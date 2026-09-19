"""One-shot scored execution of the sealed T4 utility acquisition panel."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from compose_v4.control.docking_value import identity

CONTRACT_SCHEMA = "t4_utility_data_acquisition_launch_contract_v1"
REQUEST_LOCK_SCHEMA = "t4_utility_data_acquisition_request_lock_v1"
RESULT_SCHEMA = "t4_utility_data_acquisition_launch_result_v1"
TERMINAL_SCHEMAS = {
    "result": "t4_utility_data_acquisition_docking_result_v1",
    "failure": "t4_utility_data_acquisition_docking_failure_v1",
}
CONTRACT_PATH = "configs/t4_utility_data_acquisition_launch_v1.json"
SCORING_PATH = "modal_apps/genmol_t4_opt_app.py"
REQUEST_LOCK_PATH = "diagnostics/t4_utility_data_acquisition_lock/attempt_1/request_lock.json"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_sealed(path: Path, *, expected_payload_sha256: str | None = None) -> dict:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    claimed = envelope.get("payload_sha256", envelope.get("contract_sha256"))
    if not isinstance(payload, dict) or claimed != identity(payload):
        raise ValueError(f"artifact is not self-hashed: {path}")
    if expected_payload_sha256 is not None and claimed != expected_payload_sha256:
        raise ValueError(f"artifact payload identity changed: {path}")
    return payload


def publish_once(path: Path, payload: dict, *, contract_key: str = "payload_sha256") -> str:
    """Atomically create one self-hashed state record, never overwrite it."""
    envelope = {"payload": payload, contract_key: identity(payload)}
    content = canonical_bytes(envelope) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite immutable state: {path}")
        return hashlib.sha256(content).hexdigest()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(content).hexdigest()


def load_contract(repo_root: Path) -> dict:
    path = repo_root / CONTRACT_PATH
    contract = read_sealed(path)
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError(f"unexpected scored-launch contract schema: {path}")
    authorization = contract.get("authorization", {})
    execution = contract.get("execution", {})
    if (
        authorization.get("modal_launch_authorized") is not True
        or authorization.get("modal_bundle_upload_authorized") is not True
        or authorization.get("scored_calls_authorized") is not True
        or authorization.get("first_score_call_ceiling") != 4
        or authorization.get("first_score_call_ceiling_if_later_authorized") != 4
        or authorization.get("candidate_lock_physical_sha256")
        != contract.get("inputs", {}).get("candidate_lock", {}).get("physical_sha256")
        or authorization.get("candidate_lock_payload_sha256")
        != contract.get("inputs", {}).get("candidate_lock", {}).get("payload_sha256")
        or authorization.get("request_lock_physical_sha256")
        != contract.get("inputs", {}).get("request_lock", {}).get("physical_sha256")
        or authorization.get("request_lock_payload_sha256")
        != contract.get("inputs", {}).get("request_lock", {}).get("payload_sha256")
        or execution.get("automatic_retries") != 0
        or execution.get("replacement") is not False
        or execution.get("backfill") is not False
        or execution.get("maximum_modal_workers") != 4
        or execution.get("requests_per_worker") != 1
    ):
        raise ValueError("scored-launch authority or one-shot execution contract changed")
    return contract


def _validate_request(request: dict, contract: dict) -> None:
    identity_payload = {
        field: request[field]
        for field in (
            "schema_version",
            "target",
            "canonical_smiles",
            "docking_seed",
            "evaluator",
            "evaluator_payload",
            "candidate_id",
            "cell",
        )
    }
    if identity_payload["schema_version"] != "t4_docking_request_identity_v1" or request.get(
        "request_id"
    ) != identity(identity_payload):
        raise ValueError("docking request identity changed")
    evaluator = request.get("evaluator_payload")
    expected = contract["evaluator"]
    if (
        not isinstance(evaluator, dict)
        or request["evaluator"] != identity(evaluator)
        or evaluator.get("schema_version") != expected["schema_version"]
        or evaluator.get("target") != request["target"]
        or evaluator.get("qvina02_sha256") != expected["qvina02_sha256"]
        or evaluator.get("docking", {}).get("cpu") != expected["quickvina_cpu_per_request"]
        or evaluator.get("docking", {}).get("num_modes") != expected["num_modes"]
        or evaluator.get("docking", {}).get("exhaustiveness") != expected["exhaustiveness"]
        or request.get("docking_seed") != expected["docking_seed"]
        or evaluator.get("scoring_implementation_sha256")
        != contract["inputs"]["scoring_implementation"]["sha256"]
        or evaluator.get("receptor_sha256") != expected["receptor_sha256"].get(request["target"])
        or evaluator.get("score_direction") != expected["score_direction"]
        or evaluator.get("score_unit") != expected["score_unit"]
    ):
        raise ValueError(f"evaluator identity changed: {request.get('request_id')}")
    molecule = Chem.MolFromSmiles(request["canonical_smiles"])
    if molecule is None or Chem.MolToSmiles(molecule) != request["canonical_smiles"]:
        raise ValueError(f"request is not the locked canonical molecule: {request['request_id']}")
    if (
        request.get("candidate_id") not in contract["request_contract"]["candidate_ids"]
        or request.get("cell") not in contract["request_contract"]["cells_and_counts"]
    ):
        raise ValueError(f"request acquisition provenance changed: {request['request_id']}")


def load_requests(repo_root: Path, contract: dict) -> tuple[dict, list[dict]]:
    entry = contract["inputs"]["request_lock"]
    path = repo_root / entry["path"]
    if sha256_file(path) != entry["physical_sha256"]:
        raise ValueError(f"request lock physical identity changed: {path}")
    lock = read_sealed(path, expected_payload_sha256=entry["payload_sha256"])
    requests = lock.get("requests")
    expected = contract["request_contract"]
    if (
        lock.get("schema_version") != REQUEST_LOCK_SCHEMA
        or lock.get("call_ceiling_if_later_exactly_authorized") != expected["expected_requests"]
        or lock.get("candidate_lock_payload_sha256")
        != contract["inputs"]["candidate_lock"]["payload_sha256"]
        or lock.get("contract_sha256")
        != contract["inputs"]["zero_oracle_contract"]["payload_sha256"]
        or lock.get("automatic_retries") != 0
        or lock.get("replacement_or_backfill") is not False
        or not isinstance(requests, list)
        or len(requests) != expected["expected_requests"]
    ):
        raise ValueError("sealed request-lock execution contract changed")
    for request in requests:
        _validate_request(request, contract)
    ordered = sorted(requests, key=lambda row: row["request_id"])
    if len({row["request_id"] for row in ordered}) != len(ordered):
        raise ValueError("request lock repeats a request identity")
    counts = Counter(row["target"] for row in ordered)
    cells = Counter(row["cell"] for row in ordered)
    candidates = {row["candidate_id"] for row in ordered}
    if (
        dict(sorted(counts.items())) != expected["targets_and_counts"]
        or dict(sorted(cells.items())) != expected["cells_and_counts"]
        or len(cells) != expected["expected_cells_with_requests"]
        or candidates != set(expected["candidate_ids"])
        or [row["request_id"] for row in ordered] != expected["request_ids"]
    ):
        raise ValueError("request-lock census differs from scored-launch contract")
    return lock, ordered


def validate_local_inputs(repo_root: Path) -> tuple[dict, dict, list[dict]]:
    contract = load_contract(repo_root)
    for name in ("zero_oracle_contract", "candidate_lock", "zero_oracle_result"):
        entry = contract["inputs"][name]
        path = repo_root / entry["path"]
        if sha256_file(path) != entry["physical_sha256"]:
            raise ValueError(f"bound launch input changed: {path}")
        read_sealed(path, expected_payload_sha256=entry["payload_sha256"])
    reference = contract["inputs"]["source_reference_registry"]
    if sha256_file(repo_root / reference["path"]) != reference["physical_sha256"]:
        raise ValueError("bound source-reference registry changed")
    scoring = contract["inputs"]["scoring_implementation"]
    if sha256_file(repo_root / scoring["path"]) != scoring["sha256"]:
        raise ValueError("bound QuickVina scoring implementation changed")
    lock, requests = load_requests(repo_root, contract)
    return contract, lock, requests


def validate_execution_inputs(repo_root: Path) -> tuple[dict, dict, list[dict]]:
    """Validate only inputs serialized into each scored worker image."""
    contract = load_contract(repo_root)
    scoring = contract["inputs"]["scoring_implementation"]
    if sha256_file(repo_root / scoring["path"]) != scoring["sha256"]:
        raise ValueError("bound QuickVina scoring implementation changed")
    lock, requests = load_requests(repo_root, contract)
    return contract, lock, requests


def make_run_id(contract: dict, launcher_identity: dict) -> str:
    return identity(
        {
            "schema_version": "t4_utility_data_acquisition_run_identity_v1",
            "contract_sha256": identity(contract),
            "request_lock_physical_sha256": contract["inputs"]["request_lock"]["physical_sha256"],
            "request_lock_payload_sha256": contract["inputs"]["request_lock"]["payload_sha256"],
            "launcher_identity": launcher_identity,
        }
    )


def _request_from_task(repo_root: Path, task: dict) -> tuple[dict, dict]:
    contract, _lock, requests = validate_execution_inputs(repo_root)
    expected_contract = task.get("contract_sha256")
    launcher_identity = task.get("launcher_identity")
    run_id = task.get("run_id")
    if (
        expected_contract != identity(contract)
        or not isinstance(run_id, str)
        or len(run_id) != 64
        or any(character not in "0123456789abcdef" for character in run_id)
        or not isinstance(launcher_identity, dict)
        or run_id != make_run_id(contract, launcher_identity)
    ):
        raise ValueError("worker contract identity changed")
    matches = [row for row in requests if row["request_id"] == task.get("request_id")]
    if len(matches) != 1:
        raise ValueError("worker request is outside exact four-request lock")
    return contract, matches[0]


def _copy_pose_artifacts(source: Path, destination: Path) -> dict[str, str]:
    hashes = {}
    for name in ("l.mol", "l.pdbqt", "o.pdbqt"):
        path = source / name
        if path.is_file():
            copied = destination / name
            shutil.copyfile(path, copied)
            hashes[name] = sha256_file(copied)
    return hashes


def remote_preflight(
    task: dict,
    repo_root: Path,
    *,
    validate_revision: Callable[[dict], None],
) -> dict:
    """Attest the exact serialized image and evaluator assets without docking."""
    import subprocess

    validate_revision(task["image_revision"])
    contract, _lock, requests = validate_execution_inputs(repo_root)
    if (
        rdBase.rdkitVersion != "2024.03.5"
        or task.get("contract_sha256") != identity(contract)
        or task.get("run_id") != make_run_id(contract, task.get("launcher_identity", {}))
    ):
        raise ValueError("remote scored-launch environment or identity changed")
    for field, expected in task["launcher_identity"].items():
        if sha256_file(repo_root / field) != expected:
            raise ValueError(f"serialized launch implementation changed: {field}")
    qvina = Path("/opt/dock/qvina02")
    if sha256_file(qvina) != contract["evaluator"]["qvina02_sha256"]:
        raise ValueError("QuickVina executable identity changed")
    receptors = {}
    for request in requests:
        evaluator = request["evaluator_payload"]
        path = Path("/opt/dock/receptors") / f"{request['target']}.pdbqt"
        observed = sha256_file(path)
        if observed != evaluator["receptor_sha256"]:
            raise ValueError(f"receptor identity changed: {request['target']}")
        receptors[request["target"]] = observed
    return {
        "schema_version": "t4_utility_data_acquisition_remote_preflight_v1",
        "status": "PASS_NO_DOCKING",
        "run_id": task["run_id"],
        "contract_sha256": identity(contract),
        "request_lock_physical_sha256": contract["inputs"]["request_lock"]["physical_sha256"],
        "requests": len(requests),
        "cells": dict(sorted(Counter(row["cell"] for row in requests).items())),
        "rdkit": rdBase.rdkitVersion,
        "openbabel": subprocess.check_output(["obabel", "-V"], text=True).strip(),
        "qvina02_sha256": sha256_file(qvina),
        "receptor_sha256": dict(sorted(receptors.items())),
        "docking_calls": 0,
    }


def run_worker(
    task: dict,
    repo_root: Path,
    artifact_root: Path,
    volume: Any,
    *,
    validate_revision: Callable[[dict], None],
    dock: Callable[..., float | None],
) -> dict:
    """Spend one request once, with committed state before and after docking."""
    validate_revision(task["image_revision"])
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("scored worker requires RDKit 2024.03.5")
    contract, request = _request_from_task(repo_root, task)
    for field, expected in task["launcher_identity"].items():
        if sha256_file(repo_root / field) != expected:
            raise ValueError(f"serialized launch implementation changed: {field}")
    evaluator = request["evaluator_payload"]
    if sha256_file(Path("/opt/dock/qvina02")) != evaluator["qvina02_sha256"]:
        raise ValueError("QuickVina executable identity changed")
    receptor = Path("/opt/dock/receptors") / f"{request['target']}.pdbqt"
    if sha256_file(receptor) != evaluator["receptor_sha256"]:
        raise ValueError(f"receptor identity changed: {request['target']}")

    volume.reload()
    root = artifact_root / "t4_utility_data_acquisition_launch" / task["run_id"]
    folder = root / "requests" / request["request_id"]
    reservation_path = folder / "reservation.json"
    started_path = folder / "started.json"
    result_path = folder / "result.json"
    failure_path = folder / "failure.json"
    if any(path.exists() for path in (reservation_path, started_path, result_path, failure_path)):
        raise RuntimeError("request already has durable state; no retry is permitted")

    common = {
        "run_id": task["run_id"],
        "request_id": request["request_id"],
        "request_lock_physical_sha256": contract["inputs"]["request_lock"]["physical_sha256"],
        "request_lock_payload_sha256": contract["inputs"]["request_lock"]["payload_sha256"],
        "contract_sha256": identity(contract),
    }
    reservation = {
        "schema_version": "t4_utility_data_acquisition_reservation_v1",
        **common,
        "reserved_at_utc": _stamp(),
        "attempt_ordinal": 1,
        "maximum_attempts": 1,
        "no_retry_replacement_or_backfill": True,
    }
    publish_once(reservation_path, reservation)
    volume.commit()
    started = {
        "schema_version": "t4_utility_data_acquisition_start_v1",
        **common,
        "started_at_utc": _stamp(),
        "target": request["target"],
        "docking_seed": request["docking_seed"],
        "evaluator": request["evaluator"],
    }
    publish_once(started_path, started)
    volume.commit()
    started_clock = time.perf_counter()
    print(
        json.dumps(
            {
                "phase": "docking_started",
                "request_id": request["request_id"],
                "target": request["target"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        tag = f"t4uda04_{task['run_id'][:12]}_{request['request_id']}"
        score = dock(
            request["canonical_smiles"],
            request["target"],
            tag,
            cpu=1,
            random_seed=request["docking_seed"],
        )
        pose_hashes = _copy_pose_artifacts(Path("/tmp") / tag, folder)
        if score is None:
            raise RuntimeError("bound docking wrapper returned no score")
        if not math.isfinite(score):
            raise ValueError("bound docking wrapper returned a nonfinite score")
        terminal = {
            "schema_version": TERMINAL_SCHEMAS["result"],
            **common,
            "status": "complete",
            "target": request["target"],
            "canonical_smiles": request["canonical_smiles"],
            "docking_seed": request["docking_seed"],
            "evaluator": request["evaluator"],
            "score": float(score),
            "score_direction": evaluator["score_direction"],
            "score_unit": evaluator["score_unit"],
            "pose_artifact_sha256": pose_hashes,
            "docking_seconds": time.perf_counter() - started_clock,
            "completed_at_utc": _stamp(),
            "first_score_call_charged": 1,
        }
        publish_once(result_path, terminal)
    # The terminal failure receipt is the auditable replacement for retrying any
    # operational docking error. The exception is recorded and never hidden.
    except Exception as error:  # noqa: BLE001
        terminal = {
            "schema_version": TERMINAL_SCHEMAS["failure"],
            **common,
            "status": "failed_no_retry",
            "target": request["target"],
            "canonical_smiles": request["canonical_smiles"],
            "docking_seed": request["docking_seed"],
            "evaluator": request["evaluator"],
            "error_type": type(error).__name__,
            "error_message": str(error),
            "docking_seconds": time.perf_counter() - started_clock,
            "failed_at_utc": _stamp(),
            "first_score_call_charged": 1,
            "retry_authorized": False,
        }
        publish_once(failure_path, terminal)
    volume.commit()
    print(
        json.dumps(
            {
                "phase": "docking_terminal",
                "request_id": request["request_id"],
                "status": terminal["status"],
                "score": terminal.get("score"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return terminal


def reduce_run(
    task: dict,
    repo_root: Path,
    artifact_root: Path,
    volume: Any,
    *,
    validate_revision: Callable[[dict], None],
) -> dict:
    """Reduce immutable per-request records in request-ID order without redocking."""
    validate_revision(task["image_revision"])
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("scored reducer requires RDKit 2024.03.5")
    contract, _lock, requests = validate_execution_inputs(repo_root)
    if task.get("contract_sha256") != identity(contract) or task.get("run_id") != make_run_id(
        contract, task.get("launcher_identity", {})
    ):
        raise ValueError("reducer contract identity changed")
    for field, expected in task["launcher_identity"].items():
        if sha256_file(repo_root / field) != expected:
            raise ValueError(f"serialized launch implementation changed: {field}")
    volume.reload()
    root = artifact_root / "t4_utility_data_acquisition_launch" / task["run_id"]
    rows, incomplete = [], []
    for request in requests:
        folder = root / "requests" / request["request_id"]
        reservation = folder / "reservation.json"
        started = folder / "started.json"
        result = folder / "result.json"
        failure = folder / "failure.json"
        if not reservation.exists():
            incomplete.append({"request_id": request["request_id"], "state": "unreserved"})
            continue
        read_sealed(reservation)
        terminals = [path for path in (result, failure) if path.exists()]
        if not started.exists() or len(terminals) != 1:
            incomplete.append(
                {
                    "request_id": request["request_id"],
                    "state": "reserved_without_one_terminal",
                }
            )
            continue
        read_sealed(started)
        terminal = read_sealed(terminals[0])
        if (
            terminal.get("request_id") != request["request_id"]
            or terminal.get("run_id") != task["run_id"]
            or terminal.get("first_score_call_charged") != 1
        ):
            raise ValueError(f"terminal record identity changed: {request['request_id']}")
        rows.append(
            {
                **terminal,
                "candidate_id": request["candidate_id"],
                "cell": request["cell"],
            }
        )
    rows.sort(key=lambda row: row["request_id"])
    successes = [row for row in rows if row["status"] == "complete"]
    failures = [row for row in rows if row["status"] != "complete"]
    by_target: dict[str, list[float]] = defaultdict(list)
    by_cell: dict[str, list[float]] = defaultdict(list)
    for row in successes:
        by_target[row["target"]].append(row["score"])
        by_cell[row["cell"]].append(row["score"])
    cell_pair_status = {
        cell: {
            "successful_scores": len(values),
            "finite_non_tied_pair": len(values) == 2 and values[0] != values[1],
        }
        for cell, values in sorted(by_cell.items())
    }
    result_payload = {
        "schema_version": RESULT_SCHEMA,
        "status": (
            "complete" if len(rows) == len(requests) and not incomplete else "incomplete_no_retry"
        ),
        "run_id": task["run_id"],
        "contract_sha256": identity(contract),
        "request_lock_physical_sha256": contract["inputs"]["request_lock"]["physical_sha256"],
        "request_lock_payload_sha256": contract["inputs"]["request_lock"]["payload_sha256"],
        "request_count": len(requests),
        "reservations": len(rows) + sum(row["state"] != "unreserved" for row in incomplete),
        "first_score_calls_charged": len(rows)
        + sum(row["state"] != "unreserved" for row in incomplete),
        "successful_scores": len(successes),
        "failed_scores": len(failures),
        "incomplete": incomplete,
        "target_summary": {
            target: {"successful_scores": len(values), "best_score": min(values)}
            for target, values in sorted(by_target.items())
        },
        "cell_pair_status": cell_pair_status,
        "conditional_data_gate_ready": (
            len(successes) == len(requests)
            and not incomplete
            and set(cell_pair_status) == set(contract["request_contract"]["cells_and_counts"])
            and all(row["finite_non_tied_pair"] for row in cell_pair_status.values())
        ),
        "requests": rows,
        "automatic_retries": 0,
        "replacement_or_backfill": False,
        "new_generator_calls": 0,
        "candidate_trace_available_in_bound_candidate_lock": True,
        "software": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "interpretation_limit": contract["interpretation_limit"],
    }
    publish_once(root / "result.json", result_payload)
    volume.commit()
    return result_payload
