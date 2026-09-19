"""One-shot runtime for the authorized four-candidate topology diagnostic."""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_nodistill_topology_four_call_contract import (
    AUTHORIZATION_PAYLOAD_SHA256,
    SCIENTIFIC_CONTRACT_PAYLOAD_SHA256,
    VOLUME_ROOT,
    file_sha256,
    validate_execution_contract,
)

QUERY_LOCK_SCHEMA = "t4_nodistill_topology_four_call_query_lock_v1"
RESERVATION_SCHEMA = "t4_nodistill_topology_four_call_reservation_v1"
START_SCHEMA = "t4_nodistill_topology_four_call_start_v1"
RESULT_SCHEMA = "t4_nodistill_topology_four_call_result_v1"
FAILURE_SCHEMA = "t4_nodistill_topology_four_call_failure_v1"
REDUCTION_SCHEMA = "t4_nodistill_topology_four_call_reduction_v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_sealed(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("payload_sha256") != identity(
        payload
    ):
        raise ValueError(f"receipt is not self-hashed: {path}")
    return payload


def publish_once(path: Path, payload: dict[str, Any]) -> str:
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    content = _canonical_bytes(envelope) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite immutable receipt: {path}")
        return envelope["payload_sha256"]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return envelope["payload_sha256"]


def make_run_id(
    *,
    execution_identity: str,
    capsule_identity: str,
    launcher_identity: dict[str, str],
    query_ids: list[str],
) -> str:
    return identity(
        {
            "schema_version": "t4_nodistill_topology_four_call_run_identity_v1",
            "scientific_contract_payload_sha256": SCIENTIFIC_CONTRACT_PAYLOAD_SHA256,
            "authorization_payload_sha256": AUTHORIZATION_PAYLOAD_SHA256,
            "execution_contract_payload_sha256": execution_identity,
            "source_capsule_payload_sha256": capsule_identity,
            "launcher_identity": launcher_identity,
            "query_ids": query_ids,
        }
    )


def capsule_image_revision(repository_root: Path) -> dict[str, Any]:
    """Bind the standalone runtime image to the verified exact capsule."""

    execution, execution_identity, _scientific = validate_execution_contract(
        repository_root
    )
    capsule = execution["execution_source_capsule"]
    return {
        "schema_version": "t4_nodistill_topology_four_call_capsule_image_v1",
        "execution_contract_payload_sha256": execution_identity,
        "code_revision": capsule["code_revision"],
        "git_tree": capsule["git_tree"],
        "source_capsule_payload_sha256": capsule["manifest_payload_sha256"],
        "source_capsule_manifest_sha256": capsule["manifest_sha256"],
        "file_count": capsule["file_count"],
    }


def validate_capsule_image_revision(
    value: dict[str, Any], repository_root: Path
) -> None:
    if value != capsule_image_revision(repository_root):
        raise ValueError("four-call standalone image revision changed")


def make_task(
    repository_root: Path,
    *,
    launcher_identity: dict[str, str],
    image_revision: dict[str, Any],
) -> dict[str, Any]:
    execution, execution_identity, scientific = validate_execution_contract(
        repository_root
    )
    capsule_identity = execution["execution_source_capsule"]["manifest_payload_sha256"]
    query_ids = [row["query_id"] for row in scientific["queries"]]
    run_id = make_run_id(
        execution_identity=execution_identity,
        capsule_identity=capsule_identity,
        launcher_identity=launcher_identity,
        query_ids=query_ids,
    )
    return {
        "schema_version": "t4_nodistill_topology_four_call_task_v1",
        "run_id": run_id,
        "scientific_contract_payload_sha256": SCIENTIFIC_CONTRACT_PAYLOAD_SHA256,
        "authorization_payload_sha256": AUTHORIZATION_PAYLOAD_SHA256,
        "execution_contract_payload_sha256": execution_identity,
        "source_capsule_payload_sha256": capsule_identity,
        "launcher_identity": launcher_identity,
        "image_revision": image_revision,
        "query_ids": query_ids,
    }


def validate_task(
    repository_root: Path, task: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], str]:
    execution, execution_identity, scientific = validate_execution_contract(
        repository_root
    )
    launcher_identity = task.get("launcher_identity")
    if not isinstance(launcher_identity, dict) or not launcher_identity:
        raise ValueError("four-call task has no launcher source identity")
    for relative, expected in launcher_identity.items():
        if file_sha256(repository_root / relative) != expected:
            raise ValueError(f"serialized four-call source changed: {relative}")
    query_ids = [row["query_id"] for row in scientific["queries"]]
    capsule_identity = execution["execution_source_capsule"]["manifest_payload_sha256"]
    expected_run_id = make_run_id(
        execution_identity=execution_identity,
        capsule_identity=capsule_identity,
        launcher_identity=launcher_identity,
        query_ids=query_ids,
    )
    exact = {
        "schema_version": "t4_nodistill_topology_four_call_task_v1",
        "run_id": expected_run_id,
        "scientific_contract_payload_sha256": SCIENTIFIC_CONTRACT_PAYLOAD_SHA256,
        "authorization_payload_sha256": AUTHORIZATION_PAYLOAD_SHA256,
        "execution_contract_payload_sha256": execution_identity,
        "source_capsule_payload_sha256": capsule_identity,
        "launcher_identity": launcher_identity,
        "image_revision": task.get("image_revision"),
        "query_ids": query_ids,
    }
    if task != exact:
        raise ValueError("four-call task identity changed")
    return execution, scientific, execution_identity


def _assert_evaluator_assets(scientific: dict[str, Any]) -> dict[str, str]:
    evaluator = scientific["evaluator"]
    qvina = Path("/opt/dock/qvina02")
    receptor = Path("/opt/dock/receptors/parp1.pdbqt")
    observed = {
        "qvina02_sha256": file_sha256(qvina),
        "receptor_sha256": file_sha256(receptor),
    }
    if observed != {
        "qvina02_sha256": evaluator["qvina02_sha256"],
        "receptor_sha256": evaluator["receptor_sha256"],
    }:
        raise ValueError("four-call evaluator asset identity changed")
    return observed


def remote_preflight(
    task: dict[str, Any],
    repository_root: Path,
    artifact_root: Path,
    volume: Any,
    *,
    validate_revision: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Verify serialized authority, source, environment, and empty run state."""

    validate_revision(task["image_revision"])
    execution, scientific, execution_identity = validate_task(repository_root, task)
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("four-call runtime requires RDKit 2024.03.5")
    assets = _assert_evaluator_assets(scientific)
    volume.reload()
    run_root = artifact_root / task["run_id"]
    if run_root.exists():
        raise ValueError("four-call private run prefix is not empty")
    return {
        "schema_version": "t4_nodistill_topology_four_call_preflight_v1",
        "status": "PASS_ZERO_DOCKING",
        "run_id": task["run_id"],
        "execution_contract_payload_sha256": execution_identity,
        "scientific_contract_payload_sha256": SCIENTIFIC_CONTRACT_PAYLOAD_SHA256,
        "authorization_payload_sha256": AUTHORIZATION_PAYLOAD_SHA256,
        "source_capsule_payload_sha256": execution["execution_source_capsule"][
            "manifest_payload_sha256"
        ],
        "query_ids": [row["query_id"] for row in scientific["queries"]],
        "query_count": 4,
        "rdkit": rdBase.rdkitVersion,
        "openbabel": subprocess.check_output(["obabel", "-V"], text=True).strip(),
        **assets,
        "private_volume_root": VOLUME_ROOT,
        "docking_calls": 0,
    }


def _run_root(artifact_root: Path, task: dict[str, Any]) -> Path:
    return artifact_root / task["run_id"]


def prepare_dispatch(
    task: dict[str, Any],
    repository_root: Path,
    artifact_root: Path,
    volume: Any,
    *,
    validate_revision: Callable[[dict[str, Any]], None],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Commit the complete query lock and all reservations before any spawn."""

    validate_revision(task["image_revision"])
    _, scientific, execution_identity = validate_task(repository_root, task)
    _assert_evaluator_assets(scientific)
    volume.reload()
    root = _run_root(artifact_root, task)
    if root.exists():
        raise RuntimeError("four-call run state already exists; respawn is forbidden")
    queries = scientific["queries"]
    common = {
        "run_id": task["run_id"],
        "scientific_contract_payload_sha256": SCIENTIFIC_CONTRACT_PAYLOAD_SHA256,
        "authorization_payload_sha256": AUTHORIZATION_PAYLOAD_SHA256,
        "execution_contract_payload_sha256": execution_identity,
    }
    lock = {
        "schema_version": QUERY_LOCK_SCHEMA,
        **common,
        "query_ids": [row["query_id"] for row in queries],
        "queries": queries,
        "charged_call_ceiling": 4,
        "automatic_retries": 0,
        "replacement_queries": 0,
        "backfill_queries": 0,
        "locked_at_utc": _stamp(),
    }
    publish_once(root / "query_lock.json", lock)
    reservations = []
    for ordinal, query in enumerate(queries):
        reservation = {
            "schema_version": RESERVATION_SCHEMA,
            **common,
            "query_id": query["query_id"],
            "query_ordinal": ordinal,
            "endpoint_key_sha256": query["endpoint_key_sha256"],
            "attempt_ordinal": 1,
            "maximum_attempts": 1,
            "reservation_consumes_only_authorized_attempt": True,
            "retry_replacement_or_backfill_authorized": False,
            "reserved_at_utc": _stamp(),
        }
        publish_once(
            root / "queries" / query["query_id"] / "reservation.json", reservation
        )
        reservations.append(reservation)
    intent = {
        "schema_version": "t4_nodistill_topology_four_call_dispatch_intent_v1",
        **common,
        "query_ids": lock["query_ids"],
        "all_reservations_published": True,
        "worker_spawns_planned": 4,
        "automatic_retries": 0,
        "replacement_queries": 0,
        "backfill_queries": 0,
        "published_at_utc": _stamp(),
    }
    publish_once(root / "dispatch_intent.json", intent)
    volume.commit()
    return intent, reservations


def publish_dispatch_receipt(
    *,
    task: dict[str, Any],
    artifact_root: Path,
    volume: Any,
    query_id: str,
    query_ordinal: int,
    function_call_id: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": "t4_nodistill_topology_four_call_dispatch_receipt_v1",
        "run_id": task["run_id"],
        "query_id": query_id,
        "query_ordinal": query_ordinal,
        "function_call_id": function_call_id,
        "automatic_retries": 0,
        "published_at_utc": _stamp(),
    }
    root = _run_root(artifact_root, task)
    publish_once(root / "dispatches" / f"{query_ordinal:02d}_{query_id}.json", payload)
    volume.commit()
    return payload


def _query_for_task(scientific: dict[str, Any], query_id: str) -> dict[str, Any]:
    matches = [row for row in scientific["queries"] if row["query_id"] == query_id]
    if len(matches) != 1:
        raise ValueError("worker query is outside the exact four-query lock")
    return matches[0]


def _copy_pose_artifacts(source: Path, destination: Path) -> dict[str, str]:
    hashes = {}
    for name in ("l.mol", "l.pdbqt", "o.pdbqt"):
        path = source / name
        if path.is_file():
            copied = destination / name
            shutil.copyfile(path, copied)
            hashes[name] = file_sha256(copied)
    return hashes


def run_query(
    task: dict[str, Any],
    query_id: str,
    repository_root: Path,
    artifact_root: Path,
    volume: Any,
    *,
    validate_revision: Callable[[dict[str, Any]], None],
    dock: Callable[[str, str, int, dict[str, Any]], float | None],
) -> dict[str, Any]:
    """Score one pre-reserved query once and publish one terminal receipt."""

    validate_revision(task["image_revision"])
    _, scientific, execution_identity = validate_task(repository_root, task)
    assets = _assert_evaluator_assets(scientific)
    query = _query_for_task(scientific, query_id)
    molecule = Chem.MolFromSmiles(query["canonical_smiles"])
    if molecule is None or Chem.MolToSmiles(molecule) != query["canonical_smiles"]:
        raise ValueError("locked four-call query is not canonical")

    volume.reload()
    root = _run_root(artifact_root, task)
    lock = read_sealed(root / "query_lock.json")
    folder = root / "queries" / query_id
    reservation = read_sealed(folder / "reservation.json")
    if (
        lock.get("schema_version") != QUERY_LOCK_SCHEMA
        or lock.get("query_ids") != task["query_ids"]
        or reservation.get("schema_version") != RESERVATION_SCHEMA
        or reservation.get("query_id") != query_id
        or reservation.get("maximum_attempts") != 1
    ):
        raise ValueError("four-call durable query lock or reservation changed")
    started_path = folder / "started.json"
    result_path = folder / "result.json"
    failure_path = folder / "failure.json"
    if any(path.exists() for path in (started_path, result_path, failure_path)):
        raise RuntimeError("four-call query already started; retry is forbidden")

    common = {
        "run_id": task["run_id"],
        "query_id": query_id,
        "endpoint_key_sha256": query["endpoint_key_sha256"],
        "scientific_contract_payload_sha256": SCIENTIFIC_CONTRACT_PAYLOAD_SHA256,
        "authorization_payload_sha256": AUTHORIZATION_PAYLOAD_SHA256,
        "execution_contract_payload_sha256": execution_identity,
    }
    started = {
        "schema_version": START_SCHEMA,
        **common,
        "started_at_utc": _stamp(),
        "attempt_ordinal": 1,
        "target": "parp1",
        "docking_seed": scientific["evaluator"]["docking_seed"],
        **assets,
    }
    publish_once(started_path, started)
    volume.commit()
    started_clock = time.perf_counter()
    tag = f"t4topo4_{task['run_id'][:12]}_{query_id}"
    box = {
        "receptor": "/opt/dock/receptors/parp1.pdbqt",
        "coordinates": scientific["evaluator"]["docking_box"],
    }
    try:
        score = dock(
            query["canonical_smiles"],
            tag,
            scientific["evaluator"]["docking_seed"],
            box,
        )
        if score is None or not math.isfinite(float(score)):
            raise RuntimeError("bound docking adapter returned no finite score")
        terminal = {
            "schema_version": RESULT_SCHEMA,
            **common,
            "status": "complete",
            "cell_key": query["cell_key"],
            "canonical_smiles": query["canonical_smiles"],
            "score": float(score),
            "score_direction": "minimize",
            "score_unit": "kcal/mol",
            "pose_artifact_sha256": _copy_pose_artifacts(Path("/tmp") / tag, folder),
            "docking_seconds": time.perf_counter() - started_clock,
            "first_score_call_charged": 1,
            "completed_at_utc": _stamp(),
        }
        publish_once(result_path, terminal)
    except Exception as error:  # noqa: BLE001 - terminal no-retry receipt is required
        terminal = {
            "schema_version": FAILURE_SCHEMA,
            **common,
            "status": "failed_no_retry",
            "cell_key": query["cell_key"],
            "canonical_smiles": query["canonical_smiles"],
            "error_type": type(error).__name__,
            "error_message": str(error),
            "docking_seconds": time.perf_counter() - started_clock,
            "first_score_call_charged": 1,
            "retry_authorized": False,
            "failed_at_utc": _stamp(),
        }
        publish_once(failure_path, terminal)
    volume.commit()
    return terminal


def reduce_run(
    task: dict[str, Any],
    repository_root: Path,
    artifact_root: Path,
    volume: Any,
    *,
    validate_revision: Callable[[dict[str, Any]], None],
    finalize_incomplete: bool = False,
) -> dict[str, Any]:
    """Deterministically reduce exact receipts; never score or fill a query."""

    validate_revision(task["image_revision"])
    execution, scientific, execution_identity = validate_task(repository_root, task)
    volume.reload()
    root = _run_root(artifact_root, task)
    read_sealed(root / "query_lock.json")
    rows: list[dict[str, Any]] = []
    incomplete: list[dict[str, str]] = []
    charged = 0
    for query in scientific["queries"]:
        folder = root / "queries" / query["query_id"]
        reservation = read_sealed(folder / "reservation.json")
        if reservation.get("query_id") != query["query_id"]:
            raise ValueError("four-call reservation identity changed")
        started = folder / "started.json"
        terminals = [
            path
            for path in (folder / "result.json", folder / "failure.json")
            if path.exists()
        ]
        if not started.exists():
            incomplete.append(
                {"query_id": query["query_id"], "state": "reserved_not_started"}
            )
            continue
        read_sealed(started)
        charged += 1
        if len(terminals) != 1:
            incomplete.append(
                {"query_id": query["query_id"], "state": "started_without_one_terminal"}
            )
            continue
        terminal = read_sealed(terminals[0])
        if (
            terminal.get("query_id") != query["query_id"]
            or terminal.get("endpoint_key_sha256") != query["endpoint_key_sha256"]
            or terminal.get("canonical_smiles") != query["canonical_smiles"]
            or terminal.get("first_score_call_charged") != 1
        ):
            raise ValueError("four-call terminal receipt changed")
        rows.append(terminal)
    if incomplete and not finalize_incomplete:
        raise RuntimeError(
            "four-call run is incomplete; refusing to publish an interim reduction"
        )
    successes = [row for row in rows if row["status"] == "complete"]
    failures = [row for row in rows if row["status"] != "complete"]
    payload = {
        "schema_version": REDUCTION_SCHEMA,
        "status": "complete" if not incomplete else "terminal_incomplete_no_retry",
        "run_id": task["run_id"],
        "scientific_contract_payload_sha256": SCIENTIFIC_CONTRACT_PAYLOAD_SHA256,
        "authorization_payload_sha256": AUTHORIZATION_PAYLOAD_SHA256,
        "execution_contract_payload_sha256": execution_identity,
        "source_capsule_payload_sha256": execution["execution_source_capsule"][
            "manifest_payload_sha256"
        ],
        "query_count": 4,
        "reserved_attempts_consumed": 4,
        "first_score_calls_charged": charged,
        "successful_scores": len(successes),
        "failed_scores": len(failures),
        "incomplete": incomplete,
        "results": rows,
        "best_score": min((row["score"] for row in successes), default=None),
        "automatic_retries": 0,
        "replacement_queries": 0,
        "backfill_queries": 0,
        "new_generator_calls": 0,
        "software": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "claim_boundary": scientific["claim_boundary"],
    }
    publish_once(root / "reduction.json", payload)
    volume.commit()
    return payload


__all__ = [
    "capsule_image_revision",
    "make_run_id",
    "make_task",
    "prepare_dispatch",
    "publish_dispatch_receipt",
    "publish_once",
    "read_sealed",
    "reduce_run",
    "remote_preflight",
    "run_query",
    "validate_capsule_image_revision",
    "validate_task",
]
