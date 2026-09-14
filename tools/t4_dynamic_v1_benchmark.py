#!/usr/bin/env python3
"""Prepare, preflight, launch, monitor and collect Dynamic COMPOSE v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from rdkit import rdBase

from compose_v4.control.dynamic_program_synthesis_v1 import (
    DynamicV1ProgramOptimizer,
    initial_dynamic_program_batch_v1,
)
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.experiments import t4_frozen_program_benchmark as benchmark
from compose_v4.experiments.continuation_profile import (
    publish_json,
    sha256_file,
)
from compose_v4.experiments.t4_dynamic_v1 import (
    APP,
    APP_NAME,
    CONTRACT,
    EMPTY_LIBRARY,
    KIND,
    PREFLIGHT,
    REACHABILITY,
    SELECTED_UNITS,
    SOURCE_CONTRACT,
    load_contract,
    v1_contract_payload,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = ROOT / "diagnostics/t4_dynamic_v1"
REFERENCE = ATTEMPT / "comparison_reference.json"
RECEIPT = ATTEMPT / "launch.json"
RESULT = ATTEMPT / "result.json"


def _read_sealed(path: Path) -> tuple[dict, dict]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if set(value) != {"payload", "payload_sha256"}:
        raise ValueError(f"{path}: expected one sealed JSON envelope")
    from compose_v4.control.docking_value import identity

    if identity(value["payload"]) != value["payload_sha256"]:
        raise ValueError(f"{path}: sealed payload identity changed")
    return value["payload"], {
        "source_path": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_payload_sha256": value["payload_sha256"],
    }


def _result_summary(path: Path, *, role: str) -> dict:
    payload, provenance = _read_sealed(path)
    if payload.get("status") != "complete" or payload.get("champion") is None:
        raise ValueError(f"{path}: comparison result is not durably complete")
    return {
        **provenance,
        "role": role,
        "unit_id": payload["unit"]["unit_id"],
        "oracle_calls": payload["oracle_calls"],
        "successful_oracle_calls": payload["successful_oracle_calls"],
        "failed_oracle_calls": payload["failed_oracle_calls"],
        "termination": payload["termination"],
        "best_score": payload["champion"]["score"],
        "best_query_index": payload["champion"]["query_index"],
        "endpoint_sha256": __import__(
            "compose_v4.control.docking_value", fromlist=["identity"]
        ).identity(payload["champion"]["endpoint"]),
        "curve": payload["curve"],
    }


def prepare(args) -> None:
    if any(path.exists() for path in (ROOT / CONTRACT, REFERENCE, ROOT / PREFLIGHT)):
        raise ValueError(
            "Dynamic-v1 development artifacts already exist; do not relock"
        )
    source = unseal(ROOT / SOURCE_CONTRACT)
    full_paths = {
        "5ht1b_0_r0": args.full_5ht1b,
        "braf_1_r0": args.full_braf,
        "jak2_1_r0": args.full_jak2,
    }
    v0_paths = {
        "5ht1b_0_r0": args.v0_5ht1b,
        "braf_1_r0": args.v0_braf,
        "jak2_1_r0": args.v0_jak2,
    }
    reference = {
        "schema_version": "t4_dynamic_v1_comparison_reference_v1",
        "role": "read-only Full-146 and Dynamic-v0 development outcomes",
        "full_146": {
            unit: _result_summary(path, role="Full-146")
            for unit, path in full_paths.items()
        },
        "dynamic_v0": {
            unit: _result_summary(path, role="Dynamic-v0")
            for unit, path in v0_paths.items()
        },
        "new_oracle_calls": 0,
    }
    if set(reference["full_146"]) != set(SELECTED_UNITS) or set(
        reference["dynamic_v0"]
    ) != set(SELECTED_UNITS):
        raise ValueError("comparison reference unit census changed")
    seal(REFERENCE, reference)

    contract = deepcopy(source)
    contract["library_path"] = EMPTY_LIBRARY
    contract["library_programs"] = 0
    material = (
        SOURCE_CONTRACT,
        str(REFERENCE.relative_to(ROOT)),
        REACHABILITY,
        EMPTY_LIBRARY,
        "docs/GENMOL_T4_SEEDS.json",
        "docs/T4_DYNAMIC_V1.md",
        "modal_apps/genmol_t4_opt_app.py",
        APP,
        "src/compose_v4/control/adaptive_program_optimizer.py",
        "src/compose_v4/control/dynamic_program_synthesis.py",
        "src/compose_v4/control/dynamic_program_synthesis_v1.py",
        "src/compose_v4/control/program_transfer.py",
        "src/compose_v4/experiments/t4_frozen_program_benchmark.py",
        "src/compose_v4/experiments/t4_dynamic_v1.py",
        "tools/t4_dynamic_v1.py",
        "tools/t4_dynamic_v1_benchmark.py",
    )
    contract["inputs"] = {path: sha256_file(ROOT / path) for path in material}
    contract["information_regime"] = {
        **contract["information_regime"],
        "dynamic_v1": (
            "zero initial routes; generic v1 modules; answer-known routes used only "
            "by the sealed zero-oracle reachability audit"
        ),
        "comparison": "Full-146 and Dynamic-v0 outcomes reused read-only",
    }
    contract["limitations"] = [
        *contract["limitations"],
        "Dynamic-v1 was designed on three answer-known T4 development cells.",
        "One search replicate and stochastic docking limit general conclusions.",
        "The remaining twelve T4 cells are outside this development milestone.",
    ]
    contract["dynamic_v1_development"] = v1_contract_payload()
    seal(ROOT / CONTRACT, contract)
    load_contract(ROOT)
    print(
        json.dumps(
            {
                "contract": CONTRACT,
                "contract_sha256": sha256_file(ROOT / CONTRACT),
                "reference": str(REFERENCE.relative_to(ROOT)),
                "reference_sha256": sha256_file(REFERENCE),
                "selected_units": SELECTED_UNITS,
                "new_call_ceiling": 3000,
                "new_oracle_calls": 0,
            },
            indent=2,
        )
    )


def _revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def build_preflight(*, code_revision: str | None = None) -> dict:
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("Dynamic-v1 structural preflight requires RDKit 2024.03.5")
    contract = load_contract(ROOT)
    began, rows = perf_counter(), []
    for unit_id in SELECTED_UNITS:
        unit = next(row for row in contract["units"] if row["unit_id"] == unit_id)
        cell = contract["cells"][unit["cell"]]
        source = decode_state(cell["source_state"])
        config = replace(
            benchmark.configured(contract, unit),
            seed=contract["cold_start_seed"],
            candidates_per_batch=4,
            attempts_per_batch=32,
            wall_seconds=15,
        )
        kwargs = {
            "source_group": benchmark.identity(
                {
                    "target": unit["target"],
                    "source_idx": unit["source_idx"],
                    "seed": unit["original_seed"],
                }
            ),
            "oracle_protocol": unit["oracle_protocol"],
            "eligibility": benchmark.strict_endpoint_scorer(
                unit["original_seed"], delta=contract["delta"]
            ),
        }
        first = initial_dynamic_program_batch_v1(source, (), config, **kwargs)
        second = initial_dynamic_program_batch_v1(source, (), config, **kwargs)
        if first["batch_id"] != second["batch_id"]:
            raise ValueError(f"Dynamic-v1 initial batch is nondeterministic: {unit_id}")
        if not first["candidates"]:
            raise ValueError(
                f"Dynamic-v1 representative preflight yielded no candidate: {unit_id}"
            )
        for candidate in first["candidates"]:
            program = EditProgram.from_payload(candidate["program"])
            _, replay = execute_program_graph(
                decode_state(candidate["source_state"]),
                compile_program_graph(program),
                tuple(candidate["assignment"]),
                max_primitives=config.max_primitives,
                max_blocks=config.max_blocks,
            )
            if (
                replay != candidate["trace"]
                or replay["endpoint"] != candidate["endpoint"]
            ):
                raise ValueError(f"Dynamic-v1 candidate changed on replay: {unit_id}")
        optimizer = DynamicV1ProgramOptimizer(
            benchmark.configured(contract, unit),
            source_group=kwargs["source_group"],
            oracle_protocol=kwargs["oracle_protocol"],
            hierarchy=None,
        )
        rows.append(
            {
                "unit_id": unit_id,
                "candidates": len(first["candidates"]),
                "attempts": len(first["attempts"]),
                "batch_id": first["batch_id"],
                "proposal_seconds": first["proposal_seconds"],
                "optimizer_class": type(optimizer).__name__,
            }
        )
    return {
        "schema_version": "t4_dynamic_v1_preflight_v1",
        "passed": len(rows) == 3,
        "contract_sha256": sha256_file(ROOT / CONTRACT),
        "reachability_sha256": sha256_file(ROOT / REACHABILITY),
        "representative_candidates": rows,
        "preflight_batch_size": 4,
        "seconds": perf_counter() - began,
        "rdkit": rdBase.rdkitVersion,
        "new_oracle_calls": 0,
        "code_revision": _revision() if code_revision is None else code_revision,
    }


def preflight() -> None:
    output = ROOT / PREFLIGHT
    if output.exists():
        raise ValueError("Dynamic-v1 preflight exists; do not overwrite")
    body = build_preflight()
    publish_json(output, body)
    print(json.dumps(body, indent=2))


def refresh_prequery_contract() -> None:
    """Reseal only pre-oracle implementation identities before the first commit."""
    contract_path = ROOT / CONTRACT
    if (ROOT / PREFLIGHT).exists() or RECEIPT.exists():
        raise ValueError("cannot refresh Dynamic-v1 after preflight or launch locking")
    contract = unseal(contract_path)
    if contract.get("dynamic_v1_development") != v1_contract_payload():
        raise ValueError("Dynamic-v1 scientific policy changed before refresh")
    prior_sha256 = sha256_file(contract_path)
    contract["inputs"].pop("diagnostics/t4_dynamic_v1/teacher_reachability.json", None)
    material = (
        SOURCE_CONTRACT,
        str(REFERENCE.relative_to(ROOT)),
        REACHABILITY,
        EMPTY_LIBRARY,
        "docs/GENMOL_T4_SEEDS.json",
        "docs/T4_DYNAMIC_V1.md",
        "modal_apps/genmol_t4_opt_app.py",
        APP,
        "src/compose_v4/control/adaptive_program_optimizer.py",
        "src/compose_v4/control/dynamic_program_synthesis.py",
        "src/compose_v4/control/dynamic_program_synthesis_v1.py",
        "src/compose_v4/control/program_transfer.py",
        "src/compose_v4/experiments/t4_frozen_program_benchmark.py",
        "src/compose_v4/experiments/t4_dynamic_v1.py",
        "tools/t4_dynamic_v1.py",
        "tools/t4_dynamic_v1_benchmark.py",
    )
    contract["inputs"] = {path: sha256_file(ROOT / path) for path in material}
    contract.setdefault("dynamic_v1_prequery_repairs", []).append(
        {
            "reason": (
                "local RDKit 2026.03.6 correctly rejected by the pinned 2024.03.5 "
                "gate; move zero-oracle structural preflight to the pinned image"
            ),
            "prior_contract_sha256": prior_sha256,
            "oracle_calls": 0,
            "scientific_policy_changed": False,
        }
    )
    seal(contract_path, contract)
    load_contract(ROOT)
    print(
        json.dumps(
            {
                "status": "prequery_contract_refreshed",
                "prior_contract_sha256": prior_sha256,
                "contract_sha256": sha256_file(contract_path),
                "oracle_calls": 0,
                "scientific_policy_changed": False,
            },
            indent=2,
        )
    )


def _task(*, include_preflight: bool = True):
    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    revision = local_image_revision(
        expected_commit=assert_synced(strict=True)["commit"]
    )
    paths = [CONTRACT, APP, "modal_apps/genmol_t4_opt_app.py"]
    if include_preflight:
        paths.append(PREFLIGHT)
    body = {
        "image_revision": revision,
        "files_sha256": {path: sha256_file(ROOT / path) for path in paths},
    }
    return {**body, "run_id": benchmark.identity(body)}


def remote_preflight() -> None:
    import modal

    output = ROOT / PREFLIGHT
    if output.exists():
        raise ValueError("Dynamic-v1 preflight exists; do not overwrite")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("remote preflight requires a clean committed worktree")
    task = _task(include_preflight=False)
    body = modal.Function.from_name(APP_NAME, "structural_preflight").remote(task)
    if body.get("passed") is not True or body.get("new_oracle_calls") != 0:
        raise ValueError("remote Dynamic-v1 structural preflight did not pass")
    publish_json(output, body)
    print(json.dumps(body, indent=2))


def launch() -> None:
    import modal

    if RECEIPT.exists():
        raise ValueError(
            "Dynamic-v1 launch receipt exists; monitor rather than relaunch"
        )
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Dynamic-v1 launch requires a clean committed worktree")
    contract = load_contract(ROOT)
    reference = unseal(REFERENCE)
    if any(
        reference["dynamic_v0"][unit]["unit_id"] != unit
        or reference["dynamic_v0"][unit]["oracle_calls"] < 1
        for unit in SELECTED_UNITS
    ):
        raise ValueError("Dynamic-v0 comparison units are not durably complete")
    task = _task()
    remote = modal.Function.from_name(APP_NAME, "preflight").remote(task)
    if remote.get("passed") is not True or remote.get("new_oracle_calls") != 0:
        raise ValueError("Dynamic-v1 remote zero-oracle preflight failed")
    worker = modal.Function.from_name(APP_NAME, "worker")
    calls = {
        unit: worker.spawn({**task, "unit_id": unit}).object_id
        for unit in SELECTED_UNITS
    }
    saved = {
        "schema_version": "t4_dynamic_v1_launch_v1",
        "task": task,
        "calls": calls,
        "volume": "compose-v4-artifacts",
        "volume_path": f"{KIND}/{task['run_id']}",
        "contract_sha256": sha256_file(ROOT / CONTRACT),
        "preflight_sha256": sha256_file(ROOT / PREFLIGHT),
        "new_call_ceiling": contract["dynamic_v1_development"]["new_call_ceiling"],
        "automatic_retries": 0,
    }
    seal(RECEIPT, saved)
    print(
        json.dumps(
            {key: value for key, value in saved.items() if key != "task"}, indent=2
        )
    )


def _remote_json(volume, path):
    raw = b"".join(volume.read_file(path))
    value = json.loads(raw)
    if set(value) == {"payload", "payload_sha256"}:
        from compose_v4.control.docking_value import identity

        if identity(value["payload"]) != value["payload_sha256"]:
            raise ValueError(f"corrupt sealed remote artifact: {path}")
        return value["payload"]
    return value


def status() -> None:
    import modal

    saved = unseal(RECEIPT)
    reference = unseal(REFERENCE)
    volume = modal.Volume.from_name(saved["volume"])
    rows = []
    for unit in SELECTED_UNITS:
        folder = f"{saved['volume_path']}/units/{unit}"
        try:
            names = {row.path.rsplit("/", 1)[-1] for row in volume.listdir(folder)}
        except modal.exception.NotFoundError:
            names = set()
        filename = next(
            (
                name
                for name in ("result.json", "failure.json", "progress.json")
                if name in names
            ),
            None,
        )
        current = (
            {} if filename is None else _remote_json(volume, f"{folder}/{filename}")
        )
        calls = current.get(
            "oracle_calls", current.get("queries_total", current.get("queries", 0))
        )
        best = current.get("best_score")
        if best is None and current.get("champion") is not None:
            best = current["champion"]["score"]
        full = reference["full_146"][unit]
        same = (
            None
            if not calls
            else full["curve"][min(calls, len(full["curve"])) - 1]["best_score"]
        )
        rows.append(
            {
                "unit": unit,
                "status": current.get("status", "running" if filename else "pending"),
                "calls": calls,
                "best": best,
                "full_146_at_same_calls": same,
                "gap_vs_full_at_same_calls": (
                    None if best is None or same is None else best - same
                ),
                "full_146_final": full["best_score"],
                "dynamic_v0_final": reference["dynamic_v0"][unit]["best_score"],
                "updated_at_utc": current.get("at", current.get("completed_at_utc")),
            }
        )
    print(
        json.dumps(
            {"units": rows, "total_calls": sum(row["calls"] for row in rows)}, indent=2
        )
    )


def collect() -> None:
    import modal

    if RESULT.exists():
        raise ValueError("Dynamic-v1 result exists; do not overwrite")
    saved = unseal(RECEIPT)
    reference = unseal(REFERENCE)
    volume = modal.Volume.from_name(saved["volume"])
    rows, calls = [], 0
    for unit in SELECTED_UNITS:
        path = f"{saved['volume_path']}/units/{unit}/result.json"
        current = _remote_json(volume, path)
        if current.get("status") != "complete":
            raise ValueError(f"Dynamic-v1 unit is not complete: {unit}")
        calls += current["oracle_calls"]
        rows.append(
            {
                "unit": unit,
                "dynamic_v1": {
                    "best_score": current["champion"]["score"],
                    "oracle_calls": current["oracle_calls"],
                    "termination": current["termination"],
                    "curve": current["curve"],
                },
                "dynamic_v0": reference["dynamic_v0"][unit],
                "full_146": reference["full_146"][unit],
            }
        )
    body = {
        "schema_version": "t4_dynamic_v1_result_v1",
        "contract": CONTRACT,
        "contract_sha256": sha256_file(ROOT / CONTRACT),
        "launch_receipt_sha256": sha256_file(RECEIPT),
        "new_oracle_calls": calls,
        "new_call_ceiling": 3000,
        "rows": rows,
        "evidence": "winner-informed three-cell Dynamic COMPOSE development diagnostic",
        "limitations": [
            "one search replicate per development cell",
            "stochastic docking",
            "answer-known Full-146 transformations informed the generic v1 capabilities",
            "not held-out benchmark evidence",
        ],
    }
    seal(RESULT, body)
    print(json.dumps({"new_oracle_calls": calls, "rows": rows}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "refresh-prequery-contract",
            "preflight",
            "remote-preflight",
            "launch",
            "status",
            "collect",
        ),
    )
    parser.add_argument("--full-5ht1b", type=Path)
    parser.add_argument("--full-braf", type=Path)
    parser.add_argument("--full-jak2", type=Path)
    parser.add_argument("--v0-5ht1b", type=Path)
    parser.add_argument("--v0-braf", type=Path)
    parser.add_argument("--v0-jak2", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        required = (
            args.full_5ht1b,
            args.full_braf,
            args.full_jak2,
            args.v0_5ht1b,
            args.v0_braf,
            args.v0_jak2,
        )
        if any(path is None for path in required):
            parser.error("prepare requires all six Full-146/Dynamic-v0 result paths")
        prepare(args)
    elif args.mode == "refresh-prequery-contract":
        refresh_prequery_contract()
    elif args.mode == "preflight":
        preflight()
    elif args.mode == "remote-preflight":
        remote_preflight()
    elif args.mode == "launch":
        launch()
    elif args.mode == "status":
        status()
    else:
        collect()


if __name__ == "__main__":
    main()
