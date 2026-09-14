"""Prepare, launch, monitor and collect the three-arm T4 diagnostic."""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from rdkit import rdBase

from compose_v4.control.dynamic_program_synthesis import (
    CAPACITY_AWARE_THRESHOLD,
    CAPACITY_BOOTSTRAP_PAIR_ATTEMPTS,
    CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS,
    FRESH_SYNTHESIS_PROBABILITY,
    GENERIC_MODULES,
    MAX_SEGMENT_LENGTH,
    NEAR_CAPACITY_MODULE_COUNT_PROBABILITIES,
    NEAR_CAPACITY_MODULE_WEIGHTS,
    ONLINE_COMPOSITION_PROBABILITY,
    initial_dynamic_program_batch,
)
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.control.edit_program_policy import ProgramEntry
from compose_v4.control.program_transfer import initial_program_batch
from compose_v4.experiments import t4_frozen_program_benchmark as benchmark
from compose_v4.experiments.continuation_profile import (
    canonical_bytes,
    publish_json,
    sha256_file,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_no_complete_routes import (
    APP,
    APP_NAME,
    ARM_KINDS,
    ARMS,
    COMPATIBILITY_REPINS,
    CONTRACT,
    LIBRARY,
    PREFLIGHT,
    SELECTED_UNITS,
    SOURCE_CONTRACT,
    SOURCE_LIBRARY,
    load_contract,
    partition_library,
    program_ids,
    recovered_improvement,
)
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = ROOT / "diagnostics/t4_no_complete_routes/attempt_1"
RECEIPT = ATTEMPT / "launch.json"
RESULT = ATTEMPT / "result.json"
SOURCE_RUN_ID = "54c3cb6d4a708cecc45c5037d8873a3f051538c6bc787bf84eff161f30be90da"
SOURCE_RESULT = "diagnostics/t4_program_curriculum/attempt_1/remote_result.json"


def _write_plain_json(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    temporary.replace(path)
    return sha256_file(path)


def _source_scores():
    source_path = ROOT / SOURCE_RESULT
    payload = unseal(source_path)
    selected = {}
    for row in payload["rows"]:
        unit_id = f"{row['cell']}_r0"
        if row.get("role") != "seed_control" or unit_id not in SELECTED_UNITS:
            continue
        selected[unit_id] = {
            "score": row["ds"],
            "candidate_id": row["candidate_id"],
            "docking_seed": row["docking_seed"],
            "oracle_protocol": row["oracle_protocol"],
            "pose_sha256": row["pose_sha256"],
        }
    if set(selected) != set(SELECTED_UNITS) or any(
        row["score"] is None for row in selected.values()
    ):
        raise ValueError("exact measured source controls are incomplete")
    return {
        "path": SOURCE_RESULT,
        "sha256": sha256_file(source_path),
        "role": "previously charged physical-pipeline source controls; protocol IDs differ",
        "units": selected,
    }


def prepare():
    contract_path, library_path = ROOT / CONTRACT, ROOT / LIBRARY
    if contract_path.exists() or library_path.exists() or (ROOT / PREFLIGHT).exists():
        raise ValueError("complete-route diagnostic is already prepared; do not relock")
    source_contract = unseal(ROOT / SOURCE_CONTRACT)
    source_rows = json.loads((ROOT / SOURCE_LIBRARY).read_text())
    retained, removed = partition_library(source_rows)
    if len(source_rows) != 146 or len(retained) != 69 or len(removed) != 77:
        raise ValueError("expected exact complete-route partition 146 = 69 + 77")
    library_sha = _write_plain_json(library_path, retained)

    contract = deepcopy(source_contract)
    for path, record in COMPATIBILITY_REPINS.items():
        if (
            source_contract["inputs"].get(path) != record["source_sha256"]
            or sha256_file(ROOT / path) != record["diagnostic_sha256"]
        ):
            raise ValueError(f"frozen-source compatibility lineage changed: {path}")
        contract["inputs"][path] = record["diagnostic_sha256"]
    contract["library_path"] = LIBRARY
    contract["library_programs"] = 69
    del contract["inputs"][SOURCE_LIBRARY]
    for material in (
        LIBRARY,
        "docs/T4_NO_COMPLETE_ROUTES_DIAGNOSTIC.md",
        "src/compose_v4/control/dynamic_program_synthesis.py",
        "src/compose_v4/experiments/t4_no_complete_routes.py",
        APP,
        "tools/t4_no_complete_routes.py",
    ):
        contract["inputs"][material] = sha256_file(ROOT / material)
    contract["information_regime"] = {
        **contract["information_regime"],
        "diagnostic": "77 exact whole-route entries mechanically removed; retained 69 remain T4-derived",
        "comparison": "read-only reuse of completed full-146 replicate-0 outcomes",
        "dynamic_only": "zero source-library rows; runtime generic-module composition; online reuse only after a charged observation",
    }
    contract["limitations"] = [
        *contract["limitations"],
        "The retained 69 programs remain T4-derived development artifacts.",
        "The generic module vocabulary was designed using benchmark-informed development experience.",
        "One matched replicate on three selected cells is a diagnostic, not a full benchmark.",
    ]
    contract["complete_route_ablation"] = {
        "schema_version": "t4_complete_route_ablation_v1",
        "selected_units": list(SELECTED_UNITS),
        "source_contract": SOURCE_CONTRACT,
        "source_contract_sha256": sha256_file(ROOT / SOURCE_CONTRACT),
        "source_library": SOURCE_LIBRARY,
        "source_library_sha256": sha256_file(ROOT / SOURCE_LIBRARY),
        "derived_library": LIBRARY,
        "derived_library_sha256": library_sha,
        "removal_predicate": {
            "program_block_count": 1,
            "sole_block_label": "compiled_complete_transformation",
        },
        "retained_program_ids": program_ids(retained),
        "removed_program_ids": program_ids(removed),
        "full_arm": {
            "source_run_id": SOURCE_RUN_ID,
            "source_volume": "compose-v4-artifacts",
            "source_kind": benchmark.KIND,
            "reuse_only": True,
            "new_oracle_calls": 0,
        },
        "source_scores": _source_scores(),
        "new_call_ceiling": 6000,
        "max_concurrent_workers": 6,
        "confirmation_calls": 0,
        "automatic_retries": 0,
        "arms": list(ARMS),
        "runtime_compatibility_repins": COMPATIBILITY_REPINS,
        "dynamic_only": {
            "initial_route_archive": [],
            "source_library_rows_loaded": 0,
            "generic_modules": list(GENERIC_MODULES),
            "max_modules": 3,
            "max_segment_length": MAX_SEGMENT_LENGTH,
            "module_count_probabilities": [0.4, 0.4, 0.2],
            "capacity_aware_threshold": CAPACITY_AWARE_THRESHOLD,
            "capacity_bootstrap_pair_attempts": CAPACITY_BOOTSTRAP_PAIR_ATTEMPTS,
            "capacity_bootstrap_single_attempts": (CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS),
            "near_capacity_module_count_probabilities": list(
                NEAR_CAPACITY_MODULE_COUNT_PROBABILITIES
            ),
            "near_capacity_module_weights": NEAR_CAPACITY_MODULE_WEIGHTS,
            "fresh_synthesis_probability": FRESH_SYNTHESIS_PROBABILITY,
            "online_composition_probability": ONLINE_COMPOSITION_PROBABILITY,
            "intermediate_task_evaluations": 0,
            "max_primitives": contract["controller"]["max_primitives"],
            "max_blocks": contract["controller"]["max_blocks"],
        },
    }
    seal(contract_path, contract)
    load_contract(ROOT)
    print(
        json.dumps(
            {
                "contract": CONTRACT,
                "contract_sha256": sha256_file(contract_path),
                "source_programs": 146,
                "retained_programs": 69,
                "removed_complete_routes": 77,
                "selected_units": SELECTED_UNITS,
                "arms": ARMS,
                "new_oracle_call_ceiling": 6000,
                "new_oracle_calls": 0,
                "mechanical_prepare_rdkit": rdBase.rdkitVersion,
            },
            indent=2,
        )
    )


def build_preflight():
    """Run exact chemistry checks; production authority requires pinned Modal."""
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("structural preflight requires frozen RDKit 2024.03.5")
    contract = load_contract(ROOT)
    rows = json.loads((ROOT / LIBRARY).read_text())
    entries = tuple(
        ProgramEntry(
            EditProgram.from_payload(row["program"]), tuple(row["source_groups"])
        )
        for row in rows
    )
    began, cells = perf_counter(), []
    for unit_id in SELECTED_UNITS:
        unit = next(row for row in contract["units"] if row["unit_id"] == unit_id)
        cell = contract["cells"][unit["cell"]]
        config = benchmark.configured(contract, unit)
        source = decode_state(cell["source_state"])
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
        for arm in ARMS:
            generator = (
                initial_program_batch
                if arm == "69_only"
                else initial_dynamic_program_batch
            )
            arm_entries = entries if arm == "69_only" else ()
            first = generator(source, arm_entries, config, **kwargs)
            second = generator(source, arm_entries, config, **kwargs)
            if first["batch_id"] != second["batch_id"]:
                raise ValueError(f"{arm} cold-start is nondeterministic: {unit_id}")
            if arm == "dynamic_only" and (
                first.get("initial_route_archive") != []
                or first.get("source_library_rows_loaded") != 0
            ):
                raise ValueError("dynamic-only preflight loaded an initial route")
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
                    raise ValueError(f"{arm} candidate failed exact replay: {unit_id}")
                if arm == "dynamic_only":
                    metadata = candidate["provenance"]["metadata"]
                    if (
                        not 1 <= metadata["completed_module_count"] <= 3
                        or metadata["intermediate_task_evaluations"] != 0
                        or metadata["source_library_rows_loaded"] != 0
                        or any(
                            row["family"] not in GENERIC_MODULES
                            and row["family"] not in ("pendant_ring", "fused_ring")
                            for row in metadata["modules"]
                        )
                    ):
                        raise ValueError(
                            "dynamic-only candidate violates its module contract"
                        )
            cells.append(
                {
                    "unit": unit_id,
                    "arm": arm,
                    "candidates": len(first["candidates"]),
                    "attempts": len(first["attempts"]),
                    "attempt_status": dict(
                        __import__("collections").Counter(
                            row["status"] for row in first["attempts"]
                        )
                    ),
                    "batch_id": first["batch_id"],
                    "proposal_seconds": first["proposal_seconds"],
                }
            )
    return {
        "schema_version": "t4_complete_route_dynamic_preflight_v1",
        "passed": len(cells) == 6,
        "contract_sha256": sha256_file(ROOT / CONTRACT),
        "cells": cells,
        "seconds": perf_counter() - began,
        "rdkit": rdBase.rdkitVersion,
        "new_oracle_calls": 0,
        "zero_yield_is_a_preserved_negative_result": True,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def preflight():
    output = ROOT / PREFLIGHT
    if output.exists():
        raise ValueError("complete-route preflight already exists; do not overwrite")
    body = build_preflight()
    publish_json(output, body)
    print(json.dumps(body, indent=2))


def remote_preflight():
    """Obtain the authoritative zero-oracle chemistry check from pinned Modal."""
    import modal

    output = ROOT / PREFLIGHT
    if output.exists():
        raise ValueError("complete-route preflight already exists; do not overwrite")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("remote preflight requires clean committed source")
    body = modal.Function.from_name(APP_NAME, "structural_preflight").remote(
        _task(include_preflight=False)
    )
    if body.get("passed") is not True or body.get("new_oracle_calls") != 0:
        raise ValueError("remote structural preflight did not pass")
    publish_json(output, body)
    print(json.dumps(body, indent=2))


def _task(*, include_preflight=True):
    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    revision = local_image_revision(
        expected_commit=assert_synced(strict=True)["commit"]
    )
    paths = [CONTRACT, APP, "modal_apps/genmol_t4_opt_app.py"]
    if include_preflight:
        paths.append(PREFLIGHT)
    files = {path: sha256_file(ROOT / path) for path in paths}
    body = {"image_revision": revision, "files_sha256": files}
    return {**body, "run_id": benchmark.identity(body)}


def launch(receipt: Path):
    import modal

    if receipt.exists():
        raise ValueError(
            "diagnostic launch receipt exists; monitor instead of spawning"
        )
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("diagnostic launch requires clean committed source")
    contract = load_contract(ROOT)
    task = _task()
    remote = modal.Function.from_name(APP_NAME, "preflight").remote(task)
    if remote.get("passed") is not True or remote.get("new_oracle_calls") != 0:
        raise ValueError("remote zero-oracle preflight failed")
    calls = {}
    worker = modal.Function.from_name(APP_NAME, "worker")
    for arm in ARMS:
        for unit_id in SELECTED_UNITS:
            call = worker.spawn({**task, "arm": arm, "unit_id": unit_id})
            calls[f"{arm}/{unit_id}"] = call.object_id
    saved = {
        "schema_version": "t4_no_complete_routes_launch_v1",
        "task": task,
        "calls": calls,
        "volume": "compose-v4-artifacts",
        "volume_paths": {arm: f"{ARM_KINDS[arm]}/{task['run_id']}" for arm in ARMS},
        "contract_sha256": sha256_file(ROOT / CONTRACT),
        "preflight_sha256": sha256_file(ROOT / PREFLIGHT),
        "new_call_ceiling": contract["complete_route_ablation"]["new_call_ceiling"],
        "automatic_retries": 0,
    }
    seal(receipt, saved)
    print(
        json.dumps(
            {key: value for key, value in saved.items() if key != "task"}, indent=2
        )
    )


def _remote_json(volume, path):
    raw = b"".join(volume.read_file(path))
    value = json.loads(raw)
    if set(value) == {"payload", "payload_sha256"}:
        payload = value["payload"]
        if benchmark.identity(payload) != value["payload_sha256"]:
            raise ValueError(f"corrupt remote sealed artifact: {path}")
        return payload, __import__("hashlib").sha256(raw).hexdigest()
    return value, __import__("hashlib").sha256(raw).hexdigest()


def status(receipt: Path):
    import modal

    saved = unseal(receipt)
    contract = load_contract(ROOT)
    volume = modal.Volume.from_name(saved["volume"])
    ablation = contract["complete_route_ablation"]
    full_prefix = (
        f"{ablation['full_arm']['source_kind']}/"
        f"{ablation['full_arm']['source_run_id']}"
    )
    rows = []
    for arm in ARMS:
        prefix = saved["volume_paths"][arm].strip("/")
        for unit_id in SELECTED_UNITS:
            folder = f"{prefix}/units/{unit_id}"
            try:
                names = {
                    entry.path.rsplit("/", 1)[-1] for entry in volume.listdir(folder)
                }
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
            row = (
                {}
                if filename is None
                else _remote_json(volume, f"{folder}/{filename}")[0]
            )
            calls = row.get(
                "oracle_calls", row.get("queries_total", row.get("queries", 0))
            )
            best = row.get("best_score")
            if best is None and row.get("champion") is not None:
                best = row["champion"]["score"]
            full, _ = _remote_json(volume, f"{full_prefix}/units/{unit_id}/result.json")
            full_curve = full["curve"]
            full_same_calls = (
                None
                if not calls
                else full_curve[min(calls, len(full_curve)) - 1]["best_score"]
            )
            full_final = (
                None if full.get("champion") is None else full["champion"]["score"]
            )
            rows.append(
                {
                    "arm": arm,
                    "unit": unit_id,
                    "status": row.get("status", "running" if filename else "pending"),
                    "calls": calls,
                    "best": best,
                    "full_146_best_at_same_calls": full_same_calls,
                    "gap_vs_full_at_same_calls": (
                        None
                        if best is None or full_same_calls is None
                        else best - full_same_calls
                    ),
                    "full_146_final_best": full_final,
                    "full_146_final_calls": full["oracle_calls"],
                    "ivg_mean": contract["ivg_reported"][unit_id.rsplit("_r", 1)[0]][
                        "mean"
                    ],
                    "updated_at_utc": row.get("at", row.get("completed_at_utc")),
                }
            )
    print(
        json.dumps(
            {"units": rows, "total_calls": sum(r["calls"] for r in rows)}, indent=2
        )
    )


def collect(receipt: Path, output: Path):
    import modal

    saved = unseal(receipt)
    contract = load_contract(ROOT)
    volume = modal.Volume.from_name(saved["volume"])
    source = contract["complete_route_ablation"]
    full_prefix = (
        f"{source['full_arm']['source_kind']}/{source['full_arm']['source_run_id']}"
    )
    rows, calls = [], 0
    for unit_id in SELECTED_UNITS:
        full_path = f"{full_prefix}/units/{unit_id}/result.json"
        full, full_sha = _remote_json(volume, full_path)
        full_best = None if full.get("champion") is None else full["champion"]["score"]
        start = source["source_scores"]["units"][unit_id]
        compared = {}
        for arm in ARMS:
            arm_path = (
                f"{saved['volume_paths'][arm].strip('/')}/units/{unit_id}/result.json"
            )
            result, result_sha = _remote_json(volume, arm_path)
            best = (
                None if result.get("champion") is None else result["champion"]["score"]
            )
            calls += result.get("oracle_calls", 0)
            compared[arm] = {
                "result_path": arm_path,
                "result_sha256": result_sha,
                "score": best,
                "oracle_calls": result.get("oracle_calls"),
                "termination": result.get("termination"),
                "curve": result.get("curve"),
                "improvement": None if best is None else start["score"] - best,
                "fraction_full_improvement_recovered": recovered_improvement(
                    start["score"], full_best, best
                ),
                "recovery_interpretation": (
                    "descriptive; source control protocol ID differs"
                ),
            }
        rows.append(
            {
                "unit": unit_id,
                "source_score": start,
                "full_146": {
                    "result_path": full_path,
                    "result_sha256": full_sha,
                    "score": full_best,
                    "oracle_calls": full.get("oracle_calls"),
                    "termination": full.get("termination"),
                    "curve": full.get("curve"),
                },
                "full_improvement": (
                    None if full_best is None else start["score"] - full_best
                ),
                **compared,
            }
        )
    result = {
        "schema_version": "t4_complete_route_dynamic_result_v1",
        "contract": CONTRACT,
        "contract_sha256": sha256_file(ROOT / CONTRACT),
        "launch_receipt": str(receipt.relative_to(ROOT)),
        "launch_receipt_sha256": sha256_file(receipt),
        "new_oracle_calls": calls,
        "new_call_ceiling": source["new_call_ceiling"],
        "rows": rows,
        "evidence": "matched three-cell complete-route ablation and dynamic-synthesis diagnostic",
        "limitations": [
            "one replicate per selected cell",
            "the retained 69 programs remain T4-derived",
            "the dynamic module vocabulary reflects benchmark-informed method development",
            "docking is stochastic",
            "source controls share physical docking settings but have distinct oracle-domain IDs",
        ],
    }
    seal(output, result)
    print(json.dumps({"new_oracle_calls": calls, "rows": rows}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "preflight",
            "remote-preflight",
            "launch",
            "status",
            "collect",
        ),
    )
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "preflight":
        preflight()
    elif args.mode == "remote-preflight":
        remote_preflight()
    elif args.mode == "launch":
        launch(args.receipt)
    elif args.mode == "status":
        status(args.receipt)
    else:
        collect(args.receipt, args.output)


if __name__ == "__main__":
    main()
