#!/usr/bin/env python3
"""Prepare, preflight, launch and monitor the frozen Dynamic-v0 T4 wave."""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.experiments import t4_frozen_program_benchmark as benchmark
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.t4_dynamic_v0_full_suite import (
    APP,
    APP_NAME,
    CONTRACT,
    CONTRACT_V1,
    EMPTY_LIBRARY,
    KIND,
    PREFLIGHT,
    PREFLIGHT_FAILURE_V1,
    SOURCE_CONTRACT,
    configured,
    load_contract,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "diagnostics/t4_dynamic_v0_full_suite_delta06"
LAUNCH = OUTPUT / "launch.json"
VOLUME_NAME = "compose-t4-dynamic-v0-full-suite"
CENSUS = "diagnostics/ivg_t4_census/census.json"
MATERIAL = (
    "AGENTS.md",
    CENSUS,
    SOURCE_CONTRACT,
    "docs/T4_DYNAMIC_V0_FULL_SUITE.md",
    "src/compose_v4/control/adaptive_program_optimizer.py",
    "src/compose_v4/control/dynamic_program_synthesis.py",
    "src/compose_v4/experiments/t4_frozen_program_benchmark.py",
    "src/compose_v4/experiments/t4_dynamic_v0_full_suite.py",
    "modal_apps/genmol_t4_opt_app.py",
    APP,
    "tools/t4_dynamic_v0_full_suite.py",
)
SOURCE_CONTRACT_V1_SHA256 = (
    "6549602d2da13162e1691a1583228cdd05d5bf9e0c314391ea1f96a98031a3d4"
)


def _ivg_reported(census: dict, delta: float) -> dict:
    from statistics import mean, stdev

    result = {}
    for cell in census["cells"]:
        if float(cell["delta"]) != delta:
            continue
        key = f"{cell['target']}_{cell['source_idx']}"
        scores = [float(run["reported_docking_score"]) for run in cell["runs"]]
        if len(scores) != 3:
            raise ValueError(f"IVG comparator lacks three runs: {key}")
        result[key] = {
            "run_bests": scores,
            "mean": mean(scores),
            "sample_std": stdev(scores),
            "released_rows": cell["row_count"],
        }
    if len(result) != 15:
        raise ValueError("IVG delta-0.6 census must contain 15 cells")
    return result


def prepare():
    if (
        (ROOT / CONTRACT_V1).exists()
        or (ROOT / CONTRACT).exists()
        or (ROOT / EMPTY_LIBRARY).exists()
    ):
        raise ValueError("Dynamic-v0 delta-0.6 inputs already exist; preserve them")
    source = unseal(ROOT / SOURCE_CONTRACT)
    if (
        source["complete_route_ablation"]["dynamic_only"]["source_library_rows_loaded"]
        != 0
    ):
        raise ValueError("source Dynamic-v0 contract did not use an empty initial bank")
    (ROOT / EMPTY_LIBRARY).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / EMPTY_LIBRARY).write_text("[]\n")
    cells = deepcopy(source["cells"])
    source_units = {
        row["cell"]: row for row in source["units"] if row["replicate"] == 0
    }
    units = []
    for cell, row in sorted(cells.items()):
        source_unit = source_units[cell]
        domain = deepcopy(source_unit["oracle_domain"])
        domain["schema_version"] = "t4_strict_delta06_dynamic_v0_oracle_v1"
        domain["strict_filters"]["similarity_gt"] = 0.6
        unit = {
            **source_unit,
            "unit_id": f"{cell}_r0",
            "oracle_domain": domain,
            "oracle_protocol": identity(domain),
            "budget": 1000,
        }
        units.append(unit)
    inputs = {path: sha256_file(ROOT / path) for path in MATERIAL}
    inputs[EMPTY_LIBRARY] = sha256_file(ROOT / EMPTY_LIBRARY)
    contract = {
        "schema_version": "t4_dynamic_v0_full_suite_delta06_v1",
        "delta": 0.6,
        "cells": cells,
        "units": units,
        "search_replicates": 1,
        "calls_per_unit": 1000,
        "search_call_ceiling": 15000,
        "confirmation_call_ceiling": 0,
        "total_call_ceiling": 15000,
        "cold_start_seed": 20260913,
        "max_rounds": source["max_rounds"],
        "max_consecutive_empty_rounds": source["max_consecutive_empty_rounds"],
        "plateau_stop": {"enabled": False},
        "library_path": EMPTY_LIBRARY,
        "library_programs": 0,
        "controller": deepcopy(source["controller"]),
        "runtime_input_sha256": deepcopy(source["runtime_input_sha256"]),
        "inputs": inputs,
        "ivg_reported": _ivg_reported(json.loads((ROOT / CENSUS).read_text()), 0.6),
        "container_limit": 15,
        "compute": {
            "worker_max_containers": 15,
            "cpu_per_worker": 1,
            "worker_memory_gib": 4,
            "worker_timeout_seconds": 28800,
            "gpu": False,
            "automatic_retries": 0,
        },
        "information_regime": {
            "development": "historical Dynamic-v0 was developed using public T4 information",
            "runtime_complete_routes": 0,
            "runtime_program_bank_rows": 0,
            "evaluation": "benchmark-trained prospective comparator, not held-out evidence",
        },
        "source_contract": SOURCE_CONTRACT,
        "source_contract_sha256": sha256_file(ROOT / SOURCE_CONTRACT),
        "limitations": [
            "One search replicate per cell is an initial benchmark-wide comparator, not a variance estimate.",
            "The generic module vocabulary reflects benchmark-informed development.",
            "Docking is stochastic and this wave has no confirmation calls.",
        ],
    }
    seal(ROOT / CONTRACT, contract)
    load_contract(ROOT)
    print(json.dumps({"status": "prepared", "cells": 15, "units": 15, "calls": 15000}))


def repair_preflight_contract():
    """Seal the one-time zero-oracle preflight repair without changing science."""

    source_path = ROOT / CONTRACT_V1
    target_path = ROOT / CONTRACT
    failure_path = ROOT / PREFLIGHT_FAILURE_V1
    if (
        target_path.exists()
        or failure_path.exists()
        or (ROOT / PREFLIGHT).exists()
        or LAUNCH.exists()
    ):
        raise ValueError(
            "Dynamic-v0 preflight repair already exists or launch has started"
        )
    if sha256_file(source_path) != SOURCE_CONTRACT_V1_SHA256:
        raise ValueError("Dynamic-v0 v1 contract identity changed")
    source = unseal(source_path)
    if source.get("schema_version") != "t4_dynamic_v0_full_suite_delta06_v1":
        raise ValueError("Dynamic-v0 v1 schema changed")

    failure = {
        "schema_version": "t4_dynamic_v0_full_suite_prequery_failure_v1",
        "status": "failed_zero_oracle_preflight",
        "phase": "structural_preflight_v1",
        "source_contract": CONTRACT_V1,
        "source_contract_sha256": SOURCE_CONTRACT_V1_SHA256,
        "modal_app_id": "ap-tMm3KtHs5HMDRpGwggbioF",
        "code_revision": "cef4fcd91f147d20043b6bc68a52a0c897b3c238",
        "error_type": "ValueError",
        "error": "Dynamic-v0 cold start is nondeterministic: 5ht1b_1_r0",
        "root_cause": (
            "preflight required byte-identical batches across two executions even though "
            "the frozen controller stops proposal batches at a 45-second wall-clock boundary"
        ),
        "scientific_controller_changed": False,
        "new_oracle_calls": 0,
        "query_reservations": [],
    }
    seal(failure_path, failure)

    contract = deepcopy(source)
    contract["schema_version"] = "t4_dynamic_v0_full_suite_delta06_v2"
    contract["preflight_repair"] = {
        "source_contract": CONTRACT_V1,
        "source_contract_sha256": SOURCE_CONTRACT_V1_SHA256,
        "failure_artifact": PREFLIGHT_FAILURE_V1,
        "failure_artifact_sha256": sha256_file(failure_path),
        "removed_assertion": "second wall-clock-bounded execution must have identical batch_id",
        "replacement_gate": "one batch per cell with nonempty attempts and exact candidate replay",
        "scientific_controller_changed": False,
        "oracle_calls_before_repair": 0,
    }
    contract["inputs"] = {path: sha256_file(ROOT / path) for path in MATERIAL}
    contract["inputs"][EMPTY_LIBRARY] = sha256_file(ROOT / EMPTY_LIBRARY)
    contract["inputs"][CONTRACT_V1] = SOURCE_CONTRACT_V1_SHA256
    contract["inputs"][PREFLIGHT_FAILURE_V1] = sha256_file(failure_path)
    seal(target_path, contract)
    load_contract(ROOT)
    print(
        json.dumps(
            {
                "status": "preflight_repair_sealed",
                "contract": CONTRACT,
                "contract_sha256": sha256_file(target_path),
                "failure": PREFLIGHT_FAILURE_V1,
                "new_oracle_calls": 0,
            }
        )
    )


def build_preflight(*, code_revision: str) -> dict:
    contract = load_contract(ROOT)
    results = []
    for unit in contract["units"]:
        cell = contract["cells"][unit["cell"]]
        source = decode_state(cell["source_state"])
        config = configured(contract, unit)
        kwargs = {
            "source_group": identity(
                {
                    "target": unit["target"],
                    "source_idx": unit["source_idx"],
                    "seed": unit["original_seed"],
                }
            ),
            "oracle_protocol": unit["oracle_protocol"],
            "eligibility": benchmark.strict_endpoint_scorer(
                unit["original_seed"], delta=0.6
            ),
        }
        batch = __import__(
            "compose_v4.control.dynamic_program_synthesis", fromlist=["x"]
        ).initial_dynamic_program_batch(source, (), config, **kwargs)
        if not batch["attempts"]:
            raise ValueError(
                f"Dynamic-v0 cold start made no attempts: {unit['unit_id']}"
            )
        for candidate in batch["candidates"]:
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
                raise ValueError(f"Dynamic-v0 replay changed: {unit['unit_id']}")
        results.append(
            {
                "unit": unit["unit_id"],
                "candidates": len(batch["candidates"]),
                "attempts": len(batch["attempts"]),
                "batch_id": batch["batch_id"],
            }
        )
    return {
        "schema_version": "t4_dynamic_v0_full_suite_preflight_v2",
        "passed": len(results) == 15 and all(row["attempts"] > 0 for row in results),
        "contract_sha256": sha256_file(ROOT / CONTRACT),
        "units": results,
        "code_revision": code_revision,
        "new_oracle_calls": 0,
    }


def _task(include_preflight: bool) -> dict:
    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    revision = local_image_revision(
        expected_commit=assert_synced(strict=True)["commit"], repo_root=ROOT
    )
    paths = [CONTRACT, APP, "modal_apps/genmol_t4_opt_app.py"]
    if include_preflight:
        paths.append(PREFLIGHT)
    files = {path: sha256_file(ROOT / path) for path in paths}
    body = {"image_revision": revision, "files_sha256": files}
    return {**body, "run_id": identity(body)}


def remote_preflight():
    import modal

    if (ROOT / PREFLIGHT).exists():
        raise ValueError("Dynamic-v0 preflight exists; preserve it")
    task = _task(False)
    body = modal.Function.from_name(APP_NAME, "structural_preflight").remote(task)
    if body.get("passed") is not True or body.get("new_oracle_calls") != 0:
        raise ValueError("Dynamic-v0 zero-oracle preflight failed")
    publish_json(ROOT / PREFLIGHT, body)
    print(json.dumps({"status": "preflight_passed", "units": len(body["units"])}))


def launch():
    import modal

    if LAUNCH.exists():
        raise ValueError("Dynamic-v0 launch exists; monitor instead")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Dynamic-v0 launch requires clean committed source")
    contract = load_contract(ROOT)
    task = _task(True)
    check = modal.Function.from_name(APP_NAME, "runtime_preflight").remote(task)
    if check.get("passed") is not True or check.get("oracle_calls") != 0:
        raise ValueError("Dynamic-v0 runtime preflight failed")
    worker = modal.Function.from_name(APP_NAME, "worker")
    calls = {
        unit["unit_id"]: worker.spawn({**task, "unit_id": unit["unit_id"]}).object_id
        for unit in contract["units"]
    }
    publish_json(
        LAUNCH,
        {
            "schema_version": "t4_dynamic_v0_full_suite_launch_v1",
            "task": task,
            "call_ids": calls,
            "volume": VOLUME_NAME,
            "volume_path": f"{KIND}/{task['run_id']}",
            "call_ceiling": 15000,
            "automatic_retries": 0,
        },
    )
    print(
        json.dumps(
            {"status": "launched", "workers": len(calls), "call_ids": calls}, indent=2
        )
    )


def _remote_json(volume, path: str) -> dict:
    value = json.loads(b"".join(volume.read_file(path)))
    if set(value) == {"payload", "payload_sha256"}:
        if identity(value["payload"]) != value["payload_sha256"]:
            raise ValueError(f"remote artifact changed: {path}")
        return value["payload"]
    return value


def status():
    import modal

    launch = json.loads(LAUNCH.read_text())
    contract = load_contract(ROOT)
    volume = modal.Volume.from_name(launch["volume"])
    prefix = launch["volume_path"].strip("/")
    rows, total = [], 0
    for unit in contract["units"]:
        folder = f"{prefix}/units/{unit['unit_id']}"
        try:
            names = {row.path.rsplit("/", 1)[-1] for row in volume.listdir(folder)}
        except modal.exception.NotFoundError:
            names = set()
        name = next(
            (
                item
                for item in ("result.json", "failure.json", "progress.json")
                if item in names
            ),
            None,
        )
        value = {} if name is None else _remote_json(volume, f"{folder}/{name}")
        calls = value.get(
            "oracle_calls", value.get("queries_total", value.get("queries", 0))
        )
        best = value.get("best_score")
        if best is None and value.get("champion"):
            best = value["champion"]["score"]
        total += calls
        ivg = contract["ivg_reported"][unit["cell"]]["mean"]
        rows.append(
            {
                "unit": unit["unit_id"],
                "status": value.get("status", "running" if name else "pending"),
                "calls": calls,
                "best_score": best,
                "ivg_mean": ivg,
                "margin_vs_ivg": None if best is None else best - ivg,
                "updated_at_utc": value.get("at", value.get("completed_at_utc")),
            }
        )
    print(json.dumps({"total_calls": total, "units": rows}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "repair-preflight", "remote-preflight", "launch", "status"),
    )
    action = parser.parse_args().action
    {
        "prepare": prepare,
        "repair-preflight": repair_preflight_contract,
        "remote-preflight": remote_preflight,
        "launch": launch,
        "status": status,
    }[action]()


if __name__ == "__main__":
    main()
