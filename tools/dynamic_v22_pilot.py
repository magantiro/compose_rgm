#!/usr/bin/env python3
"""Zero-oracle preflight and explicitly locked PMO Dynamic-v2.2 pilot CLI."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from dataclasses import replace
from pathlib import Path

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v22 import (
    DynamicV22PolicyConfig,
    DynamicV22ProgramOptimizer,
    advance_t4_feasibility_frontier,
    initial_frontier_state,
)
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.dynamic_v22_pilot import (
    PMO_TASKS,
    QUERY_BUDGET,
    T4_CELLS,
    quick_pilot_contract,
    run_pmo_pilot_campaign,
)
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)
from compose_v4.experiments.pmo_dynamic_v21 import (
    INITIALIZATION,
    ORACLE_ASSET_ROOT,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "configs/dynamic_v22_quick_pilot_v1.json"
T4_SOURCE_CONTRACT = "configs/t4_dynamic_v21_development_v1.json"
T4_APP = "modal_apps/t4_dynamic_v22_pilot_app.py"
T4_APP_NAME = "compose-t4-dynamic-v22-pilot"
PREFLIGHT = "diagnostics/dynamic_v22_pilot/preflight.json"
LAUNCH = "diagnostics/dynamic_v22_pilot/launch.json"
PMO_RUN_ROOT = "diagnostics/dynamic_v22_pilot/runs"
MATERIAL_FILES = (
    CONTRACT,
    "modal_apps/genmol_t4_opt_app.py",
    T4_APP,
    "tools/dynamic_v22_pilot.py",
    "src/compose_v4/control/dynamic_program_synthesis_v22.py",
    "src/compose_v4/experiments/dynamic_v22_pilot.py",
)


def configuration(*, seed=20260914, kind):
    return replace(
        ProgramSearchConfig.program_only_recipe(
            seed=seed, score_direction="minimize" if kind == "t4" else "maximize"
        ),
        proposal_cache_entries=128,
        candidates_per_batch=16,
        attempts_per_batch=128,
        wall_seconds=45.0,
    )


def load_contract(root=ROOT):
    observed = json.loads((root / CONTRACT).read_text())
    expected = quick_pilot_contract()
    if observed != expected or identity(observed["payload"]) != observed["contract_sha256"]:
        raise ValueError("Dynamic-v2.2 quick-pilot contract changed")
    return observed


def build_preflight(root=ROOT):
    contract = load_contract(root)
    source = production_state_from_smiles("C1CCCCC1CCO", max_atoms=48)
    calls = 0

    def ineligible(row):
        nonlocal calls
        return {
            **row,
            "qed": 0.55,
            "sa": 2.0,
            "sim": 0.7,
            "oracle_eligible": False,
            "endpoint_exclusion_reasons": ["qed"],
        }

    policy = DynamicV22PolicyConfig.for_t4()
    first = advance_t4_feasibility_frontier(
        source,
        config=policy,
        state=initial_frontier_state(seed=79),
        source_group="preflight",
        oracle_protocol="preflight:t4",
        eligibility=ineligible,
        delta=0.4,
        attempts_per_channel=4,
    )
    second = advance_t4_feasibility_frontier(
        source,
        config=policy,
        state=first["frontier_state"],
        source_group="preflight",
        oracle_protocol="preflight:t4",
        eligibility=ineligible,
        delta=0.4,
        attempts_per_channel=4,
    )
    pmo_optimizer = DynamicV22ProgramOptimizer(
        configuration(seed=83, kind="pmo"),
        source_group="preflight",
        oracle_protocol="preflight:pmo",
        hierarchy=None,
        policy_config=DynamicV22PolicyConfig.for_pmo(),
    )
    restored = DynamicV22ProgramOptimizer.restore(pmo_optimizer.snapshot())
    body = {
        "schema_version": "dynamic_v22_zero_oracle_preflight_v1",
        "passed": (
            calls == 0
            and first["new_oracle_calls"] == second["new_oracle_calls"] == 0
            and second["frontier_state"]["wave"] == 2
            and first["frontier_state"]["state_sha256"] != second["frontier_state"]["state_sha256"]
            and restored.policy_config == DynamicV22PolicyConfig.for_pmo()
        ),
        "contract_sha256": sha256_file(root / CONTRACT),
        "contract_identity": contract["contract_sha256"],
        "frontier_wave_ids": [
            first["frontier_state"]["state_sha256"],
            second["frontier_state"]["state_sha256"],
        ],
        "frontier_attempt_keys": len(second["frontier_state"]["attempt_keys"]),
        "snapshot_id": restored.snapshot()["snapshot_id"],
        "runtime_comparator_or_winner_inputs": [],
        "new_oracle_calls": calls,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "material_inputs_sha256": {
            path: sha256_file(root / path) for path in MATERIAL_FILES
        },
    }
    return {**body, "preflight_sha256": identity(body)}


def _load_launch_lock(path: Path, *, task: str, root=ROOT):
    artifact = json.loads(path.read_text())
    if "launch_lock" in artifact:
        receipt_body = {
            key: value for key, value in artifact.items() if key != "receipt_sha256"
        }
        if identity(receipt_body) != artifact.get("receipt_sha256"):
            raise ValueError("Dynamic-v2.2 launch receipt changed")
        lock = artifact["launch_lock"]
    else:
        lock = artifact
    body = {key: value for key, value in lock.items() if key != "launch_id"}
    if (
        identity(body) != lock.get("launch_id")
        or lock.get("authorized") is not True
        or lock.get("contract_sha256") != sha256_file(root / CONTRACT)
        or lock.get("query_budget") != QUERY_BUDGET
        or task not in lock.get("pmo_tasks", ())
        or lock.get("automatic_retry") != 0
        or lock.get("material_inputs_sha256")
        != {item: sha256_file(root / item) for item in MATERIAL_FILES}
    ):
        raise ValueError("PMO Dynamic-v2.2 launch lock is absent or incompatible")
    return lock


def _load_initialization(root=ROOT):
    initialized = json.loads((root / INITIALIZATION).read_text())
    body = {key: value for key, value in initialized.items() if key != "lock_sha256"}
    if identity(body) != initialized.get("lock_sha256"):
        raise ValueError("PMO initialization lock changed")
    if any("score" in row or "task" in row for row in initialized["candidates"]):
        raise ValueError("PMO initialization contains task information")
    return initialized


def run_pmo(task, output, launch_lock, root=ROOT):
    if task not in PMO_TASKS:
        raise ValueError("task is outside the two-task v2.2 PMO pilot")
    load_contract(root)
    _load_launch_lock(Path(launch_lock), task=task, root=root)
    initialized = _load_initialization(root)
    from tdc import Oracle

    previous = Path.cwd()
    os.chdir(root / ORACLE_ASSET_ROOT)
    try:
        oracle = Oracle(name=task)
        protocol = identity(
            {
                "task": task,
                "implementation": "native PyTDC Oracle",
                "direction": "maximize",
                "range": [0, 1],
                "prescreen": False,
                "calls_include_initialization": True,
            }
        )
        return run_pmo_pilot_campaign(
            output=output,
            task_name=task,
            oracle_protocol=protocol,
            config=configuration(kind="pmo"),
            initialization=initialized,
            evaluate=lambda smiles: float(oracle(smiles)),
            rounds=16,
        )
    finally:
        os.chdir(previous)


def _git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _launch_body(root=ROOT):
    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Dynamic-v2.2 launch requires clean committed source")
    revision = local_image_revision(
        expected_commit=assert_synced(strict=True)["commit"], repo_root=root
    )
    body = {
        "schema_version": "dynamic_v22_five_unit_launch_v1",
        "authorized": True,
        "contract_sha256": sha256_file(root / CONTRACT),
        "query_budget": QUERY_BUDGET,
        "t4_cells": list(T4_CELLS),
        "pmo_tasks": list(PMO_TASKS),
        "controller_seed": 20260913,
        "automatic_retry": 0,
        "confirmation_calls": 0,
        "max_concurrent_single_cpu_workers": 5,
        "image_revision": revision,
        "material_inputs_sha256": {
            path: sha256_file(root / path) for path in MATERIAL_FILES
        },
        "runtime_comparator_or_winner_inputs": [],
    }
    return {**body, "launch_id": identity(body)}


def _t4_tasks(lock, root=ROOT):
    from compose_v4.experiments.t4_matched_pilot import unseal

    source = unseal(root / T4_SOURCE_CONTRACT)
    units = {row["cell"]: row for row in source["units"] if row["replicate"] == 0}
    files = {
        path: lock["material_inputs_sha256"][path]
        for path in (
            CONTRACT,
            "modal_apps/genmol_t4_opt_app.py",
            T4_APP,
            "tools/dynamic_v22_pilot.py",
        )
    }
    tasks = []
    for cell in T4_CELLS:
        unit = units[cell]
        cell_record = source["cells"][cell]
        body = {
            "schema_version": "dynamic_v22_t4_unit_lock_v1",
            "authorized": True,
            "contract_sha256": lock["contract_sha256"],
            "query_budget": QUERY_BUDGET,
            "automatic_retry": 0,
            "cell": cell,
            "target": unit["target"],
            "source_idx": unit["source_idx"],
            "source_state": cell_record["source_state"],
            "source_state_sha256": cell_record["source_state_sha256"],
            "original_seed": unit["original_seed"],
            "oracle_protocol": unit["oracle_protocol"],
            "qvina02_sha256": unit["oracle_domain"]["qvina02_sha256"],
            "receptor_sha256": unit["oracle_domain"]["receptor_sha256"],
            "controller_seed": unit["controller_seed"],
            "docking_seed": unit["docking_seed"],
            "image_revision": lock["image_revision"],
            "files_sha256": files,
        }
        tasks.append({**body, "run_id": identity(body)})
    return tasks


def launch():
    import modal

    load_contract(ROOT)
    destination = ROOT / LAUNCH
    if destination.exists():
        raise ValueError("Dynamic-v2.2 launch exists; monitor rather than relaunch")
    lock = _launch_body()
    remote_preflight = modal.Function.from_name(T4_APP_NAME, "preflight").remote(
        {"image_revision": lock["image_revision"]}
    )
    if remote_preflight.get("passed") is not True or remote_preflight.get(
        "new_oracle_calls"
    ) != 0:
        raise RuntimeError("Dynamic-v2.2 remote zero-oracle preflight failed")
    publish_json(ROOT / PREFLIGHT, remote_preflight)
    worker = modal.Function.from_name(T4_APP_NAME, "worker")
    tasks = _t4_tasks(lock)
    calls = {task["cell"]: worker.spawn(task).object_id for task in tasks}
    receipt_body = {
        "schema_version": "dynamic_v22_five_unit_launch_receipt_v1",
        "launch_lock": lock,
        "t4_tasks": tasks,
        "t4_function_call_ids": calls,
        "t4_volume": "compose-v4-artifacts",
        "pmo_output_root": f"{PMO_RUN_ROOT}/{lock['launch_id']}/pmo",
        "preflight_sha256": sha256_file(ROOT / PREFLIGHT),
    }
    receipt = {**receipt_body, "receipt_sha256": identity(receipt_body)}
    publish_json(destination, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def _remote_json(volume, path):
    return json.loads(b"".join(volume.read_file(path)))


def status():
    import modal

    receipt = json.loads((ROOT / LAUNCH).read_text())
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if identity(body) != receipt.get("receipt_sha256"):
        raise ValueError("Dynamic-v2.2 launch receipt changed")
    volume = modal.Volume.from_name(receipt["t4_volume"])
    rows = []
    lock = receipt["launch_lock"]
    for task in receipt["t4_tasks"]:
        folder = f"dynamic_v22_pilot/{task['run_id']}/{task['cell']}"
        result = None
        try:
            result = _remote_json(volume, f"{folder}/result.json")
        except modal.exception.NotFoundError:
            pass
        if result is not None:
            rows.append(
                {
                    "cell": task["cell"],
                    "status": result["status"],
                    "calls": result["charged_oracle_calls"],
                    "best": result["best_score"],
                    "frontier_wave": result["frontier_state"]["wave"],
                }
            )
            continue
        try:
            names = sorted(
                row.path.rsplit("/", 1)[-1]
                for row in volume.listdir(folder)
                if row.path.rsplit("/", 1)[-1].startswith("round_")
            )
        except modal.exception.NotFoundError:
            names = []
        current = {"cell": task["cell"], "status": "pending", "calls": 0, "best": None}
        for name in reversed(names):
            try:
                saved = _remote_json(volume, f"{folder}/{name}/complete.json")
            except modal.exception.NotFoundError:
                continue
            summary = saved["summary"]
            current = {
                "cell": task["cell"],
                "status": "running",
                "calls": summary["queries_total"],
                "best": summary["best_score"],
                "frontier_wave": saved["frontier_state"]["wave"],
            }
            break
        rows.append(current)
    pmo = []
    pmo_root = ROOT / receipt["pmo_output_root"]
    for task in lock["pmo_tasks"]:
        path = pmo_root / task / "result.json"
        if path.exists():
            row = json.loads(path.read_text())
            pmo.append(
                {
                    "task": task,
                    "status": row["status"],
                    "calls": row["charged_oracle_calls"],
                    "best": row["best_score"],
                    "top10": row["final_top10"],
                    "auc_top10_256": row["auc_top10_256"],
                }
            )
        else:
            oracle = pmo_root / task / "oracle"
            calls = len(list(oracle.glob("query_*/result.json"))) if oracle.exists() else 0
            pmo.append({"task": task, "status": "running" if calls else "pending", "calls": calls})
    print(json.dumps({"t4": rows, "pmo": pmo}, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show-contract")
    subparsers.add_parser("launch")
    subparsers.add_parser("status")
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--output")
    pmo = subparsers.add_parser("run-pmo")
    pmo.add_argument("--task", choices=PMO_TASKS, required=True)
    pmo.add_argument("--output", required=True)
    pmo.add_argument("--launch-lock", required=True)
    args = parser.parse_args()
    if args.command == "show-contract":
        result = load_contract()
    elif args.command == "preflight":
        result = build_preflight()
        if args.output:
            publish_json(Path(args.output), result)
    elif args.command == "run-pmo":
        result = run_pmo(args.task, Path(args.output), Path(args.launch_lock))
    elif args.command == "launch":
        return launch()
    else:
        return status()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
