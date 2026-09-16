#!/usr/bin/env python3
"""Preflight and launch the exact hybrid Stage-20 workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.t4_hybrid_top3_v0_stage20 import (
    CONTRACT_PATH,
    STAGE_CEILING,
    load_contract,
    run_identity,
)
from modal_apps.run_process_v2_p50_app import local_image_revision

APP_NAME = "compose-t4-hybrid-top3-v0-stage20"
APP_SOURCE = "modal_apps/t4_hybrid_top3_v0_stage20_app.py"
EXPERIMENT_SOURCE = "src/compose_v4/experiments/t4_hybrid_top3_v0_stage20.py"
TOOL_SOURCE = "tools/t4_hybrid_top3_v0_stage20.py"
TEST_SOURCE = "tests/test_t4_hybrid_top3_v0_stage20.py"
DOC_SOURCE = "docs/T4_HYBRID_TOP3_V0_STAGE20_LAUNCH.md"
RECEIPT = ROOT / "diagnostics/t4_hybrid_top3_v0_stage20/spawn.json"
PREFLIGHT_RECEIPT = Path("/tmp/compose-t4-hybrid-top3-v0-stage20/preflight.json")


def _revision() -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    return local_image_revision(expected_commit=commit, repo_root=ROOT)


def material_files(contract: dict) -> list[str]:
    values = [
        CONTRACT_PATH,
        APP_SOURCE,
        EXPERIMENT_SOURCE,
        TOOL_SOURCE,
        TEST_SOURCE,
        DOC_SOURCE,
    ]
    values.extend(
        spec["path"]
        for spec in contract["launch"]["immutable_inputs"].values()
        if isinstance(spec, dict) and "path" in spec
    )
    values.extend(
        [
            "src/compose_v4/control/adaptive_program_optimizer.py",
            "src/compose_v4/control/dynamic_program_synthesis.py",
            "src/compose_v4/control/edit_program.py",
            "src/compose_v4/control/edit_program_graph.py",
            "src/compose_v4/control/macro_archive_integration.py",
            "src/compose_v4/experiments/t4_frozen_program_benchmark.py",
            "modal_apps/genmol_t4_opt_app.py",
        ]
    )
    return sorted(set(values))


def build_task() -> dict:
    contract = load_contract(ROOT)
    image_revision = _revision()
    files = {
        relative: sha256_file(ROOT / relative) for relative in material_files(contract)
    }
    run_id = run_identity(contract)
    core = {
        "run_id": run_id,
        "stage_call_ceiling": STAGE_CEILING,
        "image_revision": image_revision,
        "files_sha256": files,
    }
    return {**core, "stage_id": identity(core)}


def show_task() -> None:
    task = build_task()
    contract = load_contract(ROOT)
    print(
        json.dumps(
            {
                "task": task,
                "contract_physical_sha256": sha256_file(ROOT / CONTRACT_PATH),
                "contract_payload_sha256": identity(contract["launch"]),
                "candidate_lock": contract["launch"]["immutable_inputs"][
                    "candidate_lock"
                ],
                "admission_lock": contract["launch"]["immutable_inputs"][
                    "admission_lock"
                ],
                "workers": 5,
                "maximum_total_charged_calls": 100,
                "scored_calls_per_cell": 20,
                "automatic_retries": 0,
                "replacement_or_backfill": False,
                "confirmation_calls": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


def preflight() -> None:
    if PREFLIGHT_RECEIPT.exists():
        raise SystemExit(f"preflight receipt exists: {PREFLIGHT_RECEIPT}")
    task = build_task()
    call = modal.Function.from_name(APP_NAME, "preflight").spawn(task)
    result = call.get()
    if result.get("status") != "PASS_NO_DOCKING" or result.get("docking_calls") != 0:
        raise SystemExit(f"remote preflight failed closed: {result}")
    publish_json(
        PREFLIGHT_RECEIPT,
        {
            "schema_version": "t4_hybrid_top3_v0_stage20_preflight_receipt_v1",
            "task": task,
            "call_id": call.object_id,
            "result": result,
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def launch(*, authorized_revision: str, authorized_contract_sha256: str) -> None:
    if RECEIPT.exists():
        raise SystemExit(f"spawn receipt exists; refusing respawn: {RECEIPT}")
    task = build_task()
    if task["image_revision"]["commit"] != authorized_revision:
        raise SystemExit("exact authorized launch revision differs from clean HEAD")
    if sha256_file(ROOT / CONTRACT_PATH) != authorized_contract_sha256:
        raise SystemExit("exact authorized contract physical hash differs")
    if not PREFLIGHT_RECEIPT.exists():
        raise SystemExit("remote zero-docking preflight receipt is required")
    preflight_receipt = json.loads(PREFLIGHT_RECEIPT.read_text())
    if preflight_receipt.get("task") != task:
        raise SystemExit("remote preflight was not run on this exact task")
    receipt = {
        "schema_version": "t4_hybrid_top3_v0_stage20_spawn_v1",
        "app_name": APP_NAME,
        "function": "worker",
        "task": task,
        "authorized_revision": authorized_revision,
        "authorized_contract_physical_sha256": authorized_contract_sha256,
        "maximum_workers": 5,
        "maximum_calls_per_worker": 20,
        "maximum_total_charged_calls": 100,
        "automatic_retries": 0,
        "replacement_or_backfill": False,
        "confirmation_calls": 0,
        "calls": [],
    }
    function = modal.Function.from_name(APP_NAME, "worker")
    for unit_id in load_contract(ROOT)["launch"]["units"]:
        call = function.spawn({**task, "unit_id": unit_id})
        receipt["calls"].append({"unit_id": unit_id, "call_id": call.object_id})
        publish_json(RECEIPT, receipt)
        print(f"spawned {unit_id}: {call.object_id}", flush=True)
    print(f"run_id={task['run_id']}")
    print(f"volume=/t4_hybrid_top3_v0_stage20/{task['run_id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-task", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--authorized-revision")
    parser.add_argument("--authorized-contract-sha256")
    args = parser.parse_args()
    if sum((args.show_task, args.preflight, args.launch)) != 1:
        parser.error("choose exactly one mode")
    if args.show_task:
        show_task()
    elif args.preflight:
        preflight()
    else:
        if not args.authorized_revision or not args.authorized_contract_sha256:
            parser.error("--launch requires both exact authorization values")
        launch(
            authorized_revision=args.authorized_revision,
            authorized_contract_sha256=args.authorized_contract_sha256,
        )


if __name__ == "__main__":
    main()
