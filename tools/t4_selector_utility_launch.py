#!/usr/bin/env python3
"""Launch each of the exact two sealed selector docking requests once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json
from compose_v4.experiments.t4_selector_utility_launch import (
    make_run_id,
    publish_once,
    read_sealed,
    sha256_file,
    validate_local_inputs,
)

APP_NAME = "compose-t4-selector-utility"
APP_SOURCE = "modal_apps/t4_selector_utility_launch_app.py"
IMPLEMENTATION_SOURCE = "src/compose_v4/experiments/t4_selector_utility_launch.py"
TOOL_SOURCE = "tools/t4_selector_utility_launch.py"
PREFLIGHT_PATH = ROOT / "diagnostics/t4_selector_utility_launch_preflight.json"
RECEIPT_PATH = ROOT / "diagnostics/t4_selector_utility_launch_spawn.json"


def _common_task() -> tuple[dict, list[dict]]:
    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    preflight = assert_synced(strict=True)
    contract, _lock, requests = validate_local_inputs(ROOT)
    launcher_identity = {
        APP_SOURCE: sha256_file(ROOT / APP_SOURCE),
        IMPLEMENTATION_SOURCE: sha256_file(ROOT / IMPLEMENTATION_SOURCE),
        TOOL_SOURCE: sha256_file(ROOT / TOOL_SOURCE),
    }
    run_id = make_run_id(contract, launcher_identity)
    task = {
        "run_id": run_id,
        "contract_sha256": identity(contract),
        "launcher_identity": launcher_identity,
        "image_revision": local_image_revision(
            expected_commit=preflight["commit"], repo_root=ROOT
        ),
    }
    return task, requests


def launch(*, task: dict | None = None, requests: list[dict] | None = None) -> None:
    if RECEIPT_PATH.exists():
        raise SystemExit(
            f"launch receipt already exists; refusing any respawn: {RECEIPT_PATH}"
        )
    if task is None or requests is None:
        task, requests = _common_task()
    if len(requests) != 2:
        raise SystemExit("exact request count changed before launch")
    if not PREFLIGHT_PATH.is_file():
        raise SystemExit(
            f"required zero-score preflight receipt is missing: {PREFLIGHT_PATH}"
        )
    preflight_receipt = read_sealed(PREFLIGHT_PATH)
    if (
        preflight_receipt.get("schema_version")
        != "t4_selector_utility_preflight_receipt_v1"
        or preflight_receipt.get("task") != task
        or preflight_receipt.get("remote_result", {}).get("status") != "PASS_NO_DOCKING"
        or preflight_receipt.get("remote_result", {}).get("docking_calls") != 0
    ):
        raise SystemExit("required zero-score preflight receipt changed")
    receipt = {
        "schema_version": "t4_selector_utility_spawn_v1",
        "app_name": APP_NAME,
        "function_name": "t4_selector_utility_worker",
        "task": task,
        "preflight_receipt_physical_sha256": sha256_file(PREFLIGHT_PATH),
        "request_lock_physical_sha256": "9802da3b356ae5f41460a29377640c9a0e3fb866f2e82c46ea581624074ba046",
        "request_lock_payload_sha256": "6d49e2a397b662a4425a4a5ee0d17a6223d64c4ac3e0effa594dc8d8c1c52b29",
        "authorized_first_score_call_ceiling": 2,
        "automatic_retries": 0,
        "replacement_or_backfill": False,
        "calls": [],
    }
    function = modal.Function.from_name(APP_NAME, "t4_selector_utility_worker")
    for request in requests:
        call = function.spawn({**task, "request_id": request["request_id"]})
        receipt["calls"].append(
            {
                "request_id": request["request_id"],
                "target": request["target"],
                "call_id": call.object_id,
            }
        )
        publish_json(RECEIPT_PATH, receipt)
        print(
            f"spawned {len(receipt['calls'])}/2 {request['request_id']} "
            f"{request['target']} {call.object_id}",
            flush=True,
        )
    print(f"run_id={task['run_id']}")
    print(f"receipt={RECEIPT_PATH}")
    print("volume_result=/t4_selector_utility_launch/" f"{task['run_id']}/result.json")


def preflight(*, launch_after_pass: bool = False) -> None:
    if PREFLIGHT_PATH.exists():
        raise SystemExit(
            f"preflight receipt already exists; refusing overwrite: {PREFLIGHT_PATH}"
        )
    task, requests = _common_task()
    if len(requests) != 2:
        raise SystemExit("exact request count changed before remote preflight")
    function = modal.Function.from_name(APP_NAME, "t4_selector_utility_preflight")
    call = function.spawn(task)
    print(f"remote preflight call={call.object_id}; zero docking calls", flush=True)
    result = call.get()
    if result.get("status") != "PASS_NO_DOCKING" or result.get("docking_calls") != 0:
        raise SystemExit(f"remote preflight failed closed: {result}")
    receipt = {
        "schema_version": "t4_selector_utility_preflight_receipt_v1",
        "task": task,
        "call_id": call.object_id,
        "remote_result": result,
    }
    publish_once(PREFLIGHT_PATH, receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"receipt={PREFLIGHT_PATH}")
    if launch_after_pass:
        launch(task=task, requests=requests)


def reduce(receipt_path: Path) -> None:
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("authorized_first_score_call_ceiling") != 2
        or len(receipt.get("calls", [])) != 2
        or receipt.get("automatic_retries") != 0
    ):
        raise SystemExit(
            "reduction requires the complete exact two-call launch receipt"
        )
    task = receipt["task"]
    for relative, expected in task["launcher_identity"].items():
        if sha256_file(ROOT / relative) != expected:
            raise SystemExit(f"local launch implementation changed: {relative}")
    function = modal.Function.from_name(APP_NAME, "t4_selector_utility_reduce")
    call = function.spawn(task)
    reduction_receipt = {
        "schema_version": "t4_selector_utility_reduce_spawn_v1",
        "run_id": task["run_id"],
        "call_id": call.object_id,
        "worker_call_ids": [row["call_id"] for row in receipt["calls"]],
        "oracle_calls": 0,
    }
    path = receipt_path.with_name("t4_selector_utility_launch_reduce_spawn.json")
    publish_json(path, reduction_receipt)
    print(f"spawned reducer {call.object_id}; zero docking calls")
    print(f"receipt={path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-and-launch", action="store_true")
    parser.add_argument("--reduce", action="store_true")
    parser.add_argument("--receipt", type=Path, default=RECEIPT_PATH)
    args = parser.parse_args()
    if sum((args.preflight, args.preflight_and_launch, args.reduce)) > 1:
        parser.error("choose only one operational mode")
    if args.preflight:
        preflight()
    elif args.preflight_and_launch:
        preflight(launch_after_pass=True)
    elif args.reduce:
        reduce(args.receipt)
    else:
        launch()


if __name__ == "__main__":
    main()
