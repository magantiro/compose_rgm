#!/usr/bin/env python3
"""Operate the exact authorized four-call topology diagnostic.

This launcher has no resume, retry, replacement, or backfill mode.  A launch
intent is sealed before the single driver spawn, so an ambiguous interruption
fails closed rather than redispatching any query.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
for _path in (str(SOURCE_ROOT), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from compose_v4.experiments.t4_nodistill_topology_four_call_contract import (
    APP_NAME,
    APP_SOURCE,
    CONTRACT_SOURCE,
    DOCKING_ADAPTER_SOURCE,
    EXECUTION_ROOT,
    LAUNCHER_SOURCE,
    RUNTIME_SOURCE,
    SEALER_SOURCE,
    file_sha256,
    validate_execution_contract,
)
from compose_v4.experiments.t4_nodistill_topology_four_call_runtime import (
    capsule_image_revision,
    make_task,
    publish_once,
    read_sealed,
)

LAUNCHER_IDENTITY_SOURCES = (
    APP_SOURCE,
    LAUNCHER_SOURCE,
    SEALER_SOURCE,
    CONTRACT_SOURCE,
    RUNTIME_SOURCE,
    DOCKING_ADAPTER_SOURCE,
)


def _common_task() -> tuple[dict, dict]:
    execution, _identity, scientific = validate_execution_contract(ROOT)
    manifest_path = (
        ROOT / execution["execution_source_capsule"]["manifest_relative_path"]
    )
    manifest = json.loads(manifest_path.read_text())["payload"]
    launcher_identity = {
        relative: file_sha256(ROOT / relative) for relative in LAUNCHER_IDENTITY_SOURCES
    }
    for relative, observed in launcher_identity.items():
        if manifest["files"][relative]["sha256"] != observed:
            raise RuntimeError(
                f"local four-call source differs from exact capsule: {relative}"
            )
    task = make_task(
        ROOT,
        launcher_identity=launcher_identity,
        image_revision=capsule_image_revision(ROOT),
    )
    return task, scientific


def _paths(run_id: str) -> dict[str, Path]:
    root = ROOT / EXECUTION_ROOT
    return {
        "preflight": root / "preflights" / f"{run_id}.json",
        "intent": root / "launch_intents" / f"{run_id}.json",
        "launch": root / "launches" / f"{run_id}.json",
        "reduction": root / "reductions" / f"{run_id}.json",
    }


def preflight() -> tuple[dict, dict]:
    task, scientific = _common_task()
    paths = _paths(task["run_id"])
    if paths["preflight"].exists():
        raise SystemExit(
            f"preflight receipt already exists; refusing overwrite: {paths['preflight']}"
        )
    function = modal.Function.from_name(
        APP_NAME, "t4_nodistill_topology_four_call_preflight"
    )
    call = function.spawn(task)
    result = call.get()
    if (
        result.get("status") != "PASS_ZERO_DOCKING"
        or result.get("docking_calls") != 0
        or result.get("query_ids") != [row["query_id"] for row in scientific["queries"]]
    ):
        raise SystemExit(f"four-call remote preflight failed closed: {result}")
    receipt = {
        "schema_version": "t4_nodistill_topology_four_call_preflight_receipt_v1",
        "task": task,
        "call_id": call.object_id,
        "remote_result": result,
        "docking_calls": 0,
    }
    publish_once(paths["preflight"], receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"receipt={paths['preflight']}")
    return task, result


def launch(*, task: dict | None = None) -> None:
    if task is None:
        task, scientific = _common_task()
    else:
        _, _, scientific = validate_execution_contract(ROOT)
    paths = _paths(task["run_id"])
    if paths["intent"].exists() or paths["launch"].exists():
        raise SystemExit(
            "four-call launch state already exists; retry or redispatch is forbidden"
        )
    if not paths["preflight"].is_file():
        raise SystemExit(
            f"zero-score preflight receipt is missing: {paths['preflight']}"
        )
    preflight_receipt = read_sealed(paths["preflight"])
    if (
        preflight_receipt.get("task") != task
        or preflight_receipt.get("remote_result", {}).get("status")
        != "PASS_ZERO_DOCKING"
        or preflight_receipt.get("docking_calls") != 0
    ):
        raise SystemExit("four-call zero-score preflight receipt changed")
    query_ids = [row["query_id"] for row in scientific["queries"]]
    intent = {
        "schema_version": "t4_nodistill_topology_four_call_local_launch_intent_v1",
        "task": task,
        "query_ids": query_ids,
        "driver_spawns_planned": 1,
        "worker_spawns_planned": 4,
        "authorized_scored_call_ceiling": 4,
        "automatic_retries": 0,
        "replacement_queries": 0,
        "backfill_queries": 0,
        "redispatch_after_intent": False,
        "preflight_receipt_sha256": file_sha256(paths["preflight"]),
    }
    publish_once(paths["intent"], intent)
    function = modal.Function.from_name(
        APP_NAME, "t4_nodistill_topology_four_call_driver"
    )
    call = function.spawn(task)
    receipt = {
        "schema_version": "t4_nodistill_topology_four_call_driver_spawn_v1",
        "task": task,
        "query_ids": query_ids,
        "driver_call_id": call.object_id,
        "launch_intent_sha256": file_sha256(paths["intent"]),
        "authorized_scored_call_ceiling": 4,
        "automatic_retries": 0,
        "replacement_queries": 0,
        "backfill_queries": 0,
    }
    publish_once(paths["launch"], receipt)
    print(f"driver_call_id={call.object_id}")
    print(f"run_id={task['run_id']}")
    print(f"receipt={paths['launch']}")


def reduce(*, finalize_incomplete: bool = False) -> None:
    task, _scientific = _common_task()
    paths = _paths(task["run_id"])
    if not paths["launch"].is_file():
        raise SystemExit(
            f"four-call driver launch receipt is missing: {paths['launch']}"
        )
    launch_receipt = read_sealed(paths["launch"])
    if (
        launch_receipt.get("task") != task
        or launch_receipt.get("authorized_scored_call_ceiling") != 4
        or launch_receipt.get("automatic_retries") != 0
    ):
        raise SystemExit("four-call driver launch receipt changed")
    if paths["reduction"].exists():
        raise SystemExit(
            f"four-call reduction spawn receipt exists: {paths['reduction']}"
        )
    function = modal.Function.from_name(
        APP_NAME, "t4_nodistill_topology_four_call_reduce"
    )
    call = function.spawn(task, finalize_incomplete)
    receipt = {
        "schema_version": "t4_nodistill_topology_four_call_reduce_spawn_v1",
        "task": task,
        "call_id": call.object_id,
        "driver_call_id": launch_receipt["driver_call_id"],
        "finalize_incomplete": finalize_incomplete,
        "docking_calls_by_reducer": 0,
    }
    publish_once(paths["reduction"], receipt)
    print(f"reducer_call_id={call.object_id}")
    print(f"receipt={paths['reduction']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("preflight-and-launch")
    subparsers.add_parser("launch")
    reducer = subparsers.add_parser("reduce")
    reducer.add_argument("--finalize-incomplete", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "preflight":
        preflight()
    elif arguments.command == "preflight-and-launch":
        task, _ = preflight()
        launch(task=task)
    elif arguments.command == "launch":
        launch()
    else:
        reduce(finalize_incomplete=arguments.finalize_incomplete)


if __name__ == "__main__":
    main()
