"""Launch the sealed T4 cell drivers through the persistent Modal deployment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import modal_apps.t4_shared_controller_completion_v1_app as launcher
from compose_v4.experiments.t4_shared_controller_cell_runtime import (
    mark_driver_running,
    reserve_driver_generation,
)


def _driver(cell_key: str) -> modal.Function:
    safe = cell_key.replace("-", "_")
    return modal.Function.from_name(launcher.APP_NAME, f"driver_{safe}")


def _status(cell_key: str) -> modal.Function:
    safe = cell_key.replace("-", "_")
    return modal.Function.from_name(launcher.APP_NAME, f"status_{safe}")


def _spawn_planned(
    *, receipt_path: Path, receipt: dict, planned: dict[str, dict]
) -> dict:
    """Publish every reservation before spawning any authorized continuation."""

    updated = json.loads(json.dumps(receipt))
    for cell_key, decision in planned.items():
        if decision.get("state") is not None:
            updated["cells"][cell_key]["driver_state"] = decision["state"]
    launcher._replace_launch_receipt(receipt_path, updated)
    for cell_key in updated["launch"]["cell_keys"]:
        decision = planned[cell_key]
        if decision["action"] != "spawn":
            continue
        state = decision["state"]
        call = _driver(cell_key).spawn(
            {
                "launch": updated["launch"],
                "driver_generation": state["generation"],
                "confirmed_prior_call_terminal": state["generation"] > 0,
            }
        )
        updated["cells"][cell_key]["driver_state"] = mark_driver_running(
            state, call.object_id
        )
        launcher._replace_launch_receipt(receipt_path, updated)
    return updated


def launch() -> dict:
    """Publish all reservations, then spawn every driver from the deployment."""

    context = launcher.local_scored_context()
    launch_task = context["launch"]
    receipt_path = launcher._launch_receipt_path(launch_task["run_id"])
    if receipt_path.exists():
        raise FileExistsError("this exact scored payload was already launched")
    planned = {
        cell_key: reserve_driver_generation(
            phase_status="running", existing_state=None
        )
        for cell_key in launch_task["cell_keys"]
    }
    receipt = {
        "schema_version": "t4_shared_controller_scored_launch_receipt_v1",
        "launch": launch_task,
        "cells": {
            cell_key: {
                "volume": launcher.CELL_VOLUMES[cell_key],
                "driver_state": planned[cell_key]["state"],
            }
            for cell_key in launch_task["cell_keys"]
        },
        "driver_count": 9,
        "automatic_retries": 0,
        "replacement": False,
        "backfill": False,
    }
    launcher._publish_launch_receipt(receipt_path, receipt)
    return _spawn_planned(
        receipt_path=receipt_path, receipt=receipt, planned=planned
    )


def resume(run_id: str) -> dict:
    """Resume only terminal drivers, retaining every durable per-cell checkpoint."""

    receipt_path = launcher._launch_receipt_path(run_id)
    receipt, _ = launcher._load_envelope(receipt_path)
    launch_task = receipt["launch"]
    context = launcher.local_scored_context()
    if launch_task["run_id"] != run_id or context["launch"] != launch_task:
        raise ValueError("resume authority differs from the original launch")
    terminal_by_cell: dict[str, bool] = {}
    for cell_key in launch_task["cell_keys"]:
        state = receipt["cells"][cell_key]["driver_state"]
        function_call_id = state.get("function_call_id")
        terminal_by_cell[cell_key] = state["state"] == "terminal" or (
            bool(function_call_id)
            and launcher._modal_call_is_terminal(function_call_id)
        )
    status_rows = {
        cell_key: (
            _status(cell_key).remote({"launch": launch_task})
            if terminal_by_cell[cell_key]
            else {"status": "running"}
        )
        for cell_key in launch_task["cell_keys"]
    }
    phase_status_by_cell = {
        cell_key: (
            "running"
            if row["status"] in {"not_started", "running"}
            else row["status"]
        )
        for cell_key, row in status_rows.items()
    }
    planned = launcher._plan_driver_resume(
        receipt,
        terminal_by_cell=terminal_by_cell,
        phase_status_by_cell=phase_status_by_cell,
    )
    return _spawn_planned(
        receipt_path=receipt_path, receipt=receipt, planned=planned
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("launch", "resume"), default="launch")
    parser.add_argument("--run-id", default="")
    arguments = parser.parse_args()
    if arguments.mode == "resume":
        if not arguments.run_id:
            raise ValueError("--run-id is required for resume")
        result = resume(arguments.run_id)
    else:
        result = launch()
    print(json.dumps(result, indent=2, sort_keys=True))
