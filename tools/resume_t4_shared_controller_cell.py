"""Resume one quiescent shared-controller cell without touching live siblings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import modal
from modal.call_graph import InputStatus

from compose_v4.experiments.t4_shared_controller_cell_runtime import (
    mark_driver_running,
    mark_driver_terminal,
    reserve_driver_generation,
)
from compose_v4.experiments.t4_shared_controller_completion_contract import (
    payload_identity,
)

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "compose-t4-shared-controller-completion-v1"
LAUNCH_ROOT = ROOT / "diagnostics/t4_shared_controller_completion_v1/launches"
TERMINAL = {
    InputStatus.SUCCESS,
    InputStatus.FAILURE,
    InputStatus.INIT_FAILURE,
    InputStatus.TERMINATED,
    InputStatus.TIMEOUT,
}


def _load_envelope(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("payload_sha256") != payload_identity(
        payload
    ):
        raise ValueError(f"invalid self-hashed envelope: {path}")
    return payload


def _replace_envelope(path: Path, payload: dict[str, Any]) -> None:
    envelope = {"payload": payload, "payload_sha256": payload_identity(payload)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _graph_fingerprint(function_call_id: str) -> tuple[tuple[str, int], ...]:
    rows = modal.FunctionCall.from_id(function_call_id).get_call_graph()
    roots = [row for row in rows if row.function_call_id == function_call_id]
    if len(roots) != 1 or roots[0].status not in TERMINAL:
        raise RuntimeError("driver root is not authoritatively terminal")
    if any(row.status == InputStatus.PENDING for row in rows):
        raise RuntimeError("driver graph still contains pending work")
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.status)
        counts[key] = counts.get(key, 0) + 1
    return tuple(sorted(counts.items()))


def resume_cell(
    *,
    run_id: str,
    cell_key: str,
    expected_generation: int,
    expected_calls: int,
    expected_rounds: int,
    polls: int,
    poll_seconds: float,
) -> dict[str, Any]:
    receipt_path = LAUNCH_ROOT / f"{run_id}.json"
    receipt = _load_envelope(receipt_path)
    launch = receipt.get("launch")
    if not isinstance(launch, dict) or launch.get("run_id") != run_id:
        raise ValueError("launch receipt does not bind the requested run")
    cells = receipt.get("cells")
    if not isinstance(cells, dict) or cell_key not in cells:
        raise ValueError("cell is absent from the launch receipt")
    state = cells[cell_key].get("driver_state")
    if not isinstance(state, dict):
        raise TypeError("cell driver state is missing")
    if state.get("state") != "running" or state.get("generation") != expected_generation:
        raise ValueError("cell driver generation/state drift")
    function_call_id = state.get("function_call_id")
    if not isinstance(function_call_id, str) or not function_call_id:
        raise ValueError("cell driver call identity is missing")

    fingerprints = []
    for index in range(polls):
        fingerprints.append(_graph_fingerprint(function_call_id))
        if index + 1 < polls:
            time.sleep(poll_seconds)
    if len(set(fingerprints)) != 1:
        raise RuntimeError("call graph changed during the quiescence guard")

    safe = cell_key.replace("-", "_")
    status = modal.Function.from_name(APP_NAME, f"status_{safe}").remote(
        {"launch": launch}
    )
    expected_status = {
        "cell_key": cell_key,
        "status": "running",
        "calls": expected_calls,
        "rounds": expected_rounds,
    }
    if status != expected_status:
        raise RuntimeError(
            f"durable checkpoint drift: expected {expected_status}, observed {status}"
        )

    terminal_state = mark_driver_terminal(state)
    decision = reserve_driver_generation(
        phase_status="running",
        existing_state=terminal_state,
        confirmed_prior_call_terminal=True,
    )
    if decision.get("action") != "spawn":
        raise RuntimeError(f"continuation was not spawnable: {decision}")
    reserved = decision["state"]
    updated = json.loads(json.dumps(receipt))
    updated["cells"][cell_key]["driver_state"] = reserved
    _replace_envelope(receipt_path, updated)

    call = modal.Function.from_name(APP_NAME, f"driver_{safe}").spawn(
        {
            "launch": launch,
            "driver_generation": reserved["generation"],
            "confirmed_prior_call_terminal": True,
        }
    )
    updated["cells"][cell_key]["driver_state"] = mark_driver_running(
        reserved, call.object_id
    )
    _replace_envelope(receipt_path, updated)
    return {
        "schema_version": "t4_shared_controller_single_cell_resume_v1",
        "run_id": run_id,
        "cell_key": cell_key,
        "prior_function_call_id": function_call_id,
        "prior_generation": expected_generation,
        "new_function_call_id": call.object_id,
        "new_generation": reserved["generation"],
        "checkpoint": status,
        "quiescence_fingerprint": fingerprints[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cell-key", required=True)
    parser.add_argument("--expected-generation", required=True, type=int)
    parser.add_argument("--expected-calls", required=True, type=int)
    parser.add_argument("--expected-rounds", required=True, type=int)
    parser.add_argument("--polls", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    arguments = parser.parse_args()
    if arguments.polls < 2 or arguments.poll_seconds <= 0:
        raise ValueError("quiescence guard requires at least two positive-interval polls")
    result = resume_cell(
        run_id=arguments.run_id,
        cell_key=arguments.cell_key,
        expected_generation=arguments.expected_generation,
        expected_calls=arguments.expected_calls,
        expected_rounds=arguments.expected_rounds,
        polls=arguments.polls,
        poll_seconds=arguments.poll_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
