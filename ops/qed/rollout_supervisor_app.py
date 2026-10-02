"""Resume the bounded QED rollout driver only after its prior call is terminal."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import modal

APP_NAME = "compose-qed-shared-train16"
INITIAL_CALL_ID = "fc-01M3YT8V2N12749X11ZW6E4N7X"
SOURCE_REVISION = "9c6db0b8dc16a9cbde93b21cba080f0c72c3b648"
ROLL_OUT_COUNT = 1023
MAX_RESUMES = 3
STATE = Path("/artifacts/qed_shared_train16_supervisor_v1.json")
ROLLOUTS = Path("/artifacts/qed_shared_train16_v1")

app = modal.App("compose-qed-shared-train16-supervisor")
volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


def _write_state(state: dict) -> None:
    pending = STATE.with_name(f".{STATE.name}.pending")
    if pending.exists():
        raise FileExistsError(f"QED supervisor has an incomplete state write: {pending}")
    pending.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")
    os.rename(pending, STATE)
    volume.commit()


def _count_rollouts() -> int:
    indices = {int(path.stem.removeprefix("train_")) for path in ROLLOUTS.glob("train_*.json")}
    if any(not 0 <= index < ROLL_OUT_COUNT for index in indices):
        raise ValueError("QED rollout namespace contains an index outside the frozen split")
    return len(indices)


@app.function(
    cpu=1,
    timeout=300,
    max_containers=1,
    schedule=modal.Period(hours=1),
    volumes={"/artifacts": volume},
)
def check_and_resume() -> dict:
    volume.reload()
    count = _count_rollouts()
    now = datetime.now(UTC)
    if STATE.exists():
        state = json.loads(STATE.read_text())
        if (
            state.get("schema_version") != "compose.qed.rollout_supervisor.v1"
            or state.get("source_revision") != SOURCE_REVISION
            or state.get("initial_call_id") != INITIAL_CALL_ID
        ):
            raise ValueError("QED rollout supervisor state has a different identity")
    else:
        state = {
            "schema_version": "compose.qed.rollout_supervisor.v1",
            "source_revision": SOURCE_REVISION,
            "initial_call_id": INITIAL_CALL_ID,
            "active_call_id": INITIAL_CALL_ID,
            "generation": 0,
            "last_launch_count": 0,
            "terminal_observed_at": None,
            "terminal_count": None,
            "status": "running",
            "history": [],
        }
        _write_state(state)
    if count == ROLL_OUT_COUNT:
        if state["status"] != "complete":
            state["status"] = "complete"
            state["history"].append({"event": "complete", "at": now.isoformat(), "count": count})
            _write_state(state)
        return {"status": "complete", "rollouts": count}
    if state["status"] == "launching":
        raise RuntimeError("QED rollout resume intent exists without a call receipt")
    if state["status"] not in ("running", "terminal"):
        raise ValueError("QED rollout supervisor has an invalid status")

    if state["status"] == "running":
        call = modal.FunctionCall.from_id(state["active_call_id"])
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            return {"status": "running", "rollouts": count, "generation": state["generation"]}
        except Exception as error:
            outcome = f"{type(error).__name__}: {error}"[:500]
        else:
            outcome = f"returned: {result!r}"[:500]
        state["status"] = "terminal"
        state["terminal_observed_at"] = now.isoformat()
        state["terminal_count"] = count
        state["history"].append(
            {"event": "driver_terminal", "at": now.isoformat(), "count": count, "outcome": outcome}
        )
        _write_state(state)
        return {"status": "terminal", "rollouts": count, "outcome": outcome}

    observed_at = datetime.fromisoformat(state["terminal_observed_at"])
    if count > state["terminal_count"]:
        state["terminal_count"] = count
        state["terminal_observed_at"] = now.isoformat()
        state["history"].append({"event": "draining", "at": now.isoformat(), "count": count})
        _write_state(state)
        return {"status": "draining", "rollouts": count}
    if now - observed_at < timedelta(minutes=30):
        return {"status": "waiting_for_drain", "rollouts": count}
    if state["generation"] >= MAX_RESUMES:
        raise RuntimeError("QED rollout resume limit reached with an incomplete frozen corpus")
    if count <= state["last_launch_count"]:
        raise RuntimeError("QED rollout driver made no committed progress; refusing another resume")

    state["status"] = "launching"
    state["history"].append({"event": "resume_intent", "at": now.isoformat(), "count": count})
    _write_state(state)
    call = modal.Function.from_name(APP_NAME, "drive").spawn(0, ROLL_OUT_COUNT)
    state["active_call_id"] = call.object_id
    state["generation"] += 1
    state["last_launch_count"] = count
    state["terminal_observed_at"] = None
    state["terminal_count"] = None
    state["status"] = "running"
    state["history"].append(
        {"event": "resume_launched", "at": now.isoformat(), "count": count, "call_id": call.object_id}
    )
    _write_state(state)
    return {"status": "resumed", "rollouts": count, "call_id": call.object_id}


@app.local_entrypoint()
def main() -> None:
    print(check_and_resume.remote(), flush=True)
