"""Fifteen-way zero-oracle structural-subgoal conditional-realizer gate."""

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

from modal_apps.run_process_v2_p50_app import REMOTE_ROOT, ROOT, _validate_remote_revision
from modal_apps.run_process_v2_p50_app import image as base_image

APP_NAME = "compose-t4-structural-subgoal-realizer"
VOLUME_NAME = "compose-t4-route-distilled-artifacts"
ARTIFACT_ROOT = Path("/artifacts")
OUTPUT_ROOT = ARTIFACT_ROOT / "t4_structural_subgoal_realizer" / "attempt_1"
MATERIAL_FILES = (
    "AGENTS.md",
    "docs/T4_STRUCTURAL_SUBGOAL_POLICY.md",
    "modal_apps/t4_structural_subgoal_realizer_app.py",
    "tools/t4_structural_subgoal_realizer_audit.py",
    "tools/t4_structural_subgoal_audit.py",
    "tools/t4_program_vocabulary_audit.py",
    "tools/t4_route_distillation.py",
    "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json",
)

image = base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "PYTHONUNBUFFERED": "1"})
for relative in MATERIAL_FILES:
    if relative.startswith(("configs/", "src/")):
        continue
    image = image.add_local_file(ROOT / relative, str(REMOTE_ROOT / relative), copy=True)
image = image.add_local_dir(
    ROOT / "diagnostics/ivg_winner_paths/pairs",
    str(REMOTE_ROOT / "diagnostics/ivg_winner_paths/pairs"),
    copy=True,
)

artifact_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _validate_task(task):
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import verify_file

    body = {key: value for key, value in task.items() if key != "run_id"}
    if identity(body) != task.get("run_id"):
        raise ValueError("structural-realizer task identity changed")
    if task.get("oracle_calls") != 0 or task.get("automatic_retry") != 0:
        raise ValueError("structural-realizer task must remain zero-oracle without retry")
    expected = str(OUTPUT_ROOT / task["source_group"] / "result.json")
    if task.get("output") != expected:
        raise ValueError("structural-realizer output path changed")
    if set(task.get("files_sha256", ())) != set(MATERIAL_FILES):
        raise ValueError("structural-realizer material-file inventory changed")
    for relative, digest in task["files_sha256"].items():
        verify_file(REMOTE_ROOT / relative, digest)
    _validate_remote_revision(task["image_revision"])


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=(4096, 4096),
    timeout=60 * 60,
    retries=0,
    max_containers=15,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def worker(task):
    from compose_v4.experiments.continuation_profile import publish_json
    from tools.t4_structural_subgoal_realizer_audit import run_source

    _validate_task(task)
    progress_path = OUTPUT_ROOT / task["source_group"] / "progress.json"
    progress_lock = threading.Lock()
    state_lock = threading.Lock()
    progress_state = {
        "payload": {
            "schema_version": "t4_structural_subgoal_realizer_progress_v1",
            "event": "structural_subgoal_realizer_heartbeat",
            "source_group": task["source_group"],
            "phase": "loading_inputs",
            "current_program_id": None,
            "routes_completed": 0,
            "routes_total": None,
            "elapsed_seconds": 0.0,
            "completed_routes_per_second": 0.0,
            "search_expansions": 0,
            "action_attempts": 0,
            "expansions_per_second": 0.0,
            "current_route": {},
            "status_counts": {},
            "estimated_remaining_seconds_from_completed_routes": None,
            "estimated_remaining_seconds_to_configured_search_limit": None,
            "eta_is_operational_not_a_stopping_rule": True,
            "at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "at": time.monotonic(),
    }
    stopped = threading.Event()

    def persist(payload):
        with progress_lock:
            print(json.dumps(payload, sort_keys=True), flush=True)
            publish_json(progress_path, {**payload, "run_id": task["run_id"]})
            artifact_volume.commit()

    def publish_progress(payload):
        with state_lock:
            progress_state["payload"] = dict(payload)
            progress_state["at"] = time.monotonic()
        persist(payload)

    def heartbeat_loop():
        while not stopped.wait(30.0):
            now = time.monotonic()
            with state_lock:
                payload = dict(progress_state["payload"])
                delta = max(0.0, now - progress_state["at"])
            payload["elapsed_seconds"] = float(payload.get("elapsed_seconds", 0.0)) + delta
            for field in (
                "estimated_remaining_seconds_from_completed_routes",
                "estimated_remaining_seconds_to_configured_search_limit",
            ):
                if payload.get(field) is not None:
                    payload[field] = max(0.0, float(payload[field]) - delta)
            payload["heartbeat_repeat"] = True
            payload["at_utc"] = datetime.now(timezone.utc).isoformat()
            persist(payload)

    persist(progress_state["payload"])
    heartbeat = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat.start()
    try:
        result = run_source(
            Path(task["output"]),
            source_group=task["source_group"],
            code_revision=task["image_revision"]["commit"],
            working_tree_dirty=False,
            progress_publisher=publish_progress,
            heartbeat_seconds=30.0,
        )
    finally:
        stopped.set()
        heartbeat.join(timeout=1.0)
    artifact_volume.commit()
    return {
        "source_group": task["source_group"],
        "gates": result["gates"],
        "timing": result["timing"],
        "output": task["output"],
    }
