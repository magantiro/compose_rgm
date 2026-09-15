"""Fifteen-way zero-oracle T4 structural-subgoal audit on Modal."""

from pathlib import Path

import modal

from modal_apps.run_process_v2_p50_app import (
    REMOTE_ROOT,
    ROOT,
    _validate_remote_revision,
)
from modal_apps.run_process_v2_p50_app import image as base_image

APP_NAME = "compose-t4-structural-subgoal"
VOLUME_NAME = "compose-t4-route-distilled-artifacts"
ARTIFACT_ROOT = Path("/artifacts")
OUTPUT_ROOT = ARTIFACT_ROOT / "t4_structural_subgoal" / "attempt_2"
MATERIAL_FILES = (
    "AGENTS.md",
    "docs/T4_STRUCTURAL_SUBGOAL_POLICY.md",
    "modal_apps/t4_structural_subgoal_app.py",
    "tools/t4_structural_subgoal_audit.py",
    "tools/t4_program_vocabulary_audit.py",
    "tools/t4_route_distillation.py",
    "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json",
)

image = base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "PYTHONUNBUFFERED": "1"}
)
for relative in MATERIAL_FILES:
    if relative.startswith(("configs/", "src/")):
        continue
    image = image.add_local_file(
        ROOT / relative, str(REMOTE_ROOT / relative), copy=True
    )
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
        raise ValueError("structural-subgoal task identity changed")
    if task.get("oracle_calls") != 0 or task.get("automatic_retry") != 0:
        raise ValueError(
            "structural-subgoal task must remain zero-oracle without retry"
        )
    if task.get("output") != str(OUTPUT_ROOT / task["source_group"] / "result.json"):
        raise ValueError("structural-subgoal output path changed")
    if set(task.get("files_sha256", ())) != set(MATERIAL_FILES):
        raise ValueError("structural-subgoal material-file inventory changed")
    for relative, digest in task["files_sha256"].items():
        verify_file(REMOTE_ROOT / relative, digest)
    _validate_remote_revision(task["image_revision"])


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=(4096, 4096),
    timeout=30 * 60,
    retries=0,
    max_containers=15,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def worker(task):
    import json
    from datetime import datetime, timezone

    from tools.t4_structural_subgoal_audit import run_source

    _validate_task(task)
    print(
        json.dumps(
            {
                "event": "structural_subgoal_source_started",
                "source_group": task["source_group"],
                "at_utc": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    result = run_source(
        Path(task["output"]),
        source_group=task["source_group"],
        code_revision=task["image_revision"]["commit"],
        working_tree_dirty=False,
    )
    artifact_volume.commit()
    print(
        json.dumps(
            {
                "event": "structural_subgoal_source_complete",
                "source_group": task["source_group"],
                "routes": result["census"]["routes"],
                "gates": result["gates"],
                "at_utc": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {
        "source_group": task["source_group"],
        "routes": result["census"]["routes"],
        "gates": result["gates"],
        "output": task["output"],
    }
