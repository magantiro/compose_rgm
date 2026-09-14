"""Pinned zero-oracle fitting app for T4 route-distilled Dynamic COMPOSE."""

import json
from pathlib import Path

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    _validate_remote_revision,
    artifact_volume,
    local_image_revision,
)
from modal_apps.run_process_v2_p50_app import image as base_image

APP_NAME = "compose-t4-route-distillation"
OUTPUT = ARTIFACT_ROOT / "t4_route_distillation" / "attempt_1"
LOCAL_FILES = (
    "configs/t4_frozen_program_benchmark_v2.json",
    "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json",
    "docs/GENMOL_T4_SEEDS.json",
    "tools/t4_program_vocabulary_audit.py",
    "tools/t4_route_distillation.py",
)

image = base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
for relative in LOCAL_FILES:
    image = image.add_local_file(
        ROOT / relative, str(REMOTE_ROOT / relative), copy=True
    )

app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=(4096, 4096),
    timeout=1800,
    retries=0,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def fit(task):
    from tools.t4_route_distillation import run

    _validate_remote_revision(task["image_revision"])
    if task["output"] != str(OUTPUT):
        raise ValueError("route-distillation output path changed")
    result = run(
        Path(task["output"]),
        code_revision=task["image_revision"]["commit"],
        working_tree_dirty=False,
    )
    artifact_volume.commit()
    return {
        "decision": result["decision"],
        "gates": result["gates"],
        "costs": result["costs"],
        "output": task["output"],
    }


@app.local_entrypoint()
def main():
    import subprocess

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if status:
        raise ValueError(f"route-distillation launch requires clean source: {status}")
    task = {
        "image_revision": local_image_revision(expected_commit=commit),
        "output": str(OUTPUT),
    }
    print(json.dumps(fit.remote(task), indent=2))
