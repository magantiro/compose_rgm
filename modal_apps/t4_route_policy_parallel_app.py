"""Three-fold zero-oracle T4 route-policy comparison on Modal."""

from pathlib import Path

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    _validate_remote_revision,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as base_image

APP_NAME = "compose-t4-route-policy-parallel"
OUTPUT_ROOT = ARTIFACT_ROOT / "t4_route_policy_comparison_parallel" / "attempt_1"
MATERIAL_FILES = (
    "modal_apps/t4_route_policy_parallel_app.py",
    "tools/t4_route_policy_parallel.py",
    "tools/t4_route_policy_comparison.py",
    "tools/t4_program_vocabulary_audit.py",
    "tools/t4_route_distillation.py",
    "docs/GENMOL_T4_SEEDS.json",
    "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json",
)

image = base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
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

app = modal.App(APP_NAME)


def _validate_task(task):
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import verify_file

    body = {key: value for key, value in task.items() if key != "run_id"}
    if identity(body) != task.get("run_id"):
        raise ValueError("parallel T4 fold task identity changed")
    if task.get("fold") not in (0, 1, 2):
        raise ValueError("parallel T4 fold must be 0, 1 or 2")
    if task.get("oracle_calls") != 0 or task.get("automatic_retry") != 0:
        raise ValueError("parallel T4 fold must remain zero-oracle without retry")
    expected = str(OUTPUT_ROOT / f"fold_{task['fold']}")
    if task.get("output") != expected:
        raise ValueError("parallel T4 fold output changed")
    if set(task.get("files_sha256", ())) != set(MATERIAL_FILES):
        raise ValueError("parallel T4 material-file inventory changed")
    for relative, digest in task["files_sha256"].items():
        verify_file(REMOTE_ROOT / relative, digest)
    _validate_remote_revision(task["image_revision"])


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=(4096, 4096),
    timeout=4 * 3600,
    retries=0,
    max_containers=3,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def worker(task):
    from tools.t4_route_policy_comparison import run

    _validate_task(task)
    result = run(
        Path(task["output"]),
        fold_ids=(task["fold"],),
        code_revision=task["image_revision"]["commit"],
        working_tree_dirty=False,
    )
    artifact_volume.commit()
    return {
        "fold": task["fold"],
        "decision": result["decision"],
        "costs": result["costs"],
        "output": task["output"],
    }
