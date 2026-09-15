"""Zero-oracle regeneration and locking for the T4 route-policy utility panel."""

from pathlib import Path

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    _validate_remote_revision,
)
from modal_apps.run_process_v2_p50_app import image as base_image

APP_NAME = "compose-t4-route-candidate-utility"
VOLUME_NAME = "compose-t4-route-distilled-artifacts"
OUTPUT_ROOT = ARTIFACT_ROOT / "t4_route_candidate_utility" / "attempt_1"
MATERIAL_FILES = (
    "AGENTS.md",
    "configs/t4_frozen_program_benchmark_v2.json",
    "docs/GENMOL_T4_SEEDS.json",
    "modal_apps/t4_route_candidate_utility_app.py",
    "src/compose_v4/experiments/t4_route_candidate_utility.py",
    "tools/t4_program_vocabulary_audit.py",
    "tools/t4_route_distillation.py",
    "tools/t4_route_policy_comparison.py",
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

artifact_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
app = modal.App(APP_NAME)


def _validate(task):
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import verify_file

    body = {key: value for key, value in task.items() if key != "run_id"}
    if identity(body) != task.get("run_id"):
        raise ValueError("candidate-utility task identity changed")
    if task.get("oracle_calls") != 0 or task.get("automatic_retry") != 0:
        raise ValueError("candidate-lock generation must remain zero-oracle")
    if task.get("source_index") not in range(5):
        raise ValueError("candidate-utility source index changed")
    if set(task.get("files_sha256", ())) != set(MATERIAL_FILES):
        raise ValueError("candidate-utility code inventory changed")
    for relative, digest in task["files_sha256"].items():
        verify_file(REMOTE_ROOT / relative, digest)
    for path, digest in task["source_artifacts_sha256"].items():
        verify_file(Path(path), digest)
    _validate_remote_revision(task["image_revision"])


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=(4096, 4096),
    timeout=3600,
    retries=0,
    max_containers=15,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def lock_worker(task):
    from compose_v4.experiments.continuation_profile import publish_json
    from compose_v4.experiments.t4_route_candidate_utility import (
        lock_source_candidates,
    )

    artifact_volume.reload()
    _validate(task)
    output = OUTPUT_ROOT / task["cell"] / "candidate_lock.json"
    if output.exists():
        raise ValueError(f"preserve existing candidate lock: {output}")
    result = lock_source_candidates(
        REMOTE_ROOT,
        result_path=Path(task["result_path"]),
        model_path=Path(task["model_path"]),
        source_index=task["source_index"],
        delta=task["delta"],
    )
    publish_json(output, result)
    artifact_volume.commit()
    return {
        "cell": task["cell"],
        "output": str(output),
        "locked_candidates": len(result["locked_candidates"]),
        "costs": {"oracle_calls": 0, "docking_calls": 0},
    }
