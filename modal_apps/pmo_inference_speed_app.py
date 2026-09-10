"""CPU-only, zero-oracle frozen-model export and acceleration probe."""

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    _validate_remote_revision,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as base_image

image = base_image.env(
    {
        "PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
)
for path in (
    "modal_apps/pmo_inference_speed_app.py",
    "modal_apps/genmol_t4_opt_app.py",
    "diagnostics/pmo_inference_speed/prepared.json",
    "docs/PMO_INFERENCE_SPEED.md",
    "docs/GENMOL_T4_SEEDS.json",
    "docs/GENMOL_T4_DEV_SEEDS.json",
    "diagnostics/t4_task_search/value_check.json",
    "diagnostics/t4_whole_ring_plan/result.json",
):
    image = image.add_local_file(ROOT / path, str(REMOTE_ROOT / path), copy=True)

app = modal.App("compose-pmo-inference-speed")
resources = {
    "image": image,
    "cpu": (1.0, 1.0),
    "memory": 8192,
    "timeout": 1200,
    "max_containers": 1,
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
}


@app.function(**resources)
def export_model(task: dict) -> dict:
    from compose_v4.experiments.pmo_inference_speed import export_remote
    from modal_apps.genmol_t4_opt_app import _runtime

    return export_remote(
        task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _validate_remote_revision, _runtime
    )


@app.function(**resources)
def probe(task: dict) -> dict:
    from compose_v4.experiments.pmo_inference_speed import probe_remote

    return probe_remote(
        task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _validate_remote_revision
    )
