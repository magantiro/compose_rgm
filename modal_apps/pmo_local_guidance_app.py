"""Four-round broad controller comparison with an optional local selector."""

import modal

from modal_apps.pmo_online_policy_app import image as qualified_image
from modal_apps.run_process_v2_p50_app import ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume

image = qualified_image
for path in (
    "modal_apps/pmo_local_guidance_app.py",
    "docs/PMO_LOCAL_GUIDANCE.md",
    "diagnostics/pmo_local_guidance/prepared.json",
):
    image = image.add_local_file(ROOT / path, str(REMOTE_ROOT / path), copy=True)
app = modal.App("compose-pmo-local-guidance")
shared = {
    "image": image,
    "cpu": (1.0, 1.0),
    "memory": 8192,
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
}


@app.function(**shared, max_containers=29, timeout=180, scaledown_window=60)
def worker(task):
    from compose_v4.experiments.pmo_local_guidance import load_contract, worker_remote
    from compose_v4.experiments.pmo_online_policy import runtime
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    return worker_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda: runtime(
            REMOTE_ROOT, ARTIFACT_ROOT, load_contract(REMOTE_ROOT)["runtime_contract_sha256"]
        ),
    )


@app.function(**shared, max_containers=1, timeout=900)
def run(task):
    from compose_v4.experiments.pmo_local_guidance import driver_remote
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    return driver_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda tasks: worker.map(tasks, order_outputs=False),
    )
