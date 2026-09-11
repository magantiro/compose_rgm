"""Bounded matched donor-memory comparison, at most thirty CPU containers."""

import modal

from modal_apps.pmo_online_policy_app import image as qualified_image
from modal_apps.run_process_v2_p50_app import ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume

image = qualified_image
for path in (
    "modal_apps/pmo_evolving_memory_app.py",
    "docs/PMO_EVOLVING_MEMORY.md",
    "diagnostics/pmo_evolving_memory/prepared.json",
):
    image = image.add_local_file(ROOT / path, str(REMOTE_ROOT / path), copy=True)
app = modal.App("compose-pmo-evolving-memory")
shared = {
    "image": image,
    "cpu": (1.0, 1.0),
    "memory": 8192,
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
}


@app.function(**shared, max_containers=29, timeout=180, scaledown_window=60)
def worker(task):
    from compose_v4.experiments.pmo_donor_comparison import worker_remote
    from compose_v4.experiments.pmo_evolving_memory import load_contract, session
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
        contract_loader=load_contract,
        run_session=session,
    )


@app.function(**shared, max_containers=1, timeout=900)
def run(task):
    from compose_v4.experiments.pmo_evolving_memory import driver_remote
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    return driver_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda tasks: worker.map(tasks, order_outputs=False),
    )
