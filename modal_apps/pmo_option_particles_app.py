"""Durable option-level particle experiment, at most 25 CPU containers."""

import modal

from modal_apps.pmo_online_policy_app import image as qualified_image
from modal_apps.run_process_v2_p50_app import ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume

image = qualified_image
for path in (
    "modal_apps/pmo_option_particles_app.py",
    "docs/PMO_OPTION_PARTICLES.md",
    "docs/PMO_ROUTE_SUPPORT_DEV_SNAPSHOT.md",
    "diagnostics/pmo_option_particles/prepared.json",
):
    image = image.add_local_file(ROOT / path, str(REMOTE_ROOT / path), copy=True)

app = modal.App("compose-pmo-option-particles")
shared = {
    "image": image,
    "cpu": (1.0, 1.0),
    "memory": 8192,
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
}


@app.function(**shared, max_containers=24, timeout=180, scaledown_window=120)
def worker(task):
    from compose_v4.experiments.pmo_branch_policy import propose_remote
    from compose_v4.experiments.pmo_online_policy import runtime
    from compose_v4.experiments.pmo_option_particles import load_contract, session
    from compose_v4.experiments.pmo_route_support import validate_development_revision

    return propose_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        lambda r: validate_development_revision(r, REMOTE_ROOT),
        lambda: runtime(
            REMOTE_ROOT, ARTIFACT_ROOT, load_contract(REMOTE_ROOT)["runtime_contract_sha256"]
        ),
        run_session=session,
    )


@app.function(**shared, max_containers=1, timeout=900)
def run(task):
    from compose_v4.experiments.pmo_option_particles import driver_remote
    from compose_v4.experiments.pmo_route_support import validate_development_revision

    return driver_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        lambda r: validate_development_revision(r, REMOTE_ROOT),
        lambda tasks: worker.map(tasks, order_outputs=False),
    )
