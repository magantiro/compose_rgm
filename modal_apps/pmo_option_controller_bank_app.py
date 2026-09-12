"""Zero-oracle balanced WHERE/WHAT/HOW continuation bank on Modal."""

import modal

from modal_apps.pmo_online_policy_app import image as base_image
from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    _validate_remote_revision,
    artifact_volume,
)

for path in (
    "modal_apps/pmo_option_controller_bank_app.py",
    "docs/PMO_OPTION_CONTROLLER_BANK.md",
    "diagnostics/pmo_option_controller_bank/prepared.json",
):
    base_image = base_image.add_local_file(
        ROOT / path, str(REMOTE_ROOT / path), copy=True
    )

app = modal.App("compose-pmo-option-controller-bank")
shared = {
    "image": base_image,
    "cpu": (1.0, 1.0),
    "memory": 8192,
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
}


@app.function(**shared, max_containers=16, timeout=900, scaledown_window=300)
def propose(task):
    from compose_v4.experiments.pmo_option_controller_bank import worker_remote

    return worker_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
    )


@app.function(**shared, max_containers=1, timeout=1200)
def run(task):
    from compose_v4.experiments.pmo_option_controller_bank import driver_remote

    return driver_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda tasks: propose.map(tasks, order_outputs=False),
    )
