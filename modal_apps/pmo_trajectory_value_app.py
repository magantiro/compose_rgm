"""Bounded achieved-trajectory learning and target-free option comparison."""

import modal

from modal_apps.pmo_online_policy_app import image as qualified_image
from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    _validate_remote_revision,
    artifact_volume,
)

image = qualified_image
for path in (
    "modal_apps/pmo_trajectory_value_app.py",
    "docs/PMO_TRAJECTORY_VALUE.md",
    "diagnostics/pmo_trajectory_value/prepared.json",
    "diagnostics/pmo_continuation_choice/prepared.json",
    "diagnostics/pmo_continuation_choice/phase1_scored.json",
    "diagnostics/pmo_continuation_choice/phase2_scored.json",
    "diagnostics/pmo_continuation_choice/phase3_scored.json",
    "diagnostics/pmo_public_winner_recovery/scores.json",
    "diagnostics/pmo_public_winner_recovery/result.json",
    *(
        f"diagnostics/pmo_public_winner_recovery/{name}.json"
        for name in ("current_best", *(f"original_root_{i}" for i in range(4)))
    ),
):
    image = image.add_local_file(ROOT / path, str(REMOTE_ROOT / path), copy=True)

app = modal.App("compose-pmo-trajectory-value")
shared = {
    "image": image,
    "cpu": (1.0, 1.0),
    "memory": 8192,
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
}


@app.function(**shared, max_containers=19, timeout=1200, scaledown_window=300)
def worker(task):
    from compose_v4.experiments.pmo_branch_policy import propose_remote
    from compose_v4.experiments.pmo_online_policy import runtime
    from compose_v4.experiments.pmo_trajectory_value import coverage_remote, load_contract, session

    if task["phase"] == 0:
        return coverage_remote(
            task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _validate_remote_revision
        )
    return propose_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda: runtime(
            REMOTE_ROOT, ARTIFACT_ROOT, load_contract(REMOTE_ROOT)["runtime_contract_sha256"]
        ),
        run_session=session,
    )


@app.function(**shared, max_containers=1, timeout=1800)
def run(task):
    from compose_v4.experiments.pmo_trajectory_value import driver_remote

    return driver_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda phase, tasks: worker.map(tasks, order_outputs=False),
    )
