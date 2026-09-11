"""Durable three-round continuation comparison, at most twenty CPU workers."""

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
    "modal_apps/pmo_continuation_choice_app.py",
    "docs/PMO_CONTINUATION_CHOICE.md",
    "diagnostics/pmo_continuation_choice/prepared.json",
    "diagnostics/pmo_online_policy/result_sealed.json",
    "diagnostics/pmo_public_winner_recovery/scores.json",
):
    image = image.add_local_file(ROOT / path, str(REMOTE_ROOT / path), copy=True)

app = modal.App("compose-pmo-continuation-choice")
shared = {
    "image": image,
    "cpu": (1.0, 1.0),
    "memory": 8192,
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
}


@app.function(**shared, max_containers=20, timeout=1200, scaledown_window=300)
def propose(task):
    from compose_v4.experiments.pmo_branch_policy import propose_remote
    from compose_v4.experiments.pmo_continuation_choice import load_contract, session
    from compose_v4.experiments.pmo_online_policy import runtime

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


@app.function(**shared, max_containers=1, timeout=3600)
def run(task):
    from compose_v4.experiments.pmo_continuation_choice import driver_remote

    return driver_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda tasks: propose.map(tasks, order_outputs=False),
    )
