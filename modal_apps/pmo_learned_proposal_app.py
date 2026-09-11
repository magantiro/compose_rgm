"""Frozen proposal-learning comparison, eighteen CPU workers and one driver."""

import modal

from modal_apps.pmo_online_policy_app import image as qualified_image
from modal_apps.run_process_v2_p50_app import ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume

image = qualified_image
for path in (
    "modal_apps/pmo_learned_proposal_app.py",
    "docs/PMO_LEARNED_PROPOSAL.md",
    "diagnostics/pmo_learned_proposal/prepared.json",
    "diagnostics/pmo_learned_proposal/model.json",
    "diagnostics/pmo_learned_proposal/training.json",
):
    image = image.add_local_file(ROOT / path, str(REMOTE_ROOT / path), copy=True)
app = modal.App("compose-pmo-learned-proposal")
shared = {
    "image": image,
    "cpu": (1.0, 1.0),
    "memory": 8192,
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
}


@app.function(**shared, max_containers=18, timeout=300, scaledown_window=60)
def worker(task):
    from compose_v4.experiments.pmo_branch_policy import propose_remote
    from compose_v4.experiments.pmo_learned_proposal import load_contract, proposal_factory, session
    from compose_v4.experiments.pmo_online_policy import runtime
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

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
        proposal_factory=proposal_factory,
    )


@app.function(**shared, max_containers=1, timeout=1200)
def run(task):
    from compose_v4.experiments.pmo_learned_proposal import driver_remote
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    return driver_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda tasks: worker.map(tasks, order_outputs=False),
    )
