"""Durable driver plus up to eight complete-option proposal workers."""

import modal

from modal_apps.genmol_t4_opt_app import ARTIFACT_ROOT, REMOTE_ROOT, ROOT, _runtime, artifact_volume
from modal_apps.pmo_macro_probe_app import image as base_image
from modal_apps.run_process_v2_p50_app import _validate_remote_revision

image = (
    base_image.add_local_file(
        ROOT / "modal_apps/pmo_branch_policy_app.py",
        str(REMOTE_ROOT / "modal_apps/pmo_branch_policy_app.py"),
        copy=True,
    )
    .add_local_file(
        ROOT / "diagnostics/pmo_branch_policy/prepared.json",
        str(REMOTE_ROOT / "diagnostics/pmo_branch_policy/prepared.json"),
        copy=True,
    )
    .add_local_file(
        ROOT / "docs/PMO_BRANCH_POLICY.md",
        str(REMOTE_ROOT / "docs/PMO_BRANCH_POLICY.md"),
        copy=True,
    )
    .env({"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
)
app = modal.App("compose-pmo-branch-policy")


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=8192,
    timeout=1800,
    retries=0,
    max_containers=8,
    scaledown_window=300,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def propose(task):
    from compose_v4.experiments.pmo_branch_policy import propose_remote

    return propose_remote(
        task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _validate_remote_revision, _runtime
    )


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=4096,
    timeout=3600,
    retries=0,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run(task):
    from compose_v4.experiments.pmo_branch_policy import driver_remote

    return driver_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda tasks: propose.map(tasks, order_outputs=False),
    )
