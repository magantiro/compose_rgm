"""Repeated score feedback, one driver and up to twelve CPU proposal workers."""

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    _validate_remote_revision,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as base_image

image = base_image.pip_install(
    "PyTDC==0.3.6", "numpy==1.26.4", "scipy==1.13.1", "rdkit==2024.3.5", "requests", "networkx"
).env(
    {
        "PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
)
for path in (
    "modal_apps/pmo_online_policy_app.py",
    "docs/PMO_ONLINE_POLICY.md",
    "diagnostics/pmo_online_policy/prepared.json",
    "diagnostics/pmo_branch_policy/prepared.json",
    "diagnostics/pmo_branch_policy/result_sealed.json",
    "diagnostics/pmo_inference_speed/result_sealed.json",
    "diagnostics/pmo_inference_speed/package_manifest.json",
    "diagnostics/pmo_inference_speed/reference_law_sealed.json",
):
    image = image.add_local_file(ROOT / path, str(REMOTE_ROOT / path), copy=True)

# Construct the evaluator during the image build, without scoring a molecule.
# PyTDC 0.3.6 omits runtime imports from its declared dependencies.
image = image.run_commands(
    'python -c "from pathlib import Path; '
    "from compose_v4.experiments.pmo_macro_probe import make_oracle; "
    "make_oracle('perindopril_mpo', Path('/root/compose'), {})\""
)

app = modal.App("compose-pmo-online-policy")
shared = {
    "image": image,
    "cpu": (1.0, 1.0),
    "memory": 8192,
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
}


@app.function(**shared, max_containers=12, timeout=1200, scaledown_window=300)
def propose(task):
    from compose_v4.experiments.pmo_branch_policy import propose_remote
    from compose_v4.experiments.pmo_online_policy import runtime, session

    return propose_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda: runtime(REMOTE_ROOT, ARTIFACT_ROOT, task["contract_sha256"]),
        run_session=session,
    )


@app.function(**shared, max_containers=1, timeout=300)
def check(task):
    from compose_v4.experiments.pmo_online_policy import runtime, session

    with session(task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _validate_remote_revision) as (
        _,
        store,
        _,
    ):
        result = runtime(REMOTE_ROOT, ARTIFACT_ROOT, task["contract_sha256"])["validation"]
        store.save("runtime_check", result)
        return {k: v for k, v in result.items() if k != "actual"}


@app.function(**shared, max_containers=1, timeout=2700)
def run(task):
    from compose_v4.experiments.pmo_branch_policy import driver_remote
    from compose_v4.experiments.pmo_online_policy import session

    with session(task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _validate_remote_revision) as (
        _,
        store,
        _,
    ):
        if store.read("runtime_check") is None:
            raise ValueError("run requires the saved numerical runtime admission check")
    return driver_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda tasks: propose.map(tasks, order_outputs=False),
        run_session=session,
    )
