"""Frozen versus online complete-plan proposal learning, on CPU."""

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
    "modal_apps/pmo_plan_policy_app.py",
    "docs/PMO_PLAN_POLICY.md",
    "diagnostics/pmo_plan_policy/prepared.json",
    "diagnostics/pmo_plan_policy/training.json",
):
    image = image.add_local_file(ROOT / path, str(REMOTE_ROOT / path), copy=True)

app = modal.App("compose-pmo-plan-policy")
shared = {
    "image": image,
    "cpu": (1.0, 1.0),
    "memory": 8192,
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
}


@app.function(**shared, max_containers=29, timeout=180, scaledown_window=60)
def worker(task):
    from compose_v4.experiments.pmo_online_policy import runtime
    from compose_v4.experiments.pmo_plan_policy import load_contract, worker_remote

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


@app.function(**shared, max_containers=1, timeout=180)
def check(task):
    from compose_v4.experiments.pmo_online_policy import runtime
    from compose_v4.experiments.pmo_plan_policy import session

    with session(task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _validate_remote_revision) as (
        contract,
        store,
        _,
    ):
        result = runtime(REMOTE_ROOT, ARTIFACT_ROOT, contract["runtime_contract_sha256"])[
            "validation"
        ]
        store.save("runtime_check", result)
        return {k: v for k, v in result.items() if k != "actual"}


@app.function(**shared, max_containers=1, timeout=900)
def run(task):
    from compose_v4.experiments.pmo_plan_policy import driver_remote, session

    with session(task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _validate_remote_revision) as (
        _,
        store,
        _,
    ):
        if store.read("runtime_check") is None:
            raise ValueError("run requires the saved runtime admission check")
    return driver_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda tasks: worker.map(tasks, order_outputs=False),
    )
