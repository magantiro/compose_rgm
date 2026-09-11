"""One bounded T4 transfer batch, sharing the 30-container budget with PMO."""

import modal

from modal_apps.genmol_t4_opt_app import MOOD
from modal_apps.pmo_online_policy_app import image as qualified_image
from modal_apps.run_process_v2_p50_app import ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume

image = qualified_image.apt_install("openbabel", "curl", "ca-certificates").run_commands(
    "mkdir -p /opt/dock/receptors",
    f"curl -fLsS -o /opt/dock/qvina02 {MOOD}/qvina02",
    "chmod +x /opt/dock/qvina02",
    f"curl -fLsS -o /opt/dock/receptors/parp1.pdbqt {MOOD}/receptors/parp1.pdbqt",
)
for path in (
    "modal_apps/t4_donor_probe_app.py",
    "modal_apps/genmol_t4_opt_app.py",
    "docs/T4_DONOR_PROBE.md",
    "diagnostics/t4_donor_probe/prepared.json",
):
    image = image.add_local_file(ROOT / path, str(REMOTE_ROOT / path), copy=True)
app = modal.App("compose-t4-donor-probe")
shared = {
    "image": image,
    "cpu": (1.0, 1.0),
    "memory": 8192,
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
}


@app.function(**shared, max_containers=14, timeout=540, scaledown_window=60)
def worker(task):
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    if task.get("stage") == "dock":
        from compose_v4.experiments.t4_partial_docking import dock_saved_row
        from modal_apps.genmol_t4_opt_app import _dock

        return dock_saved_row(
            task,
            REMOTE_ROOT,
            ARTIFACT_ROOT,
            artifact_volume,
            _validate_remote_revision,
            lambda s, tag: _dock(s, "parp1", tag, cpu=1),
            run_kind="t4_donor_probe",
            batch_limit=20,
        )
    from compose_v4.experiments.pmo_donor_comparison import worker_remote
    from compose_v4.experiments.pmo_online_policy import runtime
    from compose_v4.experiments.t4_donor_probe import load_contract, session

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


@app.function(**shared, max_containers=1, timeout=1200)
def run(task):
    from compose_v4.experiments.t4_donor_probe import driver_remote
    from modal_apps.run_process_v2_p50_app import _validate_remote_revision

    return driver_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda tasks: worker.map(tasks, order_outputs=False),
    )
