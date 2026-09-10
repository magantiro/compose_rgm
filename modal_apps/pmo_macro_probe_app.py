"""Twelve bounded CPU cases, durably spawned into a deployed app."""

import modal

from modal_apps.genmol_t4_opt_app import ARTIFACT_ROOT, REMOTE_ROOT, ROOT, _runtime, artifact_volume
from modal_apps.genmol_t4_opt_app import image as base_image
from modal_apps.run_process_v2_p50_app import _validate_remote_revision

image = (
    base_image.pip_install(
        "PyTDC==0.3.6",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "rdkit==2024.3.5",
        "requests",
        "networkx",
    )
    .add_local_file(
        ROOT / "modal_apps/pmo_macro_probe_app.py",
        str(REMOTE_ROOT / "modal_apps/pmo_macro_probe_app.py"),
        copy=True,
    )
    .add_local_file(
        ROOT / "docs/MOLLEO_DEV_COHORT.json",
        str(REMOTE_ROOT / "docs/MOLLEO_DEV_COHORT.json"),
        copy=True,
    )
    .add_local_file(
        ROOT / "diagnostics/pmo_legacy_oracle_parity.json",
        str(REMOTE_ROOT / "diagnostics/pmo_legacy_oracle_parity.json"),
        copy=True,
    )
    .add_local_dir(
        ROOT / "artifacts/oracles/molleo_task3_v1",
        str(REMOTE_ROOT / "artifacts/oracles/molleo_task3_v1"),
        copy=True,
    )
    .run_commands(
        'python -c "from pathlib import Path; '
        "from compose_v4.experiments.pmo_macro_probe import make_oracle; "
        "[make_oracle(name, Path('/root/compose'), {}) "
        "for name in ('albuterol_similarity', 'perindopril_mpo')]\""
    )
)
app = modal.App("compose-pmo-macro-probe")


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=8192,
    timeout=2400,
    retries=0,
    max_containers=12,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def probe_case(task):
    from compose_v4.experiments.pmo_macro_probe import run_remote

    return run_remote(
        task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _runtime, _validate_remote_revision
    )
