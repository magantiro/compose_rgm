"""Locked paired T4 scoring, eight CPU workers and a 73-call maximum."""

import modal

from modal_apps.genmol_t4_opt_app import ARTIFACT_ROOT, REMOTE_ROOT, ROOT, _dock, artifact_volume
from modal_apps.genmol_t4_opt_app import image as base_image
from modal_apps.run_process_v2_p50_app import _validate_remote_revision

image = base_image.add_local_file(
    ROOT / "modal_apps/t4_second_generation_app.py",
    str(REMOTE_ROOT / "modal_apps/t4_second_generation_app.py"),
    copy=True,
)
app = modal.App("compose-t4-second-generation")


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=2048,
    timeout=1200,
    max_containers=1,
    retries=0,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_comparison(task):
    from compose_v4.experiments.t4_second_generation import run_remote

    return run_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda tasks: dock_one.map(tasks, order_outputs=False),
    )


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=4096,
    timeout=600,
    max_containers=8,
    retries=0,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def dock_one(task):
    from compose_v4.experiments.t4_second_generation import dock_remote

    return dock_remote(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda smiles, target, tag, seed: _dock(smiles, target, tag, cpu=1, random_seed=seed),
    )
