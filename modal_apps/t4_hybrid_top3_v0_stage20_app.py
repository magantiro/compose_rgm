"""Modal application for the exact five-cell hybrid Stage-20 run."""

from __future__ import annotations

import json

import modal

from modal_apps.genmol_t4_opt_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    _dock,
    artifact_volume,
)
from modal_apps.genmol_t4_opt_app import image as base_image
from modal_apps.run_process_v2_p50_app import _validate_remote_revision
from tools.t4_hybrid_top3_v0_stage20 import material_files

CONTRACT = "configs/t4_hybrid_top3_v0_stage20_launch_v1.json"

payload = json.loads((ROOT / CONTRACT).read_text())["payload"]
serialized = material_files({"launch": payload})
image = base_image
for relative in serialized:
    image = image.add_local_file(
        ROOT / relative,
        str(REMOTE_ROOT / relative),
        copy=True,
    )

app = modal.App("compose-t4-hybrid-top3-v0-stage20")
common = {
    "image": image,
    "cpu": (1.0, 1.0),
    "retries": 0,
    "scaledown_window": 30,
}


@app.function(**common, memory=(2048, 2048), max_containers=1, timeout=600)
def preflight(task):
    from compose_v4.experiments.t4_hybrid_top3_v0_stage20 import remote_preflight

    return remote_preflight(task, REMOTE_ROOT, _validate_remote_revision)


@app.function(
    **common,
    memory=(4096, 4096),
    max_containers=5,
    timeout=4 * 3600,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def worker(task):
    from compose_v4.experiments.t4_hybrid_top3_v0_stage20 import run_unit

    return run_unit(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda smiles, target, tag, seed: _dock(
            smiles,
            target,
            tag,
            cpu=1,
            random_seed=seed,
        ),
    )
