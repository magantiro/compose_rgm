"""Two bounded groups, twelve shared CPU workers, no automatic retries."""

import json

import modal

from modal_apps.genmol_t4_opt_app import ARTIFACT_ROOT, REMOTE_ROOT, ROOT, _dock, artifact_volume
from modal_apps.genmol_t4_opt_app import image as base_image
from modal_apps.run_process_v2_p50_app import _validate_remote_revision

image = base_image.pip_install(
    "PyTDC==0.3.6",
    "numpy==1.26.4",
    "scipy==1.13.1",
    "rdkit==2024.3.5",
    "requests",
    "fuzzywuzzy",
    "seaborn",
    "networkx==3.3",
).env({"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
manifest = json.loads((ROOT / "configs/parent_edit_cycles.json").read_text())["payload"]
for path in sorted({"modal_apps/parent_edit_cycles_app.py", *manifest["inputs"]}):
    if not path.startswith("configs/"):
        image = image.add_local_file(ROOT / path, str(REMOTE_ROOT / path), copy=True)

app = modal.App("compose-parent-edit-cycles")
common = {
    "image": image,
    "cpu": (1.0, 1.0),
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
    "scaledown_window": 10,
}


@app.function(**common, memory=(8192, 8192), max_containers=12, timeout=2100)
def worker(task):
    from compose_v4.experiments.parent_edit_cycles import run_unit

    return run_unit(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda smiles, target, tag, seed: _dock(smiles, target, tag, cpu=1, random_seed=seed),
    )


@app.function(**common, memory=(2048, 2048), max_containers=2, timeout=7200)
def run(task):
    from compose_v4.experiments.parent_edit_cycles import run_group

    return run_group(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda tasks: worker.map(tasks, order_outputs=False),
        lambda tasks: confirm.map(tasks, order_outputs=False),
    )


@app.function(**common, memory=(4096, 4096), max_containers=2, timeout=600)
def confirm(task):
    from compose_v4.experiments.parent_edit_cycles import run_confirmation

    return run_confirmation(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda smiles, target, tag, seed: _dock(smiles, target, tag, cpu=1, random_seed=seed),
    )
