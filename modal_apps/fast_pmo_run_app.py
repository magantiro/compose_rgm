"""Single CPU-only development run; no reference or docking runtime."""

import json

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
    "PyTDC==0.3.6",
    "numpy==1.26.4",
    "scipy==1.13.1",
    "rdkit==2024.3.5",
    "requests",
    "fuzzywuzzy",
    "seaborn",
    "networkx==3.3",
).env(
    {
        "PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
)
# Modal auto-mounts this package under /root/modal_apps. Its import-time ROOT
# therefore differs from the authenticated runtime /root/compose. Host files
# are needed only while building the image, never while importing the worker.
if modal.is_local():
    manifest = json.loads((ROOT / "configs/fast_pmo_run.json").read_text())["payload"]
    for path in sorted({"modal_apps/fast_pmo_run_app.py", *manifest["inputs"]}):
        if not path.startswith("configs/"):
            image = image.add_local_file(ROOT / path, str(REMOTE_ROOT / path), copy=True)

app = modal.App("compose-fast-pmo")


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=(4096, 4096),
    max_containers=1,
    timeout=600,
    retries=0,
    scaledown_window=2,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run(task):
    from compose_v4.experiments.fast_pmo_run import run_remote

    return run_remote(task, REMOTE_ROOT, ARTIFACT_ROOT, artifact_volume, _validate_remote_revision)
