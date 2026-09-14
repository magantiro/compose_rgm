"""Three-cell Dynamic COMPOSE v1 development application."""

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

CONTRACT = "configs/t4_dynamic_v1_development_v1.json"
PREFLIGHT = "diagnostics/t4_dynamic_v1/preflight.json"
APP_PATH = "modal_apps/t4_dynamic_v1_app.py"
APP_NAME = "compose-t4-dynamic-v1"

manifest = json.loads((ROOT / CONTRACT).read_text())["payload"]
image = base_image
for relative in (APP_PATH, CONTRACT, manifest["library_path"], *manifest["inputs"]):
    if relative.startswith("src/") or relative in {
        "docs/GENMOL_T4_SEEDS.json",
        "modal_apps/genmol_t4_opt_app.py",
    }:
        continue
    image = image.add_local_file(
        ROOT / relative, str(REMOTE_ROOT / relative), copy=True
    )
if (ROOT / PREFLIGHT).exists():
    image = image.add_local_file(
        ROOT / PREFLIGHT, str(REMOTE_ROOT / PREFLIGHT), copy=True
    )

app = modal.App(APP_NAME)
common = {
    "image": image,
    "cpu": (1.0, 1.0),
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
    "scaledown_window": 30,
}


@app.function(**common, memory=(2048, 2048), max_containers=1, timeout=900)
def structural_preflight(task):
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import verify_file
    from tools.t4_dynamic_v1_benchmark import build_preflight

    _validate_remote_revision(task["image_revision"])
    base = {key: value for key, value in task.items() if key != "run_id"}
    if identity(base) != task["run_id"]:
        raise ValueError("Dynamic-v1 structural-preflight identity changed")
    for path, digest in task["files_sha256"].items():
        verify_file(REMOTE_ROOT / path, digest)
    return build_preflight(code_revision=task["image_revision"]["commit"])


@app.function(**common, memory=(2048, 2048), max_containers=1, timeout=600)
def preflight(task):
    from pathlib import Path

    from compose_v4.experiments.continuation_profile import verify_file
    from compose_v4.experiments.t4_dynamic_v1 import validate_launch

    contract = validate_launch(task, REMOTE_ROOT, _validate_remote_revision)
    verified = {
        path: verify_file(Path(path), digest)
        for path, digest in contract["runtime_input_sha256"].items()
    }
    return {
        "passed": len(verified) == 6,
        "runtime_input_sha256": verified,
        "new_oracle_calls": 0,
        "image_revision": task["image_revision"],
    }


@app.function(**common, memory=(4096, 4096), max_containers=3, timeout=4 * 3600)
def worker(task):
    from compose_v4.experiments.t4_dynamic_v1 import run_unit

    return run_unit(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda smiles, target, tag, seed: _dock(
            smiles, target, tag, cpu=1, random_seed=seed
        ),
    )
