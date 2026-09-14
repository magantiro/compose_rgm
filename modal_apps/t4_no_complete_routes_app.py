"""Three-cell T4 complete-route ablation and dynamic-synthesis diagnostic."""

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

CONTRACT = "configs/t4_no_complete_routes_diagnostic_v1.json"
PREFLIGHT = "diagnostics/t4_no_complete_routes/attempt_1/preflight.json"
APP_PATH = "modal_apps/t4_no_complete_routes_app.py"
APP_NAME = "compose-t4-no-complete-routes"

manifest = json.loads((ROOT / CONTRACT).read_text())["payload"]
image = (
    base_image.add_local_file(ROOT / APP_PATH, str(REMOTE_ROOT / APP_PATH), copy=True)
    .add_local_file(ROOT / CONTRACT, str(REMOTE_ROOT / CONTRACT), copy=True)
    .add_local_file(
        ROOT / manifest["library_path"],
        str(REMOTE_ROOT / manifest["library_path"]),
        copy=True,
    )
)
if (ROOT / PREFLIGHT).exists():
    image = image.add_local_file(
        ROOT / PREFLIGHT, str(REMOTE_ROOT / PREFLIGHT), copy=True
    )
_already_serialized = {
    manifest["library_path"],
    "docs/GENMOL_T4_SEEDS.json",
    "modal_apps/genmol_t4_opt_app.py",
}
for relative in manifest["inputs"]:
    if relative.startswith(("src/", "configs/")) or relative in _already_serialized:
        continue
    image = image.add_local_file(
        ROOT / relative, str(REMOTE_ROOT / relative), copy=True
    )

app = modal.App(APP_NAME)
common = {
    "image": image,
    "cpu": (1.0, 1.0),
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
    "scaledown_window": 30,
}


@app.function(**common, memory=(2048, 2048), max_containers=1, timeout=300)
def structural_preflight(task):
    """Build the pinned-RDKit, zero-oracle preflight before it is sealed locally."""
    from compose_v4.experiments.continuation_profile import verify_file
    from tools.t4_no_complete_routes import build_preflight

    _validate_remote_revision(task["image_revision"])
    base = {key: value for key, value in task.items() if key != "run_id"}
    from compose_v4.control.docking_value import identity

    if identity(base) != task["run_id"]:
        raise ValueError("structural-preflight launch identity changed")
    for path, digest in task["files_sha256"].items():
        verify_file(REMOTE_ROOT / path, digest)
    return build_preflight()


@app.function(**common, memory=(2048, 2048), max_containers=1, timeout=300)
def preflight(task):
    """Validate the remote image and docking inputs without invoking docking."""
    from pathlib import Path

    from compose_v4.experiments.continuation_profile import verify_file
    from compose_v4.experiments.t4_no_complete_routes import validate_launch

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


@app.function(**common, memory=(4096, 4096), max_containers=6, timeout=4 * 3600)
def worker(task):
    from compose_v4.experiments.t4_no_complete_routes import run_unit

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
