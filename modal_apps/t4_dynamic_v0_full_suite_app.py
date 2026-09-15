"""Modal deployment for the frozen 15-cell Dynamic-v0 delta-0.6 wave."""

import json

import modal

from modal_apps.genmol_t4_opt_app import ARTIFACT_ROOT, REMOTE_ROOT, ROOT, _dock
from modal_apps.genmol_t4_opt_app import image as base_image
from modal_apps.run_process_v2_p50_app import _validate_remote_revision

CONTRACT = "configs/t4_dynamic_v0_full_suite_delta06_v2.json"
PREFLIGHT = "diagnostics/t4_dynamic_v0_full_suite_delta06/preflight_v2.json"
APP_PATH = "modal_apps/t4_dynamic_v0_full_suite_app.py"
APP_NAME = "compose-t4-dynamic-v0-full-suite-delta06"
VOLUME_NAME = "compose-t4-dynamic-v0-full-suite"

manifest = json.loads((ROOT / CONTRACT).read_text())["payload"]
image = base_image.add_local_file(
    ROOT / APP_PATH, str(REMOTE_ROOT / APP_PATH), copy=True
)
already_serialized = {"modal_apps/genmol_t4_opt_app.py"}
for relative in manifest["inputs"]:
    if relative.startswith(("src/", "configs/")) or relative in already_serialized:
        continue
    image = image.add_local_file(
        ROOT / relative, str(REMOTE_ROOT / relative), copy=True
    )
if (ROOT / PREFLIGHT).exists():
    image = image.add_local_file(
        ROOT / PREFLIGHT, str(REMOTE_ROOT / PREFLIGHT), copy=True
    )

artifact_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)
common = {
    "image": image,
    "cpu": (1.0, 1.0),
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
    "scaledown_window": 30,
}


def _validate_task(task):
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import verify_file

    base = {
        key: value for key, value in task.items() if key not in ("run_id", "unit_id")
    }
    if identity(base) != task["run_id"]:
        raise ValueError("Dynamic-v0 launch identity changed")
    _validate_remote_revision(task["image_revision"])
    for path, digest in task["files_sha256"].items():
        verify_file(REMOTE_ROOT / path, digest)


@app.function(**common, memory=(4096, 4096), max_containers=1, timeout=3600)
def structural_preflight(task):
    from tools.t4_dynamic_v0_full_suite import build_preflight

    _validate_task(task)
    return build_preflight(code_revision=task["image_revision"]["commit"])


@app.function(**common, memory=(2048, 2048), max_containers=1, timeout=300)
def runtime_preflight(task):
    from pathlib import Path

    from compose_v4.experiments.continuation_profile import verify_file
    from compose_v4.experiments.t4_dynamic_v0_full_suite import validate_launch

    contract = validate_launch(task, REMOTE_ROOT, _validate_remote_revision)
    verified = {
        path: verify_file(Path(path), digest)
        for path, digest in contract["runtime_input_sha256"].items()
    }
    return {"passed": len(verified) == 6, "verified": verified, "oracle_calls": 0}


@app.function(**common, memory=(4096, 4096), max_containers=15, timeout=8 * 3600)
def worker(task):
    from compose_v4.experiments.t4_dynamic_v0_full_suite import run_unit

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
