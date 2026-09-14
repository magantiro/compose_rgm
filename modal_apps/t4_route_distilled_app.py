"""Route-distilled T4 qualification on an isolated artifact volume."""

import json

import modal

from modal_apps.genmol_t4_opt_app import REMOTE_ROOT, ROOT, _dock
from modal_apps.genmol_t4_opt_app import image as base_image
from modal_apps.run_process_v2_p50_app import _validate_remote_revision

ARTIFACT_ROOT = __import__("pathlib").Path("/artifacts")
CONTRACT = "configs/t4_route_distilled_qualification_v1.json"
FULL_CONTRACTS = (
    "configs/t4_route_distilled_full_delta04_v1.json",
    "configs/t4_route_distilled_full_delta06_v1.json",
)
PREFLIGHT = "diagnostics/t4_route_distilled_qualification/preflight.json"
FULL_PREFLIGHTS = (
    "diagnostics/t4_route_distilled_full/preflight_delta04.json",
    "diagnostics/t4_route_distilled_full/preflight_delta06.json",
)
POLICY_SELECTION = "diagnostics/t4_route_policy_selection/attempt_1/result.json"
ACTOR = "diagnostics/t4_route_distillation/attempt_3/actor.json.gz"
APP_PATH = "modal_apps/t4_route_distilled_app.py"
TOOL_PATH = "tools/t4_route_distilled_benchmark.py"
FULL_TOOL_PATH = "tools/t4_route_distilled_full.py"
APP_NAME = "compose-t4-route-distilled"
VOLUME_NAME = "compose-t4-route-distilled-artifacts"

manifest_paths = [ROOT / CONTRACT, *(ROOT / relative for relative in FULL_CONTRACTS)]
manifests = [
    json.loads(path.read_text())["payload"] for path in manifest_paths if path.exists()
]
image = base_image
if manifests:
    files = {
        APP_PATH,
        TOOL_PATH,
        FULL_TOOL_PATH,
        POLICY_SELECTION,
        ACTOR,
        *(
            path.relative_to(ROOT).as_posix()
            for path in manifest_paths
            if path.exists()
        ),
        *(manifest["library_path"] for manifest in manifests),
        *(relative for manifest in manifests for relative in manifest["inputs"]),
    }
    if (ROOT / PREFLIGHT).exists():
        files.add(PREFLIGHT)
    files.update(relative for relative in FULL_PREFLIGHTS if (ROOT / relative).exists())
    for relative in sorted(files):
        if relative.startswith(("src/", "configs/")) or relative in {
            "docs/GENMOL_T4_SEEDS.json",
            "modal_apps/genmol_t4_opt_app.py",
        }:
            continue
        image = image.add_local_file(
            ROOT / relative, str(REMOTE_ROOT / relative), copy=True
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

    body = {key: value for key, value in task.items() if key != "run_id"}
    if identity(body) != task.get("run_id"):
        raise ValueError("route-distilled task identity changed")
    for path, digest in task["files_sha256"].items():
        verify_file(REMOTE_ROOT / path, digest)
    _validate_remote_revision(task["image_revision"])


@app.function(**common, memory=(4096, 4096), max_containers=1, timeout=1800)
def structural_preflight(task):
    from tools.t4_route_distilled_benchmark import build_preflight

    _validate_task(task)
    return build_preflight(code_revision=task["image_revision"]["commit"])


@app.function(**common, memory=(2048, 2048), max_containers=1, timeout=600)
def preflight(task):
    from compose_v4.experiments.t4_route_distilled_qualification import (
        validate_launch,
    )

    contract = validate_launch(task, REMOTE_ROOT, _validate_remote_revision)
    return {
        "passed": contract["route_distilled_qualification"]["plateau_stopping"]
        is False,
        "runtime_route_rows": 0,
        "new_oracle_calls": 0,
        "image_revision": task["image_revision"],
    }


@app.function(**common, memory=(4096, 4096), max_containers=5, timeout=8 * 3600)
def worker(task):
    from compose_v4.experiments.t4_route_distilled_qualification import run_unit

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


@app.function(**common, memory=(4096, 4096), max_containers=1, timeout=1800)
def structural_preflight_full(task):
    from tools.t4_route_distilled_full import build_preflight

    _validate_task(task)
    return build_preflight(
        float(task["delta"]), code_revision=task["image_revision"]["commit"]
    )


@app.function(**common, memory=(2048, 2048), max_containers=1, timeout=600)
def preflight_full(task):
    from compose_v4.experiments.t4_route_distilled_full import validate_launch

    contract = validate_launch(task, REMOTE_ROOT, _validate_remote_revision)
    return {
        "passed": contract["route_distilled_full"]["plateau_stopping"] is False,
        "delta": float(task["delta"]),
        "runtime_route_rows": 0,
        "new_oracle_calls": 0,
        "image_revision": task["image_revision"],
    }


@app.function(**common, memory=(4096, 4096), max_containers=15, timeout=8 * 3600)
def worker_full(task):
    from compose_v4.experiments.t4_route_distilled_full import run_unit

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
