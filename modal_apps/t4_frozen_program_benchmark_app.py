"""Frozen 45-unit T4 delta=0.4 program-controller benchmark."""

import json

import modal

from modal_apps.genmol_t4_opt_app import ARTIFACT_ROOT, REMOTE_ROOT, ROOT, _dock, artifact_volume
from modal_apps.genmol_t4_opt_app import image as base_image
from modal_apps.run_process_v2_p50_app import _validate_remote_revision

manifest = json.loads((ROOT / "configs/t4_frozen_program_benchmark_v2.json").read_text())[
    "payload"
]
image = (
    base_image.add_local_file(
        ROOT / "modal_apps/t4_frozen_program_benchmark_app.py",
        str(REMOTE_ROOT / "modal_apps/t4_frozen_program_benchmark_app.py"),
        copy=True,
    )
    .add_local_file(
        ROOT / manifest["library_path"],
        str(REMOTE_ROOT / manifest["library_path"]),
        copy=True,
    )
    .add_local_file(
        ROOT / "diagnostics/t4_frozen_program_benchmark/preflight_v2.json",
        str(REMOTE_ROOT / "diagnostics/t4_frozen_program_benchmark/preflight_v2.json"),
        copy=True,
    )
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
        ROOT / relative,
        str(REMOTE_ROOT / relative),
        copy=True,
    )

app = modal.App("compose-t4-frozen-program-benchmark")
common = {
    "image": image,
    "cpu": (1.0, 1.0),
    "retries": 0,
    "volumes": {str(ARTIFACT_ROOT): artifact_volume},
    "scaledown_window": 30,
}


@app.function(**common, memory=(2048, 2048), max_containers=1, timeout=300)
def preflight(task):
    """Validate the built image and every docking input without invoking docking."""
    from pathlib import Path

    from compose_v4.experiments.continuation_profile import verify_file
    from compose_v4.experiments.t4_frozen_program_benchmark import validate_launch

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


@app.function(**common, memory=(4096, 4096), max_containers=29, timeout=4 * 3600)
def worker(task):
    from compose_v4.experiments.t4_frozen_program_benchmark import run_unit

    return run_unit(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda smiles, target, tag, seed: _dock(smiles, target, tag, cpu=1, random_seed=seed),
    )


@app.function(**common, memory=(4096, 4096), max_containers=29, timeout=900)
def confirm(task):
    from compose_v4.experiments.t4_frozen_program_benchmark import run_confirmation

    return run_confirmation(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda smiles, target, tag, seed: _dock(smiles, target, tag, cpu=1, random_seed=seed),
    )


@app.function(**common, memory=(2048, 2048), max_containers=1, timeout=8 * 3600)
def run(task):
    import time

    from compose_v4.experiments.t4_frozen_program_benchmark import run_all

    def confirmation_after_worker_scale_down(tasks):
        # The search map can leave up to 29 idle containers during its declared
        # 30-second scale-down window. Wait once before starting the independent
        # confirmation pool so the app never exceeds the 30-container contract.
        time.sleep(35)
        return confirm.map(tasks, order_outputs=True, return_exceptions=True)

    return run_all(
        task,
        REMOTE_ROOT,
        ARTIFACT_ROOT,
        artifact_volume,
        _validate_remote_revision,
        lambda tasks: worker.map(tasks, order_outputs=True, return_exceptions=True),
        confirmation_after_worker_scale_down,
    )
