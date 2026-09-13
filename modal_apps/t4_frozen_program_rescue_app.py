"""Decision-equivalent worker for the frozen T4 tombstone-status repair."""

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

APP_NAME = "compose-t4-frozen-program-rescue"
LOCK_PATH = "diagnostics/t4_frozen_program_rescue/relaunch_lock_v4.json"
SOURCE_APP = "modal_apps/t4_frozen_program_benchmark_app.py"
SOURCE_PREFLIGHT = "diagnostics/t4_frozen_program_benchmark/preflight_v2.json"
SOURCE_CONTRACT = "configs/t4_frozen_program_benchmark_v2.json"

manifest = json.loads((ROOT / SOURCE_CONTRACT).read_text())["payload"]
image = (
    base_image.add_local_file(
        ROOT / "modal_apps/t4_frozen_program_rescue_app.py",
        str(REMOTE_ROOT / "modal_apps/t4_frozen_program_rescue_app.py"),
        copy=True,
    )
    .add_local_file(ROOT / SOURCE_APP, str(REMOTE_ROOT / SOURCE_APP), copy=True)
    .add_local_file(ROOT / LOCK_PATH, str(REMOTE_ROOT / LOCK_PATH), copy=True)
    .add_local_file(
        ROOT / manifest["library_path"],
        str(REMOTE_ROOT / manifest["library_path"]),
        copy=True,
    )
    .add_local_file(
        ROOT / SOURCE_PREFLIGHT,
        str(REMOTE_ROOT / SOURCE_PREFLIGHT),
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

app = modal.App(APP_NAME)


def _validate_frozen_source_revision(value, expected):
    """Validate the sealed original identity without claiming current source bytes."""
    if value != expected:
        raise RuntimeError("rescue invocation changed the original image identity")


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=(4096, 4096),
    max_containers=19,
    timeout=4 * 3600,
    retries=0,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
    scaledown_window=30,
)
def worker(task):
    from compose_v4.control.docking_value import identity
    from compose_v4.experiments import t4_frozen_program_benchmark as benchmark
    from compose_v4.experiments.continuation_profile import sha256_file, verify_file
    from compose_v4.experiments.t4_frozen_program_rescue import (
        validate_repaired_input_overrides,
        verify_with_repaired_inputs,
    )
    from compose_v4.experiments.t4_matched_pilot import unseal

    _validate_remote_revision(task["rescue_image_revision"])
    body = {
        key: value for key, value in task.items() if key not in ("run_id", "unit_id")
    }
    if identity(body) != task["run_id"]:
        raise ValueError("T4 repaired relaunch identity changed")
    for relative, digest in task["files_sha256"].items():
        verify_file(REMOTE_ROOT / relative, digest)
    lock = unseal(REMOTE_ROOT / LOCK_PATH)
    if sha256_file(REMOTE_ROOT / LOCK_PATH) != task["relaunch_lock_sha256"]:
        raise ValueError("T4 repaired relaunch lock changed")
    if task["unit_id"] not in lock["units"]:
        raise ValueError("T4 repaired relaunch unit is outside its lock")
    source_task = lock["source_task"]
    overrides = lock["contract_input_overrides"]
    source_contract = unseal(REMOTE_ROOT / SOURCE_CONTRACT)
    frozen_sha256 = {
        row["path"]: source_contract["inputs"][row["path"]] for row in overrides
    }
    repaired_sha256 = {
        row["path"]: sha256_file(REMOTE_ROOT / row["path"]) for row in overrides
    }
    validate_repaired_input_overrides(
        overrides,
        frozen_sha256=frozen_sha256,
        repaired_sha256=repaired_sha256,
    )

    frozen_verify_file = benchmark.verify_file

    def verify_rescue_input(path, expected):
        return verify_with_repaired_inputs(
            path,
            expected,
            root=REMOTE_ROOT,
            overrides=overrides,
            verify=frozen_verify_file,
        )

    benchmark.verify_file = verify_rescue_input
    try:
        return benchmark.run_unit(
            {**source_task, "unit_id": task["unit_id"]},
            REMOTE_ROOT,
            ARTIFACT_ROOT,
            artifact_volume,
            lambda revision: _validate_frozen_source_revision(
                revision, source_task["image_revision"]
            ),
            lambda smiles, target, tag, seed: _dock(
                smiles, target, tag, cpu=1, random_seed=seed
            ),
        )
    finally:
        benchmark.verify_file = frozen_verify_file
