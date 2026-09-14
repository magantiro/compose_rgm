"""Launch-locked three-cell T4 Dynamic-v2.2 pilot application."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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

APP_NAME = "compose-t4-dynamic-v22-pilot"
CONTRACT = "configs/dynamic_v22_quick_pilot_v1.json"

image = base_image
for path in (
    CONTRACT,
    "tools/dynamic_v22_pilot.py",
    "modal_apps/t4_dynamic_v22_pilot_app.py",
    "src/compose_v4/control/dynamic_program_synthesis_v22.py",
    "src/compose_v4/experiments/dynamic_v22_pilot.py",
):
    image = image.add_local_file(ROOT / path, str(REMOTE_ROOT / path), copy=True)

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
    from compose_v4.experiments.continuation_profile import sha256_file, verify_file
    from compose_v4.experiments.dynamic_v22_pilot import QUERY_BUDGET, T4_CELLS
    from tools.dynamic_v22_pilot import contains_forbidden_runtime_input

    body = {key: value for key, value in task.items() if key != "run_id"}
    if (
        identity(body) != task.get("run_id")
        or task.get("authorized") is not True
        or task.get("cell") not in T4_CELLS
        or task.get("query_budget") != QUERY_BUDGET
        or task.get("automatic_retry") != 0
        or task.get("contract_sha256") != sha256_file(REMOTE_ROOT / CONTRACT)
        or identity(task.get("source_state")) != task.get("source_state_sha256")
        or contains_forbidden_runtime_input(body)
    ):
        raise ValueError("T4 Dynamic-v2.2 task is absent from the clean launch lock")
    expected_files = {
        CONTRACT,
        "modal_apps/genmol_t4_opt_app.py",
        "modal_apps/t4_dynamic_v22_pilot_app.py",
        "tools/dynamic_v22_pilot.py",
    }
    if set(task.get("files_sha256", ())) != expected_files:
        raise ValueError("T4 Dynamic-v2.2 launch file inventory changed")
    for path, digest in task["files_sha256"].items():
        verify_file(REMOTE_ROOT / path, digest)
    verify_file(Path("/opt/dock/qvina02"), task["qvina02_sha256"])
    verify_file(
        Path(f"/opt/dock/receptors/{task['target']}.pdbqt"),
        task["receptor_sha256"],
    )
    _validate_remote_revision(task["image_revision"])


@app.function(**common, memory=(2048, 2048), max_containers=1, timeout=1800)
def preflight(task):
    from tools.dynamic_v22_pilot import build_preflight

    _validate_remote_revision(task["image_revision"])
    result = build_preflight(REMOTE_ROOT)
    if not result["passed"] or result["new_oracle_calls"] != 0:
        raise RuntimeError("Dynamic-v2.2 zero-oracle preflight failed")
    return result


@app.function(**common, memory=(4096, 4096), max_containers=3, timeout=4 * 3600)
def worker(task):
    from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
    from compose_v4.experiments.dynamic_v22_pilot import run_t4_pilot_campaign

    _validate_task(task)
    config = replace(
        ProgramSearchConfig.program_only_recipe(
            seed=task["controller_seed"], score_direction="minimize"
        ),
        proposal_cache_entries=128,
        candidates_per_batch=16,
        attempts_per_batch=128,
        wall_seconds=45.0,
    )
    output = ARTIFACT_ROOT / "dynamic_v22_pilot" / task["run_id"] / task["cell"]
    result = run_t4_pilot_campaign(
        output=output,
        source_state=task["source_state"],
        original_seed=task["original_seed"],
        target=task["target"],
        oracle_protocol=task["oracle_protocol"],
        config=config,
        evaluate=lambda smiles: _dock(
            smiles,
            task["target"],
            f"v22_{task['cell']}_{len(list((output / 'oracle').glob('query_*'))):04d}",
            cpu=1,
            random_seed=task["docking_seed"],
        ),
        delta=0.4,
        rounds=64,
        flush=artifact_volume.commit,
    )
    artifact_volume.commit()
    return result
