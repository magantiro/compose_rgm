"""Modal worker for the explicitly authorized PMO-v1 three-task run.

The scored contract and its authorization receipt remain local immutable inputs.  This
worker only supplies the pinned runtime, source capsule, and durable per-task boundary;
the PMO controller and accounting live in ``compose_v4.experiments.pmo_population_v1``.
Each task is an independent Modal call with retries disabled.  A started task is never
silently retried or backfilled.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/compose")
ARTIFACT_ROOT = Path("/artifacts")
VOLUME_NAME = "compose-v4-artifacts"
RUN_APP = "compose-pmo-population-v1-corrected"

CONTRACT = "configs/pmo_population_controller_v1_scored_contract_corrected.json"
CONTROLLER_CONTRACT = "configs/pmo_population_controller_v1.json"
ORACLE_CONTRACT = "configs/pmo_dynamic_v21_development_v1.json"
# The pinned oracle-asset capsule.  PyTDC resolves `oracle/<name>.pkl` RELATIVE to the
# working directory, lazily, on the first call -- so this directory has to be the
# working directory when the oracle is CALLED, not when it is constructed.  See
# `compose_v4.experiments.pmo_oracle_assets`.
ASSET_DIR = "diagnostics/pmo_ivg_oracle_parity/ivg_oracle_assets"
AUTHORIZATION = "diagnostics/pmo_population_controller_v1/corrected_scored_authorization_receipt.json"
SOURCE_MANIFEST = "diagnostics/pmo_population_controller_v1/source_capsule_manifest_v2.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv==0.5.31")
    .pip_install("torch==2.4.0", index_url="https://download.pytorch.org/whl/cpu")
    .run_commands(
        "uv pip install --system --no-deps 'PyTDC==1.1.15'",
        "uv pip install --system 'numpy==1.26.4' 'pandas==2.1.4' "
        "'rdkit==2023.9.6' 'requests==2.32.4' 'scikit-learn==1.2.2' "
        "'scipy==1.15.0' 'seaborn==0.13.2' 'setuptools==75.6.0' "
        "fuzzywuzzy huggingface-hub networkx packaging tqdm",
    )
    .add_local_dir(
        ROOT / "src", str(REMOTE_ROOT / "src"), copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_file(ROOT / "modal_apps/pmo_population_v1_app.py",
                    str(REMOTE_ROOT / "modal_apps/pmo_population_v1_app.py"), copy=True)
    .add_local_file(ROOT / CONTRACT, str(REMOTE_ROOT / CONTRACT), copy=True)
    .add_local_file(ROOT / CONTROLLER_CONTRACT,
                    str(REMOTE_ROOT / CONTROLLER_CONTRACT), copy=True)
    .add_local_file(ROOT / ORACLE_CONTRACT, str(REMOTE_ROOT / ORACLE_CONTRACT), copy=True)
    .add_local_file(ROOT / AUTHORIZATION, str(REMOTE_ROOT / AUTHORIZATION), copy=True)
    .add_local_file(ROOT / SOURCE_MANIFEST, str(REMOTE_ROOT / SOURCE_MANIFEST), copy=True)
    .add_local_file(
        ROOT / "diagnostics/parent_edit_cycles/prepared/init_20260921.json",
        str(REMOTE_ROOT / "diagnostics/parent_edit_cycles/prepared/init_20260921.json"),
        copy=True,
    )
    .add_local_file(
        ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json",
        str(REMOTE_ROOT / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/checkpoints.json"),
        copy=True,
    )
    .add_local_file(
        ROOT / "diagnostics/pmo_population_live_parent_gate_v1/scheduler_result.json",
        str(REMOTE_ROOT / "diagnostics/pmo_population_live_parent_gate_v1/scheduler_result.json"),
        copy=True,
    )
    .add_local_dir(
        ROOT / ASSET_DIR, str(REMOTE_ROOT / ASSET_DIR), copy=True,
    )
    .env({
        "PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
app = modal.App(RUN_APP)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _rdkit_six_shim() -> None:
    import sys
    import types
    import rdkit

    shim = types.ModuleType("rdkit.six")
    shim.iteritems = lambda value, **kwargs: iter(value.items())
    shim.itervalues = lambda value, **kwargs: iter(value.values())
    shim.iterkeys = lambda value, **kwargs: iter(value.keys())
    shim.string_types = (str,)
    sys.modules["rdkit.six"] = shim
    rdkit.six = shim


@app.function(
    image=image,
    cpu=(1.0, 1.0),
    memory=8192,
    timeout=4 * 60 * 60,
    retries=0,
    max_containers=3,
    scaledown_window=30,
    volumes={str(ARTIFACT_ROOT): volume},
)
def run_task(spec: dict) -> dict:
    """Run one task once, with no retry/backfill semantics."""

    from datetime import datetime, timezone

    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import verify_file
    from compose_v4.experiments.pmo_dynamic_v21 import verify_runtime_environment
    from compose_v4.experiments.pmo_oracle_assets import (
        PMO_ORACLE_ASSET_ROOT,
        AssetPinnedOracle,
        assert_positive_control,
        pinned_working_directory,
        requires_positive_control,
    )
    from compose_v4.experiments.pmo_population_v1 import (
        execute_task,
        load_contract,
    )

    volume.reload()
    root = REMOTE_ROOT
    run_id = spec["run_id"]
    task_name = spec["task"]
    contract = load_contract(root)
    contract_envelope = json.loads((root / CONTRACT).read_text())
    if spec["contract_payload_sha256"] != contract_envelope.get("payload_sha256"):
        raise ValueError("scored payload authorization identity changed")
    authorization = json.loads((root / AUTHORIZATION).read_text())
    if authorization.get("payload_sha256") != spec["contract_payload_sha256"]:
        raise ValueError("scored authorization receipt does not bind payload")
    old_oracle = json.loads((root / ORACLE_CONTRACT).read_text())["payload"]
    verify_runtime_environment(root, old_oracle)
    _rdkit_six_shim()
    from tdc import Oracle

    # The oracle's asset must resolve for the oracle's whole LIFETIME, not only
    # around its constructor.  PyTDC loads the pickle lazily on the first CALL from
    # the relative path `oracle/<name>.pkl` and swallows any failure into a constant
    # 0.0, so a constructor-only chdir produces a complete, plausible, entirely
    # uninformative ledger.  `AssetPinnedOracle` pins the working directory around
    # every call and primes the lazy load inside that window.
    assert ASSET_DIR == PMO_ORACLE_ASSET_ROOT, "worker and library disagree on the asset root"
    asset_root = root / ASSET_DIR
    with pinned_working_directory(asset_root):
        oracle = Oracle(name=task_name)
    scorer = AssetPinnedOracle(oracle, asset_root, name=task_name)

    folder = ARTIFACT_ROOT / "pmo_population_controller_v1" / run_id / task_name
    started = folder / "started.json"
    result_path = folder / "result.json"
    if result_path.exists() or started.exists():
        raise RuntimeError("task already started or completed; retries/backfill forbidden")

    # The positive control runs BEFORE `started.json` is written, so a control
    # failure leaves the task re-runnable instead of consuming its one attempt.  It
    # charges no PMO budget: these are local scikit-learn evaluations outside every
    # contract ledger, and they are counted and reported rather than hidden.
    positive_control = None
    if requires_positive_control(task_name):
        try:
            positive_control = assert_positive_control(scorer, task_name)
        except Exception as failure:
            _write_json(folder / "oracle_positive_control_failure.json", {
                "schema_version": "pmo_oracle_positive_control_failure_v1",
                "run_id": run_id,
                "task": task_name,
                "error_type": type(failure).__name__,
                "error": str(failure),
                "charged_oracle_calls": 0,
                "task_remains_runnable": True,
            })
            volume.commit()
            raise
        _write_json(folder / "oracle_positive_control.json",
                    {**positive_control, "run_id": run_id})
        volume.commit()
    control_calls = scorer.calls

    # The frozen contract stays fail-closed on disk.  The separately authorized
    # receipt is the only source of runtime scoring authority, bound in memory.
    runtime_contract = dict(contract)
    runtime_contract["scored_launch_authorized"] = True
    runtime_contract["modal_launch_authorized"] = True
    _write_json(started, {
        "schema_version": "pmo_population_task_started_v1",
        "run_id": run_id,
        "task": task_name,
        "contract_payload_sha256": spec["contract_payload_sha256"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "automatic_retry": False,
        "oracle_positive_control": (
            None if positive_control is None
            else {key: positive_control[key] for key in
                  ("task", "passed", "n_references", "n_agreeing",
                   "distinct_observed_values", "max_abs_delta", "reference_status")}
        ),
        "uncharged_positive_control_calls": control_calls,
    })
    volume.commit()

    def progress(row: dict) -> None:
        _write_json(folder / "progress.json", {
            **row,
            "run_id": run_id,
            "task": task_name,
            "contract_payload_sha256": spec["contract_payload_sha256"],
        })
        volume.commit()

    try:
        result = execute_task(
            runtime_contract, root, folder, task_name,
            evaluate=lambda smiles: float(scorer(smiles)),
            # The authorizing contract is the only budget authority; the runtime must not
            # fall back to a module default it was never authorized for.
            charged_calls_per_task=int(
                contract_envelope["payload"]["budget"]["charged_calls_per_task"]
            ),
            progress=progress,
        )
        result["run_id"] = run_id
        result["contract_payload_sha256"] = spec["contract_payload_sha256"]
        result["automatic_retries"] = 0
        result["oracle_positive_control"] = positive_control
        result["oracle_positive_control_passed"] = (
            True if positive_control is None else bool(positive_control["passed"])
        )
        result["uncharged_positive_control_calls"] = control_calls
        _write_json(result_path, result)
        volume.commit()
        # The AUC key carries the budget it was computed at; a fixed 1000-call denominator
        # was removed because it silently understated a 250-call run and was not comparable
        # to a published 10000-call figure.  Selecting a stale key here would KeyError AFTER
        # the full budget is charged, recording a completed run as a failure.
        return {key: result[key] for key in
                ("task", "charged_oracle_calls", "best_score", "auc_top10_at_budget",
                 "auc_budget", "oracle_positive_control_passed")}
    except Exception as error:
        _write_json(folder / "failure.json", {
            "schema_version": "pmo_population_task_failure_v1",
            "run_id": run_id,
            "task": task_name,
            "error_type": type(error).__name__,
            "error": str(error),
            "automatic_retry": False,
        })
        volume.commit()
        raise
