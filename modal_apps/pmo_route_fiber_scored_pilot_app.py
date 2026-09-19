"""Launch the frozen PMO route-prior x FiberControl factorial on eight CPUs."""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/compose")
OUTPUT = Path("/outputs")
VOLUME_NAME = "compose-pmo-route-fiber-scored-pilot-v1"
CONTRACT = "configs/pmo_route_fiber_scored_pilot_v1.json"
DYNAMIC_CONTRACT = "configs/pmo_dynamic_v21_development_v1.json"
LOCKS = "diagnostics/pmo_route_fiber_scored_pilot_v1"
INITIALIZATION = "diagnostics/parent_edit_cycles/prepared/init_20260921.json"
PRODUCTION_RESULT = "diagnostics/pmo_route_fiber_pilot/production_yield_v3.json"
ROUTE_CHECKPOINT = (
    "diagnostics/pmo_route_fiber_pilot/route_transition_checkpoint_v3.json"
)
ASSETS = "diagnostics/pmo_ivg_oracle_parity/ivg_oracle_assets"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv==0.5.31")
    .run_commands(
        "uv pip install --system --no-deps 'PyTDC==1.1.15'",
        "uv pip install --system 'numpy==1.26.4' 'pandas==2.1.4' "
        "'rdkit==2023.9.6' 'requests==2.32.4' 'scikit-learn==1.2.2' "
        "'scipy==1.15.0' 'seaborn==0.13.2' 'setuptools==75.6.0' "
        "fuzzywuzzy huggingface-hub networkx packaging tqdm",
    )
    .add_local_dir(
        ROOT / "src", str(REMOTE / "src"), copy=True, ignore=["**/__pycache__/**"]
    )
    .add_local_file(ROOT / CONTRACT, str(REMOTE / CONTRACT), copy=True)
    .add_local_file(
        ROOT / DYNAMIC_CONTRACT, str(REMOTE / DYNAMIC_CONTRACT), copy=True
    )
    .add_local_file(
        ROOT / "configs/pmo_route_fiber_production_yield_v3.json",
        str(REMOTE / "configs/pmo_route_fiber_production_yield_v3.json"),
        copy=True,
    )
    .add_local_dir(ROOT / LOCKS, str(REMOTE / LOCKS), copy=True)
    .add_local_file(ROOT / INITIALIZATION, str(REMOTE / INITIALIZATION), copy=True)
    .add_local_file(
        ROOT / PRODUCTION_RESULT, str(REMOTE / PRODUCTION_RESULT), copy=True
    )
    .add_local_file(ROOT / ROUTE_CHECKPOINT, str(REMOTE / ROUTE_CHECKPOINT), copy=True)
    .add_local_dir(ROOT / ASSETS, str(REMOTE / ASSETS), copy=True)
    .env(
        {
            "PYTHONPATH": str(REMOTE / "src"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("compose-pmo-route-fiber-scored-pilot-v1")
common = {
    "image": image,
    "cpu": (1.0, 1.0),
    "memory": 4096,
    "retries": 0,
    "volumes": {str(OUTPUT): volume},
    "scaledown_window": 20,
}


def _install_rdkit_six() -> None:
    import sys
    import types

    import rdkit

    shim = types.ModuleType("rdkit.six")
    shim.iteritems = lambda value, **kwargs: iter(value.items())
    shim.itervalues = lambda value, **kwargs: iter(value.values())
    shim.iterkeys = lambda value, **kwargs: iter(value.keys())
    shim.string_types = (str,)
    sys.modules["rdkit.six"] = rdkit.six = shim


def _seal(path: Path, payload: dict) -> dict:
    import json

    from compose_v4.control.docking_value import identity

    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)
    volume.commit()
    return envelope


@app.function(**common, max_containers=8, timeout=60 * 60)
def run_unit(spec: dict) -> dict:
    import json
    import os

    from compose_v4.experiments.continuation_profile import verify_file
    from compose_v4.experiments.pmo_ivg_oracle_parity import (
        verify_environment,
        verify_pytdc_sources,
    )
    from compose_v4.experiments.pmo_route_fiber_scored_pilot import run_locked_unit

    volume.reload()
    launch_path = OUTPUT / spec["run_id"] / "launch.json"
    launch = json.loads(launch_path.read_text())["payload"]
    if launch["code_revision"] != spec["code_revision"]:
        raise ValueError("PMO launch/source revision mismatch")

    dynamic = json.loads((REMOTE / DYNAMIC_CONTRACT).read_text())["payload"]
    verify_environment(dynamic)
    verify_pytdc_sources(dynamic)
    asset = REMOTE / ASSETS / "oracle/gsk3b_current.pkl"
    expected = dynamic["oracle"]["gsk3b_asset"]
    if asset.stat().st_size != expected["bytes"]:
        raise ValueError("GSK3B asset byte count changed")
    verify_file(asset, expected["sha256"])

    _install_rdkit_six()
    from tdc import Oracle

    previous = Path.cwd()
    os.chdir(REMOTE / ASSETS)
    try:
        oracle = Oracle(name=spec["task"])
    finally:
        os.chdir(previous)

    folder = OUTPUT / spec["run_id"] / "units" / spec["task"] / spec["arm"]

    def report(row: dict) -> None:
        print(json.dumps(row, sort_keys=True), flush=True)
        volume.commit()

    result = run_locked_unit(
        REMOTE,
        folder,
        task=spec["task"],
        arm=spec["arm"],
        oracle_protocol=launch["oracle_protocols"][spec["task"]],
        evaluate=lambda smiles: float(oracle(smiles)),
        progress=report,
        query_lock_root=folder / "query_locks",
        launch_receipt=launch_path,
        flush=volume.commit,
    )
    volume.commit()
    return result


@app.function(**common, max_containers=1, timeout=2 * 60 * 60)
def drive(code_revision: str) -> dict:
    import json

    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.pmo_route_fiber_scored_pilot import (
        aggregate_scored_units,
        build_scored_launch_receipt,
    )
    from compose_v4.experiments.pmo_route_fiber_scored_pilot_contract import (
        ARMS,
        TASKS,
    )

    launch = build_scored_launch_receipt(REMOTE, code_revision=code_revision)
    run_id = identity(launch)
    launch = {**launch, "run_id": run_id}
    root = OUTPUT / run_id
    if (root / "launch.json").exists() or (root / "result.json").exists():
        raise RuntimeError("PMO scored launch already exists; automatic relaunch forbidden")
    _seal(root / "launch.json", launch)
    specs = [
        {
            "run_id": run_id,
            "code_revision": code_revision,
            "task": task,
            "arm": arm,
        }
        for task in sorted(TASKS)
        for arm in sorted(ARMS)
    ]
    results = []
    failures = []
    for value in run_unit.map(
        specs,
        order_outputs=False,
        return_exceptions=True,
        wrap_returned_exceptions=False,
    ):
        if isinstance(value, BaseException):
            failures.append(f"{type(value).__name__}: {value}")
            continue
        results.append(value)
        print(
            json.dumps(
                {
                    "task": value["task"],
                    "arm": value["arm"],
                    "calls": value["charged_calls"],
                    "best": value["best_reward"],
                    "auc48": value["auc_top10_48"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if failures:
        failure = {
            "schema_version": "pmo_route_fiber_scored_launch_failure_v1",
            "run_id": run_id,
            "completed_units": len(results),
            "failures": failures,
            "automatic_retries": 0,
        }
        _seal(root / "failure.json", failure)
        return failure
    aggregate = aggregate_scored_units(launch, results)
    _seal(root / "result.json", aggregate)
    return aggregate


@app.function(**common, max_containers=1, timeout=300)
def read_status() -> dict:
    import json

    volume.reload()
    launches = sorted(OUTPUT.glob("*/launch.json"))
    if not launches:
        return {"status": "not_launched"}
    launch_path = launches[-1]
    launch = json.loads(launch_path.read_text())["payload"]
    root = launch_path.parent
    units = []
    for task in launch["tasks"]:
        for arm in launch["arms"]:
            folder = root / "units" / task / arm
            row = {"task": task, "arm": arm, "started": folder.exists()}
            if (folder / "progress.json").exists():
                row["progress"] = json.loads((folder / "progress.json").read_text())
            if (folder / "result.json").exists():
                row["result"] = json.loads((folder / "result.json").read_text())["payload"]
            units.append(row)
    result = {
        "status": "complete" if (root / "result.json").exists() else "running",
        "run_id": launch["run_id"],
        "units": units,
    }
    if (root / "result.json").exists():
        result["result"] = json.loads((root / "result.json").read_text())["payload"]
    if (root / "failure.json").exists():
        result["failure"] = json.loads((root / "failure.json").read_text())["payload"]
    return result


@app.local_entrypoint()
def main(action: str = "launch") -> None:
    import json
    import subprocess

    if action == "status":
        print(json.dumps(read_status.remote(), indent=2, sort_keys=True))
        return
    if action != "launch":
        raise ValueError("action must be launch or status")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    call = drive.spawn(revision)
    print(json.dumps({"status": "spawned", "call_id": call.object_id, "revision": revision}))
