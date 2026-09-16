"""Isolated, source-only T4 support probes and bounded matched scored pilot.

No historical scores, route libraries, teacher endpoints or model checkpoints
are copied into this image. Mode 'score' requires a passing hash-bound receipt.
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/compose")
OUTPUT = Path("/reset")
RUNTIME = "configs/t4_objective_reset_runtime_v1.json"
VOLUME_NAME = "compose-t4-objective-reset-20260916"
MOOD = "https://raw.githubusercontent.com/SeulLee05/MOOD/main/scorer"
TARGETS = ("5ht1b", "braf", "jak2", "parp1", "fa7")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy==1.26.4", "scipy==1.13.1", "networkx==3.3", "rdkit==2024.3.5")
    .apt_install("openbabel", "curl", "ca-certificates")
    .run_commands(
        "mkdir -p /opt/dock/receptors",
        f"curl --fail -sSL -o /opt/dock/qvina02 {MOOD}/qvina02",
        "chmod +x /opt/dock/qvina02",
        *[
            f"curl --fail -sSL -o /opt/dock/receptors/{t}.pdbqt {MOOD}/receptors/{t}.pdbqt"
            for t in TARGETS
        ],
    )
    .add_local_dir(ROOT / "src", str(REMOTE / "src"), copy=True, ignore=["**/__pycache__/**"])
    .add_local_file(ROOT / RUNTIME, str(REMOTE / RUNTIME), copy=True)
    .add_local_file(
        ROOT / "modal_apps/t4_objective_reset_app.py",
        str(REMOTE / "modal_apps/t4_objective_reset_app.py"),
        copy=True,
    )
    .env({"PYTHONPATH": str(REMOTE / "src"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("compose-t4-objective-reset")
common = {
    "image": image,
    "cpu": (1, 1),
    "memory": 4096,
    "retries": 0,
    "volumes": {str(OUTPUT): volume},
    "scaledown_window": 5,
}


def validate(task):
    import platform

    from rdkit import rdBase

    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import verify_file
    from compose_v4.experiments.t4_matched_pilot import unseal

    body = {k: v for k, v in task.items() if k not in {"run_id", "cell", "arm"}}
    if identity(body) != task["run_id"]:
        raise ValueError("reset launch identity mismatch")
    if rdBase.rdkitVersion != "2024.03.5" or not platform.python_version().startswith("3.11."):
        raise ValueError("reset pinned runtime mismatch")
    for path, digest in task["files_sha256"].items():
        verify_file(REMOTE / path, digest)
    capsule = unseal(REMOTE / RUNTIME)
    for path, digest in capsule["runtime_input_sha256"].items():
        verify_file(Path(path), digest)
    if task["mode"] == "score":
        receipt = task["verification"]
        if receipt.get("passed") is not True or receipt.get("files_sha256") != task["files_sha256"]:
            raise ValueError("scored launch needs passing tests for these exact runtime files")
    elif task["mode"] != "probe":
        raise ValueError("unknown reset mode")
    return capsule


@app.function(**common, max_containers=6, timeout=3 * 3600)
def worker(task):
    import torch
    from rdkit import RDLogger

    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.t4_docking_adapter import dock_t4
    from compose_v4.experiments.t4_objective_reset import run_scored_unit, zero_oracle_probe

    torch.set_num_threads(1)
    RDLogger.DisableLog("rdApp.*")
    capsule = validate(task)
    unit = next(u for u in capsule["units"] if u["cell"] == task["cell"])
    folder = OUTPUT / task["run_id"] / f"{task['cell']}_{task['arm']}"
    if task["mode"] == "probe":
        return zero_oracle_probe(unit, folder, rounds=4, commit=volume.commit)
    if unit["cell"] not in capsule["pilot_cells"] or task["arm"] not in capsule["arms"]:
        raise ValueError("unit outside bounded scored panel")
    return run_scored_unit(
        unit,
        folder,
        arm=task["arm"],
        run_id=identity({"launch": task["run_id"], "arm": task["arm"]}),
        volume=volume,
        calls=capsule["calls_per_unit"],
        dock=lambda smiles, target, tag, seed: dock_t4(
            smiles,
            tag,
            seed,
            box={
                "coordinates": capsule["docking_boxes"][target],
                "receptor": f"/opt/dock/receptors/{target}.pdbqt",
            },
        ),
    )


@app.function(**common, max_containers=1, timeout=4 * 3600)
def run(task):
    from compose_v4.experiments.t4_matched_pilot import seal

    capsule = validate(task)
    seal(OUTPUT / task["run_id"] / "launch.json", task)
    volume.commit()
    if task["mode"] == "probe":
        tasks = [{**task, "cell": u["cell"], "arm": "zero_oracle"} for u in capsule["units"]]
    else:
        tasks = [
            {**task, "cell": cell, "arm": arm}
            for cell in capsule["pilot_cells"]
            for arm in capsule["arms"]
        ]
    records = []
    for request, result in zip(
        tasks, worker.map(tasks, order_outputs=True, return_exceptions=True), strict=True
    ):
        record = {"cell": request["cell"], "arm": request["arm"]}
        if isinstance(result, Exception):
            record.update(status="failed", error=repr(result))
        else:
            record.update(status="complete", result=result)
        records.append(record)
        seal(
            OUTPUT / task["run_id"] / "summary.json",
            {"mode": task["mode"], "records": records, "finished": len(records) == len(tasks)},
        )
        volume.commit()
        print(record, flush=True)
    return records


@app.local_entrypoint()
def main(mode: str = "probe", verification_file: str = ""):
    import json
    import subprocess

    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.continuation_profile import sha256_file
    from compose_v4.experiments.t4_matched_pilot import seal, unseal

    if mode not in {"probe", "score"}:
        raise ValueError("mode must be probe or score")
    paths = [*sorted((ROOT / "src").rglob("*.py")), ROOT / RUNTIME, Path(__file__)]
    # Only source/config changes matter; diagnostic outputs are separate artifacts.
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "src",
            "configs",
            "modal_apps/t4_objective_reset_app.py",
        ],
        cwd=ROOT,
        check=True,
    )
    untracked = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "src",
            "configs",
            "modal_apps/t4_objective_reset_app.py",
        ],
        cwd=ROOT,
        text=True,
    )
    if untracked.strip():
        raise ValueError("uncommitted runtime source files")
    files = {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}
    receipt = unseal(Path(verification_file)) if verification_file else None
    if mode == "score" and (
        not receipt or receipt.get("passed") is not True or receipt.get("files_sha256") != files
    ):
        raise ValueError("scored mode requires --verification-file for the exact current runtime")
    body = {
        "schema_version": "t4_objective_reset_launch_v1",
        "mode": mode,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "files_sha256": files,
        "verification": receipt,
        "adapter_source_sha256": sha256_file(Path(__file__)),
        "new_call_ceiling": 0 if mode == "probe" else 120,
    }
    task = {**body, "run_id": identity(body)}
    call = run.spawn(task)
    launch = {
        "task": task,
        "function_call_id": call.object_id,
        "volume": VOLUME_NAME,
        "output_prefix": task["run_id"],
    }
    seal(ROOT / "diagnostics/t4_objective_reset/launches" / f"{mode}_{task['run_id']}.json", launch)
    print(json.dumps(launch, sort_keys=True))
