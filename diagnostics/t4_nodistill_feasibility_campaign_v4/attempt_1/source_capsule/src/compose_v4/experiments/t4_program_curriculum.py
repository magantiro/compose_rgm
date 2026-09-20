"""Bounded first scores for the shared four-target program curriculum."""

from __future__ import annotations

import json
import math
import platform
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

from rdkit import rdBase

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.experiments.t4_winner_refinement import property_scorer
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.trace_shard import decode_state

KIND = "t4_program_curriculum"
LOCK = "configs/t4_program_curriculum_lock.json"
APP = "modal_apps/t4_program_curriculum_app.py"


def validate_task(task, root, validate_revision):
    validate_revision(task["image_revision"])
    for path, digest in task["files_sha256"].items():
        verify_file(root / path, digest)
    if identity({k: v for k, v in task.items() if k != "run_id"}) != task["run_id"]:
        raise ValueError("curriculum task identity changed")
    lock = unseal(root / LOCK)
    if len(lock["take"]) != 36 or lock["new_oracle_call_limit"] != 36:
        raise ValueError("curriculum requires exactly 32 candidates and four seed controls")
    if rdBase.rdkitVersion != lock["required_rdkit"]:
        raise ValueError("curriculum runtime chemistry differs")
    for path, digest in lock["runtime_input_sha256"].items():
        verify_file(Path(path), digest)
    return lock


def run_remote(task, root, artifacts, volume, validate_revision, parallel):
    lock = validate_task(task, root, validate_revision)
    volume.reload()
    output = artifacts / KIND / task["run_id"]
    if (output / "result.json").exists():
        return unseal(output / "result.json")
    # Validate all exact endpoints and gates before any oracle is requested.
    for row in lock["take"]:
        if row["role"] == "seed_control":
            if row["smiles"] != row["original_seed"]:
                raise ValueError("seed control changed its prescribed seed")
        else:
            _, replay = execute_program(
                decode_state(row["trace"]["states"][0]), row["trace"]["actions"]
            )
            if replay["states"] != row["trace"]["states"] or replay["endpoint"] != row["smiles"]:
                raise ValueError("curriculum candidate failed exact trace replay")
            if not property_scorer(row["original_seed"])({"smiles": row["smiles"]})[
                "oracle_eligible"
            ]:
                raise ValueError("locked curriculum endpoint no longer passes its original gate")
    started = perf_counter()
    seal(output / "launch.json", task)
    seal(output / "candidate_lock.json", lock)
    publish_json(
        output / "docking_started.json",
        {"lock_sha256": task["files_sha256"][LOCK], "maximum_attempts": 36},
    )
    volume.commit()
    rows = []
    for row in parallel([{**task, "index": i} for i in range(36)]):
        rows.append(row)
        publish_json(
            output / "progress.json",
            {"completed": len(rows), "total": 36, "updated_at_utc": _stamp()},
        )
        volume.commit()
    rows.sort(key=lambda r: r["index"])
    if [r["index"] for r in rows] != list(range(36)):
        raise ValueError("curriculum is missing a row or duplicated an outcome")
    cells = []
    for cell in lock["cells"]:
        candidates = [r for r in rows if r["cell"] == cell and r["role"] == "candidate"]
        control = next(r for r in rows if r["cell"] == cell and r["role"] == "seed_control")
        best = min(
            (r for r in candidates if r["ds"] is not None),
            key=lambda r: (r["ds"], r["smiles"]),
            default=None,
        )
        champion, curve = None, []
        for i, row in enumerate(candidates, 1):
            if row["ds"] is not None:
                champion = row["ds"] if champion is None else min(champion, row["ds"])
            curve.append(
                {
                    "candidate_calls": i,
                    "calls_including_seed_control": i + 1,
                    "best_eligible_ds": champion,
                }
            )
        cells.append(
            {
                "cell": cell,
                "seed_control": control,
                "best": best,
                "program_curve": curve,
                "improvement_over_seed": None
                if best is None or control["ds"] is None
                else control["ds"] - best["ds"],
                "broad_control_curve": None,
                "adaptive_curve": None,
            }
        )
    result = {
        "schema_version": "t4_program_curriculum_result_v1",
        "task": task,
        "rows": rows,
        "cells": cells,
        "new_oracle_calls": len(rows),
        "seconds": perf_counter() - started,
        "completed_at_utc": _stamp(),
        "evidence": "first scored shared-program development batch; not a matched static/adaptive comparison",
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "openbabel": subprocess.check_output(["obabel", "-V"], text=True).strip(),
        },
        "source_registry": lock["source_registry"],
        "configuration": lock["compute"],
    }
    seal(output / "result.json", result)
    volume.commit()
    return result


def dock_remote(task, root, artifacts, volume, validate_revision, dock):
    shared = {k: v for k, v in task.items() if k != "index"}
    lock = validate_task(shared, root, validate_revision)
    index = task["index"]
    if type(index) is not int or not 0 <= index < 36:
        raise ValueError("docking index outside the fixed curriculum")
    volume.reload()
    output = artifacts / KIND / task["run_id"]
    barrier = json.loads((output / "docking_started.json").read_text())
    if barrier["lock_sha256"] != task["files_sha256"][LOCK]:
        raise ValueError("worker lacks its committed candidate-lock barrier")
    folder = output / "rows" / f"{index:02d}"
    if (folder / "result.json").exists():
        return unseal(folder / "result.json")
    if (folder / "started.json").exists():
        raise RuntimeError("ambiguous prior oracle attempt is not automatically retried")
    candidate = lock["take"][index]
    seal(
        folder / "started.json",
        {"index": index, "candidate_id": candidate["candidate_id"], "started_at_utc": _stamp()},
    )
    volume.commit()
    started = perf_counter()
    tag = f"curriculum_{task['run_id'][:16]}_{index:02d}"
    score = dock(candidate["smiles"], candidate["target"], tag, lock["docking_seed"])
    if score is not None and not math.isfinite(score):
        raise ValueError("oracle returned a nonfinite score")
    poses = {}
    for name in ("l.mol", "l.pdbqt", "o.pdbqt"):
        path = Path("/tmp") / tag / name
        if path.exists():
            destination = folder / name
            shutil.copyfile(path, destination)
            poses[name] = sha256_file(destination)
    result = {
        "index": index,
        "cell": candidate["cell"],
        "target": candidate["target"],
        "role": candidate["role"],
        "smiles": candidate["smiles"],
        "candidate_id": candidate["candidate_id"],
        "ds": score,
        "oracle_protocol": candidate["oracle_protocol"],
        "docking_seed": lock["docking_seed"],
        "seconds": perf_counter() - started,
        "pose_sha256": poses,
        "completed_at_utc": _stamp(),
        "failure": None if score is not None else "oracle_failed",
    }
    seal(folder / "result.json", result)
    volume.commit()
    return result
