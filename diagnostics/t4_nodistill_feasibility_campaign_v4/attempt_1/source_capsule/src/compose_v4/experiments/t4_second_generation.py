"""Paired locked-pool scoring and fresh confirmations, at most 73 queries."""

from __future__ import annotations

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

KIND = "t4_second_generation"
LOCK = "configs/t4_second_generation_lock.json"
APP = "modal_apps/t4_second_generation_app.py"
ARMS = ("score_rank", "score_blind")


def validate_rows(candidates, rows):
    if len(candidates) != len(rows):
        raise ValueError("missing or extra charged outcomes")
    for index, (candidate, row) in enumerate(zip(candidates, rows, strict=True)):
        if row["index"] != index:
            raise ValueError("changed outcome order or duplicated receipt")
        for field in (
            "candidate_id",
            "cell",
            "target",
            "smiles",
            "oracle_protocol",
            "docking_seed",
        ):
            if row[field] != candidate[field]:
                raise ValueError(f"row {index}: changed {field}")
        if row["ds"] is not None and not math.isfinite(row["ds"]):
            raise ValueError("nonfinite docking outcome")
        if (row["ds"] is None) != (row["failure"] == "oracle_failed"):
            raise ValueError("inconsistent docking failure")


def confirmation_candidates(lock, rows):
    """Select within each locked arm, then deduplicate physical repeats."""
    validate_rows(lock["take"], rows)
    scored = {r["candidate_id"]: r for r in rows}
    repeats = {}

    def add(candidate, role):
        for seed in (1702, 1703):
            body = {k: candidate[k] for k in ("cell", "target", "smiles", "oracle_protocol")}
            body["docking_seed"] = seed
            key = identity(body)
            if key not in repeats:
                repeats[key] = {**body, "candidate_id": key, "roles": []}
            repeats[key]["roles"].append(role)

    for cell in lock["cells"]:
        for arm in ARMS:
            requested = [m for m in lock["memberships"] if m["cell"] == cell and m["arm"] == arm]
            best = min(
                (
                    scored[m["query_id"]]
                    for m in requested
                    if scored[m["query_id"]]["ds"] is not None
                ),
                key=lambda r: (r["ds"], r["smiles"]),
                default=None,
            )
            if best is not None:
                add(best, {"cell": cell, "arm": arm, "role": "arm_champion"})
        add(lock["incumbents"][cell], {"cell": cell, "role": "incumbent"})
    result = list(repeats.values())
    if len(rows) + len(result) > lock["new_oracle_call_limit"]:
        raise ValueError("confirmation allocation exceeds the approved call budget")
    return result


def summarize(lock, rows, confirmation):
    """Keep repeat-only comparisons separate from selected first scores."""
    validate_rows(lock["take"], rows)
    expected = confirmation_candidates(lock, rows)
    validate_rows(expected, confirmation)
    scored = {r["candidate_id"]: r for r in rows}

    def repeat_scores(cell, smiles):
        return [r["ds"] for r in confirmation if r["cell"] == cell and r["smiles"] == smiles]

    cells = []
    for cell in lock["cells"]:
        incumbent = lock["incumbents"][cell]
        controls = repeat_scores(cell, incumbent["smiles"])
        control_mean = sum(controls) / 2 if len(controls) == 2 and None not in controls else None
        arms = []
        for arm in ARMS:
            members = [m for m in lock["memberships"] if m["cell"] == cell and m["arm"] == arm]
            outcomes = [scored[m["query_id"]] for m in members]
            best = min(
                (r for r in outcomes if r["ds"] is not None),
                key=lambda r: (r["ds"], r["smiles"]),
                default=None,
            )
            repeats = [] if best is None else repeat_scores(cell, best["smiles"])
            mean = sum(repeats) / 2 if len(repeats) == 2 and None not in repeats else None
            current, curve = incumbent["historical_ds"], []
            for i, row in enumerate(outcomes, 1):
                if row["ds"] is not None:
                    current = min(current, row["ds"])
                curve.append({"new_logical_queries": i, "best_including_incumbent": current})
            arms.append(
                {
                    "arm": arm,
                    "candidate_calls": len(outcomes),
                    "unfilled_slots": 8 - len(outcomes),
                    "failures": sum(r["ds"] is None for r in outcomes),
                    "best": best,
                    "repeat_scores": repeats,
                    "repeat_mean": mean,
                    "repeat_improvement_over_incumbent": None
                    if mean is None or control_mean is None
                    else control_mean - mean,
                    "first_scores": [r["ds"] for r in outcomes],
                    "curve": curve,
                }
            )
        cells.append(
            {
                "cell": cell,
                "incumbent": incumbent,
                "incumbent_repeat_scores": controls,
                "incumbent_repeat_mean": control_mean,
                "arms": arms,
            }
        )
    return cells


def validate_task(task, root, validate_revision):
    validate_revision(task["image_revision"])
    for path, digest in task["files_sha256"].items():
        verify_file(root / path, digest)
    if identity({k: v for k, v in task.items() if k != "run_id"}) != task["run_id"]:
        raise ValueError("second-generation task identity changed")
    lock = unseal(root / LOCK)
    if len(lock["take"]) != 49 or lock["new_oracle_call_limit"] != 73:
        raise ValueError("second-generation allocation differs from the approved lock")
    if rdBase.rdkitVersion != lock["required_rdkit"]:
        raise ValueError("second-generation runtime chemistry differs")
    for path, digest in lock["runtime_input_sha256"].items():
        verify_file(Path(path), digest)
    return lock


def run_remote(task, root, artifacts, volume, validate_revision, parallel):
    lock = validate_task(task, root, validate_revision)
    volume.reload()
    output = artifacts / KIND / task["run_id"]
    if (output / "result.json").exists():
        return unseal(output / "result.json")
    for row in [*lock["take"], *lock["incumbents"].values()]:
        _, replay = execute_program(
            decode_state(row["trace"]["states"][0]), row["trace"]["actions"]
        )
        if replay["states"] != row["trace"]["states"] or replay["endpoint"] != row["smiles"]:
            raise ValueError("locked second-generation trace does not replay")
        if not property_scorer(row["original_seed"])({"smiles": row["smiles"]})["oracle_eligible"]:
            raise ValueError("locked endpoint failed its original-seed gate")
    started = perf_counter()
    seal(output / "launch.json", task)
    volume.commit()

    def batch(stage, candidates):
        folder = output / stage
        body = {
            "parent_lock_sha256": task["files_sha256"][LOCK],
            "stage": stage,
            "take": candidates,
        }
        path = folder / "candidate_lock.json"
        if path.exists() and unseal(path) != body:
            raise ValueError("persisted second-generation stage changed")
        seal(path, body)
        digest = sha256_file(path)
        volume.commit()
        outcomes = []
        for row in parallel(
            [
                {**task, "stage": stage, "index": i, "stage_lock_sha256": digest}
                for i in range(len(candidates))
            ]
        ):
            outcomes.append(row)
            publish_json(
                output / "progress.json",
                {
                    "stage": stage,
                    "completed": len(outcomes),
                    "total": len(candidates),
                    "updated_at_utc": _stamp(),
                },
            )
            volume.commit()
        outcomes.sort(key=lambda r: r["index"])
        validate_rows(candidates, outcomes)
        seal(folder / "result.json", outcomes)
        volume.commit()
        return outcomes

    try:
        rows = batch("first", lock["take"])
        confirmations = batch("confirmation", confirmation_candidates(lock, rows))
        result = {
            "schema_version": "t4_second_generation_result_v1",
            "task": task,
            "rows": rows,
            "confirmation": confirmations,
            "cells": summarize(lock, rows, confirmations),
            "new_oracle_calls": len(rows) + len(confirmations),
            "seconds": perf_counter() - started,
            "completed_at_utc": _stamp(),
            "software": {
                "python": platform.python_version(),
                "rdkit": rdBase.rdkitVersion,
                "openbabel": subprocess.check_output(["obabel", "-V"], text=True).strip(),
            },
            "configuration": lock["compute"],
            "evidence": "paired measured-history allocation development; no matched external benchmark claim",
        }
        seal(output / "result.json", result)
        volume.commit()
        return result
    except Exception as error:
        publish_json(
            output / "failure.json",
            {"error_type": type(error).__name__, "message": str(error), "at": _stamp()},
        )
        volume.commit()
        raise


def dock_remote(task, root, artifacts, volume, validate_revision, dock):
    shared = {k: v for k, v in task.items() if k not in ("stage", "index", "stage_lock_sha256")}
    config = validate_task(shared, root, validate_revision)
    stage, index = task["stage"], task["index"]
    if stage not in ("first", "confirmation") or type(index) is not int:
        raise ValueError("invalid stage or docking index")
    volume.reload()
    folder = artifacts / KIND / task["run_id"] / stage
    verify_file(folder / "candidate_lock.json", task["stage_lock_sha256"])
    lock = unseal(folder / "candidate_lock.json")
    if lock["parent_lock_sha256"] != task["files_sha256"][LOCK] or lock["stage"] != stage:
        raise ValueError("worker stage lacks its committed parent lock")
    if not 0 <= index < len(lock["take"]) or len(lock["take"]) > (49 if stage == "first" else 24):
        raise ValueError("worker index or stage exceeds the budget")
    candidate = lock["take"][index]
    if stage == "first" and lock["take"] != config["take"]:
        raise ValueError("worker first-stage candidates changed")
    if stage == "confirmation":
        first = unseal(folder.parent / "first/result.json")
        if lock["take"] != confirmation_candidates(config, first):
            raise ValueError("worker confirmation selection changed")
    output = folder / "rows" / f"{index:02d}"
    if (output / "result.json").exists():
        result = unseal(output / "result.json")
        for field in (
            "candidate_id",
            "cell",
            "target",
            "smiles",
            "oracle_protocol",
            "docking_seed",
        ):
            if result[field] != candidate[field]:
                raise ValueError("cached worker receipt differs from the locked query")
        return result
    if (output / "started.json").exists():
        raise RuntimeError("ambiguous prior oracle attempt is not automatically retried")
    seal(
        output / "started.json",
        {"candidate_id": candidate["candidate_id"], "index": index, "started_at_utc": _stamp()},
    )
    volume.commit()
    started = perf_counter()
    tag = f"secondgen_{task['run_id'][:16]}_{stage}_{index:02d}"
    score = dock(candidate["smiles"], candidate["target"], tag, candidate["docking_seed"])
    if score is not None and not math.isfinite(score):
        raise ValueError("oracle returned a nonfinite score")
    poses = {}
    for name in ("l.mol", "l.pdbqt", "o.pdbqt"):
        path = Path("/tmp") / tag / name
        if path.exists():
            shutil.copyfile(path, output / name)
            poses[name] = sha256_file(output / name)
    result = {
        k: candidate[k]
        for k in ("candidate_id", "cell", "target", "smiles", "oracle_protocol", "docking_seed")
    }
    result.update(
        index=index,
        ds=score,
        failure=None if score is not None else "oracle_failed",
        pose_sha256=poses,
        seconds=perf_counter() - started,
        completed_at_utc=_stamp(),
    )
    seal(output / "result.json", result)
    volume.commit()
    return result
