"""Score the locked corrected multi-site proposal pools without surrogate selection."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from time import perf_counter

from rdkit import rdBase

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.experiments.t4_partial_docking import dock_saved_row
from compose_v4.experiments.t4_winner_refinement import property_scorer, replicate_summary
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.trace_shard import decode_state

KIND = "t4_program_pool"
LOCK = "configs/t4_program_pool_lock.json"
APP = "modal_apps/t4_program_pool_app.py"


def prepare(root: Path) -> dict:
    """Reuse existing locked candidates; do not regenerate or select by scores."""
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("pool preparation requires pinned RDKit 2024.03.5")
    seed = json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())[0]
    score = property_scorer(seed["smiles"])
    controls_path = root / "diagnostics/t4_winner_refinement/attempt_1/docked_controls.json"
    controls = unseal(controls_path)
    if len(controls) != 3 or len({r["smiles"] for r in controls}) != 1:
        raise ValueError("requires the three already completed winner controls")
    winner = controls[0]["smiles"]
    candidates, inputs, memberships = {}, {}, []
    started = perf_counter()
    for attempt, context in ((2, "original_seed"), (3, "post_linker")):
        directory = root / f"diagnostics/multi_site_proposal_probe/attempt_{attempt}"
        result_path = directory / "result.json"
        report = json.loads(result_path.read_text())
        inputs[str(result_path.relative_to(root))] = sha256_file(result_path)
        for arm in report["arms"]:
            if arm["context"] != context:
                continue
            path = directory / arm["pool_path"]
            verify_file(path, arm["pool_sha256"])
            inputs[str(path.relative_to(root))] = arm["pool_sha256"]
            pool = json.loads(gzip.decompress(path.read_bytes()))
            for smiles, index in pool["unique_endpoint_first_attempt"].items():
                entry = pool["rows"][index]
                trace = entry["receipt"]
                if entry["status"] != "complete" or trace["endpoint"] != smiles:
                    raise ValueError("pool candidate has no complete matching execution")
                properties = score({"smiles": smiles})
                membership = {
                    "context": context,
                    "arm": arm["arm"],
                    "attempt": index,
                    "pool_path": str(path.relative_to(root)),
                    "smiles": smiles,
                    "oracle_eligible": properties["oracle_eligible"],
                }
                memberships.append(membership)
                if not properties["oracle_eligible"] or smiles == winner:
                    continue
                if smiles not in candidates:
                    _, replay = execute_program(decode_state(trace["states"][0]), trace["actions"])
                    if replay["states"] != trace["states"] or replay["endpoint"] != smiles:
                        raise ValueError("locked program does not replay exactly")
                    candidates[smiles] = {
                        **properties,
                        "state": trace["states"][-1],
                        "trace": trace,
                        "ancestry": [],
                        "docking_seed": 1701,
                    }
                candidates[smiles]["ancestry"].append(membership)
    take = [candidates[key] for key in sorted(candidates)]
    if not take or len(take) + 2 > 66:
        raise ValueError("complete pool plus two repeats exceeds the approved ceiling")
    prior = json.loads((root / "configs/t4_winner_refinement.json").read_text())
    return {
        "schema_version": "t4_program_pool_lock_v1",
        "required_rdkit": "2024.03.5",
        "task": {
            "smiles": seed["smiles"],
            "target": "parp1",
            "delta": 0.4,
            "expected_input_sha256": prior["expected_input_sha256"],
        },
        "take": take,
        "memberships": memberships,
        "inputs_sha256": inputs,
        "winner_controls": {
            "path": str(controls_path.relative_to(root)),
            "sha256": sha256_file(controls_path),
        },
        "compute": {
            "new_oracle_call_limit": len(take) + 2,
            "driver_containers": 1,
            "worker_containers": 8,
            "worker_cpu": 1,
            "worker_timeout_seconds": 480,
            "driver_timeout_seconds": 3600,
            "reserved_cost_usd": 20,
            "automatic_retries": 0,
            "expected_wall_seconds": [60, 900],
            "restart_unit": "sealed row receipt; ambiguous rows not retried",
        },
        "docking": prior["docking"],
        "prepared_at_utc": _stamp(),
        "preparation_seconds": perf_counter() - started,
        "evidence": "winner-informed development; all unique eligible endpoints from corrected locked pools",
        "exclusions": [
            "confounded attempt_1",
            "attempt_2 post-linker context",
            "ineligible endpoints",
            "winner reused from three paid controls",
        ],
        "selection": "all unique eligible; no surrogate or docking-based selection",
        "preparation_implementation_sha256": sha256_file(Path(__file__)),
        "seed_manifest_sha256": sha256_file(root / "docs/GENMOL_T4_SEEDS.json"),
        "software": {"rdkit": rdBase.rdkitVersion},
        "benchmark_claim": False,
    }


def validate_task(task, root, validate_revision):
    validate_revision(task["image_revision"])
    verify_file(root / APP, task["program_app_sha256"])
    verify_file(root / LOCK, task["lock_sha256"])
    if (
        identity(
            {
                key: task[key]
                for key in ("lock_sha256", "program_app_sha256", "app_sha256", "image_revision")
            }
        )
        != task["run_id"]
    ):
        raise ValueError("program-pool run identity mismatch")
    return unseal(root / LOCK)


def run_remote(task, root, artifacts, volume, validate_revision, parallel):
    lock = validate_task(task, root, validate_revision)
    output = artifacts / KIND / task["run_id"]
    volume.reload()
    if (output / "result.json").exists():
        return unseal(output / "result.json")
    verify_file(root / lock["winner_controls"]["path"], lock["winner_controls"]["sha256"])
    controls = unseal(root / lock["winner_controls"]["path"])
    started = perf_counter()
    seal(output / "launch.json", task)
    volume.commit()

    def batch(stage, candidates):
        folder = artifacts / KIND / stage / task["run_id"]
        actual_lock = {
            "task": lock["task"],
            "required_rdkit": lock["required_rdkit"],
            "take": candidates,
        }
        path = folder / "candidate_lock.json"
        if path.exists() and unseal(path) != actual_lock:
            raise ValueError("persisted program-pool lock changed")
        seal(path, actual_lock)
        digest = sha256_file(path)
        publish_json(
            folder / "docking_started.json",
            {"candidate_lock_sha256": digest, "maximum_attempts": len(candidates)},
        )
        volume.commit()
        rows = []
        for row in parallel(
            [
                {**task, "stage": stage, "index": i, "candidate_lock_sha256": digest}
                for i in range(len(candidates))
            ]
        ):
            rows.append(row)
            publish_json(
                output / "progress.json",
                {
                    "stage": stage,
                    "completed": len(rows),
                    "total": len(candidates),
                    "updated_at_utc": _stamp(),
                },
            )
            volume.commit()
        rows.sort(key=lambda row: row["index"])
        if [r["index"] for r in rows] != list(range(len(candidates))):
            raise ValueError("missing or duplicate docking result")
        return [{**candidate, **row} for candidate, row in zip(candidates, rows, strict=True)]

    try:
        rows = batch("pool", lock["take"])
        best = min(
            (r for r in rows if r["ds"] is not None),
            key=lambda r: (r["ds"], r["smiles"]),
            default=None,
        )
        repeats = (
            []
            if best is None
            else batch("confirmation", [{**best, "docking_seed": s} for s in (1702, 1703)])
        )
        result = {
            "schema_version": "t4_program_pool_result_v1",
            "lock_sha256": task["lock_sha256"],
            "rows": rows,
            "best": best,
            "confirmation": repeats,
            "comparison": {"status": "no_scored_candidates"}
            if best is None
            else replicate_summary(controls, [best, *repeats]),
            "new_oracle_calls": len(rows) + len(repeats),
            "reused_control_calls": 3,
            "seconds": perf_counter() - started,
            "completed_at_utc": _stamp(),
            "task": task,
            "configuration": lock["compute"],
            "benchmark_claim": False,
            "model_training": False,
        }
        if result["new_oracle_calls"] > lock["compute"]["new_oracle_call_limit"]:
            raise RuntimeError("program-pool docking allowance exceeded")
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
    config = validate_task(task, root, validate_revision)
    stage = task["stage"]
    if stage not in ("pool", "confirmation"):
        raise ValueError("unexpected program-pool stage")
    folder = artifacts / KIND / stage / task["run_id"]
    volume.reload()
    verify_file(folder / "candidate_lock.json", task["candidate_lock_sha256"])
    lock = unseal(folder / "candidate_lock.json")
    candidate = lock["take"][task["index"]]
    expected = 1701 if stage == "pool" else 1702 + task["index"]
    if candidate["docking_seed"] != expected:
        raise ValueError("candidate docking seed differs from declared schedule")

    def with_pose(smiles, tag):
        tag += f"_{stage}"
        value = dock(smiles, tag, expected)
        destination = folder / "poses" / f"{task['index']:02}"
        destination.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name in ("l.mol", "l.pdbqt", "o.pdbqt"):
            path = Path("/tmp") / tag / name
            if path.exists():
                shutil.copyfile(path, destination / name)
                hashes[name] = sha256_file(destination / name)
        publish_json(
            destination / "manifest.json", {"smiles": smiles, "seed": expected, "sha256": hashes}
        )
        return value

    return dock_saved_row(
        task,
        root,
        artifacts,
        volume,
        validate_revision,
        with_pose,
        run_kind=f"{KIND}/{stage}",
        batch_limit=len(config["take"]) if stage == "pool" else 2,
    )
