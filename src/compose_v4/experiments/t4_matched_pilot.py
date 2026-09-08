"""Bounded paired cold-start audit of the existing T4 proposal path.

No docking signal enters preparation. Both immutable arm locks precede any
oracle call. This is development evidence, not a benchmark selection procedure.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from compose_v4.control.region_rewrite import kl_tilt
from compose_v4.experiments.continuation_profile import (
    ExecutorMeter,
    canonical_bytes,
    publish_json,
    verify_file,
)

CONTRACT_PATH = "configs/t4_matched_pilot.json"
ARMS = ("reference", "committor")
PLAN_FIELDS = (
    "bundle_id",
    "parent",
    "parent_lineage_id",
    "region_atoms",
    "region_id",
    "option",
    "q_option",
    "option_probabilities",
    "q_region",
    "q_region_base",
    "q_region_floor",
    "r_release",
    "region_draws",
    "particles",
)


def primitive_distribution(reference, values, guidance, kappa=1.0, exploration=0.1):
    """Exactly the existing committor mixture, or its unchanged reference law."""
    p = np.asarray(reference, dtype=float)
    if p.ndim != 1 or not p.size or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("reference must be a finite nonnegative probability vector")
    if not np.isclose(p.sum(), 1.0, rtol=0, atol=1e-10):
        raise ValueError("reference must be normalized")
    if kappa != 1.0 or not 0 <= exploration <= 1:
        raise ValueError("kappa is frozen at 1; exploration must lie in [0,1]")
    if guidance == "reference":
        return p.copy()
    if guidance != "committor":
        raise ValueError(f"unknown primitive guidance: {guidance!r}")
    h = np.asarray(values, dtype=float)
    if h.shape != p.shape or not np.isfinite(h).all() or (h < 0).any():
        raise ValueError("committor values must be finite, nonnegative and aligned")
    q, _, _, _ = kl_tilt(p, h, kappa=kappa)
    q = (1 - exploration) * q + exploration * p
    return q / q.sum()


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def seal(path: Path, payload: dict) -> str:
    return publish_json(
        path,
        {
            "payload": payload,
            "payload_sha256": hashlib.sha256(canonical_bytes(payload)).hexdigest(),
        },
    )


def unseal(path: Path) -> dict:
    envelope = json.loads(path.read_text())
    payload = envelope["payload"]
    if hashlib.sha256(canonical_bytes(payload)).hexdigest() != envelope["payload_sha256"]:
        raise ValueError(f"corrupt complete unit: {path}")
    return payload


def verify_pair(locks: dict) -> None:
    plans = []
    for arm in ARMS:
        lock = locks[arm]
        if lock["oracle_calls"] != 0 or lock["round"] != 1:
            raise ValueError("only zero-oracle, one-round candidate locks are permitted")
        candidates = lock["take"]
        if len(candidates) > 20 or len({c["smiles"] for c in candidates}) != len(candidates):
            raise ValueError("candidate lock exceeds budget or has canonical duplicates")
        for c in candidates:
            if (
                c["option"] in ("build_ring_system", "build_fused_ring")
                and not c["program_complete"]
            ):
                raise ValueError("incomplete compound program in oracle lock")
        plans.append([{f: b[f] for f in PLAN_FIELDS} for b in lock["bundles"]])
    if canonical_bytes(plans[0]) != canonical_bytes(plans[1]):
        raise ValueError("paired arms did not receive identical region/option draws")
    if locks[ARMS[0]]["input_sha256"] != locks[ARMS[1]]["input_sha256"]:
        raise ValueError("paired input identities differ")


def run_pair(task, prepare, dock, output: Path, *, commit=lambda: None, progress=None):
    """Prepare and seal both arms before invoking the supplied oracle callback.

    Completed arm locks and parent units are reusable. An interrupted oracle
    attempt is not repeated: it requires a human accounting decision first.
    """
    progress = {} if progress is None else progress
    locks, lock_hashes = {}, {}
    for arm in ARMS:
        arm_task = {**task, "arm": arm, "primitive_guidance": arm, "prepare_only": True}
        arm_dir = output / arm
        path = arm_dir / "candidate_lock.json"
        progress.update(phase="prepare", arm=arm)
        if path.exists():
            locked = unseal(path)
            if locked["task"] != arm_task:
                raise ValueError("cached lock task differs; no implicit regeneration")
        else:
            if list(arm_dir.glob("executor_attempts_*.json")):
                raise RuntimeError("failed preparation receipt exists; audit before retry")
            parent_cache = {}
            for parent_path in sorted((arm_dir / "parents").glob("*.json")):
                unit = unseal(parent_path)
                if unit["task"] != arm_task:
                    raise ValueError("cached parent task differs")
                parent_cache[unit["parent_index"]] = unit
            # Count previous complete work against the same ceiling on resume.
            prior_calls = max(
                (u["cumulative_executor_calls"] for u in parent_cache.values()), default=0
            )
            meter = ExecutorMeter(task["max_executor_applications"] - prior_calls)
            started = time.perf_counter()

            def checkpoint(
                unit,
                parent_cache=parent_cache,
                arm_dir=arm_dir,
                arm_task=arm_task,
                prior_calls=prior_calls,
                meter=meter,
            ):
                if unit["parent_index"] in parent_cache:
                    return
                seal(
                    arm_dir / "parents" / f"{unit['parent_index']:02d}.json",
                    {
                        **unit,
                        "task": arm_task,
                        "cumulative_executor_calls": prior_calls + meter.calls,
                    },
                )
                progress.update(
                    parent_complete=unit["parent_index"] + 1,
                    executor_calls=prior_calls + meter.calls,
                )
                commit()

            try:
                with meter.instrument():
                    locked = prepare(arm_task, checkpoint, parent_cache)
            finally:
                # Preserve failed/abstaining executor attempts too; not per-call I/O.
                attempt_path = arm_dir / f"executor_attempts_{prior_calls}.json"
                if attempt_path.exists():
                    raise RuntimeError("partial attempt exists; audit before retry")
                publish_json(attempt_path, {"attempts": meter.attempts, "prior_calls": prior_calls})
                commit()
            locked["total_public_executor_calls"] = prior_calls + meter.calls
            locked["prepare_wall_seconds"] = time.perf_counter() - started
            seal(path, locked)
            commit()
        locks[arm] = locked
        lock_hashes[arm] = hashlib.sha256(path.read_bytes()).hexdigest()

    verify_pair(locks)
    barrier = {"locked_at_utc": _stamp(), "candidate_lock_sha256": lock_hashes}
    barrier_path = output / "oracle_barrier.json"
    if not barrier_path.exists():
        publish_json(barrier_path, barrier)
        commit()
    else:
        barrier = json.loads(barrier_path.read_text())
        if barrier["candidate_lock_sha256"] != lock_hashes:
            raise ValueError("candidate locks changed after the oracle barrier")
    results = {}
    for arm in ARMS:
        locked = locks[arm]
        result_path, attempt_path = (
            output / arm / "docking.json",
            output / arm / "docking_started.json",
        )
        if result_path.exists():
            result = unseal(result_path)
        else:
            if attempt_path.exists():
                raise RuntimeError(f"{arm}: interrupted oracle attempt; no automatic re-docking")
            progress.update(phase="docking", arm=arm, oracle_attempts=len(locked["take"]))
            publish_json(
                attempt_path,
                {**barrier, "started_at_utc": _stamp(), "attempts": len(locked["take"])},
            )
            commit()
            started = time.perf_counter()
            scores = dock([c["smiles"] for c in locked["take"]], arm)
            if len(scores) != len(locked["take"]) or any(
                score is not None and not np.isfinite(score) for score in scores
            ):
                raise ValueError("oracle returned missing rows or nonfinite scores")
            records = [{**c, "ds": ds} for c, ds in zip(locked["take"], scores)]
            feasible = [
                c
                for c in records
                if c["v"] <= 0 and c["ds"] is not None and c["smiles"] != task["smiles"]
            ]
            best = min(feasible, key=lambda c: c["ds"]) if feasible else None
            result = {
                "arm": arm,
                "candidate_lock_sha256": lock_hashes[arm],
                "docked": records,
                "oracle_attempts": len(scores),
                "oracle_failures": sum(s is None for s in scores),
                "n_feasible": len(feasible),
                "best": best,
                "docking_seconds": time.perf_counter() - started,
                "completed_at_utc": _stamp(),
                "selected_options": dict(sorted(Counter(c["option"] for c in records).items())),
            }
            seal(result_path, result)
            commit()
        if result["candidate_lock_sha256"] != lock_hashes[arm]:
            raise ValueError("docking result is not bound to its candidate lock")
        results[arm] = result
    return {
        "schema_version": "t4_matched_cold_start_v1",
        "status": "complete",
        "task": task,
        "candidate_lock_sha256": lock_hashes,
        "arms": results,
        "interpretation_scope": "single inspected development source; no online value learning",
    }


def run_remote(
    task,
    repo_root,
    artifact_root,
    volume,
    runtime_factory,
    validate_revision,
    prepare,
    dock,
    *,
    contract_path=CONTRACT_PATH,
    run_kind="t4_matched_pilot",
    runner=run_pair,
):
    """Thin remote provenance/heartbeat boundary; never changes the proposal law."""
    import resource
    import threading

    validate_revision(task["image_revision"])
    verify_file(repo_root / "modal_apps/genmol_t4_opt_app.py", task["app_sha256"])
    path = repo_root / contract_path
    verify_file(path, task["contract_sha256"])
    contract = json.loads(path.read_text())
    body = {k: v for k, v in contract.items() if k != "contract_sha256"}
    if hashlib.sha256(canonical_bytes(body)).hexdigest() != contract["contract_sha256"]:
        raise ValueError("pilot contract self-hash mismatch")
    identity = {
        "contract_sha256": task["contract_sha256"],
        "image_revision_sha256": task["image_revision"]["image_revision_sha256"],
        "app_sha256": task["app_sha256"],
    }
    if hashlib.sha256(canonical_bytes(identity)).hexdigest() != task["run_id"]:
        raise ValueError("pilot run identity mismatch")
    output = artifact_root / run_kind / task["run_id"]
    started = time.perf_counter()
    progress = {"phase": "initialization", "run_id": task["run_id"]}
    stop, mutex = threading.Event(), threading.RLock()

    def commit():
        with mutex:
            volume.commit()

    def heartbeat():
        while not stop.wait(30):
            publish_json(
                output / "heartbeat.json",
                {
                    **progress,
                    "updated_at_utc": _stamp(),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            commit()

    publish_json(output / "launch.json", task)
    commit()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        docking_only = contract.get("docking_only", False)
        if docking_only and runtime_factory is not None:
            raise ValueError("saved docking must not initialize the generator")
        runtime = None if docking_only else runtime_factory()
        seed_path = repo_root / "docs/GENMOL_T4_SEEDS.json"
        input_paths = {
            "seed_manifest": seed_path,
            "qvina02": Path("/opt/dock/qvina02"),
            "receptor": Path("/opt/dock/receptors/parp1.pdbqt"),
        }
        if runtime is not None:
            input_paths.update(
                r_theta_checkpoint=Path(runtime["model_checkpoint"]),
                r_theta_run_paths=Path(runtime["run_paths"]),
                committor=Path("/artifacts/region_committor/committor_bellman_v1.pt"),
            )
        for name, input_path in input_paths.items():
            verify_file(input_path, contract["expected_input_sha256"][name])
        seed = json.loads(seed_path.read_text())[0]
        if seed["target"] != "parp1":
            raise ValueError("pilot requires PARP1 seed0")
        actual_task = {
            **contract["task"],
            "smiles": seed["smiles"],
            "target": seed["target"],
            "code_revision": task["image_revision"]["commit"],
            "cell": task["run_id"],
            "expected_input_sha256": contract["expected_input_sha256"],
        }
        gate = {
            "status": "docking_only_inputs_verified"
            if docking_only
            else "existing_frozen_runtime_gates_passed",
            "initialization_seconds": time.perf_counter() - started,
            "model_parameter_dtypes": sorted({str(p.dtype) for p in runtime["model"].parameters()})
            if runtime
            else [],
            "input_paths": {name: str(p) for name, p in sorted(input_paths.items())},
            "input_sha256": contract["expected_input_sha256"],
        }
        publish_json(output / "runtime_gate.json", gate)
        commit()
        result = runner(actual_task, prepare, dock, output, commit=commit, progress=progress)
        result.update(
            code_revision=actual_task["code_revision"],
            configuration=contract,
            runtime_gate=gate,
            elapsed_seconds=time.perf_counter() - started,
            peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            completed_at_utc=_stamp(),
        )
        publish_json(output / "result.json", result)
        progress.update(phase="complete")
        return result
    except Exception as error:
        publish_json(
            output / "failure.json",
            {
                **progress,
                "error_type": type(error).__name__,
                "message": str(error),
                "recorded_at_utc": _stamp(),
            },
        )
        raise
    finally:
        stop.set()
        thread.join(timeout=5)
        publish_json(
            output / "heartbeat.json",
            {
                **progress,
                "updated_at_utc": _stamp(),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        commit()
