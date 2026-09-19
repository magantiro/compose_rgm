"""One scored proposal audit for opt-in region replacement; no training."""

from __future__ import annotations

import json
from collections import Counter
from functools import partial
from time import perf_counter

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.inference_package import software
from compose_v4.experiments.pmo_branch_policy import _frozen_save, worker_identity
from compose_v4.experiments.pmo_macro_probe import DurableScores, make_oracle
from compose_v4.experiments.pmo_option_particles import archive_metrics
from compose_v4.experiments.pmo_option_particles import session as particle_session
from compose_v4.experiments.t4_matched_pilot import _stamp

KIND = "pmo_region_replacement"
APP_NAME = "compose-pmo-region-replacement"
APP = f"modal_apps/{KIND}_app.py"
CONTRACT = f"configs/{KIND}.json"
PREPARED = "diagnostics/pmo_option_particles/prepared.json"
PROTOCOL = "docs/PMO_REGION_REPLACEMENT.md"


def prepare(root):
    old = json.loads((root / "configs/pmo_option_particles.json").read_text())
    verify_file(root / PREPARED, old["prepared"]["sha256"])
    c = {
        k: old[k]
        for k in (
            "task",
            "expected_input_sha256",
            "runtime_contract_sha256",
            "prepared",
            "snapshot_authorization",
        )
    }
    c.update(
        schema_version="region_replacement_probe_contract_v1",
        authorization="2026-09-11 user approved a clean checkpoint commit and the bounded 20-proposal audit",
        seed=20260922,
        particles=20,
        boundaries=1,
        draws_per_bundle=1,
        initial_indices=[i % 5 for i in range(20)],
        new_oracle_limit=20,
        primitive_budget=40,
        include_region_replacement=True,
        law_caches=[],
        reference_training_authorized=False,
        value_training_authorized=False,
        docking_authorized=False,
        winner_input=False,
        protocol={"path": PROTOCOL, "sha256": sha256_file(root / PROTOCOL)},
        compute={
            "max_workers": 20,
            "driver_containers": 1,
            "worker_tasks_max": 20,
            "worker_timeout": 240,
            "driver_timeout": 600,
            "cpu": 1,
            "memory_mib": 8192,
            "retries": 0,
            "heartbeat_seconds": 30,
            "expected_minutes": [2, 6],
            "estimated_usd": [0.5, 4],
            "reserved_cpu_hour_bound": 1.5,
        },
    )
    c["contract_sha256"] = identity(c)
    publish_json(root / CONTRACT, c)
    return c


def load_contract(root):
    c = json.loads((root / CONTRACT).read_text())
    if identity({k: v for k, v in c.items() if k != "contract_sha256"}) != c["contract_sha256"]:
        raise ValueError("replacement probe contract hash mismatch")
    if (
        c["particles"],
        c["boundaries"],
        c["new_oracle_limit"],
        c["draws_per_bundle"],
        c["seed"],
        c["primitive_budget"],
        c["include_region_replacement"],
    ) != (20, 1, 20, 1, 20260922, 40, True):
        raise ValueError("replacement probe changed its bounded recipe")
    if c["initial_indices"] != [i % 5 for i in range(20)] or any(
        c[k]
        for k in (
            "reference_training_authorized",
            "value_training_authorized",
            "docking_authorized",
            "winner_input",
        )
    ):
        raise ValueError("replacement probe exceeds authorization")
    for key in ("prepared", "protocol", "snapshot_authorization"):
        verify_file(root / c[key]["path"], c[key]["sha256"])
    return c


session = partial(particle_session, contract_loader=load_contract, app_path=APP, kind=KIND)


def driver_remote(task, root, artifact_root, volume, validate_revision, parallel):
    with session(task, root, artifact_root, volume, validate_revision) as (c, store, progress):
        previous = store.read("result")
        if previous is not None:
            return previous
        started, at = perf_counter(), _stamp()
        data = json.loads((root / PREPARED).read_text())
        parents = [data["parents"][i] for i in c["initial_indices"]]
        tasks = [
            {**task, "phase": 1, "slot": i, "parent": p, "worker_id": worker_identity(1, i, p)}
            for i, p in enumerate(parents)
        ]
        _frozen_save(store, "task_lock", tasks)
        progress.update(phase="proposals", workers_total=len(tasks), workers_complete=0)
        results = {}
        expected = {t["worker_id"] for t in tasks}
        for result in parallel(tasks):
            key = result["worker_id"]
            if key not in expected or key in results or not result["replay_verified"]:
                raise ValueError("unexpected, repeated or unverified replacement worker")
            if len(result["attempts"]) != 1 or len(result["candidates"]) > 1:
                raise ValueError("replacement probe must return one draw, including failures")
            results[key] = result
            store.save(f"workers_complete/{key}", result)
            progress.update(workers_complete=len(results))
            print(f"[replacement] proposals={len(results)}/{len(tasks)}", flush=True)
        if set(results) != expected:
            raise ValueError("incomplete replacement task census")
        ordered = [results[t["worker_id"]] for t in tasks]
        _frozen_save(store, "candidate_lock", ordered)
        history = dict(data["observed"])
        scores = DurableScores(
            store.output,
            make_oracle(c["task"], root, c),
            c["new_oracle_limit"],
            lambda: store.flush(force=True),
            progress,
        )
        history.update({r["smiles"]: r["score"] for r in scores.rows})
        candidates = [p for r in ordered for p in r["candidates"]]
        requested = sorted({p["smiles"] for p in candidates if p["smiles"] not in history})
        scores.context = {
            "role": "replacement_proposal_audit",
            "lock_path": str(store.output / "candidate_lock.json"),
        }
        progress.update(
            phase="oracle", locked_candidates=len(candidates), new_batch_calls=len(requested)
        )
        for smiles, value in zip(requested, scores.score_many(requested), strict=True):
            history[smiles] = value["desirability"]
        for p in candidates:
            p["score"] = history[p["smiles"]]
        archive = {p["smiles"]: p["score"] for p in parents + candidates}
        result = {
            "schema_version": "region_replacement_probe_result_v1",
            "status": "complete_development",
            "run_id": task["run_id"],
            "configuration": c,
            "image_revision": task["image_revision"],
            "initial_parents": parents,
            "workers": ordered,
            "candidates": candidates,
            "attempted_options": dict(
                Counter(
                    a["bundle"]["option"] if a["bundle"] else "no_bundle"
                    for r in ordered
                    for a in r["attempts"]
                )
            ),
            "archive_metrics": archive_metrics(archive),
            "new_oracle_calls": scores.meter.spent,
            "oracle_rows": scores.rows,
            "historical_unique_labels": len(data["observed"]),
            "seconds": perf_counter() - started,
            "started_at": at,
            "finished_at": _stamp(),
            "software": software(),
            "io_timings": dict(store.timings),
            "hardware": {"device": "cpu", "threads_per_worker": 1, "neural_dtype": "float32"},
            "interpretation": "warm-development proposal audit; no paired ablation, learned guidance or matched PMO AUC claim",
        }
        store.save("result", result)
        return result
