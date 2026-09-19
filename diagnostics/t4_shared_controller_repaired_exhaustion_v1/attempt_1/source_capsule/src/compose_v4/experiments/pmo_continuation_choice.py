"""Bounded actual-score continuation choice over the unchanged option generator."""

from __future__ import annotations

import json
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.control.continuation_choice import choose_continuation
from compose_v4.control.docking_value import identity
from compose_v4.control.molecular_search_codec import decode_search_state, encode_search_state
from compose_v4.control.molecular_task_search import MolecularSearchState
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.pmo_branch_policy import _frozen_save, worker_identity
from compose_v4.experiments.pmo_macro_probe import DurableScores, make_oracle
from compose_v4.experiments.t4_matched_pilot import _stamp, unseal
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

KIND = "pmo_continuation_choice"
CONTRACT = f"configs/{KIND}.json"
PREPARED = f"diagnostics/{KIND}/prepared.json"
APP = f"modal_apps/{KIND}_app.py"
APP_NAME = "compose-pmo-continuation-choice"


def prepare(root: Path) -> dict:
    paths = [
        "diagnostics/pmo_online_policy/prepared.json",
        "diagnostics/pmo_online_policy/result_sealed.json",
        "configs/pmo_macro_probe.json",
        "configs/pmo_online_policy.json",
        "diagnostics/pmo_public_winner_recovery/scores.json",
    ]
    inputs = {p: sha256_file(root / p) for p in paths}
    prior = json.loads((root / paths[0]).read_text())
    result = unseal(root / paths[1])
    known = dict(prior["observed"])
    for arm in result["arms"].values():
        for smiles, score in arm["observed"].items():
            if smiles in known and known[smiles] != score:
                raise ValueError("historical PMO label disagreement")
            known[smiles] = score
    best = max(result["arms"]["balanced"]["archive"], key=lambda r: (r["score"], r["id"]))
    parents = [{**best, "development_source": "current_best"}]
    roots = json.loads((root / paths[2]).read_text())["roots"]
    for index, row in enumerate(roots):
        graph = decode_state(row["state"])
        smiles = canonical_state_key(graph)
        if smiles != row["smiles"] or smiles not in known:
            raise ValueError("original exact root identity/known label missing")
        node = MolecularSearchState.start(graph, budget=11, root_id=identity(row))
        parents.append(
            {
                "id": f"original_root_{index}",
                "development_source": f"original_root_{index}",
                "node": encode_search_state(node),
                "smiles": smiles,
                "score": known[smiles],
                "chain": [],
                "primitives": 0,
            }
        )
    if len(parents) != 5:
        raise ValueError("comparison requires current best and exactly four saved roots")
    for parent in parents:
        if canonical_state_key(decode_search_state(parent["node"]).graph) != parent["smiles"]:
            raise ValueError("saved parent exact-state/canonical identity mismatch")
    public = json.loads((root / paths[4]).read_text())
    # This reference is not supplied to the sampled proposal generator or chooser.
    target = next(r for r in public["oracle_ledger"] if r["role"] == "public_target")
    data = {
        "schema_version": "pmo_continuation_prepared_v1",
        "parents": parents,
        "observed": known,
        "historical_calls": prior["historical_calls"] + result["new_oracle_calls"],
        "separate_public_diagnostic_calls": public["new_oracle_calls"],
        "public_development_reference": target,
        "input_hashes": inputs,
    }
    publish_json(root / PREPARED, data)
    runtime = json.loads((root / paths[3]).read_text())
    contract = {
        "schema_version": "pmo_continuation_contract_v1",
        "authorization": "user authorizes future-aware controller development and public winner/route analysis",
        "task": "perindopril_mpo",
        "seed": 20260918,
        "new_oracle_limit": 140,
        "draws_per_bundle": 4,
        "parents": 5,
        "arms": ["immediate", "future"],
        "planning_option_depth": 2,
        "fresh_continuation_draws": 4,
        "kappa": 1,
        "exploration": 0.1,
        "prepared": {"path": PREPARED, "sha256": sha256_file(root / PREPARED)},
        "protocol": {
            "path": "docs/PMO_CONTINUATION_CHOICE.md",
            "sha256": sha256_file(root / "docs/PMO_CONTINUATION_CHOICE.md"),
        },
        "inputs": inputs,
        "expected_input_sha256": runtime["expected_input_sha256"],
        "runtime_contract_sha256": runtime["contract_sha256"],
        "law_caches": [],
        "reference_training_authorized": False,
        "docking_authorized": False,
        "winner_injected_proposals": False,
        "prescreen": False,
        "compute": {
            "max_workers": 20,
            "worker_tasks_max": 35,
            "worker_timeout": 1200,
            "driver_timeout": 3600,
            "retries": 0,
            "cpu": 1,
            "memory_mib": 8192,
            "expected_wall_minutes": [3, 15],
            "estimated_cost_usd": [0.5, 5],
            "restart_unit": "exact-parent four-draw worker; durable oracle attempt",
        },
    }
    contract["contract_sha256"] = identity(contract)
    publish_json(root / CONTRACT, contract)
    return contract


def load_contract(root: Path) -> dict:
    c = json.loads((root / CONTRACT).read_text())
    if identity({k: v for k, v in c.items() if k != "contract_sha256"}) != c["contract_sha256"]:
        raise ValueError("continuation contract hash mismatch")
    if tuple(
        c[k]
        for k in (
            "task",
            "new_oracle_limit",
            "draws_per_bundle",
            "parents",
            "planning_option_depth",
            "fresh_continuation_draws",
            "kappa",
            "exploration",
        )
    ) != ("perindopril_mpo", 140, 4, 5, 2, 4, 1, 0.1):
        raise ValueError("continuation contract outside declared scope")
    if c["arms"] != ["immediate", "future"] or any(
        c[k]
        for k in (
            "reference_training_authorized",
            "docking_authorized",
            "winner_injected_proposals",
            "prescreen",
        )
    ):
        raise ValueError("continuation contract changed scientific permissions")
    for key in ("prepared", "protocol"):
        verify_file(root / c[key]["path"], c[key]["sha256"])
    for path, sha in c["inputs"].items():
        verify_file(root / path, sha)
    return c


@contextmanager
def session(task, root, artifact_root, volume, validate_revision):
    validate_revision(task["image_revision"])
    verify_file(root / APP, task["app_sha256"])
    c = load_contract(root)
    body = {k: task[k] for k in ("contract_sha256", "image_revision", "app_sha256")}
    if c["contract_sha256"] != task["contract_sha256"] or identity(body) != task["run_id"]:
        raise ValueError("continuation deployment identity mismatch")
    output = artifact_root / KIND / task["run_id"]
    if "worker_id" in task:
        limit = {1: 5, 2: 20, 3: 10}.get(task["phase"], 0)
        if not 0 <= task["slot"] < limit or task["worker_id"] != worker_identity(
            task["phase"], task["slot"], task["parent"]
        ):
            raise ValueError("continuation worker outside exact parent census")
        output = output / "workers" / task["worker_id"]
    volume.reload()
    lock, stop = threading.RLock(), threading.Event()
    progress = {"phase": "initialization", "oracle_calls": 0}

    def commit():
        with lock:
            volume.commit()

    store, start = Store(output, commit), perf_counter()
    signature = {k: v for k, v in task.items() if k != "image_revision"}
    _frozen_save(store, "identity", signature)

    def heartbeat():
        while not stop.wait(30):
            publish_json(
                output / "heartbeat.json",
                {**progress, "at": _stamp(), "seconds": perf_counter() - start},
            )
            store.flush(force=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield c, store, progress
    except Exception as error:
        store.save("failure", {"error": repr(error), "progress": dict(progress), "at": _stamp()})
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        store.flush(force=True)


def draw_slots(worker: dict) -> list[dict | None]:
    """Preserve failed draws and actual sampling multiplicities, not success-only ranks."""
    if sorted(r["draw"] for r in worker["attempts"]) != list(range(4)):
        raise ValueError("worker must report every one of four proposal attempts")
    result = [None] * 4
    statuses = {r["draw"]: r["status"] for r in worker["attempts"]}
    for candidate in worker["candidates"]:
        prefix, draw = candidate["id"].rsplit("/", 1)
        index = int(draw)
        if (
            prefix != f"workers/{worker['worker_id']}/draws"
            or not 0 <= index < 4
            or result[index] is not None
            or statuses[index] != "complete"
        ):
            raise ValueError("candidate does not match its completed proposal draw")
        result[index] = candidate
    if any((statuses[i] == "complete") != (result[i] is not None) for i in range(4)):
        raise ValueError("completed proposal missing its candidate")
    return result


def run_comparison(task, contract, data, store, scores, progress, parallel) -> dict:
    """Three bounded rounds. Oracle locks precede labels; no surrogate is fitted."""
    history = data["observed"]
    worker_log = []

    def proposals(phase, parents):
        tasks = [
            {
                **task,
                "phase": phase,
                "slot": i,
                "parent": p,
                "worker_id": worker_identity(phase, i, p),
            }
            for i, p in enumerate(parents)
        ]
        _frozen_save(store, f"phase/{phase}/parents", {"tasks": tasks})
        by_id, results = {t["worker_id"]: t for t in tasks}, {}
        progress.update(phase=f"proposals_{phase}", workers_complete=0, workers_total=len(tasks))
        for result in parallel(tasks) if tasks else ():
            key = result["worker_id"]
            if key not in by_id or key in results or not result["replay_verified"]:
                raise ValueError("continuation worker completion identity mismatch")
            results[key] = result
            store.save(f"phase/{phase}/workers/{key}", result)
            progress.update(workers_complete=len(results))
            print(f"[continuation] phase={phase} workers={len(results)}/{len(tasks)}", flush=True)
        if results.keys() != by_id.keys():
            raise ValueError("continuation worker missing; no implicit replacement")
        workers = [results[t["worker_id"]] for t in tasks]
        lock_name = f"phase/{phase}/candidate_lock"
        _frozen_save(store, lock_name, {"workers": workers})
        slots = [draw_slots(w) for w in workers]
        progress.update(phase=f"scoring_{phase}")
        for row in slots:
            for candidate in row:
                if candidate is None:
                    continue
                smiles = candidate["smiles"]
                scores.context = {
                    "role": f"phase_{phase}",
                    "lock_path": str(store.output / f"{lock_name}.json"),
                }
                candidate["score"] = (
                    history[smiles] if smiles in history else scores.score(smiles)["desirability"]
                )
        _frozen_save(store, f"phase/{phase}/scored", {"slots": slots})
        worker_log.extend(workers)
        return slots

    first = proposals(1, data["parents"])
    positions = [(i, j) for i, row in enumerate(first) for j, c in enumerate(row) if c is not None]
    offspring = proposals(2, [first[i][j] for i, j in positions])
    second = [[[None] * 4 for _ in range(4)] for _ in first]
    for (i, j), row in zip(positions, offspring, strict=True):
        second[i][j] = row
    snapshot = identity({"first": first, "second": second})
    choices, selected_parents = [], {}
    for i, parent in enumerate(data["parents"]):
        seed = int(np.random.SeedSequence([contract["seed"], i, 101]).generate_state(1)[0])
        comparison = choose_continuation(first[i], second[i], seed=seed, snapshot_id=snapshot)
        for decision in comparison["decisions"].values():
            candidate = first[i][decision["selected"]]
            key = None if candidate is None else identity(candidate)
            decision["selected_parent_identity"] = key
            if candidate is not None:
                selected_parents[key] = candidate
        choices.append(
            {"source": parent["development_source"], "parent_score": parent["score"], **comparison}
        )
    ordered = sorted(selected_parents)
    _frozen_save(
        store,
        "choices",
        {"snapshot_sha256": snapshot, "choices": choices, "fresh_parent_keys": ordered},
    )
    fresh = proposals(3, [selected_parents[k] for k in ordered])
    fresh_by_key = dict(zip(ordered, fresh, strict=True))
    for choice in choices:
        for decision in choice["decisions"].values():
            key = decision["selected_parent_identity"]
            row = [None] * 4 if key is None else fresh_by_key[key]
            values = [0.0 if c is None else c["score"] for c in row]
            decision["fresh"] = {
                "attempts": 0 if key is None else 4,
                "successes": sum(c is not None for c in row),
                "scores_with_abstention_zero": values,
                "mean_return": float(np.mean(values)),
                "best_return": max(values),
                "canonical_diversity": len({c["smiles"] for c in row if c is not None}),
                "selected_immediate_score": None if key is None else selected_parents[key]["score"],
                "candidates": row,
            }
    all_candidates = [c for w in worker_log for c in w["candidates"]]
    arms = {
        arm: {
            "mean_fresh_return": float(
                np.mean([r["decisions"][arm]["fresh"]["mean_return"] for r in choices])
            ),
            "mean_fresh_best": float(
                np.mean([r["decisions"][arm]["fresh"]["best_return"] for r in choices])
            ),
            "logical_fresh_attempts": sum(
                r["decisions"][arm]["fresh"]["attempts"] for r in choices
            ),
        }
        for arm in contract["arms"]
    }
    return {
        "choices": choices,
        "arms": arms,
        "snapshot_sha256": snapshot,
        "changed_choices": sum(
            r["decisions"]["immediate"]["selected"] != r["decisions"]["future"]["selected"]
            for r in choices
        ),
        "fresh_paired_differences": [
            r["decisions"]["future"]["fresh"]["mean_return"]
            - r["decisions"]["immediate"]["fresh"]["mean_return"]
            for r in choices
        ],
        "workers": len(worker_log),
        "logical_planning_attempts": 20 + 4 * len(positions),
        "physical_fresh_attempts": 4 * len(ordered),
        "completed_products": len(all_candidates),
        "unique_products": len({c["smiles"] for c in all_candidates}),
        "options_attempted": dict(
            Counter(
                "none" if a["bundle"] is None else a["bundle"]["option"]
                for w in worker_log
                for a in w["attempts"]
            )
        ),
        "proposal_seconds_sum": sum(w["seconds"] for w in worker_log),
        "executor_calls": sum(w["executor_calls"] for w in worker_log),
        "law_work": {
            k: sum(w["law_work"][k] for w in worker_log) for k in worker_log[0]["law_work"]
        },
        "worker_summary": [
            {
                k: w[k]
                for k in (
                    "worker_id",
                    "phase",
                    "slot",
                    "attempts",
                    "seconds",
                    "initialization_seconds",
                    "executor_calls",
                    "law_work",
                )
            }
            for w in worker_log
        ],
    }


def driver_remote(task, root, artifact_root, volume, validate_revision, parallel):
    import importlib.metadata
    import platform
    import resource

    with session(task, root, artifact_root, volume, validate_revision) as (c, store, progress):
        previous = store.read("result")
        if previous is not None:
            return previous
        started, stamp = perf_counter(), _stamp()
        data = json.loads((root / PREPARED).read_text())
        scores = DurableScores(
            store.output,
            make_oracle(c["task"], root, c),
            c["new_oracle_limit"],
            lambda: store.flush(force=True),
            progress,
        )
        result = run_comparison(task, c, data, store, scores, progress, parallel)
        result.update(
            schema_version="pmo_continuation_result_v1",
            status="complete_bounded_development",
            new_oracle_calls=scores.meter.spent,
            new_physical_calls_by_role=dict(Counter(r["role"] for r in scores.rows)),
            historical_calls=data["historical_calls"],
            historical_unique_labels=len(data["observed"]),
            separate_public_diagnostic_calls=data["separate_public_diagnostic_calls"],
            public_development_reference=data["public_development_reference"],
            best_seen=max([*data["observed"].values(), *[r["score"] for r in scores.rows]]),
            seconds=perf_counter() - started,
            started_at=stamp,
            finished_at=_stamp(),
            run_id=task["run_id"],
            code_revision=task["image_revision"]["commit"],
            configuration=c,
            software={
                "python": platform.python_version(),
                **{
                    p: importlib.metadata.version(p)
                    for p in ("numpy", "scipy", "rdkit", "torch", "PyTDC")
                },
            },
            hardware={
                "device": "cpu",
                "policy_precision": "float64",
                "reference_precision": "float32",
                "platform": platform.platform(),
                "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            oracle_ledger=[
                sha256_file(p) for p in sorted((store.output / "oracle").glob("*/result.json"))
            ],
            interpretation="five development parents; paired sampled-tree choice, fresh continuation evaluation; no exact Doob, unbiased expectation, unseen-task or IVG-superiority claim",
        )
        store.save("result", result)
        return result
