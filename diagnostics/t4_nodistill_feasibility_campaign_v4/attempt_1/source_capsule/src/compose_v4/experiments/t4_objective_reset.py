"""Bounded T4 reset diagnostics and resumable scored pilot orchestration.

The caller supplies a frozen start state and (for scoring only) the unchanged
docking callback. Controller modules have no access to historical outcomes.
"""

from __future__ import annotations

import json
import platform
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from rdkit import rdBase

from compose_v4.control.docking_value import identity
from compose_v4.control.objective_program_search import (
    ObjectiveProgramSearch,
    ObjectiveSearchConfig,
    v0_search_config,
)
from compose_v4.control.progressive_bootstrap import ProgressiveBootstrap
from compose_v4.experiments.t4_frozen_program_benchmark import (
    _dock_candidate,
    strict_endpoint_scorer,
)
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.rewrite.trace_shard import decode_state

ARMS = {"support_control": "score_blind", "objective_search": "niche_score"}


def runtime_config(seed):
    return replace(
        v0_search_config(seed=seed),
        attempts_per_batch=64,
        candidates_per_batch=8,
        wall_seconds=20.0,
    )


def validate_unit(unit):
    required = {
        "cell",
        "unit_id",
        "source_state",
        "original_seed",
        "source_group",
        "oracle_protocol",
        "target",
        "source_idx",
        "controller_seed",
        "docking_seed",
    }
    if set(unit) != required:
        raise ValueError(f"reset unit fields differ: {sorted(set(unit) ^ required)}")
    source = decode_state(unit["source_state"])
    if source.n_atoms != 48 or not 1 <= source.n_real_atoms <= 40:
        raise ValueError("reset source violates the declared 48-slot/40-atom support")
    return source


def zero_oracle_probe(unit, output: Path, *, delta=0.4, rounds=4, commit=lambda: None):
    source = validate_unit(unit)
    config = runtime_config(unit["controller_seed"])
    bootstrap = ProgressiveBootstrap(
        source, config, source_group=unit["source_group"], oracle_protocol=unit["oracle_protocol"]
    )
    eligible = strict_endpoint_scorer(unit["original_seed"], delta=delta)
    rows = []
    began = perf_counter()
    for index in range(rounds):
        batch = bootstrap.next_batch(eligible)
        seal(output / "rounds" / f"round_{index:04d}.json", batch)
        row = {
            "round": index,
            "attempts": len(batch["attempts"]),
            "eligible": len(batch["candidates"]),
            "cursor": bootstrap.cursor,
            "status_counts": dict(Counter(a["status"] for a in batch["attempts"])),
            "lane_counts": dict(Counter(a["planner_channel"] for a in batch["attempts"])),
            "elapsed_seconds": perf_counter() - began,
            "new_oracle_calls": 0,
        }
        rows.append(row)
        seal(output / "checkpoint.json", {"bootstrap": bootstrap.snapshot(), "rounds": rows})
        seal(output / "progress.json", {"cell": unit["cell"], "phase": "bootstrap", **row})
        commit()
        print({"cell": unit["cell"], **row}, flush=True)
        if batch["candidates"]:
            break
    result = {
        "schema_version": "t4_reset_zero_oracle_probe_v1",
        "unit": unit,
        "configuration": asdict(config),
        "delta": delta,
        "maximum_rounds": rounds,
        "rounds": rows,
        "eligible": sum(r["eligible"] for r in rows),
        "attempts": sum(r["attempts"] for r in rows),
        "new_oracle_calls": 0,
        "runtime": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "completed_at_utc": _stamp(),
        "interpretation": "proposal-yield diagnostic, not a stochastic cold-start pass gate",
    }
    seal(output / "result.json", result)
    commit()
    return result


def run_scored_unit(
    unit, output: Path, *, arm, run_id, volume, dock, calls=20, delta=0.4, max_rounds=32
):
    """No automatic retry: ambiguous started docking raises in the old ledger."""
    if arm not in ARMS or type(calls) is not int or not 1 <= calls <= 20 or delta != 0.4:
        raise ValueError("scored reset pilot exceeds its declared arm/budget/delta")
    source = validate_unit(unit)
    config = runtime_config(unit["controller_seed"])
    policy = ObjectiveSearchConfig(
        parent_allocation=ARMS[arm],
        structured_attempts=16,
        structured_candidates=4,
        structured_wall_seconds=20,
        query_batch_size=8,
    )
    frozen = json.loads(
        json.dumps(
            {
                "unit": unit,
                "arm": arm,
                "run_id": run_id,
                "calls": calls,
                "delta": delta,
                "max_rounds": max_rounds,
                "configuration": asdict(config),
                "policy": asdict(policy),
            }
        )
    )
    if (output / "started.json").exists():
        if unseal(output / "started.json")["frozen"] != frozen:
            raise ValueError("resumed reset unit changed its inputs/configuration")
    else:
        seal(output / "started.json", {"frozen": frozen, "started_at_utc": _stamp()})
        volume.commit()
    if (output / "result.json").exists():
        return unseal(output / "result.json")
    eligibility = strict_endpoint_scorer(unit["original_seed"], delta=delta)
    if (output / "checkpoint.json").exists():
        saved = unseal(output / "checkpoint.json")
        bootstrap = ProgressiveBootstrap.restore(saved["bootstrap"])
        search = ObjectiveProgramSearch.restore(saved["search"])
        next_round, rows, champion = saved["next_round"], saved["rows"], saved["champion"]
    else:
        bootstrap = ProgressiveBootstrap(
            source,
            config,
            source_group=unit["source_group"],
            oracle_protocol=unit["oracle_protocol"],
        )
        search = ObjectiveProgramSearch(
            config,
            policy,
            source_group=unit["source_group"],
            oracle_protocol=unit["oracle_protocol"],
        )
        next_round, rows, champion = 0, [], None
    began = perf_counter()
    while len(rows) < calls and next_round < max_rounds:
        lock_path = output / "rounds" / f"round_{next_round:04d}.json"
        if lock_path.exists():
            lock = unseal(lock_path)
            batch, cold = lock["batch"], lock["cold_start"]
            bootstrap = ProgressiveBootstrap.restore(lock["bootstrap_after"])
            search = ObjectiveProgramSearch.restore(lock["search_after"])
        else:
            cold = not search.base.entries
            remaining = min(calls - len(rows), policy.query_batch_size)
            if cold:
                batch = bootstrap.next_batch(eligibility)
                body = {
                    k: v
                    for k, v in batch.items()
                    if k not in {"batch_id", "proposal_seconds", "new_oracle_calls"}
                }
                body["proposal_pool"] = {"candidates": body["candidates"]}
                body["candidates"] = body["candidates"][:remaining]
                batch = {**body, "batch_id": identity(body)}
            else:
                batch = search.propose_batch(eligibility, limit=remaining)
            lock = {
                "batch": batch,
                "cold_start": cold,
                "query_start": len(rows),
                "bootstrap_after": bootstrap.snapshot(),
                "search_after": search.snapshot(include_history=False),
            }
            seal(lock_path, lock)
            volume.commit()
        if lock["query_start"] != len(rows):
            raise ValueError("checkpoint and prospective query lock disagree")
        progress = {
            "cell": unit["cell"],
            "arm": arm,
            "phase": "locked_candidates" if batch["candidates"] else "empty_proposal_round",
            "round": next_round,
            "calls": len(rows),
            "ceiling": calls,
            "attempts": len(batch["attempts"]),
            "locked_candidates": len(batch["candidates"]),
            "bootstrap_cursor": bootstrap.cursor,
            "elapsed_seconds": perf_counter() - began,
            "updated_at_utc": _stamp(),
        }
        seal(output / "progress.json", progress)
        volume.commit()
        print(progress, flush=True)
        outcomes = []
        for index, candidate in enumerate(batch["candidates"]):
            result = _dock_candidate(
                folder=output,
                candidate=candidate,
                query_index=len(rows),
                round_index=next_round,
                candidate_index=index,
                unit=unit,
                run_id=run_id,
                volume=volume,
                dock=dock,
            )
            rows.append(result)
            if result["score"] is not None and (
                champion is None or result["score"] < champion["score"]
            ):
                champion = {
                    "score": result["score"],
                    "candidate": candidate,
                    "query": len(rows),
                    "receipt_id": result["receipt_id"],
                }
            outcomes.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "receipt_id": result["receipt_id"],
                    "score": result["score"],
                    "failure": result["failure"],
                    "oracle_protocol": unit["oracle_protocol"],
                }
            )
            progress = {
                "cell": unit["cell"],
                "arm": arm,
                "calls": len(rows),
                "ceiling": calls,
                "last_score": result["score"],
                "best_score": None if champion is None else champion["score"],
                "round": next_round,
                "elapsed_seconds": perf_counter() - began,
                "phase": "docking",
                "updated_at_utc": _stamp(),
            }
            seal(output / "progress.json", progress)
            volume.commit()
            print(progress, flush=True)
        if cold:
            for candidate, result in zip(batch["candidates"], outcomes, strict=True):
                if result["score"] is not None:
                    search.add_measured_program(
                        candidate, receipt_id=result["receipt_id"], score=result["score"]
                    )
                else:
                    search.channel_stats[search.lane(candidate)]["charged"] += 1
                    search.base.failed_endpoints.add(candidate["endpoint"])
        else:
            search.observe_batch(batch["batch_id"], outcomes)
        next_round += 1
        seal(
            output / "checkpoint.json",
            {
                "bootstrap": bootstrap.snapshot(),
                "search": search.snapshot(include_history=False),
                "next_round": next_round,
                "rows": rows,
                "champion": champion,
            },
        )
        volume.commit()
    result = {
        "schema_version": "t4_objective_reset_scored_unit_v1",
        "frozen": frozen,
        "query_count": len(rows),
        "rows": rows,
        "champion": champion,
        "termination": "query_budget" if len(rows) == calls else "round_limit",
        "channel_stats": search.channel_stats,
        "completed_at_utc": _stamp(),
    }
    seal(output / "result.json", result)
    volume.commit()
    return unseal(output / "result.json")
