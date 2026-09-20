#!/usr/bin/env python3
"""Phase B: run the LIVE PMO proposer from the teacher sources, zero oracle calls.

Phase A decomposed the sealed 184-route teacher corpus.  This phase asks the
complementary question the T4 archaeology asked: given the very same starting
states, what does the CURRENT proposal law actually emit?

The proposer is the production ``PmoPopulationController`` driven through the real
``run_program_campaign``, with the real sealed joint-jump checkpoint.  The only
substitution is the objective: ``evaluate`` is a deterministic hash in [0, 1], the
same device ``scripts/pmo_free_oracle_smoke.py`` uses, so no TDC oracle is ever
constructed and no call is charged.

The initialization is built HERE, in memory, from teacher route source states.  The
pinned production initialization lock is never read or modified; this is a probe
archive, not a launch.

Every eligible-pool candidate carries its own exact ``trace`` (states + actions), so
each proposal is characterized by exactly the same functions Phase A applied to the
teachers.  The comparison is therefore like-for-like by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pmo_teacher_route_gap_v1 import (
    MAX_PRIMITIVES,
    MAX_REGIONS,
    ROOT,
    characterize,
    load_routes,
    regions,
    route_objective,
)

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v21 import (
    initial_dynamic_program_batch_v21,
)
from compose_v4.control.pmo_population_controller import PmoPopulationController
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask
from compose_v4.experiments import pmo_population_v1 as population

RELAUNCH_TASKS = ("gsk3b", "celecoxib_rediscovery", "perindopril_mpo")


def free_score(smiles: str) -> float:
    """Deterministic pseudo-objective in [0, 1]. No oracle, no network, no cost."""
    return (int(hashlib.sha256(smiles.encode()).hexdigest()[:8], 16) % 10_000) / 10_000.0


def build_initialization(routes: list[dict], seed: int) -> dict:
    """A probe initialization whose parents are exactly the teacher source states."""
    candidates, seen = [], set()
    for route in routes:
        state = route["source_state"]
        key = json.dumps(state, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "endpoint": characterize(route)["source_smiles"],
                "source_id": f"teacher_source:{route['trace_identity']}",
                "state": state,
            }
        )
    body = {
        "accounting": "free-scorer probe; no oracle call is charged",
        "available_unique": len(candidates),
        "candidates": candidates,
        "count": len(candidates),
        "new_oracle_calls": 0,
        "schema_version": "program_initialization_lock_v1",
        "seed": seed,
        "source_sha256": identity([row["source_id"] for row in candidates]),
    }
    return {**body, "lock_sha256": identity(body)}


def harvest(folder: Path) -> list[dict]:
    """Every candidate the proposer emitted, across all rounds.

    The bootstrap round publishes ``proposal_pool``; later rounds publish
    ``eligible_pool``.  Both are pre-selection proposal pools, so both count as
    "what the proposer generated" -- reading only one would undercount the law.
    """
    rows, seen = [], set()
    for pending in sorted(folder.glob("campaign/round_*/pending.json")):
        batch = json.loads(pending.read_text())["batch"]
        round_name = pending.parent.name
        for pool in ("proposal_pool", "eligible_pool"):
            for candidate in batch.get(pool, {}).get("candidates", []):
                key = candidate["candidate_id"]
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"round": round_name, "pool": pool, "candidate": candidate})
    return rows


def describe_proposal(candidate: dict) -> dict | None:
    """Characterize a proposal with the Phase-A functions; None if not analyzable."""
    trace = candidate.get("trace") or {}
    states, actions = trace.get("states"), trace.get("actions")
    if not states or not actions or len(states) != len(actions) + 1:
        return None
    route = {
        "source_state": candidate["source_state"],
        "states": states,
        "actions": actions,
        "terminal_endpoint": candidate["endpoint"],
        "trace_identity": candidate["candidate_id"],
    }
    try:
        features = characterize(route)
        primary = regions(route, join=True)
        strict = regions(route, join=False)
    except (ValueError, KeyError, IndexError, TypeError) as failure:
        return {"analysis_failed": f"{type(failure).__name__}: {failure}"}
    return {
        "characterization": features,
        "primary_regions": primary["component_count"],
        "strict_regions": strict["component_count"],
        "planner_channel": candidate["provenance"].get("planner_channel"),
    }


def run_task(
    task_name: str,
    routes: list[dict],
    out: Path,
    rounds: int,
    seed: int,
    *,
    production_init: bool = False,
) -> dict:
    """Probe one task.

    ``production_init`` swaps the teacher-root parents for the PINNED production
    initialization.  That is the control which separates "binding fails because these
    teacher roots are unusual" from "binding fails generally": it is read-only, the
    lock file is never rewritten.
    """
    folder = out / task_name
    folder.mkdir(parents=True, exist_ok=True)
    if production_init:
        initialization = json.loads(
            (ROOT / "diagnostics/parent_edit_cycles/prepared/init_20260921.json").read_text()
        )
    else:
        initialization = build_initialization(routes, seed)
    # The full contract verification pins implementation_sha256 for every controller
    # source file.  Another session is editing those files while this probe runs, so a
    # strict load fails part-way through for reasons unrelated to the proposal law.
    # This probe therefore loads the sealed joint checkpoint DIRECTLY and records which
    # verification it did, rather than silently running against an unverified tree.
    contract_verified = True
    try:
        contract = population.load_contract(ROOT)
        controller_seed = contract["controller"]["seed"]
    except ValueError as failure:
        contract_verified = f"implementation pin mismatch: {failure}"
        controller_seed = seed
    checkpoint = json.loads((ROOT / population.CHECKPOINTS).read_text())["payload"][
        "checkpoints"
    ]["shared_all_routes"]
    if checkpoint.get("fit_scope") != "shared_all_routes":
        raise ValueError("PMO runtime checkpoint must be shared and task blind")

    task = ProgramTask(task_name, identity({"probe": "free_scorer", "task": task_name}), "pmo")
    calls = {"n": 0}

    def evaluate(smiles: str) -> float:
        calls["n"] += 1
        return free_score(smiles)

    ledger = ProgramQueryLedger(
        folder / "oracle", task, evaluate, budget=population.QUERY_BUDGET
    )
    config = population.configuration(controller_seed)
    status, error = "COMPLETED", None
    try:
        run_program_campaign(
            output=folder / "campaign",
            task=task,
            config=config,
            initialization=initialization,
            library=(),
            ledger=ledger,
            rounds=rounds,
            queries_per_round=population.QUERIES_PER_ROUND,
            hierarchy=None,
            fit_model=None,
            stagnation_rounds=None,
            bootstrap_rounds=1,
            initialization_mode="all_scored_pool",
            initial_parent_fraction=0.2,
            optimizer_type=PmoPopulationController,
            optimizer_kwargs={"jump_checkpoint": checkpoint},
            initial_batch_fn=initial_dynamic_program_batch_v21,
        )
    except Exception as failure:  # noqa: BLE001 - the probe reports any failure verbatim
        status, error = "FAILED", f"{type(failure).__name__}: {failure}"

    proposals = harvest(folder)
    described = []
    for row in proposals:
        detail = describe_proposal(row["candidate"])
        if detail is None or "analysis_failed" in detail:
            continue
        described.append({**detail, "round": row["round"], "pool": row["pool"]})

    # ---- teacher support: is the winning endpoint generated at all? ----
    teacher_endpoints = {route["terminal_endpoint"] for route in routes}
    proposed_endpoints = {row["candidate"]["endpoint"] for row in proposals}
    hits = sorted(teacher_endpoints & proposed_endpoints)
    return {
        "task": task_name,
        "status": status,
        "error": error,
        "contract_verified": contract_verified,
        "free_evaluations": calls["n"],
        "charged_oracle_calls": 0,
        "initialization_parents": initialization["count"],
        "pool_candidates": len(proposals),
        "distinct_proposed_endpoints": len(proposed_endpoints),
        "analyzed_proposals": len(described),
        "teacher_endpoints": len(teacher_endpoints),
        "exact_teacher_endpoint_support": len(hits),
        "exact_teacher_endpoint_hits": hits,
        "proposals": described,
    }


def summarize(described: list[dict]) -> dict:
    if not described:
        return {"proposals": 0}
    chars = [row["characterization"] for row in described]
    primitives = [c["primitive_count"] for c in chars]
    return {
        "proposals": len(described),
        "by_channel": dict(Counter(row["planner_channel"] for row in described)),
        "primitive_count": {
            "min": min(primitives),
            "median": sorted(primitives)[len(primitives) // 2],
            "mean": sum(primitives) / len(primitives),
            "max": max(primitives),
        },
        "primary_regions": dict(Counter(row["primary_regions"] for row in described)),
        "strict_regions": dict(Counter(row["strict_regions"] for row in described)),
        "direction": dict(Counter(c["direction"] for c in chars)),
        "touches_ring_topology": sum(c["touches_ring_topology"] for c in chars),
        "ring_topology_rate": sum(c["touches_ring_topology"] for c in chars) / len(chars),
        "routes_with_reused_created_handles": sum(c["created_handles_reused"] > 0 for c in chars),
        "reused_handle_rate": sum(c["created_handles_reused"] > 0 for c in chars) / len(chars),
        "retained_fraction_median": sorted(c["retained_fraction"] for c in chars)[len(chars) // 2],
        "delta_heavy_atoms_median": sorted(c["delta_heavy_atoms"] for c in chars)[len(chars) // 2],
        "delta_rings_median": sorted(
            [c["delta_rings"] for c in chars if c["delta_rings"] is not None]
        )[max(0, len([c for c in chars if c["delta_rings"] is not None]) // 2)]
        if any(c["delta_rings"] is not None for c in chars)
        else None,
        "rule_totals": dict(
            sorted(
                Counter(
                    rule for c in chars for rule, count in c["rule_counts"].items() for _ in range(count)
                ).items()
            )
        ),
        "teacher_scale_at_least_20_primitives": sum(p >= 20 for p in primitives),
        "teacher_scale_rate": sum(p >= 20 for p in primitives) / len(primitives),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--tasks", default=",".join(RELAUNCH_TASKS))
    parser.add_argument("--production-init", action="store_true")
    options = parser.parse_args()

    corpus = load_routes()
    by_objective: dict[str, list[dict]] = defaultdict(list)
    for route in corpus:
        by_objective[route_objective(route)].append(route)

    results = {}
    for task_name in options.tasks.split(","):
        routes = by_objective.get(task_name, [])
        if not routes:
            results[task_name] = {"task": task_name, "status": "NO_TEACHER_ROUTES"}
            continue
        report = run_task(
            task_name,
            routes,
            Path(options.work),
            options.rounds,
            options.seed,
            production_init=options.production_init,
        )
        report["summary"] = summarize(report.pop("proposals"))
        results[task_name] = report
        print(json.dumps({task_name: {k: v for k, v in report.items()}}, indent=1)[:1200], flush=True)

    payload = {
        "schema_version": "pmo_teacher_route_gap_phase_b_v1",
        "region_configuration": {
            "runtime_maximum_primitives": MAX_PRIMITIVES,
            "runtime_maximum_components": MAX_REGIONS,
        },
        "objective": "deterministic sha256 hash in [0,1]; no TDC oracle constructed",
        "charged_oracle_calls": 0,
        "tasks": results,
    }
    Path(options.out).parent.mkdir(parents=True, exist_ok=True)
    Path(options.out).write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")


if __name__ == "__main__":
    main()
