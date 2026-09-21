#!/usr/bin/env python3
"""Witness-stratum evidence for the PMO realization repair.

Two measurements that together decide whether a ``proven_incompatible`` verdict can be
trusted, written as repo code because the report cites their numbers.

1. RE-ENUMERABILITY CENSUS.  ``diagnostics/pmo_binder_repair_v1.json`` treats a plan
   bound back onto the teacher source that generated it as a pair where "a complete
   role-consistent realization provably exists ... so any bounded-arm failure here is
   DISCARDED_BY_SEARCH by construction".  That premise holds only if every teacher
   action is re-enumerable by ``enumerate_role_successors`` at the state it was taken
   from.  The runtime fiber applies validity gates the corpus generator did not
   necessarily apply, so it is not guaranteed.

2. SOUNDNESS GATE.  On a route whose every step IS re-enumerable, a complete
   realization exists, so the realizer must not return ``proven_incompatible``.  Any
   such verdict is an unsound prune.  This is the control that caught the real bug: a
   per-(element, charge) deletion budget is not a necessary condition while an
   ``atom_restate_semantic`` is still pending, because a restatement changes an atom's
   element in place.

Zero oracle calls.  No pinned module is modified.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

from pmo_binder_repair_v1 import TEACHER_CORPUS, load_plans
from pmo_realization_repair_v1 import (
    OUTCOME_INCOMPATIBLE,
    SPEC_R1,
    SPEC_V1_EXACT,
    realize,
    teacher_step_gaps,
)

from compose_v4.control.docking_value import identity
from compose_v4.control.pmo_joint_dependency_jump import _generic_role_sequence
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "diagnostics/pmo_realization_witness_gate_v1.json"


def _plan_routes() -> list[dict[str, Any]]:
    """One generating route per plan in the production plan bank."""

    with gzip.open(TEACHER_CORPUS, "rt") as handle:
        corpus = json.load(handle)["payload"]
    keep = {plan["plan_id"] for plan in load_plans()}
    seen: set[str] = set()
    rows = []
    for index, route in enumerate(corpus["routes"]):
        if not route["dependency_region_program"].get("complete_representation_supported"):
            continue
        roles = _generic_role_sequence(route)
        plan_id = identity({"schema_version": "pmo_joint_role_plan_v1", "roles": roles})
        if plan_id not in keep or plan_id in seen:
            continue
        seen.add(plan_id)
        rows.append({"plan_id": plan_id, "route_index": index, "primitive_count": len(roles)})
    return rows


def _census_task(entry: dict[str, Any]) -> dict[str, Any]:
    with gzip.open(TEACHER_CORPUS, "rt") as handle:
        corpus = json.load(handle)["payload"]
    route = corpus["routes"][entry["route_index"]]
    roles = _generic_role_sequence(route)
    gaps = teacher_step_gaps(roles, route["states"][:-1], route["actions"])
    return {
        "plan_id": entry["plan_id"],
        "route_index": entry["route_index"],
        "primitive_count": len(roles),
        "teacher_scale": len(roles) >= 27,
        "gaps": gaps,
    }


def _gate_task(entry: dict[str, Any]) -> dict[str, Any]:
    plans = {plan["plan_id"]: plan for plan in load_plans()}
    with gzip.open(TEACHER_CORPUS, "rt") as handle:
        corpus = json.load(handle)["payload"]
    route = corpus["routes"][entry["route_index"]]
    source = decode_state(route["states"][0])
    plan = plans[entry["plan_id"]]
    out: dict[str, Any] = {
        "plan_id": entry["plan_id"],
        "primitive_count": entry["primitive_count"],
        "teacher_scale": entry["teacher_scale"],
        "reenumerable": not entry["gaps"],
        "gaps": entry["gaps"],
    }
    for tag, spec in (("v1_exact", SPEC_V1_EXACT), ("R1_drop_environment_and_lag", SPEC_R1)):
        result = realize(
            source,
            plan,
            spec,
            node_budget=entry["node_budget"],
            seconds_cap=entry["seconds_cap"],
        )
        out[tag] = {
            "outcome": result["outcome"],
            "nodes_expanded": result["nodes_expanded"],
            "successors_enumerated": result["successors_enumerated"],
            "max_depth_reached": result["max_depth_reached"],
            "seconds": result["seconds"],
            "realizations": [
                {
                    "primitive_count": row["primitive_count"],
                    "retained_fraction": row["retained_fraction"],
                    "component_count": row["component_count"],
                    "created_dependency_edges": row["created_dependency_edges"],
                }
                for row in result["realizations"]
            ],
            # The gate: incompatibility is only admissible where the route is not
            # re-enumerable in the first place.
            "unsound": result["outcome"] == OUTCOME_INCOMPATIBLE and not entry["gaps"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gate-sample", type=int, default=24)
    parser.add_argument("--node-budget", type=int, default=200_000)
    parser.add_argument("--seconds-cap", type=float, default=120.0)
    args = parser.parse_args()

    began = perf_counter()
    entries = _plan_routes()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        census = list(pool.map(_census_task, entries))
    with_gap = [row for row in census if row["gaps"]]
    teacher_scale = [row for row in census if row["teacher_scale"]]
    rules = Counter(gap["executor_rule"] for row in with_gap for gap in row["gaps"])

    gate_entries = [
        {**row, "node_budget": args.node_budget, "seconds_cap": args.seconds_cap}
        for row in census[: args.gate_sample]
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        gate = list(pool.map(_gate_task, gate_entries))
    unsound = [
        {"plan_id": row["plan_id"], "arm": arm}
        for row in gate
        for arm in ("v1_exact", "R1_drop_environment_and_lag")
        if row[arm]["unsound"]
    ]
    payload = {
        "schema_version": "pmo_realization_witness_gate_v1",
        "reenumerability_census": {
            "definition": "teacher steps whose own action enumerate_role_successors does "
            "not offer at the state they were taken from",
            "plans": len(census),
            "plans_with_a_gap": len(with_gap),
            "plans_with_a_gap_fraction": len(with_gap) / len(census) if census else 0.0,
            "teacher_scale_plans": len(teacher_scale),
            "teacher_scale_plans_with_a_gap": sum(1 for row in teacher_scale if row["gaps"]),
            "gap_steps": sum(len(row["gaps"]) for row in with_gap),
            "teacher_steps_total": sum(row["primitive_count"] for row in census),
            "gap_steps_by_rule": dict(rules),
            "consequence": "a plan with a gap is unrealizable on its OWN generating "
            "source, so proven_incompatible there is CORRECT.  The teacher-witness "
            "stratum is therefore not a guaranteed positive control.",
        },
        "soundness_gate": {
            "definition": "on a fully re-enumerable route a realization exists, so "
            "proven_incompatible is an unsound prune",
            "pairs": len(gate),
            "unsound": len(unsound),
            "unsound_rows": unsound,
            "outcomes": {
                arm: dict(Counter(row[arm]["outcome"] for row in gate))
                for arm in ("v1_exact", "R1_drop_environment_and_lag")
            },
        },
        "census_rows": census,
        "gate_rows": gate,
        "elapsed_seconds": perf_counter() - began,
        "new_oracle_calls": 0,
        "reproduce": "KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src:scripts "
        f"python3 scripts/pmo_realization_witness_gate_v1.py --workers {args.workers} "
        f"--gate-sample {args.gate_sample}",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {REPORT}")
    print("  plans with a non-re-enumerable teacher step:", len(with_gap), "of", len(census))
    print("  gap steps by rule:", dict(rules))
    print("  soundness gate unsound verdicts:", len(unsound))


if __name__ == "__main__":
    main()
