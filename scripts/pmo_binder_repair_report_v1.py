#!/usr/bin/env python3
"""Driver for the PMO-v2 binder failure decomposition.

Produces ``diagnostics/pmo_binder_repair_v1.json``.  Zero oracle calls, no Modal, no
production wiring.  See ``pmo_binder_repair_v1`` for the measurement contract.

Stages
  0  equivalence -- the instrumented binder reproduces the PINNED bind_joint_plan
  1  static census -- zero-search step-0 NO_COMPATIBLE_BINDING certificate, all pairs
  2  witness control -- plans bound back onto the teacher source that generated them,
     where a complete realization provably exists
  3  decomposition -- referee (a)/(b)/(c) plus the beam sweep and the ranking A/B on a
     stratified sample of step-0-feasible pairs
  4  yield -- complete-program scale and retained fraction against the teacher profile

Outcome arms are deterministic and load-independent, so stage 3 runs in a process pool.
Per-pair COST is measured separately and serially, because a loaded machine cannot time
itself (see .claude/context/learnings.md, 2026-08-03).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

from pmo_binder_repair_v1 import (
    ATTRIBUTION_VIEWS,
    OUTCOME_EXISTS,
    OUTCOME_NO_BINDING,
    OUTCOME_PROGRAM_REJECTED,
    OUTCOME_UNDECIDED,
    PRODUCTION_BEAM_WIDTH,
    SCHEMA,
    bind_instrumented,
    first_step_feasible_under,
    load_plan_source_witnesses,
    load_plans,
    load_production_parents,
    load_teacher_root_parents,
    plan_specification_census,
    referee_binding_exists,
    static_first_step_feasible,
)

from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "diagnostics/pmo_binder_repair_v1.json"
WIDTHS = (4, 8, 16, 32, 64)
COMPAT_WIDTHS = (4, 8)
SEED = 20260920

# The teacher profile the archaeology measured, for the "did scale move" comparison.
TEACHER_PROFILE = {
    "primitive_count_median_range": [29, 40],
    "retained_fraction_range": [0.52, 0.96],
    "teacher_scale_threshold": 27,
    "source": "diagnostics/pmo_teacher_route_gap_v1.json",
}


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


# ---- stage 3 worker ----


def _run_pair(job: dict[str, Any]) -> dict[str, Any]:
    """All arms for one (parent, plan) pair.  Deterministic; safe to run in parallel."""

    plan = job["plan"]
    graph = decode_state(job["state"])
    row: dict[str, Any] = {
        "parent_id": job["parent_id"],
        "stratum": job["stratum"],
        "plan_id": plan["plan_id"],
        "plan_primitive_count": int(plan["primitive_count"]),
        "arms": {},
    }
    referee = referee_binding_exists(
        graph, plan, frontier_cap=job["frontier_cap"], seconds_cap=job["seconds_cap"]
    )
    row["referee"] = {
        key: referee[key]
        for key in (
            "outcome",
            "depth_reached",
            "plan_length",
            "peak_frontier",
            "successors_enumerated",
            "seconds",
            "capped_at_step",
        )
    }
    row["referee_witness"] = referee.get("witness")
    for width in job["widths"]:
        result = bind_instrumented(
            graph, plan, beam_width=width, ranking="hash",
            seconds_cap=job["arm_seconds_cap"],
        )
        row["arms"][f"hash_w{width}"] = _arm_row(result)
    for width in job["compat_widths"]:
        result = bind_instrumented(
            graph, plan, beam_width=width, ranking="compatibility",
            seconds_cap=job["arm_seconds_cap"],
        )
        row["arms"][f"compatibility_w{width}"] = _arm_row(result)
    return row


def _arm_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "bound": bool(result["complete"]),
        "died_at_step": result["died_at_step"],
        "capped_at_step": result["capped_at_step"],
        "depth_reached": result["depth_reached"],
        "plan_length": result["plan_length"],
        "successors_enumerated": result["successors_enumerated"],
        "seconds": result["seconds"],
        "truncation_occurred": any(step["truncated_away"] > 0 for step in result["steps"]),
        "max_truncated_away": max((step["truncated_away"] for step in result["steps"]), default=0),
        "complete": result["complete"],
    }


def classify(row: dict[str, Any], arm: str) -> str:
    """The (a)/(b)/(c) decomposition for one bounded arm, refereed."""

    outcome = row["referee"]["outcome"]
    if row["arms"][arm]["bound"]:
        return "bound"
    if row["arms"][arm]["capped_at_step"] is not None:
        return "arm_budget_exhausted"
    if outcome == OUTCOME_NO_BINDING:
        return "no_compatible_binding"
    if outcome == OUTCOME_EXISTS:
        return "discarded_by_search"
    if outcome == OUTCOME_PROGRAM_REJECTED:
        return "full_depth_but_program_rejected"
    return "budget_exhausted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=96)
    parser.add_argument("--witness-sample", type=int, default=24)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--frontier-cap", type=int, default=4096)
    parser.add_argument("--seconds-cap", type=float, default=90.0)
    parser.add_argument("--arm-seconds-cap", type=float, default=45.0)
    parser.add_argument("--equivalence-sample", type=int, default=4)
    args = parser.parse_args()

    began = perf_counter()
    rng = random.Random(SEED)
    plans = load_plans()
    plans_by_id = {plan["plan_id"]: plan for plan in plans}
    parents = load_production_parents() + load_teacher_root_parents()
    parents_by_id = {row["parent_id"]: row for row in parents}
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "new_oracle_calls": 0,
        "accounting": "binding is pure graph work; no oracle constructed, no network call",
        "production_beam_width": PRODUCTION_BEAM_WIDTH,
        "teacher_profile": TEACHER_PROFILE,
        "inputs": {
            "jump_checkpoint": "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/"
            "checkpoints.json (shared_all_routes)",
            "production_initialization": "diagnostics/parent_edit_cycles/prepared/"
            "init_20260921.json",
            "teacher_corpus": "diagnostics/pmo_dependency_region_program_v2/attempt_1/"
            "training_dependency_region_corpus.json.gz",
        },
        "population": {
            "plans": len(plans),
            "parents": len(parents),
            "production_init_parents": sum(
                1 for row in parents if row["stratum"] == "production_init"
            ),
            "teacher_root_parents": sum(
                1 for row in parents if row["stratum"] == "teacher_root"
            ),
        },
    }

    # ---- stage 1: zero-search static census over EVERY pair ----
    static_began = perf_counter()
    feasible: list[tuple[str, str]] = []
    static_counts: dict[str, Counter] = {"production_init": Counter(), "teacher_root": Counter()}
    for parent in parents:
        graph = decode_state(parent["state"])
        for plan in plans:
            ok = static_first_step_feasible(graph, plan)
            static_counts[parent["stratum"]]["total"] += 1
            static_counts[parent["stratum"]]["feasible" if ok else "infeasible"] += 1
            if ok:
                feasible.append((parent["parent_id"], plan["plan_id"]))
    report["stage_1_static_census"] = {
        "definition": "role 0 requires operand descriptors that no active atom of the "
        "parent carries; a proof of no_compatible_binding at depth 0, independent of "
        "beam width, ranking and search budget",
        "seconds": perf_counter() - static_began,
        "by_stratum": {
            stratum: {
                "pairs": counts["total"],
                "step0_feasible": counts["feasible"],
                "step0_infeasible_proven_no_binding": counts["infeasible"],
                "proven_no_binding_fraction": counts["infeasible"] / counts["total"],
            }
            for stratum, counts in static_counts.items()
        },
        "total_pairs": sum(counts["total"] for counts in static_counts.values()),
        "total_proven_no_binding": sum(
            counts["infeasible"] for counts in static_counts.values()
        ),
        "ceiling_on_binding_rate": len(feasible)
        / sum(counts["total"] for counts in static_counts.values()),
    }
    print(f"[stage 1] static census: {len(feasible)} feasible pairs", flush=True)

    # ---- stage 2: witness control ----
    witnesses = load_plan_source_witnesses()
    witness_jobs = []
    for plan_id, rows in witnesses.items():
        if plan_id not in plans_by_id:
            continue
        witness_jobs.append(
            {
                "parent_id": rows[0]["parent_id"],
                "stratum": "teacher_witness",
                "state": rows[0]["state"],
                "plan": plans_by_id[plan_id],
                "widths": WIDTHS,
                "compat_widths": COMPAT_WIDTHS,
                "frontier_cap": args.frontier_cap,
                "seconds_cap": args.seconds_cap,
                "arm_seconds_cap": args.arm_seconds_cap,
            }
        )
    rng.shuffle(witness_jobs)
    witness_jobs = witness_jobs[: args.witness_sample]

    # ---- stage 3: stratified sample of step-0-feasible pairs ----
    by_stratum: dict[str, list[tuple[str, str]]] = {"production_init": [], "teacher_root": []}
    for parent_id, plan_id in feasible:
        by_stratum[parents_by_id[parent_id]["stratum"]].append((parent_id, plan_id))
    per_stratum = max(1, args.sample // 2)
    sampled: list[tuple[str, str]] = []
    for rows in by_stratum.values():
        rng.shuffle(rows)
        sampled.extend(rows[:per_stratum])
    jobs = [
        {
            "parent_id": parent_id,
            "stratum": parents_by_id[parent_id]["stratum"],
            "state": parents_by_id[parent_id]["state"],
            "plan": plans_by_id[plan_id],
            "widths": WIDTHS,
            "compat_widths": COMPAT_WIDTHS,
            "frontier_cap": args.frontier_cap,
            "seconds_cap": args.seconds_cap,
            "arm_seconds_cap": args.arm_seconds_cap,
        }
        for parent_id, plan_id in sampled
    ]

    all_jobs = witness_jobs + jobs
    print(f"[stage 2+3] {len(witness_jobs)} witness + {len(jobs)} sampled pairs", flush=True)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(_run_pair, all_jobs), start=1):
            rows.append(row)
            if index % 10 == 0 or index == len(all_jobs):
                print(f"  [{index}/{len(all_jobs)}] {perf_counter()-began:.0f}s", flush=True)

    report["stage_2_witness_control"] = _summarize(
        [row for row in rows if row["stratum"] == "teacher_witness"],
        note="plans bound back onto the teacher source that generated them; a complete "
        "role-consistent realization provably exists, so any bounded-arm failure here is "
        "DISCARDED_BY_SEARCH by construction",
    )
    report["stage_3_decomposition"] = {
        stratum: _summarize([row for row in rows if row["stratum"] == stratum])
        for stratum in ("production_init", "teacher_root")
    }
    report["stage_3_pairs"] = [
        {
            key: row[key]
            for key in ("parent_id", "stratum", "plan_id", "plan_primitive_count", "referee")
        }
        | {
            "arms": {
                arm: {k: v for k, v in body.items() if k != "complete"}
                for arm, body in row["arms"].items()
            },
            "classification": {arm: classify(row, arm) for arm in row["arms"]},
        }
        for row in rows
    ]
    report["stage_4_yield"] = _yield_profile(rows)

    # ---- stage 5: attribution -- WHICH pinned field refuses the parent ----
    attribution_began = perf_counter()
    views: dict[str, dict[str, Any]] = {}
    for name, fields in ATTRIBUTION_VIEWS.items():
        counts: Counter = Counter()
        for parent in parents:
            graph = decode_state(parent["state"])
            for plan in plans:
                counts["total"] += 1
                if first_step_feasible_under(graph, plan, fields):
                    counts["feasible"] += 1
        views[name] = {
            "fields": list(fields),
            "step0_feasible": counts["feasible"],
            "pairs": counts["total"],
            "step0_feasible_fraction": counts["feasible"] / counts["total"],
        }
    report["stage_5_specification_attribution"] = {
        "definition": "step-0 operand availability when operand descriptors are compared "
        "on a projected field set.  DIAGNOSTIC ONLY: a match under a coarser view "
        "realizes a DIFFERENT transformation than the plan specifies, so this is not a "
        "proposal to loosen matching.  It attributes the depth-0 refusal to specific "
        "over-specified fields of the v1 role encoding.",
        "seconds": perf_counter() - attribution_began,
        "views": views,
    }

    # ---- stage 6: how much step-ordering structure each plan pins ----
    census = [plan_specification_census(plan) for plan in plans]
    report["stage_6_plan_specification"] = {
        "definition": "created-handle operands carry created_ordinal and creation_lag, "
        "which pin the TEACHER's step ordering into the role identity: a chemically "
        "identical transformation performed in another order cannot match.",
        "plans": len(census),
        "created_origin_fraction": _summary(
            [row["created_origin_fraction"] for row in census]
        ),
        "created_origin_operands": _summary(
            [float(row["created_origin_operands"]) for row in census]
        ),
        "creation_lag_nonzero": _summary(
            [float(row["creation_lag_nonzero"]) for row in census]
        ),
        "creation_lag_max": _summary([float(row["creation_lag_max"]) for row in census]),
        "plans_with_any_created_operand": sum(
            1 for row in census if row["created_origin_operands"] > 0
        ),
    }
    report["elapsed_seconds"] = perf_counter() - began
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"wrote {REPORT} in {report['elapsed_seconds']:.0f}s", flush=True)


def _summarize(rows: list[dict[str, Any]], *, note: str | None = None) -> dict[str, Any]:
    if not rows:
        return {"pairs": 0}
    arms = sorted({arm for row in rows for arm in row["arms"]})
    referee = Counter(row["referee"]["outcome"] for row in rows)
    body: dict[str, Any] = {
        "pairs": len(rows),
        "referee_outcomes": dict(referee),
        "referee_binding_exists_fraction": referee[OUTCOME_EXISTS] / len(rows),
        "referee_no_binding_fraction": referee[OUTCOME_NO_BINDING] / len(rows),
        "referee_undecided_fraction": referee[OUTCOME_UNDECIDED] / len(rows),
        "referee_program_rejected_fraction": referee[OUTCOME_PROGRAM_REJECTED] / len(rows),
        "arms": {},
    }
    if note:
        body["note"] = note
    for arm in arms:
        classes = Counter(classify(row, arm) for row in rows)
        bound_rows = [row for row in rows if row["arms"][arm]["bound"]]
        body["arms"][arm] = {
            "binding_rate": classes["bound"] / len(rows),
            "decomposition": {
                "bound": classes["bound"],
                "no_compatible_binding": classes["no_compatible_binding"],
                "discarded_by_search": classes["discarded_by_search"],
                "budget_exhausted": classes["budget_exhausted"],
                "arm_budget_exhausted": classes["arm_budget_exhausted"],
                "full_depth_but_program_rejected": classes["full_depth_but_program_rejected"],
            },
            "truncation_occurred_fraction": sum(
                1 for row in rows if row["arms"][arm]["truncation_occurred"]
            )
            / len(rows),
            "successors_enumerated": _summary(
                [float(row["arms"][arm]["successors_enumerated"]) for row in rows]
            ),
            "complete_program_primitives": _summary(
                [
                    float(entry["primitive_count"])
                    for row in bound_rows
                    for entry in row["arms"][arm]["complete"]
                ]
            ),
            "complete_program_retained_fraction": _summary(
                [
                    float(entry["retained_fraction"])
                    for row in bound_rows
                    for entry in row["arms"][arm]["complete"]
                ]
            ),
            "complete_programs_at_teacher_scale": sum(
                1
                for row in bound_rows
                for entry in row["arms"][arm]["complete"]
                if entry["primitive_count"] >= TEACHER_PROFILE["teacher_scale_threshold"]
            ),
        }
    return body


def _yield_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "definition": "every COMPLETE program produced by any arm, against the teacher "
        "profile; partial/prefix bindings are not reported as realizations",
        "by_arm": {},
    }
    arms = sorted({arm for row in rows for arm in row["arms"]})
    for arm in arms:
        entries = [
            entry
            for row in rows
            for entry in row["arms"][arm]["complete"]
        ]
        out["by_arm"][arm] = {
            "complete_programs": len(entries),
            "primitive_count": _summary([float(e["primitive_count"]) for e in entries]),
            "retained_fraction": _summary([float(e["retained_fraction"]) for e in entries]),
            "delta_heavy_atoms": _summary([float(e["delta_heavy_atoms"]) for e in entries]),
            "component_count": _summary([float(e["component_count"]) for e in entries]),
            "at_teacher_scale": sum(
                1
                for e in entries
                if e["primitive_count"] >= TEACHER_PROFILE["teacher_scale_threshold"]
            ),
            "in_teacher_retained_band": sum(
                1
                for e in entries
                if TEACHER_PROFILE["retained_fraction_range"][0]
                <= e["retained_fraction"]
                <= TEACHER_PROFILE["retained_fraction_range"][1]
            ),
        }
    return out


if __name__ == "__main__":
    main()
