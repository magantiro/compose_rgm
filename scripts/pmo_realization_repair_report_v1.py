#!/usr/bin/env python3
"""Measure whether a weaker role descriptor actually raises complete-program yield.

Produces ``diagnostics/pmo_realization_repair_v1.json``.

The primary metric is P(legal executable COMPLETE realization | plan, parent), reported
with the realized program's primitive count and retained fraction against the teacher
profile.  A method that raises the depth-0 match rate without moving realized SCALE and
RETAINED FRACTION has not fixed anything, and the report says so in those words.

Arms are MATCHED: every specification is run on the same sampled (plan, parent) pairs
with the same budget, so a difference is attributable to the specification and not to
the pair pool (a control this repository has been burned by omitting before).

Zero oracle calls.  No Modal launches.  No pinned module is modified.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

from pmo_binder_repair_v1 import (
    PRODUCTION_BEAM_WIDTH,
    bind_instrumented,
    load_plan_source_witnesses,
    load_plans,
    load_production_parents,
    load_teacher_root_parents,
)
from pmo_realization_repair_v1 import (
    OUTCOME_COMPLETED,
    OUTCOME_INCOMPATIBLE,
    OUTCOMES,
    SCHEMA,
    SPEC_R1,
    SPEC_R2,
    SPEC_V1_EXACT,
    _Prefix,
    plan_demand,
    plan_validity,
    propagate,
    realize,
    role_match,
    static_first_step_feasible,
)

from compose_v4.control.docking_value import identity
from compose_v4.control.pmo_action_roles import action_role_supervision
from compose_v4.control.pmo_joint_dependency_jump import (
    bind_joint_plan,
    enumerate_role_successors,
)
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "diagnostics/pmo_realization_repair_v1.json"
SEED = 20260920

TEACHER_PROFILE = {
    "primitive_count_median_range": [29, 40],
    "retained_fraction_range": [0.52, 0.96],
    "teacher_scale_threshold": 27,
    "source": "diagnostics/pmo_binder_repair_v1.json (teacher profile carried forward)",
}

# (arm name, specification, enforce the plan's declared completion conditions)
#
# Three MATCHED arms on identical pairs and identical budgets.  The declared completion
# conditions are enforced in every arm, which is the strict reading: a realization must
# meet the plan's own conditions.  How often they bite is not lost by this -- the search
# keeps looking after a rejection, and ``full_depth_but_completion_rejected`` plus
# ``completion_rejections`` record every case where a relaxed arm reached full depth and
# was refused.  That is the trap-check: a relaxation that "succeeds" only by producing a
# structurally different program is visible here rather than counted as yield.
ARMS: tuple[tuple[str, Any, bool], ...] = (
    ("v1_exact", SPEC_V1_EXACT, True),
    ("R1_drop_environment_and_lag", SPEC_R1, True),
    ("R2_plus_capacity", SPEC_R2, True),
)


def _environment() -> dict[str, Any]:
    import platform

    import networkx
    import numpy
    import rdkit

    return {
        "python": platform.python_version(),
        "rdkit": rdkit.__version__,
        "numpy": numpy.__version__,
        "networkx": networkx.__version__,
    }


def _preexisting_deletes(plan: dict[str, Any]) -> int:
    """atom_delete roles that consume a PARENT atom rather than the plan's own product."""

    return sum(
        1
        for role in plan["roles"]
        if str(role["executor_rule"]) == "atom_delete"
        and role["operands"][0]["descriptor"].get("origin") == "preexisting"
    )


def structural_retained_bound(
    plans: list[dict[str, Any]], parents: list[dict[str, Any]]
) -> dict[str, Any]:
    """What retained fraction a COMPLETE realization of each plan must have.

    Zero search.  ``origin`` is part of the role identity under EVERY specification
    here, so an ``atom_delete`` role declared ``preexisting`` consumes a parent atom in
    any realization.  The retained fraction of a complete realization is therefore
    determined by the plan and the parent's size, up to slot reuse (a freed slot
    refilled by a later insertion counts as retained, which makes this a LOWER bound on
    retained fraction and hence conservative in the direction that matters).

    This is what makes the "every realized program has retained_fraction 1.000" finding
    interpretable: it identifies exactly which plans can produce it.
    """

    threshold = TEACHER_PROFILE["teacher_scale_threshold"]
    low, high = TEACHER_PROFILE["retained_fraction_range"]
    rows = []
    for plan in plans:
        rows.append(
            {
                "plan_id": plan["plan_id"],
                "primitive_count": int(plan["primitive_count"]),
                "preexisting_deletes": _preexisting_deletes(plan),
                "created_dependency_count": int(plan["created_dependency_count"]),
            }
        )
    purely_additive = [row for row in rows if row["preexisting_deletes"] == 0]
    teacher_scale = [row for row in rows if row["primitive_count"] >= threshold]
    implied: list[float] = []
    in_band = 0
    for row in teacher_scale:
        for parent in parents:
            size = int(decode_state(parent["state"]).n_real_atoms)
            value = max(0.0, (size - row["preexisting_deletes"]) / size)
            implied.append(value)
            in_band += int(low <= value <= high)
    return {
        "definition": "retained fraction implied by the plan's own preexisting-origin "
        "deletions and the parent's heavy-atom count, with no slot reuse.",
        "plans": len(rows),
        "purely_additive_plans": {
            "count": len(purely_additive),
            "note": "a complete realization of one of these has retained_fraction 1.000 "
            "BY CONSTRUCTION -- it removes nothing from the parent",
            "primitive_counts": sorted(row["primitive_count"] for row in purely_additive),
            "created_dependency_counts": sorted(
                row["created_dependency_count"] for row in purely_additive
            ),
        },
        "teacher_scale_plans": {
            "count": len(teacher_scale),
            "preexisting_deletes": _summary(
                [float(row["preexisting_deletes"]) for row in teacher_scale]
            ),
            "created_dependency_count": _summary(
                [float(row["created_dependency_count"]) for row in teacher_scale]
            ),
            "purely_additive": sum(1 for row in teacher_scale if row["preexisting_deletes"] == 0),
        },
        "implied_retained_fraction_teacher_scale_x_parent": _summary(implied),
        "implied_in_teacher_retained_band": in_band,
        "implied_pairs": len(implied),
        "implied_in_band_fraction": in_band / len(implied) if implied else 0.0,
    }


def structural_step0_census(
    plans: list[dict[str, Any]], parents: list[dict[str, Any]]
) -> dict[str, Any]:
    """Search-free, load-independent refusal census at step 0, per specification.

    Runs the full forward propagation (slot capacity, preexisting-atom budget,
    created-handle reachability, operand availability) on the parent itself.  Every
    check is a necessary condition, so a refusal here is a proof of
    ``proven_incompatible`` that no search of any shape or budget can overturn.

    The value of separating the reasons: the descriptor-availability refusal is what a
    relaxation moves, while the atom-budget refusals are NOT movable by any
    specification here, because ``atom_type`` and ``formal_charge`` are held in all of
    them.  That gives a ceiling on what relaxing the descriptor can possibly buy.
    """

    threshold = TEACHER_PROFILE["teacher_scale_threshold"]
    specs = (SPEC_V1_EXACT, SPEC_R1, SPEC_R2)
    counts: dict[str, dict[str, dict[str, int]]] = {
        spec.name: {"teacher_scale": {}, "short": {}} for spec in specs
    }
    totals = {"teacher_scale": 0, "short": 0}
    for parent in parents:
        graph = decode_state(parent["state"])
        prefix = _Prefix(graph=graph, actions=(), created=(), next_ordinal=0)
        for plan in plans:
            scale = (
                "teacher_scale"
                if int(plan["primitive_count"]) >= threshold
                else "short"
            )
            totals[scale] += 1
            demand = plan_demand(plan)
            for spec in specs:
                reason = propagate(prefix, 0, plan, demand, spec) or "passes_step_0"
                bucket = counts[spec.name][scale]
                bucket[reason] = bucket.get(reason, 0) + 1
    return {
        "definition": "step-0 forward propagation refusal reason, by specification and "
        "plan scale.  Necessary conditions only, so a refusal is a proof.",
        "totals": totals,
        "by_specification": {
            name: {
                scale: {
                    "counts": rows,
                    "passes_fraction": rows.get("passes_step_0", 0) / totals[scale]
                    if totals[scale]
                    else 0.0,
                }
                for scale, rows in scales.items()
            }
            for name, scales in counts.items()
        },
        "note": "the atom-budget refusals are identical across every specification: "
        "atom_type and formal_charge are never relaxed, so no relaxation here can move "
        "them.  They are a hard search-free ceiling on realizability.",
    }


def build_verdict(
    yield_table: dict[str, Any], static_section: dict[str, Any]
) -> dict[str, Any]:
    """Answer the question the task asks, computed FROM the measured table.

    Written programmatically so the verdict cannot drift from the numbers it rests on.
    The guard the task names explicitly: a method that raises the match rate without
    moving realized SCALE and RETAINED FRACTION has not fixed anything.
    """

    scope = yield_table.get("teacher_scale_plans_only", yield_table["overall"])
    control = scope["v1_exact"]
    arms = {name: scope[name] for name, _, _ in ARMS if name != "v1_exact"}
    depth0 = static_section.get("feasible_fraction", {}) if not static_section.get(
        "skipped"
    ) else {}
    best_scale = max(
        (row["at_teacher_scale"] for row in arms.values()), default=0
    )
    best_band = max((row["in_teacher_retained_band"] for row in arms.values()), default=0)
    best_yield = max(
        (row["complete_realization_yield"] for row in arms.values()), default=0.0
    )
    raised_depth0 = bool(
        depth0
        and max(v for k, v in depth0.items() if k != "v1_exact")
        > depth0.get("v1_exact", 0.0)
    )
    raised_yield = best_yield > control["complete_realization_yield"]
    raised_scale = best_scale > control["at_teacher_scale"]
    raised_band = best_band > control["in_teacher_retained_band"]
    if raised_scale and raised_band:
        headline = (
            "REPAIRED: the weaker descriptor produced complete programs at teacher "
            "scale AND inside the teacher retained band."
        )
    elif raised_depth0 and not raised_scale:
        headline = (
            "NOT REPAIRED: the weaker descriptor raised depth-0 feasibility (and so the "
            "binding-rate ceiling) but produced NO additional complete program at "
            "teacher scale.  This is the failure mode the task named in advance -- a "
            "raised match rate that does not move realized scale is not a fix."
        )
    else:
        headline = (
            "INCONCLUSIVE on the primary metric: neither arm produced a complete "
            "program at teacher scale, and the control did not either."
        )
    return {
        "headline": headline,
        "relaxation_raised_depth0_feasibility": raised_depth0,
        "depth0_feasible_fraction": depth0,
        "relaxation_raised_complete_realization_yield": raised_yield,
        "relaxation_raised_teacher_scale_yield": raised_scale,
        "relaxation_raised_in_band_yield": raised_band,
        "control_at_teacher_scale": control["at_teacher_scale"],
        "best_relaxed_at_teacher_scale": best_scale,
        "control_in_teacher_retained_band": control["in_teacher_retained_band"],
        "best_relaxed_in_teacher_retained_band": best_band,
        "control_yield": control["complete_realization_yield"],
        "best_relaxed_yield": best_yield,
        "scope": "teacher_scale_plans_only -- the cut the question is about",
        "decided_fraction_by_arm": {name: scope[name]["decided_fraction"] for name, _, _ in ARMS},
        "budget_reason_by_arm": {name: scope[name]["budget_reason_counts"] for name, _, _ in ARMS},
        "caveat": "an arm whose decided_fraction is low is budget-limited: its "
        "proven_incompatible count is a lower bound and its yield is not an estimate.",
    }


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def q(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": q(0.5),
        "p90": q(0.9),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


# ---- worker ----

_CACHE: dict[str, Any] = {}


def _context() -> dict[str, Any]:
    if not _CACHE:
        plans = {plan["plan_id"]: plan for plan in load_plans()}
        _CACHE["plans"] = plans
        _CACHE["parents"] = {
            row["parent_id"]: row
            for row in (*load_production_parents(), *load_teacher_root_parents())
        }
        witnesses = load_plan_source_witnesses()
        _CACHE["witnesses"] = {
            row["parent_id"]: row for rows in witnesses.values() for row in rows
        }
    return _CACHE


def _resolve_parent(parent_id: str) -> dict[str, Any]:
    context = _context()
    if parent_id in context["parents"]:
        return context["parents"][parent_id]
    return context["witnesses"][parent_id]


def _realize_task(job: dict[str, Any]) -> dict[str, Any]:
    context = _context()
    plan = context["plans"][job["plan_id"]]
    parent = _resolve_parent(job["parent_id"])
    source = decode_state(parent["state"])
    rows = {}
    for name, spec, declared in ARMS:
        result = realize(
            source,
            plan,
            spec,
            node_budget=job["node_budget"],
            seconds_cap=job["seconds_cap"],
            enforce_declared=declared,
        )
        rows[name] = {
            "outcome": result["outcome"],
            "budget_reason": result.get("budget_reason"),
            "nodes_expanded": result["nodes_expanded"],
            "successors_enumerated": result["successors_enumerated"],
            "max_depth_reached": result["max_depth_reached"],
            "full_depth_prefixes": result["full_depth_prefixes"],
            "full_depth_but_completion_rejected": result["full_depth_but_completion_rejected"],
            "completion_rejections": result["completion_rejections"],
            "pruned_by_propagation": result["pruned_by_propagation"],
            "pruned_by_memo": result["pruned_by_memo"],
            "seconds": result["seconds"],
            "realizations": [
                {
                    key: row[key]
                    for key in (
                        "primitive_count",
                        "plan_primitive_count",
                        "retained_fraction",
                        "component_count",
                        "declared_component_count",
                        "created_dependency_edges",
                        "declared_created_dependency_count",
                        "delta_heavy_atoms",
                        "endpoint_key",
                    )
                }
                for row in result["realizations"]
            ],
        }
    return {
        "plan_id": job["plan_id"],
        "parent_id": job["parent_id"],
        "stratum": job["stratum"],
        "plan_scale": job["plan_scale"],
        "plan_primitive_count": int(plan["primitive_count"]),
        "arms": rows,
    }


def _static_task(job: dict[str, Any]) -> dict[str, Any]:
    context = _context()
    parent = _resolve_parent(job["parent_id"])
    source = decode_state(parent["state"])
    out = {"parent_id": job["parent_id"], "stratum": job["stratum"], "by_spec": {}}
    for spec in (SPEC_V1_EXACT, SPEC_R1, SPEC_R2):
        feasible = 0
        for plan in context["plans"].values():
            feasible += int(static_first_step_feasible(source, plan, spec))
        out["by_spec"][spec.name] = feasible
    out["plans"] = len(context["plans"])
    return out


# ---- equivalence proof ----


def equivalence_evidence(limit_parents: int = 6, limit_plans: int = 8) -> dict[str, Any]:
    """Prove the control arm IS the pinned predicate, on a discriminating non-empty case."""

    plans = load_plans()
    parents = (*load_production_parents(), *load_teacher_root_parents())
    agree = disagree = reference_true = reference_false = relaxed_true = 0
    for parent in parents[:limit_parents]:
        graph = decode_state(parent["state"])
        for plan in plans[:limit_plans]:
            desired = plan["roles"][0]
            desired_identity = identity(desired)
            for candidate in enumerate_role_successors(graph, desired, created={}, step=0):
                observed, _ = action_role_supervision(graph, candidate.action_record, {}, 0, 0)
                reference = identity(observed) == desired_identity
                probe = role_match(
                    desired, observed, SPEC_V1_EXACT, desired_identity=desired_identity
                )
                agree += int(reference == probe)
                disagree += int(reference != probe)
                reference_true += int(reference)
                reference_false += int(not reference)
                relaxed_true += int(role_match(desired, observed, SPEC_R1))
    # And the harness itself against the LIVE pinned entry point, on a non-empty case.
    binder_checks: list[dict[str, Any]] = []
    for parent in parents[:limit_parents]:
        graph = decode_state(parent["state"])
        for plan in plans[:limit_plans]:
            live = bind_joint_plan(graph, plan, beam_width=PRODUCTION_BEAM_WIDTH)
            probe = bind_instrumented(
                graph, plan, beam_width=PRODUCTION_BEAM_WIDTH, ranking="hash"
            )
            binder_checks.append(
                {
                    "live_bindings": len(live),
                    "probe_bindings": len(probe["complete"]),
                    "endpoint_sets_identical": sorted(row["endpoint_key"] for row in live)
                    == sorted(row["endpoint_key"] for row in probe["complete"]),
                }
            )
    non_empty = [row for row in binder_checks if row["live_bindings"] > 0]
    return {
        "predicate_equivalence": {
            "note": "role_match under SPEC_V1_EXACT vs the pinned rule "
            "identity(observed) == identity(desired) (pmo_joint_dependency_jump.py:344)",
            "comparisons": agree + disagree,
            "agree": agree,
            "disagree": disagree,
            "reference_true_cases": reference_true,
            "reference_false_cases": reference_false,
            "discriminating": reference_true > 0 and reference_false > 0,
            "relaxed_R1_true_cases": relaxed_true,
            "relaxation_is_a_strict_widening": relaxed_true >= reference_true,
        },
        "pinned_binder_equivalence": {
            "note": "live bind_joint_plan vs bind_instrumented(ranking='hash') at the "
            "production beam width",
            "checks": len(binder_checks),
            "identical": sum(1 for row in binder_checks if row["endpoint_sets_identical"]),
            "non_empty_checks": len(non_empty),
            "non_empty_identical": sum(
                1 for row in non_empty if row["endpoint_sets_identical"]
            ),
            "non_empty_binding_counts": [row["live_bindings"] for row in non_empty],
            "vacuous": not non_empty,
        },
    }


# ---- aggregation ----


def _arm_summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    outcomes = dict.fromkeys(OUTCOMES, 0)
    realized: list[dict[str, Any]] = []
    completion_rejected = 0
    for row in rows:
        entry = row["arms"][arm]
        outcomes[entry["outcome"]] += 1
        completion_rejected += int(entry["full_depth_but_completion_rejected"])
        realized.extend(entry["realizations"])
    threshold = TEACHER_PROFILE["teacher_scale_threshold"]
    low, high = TEACHER_PROFILE["retained_fraction_range"]
    at_scale = [row for row in realized if row["primitive_count"] >= threshold]
    in_band = [row for row in realized if low <= row["retained_fraction"] <= high]
    return {
        "pairs": len(rows),
        "outcomes": outcomes,
        "complete_realization_yield": outcomes[OUTCOME_COMPLETED] / len(rows) if rows else 0.0,
        "decided_fraction": (
            (outcomes[OUTCOME_COMPLETED] + outcomes[OUTCOME_INCOMPATIBLE]) / len(rows)
            if rows
            else 0.0
        ),
        "full_depth_but_completion_rejected": completion_rejected,
        "realized_programs": len(realized),
        "realized_primitive_count": _summary(
            [float(row["primitive_count"]) for row in realized]
        ),
        "realized_retained_fraction": _summary(
            [float(row["retained_fraction"]) for row in realized]
        ),
        "realized_delta_heavy_atoms": _summary(
            [float(row["delta_heavy_atoms"]) for row in realized]
        ),
        "at_teacher_scale": len(at_scale),
        "in_teacher_retained_band": len(in_band),
        "at_teacher_scale_and_in_band": sum(
            1
            for row in realized
            if row["primitive_count"] >= threshold and low <= row["retained_fraction"] <= high
        ),
        "budget_reason_counts": {
            reason: sum(
                1
                for row in rows
                if row["arms"][arm]["budget_reason"] == reason
            )
            for reason in ("node_budget", "seconds_cap")
        },
        "median_nodes_expanded": _summary(
            [float(row["arms"][arm]["nodes_expanded"]) for row in rows]
        ).get("median"),
        "median_successors_enumerated": _summary(
            [float(row["arms"][arm]["successors_enumerated"]) for row in rows]
        ).get("median"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=20, help="pairs per stratum")
    parser.add_argument(
        "--short-sample", type=int, default=5, help="short-plan control pairs per stratum"
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--node-budget",
        type=int,
        default=80,
        help="PRIMARY budget.  Node count is load-independent; the wall cap is only a "
        "safety net, because a wall-limited search measures machine load as much as "
        "the specification under test.",
    )
    parser.add_argument("--seconds-cap", type=float, default=400.0)
    parser.add_argument("--skip-static", action="store_true")
    args = parser.parse_args()

    began = perf_counter()
    rng = random.Random(SEED)
    plans = load_plans()
    plan_ids = [plan["plan_id"] for plan in plans]
    production = load_production_parents()
    teacher_roots = load_teacher_root_parents()
    witnesses = load_plan_source_witnesses()

    # ---- stage 1: plan-level validity (the invalid_plan outcome, parent-independent) ----
    validity_rows = [{"plan_id": p["plan_id"], **plan_validity(p)} for p in plans]
    invalid = [row for row in validity_rows if not row["valid"]]

    # ---- stage 2: depth-0 feasibility over the FULL grid, per specification ----
    static_section: dict[str, Any] = {"skipped": True}
    if not args.skip_static:
        static_jobs = [
            {"parent_id": row["parent_id"], "stratum": row["stratum"]}
            for row in (*production, *teacher_roots)
        ]
        started = perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            static_rows = list(pool.map(_static_task, static_jobs))
        by_stratum: dict[str, dict[str, Any]] = {}
        for row in static_rows:
            bucket = by_stratum.setdefault(
                row["stratum"], {"pairs": 0, "feasible": dict.fromkeys(
                    (SPEC_V1_EXACT.name, SPEC_R1.name, SPEC_R2.name), 0)}
            )
            bucket["pairs"] += row["plans"]
            for name, count in row["by_spec"].items():
                bucket["feasible"][name] += count
        total_pairs = sum(bucket["pairs"] for bucket in by_stratum.values())
        overall = dict.fromkeys((SPEC_V1_EXACT.name, SPEC_R1.name, SPEC_R2.name), 0)
        for bucket in by_stratum.values():
            for name, count in bucket["feasible"].items():
                overall[name] += count
        static_section = {
            "definition": "role 0 has an operand descriptor no active atom of the parent "
            "carries, under the comparison each specification makes.  A zero here is a "
            "proof of proven_incompatible at depth 0, independent of search.",
            "total_pairs": total_pairs,
            "feasible_fraction": {
                name: count / total_pairs if total_pairs else 0.0
                for name, count in overall.items()
            },
            "by_stratum": {
                name: {
                    "pairs": bucket["pairs"],
                    "feasible_fraction": {
                        spec: count / bucket["pairs"] if bucket["pairs"] else 0.0
                        for spec, count in bucket["feasible"].items()
                    },
                }
                for name, bucket in by_stratum.items()
            },
            "seconds": perf_counter() - started,
        }

    # ---- stage 3: matched realization yield on sampled pairs ----
    # Stratify by PLAN SCALE, not uniformly over the grid.  The question is whether a
    # weaker descriptor produces complete programs AT TEACHER SCALE, so teacher-scale
    # plans get the sample; short plans are carried as a positive control that the
    # pipeline does produce completions at all.  Sampling uniformly would spend most of
    # the budget on plans that cannot answer the question either way.
    threshold = TEACHER_PROFILE["teacher_scale_threshold"]
    by_id = {plan["plan_id"]: plan for plan in plans}
    teacher_plan_ids = [
        pid for pid in plan_ids if int(by_id[pid]["primitive_count"]) >= threshold
    ]
    short_plan_ids = [
        pid for pid in plan_ids if int(by_id[pid]["primitive_count"]) < threshold
    ]

    def _take(grid: list[tuple[str, str]], count: int) -> list[tuple[str, str]]:
        rng.shuffle(grid)
        return grid[:count]

    jobs: list[dict[str, Any]] = []
    for parents, stratum in ((production, "production_init"), (teacher_roots, "teacher_root")):
        for ids, scale, count in (
            (teacher_plan_ids, "teacher_scale", args.sample),
            (short_plan_ids, "short", args.short_sample),
        ):
            grid = [(plan_id, row["parent_id"]) for plan_id in ids for row in parents]
            for plan_id, parent_id in _take(grid, count):
                jobs.append(
                    {
                        "plan_id": plan_id,
                        "parent_id": parent_id,
                        "stratum": stratum,
                        "plan_scale": scale,
                    }
                )
    for ids, scale, count in (
        (set(teacher_plan_ids), "teacher_scale", args.sample),
        (set(short_plan_ids), "short", args.short_sample),
    ):
        grid = [
            (plan_id, rows[0]["parent_id"])
            for plan_id, rows in sorted(witnesses.items())
            if plan_id in ids and rows
        ]
        for plan_id, parent_id in _take(grid, count):
            jobs.append(
                {
                    "plan_id": plan_id,
                    "parent_id": parent_id,
                    "stratum": "teacher_witness",
                    "plan_scale": scale,
                }
            )
    for job in jobs:
        job["node_budget"] = args.node_budget
        job["seconds_cap"] = args.seconds_cap

    started = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pair_rows = list(pool.map(_realize_task, jobs))
    realization_seconds = perf_counter() - started

    yield_table: dict[str, Any] = {"overall": {}, "by_stratum": {}}
    for arm, _, _ in ARMS:
        yield_table["overall"][arm] = _arm_summary(pair_rows, arm)
    for stratum in ("production_init", "teacher_root", "teacher_witness"):
        subset = [row for row in pair_rows if row["stratum"] == stratum]
        yield_table["by_stratum"][stratum] = {
            arm: _arm_summary(subset, arm) for arm, _, _ in ARMS
        }
    for scale in ("teacher_scale", "short"):
        subset = [row for row in pair_rows if row["plan_scale"] == scale]
        yield_table.setdefault("by_plan_scale", {})[scale] = {
            arm: _arm_summary(subset, arm) for arm, _, _ in ARMS
        }
    for stratum in ("production_init", "teacher_root", "teacher_witness"):
        subset = [
            row
            for row in pair_rows
            if row["stratum"] == stratum and row["plan_scale"] == "teacher_scale"
        ]
        yield_table.setdefault("teacher_scale_by_stratum", {})[stratum] = {
            arm: _arm_summary(subset, arm) for arm, _, _ in ARMS
        }
    teacher_scale_rows = [row for row in pair_rows if row["plan_scale"] == "teacher_scale"]
    yield_table["teacher_scale_plans_only"] = {
        arm: _arm_summary(teacher_scale_rows, arm) for arm, _, _ in ARMS
    }

    equivalence = equivalence_evidence()

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    payload = {
        "schema_version": SCHEMA,
        "question": "Does a weaker role descriptor actually raise COMPLETE-PROGRAM yield "
        "at teacher scale and inside the teacher retained band?",
        "teacher_profile": TEACHER_PROFILE,
        "contract": {
            "new_oracle_calls": 0,
            "modal_launches": 0,
            "pinned_modules_modified": [],
            "production_wiring_changed": False,
            "note": "pmo_joint_dependency_jump, pmo_action_roles and "
            "pmo_population_controller are imported read-only.  All chemistry -- "
            "successor enumeration, role supervision, program extraction, executor "
            "replay -- is the pinned implementation in every arm.  An arm differs only "
            "in the ACCEPTANCE PREDICATE and the search.",
        },
        "specifications": {
            "v1_exact": {
                "relaxed": [],
                "note": "whole-role hash equality, the pinned predicate",
            },
            "R1_drop_environment_and_lag": {
                "relaxed": sorted(SPEC_R1.relaxed),
                "kept": list(SPEC_R1.compared_fields),
            },
            "R2_plus_capacity": {
                "relaxed": sorted(SPEC_R2.relaxed),
                "kept": list(SPEC_R2.compared_fields),
            },
            "never_relaxed": {
                "executor_rule": "the operation performed",
                "model_family": "the operator family",
                "parameters": "the full edit semantics (inserted element/charge/H, "
                "attachment bond classes, restate target class, closure bond class, "
                "ring restate bond classes)",
                "operand_atom_type": "attaching to a carbon is not the same operation "
                "as attaching to an oxygen",
                "operand_formal_charge": "charge is preserved, never optimized",
                "operand_origin": "preexisting vs route_created",
                "operand_created_ordinal": "the dataflow edge that makes the plan a "
                "program rather than N unrelated edits",
            },
            "added_back": {
                "declared_component_count": "the realized dependency-region program must "
                "have the component count the plan declares",
                "declared_created_dependency_count": "and the declared number of "
                "created-handle dependency edges",
                "forward_propagation": "element/charge budget for remaining preexisting "
                "deletions, created-handle reachability, free-slot capacity, one-step "
                "operand availability",
                "rationale": "relaxing the per-atom fingerprint removes an ACCIDENTAL "
                "proxy for constraints the v1 latent never expressed (it stores "
                "per-operand descriptors only, so pairwise operand topology such as the "
                "ring size a cycle_close produces is not recoverable from it).  The "
                "declared program structure is enforced so a relaxation cannot "
                "manufacture a 'successful' binding.",
            },
        },
        "outcomes": list(OUTCOMES),
        "stage_0_structural_retained_bound": structural_retained_bound(
            plans, [*production, *teacher_roots]
        ),
        "stage_0b_structural_step0_census": structural_step0_census(
            plans, [*production, *teacher_roots]
        ),
        "stage_1_plan_validity": {
            "definition": "defects of the LATENT, independent of any parent: a created "
            "operand referencing an ordinal no earlier role emits, a declared primitive "
            "or component count above the runtime maximum, or a declared "
            "created-dependency count disagreeing with the roles themselves.",
            "plans": len(validity_rows),
            "invalid": len(invalid),
            "invalid_rows": invalid,
        },
        "stage_2_static_depth0": static_section,
        "stage_3_yield": yield_table,
        "verdict": build_verdict(yield_table, static_section),
        "stage_4_equivalence": equivalence,
        "provenance": {
            "environment": _environment(),
            "environment_note": "NOT the pinned production image (python 3.11, rdkit "
            "2024.3.5).  Every arm runs in the SAME interpreter on the SAME pairs, so "
            "the matched comparison between specifications is internally valid; "
            "absolute rates are conditional on this chemistry kernel and should not be "
            "quoted as production numbers.  This is the interpreter the located-defect "
            "report (pmo_binder_repair_v1) also used, so the two are comparable.",
            "git_head": head,
            "branch": subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
            "seed": SEED,
            "sample_per_stratum": args.sample,
            "short_sample_per_stratum": args.short_sample,
            "node_budget": args.node_budget,
            "seconds_cap": args.seconds_cap,
            "workers": args.workers,
            "realization_seconds": realization_seconds,
            "reproduce": "KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 "
            "PYTHONPATH=src:scripts python3 scripts/pmo_realization_repair_report_v1.py "
            f"--sample {args.sample} --workers {args.workers} "
            f"--node-budget {args.node_budget} --seconds-cap {args.seconds_cap}",
        },
        "pairs": pair_rows,
        "elapsed_seconds": perf_counter() - began,
        "new_oracle_calls": 0,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {REPORT}")
    print("  VERDICT:", payload["verdict"]["headline"])
    for arm, _, _ in ARMS:
        row = yield_table["overall"][arm]
        print(
            f"  {arm:32s} yield={row['complete_realization_yield']:.3f} "
            f"at_scale={row['at_teacher_scale']} in_band={row['in_teacher_retained_band']} "
            f"outcomes={row['outcomes']}"
        )


if __name__ == "__main__":
    main()
