#!/usr/bin/env python3
"""Source-sharded zero-oracle conditional-realizer gate for T4 subgoals."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from compose_v4.control.structural_subgoal import extract_structural_goal
from compose_v4.control.structural_subgoal_realizer import RealizerConfig, realize_structural_goal
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.rewrite.trace_shard import decode_state
from tools.t4_structural_subgoal_audit import LIBRARY, ROOT, teacher_traces

RESULT_SCHEMA = "t4_structural_subgoal_realizer_source_audit_v2"
PROGRESS_SCHEMA = "t4_structural_subgoal_realizer_progress_v2"


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def run_source(
    output: Path,
    *,
    source_group: str,
    code_revision: str,
    working_tree_dirty: bool,
    progress_publisher: Callable[[dict], None] | None = None,
    heartbeat_seconds: float = 30.0,
    config: RealizerConfig | None = None,
) -> dict:
    """Run one immutable source shard without exposing teacher actions to the realizer."""

    if output.exists():
        raise ValueError(f"refusing to overwrite structural-realizer shard: {output}")
    if working_tree_dirty:
        raise ValueError("structural-realizer audit requires clean committed source")
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat interval must be positive")
    started = time.monotonic()
    selected = [row for row in teacher_traces() if row["source_group"] == source_group]
    if not selected:
        raise ValueError(f"unknown or empty T4 source group: {source_group}")
    prepared = []
    for teacher in selected:
        trace = teacher["trace"]
        states, actions = tuple(trace["states"]), tuple(trace["actions"])
        source = decode_state(states[0])
        goal, teacher_bindings, regions = extract_structural_goal(states, actions)
        prepared.append((teacher, source, goal, teacher_bindings, actions, regions))
    subgoals_total = sum(len(goal.subgoals) for _, _, goal, _, _, _ in prepared)
    work_total = len(prepared)
    config = RealizerConfig() if config is None else config
    last_heartbeat = float("-inf")
    total_expansions = total_attempts = 0
    route_status_counts: Counter[str] = Counter()
    subgoal_status_counts: Counter[str] = Counter()

    def emit(
        *,
        force: bool,
        work_completed: int,
        routes_completed: int,
        subgoals_audited: int,
        program_id: str | None,
        subgoal_index: int | None = None,
        current: dict | None = None,
        phase: str = "route_conditional_realization",
    ) -> None:
        nonlocal last_heartbeat
        now = time.monotonic()
        if not force and now - last_heartbeat < heartbeat_seconds:
            return
        elapsed = now - started
        current = current or {}
        expanded = total_expansions + int(current.get("expanded", 0))
        attempted = total_attempts + int(current.get("attempted", 0))
        current_limit = int(current.get("maximum_expansions", 0))
        current_expanded = int(current.get("expanded", 0))
        if current_limit:
            remaining_limit_work = max(0, current_limit - current_expanded)
            remaining_units = max(0, work_total - work_completed - 1)
        else:
            remaining_limit_work = 0
            remaining_units = max(0, work_total - work_completed)
        remaining_limit_work += remaining_units * config.maximum_expansions
        expansions_per_second = expanded / elapsed if elapsed > 0 else 0.0
        eta_limit = (
            remaining_limit_work / expansions_per_second if expansions_per_second > 0 else None
        )
        completed_rate = work_completed / elapsed if elapsed > 0 else 0.0
        eta_empirical = (
            (work_total - work_completed) / completed_rate if completed_rate > 0 else None
        )
        payload = {
            "schema_version": PROGRESS_SCHEMA,
            "event": "structural_subgoal_realizer_heartbeat",
            "source_group": source_group,
            "phase": phase,
            "current_program_id": program_id,
            "current_subgoal_index": subgoal_index,
            "work_units_completed": work_completed,
            "work_units_total": work_total,
            "routes_completed": routes_completed,
            "routes_total": len(prepared),
            "subgoals_audited": subgoals_audited,
            "subgoals_total": subgoals_total,
            "elapsed_seconds": elapsed,
            "completed_work_units_per_second": completed_rate,
            "search_expansions": expanded,
            "action_attempts": attempted,
            "expansions_per_second": expansions_per_second,
            "current_route": current,
            "route_status_counts": dict(sorted(route_status_counts.items())),
            "subgoal_status_counts": dict(sorted(subgoal_status_counts.items())),
            "estimated_remaining_seconds_from_completed_work": eta_empirical,
            "estimated_remaining_seconds_to_configured_search_limit": eta_limit,
            "eta_is_operational_not_a_stopping_rule": True,
            "at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if progress_publisher is not None:
            progress_publisher(payload)
        else:
            print(json.dumps(payload, sort_keys=True), flush=True)
        last_heartbeat = now

    emit(
        force=True,
        work_completed=0,
        routes_completed=0,
        subgoals_audited=0,
        program_id=None,
        phase="starting",
    )
    route_rows = []
    subgoal_rows = []
    subgoals_audited = 0
    for route_index, (
        teacher,
        source,
        goal,
        teacher_bindings,
        actions,
        regions,
    ) in enumerate(prepared):
        routes_completed = route_index
        work_completed = route_index
        route_started = time.monotonic()

        def route_progress(
            current: dict,
            *,
            completed_work: int = work_completed,
            completed_routes: int = routes_completed,
            completed_subgoals: int = subgoals_audited,
            program_id: str = teacher["program_id"],
        ) -> None:
            emit(
                force=False,
                work_completed=completed_work,
                routes_completed=completed_routes,
                subgoals_audited=completed_subgoals,
                program_id=program_id,
                current=current,
                phase="route_conditional_realization",
            )

        realized = realize_structural_goal(
            source,
            goal,
            teacher_bindings,
            config=config,
            progress=route_progress,
        )
        total_expansions += int(realized["expanded"])
        total_attempts += int(realized["attempted"])
        route_status_counts[realized["status"]] += 1
        route_elapsed_seconds = time.monotonic() - route_started
        subgoal_matches = realized["subgoal_targets_match_within_complete_goal"]
        if len(subgoal_matches) != len(goal.subgoals):
            raise AssertionError("realizer returned the wrong number of subgoal results")
        for subgoal_index, (subgoal, exact) in enumerate(
            zip(goal.subgoals, subgoal_matches, strict=True)
        ):
            subgoal_status = "realized" if realized["status"] == "realized" else "unrealized"
            subgoal_status_counts[subgoal_status] += 1
            subgoal_rows.append(
                {
                    "program_id": teacher["program_id"],
                    "teacher_receipt_sha256": teacher["receipt_sha256"],
                    "parent_goal_id": goal.goal_id,
                    "subgoal_index": subgoal_index,
                    "subgoal_id": subgoal.subgoal_id,
                    "teacher_primitive_indices": list(
                        regions["components"][subgoal_index]["primitive_indices"]
                    ),
                    "realized_within_complete_goal": realized["status"] == "realized",
                    "target_matches_within_complete_goal": exact,
                    "primitive_teacher_actions_used": realized["primitive_teacher_actions_used"],
                }
            )
        subgoals_audited += len(goal.subgoals)
        route_rows.append(
            {
                "program_id": teacher["program_id"],
                "teacher_receipt_sha256": teacher["receipt_sha256"],
                "goal_id": goal.goal_id,
                "dependency_regions": len(goal.subgoals),
                "teacher_primitive_count": len(actions),
                "realizer_status": realized["status"],
                "realized_primitive_count": len(realized["actions"]),
                "endpoint_matches_bound_target": realized["endpoint_matches_bound_target"],
                "expanded": realized["expanded"],
                "attempted": realized["attempted"],
                "best_mismatch": realized["best_mismatch"],
                "rejections": realized.get("rejections", {}),
                "realized_actions": realized["actions"],
                "realized_states": realized["states"],
                "primitive_teacher_actions_used": realized["primitive_teacher_actions_used"],
                "output_role_slots": realized.get("output_role_slots", []),
                "output_roles": sum(len(subgoal.output_atoms) for subgoal in goal.subgoals),
                "logical_role_automorphisms": realized.get("logical_role_automorphisms", 0),
                "subgoal_targets_match_within_complete_goal": subgoal_matches,
                "bound_target_receipt": realized["bound_target_receipt"],
                "elapsed_seconds": route_elapsed_seconds,
            }
        )
        work_completed = route_index + 1
        emit(
            force=True,
            work_completed=work_completed,
            routes_completed=route_index + 1,
            subgoals_audited=subgoals_audited,
            program_id=teacher["program_id"],
            phase="route_complete",
        )

    elapsed = time.monotonic() - started
    route_realized_count = route_status_counts["realized"]
    route_exact_count = sum(row["endpoint_matches_bound_target"] for row in route_rows)
    subgoal_realized_count = subgoal_status_counts["realized"]
    subgoal_exact_count = sum(row["target_matches_within_complete_goal"] for row in subgoal_rows)
    body = {
        "schema_version": RESULT_SCHEMA,
        "evidence": "answer-known zero-oracle conditional-realizer audit",
        "source_group": source_group,
        "implementation": {"revision": code_revision, "working_tree_dirty": False},
        "inputs": {
            "teacher_library": {
                "path": str(LIBRARY.relative_to(ROOT)),
                "sha256": sha256_file(LIBRARY),
            },
            "teacher_receipts": {
                row["program_id"]: row["teacher_receipt_sha256"] for row in route_rows
            },
        },
        "configuration": {
            "maximum_primitives": config.maximum_primitives,
            "maximum_active_atoms": config.maximum_active_atoms,
            "maximum_expansions": config.maximum_expansions,
            "children_per_expansion": config.children_per_expansion,
            "heartbeat_seconds": heartbeat_seconds,
        },
        "census": {
            "routes": len(route_rows),
            "subgoals": len(subgoal_rows),
            "unique_subgoal_ids": len({row["subgoal_id"] for row in subgoal_rows}),
            "route_status_counts": dict(sorted(route_status_counts.items())),
            "subgoal_status_counts": dict(sorted(subgoal_status_counts.items())),
            "search_expansions": total_expansions,
            "action_attempts": total_attempts,
        },
        "gates": {
            "route_realization_coverage": {
                "covered": route_realized_count,
                "denominator": len(route_rows),
                "coverage": route_realized_count / len(route_rows),
            },
            "route_exact_endpoint_precision": {
                "exact": route_exact_count,
                "realized": route_realized_count,
                "precision": route_exact_count / max(1, route_realized_count),
            },
            "subgoal_target_coverage_within_complete_goals": {
                "covered": subgoal_exact_count,
                "denominator": len(subgoal_rows),
                "coverage": subgoal_exact_count / len(subgoal_rows),
            },
            "subgoal_target_precision_within_realized_complete_goals": {
                "exact": subgoal_exact_count,
                "containing_route_realized": subgoal_realized_count,
                "precision": subgoal_exact_count / max(1, subgoal_realized_count),
            },
            "teacher_action_fallback": {
                "used": sum(row["primitive_teacher_actions_used"] for row in route_rows),
                "route_denominator": len(route_rows),
                "subgoal_denominator": len(subgoal_rows),
            },
        },
        "subgoals": subgoal_rows,
        "routes": route_rows,
        "timing": {
            "elapsed_seconds": elapsed,
            "expansions_per_second": total_expansions / max(elapsed, 1e-12),
            "route_seconds": [row["elapsed_seconds"] for row in route_rows],
        },
        "compiler_cost": {
            metric: {
                "median": _percentile(values, 0.5),
                "p95": _percentile(values, 0.95),
                "maximum": max(values, default=0),
            }
            for metric, values in {
                "route_seconds": [row["elapsed_seconds"] for row in route_rows],
                "expansions": [row["expanded"] for row in route_rows],
                "action_attempts": [row["attempted"] for row in route_rows],
                "realized_primitives": [row["realized_primitive_count"] for row in route_rows],
            }.items()
        },
        "costs": {"oracle_calls": 0, "docking_calls": 0, "gpu_seconds": 0},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "The structural targets and audited bindings are answer-known training data.",
            "Subgoal targets are checked inside their coordinated complete-goal endpoint; shared boundary roles make isolated-subgoal execution a different problem.",
            "The bounded compiler uses at most 12 children per expansion; this empirical gate does not prove complete search support for every representable goal.",
            "Exact realization does not establish autonomous subgoal proposal recall.",
            "No task utility or docking score was observed.",
        ],
    }
    publish_json(output, body)
    emit(
        force=True,
        work_completed=work_total,
        routes_completed=len(route_rows),
        subgoals_audited=len(subgoal_rows),
        program_id=None,
        phase="complete",
    )
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-group", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    args = parser.parse_args()
    result = run_source(
        args.output,
        source_group=args.source_group,
        code_revision=args.code_revision,
        working_tree_dirty=False,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    print(json.dumps({"source_group": args.source_group, "gates": result["gates"]}))


if __name__ == "__main__":
    main()
