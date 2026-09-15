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

RESULT_SCHEMA = "t4_structural_subgoal_realizer_source_audit_v1"
PROGRESS_SCHEMA = "t4_structural_subgoal_realizer_progress_v1"


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
    config = RealizerConfig() if config is None else config
    last_heartbeat = float("-inf")
    total_expansions = total_attempts = 0
    status_counts: Counter[str] = Counter()

    def emit(
        *,
        force: bool,
        completed: int,
        program_id: str | None,
        current: dict | None = None,
        phase: str = "conditional_realization",
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
            remaining_routes = max(0, len(selected) - completed - 1)
        else:
            remaining_limit_work = 0
            remaining_routes = max(0, len(selected) - completed)
        remaining_limit_work += remaining_routes * config.maximum_expansions
        expansions_per_second = expanded / elapsed if elapsed > 0 else 0.0
        eta_limit = (
            remaining_limit_work / expansions_per_second if expansions_per_second > 0 else None
        )
        completed_rate = completed / elapsed if elapsed > 0 else 0.0
        eta_empirical = (len(selected) - completed) / completed_rate if completed_rate > 0 else None
        payload = {
            "schema_version": PROGRESS_SCHEMA,
            "event": "structural_subgoal_realizer_heartbeat",
            "source_group": source_group,
            "phase": phase,
            "current_program_id": program_id,
            "routes_completed": completed,
            "routes_total": len(selected),
            "elapsed_seconds": elapsed,
            "completed_routes_per_second": completed_rate,
            "search_expansions": expanded,
            "action_attempts": attempted,
            "expansions_per_second": expansions_per_second,
            "current_route": current,
            "status_counts": dict(sorted(status_counts.items())),
            "estimated_remaining_seconds_from_completed_routes": eta_empirical,
            "estimated_remaining_seconds_to_configured_search_limit": eta_limit,
            "eta_is_operational_not_a_stopping_rule": True,
            "at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if progress_publisher is not None:
            progress_publisher(payload)
        else:
            print(json.dumps(payload, sort_keys=True), flush=True)
        last_heartbeat = now

    emit(force=True, completed=0, program_id=None, phase="starting")
    route_rows = []
    for route_index, teacher in enumerate(selected):
        trace = teacher["trace"]
        states, actions = tuple(trace["states"]), tuple(trace["actions"])
        source = decode_state(states[0])
        goal, teacher_bindings, _ = extract_structural_goal(states, actions)

        def route_progress(
            current: dict,
            *,
            completed_routes: int = route_index,
            program_id: str = teacher["program_id"],
        ) -> None:
            emit(
                force=False,
                completed=completed_routes,
                program_id=program_id,
                current=current,
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
        status_counts[realized["status"]] += 1
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
                "bound_target_receipt": realized["bound_target_receipt"],
            }
        )
        emit(
            force=True,
            completed=route_index + 1,
            program_id=teacher["program_id"],
            phase="route_complete",
        )

    elapsed = time.monotonic() - started
    realized_count = status_counts["realized"]
    exact_count = sum(row["endpoint_matches_bound_target"] for row in route_rows)
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
            "status_counts": dict(sorted(status_counts.items())),
            "search_expansions": total_expansions,
            "action_attempts": total_attempts,
        },
        "gates": {
            "realization_coverage": {
                "covered": realized_count,
                "denominator": len(route_rows),
                "coverage": realized_count / len(route_rows),
            },
            "exact_endpoint_precision": {
                "exact": exact_count,
                "realized": realized_count,
                "precision": exact_count / max(1, realized_count),
            },
            "teacher_action_fallback": {
                "used": sum(row["primitive_teacher_actions_used"] for row in route_rows),
                "denominator": len(route_rows),
            },
        },
        "routes": route_rows,
        "timing": {
            "elapsed_seconds": elapsed,
            "expansions_per_second": total_expansions / max(elapsed, 1e-12),
        },
        "costs": {"oracle_calls": 0, "docking_calls": 0, "gpu_seconds": 0},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "The structural targets and audited bindings are answer-known training data.",
            "Exact realization does not establish autonomous subgoal proposal recall.",
            "No task utility or docking score was observed.",
        ],
    }
    publish_json(output, body)
    emit(force=True, completed=len(selected), program_id=None, phase="complete")
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
