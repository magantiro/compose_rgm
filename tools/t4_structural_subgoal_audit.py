#!/usr/bin/env python3
"""Source-sharded zero-oracle audit of T4 structural subgoals."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from compose_v4.control.structural_subgoal import (
    attachment_bindings,
    extract_structural_goal,
    instantiate_goal,
)
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state
from tools.t4_route_distillation import _teacher_traces

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = (
    ROOT / "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json"
)
RESULT_SCHEMA = "t4_structural_subgoal_source_audit_v1"


def teacher_traces() -> list[dict]:
    traces, exclusions = _teacher_traces(json.loads(LIBRARY.read_text()))
    if exclusions or len(traces) != 77:
        raise ValueError(f"T4 teacher census changed: {len(traces)}, {exclusions}")
    return traces


def source_groups() -> tuple[str, ...]:
    groups = tuple(sorted({row["source_group"] for row in teacher_traces()}))
    if len(groups) != 15:
        raise ValueError(f"expected 15 T4 source groups, found {len(groups)}")
    return groups


def run_source(
    output: Path,
    *,
    source_group: str,
    code_revision: str,
    working_tree_dirty: bool,
) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite structural-subgoal shard: {output}")
    if working_tree_dirty:
        raise ValueError("structural-subgoal audit requires clean committed source")
    selected = [row for row in teacher_traces() if row["source_group"] == source_group]
    if not selected:
        raise ValueError(f"unknown or empty T4 source group: {source_group}")

    route_rows = []
    exact_endpoints = 0
    binding_routes = 0
    binding_subgoals = 0
    subgoal_count = Counter()
    for teacher in selected:
        trace = teacher["trace"]
        states, actions = tuple(trace["states"]), tuple(trace["actions"])
        source, target = decode_state(states[0]), decode_state(states[-1])
        goal, teacher_bindings, _regions = extract_structural_goal(states, actions)
        product, receipt = instantiate_goal(source, goal, teacher_bindings)
        exact = canonical_state_key(product) == canonical_state_key(target)
        exact_endpoints += int(exact)
        subgoal_count[len(goal.subgoals)] += 1

        binding_rows = []
        route_binding_ok = True
        for subgoal, teacher_binding in zip(
            goal.subgoals, teacher_bindings, strict=True
        ):
            census = attachment_bindings(subgoal, source)
            contains = teacher_binding in census.assignments
            binding_subgoals += int(contains)
            route_binding_ok = route_binding_ok and contains
            binding_rows.append(
                {
                    "subgoal_id": subgoal.subgoal_id,
                    "candidate_bindings": len(census.assignments),
                    "search_visits": census.visits,
                    "truncated": census.truncated,
                    "teacher_binding_in_enumeration": contains,
                }
            )
        binding_routes += int(route_binding_ok)
        route_rows.append(
            {
                "program_id": teacher["program_id"],
                "teacher_receipt_sha256": teacher["receipt_sha256"],
                "primitive_count": len(actions),
                "dependency_regions": len(goal.subgoals),
                "goal_id": goal.goal_id,
                "goal": goal.payload(),
                "teacher_binding": [list(row) for row in teacher_bindings],
                "exact_bound_target_reconstruction": exact,
                "binding_enumeration": binding_rows,
                "bound_target_receipt": receipt,
                "primitive_teacher_actions_in_goal": 0,
            }
        )

    body = {
        "schema_version": RESULT_SCHEMA,
        "evidence": "answer-known zero-oracle structural-subgoal training-data audit",
        "source_group": source_group,
        "implementation": {
            "revision": code_revision,
            "working_tree_dirty": False,
        },
        "inputs": {
            "teacher_library": {
                "path": str(LIBRARY.relative_to(ROOT)),
                "sha256": sha256_file(LIBRARY),
            },
            "teacher_receipts": {
                row["program_id"]: row["teacher_receipt_sha256"] for row in route_rows
            },
        },
        "census": {
            "routes": len(route_rows),
            "subgoals": sum(len(row["goal"]["subgoals"]) for row in route_rows),
            "subgoal_count_distribution": dict(sorted(subgoal_count.items())),
        },
        "gates": {
            "exact_endpoint_reconstruction": {
                "covered": exact_endpoints,
                "denominator": len(route_rows),
                "precision": exact_endpoints / len(route_rows),
            },
            "teacher_binding_recovered_per_route": {
                "covered": binding_routes,
                "denominator": len(route_rows),
                "coverage": binding_routes / len(route_rows),
            },
            "teacher_binding_recovered_per_subgoal": {
                "covered": binding_subgoals,
                "denominator": sum(len(row["goal"]["subgoals"]) for row in route_rows),
                "coverage": binding_subgoals
                / sum(len(row["goal"]["subgoals"]) for row in route_rows),
            },
        },
        "routes": route_rows,
        "costs": {"oracle_calls": 0, "docking_calls": 0, "gpu_seconds": 0},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Teacher bindings are used only to audit target reconstruction.",
            "Binding coverage does not establish autonomous subgoal generation.",
            "The goal payload is training-only and is not a runtime checkpoint.",
        ],
    }
    publish_json(output, body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-group", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    result = run_source(
        args.output,
        source_group=args.source_group,
        code_revision=args.code_revision,
        working_tree_dirty=False,
    )
    print(json.dumps({"source_group": args.source_group, "gates": result["gates"]}))


if __name__ == "__main__":
    main()
