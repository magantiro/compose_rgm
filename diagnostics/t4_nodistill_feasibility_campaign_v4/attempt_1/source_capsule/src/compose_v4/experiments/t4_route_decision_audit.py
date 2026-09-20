"""Where the generic proposal law loses the winners' decisions.

The 77 exact teacher routes are executable and the representation covers all of them,
so the question is not whether Dynamic *can* make the winning moves but what
probability it assigns to them. This measures that under the actual production
machinery -- `compile_generic_module`, the same call the search makes -- by
teacher-forcing the correct current state and sampling proposals there.

The measurement is factorized the way the decision is:

* **family** -- v0 draws a uniform permutation over its thirteen generic families and
  takes the first that compiles, so if every family compiles the family choice is
  uniform and the teacher's family carries 1/13 of the mass;
* **WHERE, binding and parameters** -- sampled inside `compile_generic_module`, so
  the residual is estimated by how often forcing the correct family still reproduces
  the teacher's next primitives.

A proposal counts as reproducing the continuation when its primitive sequence is a
prefix of the teacher's remaining actions. That is deliberately strict: a module that
reaches the same place by different primitives is not the teacher's decision, and
scoring it as one would overstate the law's coverage.

Read-only and zero-oracle. Teacher states are diagnostic probes; nothing here is a
runtime input.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

from compose_v4.control.dynamic_program_synthesis import GENERIC_MODULES, compile_generic_module
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA_VERSION = "t4_route_decision_audit_v1"


def action_key(action: dict) -> str:
    """Identity of one primitive, ignoring schema envelope fields."""
    return json.dumps(
        {"rule": action["executor_rule"], "payload": action["payload"]}, sort_keys=True
    )


def load_route(path: Path) -> dict:
    """The exact witness path from a sealed teacher receipt."""
    payload = json.loads(gzip.decompress(Path(path).read_bytes()))["payload"]["path"]
    if len(payload["states"]) != len(payload["actions"]) + 1:
        raise ValueError(f"teacher receipt states and actions disagree: {path}")
    return payload


def decision_points(route: dict, *, limit: int) -> list[int]:
    """Which teacher states to probe, spread across the route.

    Probing every state is the ideal and costs about 120ms per sampled proposal, so
    the points are spread evenly and the realized count is always reported rather
    than presented as full coverage.
    """
    total = len(route["actions"])
    if total <= limit:
        return list(range(total))
    step = total / limit
    return sorted({int(i * step) for i in range(limit)})


def probe_state(source, teacher_suffix, rng, *, samples: int, families=GENERIC_MODULES) -> dict:
    """Sample every family at one teacher state and record what reproduces it."""
    compiled, matched, refused = Counter(), Counter(), Counter()
    match_lengths = []
    for family in families:
        for _ in range(samples):
            try:
                _, stage = compile_generic_module(source, rng, family)
            except ValueError as error:
                refused[f"{family}:{error}"] += 1
                continue
            compiled[family] += 1
            produced = [action_key(a) for a in stage["actions"]]
            if produced and produced == teacher_suffix[: len(produced)]:
                matched[family] += 1
                match_lengths.append(len(produced))
    families_compiling = sum(1 for family in families if compiled[family])
    return {
        "samples_per_family": samples,
        "families_offered": len(families),
        "families_compiling": families_compiling,
        "compiled": dict(sorted(compiled.items())),
        "matched": dict(sorted(matched.items())),
        "matched_total": sum(matched.values()),
        "match_primitive_lengths": sorted(match_lengths),
        "family_choice_probability": 1.0 / families_compiling if families_compiling else None,
        "within_family_match_rate": {
            family: matched[family] / compiled[family]
            for family in sorted(matched)
            if compiled[family]
        },
        "refusals": dict(refused.most_common(5)),
    }


def audit_route(receipt: Path, *, rng, samples: int, points: int) -> dict:
    """One route: probe a spread of its states and report where the decision is lost."""
    route = load_route(receipt)
    teacher = [action_key(a) for a in route["actions"]]
    began, rows = perf_counter(), []
    for index in decision_points(route, limit=points):
        result = probe_state(
            decode_state(route["states"][index]), teacher[index:], rng, samples=samples
        )
        rows.append(
            {
                "state_index": index,
                "teacher_rule": route["actions"][index]["executor_rule"],
                **result,
            }
        )
    return {
        "receipt": str(receipt),
        "primitives": len(route["actions"]),
        "decision_points": rows,
        "points_probed": len(rows),
        "any_point_reproduced": any(row["matched_total"] for row in rows),
        "elapsed_seconds": perf_counter() - began,
        "new_oracle_calls": 0,
    }


def aggregate(routes) -> dict:
    """Where the mass goes, pooled over routes."""
    points = [row for route in routes for row in route["decision_points"]]
    if not points:
        return {"points": 0}
    compiled = sum(sum(row["compiled"].values()) for row in points)
    matched = sum(row["matched_total"] for row in points)
    by_family = Counter()
    for row in points:
        by_family.update(row["matched"])
    always_all = sum(1 for row in points if row["families_compiling"] == row["families_offered"])
    return {
        "routes": len(routes),
        "points": len(points),
        "proposals_compiled": compiled,
        "proposals_reproducing_the_teacher": matched,
        "reproduction_rate_per_proposal": matched / compiled if compiled else 0.0,
        "points_with_any_reproduction": sum(1 for row in points if row["matched_total"]),
        "points_where_every_family_compiled": always_all,
        "family_availability": (
            "every generic family compiled at every probed state"
            if always_all == len(points)
            else f"{always_all} of {len(points)} states had all families available"
        ),
        "reproductions_by_family": dict(by_family.most_common()),
        "routes_with_any_reproduction": sum(1 for r in routes if r["any_point_reproduced"]),
        "interpretation": (
            "family availability and within-family parameter choice are separate losses; "
            "a family that always compiles contributes 1/13 of the proposal mass, so a "
            "low reproduction rate at that family is a WHERE/binding/parameter loss"
        ),
        "new_oracle_calls": 0,
    }
