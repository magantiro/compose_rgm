#!/usr/bin/env python3
"""Assemble the zero-oracle PMO capability matrix and rank the columns to fix.

Inputs (all produced without an oracle call):
  * the teacher-side Z decomposition (``pmo_teacher_capability_matrix_v1``)
  * live proposal pools from the teacher roots (``pmo_teacher_route_gap_phase_b``)
  * the jump-lane realization audit (``pmo_teacher_capability_jump_lane_v1``)

Both sides are characterized by the SAME functions, so the comparison is
like-for-like.  Proposals are re-characterized here from the campaign pools rather
than read from a summary, so the proposer row carries the full Z (region, topology,
attachment, dependencies) and not only scale.

STAGE ATTRIBUTION.  A stage is judged failed when the proposer's BEST proposal on
that axis, over the whole pool, does not reach the LEAST demanding teacher route on
the same axis.  That is the most generous reading available for the proposer, so the
test can only under-report failures.  The first failing stage in pipeline order is
the attributed column; the full pass/fail vector is reported beside it so columns can
be fixed globally rather than task by task.

Teachers are counterfactuals only: no teacher endpoint, route or template enters any
policy here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pmo_binder_repair_v1 import static_first_step_feasible
from pmo_teacher_capability_jump_lane_v1 import load_plans
from pmo_teacher_capability_matrix_v1 import aggregate, describe_route
from pmo_teacher_route_gap_v1 import MAX_PRIMITIVES, load_routes, route_objective

from compose_v4.chem.molecular_graph import is_element
from compose_v4.rewrite.trace_shard import decode_state

STAGES = (
    "region support",
    "edit scale support",
    "structural mode support",
    "attachment alpha",
    "dependencies",
    "exact realization",
    "selection",
    "recursive exploitation",
)


def harvest_pool(campaign: Path) -> list[dict]:
    """Every candidate the proposer emitted, re-characterized with the Z functions."""
    rows, seen = [], set()
    for pending in sorted(campaign.glob("round_*/pending.json")):
        batch = json.loads(pending.read_text())["batch"]
        for pool in ("proposal_pool", "eligible_pool"):
            for candidate in batch.get(pool, {}).get("candidates", []):
                if candidate["candidate_id"] in seen:
                    continue
                seen.add(candidate["candidate_id"])
                trace = candidate.get("trace") or {}
                states, actions = trace.get("states"), trace.get("actions")
                if not states or not actions or len(states) != len(actions) + 1:
                    continue
                route = {
                    "source_state": candidate["source_state"],
                    "states": states,
                    "actions": actions,
                    "terminal_endpoint": candidate["endpoint"],
                    "trace_identity": candidate["candidate_id"],
                }
                try:
                    described = describe_route(route)
                except (ValueError, KeyError, IndexError, TypeError):
                    continue
                described["planner_channel"] = candidate["provenance"].get("planner_channel")
                described["candidate_id"] = candidate["candidate_id"]
                rows.append(described)
    return rows


def harvest_attempts(campaign: Path) -> dict:
    """Per-channel attempt accounting, straight from the campaign's own records."""
    attempts: dict[str, Counter] = defaultdict(Counter)
    reasons: dict[str, Counter] = defaultdict(Counter)
    for pending in sorted(campaign.glob("round_*/pending.json")):
        batch = json.loads(pending.read_text())["batch"]
        for attempt in batch.get("attempts", ()):
            channel = attempt.get("planner_channel")
            attempts[channel][attempt.get("status")] += 1
            if attempt.get("status") != "eligible":
                reasons[channel][str(attempt.get("reason"))[:90]] += 1
    return {
        "attempts_by_channel": {
            channel: dict(counter) for channel, counter in sorted(attempts.items())
        },
        "rejection_reasons_by_channel": {
            channel: dict(counter.most_common(6)) for channel, counter in sorted(reasons.items())
        },
    }


def pool_identity(rows: list[dict]) -> str:
    """Identity by ENDPOINT SET, not candidate id.

    ``candidate_id`` embeds the task identity, so two task-blind pools that contain
    exactly the same molecules carry different ids.  The endpoint set is what the
    "same root -> same pool" control has to compare.
    """
    return json.dumps(sorted({row["terminal_endpoint"] for row in rows}))


def attribute(teacher: dict, proposer: dict, realization: dict) -> dict:
    """Earliest stage whose support does not reach the least demanding teacher route."""
    checks: dict[str, dict] = {}

    def check(stage: str, demand: float, supply: float, unit: str) -> None:
        checks[stage] = {
            "teacher_minimum_demand": demand,
            "proposer_maximum_supply": supply,
            "unit": unit,
            "supported": supply >= demand,
        }

    check(
        "region support",
        teacher["released_atoms"]["min"],
        proposer["released_atoms"]["max"],
        "source heavy atoms released",
    )
    check(
        "edit scale support",
        teacher["primitive_count"]["min"],
        proposer["primitive_count"]["max"],
        "primitives in one program",
    )
    # Structural mode has two parts: the realized direction and the move vocabulary.
    teacher_directions = set(teacher["direction"])
    proposer_directions = set(proposer["direction"])
    mode_supported = bool(teacher_directions & proposer_directions)
    missing_moves = sorted(set(teacher["move_class_support"]) - set(proposer["move_class_support"]))
    checks["structural mode support"] = {
        "teacher_directions": sorted(teacher_directions),
        "proposer_directions": sorted(proposer_directions),
        "direction_overlap": sorted(teacher_directions & proposer_directions),
        "teacher_minimum_demand": teacher["distinct_move_classes"]["min"],
        "proposer_maximum_supply": proposer["distinct_move_classes"]["max"],
        "unit": "distinct move classes in one program",
        "move_classes_absent_from_proposals": missing_moves,
        "supported": mode_supported
        and not missing_moves
        and proposer["distinct_move_classes"]["max"] >= teacher["distinct_move_classes"]["min"],
    }
    check(
        "attachment alpha",
        teacher["attachment_bonds"]["min"],
        proposer["attachment_bonds"]["max"],
        "bonds joining created topology to the retained core",
    )
    check(
        "dependencies",
        teacher["dependency_chain_depth"]["min"],
        proposer["dependency_chain_depth"]["max"],
        "longest created-handle producer->consumer chain",
    )
    checks["exact realization"] = {
        **realization,
        "supported": realization.get("jump_eligible", 0) > 0,
    }
    checks["selection"] = {
        "testable": False,
        "reason": (
            "Phase B substitutes a deterministic hash for the objective, so selection "
            "pressure is not measured here; and the transformation is lost upstream on "
            "every task in this matrix, so selection is never the binding stage."
        ),
        "supported": None,
    }
    checks["recursive exploitation"] = {
        "testable": False,
        "reason": "not reached: no teacher-scale transformation survives to be exploited",
        "supported": None,
    }
    first = next(
        (stage for stage in STAGES if checks[stage].get("supported") is False),
        None,
    )
    return {"attributed_stage": first, "stages": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--work", required=True, help="phase-B work directory")
    parser.add_argument(
        "--jump",
        default="",
        help="optional jump-lane depth-ladder artifact; the per-root step-0 census "
        "below is computed here and does not depend on it",
    )
    parser.add_argument("--phase-b", required=True)
    parser.add_argument("--out", required=True)
    options = parser.parse_args()

    teacher_side = json.loads(Path(options.teacher).read_text())
    jump = (
        json.loads(Path(options.jump).read_text())
        if options.jump and Path(options.jump).exists()
        else {"roots": {}}
    )
    phase_b = json.loads(Path(options.phase_b).read_text())

    corpus = load_routes()
    root_ids: dict[str, int] = {}
    root_payload: dict[int, dict] = {}
    task_roots: dict[str, list[int]] = defaultdict(list)
    for route in corpus:
        key = json.dumps(route["source_state"], sort_keys=True)
        if key not in root_ids:
            root_ids[key] = len(root_ids)
            root_payload[root_ids[key]] = route["source_state"]
    for route in corpus:
        key = json.dumps(route["source_state"], sort_keys=True)
        objective = route_objective(route)
        if root_ids[key] not in task_roots[objective]:
            task_roots[objective].append(root_ids[key])

    # Proposal pools, keyed by the ROOT SET they were driven from.  The proposer is
    # task blind and the initialization is built from the roots, so two tasks with the
    # same root set share a pool; the gsk3b/qed pair is the measured control.
    pools: dict[str, dict] = {}
    work = Path(options.work)
    for folder in sorted(work.iterdir()):
        campaign = folder / "campaign"
        if not campaign.is_dir():
            continue
        rows = harvest_pool(campaign)
        pools[folder.name] = {
            "rows": rows,
            "identity": pool_identity(rows),
            "roots": tuple(sorted(task_roots[folder.name])),
            "accounting": harvest_attempts(campaign),
        }

    by_rootset: dict[tuple[int, ...], str] = {}
    identity_control = {}
    for name, pool in pools.items():
        key = pool["roots"]
        if key in by_rootset:
            other = by_rootset[key]
            identity_control[f"{other} vs {name}"] = {
                "same_root_set": list(key),
                "pool_identical": pool["identity"] == pools[other]["identity"],
                "candidates": [len(pools[other]["rows"]), len(pool["rows"])],
            }
        else:
            by_rootset[key] = name

    rows_out = {}
    for task, teacher in sorted(teacher_side["tasks"].items()):
        key = tuple(sorted(task_roots[task]))
        pool_name = by_rootset.get(key)
        if pool_name is None:
            rows_out[task] = {"status": "NO_PROPOSAL_POOL", "root_ids": list(key)}
            continue
        proposer_rows = pools[pool_name]["rows"]
        proposer = aggregate(proposer_rows)
        proposer["by_channel"] = dict(
            Counter(row["planner_channel"] for row in proposer_rows)
        )
        task_report = phase_b["tasks"].get(pool_name, {})
        summary = task_report.get("summary", {})
        accounting = pools[pool_name]["accounting"]
        jump_status = accounting["attempts_by_channel"].get(
            "joint_dependency_region_jump", {}
        )
        jump_counts = {
            "jump_attempts": sum(jump_status.values()),
            "jump_status": jump_status,
            "jump_rejection_reasons": accounting["rejection_reasons_by_channel"].get(
                "joint_dependency_region_jump", {}
            ),
            "jump_eligible": summary.get("by_channel", {}).get(
                "joint_dependency_region_jump", 0
            ),
            "exact_teacher_endpoint_support": task_report.get(
                "exact_teacher_endpoint_support"
            ),
        }
        root_jump = [
            jump["roots"][str(root)]["summary"] for root in key if str(root) in jump["roots"]
        ]
        if root_jump:
            jump_counts["role0_feasible_plans"] = sum(row["role0_feasible"] for row in root_jump)
            jump_counts["plans_probed"] = sum(row["plans"] for row in root_jump)
            jump_counts["role0_feasible_rate"] = (
                jump_counts["role0_feasible_plans"] / jump_counts["plans_probed"]
            )
            prefix = [row["prefix_bind"] for row in root_jump if "prefix_bind" in row]
            if prefix:
                jump_counts["prefix_roles"] = prefix[0]["roles_attempted"]
                jump_counts["plans_surviving_prefix"] = sum(
                    row["prefix_bindings"] for row in prefix
                )
        rows_out[task] = {
            "root_ids": list(key),
            "proposal_pool_from_task": pool_name,
            "teacher": teacher,
            "proposer": proposer,
            "channel_accounting": accounting,
            **attribute(teacher, proposer, jump_counts),
        }

    # Independent replication of the step-0 census, per ROOT.  The aggregate over the
    # teacher stratum is a published number; the per-root spread is not, and it is what
    # decides whether "role 0 is over-specified" is a bank property or a parent property.
    plans = load_plans()
    static_by_root = {}
    for root_id, state in sorted(root_payload.items()):
        graph = decode_state(state)
        feasible = sum(static_first_step_feasible(graph, plan) for plan in plans)
        static_by_root[str(root_id)] = {
            "plans": len(plans),
            "static_step0_feasible": feasible,
            "static_step0_feasible_rate": feasible / len(plans),
            "heavy_atoms": int(is_element(graph.atom_types).sum()),
            "slots": int(graph.n_atoms),
        }

    ranking: Counter = Counter()
    for task, row in rows_out.items():
        if row.get("attributed_stage"):
            ranking[row["attributed_stage"]] += 1
    blanket = {
        stage: sum(
            1
            for row in rows_out.values()
            if row.get("stages", {}).get(stage, {}).get("supported") is False
        )
        for stage in STAGES
    }

    payload = {
        "schema_version": "pmo_capability_matrix_v1",
        "charged_oracle_calls": 0,
        "runtime_maximum_primitives": MAX_PRIMITIVES,
        "pool_identity_control": identity_control,
        "static_step0_by_root": static_by_root,
        "stage_order": list(STAGES),
        "attributed_stage_counts": dict(ranking),
        "tasks_failing_each_stage": blanket,
        "tasks": rows_out,
    }
    Path(options.out).parent.mkdir(parents=True, exist_ok=True)
    Path(options.out).write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    print(f"wrote {options.out}")


if __name__ == "__main__":
    main()
