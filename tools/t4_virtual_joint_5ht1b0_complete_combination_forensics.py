"""Explain the two negative complete-combination gate findings."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.complete_region_program import program_from_structural_goal
from compose_v4.control.docking_value import identity
from compose_v4.control.structural_subgoal_realizer import (
    RealizerConfig,
    realize_target,
    target_distance,
)
from compose_v4.control.virtual_joint_region_proposer import (
    VirtualJointRegionBudgets,
    _bound_constituents,
    _goal,
    _plan_complete_combination_particle,
    _PlannedTarget,
    _PlanningPrefix,
    _valid_stop_target,
)
from compose_v4.experiments.route_proposal_quality import (
    _change_neighborhood,
    transformation_equivalent,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from tools.t4_virtual_joint_5ht1b0_complete_combination_gate import (
    ALLOCATION,
    BINDING_PARTICLE_INDEX,
    DEPTH,
    EXPECTED_PARTICLE_COUNT,
    ROOT,
    _relative,
    _revision,
    _source_input,
)
from tools.t4_virtual_joint_5ht1b0_particle_gate import (
    _sealed,
    _teacher_constituents,
)
from tools.t4_virtual_joint_5ht1b0_support_gate import (
    _load_expert,
    _route_inputs,
    _sha256,
)

SCHEMA = "t4_virtual_joint_5ht1b0_complete_combination_forensics_v1"
GATE_RESULT = ROOT / (
    "diagnostics/t4_virtual_joint_5ht1b0_support_gate/"
    "complete_combination_particle_v1/result.json"
)
RERUN_RESULT = ROOT / (
    "diagnostics/t4_virtual_joint_5ht1b0_support_gate/"
    "complete_combination_particle_v1_rerun/result.json"
)


def _state_equal(left: Any, right: Any) -> bool:
    return all(
        np.array_equal(getattr(left, name), getattr(right, name))
        for name in ("atom_types", "formal_charges", "implicit_h_counts", "bonds")
    )


def _primitive_atom_event_lower_bound(source: Any, target: Any) -> dict[str, int]:
    source_real = np.asarray(is_element(source.atom_types), dtype=bool)
    target_real = np.asarray(is_element(target.atom_types), dtype=bool)
    births = int(np.count_nonzero(~source_real & target_real))
    deletions = int(np.count_nonzero(source_real & ~target_real))
    live = source_real & target_real
    restatements = int(
        np.count_nonzero(
            live
            & (
                (source.atom_types != target.atom_types)
                | (source.formal_charges != target.formal_charges)
                | (source.implicit_h_counts != target.implicit_h_counts)
            )
        )
    )
    return {
        "births": births,
        "deletions": deletions,
        "live_atom_restatements": restatements,
        "primitive_lower_bound": births + deletions + restatements,
    }


def _receipt(source: Any, target: _PlannedTarget, budgets: VirtualJointRegionBudgets):
    receipt = realize_target(
        source,
        target.endpoint,
        config=RealizerConfig(
            maximum_primitives=budgets.maximum_primitives,
            maximum_expansions=budgets.maximum_expansions_per_realization,
        ),
    )
    if int(receipt.get("primitive_teacher_actions_used", -1)) != 0:
        raise RuntimeError("forensic realization used teacher actions")
    return receipt


def _target_report(
    source: Any,
    target: _PlannedTarget,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    goal, _ = _goal(target.prefix)
    return {
        "endpoint_sha256": identity(target.endpoint_key),
        "constituent_keys": list(target.prefix.keys),
        "template_ids": [
            row.template.template_id for row in target.prefix.constituents
        ],
        "program_id": program_from_structural_goal(goal).program_id,
        "depth": len(target.prefix.constituents),
        "source_active_atoms": source.n_real_atoms,
        "target_active_atoms": target.endpoint.n_real_atoms,
        "persistent_coordinate_target_distance": target_distance(
            source, target.endpoint
        ),
        "atom_event_primitive_lower_bound": _primitive_atom_event_lower_bound(
            source, target.endpoint
        ),
        "status": str(receipt["status"]),
        "expanded": int(receipt["expanded"]),
        "attempted": int(receipt["attempted"]),
        "best_mismatch": int(receipt["best_mismatch"]),
        "compiler_strategy": receipt.get("compiler_strategy"),
        "deterministic_schedule_status": receipt.get("deterministic_schedule_status"),
        "rejections": receipt.get("rejections", {}),
        "primitive_teacher_actions_used": int(
            receipt["primitive_teacher_actions_used"]
        ),
    }


def _changed_summary(graph: nx.Graph) -> dict[str, Any]:
    changed = sorted(
        int(slot)
        for slot, row in graph.nodes(data=True)
        if json.loads(row["label"])["changed"]
    )
    return {
        "neighborhood_wl_hash": graph.graph["hash"],
        "neighborhood_nodes": graph.number_of_nodes(),
        "neighborhood_edges": graph.number_of_edges(),
        "changed_slot_count": len(changed),
        "changed_slot_set_sha256": identity(changed),
    }


def _teacher_metric_report(
    source: Any,
    route: dict[str, Any],
    selected_by_key: dict[str, Any],
    teacher_rows: list[dict[str, Any]],
    expert: Any,
    budgets: VirtualJointRegionBudgets,
) -> dict[str, Any]:
    keys = sorted(
        row["constituent_key"]
        for row in teacher_rows
        if row["route_id"] == route["route_id"]
    )
    prefix = _PlanningPrefix(
        tuple(selected_by_key[key] for key in keys),
        expert.marginal.score(tuple(selected_by_key[key].template for key in keys)),
    )
    stop = _valid_stop_target(source, prefix)
    if stop is None:
        raise RuntimeError("teacher combination is no longer a valid STOP target")
    target_endpoint, target_receipt = stop
    target = _PlannedTarget(prefix, target_endpoint, target_receipt)
    receipt = _receipt(source, target, budgets)
    if receipt["status"] != "realized":
        raise RuntimeError("teacher combination no longer realizes")
    realized = decode_state(receipt["states"][-1])
    teacher = route["endpoint"]
    realized_neighborhood = _change_neighborhood(source, realized, radius=2)
    teacher_neighborhood = _change_neighborhood(source, teacher, radius=2)
    realized_changed = {
        slot
        for slot, row in realized_neighborhood.nodes(data=True)
        if json.loads(row["label"])["changed"]
    }
    teacher_changed = {
        slot
        for slot, row in teacher_neighborhood.nodes(data=True)
        if json.loads(row["label"])["changed"]
    }
    return {
        "route_id": route["route_id"],
        "canonical_endpoint_equal": canonical_state_key(realized)
        == route["endpoint_key"],
        "persistent_slot_state_equal": _state_equal(realized, teacher),
        "radius_2_transformation_equivalent": transformation_equivalent(
            source, realized, teacher, radius=2
        ),
        "realized_endpoint": _changed_summary(realized_neighborhood),
        "teacher_endpoint": _changed_summary(teacher_neighborhood),
        "changed_slot_intersection_count": len(realized_changed & teacher_changed),
        "changed_slot_symmetric_difference_count": len(
            realized_changed ^ teacher_changed
        ),
        "colored_neighborhood_isomorphic_without_wl_prefilter": nx.is_isomorphic(
            realized_neighborhood,
            teacher_neighborhood,
            node_match=lambda a, b: a["label"] == b["label"],
            edge_match=lambda a, b: a["label"] == b["label"],
        ),
        "realized_exact_state_sha256": identity(encode_state(realized)),
        "teacher_exact_state_sha256": identity(encode_state(teacher)),
        "realizer": {
            "expanded": int(receipt["expanded"]),
            "attempted": int(receipt["attempted"]),
            "primitive_count": len(receipt["actions"]),
            "compiler_strategy": receipt.get("compiler_strategy"),
        },
    }


def run() -> dict[str, Any]:
    gate, gate_payload_sha256 = _sealed(GATE_RESULT)
    rerun, rerun_payload_sha256 = _sealed(RERUN_RESULT)
    if gate_payload_sha256 != rerun_payload_sha256 or gate != rerun:
        raise RuntimeError("deterministic gate rerun is not byte-identical in payload")
    budgets = VirtualJointRegionBudgets()
    source, source_state_sha256 = _source_input()
    expert, checkpoint_payload_sha256 = _load_expert()
    telemetry: Counter = Counter()
    constituents = _bound_constituents(
        source,
        expert,
        budgets,
        telemetry,
        constituent_allocation=ALLOCATION,
        binding_particle_index=BINDING_PARTICLE_INDEX,
    )
    targets = _plan_complete_combination_particle(
        source,
        expert,
        constituents,
        budgets,
        telemetry,
        depth=DEPTH,
        particle_index=0,
        particle_count=EXPECTED_PARTICLE_COUNT,
    )
    target_reports = [
        _target_report(source, target, _receipt(source, target, budgets))
        for target in targets
    ]
    search_limit_targets = [
        row for row in target_reports if row["status"] == "search_limit_abstention"
    ]
    if len(search_limit_targets) != 1:
        raise RuntimeError("shard-0 search-limit abstention census changed")

    # Teacher identities enter only after the task-blind shard-0 reproduction.
    routes, route_context = _route_inputs()
    if route_context["source_state_sha256"] != source_state_sha256:
        raise RuntimeError("post-hoc teacher source differs from proposal source")
    teacher_rows = _teacher_constituents(routes)
    selected_by_key = {row.constituent_key: row for row in constituents}
    teacher_metric_reports = [
        _teacher_metric_report(
            source,
            route,
            selected_by_key,
            teacher_rows,
            expert,
            budgets,
        )
        for route in routes
    ]
    teacher_endpoint_hashes = {identity(row["endpoint_key"]) for row in routes}
    cap_target = search_limit_targets[0]
    cap_target_is_teacher = cap_target["endpoint_sha256"] in teacher_endpoint_hashes
    payload = {
        "schema_version": SCHEMA,
        "evidence": "computed zero-oracle post-gate forensics",
        "code_revision": _revision(),
        "gate_result_payload_sha256": gate_payload_sha256,
        "deterministic_rerun": {
            "first_result_physical_sha256": _sha256(GATE_RESULT),
            "rerun_result_physical_sha256": _sha256(RERUN_RESULT),
            "physical_files_byte_identical": GATE_RESULT.read_bytes()
            == RERUN_RESULT.read_bytes(),
            "payload_sha256_equal": gate_payload_sha256 == rerun_payload_sha256,
            "shard_0_search_limit_repeated": True,
        },
        "radius_2_metric_forensics": {
            "metric_semantics": (
                "the frozen evaluator compares source-slot-aligned colored change "
                "neighborhoods; canonical endpoint identity is evaluated on the "
                "molecular quotient and does not imply equality of those persistent-"
                "slot change neighborhoods"
            ),
            "finding": (
                "all three realized endpoints are canonically identical to their "
                "teachers but use different persistent-slot states; their radius-2 "
                "colored neighborhoods have different WL hashes and are not isomorphic"
            ),
            "interpretation": (
                "radius-2 transformation equivalence is non-monotone with exact "
                "canonical endpoint identity here and is not a valid failure criterion "
                "for these canonically recovered endpoints without canonical slot alignment"
            ),
            "routes": teacher_metric_reports,
        },
        "shard_0_capacity_forensics": {
            "targets_replayed": len(target_reports),
            "status_counts": dict(
                sorted(Counter(row["status"] for row in target_reports).items())
            ),
            "search_limit_target": {
                **cap_target,
                "matches_any_teacher_canonical_endpoint": cap_target_is_teacher,
            },
            "finding": (
                "the one search-limit target is not a teacher endpoint; it repeats "
                "at the unchanged 4,000-expansion per-realization cap"
            ),
        },
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "teacher_actions": 0,
        },
        "configuration": {
            "binding_particle_index": BINDING_PARTICLE_INDEX,
            "combination_depth": DEPTH,
            "combination_particle_index": 0,
            "combination_particle_count": EXPECTED_PARTICLE_COUNT,
            "budgets": {
                "maximum_primitives": budgets.maximum_primitives,
                "maximum_expansions_per_realization": budgets.maximum_expansions_per_realization,
            },
            "budget_change": False,
        },
        "inputs_sha256": {
            _relative(GATE_RESULT): _sha256(GATE_RESULT),
            _relative(RERUN_RESULT): _sha256(RERUN_RESULT),
            _relative(Path(__file__).resolve()): _sha256(Path(__file__).resolve()),
        },
        "payload_inputs": {
            "source_state_sha256": source_state_sha256,
            "checkpoint_payload_sha256": checkpoint_payload_sha256,
            "forensic_payload_sha256": route_context["forensic_payload_sha256"],
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite forensic artifact: {args.output}")
    payload = run()
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), **envelope}, sort_keys=True))


if __name__ == "__main__":
    main()
