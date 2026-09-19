"""Run the zero-oracle 5HT1B-0 joint-region known-answer support gate."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import rdkit

from compose_v4.control.complete_region_program import (
    execute_complete_region_program,
    program_from_structural_goal,
)
from compose_v4.control.docking_value import identity
from compose_v4.control.route_distilled_goal_expert import RouteDistilledGoalExpert
from compose_v4.control.structural_subgoal import StructuralGoal, instantiate_goal
from compose_v4.control.structural_subgoal_policy import (
    REWRITE_SCALE_BANDS,
    materialize_template,
    structural_rewrite_event_count,
    transfer_bindings,
)
from compose_v4.control.structural_subgoal_realizer import (
    RealizerConfig,
    realize_target,
)
from compose_v4.control.virtual_joint_region_proposer import (
    VirtualJointRegionBudgets,
    propose_virtual_joint_region_paths,
)
from compose_v4.experiments.route_proposal_quality import transformation_equivalent
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from tools.t4_structural_subgoal_policy import _rows

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "t4_virtual_joint_5ht1b0_support_gate_v1"
LEGACY_ALLOCATION_SCHEMA = "t4_virtual_joint_5ht1b0_legacy_scale_allocation_v1"
SOURCE_GROUP = "25f6f9df141a27de61a5579047ba715883345bc784c2060a2740a80c0a1816ec"
ROUTE_IDS = (
    "2b0aff1682f13aa9701f8837835f15cf5887e6bc430e86e35bab4695a3fa79af",
    "88e8705520e5037a18a5c6636dc2fdd5593819f8b4b589081b7ebd1351fb1fc4",
    "bbd7db0d5b5a6bbced5e1e82a55596a874fcc8bfab743cc6114de2cbee31d2c0",
)
CHECKPOINT = ROOT / "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json"
SOURCE_AUDIT = ROOT / (
    "diagnostics/t4_structural_subgoal/attempt_2/sources/"
    f"{SOURCE_GROUP}/result.json"
)
FORENSICS = ROOT / "diagnostics/t4_known_good_transformation_forensics/attempt_1/result.json"
OLD_AUTONOMOUS = ROOT / "diagnostics/t4_shared_retained_rewrite_v1/shards/5ht1b_0.json"
SHARED_LIBRARY = ROOT / (
    "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json"
)
SOURCE_REGISTRY = ROOT / "docs/GENMOL_T4_SEEDS.json"
T4_CONTRACT = ROOT / "configs/t4_frozen_program_benchmark_v2.json"
ROUTE_DIRECTORY = ROOT / "diagnostics/t4_complete_region_runtime/attempt_1/routes"
RECEIPTS = {
    ROUTE_IDS[0]: ROOT
    / "diagnostics/ivg_winner_paths/pairs/03726e8db2c3c5a37359fe4f0e624940bbebf46e363806006b86979d39788dcb.json.gz",
    ROUTE_IDS[1]: ROOT
    / "diagnostics/ivg_winner_paths/pairs/2a8cee05e41cf811ad125ffce006cc7339c4573817b59e12fa37d808ebd8ba3d.json.gz",
    ROUTE_IDS[2]: ROOT
    / "diagnostics/ivg_winner_paths/pairs/8247413ed66eb9b4d340ea3c777ca4c2b684a4c88732d51192f148a54b15214e.json.gz",
}
CODE_INPUTS = (
    "src/compose_v4/control/complete_region_program.py",
    "src/compose_v4/control/route_distilled_goal_expert.py",
    "src/compose_v4/control/sequential_region_proposer.py",
    "src/compose_v4/control/structural_subgoal.py",
    "src/compose_v4/control/structural_subgoal_policy.py",
    "src/compose_v4/control/structural_subgoal_realizer.py",
    "src/compose_v4/control/virtual_joint_region_proposer.py",
    "src/compose_v4/experiments/route_proposal_quality.py",
    "tools/t4_structural_subgoal_audit.py",
    "tools/t4_structural_subgoal_policy.py",
    "tools/t4_virtual_joint_5ht1b0_support_gate.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _envelope(path: Path) -> tuple[dict, str]:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    payload_sha256 = envelope.get("payload_sha256")
    if not isinstance(payload, dict) or payload_sha256 != identity(payload):
        raise ValueError(f"artifact is not self-hashed: {path}")
    return payload, str(payload_sha256)


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _route_inputs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_audit = json.loads(SOURCE_AUDIT.read_text())
    source_routes = {row["program_id"]: row for row in source_audit["routes"]}
    forensic_payload, forensic_payload_sha256 = _envelope(FORENSICS)
    forensic_routes = {row["probe_id"]: row for row in forensic_payload["routes"]}
    teacher_rows, _ = _rows()
    selected: list[dict[str, Any]] = []
    source = None
    for route_id in ROUTE_IDS:
        audit = source_routes.get(route_id)
        forensic = forensic_routes.get(route_id)
        if audit is None or forensic is None:
            raise ValueError(f"missing authoritative route row: {route_id}")
        labels = forensic["known_labels"]
        if len(labels) != 1 or float(labels[0]["delta"]) != 0.4:
            raise ValueError(f"route is not a unique delta=0.4 5HT1B label: {route_id}")
        route_path = ROUTE_DIRECTORY / f"{route_id}.json"
        route_payload, route_payload_sha256 = _envelope(route_path)
        endpoint_key = str(route_payload["execution"]["committed_endpoint_key"])
        matches = [
            row
            for row in teacher_rows
            if row["source_group"] == SOURCE_GROUP
            and canonical_state_key(row["endpoint"]) == endpoint_key
        ]
        if len(matches) != 1:
            raise ValueError(f"teacher route join is not one-to-one: {route_id}")
        teacher = matches[0]
        if source is None:
            source = teacher["source"]
        elif canonical_state_key(source) != canonical_state_key(teacher["source"]):
            raise ValueError("strong routes do not share one source")
        receipt_sha256 = _sha256(RECEIPTS[route_id])
        if (
            receipt_sha256 != audit["teacher_receipt_sha256"]
            or receipt_sha256 != route_payload["teacher_receipt_sha256"]
        ):
            raise ValueError(f"teacher receipt identity changed: {route_id}")
        if (
            int(audit["primitive_count"]) != int(route_payload["teacher_primitive_count"])
            or int(audit["dependency_regions"]) != len(teacher["templates"])
            or endpoint_key != str(audit["bound_target_receipt"]["endpoint"])
        ):
            raise ValueError(f"route provenance disagrees: {route_id}")
        selected.append(
            {
                "route_id": route_id,
                "reported_external_docking_score": float(
                    labels[0]["reported_external_docking_score"]
                ),
                "reported_run_seed": int(labels[0]["run_seed"]),
                "teacher_primitive_count": int(audit["primitive_count"]),
                "teacher_receipt_sha256": receipt_sha256,
                "endpoint_key": endpoint_key,
                "endpoint": teacher["endpoint"],
                "source": teacher["source"],
                "templates": teacher["templates"],
                "bindings": teacher["bindings"],
                "runtime_artifact_path": _relative(route_path),
                "runtime_artifact_sha256": _sha256(route_path),
                "runtime_artifact_payload_sha256": route_payload_sha256,
            }
        )
    if source is None:
        raise RuntimeError("strong route set is empty")
    return selected, {
        "source": source,
        "source_state_sha256": identity(encode_state(source)),
        "forensic_payload_sha256": forensic_payload_sha256,
    }


def _load_expert() -> tuple[RouteDistilledGoalExpert, str]:
    checkpoint, payload_sha256 = _envelope(CHECKPOINT)
    if checkpoint.get("runtime_target_conditioning") is not False:
        raise ValueError("shared checkpoint is target-conditioned")
    expert = RouteDistilledGoalExpert.from_checkpoint(checkpoint["expert"])
    return expert, payload_sha256


def _teacher_forced_joint(route: dict[str, Any], budgets: VirtualJointRegionBudgets) -> dict:
    patches = tuple(
        materialize_template(template, route["source"], binding)
        for template, binding in zip(route["templates"], route["bindings"], strict=True)
    )
    goal = StructuralGoal(patches)
    reconstructed, target_receipt = instantiate_goal(
        route["source"], goal, route["bindings"]
    )
    reconstructed_key = canonical_state_key(reconstructed)
    exact_reconstruction = reconstructed_key == route["endpoint_key"]
    receipt = realize_target(
        route["source"],
        reconstructed,
        config=RealizerConfig(
            maximum_primitives=budgets.maximum_primitives,
            maximum_expansions=budgets.maximum_expansions_per_realization,
        ),
    )
    states = receipt.get("states")
    actions = receipt.get("actions")
    realized_endpoint_key = None
    if isinstance(states, list) and states and isinstance(states[-1], dict):
        realized_endpoint_key = canonical_state_key(decode_state(states[-1]))
    primitive_count = len(actions) if isinstance(actions, list) else None
    exact_execution = (
        receipt.get("status") == "realized"
        and realized_endpoint_key == route["endpoint_key"]
    )
    teacher_actions = int(receipt.get("primitive_teacher_actions_used", -1))
    if teacher_actions != 0:
        raise RuntimeError("teacher-forced joint runtime used primitive teacher actions")
    return {
        "exact_final_target_reconstruction": exact_reconstruction,
        "exact_final_endpoint_execution": exact_execution,
        "status": str(receipt.get("status")),
        "teacher_primitive_actions_used": teacher_actions,
        "realized_primitive_count": primitive_count,
        "compiler_strategy": receipt.get("compiler_strategy"),
        "expanded": int(receipt.get("expanded", 0)),
        "attempted": int(receipt.get("attempted", 0)),
        "goal_id": goal.goal_id,
        "program_id": program_from_structural_goal(goal).program_id,
        "target_receipt_sha256": identity(target_receipt),
        "realized_actions_sha256": identity(actions) if isinstance(actions, list) else None,
    }


def _constituent_key(template, binding, patch) -> str:
    return identity(
        {
            "schema_version": "virtual_joint_bound_constituent_v1",
            "template_id": template.template_id,
            "binding": list(binding),
            "patch_id": patch.subgoal_id,
        }
    )


def _template_size_signature(template) -> dict[str, Any]:
    created = len(template.output_atoms)
    deleted = sum(row is None for row in template.target_atoms)
    restated = sum(
        after is not None and before != after
        for before, after in zip(
            template.input_atoms, template.target_atoms, strict=True
        )
    )
    events = structural_rewrite_event_count(template)
    return {
        "input_atom_count": len(template.input_atoms),
        "created_atom_count": created,
        "deleted_atom_count": deleted,
        "restated_atom_count": restated,
        "bond_change_count": events - created - deleted - restated,
        "rewrite_event_count": events,
        "rewrite_scale": "local" if events <= 3 else "medium" if events <= 11 else "large",
    }


def _binding_census(source, expert, budgets: VirtualJointRegionBudgets) -> dict[str, dict]:
    details: dict[str, dict] = {}
    visits = 0
    for template in expert.templates:
        remaining = budgets.max_binding_visits - visits
        if remaining < 1:
            details[template.template_id] = {
                "visited": False,
                "assignments": (),
                "truncated": None,
                "visits": 0,
            }
            continue
        census = transfer_bindings(
            template,
            source,
            max_bindings=budgets.max_bindings_per_template,
            max_visits=remaining,
        )
        visits += census.visits
        details[template.template_id] = {
            "visited": True,
            "assignments": census.assignments,
            "truncated": bool(census.truncated),
            "visits": int(census.visits),
        }
    details["__summary__"] = {"binding_visits": visits}
    return details


def _constituent_reports(
    routes: list[dict[str, Any]],
    expert: RouteDistilledGoalExpert,
    budgets: VirtualJointRegionBudgets,
    telemetry: dict[str, Any],
) -> list[dict]:
    source = routes[0]["source"]
    census = _binding_census(source, expert, budgets)
    if int(census["__summary__"]["binding_visits"]) != int(
        telemetry["binding_visits"]
    ):
        raise RuntimeError("independent binding census disagrees with runtime telemetry")
    ranks = {
        row["constituent_key"]: row for row in telemetry["bound_constituent_ranks"]
    }
    marginal_probabilities = dict(
        zip(expert.marginal.template_ids, expert.marginal.probabilities, strict=True)
    )
    reports = []
    for route in routes:
        for region_index, (template, binding) in enumerate(
            zip(route["templates"], route["bindings"], strict=True)
        ):
            detail = census.get(template.template_id)
            if detail is None:
                raise RuntimeError("teacher template is absent from checkpoint vocabulary")
            patch = materialize_template(template, source, binding)
            key = _constituent_key(template, binding, patch)
            rank_row = ranks.get(key)
            binding_present = binding in detail["assignments"]
            if rank_row is not None:
                absence_reason = None
            elif not detail["visited"]:
                absence_reason = "template_not_reached_before_global_binding_visit_budget"
            elif not binding_present and detail["truncated"]:
                absence_reason = "teacher_binding_not_enumerated_before_binding_truncation"
            elif not binding_present:
                absence_reason = "teacher_binding_not_in_runtime_binding_fiber"
            else:
                absence_reason = "bound_constituent_missing_after_materialization"
            selected = bool(rank_row and rank_row["selected_for_expansion"])
            reports.append(
                {
                    "route_id": route["route_id"],
                    "region_index": region_index,
                    "template_id": template.template_id,
                    "constituent_key": key,
                    "marginal_template_probability": float(
                        marginal_probabilities[template.template_id]
                    ),
                    "single_region_log_score": float(
                        expert.marginal.score((template,))
                    ),
                    "size_signature": _template_size_signature(template),
                    "bound_rank": None if rank_row is None else int(rank_row["rank"]),
                    "selected_for_expansion": selected,
                    "binding_census_truncated": detail["truncated"],
                    "binding_census_visits": int(detail["visits"]),
                    "teacher_binding_present": binding_present,
                    "absence_reason": absence_reason,
                    "selection_reason_if_not_selected": (
                        None
                        if selected
                        else absence_reason
                        or "bound_rank_exceeds_fixed_expansion_width"
                    ),
                }
            )
    if len(reports) != 9:
        raise RuntimeError("strong route constituent census changed")
    return reports


def _bound_size_census(expert, telemetry: dict[str, Any]) -> dict[str, Any]:
    templates = {row.template_id: row for row in expert.templates}
    by_signature: dict[str, dict[str, Any]] = {}
    by_scale: dict[str, dict[str, int]] = {
        scale: {"all_bound": 0, "selected_top_48": 0}
        for scale in ("local", "medium", "large")
    }
    for row in telemetry["bound_constituent_ranks"]:
        signature = _template_size_signature(templates[row["template_id"]])
        signature_id = identity(signature)
        selected = bool(row["selected_for_expansion"])
        census = by_signature.setdefault(
            signature_id,
            {
                "size_signature_sha256": signature_id,
                "size_signature": signature,
                "all_bound": 0,
                "selected_top_48": 0,
            },
        )
        census["all_bound"] += 1
        census["selected_top_48"] += int(selected)
        scale_census = by_scale[signature["rewrite_scale"]]
        scale_census["all_bound"] += 1
        scale_census["selected_top_48"] += int(selected)
    return {
        "by_rewrite_scale": by_scale,
        "by_size_signature": [by_signature[key] for key in sorted(by_signature)],
    }


def _sequential_teacher_baseline(route: dict[str, Any]) -> dict:
    status_counts: Counter[str] = Counter()
    exact_orders = 0
    for order in itertools.permutations(range(len(route["templates"]))):
        current = route["source"]
        primitives = 0
        complete = True
        for region_index in order:
            template = route["templates"][region_index]
            binding = route["bindings"][region_index]
            try:
                patch = materialize_template(template, current, binding)
                _target, _ = instantiate_goal(
                    current, StructuralGoal((patch,)), (binding,)
                )
                program = program_from_structural_goal(StructuralGoal((patch,)))
                receipt = execute_complete_region_program(
                    current,
                    program,
                    resolved_bindings=(binding,),
                    config=RealizerConfig(
                        maximum_primitives=32 - primitives,
                        maximum_expansions=4_000,
                    ),
                )
            except ValueError:
                status_counts["invalid_single_region_target"] += 1
                complete = False
                break
            status = str(receipt["status"])
            status_counts[status] += 1
            if status != "committed":
                complete = False
                break
            if int(receipt.get("primitive_teacher_actions_used", -1)) != 0:
                raise RuntimeError("sequential baseline used primitive teacher actions")
            primitives += int(receipt["realized_primitive_count"])
            current = decode_state(receipt["committed_endpoint_state"])
        if complete and canonical_state_key(current) == route["endpoint_key"]:
            exact_orders += 1
    return {
        "region_orders_attempted": 6,
        "exact_region_orders": exact_orders,
        "status_counts": dict(sorted(status_counts.items())),
    }


def _sanitized_telemetry(
    telemetry: dict[str, Any], expert: RouteDistilledGoalExpert
) -> dict[str, Any]:
    ranks = telemetry["bound_constituent_ranks"]
    templates = {row.template_id: row for row in expert.templates}
    compact_ranks = [
        {
            "rank": int(row["rank"]),
            "constituent_key": row["constituent_key"],
            "template_id": row["template_id"],
            "selected_for_expansion": bool(row["selected_for_expansion"]),
            "rewrite_event_count": structural_rewrite_event_count(
                templates[row["template_id"]]
            ),
            "rewrite_scale": _template_size_signature(
                templates[row["template_id"]]
            )["rewrite_scale"],
        }
        for row in ranks
    ]
    return {
        **{key: value for key, value in telemetry.items() if key != "bound_constituent_ranks"},
        "bound_constituent_rank_rows": len(ranks),
        "bound_constituent_rank_census_sha256": identity(compact_ranks),
        "bound_constituent_ranked_census": compact_ranks,
    }


def _autonomous_reports(
    routes: list[dict[str, Any]], expert: RouteDistilledGoalExpert, batch
) -> tuple[list[dict], dict]:
    route_reports = []
    for route in routes:
        exact_rank = next(
            (
                rank
                for rank, proposal in enumerate(batch.proposals, 1)
                if proposal.endpoint_key == route["endpoint_key"]
            ),
            None,
        )
        equivalent_rank = next(
            (
                rank
                for rank, proposal in enumerate(batch.proposals, 1)
                if transformation_equivalent(
                    route["source"], proposal.endpoint, route["endpoint"]
                )
            ),
            None,
        )
        route_reports.append(
            {
                "route_id": route["route_id"],
                "exact_endpoint_recovered": exact_rank is not None,
                "exact_endpoint_rank": exact_rank,
                "transformation_equivalent_recovered": equivalent_rank is not None,
                "transformation_equivalent_rank": equivalent_rank,
            }
        )
    proposal_signature = [
        {
            "endpoint_sha256": identity(proposal.endpoint_key),
            "constituent_keys": [step.constituent_key for step in proposal.steps],
            "program_id": proposal.program_id,
            "actions_sha256": identity(list(proposal.actions)),
            "primitive_count": proposal.primitive_count,
        }
        for proposal in batch.proposals
    ]
    return route_reports, {
        "unique_committed_endpoints": len(batch.proposals),
        "proposal_set_sha256": identity(proposal_signature),
        "telemetry": _sanitized_telemetry(batch.telemetry, expert),
    }


def _old_autonomous_baseline(routes: list[dict[str, Any]]) -> dict:
    payload, payload_sha256 = _envelope(OLD_AUTONOMOUS)
    if len(payload["cells"]) != 1 or payload["cells"][0]["cell"] != "5ht1b_0":
        raise ValueError("old autonomous baseline cell changed")
    cell = payload["cells"][0]
    candidate_keys = {str(row["endpoint_key"]) for row in cell["candidates"]}
    recovered = [
        route["route_id"]
        for route in routes
        if route["endpoint_key"] in candidate_keys
    ]
    return {
        "evidence": "sealed pre-joint same-root autonomous proposal shard",
        "artifact_payload_sha256": payload_sha256,
        "committed_endpoints": int(cell["complete_programs_committed"]),
        "strong_routes_recovered": len(recovered),
        "strong_route_count": len(routes),
        "recovered_route_ids": recovered,
    }


def run() -> dict:
    budgets = VirtualJointRegionBudgets()
    routes, route_context = _route_inputs()
    expert, checkpoint_payload_sha256 = _load_expert()
    source = route_context["source"]
    if any(canonical_state_key(row["source"]) != canonical_state_key(source) for row in routes):
        raise RuntimeError("strong route source join changed")

    teacher_forced = []
    for route in routes:
        result = _teacher_forced_joint(route, budgets)
        teacher_forced.append({"route_id": route["route_id"], **result})

    batch = propose_virtual_joint_region_paths(source, expert, budgets=budgets)
    constituents = _constituent_reports(routes, expert, budgets, batch.telemetry)
    autonomous_routes, autonomous_summary = _autonomous_reports(routes, expert, batch)
    autonomous_summary["bound_constituent_size_census"] = _bound_size_census(
        expert, batch.telemetry
    )
    autonomous_summary["budget_accounting"] = {
        "binding_visits": {
            "used": int(batch.telemetry["binding_visits"]),
            "limit": budgets.max_binding_visits,
            "exhausted": bool(
                batch.telemetry.get("binding_visit_budget_exhausted", 0)
            ),
        },
        "planning_expansions": {
            "used": int(batch.telemetry["planning_expansions"]),
            "limit": budgets.max_planning_expansions,
            "exhausted": bool(
                batch.telemetry.get("planning_expansion_budget_exhausted", 0)
            ),
        },
        "retained_targets": {
            "used": int(batch.telemetry["unique_valid_targets"]),
            "limit": budgets.max_targets,
            "exhausted": bool(batch.telemetry.get("target_budget_exhausted", 0)),
        },
        "realization_attempts": {
            "used": int(batch.telemetry["realization_attempts"]),
            "limit": budgets.max_realization_attempts,
            "exhausted": bool(
                batch.telemetry.get("realization_attempt_budget_exhausted", 0)
            ),
        },
        "global_realizer_expansions": {
            "used": int(batch.telemetry["realizer_expansions"]),
            "limit": budgets.max_realization_expansions,
            "exhausted": bool(
                batch.telemetry.get("realization_expansion_budget_exhausted", 0)
            ),
        },
        "per_realization_expansion_cap": {
            "limit": budgets.maximum_expansions_per_realization,
            "abstentions_at_cap": int(
                batch.telemetry.get("per_realization_expansion_cap_abstentions", 0)
            ),
        },
    }
    sequential_routes = [
        {"route_id": route["route_id"], **_sequential_teacher_baseline(route)}
        for route in routes
    ]
    old_baseline = _old_autonomous_baseline(routes)

    teacher_exact = sum(
        row["exact_final_endpoint_execution"] for row in teacher_forced
    )
    teacher_reconstructed = sum(
        row["exact_final_target_reconstruction"] for row in teacher_forced
    )
    autonomous_exact = sum(row["exact_endpoint_recovered"] for row in autonomous_routes)
    autonomous_equivalent = sum(
        row["transformation_equivalent_recovered"] for row in autonomous_routes
    )
    sequential_exact = sum(row["exact_region_orders"] > 0 for row in sequential_routes)

    material_paths = (
        CHECKPOINT,
        SOURCE_AUDIT,
        FORENSICS,
        OLD_AUTONOMOUS,
        SHARED_LIBRARY,
        SOURCE_REGISTRY,
        T4_CONTRACT,
        *(ROUTE_DIRECTORY / f"{route_id}.json" for route_id in ROUTE_IDS),
        *(RECEIPTS[route_id] for route_id in ROUTE_IDS),
    )
    public_routes = []
    for route in routes:
        public_routes.append(
            {
                "route_id": route["route_id"],
                "reported_external_docking_score": route[
                    "reported_external_docking_score"
                ],
                "reported_run_seed": route["reported_run_seed"],
                "teacher_primitive_count": route["teacher_primitive_count"],
                "teacher_receipt_sha256": route["teacher_receipt_sha256"],
                "endpoint_sha256": identity(route["endpoint_key"]),
                "runtime_artifact_path": route["runtime_artifact_path"],
                "runtime_artifact_sha256": route["runtime_artifact_sha256"],
                "runtime_artifact_payload_sha256": route[
                    "runtime_artifact_payload_sha256"
                ],
            }
        )

    return {
        "schema_version": SCHEMA,
        "evidence": "computed zero-oracle known-answer and autonomous runtime support gate",
        "scientific_problem": (
            "test whether deferred validation restores exact same-root joint support "
            "for the three strong 5HT1B-0 delta=0.4 routes"
        ),
        "primary_model_output": (
            "complete exactly realized endpoints from jointly planned generic region constituents"
        ),
        "central_claim_under_test": (
            "joint STOP validation preserves known-answer complete-route support without "
            "teacher primitive actions or autonomous endpoint injection"
        ),
        "experimental_setting": (
            "single-CPU zero-oracle 5HT1B-0 delta=0.4 support probe using the shared "
            "task-independent route checkpoint and fixed default joint budgets"
        ),
        "declared_support": (
            "one to four same-root address-free regions, 48 persistent slots, at most "
            "40 active atoms and 32 exactly realized primitives"
        ),
        "claim_boundary": (
            "teacher forcing supplies only known templates and bindings to measure runtime "
            "support; autonomous generation receives only the source and shared checkpoint; "
            "no docking utility or prospective optimization is measured"
        ),
        "code_revision": _revision(),
        "implementation_commit_under_test": (
            "412c9f03d1f8b3c252b2155c64ae790c203b3a71"
        ),
        "configuration": {
            "joint_budgets": asdict(budgets),
            "expansion_width_frozen_before_run": 48,
            "budget_widening_after_inspection": False,
            "transformation_equivalence_radius": 2,
            "runtime_threads": 1,
        },
        "determinism": {
            "proposal_rng": "none",
            "seed": None,
            "ordering": "deterministic score, canonical endpoint and constituent identity",
            "serialization": "sorted-key compact JSON with SHA-256 payload identity",
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "rdkit": rdkit.__version__,
            "platform_system": platform.system(),
            "machine": platform.machine(),
            "precision": "native CPU graph execution; no learned numeric inference",
        },
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "gpu_seconds": 0,
        },
        "source": {
            "source_group": SOURCE_GROUP,
            "source_state_sha256": route_context["source_state_sha256"],
        },
        "routes": public_routes,
        "teacher_forced_joint": {
            "routes": teacher_forced,
            "exact_target_reconstruction_numerator": teacher_reconstructed,
            "exact_target_reconstruction_denominator": len(routes),
            "exact_execution_precision_numerator": teacher_exact,
            "exact_execution_precision_denominator": len(routes),
            "primitive_teacher_actions_used": 0,
        },
        "teacher_constituent_support": {
            "constituents": constituents,
            "bound_numerator": sum(row["bound_rank"] is not None for row in constituents),
            "bound_denominator": len(constituents),
            "selected_for_expansion_numerator": sum(
                row["selected_for_expansion"] for row in constituents
            ),
            "selected_for_expansion_denominator": len(constituents),
        },
        "autonomous_joint": {
            "routes": autonomous_routes,
            "exact_endpoint_recovery_numerator": autonomous_exact,
            "exact_endpoint_recovery_denominator": len(routes),
            "transformation_equivalent_recovery_numerator": autonomous_equivalent,
            "transformation_equivalent_recovery_denominator": len(routes),
            **autonomous_summary,
        },
        "baselines": {
            "old_same_root_autonomous": old_baseline,
            "sequential_teacher_forced": {
                "implementation_sha256": _sha256(
                    ROOT / "src/compose_v4/control/sequential_region_proposer.py"
                ),
                "routes": sequential_routes,
                "exact_route_recovery_numerator": sequential_exact,
                "exact_route_recovery_denominator": len(routes),
            },
        },
        "gate": {
            "teacher_forced_exact_reconstruction_all_routes": teacher_reconstructed
            == len(routes),
            "teacher_forced_exact_execution_all_routes": teacher_exact == len(routes),
            "teacher_forced_exact_precision_one": teacher_exact == len(routes),
            "zero_teacher_primitive_actions": True,
            "all_teacher_constituents_bound": all(
                row["bound_rank"] is not None for row in constituents
            ),
            "all_teacher_constituents_selected_for_expansion": all(
                row["selected_for_expansion"] for row in constituents
            ),
            "autonomous_exact_recovery_all_routes": autonomous_exact == len(routes),
            "autonomous_transformation_recovery_all_routes": autonomous_equivalent
            == len(routes),
            "fixed_default_budgets_used": True,
            "zero_oracle_execution": True,
            "known_answer_support_passed": teacher_exact == len(routes),
            "autonomous_recovery_passed": autonomous_exact == len(routes),
            "passed": teacher_exact == len(routes)
            and autonomous_exact == len(routes),
        },
        "limitations": [
            "The three known endpoints are retrospective Full-146 development routes.",
            "Teacher-forced reconstruction measures runtime support, not autonomous probability.",
            "Transformation equivalence is the frozen radius-2 local colored-graph criterion.",
            "No docking value, eligibility gain or prospective optimization claim is tested.",
        ],
        "material_inputs_sha256": {
            _relative(path): _sha256(path) for path in material_paths
        },
        "material_payload_sha256": {
            _relative(CHECKPOINT): checkpoint_payload_sha256,
            _relative(FORENSICS): route_context["forensic_payload_sha256"],
        },
        "implementation_inputs_sha256": {
            path: _sha256(ROOT / path) for path in CODE_INPUTS
        },
    }


def legacy_scale_allocation_diagnostic(result_path: Path) -> dict:
    """Apply the existing scale-balanced ordering to one sealed bound census."""

    result, result_payload_sha256 = _envelope(result_path)
    if result.get("schema_version") != SCHEMA:
        raise ValueError("legacy allocation input is not the joint support gate")
    width = int(result["configuration"]["joint_budgets"]["expansion_width"])
    if width != 48:
        raise ValueError("legacy allocation diagnostic requires frozen width 48")
    rows = result["autonomous_joint"]["telemetry"].get(
        "bound_constituent_ranked_census"
    )
    if not isinstance(rows, list) or len(rows) != int(
        result["autonomous_joint"]["telemetry"]["bound_constituent_rank_rows"]
    ):
        raise ValueError("joint support gate omitted its bound constituent census")
    if [int(row["rank"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("bound constituent ranks are not contiguous")

    buckets = {
        scale: [row for row in rows if row["rewrite_scale"] == scale]
        for scale in REWRITE_SCALE_BANDS
    }
    positions = {scale: 0 for scale in REWRITE_SCALE_BANDS}
    selected = []
    while len(selected) < width:
        added = False
        for scale in REWRITE_SCALE_BANDS:
            position = positions[scale]
            if position >= len(buckets[scale]):
                continue
            selected.append(buckets[scale][position])
            positions[scale] += 1
            added = True
            if len(selected) == width:
                break
        if not added:
            break
    selected_ranks = {
        row["constituent_key"]: rank for rank, row in enumerate(selected, 1)
    }
    teachers = result["teacher_constituent_support"]["constituents"]
    teacher_rows = [
        {
            "route_id": row["route_id"],
            "region_index": int(row["region_index"]),
            "constituent_key": row["constituent_key"],
            "original_bound_rank": row["bound_rank"],
            "rewrite_event_count": int(row["size_signature"]["rewrite_event_count"]),
            "rewrite_scale": row["size_signature"]["rewrite_scale"],
            "default_selected_for_expansion": bool(row["selected_for_expansion"]),
            "legacy_scale_balanced_selected": row["constituent_key"] in selected_ranks,
            "legacy_scale_balanced_selection_rank": selected_ranks.get(
                row["constituent_key"]
            ),
        }
        for row in teachers
    ]
    scale_census = {
        scale: {
            "all_bound": len(buckets[scale]),
            "legacy_selected_at_width_48": sum(
                row["rewrite_scale"] == scale for row in selected
            ),
        }
        for scale in REWRITE_SCALE_BANDS
    }
    selected_teacher_count = sum(
        row["legacy_scale_balanced_selected"] for row in teacher_rows
    )
    return {
        "schema_version": LEGACY_ALLOCATION_SCHEMA,
        "evidence": (
            "derived zero-oracle allocation diagnostic over the sealed fixed-run "
            "bound constituent census"
        ),
        "claim_boundary": (
            "this reorders already bound constituents only; it does not rerun binding, "
            "planning or realization and does not modify the joint proposer"
        ),
        "code_revision": _revision(),
        "configuration": {
            "width": width,
            "scale_order": list(REWRITE_SCALE_BANDS),
            "allocation_semantics": (
                "existing select_scale_balanced_proposals round-robin over "
                "local, medium and large buckets preserving within-bucket rank"
            ),
            "new_learned_rule": False,
            "budget_widening": False,
        },
        "source_result": {
            "path": (
                _relative(result_path)
                if result_path.is_relative_to(ROOT)
                else str(result_path)
            ),
            "sha256": _sha256(result_path),
            "payload_sha256": result_payload_sha256,
            "bound_constituent_census_sha256": result["autonomous_joint"][
                "telemetry"
            ]["bound_constituent_rank_census_sha256"],
        },
        "costs": {
            "binding_calls": 0,
            "planning_calls": 0,
            "realization_calls": 0,
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
        },
        "bound_constituents": len(rows),
        "selected_constituents": len(selected),
        "selection_sha256": identity(
            [row["constituent_key"] for row in selected]
        ),
        "scale_census": scale_census,
        "teacher_constituents": teacher_rows,
        "teacher_selection": {
            "default_selected_numerator": sum(
                row["default_selected_for_expansion"] for row in teacher_rows
            ),
            "legacy_scale_balanced_selected_numerator": selected_teacher_count,
            "denominator": len(teacher_rows),
            "any_selected": selected_teacher_count > 0,
            "all_selected": selected_teacher_count == len(teacher_rows),
        },
        "implementation_inputs_sha256": {
            "src/compose_v4/control/structural_subgoal_policy.py": _sha256(
                ROOT / "src/compose_v4/control/structural_subgoal_policy.py"
            ),
            "tools/t4_virtual_joint_5ht1b0_support_gate.py": _sha256(Path(__file__)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--derive-legacy-scale-allocation-from",
        type=Path,
        help="derive a separate legacy scale-balanced allocation diagnostic",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite support-gate result: {output}")
    payload = (
        legacy_scale_allocation_diagnostic(
            args.derive_legacy_scale_allocation_from.resolve()
        )
        if args.derive_legacy_scale_allocation_from is not None
        else run()
    )
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(output)
    print(json.dumps(envelope, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
