"""Fail-closed validation for the COMPOSE ring-operator decision contract.

The comparison is deliberately asymmetric in only one respect: the hybrid arm
adds ``ring_system_grow``. Both arms retain primitive cycle closing and opening.
The macro can therefore be evaluated as a shortcut, but never as a replacement
for bidirectional compositional topology control.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "compose.ring_operator_decision_contract"
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_REGIME_IDS = ("primitive_ringcore", "hybrid_optional_ring_growth")
REQUIRED_PRIMITIVES = {"cycle_insert", "cycle_attach"}
REQUIRED_EXPERIMENTS = {"E2", "E3", "E4", "E7"}
REQUIRED_TASK_IDS = {
    "source_conditioned_lead_transport",
    "coupled_size_and_ring_adaptation",
    "cycle_creation",
    "cycle_opening",
    "ring_size_change",
    "fused_spiro_or_bridged_modification_where_reachable",
    "topology_simplification",
    "topology_addition_then_dynamic_reversal",
    "pareto_archive_exploration",
    "pareto_fan_from_shared_prefix",
    "protected_path_constraints",
}
REQUIRED_PRIMARY_METRICS = {
    "topology_target_success_at_matched_primitive_equivalent_cost",
    "normalized_hypervolume_auc_at_matched_oracle_calls",
    "dynamic_adaptation_regret_at_matched_oracle_and_primitive_equivalent_cost",
    "path_feasible_endpoint_yield_at_matched_oracle_calls",
}
REQUIRED_COST_AXES = {
    "event_count",
    "primitive_equivalent_chemistry_cost",
    "oracle_calls",
    "model_forward_passes",
    "canonical_successors_scored",
    "accepted_edits",
    "wall_clock_seconds",
    "gpu_hours",
}
REQUIRED_FAIRNESS_FIELDS = {
    "train_validation_and_sealed_test_partitions",
    "source_target_endpoint_units",
    "primitive_cycle_open_and_close_records",
    "backbone_and_total_parameter_shapes",
    "optimizer_and_learning_rate_schedule",
    "batch_size_and_optimizer_update_count",
    "molecular_endpoint_draw_schedule",
    "effective_endpoint_target_coefficient",
    "random_seeds",
    "checkpoint_selection_rule",
}
READINESS_KEYS = {
    "model_allows_cycle_primitives_plus_macro",
    "production_successor_evaluator_enumerates_macro",
    "dictionary_oracle_covers_macro_aliases",
    "packed_successor_training_covers_macro_aliases",
    "atomic_constraint_filter_covers_macro",
    "macro_primitive_lowering_certificate_available",
    "paired_training_corpus_compiled",
}
REQUIRED_HARD_GATES = {
    "all_committed_states_supported_connected_and_valid",
    "zero_atomic_hard_constraint_violations",
    "successor_quotient_dictionary_equivalence",
    "cycle_insert_successor_learnability",
    "cycle_attach_successor_learnability",
    "no_material_cycle_open_or_close_regression",
    "held_out_topology_generalization",
    "dynamic_reversal_noninferiority",
    "no_test_selection",
}


class RingOperatorDecisionError(ValueError):
    """The ring-operator comparison is malformed or not authorized."""


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return sorted(duplicate)


def load_ring_operator_decision(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    validate_ring_operator_decision(payload)
    return payload


def validate_ring_operator_decision(contract: dict[str, Any]) -> None:
    if contract.get("schema") != EXPECTED_SCHEMA:
        raise RingOperatorDecisionError("unexpected ring-operator schema")
    if contract.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise RingOperatorDecisionError("unexpected ring-operator schema version")
    if contract.get("current_default") != "primitive_ringcore":
        raise RingOperatorDecisionError(
            "primitive RingCore must remain the default until the hybrid passes"
        )

    boundary = contract.get("non_negotiable_operator_boundary") or {}
    if set(boundary.get("required_in_every_arm") or ()) != REQUIRED_PRIMITIVES:
        raise RingOperatorDecisionError(
            "both primitive cycle closing and opening must be required in every arm"
        )
    if boundary.get("macro_under_test") != "ring_system_grow":
        raise RingOperatorDecisionError("the registered macro must be ring_system_grow")
    for key in (
        "macro_can_replace_cycle_insert",
        "macro_can_replace_cycle_attach",
        "macro_only_arm_is_scientifically_adequate",
    ):
        if boundary.get(key) is not False:
            raise RingOperatorDecisionError(
                "ring_system_grow cannot replace primitive closing or opening"
            )

    regimes = contract.get("regimes") or []
    regime_ids = [str(regime.get("id")) for regime in regimes]
    if tuple(regime_ids) != EXPECTED_REGIME_IDS:
        raise RingOperatorDecisionError(
            "regimes must be primitive RingCore followed by the hybrid accelerator"
        )
    if _duplicates(regime_ids):
        raise RingOperatorDecisionError("duplicate ring-operator regimes")
    for regime in regimes:
        if regime.get("enable_cycle_insert") is not True:
            raise RingOperatorDecisionError(
                f"{regime.get('id')} disables primitive cycle closing"
            )
        if regime.get("enable_cycle_attach") is not True:
            raise RingOperatorDecisionError(
                f"{regime.get('id')} disables primitive cycle opening"
            )
    if regimes[0].get("enable_ring_system_grow") is not False:
        raise RingOperatorDecisionError("primitive RingCore must disable ring_system_grow")
    if regimes[1].get("enable_ring_system_grow") is not True:
        raise RingOperatorDecisionError(
            "the hybrid regime must add ring_system_grow"
        )

    quotient = contract.get("successor_quotient_contract") or {}
    if quotient.get("identity") != "canonical_molecular_successor":
        raise RingOperatorDecisionError("comparison must use canonical successors")
    for key in (
        "aggregate_all_marks_with_same_successor",
        "aggregate_slot_symmetry_template_and_operand_aliases",
        "aggregate_macro_and_non_macro_marks_when_they_share_a_one_step_successor",
        "mark_level_power_topk_or_nucleus_controller_forbidden",
    ):
        if quotient.get(key) is not True:
            raise RingOperatorDecisionError(
                "all equivalent macro and primitive marks must be quotiented "
                f"successor-level; missing {key}"
            )
    for object_name in ("training_primary_object", "evaluation_primary_object", "controller_object"):
        if quotient.get(object_name) != "canonical_successor_probability":
            raise RingOperatorDecisionError(
                f"{object_name} must be canonical_successor_probability"
            )

    execution = contract.get("macro_execution_contract") or {}
    for key in (
        "complete_executable_mark_required",
        "deterministic_executor_required",
        "certified_primitive_lowering_required",
        "all_lowering_intermediates_supported_connected_and_valid",
        "no_repair_or_posthoc_sanitization",
        "macro_failure_rejects_entire_mark",
    ):
        if execution.get(key) is not True:
            raise RingOperatorDecisionError(
                f"macro execution contract is missing {key}"
            )

    constraints = contract.get("hard_constraint_contract") or {}
    if (
        constraints.get("application")
        != "atomic_whole_action_filter_before_successor_aggregation_and_controller_renormalization"
    ):
        raise RingOperatorDecisionError(
            "hard constraints must atomically filter the complete action"
        )
    for key in (
        "partial_macro_execution_forbidden",
        "posthoc_endpoint_filtering_is_not_constraint_enforcement",
        "reject_entire_macro_if_any_touched_atom_or_bond_is_protected",
        "successor_must_satisfy_all_path_constraints",
    ):
        if constraints.get(key) is not True:
            raise RingOperatorDecisionError(
                f"atomic constraint contract is missing {key}"
            )

    support = contract.get("support_claim_contract") or {}
    if support.get("one_step_support") != (
        "expected_to_differ_because_the_macro_adds_a_shortcut_edge"
    ):
        raise RingOperatorDecisionError(
            "the contract must acknowledge different one-step support"
        )
    if (
        support.get(
            "accelerator_claim_requires_every_macro_successor_to_have_a_certified_primitive_path"
        )
        is not True
    ):
        raise RingOperatorDecisionError(
            "an accelerator claim requires primitive-path reachability"
        )
    if support.get("support_changing_result_label") != (
        "extended_support_model_not_optional_accelerator"
    ):
        raise RingOperatorDecisionError(
            "support-changing results must be labeled as an extended-support model"
        )

    training = contract.get("training_fairness") or {}
    shared_training = set(training.get("shared_across_regimes") or ())
    missing_fairness = REQUIRED_FAIRNESS_FIELDS - shared_training
    if missing_fairness:
        raise RingOperatorDecisionError(
            f"training fairness fields are missing: {sorted(missing_fairness)}"
        )
    for key in (
        "macro_data_must_be_paired_with_a_primitive_decomposition_of_the_same_endpoint_unit",
        "macro_only_molecules_or_endpoint_targets_forbidden",
        "extra_macro_rows_cannot_increase_endpoint_exposure",
        "validation_only_checkpoint_selection",
        "test_metrics_for_checkpoint_or_regime_selection_forbidden",
    ):
        if training.get(key) is not True:
            raise RingOperatorDecisionError(
                f"training comparison is not fair; missing {key}"
            )
    if set(training.get("matched_views") or ()) != {
        "equal_optimizer_updates_and_endpoint_exposures",
        "equal_training_gpu_hour_ceiling",
    }:
        raise RingOperatorDecisionError(
            "training must include exposure-matched and compute-ceiling-matched views"
        )

    budgets = contract.get("budget_contract") or {}
    chemistry = budgets.get("primitive_equivalent_chemistry_cost") or {}
    if chemistry.get("definition") != (
        "sum_of_certified_production_primitive_lowering_lengths"
    ):
        raise RingOperatorDecisionError(
            "chemistry cost must use certified primitive-equivalent lowering length"
        )
    if chemistry.get("primitive_cycle_insert_cost") != 1:
        raise RingOperatorDecisionError("primitive cycle insertion must cost one")
    if chemistry.get("primitive_cycle_attach_cost") != 1:
        raise RingOperatorDecisionError("primitive cycle opening must cost one")
    cost_axes = set(budgets.get("required_reporting_axes") or ())
    if cost_axes != REQUIRED_COST_AXES:
        raise RingOperatorDecisionError(
            f"cost axes must exactly match the frozen set: {sorted(REQUIRED_COST_AXES)}"
        )
    if budgets.get("event_count_only_win_is_insufficient_for_hybrid_selection") is not True:
        raise RingOperatorDecisionError(
            "event-count improvement alone cannot select the macro"
        )

    tasks = contract.get("task_matrix") or []
    task_ids = [str(task.get("id")) for task in tasks]
    if set(task_ids) != REQUIRED_TASK_IDS or _duplicates(task_ids):
        raise RingOperatorDecisionError(
            "task matrix must cover the frozen editing capability set exactly once"
        )
    experiments = {str(task.get("experiment")) for task in tasks}
    if experiments != REQUIRED_EXPERIMENTS:
        raise RingOperatorDecisionError(
            "ring decision must cover E2, E3, E4, and E7"
        )
    if any(task.get("required") is not True for task in tasks):
        raise RingOperatorDecisionError("every registered task must be required")

    metrics = contract.get("metric_contract") or {}
    primary = [str(metric) for metric in metrics.get("primary_non_tautological") or ()]
    if set(primary) != REQUIRED_PRIMARY_METRICS or _duplicates(primary):
        raise RingOperatorDecisionError(
            "primary metrics must be the frozen non-tautological set"
        )
    hard_gates = set(metrics.get("hard_gates") or ())
    if hard_gates != REQUIRED_HARD_GATES:
        raise RingOperatorDecisionError(
            "hard gates must protect validity, quotient semantics, both cycle "
            "directions, topology generalization, and dynamic reversal"
        )
    if "committed_event_count_alone" not in set(
        metrics.get("forbidden_as_decision_evidence") or ()
    ):
        raise RingOperatorDecisionError(
            "committed event count alone must be forbidden as decision evidence"
        )

    readiness = contract.get("current_implementation_readiness") or {}
    if set(readiness) != READINESS_KEYS:
        raise RingOperatorDecisionError(
            "implementation-readiness checklist has drifted"
        )
    if not all(isinstance(readiness[key], bool) for key in READINESS_KEYS):
        raise RingOperatorDecisionError(
            "implementation-readiness checks must be booleans"
        )

    decision = contract.get("decision_rule") or {}
    if (
        decision.get(
            "primitive_ringcore_remains_default_unless_all_hybrid_conditions_pass"
        )
        is not True
    ):
        raise RingOperatorDecisionError(
            "primitive RingCore must remain the fail-closed default"
        )
    if decision.get("committed_event_reduction_alone_selects_hybrid") is not False:
        raise RingOperatorDecisionError(
            "committed-event reduction alone cannot select the hybrid"
        )


def _shared_evaluation_blockers(contract: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    readiness = contract["current_implementation_readiness"]
    for key in sorted(READINESS_KEYS):
        if readiness[key] is not True:
            blockers.append(f"implementation readiness is false: {key}")

    thresholds = contract.get("thresholds_to_freeze_before_paired_training") or {}
    for name, value in sorted(thresholds.items()):
        if value is None:
            blockers.append(f"unfrozen threshold: {name}")

    artifacts = contract.get("task_artifacts") or {}
    for partition in ("development", "validation", "sealed_test"):
        if not artifacts.get(partition):
            blockers.append(f"missing task artifact: {partition}")
    return blockers


def paired_training_launch_blockers(contract: dict[str, Any]) -> list[str]:
    """Return every unresolved item that blocks a paired ring-regime run."""

    validate_ring_operator_decision(contract)
    blockers = _shared_evaluation_blockers(contract)
    if contract.get("paired_training_authorized") is not True:
        blockers.append("paired_training_authorized is not true")
    if contract.get("status") != "FROZEN_PAIRED_TRAINING_AUTHORIZED":
        blockers.append("status is not FROZEN_PAIRED_TRAINING_AUTHORIZED")
    return blockers


def assert_paired_training_launch_authorized(contract: dict[str, Any]) -> None:
    blockers = paired_training_launch_blockers(contract)
    if blockers:
        raise RingOperatorDecisionError(
            "ring-operator contract blocks paired training: " + "; ".join(blockers)
        )


def paired_final_evaluation_blockers(contract: dict[str, Any]) -> list[str]:
    """Return blockers for the one-time sealed final comparison."""

    validate_ring_operator_decision(contract)
    blockers = _shared_evaluation_blockers(contract)
    if contract.get("paired_training_authorized") is not True:
        blockers.append("paired_training_authorized is not true")
    if contract.get("paired_training_completed") is not True:
        blockers.append("paired_training_completed is not true")
    if contract.get("paired_final_evaluation_authorized") is not True:
        blockers.append("paired_final_evaluation_authorized is not true")
    if contract.get("status") != "FROZEN_FINAL_EVALUATION_AUTHORIZED":
        blockers.append("status is not FROZEN_FINAL_EVALUATION_AUTHORIZED")
    return blockers


__all__ = [
    "RingOperatorDecisionError",
    "assert_paired_training_launch_authorized",
    "load_ring_operator_decision",
    "paired_final_evaluation_blockers",
    "paired_training_launch_blockers",
    "validate_ring_operator_decision",
]
