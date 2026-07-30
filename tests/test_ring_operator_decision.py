from __future__ import annotations

import copy
from pathlib import Path

import pytest

from compose_v4.experiments.ring_operator_decision import (
    RingOperatorDecisionError,
    assert_paired_training_launch_authorized,
    load_ring_operator_decision,
    paired_final_evaluation_blockers,
    paired_training_launch_blockers,
    validate_ring_operator_decision,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "ring_operator_decision_v1.json"


def test_contract_is_valid_but_fail_closed_before_paired_training() -> None:
    contract = load_ring_operator_decision(CONTRACT_PATH)
    blockers = paired_training_launch_blockers(contract)
    assert "paired_training_authorized is not true" in blockers
    assert (
        "implementation readiness is false: "
        "model_allows_cycle_primitives_plus_macro"
    ) in blockers
    assert any(item.startswith("unfrozen threshold:") for item in blockers)
    assert any(item.startswith("missing task artifact:") for item in blockers)
    with pytest.raises(RingOperatorDecisionError, match="blocks paired training"):
        assert_paired_training_launch_authorized(contract)


def test_macro_cannot_replace_primitive_ring_opening() -> None:
    contract = load_ring_operator_decision(CONTRACT_PATH)
    broken = copy.deepcopy(contract)
    broken["regimes"][1]["enable_cycle_attach"] = False
    with pytest.raises(
        RingOperatorDecisionError, match="disables primitive cycle opening"
    ):
        validate_ring_operator_decision(broken)


def test_macro_arm_must_retain_primitive_ring_closing() -> None:
    contract = load_ring_operator_decision(CONTRACT_PATH)
    broken = copy.deepcopy(contract)
    broken["regimes"][1]["enable_cycle_insert"] = False
    with pytest.raises(
        RingOperatorDecisionError, match="disables primitive cycle closing"
    ):
        validate_ring_operator_decision(broken)


def test_macro_aliases_must_be_aggregated_at_successor_level() -> None:
    contract = load_ring_operator_decision(CONTRACT_PATH)
    broken = copy.deepcopy(contract)
    broken["successor_quotient_contract"][
        "aggregate_macro_and_non_macro_marks_when_they_share_a_one_step_successor"
    ] = False
    with pytest.raises(
        RingOperatorDecisionError, match="quotiented successor-level"
    ):
        validate_ring_operator_decision(broken)


def test_hard_constraints_must_reject_a_complete_macro_atomically() -> None:
    contract = load_ring_operator_decision(CONTRACT_PATH)
    broken = copy.deepcopy(contract)
    broken["hard_constraint_contract"]["partial_macro_execution_forbidden"] = False
    with pytest.raises(RingOperatorDecisionError, match="atomic constraint"):
        validate_ring_operator_decision(broken)


def test_one_step_support_cannot_be_claimed_identical() -> None:
    contract = load_ring_operator_decision(CONTRACT_PATH)
    broken = copy.deepcopy(contract)
    broken["support_claim_contract"]["one_step_support"] = "identical"
    with pytest.raises(RingOperatorDecisionError, match="different one-step support"):
        validate_ring_operator_decision(broken)


def test_event_count_alone_cannot_select_macro() -> None:
    contract = load_ring_operator_decision(CONTRACT_PATH)
    broken = copy.deepcopy(contract)
    broken["budget_contract"][
        "event_count_only_win_is_insufficient_for_hybrid_selection"
    ] = False
    with pytest.raises(RingOperatorDecisionError, match="event-count improvement"):
        validate_ring_operator_decision(broken)


def test_sealed_final_evaluation_is_more_restrictive_than_training() -> None:
    contract = load_ring_operator_decision(CONTRACT_PATH)
    blockers = paired_final_evaluation_blockers(contract)
    assert "paired_training_authorized is not true" in blockers
    assert "paired_training_completed is not true" in blockers
    assert "paired_final_evaluation_authorized is not true" in blockers
    assert "status is not FROZEN_FINAL_EVALUATION_AUTHORIZED" in blockers


def test_future_final_evaluation_state_is_satisfiable() -> None:
    contract = load_ring_operator_decision(CONTRACT_PATH)
    ready = copy.deepcopy(contract)
    ready["status"] = "FROZEN_FINAL_EVALUATION_AUTHORIZED"
    for key in ready["current_implementation_readiness"]:
        ready["current_implementation_readiness"][key] = True
    for key in ready["thresholds_to_freeze_before_paired_training"]:
        ready["thresholds_to_freeze_before_paired_training"][key] = 0
    ready["task_artifacts"] = {
        "development": "configs/tasks/ring_operator.development.json",
        "validation": "configs/tasks/ring_operator.validation.json",
        "sealed_test": "configs/tasks/ring_operator.sealed_test.json",
    }
    ready["paired_training_authorized"] = True
    ready["paired_training_completed"] = True
    ready["paired_final_evaluation_authorized"] = True
    assert paired_final_evaluation_blockers(ready) == []
