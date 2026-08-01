"""Prospective semantic T1 capacity policy is strict and nonauthorizing."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from compose_v4.experiments.editing_v2_semantic_t1_capacity_policy import (
    SemanticT1CapacityPolicyError,
    load_semantic_t1_capacity_policy,
    validate_semantic_t1_capacity_policy,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "editing_v2_semantic_t1_capacity_policy_v1.json"


def _policy() -> dict:
    return load_semantic_t1_capacity_policy(POLICY_PATH)


def test_repository_policy_is_self_hashed_and_nonauthorizing() -> None:
    policy = _policy()
    assert policy["status"].startswith("FROZEN_PROSPECTIVE")
    assert policy["training_authorized"] is False
    assert policy["bounded_p50_authorized"] is False
    assert policy["hazard_included"] is False
    assert policy["sampling_law"]["importance_correction"] == "none"


def test_policy_balances_family_then_cell_and_uses_canonical_successors() -> None:
    policy = _policy()
    assert policy["sampling_law"]["order"] == [
        "model_family",
        "semantic_capability_cell",
        "unique_panel_entry",
    ]
    assert policy["objective_unit"].endswith("canonical_successor")
    assert policy["optimization"]["maximum_optimizer_steps"] == 500
    assert policy["optimization"]["deterministic_algorithms_required"] is True
    assert policy["optimization"]["mixed_precision"] is False
    assert policy["optimization"]["trajectory_evaluation"] == (
        "every_pre_update_state_and_terminal_state"
    )


def test_joint_model_is_primary_and_family_scope_ladder_is_failure_only() -> None:
    policy = _policy()
    optimization = policy["optimization"]
    assert optimization["parameter_scope"] == "all_trainable_active8_parameters"
    assert optimization["failure_diagnostics_only_for_failing_families"] is True
    assert optimization["failure_diagnostic_scope_order"] == [
        "heads_only",
        "heads_plus_local_adapter_if_distinct",
        "all",
    ]
    assert (
        policy["thresholds"]["require_every_unique_entry_teacher_successor_top1"]
        is True
    )


def test_repeated_empirical_law_cannot_use_duplicates_or_aliases() -> None:
    policy = _policy()
    empirical = policy["empirical_repeated_state_gate"]
    assert empirical["bounded_p50_capacity_prerequisite"] is False
    assert empirical["raw_record_multiplicity_is_observation_count"] is False
    assert empirical["mark_alias_multiplicity_is_observation_count"] is False
    assert empirical["empirical_law_claims_authorized"] is False


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (None, "bounded_p50_authorized", True),
        ("sampling_law", "importance_correction", "production_over_proposal"),
        ("optimization", "maximum_optimizer_steps", 501),
        ("thresholds", "minimum_unique_state_teacher_successor_probability", 0.7),
        (
            "empirical_repeated_state_gate",
            "raw_record_multiplicity_is_observation_count",
            True,
        ),
    ],
)
def test_policy_rejects_scope_authority_and_semantic_drift(
    section: str | None, field: str, value: object
) -> None:
    policy = _policy()
    mutated = copy.deepcopy(policy)
    target = mutated if section is None else mutated[section]
    target[field] = value
    with pytest.raises(SemanticT1CapacityPolicyError):
        validate_semantic_t1_capacity_policy(mutated)


def test_policy_rejects_unknown_fields() -> None:
    policy = _policy()
    policy["future_result"] = "not prospective"
    with pytest.raises(SemanticT1CapacityPolicyError, match="missing or unknown"):
        validate_semantic_t1_capacity_policy(policy)
