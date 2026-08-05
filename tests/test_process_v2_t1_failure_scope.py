from __future__ import annotations

import copy

import pytest

from compose_v4.data.editing_v2_process_v2_schema import (
    authority_false_block,
    canonical_sha256,
)
from compose_v4.experiments.editing_v2_process_v2_t1_failure_scope import (
    NEXT_SCHEMA_VERSION,
    PASS_STATUS,
    ProcessV2T1FailureScopeError,
    SCHEMA,
    SCHEMA_VERSION,
    SUPPORTED_SCOPE,
    failing_families_from_capacity_result,
    next_scope_for_family,
    run_heads_only_failure_scope,
    run_next_failure_scope,
    validate_failure_scope_result,
)


def _capacity_policy() -> dict[str, object]:
    return {
        "required_families": ["atom_insert", "atom_delete", "cycle_attach"],
        "thresholds": {
            "minimum_every_unique_entry_teacher_successor_probability": 0.8
        },
    }


def _capacity_result() -> dict[str, object]:
    checks = {
        "family_top1": True,
        "family_probability": True,
        "family_nll": True,
        "cell_top1": True,
        "cell_probability": True,
        "cell_nll": True,
        "every_entry_top1": False,
        "every_entry_probability": False,
        "family_route_gradient": True,
        "action_route_gradient": True,
    }
    return {
        "threshold_checks": checks,
        "entry_metrics": [
            {
                "family": "atom_insert",
                "teacher_successor_probability": 0.9,
                "teacher_successor_top1": False,
            },
            {
                "family": "atom_delete",
                "teacher_successor_probability": 0.99,
                "teacher_successor_top1": True,
            },
            {
                "family": "cycle_attach",
                "teacher_successor_probability": 0.7,
                "teacher_successor_top1": True,
            },
        ],
    }


def test_failure_families_are_derived_only_from_the_entry_tail() -> None:
    assert failing_families_from_capacity_result(
        _capacity_result(), capacity_policy=_capacity_policy()
    ) == ("atom_insert", "cycle_attach")


def test_failure_scope_refuses_an_aggregate_failure() -> None:
    result = _capacity_result()
    result["threshold_checks"]["cell_probability"] = False
    with pytest.raises(ProcessV2T1FailureScopeError, match="aggregate or gradient"):
        failing_families_from_capacity_result(
            result, capacity_policy=_capacity_policy()
        )


def _result() -> dict[str, object]:
    checks = {
        "family_top1": True,
        "family_probability": True,
        "family_nll": True,
        "cell_top1": True,
        "cell_probability": True,
        "cell_nll": True,
        "every_entry_top1": True,
        "every_entry_probability": True,
        "family_route_gradient": True,
        "action_route_gradient": True,
    }
    metric = {
        "panel_entry_sha256": "8" * 64,
        "model_family": "atom_insert",
        "capability_cell_id": "cell:atom_insert",
        "teacher_successor_probability": 0.9,
        "teacher_successor_nll": 0.10536051565782628,
        "teacher_successor_rank": 1,
        "teacher_successor_top1": True,
        "canonical_successor_count": 2,
    }
    aggregate = {
        "entry_count": 1,
        "mean_teacher_successor_probability": 0.9,
        "minimum_teacher_successor_probability": 0.9,
        "mean_teacher_successor_nll": 0.10536051565782628,
        "maximum_teacher_successor_nll": 0.10536051565782628,
        "teacher_successor_top1_fraction": 1.0,
        "every_entry_teacher_successor_top1": True,
    }
    body = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": PASS_STATUS,
        **authority_false_block(),
        "family": "atom_insert",
        "scope": SUPPORTED_SCOPE,
        "input_capacity_result_file_sha256": "1" * 64,
        "input_capacity_result_sha256": "2" * 64,
        "capacity_policy_sha256": "3" * 64,
        "collated_completion_sha256": "4" * 64,
        "initial_model_state_sha256": "5" * 64,
        "panel_entry_count": 1,
        "panel_entry_inventory_sha256": "6" * 64,
        "optimizer_steps_completed": 10,
        "selected_step": 10,
        "trainable_parameter_count": 100,
        "trainable_parameter_inventory_sha256": "7" * 64,
        "trajectory": [
            {
                "step": 0,
                "minimum_entry_teacher_successor_probability": 0.1,
                "mean_entry_canonical_successor_nll": 2.302585092994046,
                "all_threshold_checks_pass": False,
            },
            {
                "step": 10,
                "minimum_entry_teacher_successor_probability": 0.9,
                "mean_entry_canonical_successor_nll": 0.10536051565782628,
                "all_threshold_checks_pass": True,
            },
        ],
        "selected_metrics": {
            "overall": aggregate,
            "by_family": {"atom_insert": aggregate},
            "by_nonempty_semantic_cell": {"cell:atom_insert": aggregate},
            "per_entry": [metric],
        },
        "selected_threshold_checks": checks,
        "gradient_evidence": {
            "atom_insert": {
                "family_route_finite_nonzero_seen": True,
                "family_route_nonzero_steps": 10,
                "family_route_cumulative_gradient_norm": 1.0,
                "action_route_finite_nonzero_seen": True,
                "action_route_nonzero_steps": 10,
                "action_route_cumulative_gradient_norm": 1.0,
            }
        },
        "diagnostic_passed": True,
    }
    return {**body, "result_sha256": canonical_sha256(body)}


def _next_result() -> dict[str, object]:
    result = _result()
    result["schema_version"] = NEXT_SCHEMA_VERSION
    result["scope"] = "all"
    result["input_prior_scope_result_file_sha256"] = "a" * 64
    result["input_prior_scope_result_sha256"] = "b" * 64
    _reseal(result)
    return result


def _reseal(value: dict[str, object]) -> None:
    body = {key: item for key, item in value.items() if key != "result_sha256"}
    value["result_sha256"] = canonical_sha256(body)


def test_failure_scope_result_is_self_validating_and_grants_no_authority() -> None:
    result = _result()
    assert validate_failure_scope_result(result) == result
    assert result["bounded_p50_authorized"] is False
    assert result["training_authorized"] is False


def test_failure_scope_result_refuses_resealed_family_metric_drift() -> None:
    result = copy.deepcopy(_result())
    result["selected_metrics"]["by_family"] = {"cycle_attach": {}}
    _reseal(result)
    with pytest.raises(ProcessV2T1FailureScopeError, match="identity disagrees"):
        validate_failure_scope_result(result)


def test_next_scope_is_local_only_when_the_family_has_a_distinct_adapter() -> None:
    policy = {
        "required_families": ["atom_insert", "cycle_attach"],
        "optimization": {
            "failure_diagnostic_scope_order": [
                "heads_only",
                "heads_plus_local_adapter_if_distinct",
                "all",
            ]
        },
    }
    assert next_scope_for_family("atom_insert", capacity_policy=policy) == "all"
    assert (
        next_scope_for_family("cycle_attach", capacity_policy=policy)
        == "heads_plus_local_adapter"
    )


def test_next_scope_result_binds_the_failed_predecessor() -> None:
    result = _next_result()
    assert validate_failure_scope_result(result) == result
    result["input_prior_scope_result_sha256"] = "c" * 64
    _reseal(result)
    assert validate_failure_scope_result(result) == result
    result["input_prior_scope_result_sha256"] = "not-a-sha256"
    _reseal(result)
    with pytest.raises(ProcessV2T1FailureScopeError, match="predecessor identity"):
        validate_failure_scope_result(result)


def test_heads_only_runner_refuses_a_family_that_did_not_fail() -> None:
    with pytest.raises(ProcessV2T1FailureScopeError, match="frozen failing family"):
        run_heads_only_failure_scope(
            object(),
            object(),
            family="atom_delete",
            failing_families=("atom_insert",),
            capacity_policy={
                "required_families": ["atom_insert", "atom_delete"],
                "optimization": {
                    "failure_diagnostic_scope_order": [
                        "heads_only",
                        "heads_plus_local_adapter_if_distinct",
                        "all",
                    ],
                    "failure_diagnostics_only_for_failing_families": True,
                },
            },
            input_capacity_result_file_sha256="1" * 64,
            input_capacity_result_sha256="2" * 64,
            collated_completion_sha256="3" * 64,
            initial_model_state_sha256="4" * 64,
        )


def test_next_scope_runner_refuses_a_passing_heads_only_predecessor() -> None:
    prior = _result()
    with pytest.raises(ProcessV2T1FailureScopeError, match="failed heads-only arm"):
        run_next_failure_scope(
            object(),
            object(),
            family="atom_insert",
            failing_families=("atom_insert",),
            capacity_policy={
                "policy_sha256": "3" * 64,
                "required_families": ["atom_insert"],
                "optimization": {
                    "failure_diagnostic_scope_order": [
                        "heads_only",
                        "heads_plus_local_adapter_if_distinct",
                        "all",
                    ],
                    "failure_diagnostics_only_for_failing_families": True,
                },
            },
            prior_scope_result=prior,
            prior_scope_result_file_sha256="a" * 64,
            input_capacity_result_file_sha256="1" * 64,
            input_capacity_result_sha256="2" * 64,
            collated_completion_sha256="4" * 64,
            initial_model_state_sha256="5" * 64,
        )
