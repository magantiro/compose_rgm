from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from compose_v4.data.editing_v2_process_v2_schema import (
    AUTHORITY_FIELDS,
    canonical_sha256,
)
from compose_v4.experiments.editing_v2_process_v2_p50_prerequisites import (
    ProcessV2P50ScopedPrerequisites,
)
from compose_v4.experiments.editing_v2_process_v2_p50_result import (
    DECISION_GO_STATUS,
    DECISION_NO_GO_STATUS,
    ProcessV2P50ResultError,
    build_process_v2_p50_decision,
    build_process_v2_p50_result,
    validate_process_v2_p50_decision,
    validate_process_v2_p50_result,
)

ROOT = Path(__file__).resolve().parents[1]
UNSUPPORTED_CELL = "editing_v2_active8_v1:atom_delete:connected_nonleaf_death"


def _policy() -> dict[str, object]:
    return json.loads((ROOT / "configs/editing_v2_process_v2_p50_recipe_policy.json").read_text())


def _required_cells() -> tuple[str, ...]:
    payload = json.loads(
        (ROOT / "configs/editing_v2_process_v2_development_cell_roles.json").read_text()
    )
    return tuple(payload["required_cell_ids"])


def _prerequisites(policy: dict[str, object]) -> ProcessV2P50ScopedPrerequisites:
    return ProcessV2P50ScopedPrerequisites(
        process_identity_sha256=policy["process_identity"]["process_identity_sha256"],
        active8_completion_sha256="1" * 64,
        gate_zero_decision_sha256="2" * 64,
        t1_capacity_policy_sha256="3" * 64,
        t1_result_sha256="4" * 64,
        t1_decision_sha256="5" * 64,
        t1_initial_model_state_sha256="6" * 64,
        t1_score_revision_receipt_sha256="7" * 64,
        p50_recipe_policy_sha256=policy["contract_sha256"],
        active_families=tuple(policy["active_families"]),
        optimizer_steps=50,
        batch_size=64,
    )


def _inputs() -> tuple[
    dict[str, object], ProcessV2P50ScopedPrerequisites, dict[str, object]
]:
    policy = _policy()
    prerequisites = _prerequisites(policy)
    families = prerequisites.active_families
    cells = _required_cells()
    observable_cells = tuple(cell for cell in cells if cell != UNSUPPORTED_CELL)
    environment_body = {
        "hardware_class": "test_gpu",
        "device_name": "test",
        "device_capability": "0.0",
        "accelerator_class": "gpu",
        "dtype": "float32",
        "mixed_precision": False,
        "batch_size": 64,
        "python_version": "test",
        "torch_version": "test",
        "cuda_version": "test",
        "cudnn_version": "test",
        "rdkit_version": "test",
        "numpy_version": "test",
    }
    provenance = {
        "p50_prerequisite_binding_sha256": prerequisites.binding_sha256,
        "process_identity_sha256": prerequisites.process_identity_sha256,
        "active8_completion_sha256": prerequisites.active8_completion_sha256,
        "gate_zero_decision_sha256": prerequisites.gate_zero_decision_sha256,
        "t1_capacity_policy_sha256": prerequisites.t1_capacity_policy_sha256,
        "t1_result_sha256": prerequisites.t1_result_sha256,
        "t1_decision_sha256": prerequisites.t1_decision_sha256,
        "t1_initial_model_state_sha256": prerequisites.t1_initial_model_state_sha256,
        "t1_score_revision_receipt_sha256": (
            prerequisites.t1_score_revision_receipt_sha256
        ),
        "p50_recipe_policy_sha256": prerequisites.p50_recipe_policy_sha256,
        "prepared_inputs_sha256": "7" * 64,
        "training_stream_sha256": "8" * 64,
        "validation_stream_sha256": "9" * 64,
        "initial_model_state_sha256": prerequisites.t1_initial_model_state_sha256,
        "runner_implementation_sha256": "a" * 64,
        "runner_source_revision_sha256": "b" * 64,
        "collated_completion_sha256": "c" * 64,
        "collated_payload_file_sha256": "d" * 64,
        "execution_environment": {
            **environment_body,
            "environment_sha256": canonical_sha256(environment_body),
        },
    }
    expected_runtime_provenance = {
        name: provenance[name]
        for name in (
            "prepared_inputs_sha256",
            "training_stream_sha256",
            "validation_stream_sha256",
            "runner_implementation_sha256",
            "runner_source_revision_sha256",
            "collated_completion_sha256",
            "collated_payload_file_sha256",
        )
    }
    run_integrity = {
        "optimizer_steps_completed": 50,
        "scheduled_example_count": 3200,
        "batch_size": 64,
        "initialization": "scratch_from_t1_bound_initial_model_state",
        "resume_requested": False,
        "resume_count": 0,
        "abort_triggered": False,
        "abort_reasons": [],
        "nonfinite_event_count": 0,
        "unsupported_teacher_count": 0,
        "missing_candidate_count": 0,
        "provenance_drift_detected": False,
        "stream_identity_drift_detected": False,
        "deterministic_algorithms_enabled": True,
        "dtype": "float32",
        "mixed_precision": False,
        "hazard_included": False,
        "t1_selected_checkpoint_loaded": False,
        "initial_model_state_sha256": prerequisites.t1_initial_model_state_sha256,
        "terminal_model_state_sha256": "c" * 64,
        "optimizer_state_sha256": "d" * 64,
        "terminal_checkpoint_file_sha256": "e" * 64,
        "hazard_initial_state_sha256": "f" * 64,
        "hazard_final_state_sha256": "f" * 64,
    }
    trajectory = [
        {
            "optimizer_step": step,
            "mean_batch_canonical_successor_nll": 1.0 - step / 1000,
            "global_gradient_finite": True,
            "global_gradient_l2": 1.0,
        }
        for step in range(1, 51)
    ]
    validation: list[dict[str, object]] = []
    for cell_index, cell in enumerate(observable_cells):
        family = next(family for family in families if f":{family}:" in cell)
        for item_index in range(4):
            validation.append(
                {
                    "validation_entry_sha256": f"{cell_index * 4 + item_index + 1:064x}",
                    "family": family,
                    "semantic_cell_id": cell,
                    "baseline_canonical_successor_nll": 1.0,
                    "final_canonical_successor_nll": 0.9,
                }
            )
    validation.sort(
        key=lambda row: (
            families.index(row["family"]),
            observable_cells.index(row["semantic_cell_id"]),
            row["validation_entry_sha256"],
        )
    )
    family_exposure = [
        {
            "family": family,
            "planned_example_count": 400,
            "observed_example_count": 400,
            "planned_optimizer_step_opportunities": 50,
            "finite_nonzero_global_gradient_exposure_steps": 40,
            "all_exposure_step_global_gradients_finite": True,
            "cumulative_exposure_step_global_gradient_l2": 1.0,
            "finite_nonzero_action_route_gradient_exposure_steps": 40,
            "all_exposure_step_action_route_gradients_finite": True,
            "cumulative_exposure_step_action_route_gradient_l2": 1.0,
            "required_revision_parameter": (
                "graft_relation_head.weight"
                if family == "bond_reroute"
                else (
                    "ring_restate_context_head.weight"
                    if family == "ring_system_restate"
                    else None
                )
            ),
            "finite_nonzero_revision_parameter_gradient_exposure_steps": (
                40 if family in {"bond_reroute", "ring_system_restate"} else 0
            ),
            "all_exposure_step_revision_parameter_gradients_finite": True,
            "cumulative_exposure_step_revision_parameter_gradient_l2": (
                1.0 if family in {"bond_reroute", "ring_system_restate"} else 0.0
            ),
        }
        for family in families
    ]
    cell_exposure = [
        {
            "semantic_cell_id": cell,
            "planned_example_count": 189 if index < 4 else 188,
            "observed_example_count": 189 if index < 4 else 188,
            "planned_optimizer_step_opportunities": 50,
            "finite_nonzero_global_gradient_exposure_steps": 40,
            "all_exposure_step_global_gradients_finite": True,
            "cumulative_exposure_step_global_gradient_l2": 1.0,
        }
        for index, cell in enumerate(cells)
    ]
    inputs = {
        "recipe_policy": policy,
        "prerequisites": prerequisites,
        "required_cell_ids": cells,
        "validation_unsupported_required_cells": (UNSUPPORTED_CELL,),
        "expected_runtime_provenance": expected_runtime_provenance,
        "provenance": provenance,
        "run_integrity": run_integrity,
        "trajectory": trajectory,
        "validation_entry_metrics": validation,
        "family_training_evidence": family_exposure,
        "semantic_cell_training_evidence": cell_exposure,
    }
    return policy, prerequisites, inputs


def _build() -> tuple[
    dict[str, object], ProcessV2P50ScopedPrerequisites, dict[str, object]
]:
    policy, prerequisites, inputs = _inputs()
    return policy, prerequisites, build_process_v2_p50_result(**inputs)


def _validate_args(
    policy: dict[str, object], prerequisites: ProcessV2P50ScopedPrerequisites
) -> dict[str, object]:
    return {
        "recipe_policy": policy,
        "prerequisites": prerequisites,
        "required_cell_ids": _required_cells(),
        "validation_unsupported_required_cells": (UNSUPPORTED_CELL,),
        "expected_runtime_provenance": {
            "prepared_inputs_sha256": "7" * 64,
            "training_stream_sha256": "8" * 64,
            "validation_stream_sha256": "9" * 64,
            "runner_implementation_sha256": "a" * 64,
            "runner_source_revision_sha256": "b" * 64,
            "collated_completion_sha256": "c" * 64,
            "collated_payload_file_sha256": "d" * 64,
        },
    }


def test_result_is_strictly_nonauthorizing_and_decision_is_the_only_p500_go() -> None:
    policy, prerequisites, result = _build()
    args = _validate_args(policy, prerequisites)

    assert validate_process_v2_p50_result(result, **args) == result
    assert result["p500_authorized"] is False
    assert all(result[field] is False for field in AUTHORITY_FIELDS)
    decision = build_process_v2_p50_decision(result, **args, result_file_sha256="f" * 64)

    assert decision["status"] == DECISION_GO_STATUS
    assert decision["p500_authorized"] is True
    assert all(decision[field] is False for field in AUTHORITY_FIELDS)
    assert decision["failed_checks"] == []
    assert (
        validate_process_v2_p50_decision(
            decision,
            result=result,
            **args,
            result_file_sha256="f" * 64,
            require_p500_go=True,
        )
        == decision
    )


def test_absent_held_out_cell_is_reported_not_fabricated_or_automatically_failed() -> None:
    policy, prerequisites, result = _build()

    assert result["validation_unsupported_required_cells"] == [UNSUPPORTED_CELL]
    assert all(
        row["semantic_cell_id"] != UNSUPPORTED_CELL for row in result["validation_entry_metrics"]
    )
    assert len(result["semantic_cell_validation_nll_checks"]) == 16
    assert len(result["semantic_cell_training_checks"]) == 17
    assert all(result["threshold_checks"].values())

    with pytest.raises(ProcessV2P50ResultError, match="identity disagrees|coverage"):
        validate_process_v2_p50_result(
            result,
            **{
                **_validate_args(policy, prerequisites),
                "validation_unsupported_required_cells": (),
            },
        )


def test_cell_nonincrease_failure_is_no_go_even_when_not_catastrophic() -> None:
    policy, prerequisites, inputs = _inputs()
    target = inputs["validation_entry_metrics"][0]["semantic_cell_id"]
    for row in inputs["validation_entry_metrics"]:
        if row["semantic_cell_id"] == target:
            row["final_canonical_successor_nll"] = 1.01
    result = build_process_v2_p50_result(**inputs)
    decision = build_process_v2_p50_decision(
        result,
        **_validate_args(policy, prerequisites),
        result_file_sha256="f" * 64,
    )

    assert result["threshold_checks"]["semantic_cell_validation_nonincrease"] is False
    assert result["threshold_checks"]["semantic_cell_validation_no_catastrophic_regression"] is True
    assert decision["status"] == DECISION_NO_GO_STATUS
    assert decision["p500_authorized"] is False
    with pytest.raises(ProcessV2P50ResultError, match="not a bounded-P500 GO"):
        validate_process_v2_p50_decision(
            decision,
            result=result,
            **_validate_args(policy, prerequisites),
            result_file_sha256="f" * 64,
            require_p500_go=True,
        )


def test_catastrophic_family_and_cell_regression_is_recorded_as_no_go() -> None:
    policy, prerequisites, inputs = _inputs()
    target_family = inputs["validation_entry_metrics"][0]["family"]
    for row in inputs["validation_entry_metrics"]:
        if row["family"] == target_family:
            row["final_canonical_successor_nll"] = 1.30
    result = build_process_v2_p50_result(**inputs)
    decision = build_process_v2_p50_decision(
        result,
        **_validate_args(policy, prerequisites),
        result_file_sha256="f" * 64,
    )

    assert result["threshold_checks"]["family_validation_no_catastrophic_regression"] is False
    assert (
        result["threshold_checks"]["semantic_cell_validation_no_catastrophic_regression"] is False
    )
    assert decision["status"] == DECISION_NO_GO_STATUS


def test_hazard_change_and_missing_gradient_exposure_each_block_p500() -> None:
    policy, prerequisites, inputs = _inputs()
    inputs["run_integrity"]["hazard_final_state_sha256"] = "0" * 64
    result = build_process_v2_p50_result(**inputs)
    assert result["threshold_checks"]["hazard_frozen_excluded_and_unchanged"] is False

    policy, prerequisites, inputs = _inputs()
    unsupported = next(
        row
        for row in inputs["semantic_cell_training_evidence"]
        if row["semantic_cell_id"] == UNSUPPORTED_CELL
    )
    unsupported["finite_nonzero_global_gradient_exposure_steps"] = 0
    result = build_process_v2_p50_result(**inputs)
    assert result["threshold_checks"]["semantic_cell_training_exposure_and_gradient"] is False


def test_resealed_provenance_or_derived_threshold_tampering_is_refused() -> None:
    policy, prerequisites, result = _build()
    changed = copy.deepcopy(result)
    changed["provenance"]["training_stream_sha256"] = "0" * 64
    changed["result_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "result_sha256"}
    )
    with pytest.raises(ProcessV2P50ResultError, match="runtime provenance"):
        validate_process_v2_p50_result(changed, **_validate_args(policy, prerequisites))

    changed = copy.deepcopy(result)
    changed["threshold_checks"]["family_validation_nonincrease"] = False
    changed["result_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "result_sha256"}
    )
    with pytest.raises(ProcessV2P50ResultError, match="derived threshold_checks"):
        validate_process_v2_p50_result(changed, **_validate_args(policy, prerequisites))


def test_builder_refuses_non_50_step_trajectory_and_missing_seventeenth_cell() -> None:
    _policy_value, _prerequisites_value, inputs = _inputs()
    inputs["trajectory"] = inputs["trajectory"][:-1]
    with pytest.raises(ProcessV2P50ResultError, match="exactly 50 rows"):
        build_process_v2_p50_result(**inputs)

    _policy_value, _prerequisites_value, inputs = _inputs()
    inputs["semantic_cell_training_evidence"] = [
        row
        for row in inputs["semantic_cell_training_evidence"]
        if row["semantic_cell_id"] != UNSUPPORTED_CELL
    ]
    with pytest.raises(ProcessV2P50ResultError, match="evidence coverage"):
        build_process_v2_p50_result(**inputs)
