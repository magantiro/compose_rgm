"""Strict Process-V2 P50 result and bounded-P500 decision boundary.

The P50 runtime supplies measurements.  This module recomputes their structural
identity, the frozen validation thresholds, and the next-stage decision.  The
result is always non-authorizing.  Only the separate decision may authorize the
bounded P500 pilot, and it never launches it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from compose_v4.data.editing_v2_process_v2_schema import (
    authority_false_block,
    canonical_sha256,
    require_authority_false,
    verify_self_hash,
)
from compose_v4.experiments.editing_v2_process_v2_p50_prerequisites import (
    ProcessV2P50ScopedPrerequisites,
)
from compose_v4.experiments.editing_v2_process_v2_contract_chain import (
    CHAIN_SCHEMA_VERSION,
)

RESULT_FILENAME = "PROCESS_V2_P50_RESULT.json"
DECISION_FILENAME = "PROCESS_V2_P50_DECISION.json"
RESULT_SCHEMA = "compose.editing_v2.process_v2_p50_result"
RESULT_SCHEMA_VERSION = 3
RESULT_STATUS = "COMPLETE_PROCESS_V2_P50_RESULT_NO_DOWNSTREAM_AUTHORITY"
DECISION_SCHEMA = "compose.editing_v2.process_v2_p50_decision"
DECISION_SCHEMA_VERSION = 1
DECISION_GO_STATUS = "GO_PROCESS_V2_BOUNDED_P500_P50"
DECISION_NO_GO_STATUS = "NO_GO_PROCESS_V2_BOUNDED_P500_P50"

_ENVIRONMENT_FIELDS = {
    "hardware_class",
    "device_name",
    "device_capability",
    "accelerator_class",
    "dtype",
    "mixed_precision",
    "batch_size",
    "python_version",
    "torch_version",
    "cuda_version",
    "cudnn_version",
    "rdkit_version",
    "numpy_version",
    "environment_sha256",
}
_PROVENANCE_FIELDS = {
    "p50_prerequisite_binding_sha256",
    "process_identity_sha256",
    "active8_completion_sha256",
    "gate_zero_decision_sha256",
    "t1_capacity_policy_sha256",
    "t1_result_sha256",
    "t1_decision_sha256",
    "t1_initial_model_state_sha256",
    "t1_score_revision_receipt_sha256",
    "p50_recipe_policy_sha256",
    "prepared_inputs_sha256",
    "training_stream_sha256",
    "validation_stream_sha256",
    "initial_model_state_sha256",
    "runner_implementation_sha256",
    "runner_source_revision_sha256",
    "collated_completion_sha256",
    "collated_payload_file_sha256",
    "execution_environment",
}
PROCESS_V2_P50_RUNTIME_PROVENANCE_FIELDS = frozenset(
    {
        "training_stream_sha256",
        "validation_stream_sha256",
        "prepared_inputs_sha256",
        "runner_implementation_sha256",
        "runner_source_revision_sha256",
        "collated_completion_sha256",
        "collated_payload_file_sha256",
    }
)
_RUN_INTEGRITY_FIELDS = {
    "optimizer_steps_completed",
    "scheduled_example_count",
    "batch_size",
    "initialization",
    "resume_requested",
    "resume_count",
    "abort_triggered",
    "abort_reasons",
    "nonfinite_event_count",
    "unsupported_teacher_count",
    "missing_candidate_count",
    "provenance_drift_detected",
    "stream_identity_drift_detected",
    "deterministic_algorithms_enabled",
    "dtype",
    "mixed_precision",
    "hazard_included",
    "t1_selected_checkpoint_loaded",
    "initial_model_state_sha256",
    "terminal_model_state_sha256",
    "optimizer_state_sha256",
    "terminal_checkpoint_file_sha256",
    "hazard_initial_state_sha256",
    "hazard_final_state_sha256",
}
_TRAJECTORY_FIELDS = {
    "optimizer_step",
    "mean_batch_canonical_successor_nll",
    "global_gradient_finite",
    "global_gradient_l2",
}
_VALIDATION_ENTRY_FIELDS = {
    "validation_entry_sha256",
    "family",
    "semantic_cell_id",
    "baseline_canonical_successor_nll",
    "final_canonical_successor_nll",
}
_EXPOSURE_FIELDS = {
    "planned_example_count",
    "observed_example_count",
    "planned_optimizer_step_opportunities",
    "finite_nonzero_global_gradient_exposure_steps",
    "all_exposure_step_global_gradients_finite",
    "cumulative_exposure_step_global_gradient_l2",
}
_FAMILY_ROUTE_EXPOSURE_FIELDS = {
    "finite_nonzero_action_route_gradient_exposure_steps",
    "all_exposure_step_action_route_gradients_finite",
    "cumulative_exposure_step_action_route_gradient_l2",
    "required_revision_parameter",
    "finite_nonzero_revision_parameter_gradient_exposure_steps",
    "all_exposure_step_revision_parameter_gradients_finite",
    "cumulative_exposure_step_revision_parameter_gradient_l2",
}
_RESULT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *authority_false_block(),
    "p500_authorized",
    "objective_unit",
    "scientific_scope",
    "learning_demonstrated",
    "scientific_interpretation",
    "provenance",
    "run_integrity",
    "trajectory",
    "required_cell_ids",
    "validation_unsupported_required_cells",
    "validation_entry_metrics",
    "validation_binding",
    "family_training_evidence",
    "semantic_cell_training_evidence",
    "family_validation_nll_checks",
    "semantic_cell_validation_nll_checks",
    "family_training_checks",
    "semantic_cell_training_checks",
    "threshold_checks",
    "result_sha256",
}


class ProcessV2P50ResultError(ValueError):
    """The P50 result or its bounded-P500 decision is invalid."""


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_mapping(value: object, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ProcessV2P50ResultError(f"{label} field set disagrees")
    return dict(value)


def _finite(value: object, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProcessV2P50ResultError(f"{label} must be finite")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ProcessV2P50ResultError(f"{label} is outside its finite range")
    return number


def _validate_policy(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...], dict[str, Any], dict[str, Any]]:
    policy = dict(value)
    try:
        verify_self_hash(policy, field="contract_sha256", label="the Process-V2 P50 policy")
        require_authority_false(policy, label="the Process-V2 P50 policy")
    except ValueError as error:
        raise ProcessV2P50ResultError(str(error)) from error
    optimization = policy.get("optimization")
    objective = policy.get("objective")
    thresholds = policy.get("thresholds")
    families = tuple(policy.get("active_families", ()))
    if (
        policy.get("schema") != "compose.editing_v2.process_v2_p50_recipe_policy"
        or policy.get("schema_version") != CHAIN_SCHEMA_VERSION
        or policy.get("status") != "FROZEN_PROCESS_V2_P50_RECIPE_POLICY_NO_DOWNSTREAM_AUTHORITY"
        or policy.get("p500_authorized") is not False
        or not isinstance(optimization, Mapping)
        or not isinstance(objective, Mapping)
        or not isinstance(thresholds, Mapping)
        or not families
        or len(families) != len(set(families))
    ):
        raise ProcessV2P50ResultError("Process-V2 P50 policy identity disagrees")
    expected_optimization = {
        "optimizer_steps": 50,
        "batch_size": 64,
        "scheduled_nonterminal_examples": 3200,
        "initialization": "scratch_from_t1_bound_initial_model_state",
        "resume": False,
        "dtype": "float32",
        "mixed_precision": False,
    }
    if any(optimization.get(name) != expected for name, expected in expected_optimization.items()):
        raise ProcessV2P50ResultError("Process-V2 P50 optimization policy disagrees")
    if (
        objective.get("unit") != "productive_embedded_canonical_successor"
        or objective.get("hazard_included") is not False
        or objective.get("hazard_weight") != 0.0
        or policy.get("scientific_scope")
        != "scratch_active8_stage_a_capability_pilot_not_production_law_calibration"
        or thresholds.get("baseline_partition_role") != "validation"
        or thresholds.get("baseline_values_inspected_when_thresholds_frozen") is not False
        or thresholds.get("zero_planned_family_or_cell_opportunities_allowed") is not False
    ):
        raise ProcessV2P50ResultError("Process-V2 P50 objective or threshold scope disagrees")
    for name in (
        "maximum_cell_final_minus_baseline_for_p50_nonincrease_nats",
        "maximum_cell_final_minus_baseline_successor_nll_nats",
        "maximum_family_final_minus_baseline_for_p50_nonincrease_nats",
        "maximum_family_final_minus_baseline_successor_nll_nats",
    ):
        _finite(thresholds.get(name), label=f"P50 threshold {name}", minimum=0.0)
    fraction = _finite(
        thresholds.get("gradient_opportunity_fraction"),
        label="P50 gradient opportunity fraction",
        minimum=0.0,
    )
    if fraction <= 0.0 or fraction > 1.0:
        raise ProcessV2P50ResultError("P50 gradient opportunity fraction must be in (0, 1]")
    return policy, tuple(str(item) for item in families), dict(optimization), dict(thresholds)


def _validate_required_cells(value: object, *, families: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProcessV2P50ResultError("P50 required cells must be a sequence")
    cells = tuple(value)
    if (
        len(cells) != 17
        or len(cells) != len(set(cells))
        or any(not isinstance(cell, str) or not cell for cell in cells)
        or any(not any(f":{family}:" in cell for family in families) for cell in cells)
        or any(sum(f":{family}:" in cell for family in families) != 1 for cell in cells)
        or any(not any(f":{family}:" in cell for cell in cells) for family in families)
    ):
        raise ProcessV2P50ResultError(
            "P50 required cells must be the ordered 17-cell Active8 editing partition"
        )
    return tuple(str(cell) for cell in cells)


def _validate_unsupported_validation_cells(
    value: object, *, required_cells: tuple[str, ...]
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProcessV2P50ResultError("unsupported validation cells must be a sequence")
    cells = tuple(value)
    if (
        len(cells) != len(set(cells))
        or any(cell not in required_cells for cell in cells)
        or cells != tuple(cell for cell in required_cells if cell in set(cells))
        or len(cells) == len(required_cells)
    ):
        raise ProcessV2P50ResultError(
            "unsupported validation cells must be an ordered proper subset of required cells"
        )
    return tuple(str(cell) for cell in cells)


def _validate_provenance(
    value: object,
    *,
    policy: Mapping[str, Any],
    prerequisites: ProcessV2P50ScopedPrerequisites,
    expected_runtime_provenance: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(prerequisites, ProcessV2P50ScopedPrerequisites):
        raise ProcessV2P50ResultError("P50 provenance requires validated Process-V2 prerequisites")
    provenance = _exact_mapping(value, _PROVENANCE_FIELDS, label="P50 provenance")
    environment = _exact_mapping(
        provenance["execution_environment"],
        _ENVIRONMENT_FIELDS,
        label="P50 execution environment",
    )
    environment_body = dict(environment)
    supplied_environment_sha = environment_body.pop("environment_sha256")
    if supplied_environment_sha != canonical_sha256(environment_body):
        raise ProcessV2P50ResultError("P50 execution environment hash disagrees")
    for name, item in provenance.items():
        if name != "execution_environment" and not _is_sha(item):
            raise ProcessV2P50ResultError(f"P50 provenance {name} is not a SHA-256")
    expected = {
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
    }
    if any(provenance[name] != expected_value for name, expected_value in expected.items()):
        raise ProcessV2P50ResultError("P50 provenance differs from its prerequisite chain")
    runtime = _exact_mapping(
        expected_runtime_provenance,
        set(PROCESS_V2_P50_RUNTIME_PROVENANCE_FIELDS),
        label="expected P50 runtime provenance",
    )
    if any(not _is_sha(item) for item in runtime.values()) or any(
        provenance[name] != item for name, item in runtime.items()
    ):
        raise ProcessV2P50ResultError("P50 runtime provenance disagrees")
    if (
        provenance["p50_recipe_policy_sha256"] != policy.get("contract_sha256")
        or provenance["initial_model_state_sha256"] != prerequisites.t1_initial_model_state_sha256
    ):
        raise ProcessV2P50ResultError("P50 scratch initialization provenance disagrees")
    return provenance


def _validate_integrity(
    value: object,
    *,
    optimization: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    integrity = _exact_mapping(value, _RUN_INTEGRITY_FIELDS, label="P50 run integrity")
    for name in (
        "initial_model_state_sha256",
        "terminal_model_state_sha256",
        "optimizer_state_sha256",
        "terminal_checkpoint_file_sha256",
        "hazard_initial_state_sha256",
        "hazard_final_state_sha256",
    ):
        if not _is_sha(integrity[name]):
            raise ProcessV2P50ResultError(f"P50 run integrity {name} is not a SHA-256")
    for name in (
        "optimizer_steps_completed",
        "scheduled_example_count",
        "batch_size",
        "resume_count",
        "nonfinite_event_count",
        "unsupported_teacher_count",
        "missing_candidate_count",
    ):
        if type(integrity[name]) is not int or integrity[name] < 0:
            raise ProcessV2P50ResultError(f"P50 run integrity {name} is not nonnegative")
    for name in (
        "resume_requested",
        "abort_triggered",
        "provenance_drift_detected",
        "stream_identity_drift_detected",
        "deterministic_algorithms_enabled",
        "mixed_precision",
        "hazard_included",
        "t1_selected_checkpoint_loaded",
    ):
        if type(integrity[name]) is not bool:
            raise ProcessV2P50ResultError(f"P50 run integrity {name} is not Boolean")
    if not isinstance(integrity["abort_reasons"], list) or any(
        not isinstance(reason, str) or not reason for reason in integrity["abort_reasons"]
    ):
        raise ProcessV2P50ResultError("P50 abort reasons are malformed")
    if (
        integrity["optimizer_steps_completed"] != optimization["optimizer_steps"]
        or integrity["scheduled_example_count"] != optimization["scheduled_nonterminal_examples"]
        or integrity["batch_size"] != optimization["batch_size"]
        or integrity["initialization"] != optimization["initialization"]
        or integrity["dtype"] != optimization["dtype"]
        or integrity["initial_model_state_sha256"] != provenance["initial_model_state_sha256"]
    ):
        raise ProcessV2P50ResultError("P50 exact 50-step run structure disagrees")
    return integrity


def _validate_trajectory(value: object, *, steps: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != steps:
        raise ProcessV2P50ResultError("P50 trajectory must contain exactly 50 rows")
    rows: list[dict[str, Any]] = []
    for expected_step, item in enumerate(value, start=1):
        row = _exact_mapping(item, _TRAJECTORY_FIELDS, label="P50 trajectory row")
        nll = _finite(
            row["mean_batch_canonical_successor_nll"],
            label="P50 batch successor NLL",
            minimum=0.0,
        )
        gradient = _finite(row["global_gradient_l2"], label="P50 global gradient", minimum=0.0)
        if (
            row["optimizer_step"] != expected_step
            or type(row["global_gradient_finite"]) is not bool
        ):
            raise ProcessV2P50ResultError("P50 trajectory step identity disagrees")
        rows.append(
            {
                "optimizer_step": expected_step,
                "mean_batch_canonical_successor_nll": nll,
                "global_gradient_finite": row["global_gradient_finite"],
                "global_gradient_l2": gradient,
            }
        )
    return rows


def _validate_validation_entries(
    value: object,
    *,
    families: tuple[str, ...],
    observable_cells: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ProcessV2P50ResultError("P50 validation entries are empty")
    rows: list[dict[str, Any]] = []
    identifiers: list[str] = []
    for item in value:
        row = _exact_mapping(item, _VALIDATION_ENTRY_FIELDS, label="P50 validation entry")
        baseline = _finite(
            row["baseline_canonical_successor_nll"],
            label="P50 baseline successor NLL",
            minimum=0.0,
        )
        final = _finite(
            row["final_canonical_successor_nll"],
            label="P50 final successor NLL",
            minimum=0.0,
        )
        if (
            not _is_sha(row["validation_entry_sha256"])
            or row["family"] not in families
            or row["semantic_cell_id"] not in observable_cells
            or f":{row['family']}:" not in row["semantic_cell_id"]
        ):
            raise ProcessV2P50ResultError("P50 validation entry identity disagrees")
        rows.append(
            {
                **row,
                "baseline_canonical_successor_nll": baseline,
                "final_canonical_successor_nll": final,
            }
        )
        identifiers.append(row["validation_entry_sha256"])
    expected = sorted(
        rows,
        key=lambda row: (
            families.index(row["family"]),
            observable_cells.index(row["semantic_cell_id"]),
            row["validation_entry_sha256"],
        ),
    )
    if (
        rows != expected
        or len(identifiers) != len(set(identifiers))
        or {row["family"] for row in rows} != set(families)
        or {row["semantic_cell_id"] for row in rows} != set(observable_cells)
    ):
        raise ProcessV2P50ResultError("P50 validation coverage or order disagrees")
    return rows


def _validate_exposure(
    value: object,
    *,
    identities: tuple[str, ...],
    identity_field: str,
    scheduled_examples: int,
    require_family_routes: bool = False,
) -> list[dict[str, Any]]:
    """Validate cheap group exposure, never isolated per-group backward passes.

    A group receives gradient exposure when it is present with positive
    coefficient in an optimizer step whose one ordinary global optimizer
    gradient is finite and nonzero.  T1 separately establishes route
    connectivity, so P50 does not repeat family/cell VJPs.
    """

    if not isinstance(value, list) or [
        row.get(identity_field) for row in value if isinstance(row, Mapping)
    ] != list(identities):
        raise ProcessV2P50ResultError(f"P50 {identity_field} evidence coverage disagrees")
    rows: list[dict[str, Any]] = []
    for item in value:
        expected_fields = {identity_field, *_EXPOSURE_FIELDS}
        if require_family_routes:
            expected_fields.update(_FAMILY_ROUTE_EXPOSURE_FIELDS)
        row = _exact_mapping(item, expected_fields, label=f"P50 {identity_field} evidence")
        for name in (
            "planned_example_count",
            "observed_example_count",
            "planned_optimizer_step_opportunities",
            "finite_nonzero_global_gradient_exposure_steps",
        ):
            if type(row[name]) is not int or row[name] < 0:
                raise ProcessV2P50ResultError(f"P50 exposure {name} is not nonnegative")
        gradient = _finite(
            row["cumulative_exposure_step_global_gradient_l2"],
            label="P50 cumulative exposure-step global gradient",
            minimum=0.0,
        )
        if (
            row["planned_example_count"] <= 0
            or not 1 <= row["planned_optimizer_step_opportunities"] <= 50
            or row["finite_nonzero_global_gradient_exposure_steps"] > 50
            or type(row["all_exposure_step_global_gradients_finite"]) is not bool
        ):
            raise ProcessV2P50ResultError("P50 exposure evidence range disagrees")
        if require_family_routes:
            for name in (
                "finite_nonzero_action_route_gradient_exposure_steps",
                "finite_nonzero_revision_parameter_gradient_exposure_steps",
            ):
                if type(row[name]) is not int or not 0 <= row[name] <= 50:
                    raise ProcessV2P50ResultError(
                        f"P50 family-route exposure {name} is outside range"
                    )
            for name in (
                "all_exposure_step_action_route_gradients_finite",
                "all_exposure_step_revision_parameter_gradients_finite",
            ):
                if type(row[name]) is not bool:
                    raise ProcessV2P50ResultError(
                        f"P50 family-route exposure {name} is not Boolean"
                    )
            action_l2 = _finite(
                row["cumulative_exposure_step_action_route_gradient_l2"],
                label="P50 action-route cumulative gradient",
                minimum=0.0,
            )
            revision_l2 = _finite(
                row["cumulative_exposure_step_revision_parameter_gradient_l2"],
                label="P50 revision-parameter cumulative gradient",
                minimum=0.0,
            )
            expected_revision = {
                "bond_reroute": "graft_relation_head.weight",
                "ring_system_restate": "ring_restate_context_head.weight",
            }.get(str(row[identity_field]))
            if (
                row["required_revision_parameter"] != expected_revision
                or action_l2 != row[
                    "cumulative_exposure_step_action_route_gradient_l2"
                ]
                or revision_l2
                != row["cumulative_exposure_step_revision_parameter_gradient_l2"]
                or (
                    expected_revision is None
                    and (
                        row[
                            "finite_nonzero_revision_parameter_gradient_exposure_steps"
                        ]
                        != 0
                        or revision_l2 != 0.0
                    )
                )
            ):
                raise ProcessV2P50ResultError(
                    "P50 scorer-revision gradient evidence disagrees"
                )
        rows.append({**row, "cumulative_exposure_step_global_gradient_l2": gradient})
    if sum(row["planned_example_count"] for row in rows) != scheduled_examples:
        raise ProcessV2P50ResultError("P50 planned exposure does not sum to 3,200")
    return rows


def _validation_checks(
    rows: Sequence[Mapping[str, Any]],
    *,
    identities: tuple[str, ...],
    identity_field: str,
    nonincrease_ceiling: float,
    catastrophic_ceiling: float,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for identity in identities:
        selected = [row for row in rows if row[identity_field] == identity]
        baseline = sum(float(row["baseline_canonical_successor_nll"]) for row in selected) / len(
            selected
        )
        final = sum(float(row["final_canonical_successor_nll"]) for row in selected) / len(selected)
        delta = final - baseline
        checks.append(
            {
                identity_field: identity,
                "entry_count": len(selected),
                "baseline_mean_canonical_successor_nll": baseline,
                "final_mean_canonical_successor_nll": final,
                "final_minus_baseline_nats": delta,
                "nonincrease_ceiling_nats": nonincrease_ceiling,
                "catastrophic_regression_ceiling_nats": catastrophic_ceiling,
                "nonincrease_passed": delta <= nonincrease_ceiling,
                "catastrophic_regression_passed": delta <= catastrophic_ceiling,
            }
        )
    return checks


def _exposure_checks(
    rows: Sequence[Mapping[str, Any]],
    *,
    identity_field: str,
    fraction: float,
    require_family_routes: bool = False,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for row in rows:
        minimum = max(1, math.ceil(fraction * row["planned_optimizer_step_opportunities"]))
        action_route_passed = (
            not require_family_routes
            or row["all_exposure_step_action_route_gradients_finite"] is True
            and row["cumulative_exposure_step_action_route_gradient_l2"] > 0.0
            and row["finite_nonzero_action_route_gradient_exposure_steps"] >= minimum
        )
        revision_required = (
            require_family_routes and row["required_revision_parameter"] is not None
        )
        revision_passed = (
            not revision_required
            or row["all_exposure_step_revision_parameter_gradients_finite"] is True
            and row["cumulative_exposure_step_revision_parameter_gradient_l2"] > 0.0
            and row["finite_nonzero_revision_parameter_gradient_exposure_steps"]
            >= minimum
        )
        global_passed = (
            row["all_exposure_step_global_gradients_finite"] is True
            and row["cumulative_exposure_step_global_gradient_l2"] > 0.0
            and row["finite_nonzero_global_gradient_exposure_steps"] >= minimum
        )
        checks.append(
            {
                identity_field: row[identity_field],
                "minimum_required_finite_nonzero_global_gradient_exposure_steps": minimum,
                "complete_example_exposure": row["observed_example_count"]
                == row["planned_example_count"],
                "finite_nonzero_global_gradient_exposure": global_passed,
                "finite_nonzero_action_route_gradient_exposure": action_route_passed,
                "required_revision_parameter_gradient_exposure": revision_passed,
                "passed": row["observed_example_count"] == row["planned_example_count"]
                and global_passed
                and action_route_passed
                and revision_passed,
            }
        )
    return checks


def _derived_sections(
    *,
    integrity: Mapping[str, Any],
    trajectory: Sequence[Mapping[str, Any]],
    validation_entries: Sequence[Mapping[str, Any]],
    family_exposure: Sequence[Mapping[str, Any]],
    cell_exposure: Sequence[Mapping[str, Any]],
    families: tuple[str, ...],
    cells: tuple[str, ...],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    family_validation = _validation_checks(
        validation_entries,
        identities=families,
        identity_field="family",
        nonincrease_ceiling=float(
            thresholds["maximum_family_final_minus_baseline_for_p50_nonincrease_nats"]
        ),
        catastrophic_ceiling=float(
            thresholds["maximum_family_final_minus_baseline_successor_nll_nats"]
        ),
    )
    cell_validation = _validation_checks(
        validation_entries,
        identities=cells,
        identity_field="semantic_cell_id",
        nonincrease_ceiling=float(
            thresholds["maximum_cell_final_minus_baseline_for_p50_nonincrease_nats"]
        ),
        catastrophic_ceiling=float(
            thresholds["maximum_cell_final_minus_baseline_successor_nll_nats"]
        ),
    )
    fraction = float(thresholds["gradient_opportunity_fraction"])
    family_training = _exposure_checks(
        family_exposure,
        identity_field="family",
        fraction=fraction,
        require_family_routes=True,
    )
    cell_training = _exposure_checks(
        cell_exposure, identity_field="semantic_cell_id", fraction=fraction
    )
    integrity_passed = (
        integrity["resume_requested"] is False
        and integrity["resume_count"] == 0
        and integrity["abort_triggered"] is False
        and integrity["abort_reasons"] == []
        and integrity["nonfinite_event_count"] == 0
        and integrity["unsupported_teacher_count"] == 0
        and integrity["missing_candidate_count"] == 0
        and integrity["provenance_drift_detected"] is False
        and integrity["stream_identity_drift_detected"] is False
        and integrity["deterministic_algorithms_enabled"] is True
        and integrity["dtype"] == "float32"
        and integrity["mixed_precision"] is False
        and integrity["t1_selected_checkpoint_loaded"] is False
    )
    threshold_checks = {
        "exact_50_step_scratch_run": integrity_passed,
        "hazard_frozen_excluded_and_unchanged": integrity["hazard_included"] is False
        and integrity["hazard_initial_state_sha256"] == integrity["hazard_final_state_sha256"],
        "trajectory_finite_nonzero_gradients": all(
            row["global_gradient_finite"] is True and row["global_gradient_l2"] > 0.0
            for row in trajectory
        ),
        "family_validation_nonincrease": all(
            row["nonincrease_passed"] is True for row in family_validation
        ),
        "family_validation_no_catastrophic_regression": all(
            row["catastrophic_regression_passed"] is True for row in family_validation
        ),
        "semantic_cell_validation_nonincrease": all(
            row["nonincrease_passed"] is True for row in cell_validation
        ),
        "semantic_cell_validation_no_catastrophic_regression": all(
            row["catastrophic_regression_passed"] is True for row in cell_validation
        ),
        "family_training_exposure_and_gradient": all(
            row["passed"] is True for row in family_training
        ),
        "semantic_cell_training_exposure_and_gradient": all(
            row["passed"] is True for row in cell_training
        ),
    }
    return {
        "family_validation_nll_checks": family_validation,
        "semantic_cell_validation_nll_checks": cell_validation,
        "family_training_checks": family_training,
        "semantic_cell_training_checks": cell_training,
        "threshold_checks": threshold_checks,
    }


def validate_process_v2_p50_result(
    value: object,
    *,
    recipe_policy: Mapping[str, Any],
    prerequisites: ProcessV2P50ScopedPrerequisites,
    required_cell_ids: Sequence[str],
    validation_unsupported_required_cells: Sequence[str],
    expected_runtime_provenance: Mapping[str, str],
) -> dict[str, Any]:
    """Recompute the exact bounded P50 result and every frozen threshold."""

    result = _exact_mapping(value, _RESULT_FIELDS, label="Process-V2 P50 result")
    try:
        verify_self_hash(result, field="result_sha256", label="the Process-V2 P50 result")
        require_authority_false(result, label="the Process-V2 P50 result")
    except ValueError as error:
        raise ProcessV2P50ResultError(str(error)) from error
    policy, families, optimization, thresholds = _validate_policy(recipe_policy)
    cells = _validate_required_cells(required_cell_ids, families=families)
    unsupported_cells = _validate_unsupported_validation_cells(
        validation_unsupported_required_cells, required_cells=cells
    )
    observable_cells = tuple(cell for cell in cells if cell not in set(unsupported_cells))
    if (
        result["schema"] != RESULT_SCHEMA
        or result["schema_version"] != RESULT_SCHEMA_VERSION
        or result["status"] != RESULT_STATUS
        or result["p500_authorized"] is not False
        or result["objective_unit"] != "productive_embedded_canonical_successor"
        or result["scientific_scope"]
        != "scratch_active8_stage_a_capability_pilot_not_production_law_calibration"
        or result["learning_demonstrated"] is not False
        or result["scientific_interpretation"]
        != "P50_NONREGRESSION_AND_GRADIENT_EXPOSURE_DO_NOT_ESTABLISH_FINAL_MODEL_QUALITY"
        or result["required_cell_ids"] != list(cells)
        or result["validation_unsupported_required_cells"] != list(unsupported_cells)
    ):
        raise ProcessV2P50ResultError("Process-V2 P50 result identity disagrees")
    provenance = _validate_provenance(
        result["provenance"],
        policy=policy,
        prerequisites=prerequisites,
        expected_runtime_provenance=expected_runtime_provenance,
    )
    integrity = _validate_integrity(
        result["run_integrity"], optimization=optimization, provenance=provenance
    )
    trajectory = _validate_trajectory(result["trajectory"], steps=optimization["optimizer_steps"])
    validation = _validate_validation_entries(
        result["validation_entry_metrics"],
        families=families,
        observable_cells=observable_cells,
    )
    expected_binding = {
        "partition_role": "validation",
        "entry_count": len(validation),
        "entry_inventory_sha256": canonical_sha256(
            sorted(row["validation_entry_sha256"] for row in validation)
        ),
        "entry_metadata_sha256": canonical_sha256(
            [
                {
                    "validation_entry_sha256": row["validation_entry_sha256"],
                    "family": row["family"],
                    "semantic_cell_id": row["semantic_cell_id"],
                }
                for row in validation
            ]
        ),
        "observable_required_cell_ids": list(observable_cells),
        "unsupported_required_cell_ids": list(unsupported_cells),
    }
    if result["validation_binding"] != expected_binding:
        raise ProcessV2P50ResultError("P50 validation binding disagrees")
    family_exposure = _validate_exposure(
        result["family_training_evidence"],
        identities=families,
        identity_field="family",
        scheduled_examples=optimization["scheduled_nonterminal_examples"],
        require_family_routes=True,
    )
    cell_exposure = _validate_exposure(
        result["semantic_cell_training_evidence"],
        identities=cells,
        identity_field="semantic_cell_id",
        scheduled_examples=optimization["scheduled_nonterminal_examples"],
    )
    expected_derived = _derived_sections(
        integrity=integrity,
        trajectory=trajectory,
        validation_entries=validation,
        family_exposure=family_exposure,
        cell_exposure=cell_exposure,
        families=families,
        cells=observable_cells,
        thresholds=thresholds,
    )
    for name, expected in expected_derived.items():
        if result[name] != expected:
            raise ProcessV2P50ResultError(f"P50 derived {name} disagrees")
    return result


def build_process_v2_p50_result(
    *,
    recipe_policy: Mapping[str, Any],
    prerequisites: ProcessV2P50ScopedPrerequisites,
    required_cell_ids: Sequence[str],
    validation_unsupported_required_cells: Sequence[str],
    expected_runtime_provenance: Mapping[str, str],
    provenance: Mapping[str, Any],
    run_integrity: Mapping[str, Any],
    trajectory: Sequence[Mapping[str, Any]],
    validation_entry_metrics: Sequence[Mapping[str, Any]],
    family_training_evidence: Sequence[Mapping[str, Any]],
    semantic_cell_training_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a non-authorizing P50 result from measured runtime evidence."""

    policy, families, optimization, thresholds = _validate_policy(recipe_policy)
    cells = _validate_required_cells(required_cell_ids, families=families)
    unsupported_cells = _validate_unsupported_validation_cells(
        validation_unsupported_required_cells, required_cells=cells
    )
    observable_cells = tuple(cell for cell in cells if cell not in set(unsupported_cells))
    validated_provenance = _validate_provenance(
        provenance,
        policy=policy,
        prerequisites=prerequisites,
        expected_runtime_provenance=expected_runtime_provenance,
    )
    integrity = _validate_integrity(
        run_integrity, optimization=optimization, provenance=validated_provenance
    )
    trajectory_rows = _validate_trajectory(list(trajectory), steps=optimization["optimizer_steps"])
    validation = _validate_validation_entries(
        list(validation_entry_metrics),
        families=families,
        observable_cells=observable_cells,
    )
    family_exposure = _validate_exposure(
        list(family_training_evidence),
        identities=families,
        identity_field="family",
        scheduled_examples=optimization["scheduled_nonterminal_examples"],
        require_family_routes=True,
    )
    cell_exposure = _validate_exposure(
        list(semantic_cell_training_evidence),
        identities=cells,
        identity_field="semantic_cell_id",
        scheduled_examples=optimization["scheduled_nonterminal_examples"],
    )
    derived = _derived_sections(
        integrity=integrity,
        trajectory=trajectory_rows,
        validation_entries=validation,
        family_exposure=family_exposure,
        cell_exposure=cell_exposure,
        families=families,
        cells=observable_cells,
        thresholds=thresholds,
    )
    authority = authority_false_block()
    body = {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": RESULT_STATUS,
        **authority,
        "p500_authorized": False,
        "objective_unit": "productive_embedded_canonical_successor",
        "scientific_scope": (
            "scratch_active8_stage_a_capability_pilot_not_production_law_calibration"
        ),
        "learning_demonstrated": False,
        "scientific_interpretation": (
            "P50_NONREGRESSION_AND_GRADIENT_EXPOSURE_DO_NOT_ESTABLISH_FINAL_MODEL_QUALITY"
        ),
        "provenance": validated_provenance,
        "run_integrity": integrity,
        "trajectory": trajectory_rows,
        "required_cell_ids": list(cells),
        "validation_unsupported_required_cells": list(unsupported_cells),
        "validation_entry_metrics": validation,
        "validation_binding": {
            "partition_role": "validation",
            "entry_count": len(validation),
            "entry_inventory_sha256": canonical_sha256(
                sorted(row["validation_entry_sha256"] for row in validation)
            ),
            "entry_metadata_sha256": canonical_sha256(
                [
                    {
                        "validation_entry_sha256": row["validation_entry_sha256"],
                        "family": row["family"],
                        "semantic_cell_id": row["semantic_cell_id"],
                    }
                    for row in validation
                ]
            ),
            "observable_required_cell_ids": list(observable_cells),
            "unsupported_required_cell_ids": list(unsupported_cells),
        },
        "family_training_evidence": family_exposure,
        "semantic_cell_training_evidence": cell_exposure,
        **derived,
    }
    result = {**body, "result_sha256": canonical_sha256(body)}
    return validate_process_v2_p50_result(
        result,
        recipe_policy=policy,
        prerequisites=prerequisites,
        required_cell_ids=cells,
        validation_unsupported_required_cells=unsupported_cells,
        expected_runtime_provenance=expected_runtime_provenance,
    )


def _decision_failures(result: Mapping[str, Any]) -> list[str]:
    return sorted(
        f"threshold:{name}"
        for name, passed in result["threshold_checks"].items()
        if passed is not True
    )


def build_process_v2_p50_decision(
    result: Mapping[str, Any],
    *,
    recipe_policy: Mapping[str, Any],
    prerequisites: ProcessV2P50ScopedPrerequisites,
    required_cell_ids: Sequence[str],
    validation_unsupported_required_cells: Sequence[str],
    expected_runtime_provenance: Mapping[str, str],
    result_file_sha256: str,
) -> dict[str, Any]:
    """Recompute the sole bounded-P500 authorization from one P50 result."""

    validated = validate_process_v2_p50_result(
        result,
        recipe_policy=recipe_policy,
        prerequisites=prerequisites,
        required_cell_ids=required_cell_ids,
        validation_unsupported_required_cells=validation_unsupported_required_cells,
        expected_runtime_provenance=expected_runtime_provenance,
    )
    if not _is_sha(result_file_sha256):
        raise ProcessV2P50ResultError("P50 result file hash is invalid")
    failures = _decision_failures(validated)
    passed = not failures
    provenance = validated["provenance"]
    integrity = validated["run_integrity"]
    body = {
        "schema": DECISION_SCHEMA,
        "schema_version": DECISION_SCHEMA_VERSION,
        "status": DECISION_GO_STATUS if passed else DECISION_NO_GO_STATUS,
        **authority_false_block(),
        "p500_authorized": passed,
        "p50_recipe_policy_sha256": provenance["p50_recipe_policy_sha256"],
        "p50_prerequisite_binding_sha256": provenance["p50_prerequisite_binding_sha256"],
        "process_identity_sha256": provenance["process_identity_sha256"],
        "active8_completion_sha256": provenance["active8_completion_sha256"],
        "gate_zero_decision_sha256": provenance["gate_zero_decision_sha256"],
        "t1_result_sha256": provenance["t1_result_sha256"],
        "t1_decision_sha256": provenance["t1_decision_sha256"],
        "training_stream_sha256": provenance["training_stream_sha256"],
        "validation_stream_sha256": provenance["validation_stream_sha256"],
        "result_file_sha256": result_file_sha256,
        "result_sha256": validated["result_sha256"],
        "terminal_checkpoint_file_sha256": integrity["terminal_checkpoint_file_sha256"],
        "terminal_model_state_sha256": integrity["terminal_model_state_sha256"],
        "optimizer_steps_completed": integrity["optimizer_steps_completed"],
        "required_families": list(prerequisites.active_families),
        "required_cell_ids": list(validated["required_cell_ids"]),
        "validation_unsupported_required_cells": list(
            validated["validation_unsupported_required_cells"]
        ),
        "failed_checks": failures,
    }
    return {**body, "decision_sha256": canonical_sha256(body)}


def validate_process_v2_p50_decision(
    value: object,
    *,
    result: Mapping[str, Any],
    recipe_policy: Mapping[str, Any],
    prerequisites: ProcessV2P50ScopedPrerequisites,
    required_cell_ids: Sequence[str],
    validation_unsupported_required_cells: Sequence[str],
    expected_runtime_provenance: Mapping[str, str],
    result_file_sha256: str,
    require_p500_go: bool = False,
) -> dict[str, Any]:
    """Recompute a P50 decision and optionally require its bounded-P500 GO."""

    if not isinstance(value, Mapping):
        raise ProcessV2P50ResultError("Process-V2 P50 decision must be an object")
    expected = build_process_v2_p50_decision(
        result,
        recipe_policy=recipe_policy,
        prerequisites=prerequisites,
        required_cell_ids=required_cell_ids,
        validation_unsupported_required_cells=validation_unsupported_required_cells,
        expected_runtime_provenance=expected_runtime_provenance,
        result_file_sha256=result_file_sha256,
    )
    decision = dict(value)
    if decision != expected:
        raise ProcessV2P50ResultError("Process-V2 P50 decision differs from recomputation")
    try:
        require_authority_false(decision, label="the Process-V2 P50 decision")
    except ValueError as error:
        raise ProcessV2P50ResultError(str(error)) from error
    if require_p500_go and (
        decision["status"] != DECISION_GO_STATUS
        or decision["p500_authorized"] is not True
        or decision["failed_checks"] != []
    ):
        raise ProcessV2P50ResultError("Process-V2 P50 decision is not a bounded-P500 GO")
    return decision


__all__ = [
    "DECISION_FILENAME",
    "DECISION_GO_STATUS",
    "DECISION_NO_GO_STATUS",
    "RESULT_FILENAME",
    "RESULT_SCHEMA",
    "RESULT_SCHEMA_VERSION",
    "PROCESS_V2_P50_RUNTIME_PROVENANCE_FIELDS",
    "ProcessV2P50ResultError",
    "build_process_v2_p50_decision",
    "build_process_v2_p50_result",
    "validate_process_v2_p50_decision",
    "validate_process_v2_p50_result",
]
