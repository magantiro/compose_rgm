"""Fail-closed prospective policy for semantic unique-state T1 capacity."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from compose_v4.experiments.successor_micro_overfit import (
    RINGCORE_EDITING_FAMILIES,
)

SCHEMA = "compose.editing_v2.semantic_t1_capacity_policy"
SCHEMA_VERSION = 1
STATUS = "FROZEN_PROSPECTIVE_UNIQUE_STATE_CAPACITY_NO_DOWNSTREAM_AUTHORITY"
PANEL_KIND = "unique_state_single_target_canonical_successor_capacity"
OBJECTIVE_UNIT = "exact_source_frozen_time_canonical_successor"
EMPIRICAL_STATUS = "BLOCKED_PENDING_VERIFIED_INDEPENDENT_OBSERVATION_RECEIPTS"
EMPIRICAL_RECEIPT_KIND = "independent_empirical_transition_v1"
EXPECTED_PANEL_POLICY_SHA256 = "ac7e6753f27493c9f221d4ff41525c9606910edb387b73baca59879b873ee770"
EXPECTED_CELL_ROLE_POLICY_SHA256 = (
    "a26c7084b2cf6f23a7690f5dc5f76f992056b205c0dc189405643a22a90b603c"
)
EXPECTED_PANEL_MINIMUM = 64
EXPECTED_PANEL_MAXIMUM = 128
EXPECTED_REPORT_POINTS = (1, 10, 50, 100, 250, 500)
EXPECTED_SAMPLING_ORDER = (
    "model_family",
    "semantic_capability_cell",
    "unique_panel_entry",
)
EXPECTED_FAILURE_DIAGNOSTIC_SCOPE_ORDER = (
    "heads_only",
    "heads_plus_local_adapter_if_distinct",
    "all",
)


class SemanticT1CapacityPolicyError(ValueError):
    """The prospective semantic T1 capacity policy is malformed or stale."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SemanticT1CapacityPolicyError(
            "semantic T1 capacity policy is not finite canonical JSON"
        ) from error


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _exact_mapping(value: object, expected: set[str], *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SemanticT1CapacityPolicyError(f"{field} has missing or unknown fields")
    return value


def validate_semantic_t1_capacity_policy(
    value: object,
) -> dict[str, Any]:
    """Validate one nonauthorizing policy without resolving any result."""

    expected_fields = {
        "schema",
        "schema_version",
        "status",
        "policy_id",
        "policy_sha256",
        "training_authorized",
        "bounded_p50_authorized",
        "long_training_authorized",
        "checkpoint_selection_authorized",
        "final_test_selection_authorized",
        "hazard_included",
        "panel_kind",
        "panel_policy_sha256",
        "objective_unit",
        "support_time_hex",
        "required_families",
        "cell_role_policy_sha256",
        "panel_cardinality",
        "sampling_law",
        "optimization",
        "thresholds",
        "empirical_repeated_state_gate",
    }
    policy = dict(_exact_mapping(value, expected_fields, field="capacity policy"))
    body = dict(policy)
    supplied_sha256 = body.pop("policy_sha256")
    if (
        policy["schema"] != SCHEMA
        or policy["schema_version"] != SCHEMA_VERSION
        or policy["status"] != STATUS
        or supplied_sha256 != _sha256(body)
        or policy["panel_kind"] != PANEL_KIND
        or policy["objective_unit"] != OBJECTIVE_UNIT
        or policy["support_time_hex"] != float(0.5).hex()
        or policy["panel_policy_sha256"] != EXPECTED_PANEL_POLICY_SHA256
        or policy["cell_role_policy_sha256"] != EXPECTED_CELL_ROLE_POLICY_SHA256
        or tuple(policy["required_families"]) != RINGCORE_EDITING_FAMILIES
        or any(
            policy[field] is not False
            for field in (
                "training_authorized",
                "bounded_p50_authorized",
                "long_training_authorized",
                "checkpoint_selection_authorized",
                "final_test_selection_authorized",
                "hazard_included",
            )
        )
    ):
        raise SemanticT1CapacityPolicyError(
            "semantic T1 capacity identity, scope, or authority disagrees"
        )
    cardinality = _exact_mapping(
        policy["panel_cardinality"],
        {"minimum_entries_by_family", "maximum_entries_by_family"},
        field="panel_cardinality",
    )
    expected_minimums = {family: EXPECTED_PANEL_MINIMUM for family in RINGCORE_EDITING_FAMILIES}
    expected_maximums = {family: EXPECTED_PANEL_MAXIMUM for family in RINGCORE_EDITING_FAMILIES}
    if (
        cardinality["minimum_entries_by_family"] != expected_minimums
        or cardinality["maximum_entries_by_family"] != expected_maximums
    ):
        raise SemanticT1CapacityPolicyError(
            "semantic T1 panel cardinality must freeze 64 to 128 entries per family"
        )

    sampling = _exact_mapping(
        policy["sampling_law"],
        {
            "order",
            "probability_within_each_level",
            "target_coefficient",
            "importance_correction",
        },
        field="sampling_law",
    )
    if (
        tuple(sampling["order"]) != EXPECTED_SAMPLING_ORDER
        or sampling["probability_within_each_level"] != "uniform_over_nonempty_children"
        or sampling["target_coefficient"] != 1.0
        or sampling["importance_correction"] != "none"
    ):
        raise SemanticT1CapacityPolicyError("semantic T1 capacity sampling law disagrees")

    optimization = _exact_mapping(
        policy["optimization"],
        {
            "accelerator_class",
            "address_stream",
            "checkpoint_selection",
            "deterministic_algorithms_required",
            "dtype",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "batch_size",
            "maximum_optimizer_steps",
            "report_points",
            "scheduler",
            "seed",
            "mixed_precision",
            "model_initialization",
            "parameter_scope",
            "failure_diagnostic_scope_order",
            "failure_diagnostics_only_for_failing_families",
            "trajectory_evaluation",
            "early_stop_rule",
        },
        field="optimization",
    )
    if (
        optimization["accelerator_class"] != "gpu"
        or optimization["address_stream"]
        != "sha256_counter_stream_bound_to_policy_and_panel_identity"
        or optimization["checkpoint_selection"]
        != "maximize_minimum_entry_teacher_successor_probability__tie_lower_mean_nll__tie_earlier_step"
        or optimization["deterministic_algorithms_required"] is not True
        or optimization["dtype"] != "float32"
        or optimization["optimizer"] != "adamw"
        or optimization["learning_rate"] != 0.001
        or optimization["weight_decay"] != 0.0
        or optimization["gradient_clip_norm"] != 10.0
        or optimization["batch_size"] != 64
        or optimization["maximum_optimizer_steps"] != 500
        or tuple(optimization["report_points"]) != EXPECTED_REPORT_POINTS
        or optimization["scheduler"] != "constant"
        or optimization["seed"] != 31
        or optimization["mixed_precision"] is not False
        or optimization["model_initialization"] != "scratch"
        or optimization["parameter_scope"] != "all_trainable_active8_parameters"
        or tuple(optimization["failure_diagnostic_scope_order"])
        != EXPECTED_FAILURE_DIAGNOSTIC_SCOPE_ORDER
        or optimization["failure_diagnostics_only_for_failing_families"] is not True
        or optimization["trajectory_evaluation"] != "every_pre_update_state_and_terminal_state"
        or optimization["early_stop_rule"]
        != "all_required_family_nonempty_cell_and_entry_thresholds_pass_at_one_evaluated_state"
    ):
        raise SemanticT1CapacityPolicyError("semantic T1 capacity optimization policy disagrees")

    thresholds = _exact_mapping(
        policy["thresholds"],
        {
            "minimum_unique_state_teacher_successor_top1",
            "minimum_unique_state_teacher_successor_probability",
            "maximum_unique_state_teacher_successor_nll",
            "require_every_unique_entry_teacher_successor_top1",
            "minimum_every_unique_entry_teacher_successor_probability",
            "minimum_nonempty_cell_teacher_successor_top1",
            "minimum_nonempty_cell_teacher_successor_probability",
            "maximum_nonempty_cell_teacher_successor_nll",
            "require_finite_nonzero_family_gate_gradient_each_family",
            "require_finite_nonzero_action_route_gradient_each_family",
        },
        field="thresholds",
    )
    probability = thresholds["minimum_unique_state_teacher_successor_probability"]
    if (
        thresholds["minimum_unique_state_teacher_successor_top1"] != 0.95
        or probability != 0.8
        or not math.isclose(
            thresholds["maximum_unique_state_teacher_successor_nll"],
            -math.log(probability),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or thresholds["require_every_unique_entry_teacher_successor_top1"] is not True
        or thresholds["minimum_every_unique_entry_teacher_successor_probability"] != probability
        or thresholds["minimum_nonempty_cell_teacher_successor_top1"] != 0.95
        or thresholds["minimum_nonempty_cell_teacher_successor_probability"] != probability
        or thresholds["maximum_nonempty_cell_teacher_successor_nll"]
        != thresholds["maximum_unique_state_teacher_successor_nll"]
        or thresholds["require_finite_nonzero_family_gate_gradient_each_family"] is not True
        or thresholds["require_finite_nonzero_action_route_gradient_each_family"] is not True
    ):
        raise SemanticT1CapacityPolicyError("semantic T1 capacity thresholds disagree")

    empirical = _exact_mapping(
        policy["empirical_repeated_state_gate"],
        {
            "status",
            "required_receipt_kind",
            "raw_record_multiplicity_is_observation_count",
            "mark_alias_multiplicity_is_observation_count",
            "bounded_p50_capacity_prerequisite",
            "empirical_law_claims_authorized",
        },
        field="empirical_repeated_state_gate",
    )
    if (
        empirical["status"] != EMPIRICAL_STATUS
        or empirical["required_receipt_kind"] != EMPIRICAL_RECEIPT_KIND
        or any(
            empirical[field] is not False
            for field in (
                "raw_record_multiplicity_is_observation_count",
                "mark_alias_multiplicity_is_observation_count",
                "bounded_p50_capacity_prerequisite",
                "empirical_law_claims_authorized",
            )
        )
    ):
        raise SemanticT1CapacityPolicyError("semantic T1 empirical-law boundary disagrees")
    return policy


def load_semantic_t1_capacity_policy(path: str | Path) -> dict[str, Any]:
    """Load and validate one prospective semantic capacity policy."""

    try:
        payload = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticT1CapacityPolicyError(
            f"cannot load semantic T1 capacity policy: {path}"
        ) from error
    return validate_semantic_t1_capacity_policy(payload)


__all__ = [
    "SemanticT1CapacityPolicyError",
    "load_semantic_t1_capacity_policy",
    "validate_semantic_t1_capacity_policy",
]
