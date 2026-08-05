"""Process-V2 T1 result artifact over the bounded canonical-successor panel.

The optimizer is shared with the legacy semantic T1 implementation, but its
physical inputs and provenance are not.  This module is the narrow adapter
that records Process-V2 inputs without inventing legacy cache or Gate-0
aliases.  It deliberately contains no training loop and grants no authority.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from compose_v4.data.editing_v2_process_v2_schema import (
    authority_false_block,
    canonical_sha256,
    require_authority_false,
    verify_self_hash,
)

RESULT_FILENAME = "PROCESS_V2_T1_CAPACITY_RESULT.json"
DECISION_FILENAME = "PROCESS_V2_T1_CAPACITY_DECISION.json"
RESULT_SCHEMA = "compose.editing_v2.process_v2_t1_capacity_result"
RESULT_SCHEMA_VERSION = 1
RESULT_STATUS = "COMPLETE_PROCESS_V2_T1_CAPACITY_RESULT_NO_DOWNSTREAM_AUTHORITY"
DECISION_SCHEMA = "compose.editing_v2.process_v2_t1_capacity_decision"
DECISION_SCHEMA_VERSION = 1
DECISION_GO_STATUS = "GO_PROCESS_V2_BOUNDED_P50_T1_CAPACITY"
DECISION_NO_GO_STATUS = "NO_GO_PROCESS_V2_BOUNDED_P50_T1_CAPACITY"

PROCESS_V2_T1_RESULT_PROVENANCE_FIELDS = frozenset(
    {
        "capacity_policy_file_sha256",
        "capacity_policy_sha256",
        "prepared_completion_file_sha256",
        "prepared_completion_sha256",
        "prepared_input_file_sha256",
        "prepared_input_artifact_sha256",
        "panel_sha256",
        "panel_entry_inventory_sha256",
        "panel_entry_metadata_sha256",
        "panel_entry_binding_count",
        "gate_zero_decision_sha256",
        "active8_completion_sha256",
        "process_identity_sha256",
        "initial_model_state_sha256",
        "runner_implementation_sha256",
        "runner_source_revision_sha256",
        "execution_environment",
    }
)
PROCESS_V2_T1_RUNTIME_ONLY_PROVENANCE_FIELDS = frozenset(
    {
        "leaf_source_revision_sha256",
        "leaf_implementation_sha256",
        "leaf_reuse",
    }
)
_EXECUTION_ENVIRONMENT_FIELDS = {
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
_RUN_INTEGRITY_FIELDS = {
    "optimizer_steps_completed",
    "evaluation_steps",
    "selected_step",
    "termination_reason",
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
    "address_stream_sha256",
    "selected_model_state_sha256",
    "selected_checkpoint_file_sha256",
    "optimizer_state_sha256",
}
_TRAJECTORY_FIELDS = {
    "step",
    "minimum_entry_teacher_successor_probability",
    "mean_entry_canonical_successor_nll",
    "model_state_sha256",
}
_ENTRY_FIELDS = {
    "panel_entry_sha256",
    "family",
    "semantic_cell_id",
    "teacher_successor_probability",
    "canonical_successor_nll",
    "teacher_successor_rank",
    "teacher_successor_top1",
}
_GRADIENT_FIELDS = {
    "family",
    "family_gate_gradient_finite",
    "family_gate_cumulative_l2",
    "family_gate_nonzero_update_steps",
    "action_route_gradient_finite",
    "action_route_cumulative_l2",
    "action_route_nonzero_update_steps",
}
_RESULT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    *authority_false_block(),
    "objective_unit",
    "panel_kind",
    "repeated_state_empirical_law_evaluated",
    "provenance",
    "run_integrity",
    "evaluation_trajectory",
    "entry_metrics",
    "gradient_evidence",
    "threshold_checks",
    "result_sha256",
}


class ProcessV2T1ResultError(ValueError):
    """The bounded Process-V2 T1 result or its provenance is invalid."""


_RUNNER_IMPLEMENTATION_FILES = (
    "src/compose_v4/experiments/editing_v2_semantic_t1_capacity_runner.py",
    "src/compose_v4/experiments/editing_v2_semantic_t1_capacity_policy.py",
    "src/compose_v4/experiments/editing_v2_process_v2_t1_runtime.py",
    "src/compose_v4/experiments/editing_v2_process_v2_t1_result.py",
    "src/compose_v4/experiments/factorized_successor_training.py",
    "src/compose_v4/experiments/factorized_mark_conditional.py",
    "src/compose_v4/model/factorized_tracelet_rate_model.py",
    "configs/editing_v2_process_v2_t1_capacity_policy.json",
)


def process_v2_t1_runner_implementation_sha256(*, repo_root: Path) -> str:
    """Hash the exact Process-V2 optimizer, runtime adapter, and result boundary."""

    root = Path(repo_root).resolve()
    rows: list[dict[str, str]] = []
    for relative in _RUNNER_IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise ProcessV2T1ResultError(f"Process-V2 T1 runner source is absent: {relative}")
        rows.append(
            {"path": relative, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    return canonical_sha256(rows)


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_mapping(value: object, fields: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ProcessV2T1ResultError(f"{label} field set disagrees")
    return dict(value)


def project_process_v2_t1_result_provenance(value: object) -> dict[str, Any]:
    """Drop authenticated preparation details that the result binds transitively."""

    runtime = _exact_mapping(
        value,
        PROCESS_V2_T1_RESULT_PROVENANCE_FIELDS | PROCESS_V2_T1_RUNTIME_ONLY_PROVENANCE_FIELDS,
        label="T1 runtime provenance",
    )
    return {name: runtime[name] for name in PROCESS_V2_T1_RESULT_PROVENANCE_FIELDS}


def _finite(value: object, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProcessV2T1ResultError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ProcessV2T1ResultError(f"{label} is outside its finite range")
    return result


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int | bool]:
    probabilities = [float(row["teacher_successor_probability"]) for row in rows]
    nlls = [float(row["canonical_successor_nll"]) for row in rows]
    top1 = [bool(row["teacher_successor_top1"]) for row in rows]
    return {
        "entry_count": len(rows),
        "mean_teacher_successor_probability": sum(probabilities) / len(probabilities),
        "minimum_teacher_successor_probability": min(probabilities),
        "mean_teacher_successor_nll": sum(nlls) / len(nlls),
        "maximum_teacher_successor_nll": max(nlls),
        "teacher_successor_top1_fraction": sum(top1) / len(top1),
        "every_entry_teacher_successor_top1": all(top1),
    }


def _threshold_checks(
    entries: Sequence[Mapping[str, Any]],
    gradients: Sequence[Mapping[str, Any]],
    *,
    capacity_policy: Mapping[str, Any],
) -> dict[str, bool]:
    thresholds = capacity_policy["thresholds"]
    families = tuple(capacity_policy["required_families"])
    by_family = {
        family: _aggregate([row for row in entries if row["family"] == family])
        for family in families
    }
    cells = sorted({str(row["semantic_cell_id"]) for row in entries})
    by_cell = {
        cell: _aggregate([row for row in entries if row["semantic_cell_id"] == cell])
        for cell in cells
    }
    gradient_by_family = {str(row["family"]): row for row in gradients}
    return {
        "family_top1": all(
            row["teacher_successor_top1_fraction"]
            >= thresholds["minimum_unique_state_teacher_successor_top1"]
            for row in by_family.values()
        ),
        "family_probability": all(
            row["mean_teacher_successor_probability"]
            >= thresholds["minimum_unique_state_teacher_successor_probability"]
            for row in by_family.values()
        ),
        "family_nll": all(
            row["mean_teacher_successor_nll"]
            <= thresholds["maximum_unique_state_teacher_successor_nll"]
            for row in by_family.values()
        ),
        "cell_top1": all(
            row["teacher_successor_top1_fraction"]
            >= thresholds["minimum_nonempty_cell_teacher_successor_top1"]
            for row in by_cell.values()
        ),
        "cell_probability": all(
            row["mean_teacher_successor_probability"]
            >= thresholds["minimum_nonempty_cell_teacher_successor_probability"]
            for row in by_cell.values()
        ),
        "cell_nll": all(
            row["mean_teacher_successor_nll"]
            <= thresholds["maximum_nonempty_cell_teacher_successor_nll"]
            for row in by_cell.values()
        ),
        "every_entry_top1": (
            not thresholds["require_every_unique_entry_teacher_successor_top1"]
            or all(bool(row["teacher_successor_top1"]) for row in entries)
        ),
        "every_entry_probability": all(
            float(row["teacher_successor_probability"])
            >= thresholds["minimum_every_unique_entry_teacher_successor_probability"]
            for row in entries
        ),
        "family_route_gradient": all(
            row["family_gate_gradient_finite"] is True
            and float(row["family_gate_cumulative_l2"]) > 0.0
            and int(row["family_gate_nonzero_update_steps"]) > 0
            for row in gradient_by_family.values()
        ),
        "action_route_gradient": all(
            row["action_route_gradient_finite"] is True
            and float(row["action_route_cumulative_l2"]) > 0.0
            and int(row["action_route_nonzero_update_steps"]) > 0
            for row in gradient_by_family.values()
        ),
    }


def _validate_provenance(value: object) -> dict[str, Any]:
    provenance = _exact_mapping(
        value,
        PROCESS_V2_T1_RESULT_PROVENANCE_FIELDS,
        label="T1 provenance",
    )
    for name, item in provenance.items():
        if name == "execution_environment":
            environment = _exact_mapping(
                item, _EXECUTION_ENVIRONMENT_FIELDS, label="T1 execution environment"
            )
            environment_body = dict(environment)
            supplied = environment_body.pop("environment_sha256")
            if supplied != canonical_sha256(environment_body):
                raise ProcessV2T1ResultError("T1 execution environment hash disagrees")
        elif name == "panel_entry_binding_count":
            if type(item) is not int or item <= 0:
                raise ProcessV2T1ResultError("T1 panel entry count must be positive")
        elif not _is_sha(item):
            raise ProcessV2T1ResultError(f"T1 provenance {name} is not a SHA-256")
    return provenance


def validate_process_v2_t1_capacity_result(
    value: object, *, capacity_policy: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute the bounded result's schedule, metrics, and frozen thresholds."""

    result = _exact_mapping(value, _RESULT_FIELDS, label="Process-V2 T1 result")
    try:
        verify_self_hash(result, field="result_sha256", label="the Process-V2 T1 result")
        require_authority_false(result, label="the Process-V2 T1 result")
    except ValueError as error:
        raise ProcessV2T1ResultError(str(error)) from error
    if (
        result["schema"] != RESULT_SCHEMA
        or result["schema_version"] != RESULT_SCHEMA_VERSION
        or result["status"] != RESULT_STATUS
        or result["objective_unit"] != "exact_source_frozen_time_canonical_successor"
        or result["panel_kind"] != "unique_state_single_target_canonical_successor_capacity"
        or result["repeated_state_empirical_law_evaluated"] is not False
    ):
        raise ProcessV2T1ResultError("Process-V2 T1 result identity disagrees")
    provenance = _validate_provenance(result["provenance"])
    families = tuple(capacity_policy.get("required_families", ()))
    optimization = capacity_policy.get("optimization", {})
    if (
        provenance["capacity_policy_sha256"]
        != capacity_policy.get("policy_sha256", capacity_policy.get("contract_sha256"))
        or not families
        or len(families) != len(set(families))
        or optimization.get("trajectory_evaluation") != "step_zero_and_report_points_only"
    ):
        raise ProcessV2T1ResultError("Process-V2 T1 required families are invalid")

    integrity = _exact_mapping(
        result["run_integrity"], _RUN_INTEGRITY_FIELDS, label="T1 run integrity"
    )
    steps = integrity["optimizer_steps_completed"]
    evaluation_steps = integrity["evaluation_steps"]
    report_points = optimization.get("report_points")
    maximum_steps = optimization.get("maximum_optimizer_steps")
    expected_schedule = (
        [point for point in (0, *report_points) if point <= steps]
        if isinstance(report_points, list) and type(steps) is int
        else None
    )
    if (
        type(steps) is not int
        or not 1 <= steps <= maximum_steps
        or evaluation_steps != expected_schedule
        or not evaluation_steps
        or evaluation_steps[-1] != steps
        or integrity["selected_step"] not in evaluation_steps
        or integrity["termination_reason"]
        not in {"all_thresholds_passed_early", "maximum_optimizer_steps_reached", "aborted"}
        or type(integrity["resume_count"]) is not int
        or integrity["resume_count"] < 0
        or not isinstance(integrity["abort_reasons"], list)
        or any(
            type(integrity[name]) is not bool
            for name in (
                "resume_requested",
                "abort_triggered",
                "provenance_drift_detected",
                "stream_identity_drift_detected",
                "deterministic_algorithms_enabled",
                "mixed_precision",
                "hazard_included",
            )
        )
        or integrity["dtype"] != "float32"
    ):
        raise ProcessV2T1ResultError("Process-V2 T1 run integrity disagrees")
    for name in (
        "address_stream_sha256",
        "selected_model_state_sha256",
        "selected_checkpoint_file_sha256",
        "optimizer_state_sha256",
    ):
        if not _is_sha(integrity[name]):
            raise ProcessV2T1ResultError(f"T1 run integrity {name} is not a SHA-256")

    trajectory = result["evaluation_trajectory"]
    if not isinstance(trajectory, list) or len(trajectory) != len(evaluation_steps):
        raise ProcessV2T1ResultError("Process-V2 T1 trajectory is incomplete")
    trajectory_rows: list[tuple[int, float, float, str]] = []
    for expected_step, item in zip(evaluation_steps, trajectory, strict=True):
        row = _exact_mapping(item, _TRAJECTORY_FIELDS, label="T1 trajectory row")
        probability = _finite(
            row["minimum_entry_teacher_successor_probability"],
            label="T1 trajectory probability",
            minimum=0.0,
        )
        nll = _finite(
            row["mean_entry_canonical_successor_nll"],
            label="T1 trajectory NLL",
            minimum=0.0,
        )
        if (
            row["step"] != expected_step
            or probability > 1.0
            or not _is_sha(row["model_state_sha256"])
        ):
            raise ProcessV2T1ResultError("Process-V2 T1 trajectory row disagrees")
        trajectory_rows.append((expected_step, probability, nll, row["model_state_sha256"]))
    selected = min(trajectory_rows, key=lambda row: (-row[1], row[2], row[0]))
    if (
        integrity["selected_step"] != selected[0]
        or integrity["selected_model_state_sha256"] != selected[3]
    ):
        raise ProcessV2T1ResultError("Process-V2 T1 checkpoint selection disagrees")

    entries = result["entry_metrics"]
    if not isinstance(entries, list) or not entries:
        raise ProcessV2T1ResultError("Process-V2 T1 entry metrics are empty")
    normalized: list[dict[str, Any]] = []
    identifiers: list[str] = []
    metadata: list[dict[str, str]] = []
    family_counts = dict.fromkeys(families, 0)
    for item in entries:
        row = _exact_mapping(item, _ENTRY_FIELDS, label="T1 entry metric")
        probability = _finite(
            row["teacher_successor_probability"], label="T1 entry probability", minimum=0.0
        )
        nll = _finite(row["canonical_successor_nll"], label="T1 entry NLL", minimum=0.0)
        if (
            not _is_sha(row["panel_entry_sha256"])
            or row["family"] not in family_counts
            or not isinstance(row["semantic_cell_id"], str)
            or not row["semantic_cell_id"]
            or probability > 1.0
            or type(row["teacher_successor_rank"]) is not int
            or row["teacher_successor_rank"] <= 0
            or type(row["teacher_successor_top1"]) is not bool
            or row["teacher_successor_top1"] is not (row["teacher_successor_rank"] == 1)
            or not math.isclose(
                probability,
                math.exp(-nll),
                rel_tol=1e-10,
                abs_tol=1e-15,
            )
        ):
            raise ProcessV2T1ResultError("Process-V2 T1 entry metric disagrees")
        normalized.append(row)
        identifiers.append(row["panel_entry_sha256"])
        metadata.append(
            {
                "panel_entry_sha256": row["panel_entry_sha256"],
                "family": row["family"],
                "semantic_cell_id": row["semantic_cell_id"],
            }
        )
        family_counts[row["family"]] += 1
    expected_order = sorted(
        normalized,
        key=lambda row: (
            families.index(row["family"]),
            row["semantic_cell_id"],
            row["panel_entry_sha256"],
        ),
    )
    minimums = capacity_policy["panel_cardinality"]["minimum_entries_by_family"]
    maximums = capacity_policy["panel_cardinality"]["maximum_entries_by_family"]
    if (
        entries != expected_order
        or len(identifiers) != len(set(identifiers))
        or len(entries) != provenance["panel_entry_binding_count"]
        or canonical_sha256(sorted(identifiers)) != provenance["panel_entry_inventory_sha256"]
        or canonical_sha256(metadata) != provenance["panel_entry_metadata_sha256"]
        or any(
            not int(minimums[family]) <= family_counts[family] <= int(maximums[family])
            for family in families
        )
    ):
        raise ProcessV2T1ResultError("Process-V2 T1 panel binding disagrees")

    gradients = result["gradient_evidence"]
    if not isinstance(gradients, list) or [
        row.get("family") for row in gradients if isinstance(row, Mapping)
    ] != list(families):
        raise ProcessV2T1ResultError("Process-V2 T1 gradients do not cover Active8")
    for item in gradients:
        row = _exact_mapping(item, _GRADIENT_FIELDS, label="T1 gradient row")
        for name in ("family_gate_gradient_finite", "action_route_gradient_finite"):
            if type(row[name]) is not bool:
                raise ProcessV2T1ResultError("Process-V2 T1 gradient flag is not Boolean")
        for name in ("family_gate_cumulative_l2", "action_route_cumulative_l2"):
            _finite(row[name], label=f"T1 gradient {name}", minimum=0.0)
        for name in ("family_gate_nonzero_update_steps", "action_route_nonzero_update_steps"):
            if type(row[name]) is not int or not 0 <= row[name] <= steps:
                raise ProcessV2T1ResultError("Process-V2 T1 gradient step count disagrees")

    checks = _threshold_checks(entries, gradients, capacity_policy=capacity_policy)
    if result["threshold_checks"] != checks:
        raise ProcessV2T1ResultError("Process-V2 T1 threshold checks do not recompute")
    selected_minimum = min(float(row["teacher_successor_probability"]) for row in entries)
    selected_mean_nll = sum(float(row["canonical_successor_nll"]) for row in entries) / len(entries)
    if not (
        math.isclose(selected[1], selected_minimum, rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(selected[2], selected_mean_nll, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ProcessV2T1ResultError("Process-V2 T1 selected metrics disagree")
    return result


def build_process_v2_t1_capacity_result(
    *,
    provenance: Mapping[str, Any],
    run_integrity: Mapping[str, Any],
    evaluation_trajectory: Sequence[Mapping[str, Any]],
    entry_metrics: Sequence[Mapping[str, Any]],
    gradient_evidence: Sequence[Mapping[str, Any]],
    capacity_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the Process-V2 result through the shared runner callback seam."""

    entries = [dict(row) for row in entry_metrics]
    gradients = [dict(row) for row in gradient_evidence]
    body = {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": RESULT_STATUS,
        **authority_false_block(),
        "objective_unit": "exact_source_frozen_time_canonical_successor",
        "panel_kind": "unique_state_single_target_canonical_successor_capacity",
        "repeated_state_empirical_law_evaluated": False,
        "provenance": dict(provenance),
        "run_integrity": dict(run_integrity),
        "evaluation_trajectory": [dict(row) for row in evaluation_trajectory],
        "entry_metrics": entries,
        "gradient_evidence": gradients,
        "threshold_checks": _threshold_checks(entries, gradients, capacity_policy=capacity_policy),
    }
    result = {**body, "result_sha256": canonical_sha256(body)}
    return validate_process_v2_t1_capacity_result(result, capacity_policy=capacity_policy)


def _decision_failures(result: Mapping[str, Any]) -> list[str]:
    integrity = result["run_integrity"]
    failures = [
        f"threshold:{name}"
        for name, passed in sorted(result["threshold_checks"].items())
        if passed is not True
    ]
    integrity_checks = {
        "resume_not_used": integrity["resume_requested"] is False
        and integrity["resume_count"] == 0,
        "abort_not_triggered": integrity["abort_triggered"] is False
        and integrity["abort_reasons"] == [],
        "no_nonfinite_events": integrity["nonfinite_event_count"] == 0,
        "all_teachers_supported": integrity["unsupported_teacher_count"] == 0,
        "all_candidates_present": integrity["missing_candidate_count"] == 0,
        "provenance_stable": integrity["provenance_drift_detected"] is False,
        "stream_stable": integrity["stream_identity_drift_detected"] is False,
        "deterministic": integrity["deterministic_algorithms_enabled"] is True,
        "float32_without_mixed_precision": integrity["dtype"] == "float32"
        and integrity["mixed_precision"] is False,
        "hazard_excluded": integrity["hazard_included"] is False,
    }
    failures.extend(
        f"run_integrity:{name}" for name, passed in sorted(integrity_checks.items()) if not passed
    )
    return sorted(failures)


def build_process_v2_t1_capacity_decision(
    result: Mapping[str, Any],
    *,
    capacity_policy: Mapping[str, Any],
    result_file_sha256: str,
) -> dict[str, Any]:
    """Derive the only bounded-P50 authority from a validated Process-V2 result."""

    validated = validate_process_v2_t1_capacity_result(result, capacity_policy=capacity_policy)
    if not _is_sha(result_file_sha256):
        raise ProcessV2T1ResultError("Process-V2 T1 result file hash is invalid")
    failures = _decision_failures(validated)
    passed = not failures
    provenance = validated["provenance"]
    integrity = validated["run_integrity"]
    authority = authority_false_block()
    authority["t1_authorized"] = passed
    authority["bounded_p50_authorized"] = passed
    body = {
        "schema": DECISION_SCHEMA,
        "schema_version": DECISION_SCHEMA_VERSION,
        "status": DECISION_GO_STATUS if passed else DECISION_NO_GO_STATUS,
        **authority,
        "capacity_policy_sha256": provenance["capacity_policy_sha256"],
        "process_identity_sha256": provenance["process_identity_sha256"],
        "active8_completion_sha256": provenance["active8_completion_sha256"],
        "gate_zero_decision_sha256": provenance["gate_zero_decision_sha256"],
        "panel_sha256": provenance["panel_sha256"],
        "panel_entry_inventory_sha256": provenance["panel_entry_inventory_sha256"],
        "prepared_completion_sha256": provenance["prepared_completion_sha256"],
        "runner_implementation_sha256": provenance["runner_implementation_sha256"],
        "runner_source_revision_sha256": provenance["runner_source_revision_sha256"],
        "execution_environment_sha256": provenance["execution_environment"]["environment_sha256"],
        "result_file_sha256": result_file_sha256,
        "result_sha256": validated["result_sha256"],
        "selected_checkpoint_file_sha256": integrity["selected_checkpoint_file_sha256"],
        "selected_model_state_sha256": integrity["selected_model_state_sha256"],
        "required_families": list(capacity_policy["required_families"]),
        "unique_entry_count": len(validated["entry_metrics"]),
        "repeated_state_empirical_law_required_for_p50": False,
        "repeated_state_empirical_law_evaluated": False,
        "failed_checks": failures,
    }
    return {**body, "decision_sha256": canonical_sha256(body)}


def validate_process_v2_t1_capacity_decision(
    value: object,
    *,
    result: Mapping[str, Any],
    capacity_policy: Mapping[str, Any],
    result_file_sha256: str,
    require_p50_go: bool = False,
) -> dict[str, Any]:
    """Recompute a Process-V2 T1 decision from its exact result bytes."""

    if not isinstance(value, Mapping):
        raise ProcessV2T1ResultError("Process-V2 T1 decision must be an object")
    decision = dict(value)
    expected = build_process_v2_t1_capacity_decision(
        result,
        capacity_policy=capacity_policy,
        result_file_sha256=result_file_sha256,
    )
    if decision != expected:
        raise ProcessV2T1ResultError("Process-V2 T1 decision differs from exact recomputation")
    if require_p50_go and (
        decision["status"] != DECISION_GO_STATUS
        or decision["bounded_p50_authorized"] is not True
        or decision["failed_checks"] != []
    ):
        raise ProcessV2T1ResultError("Process-V2 T1 decision is not a bounded-P50 GO")
    return decision


__all__ = [
    "DECISION_FILENAME",
    "DECISION_GO_STATUS",
    "DECISION_NO_GO_STATUS",
    "DECISION_SCHEMA",
    "RESULT_FILENAME",
    "RESULT_SCHEMA",
    "RESULT_SCHEMA_VERSION",
    "ProcessV2T1ResultError",
    "PROCESS_V2_T1_RESULT_PROVENANCE_FIELDS",
    "PROCESS_V2_T1_RUNTIME_ONLY_PROVENANCE_FIELDS",
    "build_process_v2_t1_capacity_decision",
    "build_process_v2_t1_capacity_result",
    "process_v2_t1_runner_implementation_sha256",
    "project_process_v2_t1_result_provenance",
    "validate_process_v2_t1_capacity_decision",
    "validate_process_v2_t1_capacity_result",
]
