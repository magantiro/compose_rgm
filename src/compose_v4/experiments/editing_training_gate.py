"""Fail-closed validation for the COMPOSE editing-training gate contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "compose.editing_training_gate_contract"
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_GATE_IDS = (
    "M0_metric_semantics",
    "S0_support_and_labels",
    "T1_true_successor_micro_overfit",
    "P50_gradient_and_collapse_sentinel",
    "P500_mixed_pilot",
    "P2000_scientific_pilot",
)
EXPECTED_MAXIMUM_OPTIMIZER_STEPS = {
    "M0_metric_semantics": 0,
    "S0_support_and_labels": 0,
    "T1_true_successor_micro_overfit": 500,
    "P50_gradient_and_collapse_sentinel": 50,
    "P500_mixed_pilot": 500,
    "P2000_scientific_pilot": 2000,
}
REQUIRED_PRIMARY_METRICS = {
    "production_weighted_canonical_successor_nll",
    "balanced_semantic_cell_canonical_successor_nll",
}
REQUIRED_S0_REQUIREMENTS = {
    "initialization_transfer_plan_complete",
    "initialized_state_matches_transfer_plan_exactly",
    "all_teacher_actions_legal",
    "all_teacher_actions_execute_to_stored_exact_successor",
    "all_teacher_successor_fibers_nonempty",
    "all_required_families_have_teachers_and_production_candidates",
    "all_legal_marks_execute",
    "canonical_successor_pushforward_finite_and_normalized",
    "all_packed_rows_preserve_persistent_slot_state",
    "all_undirected_endpoint_conventions_unique",
    "all_required_semantic_cells_nonempty",
    "split_overlap_zero",
}
MISLEADING_LEGACY_METRICS = {
    "family_accuracy",
    "family_top3_accuracy",
    "balanced_family_accuracy",
    "balanced_family_top3_accuracy",
}
REQUIRED_P50_FAMILIES = (
    "atom_insert",
    "atom_delete",
    "atom_restate",
    "bond_reorder",
    "bond_reroute",
    "cycle_insert",
    "cycle_attach",
)
P50_INITIALIZATION_REGIMES = (
    "scratch",
    "compatible_warm_start",
    "compatible_warm_start_with_retention",
)


class EditingTrainingGateError(ValueError):
    """The editing-training contract is malformed or not launch-ready."""


@dataclass(frozen=True)
class ResolvedP50Thresholds:
    """Finite, exact-family thresholds resolved before pilot construction."""

    initialization_regime: str
    required_families: tuple[str, ...]
    minimum_gradient_updates: tuple[tuple[str, int], ...]
    maximum_family_nll_regression: tuple[tuple[str, float], ...]
    inherited_retention_status: str
    maximum_inherited_probe_nll_regression: float | None

    @property
    def minimum_gradient_update_map(self) -> dict[str, int]:
        return dict(self.minimum_gradient_updates)

    @property
    def maximum_family_nll_regression_map(self) -> dict[str, float]:
        return dict(self.maximum_family_nll_regression)


def load_editing_training_gate(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    validate_editing_training_gate(payload)
    return payload


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return sorted(duplicate)


def validate_editing_training_gate(contract: dict[str, Any]) -> None:
    if contract.get("schema") != EXPECTED_SCHEMA:
        raise EditingTrainingGateError("unexpected editing-training schema")
    if contract.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise EditingTrainingGateError("unexpected editing-training schema version")

    metric_contract = contract.get("metric_contract") or {}
    primary = [str(name) for name in metric_contract.get("primary") or ()]
    duplicate_primary = _duplicates(primary)
    if duplicate_primary:
        raise EditingTrainingGateError(
            f"duplicate primary metrics: {duplicate_primary}"
        )
    if set(primary) != REQUIRED_PRIMARY_METRICS:
        raise EditingTrainingGateError(
            "primary metrics must be the production-weighted and balanced "
            "canonical-successor NLLs"
        )
    forbidden = set(metric_contract.get("forbidden_as_primary_or_checkpoint_selector") or ())
    missing_forbidden = MISLEADING_LEGACY_METRICS - forbidden
    if missing_forbidden:
        raise EditingTrainingGateError(
            "legacy family-choice metrics are not all forbidden as primary: "
            f"{sorted(missing_forbidden)}"
        )
    if set(primary) & forbidden:
        raise EditingTrainingGateError("a primary metric is also forbidden")

    gates = contract.get("gates") or []
    gate_ids = [str(gate.get("id")) for gate in gates]
    if tuple(gate_ids) != EXPECTED_GATE_IDS:
        raise EditingTrainingGateError(
            "gate sequence must be M0, S0, T1, P50, P500, P2000"
        )
    for gate in gates:
        maximum_steps = gate.get("maximum_optimizer_steps")
        if not isinstance(maximum_steps, int) or maximum_steps < 0:
            raise EditingTrainingGateError(
                f"{gate.get('id')} has an invalid optimizer-step ceiling"
            )
        expected_maximum = EXPECTED_MAXIMUM_OPTIMIZER_STEPS[gate["id"]]
        if maximum_steps != expected_maximum:
            raise EditingTrainingGateError(
                f"{gate['id']} must have exactly {expected_maximum} optimizer "
                f"steps, observed {maximum_steps}"
            )
        if not gate.get("requirements"):
            raise EditingTrainingGateError(
                f"{gate.get('id')} has no requirements"
            )
    support_gate = next(
        gate for gate in gates if gate["id"] == "S0_support_and_labels"
    )
    missing_support_requirements = REQUIRED_S0_REQUIREMENTS - set(
        support_gate["requirements"]
    )
    if missing_support_requirements:
        raise EditingTrainingGateError(
            "S0 is missing fail-closed initialization or successor-support "
            f"requirements: {sorted(missing_support_requirements)!r}"
        )

    selection = contract.get("checkpoint_selection") or {}
    if selection.get("partition") != "validation_only":
        raise EditingTrainingGateError(
            "checkpoint selection must use validation only"
        )
    if selection.get("test_metrics_forbidden") is not True:
        raise EditingTrainingGateError(
            "test metrics must be forbidden for checkpoint selection"
        )
    if selection.get("primary") not in REQUIRED_PRIMARY_METRICS:
        raise EditingTrainingGateError(
            "checkpoint primary is not a registered canonical-successor NLL"
        )


def full_training_launch_blockers(contract: dict[str, Any]) -> list[str]:
    """Return every unresolved item that blocks a replacement full run."""

    validate_editing_training_gate(contract)
    blockers: list[str] = []
    if contract.get("full_training_authorized") is not True:
        blockers.append("full_training_authorized is not true")
    if contract.get("status") != "FROZEN_FULL_TRAINING_AUTHORIZED":
        blockers.append("status is not FROZEN_FULL_TRAINING_AUTHORIZED")

    def unresolved_paths(value: object, prefix: str) -> list[str]:
        if value is None:
            return [prefix]
        if isinstance(value, dict):
            return [
                path
                for key, child in sorted(value.items())
                for path in unresolved_paths(child, f"{prefix}.{key}")
            ]
        if isinstance(value, list):
            return [
                path
                for index, child in enumerate(value)
                for path in unresolved_paths(child, f"{prefix}[{index}]")
            ]
        return []

    for gate in contract.get("gates") or ():
        thresholds = gate.get("numeric_thresholds") or {}
        for name, value in sorted(thresholds.items()):
            blockers.extend(
                f"{gate['id']} has unfrozen threshold: {path}"
                for path in unresolved_paths(value, name)
            )
    return blockers


def _p50_gate(contract: dict[str, Any]) -> dict[str, Any]:
    return next(
        gate
        for gate in contract["gates"]
        if gate["id"] == "P50_gradient_and_collapse_sentinel"
    )


def _exact_family_mapping(
    value: object,
    *,
    name: str,
    required_families: tuple[str, ...],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EditingTrainingGateError(
            f"P50 threshold {name} must be a per-family object"
        )
    if set(value) != set(required_families):
        raise EditingTrainingGateError(
            f"P50 threshold {name} must exactly cover required families"
        )
    return value


def resolve_p50_thresholds(
    contract: dict[str, Any],
    *,
    initialization_regime: str,
    required_families: tuple[str, ...] = REQUIRED_P50_FAMILIES,
) -> ResolvedP50Thresholds:
    """Resolve a non-null P50 contract before a loader or optimizer exists."""

    validate_editing_training_gate(contract)
    if (
        initialization_regime not in P50_INITIALIZATION_REGIMES
        or initialization_regime
        not in set(
            contract.get("bounded_design_comparison", {})
            .get("factors", {})
            .get("initialization", ())
        )
    ):
        raise EditingTrainingGateError(
            f"unregistered P50 initialization regime: {initialization_regime!r}"
        )
    if (
        not required_families
        or len(required_families) != len(set(required_families))
    ):
        raise EditingTrainingGateError(
            "P50 required families must be nonempty and unique"
        )
    thresholds = _p50_gate(contract).get("numeric_thresholds") or {}
    minimum_raw = _exact_family_mapping(
        thresholds.get("minimum_gradient_updates_per_required_slice"),
        name="minimum_gradient_updates_per_required_slice",
        required_families=required_families,
    )
    maximum_raw = _exact_family_mapping(
        thresholds.get("maximum_required_slice_successor_nll_regression"),
        name="maximum_required_slice_successor_nll_regression",
        required_families=required_families,
    )
    minimum: list[tuple[str, int]] = []
    maximum: list[tuple[str, float]] = []
    for family in required_families:
        minimum_value = minimum_raw[family]
        if (
            type(minimum_value) is not int
            or not 1 <= minimum_value <= 50
        ):
            raise EditingTrainingGateError(
                "P50 minimum gradient updates must be integers in [1, 50]"
            )
        maximum_value = maximum_raw[family]
        if (
            isinstance(maximum_value, bool)
            or not isinstance(maximum_value, (int, float))
            or not isfinite(float(maximum_value))
            or float(maximum_value) < 0.0
        ):
            raise EditingTrainingGateError(
                "P50 maximum family NLL regressions must be finite and "
                "nonnegative"
            )
        minimum.append((family, minimum_value))
        maximum.append((family, float(maximum_value)))

    retention = thresholds.get("maximum_inherited_probe_nll_regression")
    if not isinstance(retention, dict) or set(retention) != set(
        P50_INITIALIZATION_REGIMES
    ):
        raise EditingTrainingGateError(
            "P50 inherited-retention threshold must exactly cover "
            "initialization regimes"
        )
    selected_retention = retention[initialization_regime]
    if initialization_regime == "scratch":
        if selected_retention != "NOT_APPLICABLE":
            raise EditingTrainingGateError(
                "scratch P50 must explicitly declare inherited retention "
                "NOT_APPLICABLE"
            )
        retention_status = "NOT_APPLICABLE"
        maximum_inherited = None
    else:
        if (
            isinstance(selected_retention, bool)
            or not isinstance(selected_retention, (int, float))
            or not isfinite(float(selected_retention))
            or float(selected_retention) < 0.0
        ):
            raise EditingTrainingGateError(
                "warm-start P50 requires a finite nonnegative inherited-probe "
                "regression threshold"
            )
        retention_status = "REQUIRED"
        maximum_inherited = float(selected_retention)
    return ResolvedP50Thresholds(
        initialization_regime=initialization_regime,
        required_families=required_families,
        minimum_gradient_updates=tuple(minimum),
        maximum_family_nll_regression=tuple(maximum),
        inherited_retention_status=retention_status,
        maximum_inherited_probe_nll_regression=maximum_inherited,
    )


def p50_launch_blockers(
    contract: dict[str, Any],
    *,
    initialization_regime: str,
    required_families: tuple[str, ...] = REQUIRED_P50_FAMILIES,
) -> list[str]:
    try:
        resolve_p50_thresholds(
            contract,
            initialization_regime=initialization_regime,
            required_families=required_families,
        )
    except EditingTrainingGateError as error:
        return [str(error)]
    return []


def assert_p50_launch_authorized(
    contract: dict[str, Any],
    *,
    initialization_regime: str,
    required_families: tuple[str, ...] = REQUIRED_P50_FAMILIES,
) -> ResolvedP50Thresholds:
    blockers = p50_launch_blockers(
        contract,
        initialization_regime=initialization_regime,
        required_families=required_families,
    )
    if blockers:
        raise EditingTrainingGateError(
            "editing-training contract blocks P50 before loader construction: "
            + "; ".join(blockers)
        )
    return resolve_p50_thresholds(
        contract,
        initialization_regime=initialization_regime,
        required_families=required_families,
    )


def assert_full_training_launch_authorized(contract: dict[str, Any]) -> None:
    blockers = full_training_launch_blockers(contract)
    if blockers:
        raise EditingTrainingGateError(
            "editing-training contract blocks a full run: " + "; ".join(blockers)
        )


__all__ = [
    "P50_INITIALIZATION_REGIMES",
    "REQUIRED_P50_FAMILIES",
    "EditingTrainingGateError",
    "ResolvedP50Thresholds",
    "assert_full_training_launch_authorized",
    "assert_p50_launch_authorized",
    "full_training_launch_blockers",
    "load_editing_training_gate",
    "p50_launch_blockers",
    "resolve_p50_thresholds",
    "validate_editing_training_gate",
]
