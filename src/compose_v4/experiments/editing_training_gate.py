"""Fail-closed validation for the COMPOSE editing-training gate contract."""

from __future__ import annotations

import json
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
MISLEADING_LEGACY_METRICS = {
    "family_accuracy",
    "family_top3_accuracy",
    "balanced_family_accuracy",
    "balanced_family_top3_accuracy",
}


class EditingTrainingGateError(ValueError):
    """The editing-training contract is malformed or not launch-ready."""


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
    for gate in contract.get("gates") or ():
        thresholds = gate.get("numeric_thresholds") or {}
        for name, value in sorted(thresholds.items()):
            if value is None:
                blockers.append(f"{gate['id']} has unfrozen threshold: {name}")
    return blockers


def assert_full_training_launch_authorized(contract: dict[str, Any]) -> None:
    blockers = full_training_launch_blockers(contract)
    if blockers:
        raise EditingTrainingGateError(
            "editing-training contract blocks a full run: " + "; ".join(blockers)
        )


__all__ = [
    "EditingTrainingGateError",
    "assert_full_training_launch_authorized",
    "full_training_launch_blockers",
    "load_editing_training_gate",
    "validate_editing_training_gate",
]
