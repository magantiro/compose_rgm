"""Failure-only Process-V2 T1 parameter-scope diagnostics.

The primary T1 gate remains authoritative and immutable.  This diagnostic
answers only whether one failing family can fit its existing 64-entry panel
when optimization is restricted to the first frozen scope, ``heads_only``.
It consumes the authenticated materialized panel and never enumerates
chemistry, rebuilds fibers, or trains a family that already passed T1.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from compose_v4.data.editing_v2_process_v2_schema import (
    authority_false_block,
    canonical_sha256,
    require_authority_false,
    verify_self_hash,
)
from compose_v4.experiments.editing_v2_semantic_t1_capacity_policy import (
    semantic_t1_threshold_stop_allowed,
)
from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
    _MaterializedSemanticT1Panel,
    _batch_from_materialized_panel,
    _metric_rows,
    semantic_t1_learning_rate_for_step,
    semantic_t1_threshold_checks,
    state_dict_semantic_sha256,
    summarize_semantic_t1_metrics,
)
from compose_v4.experiments.factorized_successor_training import (
    factorized_successor_identity_loss,
    forward_teacher_successor_batch,
)
from compose_v4.experiments.successor_micro_overfit import (
    configure_micro_overfit_parameters,
)
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel

SCHEMA = "compose.editing_v2.process_v2_t1_failure_scope_result"
SCHEMA_VERSION = 1
PASS_STATUS = "PASS_PROCESS_V2_T1_FAILURE_SCOPE_NO_DOWNSTREAM_AUTHORITY"
FAIL_STATUS = "FAIL_PROCESS_V2_T1_FAILURE_SCOPE_NO_DOWNSTREAM_AUTHORITY"
SUPPORTED_SCOPE = "heads_only"


class ProcessV2T1FailureScopeError(RuntimeError):
    """The failure-only diagnostic or its frozen inputs are invalid."""


def failing_families_from_capacity_result(
    capacity_result: Mapping[str, Any],
    *,
    capacity_policy: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return only families responsible for the frozen entry-level failure.

    Family and cell aggregate checks already answer whether broad capability
    was acquired.  The registered scope ladder is reserved for the stricter
    every-entry probability and top-1 tail, so a passing family must never be
    retrained by this diagnostic.
    """

    checks = capacity_result.get("threshold_checks")
    rows = capacity_result.get("entry_metrics")
    thresholds = capacity_policy.get("thresholds")
    required = capacity_policy.get("required_families")
    if (
        not isinstance(checks, Mapping)
        or set(checks)
        != {
            "family_top1",
            "family_probability",
            "family_nll",
            "cell_top1",
            "cell_probability",
            "cell_nll",
            "every_entry_top1",
            "every_entry_probability",
            "family_route_gradient",
            "action_route_gradient",
        }
        or any(type(value) is not bool for value in checks.values())
        or not isinstance(rows, list)
        or not rows
        or not isinstance(thresholds, Mapping)
        or not isinstance(required, list)
        or not required
    ):
        raise ProcessV2T1FailureScopeError(
            "capacity result cannot define the frozen failure-only scope"
        )
    aggregate_names = set(checks) - {
        "every_entry_top1",
        "every_entry_probability",
    }
    if any(checks[name] is not True for name in aggregate_names):
        raise ProcessV2T1FailureScopeError(
            "failure-only scope is invalid when an aggregate or gradient gate failed"
        )
    if checks["every_entry_top1"] and checks["every_entry_probability"]:
        raise ProcessV2T1FailureScopeError("capacity result has no failing entry tail")
    floor = thresholds.get("minimum_every_unique_entry_teacher_successor_probability")
    if type(floor) not in {int, float} or not 0.0 < float(floor) < 1.0:
        raise ProcessV2T1FailureScopeError("capacity entry-probability floor is invalid")
    required_set = set(required)
    failing: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProcessV2T1FailureScopeError("capacity entry metric is not an object")
        family = row.get("family")
        probability = row.get("teacher_successor_probability")
        top1 = row.get("teacher_successor_top1")
        if (
            family not in required_set
            or type(probability) not in {int, float}
            or not 0.0 <= float(probability) <= 1.0
            or type(top1) is not bool
        ):
            raise ProcessV2T1FailureScopeError("capacity entry metric identity disagrees")
        if not top1 or float(probability) < float(floor):
            failing.add(str(family))
    ordered = tuple(family for family in required if family in failing)
    if not ordered:
        raise ProcessV2T1FailureScopeError(
            "capacity entry checks fail without an attributable family"
        )
    return ordered


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _criterion(metrics: Mapping[str, Any], *, step: int) -> tuple[float, float, int]:
    rows = metrics["per_entry"]
    minimum = min(float(row["teacher_successor_probability"]) for row in rows)
    mean_nll = sum(float(row["teacher_successor_nll"]) for row in rows) / len(rows)
    return minimum, -mean_nll, -step


def _subset_panel(
    panel: _MaterializedSemanticT1Panel,
    *,
    family: str,
) -> _MaterializedSemanticT1Panel:
    identifiers = tuple(
        str(entry["panel_entry_sha256"])
        for entry in panel.entries
        if entry["model_family"] == family
    )
    if not identifiers:
        raise ProcessV2T1FailureScopeError(
            f"failure-only T1 panel has no entries for {family!r}"
        )
    batch, fibers, partitions, entries = _batch_from_materialized_panel(
        panel, identifiers
    )
    return _MaterializedSemanticT1Panel(
        batch=batch,
        panel_ids=identifiers,
        fibers=fibers,
        partitions=partitions,
        entries=entries,
        index_by_panel_id={identifier: index for index, identifier in enumerate(identifiers)},
    )


def _gradient_row(
    *,
    family_seen: bool,
    family_steps: int,
    family_l2: float,
    action_seen: bool,
    action_steps: int,
    action_l2: float,
) -> dict[str, Any]:
    return {
        "family_route_finite_nonzero_seen": family_seen,
        "family_route_nonzero_steps": family_steps,
        "family_route_cumulative_gradient_norm": family_l2,
        "action_route_finite_nonzero_seen": action_seen,
        "action_route_nonzero_steps": action_steps,
        "action_route_cumulative_gradient_norm": action_l2,
    }


def validate_failure_scope_result(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcessV2T1FailureScopeError("T1 failure-scope result is not an object")
    result = dict(value)
    required = {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "family",
        "scope",
        "input_capacity_result_file_sha256",
        "input_capacity_result_sha256",
        "capacity_policy_sha256",
        "collated_completion_sha256",
        "initial_model_state_sha256",
        "panel_entry_count",
        "panel_entry_inventory_sha256",
        "optimizer_steps_completed",
        "selected_step",
        "trainable_parameter_count",
        "trainable_parameter_inventory_sha256",
        "trajectory",
        "selected_metrics",
        "selected_threshold_checks",
        "gradient_evidence",
        "diagnostic_passed",
        "result_sha256",
    }
    if set(result) != required:
        raise ProcessV2T1FailureScopeError("T1 failure-scope result fields disagree")
    try:
        verify_self_hash(result, field="result_sha256", label="the T1 failure-scope result")
        require_authority_false(result, label="the T1 failure-scope result")
    except ValueError as error:
        raise ProcessV2T1FailureScopeError(str(error)) from error
    trajectory = result["trajectory"]
    trajectory_steps = (
        [row["step"] for row in trajectory]
        if isinstance(trajectory, list)
        and trajectory
        and all(
            isinstance(row, Mapping) and type(row.get("step")) is int
            for row in trajectory
        )
        else None
    )
    if (
        result["schema"] != SCHEMA
        or result["schema_version"] != SCHEMA_VERSION
        or result["scope"] != SUPPORTED_SCOPE
        or type(result["diagnostic_passed"]) is not bool
        or result["status"]
        != (PASS_STATUS if result["diagnostic_passed"] else FAIL_STATUS)
        or type(result["panel_entry_count"]) is not int
        or result["panel_entry_count"] <= 0
        or type(result["optimizer_steps_completed"]) is not int
        or not 0 <= result["selected_step"] <= result["optimizer_steps_completed"]
        or not isinstance(result["family"], str)
        or not result["family"]
        or type(result["trainable_parameter_count"]) is not int
        or result["trainable_parameter_count"] <= 0
        or any(
            not _is_sha256(result[name])
            for name in (
                "input_capacity_result_file_sha256",
                "input_capacity_result_sha256",
                "capacity_policy_sha256",
                "collated_completion_sha256",
                "initial_model_state_sha256",
                "panel_entry_inventory_sha256",
                "trainable_parameter_inventory_sha256",
            )
        )
        or trajectory_steps is None
        or trajectory_steps != sorted(trajectory_steps)
        or trajectory_steps[-1] != result["optimizer_steps_completed"]
        or result["selected_step"]
        not in set(trajectory_steps)
        or not isinstance(result["selected_threshold_checks"], Mapping)
        or not result["selected_threshold_checks"]
        or any(type(item) is not bool for item in result["selected_threshold_checks"].values())
        or not isinstance(result["selected_metrics"], Mapping)
        or set(result["selected_metrics"].get("by_family", ())) != {result["family"]}
        or len(result["selected_metrics"].get("per_entry", ()))
        != result["panel_entry_count"]
        or not isinstance(result["gradient_evidence"], Mapping)
        or set(result["gradient_evidence"]) != {result["family"]}
        or result["diagnostic_passed"]
        is not all(result["selected_threshold_checks"].values())
    ):
        raise ProcessV2T1FailureScopeError("T1 failure-scope result identity disagrees")
    return result


def run_heads_only_failure_scope(
    model: FactorizedTraceletRateModel,
    materialized_panel: _MaterializedSemanticT1Panel,
    *,
    family: str,
    failing_families: Sequence[str],
    capacity_policy: Mapping[str, Any],
    input_capacity_result_file_sha256: str,
    input_capacity_result_sha256: str,
    collated_completion_sha256: str,
    initial_model_state_sha256: str,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run only the first frozen diagnostic scope for one failed family."""

    optimization = capacity_policy["optimization"]
    ordered_failures = tuple(dict.fromkeys(str(item) for item in failing_families))
    if (
        family not in ordered_failures
        or family not in capacity_policy["required_families"]
        or tuple(optimization["failure_diagnostic_scope_order"])[0]
        != SUPPORTED_SCOPE
        or optimization["failure_diagnostics_only_for_failing_families"] is not True
    ):
        raise ProcessV2T1FailureScopeError(
            "heads-only diagnostic is not restricted to a frozen failing family"
        )
    thresholds = capacity_policy["thresholds"]
    observed_initial = state_dict_semantic_sha256(model.state_dict())
    if observed_initial != initial_model_state_sha256:
        raise ProcessV2T1FailureScopeError(
            "heads-only diagnostic model is not the frozen scratch initialization"
        )
    panel = _subset_panel(materialized_panel, family=family)
    minimums = capacity_policy["panel_cardinality"]["minimum_entries_by_family"]
    maximums = capacity_policy["panel_cardinality"]["maximum_entries_by_family"]
    if not int(minimums[family]) <= len(panel.panel_ids) <= int(maximums[family]):
        raise ProcessV2T1FailureScopeError(
            "heads-only diagnostic panel cardinality violates the capacity policy"
        )

    torch.use_deterministic_algorithms(True)
    seed = int(optimization["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trainable_names = configure_micro_overfit_parameters(
        model, (family,), scope=SUPPORTED_SCOPE
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    family_names = tuple(name for name in trainable_names if name.startswith("family_head."))
    action_names = tuple(name for name in trainable_names if not name.startswith("family_head."))
    if not family_names or not action_names:
        raise ProcessV2T1FailureScopeError(
            "heads-only diagnostic does not expose both family and action routes"
        )
    named_parameters = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=semantic_t1_learning_rate_for_step(optimization, optimizer_step=1),
        weight_decay=float(optimization["weight_decay"]),
    )
    device_batch = panel.batch.to(model.device)
    maximum_steps = int(optimization["maximum_optimizer_steps"])
    report_points = {0, *(int(item) for item in optimization["report_points"])}
    trajectory: list[dict[str, Any]] = []
    selected_metrics: dict[str, Any] | None = None
    selected_checks: dict[str, bool] | None = None
    selected_step = 0
    selected_criterion = (float("-inf"), float("-inf"), 0)
    family_seen = False
    family_steps = 0
    family_l2 = 0.0
    action_seen = False
    action_steps = 0
    action_l2 = 0.0
    completed_steps = 0

    while completed_steps <= maximum_steps:
        gradients = {
            family: _gradient_row(
                family_seen=family_seen,
                family_steps=family_steps,
                family_l2=family_l2,
                action_seen=action_seen,
                action_steps=action_steps,
                action_l2=action_l2,
            )
        }
        if completed_steps in report_points:
            rows = _metric_rows(
                panel,
                model,
                batch_size=int(optimization["batch_size"]),
            )
            metrics = summarize_semantic_t1_metrics(rows)
            checks = semantic_t1_threshold_checks(
                metrics,
                thresholds=thresholds,
                gradient_evidence=gradients,
            )
            criterion = _criterion(metrics, step=completed_steps)
            if criterion > selected_criterion:
                selected_criterion = criterion
                selected_step = completed_steps
                selected_metrics = metrics
                selected_checks = checks
            row = {
                "step": completed_steps,
                "minimum_entry_teacher_successor_probability": criterion[0],
                "mean_entry_canonical_successor_nll": -criterion[1],
                "all_threshold_checks_pass": all(checks.values()),
            }
            trajectory.append(row)
            if progress is not None:
                progress(row)
            if semantic_t1_threshold_stop_allowed(
                optimizer_step=completed_steps,
                threshold_checks=checks,
            ):
                break
        if completed_steps == maximum_steps:
            break

        next_step = completed_steps + 1
        learning_rate = semantic_t1_learning_rate_for_step(
            optimization, optimizer_step=next_step
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_teacher_successor_batch(
            model, device_batch, panel.fibers
        )
        loss = factorized_successor_identity_loss(prediction, device_batch)
        if not bool(torch.isfinite(loss)):
            raise ProcessV2T1FailureScopeError(
                f"nonfinite heads-only loss before step {next_step}"
            )
        loss.backward()

        def route_norm(names: Sequence[str]) -> float:
            squared = 0.0
            for name in names:
                gradient = named_parameters[name].grad
                if gradient is None:
                    continue
                if not bool(torch.isfinite(gradient).all()):
                    raise ProcessV2T1FailureScopeError(
                        f"nonfinite heads-only gradient for {name!r}"
                    )
                value = float(gradient.norm())
                squared += value * value
            return squared**0.5

        family_norm = route_norm(family_names)
        action_norm = route_norm(action_names)
        if family_norm > 0.0:
            family_seen = True
            family_steps += 1
            family_l2 += family_norm
        if action_norm > 0.0:
            action_seen = True
            action_steps += 1
            action_l2 += action_norm
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(optimization["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise ProcessV2T1FailureScopeError("heads-only gradient norm is nonfinite")
        optimizer.step()
        completed_steps = next_step

    if selected_metrics is None or selected_checks is None:
        raise ProcessV2T1FailureScopeError("heads-only diagnostic selected no report point")
    gradients = {
        family: _gradient_row(
            family_seen=family_seen,
            family_steps=family_steps,
            family_l2=family_l2,
            action_seen=action_seen,
            action_steps=action_steps,
            action_l2=action_l2,
        )
    }
    # Recompute the selected checks with the terminal gradient evidence.  The
    # metric rows remain those of the prospectively selected report point.
    selected_checks = semantic_t1_threshold_checks(
        selected_metrics,
        thresholds=thresholds,
        gradient_evidence=gradients,
    )
    passed = all(selected_checks.values())
    body = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        **authority_false_block(),
        "family": family,
        "scope": SUPPORTED_SCOPE,
        "input_capacity_result_file_sha256": input_capacity_result_file_sha256,
        "input_capacity_result_sha256": input_capacity_result_sha256,
        "capacity_policy_sha256": capacity_policy["policy_sha256"],
        "collated_completion_sha256": collated_completion_sha256,
        "initial_model_state_sha256": initial_model_state_sha256,
        "panel_entry_count": len(panel.panel_ids),
        "panel_entry_inventory_sha256": canonical_sha256(list(panel.panel_ids)),
        "optimizer_steps_completed": completed_steps,
        "selected_step": selected_step,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "trainable_parameter_inventory_sha256": canonical_sha256(list(trainable_names)),
        "trajectory": trajectory,
        "selected_metrics": selected_metrics,
        "selected_threshold_checks": selected_checks,
        "gradient_evidence": gradients,
        "diagnostic_passed": passed,
    }
    return validate_failure_scope_result(
        {**body, "result_sha256": canonical_sha256(body)}
    )


__all__ = [
    "FAIL_STATUS",
    "PASS_STATUS",
    "ProcessV2T1FailureScopeError",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SUPPORTED_SCOPE",
    "failing_families_from_capacity_result",
    "run_heads_only_failure_scope",
    "validate_failure_scope_result",
]
