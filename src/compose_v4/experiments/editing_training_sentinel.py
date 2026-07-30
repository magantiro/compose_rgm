"""Fail-fast exposure, gradient, and collapse sentinels for editing pilots."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

import torch
from torch import Tensor

from compose_v4.experiments.factorized_successor_data import (
    FactorizedSuccessorBatch,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    _CYCLE_OP_EXECUTOR_TO_FAMILY,
    MARK_RULE_TO_INDEX,
    FactorizedMarkBatch,
    FactorizedTraceletRateModel,
)

DEFAULT_REQUIRED_EDITING_FAMILIES = (
    "atom_insert",
    "atom_delete",
    "atom_restate",
    "bond_reorder",
    "bond_reroute",
    "cycle_insert",
    "cycle_attach",
)

FAMILY_PARAMETER_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "atom_insert": ("grow_root_head.", "grow_query.", "grow_option."),
    "atom_delete": ("delete_head.",),
    "atom_restate": ("restate_head.",),
    "bond_reorder": ("reorder_head.",),
    "bond_reroute": ("graft_head.",),
    "cycle_insert": ("cycle_close_head.",),
    "cycle_attach": ("cycle_open_head.",),
    "ring_system_restate": ("ring_restate_head.",),
}


class EditingTrainingSentinelError(RuntimeError):
    """A short pilot must abort before spending further optimizer updates."""


def _family_name(rule_name: str | None) -> str | None:
    return (
        None
        if rule_name is None
        else _CYCLE_OP_EXECUTOR_TO_FAMILY.get(rule_name, rule_name)
    )


@dataclass(frozen=True)
class SuccessorExposurePlan:
    expected_steps: int
    batch_size: int
    teacher_examples_by_family: Mapping[str, int]
    gradient_opportunities_by_family: Mapping[str, int]
    nonterminal_examples_by_step: tuple[int, ...]
    represented_semantic_cells: tuple[str, ...]
    teacher_alias_count: int

    @property
    def total_examples(self) -> int:
        return self.expected_steps * self.batch_size


def audit_successor_exposure_plan(
    batches: Iterable[Any],
    *,
    expected_steps: int,
    required_families: tuple[str, ...],
    minimum_teacher_examples: Mapping[str, int],
    require_semantic_cells: bool = True,
) -> SuccessorExposurePlan:
    """Preflight the exact deterministic pilot stream before GPU optimization."""

    if expected_steps <= 0:
        raise ValueError("exposure audit expected_steps must be positive")
    if not required_families or len(required_families) != len(
        set(required_families)
    ):
        raise ValueError("required_families must be nonempty and unique")
    if set(minimum_teacher_examples) != set(required_families):
        raise ValueError(
            "minimum teacher-example thresholds must exactly cover required families"
        )
    if any(
        type(value) is not int or value <= 0
        for value in minimum_teacher_examples.values()
    ):
        raise ValueError("minimum teacher-example thresholds must be positive")

    iterator = iter(batches)
    family_counts: Counter[str] = Counter()
    gradient_opportunities: Counter[str] = Counter()
    per_step_nonterminal: list[int] = []
    cells: set[str] = set()
    total_aliases = 0
    batch_size: int | None = None
    for step in range(1, expected_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration as error:
            raise EditingTrainingSentinelError(
                f"pilot loader ended before required step {step}"
            ) from error
        if not isinstance(batch, FactorizedSuccessorBatch):
            raise EditingTrainingSentinelError(
                "successor exposure audit received another batch type"
            )
        if batch_size is None:
            batch_size = batch.batch_size
        elif batch.batch_size != batch_size:
            raise EditingTrainingSentinelError(
                "pilot exposure stream changes batch size"
            )
        current_families = Counter(
            family
            for family in (
                _family_name(rule_name)
                for rule_name in batch.mark_batch.teacher_rule_names
            )
            if family is not None
        )
        nonterminal = sum(current_families.values())
        if nonterminal == 0:
            raise EditingTrainingSentinelError(
                f"pilot step {step} is terminal-only and has no identity gradient"
            )
        per_step_nonterminal.append(nonterminal)
        family_counts.update(current_families)
        gradient_opportunities.update(current_families)
        for fiber, cell in zip(
            batch.fibers,
            batch.semantic_cell_ids,
            strict=True,
        ):
            if fiber is not None:
                total_aliases += len(fiber.aliases)
                if cell is None and require_semantic_cells:
                    raise EditingTrainingSentinelError(
                        "pilot stream contains a jump without a semantic cell"
                    )
                if cell is not None:
                    cells.add(cell)
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise EditingTrainingSentinelError(
            "pilot loader contains more batches than its frozen step ceiling"
        )

    shortfalls = {
        family: {
            "observed": family_counts[family],
            "required": minimum_teacher_examples[family],
        }
        for family in required_families
        if family_counts[family] < minimum_teacher_examples[family]
    }
    if shortfalls:
        raise EditingTrainingSentinelError(
            f"pilot exposure plan starves required families: {shortfalls}"
        )
    if batch_size is None:
        raise EditingTrainingSentinelError("pilot exposure stream is empty")
    return SuccessorExposurePlan(
        expected_steps=expected_steps,
        batch_size=batch_size,
        teacher_examples_by_family=dict(sorted(family_counts.items())),
        gradient_opportunities_by_family=dict(
            sorted(gradient_opportunities.items())
        ),
        nonterminal_examples_by_step=tuple(per_step_nonterminal),
        represented_semantic_cells=tuple(sorted(cells)),
        teacher_alias_count=total_aliases,
    )


class P50GradientCollapseSentinel:
    """Stateful callback that can never observe more than exactly 50 updates."""

    expected_steps = 50

    def __init__(
        self,
        *,
        required_families: tuple[str, ...],
        minimum_gradient_updates: Mapping[str, int],
        maximum_family_nll_regression: Mapping[str, float],
        baseline_validation_metrics: Mapping[str, float],
        gradient_epsilon: float = 0.0,
    ) -> None:
        if not required_families or len(required_families) != len(
            set(required_families)
        ):
            raise ValueError("required_families must be nonempty and unique")
        for family in required_families:
            if family not in FAMILY_PARAMETER_PREFIXES:
                raise ValueError(
                    f"no registered gradient route for family {family!r}"
                )
        if set(minimum_gradient_updates) != set(required_families):
            raise ValueError(
                "gradient-update thresholds must exactly cover required families"
            )
        if set(maximum_family_nll_regression) != set(required_families):
            raise ValueError(
                "NLL-regression thresholds must exactly cover required families"
            )
        if any(
            type(value) is not int or not 1 <= value <= self.expected_steps
            for value in minimum_gradient_updates.values()
        ):
            raise ValueError(
                "minimum gradient updates must be integers in [1, 50]"
            )
        if any(
            not isfinite(float(value)) or float(value) < 0.0
            for value in maximum_family_nll_regression.values()
        ):
            raise ValueError(
                "maximum family NLL regressions must be finite and nonnegative"
            )
        if not isfinite(gradient_epsilon) or gradient_epsilon < 0.0:
            raise ValueError("gradient_epsilon must be finite and nonnegative")
        for family in required_families:
            count_key = f"teacher_examples_{family}"
            nll_key = f"canonical_successor_nll_{family}"
            if float(baseline_validation_metrics.get(count_key, 0.0)) <= 0.0:
                raise ValueError(
                    f"baseline validation lacks required family {family!r}"
                )
            baseline_nll = float(
                baseline_validation_metrics.get(nll_key, float("nan"))
            )
            if not isfinite(baseline_nll):
                raise ValueError(
                    f"baseline validation NLL is nonfinite for {family!r}"
                )

        self.required_families = required_families
        self.minimum_gradient_updates = dict(minimum_gradient_updates)
        self.maximum_family_nll_regression = {
            family: float(value)
            for family, value in maximum_family_nll_regression.items()
        }
        self.baseline_validation_metrics = {
            str(key): float(value)
            for key, value in baseline_validation_metrics.items()
        }
        self.gradient_epsilon = float(gradient_epsilon)
        self.last_step = 0
        self.last_validation_step = 0
        self.teacher_examples: Counter[str] = Counter()
        self.gradient_updates: Counter[str] = Counter()
        self.minimum_global_gradient_norm = float("inf")
        self.maximum_global_gradient_norm = 0.0

    def __call__(
        self,
        model: FactorizedTraceletRateModel,
        batch: FactorizedMarkBatch,
        completed_step: int,
        loss: Tensor,
    ) -> None:
        if completed_step != self.last_step + 1:
            raise EditingTrainingSentinelError(
                "P50 gradient callback steps are not contiguous"
            )
        if completed_step > self.expected_steps:
            raise EditingTrainingSentinelError(
                "P50 callback exceeded its 50-update ceiling"
            )
        if loss.numel() != 1 or not bool(torch.isfinite(loss)):
            raise EditingTrainingSentinelError(
                f"P50 loss is nonfinite at step {completed_step}"
            )
        named_parameters = tuple(model.named_parameters())
        gradient_squared_sum = 0.0
        observed_any_gradient = False
        for name, parameter in named_parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            observed_any_gradient = True
            if not bool(torch.isfinite(gradient).all()):
                raise EditingTrainingSentinelError(
                    f"P50 gradient is nonfinite for {name!r} at "
                    f"step {completed_step}"
                )
            norm = float(gradient.norm())
            gradient_squared_sum += norm * norm
        global_norm = gradient_squared_sum**0.5
        if not observed_any_gradient or not isfinite(global_norm) or global_norm <= 0.0:
            raise EditingTrainingSentinelError(
                f"P50 has no finite nonzero global gradient at step {completed_step}"
            )
        self.minimum_global_gradient_norm = min(
            self.minimum_global_gradient_norm,
            global_norm,
        )
        self.maximum_global_gradient_norm = max(
            self.maximum_global_gradient_norm,
            global_norm,
        )

        step_families = Counter(
            family
            for family in (
                _family_name(rule_name)
                for rule_name in batch.teacher_rule_names
            )
            if family is not None
        )
        self.teacher_examples.update(step_families)
        for family in self.required_families:
            if step_families[family] == 0:
                continue
            prefixes = FAMILY_PARAMETER_PREFIXES[family]
            route_squared_sum = 0.0
            matching_parameters = 0
            for name, parameter in named_parameters:
                if not name.startswith(prefixes):
                    continue
                if not parameter.requires_grad:
                    continue
                matching_parameters += 1
                if parameter.grad is not None:
                    norm = float(parameter.grad.norm())
                    route_squared_sum += norm * norm
            family_index = MARK_RULE_TO_INDEX[family]
            for name, parameter in named_parameters:
                if name not in {
                    "family_head.2.weight",
                    "family_head.2.bias",
                }:
                    continue
                if not parameter.requires_grad:
                    continue
                matching_parameters += 1
                if parameter.grad is not None:
                    row_gradient = parameter.grad[family_index]
                    norm = float(row_gradient.norm())
                    route_squared_sum += norm * norm
            if matching_parameters == 0:
                raise EditingTrainingSentinelError(
                    f"required family {family!r} has no trainable gradient route"
                )
            if route_squared_sum**0.5 > self.gradient_epsilon:
                self.gradient_updates[family] += 1
        self.last_step = completed_step

    def observe_validation(
        self,
        *,
        completed_step: int,
        metrics: Mapping[str, float],
    ) -> None:
        if not self.last_validation_step < completed_step <= self.last_step:
            raise EditingTrainingSentinelError(
                "P50 validation steps are stale, future, or duplicated"
            )
        for family in self.required_families:
            count = float(metrics.get(f"teacher_examples_{family}", 0.0))
            if count <= 0.0:
                raise EditingTrainingSentinelError(
                    f"P50 validation lost required family {family!r}"
                )
            key = f"canonical_successor_nll_{family}"
            current = float(metrics.get(key, float("nan")))
            if not isfinite(current):
                raise EditingTrainingSentinelError(
                    f"P50 validation NLL is nonfinite for {family!r}"
                )
            baseline = self.baseline_validation_metrics[key]
            regression = current - baseline
            if regression > self.maximum_family_nll_regression[family]:
                raise EditingTrainingSentinelError(
                    f"P50 required-family collapse at step {completed_step}: "
                    f"{family} regression={regression:.6g}, allowed="
                    f"{self.maximum_family_nll_regression[family]:.6g}"
                )
        self.last_validation_step = completed_step

    def finalize(self) -> dict[str, Any]:
        if self.last_step != self.expected_steps:
            raise EditingTrainingSentinelError(
                f"P50 ended at step {self.last_step}, expected exactly 50"
            )
        if self.last_validation_step != self.expected_steps:
            raise EditingTrainingSentinelError(
                "P50 has no validation observation at exactly step 50"
            )
        shortfalls = {
            family: {
                "observed": self.gradient_updates[family],
                "required": self.minimum_gradient_updates[family],
            }
            for family in self.required_families
            if self.gradient_updates[family]
            < self.minimum_gradient_updates[family]
        }
        if shortfalls:
            raise EditingTrainingSentinelError(
                f"P50 required-family gradient shortfalls: {shortfalls}"
            )
        return {
            "schema": "compose.editing_p50_gradient_collapse_report",
            "schema_version": 1,
            "status": "PASS",
            "completed_optimizer_steps": self.last_step,
            "last_validation_step": self.last_validation_step,
            "required_families": list(self.required_families),
            "teacher_examples_by_family": dict(
                sorted(self.teacher_examples.items())
            ),
            "gradient_updates_by_family": dict(
                sorted(self.gradient_updates.items())
            ),
            "minimum_global_gradient_norm": (
                self.minimum_global_gradient_norm
            ),
            "maximum_global_gradient_norm": (
                self.maximum_global_gradient_norm
            ),
        }


__all__ = [
    "DEFAULT_REQUIRED_EDITING_FAMILIES",
    "EditingTrainingSentinelError",
    "FAMILY_PARAMETER_PREFIXES",
    "P50GradientCollapseSentinel",
    "SuccessorExposurePlan",
    "audit_successor_exposure_plan",
]
