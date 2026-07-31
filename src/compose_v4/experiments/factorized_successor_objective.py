"""Canonical-successor objectives and validation metrics for the shared trainer."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

import torch
from torch import Tensor

from compose_v4.experiments.factorized_mark_conditional import (
    assert_teachers_in_exact_candidates,
)
from compose_v4.experiments.factorized_successor_data import (
    FactorizedSuccessorBatch,
)
from compose_v4.experiments.factorized_successor_training import (
    FactorizedSuccessorPrediction,
    SuccessorTrainingError,
    factorized_hazard_bregman_loss,
    factorized_successor_bregman_loss,
    factorized_successor_identity_loss,
    forward_teacher_successor_batch,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    _CYCLE_OP_EXECUTOR_TO_FAMILY,
    MARK_RULE_NAMES,
    FactorizedMarkBatch,
    FactorizedTraceletRateModel,
)

SuccessorObjectiveMode = Literal[
    "balanced_semantic_cell_productive_identity",
    "productive_identity",
    "productive_identity_plus_hazard",
    "successor_generator_bregman",
]


def _teacher_family_name(rule_name: str | None) -> str | None:
    return None if rule_name is None else _CYCLE_OP_EXECUTOR_TO_FAMILY.get(rule_name, rule_name)


def balanced_semantic_cell_productive_identity_loss(
    prediction: FactorizedSuccessorPrediction,
    batch: FactorizedSuccessorBatch,
) -> Tensor:
    """Average productive successor NLL equally within and across cells.

    This Stage-A capability objective intentionally ignores the legacy
    importance weights.  Every represented semantic cell contributes one
    equally weighted mean, regardless of its row count or production mass.
    Terminal rows do not enter the embedded-jump identity objective and need
    no semantic-cell assignment.

    The loss depends only on productive successor probabilities.  In
    particular, it neither reads nor calibrates the total-hazard prediction.
    """

    log_probability = prediction.selected_productive_successor_log_probability
    teacher_rates = batch.mark_batch.teacher_rates.to(log_probability.device)
    nonterminal_indices = torch.nonzero(
        teacher_rates > 0,
        as_tuple=False,
    ).flatten()
    if nonterminal_indices.numel() == 0:
        raise SuccessorTrainingError(
            "balanced semantic-cell successor-identity loss requires at least one molecular jump"
        )

    rows_by_cell: dict[str, list[int]] = defaultdict(list)
    for row_index in nonterminal_indices.tolist():
        cell_id = batch.semantic_cell_ids[row_index]
        if not isinstance(cell_id, str) or not cell_id:
            raise SuccessorTrainingError(
                "balanced semantic-cell successor-identity row "
                f"{row_index} lacks a frozen semantic cell"
            )
        rows_by_cell[cell_id].append(row_index)

    per_row_nll = -log_probability
    per_cell_nll = []
    for cell_id in sorted(rows_by_cell):
        cell_rows = torch.tensor(
            rows_by_cell[cell_id],
            dtype=torch.long,
            device=log_probability.device,
        )
        per_cell_nll.append(per_row_nll.index_select(0, cell_rows).mean())
    return torch.stack(per_cell_nll).mean()


@torch.no_grad()
def factorized_successor_metrics(
    model: FactorizedTraceletRateModel,
    batch: FactorizedSuccessorBatch,
    *,
    use_bf16: bool,
    microbatch_size: int | None = None,
    require_semantic_cells: bool = True,
) -> dict[str, float]:
    """Evaluate teacher-successor likelihood and hazard as separate objects."""

    if not isinstance(batch, FactorizedSuccessorBatch):
        raise TypeError("successor metrics require FactorizedSuccessorBatch")
    if microbatch_size is not None and microbatch_size <= 0:
        raise ValueError("evaluation microbatch size must be positive")
    resolved_microbatch = min(
        batch.batch_size,
        batch.batch_size if microbatch_size is None else microbatch_size,
    )
    device = model.device
    identity_weighted_sum = 0.0
    identity_weight_sum = 0.0
    successor_probability_sum = 0.0
    family_probability_sum = 0.0
    within_family_probability_sum = 0.0
    nonterminal_count = 0
    row_count = 0
    generator_weighted_sum = 0.0
    hazard_weighted_sum = 0.0
    productive_probability_sum = 0.0
    predicted_hazard_sum = 0.0
    teacher_hazard_sum = 0.0
    hazard_absolute_error_sum = 0.0
    terminal_hazard_sum = 0.0
    terminal_count = 0
    cell_loss_sums: dict[str, float] = defaultdict(float)
    cell_weight_sums: dict[str, float] = defaultdict(float)
    family_loss_sums: dict[str, float] = defaultdict(float)
    family_weight_sums: dict[str, float] = defaultdict(float)
    family_gate_loss_sums: dict[str, float] = defaultdict(float)
    within_family_loss_sums: dict[str, float] = defaultdict(float)

    for start in range(0, batch.batch_size, resolved_microbatch):
        cpu_batch = batch.subbatch(
            start,
            min(start + resolved_microbatch, batch.batch_size),
        )
        device_batch = cpu_batch.to(
            device,
            non_blocking=device.type == "cuda",
        )
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16 and device.type == "cuda"
            else nullcontext()
        )
        with context:
            prediction = forward_teacher_successor_batch(
                model,
                device_batch.mark_batch,
                device_batch.fibers,
                state_supports=device_batch.state_supports,
            )

        mark_batch = device_batch.mark_batch
        teacher_rates = mark_batch.teacher_rates.to(device)
        weights = mark_batch.importance_weights.to(device)
        nonterminal = teacher_rates > 0
        terminal = ~nonterminal
        identity_nll = -prediction.selected_productive_successor_log_probability
        log_hazard = torch.log(prediction.total_hazard.clamp_min(1e-12))
        generator_teacher_term = teacher_rates * (
            log_hazard + prediction.selected_successor_log_probability
        )
        generator_per_example = prediction.productive_hazard - torch.where(
            nonterminal,
            generator_teacher_term,
            torch.zeros_like(generator_teacher_term),
        )
        hazard_per_example = prediction.total_hazard - torch.where(
            nonterminal,
            teacher_rates * log_hazard,
            torch.zeros_like(teacher_rates),
        )

        current_rows = mark_batch.batch_size
        row_count += current_rows
        generator_weighted_sum += float((generator_per_example * weights).sum())
        hazard_weighted_sum += float((hazard_per_example * weights).sum())
        productive_probability_sum += float(prediction.productive_log_probability.exp().sum())
        predicted_hazard_sum += float(prediction.total_hazard.sum())
        teacher_hazard_sum += float(teacher_rates.sum())
        hazard_absolute_error_sum += float((prediction.total_hazard - teacher_rates).abs().sum())

        current_nonterminal = int(nonterminal.sum())
        if current_nonterminal:
            selected_nll = identity_nll[nonterminal]
            selected_weights = weights[nonterminal]
            identity_weighted_sum += float((selected_nll * selected_weights).sum())
            identity_weight_sum += float(selected_weights.sum())
            successor_probability_sum += float(
                prediction.selected_productive_successor_log_probability[nonterminal].exp().sum()
            )
            family_probability_sum += float(
                prediction.teacher_family_log_probability[nonterminal].exp().sum()
            )
            within_family_probability_sum += float(
                prediction.selected_within_teacher_family_log_probability[nonterminal].exp().sum()
            )
            nonterminal_count += current_nonterminal

            selected_indices = (
                torch.nonzero(
                    nonterminal,
                    as_tuple=False,
                )
                .flatten()
                .tolist()
            )
            for local_index in selected_indices:
                cell = cpu_batch.semantic_cell_ids[local_index]
                if cell is None:
                    if require_semantic_cells:
                        raise SuccessorTrainingError(
                            "successor validation row lacks a frozen semantic cell"
                        )
                else:
                    weight = float(weights[local_index])
                    cell_loss_sums[cell] += float(identity_nll[local_index]) * weight
                    cell_weight_sums[cell] += weight
                family = _teacher_family_name(mark_batch.teacher_rule_names[local_index])
                if family is None:
                    raise SuccessorTrainingError(
                        "nonterminal validation row lacks a teacher family"
                    )
                weight = float(weights[local_index])
                family_loss_sums[family] += float(identity_nll[local_index]) * weight
                family_weight_sums[family] += weight
                family_gate_loss_sums[family] += (
                    -float(prediction.teacher_family_log_probability[local_index]) * weight
                )
                within_family_loss_sums[family] += (
                    -float(prediction.selected_within_teacher_family_log_probability[local_index])
                    * weight
                )
        if bool(terminal.any()):
            terminal_hazard_sum += float(prediction.total_hazard[terminal].sum())
            terminal_count += int(terminal.sum())

    if row_count != batch.batch_size:
        raise RuntimeError("successor metric microbatches lost rows")
    if nonterminal_count == 0 or identity_weight_sum <= 0.0:
        raise SuccessorTrainingError(
            "successor validation batch contains no weighted molecular jump"
        )
    if require_semantic_cells and not cell_weight_sums:
        raise SuccessorTrainingError("successor validation has no represented semantic cells")
    primary_nll = identity_weighted_sum / identity_weight_sum
    cell_nlls = tuple(
        cell_loss_sums[cell] / cell_weight_sums[cell] for cell in sorted(cell_weight_sums)
    )
    balanced_cell_nll = sum(cell_nlls) / len(cell_nlls) if cell_nlls else primary_nll
    family_nlls = {
        family: family_loss_sums[family] / family_weight_sums[family]
        for family in sorted(family_weight_sums)
    }
    metrics = {
        "production_weighted_canonical_successor_nll": primary_nll,
        "balanced_semantic_cell_canonical_successor_nll": balanced_cell_nll,
        "successor_generator_bregman_loss": (generator_weighted_sum / row_count),
        "hazard_bregman_loss": hazard_weighted_sum / row_count,
        "mean_teacher_productive_successor_probability": (
            successor_probability_sum / nonterminal_count
        ),
        "mean_teacher_family_probability": (family_probability_sum / nonterminal_count),
        "mean_teacher_within_family_successor_probability": (
            within_family_probability_sum / nonterminal_count
        ),
        "mean_productive_probability": productive_probability_sum / row_count,
        "mean_predicted_hazard": predicted_hazard_sum / row_count,
        "mean_teacher_hazard": teacher_hazard_sum / row_count,
        "mean_hazard_absolute_error": hazard_absolute_error_sum / row_count,
        "mean_terminal_hazard": (terminal_hazard_sum / terminal_count if terminal_count else 0.0),
        "nonterminal_examples": float(nonterminal_count),
        "terminal_examples": float(terminal_count),
        "represented_semantic_cells": float(len(cell_weight_sums)),
        "represented_teacher_families": float(len(family_nlls)),
        "balanced_family_canonical_successor_nll": (sum(family_nlls.values()) / len(family_nlls)),
    }
    for family in MARK_RULE_NAMES:
        metrics[f"canonical_successor_nll_{family}"] = family_nlls.get(
            family,
            0.0,
        )
        metrics[f"teacher_examples_{family}"] = float(
            sum(
                1
                for rule_name in batch.mark_batch.teacher_rule_names
                if _teacher_family_name(rule_name) == family
            )
        )
        family_weight = family_weight_sums.get(family, 0.0)
        metrics[f"teacher_family_nll_{family}"] = (
            family_gate_loss_sums[family] / family_weight if family_weight > 0.0 else 0.0
        )
        metrics[f"within_family_successor_nll_{family}"] = (
            within_family_loss_sums[family] / family_weight if family_weight > 0.0 else 0.0
        )
    if any(not isfinite(value) for value in metrics.values()):
        raise SuccessorTrainingError("successor validation produced nonfinite metrics")
    return metrics


@dataclass(frozen=True)
class CanonicalSuccessorTrainingObjective:
    """Train successor identity while preserving optional hazard semantics."""

    mode: SuccessorObjectiveMode
    hazard_weight: float
    require_semantic_cells_for_metrics: bool = True
    selection_metric: str = "production_weighted_canonical_successor_nll"

    def __post_init__(self) -> None:
        if self.mode not in {
            "balanced_semantic_cell_productive_identity",
            "productive_identity",
            "productive_identity_plus_hazard",
            "successor_generator_bregman",
        }:
            raise ValueError(f"unknown successor objective mode: {self.mode}")
        if not isfinite(self.hazard_weight) or self.hazard_weight < 0.0:
            raise ValueError("hazard_weight must be finite and nonnegative")
        if self.mode == "productive_identity_plus_hazard":
            if self.hazard_weight <= 0.0:
                raise ValueError("identity-plus-hazard mode requires positive hazard_weight")
        elif self.hazard_weight != 0.0:
            raise ValueError(
                f"{self.mode} requires hazard_weight=0; it does not use a separate hazard term"
            )

    @property
    def name(self) -> str:
        return (
            f"canonical_successor_{self.mode}_v1"
            if self.hazard_weight == 0.0
            else (f"canonical_successor_{self.mode}_hazard_{self.hazard_weight:.12g}_v1")
        )

    def validate_batch(self, batch: Any) -> None:
        if not isinstance(batch, FactorizedSuccessorBatch):
            raise TypeError("canonical-successor objective requires FactorizedSuccessorBatch")
        assert_teachers_in_exact_candidates(batch.mark_batch)

    def mark_batch(self, batch: Any) -> FactorizedMarkBatch:
        self.validate_batch(batch)
        return batch.mark_batch

    def loss(
        self,
        model: FactorizedTraceletRateModel,
        batch: Any,
    ) -> Tensor:
        self.validate_batch(batch)
        prediction = forward_teacher_successor_batch(
            model,
            batch.mark_batch,
            batch.fibers,
            state_supports=batch.state_supports,
        )
        if self.mode == "balanced_semantic_cell_productive_identity":
            return balanced_semantic_cell_productive_identity_loss(
                prediction,
                batch,
            )
        if self.mode == "successor_generator_bregman":
            return factorized_successor_bregman_loss(
                prediction,
                batch.mark_batch,
            )
        identity = factorized_successor_identity_loss(
            prediction,
            batch.mark_batch,
        )
        if self.mode == "productive_identity":
            return identity
        hazard = factorized_hazard_bregman_loss(
            prediction,
            batch.mark_batch,
        )
        return identity + self.hazard_weight * hazard

    def metrics(
        self,
        model: FactorizedTraceletRateModel,
        batch: Any,
        *,
        use_bf16: bool,
        microbatch_size: int | None,
    ) -> dict[str, float]:
        self.validate_batch(batch)
        return factorized_successor_metrics(
            model,
            batch,
            use_bf16=use_bf16,
            microbatch_size=microbatch_size,
            require_semantic_cells=self.require_semantic_cells_for_metrics,
        )


__all__ = [
    "CanonicalSuccessorTrainingObjective",
    "SuccessorObjectiveMode",
    "balanced_semantic_cell_productive_identity_loss",
    "factorized_successor_metrics",
]
