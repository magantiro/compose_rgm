"""Rate-space Bregman objectives."""

from __future__ import annotations

import torch
from torch import Tensor


def rate_bregman_loss(
    predicted_total_hazard: Tensor,
    predicted_teacher_successor_rate: Tensor,
    teacher_successor_rate: Tensor,
    *,
    eps: float = 1e-12,
    reduction: str = "mean",
) -> Tensor:
    """Poisson-KL Generator Matching loss, up to teacher-only constants.

    The predicted successor rate must already aggregate every marked rewrite
    alias that reaches the teacher successor. The objective per example is

        predicted total hazard
        - teacher successor rate * log(predicted teacher successor rate).

    This interface deliberately separates complete-successor semantics from
    any particular marked-action parameterization.
    """

    total = torch.as_tensor(predicted_total_hazard)
    predicted = torch.as_tensor(
        predicted_teacher_successor_rate,
        device=total.device,
        dtype=total.dtype,
    )
    teacher = torch.as_tensor(
        teacher_successor_rate,
        device=total.device,
        dtype=total.dtype,
    )
    total, predicted, teacher = torch.broadcast_tensors(total, predicted, teacher)
    if torch.any(total < 0) or torch.any(predicted < 0) or torch.any(teacher < 0):
        raise ValueError("all hazards and rates must be non-negative")
    if torch.any(predicted > total + 1e-6):
        raise ValueError("a successor rate cannot exceed total predicted hazard")

    per_example = total - teacher * torch.log(predicted.clamp_min(eps))
    if reduction == "none":
        return per_example
    if reduction == "mean":
        return per_example.mean()
    if reduction == "sum":
        return per_example.sum()
    raise ValueError("reduction must be one of: none, mean, sum")


def multi_successor_rate_bregman_loss(
    predicted_total_hazard: Tensor,
    predicted_successor_rates: Tensor,
    teacher_successor_rates: Tensor,
    *,
    eps: float = 1e-12,
    reduction: str = "mean",
) -> Tensor:
    """Poisson-KL rate loss for multiple nonzero teacher successors.

    The last dimension enumerates complete chemical successors. Predicted rates
    must already aggregate every marked-action alias. Successors with zero
    teacher rate may be omitted because their predicted mass is still charged
    through ``predicted_total_hazard``.
    """

    total = torch.as_tensor(predicted_total_hazard)
    predicted = torch.as_tensor(
        predicted_successor_rates,
        device=total.device,
        dtype=total.dtype,
    )
    teacher = torch.as_tensor(
        teacher_successor_rates,
        device=total.device,
        dtype=total.dtype,
    )
    predicted, teacher = torch.broadcast_tensors(predicted, teacher)
    if predicted.ndim == 0:
        predicted = predicted.unsqueeze(0)
        teacher = teacher.unsqueeze(0)
    if torch.any(total < 0) or torch.any(predicted < 0) or torch.any(teacher < 0):
        raise ValueError("all hazards and rates must be non-negative")
    teacher_prediction = predicted.sum(dim=-1)
    if torch.any(teacher_prediction > total + 1e-6):
        raise ValueError("selected successor rates cannot exceed total hazard")
    per_example = total - (
        teacher * torch.log(predicted.clamp_min(eps))
    ).sum(dim=-1)
    if reduction == "none":
        return per_example
    if reduction == "mean":
        return per_example.mean()
    if reduction == "sum":
        return per_example.sum()
    raise ValueError("reduction must be one of: none, mean, sum")
