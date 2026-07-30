"""Dependency-neutral objective interface for the shared factorized trainer."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol, runtime_checkable

from torch import Tensor

from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedMarkBatch,
    FactorizedTraceletRateModel,
)


@runtime_checkable
class FactorizedTrainingObjective(Protocol):
    """Adapt one aligned batch type to the existing optimizer machinery."""

    name: str
    selection_metric: str

    def validate_batch(self, batch: Any) -> None:
        """Fail before scoring when a batch violates objective support."""

    def mark_batch(self, batch: Any) -> FactorizedMarkBatch:
        """Return the underlying model-input batch for shared diagnostics."""

    def loss(
        self,
        model: FactorizedTraceletRateModel,
        batch: Any,
    ) -> Tensor:
        """Return one differentiable scalar training objective."""

    def metrics(
        self,
        model: FactorizedTraceletRateModel,
        batch: Any,
        *,
        use_bf16: bool,
        microbatch_size: int | None,
    ) -> dict[str, float]:
        """Evaluate one frozen validation batch."""


TrainingLoaderFactory = Callable[[int], Iterable[Any]]
GradientAuditCallback = Callable[
    [FactorizedTraceletRateModel, FactorizedMarkBatch, int, Tensor],
    None,
]


__all__ = [
    "FactorizedTrainingObjective",
    "GradientAuditCallback",
    "TrainingLoaderFactory",
]
