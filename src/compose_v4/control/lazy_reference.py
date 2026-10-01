"""Exact conditional-mixture draws without executing every candidate product.

This utility does not define molecular support or chemistry. The production
option kernel supplies candidate indices, mixture weights and product checks.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Generic, TypeVar

import numpy as np

State = TypeVar("State")


class LazyReferenceBranch(Generic[State]):
    def __init__(
        self,
        indices: tuple[int, ...],
        weights: tuple[float, ...],
        uniform_fraction: float,
        product: Callable[[int], State | None],
    ):
        w = np.asarray(weights, dtype=float)
        if (
            len(indices) != len(w)
            or len(set(indices)) != len(indices)
            or not np.isfinite(w).all()
            or (w < 0).any()
            or not 0 <= uniform_fraction <= 1
        ):
            raise ValueError("invalid lazy conditional-mixture support or weights")
        self.indices, self.weights = indices, w
        self.uniform_fraction, self.product = uniform_fraction, product
        self.products: dict[int, State | None] = {}

    def resolve(self, index: int) -> State | None:
        if index not in self.products:
            # A budget stop must not become a cached invalid product.
            self.products[index] = self.product(index)
        return self.products[index]

    def has_product(self) -> bool:
        if any(s is not None for s in self.products.values()):
            return True
        for at in np.argsort(-self.weights, kind="stable"):
            if self.resolve(self.indices[int(at)]) is not None:
                return True
        return False

    def sample(self, rng: np.random.Generator) -> State | None:
        # Condition EACH component on clean products before mixing. Restarting
        # the mixture after rejection would incorrectly change its weights.
        uniform = rng.random() < self.uniform_fraction
        remaining = [
            at
            for at, index in enumerate(self.indices)
            if index not in self.products or self.products[index] is not None
        ]
        while remaining:
            weights = np.ones(len(remaining)) if uniform else self.weights[remaining]
            if weights.sum() == 0:
                # Matches the production generic law's all-zero-clean fallback.
                weights = np.ones(len(remaining))
            selected = int(rng.choice(len(remaining), p=weights / weights.sum()))
            at = remaining[selected]
            successor = self.resolve(self.indices[at])
            if successor is not None:
                return successor
            del remaining[selected]  # only a proven invalid product is removed
        return None


class LazyReferenceRow(Generic[State]):
    """Uniform applicable-branch choice, then the branch's conditional mixture."""

    def __init__(self, branches: dict[Hashable, LazyReferenceBranch[State]]):
        self.branches = branches
        self._applicable: tuple[Hashable, ...] | None = None

    def has_product(self) -> bool:
        return any(branch.has_product() for branch in self.branches.values())

    def sample(self, rng: np.random.Generator) -> State | None:
        if self._applicable is None:
            self._applicable = tuple(
                key for key, branch in self.branches.items() if branch.has_product()
            )
        if not self._applicable:
            return None
        key = self._applicable[int(rng.integers(len(self._applicable)))]
        return self.branches[key].sample(rng)
