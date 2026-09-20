"""Bounded continuation backups over an explicitly supplied reference process.

No generator, objective, support restriction, or molecular quotient is defined
here. Callers supply complete reference rows and exact augmented-state keys.
An exact completed backup is distinct from a KL-limited sampled controller;
budget exhaustion never turns a partially expanded tree into guidance.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from compose_v4.control.region_rewrite import kl_tilt

State = TypeVar("State")


class ContinuationBudgetExceeded(RuntimeError):
    """Incomplete computation, not evidence of an unreachable terminal event."""


@dataclass(frozen=True)
class ReferenceRow(Generic[State]):
    successors: tuple[State, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        p = np.asarray(self.probabilities, dtype=float)
        if p.ndim != 1 or len(p) != len(self.successors):
            raise ValueError("reference successors and probabilities must be aligned 1D sequences")
        if not np.isfinite(p).all() or (p < 0).any():
            raise ValueError("reference probabilities must be finite and nonnegative")
        if len(p) and not math.isclose(float(p.sum()), 1.0, abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError(f"reference row must sum to one, got {p.sum()}")


@dataclass
class ContinuationWork:
    expansions: int = 0
    edges: int = 0
    terminal_evaluations: int = 0
    row_cache_hits: int = 0
    value_cache_hits: int = 0
    budget_abstentions: int = 0


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")
    return int(value)


class FiniteHorizonContinuation(Generic[State]):
    """Exact reference expectation if the whole requested backup fits the budget.

    Instances belong to ONE immutable reference/objective snapshot. The supplied
    key must include every variable affecting future transitions or terminal
    weights (including option phase, region context and lineage). Remaining
    budget is added here. An empty row contributes zero, not a self-loop.

    Limits cover this instance's entire lifetime, including failed requests.
    The reference callback must additionally meter expensive executor work.
    No timer is used as a hidden, arm-specific scientific stopping rule.
    """

    def __init__(
        self,
        reference: Callable[[State], ReferenceRow[State]],
        terminal_weight: Callable[[State], float],
        state_key: Callable[[State], Hashable],
        *,
        snapshot_id: str,
        max_expansions: int,
        max_terminal_evaluations: int,
        memoize: bool = True,
    ) -> None:
        if not snapshot_id:
            raise ValueError("an immutable reference/objective snapshot_id is required")
        self.reference = reference
        self.terminal_weight = terminal_weight
        self.state_key = state_key
        self.snapshot_id = snapshot_id
        self.max_expansions = _nonnegative_integer(max_expansions, "max_expansions")
        self.max_terminal_evaluations = _nonnegative_integer(
            max_terminal_evaluations, "max_terminal_evaluations"
        )
        self.memoize = memoize
        self.work = ContinuationWork()
        self._rows: dict[Hashable, ReferenceRow[State]] = {}
        self._values: dict[tuple[Hashable, int], float] = {}

    def value(self, state: State, remaining: int) -> float:
        remaining = _nonnegative_integer(remaining, "remaining")
        key = self.state_key(state)
        hash(key)  # fail immediately on a mutable/unhashable key
        cache_key = key, remaining
        if self.memoize and cache_key in self._values:
            self.work.value_cache_hits += 1
            return self._values[cache_key]
        if remaining == 0:
            if self.work.terminal_evaluations >= self.max_terminal_evaluations:
                raise ContinuationBudgetExceeded("terminal-evaluation budget exhausted")
            self.work.terminal_evaluations += 1
            result = float(self.terminal_weight(state))
            if not math.isfinite(result) or not 0 <= result <= 1:
                raise ValueError(f"terminal weight must be finite in [0, 1], got {result}")
        else:
            if self.memoize and key in self._rows:
                row = self._rows[key]
                self.work.row_cache_hits += 1
            else:
                if self.work.expansions >= self.max_expansions:
                    raise ContinuationBudgetExceeded("reference-expansion budget exhausted")
                self.work.expansions += 1
                row = self.reference(state)
                if not isinstance(row, ReferenceRow):
                    raise TypeError("reference callback must return a ReferenceRow")
                self.work.edges += len(row.successors)
                if self.memoize:
                    self._rows[key] = row
            result = math.fsum(
                p * self.value(s, remaining - 1)
                for s, p in zip(row.successors, row.probabilities)
                if p > 0
            )
        if self.memoize:
            self._values[cache_key] = result
        return result

    def successor_values(self, row: ReferenceRow[State], remaining: int) -> np.ndarray:
        """Values AFTER taking one transition; returns nothing partial on failure."""
        remaining = _nonnegative_integer(remaining, "remaining")
        if remaining < 1:
            raise ValueError("a decision requires at least one remaining transition")
        return np.asarray(
            [
                self.value(state, remaining - 1) if p > 0 else 0.0
                for state, p in zip(row.successors, row.probabilities)
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class ControlledDecision:
    probabilities: tuple[float, ...]
    status: str
    successor_values: tuple[float, ...] | None
    eta: float
    kl: float


def _tilted(
    base: Sequence[float], values: Sequence[float], *, kappa: float, exploration: float
) -> tuple[np.ndarray, float, float]:
    p, h = np.asarray(base, dtype=float), np.asarray(values, dtype=float)
    if h.shape != p.shape or not np.isfinite(h).all() or (h < 0).any():
        raise ValueError("guidance values must be aligned, finite and nonnegative")
    q = np.zeros_like(p)
    positive = p > 0
    controlled, eta, _, _ = kl_tilt(p[positive], h[positive], kappa=kappa)
    q[positive] = (1 - exploration) * controlled + exploration * p[positive]
    live = q > 0
    kl = float(np.sum(q[live] * np.log(q[live] / p[live])))
    if kl > kappa + 1e-10:
        raise RuntimeError(f"mixed controlled law exceeds its KL budget: {kl} > {kappa}")
    return q, float(eta), kl


def continuation_decision(
    row: ReferenceRow[State],
    remaining: int,
    continuation: FiniteHorizonContinuation[State],
    *,
    fallback_values: Sequence[float],
    kappa: float = 1.0,
    exploration: float = 0.1,
) -> ControlledDecision:
    """KL-limited guidance, with a declared baseline on zero mass or budget failure.

    Uses the qualified tilt, including its eta cap and 1e-12 value clipping;
    never describes that regularization as an exact Doob transform. The returned
    probability includes exploration and is the conditional sampling law.
    """
    _nonnegative_integer(remaining, "remaining")
    if remaining < 1:
        raise ValueError("a decision requires at least one remaining transition")
    if kappa != 1.0:
        raise ValueError("this controller freezes kappa=1.0")
    if not math.isfinite(exploration) or not 0 <= exploration <= 1:
        raise ValueError("exploration must be finite in [0, 1]")
    if not row.successors:
        return ControlledDecision((), "no_admissible_action", None, 0.0, 0.0)
    baseline, base_eta, base_kl = _tilted(
        row.probabilities, fallback_values, kappa=kappa, exploration=exploration
    )
    try:
        h = continuation.successor_values(row, remaining)
    except ContinuationBudgetExceeded:
        continuation.work.budget_abstentions += 1
        return ControlledDecision(tuple(baseline), "budget_abstention", None, base_eta, base_kl)
    if float(np.dot(row.probabilities, h)) == 0:
        return ControlledDecision(
            tuple(baseline), "zero_terminal_mass", tuple(h), base_eta, base_kl
        )
    q, eta, kl = _tilted(row.probabilities, h, kappa=kappa, exploration=exploration)
    return ControlledDecision(tuple(q), "guided", tuple(h), eta, kl)


def exact_doob_distribution(
    row: ReferenceRow[State], successor_values: Sequence[float]
) -> tuple[float, ...] | None:
    """Verification reference only; None means zero normalizer, never epsilon repair."""
    h = np.asarray(successor_values, dtype=float)
    p = np.asarray(row.probabilities, dtype=float)
    if h.shape != p.shape or not np.isfinite(h).all() or (h < 0).any():
        raise ValueError("successor values must be aligned, finite and nonnegative")
    mass = p * h
    z = float(mass.sum())
    return tuple(mass / z) if z > 0 else None
