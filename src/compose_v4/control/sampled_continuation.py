"""Fixed-sample reference rollouts, not MCTS or an exact Doob transform.

Every supported root state gets the same prospectively fixed sample count.
Only complete comparisons may guide. Confidence bounds concern Monte Carlo
error under the supplied immutable reference, not task-model calibration.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Generic

import numpy as np

from compose_v4.control.continuation import (
    ContinuationBudgetExceeded,
    ControlledDecision,
    ReferenceRow,
    State,
    _nonnegative_integer,
    _tilted,
)


@dataclass
class SampledWork:
    expansions: int = 0
    row_visits: int = 0
    row_cache_hits: int = 0
    terminal_evaluations: int = 0
    terminal_cache_hits: int = 0
    rollouts_started: int = 0
    rollouts_completed: int = 0
    dead_ends: int = 0
    budget_abstentions: int = 0


@dataclass(frozen=True)
class SampledEstimate:
    means: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    samples: tuple[int, ...]
    exact_terminal: bool


@dataclass(frozen=True)
class SampledDecision:
    decision: ControlledDecision
    estimate: SampledEstimate | None
    alpha: float


class SampledContinuation(Generic[State]):
    """Linear-in-horizon sampling with cached exact rows and terminal evaluations.

    ``state_key`` has the same full augmented-state contract as the exact solver.
    Duplicate exact root states share estimates; slot-distinct states do not.
    The caller meters all expensive executor work, including law enumeration.
    Limits are lifetime ceilings, including failed comparisons. The supplied
    reference and terminal callback must be deterministic for ``snapshot_id``.
    No learned predictions are treated as calibrated probabilities here.
    """

    def __init__(
        self,
        reference: Callable[[State], ReferenceRow[State]],
        terminal_weight: Callable[[State], float],
        state_key: Callable[[State], Hashable],
        *,
        snapshot_id: str,
        seed: int,
        samples_per_successor: int,
        alpha: float,
        max_expansions: int,
        max_terminal_evaluations: int,
        max_rollouts: int,
        memoize: bool = True,
    ) -> None:
        if not snapshot_id:
            raise ValueError("an immutable reference/objective snapshot_id is required")
        self.samples_per_successor = _nonnegative_integer(
            samples_per_successor, "samples_per_successor"
        )
        if self.samples_per_successor == 0:
            raise ValueError("samples_per_successor must be positive")
        if not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be finite in (0, 1)")
        self.seed = _nonnegative_integer(seed, "seed")
        self.rng = np.random.default_rng(self.seed)
        self.reference, self.terminal_weight, self.state_key = reference, terminal_weight, state_key
        self.snapshot_id, self.alpha, self.memoize = snapshot_id, alpha, memoize
        self.max_expansions = _nonnegative_integer(max_expansions, "max_expansions")
        self.max_terminal_evaluations = _nonnegative_integer(
            max_terminal_evaluations, "max_terminal_evaluations"
        )
        self.max_rollouts = _nonnegative_integer(max_rollouts, "max_rollouts")
        self.work = SampledWork()
        self._rows: dict[Hashable, ReferenceRow[State]] = {}
        self._terminals: dict[Hashable, float] = {}

    def _terminal(self, state: State) -> float:
        key = self.state_key(state)
        if self.memoize and key in self._terminals:
            self.work.terminal_cache_hits += 1
            return self._terminals[key]
        if self.work.terminal_evaluations >= self.max_terminal_evaluations:
            raise ContinuationBudgetExceeded("terminal-evaluation budget exhausted")
        self.work.terminal_evaluations += 1
        value = float(self.terminal_weight(state))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"terminal weight must be finite in [0, 1], got {value}")
        if self.memoize:
            self._terminals[key] = value
        return value

    def _rollout(self, state: State, remaining: int) -> float:
        if self.work.rollouts_started >= self.max_rollouts:
            raise ContinuationBudgetExceeded("rollout budget exhausted")
        self.work.rollouts_started += 1
        for _ in range(remaining):
            key = self.state_key(state)
            self.work.row_visits += 1
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
                if self.memoize:
                    self._rows[key] = row
            if not row.successors:
                self.work.dead_ends += 1
                self.work.rollouts_completed += 1
                return 0.0  # genuine empty reference support, not a budget stop
            state = row.successors[int(self.rng.choice(len(row.successors), p=row.probabilities))]
        value = self._terminal(state)
        self.work.rollouts_completed += 1
        return value

    def estimate(self, row: ReferenceRow[State], remaining: int) -> SampledEstimate:
        remaining = _nonnegative_integer(remaining, "remaining")
        if remaining < 1:
            raise ValueError("a decision requires at least one remaining transition")
        groups: dict[Hashable, list[int]] = {}
        for index, (state, probability) in enumerate(zip(row.successors, row.probabilities)):
            if probability > 0:
                groups.setdefault(self.state_key(state), []).append(index)
        count = self.samples_per_successor if remaining > 1 else 1
        if remaining > 1 and len(groups) * count > self.max_rollouts - self.work.rollouts_started:
            raise ContinuationBudgetExceeded("fixed complete comparison exceeds rollout budget")
        # Balance by sample index; do not exhaust one action's sample allocation
        # first. Nevertheless an interrupted comparison returns NO estimates.
        sums = np.zeros(len(row.successors))
        for _ in range(count):
            for indices in groups.values():
                state = row.successors[indices[0]]
                value = (
                    self._rollout(state, remaining - 1) if remaining > 1 else self._terminal(state)
                )
                sums[indices] += value
        means = sums / count
        radius = (
            math.sqrt(math.log(2 * len(groups) / self.alpha) / (2 * count))
            if groups and remaining > 1
            else 0.0
        )
        lower, upper, samples = (
            np.zeros(len(sums)),
            np.zeros(len(sums)),
            np.zeros(len(sums), dtype=int),
        )
        for indices in groups.values():
            lower[indices] = np.maximum(0, means[indices] - radius)
            upper[indices] = np.minimum(1, means[indices] + radius)
            samples[indices] = count
        return SampledEstimate(
            tuple(means), tuple(lower), tuple(upper), tuple(int(n) for n in samples), remaining == 1
        )


def sampled_continuation_decision(
    row: ReferenceRow[State],
    remaining: int,
    continuation: SampledContinuation[State],
    *,
    fallback_values: Sequence[float],
    kappa: float = 1.0,
    exploration: float = 0.1,
) -> SampledDecision:
    """Conservative sampled guidance with per-comparison simultaneous bounds.

    Bounds follow Hoeffding plus a union bound over distinct supported exact
    root states. This is not an across-round confidence guarantee or calibration
    of a learned terminal objective. Probability is conditional on this sampled
    estimate; averaging over planner randomness gives a different law.
    """
    if _nonnegative_integer(remaining, "remaining") < 1:
        raise ValueError("a decision requires at least one remaining transition")
    if kappa != 1.0:
        raise ValueError("this controller freezes kappa=1.0")
    if not math.isfinite(exploration) or not 0 <= exploration <= 1:
        raise ValueError("exploration must be finite in [0, 1]")
    if not row.successors:
        return SampledDecision(
            ControlledDecision((), "no_admissible_action", None, 0, 0), None, continuation.alpha
        )
    base, base_eta, base_kl = _tilted(
        row.probabilities, fallback_values, kappa=kappa, exploration=exploration
    )

    def fallback(status, estimate=None):
        return SampledDecision(
            ControlledDecision(tuple(base), status, None, base_eta, base_kl),
            estimate,
            continuation.alpha,
        )

    if sum(p > 0 for p in row.probabilities) == 1:
        return fallback("only_one_successor")
    try:
        estimate = continuation.estimate(row, remaining)
    except ContinuationBudgetExceeded:
        continuation.work.budget_abstentions += 1
        return fallback("budget_abstention")
    live = np.asarray(row.probabilities) > 0
    if np.max(np.asarray(estimate.means)[live]) == 0:
        return fallback(
            "zero_terminal_mass" if estimate.exact_terminal else "no_observed_success", estimate
        )
    if not estimate.exact_terminal and max(np.asarray(estimate.lower)[live]) <= min(
        np.asarray(estimate.upper)[live]
    ):
        return fallback("insufficient_evidence", estimate)
    q, eta, kl = _tilted(row.probabilities, estimate.lower, kappa=kappa, exploration=exploration)
    return SampledDecision(
        ControlledDecision(
            tuple(q),
            "guided_terminal" if estimate.exact_terminal else "guided_sampled",
            estimate.lower,
            eta,
            kl,
        ),
        estimate,
        continuation.alpha,
    )
