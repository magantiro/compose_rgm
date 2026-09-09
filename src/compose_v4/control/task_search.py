"""Budgeted, importance-weighted search over a supplied executable hierarchy.

This is approximate planning, not exact posterior sampling. Full reference
rows are retained; only completed rollouts supply return observations.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable
from dataclasses import asdict, dataclass
from typing import Generic, TypeVar

import numpy as np

from compose_v4.control.continuation import (
    ContinuationBudgetExceeded,
    ReferenceRow,
    _nonnegative_integer,
    _tilted,
)

State = TypeVar("State")


@dataclass(frozen=True)
class SearchRow(Generic[State]):
    successors: tuple[State, ...]
    labels: tuple[str, ...]
    core: tuple[float, ...]
    floor: tuple[float, ...]
    epsilon: float

    def __post_init__(self):
        ReferenceRow(self.successors, self.core)
        ReferenceRow(self.successors, self.floor)
        if len(self.labels) != len(self.successors):
            raise ValueError("search labels must align with the full successor row")
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon < 1:
            raise ValueError("a search row requires an exploration floor in (0, 1)")

    @property
    def reference(self) -> np.ndarray:
        return (1 - self.epsilon) * np.asarray(self.core) + self.epsilon * np.asarray(self.floor)


@dataclass
class ReturnEstimate:
    count: int = 0
    log_weight: float = -math.inf
    log_weighted_value: float = -math.inf

    def add(self, value: float, log_weight: float):
        if not math.isfinite(log_weight) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("return observations require finite log weights and values in [0, 1]")
        self.count += 1
        self.log_weight = float(np.logaddexp(self.log_weight, log_weight))
        if value > 0:
            self.log_weighted_value = float(
                np.logaddexp(self.log_weighted_value, log_weight + math.log(value))
            )

    def mean(self, default: float, pseudo_count: float = 0) -> float:
        if not self.count:
            return default
        numerator, denominator = self.log_weighted_value, self.log_weight
        if pseudo_count:
            denominator = float(np.logaddexp(denominator, math.log(pseudo_count)))
            if default > 0:
                numerator = float(np.logaddexp(numerator, math.log(pseudo_count * default)))
        return min(1.0, math.exp(numerator - denominator))

    def payload(self) -> dict:
        # JSON never contains Infinity, including an observed all-zero return.
        return {k: v if math.isfinite(v) else None for k, v in asdict(self).items()}


@dataclass
class SearchWork:
    rows: int = 0
    row_hits: int = 0
    terminal_evaluations: int = 0
    rollouts_started: int = 0
    rollouts_completed: int = 0
    rollouts_interrupted: int = 0
    dead_ends: int = 0


class TaskSearch(Generic[State]):
    """A persistent-in-run search graph with full-support conditional decisions.

    The row and terminal callbacks are deterministic for snapshot_id. A caller
    meters public executor work. Keys must contain all transition-relevant state
    including remaining primitive budget. Planning RNG is separate from acting.
    Return estimates are self-normalized importance estimates, NOT confidence
    bounds; adaptive observations are corrected for their actual suffix policy.
    """

    def __init__(
        self,
        reference: Callable[[State], SearchRow[State]],
        terminal: Callable[[State], float | None],
        state_key: Callable[[State], Hashable],
        *,
        snapshot_id: str,
        seed: int,
        max_rows: int = 128,
        max_terminals: int = 256,
        max_rollouts: int = 32,
        max_path_steps: int = 48,
    ):
        if not snapshot_id:
            raise ValueError("immutable reference/objective snapshot_id is required")
        self.reference, self.terminal, self.state_key = reference, terminal, state_key
        self.snapshot_id = snapshot_id
        self.rng = np.random.default_rng(_nonnegative_integer(seed, "seed"))
        for name, value in (
            ("max_rows", max_rows),
            ("max_terminals", max_terminals),
            ("max_rollouts", max_rollouts),
            ("max_path_steps", max_path_steps),
        ):
            setattr(self, name, _nonnegative_integer(value, name))
        self.work = SearchWork()
        self.rows: dict[Hashable, SearchRow[State]] = {}
        self.returns: dict[Hashable, ReturnEstimate] = {}
        self.terminals: dict[Hashable, float] = {}

    def row(self, state: State) -> SearchRow[State]:
        key = self.state_key(state)
        if key in self.rows:
            self.work.row_hits += 1
            return self.rows[key]
        if self.work.rows >= self.max_rows:
            raise ContinuationBudgetExceeded("hierarchical search row budget exhausted")
        self.work.rows += 1  # attempted rows, including interrupted construction
        result = self.reference(state)
        if not isinstance(result, SearchRow):
            raise TypeError("hierarchy callback must return SearchRow")
        self.rows[key] = result
        return result

    def terminal_value(self, state: State) -> float | None:
        key = self.state_key(state)
        if key in self.terminals:
            return self.terminals[key]
        # The callback checks terminal status before evaluating any task model.
        # Capacity is checked before the callback to avoid unmetered evaluation.
        if self.work.terminal_evaluations >= self.max_terminals:
            raise ContinuationBudgetExceeded("hierarchical terminal budget exhausted")
        value = self.terminal(state)
        if value is not None:
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("terminal task value must be finite in [0, 1]")
            self.work.terminal_evaluations += 1
            self.terminals[key] = float(value)
        return value

    def decision(self, state: State) -> dict:
        row = self.row(state)
        if not row.successors:
            return {"status": "no_admissible_action", "probabilities": [], "kl": 0.0}
        parent = self.returns.get(self.state_key(state), ReturnEstimate())
        default = parent.mean(1.0)
        estimates = [self.returns.get(self.state_key(s), ReturnEstimate()) for s in row.successors]
        values = np.asarray(
            [
                self.terminals.get(self.state_key(s), e.mean(default, pseudo_count=2.0))
                for s, e in zip(row.successors, estimates, strict=True)
            ]
        )
        if not values.any():
            q, eta = row.reference, 0.0
            status = "no_observed_value"
        else:
            tilted, eta, _ = _tilted(row.core, values, kappa=1.0, exploration=0.0)
            q = (1 - row.epsilon) * tilted + row.epsilon * np.asarray(row.floor)
            status = "task_guided" if np.ptp(values) > 1e-12 else "reference_no_contrast"
        p = row.reference
        live = q > 0
        kl = float(np.sum(q[live] * np.log(q[live] / p[live])))
        if not np.isclose(q.sum(), 1, atol=1e-12) or kl > 1 + 1e-10:
            raise RuntimeError("hierarchical probability or KL invariant violated")
        return {
            "status": status,
            "probabilities": q.tolist(),
            "reference": p.tolist(),
            "core": list(row.core),
            "floor": list(row.floor),
            "epsilon": row.epsilon,
            "values": values.tolist(),
            "visits": [e.count for e in estimates],
            "labels": list(row.labels),
            "kl": kl,
            "eta": eta,
            "snapshot_id": self.snapshot_id,
        }

    def rollout(self, root: State) -> bool:
        if self.work.rollouts_started >= self.max_rollouts:
            return False
        self.work.rollouts_started += 1
        state, path = root, []
        try:
            for _ in range(self.max_path_steps + 1):
                value = self.terminal_value(state)
                if value is not None:
                    break
                row = self.row(state)
                if not row.successors:
                    self.work.dead_ends += 1
                    value = 0.0
                    break
                if len(path) >= self.max_path_steps:
                    raise ContinuationBudgetExceeded("hierarchical rollout path bound exhausted")
                q = self.decision(state)["probabilities"]
                index = int(self.rng.choice(len(q), p=q))
                path.append((self.state_key(state), math.log(row.reference[index] / q[index])))
                state = row.successors[index]
        except ContinuationBudgetExceeded:
            self.work.rollouts_interrupted += 1
            return False
        except BaseException:
            # Account administrative/native interruption without suppressing it.
            self.work.rollouts_interrupted += 1
            raise
        self.returns.setdefault(self.state_key(state), ReturnEstimate()).add(value, 0.0)
        log_weight = 0.0
        for key, log_ratio in reversed(path):
            log_weight += log_ratio
            self.returns.setdefault(key, ReturnEstimate()).add(value, log_weight)
        self.work.rollouts_completed += 1
        return True

    def plan(self, root: State, attempts: int) -> int:
        completed = 0
        for _ in range(_nonnegative_integer(attempts, "attempts")):
            if not self.rollout(root):
                break
            completed += 1
        return completed

    def receipt(self) -> dict:
        return {
            "schema_version": "task_search_work_v1",
            "snapshot_id": self.snapshot_id,
            **asdict(self.work),
            "cached_rows": len(self.rows),
            "rng_state": self.rng.bit_generator.state,
            "estimator": "adaptive_suffix_self_normalized_importance_with_shrinkage",
            "exact_doob": False,
            "uncertainty_calibrated": False,
        }
