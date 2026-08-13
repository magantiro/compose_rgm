"""Dual oracle accounting for external-baseline comparisons.

Every comparator run logs **two** counters and never substitutes one for the
other:

``benchmark_native``
    Unique **valid canonical** molecules scored. A repeat of a canonical SMILES
    already scored in this run does not increment; an unparseable molecule does
    not increment. This is the only quantity comparable to published PMO
    numbers, because ``Oracle.score_smi`` in ``wenhao-gao/mol_opt`` keys its
    buffer on canonical SMILES and returns 0 for invalid input without charging.

``raw_compute``
    Every oracle **invocation**: duplicates, rejected proposals, invalids,
    particles, prescreened candidates, reranked endpoints and rescores. This is
    the honest efficiency number.

The two differ by orders of magnitude across the methods in the comparator
registry -- MARS has no cache anywhere and rescores its current molecule
whenever a proposal is invalid, while REINVENT caches per scoring component by
design -- so a table that reports one without naming it is misleading in a
predictable direction. Both are always logged; the choice of which to report is
made once, in a caption, and is visible.

A duplicate **request** increments ``raw_compute`` even though the underlying
evaluator is not called. That is deliberate: ``raw_compute`` measures what the
method *demands*, so bolting a cache onto a wasteful method does not silently
improve its efficiency number. ``evaluator_calls`` records the underlying calls
separately, which is what actually costs CPU.

The accountant is the only thing every baseline adapter shares, so it is also
the only place a fairness rule can be enforced rather than merely documented.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem
from rdkit import RDLogger


#: The two frozen counters. A budget must be declared against one of them by
#: name -- there is deliberately no default, because picking the counter after
#: seeing results is the exact failure this module exists to prevent.
BUDGET_COUNTERS = ("benchmark_native", "raw_compute")


class BudgetExhausted(RuntimeError):
    """The declared oracle budget has been spent."""


def canonical_smiles(smiles: str) -> str | None:
    """Canonical SMILES, or ``None`` when RDKit cannot parse the input.

    Sanitization is left at RDKit's default: a molecule that fails valence
    checking is *invalid*, which is a failed oracle proposal rather than a free
    action.
    """
    if not isinstance(smiles, str) or not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule)


@dataclass
class OracleCounts:
    raw_compute: int = 0
    benchmark_native: int = 0
    evaluator_calls: int = 0
    cache_hits: int = 0
    failed_proposals: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "raw_compute": self.raw_compute,
            "benchmark_native": self.benchmark_native,
            "evaluator_calls": self.evaluator_calls,
            "cache_hits": self.cache_hits,
            "failed_proposals": self.failed_proposals,
        }


@dataclass
class OracleAccountant:
    """Wraps a molecule objective and counts it two ways.

    Args:
        evaluate: the frozen objective, ``canonical SMILES -> float``. It is
            called at most once per distinct valid canonical molecule while
            caching is enabled.
        budget: maximum value of the counter named by ``budget_counter``.
        budget_counter: which counter the budget binds. Must be named
            explicitly.
        cache: whether repeated canonical molecules reuse the stored score.
            Allowed for every method by the fairness contract, and required to
            use identical semantics across methods.
        invalid_score: what an unparseable molecule scores. PMO returns 0.0.
    """

    evaluate: Callable[[str], float]
    budget: int
    budget_counter: str
    cache: bool = True
    invalid_score: float = 0.0
    counts: OracleCounts = field(default_factory=OracleCounts)
    _scores: dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.budget_counter not in BUDGET_COUNTERS:
            raise ValueError(
                f"budget_counter must be one of {BUDGET_COUNTERS}, "
                f"got {self.budget_counter!r}"
            )
        if self.budget <= 0:
            raise ValueError("budget must be positive")

    # ---- Budget ----

    @property
    def spent(self) -> int:
        return getattr(self.counts, self.budget_counter)

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.budget

    # ---- Scoring ----

    def score(self, smiles: str) -> float:
        """Score one candidate, charging both counters.

        Raises:
            BudgetExhausted: if the declared budget is already spent. Callers
                check ``exhausted`` to terminate cleanly; the exception exists
                so that a driver which forgets to cannot overspend silently.
        """
        if self.exhausted:
            raise BudgetExhausted(
                f"{self.budget_counter} budget of {self.budget} is spent"
            )

        # Every request is an invocation, cached or not.
        self.counts.raw_compute += 1

        key = canonical_smiles(smiles)
        if key is None:
            self.counts.failed_proposals += 1
            return self.invalid_score

        if self.cache and key in self._scores:
            self.counts.cache_hits += 1
            return self._scores[key]

        self.counts.benchmark_native += 1
        self.counts.evaluator_calls += 1
        value = float(self.evaluate(key))
        if self.cache:
            self._scores[key] = value
        return value

    def score_many(self, batch: list[str]) -> list[float]:
        """Score a batch, stopping cleanly at the budget rather than raising."""
        results: list[float] = []
        for smiles in batch:
            if self.exhausted:
                break
            results.append(self.score(smiles))
        return results

    # ---- Reporting ----

    @property
    def demand_ratio(self) -> float:
        """``raw_compute / benchmark_native`` -- the cache-and-duplicate rate.

        Reported per method by the fairness contract. It differs by orders of
        magnitude across these baselines, and hiding it is how an unfair
        comparison survives review.
        """
        if self.counts.benchmark_native == 0:
            return float("nan")
        return self.counts.raw_compute / self.counts.benchmark_native

    def manifest(self) -> dict[str, Any]:
        counts = self.counts.as_dict()
        return {
            "counters": counts,
            "budget": self.budget,
            "budget_counter": self.budget_counter,
            "budget_exhausted": self.exhausted,
            "cache_enabled": self.cache,
            "demand_ratio_raw_over_benchmark_native": self.demand_ratio,
            "distinct_valid_molecules": len(self._scores) if self.cache else None,
            "invariants": {
                "raw_compute >= benchmark_native": (
                    counts["raw_compute"] >= counts["benchmark_native"]
                ),
                "raw_compute == benchmark_native + cache_hits + failed_proposals": (
                    counts["raw_compute"]
                    == counts["benchmark_native"]
                    + counts["cache_hits"]
                    + counts["failed_proposals"]
                ),
                "evaluator_calls == benchmark_native": (
                    counts["evaluator_calls"] == counts["benchmark_native"]
                ),
            },
        }


def silence_rdkit() -> None:
    """Suppress RDKit's parse warnings; invalids are counted, not printed."""
    RDLogger.DisableLog("rdApp.*")
