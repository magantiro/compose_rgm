"""Three-counter oracle accounting for external-baseline comparisons.

Every comparator run logs **three** counters and never substitutes one for
another:

``unique_valid_canonical_evaluations``
    Distinct valid canonical molecules evaluated in this run. A repeat of a
    canonical SMILES already seen does not increment; an unparseable molecule
    does not increment. This is the **benchmark-native** number and the only
    quantity comparable to published PMO results, because ``Oracle.score_smi``
    in ``wenhao-gao/mol_opt`` keys its buffer on canonical SMILES and returns 0
    for invalid input without charging.

``oracle_requests``
    Every scoring request the algorithm makes, including duplicates, rejected
    proposals and invalids. This is **algorithmic demand, not CPU.** A duplicate
    request increments it even when the cache serves it.

``evaluator_calls``
    Actual expensive oracle executions after caching. **Real work.**

The conceptual invariant these three exist to encode:

    Caching may reduce evaluator work, but it cannot erase wasteful algorithmic
    requests.

That is why a cache hit still increments ``oracle_requests``. If it did not,
bolting a cache onto a method that re-proposes the same molecule a thousand
times would make it look efficient, and the method's actual search behaviour
would become invisible. ``evaluator_calls`` is where the saving legitimately
shows up. Neither number is "compute" in the literal sense -- for that, report
wall and core time separately.

The three differ by orders of magnitude across the comparator registry: MARS has
no cache anywhere and rescores its current molecule whenever a proposal is
invalid, while REINVENT caches per scoring component by design. A table that
reports one counter without naming it is misleading in a predictable direction.

The accountant is the only object every baseline adapter shares, so it is also
the only place a fairness rule can be enforced rather than merely documented.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem
from rdkit import RDLogger


#: Counters a budget may be declared against, by name. There is deliberately no
#: default: picking the counter after seeing results is the exact failure this
#: module exists to prevent. The first two are the reporting conventions; the
#: third is available for budgeting genuine evaluator work.
BUDGET_COUNTERS = (
    "unique_valid_canonical_evaluations",
    "oracle_requests",
    "evaluator_calls",
)

#: The PMO-compatible convention. Use only this against published PMO numbers.
BENCHMARK_NATIVE_COUNTER = "unique_valid_canonical_evaluations"


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
    #: Every scoring request the algorithm makes. Algorithmic demand, not CPU.
    oracle_requests: int = 0
    #: Distinct valid canonical molecules. The benchmark-native number.
    unique_valid_canonical_evaluations: int = 0
    #: Expensive oracle executions actually performed. Real work.
    evaluator_calls: int = 0
    #: Requests for a canonical molecule already seen this run.
    duplicate_requests: int = 0
    #: Duplicate requests served from the cache without an evaluator call.
    cache_hits: int = 0
    #: Requests RDKit could not parse.
    failed_proposals: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "oracle_requests": self.oracle_requests,
            "unique_valid_canonical_evaluations": (
                self.unique_valid_canonical_evaluations
            ),
            "evaluator_calls": self.evaluator_calls,
            "duplicate_requests": self.duplicate_requests,
            "cache_hits": self.cache_hits,
            "failed_proposals": self.failed_proposals,
        }


@dataclass
class OracleAccountant:
    """Wraps a molecule objective and counts it three ways.

    Args:
        evaluate: the frozen objective, ``canonical SMILES -> float``. Called at
            most once per distinct valid canonical molecule while caching is on.
        budget: maximum value of the counter named by ``budget_counter``.
        budget_counter: which counter the budget binds. Named explicitly.
        cache: whether repeated canonical molecules reuse the stored score.
            Allowed for every method by the fairness contract, and required to
            use identical semantics across methods. Disabling it changes
            ``evaluator_calls`` and leaves the benchmark-native counter alone --
            which is the point of separating them.
        invalid_score: what an unparseable molecule scores. PMO returns 0.0.
    """

    evaluate: Callable[[str], float]
    budget: int
    budget_counter: str
    cache: bool = True
    invalid_score: float = 0.0
    counts: OracleCounts = field(default_factory=OracleCounts)
    _scores: dict[str, float] = field(default_factory=dict, repr=False)
    _seen: set[str] = field(default_factory=set, repr=False)

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
        """Score one candidate, charging every counter it is due.

        Raises:
            BudgetExhausted: if the declared budget is already spent. Callers
                check ``exhausted`` to terminate cleanly; the exception exists
                so that a driver which forgets to cannot overspend silently.
        """
        if self.exhausted:
            raise BudgetExhausted(
                f"{self.budget_counter} budget of {self.budget} is spent"
            )

        # Every request is demand, cached or not. This is the line that stops a
        # cache from erasing wasteful search behaviour.
        self.counts.oracle_requests += 1

        key = canonical_smiles(smiles)
        if key is None:
            self.counts.failed_proposals += 1
            return self.invalid_score

        if key in self._seen:
            self.counts.duplicate_requests += 1
            if self.cache:
                self.counts.cache_hits += 1
                return self._scores[key]
        else:
            self._seen.add(key)
            self.counts.unique_valid_canonical_evaluations += 1

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
    def demand_ratio_requests_over_unique(self) -> float:
        """``oracle_requests / unique_valid_canonical_evaluations``.

        The cache-and-duplicate rate, required per method by the fairness
        contract. It differs by orders of magnitude across these baselines, and
        hiding it is how an unfair comparison survives review.
        """
        unique = self.counts.unique_valid_canonical_evaluations
        if unique == 0:
            return float("nan")
        return self.counts.oracle_requests / unique

    def manifest(self) -> dict[str, Any]:
        counts = self.counts.as_dict()
        unique = counts["unique_valid_canonical_evaluations"]
        expected_evaluator_calls = (
            unique if self.cache else unique + counts["duplicate_requests"]
        )
        return {
            "counters": counts,
            "counter_semantics": {
                "unique_valid_canonical_evaluations": (
                    "benchmark-native; the only PMO-comparable number"
                ),
                "oracle_requests": "algorithmic demand, NOT CPU",
                "evaluator_calls": "expensive oracle executions after caching; real work",
            },
            "budget": self.budget,
            "budget_counter": self.budget_counter,
            "budget_exhausted": self.exhausted,
            "cache_enabled": self.cache,
            "demand_ratio_requests_over_unique": (
                self.demand_ratio_requests_over_unique
            ),
            "distinct_valid_molecules": len(self._seen),
            "invariants": {
                "oracle_requests >= unique_valid_canonical_evaluations": (
                    counts["oracle_requests"] >= unique
                ),
                (
                    "oracle_requests == unique + duplicate_requests "
                    "+ failed_proposals"
                ): (
                    counts["oracle_requests"]
                    == unique
                    + counts["duplicate_requests"]
                    + counts["failed_proposals"]
                ),
                "evaluator_calls matches the cache setting": (
                    counts["evaluator_calls"] == expected_evaluator_calls
                ),
                "caching cannot erase a wasteful request": (
                    counts["oracle_requests"]
                    >= counts["evaluator_calls"] + counts["failed_proposals"]
                    or counts["failed_proposals"] == 0
                ),
            },
        }


def silence_rdkit() -> None:
    """Suppress RDKit's parse warnings; invalids are counted, not printed."""
    RDLogger.DisableLog("rdApp.*")
