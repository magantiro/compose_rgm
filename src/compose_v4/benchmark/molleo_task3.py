"""MOLLEO Task 3 benchmark harness — BENCHMARK ONLY, NO COMPOSE POLICY.

THE SEPARATION IS THE POINT
---------------------------
This module is the external contract and nothing else: the five objectives, the
counted-oracle meter, and hypervolume. It must never import a COMPOSE policy,
and policy development must never be able to edit the benchmark. If later policy
work can silently change what is being measured, the measurement is worthless.

    benchmark/  <- this. frozen, adversarial, policy-agnostic.
    policy/     <- COMPOSE. may call the benchmark; may never modify it.

THE ORACLE BUDGET IS ENFORCED, NOT TRUSTED
------------------------------------------
`OracleMeter` raises the moment a policy would exceed its budget. A policy
cannot overrun by accident, and cannot "notice afterwards" -- there is no
afterwards. Repeat queries of the SAME molecule are served from cache and
charged once, which is standard for this benchmark family and is what makes an
archive-based method legitimate rather than a budget exploit.

THE COUNTING UNIT -- RESOLVED AGAINST THE RELEASED CODE
--------------------------------------------------------
    1 novel CANONICAL molecule receiving the five-objective vector = 1 unit

MOLLEO describes the budget as a maximum number of MOLECULE evaluations, seeds
with 120 random ZINC-250k molecules, and caps at 10,000. In the released code a
SMILES is canonicalized first; if that canonical molecule is already in
`mol_buffer` it is not re-evaluated, and a new one gets its entire
multi-objective evaluation for a single buffer entry. So it is NOT five separate
(molecule, objective) charges. CANONICALIZATION IS PART OF THE RULE, not an
optimization -- charging two spellings of one molecule twice would be wrong.

PER_OBJECTIVE is retained only so the stricter reading stays expressible.

⚠️ OPEN -- DO NOT YET CALL THIS "EXACT MOLLEO TASK 3"
-------------------------------------------------------
The released `select_pareto_front()` calls the objective evaluators directly,
bypassing the metered `score_smi()` path, and the loop evaluates the whole
offspring population that way BEFORE passing only the Pareto survivors through
the metered oracle. That appears to let new molecules be screened for Pareto
membership without being charged to the nominal 10k counter -- which is not what
the paper describes. Until a small released-code run establishes
`N_metered` vs `N_unique molecules actually evaluated`, this harness is
"strict-10k semantics", not "exact released-code semantics".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

__all__ = ["CountingRule", "BudgetExceeded", "OracleMeter", "hypervolume",
           "OBJECTIVES", "N_OBJECTIVES"]

#: (name, higher_is_better_after_transform). Minimisation objectives are
#: transformed to higher-is-better and all five normalised to [0, 1].
OBJECTIVES: tuple[tuple[str, str], ...] = (
    ("qed",   "max"),
    ("jnk3",  "max"),
    ("sa",    "min"),      # transformed
    ("gsk3b", "min"),      # transformed
    ("drd2",  "min"),      # transformed
)
N_OBJECTIVES = len(OBJECTIVES)


class CountingRule(Enum):
    """What the benchmark charges for. MUST be pinned against MOLLEO's source."""

    #: One evaluated molecule = one call, regardless of objective count.
    #: This is the PMO convention and the likely intent, but UNVERIFIED here.
    PER_MOLECULE = "per_molecule"
    #: One (molecule, objective) pair = one call. 5x stricter.
    PER_OBJECTIVE = "per_objective"


class BudgetExceeded(RuntimeError):
    """Raised INSTEAD of returning a value that would overrun the budget."""


@dataclass
class OracleMeter:
    """Hard budget enforcement around the five benchmark objectives.

    The meter is the benchmark's contract with the policy. It is deliberately
    unforgiving: exceeding the budget raises rather than warns, because a run
    that overran is not a result and must not be silently reported as one.
    """

    evaluate: Callable[[str], Sequence[float]]
    budget: int = 10_000
    #: Canonicalizer. Part of the COUNTING RULE, not an optimization: the
    #: released benchmark canonicalizes before consulting its buffer, so two
    #: spellings of one molecule must cost ONE unit, not two.
    canonicalize: Callable[[str], str] | None = None
    #: No default. Guessing this is exactly the mistake the docstring warns of.
    counting_rule: CountingRule = CountingRule.PER_MOLECULE
    #: Optional batched evaluator. The three model-based objectives are far
    #: cheaper per molecule in a batch (one tree walk, one GEMM), and the
    #: BUDGET SEMANTICS ARE IDENTICAL either way -- a batch is charged exactly
    #: the novel canonical molecules it contains.
    evaluate_many: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None
    #: Called with (canonical_smiles, values) for each NEWLY charged molecule.
    #: This is how the durable ledger sees every evaluation without the
    #: benchmark knowing anything about persistence.
    on_evaluated: Callable[[str, tuple[float, ...]], None] | None = None
    _spent: int = field(default=0, init=False)
    _cache: dict[str, tuple[float, ...]] = field(default_factory=dict, init=False)

    @property
    def spent(self) -> int:
        return self._spent

    @property
    def remaining(self) -> int:
        return self.budget - self._spent

    @property
    def n_unique(self) -> int:
        """Distinct molecules evaluated. Equals `spent` under PER_MOLECULE."""
        return len(self._cache)

    def _charge(self) -> int:
        return 1 if self.counting_rule is CountingRule.PER_MOLECULE \
            else N_OBJECTIVES

    def _key(self, smiles: str) -> str:
        """Canonical form, or the raw string when there is no canonical form.

        A canonicalizer signals "this does not parse" with None. Using None as
        the cache key would collapse EVERY unparseable string into one entry and
        charge the policy once for unlimited junk; falling back to the raw string
        charges each distinct one, which is the conservative direction.
        """

        if self.canonicalize is None:
            return smiles
        return self.canonicalize(smiles) or smiles

    def __call__(self, smiles: str) -> tuple[float, ...]:
        """Evaluate one molecule, charging the budget. Cached repeats are free."""
        smiles = self._key(smiles)           # canonical FIRST, then look up
        if smiles in self._cache:
            return self._cache[smiles]       # already paid for
        cost = self._charge()
        if self._spent + cost > self.budget:
            raise BudgetExceeded(
                f"evaluating {smiles!r} would cost {cost} and bring the total "
                f"to {self._spent + cost:,}, over the {self.budget:,} budget "
                f"({self.remaining:,} remaining). The run is invalid if it "
                f"continues; the policy must stop on its own.")
        vals = self._commit(smiles, self.evaluate(smiles), cost)
        return vals

    def _commit(self, key: str, raw: Sequence[float], cost: int) -> tuple[float, ...]:
        vals = tuple(float(v) for v in raw)
        if len(vals) != N_OBJECTIVES:
            raise ValueError(
                f"expected {N_OBJECTIVES} objectives, got {len(vals)}")
        self._spent += cost
        self._cache[key] = vals
        if self.on_evaluated is not None:
            self.on_evaluated(key, vals)
        return vals

    def batch(self, smiles: Sequence[str]) -> list[tuple[float, ...]]:
        """Evaluate many molecules at once, charged exactly as one-at-a-time.

        The budget check is made for the WHOLE batch before anything is charged:
        a batch that does not fit raises without evaluating part of itself, so a
        policy can never end up in a state where some of its offspring were
        counted and the rest silently were not. Duplicates inside the batch --
        two spellings of one molecule, or a molecule already in the archive --
        cost nothing extra, exactly as in the single-molecule path.
        """

        keys = [self._key(s) for s in smiles]
        novel: list[str] = []
        seen: set[str] = set()
        for key in keys:
            if key not in self._cache and key not in seen:
                seen.add(key)
                novel.append(key)
        if novel:
            cost = len(novel) * self._charge()
            if self._spent + cost > self.budget:
                raise BudgetExceeded(
                    f"this batch contains {len(novel)} novel molecules costing "
                    f"{cost}, which would bring the total to "
                    f"{self._spent + cost:,} against a {self.budget:,} budget "
                    f"({self.remaining:,} remaining). Nothing was charged; trim "
                    f"the batch or stop.")
            if self.evaluate_many is not None:
                values = list(self.evaluate_many(novel))
                if len(values) != len(novel):
                    raise ValueError(
                        f"batched evaluator returned {len(values)} rows for "
                        f"{len(novel)} molecules")
            else:
                values = [self.evaluate(key) for key in novel]
            for key, row in zip(novel, values):
                self._commit(key, row, self._charge())
        return [self._cache[key] for key in keys]

    def can_afford(self, n_new_molecules: int) -> bool:
        """Whether n NEW (uncached) molecules fit. Lets a policy stop cleanly."""
        return self._spent + n_new_molecules * self._charge() <= self.budget

    def novel_in(self, smiles: Sequence[str]) -> int:
        """How many of these would actually be charged. The number a policy needs
        to decide whether it can afford a generation."""

        seen: set[str] = set()
        for item in smiles:
            key = self._key(item)
            if key not in self._cache:
                seen.add(key)
        return len(seen)

    def evaluated(self) -> dict[str, tuple[float, ...]]:
        return dict(self._cache)

    def restore(self, evaluations: dict[str, tuple[float, ...]]) -> None:
        """Re-enter a run's already-paid-for evaluations after a crash.

        The keys must already be canonical -- they come from the ledger, which
        stores what the meter charged. Spending is recomputed from the count
        rather than restored from a saved counter, so a run cannot come back
        from disk believing it has budget it already spent.
        """

        if self._cache:
            raise RuntimeError("restore() into a meter that has already scored")
        self._cache = {k: tuple(float(x) for x in v) for k, v in evaluations.items()}
        self._spent = len(self._cache) * self._charge()
        if self._spent > self.budget:
            raise BudgetExceeded(
                f"the ledger holds {len(self._cache):,} evaluations costing "
                f"{self._spent:,}, over this run's {self.budget:,} budget")


def _pareto_mask(points: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows, maximisation in every column."""
    n = len(points)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # j dominates i if j >= i everywhere and > i somewhere.
        dominates_i = np.all(points >= points[i], axis=1) & \
            np.any(points > points[i], axis=1)
        if dominates_i.any():
            keep[i] = False
    return keep


def hypervolume(points: Sequence[Sequence[float]],
                reference: Sequence[float] | None = None,
                *, samples: int = 200_000,
                rng: np.random.Generator | None = None) -> float:
    """Hypervolume dominated by `points` above `reference`, all maximisation.

    Five objectives makes exact HV expensive, so this uses Monte-Carlo
    estimation over the reference box. That is a deliberate trade: the estimate
    is unbiased and its error is reportable, whereas an exact 5-D routine would
    dominate the runtime of the very search it is meant to score.

    With the default [0,1]^5 reference the returned value is the dominated
    FRACTION of the unit box, so it is directly comparable across runs.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:            # empty front: check BEFORE ndim, since
        return 0.0               # np.asarray([]) is 1-D, not 2-D
    if pts.ndim != 2:
        raise ValueError(f"expected a 2-D array of points, got shape {pts.shape}")
    ref = np.zeros(pts.shape[1]) if reference is None else np.asarray(reference,
                                                                     dtype=float)
    if ref.shape[0] != pts.shape[1]:
        raise ValueError(f"reference has {ref.shape[0]} dims, points have "
                         f"{pts.shape[1]}")
    # Only non-dominated points contribute; dropping the rest cuts the inner
    # comparison without changing the answer.
    pts = pts[_pareto_mask(pts)]
    upper = np.maximum(pts.max(axis=0), ref)
    box = upper - ref
    if np.any(box <= 0):
        return 0.0
    g = rng if rng is not None else np.random.default_rng(0)
    u = g.random((samples, pts.shape[1])) * box + ref
    # A sample counts if ANY point dominates it.
    dominated = np.zeros(samples, dtype=bool)
    for p in pts:
        dominated |= np.all(u <= p, axis=1)
    return float(dominated.mean() * np.prod(box))
