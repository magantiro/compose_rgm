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

⚠️ SETTLED, AND THE ANSWER IS WHY THIS IS "STRICT-10k", NOT "MOLLEO TASK 3"
---------------------------------------------------------------------------
The suspicion was right, and it was measured by running the released code
(github.com/zoom-wang112358/MOLLEO @ fd138a7) with the five objectives stubbed
and every evaluator wrapped in a recorder.

`multi_objective/main/pareto_optimizer.py:223-243` -- `select_pareto_front()`
calls the five `tdc.Oracle` objects DIRECTLY (`:229`, `:232`, `:235`). No
`score_smi`, no `mol_buffer`, no budget check. The Task 3 loop
(`main/molleo_multi_pareto/run.py:158-165`) concatenates parents + all offspring,
screens the whole population through that unmetered path, and passes only the
Pareto survivors to the metered oracle. So most molecules that get all five
objectives computed are never charged:

    budget   N_metered   N_evaluated   ratio    evaluated but never charged
       300         319           814    2.55            495  (60.8%)
       500         519         1,786    3.44          1,267  (70.9%)
     1,000       1,015         4,804    4.73          3,789  (78.9%)
     2,000       1,337         7,256    5.43          5,919  (81.6%)

The ratio GROWS with the budget -- the marginal ratio over the last ten
generations of the 2,000 run was 9.3 -- because the population is never
truncated to `population_size`, so the Pareto front (and the screen) keeps
growing. A 10,000-call run would be well above 6x. Two further leaks compound
it: `clean_buffer()` empties `mol_buffer` every generation, so the `score_smi`
budget guard at `:186` is a PER-GENERATION guard that never fires (the run
actually stops on `storing_buffer` at the END of a generation, overshooting the
nominal budget by 4-6%), and every survivor is re-evaluated in full each
generation without being re-charged: 50,848 molecule-evaluations behind a
counter reading 1,007.

CONSEQUENCES, WHICH ARE BINDING:

* This harness charges every molecule that receives the objective vector. That
  is STRICT-10k SEMANTICS. It is NOT "exact MOLLEO Task 3" and must not be
  relabelled as such.
* We do not copy the leak. A screened-but-uncharged path would make the budget
  meaningless and is precisely the thing this meter exists to prevent.
* Strict-10k is closer to the benchmark's STATED semantics than reproducing the
  released path would be: the paper's own justification for the budget is that
  objective evaluation is the dominant expense and that algorithms are compared
  at equal call budgets. A path that evaluates several molecules per charged
  call does not honour that; this meter does.
* A strict-10k number must NOT be placed beside MOLLEO's published number as if
  the protocols matched. If the published figures arose under the leaky path,
  they are PUBLISHED CONTEXT, not head-to-head values. Where a baseline is
  runnable we rerun it through THIS meter and compare those numbers instead.

The counting rule itself, though, is theirs: `score_smi` (`:190-198`)
canonicalises before consulting the buffer, so one molecule written two ways
costs one unit. That part is reproduced faithfully.

THE METRIC -- THE PAPER'S, IMPLEMENTED AND QUALIFIED
-----------------------------------------------------
MOLLEO defines the metric explicitly (Eq. 5): hypervolume is the volume of the
UNION OF HYPERRECTANGLES measured FROM THE ORIGIN in the normalised [0, 1]^n
objective space. `hypervolume_qmc` implements exactly that. This is the
published definition, not a convention of ours.

It is qualified against that definition rather than against another estimate of
itself: exact inclusion-exclusion over the union of boxes, worst absolute error
3.5e-05 over random fronts, where the differences we act on are of order 0.09.
A second test pins that the value is a function of the SET of molecules found --
order-independent, so it cannot depend on scheduling.

The released CODE contains no hypervolume in the optimisation loop -- it
computes `top_auc` over the scalar sum (`optimizer.py:31`,
`pareto_optimizer.py:30`) and persists a YAML dump of `self.mol_buffer`
(`pareto_optimizer.py:105-114`), which after `clean_buffer()` (`:101-103`) holds
only the LAST GENERATION while the cumulative `storing_buffer` is never saved.
That is a property of the release, and it does NOT make the metric unknowable;
the paper states it.

What the paper's formula does not fix is which SET of molecules the front is
taken over. For DEVELOPMENT we use the cumulative all-evaluated non-dominated
set -- the natural monotone best-found-so-far object. COMPARABILITY OF OUR
NUMBERS WITH PUBLISHED ONES IS THEREFORE **PENDING**: Eq. 5 fixes the geometry
and the reference point, not the set, so protocol identity is not something we
may claim yet.
It is the standard best-found-so-far convention, it is monotone, it is
recomputable from the durable ledger alone, and it is the reading most GENEROUS
to the baselines -- which makes any eventual COMPOSE claim the conservative one.
The size of that choice is measured rather than waved at: on one Graph-GA run,
HV over all 10,000 evaluated molecules is 0.612 against 0.515 over the last 120
only.

SEED SEMANTICS -- FROZEN: EVERY SEED STARTS EMPTY
-------------------------------------------------
Each of our seeds is a genuinely independent run: a fresh oracle buffer, a fresh
budget, no state carried from any other seed. The initial 120 molecules are
charged, exactly as the released implementation charges them
(`molleo_multi_pareto/run.py:102` puts the starting population through the
metered oracle).

The released code cannot quite promise the same, and the reason is worth
recording precisely, because it is easy to overstate. `Oracle.__init__` takes
`mol_buffer={}` as a MUTABLE DEFAULT (`pareto_optimizer.py:51`, and identically
`optimizer.py:52`), and `BaseOptimizer.reset()` (`:331-333`) constructs a new
`Oracle` bound to that same shared dict. But `clean_buffer()` (`:101-103`)
REBINDS `self.mol_buffer` to a new dict at the top of every generation. So from
the structure of the code, what the shared default retains is the EARLY,
PRE-CLEAN contents -- most importantly the initial population -- rather than
necessarily the whole of a previous seed's search.

This is recorded as an IMPLEMENTATION DEFECT in the released code, not as a
protocol feature, and it is not reproduced here. Its effect on their published
five-seed numbers is UNKNOWN: it depends on how the authors actually launched
those runs, and the README's `--seed 1 2 3` form is suggestive, not proof. No
claim is made that their published results were contaminated.

Our initialization sets are drawn by us, sealed before any policy existed
(`artifacts/benchmarks/molleo_task3_init_v1/`), and recorded with the pool's
sha256. We match the benchmark's stated CONTRACT -- 120 molecules drawn at
random from ZINC-250k, five independent seeds -- and do not claim
molecule-for-molecule identity with whatever their RNG produced.
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
    """Boolean mask of non-dominated rows, maximisation in every column.

    Sorted by summed objectives, descending, then each point is compared only
    against the front found so far. That is exact, not a heuristic: if q
    dominates p then q >= p on every axis and is strictly greater somewhere, so
    sum(q) > sum(p) and q is always processed first. Points with equal sums
    cannot dominate each other, so ties need no special handling.

    The naive all-pairs sweep is O(n^2) and took 4.3 s on a 10,000-molecule
    archive; this is 0.13 s and returns the identical mask.
    """

    points = np.asarray(points, dtype=float)
    n = len(points)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    order = np.argsort(-points.sum(axis=1), kind="stable")
    front = np.empty_like(points)
    size = 0
    for i in order:
        candidate = points[i]
        if size:
            established = front[:size]
            if np.any(np.all(established >= candidate, axis=1)
                      & np.any(established > candidate, axis=1)):
                continue
        front[size] = candidate
        size += 1
        keep[i] = True
    return keep


def hypervolume_qmc(points: Sequence[Sequence[float]], *,
                    log2_samples: int = 20) -> float:
    """THE REPORTED HYPERVOLUME: deterministic, and paired across runs.

    Why not the Monte-Carlo estimator below: the benchmark's primary number is a
    COMPARISON between policies, and two independently-noisy estimates make a
    small real difference hard to see. This one integrates the FIXED unit box
    with a fixed Sobol sequence, so every front is measured against the very same
    sample points. The error is common to both sides of a comparison and largely
    cancels, and the same front always returns the same number -- no seed, no
    variance, reproducible from the archive alone.

    Only defined for objectives already normalised to [0, 1] with the origin as
    the reference point, which is what the Task 3 transforms guarantee.

    Samples outside the front's own bounding box cannot be dominated by
    anything, so they are dropped before the per-point work. That is an exact
    algebraic shortcut, not an approximation: the retained sample set is still
    the same fixed sequence.
    """

    from scipy.stats import qmc

    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return 0.0
    if pts.ndim != 2:
        raise ValueError(f"expected a 2-D array of points, got shape {pts.shape}")
    if pts.min() < 0.0 or pts.max() > 1.0:
        raise ValueError(
            "hypervolume_qmc integrates the unit box, so every objective must "
            "already be normalised to [0, 1]")
    pts = pts[_pareto_mask(pts)]
    upper = pts.max(axis=0)
    if np.any(upper <= 0):
        return 0.0

    total = 1 << log2_samples
    dominated = 0
    engine = qmc.Sobol(d=pts.shape[1], scramble=False)
    # Chunked to bound memory, but never larger than the request -- a chunk
    # bigger than `total` would silently integrate nothing and return 0.0.
    chunk = min(1 << 17, total)
    for _ in range(total // chunk):
        u = engine.random(chunk)
        # Nothing above the front's own corner can be dominated; dropping those
        # rows leaves the answer unchanged and does most of the work.
        candidates = u[np.all(u <= upper, axis=1)]
        if not len(candidates):
            continue
        hit = np.zeros(len(candidates), dtype=bool)
        for p in pts:
            hit |= np.all(candidates <= p, axis=1)
        dominated += int(hit.sum())
    return dominated / total


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
