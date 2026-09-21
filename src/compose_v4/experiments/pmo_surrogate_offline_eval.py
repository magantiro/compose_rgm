"""Matched-budget offline comparison of oracle-allocation policies.

The question this answers is narrow on purpose: HOLDING THE CANDIDATE POOL
FIXED, does ranking candidates before spending oracle calls buy more than the
current unranked selection at the same budget?

Because every arm sees the identical pool at every round and differs only in
WHICH members it charges, any measured gain is attributable to allocation and
not to proposal quality.  The converse is the other half of the finding: the
``pool_ceiling`` arm charges the true best members of the pool, so it bounds
what perfect allocation could ever buy.  If that ceiling is close to the random
arm, the pool is the problem and no ranker can rescue it.

No oracle calls are charged here.  Scores come from a caller-supplied free
scorer, and within each simulated run the surrogate trains only on the rows that
simulated run has revealed -- the same no-prescreen rule the live benchmark
requires.

Data structure
--------------
``CampaignPool``  rounds of free candidate SMILES, plus the initialization set.
``ArmResult``     one policy x one seed: the ordered purchased rewards, the
                  production PMO AUC, and the prequential surrogate diagnostics.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from compose_v4.control.program_task import pmo_top_ten_auc
from compose_v4.experiments.pmo_surrogate_allocator import (
    MorganFeaturizer,
    OnlineArchive,
    SurrogateAllocator,
    build_surrogate,
)

Scorer = Callable[[str], float]


@dataclass
class CampaignPool:
    """Initialization molecules plus one free candidate list per round."""

    initialization: list[str]
    rounds: list[list[str]]
    label: str = "pool"

    def total_candidates(self) -> int:
        return sum(len(r) for r in self.rounds)


@dataclass
class PrequentialDiagnostics:
    """Online held-out predictive quality, measured INSIDE the archive.

    Before each round's purchases are revealed, the surrogate fit on the archive
    so far predicts them; the realized scores then score the prediction.  This is
    the model-selection criterion.  Selecting a surrogate on benchmark AUC
    instead would be score hacking.
    """

    n: int = 0
    abs_error: list[float] = field(default_factory=list)
    spearman: list[float] = field(default_factory=list)
    coverage_hits: int = 0
    coverage_total: int = 0

    def record(self, mean: np.ndarray, sigma: np.ndarray, truth: np.ndarray) -> None:
        if len(truth) == 0:
            return
        self.n += len(truth)
        self.abs_error.extend(np.abs(mean - truth).tolist())
        if len(truth) >= 3 and np.std(truth) > 0 and np.std(mean) > 0:
            self.spearman.append(_spearman(mean, truth))
        lo, hi = mean - 1.96 * sigma, mean + 1.96 * sigma
        self.coverage_hits += int(np.sum((truth >= lo) & (truth <= hi)))
        self.coverage_total += len(truth)

    def summary(self) -> dict:
        return {
            "n_predicted": self.n,
            "mae": float(np.mean(self.abs_error)) if self.abs_error else None,
            "spearman_mean": float(np.mean(self.spearman)) if self.spearman else None,
            "interval_coverage_95": (
                self.coverage_hits / self.coverage_total if self.coverage_total else None
            ),
        }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rankdata(a), _rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt(float((ra**2).sum()) * float((rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


@dataclass
class ArmResult:
    arm: str
    seed: int
    rewards: list[float]
    auc: float | None
    best: float
    top10_mean: float
    rank_seconds: float
    surrogate_active_calls: int
    explore_calls: int
    diagnostics: dict


# ---- Policies ----


class Policy:
    """Choose which pool members to charge.  Never calls an oracle."""

    name = "base"
    uses_surrogate = False

    def select(self, archive: OnlineArchive, candidates: list[str], k: int) -> list[int]:
        raise NotImplementedError


class RandomPolicy(Policy):
    """Uniform over unpurchased candidates: the unranked-allocation control."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, archive, candidates, k):
        pool = [i for i, s in enumerate(candidates) if not archive.contains(s)]
        if not pool:
            return []
        k = min(k, len(pool))
        return [int(i) for i in self.rng.choice(pool, size=k, replace=False)]


class FirstNPolicy(Policy):
    """Take the pool in the order the proposer emitted it.

    This is the stand-in for an allocation step that applies no ranking at all.
    It is NOT random: if the proposer's emission order carries any quality
    signal, this arm inherits it, which makes it the more conservative control.
    """

    name = "first_n"

    def select(self, archive, candidates, k):
        pool = [i for i, s in enumerate(candidates) if not archive.contains(s)]
        return pool[:k]


class PoolCeilingPolicy(Policy):
    """Charge the truly best members of the pool.  NOT ACHIEVABLE ONLINE.

    Included to bound what perfect allocation could buy, which is what separates
    an allocation problem from a proposal-quality problem.
    """

    name = "pool_ceiling"

    def __init__(self, scorer: Scorer) -> None:
        self._scorer = scorer

    def select(self, archive, candidates, k):
        pool = [i for i, s in enumerate(candidates) if not archive.contains(s)]
        if not pool:
            return []
        ranked = sorted(pool, key=lambda i: -self._scorer(candidates[i]))
        return ranked[:k]


class SurrogatePolicy(Policy):
    """The arm under test: fit online, rank, spend on the ranked head."""

    uses_surrogate = True

    def __init__(
        self,
        *,
        surrogate_name: str = "gp_tanimoto",
        acquisition: str = "ucb",
        beta: float = 1.0,
        explore_fraction: float = 0.25,
        min_fit_rows: int = 8,
        diversity_penalty: float = 0.0,
        seed: int = 0,
        featurizer: MorganFeaturizer | None = None,
        name: str | None = None,
    ) -> None:
        self.allocator = SurrogateAllocator(
            surrogate_name=surrogate_name,
            acquisition=acquisition,
            beta=beta,
            explore_fraction=explore_fraction,
            min_fit_rows=min_fit_rows,
            diversity_penalty=diversity_penalty,
            featurizer=featurizer,
            seed=seed,
        )
        self.surrogate_name = surrogate_name
        self.name = name or (
            f"{surrogate_name}/{acquisition}"
            f"{'' if acquisition == 'greedy' else f'_b{beta:g}'}"
            f"_x{explore_fraction:g}"
            f"{f'_d{diversity_penalty:g}' if diversity_penalty else ''}"
        )
        self.last_decision = None

    def select(self, archive, candidates, k):
        decision = self.allocator.select_batch(archive, candidates, k)
        self.last_decision = decision
        return decision.selected


# ---- Campaign simulation ----


def run_arm(
    pool: CampaignPool,
    scorer: Scorer,
    policy: Policy,
    *,
    budget: int,
    seed: int,
    queries_per_round: int,
    diagnostic_surrogate: str | None = "gp_tanimoto",
    featurizer: MorganFeaturizer | None = None,
) -> ArmResult:
    """Simulate one matched-budget campaign under ``policy``.

    The initialization molecules are charged first and identically for every arm,
    so the arms are matched on both budget and starting archive.
    """
    if budget < 1 or queries_per_round < 1:
        raise ValueError("positive budget and queries_per_round required")
    archive = OnlineArchive()
    rewards: list[float] = []
    featurizer = featurizer or MorganFeaturizer()
    diagnostics = PrequentialDiagnostics()
    rank_seconds = 0.0
    surrogate_active_calls = 0
    explore_calls = 0

    for smiles in pool.initialization:
        if len(rewards) >= budget:
            break
        if archive.observe(smiles, scorer(smiles), provenance="initialization", step=0):
            rewards.append(archive.rows[-1].score)

    for round_index, candidates in enumerate(pool.rounds, start=1):
        if len(rewards) >= budget:
            break
        k = min(queries_per_round, budget - len(rewards))
        selected = policy.select(archive, candidates, k)
        if not selected:
            continue

        picked_smiles = [candidates[i] for i in selected]
        truth = np.asarray([scorer(s) for s in picked_smiles], dtype=np.float64)

        # Prequential diagnostic: predict BEFORE revealing, using only the rows
        # purchased so far.  Independent of the arm's own policy.
        if diagnostic_surrogate and len(archive) >= 8:
            X_train, kept = featurizer.featurize(archive.smiles())
            if len(kept) >= 8:
                X_eval, eval_kept = featurizer.featurize(picked_smiles)
                if eval_kept:
                    model = build_surrogate(diagnostic_surrogate)
                    model.fit(X_train, archive.scores()[kept])
                    pred = model.predict(X_eval)
                    diagnostics.record(pred.mean, pred.sigma, truth[eval_kept])

        decision = getattr(policy, "last_decision", None)
        if decision is not None:
            rank_seconds += decision.rank_seconds
            if decision.surrogate_active:
                surrogate_active_calls += len(decision.selected)
                explore_calls += decision.explore_slots

        for smiles, value in zip(picked_smiles, truth, strict=True):
            if len(rewards) >= budget:
                break
            if archive.observe(
                smiles, float(value), provenance=f"round_{round_index}", step=round_index
            ):
                rewards.append(float(value))

    scores = sorted(rewards, reverse=True)
    return ArmResult(
        arm=policy.name,
        seed=seed,
        rewards=rewards,
        auc=pmo_top_ten_auc(rewards, budget=budget) if rewards else None,
        best=scores[0] if scores else 0.0,
        top10_mean=float(np.mean(scores[:10])) if scores else 0.0,
        rank_seconds=rank_seconds,
        surrogate_active_calls=surrogate_active_calls,
        explore_calls=explore_calls,
        diagnostics=diagnostics.summary(),
    )


def aggregate(results: Sequence[ArmResult]) -> dict:
    """Mean, sd and a normal-approximation 95% CI across seeds."""

    def stats(values: list[float]) -> dict:
        values = [v for v in values if v is not None]
        if not values:
            return {"n": 0, "mean": None, "sd": None, "ci95": None}
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        half = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
        return {
            "n": len(values),
            "mean": mean,
            "sd": sd,
            "ci95": [mean - half, mean + half],
        }

    return {
        "arm": results[0].arm if results else None,
        "auc": stats([r.auc for r in results]),
        "best": stats([r.best for r in results]),
        "top10_mean": stats([r.top10_mean for r in results]),
        "rank_seconds": stats([r.rank_seconds for r in results]),
        "explore_calls": stats([float(r.explore_calls) for r in results]),
    }


def cold_start_curve(
    pool: CampaignPool,
    scorer: Scorer,
    build_policy: Callable[[int], Policy],
    *,
    budget: int,
    queries_per_round: int,
    seeds: Sequence[int],
) -> list[dict]:
    """Running top-10 mean after each charged call, averaged over seeds.

    This is what locates the cold-start crossover: the call count at which the
    surrogate arm's curve passes and stays above the control's.
    """
    curves: list[list[float]] = []
    for seed in seeds:
        result = run_arm(
            pool,
            scorer,
            build_policy(seed),
            budget=budget,
            seed=seed,
            queries_per_round=queries_per_round,
        )
        running, best10 = [], []
        for value in result.rewards:
            best10.append(value)
            best10 = sorted(best10, reverse=True)[:10]
            running.append(float(np.mean(best10)))
        curves.append(running)
    if not curves:
        return []
    width = min(len(c) for c in curves)
    return [
        {
            "calls": i + 1,
            "top10_mean": statistics.fmean([c[i] for c in curves]),
            "sd": statistics.stdev([c[i] for c in curves]) if len(curves) > 1 else 0.0,
        }
        for i in range(width)
    ]
