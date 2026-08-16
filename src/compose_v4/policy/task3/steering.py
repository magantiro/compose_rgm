"""Two ways of specifying purpose. THIS IS THE ONLY THING THE A/B VARIES.

    A  FixedScalarization    aim at "more of the weighted sum", always
    B  AdaptiveRegion        aim at the undercovered part of the current front

Everything else in the mechanism test is shared by construction: the same seeded
archive, the same surrogate instance class, the same expansion operator, the same
additional metered budget, the same candidate count. Anything else that differed
would be a confound rather than a result.

WHAT IS AND IS NOT BEING CLAIMED ABOUT SCALARIZATION
-----------------------------------------------------
A fixed scalarization is not blind to the objectives it sums. It rewards high
JNK3 and low GSK3B alike, because both are terms in it. Its limitation is
narrower: it compresses the tradeoff into a single number, so it has no
mechanism to notice that a valuable part of the current Pareto front is MISSING
and deliberately send search there. Adaptive region targeting has exactly that
mechanism. That difference -- not "one cares about selectivity and the other
does not" -- is the hypothesis.

Each rule answers the same three questions, so the loop driving them is
identical:

    target(archive)          what are we aiming at right now
    start(archive, target)   which realized molecule do we aim from
    rank(predicted, target)  which proposed molecules are worth charging
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from compose_v4.policy.task3.archive import ParetoArchive
from compose_v4.policy.task3.interfaces import Region
from compose_v4.policy.task3.regions import MarginalGainRegions

#: MOLLEO's own scalarization: the unweighted sum of the five transformed
#: objectives. Arm A uses the benchmark's scalarization rather than one invented
#: for the comparison.
UNIFORM_WEIGHTS = np.ones(5) / 5.0


@dataclass
class FixedScalarization:
    """Arm A. One fixed direction, for the whole run."""

    name: str = "fixed-scalarization"
    weights: np.ndarray = field(default_factory=lambda: UNIFORM_WEIGHTS.copy())

    def target(self, archive: ParetoArchive,
               rng: np.random.Generator) -> Region | None:
        # The "region" is the constant corner the sum points at. It never moves,
        # which is precisely the property under test.
        return Region(target=(1.0,) * 5, gain=0.0, origin=None,
                      note="fixed weighted sum")

    def start(self, archive: ParetoArchive, target: Region,
              rng: np.random.Generator) -> str | None:
        if not archive.values:
            return None
        keys = list(archive.values)
        scores = archive.points() @ self.weights
        return keys[int(np.argmax(scores))]

    def rank(self, predicted: np.ndarray, target: Region) -> np.ndarray:
        return predicted @ self.weights


@dataclass
class AdaptiveRegion:
    """Arm B. Aim at whatever valuable part of the front is currently missing."""

    name: str = "adaptive-region"
    regions: MarginalGainRegions = field(default_factory=MarginalGainRegions)

    def target(self, archive: ParetoArchive,
               rng: np.random.Generator) -> Region | None:
        return self.regions.select(archive, rng)

    def start(self, archive: ParetoArchive, target: Region,
              rng: np.random.Generator) -> str | None:
        if target.origin and target.origin in archive.values:
            return target.origin
        if not archive.values:
            return None
        keys = list(archive.values)
        shortfall = np.maximum(target.as_array()[None, :] - archive.points(),
                               0.0).sum(axis=1)
        return keys[int(np.argmin(shortfall))]

    def rank(self, predicted: np.ndarray, target: Region) -> np.ndarray:
        # Least predicted shortfall to the aspiration. Being ABOVE the target on
        # an axis is headroom, not credit -- otherwise this silently becomes a
        # scalarization with extra steps.
        shortfall = np.maximum(target.as_array()[None, :] - predicted,
                               0.0).sum(axis=1)
        return -shortfall


@dataclass
class ConstrainedAdaptiveRegion:
    """Arm B'. TESTED AND IT DID NOT WORK. Kept as a recorded negative.

    THE DIAGNOSIS THIS WAS BUILT ON WAS WRONG. I believed the summed shortfall
    let a large gain on the targeted axis pay for a loss on another, so that
    "more JNK3, everything else as it was" was being satisfied by molecules that
    quietly gave up GSK3B. A test disproved it: because being ABOVE the
    aspiration earns no credit, there is nothing to pay WITH. On a real
    aspiration, a clean candidate ranks -0.10 and a promiscuous one -0.80. The
    unconstrained rule already prefers the clean molecule.

    WHAT IS ACTUALLY HAPPENING. Of the charged molecules reaching JNK3 >= 0.5,
    0.2% have a GSK3B coordinate >= 0.9. The selective molecules are not being
    ranked below promiscuous ones -- they are not being PROPOSED, because at that
    height the reachable chemistry is promiscuous. Steering cannot select what
    expansion never offers. Constraining against the origin cannot help either,
    since the high-JNK3 front members the aspirations are built from have already
    given GSK3B up, so the floor it holds is a floor that was already low.

    MEASURED, three seeds, same seeded archive and budget as the other arms:
    mean HV lift +0.031 against +0.057 unconstrained and +0.021 for the fixed
    sum, and 3 selective molecules against 7 and 188. It is worse than the rule
    it was meant to repair.

    Kept rather than deleted because the negative is the useful part: it locates
    the problem in proposal availability rather than in purpose specification,
    which is a different thing to fix.
    """

    name: str = "constrained-adaptive-region"
    regions: MarginalGainRegions = field(default_factory=MarginalGainRegions)
    #: Weight on violating the hold-what-you-have constraint. Large enough that
    #: no gain on the targeted axis can buy a regression elsewhere.
    violation_weight: float = 10.0
    _origin: np.ndarray | None = field(default=None, init=False, repr=False)
    _axis: int | None = field(default=None, init=False, repr=False)

    def target(self, archive: ParetoArchive,
               rng: np.random.Generator) -> Region | None:
        region = self.regions.select(archive, rng)
        if region is None:
            return None
        origin = archive.values.get(region.origin) if region.origin else None
        self._origin = np.asarray(origin, dtype=float) if origin else None
        # `note` is written by MarginalGainRegions as "axis {i} +{step}".
        try:
            self._axis = int(region.note.split()[1])
        except (IndexError, ValueError):
            self._axis = None
        return region

    def start(self, archive: ParetoArchive, target: Region,
              rng: np.random.Generator) -> str | None:
        return AdaptiveRegion(regions=self.regions).start(archive, target, rng)

    def rank(self, predicted: np.ndarray, target: Region) -> np.ndarray:
        aspiration = target.as_array()
        if self._origin is None or self._axis is None:
            shortfall = np.maximum(aspiration[None, :] - predicted, 0.0).sum(axis=1)
            return -shortfall
        # Progress on the axis the aspiration is actually about.
        progress = np.maximum(aspiration[self._axis] - predicted[:, self._axis], 0.0)
        # Regression anywhere else, measured against what the origin ALREADY had.
        floor = self._origin.copy()
        floor[self._axis] = 0.0
        violation = np.maximum(floor[None, :] - predicted, 0.0).sum(axis=1)
        return -(progress + self.violation_weight * violation)


@dataclass
class ReachableHVI:
    """Arm B-v2. Aim at what the CURRENT fiber can actually deliver.

    WHY THE PREVIOUS ADAPTIVE RULE NEVER TURNED ON. Its aspiration was a front
    member with one axis raised while the other four were held at that member's
    values -- a DOMINANCE-SHAPED demand. A molecular edit moves several
    properties at once, so matching-or-exceeding on four axes while improving
    the fifth is rare, and the logging showed the consequence starkly:
    `iterations_where_target_was_represented` was ZERO for every arm and every
    seed. The mechanism was permanently in fallback, minimising shortfall to a
    target nothing could reach. It was never really tested.

    So this version does not invent a target at all. It scores each successor by
    the hypervolume it would ACTUALLY add to the current archive, and prefers the
    largest. The "region" is then whatever the best-scoring reachable successors
    occupy -- defined by the fiber and the archive, never by a wish.

    This is deliberately the simplest reachable acquisition rather than a clever
    one. A clever aspiration built on top of an unreachable one would have been
    the same mistake twice.
    """

    name: str = "reachable-hvi"
    #: OPTIMISM. A weighted-average surrogate cannot predict above its
    #: neighbours' labels, so a mean-only HVI acquisition can almost never claim
    #: a candidate beats the current front: measured on real fibers, 100% of
    #: predicted vectors were interior to the archive's box and only 0.9% had
    #: any predicted gain. Scoring the OPTIMISTIC estimate (mean + kappa *
    #: neighbour spread) lets a candidate whose neighbours disagree be worth
    #: buying, which is the whole point of spending a real evaluation on it.
    #: Zero recovers the mean-only behaviour for ablation.
    #:
    #: PROVENANCE OF 0.5, so it does not later read as a hyperparameter chosen
    #: after seeing performance: it was selected by an OFFLINE sweep on
    #: ENGAGEMENT -- the fraction of decisions on which the acquisition can
    #: discriminate at all -- measured from cached fibers and archives, which is
    #: independent of any hypervolume outcome. The sweep ran BEFORE any charged
    #: comparison at 0.5 existed. Engagement was 16% / 24% / 40% / 88% / 100% at
    #: kappa 0 / 0.1 / 0.25 / 0.5 / 1.0 while pick quality fell monotonically,
    #: so the rule applied was "smallest kappa that engages", not "kappa that
    #: won". See diagnostics/task3_surrogate_acquisition_kind_mismatch.json.
    optimism: float = 0.5
    #: Predictions are noisy, so ranking on a razor-thin HVI difference is
    #: ranking on surrogate error. Ties inside this band fall back to the sum,
    #: which is a defensible secondary preference rather than an arbitrary one.
    tie_band: float = 1e-5
    _archive: object | None = field(default=None, init=False, repr=False)

    def target(self, archive: ParetoArchive,
               rng: np.random.Generator) -> Region | None:
        # The archive IS the target: what counts as valuable is "whatever adds
        # hypervolume to this", which moves as the archive fills.
        self._archive = archive
        return Region(target=(1.0,) * 5, gain=float(1.0 - archive.covered_fraction),
                      origin=None, note="reachable marginal hypervolume")

    def start(self, archive: ParetoArchive, target: Region,
              rng: np.random.Generator) -> str | None:
        """Expand from a front member, sampled by how much room it still has.

        Uniform over the front rather than argmax of anything: the acquisition
        does the discriminating, and biasing the start as well would make it
        impossible to tell which part was doing the work.
        """

        front = archive.front()
        if not front:
            return None
        return front[int(rng.integers(len(front)))][0]

    def rank(self, predicted: np.ndarray, target: Region,
             spread: np.ndarray | None = None) -> np.ndarray:
        if self._archive is None:
            return predicted.sum(axis=1)
        estimate = predicted
        if spread is not None and self.optimism:
            estimate = np.clip(predicted + self.optimism * spread, 0.0, 1.0)
        gains = self._archive.gains_of(estimate)
        # Break near-ties on the scalar sum instead of on prediction noise.
        return gains + self.tie_band * 1e-3 * predicted.sum(axis=1)


@dataclass
class NoveltyExploration:
    """PHASE A. Find the rare active basin, not the best hypervolume.

    A deliberate change of objective. Early Phase A is judged on DISCOVERY --
    evaluations to first active, actives found, distinct active regions -- and
    not on hypervolume, because the whole later pipeline is gated on having
    actives at all. A policy that finds actives faster with worse early HV is
    the right thing at this stage.

    WHY PURE STRUCTURAL NOVELTY, AND NO SURROGATE. From an official random-120
    start, JNK3 actives are roughly 2% of draws and a surrogate fit on 120
    inactive molecules has nothing to say about where actives are -- measured
    earlier at rho 0.28 on JNK3 with zero to four actives in the training
    window. An acquisition that leans on such a surrogate is leaning on noise.
    Structural coverage needs no labels to be well defined, which is exactly the
    property the blind regime demands.

    Candidates are scored by distance to the NEAREST already-evaluated molecule,
    so the policy buys the successor least like anything it has seen. That is
    coverage, not curiosity about predicted value, and it is one mechanism
    rather than a blend.
    """

    name: str = "novelty-exploration"
    _seen: np.ndarray | None = field(default=None, init=False, repr=False)
    _norms: np.ndarray | None = field(default=None, init=False, repr=False)

    def target(self, archive: ParetoArchive,
               rng: np.random.Generator) -> Region | None:
        # Fingerprints of everything evaluated so far; the "region" being aimed
        # at is simply whatever is far from all of it.
        from compose_v4.benchmark.oracles.forest import morgan_bits

        rows = [morgan_bits(s) for s in archive.values]
        rows = [r for r in rows if r is not None]
        if rows:
            self._seen = np.vstack(rows)
            self._norms = self._seen.sum(axis=1)
        return Region(target=(1.0,) * 5, gain=0.0, origin=None,
                      note="structural coverage")

    def start(self, archive: ParetoArchive, target: Region,
              rng: np.random.Generator) -> str | None:
        """Uniform over the whole archive, front or not.

        Exploration has no reason to prefer good molecules as launch points, and
        preferring them would quietly reintroduce exploitation through the back
        door.
        """

        if not archive.values:
            return None
        keys = list(archive.values)
        return keys[int(rng.integers(len(keys)))]

    def rank(self, predicted: np.ndarray, target: Region,
             spread: np.ndarray | None = None,
             candidates: list[str] | None = None) -> np.ndarray:
        if candidates is None or self._seen is None:
            return np.zeros(len(predicted))
        from compose_v4.benchmark.oracles.forest import morgan_bits

        rows, keep = [], []
        for position, smiles in enumerate(candidates):
            row = morgan_bits(smiles)
            if row is not None:
                rows.append(row)
                keep.append(position)
        scores = np.zeros(len(candidates))
        if not rows:
            return scores
        query = np.vstack(rows)
        intersection = query @ self._seen.T
        union = query.sum(axis=1)[:, None] + self._norms[None, :] - intersection
        similarity = intersection / np.maximum(union, 1e-9)
        # Distance to the NEAREST thing already evaluated.
        scores[keep] = 1.0 - similarity.max(axis=1)
        return scores
