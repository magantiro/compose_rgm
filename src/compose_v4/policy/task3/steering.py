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
    """Arm B'. Aim at the missing region WITHOUT trading away what we already have.

    WHY THIS EXISTS. The first adaptive rule lost the mechanism test on its own
    criterion. It climbed JNK3 hard -- 65-72% of its charged molecules reached
    JNK3 >= 0.4, against 26-37% for the fixed sum -- but the molecules it found
    were promiscuous: mean GSK3B activity 0.25-0.29 among them, against
    0.03-0.15 for the fixed sum, and it produced 7 selective molecules against
    188.

    The defect is in how the aspiration is SHAPED, not in adaptivity. Aspirations
    are built by improving ONE axis of a realized molecule, and ranked by SUMMED
    shortfall. A summed shortfall lets a large gain on the targeted axis pay for
    a loss on another, so "more JNK3, everything else as it was" is satisfied by
    a molecule that raises JNK3 and quietly gives up GSK3B -- which, on this
    task's chemistry, is most of what is reachable.

    So the aspiration becomes a JOINT demand: make progress on the targeted axis
    AND do not fall below where the origin already was on any other. The
    non-targeted axes are a constraint, not a term to be traded off. That is the
    difference between "fill the missing part of the front" and "climb the
    biggest axis".
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
