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
