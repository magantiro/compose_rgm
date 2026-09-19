"""Choosing what to aim at, and what to aim from.

The valuable region is not the best point in objective space -- that is always
the (1,1,1,1,1) corner, and aiming at it says nothing. It is the point that is
BOTH worth reaching and plausibly reachable, so aspirations are built as bounded
improvements on molecules we have actually realized, and scored by what reaching
them would add to the hypervolume.

Start states are drawn from the WHOLE archive rather than the current front.
That is deliberate and is a COMPOSE affordance: a dominated molecule can still
be the best place to attack a region from, and re-using a previously realized
state costs nothing because it has already been paid for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compose_v4.policy.task3.archive import ParetoArchive
from compose_v4.policy.task3.interfaces import Region

#: How far an aspiration may reach beyond what has been realized, per axis.
#: Small steps are nearly free to reach and add little; large steps add a lot
#: and are usually fantasy. The set is deliberately short -- three scales, not a
#: swept hyperparameter.
ASPIRATION_STEPS = (0.05, 0.15, 0.35)


@dataclass
class MarginalGainRegions:
    """Aspiration points built from realized molecules, scored by HV gain.

    For each front member and each objective, propose "this molecule, but better
    on that one axis". Each proposal is scored by the probe-set hypervolume it
    would add, and the best is selected -- with an epsilon-greedy sample so the
    policy does not lock onto one axis forever.
    """

    name: str = "marginal-gain"
    epsilon: float = 0.15
    front_limit: int = 24

    def select(self, archive: ParetoArchive,
               rng: np.random.Generator) -> Region | None:
        front = archive.front()[: self.front_limit]
        if not front:
            return None
        targets, origins, notes = [], [], []
        for smiles, values in front:
            base = np.asarray(values, dtype=float)
            for axis in range(len(base)):
                for step in ASPIRATION_STEPS:
                    target = base.copy()
                    target[axis] = min(1.0, target[axis] + step)
                    if target[axis] == base[axis]:
                        continue          # already saturated on this axis
                    targets.append(target)
                    origins.append(smiles)
                    notes.append(f"axis {axis} +{step}")
        if not targets:
            return None
        gains = archive.gains_of(np.asarray(targets))
        if float(gains.max()) <= 0.0:
            return None
        if rng.random() < self.epsilon:
            # Explore among aspirations that are worth anything at all.
            worthwhile = np.flatnonzero(gains > 0)
            index = int(rng.choice(worthwhile))
        else:
            index = int(np.argmax(gains))
        return Region(target=tuple(targets[index]), gain=float(gains[index]),
                      origin=origins[index], note=notes[index])


@dataclass
class RandomRegion:
    """Control: an arbitrary undominated aspiration. No archive reasoning.

    Kept so that "the region targeting is doing the work" is a claim that can be
    tested by removing exactly the region targeting.
    """

    name: str = "random-region"
    attempts: int = 32

    def select(self, archive: ParetoArchive,
               rng: np.random.Generator) -> Region | None:
        for _ in range(self.attempts):
            target = rng.random(5)
            gain = archive.gain_of(target)
            if gain > 0:
                return Region(target=tuple(target), gain=gain,
                              origin=None, note="uniform")
        return None


@dataclass
class NearestRealizedStart:
    """The realized molecule with the least ground to make up to the region.

    Shortfall is summed only over axes where the molecule falls SHORT of the
    target: being far above the target on some axis is not a cost, it is
    headroom.

    A PROPERTY WORTH KNOWING BEFORE RELYING ON THIS FOR STATE REUSE. If x
    dominates y then x >= y on every axis, so x's shortfall to any target is <=
    y's. Objective-space proximity can therefore TIE with a dominated state but
    can never strictly prefer one. Searching the whole archive is still right --
    ties are common and the temperature makes them reachable -- but genuine
    reuse of dominated states needs a criterion from OUTSIDE objective space
    (structural novelty, edit distance, how well a scaffold has served before).
    That is a real Phase 2 design question and is deliberately left open rather
    than faked by a selector that cannot deliver it.
    """

    name: str = "nearest-realized"
    temperature: float = 0.05

    def select(self, archive: ParetoArchive, region: Region,
               rng: np.random.Generator) -> str | None:
        if not archive.values:
            return None
        keys = list(archive.values)
        points = archive.points()
        target = region.as_array()
        shortfall = np.maximum(target[None, :] - points, 0.0).sum(axis=1)
        if self.temperature <= 0:
            return keys[int(np.argmin(shortfall))]
        # Softmin: usually the closest, sometimes a near neighbour, so one
        # unlucky start does not pin the whole run.
        weights = np.exp(-shortfall / self.temperature)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return keys[int(np.argmin(shortfall))]
        return keys[int(rng.choice(len(keys), p=weights / total))]


@dataclass
class RegionOriginStart:
    """Start from the molecule the aspiration was built out of.

    A different principled answer to the same question, and the natural partner
    to `MarginalGainRegions`: the region was defined as "this molecule, but
    better on one axis", so this molecule is the obvious place to attack it from.
    """

    name: str = "region-origin"

    def select(self, archive: ParetoArchive, region: Region,
               rng: np.random.Generator) -> str | None:
        if region.origin and region.origin in archive.values:
            return region.origin
        return NearestRealizedStart().select(archive, region, rng)
