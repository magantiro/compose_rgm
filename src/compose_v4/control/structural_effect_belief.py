"""Hierarchical belief over what intervention coordinates do to docking score.

Latent utility is separated from measurement throughout:

    Y(x) = U(x) + epsilon,        epsilon ~ N(0, sigma^2)

Under an extreme-value objective this is not pedantry. A controller that treats one
lucky -12 as truth will chase measurement noise, and the objective rewards exactly that
mistake by taking a maximum. So the posterior is over the latent effect and the reported
incumbent is a posterior quantity, not the best number ever seen.

The effect of setting coordinate `c` to value `v` is modelled as

    theta[c, v] ~ N(theta[c], tau^2)        theta[c] ~ N(0, tau0^2)

so a value borrows strength from its coordinate and a coordinate from zero. With twenty
observations that partial pooling is the whole game: an unpooled estimate of a value seen
twice is noise, and the earlier failures on this branch came from point estimates that
hid exactly that.

Observations arrive as CONTRASTS. For two options from the same parent differing in one
coordinate, the difference cancels every parent-level term exactly, which is what makes
the effect identifiable (see docs/CONTRASTIVE_PROGRAM_CONTROL.md section 3.2). Absolute
observations are accepted too but are far less informative, because they confound the
effect with the parent.

All updates are conjugate and closed form. There is no sampler to tune and no seed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

SCHEMA_VERSION = "structural_effect_belief_v1"


@dataclass
class EffectBelief:
    """Posterior over coordinate-value effects, updated from contrasts."""

    noise: float = 0.35  # docking measurement sd, kcal/mol
    value_scale: float = 0.8  # tau: spread of values within a coordinate
    coordinate_scale: float = 0.5  # tau0: spread of coordinates around zero
    parent_confound: float = 1.0
    """Spread of the unmeasured parent effect.

    This is what makes an unmatched observation weaker, and omitting it is not a
    simplification -- it silently makes absolute and contrastive evidence equally
    informative, which contradicts the entire argument for matched bundles. Left out,
    both routes reduced to precision 1/(4 sigma^2) and were numerically identical; the
    docstring claimed an advantage the code did not implement."""
    _precision: dict = field(default_factory=lambda: defaultdict(float))
    _weighted: dict = field(default_factory=lambda: defaultdict(float))
    _coordinate: dict = field(default_factory=lambda: defaultdict(list))

    def observe_contrast(self, coordinate: str, better_value, worse_value, gain: float) -> None:
        """One matched pair: setting `better_value` beat `worse_value` by `gain`.

        The contrast variance is 2 sigma^2 because both arms are measured. Credit is
        split symmetrically -- the pair says the difference, not which side moved -- and
        an asymmetric attribution would invent information the design does not carry.
        """
        variance = 2.0 * self.noise**2
        precision = 1.0 / variance
        for value, sign in ((better_value, +1.0), (worse_value, -1.0)):
            key = (coordinate, value)
            self._precision[key] += precision * 0.5
            self._weighted[key] += precision * 0.5 * sign * gain
            self._coordinate[coordinate].append(sign * gain * 0.5)

    def observe_absolute(self, coordinate: str, value, gain: float) -> None:
        """An unmatched observation: the effect is confounded with the parent's own quality.

        Its variance carries the parent spread as well as measurement noise, so it is
        strictly less informative than a contrast between two children of one parent,
        where the parent term cancels exactly.
        """
        precision = 1.0 / (self.noise**2 + self.parent_confound**2)
        key = (coordinate, value)
        self._precision[key] += precision
        self._weighted[key] += precision * gain
        self._coordinate[coordinate].append(gain)

    def coordinate_mean(self, coordinate: str) -> float:
        """Partially pooled mean for a coordinate, shrunk toward zero."""
        observations = self._coordinate.get(coordinate, [])
        if not observations:
            return 0.0
        n = len(observations)
        shrink = (self.coordinate_scale**2) / (self.coordinate_scale**2 + self.noise**2 / n)
        return shrink * float(np.mean(observations))

    def posterior(self, coordinate: str, value) -> tuple[float, float]:
        """Mean and sd of the latent effect of setting `coordinate` to `value`."""
        prior_mean = self.coordinate_mean(coordinate)
        prior_precision = 1.0 / (self.value_scale**2)
        key = (coordinate, value)
        precision = prior_precision + self._precision.get(key, 0.0)
        mean = (prior_precision * prior_mean + self._weighted.get(key, 0.0)) / precision
        return mean, math.sqrt(1.0 / precision)

    def sample(self, coordinate: str, value, rng) -> float:
        mean, sd = self.posterior(coordinate, value)
        return float(rng.normal(mean, sd))

    def observations(self) -> int:
        return sum(len(v) for v in self._coordinate.values())


def posterior_incumbent(scores, *, noise: float) -> float:
    """A noise-aware incumbent.

    The best measured score is a biased estimate of the best latent utility, because
    taking a maximum over noisy measurements selects for upward errors. Shrinking the
    extreme toward the runner-up keeps an outlier from anchoring the whole search.
    """
    if not scores:
        return float("inf")
    ordered = sorted(scores)
    if len(ordered) == 1:
        return ordered[0]
    best, second = ordered[0], ordered[1]
    gap = second - best
    # with a gap far above the noise the best is trusted; within noise it is pulled in
    trust = gap / (gap + 2.0 * noise) if gap > 0 else 0.0
    return best + (1.0 - trust) * min(gap, 2.0 * noise) * 0.5
