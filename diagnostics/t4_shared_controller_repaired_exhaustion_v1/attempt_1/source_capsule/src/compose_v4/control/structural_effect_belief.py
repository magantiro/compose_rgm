"""Joint Gaussian belief over what intervention coordinates do to docking score.

Latent utility is separated from measurement throughout:

    Y_{p,i} = mu_p + theta_i + eps_{p,i},     eps ~ N(0, sigma^2)

`mu_p` is the parent's own unknown quality. For two siblings of the SAME parent

    Y_{p,i} - Y_{p,j} = theta_i - theta_j + (eps_{p,i} - eps_{p,j})

and `mu_p` cancels identically. That cancellation, not a noise discount, is the reason
matched interventions are informative, so it is modelled rather than approximated: a
contrast is a linear observation on `theta_i - theta_j`, and an unmatched observation
carries the full variance of `mu_p` on top of the measurement noise.

The posterior is therefore a JOINT Gaussian over the effect vector with a full precision
matrix. A contrast contributes

    (1 / 2 sigma^2) * (e_i - e_j)(e_i - e_j)^T

which is rank one and correlates the two effects; it constrains their DIFFERENCE and says
nothing about their common level. An independent per-effect update cannot represent that,
and splitting a contrast's credit evenly between two independent precisions -- which is
what this module did first -- makes matched and unmatched evidence numerically identical.
A test caught it; the docstring had claimed an advantage the arithmetic did not deliver.

Partial pooling toward a per-coordinate mean, and that mean toward zero, is what keeps
twenty observations from being over-read.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

SCHEMA_VERSION = "structural_effect_belief_v2"


class EffectBelief:
    """Joint posterior over coordinate-value effects, updated from contrasts."""

    def __init__(
        self,
        *,
        noise: float = 0.35,
        value_scale: float = 0.8,
        coordinate_scale: float = 0.5,
        parent_confound: float = 1.0,
    ):
        self.noise = noise
        self.value_scale = value_scale
        self.coordinate_scale = coordinate_scale
        self.parent_confound = parent_confound
        self._index: dict[tuple, int] = {}
        self._keys: list[tuple] = []
        self._information = np.zeros((0, 0))  # accumulated precision from data
        self._score = np.zeros(0)  # accumulated precision-weighted evidence
        self._coordinate_evidence = defaultdict(list)

    # ---- structure ----

    def _slot(self, coordinate: str, value) -> int:
        key = (coordinate, value)
        if key not in self._index:
            self._index[key] = len(self._keys)
            self._keys.append(key)
            size = len(self._keys)
            grown = np.zeros((size, size))
            grown[: size - 1, : size - 1] = self._information
            self._information = grown
            self._score = np.append(self._score, 0.0)
        return self._index[key]

    def _prior(self):
        """Prior mean and precision, with each effect pooled toward its coordinate."""
        n = len(self._keys)
        mean = np.asarray([self.coordinate_mean(c) for c, _ in self._keys])
        precision = np.eye(n) / (self.value_scale**2)
        return mean, precision

    # ---- evidence ----

    def observe_contrast(self, coordinate: str, better_value, worse_value, gain: float) -> None:
        """A matched pair: `better_value` beat `worse_value` by `gain` from one parent.

        Contributes a rank-one precision on the DIFFERENCE. The pair is silent about the
        common level of the two effects, and the joint form represents that honestly
        instead of inventing a level for each.
        """
        i, j = self._slot(coordinate, better_value), self._slot(coordinate, worse_value)
        contrast = np.zeros(len(self._keys))
        contrast[i], contrast[j] = 1.0, -1.0
        precision = 1.0 / (2.0 * self.noise**2)
        self._information += precision * np.outer(contrast, contrast)
        self._score += precision * contrast * gain
        self._coordinate_evidence[coordinate].append(gain / 2.0)

    def observe_absolute(self, coordinate: str, value, gain: float) -> None:
        """An unmatched observation: the parent's own quality is not cancelled."""
        i = self._slot(coordinate, value)
        direction = np.zeros(len(self._keys))
        direction[i] = 1.0
        precision = 1.0 / (self.noise**2 + self.parent_confound**2)
        self._information += precision * np.outer(direction, direction)
        self._score += precision * direction * gain
        self._coordinate_evidence[coordinate].append(gain)

    # ---- posterior ----

    def coordinate_mean(self, coordinate: str) -> float:
        evidence = self._coordinate_evidence.get(coordinate, [])
        if not evidence:
            return 0.0
        n = len(evidence)
        shrink = (self.coordinate_scale**2) / (self.coordinate_scale**2 + self.noise**2 / n)
        return shrink * float(np.mean(evidence))

    def _moments(self):
        if not self._keys:
            return np.zeros(0), np.zeros((0, 0))
        prior_mean, prior_precision = self._prior()
        precision = prior_precision + self._information
        covariance = np.linalg.inv(precision)
        mean = covariance @ (prior_precision @ prior_mean + self._score)
        return mean, covariance

    def posterior(self, coordinate: str, value) -> tuple[float, float]:
        key = (coordinate, value)
        if key not in self._index:
            return self.coordinate_mean(coordinate), self.value_scale
        mean, covariance = self._moments()
        i = self._index[key]
        return float(mean[i]), float(math.sqrt(max(covariance[i, i], 1e-12)))

    def sample(self, coordinate: str, value, rng) -> float:
        mean, sd = self.posterior(coordinate, value)
        return float(rng.normal(mean, sd))

    def sample_joint(self, pairs, rng) -> np.ndarray:
        """One coherent draw over several effects, respecting their correlations.

        Sampling each effect independently would break the correlation a contrast
        induces, and a rollout that does so can believe both arms of a measured pair are
        simultaneously excellent -- which the pair explicitly did not say.
        """
        if not pairs:
            return np.zeros(0)
        known = [p for p in pairs if p in self._index]
        draws = {}
        if known:
            mean, covariance = self._moments()
            rows = [self._index[p] for p in known]
            sub_mean = mean[rows]
            sub_cov = covariance[np.ix_(rows, rows)]
            sub_cov = sub_cov + 1e-9 * np.eye(len(rows))
            values = rng.multivariate_normal(sub_mean, sub_cov)
            draws = dict(zip(known, values))
        return np.asarray(
            [draws.get(p, rng.normal(self.coordinate_mean(p[0]), self.value_scale)) for p in pairs]
        )

    def observations(self) -> int:
        return sum(len(v) for v in self._coordinate_evidence.values())


def posterior_incumbent(scores, *, noise: float) -> float:
    """A noise-aware incumbent.

    The best measured score is a biased estimate of the best latent utility, because a
    maximum over noisy measurements selects for upward error. Under an extreme-value
    objective that is the mistake the objective would reward, so an extreme sitting
    inside the noise is pulled toward the field.
    """
    if not scores:
        return float("inf")
    ordered = sorted(scores)
    if len(ordered) == 1:
        return ordered[0]
    best, second = ordered[0], ordered[1]
    gap = second - best
    trust = gap / (gap + 2.0 * noise) if gap > 0 else 0.0
    return best + (1.0 - trust) * min(gap, 2.0 * noise) * 0.5
