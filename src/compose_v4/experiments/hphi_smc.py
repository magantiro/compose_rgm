"""Twisted-SMC mathematics for the QED/GrIDDD controller. MODEL-FREE CORE.

WHY THIS MODULE HAS NO MODEL IN IT
----------------------------------
The SMC mathematics -- weight accumulation, ESS, the resampling trigger, the
systematic resampler, and the terminal output rule -- is entirely independent of
`R_theta`, `h_phi` and chemistry. Separating it means it can be qualified
EXACTLY and locally: uniform weights must give ESS = N, a degenerate weight
vector must give ESS = 1, the trigger must fire precisely at ESS < N/2, and the
resampler must be reproducible from a seed. None of that needs a molecule.

That matters because SMC is *supposed* to produce different trajectories from
rejection, so it cannot be qualified by parity against the previous sampler.
The only available oracle is the mathematics itself.

WHAT IS PRESERVED FROM THE EXISTING CONTROLLER
-----------------------------------------------
`systematic_resample` is copied VERBATIM from
`scripts/griddd_value_guided_smc_controller.py`, whose SMC machinery already
matches the frozen specification (ESS as the inverse sum of squared normalized
weights, resampling at ESS < N/2, low-variance systematic resampling). It is
reproduced rather than reimplemented so the preserved layer cannot drift.

WHAT IS REPLACED
----------------
The old task wrapper tracked `best_qed` / `best_state` and returned
`best_feasible_qed` -- best-of-population -- under a 1000-oracle-call budget.
Preregistration section 13 bars both. One SMC run returns exactly ONE molecule,
sampled from the normalized terminal particle measure, and a source gets 20
INDEPENDENT runs. The N = 32 particles are inference state, never candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "N_PARTICLES", "ESS_FRACTION", "CANDIDATES_PER_SOURCE",
    "systematic_resample", "normalized_weights", "effective_sample_size",
    "should_resample", "sample_terminal_particle", "SMCStepOutcome",
]

#: Frozen by preregistration section 13. Inherited as the single pre-existing
#: operating point; NO particle-count sweep is permitted.
N_PARTICLES = 32
#: Resample when ESS < N * ESS_FRACTION. Frozen at one half.
ESS_FRACTION = 0.5
#: 20 INDEPENDENT SMC runs per source, one returned molecule each.
CANDIDATES_PER_SOURCE = 20


def systematic_resample(weights: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """Return resampled indices (low-variance/systematic).

    VERBATIM from the existing qualified controller -- do not "improve" it.
    A single uniform draw positions the whole comb, which is what makes this
    low-variance rather than multinomial.
    """
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    """Max-shifted softmax. The shift is numerical only and changes nothing."""
    shifted = np.asarray(log_weights, dtype=float)
    shifted = shifted - shifted.max()
    w = np.exp(shifted)
    return w / w.sum()


def effective_sample_size(weights: np.ndarray) -> float:
    """ESS = 1 / sum_i w_i^2 for NORMALIZED weights.

    Uniform weights give exactly N; a point mass gives exactly 1.
    """
    w = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(w ** 2))


def should_resample(weights: np.ndarray, n_particles: int | None = None) -> bool:
    """The frozen degeneracy trigger: ESS < N/2. Strict inequality."""
    n = len(weights) if n_particles is None else n_particles
    return effective_sample_size(weights) < n * ESS_FRACTION


def sample_terminal_particle(weights: np.ndarray,
                             rng: np.random.Generator) -> int:
    """THE OUTPUT RULE. One index drawn from the normalized terminal measure.

        J ~ Categorical(w_1/sum w, ..., w_N/sum w)

    This is what preregistration section 13 freezes, and it is deliberately NOT
    best-of-population: no highest-QED particle, no first particle to hit the
    region, no top-k. Those would convert inference state into candidate budget.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return int(rng.choice(len(w), p=w))


@dataclass(frozen=True)
class SMCStepOutcome:
    """One synchronisation point: normalize, measure ESS, resample if degenerate.

    Particles propagate independently BEFORE this; the population is coupled
    only here. That is the entire parallel structure of the algorithm.
    """

    weights: np.ndarray
    ess: float
    resampled: bool
    indices: np.ndarray

    @property
    def n_unique(self) -> int:
        """Distinct particle indices surviving. After resampling this is
        typically far below N, which is what makes exact duplicate-state
        computation reuse worthwhile later."""
        return len(np.unique(self.indices))


def smc_synchronisation(log_weights: np.ndarray,
                        rng: np.random.Generator) -> SMCStepOutcome:
    """The population-coupling step, in one place so it can be tested alone."""
    w = normalized_weights(log_weights)
    n = len(w)
    if should_resample(w, n):
        idx = systematic_resample(w, rng)
        return SMCStepOutcome(weights=w, ess=effective_sample_size(w),
                              resampled=True, indices=idx)
    return SMCStepOutcome(weights=w, ess=effective_sample_size(w),
                          resampled=False, indices=np.arange(n))
