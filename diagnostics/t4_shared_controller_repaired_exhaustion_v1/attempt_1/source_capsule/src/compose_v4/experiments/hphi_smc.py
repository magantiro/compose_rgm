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
    "EXTINCT_NO_HIT", "is_extinct", "terminal_output",
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


#: Frozen extinction status. See preregistration section 13.1.
EXTINCT_NO_HIT = "EXTINCT_NO_HIT"


def is_extinct(log_weights: np.ndarray) -> bool:
    """True when the terminal normalizing constant Z_H is exactly zero.

    With the exact terminal potential h_0(x) = 1[x in B], every particle
    outside the region receives weight zero at the final step. If no particle
    entered B the whole population dies, and the target-conditioned measure has
    NO SAMPLED SUPPORT -- so "sample from the normalized terminal weights" is
    undefined, not merely awkward. This is the known collapse of particle
    filters under indicator potentials.
    """
    lw = np.asarray(log_weights, dtype=float)
    return bool(np.all(np.isneginf(lw)) or np.all(np.exp(
        lw - (lw.max() if np.isfinite(lw.max()) else 0.0)) == 0.0))


def terminal_output(log_weights: np.ndarray, rng: np.random.Generator):
    """FROZEN output rule including extinction. Returns (index_or_None, status).

    Z_H > 0  -> sample one particle from the normalized terminal measure.
    Z_H == 0 -> EXTINCT_NO_HIT. The caller returns the canonical source x_0
                SOLELY to satisfy the benchmark's fixed 20-output interface.

    An extinct slot is ALWAYS a benchmark failure: no particle ever entered B,
    so no fallback could have qualified. Deliberately NOT last-nondegenerate,
    best-particle, uniform-particle, or reward-ranked -- each of those would
    return the molecule the sampler liked just before failing, which flatters
    secondary statistics without changing primary success.
    """
    if is_extinct(log_weights):
        return None, EXTINCT_NO_HIT
    w = normalized_weights(log_weights)
    return sample_terminal_particle(w, rng), "OK"


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


# ---------------------------------------------------------------------------
# The composed recursion. Generic over the process so it can be qualified
# END-TO-END against an analytically known controlled law before any molecule
# is involved. The molecular runner supplies molecular callables; the toy
# qualification supplies enumerable ones. Same code path either way.
# ---------------------------------------------------------------------------


def run_twisted_smc(
    x0,
    *,
    horizon: int,
    n_particles: int,
    propose,          # (x, budget, rng) -> y     sample from the REFERENCE law R
    twist,            # (x, budget) -> h_b(x)     the learned/known value
    in_target,        # (x) -> bool               goal region B_z
    rng: np.random.Generator,
    resample: bool = True,
):
    """Twisted SMC targeting the Doob h-transform of the reference process.

    THE WEIGHT IS THE WHOLE POINT. Proposing from R while targeting R*h makes
    the incremental weight

        w_t  =  h_{b-1}(y) / h_b(x)

    which telescopes over a path to h_0(x_T) / h_H(x_0). Multiplying h in
    "somewhere intuitive" instead -- using h(y) alone, or h(y)/h(y), or applying
    it twice -- yields a sampler that still has a correct ESS and a correct
    resampler and is confidently wrong. That is the failure mode the end-to-end
    test exists to catch.

    Absorption: a particle in the target set STOPS and is frozen thereafter, so
    its state and weight stop evolving. This is the STOP semantics of the
    controller, not a convergence trick.
    """
    states = [x0] * n_particles
    absorbed = [bool(in_target(x0))] * n_particles
    log_w = np.zeros(n_particles)
    n_resamples = 0
    unique_after_resample: list[int] = []

    for step in range(horizon):
        b = horizon - step                      # remaining budget BEFORE the move
        for i in range(n_particles):
            if absorbed[i]:
                continue                        # STOP: frozen, weight unchanged
            x = states[i]
            hx = twist(x, b)
            if hx <= 0.0:
                log_w[i] = -np.inf              # dead end under the target law
                absorbed[i] = True
                continue
            y = propose(x, b, rng)
            hy = twist(y, b - 1)
            # incremental Feynman-Kac weight, in log space
            log_w[i] += (np.log(hy) if hy > 0 else -np.inf) - np.log(hx)
            states[i] = y
            if in_target(y):
                absorbed[i] = True
        if np.all(np.isneginf(log_w)):
            break                               # every particle died
        if resample:
            out = smc_synchronisation(log_w, rng)
            if out.resampled:
                n_resamples += 1
                unique_after_resample.append(out.n_unique)
                states = [states[j] for j in out.indices]
                absorbed = [absorbed[j] for j in out.indices]
                log_w = np.zeros(n_particles)   # weights reset after resampling
        if all(absorbed):
            break

    w = normalized_weights(log_w)
    j = sample_terminal_particle(w, rng)
    return {
        "returned": states[j],                  # THE one candidate for this run
        "states": states, "weights": w, "absorbed": absorbed,
        "ess": effective_sample_size(w),
        "n_resamples": n_resamples,
        "mean_unique_after_resample": (float(np.mean(unique_after_resample))
                                       if unique_after_resample else None),
    }
