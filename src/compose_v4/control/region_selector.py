"""mu_exec(M | x) and Q(M | x, z): which region to rewrite, and at what scale.

The inner controller is frozen and qualified -- given a viable region it
navigates the rewrite. This is the outer half: choosing a region worth spending
that controller on. It splits into a TASK-INDEPENDENT prior and a task tilt:

    mu_exec(M|x)  ∝  p_feas(M|x) / E[C(M)|x]  *  pi(s(M)) / |M_s(x)|

        which regions are executable and computationally worthwhile at all.
        Dividing by measured cost puts the units right: a budgeted search
        maximises expected successes per SECOND, not probability of success, so
        a 0.8-scope region succeeding 46% at 448s is worth less than a local
        edit succeeding 40% at 60s. The last factor balances scale.

    Q(M|x,z)      ∝  mu_exec(M|x) * exp( V_z(M,x) / tau_M )

        then tilt toward regions whose futures look promising for the current
        task. The tilt is exponential rather than multiplicative because task
        value has arbitrary sign and scale -- multiplying by a negative or
        unnormalised V would not even be a measure -- and because this is the
        same KL-control form the rest of COMPOSE uses.

Everything here is calibrated from the measured local-to-global gate (239
sampled regions, one particle each) rather than assumed. Two of those
measurements contradict the obvious prior and are worth stating:

  * interface class dominates scope. pendant 75%, segment 74%, multi 20%,
    splitting 1/91. A selector that reasons about size and ignores interface is
    reasoning about the wrong variable.
  * size is NOT monotone. 1-3 atoms 56%, 4-6 18%, 7-10 19%, 11-20 32%, 21+ 38%.
    Large regions are not the hard case; mid-size ones are.

Feasibility here is a calibrated empirical prior, deliberately not a network:
239 labelled regions do not support a learned model, and a smoothed contingency
table is honest about that. The structural committor already supplies the
per-state future value; this supplies the per-region prior that decides where to
point it.
"""

from __future__ import annotations

from dataclasses import dataclass

# Measured on the 2026-09-02 gate: successes / attempts.
_INTERFACE = {
    "pendant": (18, 24),
    "segment": (50, 68),
    "multi": (11, 56),
    "splitting": (1, 91),
}
_SIZE_BUCKETS = (
    (1, 3, 15, 27),
    (4, 6, 4, 22),
    (7, 10, 7, 36),
    (11, 20, 24, 76),
    (21, 10**6, 30, 78),
)
# Median wall-clock seconds for one attempt, by released fraction.
_COST_BANDS = (
    (0.0, 0.2, 60.0),
    (0.2, 0.4, 108.0),
    (0.4, 0.6, 190.0),
    (0.6, 0.8, 256.0),
    (0.8, 1.01, 448.0),
)
_GLOBAL = (80, 239)
_PRIOR_STRENGTH = 8.0  # pseudo-counts pulling a cell toward its backoff


def _smoothed(hits: int, n: int, backoff: float, strength: float = _PRIOR_STRENGTH) -> float:
    """Laplace-style shrink toward a backoff rate.

    Cells here are small -- pendant has 24 attempts -- so a raw ratio would
    swing wildly on one extra success. Shrinking toward the coarser rate keeps a
    thin cell from dominating the selector on noise.
    """
    return (hits + strength * backoff) / (n + strength)


def _size_rate(size: int, backoff: float) -> float:
    for lo, hi, hits, n in _SIZE_BUCKETS:
        if lo <= size <= hi:
            return _smoothed(hits, n, backoff)
    return backoff


def feasibility(interface: str, size: int) -> float:
    """P(a valid rewrite exists and the frozen controller finds it).

    Interface is the primary factor and size a correction, in that order,
    because that is the order the measurement puts them in.
    """
    g = _GLOBAL[0] / _GLOBAL[1]
    hits, n = _INTERFACE.get(interface, _GLOBAL)
    p_iface = _smoothed(hits, n, g)
    p_size = _size_rate(int(size), g)
    # geometric mean: neither factor alone should be able to veto the other,
    # and the product of two rates would be badly miscalibrated as a probability
    return float((p_iface * p_size) ** 0.5)


def cost_seconds(released_fraction: float) -> float:
    """Expected wall-clock for one attempt at this scope, from the gate."""
    r = float(released_fraction)
    for lo, hi, sec in _COST_BANDS:
        if lo <= r < hi:
            return sec
    return _COST_BANDS[-1][2]


@dataclass
class RegionScore:
    feasibility: float
    value: float
    cost_s: float
    mu_exec: float
    score: float
    interface: str
    size: int
    released_fraction: float
    # Populated by sample_region.  Keeping the normalized probabilities on the
    # receipt makes a selected (parent, region, option) bundle auditable without
    # changing the rank_regions public contract.
    base_probability: float | None = None
    floor_probability: float | None = None
    selection_probability: float | None = None


def score_region(region, value: float = 0.0, tau: float = 1.0) -> RegionScore:
    """mu_exec, and Q = mu_exec * exp(V/tau) when a task value is supplied.

    `value` defaults to 0.0, not 1.0: with no task the tilt must be exactly
    exp(0) = 1, leaving the task-independent prior untouched.
    """
    import math

    f = feasibility(region.interface, region.size)
    c = cost_seconds(region.released_fraction)
    mu = f / c
    tilt = math.exp(float(value) / max(1e-9, float(tau)))
    return RegionScore(
        feasibility=f,
        value=float(value),
        cost_s=c,
        mu_exec=mu,
        score=mu * tilt,
        interface=region.interface,
        size=int(region.size),
        released_fraction=float(region.released_fraction),
    )


def rank_regions(regions, value_fn=None, scale_balance: bool = True, tau: float = 1.0):
    """Rank regions by mu_exec, or by Q when a task value is supplied.

    `scale_balance` divides by how many regions share a scope band. Without it
    the selector drowns in small regions -- not because they score well, but
    because enumeration produces far more of them, so an unbalanced ranking
    reports the shape of the enumerator rather than a decision.
    """
    scored = [(r, score_region(r, 0.0 if value_fn is None else value_fn(r), tau)) for r in regions]
    if scale_balance:
        from collections import Counter

        band = lambda r: min(int(r.released_fraction * 5), 4)
        counts = Counter(band(r) for r, _ in scored)
        for r, s in scored:
            s.score /= max(1, counts[band(r)])
    return sorted(scored, key=lambda rs: -rs[1].score)


def region_distribution(
    regions, value_fn=None, tau: float = 1.0, epsilon: float = 0.2, scale_balance: bool = True
):
    """Return the existing ranked regions, scores and complete mixed law.

        Q = (1 - epsilon) Q_learned + epsilon mu_scale_balanced

    The floor is not decoration. mu_exec rewards successes per second, and cheap
    pendant edits win that trade almost every time -- 75% feasible at 60s
    against 46% at 448s. Left alone the search would collapse onto small local
    moves, which is precisely the failure mode the local-to-global work exists
    to avoid: a large region may have worse immediate success-per-second and
    still be the only route out of a basin. The floor guarantees every
    structural scale keeps a share of the budget.
    """
    import numpy as np

    if not regions:
        return [], [], np.asarray([], dtype=float)
    ranked = rank_regions(regions, value_fn=value_fn, tau=tau, scale_balance=scale_balance)
    regs = [r for r, _ in ranked]
    q = np.array([s.score for _, s in ranked], float)
    q = q / q.sum() if q.sum() > 0 else np.full(len(q), 1.0 / len(q))

    band = lambda r: min(int(r.released_fraction * 5), 4)
    from collections import Counter

    counts = Counter(band(r) for r in regs)
    nb = len(counts)
    # uniform over scale bands first, then uniform within a band: an unweighted
    # uniform would still be dominated by whichever band enumerates most
    floor = np.array([1.0 / (nb * counts[band(r)]) for r in regs], float)
    floor = floor / floor.sum()

    mix = (1.0 - epsilon) * q + epsilon * floor
    mix = mix / mix.sum()
    for j, (_region, score) in enumerate(ranked):
        score.base_probability = float(q[j])
        score.floor_probability = float(floor[j])
        score.selection_probability = float(mix[j])
    return regs, [score for _, score in ranked], mix


def sample_region(
    regions, rng, value_fn=None, tau: float = 1.0, epsilon: float = 0.2, scale_balance: bool = True
):
    """Sample the qualified region law without changing its arithmetic or RNG use."""
    regs, scores, mix = region_distribution(
        regions, value_fn=value_fn, tau=tau, epsilon=epsilon, scale_balance=scale_balance
    )
    if not regs:
        return None, None
    i = int(rng.choice(len(regs), p=mix))
    return regs[i], scores[i]
