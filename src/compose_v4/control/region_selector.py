"""Q(M | x, z): which region to rewrite, and at what scale.

The inner controller is frozen and qualified -- given a viable region it
navigates the rewrite. This is the outer half: choosing a region worth spending
that controller on. It factorises the way the search actually needs:

    score(M) = feasibility(M) * value(M) / cost(M)

    feasibility   can this region be rewritten at all
    value         would rewriting it help (task-dependent, injected)
    cost          what the inner controller will spend trying

Dividing by cost is what makes the units right: the quantity a budgeted search
maximises is expected successes per second, not probability of success. A
scaffold rewrite that succeeds 46% of the time but costs 448s is worth less per
second than a local edit succeeding 40% at 60s, and only this form says so.

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

from dataclasses import dataclass, field

# Measured on the 2026-09-02 gate: successes / attempts.
_INTERFACE = {
    "pendant": (18, 24),
    "segment": (50, 68),
    "multi": (11, 56),
    "splitting": (1, 91),
}
_SIZE_BUCKETS = ((1, 3, 15, 27), (4, 6, 4, 22), (7, 10, 7, 36),
                 (11, 20, 24, 76), (21, 10 ** 6, 30, 78))
# Median wall-clock seconds for one attempt, by released fraction.
_COST_BANDS = ((0.0, 0.2, 60.0), (0.2, 0.4, 108.0), (0.4, 0.6, 190.0),
               (0.6, 0.8, 256.0), (0.8, 1.01, 448.0))
_GLOBAL = (80, 239)
_PRIOR_STRENGTH = 8.0        # pseudo-counts pulling a cell toward its backoff


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
    score: float
    interface: str
    size: int
    released_fraction: float


def score_region(region, value: float = 1.0) -> RegionScore:
    """Expected successful rewrites per second of controller time."""
    f = feasibility(region.interface, region.size)
    c = cost_seconds(region.released_fraction)
    return RegionScore(feasibility=f, value=float(value), cost_s=c,
                       score=f * float(value) / c, interface=region.interface,
                       size=int(region.size),
                       released_fraction=float(region.released_fraction))


def rank_regions(regions, value_fn=None, scale_balance: bool = True):
    """Rank regions by expected successes per second.

    `scale_balance` divides by how many regions share a scope band. Without it
    the selector drowns in small regions -- not because they score well, but
    because enumeration produces far more of them, so an unbalanced ranking
    reports the shape of the enumerator rather than a decision.
    """
    scored = [(r, score_region(r, 1.0 if value_fn is None else value_fn(r)))
              for r in regions]
    if scale_balance:
        from collections import Counter
        band = lambda r: min(int(r.released_fraction * 5), 4)
        counts = Counter(band(r) for r, _ in scored)
        for r, s in scored:
            s.score /= max(1, counts[band(r)])
    return sorted(scored, key=lambda rs: -rs[1].score)
