"""Segmentation of a primitive route into macro-options with monotone boundary scores.

A COMPOSE macro executes ATOMICALLY and only its endpoint is oracle-scored: measured,
`intermediate_task_evaluations` is 0 across every composed program, and no composed
program truncates.  So the primitive-level score dips a route shows are INVISIBLE to a
score-greedy controller -- it only ever observes the states at MACRO BOUNDARIES.

The question that bears on controller design is therefore not whether the primitive
trajectory is monotone, but whether a segmentation into macros exists whose boundary
scores never decrease, subject to each macro fitting the measured realization ceiling.

`min_segments` answers that exactly by dynamic programming over boundary indices.
`source_relative_tolerance` answers the follow-up: when no monotone staging exists, how
far below its own starting score must the controller be willing to go.

INVARIANT the tolerance respects, and the reason a per-boundary tolerance is the wrong
quantity: a per-boundary allowance accumulates without bound across many boundaries, so
it converges to zero on every route and measures nothing.  The tolerance here is
measured against the SOURCE score, which does not accumulate.
"""
from __future__ import annotations


def min_segments(
    scores: list[float], max_len: int | None = None, strict: bool = False
) -> tuple[int | None, list[int] | None]:
    """Fewest contiguous segments whose boundary scores never decrease.

    Returns ``(k, boundaries)``, or ``(None, None)`` when no such segmentation exists.
    ``max_len`` caps primitives per segment; ``None`` leaves it unbounded.
    """
    n = len(scores) - 1
    if n < 0:
        return None, None
    if n == 0:
        return 0, [0]
    inf = float("inf")
    best = [inf] * (n + 1)
    prev = [-1] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        lo = 0 if max_len is None else max(0, j - max_len)
        for i in range(lo, j):
            if best[i] == inf:
                continue
            ordered = scores[i] < scores[j] if strict else scores[i] <= scores[j]
            if ordered and best[i] + 1 < best[j]:
                best[j] = best[i] + 1
                prev[j] = i
    if best[n] == inf:
        return None, None
    bounds, j = [n], n
    while j > 0:
        j = prev[j]
        bounds.append(j)
    return int(best[n]), list(reversed(bounds))


def _feasible_within(scores: list[float], max_len: int, slack: float) -> bool:
    """Is a segmentation available in which every boundary is >= source - slack?"""
    n = len(scores) - 1
    inf = float("inf")
    best = [inf] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        if j < n and scores[j] < scores[0] - slack:
            continue
        for i in range(max(0, j - max_len), j):
            if best[i] < inf:
                best[j] = min(best[j], best[i] + 1)
    return best[n] < inf


def source_relative_tolerance(scores: list[float], max_len: int, iterations: int = 48) -> float:
    """Smallest drop below the SOURCE score that admits a staging; 0.0 when monotone."""
    if min_segments(scores, max_len)[0] is not None:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if _feasible_within(scores, max_len, mid):
            hi = mid
        else:
            lo = mid
    return hi


def primitive_dips_below_source(scores: list[float]) -> bool:
    """Does the PRIMITIVE trajectory pass below where it started?"""
    return min(scores) < scores[0] - 1e-12


def describe(scores: list[float], max_len: int = 23) -> dict:
    """Full segmentation read of one route.  ``max_len`` defaults to the measured ceiling."""
    k_free, b_free = min_segments(scores, None)
    k_cap, b_cap = min_segments(scores, max_len)
    sizes = None if b_cap is None else [b_cap[i + 1] - b_cap[i] for i in range(len(b_cap) - 1)]
    return {
        "n_primitives": len(scores) - 1,
        "source": scores[0],
        "endpoint": scores[-1],
        "min_primitive_score": min(scores),
        "primitive_dips_below_source": primitive_dips_below_source(scores),
        "min_segments_unbounded": k_free,
        "min_segments_capped": k_cap,
        "boundaries_capped": b_cap,
        "boundary_scores_capped": None if b_cap is None else [scores[i] for i in b_cap],
        "macro_sizes": sizes,
        "source_relative_tolerance": source_relative_tolerance(scores, max_len),
        "macro_feasible_within_five": k_cap is not None and k_cap <= 5,
    }
