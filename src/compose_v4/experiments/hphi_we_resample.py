"""Committor-stratified Weighted Ensemble resampling. ONLY the resampler.

Everything else in the controller is frozen: R_theta, the qualified lazy
transition sampler, h_phi(x, b), the target definition, the ESS trigger, the
horizon, the particle count, and the Feynman-Kac incremental weights. This
module replaces one operation.

WHY. Global resampling lets a single high-h_phi basin win every slot, so
distinct progress frontiers are wiped out precisely when the population should
be hedging across them. Weighted Ensemble instead resamples WITHIN progress
strata, so particles compete only against others at similar committor value.

THE WEIGHT RULE IS THE CORRECTNESS CONDITION. If stratum s carries total
incoming weight W_s and receives n_s descendants, every descendant gets W_s/n_s.
Resetting all particles to 1/N -- which ordinary SMC may do because its single
stratum makes that identical -- would destroy the relative weighting BETWEEN
strata and silently change the target measure. Splitting and merging
trajectories while preserving per-stratum weight is what makes WE statistically
exact for a broad class of processes and binning rules; the binning may even be
adaptive, provided weights are preserved.

h_phi is used as an approximate finite-horizon committor, i.e. a progress
coordinate. It does not need to be a calibrated probability for the resampling
to be valid -- a wrong coordinate makes WE inefficient, not incorrect, because
the weights carry the distribution.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

__all__ = ["systematic_n", "we_resample"]


def systematic_n(probabilities: np.ndarray, n: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Low-variance systematic resampling generalised to n != len(p) draws.

    One uniform draw positions the whole comb, which is what makes this
    systematic rather than multinomial. The frozen controller's resampler is the
    n == len(p) special case of exactly this.
    """
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def we_resample(
    keys: Sequence[Any],
    weights: np.ndarray,
    score: Callable[[Any], float],
    rng: np.random.Generator,
    *,
    n_slots: int,
    n_strata: int = 4,
) -> tuple[list[int], np.ndarray, dict[str, Any]]:
    """One WE resampling event over LIVE particles.

    `keys[i]` identifies particle i's state; particles sharing a key occupy the
    same Markov state and are merged. `weights` are the incoming (unnormalised,
    non-negative) weights. `score` maps a key to its committor coordinate.

    Returns (parent indices, descendant weights, diagnostics). The parent list
    has length `n_slots`; descendant weights are absolute, NOT normalised, so
    the caller must not renormalise them to uniform.
    """
    live = [i for i in range(len(keys)) if weights[i] > 0.0
            and np.isfinite(weights[i])]
    if not live:
        return [], np.zeros(0), {"strata": 0, "merged": 0, "unique": 0}

    # ---- merge identical states, preserving summed weight -----------------
    # Same canonical state at the same remaining budget IS the same Markov
    # state, so treating copies as distinct frontiers would let duplication
    # alone dominate a stratum. The representative is drawn in proportion to
    # incoming weight rather than taken deterministically, so no lineage is
    # discarded by position.
    groups: dict[Any, list[int]] = {}
    for i in live:
        groups.setdefault(keys[i], []).append(i)
    merged: list[tuple[float, Any, int, float]] = []
    for key, members in groups.items():
        total = float(sum(weights[i] for i in members))
        if len(members) == 1:
            rep = members[0]
        else:
            p = np.array([weights[i] for i in members], dtype=float)
            rep = members[int(rng.choice(len(members), p=p / p.sum()))]
        merged.append((float(score(key)), key, rep, total))

    # ---- sort by committor, deterministic tie-break ------------------------
    merged.sort(key=lambda t: (t[0], repr(t[1])))
    m = len(merged)
    k = max(1, min(int(n_strata), m))

    # ---- contiguous RANK strata, no tuned thresholds ----------------------
    edges = [round(j * m / k) for j in range(k + 1)]
    strata = [merged[edges[j]:edges[j + 1]] for j in range(k)]
    strata = [s for s in strata if s]

    # ---- slots split as evenly as possible over OCCUPIED strata -----------
    base, extra = divmod(int(n_slots), len(strata))
    counts = [base + (1 if j < extra else 0) for j in range(len(strata))]

    parents: list[int] = []
    out_w: list[float] = []
    for stratum, n_s in zip(strata, counts):
        if n_s <= 0:
            continue
        w_s = np.array([t[3] for t in stratum], dtype=float)
        total = float(w_s.sum())
        idx = systematic_n(w_s / total, n_s, rng)
        for j in idx:
            parents.append(stratum[int(j)][2])
            # THE correctness condition: every descendant of stratum s carries
            # W_s / n_s, so the stratum's total weight is preserved exactly and
            # the relative weighting BETWEEN strata survives.
            out_w.append(total / n_s)

    return parents, np.asarray(out_w, dtype=float), {
        "strata": len(strata), "merged": len(live) - m, "unique": m,
        "counts": counts,
    }
