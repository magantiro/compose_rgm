"""Reference-propagated particles at completed-option boundaries.

No chemistry or proposal truncation here. q=B, so only potential ratios enter
weights. The unnormalized initial potential is one for every initial particle.
"""

from __future__ import annotations

import numpy as np

from compose_v4.experiments.hphi_smc import (
    effective_sample_size,
    normalized_weights,
    systematic_resample,
)


def advance(log_weights, previous_log_potential, next_log_potential, alive, rng, *, resample):
    """Accumulate before resampling; None is a serialized zero weight/death."""
    n = len(log_weights)
    if not n or any(len(x) != n for x in (previous_log_potential, next_log_potential, alive)):
        raise ValueError("particle arrays must have equal nonzero length")
    updated = np.full(n, -np.inf)
    for i in range(n):
        if alive[i] and log_weights[i] is not None:
            old, new, weight = previous_log_potential[i], next_log_potential[i], log_weights[i]
            if old is None or new is None or not np.isfinite([old, new, weight]).all():
                raise ValueError("live particles require finite log weights and potentials")
            updated[i] = weight + new - old
    if np.isneginf(updated).all():
        return {
            "status": "extinct",
            "weights": [0.0] * n,
            "ess": 0.0,
            "resampled": False,
            "indices": list(range(n)),
            "log_weights": [None] * n,
            "log_potential": list(next_log_potential),
        }
    weights = normalized_weights(updated)
    ess = effective_sample_size(weights)
    trigger = bool(resample and ess < n / 2)
    indices = systematic_resample(weights, rng) if trigger else np.arange(n)
    # Normalize also when not resampling to avoid log-weight drift. This is a
    # particle-independent constant, not an additional potential or selection.
    normalized = [None if w == 0 else float(np.log(w)) for w in weights]
    return {
        "status": "live",
        "weights": weights.tolist(),
        "ess": ess,
        "resampled": trigger,
        "indices": indices.tolist(),
        "log_weights": [-float(np.log(n))] * n if trigger else normalized,
        "log_potential": [next_log_potential[i] for i in indices],
    }


def log_potentials(arm, scores, future, *, terminal, beta):
    if arm not in ("reference", "immediate", "future") or not np.isfinite(beta) or beta <= 0:
        raise ValueError("invalid particle arm or positive potential scale")
    if len(scores) != len(future):
        raise ValueError("score and future arrays differ")
    result = []
    for score, value in zip(scores, future, strict=True):
        if score is None:
            result.append(None)
            continue
        selected = score if terminal or arm == "immediate" else value if arm == "future" else 0
        if selected is None or not np.isfinite(selected) or not 0 <= selected <= 1:
            raise ValueError("potential requires a finite score in [0,1]")
        result.append(float(beta * selected))
    return result
