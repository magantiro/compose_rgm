"""Time-feature convention for the factorized rate model.

The rate model consumes a *frozen time* feature in ``[0, 1)``, NOT the raw
operational time of the CTMC. The canonical map -- used by the trained/committed
ancestral sampler (``tracelet_conditional.sample_tracelet_ancestral``) -- is

    frozen_time = 1 - exp(-(operational_time + interval_end) / 2)        # "k = 2"

Passing raw operational time is a silent, severe bug: real event-times routinely
exceed 1 (per-family means ~1.7-5.4 in the pancake event audit), so the model is
evaluated far outside its trained ``[0, 1]`` range and its productive hazard
collapses toward zero. Route EVERY ``sample_rewrite_mark`` / ``rate_table`` call
through :func:`frozen_time` -- there is no reason to pass raw operational time to
the model.
"""

from __future__ import annotations

from math import exp

# Rate-model time constant: frozen_time = 1 - exp(-t / TIME_CONSTANT).
TIME_CONSTANT = 2.0


def frozen_time(operational_time: float, interval_end: float = 0.0) -> float:
    """Map operational time (``>= 0``) to the model's frozen-time feature in ``[0, 1)``.

    ``interval_end`` is the within-step offset used by the canonical ancestral
    sampler (the midpoint of a holding interval); accumulate-and-step callers pass
    ``0.0`` and rely on the accumulated ``operational_time`` alone.
    """
    return 1.0 - exp(-(operational_time + interval_end) / TIME_CONSTANT)
