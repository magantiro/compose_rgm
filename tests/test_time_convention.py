"""Regression guard for the rate-model time-feature convention.

The rate model consumes ``frozen_time = 1 - exp(-t/2) in [0, 1)``, never raw
operational time. A raw-``t`` regression (passing operational time, which reaches
the horizon ~16, straight to ``sample_rewrite_mark``) silently collapses the
productive hazard -- it cost a retracted "uniform beats learned" result. These
tests pin the mapping so any such regression fails loudly here instead.
"""

from __future__ import annotations

from math import exp

from compose_v4.model.time_convention import TIME_CONSTANT, frozen_time


def test_zero_maps_to_zero() -> None:
    assert frozen_time(0.0) == 0.0


def test_stays_in_unit_interval_across_the_whole_horizon() -> None:
    # Operational time runs to the horizon (~16); the model feature must never exceed 1
    # (the raw-t bug fed values up to ~16). It saturates to exactly 1.0 only far past
    # the horizon (float underflow), so the bound is inclusive.
    for t in [0.0, 0.1, 1.0, 1.4, 2.0, 4.0, 6.4, 12.0, 16.0, 100.0]:
        value = frozen_time(t)
        assert 0.0 <= value <= 1.0, (t, value)
    # Within the real operational range the feature stays strictly inside the interval.
    assert frozen_time(16.0) < 1.0


def test_strictly_increasing() -> None:
    grid = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    values = [frozen_time(t) for t in grid]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_matches_canonical_formula_and_saturates() -> None:
    # Same "k=2" map the committed ancestral path uses (tracelet_conditional).
    for t in [0.3, 1.7, 5.4]:
        assert abs(frozen_time(t) - (1.0 - exp(-t / TIME_CONSTANT))) < 1e-12
    assert frozen_time(50.0) > 0.999  # deep into the trajectory -> ~1


def test_interval_end_offset() -> None:
    assert frozen_time(1.0, interval_end=1.0) == frozen_time(2.0)
