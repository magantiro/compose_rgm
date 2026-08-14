"""Region-conditioned h_φ: contract, budget axis, and the exact boundary."""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.experiments.hphi_region_features import (
    BUDGET_MAX,
    EMBED_DIM,
    INPUT_DIM,
    acceptance_probabilities,
    build_features,
    h_with_boundary,
    in_region,
    should_stop,
)
from compose_v4.experiments.hphi_rollout import BENCHMARK_REGION

E = np.arange(EMBED_DIM, dtype=np.float64) / EMBED_DIM
S = np.ones(EMBED_DIM, dtype=np.float64) * 0.5


def test_budget_axis_covers_the_amended_horizon():
    """The old contract's budget_max=8 would silently truncate b in [0,24]."""
    assert BUDGET_MAX == 24
    assert INPUT_DIM == 4 * EMBED_DIM + 2 + 2 + 2 + 25 == 1055


@pytest.mark.parametrize("b", [0, 1, 6, 12, 24])
def test_features_have_the_declared_width_at_every_budget(b):
    v = build_features(E, S, 0.77, 0.62, BENCHMARK_REGION, b)
    assert v.shape == (INPUT_DIM,)
    assert np.all(np.isfinite(v))


def test_budget_is_one_hot_and_positioned_correctly():
    for b in (0, 7, 24):
        v = build_features(E, S, 0.77, 0.62, BENCHMARK_REGION, b)
        oh = v[-(BUDGET_MAX + 1):]
        assert oh.sum() == 1.0 and oh[b] == 1.0


def test_budget_outside_the_horizon_raises():
    with pytest.raises(ValueError):
        build_features(E, S, 0.77, 0.62, BENCHMARK_REGION, 25)
    with pytest.raises(ValueError):
        build_features(E, S, 0.77, 0.62, BENCHMARK_REGION, -1)


# --------------------------------------------------------------------------
# The margin features -- the reason the head does not rediscover chemistry
# --------------------------------------------------------------------------

def test_margins_are_present_and_signed():
    """qed - tau_q and sim - tau_s must appear explicitly."""
    v = build_features(E, S, 0.93, 0.62, BENCHMARK_REGION, 6)   # 0.90 / 0.40
    tail = v[4 * EMBED_DIM: 4 * EMBED_DIM + 6]
    qed, sim, tq, ts, mq, ms = tail
    assert (qed, sim, tq, ts) == (0.93, 0.62, 0.90, 0.40)
    assert mq == pytest.approx(0.03)      # inside on QED
    assert ms == pytest.approx(0.22)      # inside on similarity


def test_margins_go_NEGATIVE_outside_the_region():
    v = build_features(E, S, 0.81, 0.35, BENCHMARK_REGION, 6)
    mq, ms = v[4 * EMBED_DIM + 4], v[4 * EMBED_DIM + 5]
    assert mq < 0 and ms < 0


def test_the_same_state_gets_DIFFERENT_features_per_region():
    """Goal conditioning must actually condition."""
    a = build_features(E, S, 0.86, 0.55, (0.85, 0.50), 6)
    b = build_features(E, S, 0.86, 0.55, (0.90, 0.40), 6)
    assert not np.allclose(a, b), "region must change the representation"


# --------------------------------------------------------------------------
# The boundary condition -- ENFORCED, never learned
# --------------------------------------------------------------------------

def test_in_region_forces_h_to_ONE_regardless_of_the_network():
    """Even a confidently wrong network cannot override a fact."""
    for raw in (0.0, 0.13, 0.5, 1.0):
        assert h_with_boundary(raw, 0.95, 0.80, BENCHMARK_REGION) == 1.0


def test_boundary_holds_at_every_budget_including_zero():
    assert h_with_boundary(0.0, 0.91, 0.41, BENCHMARK_REGION) == 1.0


def test_inclusive_on_both_constraints():
    """Exactly on the threshold counts as inside, matching the census."""
    assert in_region(0.90, 0.40, BENCHMARK_REGION)
    assert not in_region(0.8999, 0.40, BENCHMARK_REGION)
    assert not in_region(0.90, 0.3999, BENCHMARK_REGION)


def test_outside_the_region_the_network_prediction_is_used_and_clamped():
    assert h_with_boundary(0.37, 0.80, 0.50, BENCHMARK_REGION) == pytest.approx(0.37)
    assert h_with_boundary(-0.2, 0.80, 0.50, BENCHMARK_REGION) == 0.0
    assert h_with_boundary(1.4, 0.80, 0.50, BENCHMARK_REGION) == 1.0


def test_should_stop_matches_the_boundary_exactly():
    assert should_stop(0.92, 0.55, BENCHMARK_REGION)
    assert not should_stop(0.92, 0.30, BENCHMARK_REGION)


# --------------------------------------------------------------------------
# Acceptance for the Stage-A1 sampler
# --------------------------------------------------------------------------

def test_a_successor_already_inside_the_region_is_accepted_with_probability_one():
    """This is what makes the controller reach the region and stop."""
    p = acceptance_probabilities([0.02, 0.02], [0.95, 0.70], [0.60, 0.60],
                                 BENCHMARK_REGION)
    assert p[0] == 1.0 and p[1] == pytest.approx(0.02)


def test_acceptance_is_a_valid_probability_vector():
    rng = np.random.default_rng(0)
    p = acceptance_probabilities(rng.normal(0.5, 0.6, 50), rng.uniform(0.5, 0.95, 50),
                                 rng.uniform(0.2, 0.9, 50), BENCHMARK_REGION)
    assert np.all((p >= 0.0) & (p <= 1.0))
