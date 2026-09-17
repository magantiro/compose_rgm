"""Guards for the structural-effect belief and the finite-budget acquisition."""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.control.bundle_acquisition import (
    TRANSITION_PRIOR,
    bundle_value,
    select_bundle,
)
from compose_v4.control.structural_effect_belief import EffectBelief, posterior_incumbent


def _coordinates():
    return {"G": [("scale", 6), ("scale", 5)], "L": [("element", 7), ("bond_order", 2)]}


def test_a_matched_contrast_moves_both_arms_in_opposite_directions():
    belief = EffectBelief()
    belief.observe_contrast("scale", 6, 5, gain=1.0)
    better, _ = belief.posterior("scale", 6)
    worse, _ = belief.posterior("scale", 5)
    assert better > 0 > worse, "a contrast says which side won, symmetrically"


def test_repeated_evidence_narrows_the_posterior():
    belief = EffectBelief()
    _, wide = belief.posterior("scale", 6)
    for _ in range(8):
        belief.observe_contrast("scale", 6, 5, gain=0.5)
    _, narrow = belief.posterior("scale", 6)
    assert narrow < wide / 2


def test_an_unmatched_observation_is_weaker_than_a_matched_one():
    """Absolute observations confound the effect with the parent, so they count for less."""
    matched, absolute = EffectBelief(), EffectBelief()
    matched.observe_contrast("scale", 6, 5, gain=1.0)
    absolute.observe_absolute("scale", 6, gain=1.0)
    _, matched_sd = matched.posterior("scale", 6)
    _, absolute_sd = absolute.posterior("scale", 6)
    assert matched_sd < absolute_sd


def test_an_unobserved_coordinate_keeps_its_prior_width():
    belief = EffectBelief()
    for _ in range(10):
        belief.observe_contrast("element", 7, 6, gain=0.4)
    mean, sd = belief.posterior("scale", 6)
    assert mean == pytest.approx(0.0)
    assert sd == pytest.approx(belief.value_scale)


def test_the_incumbent_shrinks_an_extreme_that_sits_inside_the_noise():
    """Taking a max over noisy measurements selects for upward error.

    With an extreme-value objective this is the failure the controller would be rewarded
    for making, so a clear winner is trusted and a within-noise winner is pulled in.
    """
    clear = posterior_incumbent([-12.0, -9.5, -9.4], noise=0.35)
    tied = posterior_incumbent([-9.6, -9.5, -9.4], noise=0.35)
    assert clear < -11.8, "a winner far outside the noise is trusted"
    assert tied > -9.6, "a winner inside the noise is shrunk toward the field"


def test_deeper_rollout_never_reports_a_worse_continuation_value():
    """The incumbent cannot degrade: a bad option is simply not taken."""
    belief = EffectBelief()
    bundle = [
        {"coordinate": "scale", "value": 6, "lane": "G"},
        {"coordinate": "scale", "value": 5, "lane": "G"},
    ]
    shallow = bundle_value(
        belief,
        bundle,
        parent_score=-9.0,
        incumbent=-9.0,
        coordinates=_coordinates(),
        rng=np.random.default_rng(1),
        samples=200,
        depth=1,
    )
    deep = bundle_value(
        belief,
        bundle,
        parent_score=-9.0,
        incumbent=-9.0,
        coordinates=_coordinates(),
        rng=np.random.default_rng(1),
        samples=200,
        depth=3,
    )
    assert deep["value"] <= shallow["value"] + 1e-9
    assert deep["continuation_gain"] >= shallow["continuation_gain"]


def test_selection_returns_the_lowest_value_bundle_and_its_margin():
    belief = EffectBelief()
    for _ in range(6):
        belief.observe_contrast("scale", 6, 5, gain=1.5)
    candidates = [
        [
            {"coordinate": "scale", "value": 6, "lane": "G"},
            {"coordinate": "scale", "value": 5, "lane": "G"},
        ],
        [
            {"coordinate": "bond_order", "value": 2, "lane": "L"},
            {"coordinate": "bond_order", "value": 1, "lane": "L"},
        ],
    ]
    chosen = select_bundle(
        belief,
        candidates,
        parent_score=-9.0,
        incumbent=-9.0,
        coordinates=_coordinates(),
        rng=np.random.default_rng(3),
        samples=200,
    )
    assert chosen["bundle"][0]["coordinate"] == "scale", "evidence should steer the choice"
    assert chosen["considered"] == 2
    assert chosen["runner_up_gap"] >= 0


def test_the_transition_prior_is_a_normalised_length_two_table():
    """Length-3 motifs rest on 19 observations and are deliberately absent."""
    for lane, row in TRANSITION_PRIOR.items():
        assert set(row) == {"G", "L"}, lane
        assert sum(row.values()) == pytest.approx(1.0)
