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


def _difference_sd(belief, coordinate, left, right):
    _, covariance = belief._moments()
    i, j = belief._index[(coordinate, left)], belief._index[(coordinate, right)]
    return float(np.sqrt(covariance[i, i] + covariance[j, j] - 2 * covariance[i, j]))


def test_repeated_contrasts_narrow_the_difference_they_measure():
    """Evidence must accumulate on the quantity the pair actually reports.

    The marginal of a single effect stays wide under any number of contrasts, because a
    pair never pins the common level. Asserting on the marginal is asserting the model is
    wrong.
    """
    belief = EffectBelief()
    belief.observe_contrast("scale", 6, 5, gain=0.5)
    wide = _difference_sd(belief, "scale", 6, 5)
    for _ in range(8):
        belief.observe_contrast("scale", 6, 5, gain=0.5)
    assert _difference_sd(belief, "scale", 6, 5) < wide / 2


def test_a_contrast_constrains_the_difference_not_the_common_level():
    """What a matched pair informs is `theta_i - theta_j`, and the test must ask that.

    An earlier version of this test compared MARGINAL widths and failed, which was the
    test being wrong rather than the model: a pair says nothing about the common level of
    the two effects, so each marginal stays wide while their difference becomes sharp.
    The controller ranks values WITHIN a coordinate, so the difference is the quantity it
    actually uses.
    """
    matched, absolute = EffectBelief(), EffectBelief()
    matched.observe_contrast("scale", 6, 5, gain=1.0)
    absolute.observe_absolute("scale", 6, gain=1.0)
    absolute.observe_absolute("scale", 5, gain=0.0)

    def difference_sd(belief):
        _, covariance = belief._moments()
        i, j = belief._index[("scale", 6)], belief._index[("scale", 5)]
        return np.sqrt(covariance[i, i] + covariance[j, j] - 2 * covariance[i, j])

    assert difference_sd(matched) < difference_sd(absolute)


def test_a_contrast_leaves_the_two_effects_positively_correlated():
    """Pinning a difference leaves the common level free, so the pair moves together."""
    belief = EffectBelief()
    belief.observe_contrast("scale", 6, 5, gain=1.0)
    _, covariance = belief._moments()
    i, j = belief._index[("scale", 6)], belief._index[("scale", 5)]
    correlation = covariance[i, j] / np.sqrt(covariance[i, i] * covariance[j, j])
    assert correlation > 0.5


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


def test_lookahead_depth_changes_the_policy_when_continuation_value_differs():
    """The depth parameter must be able to change a DECISION, not only a score.

    E_A pays best immediately and opens nothing. E_B pays less and opens a strong
    follow-on. A one-step controller should take E_A and a three-step controller E_B.
    The converse case pins the other side: when E_B opens nothing useful, both depths
    must take E_A, so depth cannot be flipping choices for its own sake.
    """
    import compose_v4.control.bundle_acquisition as acquisition

    original = acquisition.TRANSITION_PRIOR
    acquisition.TRANSITION_PRIOR = {"A": {"A": 1.0}, "B": {"B": 1.0}}
    try:
        for strong_continuation, expected in ((True, "E_B"), (False, "E_A")):
            belief = EffectBelief()
            for _ in range(8):
                belief.observe_contrast("capA", 1, 0, gain=1.2)
                belief.observe_contrast("capB", 1, 0, gain=0.4)
                if strong_continuation:
                    belief.observe_contrast("followup", 1, 0, gain=1.6)
            coordinates = {
                "A": [("capA", 1)],
                "B": [("followup", 1)] if strong_continuation else [("capB", 0)],
            }
            a = [
                {"coordinate": "capA", "value": 1, "lane": "A"},
                {"coordinate": "capA", "value": 0, "lane": "A"},
            ]
            b = [
                {"coordinate": "capB", "value": 1, "lane": "B"},
                {"coordinate": "capB", "value": 0, "lane": "B"},
            ]
            picks = {}
            for depth in (1, 3):
                qa = bundle_value(
                    belief,
                    a,
                    parent_score=-9.0,
                    incumbent=-9.0,
                    coordinates=coordinates,
                    samples=400,
                    depth=depth,
                    seed=11,
                )
                qb = bundle_value(
                    belief,
                    b,
                    parent_score=-9.0,
                    incumbent=-9.0,
                    coordinates=coordinates,
                    samples=400,
                    depth=depth,
                    seed=11,
                )
                picks[depth] = "E_A" if qa["value"] < qb["value"] else "E_B"
            assert picks[1] == "E_A", strong_continuation
            assert picks[3] == expected, strong_continuation
    finally:
        acquisition.TRANSITION_PRIOR = original


def test_common_random_numbers_make_the_comparison_seed_stable():
    """Competing bundles must not win by drawing luckier simulations than their rivals."""
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
    picks = {
        select_bundle(
            belief,
            candidates,
            parent_score=-9.0,
            incumbent=-9.0,
            coordinates=_coordinates(),
            samples=300,
            seed=s,
        )["bundle"][0]["coordinate"]
        for s in range(6)
    }
    assert len(picks) == 1, f"selection flipped across seeds: {picks}"


def test_the_rollout_pays_an_oracle_call_for_what_it_learns():
    """A rollout step must not discover which option is best without measuring it.

    Taking argmax over sampled latent gains would let the simulation peek. The honest
    version acts on the posterior mean and then pays for the outcome, so a step adds one
    observation to the belief it was handed.
    """
    from compose_v4.control.bundle_acquisition import _rollout

    belief = EffectBelief()
    before = belief.observations()
    _rollout(belief, -9.0, "G", 3, _coordinates(), np.random.default_rng(0))
    assert belief.observations() == before + 3


def test_the_transition_prior_is_a_normalised_length_two_table():
    """Length-3 motifs rest on 19 observations and are deliberately absent."""
    for lane, row in TRANSITION_PRIOR.items():
        assert set(row) == {"G", "L"}, lane
        assert sum(row.values()) == pytest.approx(1.0)
