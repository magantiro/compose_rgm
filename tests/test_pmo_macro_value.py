"""The PMO reward target, and the ways it must DISAGREE with parent-relative improvement."""

from __future__ import annotations

import math

import numpy as np

from compose_v4.control.pmo_macro_value import (
    FEATURE_NAMES,
    MacroValue,
    archive_gain,
    archive_threshold,
    macro_features,
)

FULL_ARCHIVE = [(f"m{i}", 0.40 + 0.01 * i) for i in range(10)]  # top ten spans 0.40 .. 0.49


def test_a_candidate_below_the_frontier_is_worth_nothing():
    """The whole point of the target: not beating the archive means adding nothing."""
    assert archive_gain(FULL_ARCHIVE, 0.30) == 0.0
    assert archive_gain(FULL_ARCHIVE, 0.39) == 0.0
    assert archive_gain(FULL_ARCHIVE, 0.55) > 0.0


def test_archive_gain_disagrees_with_parent_relative_delta():
    """The case that motivated the new target, stated as an executable contrast.

    A improves a strong parent slightly and enters the frontier. B improves a weak parent
    enormously and lands below the tenth best. Parent-relative learning prefers B by a wide
    margin; archive gain prefers A, and the benchmark scores the archive.
    """
    a_parent, a_child = 0.40, 0.505
    b_parent, b_child = 0.25, 0.39
    assert (b_child - b_parent) > (a_child - a_parent), "B wins on parent-relative delta"
    assert archive_gain(FULL_ARCHIVE, a_child) > archive_gain(FULL_ARCHIVE, b_child), (
        "archive gain must prefer the candidate that actually moves the frontier"
    )
    assert archive_gain(FULL_ARCHIVE, b_child) == 0.0


def test_an_empty_archive_gives_a_candidate_its_full_value():
    assert archive_gain([], 0.30) == 0.30


def test_threshold_is_open_until_the_archive_is_full():
    assert archive_threshold(FULL_ARCHIVE[:3], k=10) == -math.inf
    assert math.isclose(archive_threshold(FULL_ARCHIVE, k=10), 0.40, rel_tol=1e-9)


def test_features_describe_the_macro_not_only_the_molecule():
    """Two candidates with the SAME endpoint score but different macros must differ."""
    base = {
        "provenance": {
            "parent_measured_score": 0.25,
            "planner_channel": "structured",
            "retention_realized": 0.6,
            "retention_target": 0.6,
        },
        "program_size": {"primitive_edits": 14, "block_count": 3, "delta_from_measured_parent": -5},
        "program_rule_histogram": {"cycle_close": 2, "atom_insert": 4, "atom_delete": 9},
    }
    small = {
        "provenance": dict(base["provenance"], planner_channel="shallow"),
        "program_size": {"primitive_edits": 3, "block_count": 1, "delta_from_measured_parent": 1},
        "program_rule_histogram": {"atom_insert": 1},
    }
    big = macro_features(base, observations=FULL_ARCHIVE)
    little = macro_features(small, observations=FULL_ARCHIVE)
    assert len(big) == len(FEATURE_NAMES)
    assert not np.allclose(big, little), "a 14-primitive ring macro must not look like a 3-edit growth"
    names = dict(zip(FEATURE_NAMES, big, strict=True))
    assert names["primitive_edits_scaled"] > dict(zip(FEATURE_NAMES, little, strict=True))[
        "primitive_edits_scaled"
    ]
    assert names["is_structured_channel"] == 1.0 and names["is_shallow_channel"] == 0.0


def test_uncertainty_is_wide_without_evidence_and_narrows_with_it():
    """Exploration must come from what the model does not know, not a fixed constant."""
    model = MacroValue()
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, len(FEATURE_NAMES)))
    y = x @ np.linspace(0.0, 1.0, len(FEATURE_NAMES))
    probe = x[:5]
    unfitted = model.spread(probe)
    assert np.all(unfitted > 0), "an unfitted model must still offer a positive bonus"

    model.fit(x[:8], y[:8])
    thin = float(np.mean(model.spread(probe)))
    model.fit(x, y)
    thick = float(np.mean(model.spread(probe)))
    assert thick < thin, "the bonus must shrink as evidence accumulates"


def test_a_constant_bonus_would_not_reorder_but_uncertainty_does():
    """Why the shipped optimism constant was inert: it applied to nearly every candidate.

    Adding the same number to every score cannot change a ranking. A spread that varies by
    candidate can, and must.
    """
    model = MacroValue()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, len(FEATURE_NAMES)))
    model.fit(x, x @ np.linspace(0.0, 1.0, len(FEATURE_NAMES)))
    probe = rng.normal(size=(12, len(FEATURE_NAMES))) * 3.0  # off-support, so spread varies
    spreads = model.spread(probe)
    assert spreads.std() > 0, "predictive spread must differ between candidates"
    plain = np.argsort(-model.predict(probe))
    explored = np.argsort(-model.acquire(probe, beta=5.0))
    assert not np.array_equal(plain, explored), "uncertainty must be able to reorder candidates"


# ---- Continuation value ----------------------------------------------------------------

from compose_v4.control.pmo_macro_value import (  # noqa: E402
    acquire_with_continuation,
    lineage_continuation,
)


def _observed(rows):
    return [{"endpoint": e, "score": s, "parent": p} for e, s, p in rows]


def test_a_flat_state_whose_descendant_pays_off_gets_credit():
    """The exact shape the beam showed: 0.44 sits still, then a descendant reaches 0.51."""
    labels = lineage_continuation(
        _observed(
            [
                ("a", 0.4444, None),
                ("b", 0.4400, "a"),   # looks like a waste under an immediate target
                ("c", 0.4380, "b"),
                ("d", 0.5116, "c"),   # the payoff, three decisions later
                ("z", 0.1000, None),  # an unrelated root that went nowhere
            ]
        ),
        horizon=5,
    )
    assert labels["b"] > 0.0, "a state whose lineage later paid must carry positive continuation"
    assert labels["c"] > labels["b"], "the state nearer the payoff carries more"
    assert labels["z"] == 0.0, "a barren lineage carries none"


def test_continuation_is_attributed_by_lineage_not_by_wall_clock():
    """An unrelated branch's success must not credit a state that had nothing to do with it."""
    labels = lineage_continuation(
        _observed([("root_a", 0.20, None), ("root_b", 0.20, None), ("b_child", 0.90, "root_b")]),
        horizon=5,
    )
    assert labels["root_b"] > 0.0
    assert labels["root_a"] == 0.0, "a sibling root must not inherit another lineage's payoff"


def test_the_horizon_actually_bounds_attribution():
    rows = _observed(
        [("a", 0.10, None)] + [(f"f{i}", 0.10, "a") for i in range(6)] + [("late", 0.90, "f5")]
    )
    near = lineage_continuation(rows, horizon=2)
    far = lineage_continuation(rows, horizon=20)
    assert near["a"] == 0.0, "a payoff beyond the horizon must not be credited"
    assert far["a"] > 0.0


def test_semi_markov_acquisition_discounts_by_macro_duration():
    """A long macro is a larger commitment and its continuation is discounted further."""
    immediate, continuation = MacroValue(), MacroValue()
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, len(FEATURE_NAMES)))
    immediate.fit(x, np.zeros(len(x)))
    continuation.fit(x, np.ones(len(x)))

    probe = x[:4]
    short = acquire_with_continuation(
        immediate, continuation, probe, beta=0.0, discount=0.9, macro_lengths=[3, 3, 3, 3]
    )
    long = acquire_with_continuation(
        immediate, continuation, probe, beta=0.0, discount=0.9, macro_lengths=[14, 14, 14, 14]
    )
    assert np.all(long < short), "a longer macro must discount its continuation more"


def test_acquisition_explores_when_either_head_is_unsure():
    immediate, continuation = MacroValue(), MacroValue()
    rng = np.random.default_rng(4)
    x = rng.normal(size=(60, len(FEATURE_NAMES)))
    immediate.fit(x, x @ np.linspace(0, 1, len(FEATURE_NAMES)))
    continuation.fit(x, np.zeros(len(x)))
    probe = rng.normal(size=(10, len(FEATURE_NAMES))) * 4.0
    greedy = acquire_with_continuation(immediate, continuation, probe, beta=0.0)
    curious = acquire_with_continuation(immediate, continuation, probe, beta=6.0)
    assert not np.array_equal(np.argsort(-greedy), np.argsort(-curious))


# ---- Expected archive gain and AUC weighting ---------------------------------------------

from compose_v4.control.pmo_macro_value import (  # noqa: E402
    auc_weighted_gain,
    expected_archive_gain,
)


def test_a_near_miss_is_worth_nothing_only_when_we_are_certain():
    """Stage A's blind spot, stated as a test.

    Learning `archive_gain` directly makes every sub-threshold candidate worth exactly zero.
    Predicting the SCORE and integrating recovers the option value: a 0.48 against a 0.50
    threshold is worthless only if we are sure it is 0.48.
    """
    certain = float(expected_archive_gain(0.48, 0.0, threshold=0.50))
    unsure = float(expected_archive_gain(0.48, 0.05, threshold=0.50))
    very_unsure = float(expected_archive_gain(0.48, 0.20, threshold=0.50))
    assert certain == 0.0
    assert 0.0 < unsure < very_unsure


def test_a_hopeless_candidate_is_negligible_but_not_exactly_zero():
    """A Gaussian posterior has infinite support, so nothing is ever strictly impossible.

    The honest property is that a hopeless candidate is worth ORDERS less than a live one,
    not that it is worth exactly nothing. Asserting equality with zero would be asserting a
    falsehood that happens to survive rounding in a printout.
    """
    hopeless = float(expected_archive_gain(0.30, 0.05, threshold=0.50))
    live = float(expected_archive_gain(0.48, 0.05, threshold=0.50))
    assert 0.0 < hopeless < 1e-6
    assert hopeless < live / 1000, "a hopeless candidate must be negligible beside a near miss"


def test_the_degenerate_posterior_reduces_to_the_plug_in():
    """With no uncertainty the closed form must equal `max(0, u - tau) / k` exactly."""
    for score in (0.30, 0.50, 0.55, 0.90):
        expected = max(0.0, score - 0.50) / 10
        assert math.isclose(
            float(expected_archive_gain(score, 0.0, threshold=0.50)), expected, abs_tol=1e-12
        )


def test_expected_gain_is_monotone_in_the_predicted_score():
    grid = [float(expected_archive_gain(m, 0.05, threshold=0.50)) for m in np.linspace(0.3, 0.7, 9)]
    assert all(b >= a for a, b in zip(grid, grid[1:], strict=False))


def test_an_early_improvement_is_worth_more_of_the_query_curve():
    """PMO scores the top-ten trajectory, so a gain persists through the calls that remain."""
    gain = 0.005
    early = float(auc_weighted_gain(gain, calls_remaining=200, budget=200))
    middle = float(auc_weighted_gain(gain, calls_remaining=100, budget=200))
    late = float(auc_weighted_gain(gain, calls_remaining=10, budget=200))
    assert early > middle > late
    assert math.isclose(early, gain, rel_tol=1e-12), "a full budget keeps the full gain"
    assert math.isclose(late, gain * 0.05, rel_tol=1e-12)


def test_auc_weighting_refuses_an_impossible_budget():
    for bad in ({"calls_remaining": -1, "budget": 10}, {"calls_remaining": 5, "budget": 0}):
        try:
            auc_weighted_gain(0.01, **bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted an invalid budget: {bad}")
