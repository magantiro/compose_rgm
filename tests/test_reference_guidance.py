from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from compose_v4.control.reference_guidance import (
    GuidanceConfig,
    ProgramScore,
    ScoredPanel,
    guide_panel,
)


def panel(values=(-4.0, -1.0)):
    return ScoredPanel(
        "a" * 64,
        tuple(ProgramScore(str(index), value, "scored") for index, value in enumerate(values)),
        progress=0.0,
    )


def forbidden():
    raise AssertionError("off mode must not touch a reference")


def test_off_bypasses_reference_and_returns_exact_baseline():
    baseline = (0.123456789012345, 0.876543210987655)
    result = guide_panel(("0", "1"), baseline, score=forbidden)
    assert result.probabilities is baseline
    assert not result.probabilities_changed
    assert not result.receipt()["reference_evaluated"]


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_observational_modes_preserve_draws_and_rng_state(mode):
    left, right = np.random.default_rng(12), np.random.default_rng(12)
    global_state = copy.deepcopy(np.random.get_state())
    result = guide_panel(("0", "1"), (0.3, 0.7), config=GuidanceConfig(mode), score=panel)
    assert np.array_equal(
        left.choice(2, 50, p=(0.3, 0.7)), right.choice(2, 50, p=result.probabilities)
    )
    assert left.bit_generator.state == right.bit_generator.state
    now = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(global_state, now))
    assert result.probabilities == (0.3, 0.7)


def test_active_reference_changes_odds_with_declared_cap():
    result = guide_panel(
        ("0", "1"),
        (0.3, 0.7),
        config=GuidanceConfig("active", strength=2, log_weight_cap=0.4),
        score=panel,
    )
    assert result.probabilities_changed
    assert result.probabilities[1] > 0.7
    odds_change = (result.probabilities[1] / result.probabilities[0]) / (0.7 / 0.3)
    assert odds_change == pytest.approx(math.exp(0.4))
    assert sum(result.probabilities) == pytest.approx(1)
    # Inverse-CDF uniforms demonstrate a changed decision without a probabilistic test.
    uniform = (result.probabilities[0] + 0.3) / 2
    assert np.searchsorted(np.cumsum((0.3, 0.7)), uniform) != np.searchsorted(
        np.cumsum(result.probabilities), uniform
    )


def test_active_retains_zeros_and_does_not_duplicate_or_reorder_candidates():
    result = guide_panel(
        ("0", "1", "2"),
        (0, 0.9, 0.1),
        config=GuidanceConfig("active", strength=1),
        score=lambda: panel((0, -3, -1)),
    )
    assert result.candidate_ids == ("0", "1", "2")
    assert result.probabilities[0] == 0
    assert all(q > 0 for q in result.probabilities[1:])


def test_equal_scores_leave_no_claim_of_changed_probabilities():
    result = guide_panel(
        ("0", "1"),
        (0.25, 0.75),
        config=GuidanceConfig("active", strength=1),
        score=lambda: panel((-2, -2)),
    )
    assert not result.probabilities_changed
    assert result.outcome == "active"


def incomplete():
    return ScoredPanel(
        "b" * 64,
        (
            ProgramScore("0", -1, "scored"),
            ProgramScore(
                "1", None, "unsupported_native_mark", "step 2 is outside native mark support"
            ),
        ),
        0.0,
    )


def test_missing_coverage_preserves_whole_panel_and_reports_it():
    result = guide_panel(
        ("0", "1"),
        (0.5, 0.5),
        config=GuidanceConfig("active", strength=1),
        score=incomplete,
    )
    assert result.outcome == "baseline_fallback"
    assert result.probabilities == (0.5, 0.5)
    assert len(result.reference.scores) == 2
    assert result.reference.scores[1].value is None
    assert not result.probabilities_changed


def test_strict_coverage_fails_with_candidate_identity():
    with pytest.raises(ValueError, match="reference coverage incomplete: 1: step 2"):
        guide_panel(
            ("0", "1"),
            (0.5, 0.5),
            score=incomplete,
            config=GuidanceConfig("active", strength=1, on_unsupported="error"),
        )


def test_shadow_records_missing_coverage_without_changing_decisions():
    result = guide_panel(
        ("0", "1"),
        (0.5, 0.5),
        score=incomplete,
        config=GuidanceConfig("shadow", on_unsupported="error"),
    )
    assert result.outcome == "shadow"
    assert result.probabilities == (0.5, 0.5)


def mixed_panel(values):
    return ScoredPanel(
        "c" * 64,
        tuple(
            ProgramScore(str(i), value, "scored")
            if value is not None
            else ProgramScore(str(i), None, "missing_trace", "no executed trace")
            for i, value in enumerate(values)
        ),
        0.0,
    )


def guide_mixed(values, probabilities):
    return guide_panel(
        tuple(str(i) for i in range(len(values))),
        probabilities,
        config=GuidanceConfig("active", strength=1, on_unsupported="preserve_mass"),
        score=lambda: mixed_panel(values),
    )


def test_partial_guidance_preserves_each_unscored_probability_and_covered_mass():
    result = guide_mixed((-4, None, -1, None), (0.1, 0.2, 0.3, 0.4))
    assert result.outcome == "active_partial"
    assert result.probabilities_changed
    assert result.probabilities[1::2] == (0.2, 0.4)
    assert math.fsum(result.probabilities[::2]) == pytest.approx(0.4)
    assert result.probabilities[2] > 0.3
    assert sum(result.probabilities) == pytest.approx(1)
    coverage = result.receipt()["coverage"]
    assert coverage["scored_count"] == 2 and coverage["candidate_count"] == 4
    assert coverage["scored_guided_mass"] == pytest.approx(coverage["scored_baseline_mass"])
    assert result.reference.scores[1].value is None


@pytest.mark.parametrize("values", [(None, None), (-1, None), (None, -1)])
def test_partial_guidance_needs_two_scored_candidates(values):
    result = guide_mixed(values, (0.2, 0.8))
    assert result.outcome == "baseline_fallback"
    assert result.probabilities == result.baseline_probabilities


def test_zero_probability_scored_candidate_does_not_count_as_an_alternative():
    result = guide_mixed((-10, -1, None), (0, 0.4, 0.6))
    assert result.probabilities == (0, 0.4, 0.6)
    assert result.outcome == "baseline_fallback"


def test_partial_guidance_does_not_gain_or_lose_small_scored_mass():
    result = guide_mixed((-4, -1, None), (1e-200, 2e-200, 1.0))
    assert all(p > 0 for p in result.probabilities)
    assert math.fsum(result.probabilities[:2]) / 3e-200 == pytest.approx(1)
    assert result.probabilities[2] == 1.0


def test_partial_guidance_is_equivariant_to_panel_order():
    left = guide_mixed((-4, None, -1, None), (0.1, 0.2, 0.3, 0.4))
    right = guide_mixed((None, -1, None, -4), (0.4, 0.3, 0.2, 0.1))
    assert left.probabilities == right.probabilities[::-1]


def test_equal_scores_do_not_report_rounding_as_guidance():
    result = guide_mixed((-1, -1, None), (0.1, 0.3, 0.6))
    assert result.probabilities == (0.1, 0.3, 0.6)
    assert not result.probabilities_changed


def test_partial_policy_matches_active_policy_when_every_candidate_is_scored():
    partial = guide_mixed((-4, -1), (0.1, 0.9))
    complete = guide_panel(
        ("0", "1"),
        (0.1, 0.9),
        score=panel,
        config=GuidanceConfig("active", strength=1),
    )
    assert partial.probabilities == complete.probabilities
    assert partial.outcome == "active"


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_partial_policy_does_not_change_off_or_shadow(mode):
    result = guide_panel(
        ("0", "1"),
        (0.3, 0.7),
        score=incomplete,
        config=GuidanceConfig(mode, on_unsupported="preserve_mass"),
    )
    assert result.probabilities == (0.3, 0.7)


def test_unexpected_model_errors_are_not_hidden_as_coverage_fallback():
    def broken():
        raise RuntimeError("model implementation failed")

    with pytest.raises(RuntimeError, match="model implementation failed"):
        guide_panel(
            ("0", "1"), (0.5, 0.5), score=broken, config=GuidanceConfig("active", strength=1)
        )


@pytest.mark.parametrize(
    "ids,probabilities",
    [
        ((), ()),
        (("x", "x"), (0.5, 0.5)),
        (("",), (1,)),
        (("x",), ()),
        (("x",), (2,)),
        (("x", "y"), (-1, 2)),
        (("x",), (float("nan"),)),
        (("x",), (float("inf"),)),
    ],
)
def test_malformed_baseline_is_not_repaired(ids, probabilities):
    with pytest.raises(ValueError):
        guide_panel(ids, probabilities)


def test_mismatched_order_is_rejected():
    with pytest.raises(ValueError, match="exact candidate IDs and order"):
        guide_panel(("1", "0"), (0.5, 0.5), config=GuidanceConfig("shadow"), score=panel)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "unknown"},
        {"mode": "active"},
        {"strength": 1},
        {"mode": "shadow", "strength": 1},
        {"strength": -1},
        {"log_weight_cap": 0},
        {"log_weight_cap": 1000},
        {"log_weight_cap": float("nan")},
        {"on_unsupported": "drop"},
    ],
)
def test_bad_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        GuidanceConfig(**kwargs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0.1, None])
def test_invalid_scores_cannot_enter_selection(value):
    with pytest.raises(ValueError):
        ProgramScore("x", value, "scored")
