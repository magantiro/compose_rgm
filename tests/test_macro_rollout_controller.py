"""Guards for continuation-value macro selection.

The central fixture is the MEASURED thiothixene landscape, where immediate score and
continuation value disagree in sign. A selector that argmaxes immediate reward fails it.
"""
from __future__ import annotations

import pytest

from compose_v4.control.macro_rollout_controller import (
    DEFAULT_RUNGS,
    ArmOutcome,
    MacroCandidate,
    StagnationDetector,
    macro_candidates,
    rollout_cost,
    select_by_continuation,
)

# MEASURED, 5 matched 64-call arms: (immediate score, value reached after continuation).
# The two junction arms have the HIGHEST immediate score and go nowhere; the winner has
# the lowest immediate score of the three macro siblings.
THIOTHIXENE = {
    "incumbent":     (0.4839, 0.4839),
    "bare_graft":    (0.5076, 0.5810),
    "junction_cl":   (0.5118, 0.5118),
    "junction_f":    (0.5118, 0.5118),
    "junction_n":    (0.5000, 0.5966),
}


def _landscape(table):
    """A continuation that reveals its arm's true value only when it is actually run."""
    calls = {"n": 0}

    def advance(endpoint, steps):
        calls["n"] += steps
        immediate, reachable = table[endpoint]
        # More charged steps get closer to what the basin can actually reach; one step
        # reveals essentially nothing beyond the immediate score.
        fraction = min(1.0, steps / 8.0)
        return ArmOutcome(value=immediate + fraction * (reachable - immediate),
                          calls=steps)

    return advance, calls


def _arms():
    return [MacroCandidate("no_macro", "incumbent", "no_macro"),
            MacroCandidate("bare", "bare_graft", "region_replace"),
            MacroCandidate("jcl", "junction_cl", "region_replace"),
            MacroCandidate("jf", "junction_f", "region_replace"),
            MacroCandidate("jn", "junction_n", "region_replace")]


def test_selection_prefers_continuation_value_over_immediate_score():
    advance, _ = _landscape(THIOTHIXENE)
    result = select_by_continuation(_arms(), advance)
    assert result.winner.endpoint == "junction_n"
    # The arm it beat is the one an immediate-reward argmax would have taken.
    best_immediate = max(THIOTHIXENE, key=lambda k: THIOTHIXENE[k][0])
    assert best_immediate in ("junction_cl", "junction_f")
    assert result.winner.endpoint != best_immediate


def test_an_immediate_reward_selector_fails_this_landscape():
    """The negative control: without rollouts the wrong arm wins, so the test can fail."""
    greedy = max(THIOTHIXENE, key=lambda k: THIOTHIXENE[k][0])
    truth = max(THIOTHIXENE, key=lambda k: THIOTHIXENE[k][1])
    assert greedy != truth, "fixture no longer discriminates the two rules"


def test_no_macro_is_mandatory():
    advance, _ = _landscape(THIOTHIXENE)
    without = [c for c in _arms() if not c.is_no_macro]
    with pytest.raises(ValueError, match="no-macro arm is mandatory"):
        select_by_continuation(without, advance)


def test_no_macro_wins_when_local_search_is_still_productive():
    """A macro must EARN the transition; stagnation firing is not sufficient."""
    productive = {"incumbent": (0.50, 0.95), "bare_graft": (0.60, 0.61),
                  "junction_cl": (0.58, 0.59), "junction_f": (0.57, 0.58),
                  "junction_n": (0.55, 0.56)}
    advance, _ = _landscape(productive)
    result = select_by_continuation(_arms(), advance)
    assert result.winner.is_no_macro


def test_charged_cost_is_exact_and_declared_before_spending():
    advance, calls = _landscape(THIOTHIXENE)
    predicted = rollout_cost(5, DEFAULT_RUNGS)
    result = select_by_continuation(_arms(), advance)
    assert predicted == 5 * 4 + 4 * 8 == 52
    assert result.calls_spent == predicted == calls["n"]


def test_rollout_refuses_rather_than_truncating_when_the_budget_is_short():
    advance, calls = _landscape(THIOTHIXENE)
    with pytest.raises(ValueError, match="rollout needs"):
        select_by_continuation(_arms(), advance, budget=rollout_cost(5) - 1)
    assert calls["n"] == 0, "refused rollout must not charge anything"


def test_an_arm_that_undercharges_is_refused():
    """Unequal evidence makes the comparison meaningless, so it must raise, not rank."""
    def advance(endpoint, steps):
        return ArmOutcome(value=1.0, calls=steps - 1 if endpoint == "bare_graft" else steps)

    with pytest.raises(ValueError, match="unequal evidence"):
        select_by_continuation(_arms(), advance)


def test_a_surviving_arm_keeps_its_best_rung():
    """An arm must not be punished for a weak later rung after surviving an early one."""
    seen = {"n": 0}

    def advance(endpoint, steps):
        seen["n"] += 1
        if endpoint == "junction_n":
            # Strong first rung, then a deliberately terrible second rung.
            return ArmOutcome(value=0.9 if seen["n"] <= 5 else 0.0, calls=steps)
        return ArmOutcome(value=0.5, calls=steps)

    result = select_by_continuation(_arms(), advance)
    assert result.winner.endpoint == "junction_n"
    assert result.value == pytest.approx(0.9)


def test_stagnation_fires_on_a_frozen_frontier_not_on_a_noisy_mean():
    detector = StagnationDetector(patience=3)
    assert not detector.update(0.40)
    assert not detector.update(0.45)
    assert not detector.update(0.45)
    assert not detector.update(0.45)
    assert detector.update(0.45)
    # A real improvement clears it.
    assert not detector.update(0.50)


def test_candidates_always_lead_with_the_incumbent_and_drop_duplicates():
    proposals = [{"endpoint": "A", "family": "region_replace"},
                 {"endpoint": "A", "family": "region_replace"},
                 {"endpoint": "incumbent", "family": "region_excise"},
                 {"endpoint": "B", "family": "region_excise"}]
    arms = macro_candidates("incumbent", proposals, limit=7)
    assert arms[0].is_no_macro and arms[0].endpoint == "incumbent"
    assert [c.endpoint for c in arms[1:]] == ["A", "B"]


def test_candidates_respect_a_diversity_rule():
    """Near-identical arms waste rungs -- measured: the Cl and F siblings were both dead."""
    def diversity(kept, endpoint):
        return all(endpoint[0] != k[0] for k in kept)

    proposals = [{"endpoint": "Xa"}, {"endpoint": "Xb"}, {"endpoint": "Yc"}]
    arms = macro_candidates("inc", proposals, diversity=diversity)
    assert [c.endpoint for c in arms[1:]] == ["Xa", "Yc"]


def test_cost_helper_matches_the_schedule_it_is_given():
    assert rollout_cost(8, ((4, 4), (8, 1))) == 8 * 4 + 4 * 8
    assert rollout_cost(3, ((2, 1),)) == 6
    assert rollout_cost(1, ((5, 1), (5, 1))) == 10
