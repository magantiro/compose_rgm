"""Guards for backed-up lineage allocation.

The point of the allocator is that value propagates upward: a node whose descendant
scored well must become attractive. The failures that matter are value not
propagating, an outlier capturing the budget, and the exploration term ceasing to
recover the current score-blind behaviour in its limit.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.control.lineage_allocation import (
    backed_up_values,
    lineage_of,
    rank_scores,
    replay,
    score_blind_parent,
    select_parent,
)


def _chain():
    """root -> a -> b, where only b is good."""
    return {
        "root": {"parent": None, "score": -8.0},
        "a": {"parent": "root", "score": -8.5},
        "b": {"parent": "a", "score": -14.0},
        "side": {"parent": "root", "score": -9.0},
    }


# ---- Backed-up value ----


def test_value_propagates_all_the_way_up_the_lineage():
    values = backed_up_values(_chain())
    assert values["b"] == -14.0
    assert values["a"] == -14.0, "a enabled b and must inherit its value"
    assert values["root"] == -14.0
    assert values["side"] == -9.0, "an unrelated branch must not inherit"


def test_a_node_with_no_better_descendant_keeps_its_own_score():
    nodes = {"root": {"parent": None, "score": -12.0}, "c": {"parent": "root", "score": -7.0}}
    assert backed_up_values(nodes)["root"] == -12.0


def test_a_parent_cycle_does_not_hang():
    nodes = {"x": {"parent": "y", "score": -1.0}, "y": {"parent": "x", "score": -2.0}}
    assert backed_up_values(nodes)["x"] <= -1.0


def test_a_missing_parent_is_tolerated():
    nodes = {"orphan": {"parent": "gone", "score": -5.0}}
    assert backed_up_values(nodes) == {"orphan": -5.0}


# ---- Ranking ----


def test_ranking_is_by_order_not_magnitude():
    plain = rank_scores({"a": -10.0, "b": -9.0, "c": -8.0})
    outlier = rank_scores({"a": -1000.0, "b": -9.0, "c": -8.0})
    assert plain == outlier, "an outlier must move a rank step, not the whole scale"
    assert plain["a"] == 1.0 and plain["c"] == 0.0


def test_ties_share_a_rank():
    ranks = rank_scores({"a": -9.0, "b": -9.0, "c": -8.0})
    assert ranks["a"] == ranks["b"]


def test_ranking_handles_degenerate_inputs():
    assert rank_scores({}) == {}
    assert rank_scores({"only": -3.0}) == {"only": 1.0}


# ---- Selection ----


def test_with_no_exploration_the_best_backed_up_lineage_is_chosen():
    chosen = select_parent(_chain(), {}, exploration=0.0)
    assert chosen in {"root", "a", "b"}, "the winning lineage, not the side branch"


def test_exploration_moves_budget_off_a_heavily_used_node():
    nodes, visits = _chain(), {"root": 0, "a": 0, "b": 50, "side": 0}
    assert select_parent(nodes, visits, exploration=5.0) != "b"


def test_a_large_exploration_constant_recovers_least_used_selection():
    """The score-blind allocator is the limit of this one, not a different thing."""
    nodes = _chain()
    visits = {"root": 9, "a": 9, "b": 9, "side": 0}
    assert select_parent(nodes, visits, exploration=1e6) == "side"


def test_selection_is_deterministic_given_the_same_archive():
    nodes = _chain()
    first = select_parent(nodes, {}, exploration=1.0)
    second = select_parent(dict(reversed(list(nodes.items()))), {}, exploration=1.0)
    assert first == second


def test_an_empty_archive_is_refused_by_both_allocators():
    with pytest.raises(ValueError, match="empty archive"):
        select_parent({}, {}, exploration=1.0)
    with pytest.raises(ValueError, match="empty archive"):
        score_blind_parent({}, {}, np.random.default_rng(0))


def test_a_negative_or_nonfinite_exploration_constant_is_refused():
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and non-negative"):
            select_parent(_chain(), {}, exploration=bad)


def test_a_supplied_prior_steers_selection():
    nodes = _chain()
    biased = select_parent(nodes, {}, exploration=50.0, prior={"side": 1.0})
    assert biased == "side"


# ---- Lineage and replay ----


def test_lineage_walks_to_the_root_and_stops():
    assert lineage_of("b", _chain()) == ["b", "a", "root"]


def test_replay_reports_concentration_without_inventing_a_counterfactual_score():
    records = [
        {"call": 0, "id": "root", "parent": None, "score": -8.0},
        {"call": 1, "id": "a", "parent": "root", "score": -8.5},
        {"call": 2, "id": "side", "parent": "root", "score": -9.0},
        {"call": 3, "id": "b", "parent": "a", "score": -14.0},
    ]
    result = replay(records, exploration=1.0, rng=np.random.default_rng(0))
    assert result["calls"] == 4 and result["decisions"] == 3
    assert result["champion_score"] == -14.0
    assert result["new_oracle_calls"] == 0
    assert "no molecule that was never made" in result["interpretation"]
    for key in ("score_blind_rate", "backed_up_rate"):
        assert 0.0 <= result[key] <= 1.0
