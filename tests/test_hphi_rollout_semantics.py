"""The three execution invariants, made mechanical BEFORE the cloud launch.

1. the rollout law is the controller's base chain (parity fixture lives in the
   Modal app; the step-accounting half is here)
2. terminal reachability, NOT "ever hit the goal"
3. early termination is explicit, never a silent H=6 endpoint
"""

from __future__ import annotations

import pytest

from compose_v4.experiments.hphi_rollout import (
    BENCHMARK_REGION,
    HORIZON,
    QED_THRESHOLDS,
    SIMILARITY_FLOORS,
    Termination,
    ever_hit_label,
    first_hit_step,
    prefix_examples,
    registered_regions,
    terminal_label,
)


# --------------------------------------------------------------------------
# The frozen goal grid
# --------------------------------------------------------------------------

def test_registered_grid_is_the_preregistered_5x4():
    assert QED_THRESHOLDS == (0.75, 0.80, 0.85, 0.90, 0.95)
    assert SIMILARITY_FLOORS == (0.30, 0.40, 0.50, 0.60)
    assert len(registered_regions()) == 20
    assert BENCHMARK_REGION in registered_regions(), (
        "the benchmark region must be a MEMBER of the grid, not a special case")


# --------------------------------------------------------------------------
# INVARIANT 2 -- the load-bearing one
# --------------------------------------------------------------------------

def test_TERMINAL_not_ever_hit__crossing_then_falling_back_is_NEGATIVE():
    """The exact scenario that separates Doob value from hitting probability.

    QED crosses 0.90 at step 4 and falls back to 0.86 by step 6. Terminal
    reachability says NEGATIVE. "Ever hit" would say positive.
    """
    qed = [0.75, 0.79, 0.84, 0.88, 0.93, 0.90, 0.86]
    sim = [1.00, 0.92, 0.85, 0.78, 0.71, 0.66, 0.62]
    term = Termination(kind="complete", edits=6)

    assert terminal_label(qed[-1], sim[-1], BENCHMARK_REGION, term) == 0
    assert ever_hit_label(qed, sim, BENCHMARK_REGION) == 1, "fixture must differ"

    for ex in prefix_examples(list("abcdefg"), qed, sim, term):
        assert ex["labels"]["0.90_0.40"] == 0, (
            "every prefix must carry the TERMINAL label, not its own membership")


def test_every_prefix_carries_the_same_terminal_label():
    qed = [0.72, 0.80, 0.88, 0.91, 0.93, 0.95, 0.96]
    sim = [1.00, 0.90, 0.80, 0.70, 0.60, 0.55, 0.52]
    exs = prefix_examples(list("abcdefg"), qed, sim,
                          Termination(kind="complete", edits=6))
    assert {e["labels"]["0.90_0.40"] for e in exs} == {1}


def test_budget_counts_down_from_the_horizon():
    exs = prefix_examples(list("abcdefg"), [0.7] * 7, [1.0] * 7,
                          Termination(kind="complete", edits=6))
    assert [e["budget_remaining"] for e in exs] == [6, 5, 4, 3, 2, 1, 0]


def test_similarity_is_to_the_IMMUTABLE_SOURCE_not_the_previous_state():
    """Source similarity must be monotone-ish and anchored, never pairwise.

    Encoded as a contract check: the first prefix is the source itself, so its
    similarity to the source is 1.0 by construction.
    """
    sim = [1.00, 0.88, 0.71, 0.64, 0.58, 0.51, 0.47]
    exs = prefix_examples(list("abcdefg"), [0.7] * 7, sim,
                          Termination(kind="complete", edits=6))
    assert exs[0]["state_similarity_to_source"] == 1.0
    assert [e["state_similarity_to_source"] for e in exs] == sim


def test_a_region_is_a_CONJUNCTION_both_constraints_bind():
    term = Termination(kind="complete", edits=6)
    assert terminal_label(0.95, 0.30, BENCHMARK_REGION, term) == 0   # sim fails
    assert terminal_label(0.85, 0.90, BENCHMARK_REGION, term) == 0   # qed fails
    assert terminal_label(0.95, 0.90, BENCHMARK_REGION, term) == 1


# --------------------------------------------------------------------------
# INVARIANT 3 -- early termination
# --------------------------------------------------------------------------

def test_CEMETERY_is_negative_for_every_region_even_if_properties_qualify():
    """A short rollout must NOT be promoted to an H=6 endpoint.

    Even with terminal properties that would qualify, an incomplete chain did
    not reach the goal within the budget.
    """
    dead = Termination(kind="cemetery", edits=3, reason="empty_fiber")
    for region in registered_regions():
        assert terminal_label(0.99, 0.99, region, dead) == 0


def test_cemetery_prefixes_are_still_emitted_and_labelled_negative():
    dead = Termination(kind="cemetery", edits=2, reason="empty_fiber")
    exs = prefix_examples(list("abc"), [0.7, 0.85, 0.99], [1.0, 0.8, 0.7], dead)
    assert len(exs) == 3
    assert all(e["labels"]["0.90_0.40"] == 0 for e in exs)
    assert all(e["termination"] == "cemetery" for e in exs)


def test_termination_kind_is_recorded_on_every_example():
    exs = prefix_examples(list("abcdefg"), [0.7] * 7, [1.0] * 7,
                          Termination(kind="complete", edits=6))
    assert all(e["termination"] == "complete" for e in exs)


def test_complete_requires_the_full_horizon():
    assert Termination(kind="complete", edits=6).is_complete
    assert not Termination(kind="cemetery", edits=5).is_complete


# --------------------------------------------------------------------------
# Structural
# --------------------------------------------------------------------------

def test_every_example_is_labelled_against_all_twenty_regions():
    exs = prefix_examples(list("abcdefg"), [0.7] * 7, [1.0] * 7,
                          Termination(kind="complete", edits=6))
    for e in exs:
        assert len(e["labels"]) == 20


def test_misaligned_sequences_raise_rather_than_silently_truncate():
    with pytest.raises(ValueError):
        prefix_examples(list("abc"), [0.7, 0.8], [1.0, 0.9, 0.8],
                        Termination(kind="complete", edits=6))


def test_horizon_is_six():
    assert HORIZON == 6


# --------------------------------------------------------------------------
# BOTH value definitions -- the corpus must support either controller
# --------------------------------------------------------------------------

def test_both_label_families_are_emitted_for_every_region():
    qed = [0.75, 0.79, 0.84, 0.88, 0.93, 0.90, 0.86]
    sim = [1.00, 0.92, 0.85, 0.78, 0.71, 0.66, 0.62]
    exs = prefix_examples(list("abcdefg"), qed, sim,
                          Termination(kind="complete", edits=6))
    for e in exs:
        assert len(e["labels"]) == 20 and len(e["labels_hit"]) == 20


def test_terminal_and_hit_DISAGREE_on_the_crossing_trajectory():
    """The whole point of storing both: they are different objects."""
    qed = [0.75, 0.79, 0.84, 0.88, 0.93, 0.90, 0.86]
    sim = [1.00, 0.92, 0.85, 0.78, 0.71, 0.66, 0.62]
    exs = prefix_examples(list("abcdefg"), qed, sim,
                          Termination(kind="complete", edits=6))
    assert exs[0]["labels"]["0.90_0.40"] == 0        # fixed-horizon: missed
    assert exs[0]["labels_hit"]["0.90_0.40"] == 1    # first-passage: reached


def test_hit_label_is_relative_to_the_PREFIX_not_the_whole_trajectory():
    """Hit is evaluated from the prefix forward, so it can flip to 0 late.

    Note the threshold is INCLUSIVE: qed[5] = 0.90 still qualifies at
    `>= 0.90`. The region is occupied at steps 4 and 5 and left at step 6.
    """
    qed = [0.75, 0.79, 0.84, 0.88, 0.93, 0.90, 0.86]
    sim = [1.00, 0.92, 0.85, 0.78, 0.71, 0.66, 0.62]
    exs = prefix_examples(list("abcdefg"), qed, sim,
                          Termination(kind="complete", edits=6))
    assert exs[4]["labels_hit"]["0.90_0.40"] == 1    # step 4 is in the region
    assert exs[5]["labels_hit"]["0.90_0.40"] == 1    # 0.90 >= 0.90, inclusive
    assert exs[6]["labels_hit"]["0.90_0.40"] == 0    # 0.86 -- left, unreachable


def test_first_hit_step_is_the_stopping_time_not_the_best_state():
    """STOP at the FIRST qualifying state, even if a later one scores higher."""
    qed = [0.70, 0.91, 0.99, 0.99]
    sim = [1.00, 0.55, 0.50, 0.45]
    assert first_hit_step(qed, sim, BENCHMARK_REGION) == 1, (
        "must return the first qualifying index, never the highest-QED index")


def test_first_hit_step_is_None_when_the_region_is_never_entered():
    assert first_hit_step([0.7] * 7, [1.0] * 7, BENCHMARK_REGION) is None


def test_first_hit_requires_BOTH_constraints_simultaneously():
    """A state passing QED and a later state passing similarity is not a hit."""
    qed = [0.95, 0.70, 0.70]
    sim = [0.20, 0.90, 0.90]
    assert first_hit_step(qed, sim, BENCHMARK_REGION) is None
