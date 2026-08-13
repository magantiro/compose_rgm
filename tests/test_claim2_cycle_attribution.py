"""Cycle-attribution definitions, pinned before they are read on real traces.

Written and committed BEFORE running the diagnostic. "What counts as a
cancelled edit" is exactly the kind of choice that, made after seeing the
numbers, can turn any result into any other result.
"""

from __future__ import annotations

import math

import pytest

from compose_v4.experiments.claim2_cycle_attribution import (
    DIRECTIONAL_CORPUS_THRESHOLD,
    INCONCLUSIVE,
    INHERITED,
    INHERITED_CORPUS_THRESHOLD,
    INVERSE_FAMILIES,
    MODEL_SPECIFIC,
    SELF_INVERTING_FAMILIES,
    CycleAttributionError,
    attribute_cycles,
    attribution_verdict,
    edges_from_trajectory,
    net_displacement_efficiency,
    reduce_out_and_back,
    reversibility_census,
    summarize,
)


# ---- out-and-back reduction ----------------------------------------------


def test_a_forward_walk_cancels_nothing():
    assert reduce_out_and_back(["a", "b", "c", "d"]) == ("a", "b", "c", "d")


def test_an_immediate_out_and_back_cancels():
    assert reduce_out_and_back(["a", "b", "a"]) == ("a",)


def test_nested_out_and_back_collapses_through_the_stack():
    """a b c b a -> a: once c b c cancels, the outer pair becomes adjacent."""
    assert reduce_out_and_back(["a", "b", "c", "b", "a"]) == ("a",)


def test_a_long_loop_is_not_cancelled():
    """Wandering back after five steps genuinely traversed those states."""
    assert reduce_out_and_back(["a", "b", "c", "d", "a"]) == ("a", "b", "c", "d", "a")


def test_reduction_refuses_an_empty_trajectory():
    with pytest.raises(CycleAttributionError, match="empty"):
        reduce_out_and_back([])


# ---- the headline decomposition ------------------------------------------


def test_a_clean_forward_trajectory_has_no_cancellation():
    a = attribute_cycles(
        ["a", "b", "c", "d", "e", "f", "g"],
        [["atom_insert"]] * 6,
    )
    assert a.committed_edits == 6
    assert a.net_edits == 6
    assert a.cancelled_edits == 0
    assert a.cancelled_fraction == pytest.approx(0.0)
    assert a.immediate_two_cycles == 0
    assert a.longer_revisits == 0


def test_a_pure_oscillation_cancels_almost_everything():
    """a b a b a b a -- six edits, no net motion. This is the case that must
    not receive credit for six units of displacement."""
    a = attribute_cycles(
        ["a", "b", "a", "b", "a", "b", "a"],
        [["atom_insert"], ["atom_delete"]] * 3,
    )
    assert a.committed_edits == 6
    assert a.net_edits == 0
    assert a.cancelled_edits == 6
    assert a.cancelled_fraction == pytest.approx(1.0)
    assert a.immediate_two_cycles == 5


def test_immediate_two_cycles_and_longer_revisits_are_counted_separately():
    """Oscillating and wandering back are different behaviours."""
    immediate = attribute_cycles(["a", "b", "a"], [["atom_insert"], ["atom_delete"]])
    assert immediate.immediate_two_cycles == 1
    assert immediate.longer_revisits == 0

    wandering = attribute_cycles(
        ["a", "b", "c", "a"], [["atom_insert"], ["atom_insert"], ["atom_delete"]]
    )
    assert wandering.immediate_two_cycles == 0
    assert wandering.longer_revisits == 1


def test_inverse_family_pairs_are_necessary_but_not_sufficient():
    """An insert then a delete elsewhere is an inverse-family pair that did NOT
    cancel. Counting family pairs alone would overstate cancellation."""
    a = attribute_cycles(
        ["a", "b", "c"],
        [["atom_insert"], ["atom_delete"]],
    )
    assert a.inverse_family_pairs == 1
    assert a.inverse_family_pairs_that_cancelled == 0

    b = attribute_cycles(
        ["a", "b", "a"],
        [["atom_insert"], ["atom_delete"]],
    )
    assert b.inverse_family_pairs == 1
    assert b.inverse_family_pairs_that_cancelled == 1


def test_non_inverse_families_are_not_counted_as_a_pair():
    a = attribute_cycles(["a", "b", "c"], [["atom_insert"], ["bond_reorder"]])
    assert a.inverse_family_pairs == 0


def test_reversal_rate_is_attributed_to_the_family_that_was_undone():
    a = attribute_cycles(
        ["a", "b", "a", "c"],
        [["atom_insert"], ["atom_delete"], ["bond_reorder"]],
    )
    assert a.reversal_by_family["atom_insert"]["rate"] == pytest.approx(1.0)
    assert a.reversal_by_family["bond_reorder"]["rate"] == pytest.approx(0.0)


def test_attribution_refuses_mismatched_family_labels():
    with pytest.raises(CycleAttributionError, match="family labels"):
        attribute_cycles(["a", "b", "c"], [["atom_insert"]])


def test_a_source_only_trajectory_is_not_a_crash():
    a = attribute_cycles(["a"], [])
    assert a.committed_edits == 0
    assert a.net_edits == 0
    assert a.cancelled_fraction == 0.0


# ---- the inverse-family map ----------------------------------------------


def test_the_inverse_map_is_symmetric():
    """If f can undo g then g must be able to undo f, or the map is incoherent."""
    for family, inverses in INVERSE_FAMILIES.items():
        for other in inverses:
            assert family in INVERSE_FAMILIES[other], f"{family}/{other} asymmetric"


def test_ring_close_and_ring_open_are_paired():
    """cycle_insert closes to a ring system; cycle_attach opens from one."""
    assert INVERSE_FAMILIES["cycle_insert"] == frozenset({"cycle_attach"})
    assert INVERSE_FAMILIES["cycle_attach"] == frozenset({"cycle_insert"})


def test_self_inverting_families_are_identified():
    assert SELF_INVERTING_FAMILIES == {
        "bond_reorder",
        "atom_restate",
        "ring_system_restate",
        "bond_reroute",
    }


def test_the_map_covers_every_active_family():
    import json
    from pathlib import Path

    cells = json.loads(
        (
            Path(__file__).resolve().parent.parent
            / "configs/editing_v2_process_v2_capability_cells.json"
        ).read_text()
    )
    assert set(INVERSE_FAMILIES) == set(cells["active_families"])


# ---- net displacement -----------------------------------------------------


def test_cancelled_motion_does_not_get_credit_for_its_committed_length():
    """The lead's requirement: six edits with four cancelled is not six units."""
    a = attribute_cycles(
        ["a", "b", "a", "b", "c", "d", "e"],
        [["atom_insert"], ["atom_delete"], ["atom_insert"], ["atom_insert"],
         ["atom_insert"], ["atom_insert"]],
    )
    assert a.committed_edits == 6
    assert a.net_edits == 4
    efficiency = net_displacement_efficiency(0.6, a)
    assert efficiency["per_committed_edit"] == pytest.approx(0.1)
    assert efficiency["per_net_edit"] == pytest.approx(0.15)
    assert efficiency["inflation"] == pytest.approx(1.5)


def test_displacement_efficiency_tolerates_a_missing_distance():
    a = attribute_cycles(["a", "b"], [["atom_insert"]])
    assert net_displacement_efficiency(None, a)["per_net_edit"] is None


# ---- the central causal question -----------------------------------------


def test_a_fully_reversible_edge_set_scores_one():
    census = reversibility_census([("a", "b"), ("b", "a")])
    assert census.mutual_edge_fraction == pytest.approx(1.0)
    assert census.mutual_pairs == 1


def test_a_purely_directional_edge_set_scores_zero():
    census = reversibility_census([("a", "b"), ("b", "c"), ("c", "d")])
    assert census.mutual_edge_fraction == pytest.approx(0.0)
    assert census.mutual_pairs == 0


def test_the_census_is_over_distinct_edges_not_occurrences():
    """One heavily repeated transition must not dominate the statistic."""
    edges = [("a", "b")] * 100 + [("b", "c"), ("c", "b")]
    census = reversibility_census(edges)
    assert census.distinct_directed_edges == 3
    assert census.mutual_edge_fraction == pytest.approx(2 / 3)


def test_self_loops_are_excluded_from_the_census():
    census = reversibility_census([("a", "a"), ("a", "b")])
    assert census.distinct_directed_edges == 1


def test_edges_from_trajectory_round_trips():
    assert edges_from_trajectory(["a", "b", "c"]) == [("a", "b"), ("b", "c")]


# ---- the preregistered reading -------------------------------------------


def test_a_reversible_corpus_licenses_the_inherited_reading():
    assert attribution_verdict(0.55, 0.5) == INHERITED


def test_a_directional_corpus_licenses_the_pathology_reading():
    """The diagnostic must be able to come out against R_theta."""
    assert attribution_verdict(0.02, 0.5) == MODEL_SPECIFIC


def test_a_middling_corpus_is_inconclusive_rather_than_guessed():
    middle = (INHERITED_CORPUS_THRESHOLD + DIRECTIONAL_CORPUS_THRESHOLD) / 2
    assert attribution_verdict(middle, 0.5) == INCONCLUSIVE


def test_a_missing_corpus_census_is_inconclusive_not_favourable():
    assert attribution_verdict(None, 0.5) == INCONCLUSIVE


def test_a_directional_corpus_with_no_model_cancellation_is_not_a_pathology():
    """No cycling to explain means nothing to blame on the model."""
    assert attribution_verdict(0.02, 0.0) == INCONCLUSIVE


def test_thresholds_are_ordered_and_leave_an_inconclusive_band():
    assert 0.0 < DIRECTIONAL_CORPUS_THRESHOLD < INHERITED_CORPUS_THRESHOLD < 1.0


# ---- aggregation ----------------------------------------------------------


def test_summarize_averages_across_trajectories():
    clean = attribute_cycles(["a", "b", "c"], [["atom_insert"]] * 2)
    cycled = attribute_cycles(["a", "b", "a"], [["atom_insert"], ["atom_delete"]])
    mean = summarize([clean, cycled])
    assert mean["net_edits"] == pytest.approx(1.0)
    assert mean["cancelled_fraction"] == pytest.approx(0.5)


def test_summarize_of_nothing_is_empty_not_a_crash():
    assert summarize([]) == {}
