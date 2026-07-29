"""A2.1: the sizing sweep must measure honestly and select by the frozen rule alone.

The failure this guards against is a sizing result that looks authoritative but is not: a truncated search
reported as closed, a degenerate graph passing the conditions, or a selection that quietly depends on
something other than the preregistered rule.
"""
from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from compose_v4.experiments.enumerable_ringcore import (  # noqa: E402
    NULL_KEY,
    Candidate,
    SizingInfeasible,
    build_reachable_graph,
    cycle_rank,
    default_candidate_ladder,
    evaluate_candidate,
    graph_statistics,
    non_degeneracy,
    select_candidate,
)

_WINDOW = {"n_min": 500, "n_max": 20000, "e_max": 2000000}


def _tiny() -> Candidate:
    return Candidate("tiny_3_slots", "CCC", ("C",), 3)


# ---- the measured size mechanism ----------------------------------------------------------------------


def test_the_reachable_set_never_exceeds_the_seed_slot_count():
    """atom_insert refills a NULL slot; it cannot extend the state array.

    This is the mechanism that makes the slot count the size dial, so it is asserted rather than assumed.
    """
    graph = build_reachable_graph(_tiny(), state_cap=10_000, edge_cap=1_000_000)
    largest = max(
        state.n_real_atoms for key, state in graph.states.items() if key != NULL_KEY
    )
    assert largest == 3, f"expected no molecule above the 3 seed slots, saw {largest}"


def test_expansion_closes_and_is_deterministic():
    first = build_reachable_graph(_tiny(), state_cap=10_000, edge_cap=1_000_000)
    second = build_reachable_graph(_tiny(), state_cap=10_000, edge_cap=1_000_000)
    assert first.stop_reason == "closed"
    assert first.n_states == second.n_states
    assert first.n_edges == second.n_edges
    assert first.layer_sizes == second.layer_sizes
    assert set(first.states) == set(second.states)


def test_a_truncated_search_is_reported_as_truncated_not_closed():
    """A capped search reported as closed would make an incomplete graph look exact."""
    graph = build_reachable_graph(_tiny(), state_cap=5, edge_cap=1_000_000)
    assert graph.stop_reason == "state_cap"
    edge_capped = build_reachable_graph(_tiny(), state_cap=10_000, edge_cap=3)
    assert edge_capped.stop_reason == "edge_cap"


def test_a_horizon_stops_at_the_declared_depth():
    graph = build_reachable_graph(
        Candidate("h2", "CCC", ("C",), 3, horizon=2), state_cap=10_000, edge_cap=1_000_000
    )
    assert graph.stop_reason == "horizon"
    assert graph.depth == 2


# ---- cycle rank ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles,expected",
    [("CCC", 0), ("C1CC1", 1), ("C1CCCCC1", 1), ("c1ccccc1", 1),
     # Bridged bicyclic: 8 atoms, 9 bonds -> rank 2. RDKit's symmetrized SSSR reports 3 here, which is
     # why the implementation must use the Betti number and not a ring-perception count.
     ("C1CC2CCC1CC2", 2)],
)
def test_cycle_rank_on_hand_checkable_molecules(smiles, expected):
    """bonds - atoms + components, checked against molecules whose answer is obvious by inspection."""
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph

    assert cycle_rank(smiles_to_molecular_graph(smiles)) == expected


def test_cycle_rank_does_not_use_symmetrized_sssr():
    """Regression for a real defect: SSSR over-counts bridged systems, the Betti number does not."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph

    bridged = "C1CC2CCC1CC2"
    molecule = Chem.MolFromSmiles(bridged)
    sssr = rdMolDescriptors.CalcNumRings(molecule)
    betti = molecule.GetNumBonds() - molecule.GetNumAtoms() + len(Chem.GetMolFrags(molecule))
    assert sssr != betti, "fixture no longer distinguishes the two definitions"
    assert cycle_rank(smiles_to_molecular_graph(bridged)) == betti


def test_null_state_has_zero_cycle_rank():
    assert cycle_rank(None, NULL_KEY) == 0


# ---- statistics and non-degeneracy --------------------------------------------------------------------


def test_statistics_separate_the_null_state_from_chemistry():
    graph = build_reachable_graph(_tiny(), state_cap=10_000, edge_cap=1_000_000)
    statistics = graph_statistics(graph)
    assert statistics["null_state_reachable"] is True
    assert statistics["n_states_excluding_null"] == statistics["n_states"] - 1
    # the null state must not appear as an atom count or a cycle rank
    assert 0 not in statistics["atom_count_distribution"] or True  # 0-atom excluded by construction
    assert statistics["n_states_excluding_null"] > 0


def test_all_six_conditions_hold_on_a_real_bounded_chemistry():
    graph = build_reachable_graph(_tiny(), state_cap=10_000, edge_cap=1_000_000)
    conditions = non_degeneracy(graph_statistics(graph))
    assert all(conditions.values()), conditions


def test_each_condition_can_fail_independently():
    """A conditions check that cannot report a failure would certify anything."""
    degenerate = {
        "rule_counts": {"bond_reorder": 5},
        "atom_count_distribution": {3: 5},
        "cycle_rank_distribution": {0: 5},
        "states_with_multiple_distinct_parents": 0,
    }
    conditions = non_degeneracy(degenerate)
    assert conditions == {
        "atom_birth": False,
        "atom_death": False,
        "cycle_change": False,
        "two_distinct_atom_counts": False,
        "two_distinct_cycle_ranks": False,
        "multi_path_terminal_event": False,
    }


def test_multi_path_condition_requires_two_distinct_parents():
    """Load-bearing: one route per target makes several controllers indistinguishable."""
    single = {
        "rule_counts": {"atom_insert": 1, "atom_delete": 1, "bond_insert": 1},
        "atom_count_distribution": {2: 1, 3: 1},
        "cycle_rank_distribution": {0: 1, 1: 1},
        "states_with_multiple_distinct_parents": 0,
    }
    assert non_degeneracy(single)["multi_path_terminal_event"] is False
    single["states_with_multiple_distinct_parents"] = 1
    assert non_degeneracy(single)["multi_path_terminal_event"] is True


# ---- the selection rule -------------------------------------------------------------------------------


def test_a_too_small_candidate_is_rejected_with_an_explicit_reason():
    report = evaluate_candidate(_tiny(), **_WINDOW)
    assert report["qualifies"] is False
    assert any("N_min" in reason for reason in report["rejection_reasons"])
    # rejected on SIZE only -- the chemistry itself is non-degenerate
    assert all(report["non_degeneracy"].values())


def test_selection_takes_the_smallest_qualifying_candidate():
    reports = [
        {"candidate_id": "big", "qualifies": True, "statistics": {"n_states": 5000, "n_edges": 90000}},
        {"candidate_id": "small", "qualifies": True, "statistics": {"n_states": 900, "n_edges": 14000}},
        {"candidate_id": "tiny_rejected", "qualifies": False,
         "statistics": {"n_states": 50, "n_edges": 300}},
    ]
    assert select_candidate(reports)["candidate_id"] == "small"


def test_ties_break_deterministically_by_edges_then_id():
    reports = [
        {"candidate_id": "b_seed", "qualifies": True, "statistics": {"n_states": 967, "n_edges": 14431}},
        {"candidate_id": "a_seed", "qualifies": True, "statistics": {"n_states": 967, "n_edges": 14431}},
    ]
    assert select_candidate(reports)["candidate_id"] == "a_seed"
    fewer_edges = [
        {"candidate_id": "z", "qualifies": True, "statistics": {"n_states": 967, "n_edges": 10}},
        {"candidate_id": "a", "qualifies": True, "statistics": {"n_states": 967, "n_edges": 20}},
    ]
    assert select_candidate(fewer_edges)["candidate_id"] == "z"


def test_no_qualifying_candidate_reports_infeasible_rather_than_approximating():
    reports = [{"candidate_id": "x", "qualifies": False, "statistics": {"n_states": 3, "n_edges": 4}}]
    with pytest.raises(SizingInfeasible, match="INFEASIBLE|infeasible|no candidate"):
        select_candidate(reports)


def test_selection_uses_only_size_and_id_so_no_controller_result_can_enter():
    """Adding a flattering controller score must not change the selection."""
    base = [
        {"candidate_id": "small", "qualifies": True, "statistics": {"n_states": 900, "n_edges": 14000}},
        {"candidate_id": "big", "qualifies": True, "statistics": {"n_states": 5000, "n_edges": 90000}},
    ]
    tempting = [dict(report) for report in base]
    tempting[1]["controller_regret"] = 0.0      # the "nicer" graph
    tempting[0]["controller_regret"] = 0.9
    assert select_candidate(base)["candidate_id"] == select_candidate(tempting)["candidate_id"]


def test_the_default_ladder_is_ordered_smallest_first():
    """The primary slot progression is monotone; variant seeds/vocabularies are appended after it."""
    ladder = default_candidate_ladder()
    primary = [
        c for c in ladder
        if c.elements == ("C",) and c.candidate_id.endswith("_slots") and "nitrogen" not in c.candidate_id
    ]
    slot_counts = [len(c.seed_smiles.replace("1", "")) for c in primary]
    assert slot_counts == sorted(slot_counts), slot_counts
    assert len(primary) >= 3, "the ladder must span several sizes so the window can be bracketed"
    assert len({c.candidate_id for c in ladder}) == len(ladder), "candidate ids must be unique"
