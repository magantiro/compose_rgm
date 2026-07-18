from __future__ import annotations

import pytest

import numpy as np

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    MolecularGraph,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import (
    empty_molecular_graph,
    is_connected_or_null,
    is_valid_state,
    pad_molecular_graph,
)
from compose_v4.rewrite import (
    AtomDelete,
    AtomInsert,
    BondInsert,
    BondReorder,
    BondReroute,
    InvalidRewrite,
    default_rewrite_system,
)


C = ELEMENT_TO_IDX["C"]


def test_null_to_ethene_is_a_valid_rewrite_program() -> None:
    system = default_rewrite_system()
    state = empty_molecular_graph(6)

    state = system.apply(state, "atom_insert", AtomInsert(0, C, 0, 4))
    state = system.apply(
        state,
        "atom_insert",
        AtomInsert(1, C, 0, 3, neighbors=((0, 1),)),
    )
    state = system.apply(state, "bond_reorder", BondReorder(0, 1, 2))

    assert is_valid_state(state)
    assert state.implicit_h_counts[0] == 2
    assert state.implicit_h_counts[1] == 2


def test_arbitrary_ring_closure_is_a_legal_micro_rewrite() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("CCC"), 8)
    successor = default_rewrite_system().apply(
        state, "bond_insert", BondInsert(0, 2, 1)
    )

    assert is_valid_state(successor)
    assert successor.bonds[0, 2] == 1


def test_invalid_rewrite_never_commits() -> None:
    neopentane = pad_molecular_graph(smiles_to_molecular_graph("CC(C)(C)C"), 8)
    assert neopentane.implicit_h_counts[1] == 0
    with pytest.raises(InvalidRewrite):
        default_rewrite_system().apply(
            neopentane,
            "atom_insert",
            AtomInsert(5, C, 0, 3, neighbors=((1, 1),)),
        )


def test_hard_condition_restricts_the_rewrite_space() -> None:
    def preserve_atom_zero(before, action, after) -> bool:
        return (
            before.atom_types[0] == after.atom_types[0]
            and before.formal_charges[0] == after.formal_charges[0]
        )

    methane = pad_molecular_graph(smiles_to_molecular_graph("C"), 4)
    system = default_rewrite_system(constraints=(preserve_atom_zero,))
    with pytest.raises(InvalidRewrite):
        system.apply(methane, "atom_delete", AtomDelete(0))


def test_alias_rates_are_aggregated_at_complete_successors() -> None:
    propane = pad_molecular_graph(smiles_to_molecular_graph("CCC"), 6)
    aggregated = default_rewrite_system().aggregate_successor_rates(
        propane,
        (
            ("bond_insert", BondInsert(0, 2, 1), 0.25),
            ("bond_insert", BondInsert(2, 0, 1), 0.75),
        ),
    )

    assert len(aggregated) == 1
    successor = next(iter(aggregated.values()))
    assert successor.rate == pytest.approx(1.0)
    assert len(successor.provenance) == 2


def test_bridge_reroute_changes_tree_wiring_without_disconnected_state() -> None:
    # Slot wiring A-C-B-D, with every atom a neutral saturated carbon.
    bonds = np.zeros((4, 4), dtype=np.int32)
    for a, b in ((0, 2), (2, 1), (1, 3)):
        bonds[a, b] = bonds[b, a] = 1
    degree = (bonds != 0).sum(axis=1)
    state = MolecularGraph(
        atom_types=np.full(4, C, dtype=np.int32),
        formal_charges=np.zeros(4, dtype=np.int32),
        implicit_h_counts=(4 - degree).astype(np.int32),
        bonds=bonds,
    )
    system = default_rewrite_system()

    state = system.apply(
        state,
        "bond_reroute",
        BondReroute(a=0, b=2, u=0, v=1),
    )
    assert is_valid_state(state) and is_connected_or_null(state)
    assert int(state.bonds[0, 2]) == 0
    assert int(state.bonds[0, 1]) == 1

    state = system.apply(
        state,
        "bond_reroute",
        BondReroute(a=1, b=3, u=2, v=3),
    )
    assert is_valid_state(state) and is_connected_or_null(state)
    assert {
        frozenset((a, b))
        for a in range(4)
        for b in range(a + 1, 4)
        if int(state.bonds[a, b]) != 0
    } == {
        frozenset((0, 1)),
        frozenset((1, 2)),
        frozenset((2, 3)),
    }


def test_bridge_reroute_rejects_identity_reconnection() -> None:
    propane = pad_molecular_graph(smiles_to_molecular_graph("CCC"), 4)
    with pytest.raises(InvalidRewrite):
        default_rewrite_system().apply(
            propane,
            "bond_reroute",
            BondReroute(a=0, b=1, u=0, v=1),
        )
