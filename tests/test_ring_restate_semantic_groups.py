"""Semantic successor-group evidence for ring-system restatement."""

from __future__ import annotations

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.ring_restate_semantics import (
    enumerate_ring_restate_semantic_groups,
)


def test_cyclohexane_aliases_share_one_six_edge_aromatization_descriptor() -> None:
    groups = enumerate_ring_restate_semantic_groups(
        smiles_to_molecular_graph("C1CCCCC1"),
        system=de_novo_rewrite_system(),
    )
    assert len(groups.actions) == 2
    assert groups.successor_keys == ("c1ccccc1",)
    assert groups.successor_group_ids == (0, 0)
    assert groups.group_multiplicities == (2,)
    assert len(groups.group_descriptors[0]) == 6
    assert {
        (source_class, target_class)
        for _, _, source_class, target_class in groups.group_descriptors[0]
    } == {(1, 4)}


def test_two_ring_fixture_keeps_distinct_semantic_successor_groups() -> None:
    groups = enumerate_ring_restate_semantic_groups(
        smiles_to_molecular_graph("Cc1ccccc1CCc1cccc(Cl)c1"),
        system=de_novo_rewrite_system(),
    )
    assert len(groups.actions) == 2
    assert groups.successor_group_ids == (0, 1)
    assert groups.group_multiplicities == (1, 1)
    assert len(groups.successor_keys) == 2
    assert all(len(descriptor) == 6 for descriptor in groups.group_descriptors)
    assert groups.group_descriptors[0] != groups.group_descriptors[1]


def test_dearomatization_descriptor_is_complete_not_raw_change_subset() -> None:
    groups = enumerate_ring_restate_semantic_groups(
        smiles_to_molecular_graph("c1ccccc1"),
        system=de_novo_rewrite_system(),
    )
    assert len(groups.actions) == 1
    assert groups.successor_keys == ("C1CCCCC1",)
    assert len(groups.actions[0].changes) == 3
    assert len(groups.group_descriptors[0]) == 6
    assert {
        (source_class, target_class)
        for _, _, source_class, target_class in groups.group_descriptors[0]
    } == {(4, 1)}
