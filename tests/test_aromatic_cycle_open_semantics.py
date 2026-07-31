"""Focused tests for the non-authorizing aromatic cycle-open prototype."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import is_connected_or_null
from compose_v4.experiments.aromatic_cycle_open_semantics import (
    PROTOTYPE_STATUS,
    AromaticCycleOpenAliasOverflow,
    AromaticCycleOpenRejectionCode,
    enumerate_charge_h_preserving_kekule_aliases,
    inverse_mark_for_resolution,
    rdkit_kekule_supplier,
    resolve_edge_anchored_cycle_open,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import (
    BondDelete,
    apply_bond_delete,
    apply_bond_insert,
    is_valid_bond_insert,
)


def _aromatic_edges(state) -> tuple[tuple[int, int], ...]:
    perceived = resonance_invariant_bond_classes(state)
    return tuple(
        (left, right)
        for left in range(state.n_atoms)
        for right in range(left + 1, state.n_atoms)
        if int(perceived[left, right]) == BOND_AROMATIC
    )


def _same_exact_state(left, right) -> bool:
    return all(
        np.array_equal(getattr(left, field), getattr(right, field))
        for field in (
            "atom_types",
            "formal_charges",
            "implicit_h_counts",
            "bonds",
        )
    )


def _reverse_supplier(molecule: Chem.Mol, maximum: int):
    return tuple(reversed(tuple(rdkit_kekule_supplier(molecule, maximum))))


def test_benzene_two_phases_have_identical_edge_anchored_products() -> None:
    source = smiles_to_molecular_graph("c1ccccc1")
    enumeration = enumerate_charge_h_preserving_kekule_aliases(source)
    assert len(enumeration.aliases) == 2
    assert all(
        canonical_state_key(alias) == canonical_state_key(source) for alias in enumeration.aliases
    )
    assert all(is_connected_or_null(alias) for alias in enumeration.aliases)
    assert all(
        np.array_equal(alias.atom_types, source.atom_types)
        and np.array_equal(alias.formal_charges, source.formal_charges)
        and np.array_equal(alias.implicit_h_counts, source.implicit_h_counts)
        for alias in enumeration.aliases
    )

    for edge in enumeration.aromatic_edges:
        resolutions = tuple(
            resolve_edge_anchored_cycle_open(alias, BondDelete(*edge))
            for alias in enumeration.aliases
        )
        assert all(resolution.admitted for resolution in resolutions)
        assert all(resolution.prototype_status == PROTOTYPE_STATUS for resolution in resolutions)
        assert all(resolution.semantic_aromatic_edge for resolution in resolutions)
        assert all(len(resolution.canonical_product_keys) == 1 for resolution in resolutions)
        assert _same_exact_state(resolutions[0].successor, resolutions[1].successor)
        for alias, resolution in zip(enumeration.aliases, resolutions, strict=True):
            assert (
                int(resolution.successor.implicit_h_counts[edge[0]])
                == int(alias.implicit_h_counts[edge[0]]) + 1
            )
            assert (
                int(resolution.successor.implicit_h_counts[edge[1]])
                == int(alias.implicit_h_counts[edge[1]]) + 1
            )


@pytest.mark.parametrize(
    "smiles",
    (
        "Cc1cccc(Cl)c1",
        "COc1cc(Cl)ccc1N",
    ),
)
def test_asymmetric_alternate_phases_have_identical_edge_anchored_products(
    smiles: str,
) -> None:
    source = smiles_to_molecular_graph(smiles)
    enumeration = enumerate_charge_h_preserving_kekule_aliases(source)
    assert len(enumeration.aliases) >= 2

    for edge in enumeration.aromatic_edges:
        resolutions = tuple(
            resolve_edge_anchored_cycle_open(alias, BondDelete(*edge))
            for alias in enumeration.aliases
        )
        assert all(resolution.admitted for resolution in resolutions)
        assert len({resolution.canonical_product_keys for resolution in resolutions}) == 1
        assert all(
            _same_exact_state(resolutions[0].successor, resolution.successor)
            for resolution in resolutions[1:]
        )


def test_pyridine_all_semantic_aromatic_edges_are_admitted() -> None:
    source = smiles_to_molecular_graph("c1ccncc1")
    edges = _aromatic_edges(source)
    assert len(edges) == 6
    resolutions = tuple(
        resolve_edge_anchored_cycle_open(source, BondDelete(*edge)) for edge in edges
    )
    assert all(resolution.admitted for resolution in resolutions)
    assert all(resolution.canonical_product_keys for resolution in resolutions)


def test_pyrrole_exposes_partial_forced_single_support_without_fallback() -> None:
    source = smiles_to_molecular_graph("c1cc[nH]c1")
    edges = _aromatic_edges(source)
    assert len(edges) == 5
    resolutions = tuple(
        resolve_edge_anchored_cycle_open(source, BondDelete(*edge)) for edge in edges
    )
    admitted = tuple(resolution for resolution in resolutions if resolution.admitted)
    rejected = tuple(resolution for resolution in resolutions if not resolution.admitted)
    assert len(admitted) == 3
    assert len(rejected) == 2
    assert {resolution.rejection_code for resolution in rejected} == {
        AromaticCycleOpenRejectionCode.NO_FORCED_SINGLE_ALIAS
    }


def test_naphthalene_groups_multiple_forced_aliases_into_one_product() -> None:
    source = smiles_to_molecular_graph("c1ccc2ccccc2c1")
    enumeration = enumerate_charge_h_preserving_kekule_aliases(source)
    assert len(enumeration.aliases) == 3
    resolutions = tuple(
        resolve_edge_anchored_cycle_open(source, BondDelete(*edge))
        for edge in enumeration.aromatic_edges
    )
    assert len(resolutions) == 11
    assert all(resolution.admitted for resolution in resolutions)
    assert all(len(resolution.canonical_product_keys) == 1 for resolution in resolutions)
    assert any(resolution.forced_single_alias_count > 1 for resolution in resolutions)


def test_cyclohexane_is_an_exact_nonaromatic_passthrough() -> None:
    source = smiles_to_molecular_graph("C1CCCCC1")
    action = BondDelete(0, 1)
    expected = apply_bond_delete(source, action)
    resolution = resolve_edge_anchored_cycle_open(source, action)

    assert resolution.admitted
    assert not resolution.semantic_aromatic_edge
    assert resolution.enumerated_alias_count == 0
    assert resolution.inverse_bond_order == int(source.bonds[0, 1])
    assert _same_exact_state(resolution.successor, expected)


@pytest.mark.parametrize(
    "smiles",
    ("c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1"),
)
def test_admitted_aromatic_open_has_single_bond_canonical_inverse(smiles: str) -> None:
    source = smiles_to_molecular_graph(smiles)
    resolution = next(
        resolution
        for edge in _aromatic_edges(source)
        if (
            resolution := resolve_edge_anchored_cycle_open(
                source,
                BondDelete(*edge),
            )
        ).admitted
    )
    inverse = inverse_mark_for_resolution(resolution)
    assert inverse.order == 1
    assert is_valid_bond_insert(resolution.successor, inverse)
    recovered = apply_bond_insert(resolution.successor, inverse)
    assert canonical_state_key(recovered) == canonical_state_key(source)


def test_supplier_order_does_not_change_aliases_or_product_representative() -> None:
    source = smiles_to_molecular_graph("c1ccc2ccccc2c1")
    edge = _aromatic_edges(source)[0]
    forward_aliases = enumerate_charge_h_preserving_kekule_aliases(source)
    reverse_aliases = enumerate_charge_h_preserving_kekule_aliases(
        source,
        supplier_factory=_reverse_supplier,
    )
    assert len(forward_aliases.aliases) == len(reverse_aliases.aliases)
    assert all(
        _same_exact_state(left, right)
        for left, right in zip(
            forward_aliases.aliases,
            reverse_aliases.aliases,
            strict=True,
        )
    )

    forward = resolve_edge_anchored_cycle_open(source, BondDelete(*edge))
    reverse = resolve_edge_anchored_cycle_open(
        source,
        BondDelete(*edge),
        supplier_factory=_reverse_supplier,
    )
    assert forward.admitted and reverse.admitted
    assert forward.canonical_product_keys == reverse.canonical_product_keys
    assert _same_exact_state(forward.successor, reverse.successor)


def test_cap_plus_one_resonance_structure_fails_loudly() -> None:
    source = smiles_to_molecular_graph("c1ccccc1")

    def overflow_supplier(molecule: Chem.Mol, maximum: int):
        return tuple(Chem.Mol(molecule) for _ in range(maximum))

    with pytest.raises(AromaticCycleOpenAliasOverflow) as caught:
        enumerate_charge_h_preserving_kekule_aliases(
            source,
            maximum_aliases=1,
            supplier_factory=overflow_supplier,
        )
    assert caught.value.reason_code == "kekule_alias_enumeration_overflow"
    assert caught.value.maximum_aliases == 1
    assert caught.value.observed_structures == 2


def test_empty_preserving_alias_set_has_explicit_rejection_code() -> None:
    source = smiles_to_molecular_graph("c1ccccc1")
    resolution = resolve_edge_anchored_cycle_open(
        source,
        BondDelete(*_aromatic_edges(source)[0]),
        supplier_factory=lambda _molecule, _maximum: (),
    )
    assert not resolution.admitted
    assert resolution.rejection_code == AromaticCycleOpenRejectionCode.NO_PRESERVING_KEKULE_ALIAS


def test_bridge_and_absent_edges_have_explicit_rejection_codes() -> None:
    source = smiles_to_molecular_graph("CCC")
    bridge = resolve_edge_anchored_cycle_open(source, BondDelete(0, 1))
    absent = resolve_edge_anchored_cycle_open(source, BondDelete(0, 2))
    assert bridge.rejection_code == AromaticCycleOpenRejectionCode.EDGE_IS_BRIDGE
    assert absent.rejection_code == AromaticCycleOpenRejectionCode.EDGE_ABSENT
