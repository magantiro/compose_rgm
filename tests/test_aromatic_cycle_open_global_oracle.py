"""Independent-oracle tests for bounded semantic aromatic cycle opening."""

from __future__ import annotations

from math import prod

import numpy as np
import pytest

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    MolecularGraph,
    smiles_to_molecular_graph,
)
from compose_v4.experiments.aromatic_cycle_open_global_oracle import (
    enumerate_global_milp_kekule_aliases,
)
from compose_v4.experiments.aromatic_cycle_open_semantics import (
    AromaticCycleOpenAliasOverflow,
    enumerate_charge_h_preserving_kekule_aliases,
    enumerate_component_factored_kekule_assignments,
    resolve_component_factored_cycle_open,
    resolve_edge_anchored_cycle_open,
)
from compose_v4.rewrite.operators import BondDelete


def _aromatic_edges(state: MolecularGraph) -> tuple[tuple[int, int], ...]:
    perceived = resonance_invariant_bond_classes(state)
    return tuple(
        (left, right)
        for left in range(state.n_atoms)
        for right in range(left + 1, state.n_atoms)
        if int(perceived[left, right]) == BOND_AROMATIC
    )


def _same_exact_state(left: MolecularGraph, right: MolecularGraph) -> bool:
    return all(
        np.array_equal(getattr(left, field), getattr(right, field))
        for field in (
            "atom_types",
            "formal_charges",
            "implicit_h_counts",
            "bonds",
        )
    )


def _assert_factored_matches_global_oracle(state: MolecularGraph) -> None:
    oracle = enumerate_global_milp_kekule_aliases(state, maximum_assignments=4096)
    components = enumerate_component_factored_kekule_assignments(state)
    assert oracle.diagnostics.preserving_alias_count == prod(
        len(component.bond_orders) for component in components
    )
    for edge in _aromatic_edges(state):
        expected = resolve_edge_anchored_cycle_open(
            state,
            BondDelete(*edge),
            maximum_aliases=4096,
            enumeration=oracle.enumeration,
        )
        observed = resolve_component_factored_cycle_open(state, BondDelete(*edge))
        assert observed.admitted is expected.admitted
        assert observed.rejection_code == expected.rejection_code
        assert observed.source_key == expected.source_key
        assert observed.edge == expected.edge
        assert observed.semantic_aromatic_edge is expected.semantic_aromatic_edge
        assert observed.enumerated_alias_count == expected.enumerated_alias_count
        assert observed.forced_single_alias_count == expected.forced_single_alias_count
        assert observed.canonical_product_keys == expected.canonical_product_keys
        assert observed.inverse_bond_order == expected.inverse_bond_order
        if expected.successor is None:
            assert observed.successor is None
        else:
            assert observed.successor is not None
            assert _same_exact_state(observed.successor, expected.successor)


@pytest.mark.parametrize(
    "smiles",
    (
        "c1ccccc1",
        "c1cc[nH]c1",
        "c1ccc2ccccc2c1",
        "c1ccc2ncccc2c1",
        "c1ccccc1Oc2ccccc2",
        "c1ccncc1CCc2cc[nH]c2",
    ),
)
def test_factored_resolver_matches_independent_global_milp_oracle(smiles: str) -> None:
    _assert_factored_matches_global_oracle(smiles_to_molecular_graph(smiles))


@pytest.mark.parametrize(
    "smiles",
    (
        "CC(C)(C)c1cc[n+](CCCCC[n+]2ccc(C(C)(C)C)cc2)cc1",
        "CC(C)(C)c1cc[n+](CCCCCCCCCC[n+]2ccc(C(C)(C)C)cc2)cc1",
    ),
)
def test_rdkit_supplier_is_incomplete_for_two_validation_mismatch_patterns(
    smiles: str,
) -> None:
    """Preserve the exact failure mechanism found by the V1 validation audit."""

    source = smiles_to_molecular_graph(smiles)
    rdkit = enumerate_charge_h_preserving_kekule_aliases(source, maximum_aliases=4096)
    global_oracle = enumerate_global_milp_kekule_aliases(
        source, maximum_assignments=4096
    )
    components = enumerate_component_factored_kekule_assignments(source)

    assert tuple(len(component.bond_orders) for component in components) == (2, 2)
    assert len(rdkit.aliases) == 2
    assert global_oracle.diagnostics.feasible_assignment_count == 4
    assert len(global_oracle.enumeration.aliases) == 4
    _assert_factored_matches_global_oracle(source)


def test_global_oracle_resolves_the_previous_whole_molecule_overflow_fixture() -> None:
    source = smiles_to_molecular_graph(
        "CC(=O)Nc1ccc(Nc2nc3ccc([N+](=O)[O-])cc3nc2Nc2ccc(NC(C)=O)cc2)cc1"
    )
    components = enumerate_component_factored_kekule_assignments(source)
    assert tuple(len(component.bond_orders) for component in components) == (2, 3, 2)

    global_oracle = enumerate_global_milp_kekule_aliases(
        source, maximum_assignments=4096
    )
    assert global_oracle.diagnostics.feasible_assignment_count == 12
    assert global_oracle.diagnostics.preserving_alias_count == 12
    assert global_oracle.diagnostics.solver_call_count == 13
    _assert_factored_matches_global_oracle(source)


def test_assignment_cap_fails_instead_of_truncating() -> None:
    source = smiles_to_molecular_graph("c1ccccc1Oc2ccccc2")
    with pytest.raises(AromaticCycleOpenAliasOverflow) as caught:
        enumerate_global_milp_kekule_aliases(source, maximum_assignments=3)
    assert caught.value.maximum_aliases == 3
    assert caught.value.observed_structures == 4


def test_nonaromatic_state_has_one_identity_alias_without_solver_work() -> None:
    source = smiles_to_molecular_graph("CCC")
    result = enumerate_global_milp_kekule_aliases(source)
    assert result.enumeration.aromatic_edges == ()
    assert result.enumeration.aliases == (source,)
    assert result.diagnostics.semantic_aromatic_edge_count == 0
    assert result.diagnostics.feasible_assignment_count == 0
    assert result.diagnostics.preserving_alias_count == 1
    assert result.diagnostics.solver_call_count == 0


def test_invalid_source_is_rejected_at_the_oracle_boundary() -> None:
    disconnected = smiles_to_molecular_graph("c1ccccc1.c2ccccc2")
    with pytest.raises(ValueError, match="valid connected source"):
        enumerate_global_milp_kekule_aliases(disconnected)
