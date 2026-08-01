"""Oracle tests for the non-authorizing component-factored cycle-open prototype."""

from __future__ import annotations

from collections import Counter
from math import prod

import numpy as np
import pytest

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_SINGLE,
    MolecularGraph,
    smiles_to_molecular_graph,
)
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.aromatic_cycle_open_semantics import (
    COMPONENT_FACTORED_PROTOTYPE_STATUS,
    AromaticCycleOpenAliasOverflow,
    AromaticCycleOpenRejectionCode,
    enumerate_charge_h_preserving_kekule_aliases,
    enumerate_component_factored_kekule_assignments,
    inverse_mark_for_resolution,
    resolve_component_factored_cycle_open,
    resolve_edge_anchored_cycle_open,
)
from compose_v4.experiments.cycle_open_kekule_invariance import transport_bond_delete
from compose_v4.experiments.quotient_invariance import permute_persistent_slots
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import (
    BondDelete,
    apply_bond_insert,
    is_valid_bond_insert,
)


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


def _assert_oracle_equivalent(
    state: MolecularGraph,
    action: BondDelete,
) -> None:
    oracle = resolve_edge_anchored_cycle_open(
        state,
        action,
        maximum_aliases=4096,
    )
    factored = resolve_component_factored_cycle_open(state, action)
    assert factored.prototype_status == COMPONENT_FACTORED_PROTOTYPE_STATUS
    assert factored.admitted is oracle.admitted
    assert factored.rejection_code == oracle.rejection_code
    assert factored.source_key == oracle.source_key
    assert factored.edge == oracle.edge
    assert factored.semantic_aromatic_edge is oracle.semantic_aromatic_edge
    assert factored.enumerated_alias_count == oracle.enumerated_alias_count
    assert factored.forced_single_alias_count == oracle.forced_single_alias_count
    assert factored.canonical_product_keys == oracle.canonical_product_keys
    assert factored.inverse_bond_order == oracle.inverse_bond_order
    if factored.semantic_aromatic_edge and factored.admitted:
        components = enumerate_component_factored_kekule_assignments(state)
        selected = tuple(component for component in components if factored.edge in component.edges)
        assert len(selected) == 1
        edge_offset = selected[0].edges.index(factored.edge)
        expected_executions = sum(
            int(orders[edge_offset]) == BOND_SINGLE for orders in selected[0].bond_orders
        )
        assert factored.executed_product_count == expected_executions
        assert oracle.executed_product_count == factored.forced_single_alias_count
    else:
        assert factored.executed_product_count == oracle.executed_product_count
    if oracle.successor is None:
        assert factored.successor is None
    else:
        assert factored.successor is not None
        assert _same_exact_state(factored.successor, oracle.successor)


@pytest.mark.parametrize(
    "smiles",
    (
        "c1ccccc1",
        "Cc1cccc(Cl)c1",
        "COc1cc(Cl)ccc1N",
        "c1ccncc1",
        "c1cc[nH]c1",
        "c1ccc2ccccc2c1",
        "c1ccc2ncccc2c1",
        "c1ccc2[nH]ccc2c1",
    ),
)
def test_every_completed_single_system_fixture_matches_exhaustive_oracle(
    smiles: str,
) -> None:
    source = smiles_to_molecular_graph(smiles)
    edges = _aromatic_edges(source)
    assert edges
    for edge in edges:
        _assert_oracle_equivalent(source, BondDelete(*edge))


def test_nonaromatic_bridge_and_absent_behavior_matches_oracle() -> None:
    cyclohexane = smiles_to_molecular_graph("C1CCCCC1")
    chain = smiles_to_molecular_graph("CCC")
    _assert_oracle_equivalent(cyclohexane, BondDelete(0, 1))
    _assert_oracle_equivalent(chain, BondDelete(0, 1))
    _assert_oracle_equivalent(chain, BondDelete(0, 2))


@pytest.mark.parametrize(
    "smiles",
    (
        "c1ccccc1-c2ccccc2",
        "c1ccccc1Oc2ccccc2",
        "c1ccncc1CCc2cc[nH]c2",
    ),
)
def test_disconnected_aromatic_components_factor_exactly(smiles: str) -> None:
    source = smiles_to_molecular_graph(smiles)
    components = enumerate_component_factored_kekule_assignments(source)
    assert len(components) == 2
    oracle_aliases = enumerate_charge_h_preserving_kekule_aliases(
        source,
        maximum_aliases=4096,
    )
    factored_alias_count = prod(len(component.bond_orders) for component in components)
    assert factored_alias_count == len(oracle_aliases.aliases)
    for edge in _aromatic_edges(source):
        _assert_oracle_equivalent(source, BondDelete(*edge))


def test_global_alias_cardinality_is_not_reported_as_factored_execution() -> None:
    source = smiles_to_molecular_graph("c1ccccc1Oc2ccccc2")
    edge = _aromatic_edges(source)[0]
    oracle = resolve_edge_anchored_cycle_open(
        source,
        BondDelete(*edge),
        maximum_aliases=16,
    )
    factored = resolve_component_factored_cycle_open(source, BondDelete(*edge))

    assert factored.admitted and oracle.admitted
    assert factored.forced_single_alias_count == oracle.executed_product_count
    assert factored.executed_product_count < factored.forced_single_alias_count


def test_disconnected_molecular_source_remains_outside_declared_support() -> None:
    source = smiles_to_molecular_graph("c1ccccc1.c2ccccc2")
    action = BondDelete(*_aromatic_edges(source)[0])
    oracle = resolve_edge_anchored_cycle_open(source, action)
    factored = resolve_component_factored_cycle_open(source, action)
    assert oracle.rejection_code == AromaticCycleOpenRejectionCode.INVALID_SOURCE
    assert factored.rejection_code == AromaticCycleOpenRejectionCode.INVALID_SOURCE


def test_inspected_validation_overflow_source_is_resolved_without_censoring() -> None:
    # This molecule is the one validation source identified by the frozen impact
    # audit as overflowing the whole-molecule 256-structure supplier bound. It is
    # a regression fixture, not new held-out evidence.
    source = pad_molecular_graph(
        smiles_to_molecular_graph(
            "CC(=O)Nc1ccc(Nc2nc3ccc([N+](=O)[O-])cc3nc2Nc2ccc(NC(C)=O)cc2)cc1"
        ),
        40,
    )
    assert persistent_slot_state_sha256(source) == (
        "ff944f22582562fb02ee277b26b3ade5e595e89fed64d128029f2a6e370d9f01"
    )
    components = enumerate_component_factored_kekule_assignments(source)
    assert tuple(len(component.bond_orders) for component in components) == (2, 3, 2)
    assert prod(len(component.bond_orders) for component in components) == 12

    with pytest.raises(AromaticCycleOpenAliasOverflow):
        enumerate_charge_h_preserving_kekule_aliases(source, maximum_aliases=256)
    complete_oracle = enumerate_charge_h_preserving_kekule_aliases(
        source,
        maximum_aliases=512,
    )
    assert complete_oracle.raw_structure_count == 418
    assert len(complete_oracle.aliases) == 12

    edges = _aromatic_edges(source)
    assert len(edges) == 23
    for edge in edges:
        _assert_oracle_equivalent(source, BondDelete(*edge))


def test_low_global_oracle_cap_does_not_censor_component_resolution() -> None:
    source = smiles_to_molecular_graph("c1ccccc1Oc2ccccc2")
    edge = _aromatic_edges(source)[0]
    with pytest.raises(AromaticCycleOpenAliasOverflow):
        resolve_edge_anchored_cycle_open(
            source,
            BondDelete(*edge),
            maximum_aliases=2,
        )

    factored = resolve_component_factored_cycle_open(source, BondDelete(*edge))
    complete_oracle = resolve_edge_anchored_cycle_open(
        source,
        BondDelete(*edge),
        maximum_aliases=16,
    )
    assert factored.admitted and complete_oracle.admitted
    assert factored.enumerated_alias_count == complete_oracle.enumerated_alias_count == 4
    assert factored.canonical_product_keys == complete_oracle.canonical_product_keys
    assert _same_exact_state(factored.successor, complete_oracle.successor)


def test_exhaustive_resolver_rejects_an_enumeration_from_another_source() -> None:
    benzene = smiles_to_molecular_graph("c1ccccc1")
    pyridine = smiles_to_molecular_graph("c1ccncc1")
    enumeration = enumerate_charge_h_preserving_kekule_aliases(
        benzene,
        maximum_aliases=16,
    )
    edge = _aromatic_edges(pyridine)[0]
    with pytest.raises(ValueError, match="another semantic source"):
        resolve_edge_anchored_cycle_open(
            pyridine,
            BondDelete(*edge),
            maximum_aliases=16,
            enumeration=enumeration,
        )


def test_every_alternate_kekule_phase_preserves_factored_oracle_equivalence() -> None:
    source = smiles_to_molecular_graph("c1ccc2ccccc2c1Oc3ccncc3")
    aliases = enumerate_charge_h_preserving_kekule_aliases(
        source,
        maximum_aliases=4096,
    ).aliases
    assert len(aliases) > 2

    products_by_edge: dict[tuple[int, int], set[str]] = {}
    for alias in aliases:
        for edge in _aromatic_edges(alias):
            _assert_oracle_equivalent(alias, BondDelete(*edge))
            resolution = resolve_component_factored_cycle_open(alias, BondDelete(*edge))
            if resolution.admitted:
                products_by_edge.setdefault(edge, set()).add(resolution.canonical_product_keys[0])
    assert products_by_edge
    assert all(len(keys) == 1 for keys in products_by_edge.values())


def test_pyrrole_no_forced_single_is_a_typed_support_exclusion() -> None:
    source = smiles_to_molecular_graph("c1cc[nH]c1")
    resolutions = tuple(
        resolve_component_factored_cycle_open(source, BondDelete(*edge))
        for edge in _aromatic_edges(source)
    )
    assert sum(resolution.admitted for resolution in resolutions) == 3
    rejected = tuple(resolution for resolution in resolutions if not resolution.admitted)
    assert len(rejected) == 2
    assert {resolution.rejection_code for resolution in rejected} == {
        AromaticCycleOpenRejectionCode.NO_FORCED_SINGLE_ALIAS
    }
    for edge in _aromatic_edges(source):
        _assert_oracle_equivalent(source, BondDelete(*edge))


def test_slot_relabeling_preserves_admission_and_molecular_successor() -> None:
    source = pad_molecular_graph(
        smiles_to_molecular_graph("c1ccc2ncccc2c1Oc3ccccc3"),
        24,
    )
    permutation = (
        7,
        2,
        19,
        0,
        4,
        1,
        16,
        3,
        5,
        8,
        15,
        10,
        14,
        11,
        13,
        12,
        6,
        17,
        18,
        9,
        20,
        23,
        22,
        21,
    )
    relabeled = permute_persistent_slots(source, permutation)

    for edge in _aromatic_edges(source):
        action = BondDelete(*edge)
        transported_action = transport_bond_delete(
            action,
            permutation,
            n_slots=source.n_atoms,
        )
        original = resolve_component_factored_cycle_open(source, action)
        transported = resolve_component_factored_cycle_open(relabeled, transported_action)
        _assert_oracle_equivalent(source, action)
        _assert_oracle_equivalent(relabeled, transported_action)
        assert transported.admitted is original.admitted
        assert transported.rejection_code == original.rejection_code
        assert transported.enumerated_alias_count == original.enumerated_alias_count
        assert transported.forced_single_alias_count == original.forced_single_alias_count
        if original.admitted:
            assert canonical_state_key(transported.successor) == canonical_state_key(
                original.successor
            )


@pytest.mark.parametrize(
    "smiles",
    (
        "c1ccccc1",
        "c1ccc2ccccc2c1",
        "c1ccc2ncccc2c1",
        "c1ccccc1Oc2ccccc2",
    ),
)
def test_every_admitted_factored_open_has_the_declared_inverse(smiles: str) -> None:
    source = smiles_to_molecular_graph(smiles)
    admitted = 0
    for edge in _aromatic_edges(source):
        resolution = resolve_component_factored_cycle_open(source, BondDelete(*edge))
        if not resolution.admitted:
            continue
        admitted += 1
        inverse = inverse_mark_for_resolution(resolution)
        assert is_valid_bond_insert(resolution.successor, inverse)
        recovered = apply_bond_insert(resolution.successor, inverse)
        assert canonical_state_key(recovered) == canonical_state_key(source)
    assert admitted > 0


@pytest.mark.parametrize(
    "smiles",
    (
        "c1ccc2ccccc2c1",
        "c1ccccc1Oc2ccccc2",
        "c1ccc2ncccc2c1",
    ),
)
def test_two_step_successor_law_matches_exhaustive_oracle(smiles: str) -> None:
    source = smiles_to_molecular_graph(smiles)
    compared_second_steps = 0
    first_law: Counter[str] = Counter()
    for first_edge in _aromatic_edges(source):
        first_action = BondDelete(*first_edge)
        oracle_first = resolve_edge_anchored_cycle_open(
            source,
            first_action,
            maximum_aliases=4096,
        )
        factored_first = resolve_component_factored_cycle_open(source, first_action)
        _assert_oracle_equivalent(source, first_action)
        if not factored_first.admitted:
            continue
        first_law[factored_first.canonical_product_keys[0]] += 1
        assert _same_exact_state(factored_first.successor, oracle_first.successor)

        intermediate = factored_first.successor
        for left in range(intermediate.n_atoms):
            for right in range(left + 1, intermediate.n_atoms):
                if int(intermediate.bonds[left, right]) == 0:
                    continue
                second_action = BondDelete(left, right)
                oracle_second = resolve_edge_anchored_cycle_open(
                    intermediate,
                    second_action,
                    maximum_aliases=4096,
                )
                factored_second = resolve_component_factored_cycle_open(
                    intermediate,
                    second_action,
                )
                _assert_oracle_equivalent(intermediate, second_action)
                if oracle_second.admitted or factored_second.admitted:
                    compared_second_steps += 1
    assert first_law
    assert compared_second_steps > 0
