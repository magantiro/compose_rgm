"""Production-boundary tests for the explicit semantic cycle-close action."""

from __future__ import annotations

import inspect

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.experiments.cycle_open_kekule_invariance import (
    build_alternate_kekule_pair,
)
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    de_novo_rewrite_system,
    editing_v2_semantic_cycle_rewrite_system,
)
from compose_v4.rewrite import operators as operator_module
from compose_v4.rewrite import semantic_cycle_close, semantic_cycle_open
from compose_v4.rewrite.aromatic_kekule import (
    enumerate_component_factored_kekule_assignments,
    instantiate_component_factored_kekule_alias,
)
from compose_v4.rewrite.operators import (
    BondInsert,
    CycleCloseEdge,
    CycleOpenEdge,
    apply_cycle_close_edge,
    apply_cycle_open_edge,
    enumerate_cycle_close_edges,
    enumerate_cycle_open_edges,
    inverse_cycle_close_edge,
)


def _two_ring_source_with_nonminimal_remote_phase():
    source = pad_molecular_graph(
        smiles_to_molecular_graph("c1ccccc1-c1ccccc1"),
        20,
    )
    components = enumerate_component_factored_kekule_assignments(source)
    assert len(components) == 2
    source_orders = [
        tuple(int(source.bonds[edge]) for edge in component.edges)
        for component in components
    ]
    remote_index = 1
    alternatives = tuple(
        orders
        for orders in components[remote_index].bond_orders
        if orders != source_orders[remote_index]
    )
    assert alternatives
    source_orders[remote_index] = max(alternatives)
    phased = instantiate_component_factored_kekule_alias(
        source,
        components,
        tuple(source_orders),
    )
    return phased, components[0], components[remote_index]


def test_ambiguous_aromatic_closure_is_excluded_from_the_semantic_fiber() -> None:
    source = pad_molecular_graph(
        smiles_to_molecular_graph("Cc1cccc(Cl)c1"),
        16,
    )
    assert CycleCloseEdge(0, 4, 1) not in enumerate_cycle_close_edges(source)
    with pytest.raises(InvalidRewrite, match="invalid cycle_close"):
        editing_v2_semantic_cycle_rewrite_system().apply(
            source,
            "cycle_close",
            CycleCloseEdge(0, 4, 1),
        )


def test_retained_aromatic_closure_is_kekule_invariant() -> None:
    pair = build_alternate_kekule_pair(
        pad_molecular_graph(smiles_to_molecular_graph("Cc1cccc(Cl)c1"), 16)
    )
    action = CycleCloseEdge(0, 3, 1)
    runtime = editing_v2_semantic_cycle_rewrite_system()
    original = runtime.apply(pair.original, "cycle_close", action)
    alternate = runtime.apply(pair.alternate, "cycle_close", action)
    assert canonical_state_key(original) == canonical_state_key(alternate)


def test_enumerator_equals_runtime_admission_and_every_mark_executes() -> None:
    source = smiles_to_molecular_graph("CCCCCC")
    runtime = editing_v2_semantic_cycle_rewrite_system()
    enumerated = set(enumerate_cycle_close_edges(source))
    candidates = {
        CycleCloseEdge(left, right, order)
        for left in range(source.n_atoms)
        for right in range(left + 1, source.n_atoms)
        for order in (1, 2, 3)
    }
    admitted = set()
    for action in candidates:
        try:
            successor = runtime.apply(source, "cycle_close", action)
        except InvalidRewrite:
            continue
        admitted.add(action)
        assert canonical_state_key(successor)
    assert enumerated == admitted


def test_semantic_close_and_open_are_canonical_inverses() -> None:
    source = smiles_to_molecular_graph("CCCCCC")
    action = CycleCloseEdge(0, 5, 1)
    runtime = editing_v2_semantic_cycle_rewrite_system()
    closed = runtime.apply(source, "cycle_close", action)
    inverse = inverse_cycle_close_edge(source, action)
    assert inverse == CycleOpenEdge(0, 5)
    restored = runtime.apply(closed, "cycle_open", inverse)
    assert canonical_state_key(restored) == canonical_state_key(source)


def test_semantic_action_is_not_interchangeable_with_raw_bond_insert() -> None:
    source = smiles_to_molecular_graph("CCCCCC")
    action = CycleCloseEdge(0, 5, 1)
    runtime = editing_v2_semantic_cycle_rewrite_system()
    with pytest.raises(InvalidRewrite, match="expects CycleCloseEdge"):
        runtime.apply(source, "cycle_close", BondInsert(0, 5, 1))
    with pytest.raises(InvalidRewrite, match="unknown rewrite rule: bond_insert"):
        runtime.apply(source, "bond_insert", action)


def test_endpoint_order_and_bond_order_are_part_of_the_contract() -> None:
    source = smiles_to_molecular_graph("CCCCCC")
    runtime = editing_v2_semantic_cycle_rewrite_system()
    with pytest.raises(InvalidRewrite, match="invalid cycle_close"):
        runtime.apply(source, "cycle_close", CycleCloseEdge(5, 0, 1))
    with pytest.raises(InvalidRewrite, match="invalid cycle_close"):
        runtime.apply(source, "cycle_close", CycleCloseEdge(0, 5, 4))


def test_legacy_and_de_novo_runtimes_do_not_gain_semantic_close() -> None:
    source = smiles_to_molecular_graph("CCCCCC")
    with pytest.raises(InvalidRewrite, match="unknown rewrite rule"):
        de_novo_rewrite_system().apply(
            source,
            "cycle_close",
            CycleCloseEdge(0, 5, 1),
        )


def test_production_semantic_cycle_runtime_has_no_experiment_imports() -> None:
    for module in (operator_module, semantic_cycle_close, semantic_cycle_open):
        assert "compose_v4.experiments" not in inspect.getsource(module)


def test_local_cycle_edits_preserve_unaffected_component_exact_phase() -> None:
    source, local_component, remote_component = (
        _two_ring_source_with_nonminimal_remote_phase()
    )
    local_vertices = {vertex for edge in local_component.edges for vertex in edge}
    remote_before = tuple(int(source.bonds[edge]) for edge in remote_component.edges)

    open_action = next(
        action
        for action in enumerate_cycle_open_edges(source)
        if {int(action.a), int(action.b)} <= local_vertices
    )
    opened = apply_cycle_open_edge(source, open_action)
    assert (
        tuple(int(opened.bonds[edge]) for edge in remote_component.edges)
        == remote_before
    )

    close_action = next(
        action
        for action in enumerate_cycle_close_edges(source)
        if {int(action.a), int(action.b)} <= local_vertices
    )
    closed = apply_cycle_close_edge(source, close_action)
    assert (
        tuple(int(closed.bonds[edge]) for edge in remote_component.edges)
        == remote_before
    )


@pytest.mark.parametrize("smiles", ("[nH+]1ccccc1", "C1CC[NH2+]CC1"))
def test_every_admitted_charged_cycle_successor_preserves_policy(smiles: str) -> None:
    source = pad_molecular_graph(smiles_to_molecular_graph(smiles), 20)
    open_actions = enumerate_cycle_open_edges(source)
    close_actions = enumerate_cycle_close_edges(source)
    assert open_actions
    assert close_actions
    assert all(
        charge_policy_preserved(source, apply_cycle_open_edge(source, action))
        for action in open_actions
    )
    assert all(
        charge_policy_preserved(source, apply_cycle_close_edge(source, action))
        for action in close_actions
    )
