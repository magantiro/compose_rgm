from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import is_connected_or_null, is_valid_state, pad_molecular_graph
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace import execute_trace, invert_trace
from compose_v4.rewrite.ring_system_fiber import enumerate_ring_system_grows
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog


@pytest.mark.parametrize(
    "smiles",
    (
        "CC(C)CO",
        "C1CCCCC1",
        "c1ccncc1",
        "C1CCC2CCCCC2C1",
        "C1CC2CCC1C2",
        "C1CCC2(CC1)CCCC2",
    ),
)
def test_random_carbon_tree_transport_reaches_exact_target(smiles: str) -> None:
    target_raw = smiles_to_molecular_graph(smiles)
    target = pad_molecular_graph(target_raw, target_raw.n_real_atoms + 3)
    source = DegreeBoundedCarbonTreePrior(
        sizes=(target.n_real_atoms,),
    ).sample(np.random.default_rng(31), n_slots=target.n_atoms)
    trace = compile_carbon_tree_to_target(source, target)
    endpoint, states = execute_trace(trace.source, trace.steps, return_states=True)

    assert all(is_valid_state(state) for state in states)
    assert all(is_connected_or_null(state) for state in states)
    assert np.array_equal(endpoint.atom_types, target.atom_types)
    assert np.array_equal(endpoint.formal_charges, target.formal_charges)
    assert np.array_equal(endpoint.implicit_h_counts, target.implicit_h_counts)
    assert np.array_equal(endpoint.bonds, target.bonds)
    assert trace.metadata["ring_commitment_status"] == "topology_committed_tracelets"

    inverse = invert_trace(trace.source, trace.steps)
    recovered = execute_trace(endpoint, inverse)
    assert np.array_equal(recovered.atom_types, source.atom_types)
    assert np.array_equal(recovered.bonds, source.bonds)


def test_primitive_tree_transport_can_change_source_size() -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph("c1ccncc1O"), 12)
    for source_size in (3, 10):
        source = DegreeBoundedCarbonTreePrior(sizes=(source_size,)).sample(
            np.random.default_rng(100 + source_size),
            n_slots=12,
        )
        trace = compile_carbon_tree_to_target(source, target)
        endpoint, states = execute_trace(trace.source, trace.steps, return_states=True)
        assert all(is_valid_state(state) for state in states)
        assert all(is_connected_or_null(state) for state in states)
        assert np.array_equal(endpoint.atom_types, target.atom_types)
        assert np.array_equal(endpoint.bonds, target.bonds)
        assert (
            trace.metadata["transport_strategy"]
            == "primitive_leaf_delete_tracelet_regrow"
        )
        assert trace.metadata["atom_delete_steps"] == source_size - 1
        assert trace.metadata["bond_reroute_steps"] == 0


@pytest.mark.parametrize("source_size", (3, 10))
def test_flexible_graft_transport_grows_or_shrinks_to_exact_target(
    source_size: int,
) -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph("c1ccncc1O"), 12)
    source = DegreeBoundedCarbonTreePrior(sizes=(source_size,)).sample(
        np.random.default_rng(900 + source_size),
        n_slots=12,
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        flexible_size=True,
    )
    endpoint, states = execute_trace(trace.source, trace.steps, return_states=True)

    assert all(is_valid_state(state) for state in states)
    assert all(is_connected_or_null(state) for state in states)
    assert np.array_equal(endpoint.atom_types, target.atom_types)
    assert np.array_equal(endpoint.formal_charges, target.formal_charges)
    assert np.array_equal(endpoint.implicit_h_counts, target.implicit_h_counts)
    assert np.array_equal(endpoint.bonds, target.bonds)
    assert trace.metadata["compiler"] == "flexible_size_graft_transport_v1"
    if source_size < target.n_real_atoms:
        assert trace.metadata["atom_insert_steps"] == target.n_real_atoms - source_size
        assert trace.metadata["atom_delete_steps"] == 0
    else:
        assert trace.metadata["atom_delete_steps"] == source_size - target.n_real_atoms
        assert trace.metadata["atom_insert_steps"] == 0
        assert trace.metadata["resize_strategy"] == "direct_leaf_shrink"
    assert trace.metadata["compiler_revision"] == "quotient_common_topology_v2"
    assert trace.metadata["graft_planner"] == "quotient_common_topology"


def test_bond_reroute_is_an_optional_path_compression_ablation() -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph("CC(C)CO"), 9)
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(71),
        n_slots=9,
    )
    primitive = compile_carbon_tree_to_target(source, target)
    compressed = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
    )
    assert primitive.metadata["bond_reroute_steps"] == 0
    assert compressed.metadata["atom_delete_steps"] == 0
    assert compressed.metadata["atom_insert_steps"] == 0
    assert len(compressed.steps) <= len(primitive.steps)


@pytest.mark.parametrize("source_size", (4, 11, 13))
def test_quotient_graft_teacher_has_no_gauge_only_or_repeated_grafts(
    source_size: int,
) -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph("C1CCC2CCCCC2C1O"), 14)
    source = DegreeBoundedCarbonTreePrior(sizes=(source_size,)).sample(
        np.random.default_rng(4100 + source_size),
        n_slots=target.n_atoms,
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        flexible_size=True,
    )
    endpoint, states = execute_trace(trace.source, trace.steps, return_states=True)

    assert canonical_state_key(trace.source) == canonical_state_key(source)
    assert np.array_equal(endpoint.atom_types, target.atom_types)
    assert np.array_equal(endpoint.bonds, target.bonds)
    graft_successor_keys = []
    for index, step in enumerate(trace.steps):
        if step.rule_name != "bond_reroute":
            continue
        before = canonical_state_key(states[index])
        after = canonical_state_key(states[index + 1])
        assert before != after
        graft_successor_keys.append(after)
    assert len(graft_successor_keys) == len(set(graft_successor_keys))


@pytest.mark.parametrize(
    "smiles",
    (
        "CC(C)CO",
        "C1CCCCC1",
        "c1ccncc1",
        "C1CCC2CCCCC2C1",
        "C1CC2CCC1C2",
        "C1CCC2(CC1)CCCC2",
    ),
)
def test_size_matched_graft_transport_retains_atoms_and_reaches_target(
    smiles: str,
) -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 16)
    sources = tuple(
        DegreeBoundedCarbonTreePrior(
            sizes=(target.n_real_atoms,),
        ).sample(np.random.default_rng(seed + 700), n_slots=target.n_atoms)
        for seed in range(8)
    )
    proposals = tuple(
        compile_carbon_tree_to_target(
            source,
            target,
            use_bond_reroute=True,
            align_source=True,
        )
        for source in sources
    )
    catalog = build_typed_ring_catalog(proposals)
    for source in sources:
        trace = compile_carbon_tree_to_target(
            source,
            target,
            use_bond_reroute=True,
            align_source=True,
            ring_catalog=catalog,
        )
        endpoint, states = execute_trace(
            trace.source,
            trace.steps,
            return_states=True,
        )

        assert all(is_valid_state(state) for state in states)
        assert all(is_connected_or_null(state) for state in states)
        assert np.array_equal(endpoint.atom_types, target.atom_types)
        assert np.array_equal(endpoint.formal_charges, target.formal_charges)
        assert np.array_equal(endpoint.implicit_h_counts, target.implicit_h_counts)
        assert np.array_equal(endpoint.bonds, target.bonds)
        assert not {
            "atom_insert",
            "atom_delete",
        } & {step.rule_name for step in trace.steps}
        assert all(
            step.action.u in (step.action.a, step.action.b)
            for step in trace.steps
            if step.rule_name == "bond_reroute"
        )
        assert catalog.supports_trace(trace)
        assert (
            trace.metadata["transport_strategy"]
            == "graft_then_atomic_ring_system_grow"
        )
        assert trace.metadata["zero_span_ring_closure_steps"] == 0
        assert not any(step.rule_name == "ring_ear_insert" for step in trace.steps)


def test_typed_tree_transport_teacher_is_in_topology_committed_fiber() -> None:
    targets = tuple(
        pad_molecular_graph(smiles_to_molecular_graph(smiles), 14)
        for smiles in ("c1ccncc1", "C1CC2CCC1C2", "CCOC")
    )
    sources = tuple(
        DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
            np.random.default_rng(200 + index),
            n_slots=target.n_atoms,
        )
        for index, target in enumerate(targets)
    )
    proposal_traces = tuple(
        compile_carbon_tree_to_target(
            source,
            target,
            use_bond_reroute=True,
            align_source=True,
            typed_ring_payloads=True,
        )
        for source, target in zip(sources, targets)
    )
    catalog = build_typed_ring_catalog(proposal_traces)
    for source, target in zip(sources, targets):
        trace = compile_carbon_tree_to_target(
            source,
            target,
            use_bond_reroute=True,
            align_source=True,
            typed_ring_payloads=True,
            ring_catalog=catalog,
        )
        state = trace.source
        for step in trace.steps:
            successor = execute_trace(state, (step,))
            if step.rule_name == "ring_system_grow":
                assert step.action in enumerate_ring_system_grows(state, catalog)
            state = successor
