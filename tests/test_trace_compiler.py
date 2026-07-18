from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import is_valid_state, pad_molecular_graph
from compose_v4.rewrite.compiler import compile_null_to_target, compile_source_to_target
from compose_v4.rewrite.kernel import canonical_state_key, default_rewrite_system
from compose_v4.rewrite.trace import execute_trace, invert_trace


MOLECULES = (
    "CC(=O)NCCO",          # acyclic, heteroatoms, carbonyl
    "N#CCO",               # triple bond
    "C[N+](C)(C)C",        # formal charge
    "c1ccccc1",            # monocycle
    "c1ccc2ccccc2c1",      # fused aromatic system
    "C1CC2CCC1C2",         # bridged bicyclic system
    "C1CCC2(CC1)CCCC2",    # spiro system
)


def _same_arrays(left, right) -> bool:
    return bool(
        np.array_equal(left.atom_types, right.atom_types)
        and np.array_equal(left.formal_charges, right.formal_charges)
        and np.array_equal(left.implicit_h_counts, right.implicit_h_counts)
        and np.array_equal(left.bonds, right.bonds)
    )


@pytest.mark.parametrize("smiles", MOLECULES)
def test_compiler_reconstructs_exact_target_and_exact_inverse(smiles: str) -> None:
    target = smiles_to_molecular_graph(smiles)
    target = pad_molecular_graph(target, target.n_atoms + 4)
    runtime = default_rewrite_system()

    trace = compile_null_to_target(target, system=runtime)
    reconstructed, states = execute_trace(
        trace.source, trace.steps, system=runtime, return_states=True
    )
    reverse = invert_trace(trace.source, trace.steps, system=runtime)
    restored_source = execute_trace(reconstructed, reverse, system=runtime)

    assert _same_arrays(reconstructed, target)
    assert _same_arrays(restored_source, trace.source)
    assert all(is_valid_state(state) for state in states)


def test_compiler_is_invariant_to_padding_slot_permutation() -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph("c1ccc2ccccc2c1"), 16)
    permutation = np.array([7, 3, 12, 0, 10, 5, 1, 14, 8, 2, 4, 6, 9, 11, 13, 15])
    permuted = type(target)(
        atom_types=target.atom_types[permutation],
        formal_charges=target.formal_charges[permutation],
        implicit_h_counts=target.implicit_h_counts[permutation],
        bonds=target.bonds[np.ix_(permutation, permutation)],
    )

    original_trace = compile_null_to_target(target)
    permuted_trace = compile_null_to_target(permuted)
    original_out = execute_trace(original_trace.source, original_trace.steps)
    permuted_out = execute_trace(permuted_trace.source, permuted_trace.steps)

    assert canonical_state_key(original_out) == canonical_state_key(permuted_out)


def test_randomized_compiler_has_trace_diversity_without_endpoint_drift() -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph("c1ccc2ccccc2c1"), 16)
    programs = set()
    for seed in range(32):
        trace = compile_null_to_target(target, rng=np.random.default_rng(seed))
        programs.add(tuple((step.rule_name, repr(step.action)) for step in trace.steps))
        assert _same_arrays(execute_trace(trace.source, trace.steps), target)
    assert len(programs) >= 8


@pytest.mark.parametrize(
    "smiles",
    ("CC(=O)NCCO", "N#CCO", "c1ccccc1", "c1ccc2ccccc2c1"),
)
def test_deferred_bond_order_compiler_is_exact_and_pathwise_valid(smiles: str) -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 16)
    trace = compile_null_to_target(
        target,
        rng=np.random.default_rng(73),
        defer_bond_orders=True,
    )
    endpoint, states = execute_trace(trace.source, trace.steps, return_states=True)
    assert _same_arrays(endpoint, target)
    assert all(is_valid_state(state) for state in states)
    assert trace.metadata["defer_bond_orders"] is True
    assert sum(step.rule_name == "bond_reorder" for step in trace.steps) == int(
        np.triu(target.bonds > 1, k=1).sum()
    )


def test_arbitrary_source_target_bridge_is_exact_and_valid() -> None:
    source = smiles_to_molecular_graph("CC(=O)NCCO")
    target = smiles_to_molecular_graph("c1ncc2ccccc2n1")
    trace = compile_source_to_target(
        source,
        target,
        rng=np.random.default_rng(20260714),
    )
    endpoint, states = execute_trace(trace.source, trace.steps, return_states=True)
    assert _same_arrays(endpoint, trace.target)
    assert all(is_valid_state(state) for state in states)
    assert any(state.n_real_atoms == 0 for state in states)
