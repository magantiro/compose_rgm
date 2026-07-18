from __future__ import annotations

from collections import Counter

import numpy as np

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import is_connected_or_null, is_valid_state, pad_molecular_graph
from compose_v4.rewrite.compiler import compile_null_to_target
from compose_v4.rewrite.fiber import enumerate_action_fiber
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace import execute_trace


def _padded(smiles: str, n_slots: int = 6):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), n_slots)


def test_every_enumerated_successor_is_valid_connected_and_nonidentity() -> None:
    state = _padded("CCC")
    source_key = canonical_state_key(state)
    transitions = enumerate_action_fiber(state)
    assert transitions
    assert all(is_valid_state(item.successor) for item in transitions)
    assert all(is_connected_or_null(item.successor) for item in transitions)
    assert all(item.successor_key != source_key for item in transitions)


def test_connected_fiber_rejects_fragmenting_bridge_deletion() -> None:
    chain_rules = {item.rule_name for item in enumerate_action_fiber(_padded("CCC"))}
    ring_rules = {item.rule_name for item in enumerate_action_fiber(_padded("C1CC1"))}
    assert "bond_delete" not in chain_rules
    assert "bond_delete" in ring_rules


def test_fiber_exercises_all_six_micro_rule_families() -> None:
    observed = set()
    for smiles in ("C", "CC", "CCC", "C1CC1"):
        observed.update(item.rule_name for item in enumerate_action_fiber(_padded(smiles)))
    assert observed == {
        "atom_insert",
        "atom_delete",
        "atom_restate",
        "bond_insert",
        "bond_delete",
        "bond_reorder",
    }


def test_fiber_successor_multiset_is_slot_permutation_invariant() -> None:
    state = _padded("CCO")
    permutation = np.asarray([3, 0, 5, 2, 1, 4])
    permuted = type(state)(
        atom_types=state.atom_types[permutation],
        formal_charges=state.formal_charges[permutation],
        implicit_h_counts=state.implicit_h_counts[permutation],
        bonds=state.bonds[np.ix_(permutation, permutation)],
    )
    original_keys = Counter(item.successor_key for item in enumerate_action_fiber(state))
    permuted_keys = Counter(item.successor_key for item in enumerate_action_fiber(permuted))
    assert original_keys == permuted_keys


def test_every_cnof_teacher_successor_is_in_model_fiber() -> None:
    target = _padded("C1COC1")
    trace = compile_null_to_target(target, rng=np.random.default_rng(11))
    _, states = execute_trace(trace.source, trace.steps, return_states=True)
    for progress, teacher_successor in enumerate(states[1:]):
        teacher_key = canonical_state_key(teacher_successor)
        keys = {item.successor_key for item in enumerate_action_fiber(states[progress])}
        assert teacher_key in keys
