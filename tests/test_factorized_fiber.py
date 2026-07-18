from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.compiler import compile_null_to_target
from compose_v4.rewrite.factorized_fiber import enumerate_factorized_cnof_fiber
from compose_v4.rewrite.fiber import ActionFiberSpec, enumerate_action_fiber
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace import execute_trace


MOLECULES = (
    "C",
    "CO",
    "C=C",
    "C#N",
    "CCO",
    "C1CC1",
    "C1=COC1",
    "c1ccncc1",
)


def _successor_set(transitions):
    return {(item.rule_name, item.successor_key) for item in transitions}


@pytest.mark.parametrize("smiles", MOLECULES)
def test_factorized_fiber_matches_exhaustive_chemical_successors(smiles: str) -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 9)
    trace = compile_null_to_target(target, rng=np.random.default_rng(17))
    _, states = execute_trace(trace.source, trace.steps, return_states=True)
    for state in states:
        exhaustive = enumerate_action_fiber(state)
        factorized = enumerate_factorized_cnof_fiber(state).transitions
        assert _successor_set(factorized) == _successor_set(exhaustive)


def test_factorization_removes_padding_slot_aliases() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("CCO"), 12)
    exhaustive = enumerate_action_fiber(state)
    factorized = enumerate_factorized_cnof_fiber(state).transitions
    assert _successor_set(factorized) == _successor_set(exhaustive)
    assert len(factorized) < len(exhaustive) / 3


def test_factorized_fiber_rejects_out_of_scope_chemistry() -> None:
    charged = pad_molecular_graph(smiles_to_molecular_graph("C[N+](C)(C)C"), 8)
    with pytest.raises(ValueError):
        enumerate_factorized_cnof_fiber(charged)


def test_transport_fiber_matches_exhaustive_reroute_successors() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("CC(C)C"), 7)
    language = ActionFiberSpec.neutral_cnof()
    language = ActionFiberSpec(
        atom_states=language.atom_states,
        allow_atom_insert=False,
        allow_atom_delete=False,
        allow_atom_restate=False,
        allow_bond_insert=False,
        allow_bond_delete=False,
        allow_bond_reorder=False,
        allow_bond_reroute=True,
    )
    exhaustive = tuple(
        item
        for item in enumerate_action_fiber(state, spec=language)
        if item.rule_name == "bond_reroute"
    )
    factorized = enumerate_factorized_cnof_fiber(
        state,
        allow_bond_reroute=True,
    ).by_family["bond_reroute"]
    assert _successor_set(factorized) == _successor_set(exhaustive)
    assert factorized


def test_transport_fiber_removes_graft_identity_transitions() -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph("CCCCCCCC"), 12)
    transitions = enumerate_factorized_cnof_fiber(
        state,
        allow_bond_reroute=True,
    ).by_family["bond_reroute"]
    current_key = canonical_state_key(state)
    assert transitions
    assert all(item.successor_key != current_key for item in transitions)
