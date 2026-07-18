from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    ELEMENT_TO_IDX,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.operators import AtomInsert
from compose_v4.rewrite.trace import execute_trace, invert_trace
from compose_v4.rewrite.tracelets import (
    RingBond,
    RingSystemGrow,
    apply_ring_system_delete,
    apply_ring_system_grow,
    inverse_ring_system_grow,
)
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target


def _same_state(left, right) -> bool:
    return all(
        np.array_equal(getattr(left, field), getattr(right, field))
        for field in ("atom_types", "formal_charges", "implicit_h_counts", "bonds")
    )


def test_ring_system_grow_can_allocate_a_complete_new_ring_and_delete_it() -> None:
    source = pad_molecular_graph(smiles_to_molecular_graph("C"), 10)
    carbon = int(ELEMENT_TO_IDX["C"])
    insertions = tuple(
        AtomInsert(
            slot=slot,
            atom_type=carbon,
            formal_charge=0,
            implicit_h_count=3,
            neighbors=((0 if slot == 1 else slot - 1, 1),),
        )
        for slot in range(1, 7)
    )
    action = RingSystemGrow(
        system_atoms=tuple(range(1, 7)),
        interface_atoms=(),
        scaffold_bonds=(),
        bond_reorders=(),
        atom_payloads=(),
        atom_insertions=insertions,
        bond_insertions=(RingBond(6, 1, 1),),
        source_aromatic_edges=(),
        aromatic_edges=(),
        topology_class="single_ring",
    )

    product = apply_ring_system_grow(source, action)
    inverse = inverse_ring_system_grow(source, action)
    restored = apply_ring_system_delete(product, inverse)

    assert molecular_graph_to_smiles(product) == "CC1CCCCC1"
    assert _same_state(restored, source)
    assert inverse.atom_deletions == (6, 5, 4, 3, 2, 1)


@pytest.mark.parametrize(
    "smiles",
    (
        "C1CCCCC1",
        "c1ccncc1",
        "C1CCC2CCCCC2C1",
        "C1CC2CCC1C2",
        "C1CCC2(CC1)CCCC2",
    ),
)
def test_tree_transport_commits_one_full_system_and_has_exact_inverse(
    smiles: str,
) -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 20)
    source = DegreeBoundedCarbonTreePrior(
        sizes=(target.n_real_atoms,),
    ).sample(np.random.default_rng(91), n_slots=20)
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    ring_steps = tuple(
        step for step in trace.steps if step.rule_name == "ring_system_grow"
    )
    inverse = invert_trace(trace.source, trace.steps)
    restored = execute_trace(target, inverse)

    assert len(ring_steps) == 1
    assert not any(step.rule_name == "ring_ear_insert" for step in trace.steps)
    assert not any(step.rule_name == "bond_insert" for step in trace.steps)
    assert _same_state(restored, trace.source)


def test_aromatic_system_is_semantic_class_four_but_executes_in_kekule_form() -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph("c1ncccc1"), 12)
    source = DegreeBoundedCarbonTreePrior(
        sizes=(target.n_real_atoms,),
    ).sample(np.random.default_rng(19), n_slots=12)
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    ring_action = next(
        step.action for step in trace.steps if step.rule_name == "ring_system_grow"
    )
    endpoint = execute_trace(trace.source, trace.steps)
    perceived = resonance_invariant_bond_classes(endpoint)

    assert ring_action.aromatic_edges
    assert all(
        int(perceived[a, b]) == BOND_AROMATIC
        for a, b in ring_action.aromatic_edges
    )
    assert not np.any(endpoint.bonds == BOND_AROMATIC)


@pytest.mark.parametrize("source_delta", (-3, 3))
def test_variable_size_transport_uses_full_ring_system_grow(source_delta: int) -> None:
    target = pad_molecular_graph(
        smiles_to_molecular_graph("CCOc1ccc2ncc(C)cc2c1"),
        40,
    )
    source = DegreeBoundedCarbonTreePrior(
        sizes=(target.n_real_atoms + source_delta,),
    ).sample(np.random.default_rng(101 + source_delta), n_slots=40)
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        flexible_size=True,
    )
    endpoint = execute_trace(trace.source, trace.steps)

    assert _same_state(endpoint, target)
    assert any(step.rule_name == "ring_system_grow" for step in trace.steps)
    assert not any(step.rule_name == "ring_ear_insert" for step in trace.steps)
    assert any(
        step.rule_name == ("atom_insert" if source_delta < 0 else "atom_delete")
        for step in trace.steps
    )
