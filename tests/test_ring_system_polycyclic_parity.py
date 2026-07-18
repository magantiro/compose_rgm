from __future__ import annotations

import numpy as np

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.ring_system_fiber import (
    build_semantic_ring_system_decoder,
    instantiate_semantic_ring_system_grow,
    ring_system_placement,
    semantic_ring_categories_for_action,
    semantic_ring_prefix_is_completable,
)
from compose_v4.rewrite.tracelets import (
    apply_ring_system_grow,
    is_valid_ring_system_grow,
)
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target


def _aromatic_teacher(smiles: str, seed: int, n_slots: int):
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), n_slots)
    source = DegreeBoundedCarbonTreePrior(
        sizes=(target.n_real_atoms,),
    ).sample(np.random.default_rng(seed), n_slots=n_slots)
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    path = TraceProgressCTMC(trace)
    progress = next(
        index
        for index, step in enumerate(trace.steps)
        if step.rule_name == "ring_system_grow"
    )
    return path.state_at(progress), trace.steps[progress].action


def test_polycyclic_aromatic_skips_unsound_global_huckel_parity() -> None:
    # Pyrene has one connected 16-atom aromatic edge graph with cycle rank four.
    # A global 4n+2 predicate rejects its executable teacher even though RDKit
    # accepts the complete polycyclic aromatic successor.
    state, teacher = _aromatic_teacher(
        "c1cc2ccc3cccc4ccc(c1)c2c34",
        991,
        n_slots=24,
    )
    decoder = build_semantic_ring_system_decoder(
        state,
        ring_system_placement(teacher),
    )
    categories = semantic_ring_categories_for_action(decoder, teacher)

    assert len(decoder.aromatic_components) == 1
    component = decoder.aromatic_components[0]
    edge_count = sum(mask.bit_count() for mask in component.adjacency_masks) // 2
    assert edge_count > len(component.positions)
    assert not component.enforce_huckel_parity
    assert is_valid_ring_system_grow(state, teacher)
    assert semantic_ring_prefix_is_completable(decoder, categories)

    canonical = instantiate_semantic_ring_system_grow(state, decoder, categories)
    assert is_valid_ring_system_grow(state, canonical)
    assert canonical_state_key(apply_ring_system_grow(state, canonical)) == (
        canonical_state_key(apply_ring_system_grow(state, teacher))
    )
