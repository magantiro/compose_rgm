from __future__ import annotations

import itertools

import numpy as np

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.ring_system_fiber import (
    build_semantic_ring_system_decoder,
    instantiate_semantic_ring_system_grow,
    ring_atom_electronic_category,
    ring_system_grow_final_atom_states,
    ring_system_placement,
    semantic_ring_categories_for_action,
    semantic_ring_next_category_mask,
    semantic_ring_prefix_is_completable,
)
from compose_v4.rewrite.tracelets import (
    apply_ring_system_grow,
    is_valid_ring_system_grow,
)
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target


def _aromatic_teacher(smiles: str, seed: int, n_slots: int = 16):
    target = pad_molecular_graph(
        smiles_to_molecular_graph(smiles),
        n_slots,
    )
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


def test_semantic_decoder_canonicalizes_unseen_kekule_teacher() -> None:
    state, teacher = _aromatic_teacher("O=c1cccco1", 821)
    decoder = build_semantic_ring_system_decoder(
        state,
        ring_system_placement(teacher),
    )
    categories = semantic_ring_categories_for_action(decoder, teacher)
    canonical = instantiate_semantic_ring_system_grow(
        state,
        decoder,
        categories,
    )

    assert semantic_ring_prefix_is_completable(decoder, categories)
    assert is_valid_ring_system_grow(state, canonical)
    assert canonical_state_key(apply_ring_system_grow(state, canonical)) == (
        canonical_state_key(apply_ring_system_grow(state, teacher))
    )


def test_semantic_decoder_rejects_matching_feasible_non_huckel_labels() -> None:
    state, teacher = _aromatic_teacher("c1ccccc1", 811)
    decoder = build_semantic_ring_system_decoder(
        state,
        ring_system_placement(teacher),
    )
    donor_categories = []
    for options in decoder.options_by_member:
        donor = next(
            option
            for option in options
            if option.aromatic_demand == 0 and option.pi_electrons == 2
        )
        donor_categories.append(ring_atom_electronic_category(donor))

    # Six donor sites contribute twelve pi electrons. The empty matching exists,
    # but the semantic molecule is not aromatic under the 4k+2 condition.
    assert not semantic_ring_prefix_is_completable(
        decoder,
        tuple(donor_categories),
    )


def test_semantic_prefix_masks_define_one_normalized_finite_language() -> None:
    state, teacher = _aromatic_teacher("c1cc[nH]c1", 701)
    decoder = build_semantic_ring_system_decoder(
        state,
        ring_system_placement(teacher),
    )
    complete = []

    def visit(prefix: tuple[int, ...]) -> None:
        if len(prefix) == decoder.span:
            complete.append(prefix)
            return
        mask = semantic_ring_next_category_mask(decoder, prefix)
        for category, enabled in enumerate(mask):
            if enabled:
                visit((*prefix, category))

    visit(())
    assert complete
    assert all(semantic_ring_prefix_is_completable(decoder, item) for item in complete)
    assert semantic_ring_categories_for_action(decoder, teacher) in complete
    for categories in complete:
        action = instantiate_semantic_ring_system_grow(
            state,
            decoder,
            categories,
        )
        assert is_valid_ring_system_grow(state, action)


def test_semantic_key_retains_final_hydrogen_role() -> None:
    state, teacher = _aromatic_teacher("c1cc[nH]c1", 702)
    final_states = ring_system_grow_final_atom_states(teacher)
    assert any(
        atom_type == 3 and hydrogens == 1
        for _, atom_type, _, hydrogens in final_states
    )
