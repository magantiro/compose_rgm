"""Scaffold protection restricts the existing ring law, not the ring vocabulary."""

from itertools import product

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import empty_molecular_graph, pad_molecular_graph
from compose_v4.rewrite.kernel import InvalidRewrite, de_novo_rewrite_system
from compose_v4.rewrite.ring_system_fiber import (
    RingSystemPlacement,
    build_semantic_ring_system_decoder,
    instantiate_semantic_ring_system_grow,
    ring_atom_electronic_category,
    semantic_ring_next_category_mask,
    semantic_ring_prefix_is_completable,
)
from compose_v4.rewrite.scaffold_construction import ScaffoldContext
from compose_v4.rewrite.tracelets import RingBond


def graph(smiles):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 8)


def placement(state):
    return RingSystemPlacement(
        system_atoms=(0, 1, 2),
        scaffold_bonds=(RingBond(0, 1, int(state.bonds[0, 1])),
                        RingBond(1, 2, int(state.bonds[1, 2]))),
        bond_reorders=(), bond_insertions=(RingBond(0, 2, 1),),
        aromatic_edges=(), topology_class="monocyclic",
    )


def language(decoder):
    complete = set()

    def visit(prefix):
        if len(prefix) == decoder.span:
            complete.add(prefix)
            return
        for category, enabled in enumerate(semantic_ring_next_category_mask(decoder, prefix)):
            if enabled:
                visit((*prefix, category))

    visit(())
    return complete


@pytest.mark.parametrize("source,current", [("C", "CCC"), ("N", "NCC"), ("O", "OCC")])
def test_conditional_language_equals_executor_filtered_base_language(source, current):
    state = graph(current)
    context = ScaffoldContext.from_source(graph(source), (0,))
    base = build_semantic_ring_system_decoder(state, placement(state))
    conditioned = build_semantic_ring_system_decoder(
        state, placement(state), scaffold_context=context)
    runtime = context.rewrite_system()
    expected = set()
    # Independent small oracle: enumerate the UNCONDITIONED label combinations,
    # execute each legal base action, then enforce the supplied-core invariant.
    categories = tuple(tuple(ring_atom_electronic_category(o) for o in options)
                       for options in base.options_by_member)
    for labels in product(*categories):
        if not semantic_ring_prefix_is_completable(base, labels):
            continue
        action = instantiate_semantic_ring_system_grow(state, base, labels)
        try:
            runtime.apply(state, "ring_system_grow", action)
        except InvalidRewrite:
            continue
        expected.add(labels)
    assert expected
    assert expected < language(base)
    assert language(conditioned) == expected
    for labels in expected:
        action = instantiate_semantic_ring_system_grow(state, conditioned, labels)
        assert action == instantiate_semantic_ring_system_grow(state, base, labels)
        assert context.accepts(runtime.apply(state, "ring_system_grow", action))


def test_empty_context_does_not_restrict_ring_choices_or_change_actions():
    state = graph("CCC")
    base = build_semantic_ring_system_decoder(state, placement(state))
    empty = ScaffoldContext.from_source(empty_molecular_graph(state.n_atoms))
    conditioned = build_semantic_ring_system_decoder(state, placement(state), scaffold_context=empty)
    assert language(base) == language(conditioned)
    for labels in language(base):
        assert instantiate_semantic_ring_system_grow(state, base, labels) == (
            instantiate_semantic_ring_system_grow(state, conditioned, labels))


def test_attachment_permission_and_induced_core_bonds_are_checked_inside_ring_choice():
    state = graph("NCC")
    allowed = ScaffoldContext.from_source(graph("NC"), (0, 1))
    closed_end = ScaffoldContext.from_source(graph("NC"), (1,))
    frozen = ScaffoldContext.from_source(state, (0, 2))
    assert all(c.accepts(state) for c in (allowed, closed_end, frozen))
    # Different contexts on the exact SAME state must not share prefix caches.
    for context, expected in ((allowed, True), (closed_end, False), (frozen, False),
                              (allowed, True), (closed_end, False)):
        decoder = build_semantic_ring_system_decoder(
            state, placement(state), scaffold_context=context)
        assert bool(language(decoder)) is expected
        assert semantic_ring_prefix_is_completable(decoder, ()) is expected
    assert allowed.state_cache_key(state) != closed_end.state_cache_key(state)


def test_unprotected_ring_positions_still_have_multiple_element_choices():
    state = graph("NCC")
    context = ScaffoldContext.from_source(graph("N"), (0,))
    decoder = build_semantic_ring_system_decoder(state, placement(state), scaffold_context=context)
    runtime = de_novo_rewrite_system()
    exteriors = set()
    for labels in language(decoder):
        action = instantiate_semantic_ring_system_grow(state, decoder, labels)
        endpoint = runtime.apply(state, "ring_system_grow", action)
        assert endpoint.atom_types[0] == state.atom_types[0]
        exteriors.add(tuple(int(v) for v in endpoint.atom_types[1:3]))
    assert len(exteriors) > 1


def test_mismatched_context_is_rejected_before_decoding():
    state = graph("NCC")
    context = ScaffoldContext.from_source(graph("O"), (0,))
    with pytest.raises(ValueError, match="violates its supplied scaffold"):
        build_semantic_ring_system_decoder(state, placement(state), scaffold_context=context)
    with pytest.raises(ValueError, match="violates its supplied scaffold"):
        context.state_cache_key(state)


def test_condition_state_key_binds_exact_executable_arrays():
    context = ScaffoldContext.from_source(graph("N"), (0,))
    assert context.state_cache_key(graph("NC")) != context.state_cache_key(graph("NCC"))
    permutation = np.arange(8)[::-1]
    state = graph("NCC")
    inverse = np.argsort(permutation)
    permuted = type(state)(state.atom_types[inverse], state.formal_charges[inverse],
                          state.implicit_h_counts[inverse], state.bonds[np.ix_(inverse, inverse)])
    assert context.permuted(permutation).accepts(permuted)
    assert context.permuted(permutation).state_cache_key(permuted) != context.state_cache_key(state)
