from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import empty_molecular_graph, is_valid_state, pad_molecular_graph
from compose_v4.rewrite.compiler import TraceCompilationError, compile_null_to_target
from compose_v4.rewrite.kernel import InvalidRewrite, canonical_state_key, default_rewrite_system
from compose_v4.rewrite.operators import AtomDelete, AtomInsert, AtomRestate, BondReorder
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.scaffold_construction import (
    ScaffoldContext,
    compile_scaffold_to_target,
    compile_scaffold_to_target_tracelets,
)
from compose_v4.rewrite.trace import execute_trace
from compose_v4.rewrite.tracelet_compiler import compile_null_to_target_tracelets


def graph(smiles, capacity=24):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), capacity)


def same(left, right):
    return all(np.array_equal(getattr(left, name), getattr(right, name)) for name in (
        "atom_types", "formal_charges", "implicit_h_counts", "bonds"))


def permute(state, old_to_new):
    index = np.argsort(old_to_new)
    return type(state)(state.atom_types[index], state.formal_charges[index],
                       state.implicit_h_counts[index], state.bonds[np.ix_(index, index)])


CASES = (
    ("N", "NCC", {0: 0}),
    ("N", "N(CC)CC", {0: 0}),
    ("N", "N(CC)(CC)CC", {0: 0}),
    ("N1CCCCC1", "N1CCCCC1CC", dict(enumerate(range(6)))),
    ("C", "C1CCCCC1", {0: 0}),
    ("C", "CC(=O)OCC", {0: 0}),
    ("N", "NCCSSCC", {0: 0}),
    ("O", "OP(=O)(OCC)OCC", {0: 0}),
    ("C=O", "CC=O", {0: 1, 1: 2}),
    ("C", "N#C", {0: 1}),
    ("c1ccccc1", "c1ccccc1CC", dict(enumerate(range(6)))),
)


@pytest.mark.parametrize("source_smiles,target_smiles,mapping", CASES)
def test_exact_replay_preserves_source_at_every_step(source_smiles, target_smiles, mapping):
    source, target = graph(source_smiles), graph(target_smiles)
    boundary = tuple(int(v) for v in np.flatnonzero(source.implicit_h_counts > 0))
    result = compile_scaffold_to_target(source, target, mapping, attachment_slots=boundary)
    runtime = result.context.rewrite_system()
    endpoint, states = execute_trace(source, result.trace.steps, system=runtime, return_states=True)
    assert same(result.trace.source, source)
    assert same(endpoint, result.trace.target)
    assert canonical_state_key(endpoint) == canonical_state_key(target)
    assert all(result.context.accepts(state) and is_valid_state(state) for state in states)
    assert min(state.n_real_atoms for state in states) == source.n_real_atoms
    assert all(step.rule_name in ("atom_insert", "bond_insert") for step in result.trace.steps)
    correspondence = dict(result.target_to_execution)
    for source_slot, target_slot in mapping.items():
        assert correspondence[target_slot] == source_slot
    for t, v in correspondence.items():
        assert endpoint.atom_types[v] == target.atom_types[t]
        assert endpoint.implicit_h_counts[v] == target.implicit_h_counts[t]
        for u, w in correspondence.items():
            assert endpoint.bonds[v, w] == target.bonds[t, u]


def test_closed_context_prevents_otherwise_legal_attachment():
    source = graph("N")
    context = ScaffoldContext.from_source(source)
    carbon = int(graph("C").atom_types[0])
    action = AtomInsert(1, carbon, 0, 3, ((0, 1),))
    assert is_valid_state(default_rewrite_system().apply(source, "atom_insert", action))
    with pytest.raises(InvalidRewrite, match="hard condition"):
        context.rewrite_system().apply(source, "atom_insert", action)
    with pytest.raises(TraceCompilationError, match="attachment permission"):
        compile_scaffold_to_target(source, graph("NCC"), {0: 0})


@pytest.mark.parametrize("rule,action", (
    ("atom_delete", AtomDelete(0)),
    ("atom_restate", AtomRestate(0, int(graph("O").atom_types[0]), 0, 1)),
    ("bond_reorder", BondReorder(0, 1, 2)),
))
def test_protected_chemistry_edits_are_rejected(rule, action):
    source = graph("CC")
    context = ScaffoldContext.from_source(source, (0, 1))
    default_rewrite_system().apply(source, rule, action)  # not merely a chemical rejection
    with pytest.raises(InvalidRewrite, match="hard condition"):
        context.rewrite_system().apply(source, rule, action)


def test_attachment_hydrogens_change_but_supplied_atoms_do_not():
    result = compile_scaffold_to_target(graph("N"), graph("N(C)(C)C"), {0: 0}, attachment_slots=(0,))
    _, states = execute_trace(result.trace.source, result.trace.steps,
                              system=result.context.rewrite_system(), return_states=True)
    assert [int(state.implicit_h_counts[0]) for state in states] == [3, 2, 1, 0]
    restored = result.context.rewrite_system().apply(states[-1], "atom_delete", AtomDelete(3))
    assert restored.implicit_h_counts[0] == 1
    assert result.context.accepts(restored)


def test_context_is_source_only_immutable_and_permission_bound():
    source = graph("N")
    one = compile_scaffold_to_target(source, graph("NC"), {0: 0}, attachment_slots=(0,))
    three = compile_scaffold_to_target(source, graph("N(C)(C)C"), {0: 0}, attachment_slots=(0,))
    assert one.context == three.context
    assert one.context.identity == three.context.identity
    assert one.context.identity != ScaffoldContext.from_source(source).identity
    digest = one.context.identity
    source.implicit_h_counts[0] = 0
    assert one.context.identity == digest
    assert one.trace.source.implicit_h_counts[0] == 3
    assert not one.context.accepts(source)
    flags = one.context.node_flags()
    assert flags.shape == (24, 2) and flags.sum() == 2
    flags[:] = 100
    assert one.context.node_flags().sum() == 2


def test_slot_permutation_carries_context_and_teacher_mapping():
    source, target = graph("NC"), graph("N(CC)CC")
    result = compile_scaffold_to_target(source, target, {0: 0, 1: 1}, attachment_slots=(0, 1))
    p = np.random.default_rng(13).permutation(24)
    ps = permute(source, p)
    moved = compile_scaffold_to_target(ps, target, {int(p[0]): 0, int(p[1]): 1},
                                      attachment_slots=(int(p[0]), int(p[1])))
    context = result.context.permuted(p)
    assert context == moved.context
    assert context.identity != result.context.identity
    assert np.array_equal(context.node_flags()[p], result.context.node_flags())
    assert context.accepts(permute(result.trace.target, p))
    assert canonical_state_key(moved.trace.target) == canonical_state_key(result.trace.target)
    assert same(moved.trace.source, ps)
    assert result.context == context.permuted(np.argsort(p))
    with pytest.raises(ValueError, match="complete slot permutation"):
        result.context.permuted([0] * 24)


def test_compact_progress_replays_same_nonempty_source_and_clock():
    result = compile_scaffold_to_target(graph("N"), graph("N(CC)(CC)CC"), {0: 0}, attachment_slots=(0,))
    runtime = result.context.rewrite_system()
    dense = TraceProgressCTMC(result.trace, system=runtime)
    compact = TraceProgressCTMC(result.trace, system=runtime, checkpoint_interval=2)
    for k in range(len(result.trace.steps) + 1):
        assert same(dense.state_at(k), compact.state_at(k))
        assert result.context.accepts(compact.state_at(k))
    assert same(compact.state_at(0), result.trace.source)
    assert same(compact.state_at(len(result.trace.steps)), result.trace.target)
    for t in (0.0, 0.2, 0.9):
        a = dense.sample(t, np.random.default_rng(123))
        b = compact.sample(t, np.random.default_rng(123))
        assert a.progress == b.progress and a.teacher_rate == b.teacher_rate
        assert same(a.state, b.state)


def test_empty_context_delegates_without_altering_existing_compiler():
    target = graph("NCCO")
    result = compile_scaffold_to_target(empty_molecular_graph(24), target, {})
    expected = compile_null_to_target(target)
    assert result.trace.steps == expected.steps
    assert same(result.trace.source, expected.source)
    assert same(result.trace.target, expected.target)
    assert result.context.node_flags().sum() == 0


def test_identity_completion_has_no_destructive_excursion():
    source = graph("NCC")
    result = compile_scaffold_to_target(source, source, {0: 0, 1: 1, 2: 2})
    assert result.trace.steps == () and same(result.trace.source, result.trace.target)


@pytest.mark.parametrize("mapping", ({}, {1: 0}, {0: 100}, {0: 0.0}))
def test_bad_correspondence_is_not_recovered_by_canonicalization(mapping):
    with pytest.raises(TraceCompilationError):
        compile_scaffold_to_target(graph("N"), graph("NCC"), mapping, attachment_slots=(0,))


@pytest.mark.parametrize("attachments", ((1,), (0, 0), (0.0,)))
def test_invalid_attachment_slots_fail(attachments):
    with pytest.raises(ValueError):
        ScaffoldContext.from_source(graph("N"), attachments)


@pytest.mark.parametrize("source,target,mapping", (
    ("N.N", "NCC", {0: 0, 1: 1}),
    ("N", "N.CC", {0: 0}),
    ("N", "CCC", {0: 0}),
))
def test_invalid_completion_contracts_fail(source, target, mapping):
    with pytest.raises(TraceCompilationError):
        compile_scaffold_to_target(graph(source), graph(target), mapping, attachment_slots=(0,))


def test_capacity_and_existing_runtime_constraints_are_respected():
    with pytest.raises(TraceCompilationError, match="capacity"):
        compile_scaffold_to_target(graph("N", 2), graph("NCC"), {0: 0}, attachment_slots=(0,))
    base = default_rewrite_system(constraints=(lambda before, action, after: after.n_real_atoms <= 2,))
    with pytest.raises(TraceCompilationError, match="hard condition"):
        compile_scaffold_to_target(graph("N"), graph("NCC"), {0: 0},
                                   attachment_slots=(0,), system=base)


def test_kekule_phase_disagreement_is_explicit_not_a_silent_representation_repair():
    source = graph("c1ccccc1")
    target = graph("c1ccccc1C")
    for i in range(6):
        j = (i + 1) % 6
        source.bonds[i, j] = source.bonds[j, i] = 3 - source.bonds[i, j]
    assert is_valid_state(source)
    with pytest.raises(TraceCompilationError, match="exact supplied chemistry"):
        compile_scaffold_to_target(source, target, dict(enumerate(range(6))), attachment_slots=tuple(range(6)))


@pytest.mark.parametrize("typed", (False, True))
@pytest.mark.parametrize("target", ("NC1COCCN1", "Nc1ccncc1", "NCC(=O)OCC", "NCCSSCC"))
def test_tracelet_completion_reuses_ring_actions_without_primitive_chords(typed, target):
    result = compile_scaffold_to_target_tracelets(
        graph("N"), graph(target), {0: 0}, attachment_slots=(0,), typed_ring_payloads=typed)
    endpoint, states = execute_trace(result.trace.source, result.trace.steps,
                                    system=result.context.rewrite_system(), return_states=True)
    assert same(endpoint, result.trace.target)
    assert all(result.context.accepts(state) for state in states)
    assert not any(s.rule_name == "bond_insert" for s in result.trace.steps)
    if "1" in target:
        birth = next(s for s in result.trace.steps if s.rule_name == "cycle_attach")
        labels = {atom.atom_type for atom in birth.action.atoms}
        assert (len(labels) > 1) is typed  # typed birth includes ring heteroatoms immediately
    assert result.trace.metadata["typed_ring_payloads"] is typed


@pytest.mark.parametrize("target", ("NCCO", "c1ccncc1", "C1CC2CCC1C2", "C1CCC2(CC1)CCCC2"))
@pytest.mark.parametrize("typed", (False, True))
def test_empty_tracelet_behavior_is_unchanged(target, typed):
    molecule = graph(target)
    result = compile_scaffold_to_target_tracelets(
        empty_molecular_graph(24), molecule, {}, typed_ring_payloads=typed)
    expected = compile_null_to_target_tracelets(molecule, typed_ring_payloads=typed)
    assert result.trace.steps == expected.steps
    assert same(result.trace.target, expected.target)


@pytest.mark.parametrize("typed", (False, True))
def test_supplied_ring_is_never_rebuilt_or_retyped(typed):
    source = graph("c1ccncc1")
    target = graph("c1ccncc1CCN")
    # Fresh fixture alignment, not a production repair or replay reserialization.
    # The explicit opposite-Kekule rejection regression above remains required.
    identity = canonical_state_key(target)
    target.bonds[:6, :6] = source.bonds[:6, :6]
    assert is_valid_state(target) and canonical_state_key(target) == identity
    result = compile_scaffold_to_target_tracelets(
        source, target, dict(enumerate(range(6))), attachment_slots=(0, 1, 2, 4, 5),
        typed_ring_payloads=typed)
    assert all(s.rule_name == "atom_insert" for s in result.trace.steps)
    endpoint = execute_trace(source, result.trace.steps, system=result.context.rewrite_system())
    assert same(endpoint, result.trace.target)


def test_partial_ring_context_requires_a_different_teacher_policy():
    with pytest.raises(TraceCompilationError, match="complete cyclic blocks"):
        compile_scaffold_to_target_tracelets(graph("C"), graph("C1CCCCC1"),
                                             {0: 0}, attachment_slots=(0,))
