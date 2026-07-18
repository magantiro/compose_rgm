from __future__ import annotations

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import empty_molecular_graph, is_valid_state, pad_molecular_graph
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    de_novo_rewrite_system,
)
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import execute_trace, invert_trace
from compose_v4.rewrite.tracelet_compiler import compile_null_to_target_tracelets
from compose_v4.rewrite.tracelet_fiber import enumerate_tracelet_cnof_fiber
from compose_v4.rewrite.tracelets import (
    AtomPayload,
    BondOrderChange,
    CycleInsert,
    RingSystemRestate,
    lower_cycle_insert,
    lower_ring_system_restate,
)


C = ELEMENT_TO_IDX["C"]


def _same_arrays(left, right) -> bool:
    return bool(
        np.array_equal(left.atom_types, right.atom_types)
        and np.array_equal(left.formal_charges, right.formal_charges)
        and np.array_equal(left.implicit_h_counts, right.implicit_h_counts)
        and np.array_equal(left.bonds, right.bonds)
    )


def _cyclohexane_seed() -> CycleInsert:
    return CycleInsert(
        atoms=tuple(AtomPayload(i, C, 0, 2) for i in range(6)),
        bond_orders=(1,) * 6,
    )


def test_cycle_and_electronic_tracelets_have_valid_micro_lowerings() -> None:
    runtime = de_novo_rewrite_system()
    source = empty_molecular_graph(8)
    cycle = _cyclohexane_seed()
    saturated = runtime.apply(source, "cycle_insert", cycle)
    aromatic = RingSystemRestate(
        tuple(BondOrderChange(i, (i + 1) % 6, 2) for i in (0, 2, 4))
    )
    benzene = runtime.apply(saturated, "ring_system_restate", aromatic)

    cycle_steps = lower_cycle_insert(source, cycle)
    electronic_steps = lower_ring_system_restate(saturated, aromatic)
    assert len(cycle_steps) == 7
    assert len(electronic_steps) == 3
    assert canonical_state_key(saturated) == "C1CCCCC1"
    assert canonical_state_key(benzene) == "c1ccccc1"


@pytest.mark.parametrize(
    "smiles",
    (
        "c1ccccc1",            # aromatic cycle
        "c1ccncc1",            # heteroaromatic cycle
        "c1ccoc1",             # lone-pair heteroaromatic cycle
        "c1ccc2ccccc2c1",      # fused aromatic ears
        "C1CC2CCC1C2",         # bridged bicyclic ears
        "C1CCC2(CC1)CCCC2",    # spiro cyclic blocks
        "CC1=CC=CC=C1",        # cyclic block with an acyclic branch
        "N#CC1CCCCC1",         # ring plus non-ring multiple bond
        "C1CC1",               # three-membered ring
        "C1CCC1",              # four-membered ring
        "C1CCCCCCCCCCC1",      # macrocycle
        "O1CCNCC1",            # saturated heterocycle
        "C1=CCCC=C1",          # partially unsaturated ring
        "c1ncc2ccccc2n1",      # fused heteroaromatic system
        "C12C3C4C1C5C2C3C45", # cage / polybridged system
        "[nH+]1ccccc1",        # charged heteroaromatic ring
    ),
)
def test_tracelet_compiler_is_exact_valid_and_exactly_invertible(smiles: str) -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 16)
    runtime = de_novo_rewrite_system()
    trace = compile_null_to_target_tracelets(target, system=runtime)
    endpoint, states = execute_trace(
        trace.source,
        trace.steps,
        system=runtime,
        return_states=True,
    )
    inverse = invert_trace(trace.source, trace.steps, system=runtime)
    restored = execute_trace(endpoint, inverse, system=runtime)

    assert _same_arrays(endpoint, target)
    assert _same_arrays(restored, trace.source)
    assert all(is_valid_state(state) for state in states)
    assert trace.metadata["micro_lowered_steps"] >= len(trace.steps)


def test_tracelets_compress_the_causally_coupled_ring_program() -> None:
    benzene = pad_molecular_graph(smiles_to_molecular_graph("c1ccccc1"), 12)
    naphthalene = pad_molecular_graph(
        smiles_to_molecular_graph("c1ccc2ccccc2c1"),
        16,
    )
    benzene_trace = compile_null_to_target_tracelets(benzene)
    naphthalene_trace = compile_null_to_target_tracelets(naphthalene)

    assert [step.rule_name for step in benzene_trace.steps] == [
        "cycle_insert",
        "ring_system_restate",
    ]
    assert [step.rule_name for step in naphthalene_trace.steps] == [
        "cycle_insert",
        "ring_ear_insert",
        "ring_system_restate",
    ]


def test_partial_unsaturation_uses_complete_micro_fallback() -> None:
    target = pad_molecular_graph(
        smiles_to_molecular_graph("C1=CCCC=C1"),
        12,
    )
    trace = compile_null_to_target_tracelets(target)

    assert [step.rule_name for step in trace.steps] == [
        "cycle_insert",
        "bond_reorder",
        "bond_reorder",
    ]


def test_nonaromatic_maximum_matching_uses_micro_fallback() -> None:
    target = pad_molecular_graph(
        smiles_to_molecular_graph("C1=CC=CC=CC=C1"),
        12,
    )
    trace = compile_null_to_target_tracelets(target)

    assert [step.rule_name for step in trace.steps] == [
        "cycle_insert",
        "bond_reorder",
        "bond_reorder",
        "bond_reorder",
        "bond_reorder",
    ]


def test_inference_restate_family_contains_only_aromatic_closures() -> None:
    benzene = pad_molecular_graph(smiles_to_molecular_graph("c1ccccc1"), 12)
    trace = compile_null_to_target_tracelets(benzene)
    precursor = TraceProgressCTMC(trace).states[-2]
    restates = enumerate_tracelet_cnof_fiber(precursor).by_family[
        "ring_system_restate"
    ]

    assert restates
    assert all(transition.successor_key == "c1ccccc1" for transition in restates)


@pytest.mark.parametrize(
    "smiles",
    (
        "c1ccccc1",
        "c1ccncc1",
        "c1ccc2ccccc2c1",
        "C1CC2CCC1C2",
        "C1CCC2(CC1)CCCC2",
        "C1CCCCCCCCCCC1",
    ),
)
def test_neutral_tracelet_teacher_successors_are_in_the_finite_fiber(smiles: str) -> None:
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 16)
    runtime = de_novo_rewrite_system()
    trace = compile_null_to_target_tracelets(target, system=runtime)
    state = trace.source

    for step in trace.steps:
        successor = runtime.apply(state, step.rule_name, step.action)
        successor_keys = {
            item.successor_key for item in enumerate_tracelet_cnof_fiber(state).transitions
        }
        assert canonical_state_key(successor) in successor_keys
        state = successor


@pytest.mark.parametrize(
    "smiles",
    ("C", "CC", "CCC", "C1CCCCC1", "CC1CCCCC1"),
)
def test_fast_generic_tracelet_fiber_matches_verified_runtime(smiles: str) -> None:
    state = pad_molecular_graph(smiles_to_molecular_graph(smiles), 10)
    runtime = de_novo_rewrite_system()
    fiber = enumerate_tracelet_cnof_fiber(state, system=runtime)

    for transition in (
        *fiber.by_family.get("cycle_insert", ()),
        *fiber.by_family.get("ring_ear_insert", ()),
    ):
        verified = runtime.apply(state, transition.rule_name, transition.action)
        assert _same_arrays(verified, transition.successor)
        assert is_valid_state(transition.successor)


def test_resonance_aliases_aggregate_at_the_chemical_successor() -> None:
    runtime = de_novo_rewrite_system()
    saturated = runtime.apply(
        empty_molecular_graph(8),
        "cycle_insert",
        _cyclohexane_seed(),
    )
    left = RingSystemRestate(
        tuple(BondOrderChange(i, (i + 1) % 6, 2) for i in (0, 2, 4))
    )
    right = RingSystemRestate(
        tuple(BondOrderChange(i, (i + 1) % 6, 2) for i in (1, 3, 5))
    )
    aggregated = runtime.aggregate_successor_rates(
        saturated,
        (
            ("ring_system_restate", left, 0.4),
            ("ring_system_restate", right, 0.6),
        ),
    )

    assert len(aggregated) == 1
    successor = next(iter(aggregated.values()))
    assert successor.rate == pytest.approx(1.0)
    assert canonical_state_key(successor.state) == "c1ccccc1"


def test_invalid_cycle_tracelet_never_commits() -> None:
    invalid = CycleInsert(
        atoms=tuple(AtomPayload(i, C, 0, 3) for i in range(6)),
        bond_orders=(1,) * 6,
    )
    with pytest.raises(InvalidRewrite):
        de_novo_rewrite_system().apply(
            empty_molecular_graph(8),
            "cycle_insert",
            invalid,
        )
