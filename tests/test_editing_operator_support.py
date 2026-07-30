from __future__ import annotations

import numpy as np
import pytest

import compose_v4.experiments.editing_operator_support as support_module
from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.chem.source_prior import DegreeBoundedCarbonTreePrior
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.editing_operator_support import (
    OperatorBasis,
    OperatorSupportError,
    enumerate_production_editing_support,
    lower_two_neighbor_atom_insert,
)
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    de_novo_rewrite_system,
)
from compose_v4.rewrite.operators import AtomInsert
from compose_v4.rewrite.trace import execute_trace
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog


def _state(smiles: str, n_slots: int = 12):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), n_slots)


def _only(**enabled: bool) -> OperatorBasis:
    values = {
        "atom_insert": False,
        "atom_delete": False,
        "atom_restate": False,
        "bond_reorder": False,
        "bond_reroute": False,
        "cycle_insert": False,
        "cycle_attach": False,
        "ring_system_restate": False,
        "ring_system_delete": False,
        "two_neighbor_atom_insert": False,
        "ring_system_grow": False,
    }
    values.update(enabled)
    return OperatorBasis(basis_id="fixture", **values)


def test_public_support_uses_model_cycle_names_and_canonical_quotient() -> None:
    chain = _state("CCC", 6)
    cyclopropane = _state("C1CC1", 6)
    closing = enumerate_production_editing_support(
        chain,
        _only(cycle_insert=True),
    )

    target_key = canonical_state_key(cyclopropane)
    target_marks = [mark for mark in closing.marks if mark.successor_key == target_key]
    assert target_marks
    assert {mark.family_name for mark in target_marks} == {"cycle_insert"}
    assert {mark.rule_name for mark in target_marks} == {"bond_insert"}
    assert all(mark.primitive_equivalent_cost == 1 for mark in target_marks)

    opening = enumerate_production_editing_support(
        cyclopropane,
        _only(cycle_attach=True),
    )
    chain_key = canonical_state_key(chain)
    inverse_marks = [mark for mark in opening.marks if mark.successor_key == chain_key]
    assert inverse_marks
    assert {mark.family_name for mark in inverse_marks} == {"cycle_attach"}
    assert {mark.rule_name for mark in inverse_marks} == {"bond_delete"}


def test_two_neighbor_birth_has_exact_two_or_three_step_lowering() -> None:
    source = _state("CC", 5)
    runtime = de_novo_rewrite_system()
    carbon_action = AtomInsert(
        slot=2,
        atom_type=ELEMENT_TO_IDX["C"],
        formal_charge=0,
        implicit_h_count=2,
        neighbors=((0, 1), (1, 1)),
    )
    carbon_target = runtime.apply(source, "atom_insert", carbon_action)
    carbon_steps = lower_two_neighbor_atom_insert(source, carbon_action)
    assert len(carbon_steps) == 2
    assert tuple(step.rule_name for step in carbon_steps) == (
        "atom_insert",
        "bond_insert",
    )
    carbon_replayed = execute_trace(source, carbon_steps, system=runtime)
    assert persistent_slot_state_sha256(carbon_replayed) == (
        persistent_slot_state_sha256(carbon_target)
    )

    sulfur_six_class = next(
        index
        for index, (element, valence) in enumerate(ORGANIC_VOCABULARY.classes)
        if element == ELEMENT_TO_IDX["S"] and valence == 6
    )
    sulfur_h = ORGANIC_VOCABULARY.h_count(sulfur_six_class, 2)
    assert sulfur_h == 4
    sulfur_action = AtomInsert(
        slot=2,
        atom_type=ELEMENT_TO_IDX["S"],
        formal_charge=0,
        implicit_h_count=sulfur_h,
        neighbors=((0, 1), (1, 1)),
    )
    sulfur_target = runtime.apply(source, "atom_insert", sulfur_action)
    sulfur_steps = lower_two_neighbor_atom_insert(source, sulfur_action)
    assert len(sulfur_steps) == 3
    assert tuple(step.rule_name for step in sulfur_steps) == (
        "atom_insert",
        "bond_insert",
        "atom_restate",
    )
    sulfur_replayed = execute_trace(source, sulfur_steps, system=runtime)
    assert persistent_slot_state_sha256(sulfur_replayed) == (
        persistent_slot_state_sha256(sulfur_target)
    )


def test_two_neighbor_support_reports_event_shortcut_at_primitive_cost() -> None:
    source = _state("CC", 5)
    action = AtomInsert(
        slot=2,
        atom_type=ELEMENT_TO_IDX["C"],
        formal_charge=0,
        implicit_h_count=2,
        neighbors=((0, 1), (1, 1)),
    )
    target = de_novo_rewrite_system().apply(source, "atom_insert", action)
    support = enumerate_production_editing_support(
        source,
        _only(two_neighbor_atom_insert=True),
    )
    matches = [
        mark
        for mark in support.marks
        if mark.successor_state_sha256 == persistent_slot_state_sha256(target)
    ]
    assert len(matches) == 1
    assert matches[0].primitive_equivalent_cost == 2
    assert matches[0].primitive_lowering_supported is True


def test_executable_two_neighbor_action_is_retained_when_lowering_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _state("CC", 5)
    action = AtomInsert(
        slot=2,
        atom_type=ELEMENT_TO_IDX["C"],
        formal_charge=0,
        implicit_h_count=2,
        neighbors=((0, 1), (1, 1)),
    )
    target = de_novo_rewrite_system().apply(source, "atom_insert", action)

    def reject_lowering(*_args, **_kwargs):
        raise OperatorSupportError("fixture has no primitive lowering")

    monkeypatch.setattr(
        support_module,
        "lower_two_neighbor_atom_insert",
        reject_lowering,
    )
    support = enumerate_production_editing_support(
        source,
        _only(two_neighbor_atom_insert=True),
    )
    matches = [
        mark
        for mark in support.marks
        if mark.successor_state_sha256 == persistent_slot_state_sha256(target)
    ]
    assert len(matches) == 1
    assert matches[0].primitive_equivalent_cost is None
    assert matches[0].primitive_lowering == ()
    assert matches[0].primitive_lowering_supported is False
    assert matches[0].primitive_lowering_failure_reason == ("fixture has no primitive lowering")
    group = next(
        item for item in support.successors if item.successor_key == canonical_state_key(target)
    )
    assert group.minimum_primitive_equivalent_cost is None
    assert group.contains_unsupported_lowering is True
    assert group.all_lowerings_supported is False


def test_clean_ring_delete_is_costed_by_productive_lowering_only() -> None:
    target = _state("C1CCC2CCCCC2C1", 20)
    source = DegreeBoundedCarbonTreePrior(sizes=(target.n_real_atoms,)).sample(
        np.random.default_rng(91), n_slots=20
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
    )
    catalog = build_typed_ring_catalog((trace,))
    support = enumerate_production_editing_support(
        target,
        _only(ring_system_delete=True),
        ring_catalog=catalog,
    )

    assert len(support.marks) == 1
    mark = support.marks[0]
    assert mark.family_name == "ring_system_delete"
    assert mark.primitive_equivalent_cost == 2
    assert mark.primitive_lowering_supported is True
    assert tuple(step.rule_name for step in mark.primitive_lowering) == (
        "bond_delete",
        "bond_delete",
    )
    replayed = execute_trace(
        target,
        mark.primitive_lowering,
        system=de_novo_rewrite_system(),
    )
    assert persistent_slot_state_sha256(replayed) == (mark.successor_state_sha256)


def test_optional_macros_fail_closed_without_explicit_support_inputs() -> None:
    state = _state("C1CCCCC1", 12)
    with pytest.raises(OperatorSupportError, match="typed ring catalog"):
        enumerate_production_editing_support(
            state,
            _only(ring_system_delete=True),
        )
    with pytest.raises(OperatorSupportError, match="pre-enumerated"):
        enumerate_production_editing_support(
            _state("CCCCCC", 12),
            _only(ring_system_grow=True),
        )
