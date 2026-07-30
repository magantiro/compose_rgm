from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import compose_v4.experiments.editing_operator_support as support_module
from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    smiles_to_molecular_graph,
)
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.editing_operator_o0 import (
    EditingOperatorO0Error,
    OperatorTask,
    SearchLimits,
    compare_operator_bases,
    evaluate_basis_reachability,
    load_o0_contract,
    validate_o0_contract,
)
from compose_v4.experiments.editing_operator_support import (
    OperatorBasis,
    OperatorSupportError,
    SupportLimits,
)
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    de_novo_rewrite_system,
)
from compose_v4.rewrite.operators import AtomInsert


ROOT = Path(__file__).resolve().parents[1]


def _state(smiles: str, n_slots: int = 6):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), n_slots)


def _basis(basis_id: str, *, two_neighbor: bool = False) -> OperatorBasis:
    return OperatorBasis(
        basis_id=basis_id,
        atom_insert=True,
        atom_delete=False,
        atom_restate=False,
        bond_reorder=False,
        bond_reroute=False,
        cycle_insert=True,
        cycle_attach=False,
        two_neighbor_atom_insert=two_neighbor,
    )


def _limits() -> SearchLimits:
    return SearchLimits(
        max_expanded_labels=200,
        max_generated_marks=50_000,
        max_wall_seconds=10.0,
        support_limits=SupportLimits(
            max_candidate_attempts=20_000,
            max_executable_actions=10_000,
            max_wall_seconds=5.0,
        ),
    )


def _ear_task(*, max_events: int, constrained: bool = False) -> OperatorTask:
    source = _state("CC", 4)
    action = AtomInsert(
        slot=2,
        atom_type=ELEMENT_TO_IDX["C"],
        formal_charge=0,
        implicit_h_count=2,
        neighbors=((0, 1), (1, 1)),
    )
    target = de_novo_rewrite_system().apply(source, "atom_insert", action)

    def no_pendant_intermediate(memory, _source, marked_action):
        successor = marked_action.successor
        if int(successor.atom_types[2]) != 0:
            degree = int((successor.bonds[2] != 0).sum())
            if degree < 2:
                return False, memory
        return True, memory

    return OperatorTask(
        task_id="ring_ear_fixture",
        experiment="E7" if constrained else "E4",
        task_kind=("protected_path_fixture" if constrained else "ring_size_change"),
        source=source,
        target_key=canonical_state_key(target),
        max_events=max_events,
        constraint_id="no_pendant_new_atom" if constrained else "none",
        constraint_transition=(no_pendant_intermediate if constrained else None),
    )


def test_frozen_contract_covers_tasks_and_has_no_invented_thresholds() -> None:
    payload = load_o0_contract(ROOT / "configs" / "editing_operator_o0_v1.json")
    assert payload["status"] == "METHOD_FROZEN_TASK_ARTIFACT_PENDING"
    assert payload["task_artifact"] is None
    assert payload["launch_gate"]["open"] is False
    assert "decision_thresholds" not in payload
    assert payload["required_experiments"] == ["E3", "E4", "E7"]
    assert payload["search_state_identity"][1] == ("persistent_slot_state_sha256")
    assert (
        payload["cost_views"]["unsupported_lowering_result"]["primitive_search"]
        == "primitive_cost_undefined"
    )

    mutated = deepcopy(payload)
    mutated["decision_thresholds"] = {"event_savings": 1}
    with pytest.raises(EditingOperatorO0Error, match="threshold"):
        validate_o0_contract(mutated)


def test_two_neighbor_head_changes_one_event_support_but_not_primitive_cost() -> None:
    task = _ear_task(max_events=1)
    base, extended = compare_operator_bases(
        task,
        (
            _basis("primitive_birth_close"),
            _basis("plus_two_neighbor", two_neighbor=True),
        ),
        limits=_limits(),
    )

    assert base.events_then_primitive.status == ("not_reached_within_complete_event_bound")
    assert base.events_then_primitive.exhausted_complete_event_bound is True
    assert extended.events_then_primitive.status == "reached"
    assert extended.events_then_primitive.path is not None
    assert extended.events_then_primitive.path.event_cost == 1
    assert extended.events_then_primitive.path.primitive_equivalent_cost == 2
    assert extended.events_then_primitive.path.family_names == ("two_neighbor_atom_insert",)
    assert extended.primitive_then_events.path is not None
    assert extended.primitive_then_events.path.primitive_equivalent_cost == 2


def test_path_constraint_is_applied_before_successor_quotient() -> None:
    task = _ear_task(max_events=2, constrained=True)
    base, extended = compare_operator_bases(
        task,
        (
            _basis("primitive_birth_close"),
            _basis("plus_two_neighbor", two_neighbor=True),
        ),
        limits=_limits(),
    )

    # The primitive endpoint is reachable in two unconstrained events, but the
    # first pendant intermediate violates this task's explicit path condition.
    assert base.events_then_primitive.status == ("not_reached_within_complete_event_bound")
    assert base.events_then_primitive.exhausted_complete_event_bound is True
    assert extended.events_then_primitive.status == "reached"
    assert extended.events_then_primitive.path is not None
    assert extended.events_then_primitive.path.event_cost == 1


def test_unsupported_lowering_is_not_assigned_a_primitive_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_lowering(*_args, **_kwargs):
        raise OperatorSupportError("fixture has no primitive lowering")

    monkeypatch.setattr(
        support_module,
        "lower_two_neighbor_atom_insert",
        reject_lowering,
    )
    result = evaluate_basis_reachability(
        _ear_task(max_events=1),
        _basis("unsupported_two_neighbor", two_neighbor=True),
        limits=_limits(),
    )

    assert result.events_then_primitive.status == ("reached_with_unsupported_lowering")
    assert result.events_then_primitive.path is not None
    assert result.events_then_primitive.path.primitive_equivalent_cost is None
    assert result.events_then_primitive.path.contains_unsupported_lowering is True
    assert result.primitive_then_events.status == "primitive_cost_undefined"
    assert result.primitive_then_events.path is None
    assert result.primitive_then_events.incomplete_reason is None
    assert result.primitive_then_events.unsupported_lowering_families == (
        "two_neighbor_atom_insert",
    )
    assert result.primitive_cost_undefined is True
    assert result.incomplete is False


def test_persistent_aliases_remain_distinct_for_future_reachability() -> None:
    source = _state("CC", 4)
    runtime = de_novo_rewrite_system()
    left_alias = runtime.apply(
        source,
        "atom_insert",
        AtomInsert(
            slot=2,
            atom_type=ELEMENT_TO_IDX["C"],
            formal_charge=0,
            implicit_h_count=3,
            neighbors=((0, 1),),
        ),
    )
    right_alias = runtime.apply(
        source,
        "atom_insert",
        AtomInsert(
            slot=2,
            atom_type=ELEMENT_TO_IDX["C"],
            formal_charge=0,
            implicit_h_count=3,
            neighbors=((1, 1),),
        ),
    )
    assert canonical_state_key(left_alias) == canonical_state_key(right_alias)
    assert persistent_slot_state_sha256(left_alias) != (persistent_slot_state_sha256(right_alias))
    target = runtime.apply(
        right_alias,
        "atom_insert",
        AtomInsert(
            slot=3,
            atom_type=ELEMENT_TO_IDX["F"],
            formal_charge=0,
            implicit_h_count=0,
            neighbors=((0, 1),),
        ),
    )
    target_digest = persistent_slot_state_sha256(target)
    task = OperatorTask(
        task_id="persistent_alias_future",
        experiment="E7",
        task_kind="protected_scaffold_path",
        source=source,
        max_events=2,
        target_predicate=lambda candidate, _memory: (
            persistent_slot_state_sha256(candidate) == target_digest
        ),
    )
    result = evaluate_basis_reachability(
        task,
        OperatorBasis(
            basis_id="atom_insert_only",
            atom_insert=True,
            atom_delete=False,
            atom_restate=False,
            bond_reorder=False,
            bond_reroute=False,
            cycle_insert=False,
            cycle_attach=False,
        ),
        limits=_limits(),
    )

    assert result.events_then_primitive.status == "reached"
    assert result.events_then_primitive.path is not None
    assert result.events_then_primitive.path.event_cost == 2
    assert result.events_then_primitive.path.successor_state_sha256s[-1] == (target_digest)
    assert result.primitive_then_events.status == "reached"


def test_resource_exhaustion_is_never_reported_as_nonreachability() -> None:
    task = OperatorTask(
        task_id="deliberately_capped",
        experiment="E3",
        task_kind="size_grow",
        source=_state("CC", 8),
        target_key=canonical_state_key(_state("c1ccccc1", 8)),
        max_events=4,
    )
    capped = SearchLimits(
        max_expanded_labels=1,
        max_generated_marks=50_000,
        max_wall_seconds=10.0,
        support_limits=SupportLimits(max_wall_seconds=5.0),
    )
    result = evaluate_basis_reachability(
        task,
        _basis("capped"),
        limits=capped,
    )

    assert result.incomplete is True
    assert result.events_then_primitive.status == "incomplete_resource_bound"
    assert result.events_then_primitive.exhausted_complete_event_bound is False
    assert result.events_then_primitive.incomplete_reason == ("expanded_label_limit")


def test_ring_grow_screen_requires_audited_action_provider() -> None:
    task = OperatorTask(
        task_id="grow_provider_gate",
        experiment="E4",
        task_kind="cycle_create",
        source=_state("CCCCCC", 12),
        target_key=canonical_state_key(_state("C1CCCCC1", 12)),
        max_events=1,
    )
    basis = OperatorBasis(
        basis_id="grow_screen",
        ring_system_grow=True,
    )
    with pytest.raises(EditingOperatorO0Error, match="action provider"):
        evaluate_basis_reachability(task, basis, limits=_limits())


def test_ring_delete_search_requires_typed_catalog_preflight() -> None:
    task = OperatorTask(
        task_id="delete_catalog_gate",
        experiment="E4",
        task_kind="cycle_open",
        source=_state("C1CCCCC1", 12),
        target_key=canonical_state_key(_state("CCCCCC", 12)),
        max_events=1,
    )
    basis = OperatorBasis(
        basis_id="delete_without_catalog",
        ring_system_delete=True,
    )
    with pytest.raises(EditingOperatorO0Error, match="typed ring catalog"):
        evaluate_basis_reachability(task, basis, limits=_limits())
