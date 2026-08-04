"""Family-local Process-V2 teacher admission.

The corpus-wide Active8 question is whether the stored teacher is one exact
coordinate of its declared model family and executes to the stored productive
successor.  It does not require constructing the other seven family fibers.
The exhaustive production quotient remains the independent bounded sentinel.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from compose_v4.chem.molecular_graph import (
    BOND_CLASS_TO_H_CHANGE,
    MAX_H_COUNT,
    NULL_IDX,
    MolecularGraph,
    is_element,
)
from compose_v4.chem.persistent_state_identity import persistent_slot_state_sha256
from compose_v4.data.editing_v2_process_v2_active8_admission import (
    ProcessV2TeacherAdmissionEvidence,
    SemanticActive8AdmissionError,
    SemanticActive8AdmissionPolicy,
    _action_sha256,
    _validate_process_v2_model_modes,
    build_process_v2_semantic_active8_admission_policy,
    classify_process_v2_semantic_action,
)
from compose_v4.data.packed_trace_store import AddressedPackedTrace
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel
from compose_v4.rewrite.factorized_fiber import pendant_graft_candidates
from compose_v4.rewrite.kernel import (
    RewriteSystem,
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    BondReorder,
    BondReroute,
    CycleCloseEdge,
    CycleOpenEdge,
    SemanticAtomRestate,
    resolve_cycle_close_edge,
    resolve_cycle_open_edge,
    resolve_semantic_atom_restate_action,
)
from compose_v4.rewrite.process_v2_atom_delete import resolve_process_v2_atom_delete
from compose_v4.rewrite.ring_restate_semantics import (
    enumerate_ring_restate_semantic_groups,
)
from compose_v4.rewrite.tracelets import RingSystemRestate


_PROGRESS_INTERVAL = 64


def _edge_is_bridge(state: MolecularGraph, left: int, right: int) -> bool:
    """Return whether removing one real edge disconnects its endpoints."""

    if (
        not 0 <= left < state.n_atoms
        or not 0 <= right < state.n_atoms
        or left == right
        or not bool(is_element(np.asarray(state.atom_types[left])))
        or not bool(is_element(np.asarray(state.atom_types[right])))
        or int(state.bonds[left, right]) == 0
    ):
        return False
    seen = {left}
    stack = [left]
    while stack:
        vertex = stack.pop()
        for neighbor_value in np.flatnonzero(state.bonds[vertex] != 0):
            neighbor = int(neighbor_value)
            if frozenset((vertex, neighbor)) == frozenset((left, right)):
                continue
            if not bool(is_element(np.asarray(state.atom_types[neighbor]))):
                continue
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return right not in seen


def _atom_insert_is_one_model_coordinate(
    model: FactorizedTraceletRateModel,
    source: MolecularGraph,
    action: object,
) -> bool:
    if type(action) is not AtomInsert:
        return False
    null_slots = tuple(int(slot) for slot in np.flatnonzero(source.atom_types == NULL_IDX))
    if not null_slots or int(action.slot) != null_slots[0] or int(action.formal_charge) != 0:
        return False
    neighbors = tuple((int(slot), int(order)) for slot, order in action.neighbors)
    if len(neighbors) > 1:
        return False
    if any(order not in {1, 2, 3} for _slot, order in neighbors):
        return False
    bond_sum = sum(int(BOND_CLASS_TO_H_CHANGE[order]) for _slot, order in neighbors)
    atom_index = model.atom_vocabulary.class_index(
        int(action.atom_type),
        bond_sum,
        int(action.implicit_h_count),
        formal_charge=int(action.formal_charge),
    )
    if atom_index is None:
        return False
    expected_h = model.atom_vocabulary.h_count(atom_index, bond_sum, formal_charge=0)
    if expected_h is None or int(action.implicit_h_count) != expected_h:
        return False
    real_count = int(is_element(source.atom_types).sum())
    if not neighbors:
        return real_count == 0
    neighbor, order = neighbors[0]
    return bool(
        real_count > 0
        and 0 <= neighbor < source.n_atoms
        and is_element(np.asarray(source.atom_types[neighbor]))
        and order in {1, 2, 3}
        and int(source.implicit_h_counts[neighbor])
        >= int(BOND_CLASS_TO_H_CHANGE[order])
        and int(source.formal_charges[neighbor]) == 0
    )


def _bond_reorder_is_one_model_coordinate(
    source: MolecularGraph,
    action: object,
) -> bool:
    if type(action) is not BondReorder:
        return False
    left, right = int(action.a), int(action.b)
    if not left < right or not _edge_is_bridge(source, left, right):
        return False
    old_order = int(source.bonds[left, right])
    new_order = int(action.new_order)
    if old_order not in {1, 2, 3} or new_order not in {1, 2, 3} or old_order == new_order:
        return False
    delta = new_order - old_order
    return bool(
        int(source.formal_charges[left]) == 0
        and int(source.formal_charges[right]) == 0
        and 0 <= int(source.implicit_h_counts[left]) - delta <= MAX_H_COUNT
        and 0 <= int(source.implicit_h_counts[right]) - delta <= MAX_H_COUNT
    )


def _bond_reroute_is_one_model_coordinate(
    source: MolecularGraph,
    action: object,
) -> bool:
    if type(action) is not BondReroute or int(action.new_order) != 1:
        return False
    moved = int(action.u)
    target = int(action.v)
    removed = {int(action.a), int(action.b)} - {moved}
    if len(removed) != 1:
        return False
    removed_neighbor = removed.pop()
    if (moved, removed_neighbor, target) not in pendant_graft_candidates(source):
        return False
    charge_safe = bool(
        int(source.formal_charges[moved]) == 0
        and int(source.formal_charges[target]) == 0
        and int(source.formal_charges[removed_neighbor]) == 0
    )
    return charge_safe


def _teacher_coordinate_is_legal(
    model: FactorizedTraceletRateModel,
    source: MolecularGraph,
    rule_name: str,
    action: object,
    *,
    system: RewriteSystem,
) -> bool:
    if rule_name == "atom_insert":
        return _atom_insert_is_one_model_coordinate(model, source, action)
    if rule_name == "atom_delete":
        return bool(
            type(action) is AtomDelete
            and resolve_process_v2_atom_delete(source, action).admitted
        )
    if rule_name == "atom_restate_semantic":
        if type(action) is not SemanticAtomRestate:
            return False
        resolution = resolve_semantic_atom_restate_action(source, action)
        return bool(resolution is not None and resolution.admitted)
    if rule_name == "bond_reorder":
        return _bond_reorder_is_one_model_coordinate(source, action)
    if rule_name == "bond_reroute":
        return _bond_reroute_is_one_model_coordinate(source, action)
    if rule_name == "cycle_close":
        if type(action) is not CycleCloseEdge:
            return False
        resolution = resolve_cycle_close_edge(source, action)
        return bool(resolution is not None and resolution.admitted)
    if rule_name == "cycle_open":
        if type(action) is not CycleOpenEdge:
            return False
        resolution = resolve_cycle_open_edge(source, action)
        return bool(resolution is not None and resolution.admitted)
    if rule_name == "ring_system_restate":
        if type(action) is not RingSystemRestate:
            return False
        groups = enumerate_ring_restate_semantic_groups(
            source,
            system=system,
        )
        return sum(candidate == action for candidate in groups.actions) == 1
    raise SemanticActive8AdmissionError(
        f"teacher admission received unknown Process-V2 executor rule {rule_name!r}"
    )


class ProductionProcessV2FamilyTeacherAdmissionChecker:
    """Check only the exact teacher family, then replay exactly once."""

    def __init__(
        self,
        model: FactorizedTraceletRateModel,
        *,
        policy: SemanticActive8AdmissionPolicy | None = None,
    ) -> None:
        selected = policy or build_process_v2_semantic_active8_admission_policy()
        _validate_process_v2_model_modes(model, selected)
        self.model = model
        self.policy = selected
        self.system = editing_v2_semantic_rewrite_system()

    def evaluate_many(
        self,
        queries: Sequence[tuple[AddressedPackedTrace, int]],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[ProcessV2TeacherAdmissionEvidence, ...]:
        results: list[ProcessV2TeacherAdmissionEvidence] = []
        total = len(queries)
        for index, (addressed, step_index) in enumerate(queries, start=1):
            results.append(self.evaluate(addressed, step_index))
            if progress_callback is not None and (
                index % _PROGRESS_INTERVAL == 0 or index == total
            ):
                progress_callback(index, total)
        return tuple(results)

    def evaluate(
        self,
        addressed: AddressedPackedTrace,
        step_index: int,
    ) -> ProcessV2TeacherAdmissionEvidence:
        if type(step_index) is not int or not 0 <= step_index < len(addressed.trace.steps):
            raise SemanticActive8AdmissionError(
                "Process-V2 teacher-admission step_index lies outside the trace"
            )
        step = addressed.trace.steps[step_index]
        classification = classify_process_v2_semantic_action(
            step,
            step_index=step_index,
            policy=self.policy,
        )
        if not classification.policy_eligible or classification.action_sha256 is None:
            raise SemanticActive8AdmissionError(
                "teacher-admission checker received a policy-excluded action"
            )
        source = addressed.path.state_at(step_index)
        target = addressed.path.state_at(step_index + 1)
        source_sha256 = persistent_slot_state_sha256(source)
        target_sha256 = persistent_slot_state_sha256(target)
        source_key = canonical_state_key(source)
        target_key = canonical_state_key(target)
        legal = _teacher_coordinate_is_legal(
            self.model,
            source,
            str(step.rule_name),
            step.action,
            system=self.system,
        )
        exact = False
        executed_successor = None
        if legal:
            try:
                executed_successor = self.system.apply(
                    source,
                    str(step.rule_name),
                    step.action,
                )
            except ValueError as error:
                raise SemanticActive8AdmissionError(
                    "a family-legal Process-V2 teacher is rejected by the executor"
                ) from error
            if (
                str(step.rule_name) == "bond_reroute"
                and canonical_state_key(executed_successor) == source_key
            ):
                legal = False
                executed_successor = None
            else:
                exact = (
                    persistent_slot_state_sha256(executed_successor) == target_sha256
                )
        productive = target_key != source_key
        reason: str | None = None
        if not legal:
            reason = "teacher_action_not_a_unique_process_v2_coordinate"
        elif not exact:
            reason = "teacher_action_does_not_reproduce_exact_successor"
        elif not productive:
            reason = "teacher_successor_is_virtual_self_transition"
        action_sha256, observed_family = _action_sha256(str(step.rule_name), step.action)
        if action_sha256 != classification.action_sha256:
            raise SemanticActive8AdmissionError(
                "teacher-admission action identity changed after classification"
            )
        if observed_family != classification.model_family:
            raise SemanticActive8AdmissionError(
                "teacher-admission family changed after classification"
            )
        return ProcessV2TeacherAdmissionEvidence(
            supported=reason is None,
            exclusion_reason=reason,
            action_sha256=action_sha256,
            source_state_sha256=source_sha256,
            target_state_sha256=target_sha256,
            source_canonical_key=source_key,
            canonical_successor_key=target_key,
            teacher_coordinate_legal=legal,
            teacher_executes_to_exact_successor=exact,
            productive_canonical_successor=productive,
        )

    def __call__(
        self,
        addressed: AddressedPackedTrace,
        step_index: int,
    ) -> ProcessV2TeacherAdmissionEvidence:
        return self.evaluate(addressed, step_index)


__all__ = [
    "ProcessV2TeacherAdmissionEvidence",
    "ProductionProcessV2FamilyTeacherAdmissionChecker",
]
