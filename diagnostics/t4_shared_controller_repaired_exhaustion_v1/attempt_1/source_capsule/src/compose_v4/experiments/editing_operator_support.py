"""Checkpoint-independent support enumeration for editing-operator Gate O0.

This module is deliberately *not* a sampler and does not assign learned
probabilities.  It exposes the executable support needed to answer the earlier,
cheaper question:

    Which molecular successors are reachable under a declared editing basis,
    and is an optional operator new support or only an event-count shortcut?

All candidates are committed through the production rewrite system and are
then quotiented by the production canonical molecular identity.  Optional
macro actions must be supplied explicitly by a caller; importing this module
cannot enable an operator in a model, checkpoint, or trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import time
from typing import Any, Iterable

import numpy as np

from compose_v4.chem.molecular_graph import (
    MAX_H_COUNT,
    NULL_IDX,
    ORGANIC_VOCABULARY,
    MolecularGraph,
    is_element,
)
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.rewrite.factorized_fiber import (
    _factorized_candidates,
    enumerate_pendant_graft_actions,
)
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    RewriteSystem,
    canonical_state_key,
    de_novo_rewrite_system,
)
from compose_v4.rewrite.operators import (
    MICRO_BOND_CLASSES,
    AtomInsert,
    AtomRestate,
    BondInsert,
)
from compose_v4.rewrite.ring_system_fiber import (
    enumerate_clean_ring_system_deletes,
)
from compose_v4.rewrite.trace import RewriteStep
from compose_v4.rewrite.tracelet_fiber import (
    enumerate_ring_system_restate_actions,
)
from compose_v4.rewrite.tracelets import (
    RingSystemDelete,
    RingSystemGrow,
    RingSystemRestate,
    lower_ring_system_delete,
    lower_ring_system_grow,
    lower_ring_system_restate,
)
from compose_v4.rewrite.typed_ring_catalog import TypedRingCatalog


SUPPORT_CONTRACT = "compose.editing_operator_support"
SUPPORT_CONTRACT_VERSION = 1

_CORE_FAMILIES = (
    "atom_insert",
    "atom_delete",
    "atom_restate",
    "bond_reorder",
    "bond_reroute",
    "cycle_insert",
    "cycle_attach",
)
_RULE_TO_FAMILY = {
    "atom_insert": "atom_insert",
    "atom_delete": "atom_delete",
    "atom_restate": "atom_restate",
    "bond_reorder": "bond_reorder",
    "bond_reroute": "bond_reroute",
    # The model namespace deliberately differs from the executor namespace.
    "bond_insert": "cycle_insert",
    "bond_delete": "cycle_attach",
}


class OperatorSupportError(RuntimeError):
    """The requested O0 support could not be enumerated exactly."""


@dataclass(frozen=True)
class OperatorBasis:
    """One explicit editing support hypothesis.

    The defaults are the seven first-principles primitive families.  Optional
    operators are off unless a caller names and enables them.
    """

    basis_id: str = "primitive7"
    atom_insert: bool = True
    atom_delete: bool = True
    atom_restate: bool = True
    bond_reorder: bool = True
    bond_reroute: bool = True
    cycle_insert: bool = True
    cycle_attach: bool = True
    ring_system_restate: bool = False
    ring_system_delete: bool = False
    two_neighbor_atom_insert: bool = False
    ring_system_grow: bool = False

    def __post_init__(self) -> None:
        if not self.basis_id.strip():
            raise ValueError("operator basis requires a non-empty basis_id")

    def enables(self, family_name: str) -> bool:
        try:
            return bool(getattr(self, family_name))
        except AttributeError as exc:
            raise ValueError(f"unknown operator family {family_name!r}") from exc


@dataclass(frozen=True)
class SupportLimits:
    """Fail-closed resource limits for one state's legal fiber."""

    max_candidate_attempts: int = 250_000
    max_executable_actions: int = 100_000
    max_wall_seconds: float = 15.0

    def __post_init__(self) -> None:
        if self.max_candidate_attempts <= 0:
            raise ValueError("max_candidate_attempts must be positive")
        if self.max_executable_actions <= 0:
            raise ValueError("max_executable_actions must be positive")
        if self.max_wall_seconds <= 0:
            raise ValueError("max_wall_seconds must be positive")


@dataclass(frozen=True)
class SupportedAction:
    """One executable mark and its primitive-lowering certificate."""

    family_name: str
    rule_name: str
    action: Any
    successor: MolecularGraph
    successor_key: str
    successor_state_sha256: str
    primitive_equivalent_cost: int | None
    primitive_lowering: tuple[RewriteStep, ...]
    primitive_lowering_supported: bool
    primitive_lowering_failure_reason: str | None

    def __post_init__(self) -> None:
        if self.primitive_lowering_supported:
            if self.primitive_equivalent_cost is None:
                raise ValueError("supported lowering requires a primitive cost")
            if self.primitive_equivalent_cost <= 0:
                raise ValueError("primitive cost must be positive")
            if not self.primitive_lowering:
                raise ValueError("supported lowering requires at least one step")
            if self.primitive_lowering_failure_reason is not None:
                raise ValueError("supported lowering cannot have a failure reason")
        else:
            if self.primitive_equivalent_cost is not None:
                raise ValueError("unsupported lowering must have undefined cost")
            if not self.primitive_lowering_failure_reason:
                raise ValueError("unsupported lowering requires an explicit reason")


@dataclass(frozen=True)
class CanonicalSupportGroup:
    """All retained marks that induce one molecular successor."""

    successor: MolecularGraph
    successor_key: str
    marks: tuple[SupportedAction, ...]
    minimum_primitive_equivalent_cost: int | None
    all_lowerings_supported: bool
    contains_unsupported_lowering: bool


@dataclass(frozen=True)
class SupportEnumeration:
    source_key: str
    source_state_sha256: str
    basis_id: str
    marks: tuple[SupportedAction, ...]
    successors: tuple[CanonicalSupportGroup, ...]
    candidate_attempts: int


def _cycle_edges(state: MolecularGraph) -> set[frozenset[int]]:
    """Edges whose deletion preserves connectivity, without an RDKit ring basis."""

    real = {int(v) for v in np.flatnonzero(is_element(state.atom_types))}
    cyclic: set[frozenset[int]] = set()
    for a, b in combinations(sorted(real), 2):
        if int(state.bonds[a, b]) == 0:
            continue
        seen = {a}
        stack = [a]
        forbidden = frozenset((a, b))
        while stack:
            current = stack.pop()
            for neighbor in np.flatnonzero(state.bonds[current] != 0):
                neighbor = int(neighbor)
                if neighbor not in real:
                    continue
                if frozenset((current, neighbor)) == forbidden:
                    continue
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if b in seen:
            cyclic.add(forbidden)
    return cyclic


def _touches_charged_atom(
    state: MolecularGraph,
    rule_name: str,
    action: Any,
) -> bool:
    if rule_name == "atom_restate":
        vertices = (int(action.v),)
    elif rule_name in {"bond_insert", "bond_delete", "bond_reorder"}:
        vertices = (int(action.a), int(action.b))
    elif rule_name == "bond_reroute":
        vertices = (
            int(action.a),
            int(action.b),
            int(action.u),
            int(action.v),
        )
    elif rule_name == "atom_delete":
        vertices = (int(action.v),)
    elif rule_name == "atom_insert":
        vertices = tuple(int(v) for v, _ in action.neighbors)
    else:
        return False
    return any(int(state.formal_charges[vertex]) != 0 for vertex in vertices)


def _charge_policy_preserved(
    source: MolecularGraph,
    successor: MolecularGraph,
) -> bool:
    """Protect exact charged centers and forbid creating/deleting/moving charge."""

    if not np.array_equal(source.formal_charges, successor.formal_charges):
        return False
    charged = np.flatnonzero(source.formal_charges != 0)
    return all(
        int(source.atom_types[v]) == int(successor.atom_types[v])
        and int(source.implicit_h_counts[v]) == int(successor.implicit_h_counts[v])
        and np.array_equal(source.bonds[v], successor.bonds[v])
        for v in charged
    )


def _direct_candidate_allowed(
    state: MolecularGraph,
    rule_name: str,
    action: Any,
    basis: OperatorBasis,
    cycle_edges: set[frozenset[int]],
) -> bool:
    family_name = _RULE_TO_FAMILY[rule_name]
    if not basis.enables(family_name):
        return False
    if _touches_charged_atom(state, rule_name, action):
        return False
    if rule_name == "atom_restate" and int(action.formal_charge) != 0:
        return False
    if rule_name == "bond_reorder":
        edge = frozenset((int(action.a), int(action.b)))
        old_order = int(state.bonds[int(action.a), int(action.b)])
        if edge in cycle_edges or old_order not in MICRO_BOND_CLASSES:
            return False
    return True


def _primitive_candidate_is_declared(
    state: MolecularGraph,
    step: RewriteStep,
) -> bool:
    """Exact membership in the declared primitive-seven action fiber."""

    if step.rule_name == "bond_reroute":
        return not _touches_charged_atom(
            state, step.rule_name, step.action
        ) and step.action in enumerate_pendant_graft_actions(state)
    if step.rule_name not in _RULE_TO_FAMILY:
        return False
    cycle_edges = _cycle_edges(state)
    for rule_name, action in _factorized_candidates(
        state,
        allow_bond_reroute=False,
        vocabulary=ORGANIC_VOCABULARY,
    ):
        if rule_name == step.rule_name and action == step.action:
            return _direct_candidate_allowed(
                state,
                rule_name,
                action,
                OperatorBasis(),
                cycle_edges,
            )
    return False


def _certify_lowering(
    source: MolecularGraph,
    successor: MolecularGraph,
    instructions: Iterable[tuple[str, Any]],
    *,
    system: RewriteSystem,
) -> tuple[tuple[RewriteStep, ...], bool]:
    current = source
    supported = True
    retained_steps: list[RewriteStep] = []
    for rule_name, action in instructions:
        step = RewriteStep(str(rule_name), action)
        next_state = system.apply(current, step.rule_name, step.action)
        # Tracelet lowerers can carry explicit payload restorations that are
        # exact no-ops for a particular molecule.  They are implementation
        # bookkeeping, not primitive events, and the production fiber correctly
        # excludes them as canonical self transitions.
        if persistent_slot_state_sha256(next_state) == persistent_slot_state_sha256(current):
            current = next_state
            continue
        supported = supported and _primitive_candidate_is_declared(current, step)
        retained_steps.append(step)
        current = next_state
    if persistent_slot_state_sha256(current) != persistent_slot_state_sha256(successor):
        raise OperatorSupportError("primitive lowering did not reproduce the macro successor")
    if not retained_steps:
        raise OperatorSupportError("macro lowering contained no productive primitive step")
    return tuple(retained_steps), bool(supported)


def _two_neighbor_candidates(
    state: MolecularGraph,
) -> Iterable[AtomInsert]:
    null = tuple(int(v) for v in np.flatnonzero(state.atom_types == NULL_IDX))
    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    if not null or len(real) < 2:
        return
    slot = null[0]
    for left, right in combinations(real, 2):
        for left_order, right_order in product(MICRO_BOND_CLASSES, repeat=2):
            if int(state.implicit_h_counts[left]) < int(left_order):
                continue
            if int(state.implicit_h_counts[right]) < int(right_order):
                continue
            bond_sum = int(left_order) + int(right_order)
            for class_index in range(len(ORGANIC_VOCABULARY)):
                hydrogen_count = ORGANIC_VOCABULARY.h_count(
                    class_index,
                    bond_sum,
                )
                if hydrogen_count is None:
                    continue
                yield AtomInsert(
                    slot=slot,
                    atom_type=ORGANIC_VOCABULARY.element_of(class_index),
                    formal_charge=0,
                    implicit_h_count=hydrogen_count,
                    neighbors=(
                        (int(left), int(left_order)),
                        (int(right), int(right_order)),
                    ),
                )


def lower_two_neighbor_atom_insert(
    state: MolecularGraph,
    action: AtomInsert,
    *,
    system: RewriteSystem | None = None,
) -> tuple[RewriteStep, ...]:
    """Certify a two-neighbor birth as primitive birth + close (+ restate).

    The exact persistent-slot endpoint must match.  The S/P valence classes can
    require a final restate because the one-neighbor precursor and the
    two-neighbor endpoint may occupy different valence-class rows.
    """

    if len(action.neighbors) != 2:
        raise OperatorSupportError("two-neighbor lowering requires exactly two neighbors")
    runtime = system or de_novo_rewrite_system()
    try:
        target = runtime.apply(state, "atom_insert", action)
    except InvalidRewrite as exc:
        raise OperatorSupportError("two-neighbor action is not executable") from exc

    neighbors = tuple((int(neighbor), int(order)) for neighbor, order in action.neighbors)
    final_digest = persistent_slot_state_sha256(target)
    trials: list[tuple[int, int, int, int]] = []
    for anchor_index in (0, 1):
        anchor, anchor_order = neighbors[anchor_index]
        other, other_order = neighbors[1 - anchor_index]
        trials.append((anchor, anchor_order, other, other_order))

    for anchor, anchor_order, other, other_order in trials:
        precursor_h_counts: list[int] = []
        direct_h = int(action.implicit_h_count) + int(other_order)
        if 0 <= direct_h <= MAX_H_COUNT:
            precursor_h_counts.append(direct_h)
        for class_index in range(len(ORGANIC_VOCABULARY)):
            if ORGANIC_VOCABULARY.element_of(class_index) != int(action.atom_type):
                continue
            candidate_h = ORGANIC_VOCABULARY.h_count(class_index, anchor_order)
            if candidate_h is not None and candidate_h not in precursor_h_counts:
                precursor_h_counts.append(int(candidate_h))

        for precursor_h in precursor_h_counts:
            birth = AtomInsert(
                slot=int(action.slot),
                atom_type=int(action.atom_type),
                formal_charge=int(action.formal_charge),
                implicit_h_count=int(precursor_h),
                neighbors=((int(anchor), int(anchor_order)),),
            )
            first = RewriteStep("atom_insert", birth)
            if not _primitive_candidate_is_declared(state, first):
                continue
            try:
                after_birth = runtime.apply(state, first.rule_name, first.action)
            except InvalidRewrite:
                continue
            close = RewriteStep(
                "bond_insert",
                BondInsert(
                    min(int(action.slot), int(other)),
                    max(int(action.slot), int(other)),
                    int(other_order),
                ),
            )
            if not _primitive_candidate_is_declared(after_birth, close):
                continue
            try:
                after_close = runtime.apply(
                    after_birth,
                    close.rule_name,
                    close.action,
                )
            except InvalidRewrite:
                continue
            steps = [first, close]
            if persistent_slot_state_sha256(after_close) != final_digest:
                restate = RewriteStep(
                    "atom_restate",
                    AtomRestate(
                        v=int(action.slot),
                        atom_type=int(action.atom_type),
                        formal_charge=int(action.formal_charge),
                        implicit_h_count=int(action.implicit_h_count),
                    ),
                )
                if not _primitive_candidate_is_declared(after_close, restate):
                    continue
                try:
                    after_close = runtime.apply(
                        after_close,
                        restate.rule_name,
                        restate.action,
                    )
                except InvalidRewrite:
                    continue
                steps.append(restate)
            if persistent_slot_state_sha256(after_close) == final_digest:
                return tuple(steps)
    raise OperatorSupportError("two-neighbor insertion has no exact primitive-seven lowering")


def _macro_lowering(
    state: MolecularGraph,
    successor: MolecularGraph,
    action: RingSystemRestate | RingSystemDelete | RingSystemGrow,
    *,
    system: RewriteSystem,
) -> tuple[tuple[RewriteStep, ...], bool]:
    if isinstance(action, RingSystemRestate):
        instructions = lower_ring_system_restate(state, action)
    elif isinstance(action, RingSystemDelete):
        instructions = lower_ring_system_delete(state, action)
    elif isinstance(action, RingSystemGrow):
        instructions = lower_ring_system_grow(state, action)
    else:  # pragma: no cover - protected by the public signature.
        raise TypeError(type(action))
    return _certify_lowering(state, successor, instructions, system=system)


def enumerate_production_editing_support(
    state: MolecularGraph,
    basis: OperatorBasis,
    *,
    ring_catalog: TypedRingCatalog | None = None,
    ring_system_grow_actions: Iterable[RingSystemGrow] | None = None,
    system: RewriteSystem | None = None,
    limits: SupportLimits | None = None,
) -> SupportEnumeration:
    """Enumerate, execute, and quotient one declared O0 operator basis.

    ``ring_system_grow`` is intentionally provider-driven.  A caller must
    supply complete executable macro actions produced by an independently
    audited enumerator; this support-only function never constructs a model or
    turns the production hybrid on.
    """

    runtime = system or de_novo_rewrite_system()
    resource_limits = limits or SupportLimits()
    start = time.monotonic()
    source_key = canonical_state_key(state)
    attempts = 0
    retained: list[SupportedAction] = []

    def add(
        family_name: str,
        rule_name: str,
        action: Any,
        *,
        lowering: tuple[RewriteStep, ...] | None = None,
        lowering_supported: bool | None = None,
        lowering_failure_reason: str | None = None,
    ) -> None:
        nonlocal attempts
        attempts += 1
        if attempts > resource_limits.max_candidate_attempts:
            raise OperatorSupportError("candidate-attempt limit exhausted")
        if time.monotonic() - start > resource_limits.max_wall_seconds:
            raise OperatorSupportError("support wall-time limit exhausted")
        try:
            successor = runtime.apply(state, rule_name, action)
        except InvalidRewrite:
            return
        if not _charge_policy_preserved(state, successor):
            return
        successor_key = canonical_state_key(successor)
        if successor_key == source_key:
            return
        primitive_lowering = lowering
        primitive_supported = lowering_supported
        if primitive_lowering is None:
            if primitive_supported is False:
                primitive_lowering = ()
            else:
                primitive_lowering = (RewriteStep(rule_name, action),)
                primitive_supported = family_name in _CORE_FAMILIES
        if primitive_supported is None:
            raise OperatorSupportError("lowering support status was not resolved")
        if primitive_supported and lowering_failure_reason is not None:
            raise OperatorSupportError("supported lowering cannot carry a failure reason")
        if not primitive_supported and lowering_failure_reason is None:
            lowering_failure_reason = "exact lowering contains action outside primitive7 support"
        retained.append(
            SupportedAction(
                family_name=family_name,
                rule_name=rule_name,
                action=action,
                successor=successor,
                successor_key=successor_key,
                successor_state_sha256=persistent_slot_state_sha256(successor),
                primitive_equivalent_cost=(
                    len(primitive_lowering) if primitive_supported else None
                ),
                primitive_lowering=primitive_lowering,
                primitive_lowering_supported=bool(primitive_supported),
                primitive_lowering_failure_reason=lowering_failure_reason,
            )
        )
        if len(retained) > resource_limits.max_executable_actions:
            raise OperatorSupportError("executable-action limit exhausted")

    cycle_edges = _cycle_edges(state)
    for rule_name, action in _factorized_candidates(
        state,
        allow_bond_reroute=False,
        vocabulary=ORGANIC_VOCABULARY,
    ):
        if _direct_candidate_allowed(
            state,
            rule_name,
            action,
            basis,
            cycle_edges,
        ):
            add(_RULE_TO_FAMILY[rule_name], rule_name, action)

    if basis.bond_reroute:
        for action in enumerate_pendant_graft_actions(state):
            if not _touches_charged_atom(state, "bond_reroute", action):
                add("bond_reroute", "bond_reroute", action)

    if basis.two_neighbor_atom_insert:
        for action in _two_neighbor_candidates(state):
            try:
                lowering = lower_two_neighbor_atom_insert(
                    state,
                    action,
                    system=runtime,
                )
            except OperatorSupportError as exc:
                # Executable extended-support actions remain O0 evidence even
                # when no primitive-seven lowering can be certified.  Their
                # primitive cost is undefined; silently dropping them would
                # erase exactly the new-support case this gate must detect.
                add(
                    "two_neighbor_atom_insert",
                    "atom_insert",
                    action,
                    lowering=(),
                    lowering_supported=False,
                    lowering_failure_reason=str(exc),
                )
            else:
                add(
                    "two_neighbor_atom_insert",
                    "atom_insert",
                    action,
                    lowering=lowering,
                    lowering_supported=True,
                )

    if basis.ring_system_restate:
        for action in enumerate_ring_system_restate_actions(
            state,
            system=runtime,
        ):
            successor = runtime.apply(state, "ring_system_restate", action)
            lowering, supported = _macro_lowering(
                state,
                successor,
                action,
                system=runtime,
            )
            add(
                "ring_system_restate",
                "ring_system_restate",
                action,
                lowering=lowering,
                lowering_supported=supported,
            )

    if basis.ring_system_delete:
        if ring_catalog is None:
            raise OperatorSupportError("ring_system_delete requires an explicit typed ring catalog")
        for action in enumerate_clean_ring_system_deletes(state, ring_catalog):
            successor = runtime.apply(state, "ring_system_delete", action)
            lowering, supported = _macro_lowering(
                state,
                successor,
                action,
                system=runtime,
            )
            add(
                "ring_system_delete",
                "ring_system_delete",
                action,
                lowering=lowering,
                lowering_supported=supported,
            )

    if basis.ring_system_grow:
        if ring_system_grow_actions is None:
            raise OperatorSupportError(
                "ring_system_grow requires explicit pre-enumerated complete actions"
            )
        for action in ring_system_grow_actions:
            successor = runtime.apply(state, "ring_system_grow", action)
            lowering, supported = _macro_lowering(
                state,
                successor,
                action,
                system=runtime,
            )
            add(
                "ring_system_grow",
                "ring_system_grow",
                action,
                lowering=lowering,
                lowering_supported=supported,
            )
    elif ring_system_grow_actions is not None:
        raise OperatorSupportError(
            "ring-system grow actions were supplied to a basis that disables them"
        )

    retained.sort(
        key=lambda item: (
            item.successor_key,
            item.family_name,
            item.rule_name,
            repr(item.action),
        )
    )
    groups: list[CanonicalSupportGroup] = []
    for successor_key in sorted({item.successor_key for item in retained}):
        marks = tuple(item for item in retained if item.successor_key == successor_key)
        supported_costs = tuple(
            item.primitive_equivalent_cost
            for item in marks
            if item.primitive_equivalent_cost is not None
        )
        groups.append(
            CanonicalSupportGroup(
                successor=marks[0].successor,
                successor_key=successor_key,
                marks=marks,
                minimum_primitive_equivalent_cost=(
                    min(supported_costs) if supported_costs else None
                ),
                all_lowerings_supported=all(item.primitive_lowering_supported for item in marks),
                contains_unsupported_lowering=any(
                    not item.primitive_lowering_supported for item in marks
                ),
            )
        )
    return SupportEnumeration(
        source_key=source_key,
        source_state_sha256=persistent_slot_state_sha256(state),
        basis_id=basis.basis_id,
        marks=tuple(retained),
        successors=tuple(groups),
        candidate_attempts=attempts,
    )


__all__ = [
    "CanonicalSupportGroup",
    "OperatorBasis",
    "OperatorSupportError",
    "SUPPORT_CONTRACT",
    "SUPPORT_CONTRACT_VERSION",
    "SupportEnumeration",
    "SupportLimits",
    "SupportedAction",
    "enumerate_production_editing_support",
    "lower_two_neighbor_atom_insert",
]
