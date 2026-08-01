"""Minimal universal molecular rewrite basis.

These operators are intentionally deterministic and model-free. The learned
Generator Matching layer will assign time-dependent rates to valid instances;
the executor remains the single source of truth for transition semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from compose_v4.rewrite.semantic_atom_restate import (
        SemanticAtomRestateContext,
        SemanticAtomRestateResolution,
    )
    from compose_v4.rewrite.semantic_cycle_close import (
        SemanticCycleCloseContext,
        SemanticCycleCloseResolution,
    )
    from compose_v4.rewrite.semantic_cycle_open import (
        SemanticCycleOpenResolution,
    )

from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_CLASS_TO_H_CHANGE,
    BOND_DOUBLE,
    BOND_SINGLE,
    BOND_TRIPLE,
    FORMAL_CHARGES,
    MAX_H_COUNT,
    NULL_IDX,
    SCAR_IDX,
    M,
    MolecularGraph,
    is_element,
)
from compose_v4.chem.state import is_valid_state

MICRO_BOND_CLASSES = (BOND_SINGLE, BOND_DOUBLE, BOND_TRIPLE)
REAL_ATOM_TYPES = set(range(1, M)) - {SCAR_IDX}


@dataclass(frozen=True)
class BondInsert:
    a: int
    b: int
    order: int


@dataclass(frozen=True)
class BondDelete:
    a: int
    b: int


@dataclass(frozen=True)
class CycleOpenEdge:
    """One semantic undirected cycle edge selected for ring opening.

    This action is deliberately distinct from :class:`BondDelete`. Editing V2
    may therefore change aromatic ring-opening semantics without silently
    reinterpreting historical raw bond-deletion records.
    """

    a: int
    b: int


@dataclass(frozen=True)
class CycleCloseEdge:
    """One semantic undirected endpoint pair and order for cycle closing.

    The action is distinct from :class:`BondInsert` so Editing V2 can reject
    closures whose molecular product depends on the stored Kekule phase while
    preserving the historical raw bond-insertion executor unchanged.
    """

    a: int
    b: int
    order: int


@dataclass(frozen=True)
class BondReorder:
    a: int
    b: int
    new_order: int


@dataclass(frozen=True)
class BondReroute:
    """Atomically exchange one bridge for a new cross-component bond.

    ``(a, b)`` is removed and ``(u, v)`` is inserted in one committed rewrite.
    The removed edge must be a bridge and the new endpoints must lie in the two
    components exposed by that cut.  No disconnected state is ever visible.
    """

    a: int
    b: int
    u: int
    v: int
    new_order: int = BOND_SINGLE


@dataclass(frozen=True)
class AtomInsert:
    slot: int
    atom_type: int
    formal_charge: int
    implicit_h_count: int
    neighbors: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class AtomDelete:
    v: int


@dataclass(frozen=True)
class AtomRestate:
    v: int
    atom_type: int
    formal_charge: int
    implicit_h_count: int


@dataclass(frozen=True)
class SemanticAtomRestate:
    """Editing-V2 atom restatement by persistent slot and valence class."""

    v: int
    target_class_index: int


def _real_atom(mg: MolecularGraph, v: int) -> bool:
    return bool(0 <= v < mg.n_atoms and is_element(np.asarray(mg.atom_types[v])))


def _h_delta(order: int) -> int:
    return int(BOND_CLASS_TO_H_CHANGE[int(order)])


def apply_bond_insert(mg: MolecularGraph, op: BondInsert) -> MolecularGraph:
    bonds = mg.bonds.copy()
    implicit_h = mg.implicit_h_counts.copy()
    bonds[op.a, op.b] = bonds[op.b, op.a] = int(op.order)
    delta = _h_delta(op.order)
    implicit_h[op.a] -= delta
    implicit_h[op.b] -= delta
    return MolecularGraph(
        mg.atom_types.copy(), mg.formal_charges.copy(), implicit_h, bonds
    )


def is_valid_bond_insert(mg: MolecularGraph, op: BondInsert) -> bool:
    if not (_real_atom(mg, op.a) and _real_atom(mg, op.b)):
        return False
    if op.a == op.b or int(op.order) not in MICRO_BOND_CLASSES:
        return False
    if int(mg.bonds[op.a, op.b]) != 0:
        return False
    delta = _h_delta(op.order)
    if int(mg.implicit_h_counts[op.a]) < delta:
        return False
    if int(mg.implicit_h_counts[op.b]) < delta:
        return False
    return is_valid_state(apply_bond_insert(mg, op))


def apply_bond_delete(mg: MolecularGraph, op: BondDelete) -> MolecularGraph:
    bonds = mg.bonds.copy()
    implicit_h = mg.implicit_h_counts.copy()
    order = int(bonds[op.a, op.b])
    bonds[op.a, op.b] = bonds[op.b, op.a] = 0
    delta = _h_delta(order)
    implicit_h[op.a] += delta
    implicit_h[op.b] += delta
    return MolecularGraph(
        mg.atom_types.copy(), mg.formal_charges.copy(), implicit_h, bonds
    )


def is_valid_bond_delete(mg: MolecularGraph, op: BondDelete) -> bool:
    if not (_real_atom(mg, op.a) and _real_atom(mg, op.b)) or op.a == op.b:
        return False
    order = int(mg.bonds[op.a, op.b])
    if order == 0:
        return False
    delta = _h_delta(order)
    if int(mg.implicit_h_counts[op.a]) + delta > MAX_H_COUNT:
        return False
    if int(mg.implicit_h_counts[op.b]) + delta > MAX_H_COUNT:
        return False
    return is_valid_state(apply_bond_delete(mg, op))


def resolve_cycle_open_edge(
    mg: MolecularGraph,
    op: CycleOpenEdge,
) -> SemanticCycleOpenResolution | None:
    """Resolve the frozen Editing-V2 semantic cycle-opening action.

    The returned production resolution is the single source for validation,
    execution, enumeration, and inverse construction.
    """

    from compose_v4.rewrite.semantic_cycle_open import (
        resolve_semantic_cycle_open,
    )

    if type(op) is not CycleOpenEdge or not (int(op.a) < int(op.b)):
        return None
    return resolve_semantic_cycle_open(
        mg,
        BondDelete(int(op.a), int(op.b)),
    )


def is_valid_cycle_open_edge(mg: MolecularGraph, op: CycleOpenEdge) -> bool:
    resolution = resolve_cycle_open_edge(mg, op)
    return bool(resolution is not None and resolution.admitted)


def apply_cycle_open_edge(mg: MolecularGraph, op: CycleOpenEdge) -> MolecularGraph:
    resolution = resolve_cycle_open_edge(mg, op)
    if resolution is None or not resolution.admitted or resolution.successor is None:
        raise ValueError(f"invalid semantic cycle_open instance: {op!r}")
    return resolution.successor


def enumerate_cycle_open_edges(mg: MolecularGraph) -> tuple[CycleOpenEdge, ...]:
    """Enumerate the exact semantic cycle-opening fiber once per endpoint pair."""

    real = tuple(int(v) for v in np.flatnonzero(is_element(mg.atom_types)))
    return tuple(
        action
        for a, b in combinations(real, 2)
        if int(mg.bonds[a, b]) != 0
        and is_valid_cycle_open_edge(mg, action := CycleOpenEdge(a, b))
    )


def inverse_cycle_open_edge(
    mg: MolecularGraph,
    op: CycleOpenEdge,
) -> BondInsert:
    """Return the exact cycle-close mark declared by an admitted resolution."""

    resolution = resolve_cycle_open_edge(mg, op)
    if (
        resolution is None
        or not resolution.admitted
        or resolution.inverse_bond_order is None
    ):
        raise ValueError(f"rejected semantic cycle_open has no inverse: {op!r}")
    return BondInsert(int(op.a), int(op.b), int(resolution.inverse_bond_order))


def resolve_cycle_close_edge(
    mg: MolecularGraph,
    op: CycleCloseEdge,
    *,
    context: SemanticCycleCloseContext | None = None,
) -> SemanticCycleCloseResolution | None:
    """Resolve the Editing-V2 semantic cycle-closing action."""

    from compose_v4.rewrite.semantic_cycle_close import (
        resolve_semantic_cycle_close,
    )

    if (
        type(op) is not CycleCloseEdge
        or not (int(op.a) < int(op.b))
        or int(op.order) not in MICRO_BOND_CLASSES
    ):
        return None
    return resolve_semantic_cycle_close(
        mg,
        BondInsert(int(op.a), int(op.b), int(op.order)),
        context=context,
    )


def is_valid_cycle_close_edge(mg: MolecularGraph, op: CycleCloseEdge) -> bool:
    resolution = resolve_cycle_close_edge(mg, op)
    return bool(resolution is not None and resolution.admitted)


def apply_cycle_close_edge(mg: MolecularGraph, op: CycleCloseEdge) -> MolecularGraph:
    resolution = resolve_cycle_close_edge(mg, op)
    if resolution is None or not resolution.admitted or resolution.successor is None:
        raise ValueError(f"invalid semantic cycle_close instance: {op!r}")
    return resolution.successor


def enumerate_cycle_close_edges(mg: MolecularGraph) -> tuple[CycleCloseEdge, ...]:
    """Enumerate the complete admitted semantic cycle-closing fiber."""

    from compose_v4.rewrite.semantic_cycle_close import (
        prepare_semantic_cycle_close_context,
    )

    context = prepare_semantic_cycle_close_context(mg)
    real = tuple(int(v) for v in np.flatnonzero(is_element(mg.atom_types)))
    return tuple(
        action
        for a, b in combinations(real, 2)
        for order in MICRO_BOND_CLASSES
        if (
            resolution := resolve_cycle_close_edge(
                mg,
                action := CycleCloseEdge(a, b, order),
                context=context,
            )
        )
        is not None
        and resolution.admitted
    )


def inverse_cycle_close_edge(
    mg: MolecularGraph,
    op: CycleCloseEdge,
) -> CycleOpenEdge:
    """Return the semantic ring-opening mark for an admitted closure."""

    resolution = resolve_cycle_close_edge(mg, op)
    if resolution is None or not resolution.admitted:
        raise ValueError(f"rejected semantic cycle_close has no inverse: {op!r}")
    return CycleOpenEdge(int(op.a), int(op.b))


def apply_bond_reorder(mg: MolecularGraph, op: BondReorder) -> MolecularGraph:
    bonds = mg.bonds.copy()
    implicit_h = mg.implicit_h_counts.copy()
    old_order = int(bonds[op.a, op.b])
    bonds[op.a, op.b] = bonds[op.b, op.a] = int(op.new_order)
    delta = _h_delta(op.new_order) - _h_delta(old_order)
    implicit_h[op.a] -= delta
    implicit_h[op.b] -= delta
    return MolecularGraph(
        mg.atom_types.copy(), mg.formal_charges.copy(), implicit_h, bonds
    )


def is_valid_bond_reorder(mg: MolecularGraph, op: BondReorder) -> bool:
    if not (_real_atom(mg, op.a) and _real_atom(mg, op.b)) or op.a == op.b:
        return False
    old_order = int(mg.bonds[op.a, op.b])
    if old_order == 0 or old_order == BOND_AROMATIC:
        return False
    if int(op.new_order) not in MICRO_BOND_CLASSES or int(op.new_order) == old_order:
        return False
    delta = _h_delta(op.new_order) - _h_delta(old_order)
    for v in (op.a, op.b):
        new_h = int(mg.implicit_h_counts[v]) - delta
        if not 0 <= new_h <= MAX_H_COUNT:
            return False
    return is_valid_state(apply_bond_reorder(mg, op))


def apply_bond_reroute(mg: MolecularGraph, op: BondReroute) -> MolecularGraph:
    """Commit a bridge cut and cross-component reconnection atomically."""

    bonds = mg.bonds.copy()
    implicit_h = mg.implicit_h_counts.copy()
    old_order = int(bonds[op.a, op.b])
    bonds[op.a, op.b] = bonds[op.b, op.a] = 0
    old_delta = _h_delta(old_order)
    implicit_h[op.a] += old_delta
    implicit_h[op.b] += old_delta

    bonds[op.u, op.v] = bonds[op.v, op.u] = int(op.new_order)
    new_delta = _h_delta(op.new_order)
    implicit_h[op.u] -= new_delta
    implicit_h[op.v] -= new_delta
    return MolecularGraph(
        mg.atom_types.copy(), mg.formal_charges.copy(), implicit_h, bonds
    )


def is_valid_bond_reroute(mg: MolecularGraph, op: BondReroute) -> bool:
    vertices = (int(op.a), int(op.b), int(op.u), int(op.v))
    if not all(_real_atom(mg, vertex) for vertex in vertices):
        return False
    if op.a == op.b or op.u == op.v:
        return False
    if int(op.new_order) not in MICRO_BOND_CLASSES:
        return False
    old_order = int(mg.bonds[op.a, op.b])
    if old_order == 0 or old_order == BOND_AROMATIC:
        return False
    if int(mg.bonds[op.u, op.v]) != 0:
        return False
    if frozenset((int(op.a), int(op.b))) == frozenset((int(op.u), int(op.v))):
        return False

    left = _component_after_cut(mg, int(op.a), int(op.a), int(op.b))
    # If b remains reachable, the removed edge was not a bridge.
    if int(op.b) in left:
        return False
    if (int(op.u) in left) == (int(op.v) in left):
        return False

    successor = apply_bond_reroute(mg, op)
    if np.any(successor.implicit_h_counts < 0):
        return False
    if np.any(successor.implicit_h_counts > MAX_H_COUNT):
        return False
    return is_valid_state(successor)


def _component_after_cut(
    mg: MolecularGraph,
    root: int,
    cut_a: int,
    cut_b: int,
) -> set[int]:
    seen = {int(root)}
    stack = [int(root)]
    cut = frozenset((int(cut_a), int(cut_b)))
    while stack:
        vertex = stack.pop()
        for neighbor in np.flatnonzero(mg.bonds[vertex] != 0):
            neighbor = int(neighbor)
            if not _real_atom(mg, neighbor):
                continue
            if frozenset((vertex, neighbor)) == cut:
                continue
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def apply_atom_insert(mg: MolecularGraph, op: AtomInsert) -> MolecularGraph:
    atom_types = mg.atom_types.copy()
    formal_charges = mg.formal_charges.copy()
    implicit_h = mg.implicit_h_counts.copy()
    bonds = mg.bonds.copy()
    atom_types[op.slot] = int(op.atom_type)
    formal_charges[op.slot] = int(op.formal_charge)
    implicit_h[op.slot] = int(op.implicit_h_count)
    for neighbor, order in op.neighbors:
        bonds[op.slot, neighbor] = bonds[neighbor, op.slot] = int(order)
        implicit_h[neighbor] -= _h_delta(order)
    return MolecularGraph(atom_types, formal_charges, implicit_h, bonds)


def is_valid_atom_insert(mg: MolecularGraph, op: AtomInsert) -> bool:
    if not 0 <= int(op.slot) < mg.n_atoms:
        return False
    if int(mg.atom_types[op.slot]) != NULL_IDX:
        return False
    if np.any(mg.bonds[op.slot] != 0):
        return False
    if int(op.atom_type) not in REAL_ATOM_TYPES:
        return False
    if int(op.formal_charge) not in FORMAL_CHARGES:
        return False
    if not 0 <= int(op.implicit_h_count) <= MAX_H_COUNT:
        return False
    neighbors = tuple((int(v), int(order)) for v, order in op.neighbors)
    if len({v for v, _ in neighbors}) != len(neighbors):
        return False
    real_count = int(is_element(mg.atom_types).sum())
    if not neighbors and real_count != 0:
        return False
    for v, order in neighbors:
        if not _real_atom(mg, v) or order not in MICRO_BOND_CLASSES:
            return False
        if int(mg.implicit_h_counts[v]) < _h_delta(order):
            return False
    return is_valid_state(apply_atom_insert(mg, op))


def is_valid_editing_v2_atom_insert(mg: MolecularGraph, op: AtomInsert) -> bool:
    """Editing-V2 birth supports only root or one-neighbor insertion."""

    return len(tuple(op.neighbors)) <= 1 and is_valid_atom_insert(mg, op)


def apply_atom_delete(mg: MolecularGraph, op: AtomDelete) -> MolecularGraph:
    atom_types = mg.atom_types.copy()
    formal_charges = mg.formal_charges.copy()
    implicit_h = mg.implicit_h_counts.copy()
    bonds = mg.bonds.copy()
    for neighbor in np.flatnonzero(bonds[op.v] != 0):
        implicit_h[neighbor] += _h_delta(int(bonds[op.v, neighbor]))
    bonds[op.v, :] = 0
    bonds[:, op.v] = 0
    atom_types[op.v] = NULL_IDX
    formal_charges[op.v] = 0
    implicit_h[op.v] = 0
    return MolecularGraph(atom_types, formal_charges, implicit_h, bonds)


def is_valid_atom_delete(mg: MolecularGraph, op: AtomDelete) -> bool:
    if not _real_atom(mg, op.v):
        return False
    for neighbor in np.flatnonzero(mg.bonds[op.v] != 0):
        new_h = int(mg.implicit_h_counts[neighbor]) + _h_delta(
            int(mg.bonds[op.v, neighbor])
        )
        if new_h > MAX_H_COUNT:
            return False
    return is_valid_state(apply_atom_delete(mg, op))


def apply_atom_restate(mg: MolecularGraph, op: AtomRestate) -> MolecularGraph:
    atom_types = mg.atom_types.copy()
    formal_charges = mg.formal_charges.copy()
    implicit_h = mg.implicit_h_counts.copy()
    atom_types[op.v] = int(op.atom_type)
    formal_charges[op.v] = int(op.formal_charge)
    implicit_h[op.v] = int(op.implicit_h_count)
    return MolecularGraph(atom_types, formal_charges, implicit_h, mg.bonds.copy())


def is_valid_atom_restate(mg: MolecularGraph, op: AtomRestate) -> bool:
    if not _real_atom(mg, op.v):
        return False
    if int(op.atom_type) not in REAL_ATOM_TYPES:
        return False
    if int(op.formal_charge) not in FORMAL_CHARGES:
        return False
    if not 0 <= int(op.implicit_h_count) <= MAX_H_COUNT:
        return False
    return is_valid_state(apply_atom_restate(mg, op))


def resolve_semantic_atom_restate_action(
    mg: MolecularGraph,
    op: SemanticAtomRestate,
    *,
    context: SemanticAtomRestateContext | None = None,
) -> SemanticAtomRestateResolution | None:
    """Resolve the frozen Editing-V2 semantic atom-restatement action."""

    from compose_v4.rewrite.semantic_atom_restate import (
        resolve_semantic_atom_restate,
    )

    if type(op) is not SemanticAtomRestate:
        return None
    return resolve_semantic_atom_restate(
        mg,
        vertex=int(op.v),
        target_class_index=int(op.target_class_index),
        context=context,
    )


def is_valid_semantic_atom_restate(
    mg: MolecularGraph,
    op: SemanticAtomRestate,
) -> bool:
    resolution = resolve_semantic_atom_restate_action(mg, op)
    return bool(resolution is not None and resolution.admitted)


def apply_semantic_atom_restate(
    mg: MolecularGraph,
    op: SemanticAtomRestate,
) -> MolecularGraph:
    resolution = resolve_semantic_atom_restate_action(mg, op)
    if resolution is None or not resolution.admitted or resolution.successor is None:
        raise ValueError(f"invalid atom_restate_semantic instance: {op!r}")
    return resolution.successor


def enumerate_semantic_atom_restates(
    mg: MolecularGraph,
) -> tuple[SemanticAtomRestate, ...]:
    """Enumerate every admitted productive semantic atom-restatement mark."""

    from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
    from compose_v4.rewrite.kernel import canonical_state_key
    from compose_v4.rewrite.semantic_atom_restate import (
        prepare_semantic_atom_restate_context,
    )

    context = prepare_semantic_atom_restate_context(mg)
    source_key = context.source_key
    actions: list[SemanticAtomRestate] = []
    for vertex in np.flatnonzero(is_element(mg.atom_types)):
        for target_class_index in range(len(ORGANIC_VOCABULARY)):
            action = SemanticAtomRestate(int(vertex), target_class_index)
            resolution = resolve_semantic_atom_restate_action(
                mg,
                action,
                context=context,
            )
            if (
                resolution is not None
                and resolution.admitted
                and resolution.successor is not None
                and canonical_state_key(resolution.successor) != source_key
            ):
                actions.append(action)
    return tuple(actions)
