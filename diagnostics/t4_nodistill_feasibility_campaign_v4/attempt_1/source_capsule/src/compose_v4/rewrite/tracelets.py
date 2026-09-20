"""Chemically verified composite rewrites for coordinated ring events.

The public actions in this module are *derived* rules: each lowers to the
universal micro instruction set, and every state in that lowering is checked by
the same validity-closed runtime. The production CTMC may therefore fire the
derived rule atomically without making an unverified shortcut part of the
chemistry semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx
import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_CLASS_TO_H_CHANGE,
    MAX_H_COUNT,
    MolecularGraph,
    NULL_IDX,
    is_element,
)
from compose_v4.rewrite import operators as micro


class TraceletLoweringError(ValueError):
    """Raised when a composite rule has no validity-closed micro lowering."""


@dataclass(frozen=True)
class AtomPayload:
    """State of a new atom after the complete tracelet has committed."""

    slot: int
    atom_type: int
    formal_charge: int
    implicit_h_count: int


@dataclass(frozen=True)
class CycleInsert:
    """Insert one labelled cycle into the null state.

    ``atoms`` are in cyclic order. ``bond_orders[i]`` labels the edge from
    atom ``i`` to atom ``(i + 1) mod n``.
    """

    atoms: tuple[AtomPayload, ...]
    bond_orders: tuple[int, ...]


@dataclass(frozen=True)
class CycleDelete:
    """Delete an isolated cycle, with ``slots`` supplied in cyclic order."""

    slots: tuple[int, ...]


@dataclass(frozen=True)
class CycleAttach:
    """Attach a completely formed labelled cycle to an existing atom.

    ``atoms[0]`` is connected to ``anchor`` by ``attachment_order``.  Unlike
    growing a path and closing it later, every new atom becomes visible in a
    cyclic role in this single derived event.
    """

    anchor: int
    atoms: tuple[AtomPayload, ...]
    bond_orders: tuple[int, ...]
    attachment_order: int


@dataclass(frozen=True)
class CycleDetach:
    """Delete a pendant cycle and its single attachment edge."""

    anchor: int
    slots: tuple[int, ...]


@dataclass(frozen=True)
class RingEarInsert:
    """Insert a labelled open ear between existing anchors.

    For distinct ``a`` and ``b`` this inserts the path
    ``a-z1-...-zk-b``. For ``a == b`` it inserts a cycle petal sharing one
    anchor, covering both initial anchored cycles and spiro blocks.
    """

    a: int
    b: int
    atoms: tuple[AtomPayload, ...]
    bond_orders: tuple[int, ...]


@dataclass(frozen=True)
class RingEarDelete:
    """Delete a degree-two ear path and retain its two anchors."""

    a: int
    b: int
    slots: tuple[int, ...]


@dataclass(frozen=True, order=True)
class BondOrderChange:
    a: int
    b: int
    new_order: int


@dataclass(frozen=True)
class RingSystemRestate:
    """Apply a correlated bond-order change inside one cyclic block."""

    changes: tuple[BondOrderChange, ...]


@dataclass(frozen=True, order=True)
class RingBond:
    """One normalized executable bond inside a coordinated ring system."""

    a: int
    b: int
    order: int


@dataclass(frozen=True)
class RingSystemGrow:
    """Install one complete typed cyclic system as a single visible event.

    The learned CTMC sees this action atomically.  ``bond_reorders`` and
    ``atom_payloads`` prepare the complete typed cyclic block, after which all
    missing cycle-rank edges are committed.  They are retained only as a
    validity-checked lowering; none is exposed as an independently sampled
    closure mark.
    """

    system_atoms: tuple[int, ...]
    interface_atoms: tuple[int, ...]
    scaffold_bonds: tuple[RingBond, ...]
    bond_reorders: tuple[BondOrderChange, ...]
    atom_payloads: tuple[AtomPayload, ...]
    atom_insertions: tuple[micro.AtomInsert, ...]
    bond_insertions: tuple[RingBond, ...]
    source_aromatic_edges: tuple[tuple[int, int], ...] = ()
    aromatic_edges: tuple[tuple[int, int], ...] = ()
    topology_class: str = "cyclic"


@dataclass(frozen=True)
class RingSystemDelete:
    """Exact whole-system inverse of :class:`RingSystemGrow`.

    This removes every cycle-rank edge introduced by the installation and
    restores the complete typed acyclic precursor in one visible event.
    """

    system_atoms: tuple[int, ...]
    retained_system_atoms: tuple[int, ...]
    bond_deletions: tuple[RingBond, ...]
    atom_deletions: tuple[int, ...]
    atom_payloads: tuple[AtomPayload, ...]
    bond_reorders: tuple[BondOrderChange, ...]
    source_aromatic_edges: tuple[tuple[int, int], ...] = ()
    aromatic_edges: tuple[tuple[int, int], ...] = ()
    topology_class: str = "cyclic"


MicroInstruction = tuple[str, Any]


def lower_cycle_insert(
    state: MolecularGraph,
    action: CycleInsert,
) -> tuple[MicroInstruction, ...]:
    _check_cycle_insert_shape(state, action)
    atoms = action.atoms
    orders = action.bond_orders
    first = atoms[0]
    first_h = (
        int(first.implicit_h_count)
        + _h_delta(orders[-1])
        + _h_delta(orders[0])
    )
    _check_h(first_h)
    steps: list[MicroInstruction] = [
        (
            "atom_insert",
            micro.AtomInsert(
                slot=int(first.slot),
                atom_type=int(first.atom_type),
                formal_charge=int(first.formal_charge),
                implicit_h_count=first_h,
                neighbors=(),
            ),
        )
    ]
    for index, atom in enumerate(atoms[1:], start=1):
        insertion_h = int(atom.implicit_h_count) + _h_delta(orders[index])
        _check_h(insertion_h)
        steps.append(
            (
                "atom_insert",
                micro.AtomInsert(
                    slot=int(atom.slot),
                    atom_type=int(atom.atom_type),
                    formal_charge=int(atom.formal_charge),
                    implicit_h_count=insertion_h,
                    neighbors=((int(atoms[index - 1].slot), int(orders[index - 1])),),
                ),
            )
        )
    steps.append(
        (
            "bond_insert",
            micro.BondInsert(
                int(atoms[-1].slot),
                int(first.slot),
                int(orders[-1]),
            ),
        )
    )
    _execute_micro(state, steps)
    return tuple(steps)


def apply_cycle_insert(state: MolecularGraph, action: CycleInsert) -> MolecularGraph:
    return _execute_micro(state, lower_cycle_insert(state, action))


def is_valid_cycle_insert(state: MolecularGraph, action: CycleInsert) -> bool:
    return _admits_lowering(lower_cycle_insert, state, action)


def lower_cycle_delete(
    state: MolecularGraph,
    action: CycleDelete,
) -> tuple[MicroInstruction, ...]:
    slots = tuple(int(v) for v in action.slots)
    if len(slots) < 3 or len(set(slots)) != len(slots):
        raise TraceletLoweringError("a cycle deletion needs at least three unique atoms")
    if any(not _real_atom(state, v) for v in slots):
        raise TraceletLoweringError("cycle deletion refers to a non-real atom")
    cycle_edges = {
        _edge(slots[index], slots[(index + 1) % len(slots)])
        for index in range(len(slots))
    }
    for edge in cycle_edges:
        a, b = tuple(edge)
        if int(state.bonds[a, b]) == 0:
            raise TraceletLoweringError("cycle deletion is missing a cycle edge")
    for v in slots:
        incident = {
            _edge(v, int(u)) for u in np.flatnonzero(state.bonds[v] != 0)
        }
        if not incident.issubset(cycle_edges) or len(incident) != 2:
            raise TraceletLoweringError("cycle deletion atoms must form an isolated cycle")
    steps: list[MicroInstruction] = [
        ("bond_delete", micro.BondDelete(slots[-1], slots[0]))
    ]
    steps.extend(
        ("atom_delete", micro.AtomDelete(v)) for v in reversed(slots[1:])
    )
    steps.append(("atom_delete", micro.AtomDelete(slots[0])))
    _execute_micro(state, steps)
    return tuple(steps)


def apply_cycle_delete(state: MolecularGraph, action: CycleDelete) -> MolecularGraph:
    return _execute_micro(state, lower_cycle_delete(state, action))


def is_valid_cycle_delete(state: MolecularGraph, action: CycleDelete) -> bool:
    return _admits_lowering(lower_cycle_delete, state, action)


def lower_cycle_attach(
    state: MolecularGraph,
    action: CycleAttach,
) -> tuple[MicroInstruction, ...]:
    _check_cycle_attach_shape(state, action)
    atoms = action.atoms
    orders = action.bond_orders
    first = atoms[0]
    first_h = (
        int(first.implicit_h_count)
        + _h_delta(orders[-1])
        + _h_delta(orders[0])
    )
    _check_h(first_h)
    steps: list[MicroInstruction] = [
        (
            "atom_insert",
            micro.AtomInsert(
                slot=int(first.slot),
                atom_type=int(first.atom_type),
                formal_charge=int(first.formal_charge),
                implicit_h_count=first_h,
                neighbors=((int(action.anchor), int(action.attachment_order)),),
            ),
        )
    ]
    for index, atom in enumerate(atoms[1:], start=1):
        insertion_h = int(atom.implicit_h_count) + _h_delta(orders[index])
        _check_h(insertion_h)
        steps.append(
            (
                "atom_insert",
                micro.AtomInsert(
                    slot=int(atom.slot),
                    atom_type=int(atom.atom_type),
                    formal_charge=int(atom.formal_charge),
                    implicit_h_count=insertion_h,
                    neighbors=((int(atoms[index - 1].slot), int(orders[index - 1])),),
                ),
            )
        )
    steps.append(
        (
            "bond_insert",
            micro.BondInsert(
                int(atoms[-1].slot),
                int(first.slot),
                int(orders[-1]),
            ),
        )
    )
    _execute_micro(state, steps)
    return tuple(steps)


def apply_cycle_attach(state: MolecularGraph, action: CycleAttach) -> MolecularGraph:
    return _execute_micro(state, lower_cycle_attach(state, action))


def is_valid_cycle_attach(state: MolecularGraph, action: CycleAttach) -> bool:
    return _admits_lowering(lower_cycle_attach, state, action)


def lower_cycle_detach(
    state: MolecularGraph,
    action: CycleDetach,
) -> tuple[MicroInstruction, ...]:
    anchor = int(action.anchor)
    slots = tuple(int(v) for v in action.slots)
    if not _real_atom(state, anchor):
        raise TraceletLoweringError("cycle detachment anchor must be real")
    if len(slots) < 3 or len(set(slots)) != len(slots) or anchor in slots:
        raise TraceletLoweringError("cycle detachment needs three unique cycle atoms")
    if any(not _real_atom(state, v) for v in slots):
        raise TraceletLoweringError("cycle detachment refers to a non-real atom")
    cycle_edges = {
        _edge(slots[index], slots[(index + 1) % len(slots)])
        for index in range(len(slots))
    }
    attachment = _edge(anchor, slots[0])
    if int(state.bonds[anchor, slots[0]]) == 0:
        raise TraceletLoweringError("cycle detachment is missing its attachment edge")
    for edge in cycle_edges:
        a, b = tuple(edge)
        if int(state.bonds[a, b]) == 0:
            raise TraceletLoweringError("cycle detachment is missing a cycle edge")
    for v in slots:
        incident = {
            _edge(v, int(u)) for u in np.flatnonzero(state.bonds[v] != 0)
        }
        allowed = cycle_edges | ({attachment} if v == slots[0] else set())
        if incident != {edge for edge in allowed if v in edge}:
            raise TraceletLoweringError("attached cycle has unsupported external edges")
    steps: list[MicroInstruction] = [
        ("bond_delete", micro.BondDelete(slots[-1], slots[0]))
    ]
    steps.extend(("atom_delete", micro.AtomDelete(v)) for v in reversed(slots))
    _execute_micro(state, steps)
    return tuple(steps)


def apply_cycle_detach(state: MolecularGraph, action: CycleDetach) -> MolecularGraph:
    return _execute_micro(state, lower_cycle_detach(state, action))


def is_valid_cycle_detach(state: MolecularGraph, action: CycleDetach) -> bool:
    return _admits_lowering(lower_cycle_detach, state, action)


def lower_ring_ear_insert(
    state: MolecularGraph,
    action: RingEarInsert,
) -> tuple[MicroInstruction, ...]:
    _check_ear_insert_shape(state, action)
    atoms = action.atoms
    orders = action.bond_orders
    steps: list[MicroInstruction] = []
    previous = int(action.a)
    for index, atom in enumerate(atoms):
        insertion_h = int(atom.implicit_h_count) + _h_delta(orders[index + 1])
        _check_h(insertion_h)
        steps.append(
            (
                "atom_insert",
                micro.AtomInsert(
                    slot=int(atom.slot),
                    atom_type=int(atom.atom_type),
                    formal_charge=int(atom.formal_charge),
                    implicit_h_count=insertion_h,
                    neighbors=((previous, int(orders[index])),),
                ),
            )
        )
        previous = int(atom.slot)
    steps.append(
        (
            "bond_insert",
            micro.BondInsert(previous, int(action.b), int(orders[-1])),
        )
    )
    _execute_micro(state, steps)
    return tuple(steps)


def apply_ring_ear_insert(
    state: MolecularGraph,
    action: RingEarInsert,
) -> MolecularGraph:
    return _execute_micro(state, lower_ring_ear_insert(state, action))


def is_valid_ring_ear_insert(state: MolecularGraph, action: RingEarInsert) -> bool:
    return _admits_lowering(lower_ring_ear_insert, state, action)


def lower_ring_ear_delete(
    state: MolecularGraph,
    action: RingEarDelete,
) -> tuple[MicroInstruction, ...]:
    a = int(action.a)
    b = int(action.b)
    slots = tuple(int(v) for v in action.slots)
    minimum = 2 if a == b else 0
    if len(slots) < minimum or len(set(slots)) != len(slots):
        raise TraceletLoweringError("ear deletion has an invalid internal path")
    if not (_real_atom(state, a) and _real_atom(state, b)):
        raise TraceletLoweringError("ear deletion anchors must be real")
    if a in slots or b in slots or any(not _real_atom(state, v) for v in slots):
        raise TraceletLoweringError("ear deletion path and anchors must be disjoint")
    path = (a, *slots, b)
    path_edges = {_edge(path[i], path[i + 1]) for i in range(len(path) - 1)}
    if len(path_edges) != len(path) - 1:
        raise TraceletLoweringError("ear deletion path repeats an edge")
    for edge in path_edges:
        u, v = tuple(edge)
        if int(state.bonds[u, v]) == 0:
            raise TraceletLoweringError("ear deletion is missing a path edge")
    for v in slots:
        incident = {
            _edge(v, int(u)) for u in np.flatnonzero(state.bonds[v] != 0)
        }
        if incident != {edge for edge in path_edges if v in edge}:
            raise TraceletLoweringError("ear internal atoms must have path degree two")
    last = slots[-1] if slots else a
    steps: list[MicroInstruction] = [
        ("bond_delete", micro.BondDelete(last, b))
    ]
    steps.extend(
        ("atom_delete", micro.AtomDelete(v)) for v in reversed(slots)
    )
    _execute_micro(state, steps)
    return tuple(steps)


def apply_ring_ear_delete(
    state: MolecularGraph,
    action: RingEarDelete,
) -> MolecularGraph:
    return _execute_micro(state, lower_ring_ear_delete(state, action))


def is_valid_ring_ear_delete(state: MolecularGraph, action: RingEarDelete) -> bool:
    return _admits_lowering(lower_ring_ear_delete, state, action)


def lower_ring_system_restate(
    state: MolecularGraph,
    action: RingSystemRestate,
) -> tuple[MicroInstruction, ...]:
    changes = tuple(action.changes)
    if not changes:
        raise TraceletLoweringError("ring restate needs at least one changed bond")
    edges = [_edge(int(change.a), int(change.b)) for change in changes]
    if len(set(edges)) != len(edges):
        raise TraceletLoweringError("ring restate repeats a bond")
    for change in changes:
        a, b, order = int(change.a), int(change.b), int(change.new_order)
        if not (_real_atom(state, a) and _real_atom(state, b)) or a == b:
            raise TraceletLoweringError("ring restate refers to invalid atoms")
        old = int(state.bonds[a, b])
        if old == 0 or order not in micro.MICRO_BOND_CLASSES or order == old:
            raise TraceletLoweringError("ring restate contains an invalid order change")
    _require_one_ring_system(state, edges)

    # A resonance move may need to lower one bond before raising its neighbor.
    # Greedily commit any currently legal decrease first, then any legal
    # increase. Failure means the macro has no verified lowering in this order
    # class, so it is not admitted.
    remaining = list(changes)
    current = state
    steps: list[MicroInstruction] = []
    while remaining:
        ranked = sorted(
            remaining,
            key=lambda item: (
                int(item.new_order) - int(current.bonds[item.a, item.b]) > 0,
                abs(int(item.new_order) - int(current.bonds[item.a, item.b])),
                min(int(item.a), int(item.b)),
                max(int(item.a), int(item.b)),
            ),
        )
        committed = False
        for change in ranked:
            step = (
                "bond_reorder",
                micro.BondReorder(
                    int(change.a), int(change.b), int(change.new_order)
                ),
            )
            try:
                successor = _execute_micro(current, (step,))
            except Exception:
                continue
            steps.append(step)
            current = successor
            remaining.remove(change)
            committed = True
            break
        if not committed:
            raise TraceletLoweringError(
                "ring restate has no validity-closed sequential lowering"
            )
    return tuple(steps)


def apply_ring_system_restate(
    state: MolecularGraph,
    action: RingSystemRestate,
) -> MolecularGraph:
    return _execute_micro(state, lower_ring_system_restate(state, action))


def is_valid_ring_system_restate(
    state: MolecularGraph,
    action: RingSystemRestate,
) -> bool:
    return _admits_lowering(lower_ring_system_restate, state, action)


def lower_ring_system_grow(
    state: MolecularGraph,
    action: RingSystemGrow,
) -> tuple[MicroInstruction, ...]:
    insertion_slots = tuple(int(item.slot) for item in action.atom_insertions)
    members = _check_ring_system_members(
        state,
        action.system_atoms,
        new_slots=insertion_slots,
    )
    new_members = set(insertion_slots)
    existing_members = members - new_members
    interface = {int(slot) for slot in action.interface_atoms}
    if not interface.issubset(existing_members):
        raise TraceletLoweringError("ring-system interface must already be real")
    if len(interface) != len(action.interface_atoms):
        raise TraceletLoweringError("ring-system interface repeats a slot")
    cyclic_before = _cyclic_atoms(state)
    if cyclic_before & members != interface:
        raise TraceletLoweringError(
            "ring-system growth must declare its complete existing cyclic interface"
        )
    _check_aromatic_edge_shape(action.source_aromatic_edges, interface)
    if interface:
        _require_exact_ring_system(
            state,
            interface,
            aromatic_edges=action.source_aromatic_edges,
        )
    elif action.source_aromatic_edges:
        raise TraceletLoweringError("acyclic ring growth has aromatic source edges")
    _check_aromatic_edge_shape(action.aromatic_edges, members)
    scaffold_edges: dict[frozenset[int], int] = {}
    for bond in action.scaffold_bonds:
        a, b, order = int(bond.a), int(bond.b), int(bond.order)
        edge = _edge(a, b)
        if (
            a not in existing_members
            or b not in existing_members
            or a == b
            or edge in scaffold_edges
            or order not in micro.MICRO_BOND_CLASSES
        ):
            raise TraceletLoweringError("ring-system scaffold has invalid support")
        scaffold_edges[edge] = order
    actual_existing_edges = {
        _edge(a, b): int(state.bonds[a, b])
        for a in existing_members
        for b in existing_members
        if b > a and int(state.bonds[a, b]) != 0
    }
    if actual_existing_edges != scaffold_edges:
        raise TraceletLoweringError(
            "ring-system action does not match the complete acyclic scaffold"
        )
    steps: list[MicroInstruction] = []
    for change in action.bond_reorders:
        a, b, order = int(change.a), int(change.b), int(change.new_order)
        if a not in members or b not in members or a == b:
            raise TraceletLoweringError("ring-system reorder leaves its member set")
        if int(state.bonds[a, b]) == 0 or order not in micro.MICRO_BOND_CLASSES:
            raise TraceletLoweringError("ring-system reorder is not executable")
        steps.append(("bond_reorder", micro.BondReorder(a, b, order)))
    payload_slots = tuple(int(payload.slot) for payload in action.atom_payloads)
    if (
        not (existing_members - interface).issubset(payload_slots)
        or not set(payload_slots).issubset(existing_members)
        or len(set(payload_slots)) != len(payload_slots)
    ):
        raise TraceletLoweringError(
            "ring-system growth must jointly type every new acyclic member"
        )
    for payload in action.atom_payloads:
        steps.append(
            (
                "atom_restate",
                micro.AtomRestate(
                    int(payload.slot),
                    int(payload.atom_type),
                    int(payload.formal_charge),
                    int(payload.implicit_h_count),
                ),
            )
        )
    if len(new_members) != len(insertion_slots):
        raise TraceletLoweringError("ring-system atom insertions repeat a slot")
    for insertion in action.atom_insertions:
        if int(insertion.slot) not in members or len(insertion.neighbors) > 1:
            raise TraceletLoweringError(
                "ring-system growth atoms must extend one connected path at a time"
            )
        steps.append(("atom_insert", insertion))
    insertion_edges: set[frozenset[int]] = set()
    for bond in action.bond_insertions:
        a, b, order = int(bond.a), int(bond.b), int(bond.order)
        edge = _edge(a, b)
        if (
            a not in members
            or b not in members
            or a == b
            or edge in insertion_edges
            or order not in micro.MICRO_BOND_CLASSES
            or int(state.bonds[a, b]) != 0
        ):
            raise TraceletLoweringError("ring-system insertion has invalid support")
        insertion_edges.add(edge)
        steps.append(("bond_insert", micro.BondInsert(a, b, order)))
    if not insertion_edges:
        raise TraceletLoweringError("ring-system growth must increase cycle rank")
    successor = _execute_micro(state, steps)
    _require_exact_ring_system(
        successor,
        members,
        aromatic_edges=action.aromatic_edges,
    )
    if _cycle_rank(successor) - _cycle_rank(state) != len(insertion_edges):
        raise TraceletLoweringError("ring-system growth has the wrong cycle-rank delta")
    return tuple(steps)


def apply_ring_system_grow(
    state: MolecularGraph,
    action: RingSystemGrow,
) -> MolecularGraph:
    return _execute_micro(state, lower_ring_system_grow(state, action))


def is_valid_ring_system_grow(
    state: MolecularGraph,
    action: RingSystemGrow,
) -> bool:
    return _admits_lowering(lower_ring_system_grow, state, action)


def lower_ring_system_delete(
    state: MolecularGraph,
    action: RingSystemDelete,
) -> tuple[MicroInstruction, ...]:
    members = _check_ring_system_members(state, action.system_atoms)
    retained = {int(slot) for slot in action.retained_system_atoms}
    if not retained.issubset(members) or len(retained) != len(
        action.retained_system_atoms
    ):
        raise TraceletLoweringError("ring-system retained interface is invalid")
    _check_aromatic_edge_shape(action.aromatic_edges, members)
    _check_aromatic_edge_shape(action.source_aromatic_edges, retained)
    _require_exact_ring_system(
        state,
        members,
        aromatic_edges=action.aromatic_edges,
    )
    steps: list[MicroInstruction] = []
    deletion_edges: set[frozenset[int]] = set()
    for bond in action.bond_deletions:
        a, b, order = int(bond.a), int(bond.b), int(bond.order)
        edge = _edge(a, b)
        if (
            a not in members
            or b not in members
            or a == b
            or edge in deletion_edges
            or int(state.bonds[a, b]) != order
        ):
            raise TraceletLoweringError("ring-system deletion has invalid support")
        deletion_edges.add(edge)
        steps.append(("bond_delete", micro.BondDelete(a, b)))
    if not deletion_edges:
        raise TraceletLoweringError("ring-system deletion must reduce cycle rank")
    deleted_atoms = tuple(int(slot) for slot in action.atom_deletions)
    if (
        len(set(deleted_atoms)) != len(deleted_atoms)
        or not set(deleted_atoms).issubset(members - retained)
    ):
        raise TraceletLoweringError("ring-system atom deletions are invalid")
    for slot in deleted_atoms:
        steps.append(("atom_delete", micro.AtomDelete(slot)))
    payload_slots = tuple(int(payload.slot) for payload in action.atom_payloads)
    restored_members = members - set(deleted_atoms)
    if (
        not (restored_members - retained).issubset(payload_slots)
        or not set(payload_slots).issubset(restored_members)
        or len(set(payload_slots)) != len(payload_slots)
    ):
        raise TraceletLoweringError(
            "ring-system deletion must restore every retained acyclic member"
        )
    for payload in action.atom_payloads:
        steps.append(
            (
                "atom_restate",
                micro.AtomRestate(
                    int(payload.slot),
                    int(payload.atom_type),
                    int(payload.formal_charge),
                    int(payload.implicit_h_count),
                ),
            )
        )
    for change in action.bond_reorders:
        a, b, order = int(change.a), int(change.b), int(change.new_order)
        if a not in members or b not in members or a == b:
            raise TraceletLoweringError("ring-system inverse reorder leaves its members")
        steps.append(("bond_reorder", micro.BondReorder(a, b, order)))
    successor = _execute_micro(state, steps)
    if retained:
        _require_exact_ring_system(
            successor,
            retained,
            aromatic_edges=action.source_aromatic_edges,
        )
    if _cyclic_atoms(successor) & (restored_members - retained):
        raise TraceletLoweringError("ring-system deletion left a new cyclic member")
    if any(int(successor.atom_types[slot]) != NULL_IDX for slot in deleted_atoms):
        raise TraceletLoweringError("ring-system deletion did not release its slots")
    if _cycle_rank(state) - _cycle_rank(successor) != len(deletion_edges):
        raise TraceletLoweringError("ring-system deletion has the wrong cycle-rank delta")
    return tuple(steps)


def apply_ring_system_delete(
    state: MolecularGraph,
    action: RingSystemDelete,
) -> MolecularGraph:
    return _execute_micro(state, lower_ring_system_delete(state, action))


def is_valid_ring_system_delete(
    state: MolecularGraph,
    action: RingSystemDelete,
) -> bool:
    return _admits_lowering(lower_ring_system_delete, state, action)


def inverse_ring_system_grow(
    source: MolecularGraph,
    action: RingSystemGrow,
) -> RingSystemDelete:
    inverse = _invert_micro_program(source, lower_ring_system_grow(source, action))
    return RingSystemDelete(
        system_atoms=action.system_atoms,
        retained_system_atoms=action.interface_atoms,
        bond_deletions=tuple(
            RingBond(int(item.a), int(item.b), int(bond.order))
            for (rule, item), bond in zip(
                (step for step in inverse if step[0] == "bond_delete"),
                reversed(action.bond_insertions),
            )
        ),
        atom_deletions=tuple(
            int(item.v)
            for rule, item in inverse
            if rule == "atom_delete"
        ),
        atom_payloads=tuple(
            AtomPayload(
                int(item.v),
                int(item.atom_type),
                int(item.formal_charge),
                int(item.implicit_h_count),
            )
            for rule, item in inverse
            if rule == "atom_restate"
        ),
        bond_reorders=tuple(
            BondOrderChange(int(item.a), int(item.b), int(item.new_order))
            for rule, item in inverse
            if rule == "bond_reorder"
        ),
        source_aromatic_edges=action.source_aromatic_edges,
        aromatic_edges=action.aromatic_edges,
        topology_class=action.topology_class,
    )


def inverse_ring_system_delete(
    source: MolecularGraph,
    action: RingSystemDelete,
) -> RingSystemGrow:
    inverse = _invert_micro_program(source, lower_ring_system_delete(source, action))
    # The grow's scaffold is the OPENED graph (post-delete): the ring bonds MINUS the closing bonds that
    # this inverse re-adds as bond_insertions. Reading scaffold_bonds straight off ``source`` (the still-
    # closed ring) would double-count those bonds and produce an invalid grow (scaffold ∩ insertions ≠ ∅).
    reinserted = {
        frozenset((int(item.a), int(item.b)))
        for rule, item in inverse
        if rule == "bond_insert"
    }
    return RingSystemGrow(
        system_atoms=action.system_atoms,
        interface_atoms=action.retained_system_atoms,
        scaffold_bonds=tuple(
            RingBond(int(a), int(b), int(source.bonds[a, b]))
            for offset, a in enumerate(action.system_atoms)
            for b in action.system_atoms[offset + 1 :]
            if int(source.bonds[a, b]) != 0
            and frozenset((int(a), int(b))) not in reinserted
        ),
        bond_reorders=tuple(
            BondOrderChange(int(item.a), int(item.b), int(item.new_order))
            for rule, item in inverse
            if rule == "bond_reorder"
        ),
        atom_payloads=tuple(
            AtomPayload(
                int(item.v),
                int(item.atom_type),
                int(item.formal_charge),
                int(item.implicit_h_count),
            )
            for rule, item in inverse
            if rule == "atom_restate"
        ),
        atom_insertions=tuple(
            item
            for rule, item in inverse
            if rule == "atom_insert"
        ),
        bond_insertions=tuple(
            RingBond(int(item.a), int(item.b), int(item.order))
            for rule, item in inverse
            if rule == "bond_insert"
        ),
        source_aromatic_edges=action.source_aromatic_edges,
        aromatic_edges=action.aromatic_edges,
        topology_class=action.topology_class,
    )


def _check_cycle_insert_shape(state: MolecularGraph, action: CycleInsert) -> None:
    atoms = tuple(action.atoms)
    if state.n_real_atoms != 0:
        raise TraceletLoweringError("cycle insertion is the null-state seed rule")
    if len(atoms) < 3 or len(action.bond_orders) != len(atoms):
        raise TraceletLoweringError("cycle insertion has inconsistent cycle dimensions")
    slots = tuple(int(atom.slot) for atom in atoms)
    if len(set(slots)) != len(slots):
        raise TraceletLoweringError("cycle insertion repeats a slot")
    for atom in atoms:
        _check_new_atom_payload(state, atom)
    _check_orders(action.bond_orders)


def _check_cycle_attach_shape(state: MolecularGraph, action: CycleAttach) -> None:
    atoms = tuple(action.atoms)
    anchor = int(action.anchor)
    if state.n_real_atoms == 0 or not _real_atom(state, anchor):
        raise TraceletLoweringError("cycle attachment needs an existing anchor")
    if len(atoms) < 3 or len(action.bond_orders) != len(atoms):
        raise TraceletLoweringError("cycle attachment has inconsistent dimensions")
    slots = tuple(int(atom.slot) for atom in atoms)
    if len(set(slots)) != len(slots) or anchor in slots:
        raise TraceletLoweringError("cycle attachment repeats an atom")
    for atom in atoms:
        _check_new_atom_payload(state, atom)
    _check_orders(action.bond_orders)
    _check_orders((action.attachment_order,))


def _check_ear_insert_shape(state: MolecularGraph, action: RingEarInsert) -> None:
    a, b = int(action.a), int(action.b)
    atoms = tuple(action.atoms)
    minimum = 2 if a == b else 0
    if len(atoms) < minimum or len(action.bond_orders) != len(atoms) + 1:
        raise TraceletLoweringError("ear insertion has inconsistent path dimensions")
    if not (_real_atom(state, a) and _real_atom(state, b)):
        raise TraceletLoweringError("ear insertion anchors must be real")
    slots = tuple(int(atom.slot) for atom in atoms)
    if len(set(slots)) != len(slots) or a in slots or b in slots:
        raise TraceletLoweringError("ear insertion path repeats an atom")
    for atom in atoms:
        _check_new_atom_payload(state, atom)
    _check_orders(action.bond_orders)


def _check_new_atom_payload(state: MolecularGraph, atom: AtomPayload) -> None:
    slot = int(atom.slot)
    if not 0 <= slot < state.n_atoms or int(state.atom_types[slot]) != NULL_IDX:
        raise TraceletLoweringError("tracelet atom payload needs a null slot")
    if not 0 <= int(atom.implicit_h_count) <= MAX_H_COUNT:
        raise TraceletLoweringError("tracelet atom payload has invalid hydrogens")


def _check_orders(orders: Iterable[int]) -> None:
    if any(int(order) not in micro.MICRO_BOND_CLASSES for order in orders):
        raise TraceletLoweringError("tracelet uses a non-micro bond order")


def _check_ring_system_members(
    state: MolecularGraph,
    slots: Iterable[int],
    *,
    new_slots: Iterable[int] = (),
) -> set[int]:
    members = tuple(int(slot) for slot in slots)
    new_members = {int(slot) for slot in new_slots}
    if len(members) < 3 or len(set(members)) != len(members):
        raise TraceletLoweringError(
            "ring-system action needs at least three unique members"
        )
    if not new_members.issubset(members):
        raise TraceletLoweringError("ring-system growth inserts a non-member slot")
    for slot in members:
        if slot in new_members:
            if not 0 <= slot < state.n_atoms or int(state.atom_types[slot]) != NULL_IDX:
                raise TraceletLoweringError(
                    "ring-system growth requires an unoccupied insertion slot"
                )
        elif not _real_atom(state, slot):
            raise TraceletLoweringError("ring-system action refers to a non-real atom")
    return set(members)


def _state_graph(state: MolecularGraph) -> nx.Graph:
    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    graph = nx.Graph()
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (a, b)
        for offset, a in enumerate(real)
        for b in real[offset + 1 :]
        if int(state.bonds[a, b]) != 0
    )
    return graph


def _cyclic_atoms(state: MolecularGraph) -> set[int]:
    graph = _state_graph(state)
    bridges = {_edge(a, b) for a, b in nx.bridges(graph)}
    return {
        int(vertex)
        for a, b in graph.edges()
        if _edge(a, b) not in bridges
        for vertex in (a, b)
    }


def _cycle_rank(state: MolecularGraph) -> int:
    graph = _state_graph(state)
    if graph.number_of_nodes() == 0:
        return 0
    return int(
        graph.number_of_edges()
        - graph.number_of_nodes()
        + nx.number_connected_components(graph)
    )


def _check_aromatic_edge_shape(
    aromatic_edges: Iterable[tuple[int, int]],
    members: set[int],
) -> None:
    normalized = []
    for raw_a, raw_b in aromatic_edges:
        a, b = int(raw_a), int(raw_b)
        if a >= b or a not in members or b not in members:
            raise TraceletLoweringError(
                "aromatic ring-system edges must be normalized member pairs"
            )
        normalized.append((a, b))
    if len(set(normalized)) != len(normalized):
        raise TraceletLoweringError("ring-system aromatic edges repeat")


def _require_exact_ring_system(
    state: MolecularGraph,
    members: set[int],
    *,
    aromatic_edges: Iterable[tuple[int, int]],
) -> None:
    graph = _state_graph(state)
    bridges = {_edge(a, b) for a, b in nx.bridges(graph)}
    cyclic_graph = nx.Graph()
    cyclic_graph.add_edges_from(
        (int(a), int(b))
        for a, b in graph.edges()
        if _edge(a, b) not in bridges
    )
    components = [set(component) for component in nx.connected_components(cyclic_graph)]
    if members not in components:
        raise TraceletLoweringError(
            "action result does not contain exactly the declared ring system"
        )
    perceived = resonance_invariant_bond_classes(state)
    actual_aromatic = {
        (a, b)
        for offset, a in enumerate(sorted(members))
        for b in sorted(members)[offset + 1 :]
        if int(perceived[a, b]) == BOND_AROMATIC
    }
    expected_aromatic = {
        (int(a), int(b)) for a, b in aromatic_edges
    }
    if actual_aromatic != expected_aromatic:
        raise TraceletLoweringError(
            "ring-system action does not realize its aromatic semantic state"
        )


def _invert_micro_program(
    source: MolecularGraph,
    steps: Iterable[MicroInstruction],
) -> tuple[MicroInstruction, ...]:
    current = source
    states = []
    program = tuple(steps)
    for step in program:
        states.append(current)
        current = _execute_micro(current, (step,))
    inverse: list[MicroInstruction] = []
    for state, (rule_name, action) in reversed(tuple(zip(states, program))):
        if rule_name == "bond_insert":
            inverse.append(("bond_delete", micro.BondDelete(action.a, action.b)))
        elif rule_name == "bond_delete":
            inverse.append(
                (
                    "bond_insert",
                    micro.BondInsert(
                        action.a,
                        action.b,
                        int(state.bonds[action.a, action.b]),
                    ),
                )
            )
        elif rule_name == "bond_reorder":
            inverse.append(
                (
                    "bond_reorder",
                    micro.BondReorder(
                        action.a,
                        action.b,
                        int(state.bonds[action.a, action.b]),
                    ),
                )
            )
        elif rule_name == "atom_restate":
            vertex = int(action.v)
            inverse.append(
                (
                    "atom_restate",
                    micro.AtomRestate(
                        vertex,
                        int(state.atom_types[vertex]),
                        int(state.formal_charges[vertex]),
                        int(state.implicit_h_counts[vertex]),
                    ),
                )
            )
        elif rule_name == "atom_insert":
            inverse.append(("atom_delete", micro.AtomDelete(int(action.slot))))
        elif rule_name == "atom_delete":
            vertex = int(action.v)
            neighbors = tuple(
                (int(neighbor), int(state.bonds[vertex, neighbor]))
                for neighbor in np.flatnonzero(state.bonds[vertex] != 0)
            )
            inverse.append(
                (
                    "atom_insert",
                    micro.AtomInsert(
                        slot=vertex,
                        atom_type=int(state.atom_types[vertex]),
                        formal_charge=int(state.formal_charges[vertex]),
                        implicit_h_count=int(state.implicit_h_counts[vertex]),
                        neighbors=neighbors,
                    ),
                )
            )
        else:
            raise TraceletLoweringError(
                f"ring-system program contains unsupported rule {rule_name!r}"
            )
    return tuple(inverse)


def _require_one_ring_system(
    state: MolecularGraph,
    edges: Iterable[frozenset[int]],
) -> None:
    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    graph = nx.Graph()
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (int(a), int(b))
        for a in real
        for b in real
        if b > a and int(state.bonds[a, b]) != 0
    )
    bridges = {_edge(a, b) for a, b in nx.bridges(graph)}
    ring_graph = graph.copy()
    ring_graph.remove_edges_from(tuple(tuple(edge) for edge in bridges))
    component_by_edge: dict[frozenset[int], int] = {}
    for component_id, vertices in enumerate(nx.connected_components(ring_graph)):
        subgraph = ring_graph.subgraph(vertices)
        for a, b in subgraph.edges():
            component_by_edge[_edge(a, b)] = component_id
    ids = {component_by_edge.get(edge) for edge in edges}
    if None in ids or len(ids) != 1:
        raise TraceletLoweringError("bond changes must lie in one cyclic ring system")


def _micro_runtime():
    # Local import avoids a module cycle while the production kernel registers
    # these tracelet actions.
    from compose_v4.rewrite.kernel import (
        RewriteRule,
        RewriteSystem,
        connected_successor_constraint,
    )

    return RewriteSystem(
        rules=(
            RewriteRule(
                "atom_insert",
                micro.AtomInsert,
                micro.is_valid_atom_insert,
                micro.apply_atom_insert,
            ),
            RewriteRule(
                "atom_delete",
                micro.AtomDelete,
                micro.is_valid_atom_delete,
                micro.apply_atom_delete,
            ),
            RewriteRule(
                "atom_restate",
                micro.AtomRestate,
                micro.is_valid_atom_restate,
                micro.apply_atom_restate,
            ),
            RewriteRule(
                "bond_insert",
                micro.BondInsert,
                micro.is_valid_bond_insert,
                micro.apply_bond_insert,
            ),
            RewriteRule(
                "bond_delete",
                micro.BondDelete,
                micro.is_valid_bond_delete,
                micro.apply_bond_delete,
            ),
            RewriteRule(
                "bond_reorder",
                micro.BondReorder,
                micro.is_valid_bond_reorder,
                micro.apply_bond_reorder,
            ),
        ),
        constraints=(connected_successor_constraint,),
    )


def _execute_micro(
    state: MolecularGraph,
    steps: Iterable[MicroInstruction],
) -> MolecularGraph:
    runtime = _micro_runtime()
    current = state
    for rule_name, action in steps:
        current = runtime.apply(current, rule_name, action)
    return current


def _admits_lowering(lower, state: MolecularGraph, action: Any) -> bool:
    try:
        lower(state, action)
        return True
    except Exception:
        return False


def _real_atom(state: MolecularGraph, v: int) -> bool:
    return bool(0 <= int(v) < state.n_atoms and is_element(state.atom_types[int(v)]))


def _h_delta(order: int) -> int:
    return int(BOND_CLASS_TO_H_CHANGE[int(order)])


def _check_h(value: int) -> None:
    if not 0 <= int(value) <= MAX_H_COUNT:
        raise TraceletLoweringError("tracelet lowering needs an invalid transient H count")


def _edge(a: int, b: int) -> frozenset[int]:
    return frozenset((int(a), int(b)))
