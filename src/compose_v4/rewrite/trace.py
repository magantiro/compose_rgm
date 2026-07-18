"""Executable rewrite programs and exact inverse compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.rewrite.kernel import RewriteSystem, default_rewrite_system
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondDelete,
    BondInsert,
    BondReorder,
    BondReroute,
)
from compose_v4.rewrite.tracelets import (
    AtomPayload,
    BondOrderChange,
    CycleAttach,
    CycleDetach,
    CycleDelete,
    CycleInsert,
    RingEarDelete,
    RingEarInsert,
    RingSystemRestate,
    inverse_ring_system_delete,
    inverse_ring_system_grow,
)


@dataclass(frozen=True)
class RewriteStep:
    rule_name: str
    action: Any


@dataclass(frozen=True)
class RewriteTrace:
    source: MolecularGraph
    target: MolecularGraph
    steps: tuple[RewriteStep, ...]
    metadata: dict[str, Any]


def execute_trace(
    source: MolecularGraph,
    steps: Iterable[RewriteStep],
    *,
    system: RewriteSystem | None = None,
    return_states: bool = False,
) -> MolecularGraph | tuple[MolecularGraph, tuple[MolecularGraph, ...]]:
    runtime = system or default_rewrite_system()
    state = source
    states = [source]
    for step in steps:
        state = runtime.apply(state, step.rule_name, step.action)
        states.append(state)
    if return_states:
        return state, tuple(states)
    return state


def inverse_step(source: MolecularGraph, step: RewriteStep) -> RewriteStep:
    """Return the exact marked inverse using the pre-step source state."""

    action = step.action
    if step.rule_name == "atom_insert":
        return RewriteStep("atom_delete", AtomDelete(int(action.slot)))
    if step.rule_name == "atom_delete":
        v = int(action.v)
        neighbors = tuple(
            (int(u), int(source.bonds[v, u]))
            for u in np.flatnonzero(source.bonds[v] != 0)
        )
        return RewriteStep(
            "atom_insert",
            AtomInsert(
                slot=v,
                atom_type=int(source.atom_types[v]),
                formal_charge=int(source.formal_charges[v]),
                implicit_h_count=int(source.implicit_h_counts[v]),
                neighbors=neighbors,
            ),
        )
    if step.rule_name == "atom_restate":
        v = int(action.v)
        return RewriteStep(
            "atom_restate",
            AtomRestate(
                v=v,
                atom_type=int(source.atom_types[v]),
                formal_charge=int(source.formal_charges[v]),
                implicit_h_count=int(source.implicit_h_counts[v]),
            ),
        )
    if step.rule_name == "bond_insert":
        return RewriteStep("bond_delete", BondDelete(int(action.a), int(action.b)))
    if step.rule_name == "bond_delete":
        return RewriteStep(
            "bond_insert",
            BondInsert(
                int(action.a),
                int(action.b),
                int(source.bonds[action.a, action.b]),
            ),
        )
    if step.rule_name == "bond_reorder":
        return RewriteStep(
            "bond_reorder",
            BondReorder(
                int(action.a),
                int(action.b),
                int(source.bonds[action.a, action.b]),
            ),
        )
    if step.rule_name == "bond_reroute":
        return RewriteStep(
            "bond_reroute",
            BondReroute(
                a=int(action.u),
                b=int(action.v),
                u=int(action.a),
                v=int(action.b),
                new_order=int(source.bonds[action.a, action.b]),
            ),
        )
    if step.rule_name == "cycle_insert":
        return RewriteStep(
            "cycle_delete",
            CycleDelete(tuple(int(atom.slot) for atom in action.atoms)),
        )
    if step.rule_name == "cycle_delete":
        slots = tuple(int(v) for v in action.slots)
        return RewriteStep(
            "cycle_insert",
            CycleInsert(
                atoms=tuple(
                    AtomPayload(
                        slot=v,
                        atom_type=int(source.atom_types[v]),
                        formal_charge=int(source.formal_charges[v]),
                        implicit_h_count=int(source.implicit_h_counts[v]),
                    )
                    for v in slots
                ),
                bond_orders=tuple(
                    int(source.bonds[slots[i], slots[(i + 1) % len(slots)]])
                    for i in range(len(slots))
                ),
            ),
        )
    if step.rule_name == "cycle_attach":
        return RewriteStep(
            "cycle_detach",
            CycleDetach(
                anchor=int(action.anchor),
                slots=tuple(int(atom.slot) for atom in action.atoms),
            ),
        )
    if step.rule_name == "cycle_detach":
        anchor = int(action.anchor)
        slots = tuple(int(v) for v in action.slots)
        return RewriteStep(
            "cycle_attach",
            CycleAttach(
                anchor=anchor,
                atoms=tuple(
                    AtomPayload(
                        slot=v,
                        atom_type=int(source.atom_types[v]),
                        formal_charge=int(source.formal_charges[v]),
                        implicit_h_count=int(source.implicit_h_counts[v]),
                    )
                    for v in slots
                ),
                bond_orders=tuple(
                    int(source.bonds[slots[i], slots[(i + 1) % len(slots)]])
                    for i in range(len(slots))
                ),
                attachment_order=int(source.bonds[anchor, slots[0]]),
            ),
        )
    if step.rule_name == "ring_ear_insert":
        return RewriteStep(
            "ring_ear_delete",
            RingEarDelete(
                a=int(action.a),
                b=int(action.b),
                slots=tuple(int(atom.slot) for atom in action.atoms),
            ),
        )
    if step.rule_name == "ring_ear_delete":
        a = int(action.a)
        b = int(action.b)
        slots = tuple(int(v) for v in action.slots)
        path = (a, *slots, b)
        return RewriteStep(
            "ring_ear_insert",
            RingEarInsert(
                a=a,
                b=b,
                atoms=tuple(
                    AtomPayload(
                        slot=v,
                        atom_type=int(source.atom_types[v]),
                        formal_charge=int(source.formal_charges[v]),
                        implicit_h_count=int(source.implicit_h_counts[v]),
                    )
                    for v in slots
                ),
                bond_orders=tuple(
                    int(source.bonds[path[i], path[i + 1]])
                    for i in range(len(path) - 1)
                ),
            ),
        )
    if step.rule_name == "ring_system_restate":
        return RewriteStep(
            "ring_system_restate",
            RingSystemRestate(
                tuple(
                    BondOrderChange(
                        int(change.a),
                        int(change.b),
                        int(source.bonds[change.a, change.b]),
                    )
                    for change in action.changes
                )
            ),
        )
    if step.rule_name == "ring_system_grow":
        return RewriteStep(
            "ring_system_delete",
            inverse_ring_system_grow(source, action),
        )
    if step.rule_name == "ring_system_delete":
        return RewriteStep(
            "ring_system_grow",
            inverse_ring_system_delete(source, action),
        )
    raise ValueError(f"no inverse registered for {step.rule_name!r}")


def invert_trace(
    source: MolecularGraph,
    steps: Iterable[RewriteStep],
    *,
    system: RewriteSystem | None = None,
) -> tuple[RewriteStep, ...]:
    runtime = system or default_rewrite_system()
    step_tuple = tuple(steps)
    _, states = execute_trace(source, step_tuple, system=runtime, return_states=True)
    inverse = [
        inverse_step(states[i], step_tuple[i])
        for i in range(len(step_tuple))
    ]
    inverse.reverse()
    return tuple(inverse)
