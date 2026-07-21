"""Learned chain-continuation rewrite for lipid tails: `AlkylGraft` / `AlkylPrune`.

`AlkylGraft` grafts a run of carbons onto an existing anchor atom in ONE derived
event -- it is `cycle_attach` minus the ring-closure bond (a path, not a cycle).
Unsaturation is the per-unit `bond_orders` (1 = CH2-CH2, 2 = CH=CH); branching is
the same move grafted onto a mid-chain carbon. Grafted atoms are constrained to
CARBON (the specialized-macro typing the factorized model assumes).

This module is additive: it reuses the tracelet lowering helpers and the micro
runtime, and lowers to micro atom-inserts (the executor stays the source of truth). It lowers to N `atom_insert` micro
steps, so the executor remains the single source of truth for validity.

See docs/CHAIN_REWRITE_DESIGN.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compose_v4.chem.molecular_graph import (
    BOND_CLASS_TO_H_CHANGE,
    BOND_SINGLE,
    ELEMENT_TO_IDX,
    MolecularGraph,
)
from compose_v4.rewrite import operators as micro
from compose_v4.rewrite.tracelets import (
    AtomPayload,
    MicroInstruction,
    TraceletLoweringError,
    _check_h,
    _check_new_atom_payload,
    _check_orders,
    _edge,
    _execute_micro,
    _real_atom,
)

_C_IDX = ELEMENT_TO_IDX["C"]
_CARBON_VALENCE = 4


def _hd(order: int) -> int:
    return int(BOND_CLASS_TO_H_CHANGE[int(order)])


@dataclass(frozen=True)
class AlkylGraft:
    """Graft a carbon path onto `anchor`. `atoms[0]` bonds `anchor` by
    `attachment_order`; `atoms[i]` bonds `atoms[i-1]` by `bond_orders[i-1]`.
    `len(bond_orders) == len(atoms) - 1`. No ring-closure bond."""

    anchor: int
    atoms: tuple[AtomPayload, ...]
    bond_orders: tuple[int, ...]
    attachment_order: int = BOND_SINGLE


@dataclass(frozen=True)
class AlkylPrune:
    """Remove a pendant carbon path (reverse of `AlkylGraft`). `slots` are the
    chain atoms in order; `slots[0]` is bonded to `anchor`."""

    anchor: int
    slots: tuple[int, ...]


def _check_alkyl_graft_shape(state: MolecularGraph, action: AlkylGraft) -> None:
    atoms = tuple(action.atoms)
    anchor = int(action.anchor)
    if state.n_real_atoms == 0 or not _real_atom(state, anchor):
        raise TraceletLoweringError("alkyl graft needs an existing anchor")
    if len(atoms) < 1 or len(action.bond_orders) != len(atoms) - 1:
        raise TraceletLoweringError("alkyl graft has inconsistent dimensions")
    slots = tuple(int(a.slot) for a in atoms)
    if len(set(slots)) != len(slots) or anchor in slots:
        raise TraceletLoweringError("alkyl graft repeats an atom")
    for atom in atoms:
        if int(atom.atom_type) != _C_IDX:
            raise TraceletLoweringError("alkyl graft atoms must be carbon")
        _check_new_atom_payload(state, atom)
    _check_orders(action.bond_orders)
    _check_orders((action.attachment_order,))


def lower_alkyl_graft(state: MolecularGraph, action: AlkylGraft) -> tuple[MicroInstruction, ...]:
    _check_alkyl_graft_shape(state, action)
    atoms = action.atoms
    orders = action.bond_orders  # len == len(atoms) - 1
    first = atoms[0]
    # first's implicit-H must add back the not-yet-formed bond to atoms[1]
    first_h = int(first.implicit_h_count) + (_hd(orders[0]) if len(atoms) > 1 else 0)
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
    for index in range(1, len(atoms)):
        atom = atoms[index]
        # add back the bond to atoms[index+1] if it exists (formed at the next insert)
        after = _hd(orders[index]) if index < len(orders) else 0
        insertion_h = int(atom.implicit_h_count) + after
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
    _execute_micro(state, steps)  # authoritative validation via the micro runtime
    return tuple(steps)


def apply_alkyl_graft(state: MolecularGraph, action: AlkylGraft) -> MolecularGraph:
    return _execute_micro(state, lower_alkyl_graft(state, action))


def is_valid_alkyl_graft(state: MolecularGraph, action: AlkylGraft) -> bool:
    try:
        lower_alkyl_graft(state, action)
        return True
    except Exception:
        return False


def lower_alkyl_prune(state: MolecularGraph, action: AlkylPrune) -> tuple[MicroInstruction, ...]:
    anchor = int(action.anchor)
    slots = tuple(int(v) for v in action.slots)
    if not _real_atom(state, anchor):
        raise TraceletLoweringError("alkyl prune anchor must be real")
    if len(slots) < 1 or len(set(slots)) != len(slots) or anchor in slots:
        raise TraceletLoweringError("alkyl prune needs unique chain atoms")
    if any(not _real_atom(state, v) for v in slots):
        raise TraceletLoweringError("alkyl prune refers to a non-real atom")
    if int(state.bonds[anchor, slots[0]]) == 0:
        raise TraceletLoweringError("alkyl prune is missing its attachment edge")
    chain_edges = {_edge(anchor, slots[0])}
    for i in range(len(slots) - 1):
        if int(state.bonds[slots[i], slots[i + 1]]) == 0:
            raise TraceletLoweringError("alkyl prune is missing a chain edge")
        chain_edges.add(_edge(slots[i], slots[i + 1]))
    # every chain atom must be PENDANT: its only bonds are the chain bonds
    for v in slots:
        incident = {_edge(v, int(u)) for u in np.flatnonzero(state.bonds[v] != 0)}
        allowed = {edge for edge in chain_edges if v in edge}
        if incident != allowed:
            raise TraceletLoweringError("alkyl prune target has external edges")
    # delete terminal-inward so each deletion removes a leaf
    steps: list[MicroInstruction] = [
        ("atom_delete", micro.AtomDelete(v)) for v in reversed(slots)
    ]
    _execute_micro(state, steps)
    return tuple(steps)


def apply_alkyl_prune(state: MolecularGraph, action: AlkylPrune) -> MolecularGraph:
    return _execute_micro(state, lower_alkyl_prune(state, action))


def is_valid_alkyl_prune(state: MolecularGraph, action: AlkylPrune) -> bool:
    try:
        lower_alkyl_prune(state, action)
        return True
    except Exception:
        return False


def build_graft(anchor: int, bond_orders, null_slots, attachment_order: int = BOND_SINGLE) -> AlkylGraft:
    """Construct an `AlkylGraft` of `len(bond_orders)+1` carbons with the correct
    final implicit-H per atom. `bond_orders[i]` = bond between atoms[i] and atoms[i+1]
    (1 = CH2-CH2, 2 = CH=CH). Saturated linear tail = all 1s; oleyl = 8x1,2,8x1."""
    bond_orders = tuple(int(o) for o in bond_orders)
    n = len(bond_orders) + 1
    slots = [int(s) for s in null_slots][:n]
    if len(slots) != n:
        raise ValueError("need exactly len(bond_orders)+1 null slots")
    atoms = []
    for i in range(n):
        prev_order = attachment_order if i == 0 else bond_orders[i - 1]
        next_order = bond_orders[i] if i < len(bond_orders) else 0
        implicit_h = _CARBON_VALENCE - prev_order - next_order
        atoms.append(AtomPayload(slot=slots[i], atom_type=_C_IDX, formal_charge=0, implicit_h_count=implicit_h))
    return AlkylGraft(anchor=int(anchor), atoms=tuple(atoms), bond_orders=bond_orders, attachment_order=int(attachment_order))
