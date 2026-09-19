"""State helpers for padded, trans-dimensional molecular graphs."""

from __future__ import annotations

import numpy as np

from compose_v4.chem.molecular_graph import (
    MolecularGraph,
    NULL_IDX,
    is_element,
    is_rdkit_valid,
    per_atom_valence_check,
)


def empty_molecular_graph(n_slots: int) -> MolecularGraph:
    """Return the formal null source state with ``n_slots`` available slots."""

    if n_slots <= 0:
        raise ValueError("n_slots must be positive")
    return MolecularGraph(
        atom_types=np.full(n_slots, NULL_IDX, dtype=np.int32),
        formal_charges=np.zeros(n_slots, dtype=np.int32),
        implicit_h_counts=np.zeros(n_slots, dtype=np.int32),
        bonds=np.zeros((n_slots, n_slots), dtype=np.int32),
    )


def pad_molecular_graph(mg: MolecularGraph, n_slots: int) -> MolecularGraph:
    """Pad ``mg`` with chemically inert NULL slots without changing its state."""

    if n_slots < mg.n_atoms:
        raise ValueError("n_slots cannot be smaller than the current slot count")
    if n_slots == mg.n_atoms:
        return MolecularGraph(
            mg.atom_types.copy(),
            mg.formal_charges.copy(),
            mg.implicit_h_counts.copy(),
            mg.bonds.copy(),
        )
    pad = n_slots - mg.n_atoms
    atom_types = np.concatenate(
        [mg.atom_types, np.full(pad, NULL_IDX, dtype=mg.atom_types.dtype)]
    )
    formal_charges = np.concatenate(
        [mg.formal_charges, np.zeros(pad, dtype=mg.formal_charges.dtype)]
    )
    implicit_h = np.concatenate(
        [mg.implicit_h_counts, np.zeros(pad, dtype=mg.implicit_h_counts.dtype)]
    )
    bonds = np.zeros((n_slots, n_slots), dtype=mg.bonds.dtype)
    bonds[: mg.n_atoms, : mg.n_atoms] = mg.bonds
    return MolecularGraph(atom_types, formal_charges, implicit_h, bonds)


def is_valid_state(mg: MolecularGraph) -> bool:
    """Validity predicate shared by rewrite validators and the runtime.

    The all-NULL graph is a distinguished valid source state. Every non-empty
    state must satisfy the exact local valence checks and RDKit sanitization.
    """

    real = is_element(mg.atom_types)
    if not bool(real.any()):
        return bool(
            np.all(mg.bonds == 0)
            and np.all(mg.formal_charges == 0)
            and np.all(mg.implicit_h_counts == 0)
        )
    return bool(per_atom_valence_check(mg).all() and is_rdkit_valid(mg))


def is_connected_or_null(mg: MolecularGraph) -> bool:
    """Return whether the real-atom graph is connected or formally null."""

    real_slots = [int(v) for v in np.flatnonzero(is_element(mg.atom_types))]
    if not real_slots:
        return is_valid_state(mg)
    real = set(real_slots)
    seen = {real_slots[0]}
    stack = [real_slots[0]]
    while stack:
        v = stack.pop()
        for u in np.flatnonzero(mg.bonds[v] != 0):
            u = int(u)
            if u in real and u not in seen:
                seen.add(u)
                stack.append(u)
    return seen == real
