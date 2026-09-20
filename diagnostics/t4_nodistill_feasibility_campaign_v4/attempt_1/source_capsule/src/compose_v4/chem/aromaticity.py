"""Resonance-invariant neural bond views of executable molecular states."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_NULL,
    BOND_SINGLE,
    BOND_TRIPLE,
    IDX_TO_ELEMENT,
    MolecularGraph,
    is_element,
    molecular_graph_to_smiles,
)


def resonance_invariant_bond_classes(state: MolecularGraph) -> np.ndarray:
    """Return bond classes with perceived aromatic systems labeled class four.

    The executable state is not changed: it remains an integer-order Kekule
    graph for exact valence updates and rewrite lowering.  This function builds
    an index-preserving RDKit molecule, perceives aromaticity on the complete
    valid state, and returns a neural edge view in which every perceived
    aromatic bond has class four.  Nonaromatic single/double/triple bonds retain
    their operational classes.
    """

    bonds = state.bonds.copy()
    real_slots = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    if not real_slots:
        return bonds

    aromatic_atoms = (state.bonds == BOND_AROMATIC).any(axis=1)
    editable = Chem.RWMol()
    slot_to_rdkit = {}
    rdkit_to_slot = {}
    for slot in real_slots:
        atom = Chem.Atom(IDX_TO_ELEMENT[int(state.atom_types[slot])])
        atom.SetFormalCharge(int(state.formal_charges[slot]))
        atom.SetNumExplicitHs(int(state.implicit_h_counts[slot]))
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(bool(aromatic_atoms[slot]))
        rdkit_index = editable.AddAtom(atom)
        slot_to_rdkit[slot] = rdkit_index
        rdkit_to_slot[rdkit_index] = slot

    bond_types = {
        BOND_SINGLE: Chem.BondType.SINGLE,
        BOND_DOUBLE: Chem.BondType.DOUBLE,
        BOND_TRIPLE: Chem.BondType.TRIPLE,
        BOND_AROMATIC: Chem.BondType.AROMATIC,
    }
    for left_offset, left in enumerate(real_slots):
        for right in real_slots[left_offset + 1 :]:
            bond_class = int(state.bonds[left, right])
            if bond_class == BOND_NULL:
                continue
            editable.AddBond(
                slot_to_rdkit[left],
                slot_to_rdkit[right],
                bond_types[bond_class],
            )

    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    for bond in molecule.GetBonds():
        if not bond.GetIsAromatic():
            continue
        left = rdkit_to_slot[bond.GetBeginAtomIdx()]
        right = rdkit_to_slot[bond.GetEndAtomIdx()]
        bonds[left, right] = bonds[right, left] = BOND_AROMATIC
    return bonds


def perceived_aromatic_bond_count(state: MolecularGraph) -> int:
    """Count undirected bonds in RDKit-perceived aromatic systems."""

    perceived = resonance_invariant_bond_classes(state)
    return int(np.count_nonzero(np.triu(perceived == BOND_AROMATIC, k=1)))


def perceived_aromatic_ring_count(state: MolecularGraph) -> int:
    """Count RDKit-perceived aromatic rings in a valid executable state."""

    smiles = molecular_graph_to_smiles(state)
    if smiles is None:
        raise ValueError("cannot perceive aromatic rings in an invalid state")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("serialized molecular state did not parse")
    return int(rdMolDescriptors.CalcNumAromaticRings(molecule))


__all__ = [
    "perceived_aromatic_bond_count",
    "perceived_aromatic_ring_count",
    "resonance_invariant_bond_classes",
]
