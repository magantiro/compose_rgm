"""Which local contexts tolerate which attachments, mined rather than asserted.

WHY THIS EXISTS.  A context-preserving macro protects everything outside its cut, but the
ONE bond it creates is unconstrained, and an unconstrained join invents chemistry that does
not occur.  MEASURED while building a celecoxib initialization bank: joining a sulfonyl-
bearing payload onto a sulfonyl anchor produced `S(=O)(=O)S(=O)(=O)` disulfones in 4 of 8
selected products.  Every one is valence-legal, RDKit-sanitizable and executor-constructible
-- validity closure is a VALENCE guarantee and says nothing about whether a bond environment
occurs in real chemistry.

WHAT IT IS.  A frequency table over the ENVIRONMENT PAIR of a real single bond, taken from a
corpus of real molecules: for each endpoint, its element, aromaticity, heavy degree and the
multiset of its other neighbours' elements.  A proposed join is admissible iff its
environment pair occurs in the table.  This is a chemical prior, not a task signal: the table
is built from molecular structure alone and never reads an oracle score, a task identity or
a reference.

WHY NOT A HAND-WRITTEN RULE.  "No sulfonyl-sulfonyl" would be one patch for one observed
defect.  The table refuses every environment pair the corpus does not contain, and the
corpus decides which those are.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

from rdkit import Chem


def _environment(atom, *, exclude) -> tuple:
    """Element, aromaticity, heavy degree and the other neighbours' elements."""
    neighbours = sorted(n.GetSymbol() for n in atom.GetNeighbors()
                        if n.GetIdx() != exclude and n.GetAtomicNum() > 1)
    return (atom.GetSymbol(), bool(atom.GetIsAromatic()),
            atom.GetDegree(), tuple(neighbours))


def bond_environment_key(mol, begin_idx, end_idx) -> tuple:
    """An order-independent key for the environment pair of one bond."""
    begin, end = mol.GetAtomWithIdx(begin_idx), mol.GetAtomWithIdx(end_idx)
    left = _environment(begin, exclude=end_idx)
    right = _environment(end, exclude=begin_idx)
    return tuple(sorted((left, right), key=repr))


@dataclass(frozen=True)
class AttachmentTable:
    """Environment pairs observed on real acyclic single bonds, with their counts."""

    counts: dict
    molecules: int

    def occurs(self, key, *, min_support: int = 1) -> bool:
        return self.counts.get(key, 0) >= min_support

    def support(self, key) -> int:
        return self.counts.get(key, 0)


def build_attachment_table(smiles_iterable) -> AttachmentTable:
    """Every acyclic single bond of every parseable molecule contributes one key."""
    counts: dict = collections.Counter()
    molecules = 0
    for smiles in smiles_iterable:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        molecules += 1
        for bond in mol.GetBonds():
            if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            counts[bond_environment_key(
                mol, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] += 1
    return AttachmentTable(counts=dict(counts), molecules=molecules)


def new_bond_indices(source_smiles, endpoint_smiles):
    """The bonds present in the endpoint whose environment pair is absent from the source.

    Used to score a macro product without needing the proposal's internal bookkeeping, so
    the check works on any endpoint however it was produced.
    """
    source, endpoint = Chem.MolFromSmiles(source_smiles), Chem.MolFromSmiles(endpoint_smiles)
    if source is None or endpoint is None:
        return None
    seen = collections.Counter()
    for bond in source.GetBonds():
        if not bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE:
            seen[bond_environment_key(
                source, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] += 1
    novel = []
    for bond in endpoint.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        key = bond_environment_key(endpoint, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        if seen[key] > 0:
            seen[key] -= 1
        else:
            novel.append(key)
    return novel


def admissible(source_smiles, endpoint_smiles, table, *, min_support=1) -> bool:
    """True iff every bond environment the macro INTRODUCED occurs in the corpus."""
    novel = new_bond_indices(source_smiles, endpoint_smiles)
    if novel is None:
        return False
    return all(table.occurs(key, min_support=min_support) for key in novel)
