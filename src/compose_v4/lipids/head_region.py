"""Lightweight ionizable-head detection (RDKit only, no oracle/joblib deps).

Shared by the oracle's head-aware AD gate AND the generator's region labeler, so
region-awareness in the (Modal, minimal-deps) training image does not pull in the
oracle's heavy dependencies. The definition of a protonatable head is here so both
consumers agree on it; the BEAE central enamine N is excluded (a vinylogous amide,
weakly basic linker junction, not the head).
"""

from __future__ import annotations

from collections import deque

from rdkit import Chem

# A protonatable ionizable-head amine: exclude amides, imines, cations, AND
# vinylogous amides / enamine-esters (N-C=C-C=O, e.g. the BEAE central N).
_BASIC_AMINE = Chem.MolFromSmarts(
    "[NX3;!$(NC=O);!$(N=*);!$([N+]);!$([NX3][CX3]=[CX3][CX3]=[OX1])]"
)
_HEAD_RADIUS_BONDS = 3


def basic_nitrogens(mol: Chem.Mol) -> list[int]:
    return [m[0] for m in mol.GetSubstructMatches(_BASIC_AMINE)]


def head_region_atoms(mol: Chem.Mol, radius: int = _HEAD_RADIUS_BONDS) -> set[int]:
    """Atoms within `radius` bonds of any basic amine N, plus whole ring systems
    touched. This is the ionizable-head neighborhood (excludes long tails)."""
    starts = basic_nitrogens(mol)
    if not starts:
        return set()
    keep: set[int] = set()
    ring_info = mol.GetRingInfo().AtomRings()
    for start in starts:
        seen = {start: 0}
        queue: deque[int] = deque([start])
        while queue:
            idx = queue.popleft()
            depth = seen[idx]
            keep.add(idx)
            if depth >= radius:
                continue
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                j = nbr.GetIdx()
                if j not in seen:
                    seen[j] = depth + 1
                    queue.append(j)
    for ring in ring_info:
        if keep & set(ring):
            keep.update(ring)
    return keep
