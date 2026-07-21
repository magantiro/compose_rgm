"""The BEAE linker: the invariant connecting core, for the Fig-6 linker-frozen campaign.

A BEAE lipid is (variable head) + (variable tail1) + (variable tail2) fused by a
FIXED linker. That linker is the 12-atom connecting core: the central branch-point
N + the acrylate-derived beta-amino-ester arm + the propiolate-derived E-beta-
enamine-ester arm. It is identical across every BEAE lipid; only the head and the
two tails vary.

Two consumers:
- Fig-6 linker-frozen optimization: `beae_linker_atoms(mol)` returns the atoms to
  FREEZE (the legal-fiber constraint removes any rewrite touching them, so the
  linker survives by construction while the generator optimizes head + tails).
- Analysis/decomposition: `beae_decompose(mol)` splits a BEAE lipid into
  {linker, head, tail1, tail2} at the three attachment points.

The central N is part of the LINKER (a weakly-basic enamine junction), NOT the
head; the ionizable head is whatever amine hangs off the third arm -- see
`region_labels` for the per-atom soft-labeling counterpart.
"""

from __future__ import annotations

from collections import deque

from rdkit import Chem

# maps 1-11 = the frozen linker core; 12/13 = tail attachment carbons; 14 = head attachment carbon.
_BEAE_LINKER_SMARTS = (
    "[NX3:1]([CH2:2][CH2:3][CX3:4](=[OX1:5])[OX2:6][#6:12])"
    "([CH:7]=[CH:8][CX3:9](=[OX1:10])[OX2:11][#6:13])[#6:14]"
)
_PATT = Chem.MolFromSmarts(_BEAE_LINKER_SMARTS)
_CORE_MAPS = frozenset(range(1, 12))            # frozen linker atoms
_ATTACH_MAPS = {"tail1": 12, "tail2": 13, "head": 14}
_CENTRAL_N_MAP = 1
_ESTER_O_MAPS = {"tail1": 6, "tail2": 11}       # the ester O that each tail hangs off


def _mapnum_to_qidx() -> dict[int, int]:
    return {_PATT.GetAtomWithIdx(i).GetAtomMapNum(): i
            for i in range(_PATT.GetNumAtoms())
            if _PATT.GetAtomWithIdx(i).GetAtomMapNum()}


_M2Q = _mapnum_to_qidx()


def is_beae(mol: Chem.Mol) -> bool:
    """Whether the molecule contains the invariant BEAE linker core."""
    return mol.HasSubstructMatch(_PATT)


def _match_by_mapnum(mol: Chem.Mol) -> dict[int, int] | None:
    match = mol.GetSubstructMatch(_PATT)  # mol indices in query-atom order
    if not match:
        return None
    return {mapnum: match[qidx] for mapnum, qidx in _M2Q.items()}


def beae_linker_atoms(mol: Chem.Mol) -> frozenset[int]:
    """The atoms of the invariant 12-atom linker core to FREEZE (11 mapped core atoms;
    the 3 attachment carbons stay editable). Empty if the molecule is not BEAE."""
    mm = _match_by_mapnum(mol)
    if mm is None:
        return frozenset()
    return frozenset(mm[mn] for mn in _CORE_MAPS)


def beae_attachment_points(mol: Chem.Mol) -> dict[str, int]:
    """The first atom of each variable region: {head, tail1, tail2} -> atom index."""
    mm = _match_by_mapnum(mol)
    return {} if mm is None else {name: mm[mn] for name, mn in _ATTACH_MAPS.items()}


def _grow_region(mol: Chem.Mol, start: int, blocked: set[int]) -> set[int]:
    """BFS outward from an attachment atom, never crossing into the frozen linker."""
    seen = {start}
    q: deque[int] = deque([start])
    while q:
        i = q.popleft()
        for nbr in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nbr.GetIdx()
            if j not in seen and j not in blocked:
                seen.add(j)
                q.append(j)
    return seen


def beae_decompose(mol: Chem.Mol) -> dict[str, frozenset[int]] | None:
    """Partition a BEAE lipid into {linker, head, tail1, tail2} atom sets."""
    mm = _match_by_mapnum(mol)
    if mm is None:
        return None
    linker = {mm[mn] for mn in _CORE_MAPS}
    out = {"linker": frozenset(linker)}
    for name in ("head", "tail1", "tail2"):
        anchor = mm[_ATTACH_MAPS[name]]
        out[name] = frozenset(_grow_region(mol, anchor, blocked=linker) - linker)
    return out
