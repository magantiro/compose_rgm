"""Head-aware applicability-domain gate for the pan-lung oracle.

Milestone 4 established that the ionizable-head axis does not transfer
(held-head Spearman ~0.16), while tails/pKa do. The honest consequence is not
"the oracle is weak" but "the oracle has a domain": it ranks confidently within
KNOWN ionizable heads (a large space -- ~435 LNPDB head SMILES, pKa 7.2-10.6)
and ABSTAINS on novel heads, which then trigger a small active-learning round.

This module extracts a candidate's ionizable-head region, compares it to the
frozen known-head domain (head-scaffold Morgan similarity + intrinsic head-pKa
range), and returns an admit/abstain decision. It does not import torch; head
pKa, when used, is supplied from the MolGpKa cache/provenance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

# A protonatable ionizable-head amine: exclude amides, imines, cations, AND
# vinylogous amides / enamine-esters (N-C=C-C=O, e.g. the BEAE central N) -- those
# are conjugated into a carbonyl and are weakly basic linker junctions, not the head.
_BASIC_AMINE = Chem.MolFromSmarts(
    "[NX3;!$(NC=O);!$(N=*);!$([N+]);!$([NX3][CX3]=[CX3][CX3]=[OX1])]"
)
_HEAD_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
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
    # pull in any ring fully/partly reached so head heterocycles stay intact
    for ring in ring_info:
        if keep & set(ring):
            keep.update(ring)
    return keep


def head_region_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    atoms = head_region_atoms(mol)
    if not atoms:
        return None
    with rdBase.BlockLogs():
        try:
            frag = Chem.PathToSubmol(mol, [b.GetIdx() for b in mol.GetBonds()
                                           if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms])
            if frag.GetNumAtoms() == 0:
                frag = mol
            else:
                frag.UpdatePropertyCache(strict=False)
                Chem.FastFindRings(frag)
        except Exception:
            frag = mol
    return _HEAD_FP.GetFingerprint(frag)


def head_region_smiles(smiles: str) -> str | None:
    """Canonical SMILES of the ionizable-head region (for a serializable domain)."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    atoms = head_region_atoms(mol)
    if not atoms:
        return None
    with rdBase.BlockLogs():
        try:
            frag = Chem.PathToSubmol(mol, [b.GetIdx() for b in mol.GetBonds()
                                           if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms])
            if frag.GetNumAtoms() == 0:
                return None
            frag.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(frag)
            return Chem.MolToSmiles(frag)
        except Exception:
            return None


@dataclass(frozen=True)
class HeadDomain:
    """Frozen known-head applicability domain."""

    head_fingerprints: tuple
    pka_min: float
    pka_max: float
    similarity_threshold: float
    n_known_heads: int

    def admission(self, candidate_smiles: str, head_pka: float | None = None) -> dict:
        fp = head_region_fingerprint(candidate_smiles)
        reasons = []
        if fp is None:
            return {"admitted": False, "reasons": ["no ionizable head found"],
                    "max_head_similarity": None, "head_pka": head_pka}
        sims = DataStructs.BulkTanimotoSimilarity(fp, list(self.head_fingerprints))
        max_sim = float(max(sims)) if sims else 0.0
        if max_sim < self.similarity_threshold:
            reasons.append(f"head novel (max head-scaffold Tanimoto {max_sim:.2f} < {self.similarity_threshold})")
        if head_pka is not None and not (self.pka_min - 0.5 <= head_pka <= self.pka_max + 0.5):
            reasons.append(f"head pKa {head_pka:.1f} outside training range [{self.pka_min:.1f},{self.pka_max:.1f}]")
        return {
            "admitted": not reasons,
            "max_head_similarity": max_sim,
            "head_pka": head_pka,
            "in_head_pka_range": None if head_pka is None
            else bool(self.pka_min - 0.5 <= head_pka <= self.pka_max + 0.5),
            "reasons": reasons,
            "recommendation": "rank with pKa+tail model" if not reasons
            else "abstain -> small active-learning calibration round",
        }


def unique_head_smiles(training_smiles: list[str]) -> list[str]:
    """Deduplicated head-region SMILES over a training set (the serializable domain)."""
    seen: set[str] = set()
    for smiles in training_smiles:
        head = head_region_smiles(smiles)
        if head and head not in seen:
            seen.add(head)
    return sorted(seen)


def domain_from_head_smiles(head_smiles: list[str], pka_min: float, pka_max: float,
                            similarity_threshold: float = 0.4) -> HeadDomain:
    fps = []
    for head in head_smiles:
        mol = Chem.MolFromSmiles(head)
        if mol is None:
            continue
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
        fps.append(_HEAD_FP.GetFingerprint(mol))
    return HeadDomain(tuple(fps), float(pka_min), float(pka_max), similarity_threshold, len(fps))


def build_head_domain(training_smiles: list[str], pka_values: list[float],
                      similarity_threshold: float = 0.4) -> HeadDomain:
    heads = unique_head_smiles(training_smiles)
    pk = np.array([p for p in pka_values if p is not None], dtype=float)
    return domain_from_head_smiles(heads, float(pk.min()), float(pk.max()), similarity_threshold)
