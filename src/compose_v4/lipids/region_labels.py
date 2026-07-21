"""Per-atom head / linker / tail region labels for lipid-aware generation.

A lipid has functionally distinct regions with characteristically different
heteroatom abundances: the ionizable **head** is nitrogen-rich, the **linker(s)**
are the O/N/S/P-dense degradable junctions, and the **tails** are near-pure
carbon. A lipid-native generator should treat them differently -- both as encoder
context (a region embedding) and, more directly, via a region-conditioned
element-insertion prior (the right atom to add depends on the region).

This module assigns each atom a region by structural chemistry, reusing the
oracle's head detection so the oracle and generator share one definition of
head/linker/tail. It is a deterministic function of the molecular graph, so it is
computable both from a complete target (training) and from a partial molecule
during ancestral generation (the region emerges as the molecule grows).
"""

from __future__ import annotations

from rdkit import Chem

from compose_v4.oracles.head_domain import head_region_atoms

# region codes (stable ordering for embeddings / prior tables)
OTHER, HEAD, LINKER, TAIL = 0, 1, 2, 3
REGION_NAMES = {OTHER: "other", HEAD: "head", LINKER: "linker", TAIL: "tail"}

# degradable / connecting linker motifs -- the heteroatom-dense junctions.
_LINKER_SMARTS = {
    "ester": "[CX3](=[OX1])[OX2][#6]",
    "amide": "[CX3](=[OX1])[NX3]",
    "carbonate": "[OX2][CX3](=[OX1])[OX2]",
    "carbamate": "[NX3][CX3](=[OX1])[OX2]",
    "urea": "[NX3][CX3](=[OX1])[NX3]",
    "disulfide": "[SX2][SX2]",
    "thioether": "[#6][SX2][#6]",
    "acetal": "[CX4]([OX2])[OX2]",
    "phosphate": "[PX4](=[OX1])([OX2])[OX2]",
    "ether": "[#6;!$([CX3]=[OX1])][OX2][#6;!$([CX3]=[OX1])]",  # non-ester C-O-C
}
_LINKER_PATTS = {k: Chem.MolFromSmarts(v) for k, v in _LINKER_SMARTS.items()}
_MIN_TAIL_RUN = 3  # aliphatic carbon run length to count as a tail (not a short connector)


def _linker_atoms(mol: Chem.Mol) -> set[int]:
    keep: set[int] = set()
    for patt in _LINKER_PATTS.values():
        for match in mol.GetSubstructMatches(patt):
            keep.update(match)
    return keep


def _tail_atoms(mol: Chem.Mol, exclude: set[int]) -> set[int]:
    """Aliphatic, non-ring carbons outside head/linker that sit in a chain of
    length >= _MIN_TAIL_RUN (the alkyl tails; short connectors stay 'other')."""
    candidate = {
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and not a.GetIsAromatic() and not a.IsInRing()
        and a.GetIdx() not in exclude
    }
    # connected components over candidate carbons
    tails: set[int] = set()
    unvisited = set(candidate)
    while unvisited:
        seed = unvisited.pop()
        comp = {seed}
        stack = [seed]
        while stack:
            i = stack.pop()
            for nbr in mol.GetAtomWithIdx(i).GetNeighbors():
                j = nbr.GetIdx()
                if j in candidate and j not in comp:
                    comp.add(j)
                    unvisited.discard(j)
                    stack.append(j)
        if len(comp) >= _MIN_TAIL_RUN:
            tails.update(comp)
    return tails


def lipid_region_labels(mol: Chem.Mol) -> list[int]:
    """Return a per-atom region code in {OTHER, HEAD, LINKER, TAIL}.

    Precedence head > linker > tail > other: the ionizable head wins (an amide
    carbonyl next to the amine is head context), then degradable linker motifs,
    then long aliphatic tails, else short connectors / unclassified.
    """
    n = mol.GetNumAtoms()
    labels = [OTHER] * n
    head = head_region_atoms(mol)
    linker = _linker_atoms(mol) - head
    tail = _tail_atoms(mol, exclude=head | linker)
    for i in head:
        labels[i] = HEAD
    for i in linker:
        labels[i] = LINKER
    for i in tail:
        labels[i] = TAIL
    return labels


def region_label_counts(mol: Chem.Mol) -> dict[str, int]:
    labels = lipid_region_labels(mol)
    return {name: labels.count(code) for code, name in REGION_NAMES.items()}
