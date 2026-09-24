"""Context-preserving macro transitions: replace or excise a region, keep the rest exact.

WHY THIS EXISTS.  COMPOSE's local edits are exploitation inside a basin.  Measured on two
independent PMO tasks, a leader can sit in a basin where 64 further charged calls improve
it by EXACTLY ZERO, while a single macro transition to a different basin leaves ordinary
local search productive again.  The transitions that work are not merely BIGGER edits --
the first generic ring graft scored -0.246 on celecoxib because it replaced the correct
pyrazole and kept the wrong thiophene.  What distinguishes a useful macro is that it
leaves already-correct chemistry UNTOUCHED.

THE FAMILY.  Every proposal here cuts a bounded number of acyclic single bonds and
rebuilds, so everything outside the cut set is preserved BY CONSTRUCTION rather than by a
heuristic that can be wrong:

  substituent_replacements   ONE cut    a terminal branch leaves, a retrieved branch arrives
  region_replacements        TWO cuts   an interior region leaves, a retrieved region arrives
  region_excisions           TWO cuts   an interior region leaves and the flanks rejoin

One cut cannot express a move whose payload is fused to what it replaces -- a ring system
and the exocyclic junction carbon attached to it are one unit, and cutting only the
pendant side leaves the junction behind.  That is the measured reason two-cut exists:
on thiothixene the one-cut form reproduced 1 of 2 reference ring systems and the two-cut
form reproduced 2 of 2.

INFORMATION BOUNDARY.  Nothing here reads a task, an oracle, a reference structure or a
similarity to one.  A caller supplies payloads; how those were retrieved is the caller's
concern and is where any information claim has to be made.

MOLZIP GOTCHA, paid for once.  `Chem.molzip` defaults to AtomMapNumber labels while
`FragmentOnBonds` writes ISOTOPE dummies, so the default silently returns a DISCONNECTED
molecule rather than raising -- and a "." guard then discards every join as if nothing
matched.  Every join here sets `MolzipParams.label = MolzipLabel.Isotope`.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from rdkit import Chem

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import is_valid_state, pad_molecular_graph

#: PMO proposal states carry 48 slots; the executor refuses anything else outright.
PMO_SLOTS = 48
#: The executor refuses endpoints above this many heavy atoms.
REPRESENTABLE_HEAVY_ATOMS = 40

_LABEL_SWAP = {1: 2, 2: 1}


@dataclass(frozen=True)
class RegionProposal:
    """One macro transition.  `family` names which of the three produced it."""

    endpoint: str
    family: str
    removed: str
    removed_atoms: int
    installed: str | None
    installed_atoms: int
    heavy_atoms: int
    detail: dict


def _zip_params():
    params = Chem.MolzipParams()
    params.label = Chem.MolzipLabel.Isotope
    return params


def _heavy(mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (0, 1))


def _cuttable(mol) -> list[int]:
    """Acyclic single bonds -- the only bonds a region boundary may cross.

    Cutting a ring bond opens the ring rather than detaching a region, which is a
    different transformation with a different executor cost.

    MEASURED: this filter is REDUNDANT with the fragment-count requirements downstream --
    over three real sources, allowing every single bond produced 0 additional usable
    splits (21/21, 91/91, 55/55), because opening a ring does not disconnect the molecule
    and so never yields the 3 fragments a two-cut split needs. A mutation removing it
    therefore survives the suite BY CONSTRUCTION rather than through a fixture gap. It is
    kept as cheap defence in depth and because it avoids enumerating pairs that cannot
    contribute; do not read its presence as evidence that ring-crossing cuts were a live
    hazard.
    """
    return [bond.GetIdx() for bond in mol.GetBonds()
            if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing()]


def _relabel(mol):
    """Swap the two attachment labels, so a donor can arrive either way round."""
    editable = Chem.RWMol(mol)
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() in _LABEL_SWAP:
            atom.SetIsotope(_LABEL_SWAP[atom.GetIsotope()])
    return editable.GetMol()


def executable_endpoint(mol_or_smiles) -> str | None:
    """The canonical SMILES iff the production executor can actually carry this state.

    RDKit-parseable is NOT the same as COMPOSE-constructible -- radicals parse and cannot
    be built -- so this asks the production validity predicate on a real padded state
    rather than trusting `MolFromSmiles is not None`.
    """
    if isinstance(mol_or_smiles, str):
        smiles = mol_or_smiles
    else:
        try:
            Chem.SanitizeMol(mol_or_smiles)
            smiles = Chem.MolToSmiles(mol_or_smiles)
        except (ValueError, RuntimeError):
            return None
    if not smiles or "." in smiles or "*" in smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() > REPRESENTABLE_HEAVY_ATOMS:
        return None
    try:
        state = pad_molecular_graph(smiles_to_molecular_graph(smiles), PMO_SLOTS)
    except (ValueError, RuntimeError, KeyError, IndexError):
        return None
    return smiles if is_valid_state(state) else None


def _split_one(mol, bond):
    try:
        fragmented = Chem.FragmentOnBonds(mol, [bond], dummyLabels=[(1, 1)])
        parts = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True)
    except (ValueError, RuntimeError):
        return None
    return parts if len(parts) == 2 else None


def _split_two(mol, first, second, *, labels=((1, 1), (2, 2))):
    """Cut two bonds and return (flanks, middle); the middle carries BOTH labels."""
    try:
        fragmented = Chem.FragmentOnBonds(mol, [first, second], dummyLabels=list(labels))
        parts = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True)
    except (ValueError, RuntimeError):
        return None
    if len(parts) != 3:
        return None
    middle, flanks = None, []
    for part in parts:
        marks = sorted(a.GetIsotope() for a in part.GetAtoms() if a.GetAtomicNum() == 0)
        if marks == [1, 2] and middle is None:
            middle = part
        else:
            flanks.append(part)
    if middle is None or len(flanks) != 2:
        return None
    return flanks, middle


def substituent_replacements(source, payloads, *, min_removed=2, max_removed=14,
                             min_kept=8, limit=400):
    """ONE cut: a terminal branch leaves and a retrieved decorated branch arrives.

    `payloads` are `(smiles_with_one_[1*], detail)` pairs.  A payload that is decorated
    matters: a BARE skeleton library can only ever offer the ring, and the measured
    failure it produces is that the correct decorated context is stripped.
    """
    mol = Chem.MolFromSmiles(source)
    if mol is None:
        return []
    made, seen, params = [], {source}, _zip_params()
    for bond in _cuttable(mol):
        parts = _split_one(mol, bond)
        if parts is None:
            continue
        small, big = sorted(parts, key=lambda p: p.GetNumAtoms())
        removed = _heavy(small)
        if not min_removed <= removed <= max_removed or _heavy(big) < min_kept:
            continue
        removed_key = Chem.MolToSmiles(small)
        for payload, detail in payloads:
            if payload == removed_key:
                continue
            arriving = Chem.MolFromSmiles(payload)
            if arriving is None:
                continue
            try:
                joined = Chem.molzip(big, arriving, params)
            except (ValueError, RuntimeError):
                continue
            endpoint = executable_endpoint(joined)
            if endpoint is None or endpoint in seen:
                continue
            seen.add(endpoint)
            made.append(RegionProposal(
                endpoint=endpoint, family="substituent_replace", removed=removed_key,
                removed_atoms=removed, installed=payload,
                installed_atoms=_heavy(arriving),
                heavy_atoms=Chem.MolFromSmiles(endpoint).GetNumHeavyAtoms(),
                detail=dict(detail)))
            if len(made) >= limit:
                return made
    return made


def region_replacements(source, regions, *, min_removed=3, max_removed=24, min_kept=6,
                        limit=600):
    """TWO cuts: an interior region leaves and a retrieved region arrives.

    `regions` are `(smiles_with_[1*]_and_[2*], detail)` pairs.  Both label orientations
    are tried, because which flank received label 1 is an artifact of bond ordering.
    """
    mol = Chem.MolFromSmiles(source)
    if mol is None:
        return []
    arriving = []
    for payload, detail in regions:
        base = Chem.MolFromSmiles(payload)
        if base is not None:
            arriving.append((payload, detail, base))
    made, seen, params = [], {source}, _zip_params()
    for first, second in itertools.combinations(_cuttable(mol), 2):
        split = _split_two(mol, first, second)
        if split is None:
            continue
        flanks, middle = split
        removed = _heavy(middle)
        if not min_removed <= removed <= max_removed:
            continue
        if sum(_heavy(f) for f in flanks) < min_kept:
            continue
        removed_key = Chem.MolToSmiles(middle)
        for payload, detail, base in arriving:
            if payload == removed_key:
                continue
            for orientation, donor in (("as_mined", base), ("flipped", _relabel(base))):
                try:
                    joined = Chem.molzip(Chem.molzip(flanks[0], donor, params),
                                         flanks[1], params)
                except (ValueError, RuntimeError):
                    continue
                endpoint = executable_endpoint(joined)
                if endpoint is None or endpoint in seen:
                    continue
                seen.add(endpoint)
                made.append(RegionProposal(
                    endpoint=endpoint, family="region_replace", removed=removed_key,
                    removed_atoms=removed, installed=payload,
                    installed_atoms=_heavy(base),
                    heavy_atoms=Chem.MolFromSmiles(endpoint).GetNumHeavyAtoms(),
                    detail={**dict(detail), "orientation": orientation}))
                if len(made) >= limit:
                    return made
    return made


def _merge_label(mol, frm=2, to=1):
    """Rewrite one attachment label so two flanks can zip to each other."""
    editable = Chem.RWMol(mol)
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() == frm:
            atom.SetIsotope(to)
    return editable.GetMol()


def region_excisions(source, *, min_removed=1, max_removed=24, min_kept=6, limit=600):
    """TWO cuts: an interior region leaves and the flanks rejoin directly.

    No donor library exists for this family -- the proposal IS the enumeration -- so it
    is the one macro that needs no retrieval at all.

    The two cuts are labelled DISTINCTLY so the middle is identifiable (it is the only
    piece carrying both marks), and the surviving flank's label is then rewritten to
    match the other's so the two can zip.  Cutting both bonds with one shared label
    instead makes the middle indistinguishable from a flank, which yields zero
    proposals -- measured, and the reason the labels are separated here.
    """
    mol = Chem.MolFromSmiles(source)
    if mol is None:
        return []
    made, seen, params = [], {source}, _zip_params()
    for first, second in itertools.combinations(_cuttable(mol), 2):
        split = _split_two(mol, first, second)
        if split is None:
            continue
        flanks, middle = split
        removed = _heavy(middle)
        if not min_removed <= removed <= max_removed:
            continue
        if sum(_heavy(f) for f in flanks) < min_kept:
            continue
        try:
            joined = Chem.molzip(_merge_label(flanks[0]), _merge_label(flanks[1]), params)
        except (ValueError, RuntimeError):
            continue
        endpoint = executable_endpoint(joined)
        if endpoint is None or endpoint in seen:
            continue
        seen.add(endpoint)
        made.append(RegionProposal(
            endpoint=endpoint, family="region_excise",
            removed=Chem.MolToSmiles(middle), removed_atoms=removed, installed=None,
            installed_atoms=0,
            heavy_atoms=Chem.MolFromSmiles(endpoint).GetNumHeavyAtoms(), detail={}))
        if len(made) >= limit:
            return made
    return made


FAMILIES = ("substituent_replace", "region_replace", "region_excise")
