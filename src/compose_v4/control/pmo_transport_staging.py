"""Split a (G, T) transport into stages each under the realization ceiling.

WHY STAGING IS FORCED, not preferred
------------------------------------
Measured over 108 correspondences on 15 declared-structure PMO tasks, a transport from
COMPOSE's initialization bank to a declared target is **scale 19 to 56, median 36**
primitives. The measured realization ceiling on this path is a median of 16 primitives
with a maximum of 23, and teacher-scale plans at 29-40 never bound at all. So most
transports cannot be one program, and the question is whether a sequence of programs can
do it while every intermediate remains a molecule.

This module answers the part that needs no realization machinery: does a staged PATH
EXIST, and does it approach the target monotonically? Nothing here executes a COMPOSE
program. Intermediates are constructed with RDKit from the correspondence, so a failure
here is a failure of the PLAN, and cannot be blamed on the executor.

WHAT A STAGE IS
---------------
The correspondence supplies three ordered operation lists -- deletions (in an order that
keeps survivors connected), bond-order changes inside the retained core, and installations
(ring systems before acyclic substituents). Concatenated, they are the transport. A stage
is a contiguous run of at most ``max_primitives`` of them.

Deletions come first because they free slots and shrink the molecule, and installations
last because a substituent cannot attach to structure that does not exist yet.

THE PREDECLARED FALSIFIER
-------------------------
If staged intermediates cannot be kept connected and valid, or if similarity to the target
does not improve monotonically, then the REALIZATION CEILING -- not the planner -- is what
blocks long-range transport, and that is a statement about the architecture rather than a
bug to fix. It is written down here before the measurement so a negative stays
interpretable.

A NOTE ON MONOTONICITY. Morgan similarity is context-sensitive: removing one atom changes
the environment of every atom within the fingerprint radius. So a prune step can LOWER
similarity to T even while removing structure that T does not contain. Whether it does is
the empirical question, not an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.control.pmo_transport_correspondence import Correspondence

#: The measured realization ceiling: median 16 primitives, maximum 23 observed binding.
#: A SETTING, not an optimum -- and deliberately the observed MAXIMUM rather than the
#: median, so staging is not tuned to look easy.
DEFAULT_MAX_PRIMITIVES = 23

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@dataclass(frozen=True)
class Stage:
    """One sub-transport: the operations it performs and the molecule it produces."""

    index: int
    deletes: tuple[int, ...]
    bond_changes: tuple[tuple[int, int], ...]
    installs: tuple[int, ...]
    endpoint_smiles: str | None
    #: False when the stage could not be built into a single valid connected molecule.
    is_valid: bool
    is_connected: bool
    similarity_to_target: float | None

    @property
    def size(self) -> int:
        return len(self.deletes) + len(self.bond_changes) + len(self.installs)

    def payload(self) -> dict:
        return {
            "index": self.index,
            "deletes": list(self.deletes),
            "bond_changes": [list(pair) for pair in self.bond_changes],
            "installs": list(self.installs),
            "endpoint_smiles": self.endpoint_smiles,
            "is_valid": self.is_valid,
            "is_connected": self.is_connected,
            "similarity_to_target": self.similarity_to_target,
            "size": self.size,
        }


def _without_stereo(mol):
    """A copy with every stereo annotation removed."""
    flat = Chem.Mol(mol)
    Chem.RemoveStereochemistry(flat)
    return flat


def _similarity(smiles: str, target_fingerprint) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(
        DataStructs.TanimotoSimilarity(target_fingerprint, _GEN.GetFingerprint(mol))
    )


def build_intermediate(
    correspondence: Correspondence, *, deleted: int, changed: int, installed: int
) -> str | None:
    """The molecule after the first `deleted` deletions, `changed` bond changes and
    `installed` installations.

    Built FRESH from the source each time rather than by mutating one editable molecule,
    because ``RemoveAtom`` renumbers every index above the one removed and a running map
    is the kind of bookkeeping that silently produces a plausible wrong molecule.

    Returns None when the result does not sanitize -- which is itself a finding, not an
    error to swallow: it means the staged plan passes through a non-molecule.
    """
    source = Chem.MolFromSmiles(correspondence.source_smiles)
    target = Chem.MolFromSmiles(correspondence.target_smiles)
    # Build from KEKULE bond orders and let sanitization re-perceive aromaticity.
    # Copying aromatic bonds directly does not work: a ring installed WITHOUT its
    # substituents is not aromatic on its own -- a bare pyrazole whose N-substituent has
    # not arrived yet cannot be kekulized -- so forcing the target's aromatic flags
    # produces a fragment RDKit rejects. Integer bond orders are also what COMPOSE's own
    # executable states carry.
    kekule_source = Chem.Mol(source)
    kekule_target = Chem.Mol(target)
    Chem.Kekulize(kekule_source, clearAromaticFlags=True)
    Chem.Kekulize(kekule_target, clearAromaticFlags=True)
    source, target_bonds = kekule_source, kekule_target
    editable = Chem.RWMol(source)

    # ---- deletions, highest index first so earlier indices stay valid ----
    victims = set(correspondence.delete_order[:deleted])
    for index in sorted(victims, reverse=True):
        editable.RemoveAtom(index)
    # Surviving source atoms, in their new numbering.
    survivors = [i for i in range(source.GetNumAtoms()) if i not in victims]
    source_to_new = {old: new for new, old in enumerate(survivors)}

    # ---- bond-order changes inside the retained core ----
    for change in correspondence.core_bond_changes[:changed]:
        begin, end = change.source_begin, change.source_end
        if begin in source_to_new and end in source_to_new:
            bond = editable.GetBondBetweenAtoms(source_to_new[begin], source_to_new[end])
            if bond is not None:
                mirrored = target_bonds.GetBondBetweenAtoms(
                    *[dict(correspondence.core_map)[a] for a in (begin, end)]
                )
                if mirrored is not None:
                    bond.SetBondType(mirrored.GetBondType())

    # ---- installations, bonded to whatever of T is already present ----
    forward = dict(correspondence.core_map)  # source index -> target index
    target_to_new = {
        forward[old]: new for old, new in source_to_new.items() if old in forward
    }
    for target_index in correspondence.install_order[:installed]:
        atom = target_bonds.GetAtomWithIdx(target_index)
        fresh = Chem.Atom(atom.GetAtomicNum())
        fresh.SetFormalCharge(atom.GetFormalCharge())
        fresh.SetNoImplicit(False)
        new_index = editable.AddAtom(fresh)
        target_to_new[target_index] = new_index
        for neighbour in atom.GetNeighbors():
            other = neighbour.GetIdx()
            if other in target_to_new and other != target_index:
                bond = target_bonds.GetBondBetweenAtoms(target_index, other)
                if editable.GetBondBetweenAtoms(new_index, target_to_new[other]) is None:
                    editable.AddBond(
                        new_index, target_to_new[other], bond.GetBondType()
                    )

    molecule = editable.GetMol()
    try:
        Chem.SanitizeMol(molecule)
    except Exception:  # noqa: BLE001 - a non-sanitizing intermediate IS the finding
        return None
    return Chem.MolToSmiles(molecule)


def _atomic_groups(mol, ordered: tuple[int, ...]) -> list[list[int]]:
    """Group an atom ordering so no RING SYSTEM is split across a stage boundary.

    Measured, and this is why the grouping exists: chunking celecoxib's transport at a
    flat 23 operations cut an aromatic ring in half, and the intermediate would not
    sanitize -- "Can't kekulize mol. Unkekulized atoms: 6 7 8 9 10". Half an aromatic ring
    is not a molecule, so a stage boundary there produces a state no executor could
    commit.

    Atoms of one ring system therefore move together. Acyclic atoms stay individually
    divisible, so this constrains the split as little as it can while keeping every
    intermediate a molecule.
    """
    rings = mol.GetRingInfo().AtomRings()
    # Fuse rings that share atoms into one system.
    systems: list[set[int]] = []
    for ring in rings:
        merged = set(ring)
        rest = []
        for system in systems:
            if system & merged:
                merged |= system
            else:
                rest.append(system)
        rest.append(merged)
        systems = rest
    owner = {atom: index for index, system in enumerate(systems) for atom in system}

    groups: list[list[int]] = []
    emitted: set[int] = set()
    for atom in ordered:
        if atom in emitted:
            continue
        system = owner.get(atom)
        if system is None:
            groups.append([atom])
            emitted.add(atom)
            continue
        # Take the whole system, in the order the caller gave, then anything left of it.
        members = [a for a in ordered if owner.get(a) == system and a not in emitted]
        members += [a for a in sorted(systems[system]) if a not in emitted and a not in members]
        groups.append(members)
        emitted.update(members)
    return groups


def split(
    correspondence: Correspondence, *, max_primitives: int = DEFAULT_MAX_PRIMITIVES
) -> list[Stage]:
    """Cut the transport into stages of at most `max_primitives` operations.

    Boundaries respect ring systems: a stage never contains part of a ring, because a
    partial ring does not sanitize. A single ring system larger than `max_primitives`
    would therefore force an oversized stage; `validate_staging` reports
    `respects_ceiling` so that case is visible rather than silently accepted.
    """
    if max_primitives < 1:
        raise ValueError("a stage must hold at least one primitive")
    target = Chem.MolFromSmiles(correspondence.target_smiles)
    target_fingerprint = _GEN.GetFingerprint(target)

    source = Chem.MolFromSmiles(correspondence.source_smiles)
    units: list[list[tuple[str, object]]] = []
    for group in _atomic_groups(source, correspondence.delete_order):
        units.append([("delete", index) for index in group])
    units.append(
        [
            ("bond", (change.source_begin, change.source_end))
            for change in correspondence.core_bond_changes
        ]
    )
    for group in _atomic_groups(target, correspondence.install_order):
        units.append([("install", index) for index in group])
    units = [unit for unit in units if unit]

    chunks: list[list[tuple[str, object]]] = []
    current: list[tuple[str, object]] = []
    for unit in units:
        if current and len(current) + len(unit) > max_primitives:
            chunks.append(current)
            current = []
        current.extend(unit)
    if current:
        chunks.append(current)

    stages: list[Stage] = []
    deleted = changed = installed = 0
    for chunk in chunks:
        deletes = tuple(value for kind, value in chunk if kind == "delete")
        bonds = tuple(value for kind, value in chunk if kind == "bond")
        installs = tuple(value for kind, value in chunk if kind == "install")
        deleted += len(deletes)
        changed += len(bonds)
        installed += len(installs)
        smiles = build_intermediate(
            correspondence, deleted=deleted, changed=changed, installed=installed
        )
        connected = False
        if smiles is not None:
            connected = "." not in smiles
        stages.append(
            Stage(
                index=len(stages),
                deletes=deletes,
                bond_changes=bonds,
                installs=installs,
                endpoint_smiles=smiles,
                is_valid=smiles is not None,
                is_connected=connected,
                similarity_to_target=(
                    _similarity(smiles, target_fingerprint) if smiles else None
                ),
            )
        )
    return stages


def validate_staging(
    correspondence: Correspondence, stages: list[Stage], *, max_primitives: int
) -> dict:
    """Every offline property the staged plan must have, reported as data.

    `reaches_target` is canonical-SMILES equality, not a similarity threshold: "close to
    the target" is not "is the target", and a staging that stops short must say so.
    """
    target = Chem.MolFromSmiles(correspondence.target_smiles)
    # STEREOCHEMISTRY IS OUT OF SCOPE, and comparing with it in makes a correct transport
    # read as a failure. COMPOSE's MolecularGraph carries no stereo -- the T4 contract
    # states `stereochemistry_claim: false` -- so a rebuilt molecule can never reproduce
    # `[C@H]`. mestranol_similarity is the case that surfaced it. Both comparisons are
    # reported: the stereo-blind one is the verdict, the strict one is the diagnosis.
    canonical_target = Chem.MolToSmiles(target)
    flat_target = Chem.MolToSmiles(_without_stereo(target))
    # The trajectory MUST start at the SOURCE. Measuring monotonicity over stage
    # endpoints alone cannot see an initial DIP -- the prune phase strips structure before
    # the install phase rebuilds it, so similarity can fall below where the run started.
    # Omitting the source made celecoxib read as monotone when its first stage is below
    # its own starting point, and a controller selecting on score would abandon exactly
    # that intermediate.
    source_similarity = _similarity(correspondence.source_smiles, _GEN.GetFingerprint(target))
    similarities = [source_similarity] if source_similarity is not None else []
    similarities += [s.similarity_to_target for s in stages if s.similarity_to_target is not None]
    # A monotonicity verdict over fewer than two points is VACUOUS -- it is True by
    # construction and says nothing. Report it as unevaluated rather than as a pass;
    # a single-point "monotone: True" beside an invalid intermediate reads as success.
    monotone = (
        None
        if len(similarities) < 2
        else all(earlier <= later + 1e-9 for earlier, later in pairwise(similarities))
    )
    final = stages[-1].endpoint_smiles if stages else None
    final_mol = Chem.MolFromSmiles(final) if final else None
    flat_final = Chem.MolToSmiles(_without_stereo(final_mol)) if final_mol else None
    return {
        "stages": len(stages),
        "max_stage_size": max((s.size for s in stages), default=0),
        "respects_ceiling": all(s.size <= max_primitives for s in stages),
        "all_intermediates_valid": all(s.is_valid for s in stages),
        "all_intermediates_connected": all(s.is_connected for s in stages),
        "similarity_monotone_nondecreasing": monotone,
        "similarity_points": len(similarities),
        "source_similarity": source_similarity,
        "dips_below_source": (
            None
            if source_similarity is None
            else any(
                value < source_similarity - 1e-9
                for value in similarities[1:]
            )
        ),
        "minimum_similarity_on_path": min(similarities) if similarities else None,
        "similarity_trajectory": similarities,
        "reaches_target": flat_final is not None and flat_final == flat_target,
        "reaches_target_with_stereochemistry": final is not None and final == canonical_target,
        "final_smiles": final,
    }
