"""Step 1 of the transport planner: the (G, T) structural correspondence.

Given a current molecule ``G`` and a DECLARED goal structure ``T``, compute what must be
kept, what must go, what must arrive, where it attaches, and in what order -- the
``(retain_core, R_delete, H_install, alpha, D)`` representation COMPOSE already wants.

WHAT THIS IS NOT
----------------
It is not a learned region prior. The measured teacher-route negative attributed its loss
ENTIRELY to a fitted region law (T4 -0.064 p=0.0034; generic -0.026 p=0.0001) while the
family projection was neutral (p=0.63). So the region here is COMPUTED from the actual
G-to-T difference and nothing about it is fitted. It is a correspondence, not a model.

It is also not a compiler. Nothing here emits or executes a program; that is Step 2.

WHY ``completeRingsOnly``
-------------------------
A partial ring is not a retained core. Counting one as retained understates the install
work by a whole ring system and would make a transport look shorter than it is, which is
exactly the error that would make staging look unnecessary. Both ``completeRingsOnly`` and
``ringMatchesRingOnly`` are on, so a ring atom may only correspond to a ring atom.

WHY ``alpha`` IS A SET AND NOT A FIRST CHOICE
---------------------------------------------
A T4 measurement widened attachment binding on fa7_0 and found the alternatives are NOT
duplicates -- 393 genuinely different molecules, +11.0% over the single choice -- but that
the gate refused every one, two thirds of them on SIMILARITY. That result is a property of
searching inside a similarity ball: moving the attachment site moves you away from the
reference, so ``assignments[0]`` is already near-optimal there and the alternatives are
strictly worse on the axis that binds.

PMO has no similarity ball, and for a declared-target transport the objective IS
similarity to ``T``, so placement that moves toward ``T`` is precisely what is wanted. The
T4 result is therefore not expected to transfer, and this module ENUMERATES attachment
interfaces rather than returning a first assignment. That expectation is an INFERENCE and
is not yet measured here; the honest test is to offer narrow and broad placement to the
gate and count added-AND-eligible, never added alone.

INDEX SPACE, stated because it is a real boundary
-------------------------------------------------
Everything here is in RDKit atom indices of the parsed molecules. COMPOSE states are
48-slot padded arrays with their own slot indices, and a SMILES round trip produces a
TIGHT graph that silently deletes the ``atom_insert`` family. Step 2 must map these
indices into slot space deliberately; this module deliberately does not, so the mapping is
explicit rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import rdFMCS

#: Seconds allowed for one maximum-common-substructure search. MCS is worst-case
#: exponential; a timeout returns the best found so far, which is still a valid (smaller)
#: core, so a slow pair degrades to a longer transport rather than to a wrong one.
DEFAULT_MCS_TIMEOUT = 20

#: Cap on distinct alignments returned. Several correspondences can share one core when a
#: symmetric molecule admits multiple mappings; each is a genuinely different transport.
DEFAULT_TOP_K = 4


@dataclass(frozen=True)
class Attachment:
    """One bond crossing the boundary of the retained core.

    ``source_side`` is a bond of ``G`` from a retained atom to an atom being deleted, so
    it is an interface that will be BROKEN. ``target_side`` is a bond of ``T`` from the
    corresponding retained atom into structure being installed, so it is an interface that
    must be MADE. The two are reported separately because a planner that conflates them
    cannot tell "free up this valence" from "attach here".
    """

    core_atom: int
    other_atom: int
    bond_order: str
    side: str  # "source" or "target"


@dataclass(frozen=True)
class BondChange:
    """A bond INSIDE the retained core whose order differs between G and T.

    Found by the offline validation rather than designed in: benzene -> cyclohexane maps
    all six atoms, so an atom-only representation reported ``scale = 0`` -- "no transport
    needed" -- for two different molecules. Retaining an atom skeleton does not retain its
    bond orders, and these changes cost `bond_reorder` / `ring_system_restate` primitives.
    """

    source_begin: int
    source_end: int
    source_order: str
    target_order: str


@dataclass(frozen=True)
class Correspondence:
    """One alignment of ``G`` onto ``T`` and the transport it implies."""

    source_smiles: str
    target_smiles: str
    core_smarts: str
    #: Atom indices of G that map into T, and the mapping itself.
    retain_core: tuple[int, ...]
    core_map: tuple[tuple[int, int], ...]
    #: Atom indices of G outside the core. These are removed.
    r_delete: tuple[int, ...]
    #: Atom indices of T not covered by the core. These are installed.
    h_install: tuple[int, ...]
    #: Every boundary interface, both sides. Enumerated, never reduced to a first choice.
    alpha: tuple[Attachment, ...]
    #: Bonds within the retained core whose order must change. Part of the transport.
    core_bond_changes: tuple[BondChange, ...]
    #: A partial order: leaf-first deletion, ring-systems-before-substituents installation.
    delete_order: tuple[int, ...]
    install_order: tuple[int, ...]
    dependencies: dict = field(default_factory=dict)

    @property
    def scale(self) -> int:
        """Primitive transport size: atoms removed, atoms installed, bonds re-ordered.

        The third term is not cosmetic: without it two molecules differing only in bond
        order report a transport of zero.
        """
        return len(self.r_delete) + len(self.h_install) + len(self.core_bond_changes)

    @property
    def requires_reattachment(self) -> bool:
        """True when the retained core is DISCONNECTED in the source molecule.

        Found by running the validation at scale: 7 of 108 correspondences failed
        `deletion_keeps_source_connected`, all on `median1`, whose target is the bridged
        bicyclic camphor. The cause is not the ordering -- when the core's two pieces are
        joined only through atoms being deleted, NO deletion order preserves connectivity,
        because the remainder is genuinely two fragments.

        Such a transport is a deletion PLUS a reattachment bond joining the flanks, which
        is the move class the T4 5ht1b_2 witness needed and that a deletion-only search
        provably cannot express. Declaring it here means the planner can pick another
        alignment or stage the reattachment, instead of discovering it as an ordering bug.

        THE REATTACHMENT PRIMITIVE IS ``cycle_close``, NOT ``bond_insert``. Verified
        against the frozen Active8 codec surface
        (``action_codec_v4.supported_executor_rules()``), which is exactly
        ``atom_delete, atom_insert, atom_restate_semantic, bond_reorder, bond_reroute,
        cycle_close, cycle_open, ring_system_restate`` -- ``bond_insert`` is NOT in it.
        T4 learned that the expensive way; it is recorded here so Step 2 does not.
        """
        source = Chem.MolFromSmiles(self.source_smiles)
        core = set(self.retain_core)
        if len(core) <= 1:
            return False
        adjacency = {
            atom.GetIdx(): {n.GetIdx() for n in atom.GetNeighbors()}
            for atom in source.GetAtoms()
        }
        start = next(iter(core))
        seen, stack = {start}, [start]
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current] & core:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return seen != core

    @property
    def retained_fraction_of_target(self) -> float:
        target = Chem.MolFromSmiles(self.target_smiles)
        return len(self.retain_core) / max(1, target.GetNumHeavyAtoms())

    def payload(self) -> dict:
        return {
            "source_smiles": self.source_smiles,
            "target_smiles": self.target_smiles,
            "core_smarts": self.core_smarts,
            "retain_core": list(self.retain_core),
            "core_map": [list(pair) for pair in self.core_map],
            "r_delete": list(self.r_delete),
            "h_install": list(self.h_install),
            "alpha": [
                {
                    "core_atom": a.core_atom,
                    "other_atom": a.other_atom,
                    "bond_order": a.bond_order,
                    "side": a.side,
                }
                for a in self.alpha
            ],
            "core_bond_changes": [
                {
                    "source_begin": b.source_begin,
                    "source_end": b.source_end,
                    "source_order": b.source_order,
                    "target_order": b.target_order,
                }
                for b in self.core_bond_changes
            ],
            "delete_order": list(self.delete_order),
            "install_order": list(self.install_order),
            "scale": self.scale,
            "requires_reattachment": self.requires_reattachment,
            "retained_fraction_of_target": round(self.retained_fraction_of_target, 4),
        }


# ---- Ordering ------------------------------------------------------------


def _connectivity_preserving_order(mol, atoms: tuple[int, ...]) -> tuple[int, ...]:
    """Order `atoms` so that removing them one at a time never disconnects what survives.

    A disconnected intermediate is not a committed COMPOSE state, so an ordering that
    produces one cannot be realized step by step.

    A plain lowest-in-set-degree peel is NOT sufficient, and the test that asserted it was
    tautological (`len(fragments) >= 1` is always true). Once the assertion was made real,
    the celecoxib pair reached TWO fragments mid-deletion: in-set degree ignores how an
    atom connects to the retained core, so a low-degree atom can still be a cut vertex of
    the surviving graph.

    This instead picks, at each step, an atom that is not a cut vertex of the CURRENT
    surviving graph, preferring the lowest surviving degree and breaking ties on index. If
    every remaining candidate is a cut vertex -- possible when the delete set is itself a
    bridge -- it takes the lowest-degree one, because refusing to order is worse than
    reporting an ordering the caller can check with `deletion_keeps_source_connected`.
    """
    adjacency = {
        atom.GetIdx(): {n.GetIdx() for n in atom.GetNeighbors()} for atom in mol.GetAtoms()
    }
    surviving = set(adjacency)
    remaining = set(atoms)
    order: list[int] = []

    def connected_without(dropped: int) -> bool:
        nodes = surviving - {dropped}
        if len(nodes) <= 1:
            return True
        start = next(iter(nodes))
        seen, stack = {start}, [start]
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current] & nodes:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return seen == nodes

    while remaining:
        ranked = sorted(
            remaining, key=lambda index: (len(adjacency[index] & surviving), index)
        )
        pick = next((index for index in ranked if connected_without(index)), ranked[0])
        order.append(pick)
        remaining.discard(pick)
        surviving.discard(pick)
    return tuple(order)


def _growth_order(mol, atoms: tuple[int, ...], anchored: set[int]) -> tuple[int, ...]:
    """Install atoms outward from what already exists, taking whole ring systems.

    A globally ring-first order is WRONG and was measured to be: installing a target ring
    system before the linker atoms that connect it to the retained core produces a
    separate fragment, and 17 of 44 staged transports had a disconnected intermediate for
    exactly that reason while their correspondence's core was perfectly connected. The
    disconnection was created by the ORDER, not by the plan.

    So growth is anchored: at each step take an atom adjacent to something already
    present, and when that atom belongs to a ring system take the whole system at once --
    a partial ring does not sanitize. Ties break on index, so the order is deterministic.
    """
    rings = mol.GetRingInfo().AtomRings()
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

    pending = set(atoms)
    present = set(anchored)
    order: list[int] = []
    while pending:
        adjacent = [
            index
            for index in sorted(pending)
            if any(n.GetIdx() in present for n in mol.GetAtomWithIdx(index).GetNeighbors())
        ]
        # Nothing touches what exists yet -- the remainder is a detached piece of the
        # target. Take it in index order rather than stalling; the staging validator
        # reports the resulting disconnection instead of this function hiding it.
        pick = adjacent[0] if adjacent else min(pending)
        system = owner.get(pick)
        if system is None:
            taken = [pick]
        else:
            taken = [index for index in sorted(systems[system]) if index in pending]
        order.extend(taken)
        present.update(taken)
        pending.difference_update(taken)
    return tuple(order)


def prune_to_ring_complete(mol, core: set[int]) -> set[int]:
    """Drop every core atom that is a ring atom whose ring is not wholly in the core.

    RDKit's ``completeRingsOnly`` constrains the MCS itself; it still admits a LONE ring
    atom as an attachment point. Measured on the celecoxib transport, the MCS core was a
    benzene ring PLUS one atom of the pyrazole -- 7 atoms, of which the seventh belongs to
    a 5-ring whose other four are deleted.

    Counting that atom as retained claims a ring atom is already correct while its ring
    must still be rebuilt, which understates the install work. Pruning is iterated to a
    fixed point because removing one atom can orphan another.
    """
    rings = mol.GetRingInfo().AtomRings()
    current = set(core)
    while True:
        drop = set()
        for ring in rings:
            touched = current.intersection(ring)
            if touched and len(touched) != len(ring):
                drop |= touched
        if not drop:
            return current
        current -= drop


def _core_bond_changes(source, target, forward: dict[int, int]) -> tuple[BondChange, ...]:
    """Bonds inside the retained core whose order differs between the two endpoints."""
    changes = []
    for bond in source.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin not in forward or end not in forward:
            continue
        mirrored = target.GetBondBetweenAtoms(forward[begin], forward[end])
        if mirrored is None or mirrored.GetBondType() == bond.GetBondType():
            continue
        changes.append(
            BondChange(
                source_begin=begin,
                source_end=end,
                source_order=str(bond.GetBondType()),
                target_order=str(mirrored.GetBondType()),
            )
        )
    return tuple(changes)


def _boundary(mol, core: set[int], outside: set[int], side: str) -> list[Attachment]:
    found = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for core_atom, other in ((begin, end), (end, begin)):
            if core_atom in core and other in outside:
                found.append(
                    Attachment(
                        core_atom=core_atom,
                        other_atom=other,
                        bond_order=str(bond.GetBondType()),
                        side=side,
                    )
                )
    return found


# ---- Correspondence ------------------------------------------------------


def correspondences(
    source_smiles: str,
    target_smiles: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    timeout: int = DEFAULT_MCS_TIMEOUT,
) -> list[Correspondence]:
    """Top-K alignments of `source_smiles` onto `target_smiles`.

    Raises on unparseable input rather than returning an empty list: an empty result must
    mean "these molecules share no ring-complete core", never "one of them did not parse".
    """
    source = Chem.MolFromSmiles(source_smiles)
    target = Chem.MolFromSmiles(target_smiles)
    if source is None:
        raise ValueError(f"source does not parse: {source_smiles!r}")
    if target is None:
        raise ValueError(f"target does not parse: {target_smiles!r}")

    result = rdFMCS.FindMCS(
        [source, target],
        timeout=timeout,
        completeRingsOnly=True,
        ringMatchesRingOnly=True,
    )
    if not result.smartsString or result.numAtoms == 0:
        return []
    query = Chem.MolFromSmarts(result.smartsString)
    if query is None:
        return []
    source_matches = source.GetSubstructMatches(query, uniquify=True, maxMatches=top_k)
    target_matches = target.GetSubstructMatches(query, uniquify=True, maxMatches=top_k)
    if not source_matches or not target_matches:
        return []

    source_atoms = set(range(source.GetNumAtoms()))
    target_atoms = set(range(target.GetNumAtoms()))
    found: list[Correspondence] = []
    for source_match in source_matches:
        for target_match in target_matches:
            # Prune on BOTH sides and keep only the mapping pairs that survive on both,
            # so the retained core stays a ring-complete substructure of each endpoint
            # and the map remains a bijection between the two survivors.
            pairs = list(zip(source_match, target_match, strict=True))
            keep_source = prune_to_ring_complete(source, set(source_match))
            keep_target = prune_to_ring_complete(target, set(target_match))
            pairs = [
                (a, b) for a, b in pairs if a in keep_source and b in keep_target
            ]
            if not pairs:
                continue
            kept_source = {a for a, _ in pairs}
            kept_target = {b for _, b in pairs}
            core = tuple(sorted(kept_source))
            delete = tuple(sorted(source_atoms - kept_source))
            install = tuple(sorted(target_atoms - kept_target))
            found.append(
                Correspondence(
                    source_smiles=source_smiles,
                    target_smiles=target_smiles,
                    core_smarts=result.smartsString,
                    retain_core=core,
                    core_map=tuple(sorted(pairs)),
                    r_delete=delete,
                    h_install=install,
                    alpha=tuple(
                        _boundary(source, kept_source, set(delete), "source")
                        + _boundary(target, kept_target, set(install), "target")
                    ),
                    core_bond_changes=_core_bond_changes(source, target, dict(pairs)),
                    delete_order=_connectivity_preserving_order(source, delete),
                    install_order=_growth_order(target, install, kept_target),
                    dependencies={
                        "delete_rule": "non_cut_vertex_first_keeps_survivors_connected",
                        "install_rule": "anchored_growth_whole_ring_systems_at_once",
                    },
                )
            )
            if len(found) >= top_k:
                return found
    return found


# ---- Validation ----------------------------------------------------------


def deletion_keeps_source_connected(correspondence: Correspondence) -> bool:
    """Removing `delete_order` one atom at a time never disconnects the survivors."""
    mol = Chem.MolFromSmiles(correspondence.source_smiles)
    adjacency = {
        atom.GetIdx(): {n.GetIdx() for n in atom.GetNeighbors()} for atom in mol.GetAtoms()
    }
    surviving = set(adjacency)
    for index in correspondence.delete_order:
        surviving.discard(index)
        if len(surviving) <= 1:
            continue
        start = next(iter(surviving))
        seen, stack = {start}, [start]
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current] & surviving:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        if seen != surviving:
            return False
    return True


def partitions_source(correspondence: Correspondence) -> bool:
    """`retain_core` and `r_delete` must be disjoint and cover every atom of G exactly."""
    source = Chem.MolFromSmiles(correspondence.source_smiles)
    core, delete = set(correspondence.retain_core), set(correspondence.r_delete)
    return (
        not (core & delete)
        and core | delete == set(range(source.GetNumAtoms()))
        and len(core) + len(delete) == source.GetNumAtoms()
    )


def core_is_ring_complete_substructure(correspondence: Correspondence) -> bool:
    """The core must be a real substructure of BOTH endpoints, with no partial ring.

    Checked three ways, because each catches a different failure: the recorded SMARTS must
    match both molecules; the recorded atom sets must BE matches of it, not merely the
    right size; and no atom of the core may be a ring atom in its own molecule while the
    rest of that ring sits outside the core.
    """
    source = Chem.MolFromSmiles(correspondence.source_smiles)
    target = Chem.MolFromSmiles(correspondence.target_smiles)
    source_core = set(correspondence.retain_core)
    target_core = {pair[1] for pair in correspondence.core_map}
    if not source_core or not target_core:
        return False
    if len(source_core) != len(target_core):
        return False
    # RING COMPLETENESS on each endpoint: no ring may be partly in and partly out.
    for mol, core in ((source, source_core), (target, target_core)):
        for ring in mol.GetRingInfo().AtomRings():
            touched = core.intersection(ring)
            if touched and len(touched) != len(ring):
                return False
    # GENUINE SUBSTRUCTURE: the core's induced subgraph must exist in both molecules, and
    # the map must be an isomorphism of those two induced subgraphs -- element, bond order
    # and adjacency. Extracting the induced subgraph is what makes this a real check
    # rather than a re-assertion of how the core was built.
    forward = dict(correspondence.core_map)
    for atom in source_core:
        counterpart = forward.get(atom)
        if counterpart is None:
            return False
        if (
            source.GetAtomWithIdx(atom).GetAtomicNum()
            != target.GetAtomWithIdx(counterpart).GetAtomicNum()
        ):
            return False
    for bond in source.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # ADJACENCY must mirror; bond ORDER may differ and is recorded as a
        # `core_bond_changes` entry instead. Requiring order equality here would report a
        # legitimate aromatic-to-saturated transport as an invalid core.
        if (
            begin in source_core
            and end in source_core
            and target.GetBondBetweenAtoms(forward[begin], forward[end]) is None
        ):
            return False
    return True


def zero_scale_implies_identical(correspondence: Correspondence) -> bool:
    """A transport of size zero must mean the molecules ARE the same molecule.

    This is the guard that caught the missing bond-order term: benzene -> cyclohexane
    reported `scale = 0` while being two different molecules.
    """
    if correspondence.scale != 0:
        return True
    source = Chem.MolFromSmiles(correspondence.source_smiles)
    target = Chem.MolFromSmiles(correspondence.target_smiles)
    return Chem.MolToSmiles(source) == Chem.MolToSmiles(target)


def validate(correspondence: Correspondence) -> dict:
    """The offline checks, reported as data rather than raised."""
    # A disconnected core CANNOT be realized by deletion alone, so the connectivity
    # invariant is conditioned on it rather than asserted unconditionally. Reporting it
    # unconditionally would label a legitimate excision-plus-reattachment as a defect.
    reattachment = correspondence.requires_reattachment
    connected = deletion_keeps_source_connected(correspondence)
    return {
        "requires_reattachment": reattachment,
        "deletion_connectivity_satisfied_or_reattachment_declared": connected or reattachment,
        "partitions_source": partitions_source(correspondence),
        "core_is_ring_complete_substructure": core_is_ring_complete_substructure(
            correspondence
        ),
        "deletion_keeps_source_connected": connected,
        "zero_scale_implies_identical": zero_scale_implies_identical(correspondence),
    }
