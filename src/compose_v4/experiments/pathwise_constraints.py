"""Exact labeled-subgraph pathwise constraints for COMPOSE trajectories.

WHAT THIS IS
------------
A protected motif is a labeled subgraph derived deterministically from the
SOURCE molecule alone. The pathwise constraint is

    at every committed state x_t, the protected motif embeds into x_t as an
    exact atom/bond-labeled subgraph.

Preservation is *subgraph monomorphism under exact labels*, not fingerprint
similarity and not endpoint recovery. Two atoms match only when element,
aromaticity flag and formal charge all agree; two bonds match only when the
bond order (or aromaticity) agrees. Substitution OUTSIDE the motif is
unconstrained -- that is exactly the room left to act.

WHY A HAND-WRITTEN SMARTS WRITER
--------------------------------
`Chem.MolFragmentToSmarts` emits bare `[#6]` primitives: it drops the
aromaticity flag and the formal charge, so `[#7]` matches a neutral amine and a
quaternary ammonium alike. That is weaker than "exact atom-labeled" and would
silently let a charge-changing edit pass the constraint. `fragment_smarts`
below emits every primitive explicitly. Every derivation self-checks that the
query it built actually matches the molecule it came from, so a writer bug
fails loudly at derivation time rather than quietly widening the support.

NO MODEL, NO MODAL
------------------
Everything here is pure RDKit and runs locally. The successor kernel is the
only part of the experiment that needs the Modal runtime; the constraint is
evaluated on successor KEYS (SMILES), so masking costs zero extra kernel calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from rdkit import Chem

#: Bumped whenever the derivation rule or the label semantics change. Any
#: artifact produced under a different version is not comparable.
MOTIF_RULE_VERSION = "largest_ring_system_v1"

_BOND_TOKEN = {
    Chem.BondType.SINGLE: "-",
    Chem.BondType.DOUBLE: "=",
    Chem.BondType.TRIPLE: "#",
}


class MotifDerivationError(RuntimeError):
    """The derived query failed to match the molecule it was derived from."""


# --------------------------------------------------------------------------
# labeled SMARTS construction
# --------------------------------------------------------------------------


def atom_primitive(atom: Chem.Atom) -> str:
    """Element, aromaticity and formal charge. All three are load-bearing."""
    charge = atom.GetFormalCharge()
    return (
        f"[#{atom.GetAtomicNum()};"
        f"{'a' if atom.GetIsAromatic() else 'A'};"
        f"{charge:+d}]"
    )


def bond_primitive(bond: Chem.Bond) -> str:
    """Aromatic bonds are `:`; everything else is its exact order."""
    if bond.GetIsAromatic():
        return ":"
    token = _BOND_TOKEN.get(bond.GetBondType())
    if token is None:  # dative/unspecified orders are not part of this chemistry
        raise MotifDerivationError(f"unsupported bond type {bond.GetBondType()!r}")
    return token


def _ring_closure_token(digit: int) -> str:
    return str(digit) if digit < 10 else f"%{digit}"


def fragment_smarts(
    mol: Chem.Mol,
    atom_indices: list[int],
    bond_indices: list[int],
) -> str:
    """Serialise an induced labeled subgraph as SMARTS.

    A depth-first spanning tree gives the linear backbone; every remaining
    (cycle-closing) bond becomes a ring-closure digit carrying its own bond
    primitive. Disconnected components are joined with `.`.
    """
    atoms = sorted(atom_indices)
    adjacency: dict[int, list[tuple[int, int]]] = {a: [] for a in atoms}
    for bond_index in bond_indices:
        bond = mol.GetBondWithIdx(bond_index)
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin not in adjacency or end not in adjacency:
            raise MotifDerivationError("bond leaves the atom set")
        adjacency[begin].append((end, bond_index))
        adjacency[end].append((begin, bond_index))
    for neighbours in adjacency.values():
        neighbours.sort()

    tree_bonds: set[int] = set()
    closure_bonds: list[tuple[int, int, int]] = []
    children: dict[int, list[tuple[int, int]]] = {a: [] for a in atoms}
    seen: set[int] = set()

    def walk(node: int, incoming: int | None) -> None:
        seen.add(node)
        for neighbour, bond_index in adjacency[node]:
            if bond_index == incoming or bond_index in tree_bonds:
                continue
            if neighbour in seen:
                if all(bond_index != b for _, _, b in closure_bonds):
                    closure_bonds.append((node, neighbour, bond_index))
                continue
            tree_bonds.add(bond_index)
            children[node].append((neighbour, bond_index))
            walk(neighbour, bond_index)

    components: list[int] = []
    for atom in atoms:
        if atom not in seen:
            components.append(atom)
            walk(atom, None)

    closures_at: dict[int, list[tuple[int, int]]] = {a: [] for a in atoms}
    for digit, (begin, end, bond_index) in enumerate(closure_bonds, start=1):
        closures_at[begin].append((digit, bond_index))
        closures_at[end].append((digit, bond_index))

    def emit(node: int) -> str:
        text = atom_primitive(mol.GetAtomWithIdx(node))
        for digit, bond_index in closures_at[node]:
            text += bond_primitive(mol.GetBondWithIdx(bond_index))
            text += _ring_closure_token(digit)
        kids = children[node]
        for position, (child, bond_index) in enumerate(kids):
            branch = bond_primitive(mol.GetBondWithIdx(bond_index)) + emit(child)
            text += branch if position == len(kids) - 1 else f"({branch})"
        return text

    return ".".join(emit(root) for root in components)


# --------------------------------------------------------------------------
# motif derivation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtectedMotif:
    """A source-derived labeled subgraph that every committed state must keep."""

    smarts: str
    rule: str
    atom_count: int
    bond_count: int
    ring_count: int
    source_smiles: str
    source_heavy_atoms: int
    atom_indices: tuple[int, ...] = field(default=())

    @property
    def fraction(self) -> float:
        return self.atom_count / self.source_heavy_atoms

    @property
    def free_atoms(self) -> int:
        """Heavy atoms of the source outside the motif -- the room to act."""
        return self.source_heavy_atoms - self.atom_count

    def as_dict(self) -> dict:
        return {
            "smarts": self.smarts,
            "rule": self.rule,
            "atom_count": self.atom_count,
            "bond_count": self.bond_count,
            "ring_count": self.ring_count,
            "source_heavy_atoms": self.source_heavy_atoms,
            "fraction": round(self.fraction, 4),
            "free_atoms": self.free_atoms,
        }


def ring_systems(mol: Chem.Mol) -> list[list[int]]:
    """Fused ring systems: connected components of the ring-bond graph."""
    parent: dict[int, int] = {}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[a] = b

    for bond in mol.GetBonds():
        if bond.IsInRing():
            union(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    groups: dict[int, set[int]] = {}
    for atom in list(parent):
        groups.setdefault(find(atom), set()).add(atom)
    return [sorted(group) for group in groups.values()]


def _induced_bonds(mol: Chem.Mol, atom_indices: list[int]) -> list[int]:
    members = set(atom_indices)
    return [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in members and bond.GetEndAtomIdx() in members
    ]


def _ring_count(mol: Chem.Mol, atom_indices: list[int]) -> int:
    members = set(atom_indices)
    return sum(
        1 for ring in mol.GetRingInfo().AtomRings() if members.issuperset(ring)
    )


def derive_protected_motif(smiles: str) -> ProtectedMotif | None:
    """The largest fused ring system of the source, with exact labels.

    Deterministic and outcome-independent: it reads the source molecule and
    nothing else -- no trajectory, no successor set, no objective value. Ties
    on atom count are broken by ring count and then by the sorted atom-index
    tuple, so the rule never depends on RDKit iteration order.

    Returns None for an unparseable or acyclic source.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    systems = ring_systems(mol)
    if not systems:
        return None

    best = max(
        systems,
        key=lambda atoms: (len(atoms), _ring_count(mol, atoms), tuple(atoms)),
    )
    bonds = _induced_bonds(mol, best)
    smarts = fragment_smarts(mol, best, bonds)

    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise MotifDerivationError(f"unparseable derived SMARTS: {smarts}")
    match = mol.GetSubstructMatch(query)
    if len(match) != len(best):
        raise MotifDerivationError(
            f"derived motif does not match its own source: {smiles} -> {smarts}"
        )

    return ProtectedMotif(
        smarts=smarts,
        rule=MOTIF_RULE_VERSION,
        atom_count=len(best),
        bond_count=len(bonds),
        ring_count=_ring_count(mol, best),
        source_smiles=smiles,
        source_heavy_atoms=mol.GetNumHeavyAtoms(),
        atom_indices=tuple(best),
    )


# --------------------------------------------------------------------------
# the constraint predicate
# --------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _compiled(smarts: str) -> Chem.Mol:
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise MotifDerivationError(f"unparseable motif SMARTS: {smarts}")
    return query


def preserves_motif(smarts: str, smiles: str) -> bool:
    """True when the protected motif embeds in `smiles` under exact labels.

    An unparseable state is treated as a violation: a state COMPOSE cannot
    read is not a state that demonstrably contains the motif.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return mol.HasSubstructMatch(_compiled(smarts))


def motif_embedding_count(smarts: str, smiles: str) -> int:
    """Number of distinct embeddings. Diagnostic only; the predicate is >=1."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return len(mol.GetSubstructMatches(_compiled(smarts), uniquify=True))


def first_violation(smarts: str, trajectory: list[str]) -> int | None:
    """Index of the first committed state that breaks the motif, else None.

    Index 0 is the source. A well-formed source never violates its own motif,
    so a return value of 0 signals a derivation bug, not a trajectory event.
    """
    for index, key in enumerate(trajectory):
        if not preserves_motif(smarts, key):
            return index
    return None


def path_violation_summary(smarts: str, trajectory: list[str]) -> dict:
    """The pathwise audit of one realised trajectory.

    `endpoint_valid_path_invalid` is the cell that decides this workstream:
    a trajectory endpoint-only filtering would ACCEPT, having passed through a
    state the pathwise constraint forbids. It is 0 whenever the hypothesis is
    false, which is what makes it a measurement rather than a definition.
    """
    flags = [preserves_motif(smarts, key) for key in trajectory]
    endpoint_valid = bool(flags[-1]) if flags else False
    # index 0 is the source; a violation there is a derivation bug
    intermediate = flags[1:]
    violated = [i + 1 for i, ok in enumerate(intermediate) if not ok]
    return {
        "states": len(trajectory),
        "endpoint_valid": endpoint_valid,
        "any_violation": bool(violated),
        "first_violation_index": violated[0] if violated else None,
        "violation_count": len(violated),
        "endpoint_valid_path_invalid": bool(endpoint_valid and violated),
        "source_violates_own_motif": bool(flags and not flags[0]),
    }


def mask_successors(
    smarts: str,
    rows: list[tuple[str, float]],
) -> tuple[list[tuple[str, float]], dict]:
    """Drop every successor whose state breaks the motif.

    Costs no kernel calls: the kernel already produced the keys, and the
    predicate reads the key. Returns the surviving rows plus the removal
    census used by the "mask removes nearly all support" stop rule.
    """
    kept = [row for row in rows if preserves_motif(smarts, row[0])]
    total = len(rows)
    kept_mass = sum(row[1] for row in kept)
    total_mass = sum(row[1] for row in rows)
    return kept, {
        "candidates": total,
        "kept": len(kept),
        "removed": total - len(kept),
        "removed_fraction": (total - len(kept)) / total if total else 0.0,
        "kept_reference_mass": kept_mass,
        "removed_reference_mass": total_mass - kept_mass,
    }


# --------------------------------------------------------------------------
# eligibility -- outcome-independent, source-only
# --------------------------------------------------------------------------

#: Reused verbatim from the frozen retargeting cohort so the pathwise panel
#: sits in the same size band as every other held-in COMPOSE panel.
HEAVY_ATOM_BAND = (18, 38)
#: A motif smaller than a benzene ring is a "trivial one-atom pattern" in the
#: workstream's sense and cannot carry a pharmacophore reading.
MIN_MOTIF_ATOMS = 6
#: Neither nearly the whole molecule nor trivial.
MOTIF_FRACTION_BAND = (0.20, 0.70)
#: Enough atoms outside the motif to permit meaningful edits.
MIN_FREE_ATOMS = 8


def eligibility(
    smiles: str,
    motif: ProtectedMotif | None,
    *,
    already_satisfies_goal: bool | None = None,
) -> dict:
    """Decide whether a source may enter the pathwise panel.

    Every criterion reads the SOURCE only. None of them reads a trajectory, a
    successor set, or an arm outcome, so admitting a source cannot depend on
    whether the constraint turns out to bite.

    DELIBERATELY NOT A CRITERION: "the frozen kernel offers both
    motif-preserving and motif-destroying legal successors". That property is
    the numerator of the vacuity gate. Filtering on it would guarantee a
    non-vacuous mask by construction and convert the headline measurement into
    a definition. It is measured and reported per source instead.
    """
    reasons: list[str] = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"eligible": False, "reasons": ["unparseable"]}
    heavy = mol.GetNumHeavyAtoms()
    if not HEAVY_ATOM_BAND[0] <= heavy <= HEAVY_ATOM_BAND[1]:
        reasons.append("size_band")
    if motif is None:
        reasons.append("no_ring_system")
    else:
        if motif.atom_count < MIN_MOTIF_ATOMS:
            reasons.append("motif_too_small")
        if not (
            MOTIF_FRACTION_BAND[0] <= motif.fraction <= MOTIF_FRACTION_BAND[1]
        ):
            reasons.append("motif_fraction")
        if motif.free_atoms < MIN_FREE_ATOMS:
            reasons.append("too_few_free_atoms")
    if already_satisfies_goal:
        reasons.append("already_satisfies_goal")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "heavy_atoms": heavy,
        "motif": motif.as_dict() if motif else None,
    }
