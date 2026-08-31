"""Connected mutable regions of a molecule: the unit of a variable-scope proposal.

A proposal is defined by a connected mutable region `M`, the preserved context
`C = x \\ M`, and the attachment boundary `dM`. The SCALE of the move is set by
the size and structure of `M`, never by choosing a macro name -- so "local" and
"global" become measured properties rather than operator labels.

Boundary arity ALONE does not say whether deletion is safe, so the interface is
classified by arity together with the connectivity of the preserved context:

    whole     |dM| = 0                              near-de-novo
    pendant   |dM| = 1, C connected                 prune then regrow
    segment   |dM| = 2, C still connected           open/restructure then regrow;
                                                    the rest of the molecule
                                                    already joins the anchors
    splitting |dM| = 2, C splits in two             CANNOT delete first without
                                                    disconnecting -- needs a
                                                    connect-first or coupled
                                                    schedule
    multi     |dM| >= 3                             deferred

Those are three genuinely different executable problems, not one compiler with a
parameter.

Enumeration is DETERMINISTIC and returns the complete candidate set for the
generators it declares, so a scope proposal `Q_scope(M | x)` over it is
likelihood-evaluable. That is a hard design constraint: every proposal mechanism
in this line must admit an evaluable density, or the exact
importance-corrected story cannot be told later.

Nothing here proposes chemistry or touches the executor. It only says WHICH part
of the molecule a replacement is allowed to rewrite.

RING-TOPOLOGY POLICY. "splitting" is graph-theoretic -- removing the region
splits the preserved context -- and has NOTHING to do with a chemically bridged
bicyclic ring. Those are low priority and get no dedicated compiler: the 22-cell
development panel carries zero bridgehead and zero spiro atoms, matching zero in
the 25 IVG T4 winners, and in the approved-drug landscape fused systems appear in
roughly 59% of recent EMA heterocyclic approvals against roughly 5% bridged.
They are NOT banned, however: if a generic region rewrite legally produces a
spiro or bridged system through ordinary primitive support, it stands. We simply
do not build a proposal whose only purpose is to find one. A spiro or bridged
shortcut earns its way in later only if held-out geometry shows it recurs among
strong molecules AND generic region resampling cannot reach it efficiently.
Ordinary pendant, linked and fused systems, segment replacement, expansion and
contraction, and composition refinement remain core.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class Region:
    """A connected mutable region and how it attaches to the preserved context."""

    atoms: frozenset          # indices of the mutable region M
    boundary: tuple           # ((inside_idx, outside_idx, bond_order_as_float), ...)
    kind: str                 # 'whole' | 'substituent' | 'linker' | 'multi'
    generator: str            # generators that produced it, "a+b" (provenance)
    n_atoms_total: int
    n_context_components: int = 1   # connected components of C = x \ M
    interface: str = "pendant"      # 'whole'|'pendant'|'segment'|'splitting'|'multi'
    n_ring_boundary_bonds: int = 0  # boundary bonds that are ring bonds
    region_has_ring: bool = False

    @property
    def size(self) -> int:
        return len(self.atoms)

    @property
    def arity(self) -> int:
        return len(self.boundary)

    @property
    def released_fraction(self) -> float:
        return self.size / max(1, self.n_atoms_total)

    def key(self) -> tuple:
        return (tuple(sorted(self.atoms)), self.boundary)


def _boundary(mol, atoms: frozenset) -> tuple:
    out = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (i in atoms) != (j in atoms):
            inside, outside = (i, j) if i in atoms else (j, i)
            out.append((int(inside), int(outside), float(b.GetBondTypeAsDouble())))
    return tuple(sorted(out))


def _kind(arity: int) -> str:
    return {0: "whole", 1: "substituent", 2: "linker"}.get(arity, "multi")


def _interface(arity: int, n_context_components: int) -> str:
    """Deletion safety, not just attachment count."""
    if arity == 0:
        return "whole"
    if arity >= 3:
        return "multi"
    if arity == 1:
        return "pendant"
    return "segment" if n_context_components <= 1 else "splitting"


def _components(mol, removed_bonds: set) -> list:
    """Connected components of the atom graph with `removed_bonds` deleted."""
    adj = {a.GetIdx(): set() for a in mol.GetAtoms()}
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (min(i, j), max(i, j)) in removed_bonds:
            continue
        adj[i].add(j); adj[j].add(i)
    seen, comps = set(), []
    for start in adj:
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            v = stack.pop()
            if v in comp:
                continue
            comp.add(v); seen.add(v)
            stack.extend(adj[v] - comp)
        comps.append(frozenset(comp))
    return comps


def _ring_systems(mol) -> list:
    """Fused ring systems: rings sharing >=1 atom are one system."""
    rings = [frozenset(r) for r in mol.GetRingInfo().AtomRings()]
    systems: list = []
    for r in rings:
        merged, rest = [r], []
        for s in systems:
            (merged if s & r else rest).append(s)
        systems = rest + [frozenset().union(*merged)]
    return systems


def _connected_parts(mol, atoms: frozenset) -> list:
    """Split an atom set into its connected components inside the molecule.

    Complement generators (molecule minus a ball or ring system) routinely
    produce fragmented sets -- CCO minus its middle carbon is {0, 2}. A region
    is connected BY DEFINITION, so every candidate is split before it is
    offered rather than silently admitted.
    """
    remaining, parts = set(atoms), []
    while remaining:
        start = next(iter(remaining))
        comp, stack = set(), [start]
        while stack:
            v = stack.pop()
            if v in comp:
                continue
            comp.add(v)
            for nb in mol.GetAtomWithIdx(v).GetNeighbors():
                j = nb.GetIdx()
                if j in remaining and j not in comp:
                    stack.append(j)
        remaining -= comp
        parts.append(frozenset(comp))
    return parts


def _ball(mol, center: int, radius: int) -> frozenset:
    cur = {center}
    for _ in range(radius):
        nxt = set(cur)
        for a in cur:
            nxt.update(n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors())
        cur = nxt
    return frozenset(cur)


def enumerate_regions(smiles: str, *, max_released: float = 0.90,
                      ball_radii=(0, 1, 2), include_complement: bool = True):
    """Every connected mutable region this molecule offers, across all scales.

    Generators, in increasing scope:

      ball        single atoms and small neighbourhoods   -> local edits
      one_cut     split at each acyclic bond              -> substituent swap
      ring_system each fused ring system                  -> ring replacement
      two_cut     the middle piece between two acyclic bonds -> linker redesign
      complement  the LARGE side of a one-cut             -> scaffold-scale move
      co_*        the molecule MINUS a ring system or ball -> large scaffold region

    Deterministic: same molecule in, same list out, so a scope distribution over
    the result is evaluable. Regions releasing more than `max_released` of the
    molecule are dropped (they are de-novo generation, not replacement).
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    n = mol.GetNumAtoms()
    acyclic = [(min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
               for b in mol.GetBonds() if not b.IsInRing()]

    found: dict = {}

    def offer(candidate: frozenset, generator: str):
        for atoms in _connected_parts(mol, candidate):
            _offer_connected(atoms, generator)

    def _offer_connected(atoms: frozenset, generator: str):
        if not atoms or len(atoms) >= n:
            return
        if len(atoms) / n > max_released:
            return
        bnd = _boundary(mol, atoms)
        context = frozenset(range(n)) - atoms
        ncc = len(_connected_parts(mol, context)) if context else 0
        nrb = sum(1 for (i, j, _o) in bnd
                  if mol.GetBondBetweenAtoms(int(i), int(j)).IsInRing())
        has_ring = any(mol.GetAtomWithIdx(int(a)).IsInRing() for a in atoms)
        r = Region(atoms=atoms, boundary=bnd, kind=_kind(len(bnd)),
                   generator=generator, n_atoms_total=n,
                   n_context_components=ncc,
                   interface=_interface(len(bnd), ncc),
                   n_ring_boundary_bonds=nrb, region_has_ring=has_ring)
        prev = found.get(r.key())
        if prev is None:
            found[r.key()] = r
        elif generator not in prev.generator.split("+"):
            found[r.key()] = Region(
                atoms=prev.atoms, boundary=prev.boundary, kind=prev.kind,
                generator="+".join(sorted(set(prev.generator.split("+")) | {generator})),
                n_atoms_total=n, n_context_components=prev.n_context_components,
                interface=prev.interface,
                n_ring_boundary_bonds=prev.n_ring_boundary_bonds,
                region_has_ring=prev.region_has_ring)

    for radius in ball_radii:                                   # local
        for a in range(n):
            offer(_ball(mol, a, radius), f"ball{radius}")

    for e in acyclic:                                           # one-cut
        for comp in _components(mol, {e}):
            offer(comp, "one_cut")
            if include_complement:
                offer(frozenset(range(n)) - comp, "complement")

    for sysm in _ring_systems(mol):                             # ring system
        offer(sysm, "ring_system")

    for sysm in _ring_systems(mol):            # everything EXCEPT one ring system
        offer(frozenset(range(n)) - sysm, "co_ring_system")

    for radius in ball_radii:                  # everything EXCEPT a local ball
        for a in range(n):
            offer(frozenset(range(n)) - _ball(mol, a, radius), f"co_ball{radius}")

    for e1, e2 in combinations(acyclic, 2):                     # two-cut
        comps = _components(mol, {e1, e2})
        if len(comps) < 3:
            continue
        ends = {a for e in (e1, e2) for a in e}
        for comp in comps:
            if len(comp & ends) == 2:        # the middle piece touches both cuts
                offer(comp, "two_cut")

    return sorted(found.values(), key=lambda r: (r.size, sorted(r.atoms)))


def scope_summary(regions) -> dict:
    """Coverage of the scale axis, for the geometry qualification gate."""
    from collections import Counter
    if not regions:
        return {"n": 0}
    by_kind = Counter(r.kind for r in regions)
    by_iface = Counter(r.interface for r in regions)
    by_gen = Counter(r.generator for r in regions)
    fr = sorted(r.released_fraction for r in regions)
    return {
        "n": len(regions),
        "by_kind": dict(by_kind),
        "by_interface": dict(by_iface),
        "by_generator": dict(by_gen),
        "released_fraction_min": fr[0],
        "released_fraction_median": fr[len(fr) // 2],
        "released_fraction_max": fr[-1],
        "n_substituent": by_kind.get("substituent", 0),
        "n_linker": by_kind.get("linker", 0),
        "size_range": (min(r.size for r in regions), max(r.size for r in regions)),
    }


def actual_changed_fraction(x_smiles: str, y_smiles: str, timeout: int = 5):
    """Fraction of x's atoms NOT preserved in y, by maximum common substructure.

    Guards the next false-success mode: selecting a 70% mutable region and then
    changing one atom is not a global move. Compare this against the region's
    `released_fraction` -- a large gap means the scope selector is writing
    cheques the replacement dynamics do not cash.

    Returns None when no common substructure is found (a total rewrite).
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    a, b = Chem.MolFromSmiles(x_smiles), Chem.MolFromSmiles(y_smiles)
    if a is None or b is None:
        return None
    res = rdFMCS.FindMCS([a, b], timeout=timeout,
                         ringMatchesRingOnly=True, completeRingsOnly=False)
    if res.canceled or res.numAtoms == 0:
        return 1.0
    return 1.0 - (res.numAtoms / a.GetNumAtoms())
