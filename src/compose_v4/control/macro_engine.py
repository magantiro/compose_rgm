"""Macro-endpoint search: program selection, tail exploration, endpoint credit.

Three measured failures motivate every piece of this, and none of them is fixed
by the others:

  1 GLOBAL FAMILY SUPPRESSION. On the parp1 witness seed all 57 legal
    cycle_close actions sit past the global top-300; the route's entry closure
    is at rank 414. Fix: support = top-K global UNION top-k per family.

  2 WITHIN-FAMILY TAIL. Naming the family is not enough -- two witness steps
    need actions ranked 137th and 109th INSIDE their own family, and greedy
    top-1-within-family walks away from the target from the first step
    (similarity 0.424 -> 0.125 over 7 edits). Fix: temper/mix inside the macro
    so the clean tail stays reachable.

  3 DELAYED CREDIT. The witness binding curve is flat for six edits and then
    drops 6 kcal/mol at the reclose. Any controller that scores every primitive
    edit prunes the program before it pays. Fix: execute L edits, score ONLY the
    endpoint.

Bad chemistry is MASKED and the remainder renormalised -- measured as ~1/3 of
actions by count but under 1.5% of R_theta mass, so masking costs almost no
probability and needs no retraining.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: A macro names a PURPOSE; the families are how that purpose is executed.
MACRO_FAMILIES: dict[str, tuple[str, ...]] = {
    "local":     ("atom_restate_semantic", "bond_reorder"),
    "grow":      ("atom_insert",),
    # Ring formation is NOT confined to cycle_close. Measured on a real
    # six-carbon precursor, that state offered ZERO cycle_close actions and 127
    # bond_insert actions, two of which close the chain into a six-membered
    # carbocycle. A cyclize macro scoped to cycle_close alone could never find
    # the hexagon sitting in front of it, and closed whatever ring it could
    # reach elsewhere -- epoxides, cyclobutenes, bridged bicycles.
    "cyclize":   ("cycle_close", "bond_insert"),
    "aromatize": ("atom_restate_semantic", "ring_system_restate", "bond_reorder"),
    "rebuild":   ("cycle_open", "atom_insert", "bond_reroute"),
    # The measured constructive program is open/rearrange -> grow -> cyclize ->
    # restate/reorder, NOT grow+cyclize alone. IVG topology is not merely "more
    # rings", it is more ORGANISED aromatic ring systems, and old COMPOSE proved
    # it can add mass (19->26, 16->33 heavy atoms) without ever getting there.
    "restate":   ("atom_restate_semantic", "bond_reorder", "ring_system_restate"),
    "open":      ("cycle_open", "bond_reroute"),
    # APPEND is grow+cyclize with a DISJOINT-closure constraint. Measured: 81%
    # of our macro endpoints had exactly ONE fused ring system, while all five
    # published IVG winners have TWO or THREE joined by flexible linkers. The
    # cause was mechanical, not chemical -- a pendant ring must close among
    # NEWLY ADDED atoms, the simplest IVG pendant (CH2CH2 linker + benzene)
    # needs 8 new atoms, and our grow horizons were 3/4/6. There was never
    # enough chain, so every legal closure reached back to the scaffold and
    # cyclize could only annulate. No new operators are required.
    "append":    ("atom_insert",),
    # SCAFFOLD_EXTEND is grow restricted to BACKBONE-CAPABLE atoms. Measured on
    # 80 grows: only ~2.6 of ~9 added atoms were carbon and ~2 were halogens,
    # which are terminal by definition and cannot extend a chain at all. An
    # epsilon A/B (eps 0 vs 0.15, paired seeds) showed this is R_theta's own
    # insert law, not our exploration floor -- so it is fixed by restricting the
    # macro's support, not by retuning exploration. Halogens are DECORATION and
    # belong in a later local/decorate macro, not in backbone construction.
    "scaffold_extend": ("atom_insert",),
    "shrink":    ("atom_delete", "cycle_open"),
}
MACROS = tuple(MACRO_FAMILIES)

GLOBAL_CAP = 300
FAMILY_FLOOR = 20


def proposal_support(families: list[str], probs: np.ndarray,
                     cap: int = GLOBAL_CAP,
                     floor: int = FAMILY_FLOOR) -> np.ndarray:
    """Indices in top-`cap` globally UNION top-`floor` within each family.

    Pure function of (families, probs) so it is testable without the model.
    """
    n = len(probs)
    order = np.argsort(-probs, kind="stable")
    keep = np.zeros(n, dtype=bool)
    keep[order[:cap]] = True
    fam = np.asarray(families)
    for f in np.unique(fam):
        idx = np.flatnonzero(fam == f)
        idx = idx[np.argsort(-probs[idx], kind="stable")]
        keep[idx[:floor]] = True
    return np.flatnonzero(keep)


def macro_action_distribution(families: list[str], probs: np.ndarray,
                              support: np.ndarray, macro: str,
                              clean: np.ndarray,
                              temperature: float = 2.0,
                              epsilon: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Distribution over CLEAN in-support actions of `macro`'s families.

    q = (1-eps) * renormalised R_theta^(1/T)  +  eps * uniform over the same set.
    Temperature alone cannot lift a rank-137 action into reach without also
    flattening the head; the uniform floor is what makes the deep tail
    genuinely reachable, and eps is the only knob that guarantees it.

    Returns (indices, probabilities); indices is empty if nothing qualifies.
    """
    if macro not in MACRO_FAMILIES:
        raise KeyError(f"unknown macro {macro!r}; known: {MACROS}")
    fam = np.asarray(families)
    want = set(MACRO_FAMILIES[macro])
    sel = np.array([i for i in support if fam[i] in want and clean[i]], dtype=int)
    if sel.size == 0:
        return sel, np.zeros(0)
    p = np.asarray(probs, float)[sel]
    if not np.isfinite(p).all() or p.sum() <= 0:
        q = np.full(sel.size, 1.0 / sel.size)
        return sel, q
    tempered = np.power(np.maximum(p, 1e-300), 1.0 / float(temperature))
    tempered = tempered / tempered.sum()
    q = (1.0 - epsilon) * tempered + epsilon / sel.size
    return sel, q / q.sum()


@dataclass
class MacroStep:
    macro: str
    smiles: str
    action_index: int
    r_theta_prob: float
    within_family_rank: int


@dataclass
class MacroRollout:
    """One executed program. `scored` stays False until endpoint evaluation."""
    start: str
    macro: str
    steps: list[MacroStep] = field(default_factory=list)
    halted: str | None = None

    @property
    def endpoint(self) -> str:
        return self.steps[-1].smiles if self.steps else self.start

    @property
    def length(self) -> int:
        return len(self.steps)


def rollout(enumerate_fn, apply_fn, gate_fn, start: str, macro: str,
            length: int, rng, temperature: float = 2.0,
            epsilon: float = 0.15, cap: int = GLOBAL_CAP,
            floor: int = FAMILY_FLOOR, prefer_fn=None) -> MacroRollout:
    """Execute `length` primitive edits under one macro. NO intermediate scoring.

    The witness binding curve is flat for six edits and then drops 6 kcal/mol at
    the reclose, so a controller that evaluates purpose after each edit prunes
    the program before it pays. Purpose is applied to `.endpoint` by the caller.

    `enumerate_fn(smi) -> (families, probs, handles)` and
    `apply_fn(smi, handle) -> smiles|None` keep the model out of this module so
    the control logic stays unit-testable; `gate_fn(smi) -> bool` is the
    med-chem gate.
    """
    out = MacroRollout(start=start, macro=macro)
    cur = start
    for _ in range(int(length)):
        try:
            families, probs, handles = enumerate_fn(cur)
        except Exception as exc:
            out.halted = f"enumerate failed: {type(exc).__name__}"; break
        if not families:
            out.halted = "no legal successors"; break
        probs = np.asarray(probs, float)
        support = proposal_support(list(families), probs, cap=cap, floor=floor)

        # Gate is applied to PRODUCTS, so it must be evaluated lazily -- applying
        # every in-support action would cost more than the search itself. Draw,
        # then check, then redraw without the rejected action.
        fam = np.asarray(families)
        want = set(MACRO_FAMILIES[macro])
        pool = [int(i) for i in support if fam[i] in want]
        if not pool:
            out.halted = f"no legal {macro} action"; break
        clean = np.ones(len(probs), dtype=bool)
        picked = None
        for _attempt in range(12):
            idx, q = macro_action_distribution(list(families), probs, support,
                                               macro, clean, temperature, epsilon)
            if idx.size == 0:
                break
            j = int(rng.choice(idx, p=q))
            y = apply_fn(cur, handles[j])
            # prefer_fn is a SELECTION preference, deliberately separate from
            # gate_fn. Bridgeheads and stereocentres are legal chemistry -- they
            # are merely expensive to synthesise, which is what SA encodes. So
            # they belong here, not in the validity gate. Measured on 96 parp1
            # endpoints: 0/67 with a bridgehead were T4-feasible, versus 3/8
            # (37.5%) with neither bridgehead nor stereocentre, against a 3.1%
            # baseline. If nothing passes the preference we fall back to the
            # gate alone rather than halting -- a preference must never make a
            # macro unexecutable.
            if y and gate_fn(y) and (prefer_fn is None or prefer_fn(y)):
                p = float(probs[j])
                wf = int(np.sum((fam == fam[j]) & (probs > p)))
                picked = MacroStep(macro=macro, smiles=y, action_index=j,
                                   r_theta_prob=p, within_family_rank=wf)
                break
            clean[j] = False          # mask this action and redraw
        if picked is None and prefer_fn is not None:
            # second pass: preference relaxed, gate still enforced
            clean2 = np.ones(len(probs), dtype=bool)
            for _attempt in range(8):
                idx, q = macro_action_distribution(list(families), probs, support,
                                                   macro, clean2, temperature, epsilon)
                if idx.size == 0:
                    break
                j = int(rng.choice(idx, p=q))
                y = apply_fn(cur, handles[j])
                if y and gate_fn(y):
                    p = float(probs[j])
                    picked = MacroStep(macro=macro, smiles=y, action_index=j,
                                       r_theta_prob=p,
                                       within_family_rank=int(np.sum((fam == fam[j])
                                                                     & (probs > p))))
                    break
                clean2[j] = False
        if picked is None:
            out.halted = f"no clean {macro} action after 12 draws"; break
        out.steps.append(picked)
        cur = picked.smiles
    return out


def synthesis_preference(smiles: str) -> bool:
    """Reject products that add bridgeheads or stereocentres.

    NOT a validity judgement -- these are real, legal structures. It is a
    synthesis-cost preference, measured on 96 parp1 macro endpoints:

        bridgeheads > 0            0/67 T4-feasible   median SA 5.55
        bridge == 0 and stereo == 0  3/8 T4-feasible   median SA 3.75
        baseline                    3/96               median SA 5.10

    Our cyclize builds BRIDGED polycyclics; the published IVG winners contain
    zero bridgeheads and zero stereocentres while carrying MORE aliphatic rings
    than we do, so this is not a penalty on ring count or saturation.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdMD
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False
    if rdMD.CalcNumBridgeheadAtoms(m) > 0:
        return False
    return not Chem.FindMolChiralCenters(m, includeUnassigned=True,
                                         useLegacyImplementation=False)


def ring_systems(mol):
    """Fused rings grouped into systems (rings sharing atoms are one system)."""
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    groups: list[list[set]] = []
    for r in rings:
        touching = [g for g in groups if any(r & x for x in g)]
        if not touching:
            groups.append([r])
        else:
            merged = [r]
            for g in touching:
                merged += g
                groups.remove(g)
            groups.append(merged)
    return groups


def disjoint_closure_only(before_smiles: str):
    """Accept a closure ONLY if it creates a ring sharing no atom with any ring
    that already existed -- i.e. a PENDANT system, not an annulation.

    Returns a predicate over the product SMILES, so it plugs into rollout's
    prefer_fn. Comparing ring-system COUNT is what distinguishes the two: welding
    a ring onto the existing system leaves the count unchanged, while a pendant
    ring increments it.
    """
    from rdkit import Chem
    m0 = Chem.MolFromSmiles(before_smiles)
    n0 = len(ring_systems(m0)) if m0 is not None else 0

    def ok(smiles: str) -> bool:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return False
        return len(ring_systems(m)) > n0

    return ok


TERMINAL_ELEMENTS = frozenset({"F", "Cl", "Br", "I"})

#: Elements that can carry a drug scaffold. Measured on the parp1 seed's 216
#: legal inserts: carbon is 7.9% unrestricted, 12.3% if only halogens are
#: dropped, and 34.7% restricted to C/N/O. Dropping halogens alone is far too
#: weak because S and P -- not halogens -- are the bulk of the imbalance (72 of
#: the 216 actions). The cause is structural: the insert fiber is over
#: (element, valence) CLASSES, so S (3 valences), I (3) and P (2) each get
#: multiple actions while carbon gets one. I/S/P are ~53% of every insert fiber
#: across all five T4 targets. R_theta already upweights carbon ~3.7x against
#: this, so the prior is not at fault; the macro's support is.
BACKBONE_ELEMENTS = frozenset({"C", "N", "O"})

#: SCAFFOLD_EXTEND builds an EXTENDABLE BACKBONE. Carbon-rich is one MODE of
#: that macro, not its meaning -- a nitrogen linker or a heteroaromatic
#: precursor is the right backbone for some targets, and hard-coding carbon
#: would bake a parp1/5ht1b observation into the architecture. The controller
#: picks the mode; R_theta still picks the site, valence and local state.
#:
#: Terminal halogens are excluded from EVERY mode because they cannot extend a
#: chain at all -- they belong to DECORATE, after construction.
COMPOSITION_MODES: dict[str, frozenset] = {
    "carbon_rich": frozenset({"C"}),
    "mixed":       frozenset({"C", "N", "O"}),
    "hetero_rich": frozenset({"N", "O", "S"}),
}
#: Smallest ring SYSTEM (in newly added atoms) that APPEND_SYSTEM will accept.
#: Derived from the IVG winners, not chosen: their major pendant systems are
#: 9-10 atoms on parp1 and 6-atom rings on 5ht1b. Deliberately NOT applied as a
#: validity rule -- one parp1 winner carries a 3-atom cyclopropane spacer, so a
#: global minimum would reject real winning chemistry. This is a statement about
#: what APPEND_SYSTEM is FOR; a decorate/spacer macro builds the small rings.
APPEND_MIN_NEW_ATOMS = 6


def backbone_only(before_smiles: str, composition: str = "mixed"):
    """Restrict inserts to the chosen backbone COMPOSITION MODE.

    A conditioning, not a ban: other macros and other modes remain available,
    and rollout falls back to the gate alone if nothing qualifies, so a mode can
    never make the macro unexecutable.

    Direct insertion is preferred over insert-then-restate for construction:
    restating a heteroatom to carbon costs a second primitive, adds an
    intermediate the controller can wander from, and may not even be legal for a
    terminal halogen. Restatement is a REFINEMENT operation -- use it after gross
    topology exists, to place heteroatoms or set saturation.
    """
    from rdkit import Chem
    m0 = Chem.MolFromSmiles(before_smiles)
    if m0 is None:
        return lambda smiles: False
    import collections
    before = collections.Counter(a.GetSymbol() for a in m0.GetAtoms())

    allowed = COMPOSITION_MODES.get(composition)
    if allowed is None:
        raise KeyError(f"unknown composition {composition!r}; "
                       f"known: {tuple(COMPOSITION_MODES)}")

    def ok(smiles: str) -> bool:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return False
        after = collections.Counter(a.GetSymbol() for a in m.GetAtoms())
        added = set(after - before)
        if added & TERMINAL_ELEMENTS:
            return False              # halogens cannot extend a chain, any mode
        return bool(added) and added <= allowed

    return ok


def append_system_closure(before_smiles: str, min_new: int = APPEND_MIN_NEW_ATOMS):
    """Accept a closure only if it creates a NEW ring system of >= min_new atoms
    that does not weld back onto an existing one.

    Two conditions, both measured against the pre-closure state:
      - ring-system count must INCREASE (a pendant, not an annulation)
      - the new system must contain at least `min_new` atoms, so the macro stops
        producing the 3-6 atom aziridines/azetines/oxathiazolidines it currently
        makes instead of the 9-10 atom bicycles the winners carry
    """
    from rdkit import Chem
    m0 = Chem.MolFromSmiles(before_smiles)
    if m0 is None:
        return lambda smiles: False
    old_systems = [frozenset().union(*g) for g in ring_systems(m0)]
    n0 = len(old_systems)
    old_atoms = frozenset().union(*old_systems) if old_systems else frozenset()

    def ok(smiles: str) -> bool:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return False
        groups = ring_systems(m)
        if len(groups) <= n0:
            return False                      # annulated, or nothing new
        for g in groups:
            atoms = frozenset().union(*g)
            # a system disjoint from every pre-existing ring atom, big enough
            if not (atoms & old_atoms) and len(atoms) >= int(min_new):
                return True
        return False

    return ok


# ---------------------------------------------------------------------------
# MACRO CONTRACTS -- layer 2 of four.
#
#   1 global validity   narrow, universal, med_chem_gate. Calibrated so every
#                       benchmark seed and every published IVG winner passes.
#   2 macro contract    hard WITHIN a macro, never globally. "Not what this
#                       macro does" is not "bad chemistry".
#   3 soft prior        ranks legal endpoints (SA/QED/sim). Never prohibits.
#   4 purpose           docking decides which macro was the right move.
#
# The test every new hard condition must answer:
#     "Is this universally invalid chemistry, or merely not the behaviour of
#      this macro?"  If the latter, it belongs HERE, not in the gate.
#
# Concretely: one parp1 IVG winner carries a cyclopropane. It is perfectly valid
# chemistry, it is simply not what APPEND_SYSTEM builds -- SMALL_RING does. A
# global "ring >= 5" rule would have rejected a winning molecule, which is the
# same mistake the 7-ring ban would have made.
# ---------------------------------------------------------------------------

#: BUILD_RING_SYSTEM is a PROGRAM, not a primitive. COMPOSE already has every
#: operator it needs -- cycle_close/cycle_open, bond_reorder/reroute,
#: atom_restate, ring_system_restate -- and the experiments showed ring FORMATION
#: works once the macro semantics are right (multi-system construction went
#: 12% -> 33% with longer growth -> 80% with a disjoint-closure contract). What
#: was missing was never a ring operator; it was growing the right precursor
#: before closing. So this compiles to existing operators.
BUILD_RING_SYSTEM = (("scaffold_extend", 8), ("append_system", 1), ("restate", 2))

#: Refinement variants explored AFTER a ring closes. We do not decide in advance
#: that the new ring "must become benzene": turning -C-C-C-C-C-C- aromatic needs
#: an alternating sequence of bond_reorder edits, and any single intermediate
#: Kekule state earns nothing. So the macro emits several legal realizations --
#: saturated, partially unsaturated, fully conjugated -- and PURPOSE decides
#: which mattered. Same delayed-credit logic as the macro level, one layer down.
REFINEMENT_BRANCHES = (("none", 0), ("light", 1), ("conjugate", 3), ("deep", 5))

MACRO_CONTRACTS: dict[str, str] = {
    "scaffold_extend": "add backbone-capable atoms; no terminal decoration",
    "append_system":   "closure forms a NEW ring system from newly grown material",
    "annulate":        "closure fuses INTO an existing ring system",
    "small_ring":      "form a 3-4 membered ring",
    "aromatize":       "reorganise bond orders/types; no atom count change",
    "decorate":        "add terminal substituents (halogens etc.) after construction",
    "shrink":          "remove material while preserving the core",
}

MACRO_FAMILIES.update({
    "append_system": ("cycle_close", "bond_insert"),
    "annulate":      ("cycle_close", "bond_insert"),
    "small_ring":    ("cycle_close", "bond_insert"),
    "decorate":      ("atom_insert",),
})


def contract_for(macro: str, before_smiles: str, **kwargs):
    """The hard, macro-local predicate. None means the macro has no contract
    beyond its family restriction."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdMD

    if macro == "scaffold_extend":
        return backbone_only(before_smiles, kwargs.get("composition", "mixed"))
    if macro == "append_system":
        return append_system_closure(before_smiles)
    if macro == "annulate":
        m0 = Chem.MolFromSmiles(before_smiles)
        n0 = len(ring_systems(m0)) if m0 is not None else 0
        def fused(smiles: str) -> bool:
            m = Chem.MolFromSmiles(smiles)
            if m is None:
                return False
            # a ring was added but the SYSTEM count did not rise -> it fused in
            return (len(m.GetRingInfo().AtomRings())
                    > (len(m0.GetRingInfo().AtomRings()) if m0 else 0)
                    and len(ring_systems(m)) <= n0)
        return fused
    if macro == "small_ring":
        m0 = Chem.MolFromSmiles(before_smiles)
        before = sorted(len(r) for r in m0.GetRingInfo().AtomRings()) if m0 else []
        def small(smiles: str) -> bool:
            m = Chem.MolFromSmiles(smiles)
            if m is None:
                return False
            after = sorted(len(r) for r in m.GetRingInfo().AtomRings())
            b = list(before)
            for x in after:
                if x in b:
                    b.remove(x)
                elif x <= 4:
                    return True
            return False
        return small
    if macro == "decorate":
        m0 = Chem.MolFromSmiles(before_smiles)
        import collections
        before_c = collections.Counter(a.GetSymbol() for a in m0.GetAtoms()) if m0 else {}
        def deco(smiles: str) -> bool:
            m = Chem.MolFromSmiles(smiles)
            if m is None:
                return False
            after = collections.Counter(a.GetSymbol() for a in m.GetAtoms())
            added = after - before_c
            return bool(set(added) & TERMINAL_ELEMENTS)
        return deco
    if macro == "shrink":
        m0 = Chem.MolFromSmiles(before_smiles)
        n0 = m0.GetNumHeavyAtoms() if m0 is not None else 0
        def shrunk(smiles: str) -> bool:
            m = Chem.MolFromSmiles(smiles)
            return m is not None and m.GetNumHeavyAtoms() < n0
        return shrunk
    if macro == "aromatize":
        m0 = Chem.MolFromSmiles(before_smiles)
        n0 = m0.GetNumAtoms() if m0 is not None else 0
        def arom(smiles: str) -> bool:
            m = Chem.MolFromSmiles(smiles)
            return m is not None and m.GetNumAtoms() == n0
        return arom
    return None


def build_ring_system(enumerate_fn, apply_fn, gate_fn, start: str, rng,
                      extend_len: int = 8, topology: str = "append_system",
                      branches=REFINEMENT_BRANCHES, **kw):
    """Compile BUILD_RING_SYSTEM into primitive transitions and return every
    refinement realization, unjudged.

    Phases: extend precursor -> choose topology -> close -> refine (branching)
    -> (decorate happens later, as its own macro, so a halogen never consumes a
    backbone growth step).

    Returns a list of (label, MacroRollout-like dict). The caller scores the
    ENDPOINTS; nothing here inspects purpose, so an intermediate that looks
    worse cannot be pruned before the program pays.
    """
    out = []
    # 1. precursor: backbone-capable atoms only
    pre = rollout(enumerate_fn, apply_fn, gate_fn, start, "scaffold_extend",
                  extend_len, rng, prefer_fn=contract_for("scaffold_extend", start), **kw)
    if pre.halted and pre.length == 0:
        return [dict(label="halted_extend", smiles=start, halted=pre.halted)]
    grown = pre.endpoint

    # 2+3. topology choice and closure, under that macro's contract
    closed = rollout(enumerate_fn, apply_fn, gate_fn, grown, topology, 1, rng,
                     prefer_fn=contract_for(topology, grown), **kw)
    if closed.length == 0:
        return [dict(label="halted_close", smiles=grown, halted=closed.halted,
                     extend_n=pre.length)]
    ring = closed.endpoint

    # 4. refinement BRANCHES from the same closed state -- not one path
    for label, k in branches:
        if k == 0:
            out.append(dict(label="refine:none", smiles=ring, extend_n=pre.length,
                            refine_n=0))
            continue
        r = rollout(enumerate_fn, apply_fn, gate_fn, ring, "restate", k,
                    np.random.default_rng(int(rng.integers(0, 2**31))), **kw)
        out.append(dict(label=f"refine:{label}", smiles=r.endpoint,
                        extend_n=pre.length, refine_n=r.length, halted=r.halted))
    return out


# ---------------------------------------------------------------------------
# ORTHOGONAL RING TAXONOMY
#
#   BUILD_RING_SYSTEM(topology, scale, refinement)
#
# topology and electronic state are INDEPENDENT axes. There is no
# BUILD_AROMATIC_SPIRO_RING: you ask for spiro, then refine the electronic
# state separately. Every combination compiles to the same primitives.
#
# NONE of these is globally banned. A bridged ring is not wrong, a spirocycle
# is not right, a cyclopropane can be excellent. The three questions stay
# separate:
#     global gate      is this chemically legitimate?
#     macro contract   did you build the topology I asked for?
#     purpose          was that topology useful here?
# ---------------------------------------------------------------------------

RING_TOPOLOGIES = ("pendant", "fused", "spiro", "bridged")
RING_SCALES = {"small": (3, 4), "medium": (5, 6), "large": (7, 10)}


def _ring_relationship(before_smiles: str):
    """Classify how a product's new ring relates to the pre-existing systems.

    Uses RDKit DESCRIPTOR DELTAS rather than atom-index correspondence. The
    first version substructure-matched the whole before-molecule against the
    product, which returns nothing whenever the product is not a superstructure
    (a closure rearranges bonds, an annulation consumes the chain), so every
    case classified as None. Deltas need no index mapping and work for closures
    and insertions alike.

      pendant  ring-SYSTEM count rises        (a separate system appeared)
      spiro    spiro-atom count rises         (systems joined at one atom)
      bridged  bridgehead count rises         (joined at >=2 non-adjacent atoms)
      fused    ring count rises, systems flat, no new spiro/bridgehead
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdMD
    m0 = Chem.MolFromSmiles(before_smiles)
    if m0 is None:
        return lambda smiles: None
    b = dict(systems=len(ring_systems(m0)),
             rings=len(m0.GetRingInfo().AtomRings()),
             spiro=int(rdMD.CalcNumSpiroAtoms(m0)),
             bridge=int(rdMD.CalcNumBridgeheadAtoms(m0)))

    def classify(smiles: str):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        a = dict(systems=len(ring_systems(m)),
                 rings=len(m.GetRingInfo().AtomRings()),
                 spiro=int(rdMD.CalcNumSpiroAtoms(m)),
                 bridge=int(rdMD.CalcNumBridgeheadAtoms(m)))
        if a["rings"] <= b["rings"]:
            return None                       # no new ring
        if a["bridge"] > b["bridge"]:
            return "bridged"
        if a["spiro"] > b["spiro"]:
            return "spiro"
        if a["systems"] > b["systems"]:
            return "pendant"
        return "fused"

    return classify


def ring_topology_contract(before_smiles: str, topology: str, scale: str | None = None):
    """Hard contract for BUILD_RING_SYSTEM(topology, scale)."""
    if topology not in RING_TOPOLOGIES:
        raise KeyError(f"unknown topology {topology!r}; known: {RING_TOPOLOGIES}")
    classify = _ring_relationship(before_smiles)
    lo, hi = RING_SCALES.get(scale, (3, 20)) if scale else (3, 20)
    from rdkit import Chem
    m0 = Chem.MolFromSmiles(before_smiles)
    before_sizes = sorted(len(r) for r in m0.GetRingInfo().AtomRings()) if m0 else []

    def ok(smiles: str) -> bool:
        if classify(smiles) != topology:
            return False
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return False
        after = sorted(len(r) for r in m.GetRingInfo().AtomRings())
        b = list(before_sizes)
        new_sizes = []
        for x in after:
            if x in b:
                b.remove(x)
            else:
                new_sizes.append(x)
        return any(lo <= x <= hi for x in new_sizes) if new_sizes else False

    return ok


# ---------------------------------------------------------------------------
# ELECTRONIC STATE AS MACRO INTENT
#
# A ring cannot be aromatic while half-built -- aromaticity is a property of the
# COMPLETED cycle. So the state is not something to repair afterwards; it is the
# macro's ENDPOINT REQUIREMENT. Measured: of 153 pendant rings built by the
# carbon-rich arms, 87 (57%) are aromatisable in principle (size 5/6, no
# quaternary carbon, C/N/O/S only) yet only ONE came out aromatic, because
# nothing ever asked. Adding more generic `restate` steps made it worse, not
# better -- the restate5 arm produced 0 aromatic rings and halved feasible
# yield, because unguided restatement wanders rather than conjugating.
#
# Nothing is forced: saturated is a legitimate branch, and purpose decides.
# ---------------------------------------------------------------------------

RING_STATES = ("saturated", "unsaturated", "aromatic")


def ring_state_contract(before_smiles: str, state: str):
    """Endpoint requirement on the electronic state of the NEW ring system."""
    if state not in RING_STATES:
        raise KeyError(f"unknown state {state!r}; known: {RING_STATES}")
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdMD
    m0 = Chem.MolFromSmiles(before_smiles)
    n_arom0 = int(rdMD.CalcNumAromaticRings(m0)) if m0 is not None else 0
    n_ring0 = len(m0.GetRingInfo().AtomRings()) if m0 is not None else 0

    def ok(smiles: str) -> bool:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return False
        rings = m.GetRingInfo().AtomRings()
        if len(rings) <= n_ring0:
            return False                      # no new ring at all
        n_arom = int(rdMD.CalcNumAromaticRings(m))
        if state == "aromatic":
            return n_arom > n_arom0
        if state == "saturated":
            return n_arom == n_arom0
        # unsaturated: a new ring with some multiple bond but not aromatic
        if n_arom != n_arom0:
            return False
        return any(b.GetBondType() != Chem.BondType.SINGLE and b.IsInRing()
                   for b in m.GetBonds())

    return ok


def aromatisable_ring(mol, ring) -> bool:
    """Could this ring plausibly become aromatic? Size 5/6, no atom carrying more
    than one exocyclic heavy neighbour, C/N/O/S only. Used to decide whether an
    aromatic-state request is even worth attempting from a given precursor."""
    if len(ring) not in (5, 6):
        return False
    for a in ring:
        at = mol.GetAtomWithIdx(a)
        if at.GetSymbol() not in ("C", "N", "O", "S"):
            return False
        if sum(1 for nb in at.GetNeighbors() if nb.GetIdx() not in ring) > 1:
            return False
    return True


def local_extend(before_smiles: str, seed_smiles: str, max_anchors: int = 2,
                 max_separation: int = 2, avoid_aromatic: bool = True):
    """Confine construction to ONE attachment neighbourhood on the seed.

    Measured on 200 parp1 endpoints. The three that reached T4 similarity
    (0.508-0.557 at +8 heavy atoms) all independently anchored at C0/C2 or
    C0/C3 -- the peripheral ALIPHATIC dimethylamino arm, separation 2, changing
    3-4 radius-2 environments. Every failure (sim 0.115-0.131) anchored at 5-8
    atoms spread 1-9 apart INCLUDING the aromatic core (C5ar, C6ar, C15ar,
    C16ar), changing 12-16 environments.

    Direct anchors and changed environments are not the same thing: Morgan
    radius propagates, so each extra anchor costs ~2.5 environments and each
    environment costs fingerprint bits. Growth AMOUNT was identical (+8) across
    both groups -- locality, not size, is what preserves similarity.
    """
    from rdkit import Chem
    sm = Chem.MolFromSmiles(seed_smiles)
    if sm is None:
        return lambda smiles: False
    dmat = Chem.GetDistanceMatrix(sm)

    def ok(smiles: str) -> bool:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return False
        mt = m.GetSubstructMatch(sm)
        if not mt:
            return False                     # seed rewritten -> unbounded cost
        inseed = set(mt)
        anchors = [i for i, a in enumerate(mt)
                   if any(nb.GetIdx() not in inseed
                          for nb in m.GetAtomWithIdx(a).GetNeighbors())]
        if not anchors or len(anchors) > int(max_anchors):
            return False
        if len(anchors) > 1:
            worst = max(dmat[i][j] for i in anchors for j in anchors if i < j)
            if worst > max_separation:
                return False
        if avoid_aromatic and any(sm.GetAtomWithIdx(i).GetIsAromatic()
                                  for i in anchors):
            return False
        return True

    return ok


def environment_complexity(mol, atoms, radius: int = 2) -> float:
    """Distinct radius-r environments per atom in a substructure.

    A benzene scores low because its six carbons are equivalent; an irregular
    heterocycle scores high because every atom sits in a different environment.
    This is the quantity that costs fingerprint budget: novel STRUCTURE is only
    expensive to the extent it is novel in many DISTINCT ways.
    """
    from rdkit import Chem
    from rdkit.Chem import rdmolops
    envs = set()
    for a in atoms:
        try:
            env = rdmolops.FindAtomEnvironmentOfRadiusN(mol, radius, int(a))
            if env:
                amap = {}
                envs.add(Chem.MolToSmiles(Chem.PathToSubmol(mol, env, atomMap=amap)))
            else:
                envs.add(mol.GetAtomWithIdx(int(a)).GetSymbol())
        except Exception:
            envs.add(f"?{a}")
    return len(envs) / max(len(atoms), 1)


def regular_ring_closure(before_smiles: str, max_complexity: float = 0.75,
                         allowed=frozenset({"C", "N"}), sizes=(5, 6)):
    """Closure must produce a COMPACT, REGULAR, aromatic-compatible ring.

    Deliberately NOT a scaffold whitelist. Measured: our locality-constrained
    endpoints already beat IVG on similarity (0.561 at 4 lost bits vs their
    0.424 at 13) but fail SA in 20/24 and QED in 18/24, because the new rings
    are bespoke -- dithiolane, morpholine, C1=NN=CC1, one different heterocycle
    every time -- against IVG's two repeated benzene-fused bicycles. We spend 33
    new fingerprint bits where they spend 20.

    So the criterion is the generic property, not their scaffolds: add mass with
    FEW DISTINCT new environments. That attacks the new-bit cost and the SA cost
    with one condition, and leaves COMPOSE free to discover any regular ring.
    """
    from rdkit import Chem
    m0 = Chem.MolFromSmiles(before_smiles)
    before = sorted(len(r) for r in m0.GetRingInfo().AtomRings()) if m0 else []

    def ok(smiles: str) -> bool:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return False
        rings = m.GetRingInfo().AtomRings()
        b = list(before)
        new = []
        for r in rings:
            if len(r) in b:
                b.remove(len(r))
            else:
                new.append(r)
        if not new:
            return False
        for r in new:
            if len(r) not in sizes:
                return False
            if any(m.GetAtomWithIdx(a).GetSymbol() not in allowed for a in r):
                return False
            if not aromatisable_ring(m, r):
                return False
            if environment_complexity(m, r) > max_complexity:
                return False
        return True

    return ok


def aromatic_target_of(smiles: str, ring):
    """The aromatic form of one ring, or None if it cannot exist.

    Used to give the AROMATIZE macro an explicit intermediate target. Without
    one, aromatisation is unguidable: building a benzene from a hexyl chain
    needs a closure plus alternating bond changes, and NO individual edit
    improves any local measure. Measured on the 5ht1b seed, a greedy compiler
    laid down the six-carbon chain in six well-ranked steps (R_theta ranks
    13,16,6,9,12,10) and then oscillated between two Kekule states forever,
    flipping one bond back and forth at ranks 7 and 1 until its budget ran out.
    """
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    rw = Chem.RWMol(m)
    rs = set(ring)
    for a in ring:
        rw.GetAtomWithIdx(int(a)).SetIsAromatic(True)
    for b in rw.GetBonds():
        if b.GetBeginAtomIdx() in rs and b.GetEndAtomIdx() in rs:
            b.SetBondType(Chem.BondType.AROMATIC)
            b.SetIsAromatic(True)
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return Chem.MolToSmiles(out)
    except Exception:
        return None


def aromatisation_targets(smiles: str):
    """Every aromatic form reachable by aromatising one currently-saturated
    ring of this molecule. These are explicit endpoints an AROMATIZE macro can
    compile toward, so the intermediate Kekule states need no local reward."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdMD
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return []
    n0 = int(rdMD.CalcNumAromaticRings(m))
    out = []
    for ring in m.GetRingInfo().AtomRings():
        if all(m.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            continue
        if not aromatisable_ring(m, ring):
            continue
        t = aromatic_target_of(smiles, ring)
        if t is None:
            continue
        tm = Chem.MolFromSmiles(t)
        if tm is not None and int(rdMD.CalcNumAromaticRings(tm)) > n0:
            out.append(t)
    return sorted(set(out))


def _compile_to_target(enumerate_fn, apply_fn, gate_fn, start: str, target: str,
                       max_steps: int = 6):
    """Greedy walk toward an EXPLICIT target molecule.

    Only usable because the target is known: the intermediate Kekule states of
    an aromatisation are individually worthless, so distance-to-target is the
    only signal that orders them. Given no target, a greedy walk oscillates --
    measured on the 5ht1b seed, it flipped one bond between ranks 7 and 1 for
    eight consecutive steps without converging.
    """
    from rdkit import Chem
    tm = Chem.MolFromSmiles(target)
    if tm is None:
        return start, []
    tgt_can = Chem.MolToSmiles(tm)
    tb = set()
    from rdkit.Chem import rdFingerprintGenerator
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    tfp = gm.GetFingerprint(tm)
    cur, steps = start, []
    for _ in range(int(max_steps)):
        if Chem.MolToSmiles(Chem.MolFromSmiles(cur)) == tgt_can:
            return cur, steps
        try:
            families, probs, handles = enumerate_fn(cur)
        except Exception:
            return cur, steps
        best = None
        for j in range(len(families)):
            y = apply_fn(cur, handles[j])
            if not y or not gate_fn(y):
                continue
            ym = Chem.MolFromSmiles(y)
            if ym is None:
                continue
            from rdkit import DataStructs
            d = DataStructs.TanimotoSimilarity(tfp, gm.GetFingerprint(ym))
            if best is None or d > best[0]:
                best = (d, y, families[j])
        if best is None or best[1] == cur:
            return cur, steps
        steps.append(best[2])
        cur = best[1]
    return cur, steps


def build_ring_system(enumerate_fn, apply_fn, gate_fn, start: str, rng,
                      topology: str = "pendant", size: int = 6,
                      composition: str = "carbon_rich", state: str = "aromatic",
                      **kw):
    """ONE temporally extended action. Judged only at the endpoint.

    grow -> close -> complete the requested electronic state, as an indivisible
    unit. This is not "build an arbitrary saturated ring and aromatise it
    later": when state='aromatic' the whole realisation is planned with an
    aromatic endpoint, and the electronic completion is an internal subroutine
    rather than a separate macro the controller has to rediscover.

    The earlier implementation attached state as a condition on the CLOSURE
    action, which no single action can satisfy -- a closure cannot make a ring
    aromatic on its own -- so it never fired and fell back. Splitting the
    sequence into locally judged steps then let it wander: R_theta ranks every
    required edit highly (carbon insertions at ranks 6-16, bond reorders at 1-7),
    but no individual edit improves any local measure until all of them are done.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdMD
    m0 = Chem.MolFromSmiles(start)
    arom0 = int(rdMD.CalcNumAromaticRings(m0)) if m0 is not None else 0
    sys0 = len(ring_systems(m0)) if m0 is not None else 0

    grown = rollout(enumerate_fn, apply_fn, gate_fn, start, "scaffold_extend",
                    int(size), rng,
                    prefer_fn=backbone_only(start, composition), **kw)
    cur = grown.endpoint
    # The closure must build the ring from the material THIS macro grew, and it
    # must accept EITHER closure route: the local primitive fiber offers 127
    # bond_insert and zero cycle_close at a six-carbon precursor, while the
    # production law at the same kind of state offers cycle_close in quantity.
    # CycleCloseEdge and BondInsert share the (a, b, order) shape, so one
    # descriptor match covers both; hard-coding either family alone silently
    # empties the candidate set on the other support. Requiring only "one more
    # ring system" produced epoxides and bridged bicycles while the chain sat
    # unclosed.
    closed = rollout(enumerate_fn, apply_fn, gate_fn, cur, "cyclize", 1, rng,
                     prefer_fn=closure_from_new_atoms(start, cur, topology,
                                                      int(size)), **kw)
    cur = closed.endpoint
    trace = [f"grow{grown.length}", f"close{closed.length}"]

    if state == "aromatic":
        # ring_system_restate reaches the aromatic form of a saturated pendant in
        # ONE action, ranked 2nd by R_theta (measured: cyclohexyl pendant ->
        # phenyl pendant, depth 1). No multi-bond Kekule compilation is needed;
        # an earlier diagnosis that it required three coordinated bond_reorders
        # was wrong. The contract simply requires the aromatic count to rise.
        def _more_aromatic(y, _a0=arom0):
            mm = Chem.MolFromSmiles(y)
            return mm is not None and int(rdMD.CalcNumAromaticRings(mm)) > _a0
        r = rollout(enumerate_fn, apply_fn, gate_fn, cur, "restate", 1, rng,
                    prefer_fn=_more_aromatic, **kw)
        em = Chem.MolFromSmiles(r.endpoint)
        ok = em is not None and int(rdMD.CalcNumAromaticRings(em)) > arom0
        return dict(smiles=r.endpoint, trace=trace + [f"restate{r.length}"],
                    satisfied=ok,
                    arom_gain=(int(rdMD.CalcNumAromaticRings(em)) - arom0) if em else 0,
                    sys_gain=(len(ring_systems(em)) - sys0) if em else 0,
                    reason=None if ok else "restate did not aromatise")
    em = Chem.MolFromSmiles(cur)
    return dict(smiles=cur, trace=trace, satisfied=em is not None,
                arom_gain=(int(rdMD.CalcNumAromaticRings(em)) - arom0) if em else 0,
                sys_gain=(len(ring_systems(em)) - sys0) if em else 0)


def closure_from_new_atoms(before_growth: str, after_growth: str,
                           topology: str = "pendant", size: int | None = None):
    """Require the requested ring to be built from the atoms THIS MACRO grew.

    Provenance belongs to the topology, not to validity. A pendant system must
    close among new atoms; an annulation must deliberately involve old ring
    atoms. Applying a global "new atoms only" rule would break fused and spiro
    construction, which are legitimate requested topologies.

    Motivating measurement: the pendant contract previously required only that
    the ring-SYSTEM count rise, so the closure fired on whatever atoms were
    convenient and returned epoxides, cyclobutenes and bridged bicycles, none of
    them aromatisable. The six-carbon chain the macro had just grown sat
    unclosed.
    """
    from rdkit import Chem
    b = Chem.MolFromSmiles(before_growth)
    a = Chem.MolFromSmiles(after_growth)
    if b is None or a is None:
        return lambda smiles: False
    mt = a.GetSubstructMatch(b)
    new = set(range(a.GetNumAtoms())) - set(mt) if mt else set()
    if not new:
        return lambda smiles: False
    # identify the grown material by its canonical fragment, so it can be found
    # again in the product regardless of atom renumbering
    frag = Chem.MolFragmentToSmiles(a, atomsToUse=sorted(new))
    q = Chem.MolFromSmarts(frag) if frag else None
    n_new = len(new)

    def ok(smiles: str) -> bool:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return False
        rings = m.GetRingInfo().AtomRings()
        if not rings:
            return False
        prior = b.GetRingInfo().NumRings()
        if len(rings) <= prior:
            return False                      # no new ring
        old = set()
        mt2 = m.GetSubstructMatch(b)
        if mt2:
            old = set(mt2)
        for r in rings:
            if size is not None and len(r) != int(size):
                continue
            outside = [x for x in r if x not in old]
            if topology == "pendant":
                # every ring atom must be material this macro added
                if len(outside) == len(r) and len(r) <= n_new:
                    return True
            else:
                # fused / spiro / bridged deliberately involve old ring atoms
                if outside and len(outside) < len(r):
                    return True
        return False

    return ok


# ---------------------------------------------------------------------------
# EXACT CLOSURE BY ENUMERATE-AND-FILTER.
#
# Rejection sampling cannot find this action. Measured on a real six-carbon
# precursor: 127 legal bond_insert actions, exactly 2 of which close the chain
# into a six-membered carbocycle (1.6%). With 12 draws that succeeds ~18% of the
# time, and 2 of 16 qualification runs reached the intended topology while the
# rest fell back and closed whatever was legal.
#
# The admissible set is small and enumerable, so enumerate it, filter it
# exactly, and let R_theta rank the survivors. An empty filtered set returns
# UNSAT rather than falling back -- every predicate in this file that could not
# evaluate previously failed OPEN, which is how fused rings passed a pendant
# contract.
#
# Provenance is tracked through execution in the STATE index space, never
# recovered by rematching the seed afterwards: 5 of 16 products had the seed
# substructure destroyed, and a rematch then reports "no old atoms", making
# every ring look newly grown.
# ---------------------------------------------------------------------------

def enumerate_closures(law_families, law_probs, law_handles, apply_state_fn,
                       n_before: int, new_slots, size: int, topology: str = "pendant"):
    """Return (index, prob, product) for closures satisfying the topology
    contract exactly, ranked by R_theta. Empty list means UNSAT.

    `new_slots` are the state-space indices this macro created. For a pendant,
    BOTH endpoints must be new and the ring formed must consist only of new
    atoms and have the requested size.
    """
    import numpy as np
    from rdkit import Chem
    out = []
    newset = set(int(x) for x in new_slots)
    for j, fam in enumerate(law_families):
        if fam not in ("bond_insert", "cycle_close"):
            continue
        y = apply_state_fn(j)
        if y is None:
            continue
        m = Chem.MolFromSmiles(y)
        if m is None:
            continue
        rings = m.GetRingInfo().AtomRings()
        hit = False
        for r in rings:
            if len(r) != int(size):
                continue
            # the ring must be entirely carbon-and-new for a pendant; we verify
            # composition on the product because slot identity is preserved by
            # the executor, so a ring of exactly `size` new atoms is the one the
            # macro grew
            if topology == "pendant":
                if all(m.GetAtomWithIdx(a).GetSymbol() == "C" for a in r):
                    hit = True
                    break
            else:
                hit = True
                break
        if hit:
            out.append((j, float(law_probs[j]), y))
    out.sort(key=lambda t: -t[1])
    return out


def enumerate_aromatisations(law_families, law_probs, apply_state_fn, arom_before: int):
    """Closures done; now the electronic state. ring_system_restate reaches the
    aromatic form of a saturated carbocycle in ONE action (measured rank 2), so
    this filters for an aromatic-count increase and ranks by R_theta."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdMD
    out = []
    for j, fam in enumerate(law_families):
        if fam not in ("ring_system_restate", "atom_restate_semantic",
                       "atom_restate", "bond_reorder"):
            continue
        y = apply_state_fn(j)
        if y is None:
            continue
        m = Chem.MolFromSmiles(y)
        if m is None:
            continue
        if int(rdMD.CalcNumAromaticRings(m)) > int(arom_before):
            out.append((j, float(law_probs[j]), y))
    out.sort(key=lambda t: -t[1])
    return out


# ---------------------------------------------------------------------------
# RING FRONTIER: compile the requested topology into a MINIMAL candidate set.
#
# Macros restrict support; they never invent it:
#       A_m(x) = A(x) INTERSECT C_m(x)
# where A(x) is the executor's exact legal support and C_m(x) is the macro's
# topology condition. C_m is cheap graph logic computed FIRST, so only a couple
# of descriptors are ever checked against A(x).
#
# Materialising the whole closure fiber is a reference test, not production: at
# a real six-carbon precursor that is 127 bond_insert applications (local
# fiber) or ~150 (production law) to find the 2 that matter. Tracking the frontier during growth turns it into one endpoint
# pair. An earlier attempt filtered the PRODUCT for "contains a size-6
# all-carbon ring" instead, which is vacuous on any seed that already has a
# benzene -- as this one does -- so 28-91 irrelevant closures passed.
# ---------------------------------------------------------------------------

@dataclass
class RingFrontier:
    """Atoms this macro grew, in attachment order, plus the requested shape."""
    path: list = field(default_factory=list)   # state-space slots, in order
    anchors: list = field(default_factory=list)  # pre-existing atoms attached to
    size: int = 6
    topology: str = "pendant"

    def observe_insert(self, action) -> None:
        """Record one atom_insert from its descriptor: `slot` is the new atom,
        `neighbors` is what it attached to."""
        slot = int(getattr(action, "slot"))
        nbrs = [int(n[0]) for n in (getattr(action, "neighbors", ()) or ())]
        if self.path and any(n == self.path[-1] for n in nbrs):
            self.path.append(slot)              # extends the chain
        elif not self.path:
            self.path.append(slot)
            self.anchors.extend(nbrs)
        else:
            self.path.append(slot)              # branch; closure may not apply

    def closure_pairs(self):
        """The endpoint pairs a closure should consider, given the topology."""
        if self.topology == "pendant":
            # close the newly grown path end to end: exactly one pair
            if len(self.path) >= 3:
                return [(self.path[0], self.path[-1])]
            return []
        if self.topology == "fused":
            # attach the new path across an existing edge: both anchors
            if self.anchors and self.path:
                return [(a, self.path[-1]) for a in self.anchors[:2]]
            return []
        if self.topology == "spiro":
            if self.anchors and len(self.path) >= 2:
                return [(self.anchors[0], self.path[-1])]
            return []
        if self.topology == "bridged":
            return [(a, b) for a in self.anchors for b in self.path[1:-1]][:4]
        return []


def match_closure_descriptors(law_families, law_actions, pairs):
    """Indices of legal actions realising one of `pairs`. No application, no
    successor materialisation -- descriptor matching only.

    Returns [] when the intersection is empty, which the caller must treat as
    UNSAT. Nothing here falls back to an unrestricted set: every predicate in
    this file that could not evaluate previously failed OPEN, which is how
    fused rings passed a pendant contract and how a vacuous product filter
    admitted 28-91 irrelevant closures.
    """
    want = {(min(a, b), max(a, b)) for a, b in pairs}
    out = []
    for j, fam in enumerate(law_families):
        if fam not in ("bond_insert", "cycle_close"):
            continue
        act = law_actions[j]
        a = getattr(act, "a", None)
        b = getattr(act, "b", None)
        if a is None or b is None:
            continue
        if (min(int(a), int(b)), max(int(a), int(b))) in want:
            out.append(j)
    return out


# `atom_type` is an ELEMENT INDEX into ELEMENTS, not a vocabulary row. Decoding
# it twice with element_of() produced "R_theta(C)=0.000" twice in a row.
ELEMENT_CODE = {"C": 2, "N": 3, "O": 4, "F": 5, "P": 6, "S": 7,
                "Cl": 8, "Br": 9, "I": 10}
COMPOSITION_CODES = {
    "carbon_rich": {ELEMENT_CODE["C"]},
    "mixed": {ELEMENT_CODE["C"], ELEMENT_CODE["N"], ELEMENT_CODE["O"]},
    "hetero_rich": {ELEMENT_CODE["N"], ELEMENT_CODE["O"], ELEMENT_CODE["S"]},
}


def match_growth_descriptors(fams, acts, probs, tip, allowed_elements,
                             anchors=None, bond_order=None):
    """Legal atom_insert actions that extend the chain at `tip` with an atom of
    the requested composition, ranked by R_theta. Descriptor-only.

    When `tip` is None this is the first insertion, so any attachment point in
    `anchors` (or anywhere, if anchors is None) is admissible.

    Restricting growth by descriptor rather than by rejection matters for a
    reason measured directly: carbon is only 7.5-7.9% of legal atom_insert
    actions, because the action space is over (element, valence) classes and S
    has three valences, I three, P two, while C has one. Sampling and testing
    wastes ~13 draws per carbon; selecting on atom_type costs nothing.
    """
    ok = []
    anchorset = None if anchors is None else {int(a) for a in anchors}
    for j, fam in enumerate(fams):
        if fam != "atom_insert":
            continue
        a = acts[j]
        if int(getattr(a, "atom_type", -1)) not in allowed_elements:
            continue
        nb_raw = list(getattr(a, "neighbors", ()) or ())
        nbrs = [int(n[0]) for n in nb_raw]
        if len(nbrs) != 1:
            continue                      # a chain step attaches at one point
        if bond_order is not None and int(nb_raw[0][1]) != int(bond_order):
            # Aromatic rings are built by laying an ALTERNATING pattern during
            # growth, because no insertion or closure in this process can
            # create an aromatic-order bond (measured: orders 1/2/3 only).
            # Aromaticity is perceived from the Kekule parity, never inserted.
            continue
        if tip is None:
            if anchorset is not None and nbrs[0] not in anchorset:
                continue
        elif nbrs[0] != int(tip):
            continue
        ok.append(j)
    ok.sort(key=lambda j: -float(probs[j]))
    return ok


def match_aromatisation_descriptors(fams, acts, probs, ring_atoms):
    """Restate/reorder actions confined to the ring this macro built.

    `ring_system_restate` carries no atom-scoped field: it is a
    RingSystemRestate whose payload is `changes`, a tuple of BondOrderChange
    (a, b, new_order) applied as ONE correlated event inside a cyclic block.
    That is exactly the aromatisation -- a hexagon's three alternating bonds go
    to order 2 together -- and it is what makes the electronic completion a
    single action rather than a three-step Kekule walk.

    Filtering on `.v` alone silently dropped every one of them: at the closed
    cyclohexyl state the production law offers 3 ring_system_restate actions at
    ranks 2, 3 and 27, and the macro reported zero candidates and returned
    UNSAT. The local primitive fiber has no ring_system_restate at all and
    reaches the same endpoint through bond_reorder, which production does not
    enable -- so the two supports need different descriptors for the same
    chemical step.
    """
    ring = {int(a) for a in ring_atoms}
    ok = []
    for j, fam in enumerate(fams):
        a = acts[j]
        changes = getattr(a, "changes", None)
        if changes:
            pts = []
            for c in changes:
                x, y = getattr(c, "a", None), getattr(c, "b", None)
                if x is None or y is None:
                    pts = None
                    break
                pts.extend((int(x), int(y)))
            # every bond the correlated event touches must lie in the ring this
            # macro built, so a restate cannot silently rewrite the seed
            if pts and all(v in ring for v in pts):
                ok.append(j)
            continue
        if fam in ("atom_restate", "atom_restate_semantic", "ring_system_restate"):
            v = getattr(a, "v", None)
            if v is not None and int(v) in ring:
                ok.append(j)
        elif fam == "bond_reorder":
            x, y = getattr(a, "a", None), getattr(a, "b", None)
            if x is not None and y is not None and int(x) in ring and int(y) in ring:
                ok.append(j)
    ok.sort(key=lambda j: -float(probs[j]))
    return ok


def build_ring_system_exact(enum_full_fn, apply_fn, to_smiles, gate_fn, seed_state,
                            size: int = 6, topology: str = "pendant",
                            composition: str = "carbon_rich",
                            state: str = "aromatic", anchors=None,
                            anchor_rank: int = 0, max_aromatise: int = 24,
                            stoich=None, refine: int = 0, refine_rng=None):
    """BUILD_RING_SYSTEM by descriptor compilation.

    Operates on STATES, not SMILES: `enum_full_fn(state) -> (families, actions,
    probabilities)`, `apply_fn(state, j) -> state|None`, `to_smiles(state) ->
    str`. Threading the state is what makes slot provenance meaningful --
    round-tripping through SMILES between steps renumbers the atoms, so the
    slot recorded at insertion names a different atom one step later and the
    chain never extends past its first bond.

    At every step the macro compiles its contract
    into candidate descriptors, intersects them with the executor's exact legal
    support, and lets frozen R_theta rank what survives. An empty intersection
    is UNSAT: there is no fallback to an unrestricted gate anywhere in here.

    Provenance is carried forward in state-space slots through the whole
    construction and never recovered by rematching the seed against the
    product -- a rematch fails on 5 of 16 products, and it fails OPEN, so every
    ring then looks newly grown.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdMD
    seed = to_smiles(seed_state)
    m0 = Chem.MolFromSmiles(seed) if seed else None
    if m0 is None:
        return dict(status="UNSAT", stage="seed")
    arom0 = int(rdMD.CalcNumAromaticRings(m0))
    sys0 = len(ring_systems(m0))
    allowed = COMPOSITION_CODES.get(composition, COMPOSITION_CODES["carbon_rich"])

    # EXACT STOICHIOMETRY BY QUOTA, not by a permissive allowed-element set.
    #
    # `composition` is a SET: {C,N,O} only says "N is permitted somewhere", so a
    # request for "6-ring with exactly one N" is inexpressible -- the ranker may
    # place none, or six. `stoich` is a count vector ((C,5),(N,1)) and is
    # enforced as a RUNNING QUOTA: at each growth step only elements with
    # positive remaining quota are offered, and because the quota sums to the
    # number of positions still to fill, every admissible choice leaves the
    # residual completable. The invariant is checked, not assumed.
    #
    # R_theta ranking is untouched: the descriptor space is narrowed, and the
    # frozen law still orders whatever remains.
    _quota = None
    if stoich:
        _quota = {}
        for _e, _n in stoich:
            _z = ELEMENT_CODE.get(_e) if isinstance(_e, str) else int(_e)
            if _z is None:
                return dict(status="UNSAT", stage="stoich_element", element=_e)
            _quota[_z] = _quota.get(_z, 0) + int(_n)
        if sum(_quota.values()) != int(size):
            return dict(status="UNSAT", stage="stoich_sum",
                        requested=sum(_quota.values()), size=int(size))

    fr = RingFrontier(size=int(size), topology=topology)

    def _maybe_refine(st_ok):
        """REFINE_RING, applied to the ring THIS call just built.

        Runs here rather than in the caller because provenance dies at the
        exit: the caller receives SMILES, and one canonicalisation renumbers
        the atoms so `fr.path` would name a different ring. Default refine=0
        leaves every existing call byte-identical.
        """
        if int(refine) <= 0:
            return st_ok, []
        import numpy as _np
        _rng = refine_rng if refine_rng is not None else _np.random.default_rng(0)
        _st, _tr = refine_ring(enum_full_fn, apply_fn, to_smiles, gate_fn,
                               st_ok, list(fr.path), int(refine), _rng)
        if _st is None:
            return st_ok, [f"refine_unsat({_tr})"]
        return _st, list(_tr) if isinstance(_tr, list) else [str(_tr)]
    cur, trace = seed_state, []

    # --- 1. grow the chain, one descriptor-selected insertion at a time ------
    for step in range(int(size)):
        fams, acts, probs = enum_full_fn(cur)
        tip = fr.path[-1] if fr.path else None
        _allowed_step = allowed
        if _quota is not None:
            _allowed_step = {z for z, n in _quota.items() if n > 0}
            if not _allowed_step:
                return dict(status="UNSAT", stage=f"quota_exhausted{step}",
                            smiles=to_smiles(cur), trace=trace)
        cands = match_growth_descriptors(fams, acts, probs, tip, _allowed_step,
                                         anchors=anchors if tip is None else None)
        if tip is None and cands:
            # `anchor_rank` selects WHICH attachment site on the seed, by
            # R_theta order over distinct sites. The construction itself is
            # deterministic, so varying the RNG seed would run the same
            # program eight times; varying the anchor asks the real question,
            # which is whether the macro closes from eight different sites.
            by_site = {}
            for j in cands:
                site = int(acts[j].neighbors[0][0])
                by_site.setdefault(site, []).append(j)
            order = sorted(by_site, key=lambda st_: -float(probs[by_site[st_][0]]))
            if int(anchor_rank) >= len(order):
                return dict(status="UNSAT", stage="anchor_rank", smiles=to_smiles(cur),
                            trace=trace, n_anchor_sites=len(order))
            cands = by_site[order[int(anchor_rank)]]
            fr.anchors = [order[int(anchor_rank)]]
        if not cands:
            return dict(status="UNSAT", stage=f"grow{step}", smiles=to_smiles(cur),
                        trace=trace, frontier=list(fr.path))
        placed = False
        for j in cands[:6]:
            y = apply_fn(cur, j)
            if y is None or not gate_fn(to_smiles(y)):
                continue
            fr.observe_insert(acts[j])
            rank = sorted(range(len(probs)), key=lambda k: -probs[k]).index(j)
            trace.append(f"atom_insert@{acts[j].slot}(r{rank})")
            if _quota is not None:
                _zp = int(getattr(acts[j], "atom_type", -1))
                _quota[_zp] = _quota.get(_zp, 0) - 1
            cur, placed = y, True
            break
        if not placed:
            return dict(status="UNSAT", stage=f"grow{step}_blocked",
                        smiles=to_smiles(cur), trace=trace,
                        frontier=list(fr.path))

    # --- 2. closure: ONE candidate pair, not the 127-action fiber -----------
    pairs = fr.closure_pairs()
    if not pairs:
        return dict(status="UNSAT", stage="closure_pair",
                    smiles=to_smiles(cur), trace=trace)
    fams, acts, probs = enum_full_fn(cur)
    cands = match_closure_descriptors(fams, acts, pairs)
    n_fiber = sum(1 for f in fams if f in ("bond_insert", "cycle_close"))
    if not cands:
        return dict(status="UNSAT", stage="closure", smiles=to_smiles(cur),
                    trace=trace, closure_pair=pairs[0], n_fiber=n_fiber,
                    n_candidates=0)
    cands.sort(key=lambda j: -float(probs[j]))
    closed = None
    for j in cands:
        y = apply_fn(cur, j)
        if y is not None and gate_fn(to_smiles(y)):
            rank = sorted(range(len(probs)), key=lambda k: -probs[k]).index(j)
            trace.append(f"bond_insert({acts[j].a},{acts[j].b})(r{rank})")
            closed = y
            break
    if closed is None:
        return dict(status="UNSAT", stage="closure_apply", smiles=to_smiles(cur),
                    trace=trace, n_candidates=len(cands), n_fiber=n_fiber)
    cur = closed

    if state != "aromatic":
        cur, rf_trace = _maybe_refine(cur)
        m = Chem.MolFromSmiles(to_smiles(cur))
        return dict(status="OK", smiles=to_smiles(cur), trace=trace + rf_trace,
                    sys_gain=len(ring_systems(m)) - sys0,
                    arom_gain=int(rdMD.CalcNumAromaticRings(m)) - arom0,
                    n_fiber=n_fiber, n_candidates=len(cands),
                    refined=int(refine), frontier=list(fr.path))

    # --- 3. electronic state, confined to the ring just built ---------------
    fams, acts, probs = enum_full_fn(cur)
    aro = match_aromatisation_descriptors(fams, acts, probs, fr.path)
    n_restate = sum(1 for f in fams if f in ("atom_restate", "ring_system_restate",
                                             "bond_reorder"))
    ar_state, ar_trace = aromatise_ring(enum_full_fn, apply_fn, to_smiles,
                                        gate_fn, cur, fr.path, arom0,
                                        max_one_shot=max_aromatise)
    if ar_state is not None:
        ar_state, rf_trace = _maybe_refine(ar_state)
        sm = to_smiles(ar_state)
        mm = Chem.MolFromSmiles(sm)
        return dict(status="OK", smiles=sm,
                    trace=trace + list(ar_trace) + rf_trace,
                    sys_gain=len(ring_systems(mm)) - sys0,
                    arom_gain=int(rdMD.CalcNumAromaticRings(mm)) - arom0,
                    n_fiber=n_fiber, n_candidates=len(cands),
                    n_restate_fiber=n_restate, n_restate_candidates=len(aro),
                    refined=int(refine), frontier=list(fr.path))
    aromatise_reason = ar_trace
    for j in []:
        y = apply_fn(cur, j)
        ys = to_smiles(y) if y is not None else None
        if ys is None or not gate_fn(ys):
            continue
        mm = Chem.MolFromSmiles(ys)
        if mm is None or int(rdMD.CalcNumAromaticRings(mm)) <= arom0:
            continue
        rank = sorted(range(len(probs)), key=lambda k: -probs[k]).index(j)
        trace.append(f"{fams[j]}(r{rank})")
        return dict(status="OK", smiles=ys, trace=trace,
                    sys_gain=len(ring_systems(mm)) - sys0,
                    arom_gain=int(rdMD.CalcNumAromaticRings(mm)) - arom0,
                    n_fiber=n_fiber, n_candidates=len(cands),
                    n_restate_fiber=n_restate, n_restate_candidates=len(aro))
    m = Chem.MolFromSmiles(to_smiles(cur))
    return dict(status="UNSAT", stage="aromatise", smiles=to_smiles(cur),
                trace=trace, reason=aromatise_reason,
                n_restate_fiber=n_restate, n_restate_candidates=len(aro),
                sys_gain=len(ring_systems(m)) - sys0 if m else 0)


def kekule_reorder_descriptors(fams, acts, path):
    """The exact bond_reorder actions that Kekulise the ring `path`.

    A saturated carbocycle does not become aromatic in one primitive edit --
    measured here, 68 restate/reorder actions confined to the new ring and not
    one of them raises the aromatic count. It becomes aromatic in exactly
    len(path)//2 of them: alternate bonds (p0,p1), (p2,p3), (p4,p5) go to
    order 2. That set is fully determined by the ring, so it is compiled, not
    searched.

    Returns [(index, action), ...] in ring order, or [] if any required
    descriptor is missing from the legal support -- which is UNSAT, not a cue
    to reorder whatever else is available.
    """
    n = len(path)
    if n < 6 or n % 2:
        return []
    want = [(min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            for i in range(0, n, 2)]
    found = {}
    for j, fam in enumerate(fams):
        if fam != "bond_reorder":
            continue
        a = acts[j]
        x, y = getattr(a, "a", None), getattr(a, "b", None)
        if x is None or y is None or int(getattr(a, "new_order", 0)) != 2:
            continue
        e = (min(int(x), int(y)), max(int(x), int(y)))
        if e in want and e not in found:
            found[e] = (j, a)
    if len(found) != len(want):
        return []
    return [found[e] for e in want]


def aromatise_ring(enum_full_fn, apply_fn, to_smiles, gate_fn, state, path,
                   arom0: int, max_one_shot: int = 24):
    """Complete the electronic state of the ring `path`. Returns (state, trace)
    or (None, reason).

    Tries the one-shot route first -- the production kernel exposes a
    ring_system_restate family that the primitive fiber does not -- then the
    compiled Kekule sequence. Both are exact consequences of the contract; the
    second is not a relaxation of the first.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdMD

    def _arom(st):
        sm = to_smiles(st)
        m = Chem.MolFromSmiles(sm) if sm else None
        return (int(rdMD.CalcNumAromaticRings(m)) if m else -1), sm

    fams, acts, probs = enum_full_fn(state)
    for j in match_aromatisation_descriptors(fams, acts, probs, path)[:max_one_shot]:
        y = apply_fn(state, j)
        if y is None:
            continue
        a, sm = _arom(y)
        if a > arom0 and sm and gate_fn(sm):
            rank = sorted(range(len(probs)), key=lambda k: -probs[k]).index(j)
            return y, [f"{fams[j]}(r{rank})"]

    cur, trace = state, []
    for step in range(len(path) // 2):
        # Only the edge for THIS step is required. An earlier attempt demanded
        # all len(path)//2 descriptors at every step, which is unsatisfiable
        # after the first one: once (p0,p1) is order 2, "reorder (p0,p1) to
        # order 2" is no longer a legal action, so the plan came back empty at
        # step 1 and the ring stayed saturated.
        edge = (min(path[2 * step], path[2 * step + 1]),
                max(path[2 * step], path[2 * step + 1]))
        fams, acts, probs = enum_full_fn(cur)
        j = None
        for jj, fam in enumerate(fams):
            if fam != "bond_reorder":
                continue
            aa = acts[jj]
            x, y = getattr(aa, "a", None), getattr(aa, "b", None)
            if x is None or y is None:
                continue
            if int(getattr(aa, "new_order", 0)) != 2:
                continue
            if (min(int(x), int(y)), max(int(x), int(y))) == edge:
                j = jj
                break
        if j is None:
            return None, f"kekule_missing{edge}@{step}"
        y = apply_fn(cur, j)
        if y is None:
            return None, f"kekule_apply{edge}@{step}"
        cur = y
        trace.append(f"bond_reorder{edge}")
    a, sm = _arom(cur)
    if a > arom0 and sm and gate_fn(sm):
        return cur, trace
    return None, f"kekule_not_aromatic(arom={a})"


# ---------------------------------------------------------------------------
# FUSED RING CONSTRUCTION
#
# Pendant closes the grown path back on ITSELF. Fused is a different structural
# intent: the new ring must SHARE AN EDGE with an existing ring, so two of its
# atoms are pre-existing and adjacent.
#
#   pendant : grow k atoms, close path[0]-path[-1]
#   fused   : pick ring edge (u,v), grow k-2 atoms from u, close path[-1]-v
#
# Measured motivation: 10 of the 25 published InVirtuoGen T4 winners contain a
# fused ring pair, including all five of the strongest PARP1 molecules
# (DS -13.6, three fused pairs each, e.g. a c3ccc4c(c3)CNC(=O)c3cccn3C4=O
# tricyclic core). A pendant-only macro cannot construct those at all. Spiro and
# bridged appear in ZERO winners -- confirmed independently by RDKit's
# CalcNumSpiroAtoms and CalcNumBridgeheadAtoms both summing to zero -- so they
# are deliberately not built here.
#
# Everything stays in STATE SLOT space, read off the adjacency matrix. Going via
# SMILES to find rings would renumber the atoms and void the provenance, which
# is exactly the bug that made the first pendant implementation close nothing.
# ---------------------------------------------------------------------------

def ring_edges_in_state(state):
    """Bonds lying on a cycle, as (u, v) slot pairs.

    An edge is on a cycle iff its endpoints remain connected after removing it.
    Computed on `state.bonds` directly, never via RDKit, so the indices returned
    are the same slots the executor's action descriptors use.
    """
    import numpy as np
    B = np.asarray(state.bonds)
    n = int(getattr(state, "n_real_atoms", B.shape[0]))
    out = []
    for u in range(n):
        for v in range(u + 1, n):
            if B[u, v] <= 0:
                continue
            seen, stack = {u}, [u]
            while stack:
                x = stack.pop()
                if x == v:
                    break
                for y in range(n):
                    if y in seen or B[x, y] <= 0:
                        continue
                    if (x == u and y == v) or (x == v and y == u):
                        continue          # the edge under test is removed
                    seen.add(y); stack.append(y)
            if v in seen:
                out.append((u, v))
    return out


def fusable_edges(state, require_h: bool = True):
    """Ring edges whose BOTH endpoints can accept one more heavy neighbour.

    Fusing onto an edge turns both shared atoms into ring-fusion atoms, so each
    must give up an implicit hydrogen. An edge whose atoms are already fully
    substituted cannot be fused onto, and offering it would make the macro
    return UNSAT after paying for the whole growth phase.
    """
    import numpy as np
    h = np.asarray(state.implicit_h_counts)
    out = []
    for u, v in ring_edges_in_state(state):
        if require_h and (int(h[u]) < 1 or int(h[v]) < 1):
            continue
        out.append((u, v))
    return out


def ring_pair_relations(smiles: str):
    """Counts of {linked, fused, spiro, bridged} ring-pair relations.

    Shared-atom count is what separates them: 1 = spiro, 2 = ortho-fused,
    >=3 = bridged, 0 with a connecting bond = linked. Used as the fused
    macro's ENDPOINT CONTRACT so a realisation that accidentally produced a
    pendant or spiro cannot pass as fused -- the pendant contract failing open
    is how bridged bicycles and epoxides once satisfied a pendant request.
    """
    from collections import Counter
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles) if smiles else None
    c = Counter()
    if m is None:
        return c
    rings = [set(r) for r in m.GetRingInfo().AtomRings()]
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            sh = len(rings[i] & rings[j])
            if sh == 0:
                if any(m.GetBondBetweenAtoms(a, b) for a in rings[i] for b in rings[j]):
                    c["linked"] += 1
            elif sh == 1:
                c["spiro"] += 1
            elif sh == 2:
                c["fused"] += 1
            else:
                c["bridged"] += 1
    return c


def build_fused_ring_exact(enum_full_fn, apply_fn, to_smiles, gate_fn, seed_state,
                           size: int = 6, composition: str = "carbon_rich",
                           state: str = "aromatic", anchor_rank: int = 0,
                           candidate_edges=None, max_aromatise: int = 24,
                           stoich=None, refine: int = 0, refine_rng=None):
    """Build a ring FUSED to an existing one, by descriptor compilation.

    Pick an existing ring edge (u, v); grow size-2 new atoms starting at u;
    close the last new atom to v. Those two pre-existing adjacent atoms become
    the shared edge, which is what makes the product fused rather than pendant.

        A_fused(x) = A(x) INTERSECT C_fused(x)

    Same discipline as pendant: the contract is compiled into candidate
    descriptors first and intersected with the executor's exact legal support;
    frozen R_theta ranks only survivors; an empty intersection is UNSAT with no
    fallback.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdMD
    seed = to_smiles(seed_state)
    m0 = Chem.MolFromSmiles(seed) if seed else None
    if m0 is None:
        return dict(status="UNSAT", stage="seed")
    arom0 = int(rdMD.CalcNumAromaticRings(m0))
    sys0 = len(ring_systems(m0))
    rel0 = ring_pair_relations(seed)
    allowed = COMPOSITION_CODES.get(composition, COMPOSITION_CODES["carbon_rich"])

    edges = list(candidate_edges) if candidate_edges else fusable_edges(seed_state)
    if not edges:
        return dict(status="UNSAT", stage="no_fusable_edge")
    if int(anchor_rank) >= len(edges):
        return dict(status="UNSAT", stage="anchor_rank", n_edges=len(edges))
    u, v = edges[int(anchor_rank)]

    n_new = int(size) - 2                 # two ring atoms are the shared edge

    # EXACT STOICHIOMETRY ON A FUSED RING.
    #
    # The shared edge is INHERITED: fusing cannot transmute (u, v), so the
    # requested FULL-RING stoichiometry must first have the edge's own element
    # identities subtracted. What remains is the composition of the n_new atoms
    # this macro actually places.
    #
    # If the edge already spends more of an element than the request allows --
    # e.g. C5N1 requested onto an edge that is already N,N -- then no sequence
    # of insertions can satisfy it FROM THIS EDGE. That is a genuine
    # executor-UNSAT for this (edge, request) pair, not a compiler gap, and it
    # is reported per-edge so a different edge may still succeed.
    _quota = None
    if stoich:
        _req: dict[int, int] = {}
        for _e, _n in stoich:
            _z = ELEMENT_CODE.get(_e) if isinstance(_e, str) else int(_e)
            if _z is None:
                return dict(status="UNSAT", stage="stoich_element", element=_e)
            _req[_z] = _req.get(_z, 0) + int(_n)
        if sum(_req.values()) != int(size):
            return dict(status="UNSAT", stage="stoich_sum",
                        requested=sum(_req.values()), size=int(size))
        for _shared in (u, v):
            _zs = int(seed_state.atom_types[int(_shared)])
            if _req.get(_zs, 0) <= 0:
                return dict(status="UNSAT", stage="stoich_shared_edge",
                            edge=(int(u), int(v)), shared_element=_zs,
                            reason="shared edge element not available in request")
            _req[_zs] -= 1
        _quota = {z: n for z, n in _req.items() if n > 0}
        if sum(_quota.values()) != n_new:
            return dict(status="UNSAT", stage="stoich_residual",
                        residual=sum(_quota.values()), n_new=n_new)
    if n_new < 1:
        return dict(status="UNSAT", stage="size_too_small")

    fr = RingFrontier(size=int(size), topology="fused")
    fr.anchors = [u, v]
    cur, trace = seed_state, [f"fuse_edge({u},{v})"]
    # For an aromatic request, alternate the bond order laid down at each step
    # so the finished ring carries a Kekule pattern. Building saturated and
    # asking for aromatisation afterwards does not work: at the saturated fused
    # state the production law offered exactly ONE ring_system_restate and it
    # sets orders to 1, and the fusion atoms have no capacity for an
    # independent Kekule walk. This is an ORDERING fix, not new chemistry.
    parity = [1 if i % 2 == 0 else 2 for i in range(n_new)] if state == "aromatic" \
        else [None] * n_new

    for step in range(n_new):
        fams, acts, probs = enum_full_fn(cur)
        tip = fr.path[-1] if fr.path else None
        _allowed_step = allowed
        if _quota is not None:
            _allowed_step = {z for z, n in _quota.items() if n > 0}
            if not _allowed_step:
                return dict(status="UNSAT", stage=f"quota_exhausted{step}",
                            smiles=to_smiles(cur), trace=trace)
        cands = match_growth_descriptors(fams, acts, probs, tip, _allowed_step,
                                         anchors=[u] if tip is None else None,
                                         bond_order=parity[step])
        if not cands:
            return dict(status="UNSAT", stage=f"grow{step}", smiles=to_smiles(cur),
                        trace=trace, frontier=list(fr.path), edge=(u, v))
        placed = False
        for j in cands[:6]:
            y = apply_fn(cur, j)
            if y is None or not gate_fn(to_smiles(y)):
                continue
            slot = int(acts[j].slot)
            fr.path.append(slot)
            rank = sorted(range(len(probs)), key=lambda k: -probs[k]).index(j)
            trace.append(f"atom_insert@{slot}(r{rank})")
            if _quota is not None:
                _zp = int(getattr(acts[j], "atom_type", -1))
                _quota[_zp] = _quota.get(_zp, 0) - 1
            cur, placed = y, True
            break
        if not placed:
            return dict(status="UNSAT", stage=f"grow{step}_blocked",
                        smiles=to_smiles(cur), trace=trace, edge=(u, v))

    # closure: last new atom back to the OTHER end of the shared edge
    fams, acts, probs = enum_full_fn(cur)
    pair = (fr.path[-1], v)
    cands = match_closure_descriptors(fams, acts, [pair])
    n_fiber = sum(1 for f in fams if f in ("bond_insert", "cycle_close"))
    if not cands:
        return dict(status="UNSAT", stage="closure", smiles=to_smiles(cur),
                    trace=trace, closure_pair=pair, n_fiber=n_fiber,
                    n_candidates=0, edge=(u, v))
    closed = None
    for j in sorted(cands, key=lambda j: -float(probs[j])):
        y = apply_fn(cur, j)
        if y is not None and gate_fn(to_smiles(y)):
            rank = sorted(range(len(probs)), key=lambda k: -probs[k]).index(j)
            trace.append(f"bond_insert({acts[j].a},{acts[j].b})(r{rank})")
            closed = y
            break
    if closed is None:
        return dict(status="UNSAT", stage="closure_apply", smiles=to_smiles(cur),
                    trace=trace, n_candidates=len(cands), edge=(u, v))
    cur = closed

    def _finish(st_, extra_trace):
        # REFINE_RING on the fused product. Scope is the whole fused BLOCK, not
        # just the new atoms, for the reason already documented for the
        # electronic step below: a restate inside a fused ring necessarily
        # changes bond orders in the ring it is fused to, so a new-atoms-only
        # scope makes the legal refinements unreachable and reproduces the
        # kekule_missing(u,v) failure. Still vastly narrower than the molecule.
        extra_trace = list(extra_trace)
        if int(refine) > 0:
            import numpy as _np
            # EXACT BUILD PROVENANCE, not a traversal. `fr.path` is the new
            # atoms and (u, v) is the shared fusion edge, so this set IS the
            # six-cycle this call just constructed. Any traversal -- even one
            # restricted to ring bonds -- returns the whole fused system when
            # the new ring is built onto a benzene, and REFINE_RING could then
            # retype an atom of the parent ring it did not build.
            #
            # Scoping this tightly does NOT reproduce the kekule_missing
            # failure that forced aromatise_ring to widen: admission here is
            # INTERSECTION (locus & R), not containment, so a correlated block
            # restate spanning both rings still meets the new cycle and stays
            # in the fiber.
            _scope = sorted({int(u), int(v)} | {int(a) for a in fr.path})
            _rng = refine_rng if refine_rng is not None else _np.random.default_rng(0)
            _rs, _rt = refine_ring(enum_full_fn, apply_fn, to_smiles, gate_fn,
                                   st_, _scope, int(refine), _rng)
            if _rs is not None:
                st_ = _rs
                extra_trace = extra_trace + (list(_rt) if isinstance(_rt, list)
                                             else [str(_rt)])
            else:
                extra_trace = extra_trace + [f"refine_unsat({_rt})"]
        sm = to_smiles(st_)
        mm = Chem.MolFromSmiles(sm) if sm else None
        rel = ring_pair_relations(sm)
        gained_fused = rel.get("fused", 0) - rel0.get("fused", 0)
        # ENDPOINT CONTRACT: a fused request must deliver a NEW fused pair, and
        # must not have quietly produced a spiro or bridged system instead.
        ok = (mm is not None and gained_fused >= 1
              and rel.get("spiro", 0) <= rel0.get("spiro", 0)
              and rel.get("bridged", 0) <= rel0.get("bridged", 0))
        return dict(status="OK" if ok else "UNSAT",
                    stage=None if ok else "topology_contract",
                    smiles=sm, trace=trace + list(extra_trace),
                    edge=(u, v), fused_gain=gained_fused,
                    spiro_gain=rel.get("spiro", 0) - rel0.get("spiro", 0),
                    bridged_gain=rel.get("bridged", 0) - rel0.get("bridged", 0),
                    sys_gain=(len(ring_systems(mm)) - sys0) if mm else 0,
                    arom_gain=(int(rdMD.CalcNumAromaticRings(mm)) - arom0) if mm else 0,
                    n_fiber=n_fiber, n_candidates=len(cands),
                    refined=int(refine), frontier=list(fr.path),
                    new_atoms=sorted(int(a) for a in fr.path),
                    fusion_atoms=sorted({int(u), int(v)}),
                    refine_scope=sorted({int(u), int(v)}
                                        | {int(a) for a in fr.path}))

    if state != "aromatic":
        return _finish(cur, [])
    # If the alternating-parity growth ALREADY delivered the aromatic ring, stop
    # here. Running the electronic step anyway accepts any action satisfying
    # "aromatic count > seed count", which is trivially true once the ring is
    # aromatic -- so it walked atom_restate through S, SH2, PH, IH and finally
    # N, turning a correct C19H19N3O benzo-fused product into C18H18N4O and
    # silently violating the requested carbon_rich composition.
    _sm_now = to_smiles(cur)
    _m_now = Chem.MolFromSmiles(_sm_now) if _sm_now else None
    if _m_now is not None and int(rdMD.CalcNumAromaticRings(_m_now)) > arom0:
        return _finish(cur, [])
    # CYCLE ORDER, not set order: the new ring traverses u -> path -> v -> u,
    # so the alternating Kekule bonds are (u,p0), (p1,p2), (p3,v) and the
    # shared edge (v,u) is left alone. Passing path + [u, v] instead put the
    # shared edge in an alternating position, and since it is already aromatic
    # in the existing ring no bond_reorder for it is ever legal -- every fused
    # realisation on both parp1 and braf died at kekule_missing(u,v).
    # Scope the electronic step to the whole fused BLOCK, not the new ring:
    # the restate that aromatises a fused ring necessarily changes bond orders
    # in the ring it is fused to.
    ring_atoms = sorted(ring_block_atoms(cur, [u] + list(fr.path) + [v]))
    ar_state, ar_trace = aromatise_ring(enum_full_fn, apply_fn, to_smiles, gate_fn,
                                        cur, ring_atoms, arom0,
                                        max_one_shot=max_aromatise)
    if ar_state is None:
        r = _finish(cur, [])
        r["status"], r["stage"] = "UNSAT", "aromatise"
        r["reason"] = ar_trace
        return r
    return _finish(ar_state, ar_trace)


def ring_system_atoms(state, atoms):
    """Atoms of the fused ring SYSTEM containing `atoms`, in slot space.

    Differs from ring_block_atoms ONLY in the traversal: it walks RING edges,
    where ring_block_atoms walks any bond joining two on-ring atoms. That
    distinction is invisible on naphthalene and decisive on a biaryl: in
    FC(F)(F)c1cccc(N2CC[NH2+]C3CCCCC32)c1 the piperazine N and a benzene carbon
    are both on rings but are joined by an ACYCLIC single bond, so the
    block walk crosses it and returns all 16 ring atoms -- two independent ring
    systems merged into one.

    REFINE_RING does NOT use this function: it scopes to the exact constructed
    cycle from builder provenance, because even a ring-bond traversal returns
    the whole fused system for a ring built onto a benzene. This is kept as a
    general utility and for the ring-atom/ring-bond distinction it records.

    ring_block_atoms is correct for the ELECTRONIC step and is left alone: a
    fused aromatic system is one correlated pi block, and scoping aromatisation
    to the new ring alone is what caused kekule_missing. REFINE_RING wants the
    opposite guarantee -- that refinement cannot reach a ring the macro did not
    build -- and measured on that biaryl the block scope let refinement retype a
    scaffold benzene carbon to N, which is exactly the all-molecule behaviour
    REFINE_RING exists to remove.
    """
    adj: dict = {}
    for u, v in ring_edges_in_state(state):
        adj.setdefault(int(u), set()).add(int(v))
        adj.setdefault(int(v), set()).add(int(u))
    seed = {int(a) for a in atoms} & set(adj)
    if not seed:
        return {int(a) for a in atoms}
    block, stack = set(seed), list(seed)
    while stack:
        x = stack.pop()
        for y in adj.get(x, ()):
            if y not in block:
                block.add(y); stack.append(y)
    return block | {int(a) for a in atoms}


def ring_block_atoms(state, atoms):
    """All atoms in the cyclic block(s) containing `atoms`, in slot space.

    A fused aromatic system is ONE correlated electronic block: naphthalene is
    a 10-pi system, not two independently aromatic rings. So a
    ring_system_restate that aromatises a newly fused ring legitimately changes
    bond orders in the PRE-EXISTING ring as well.

    Scoping the restate filter to the new ring alone therefore rejected every
    correct action: the production law offered block restates, all of them
    touched old-ring atoms, `match_aromatisation_descriptors` dropped them, and
    all four fused aromatic realisations fell through to the Kekule walk and
    died at kekule_missing. Growth, closure and the topology contract were fine
    -- the electronic scope was wrong.
    """
    import numpy as np
    B = np.asarray(state.bonds)
    n = int(getattr(state, "n_real_atoms", B.shape[0]))
    on_ring = set()
    for u, v in ring_edges_in_state(state):
        on_ring.add(u); on_ring.add(v)
    seed = {int(a) for a in atoms} & on_ring
    if not seed:
        return {int(a) for a in atoms}
    block, stack = set(seed), list(seed)
    while stack:
        x = stack.pop()
        for y in range(n):
            if y in block or B[x, y] <= 0 or y not in on_ring:
                continue
            block.add(y); stack.append(y)
    return block | {int(a) for a in atoms}


# ---------------------------------------------------------------------------
# CHEAP-FIRST REALIZATION FILTER
#
# Constraints define ADMISSIBILITY; R_theta chooses among admissible
# realizations. Those are different jobs and they cost different amounts:
#
#     A_m^feas(x) = { p : p realizes m, T(p) satisfies the endpoint gate }
#     P(p | x, m) proportional to R_theta(p | x),  p in A_m^feas(x)
#
# The first implementation ran the FULL macro for all six anchors and then
# picked the highest-similarity endpoint -- 6 x ~8 = ~48 law evaluations per
# novel state. That inverted the cost: round 0 correctly raised the ring macro
# from 0.05 to 0.29, and round 1 got SLOWER because the controller had learned
# the right thing. Measured: 532 memo entries over 89 states, 6.0 anchors each.
#
# Predicting the product is pure graph surgery on the MolecularGraph arrays --
# no model, no executor. On the delta=0.6 5HT1B cell only anchor a2 is
# admissible, so this turns ~48 law evaluations into ~6 cheap graph builds plus
# ~8 law evaluations.
#
# Selecting on maximum similarity was also the wrong OBJECT: similarity is a
# constraint, not the objective, and R_theta should rank what survives it.
# ---------------------------------------------------------------------------

def predict_pendant_product(state, anchor: int, size: int = 6,
                            aromatic: bool = True, element: int = 2,
                            hetero=None):
    """SMILES of the product of attaching a `size`-ring at `anchor`.

    Pure graph surgery, model-free. Returns None if the request is not
    expressible (no hydrogen at the anchor, or no free slots)."""
    import numpy as np
    from dataclasses import replace
    from compose_v4.chem.molecular_graph import (molecular_graph_to_smiles,
        BOND_SINGLE, BOND_AROMATIC)
    n_tot = int(state.atom_types.shape[0])
    n = int(getattr(state, "n_real_atoms", n_tot))
    if int(state.implicit_h_counts[anchor]) < 1 or n + int(size) > n_tot:
        return None
    at = np.array(state.atom_types, copy=True)
    fc = np.array(state.formal_charges, copy=True)
    hc = np.array(state.implicit_h_counts, copy=True)
    bd = np.array(state.bonds, copy=True)
    ring = list(range(n, n + int(size)))
    order = BOND_AROMATIC if aromatic else BOND_SINGLE
    # STOICHIOMETRIC composition: `hetero` maps ring POSITION -> element code,
    # e.g. {3: 3} builds C5N1 with the nitrogen para to the attachment.
    #
    # Motivated by the same measurement on both benchmarks. On T4, stacking
    # all-carbon aromatic rings drove QED 0.904 -> 0.710 -> 0.392 and the
    # winner audit found 28% of added rings are saturated N heterocycles. On
    # MOLLEO Task 3 the carbocycle macro moves jnk3 +0.011 but qed -0.064 and
    # sa -0.042, a net -0.170 under the unweighted scalarization -- it buys
    # activity with drug-likeness. A ring with one nitrogen is the standard
    # med-chem answer and was not expressible while `composition` was only an
    # allowed-element SET.
    hetero = dict(hetero or {})
    for k, idx in enumerate(ring):
        at[idx] = int(hetero.get(k, element))
        fc[idx] = 0
        # Hydrogens from VALENCE minus ring bonds, not a per-element special
        # case. A ring atom has two ring bonds; an aromatic one effectively
        # spends a third on the pi system. Hard-coding "1 if N else 2" gave
        # oxygen two hydrogens (water) and the ring never parsed.
        _z = int(at[idx])
        _val = {2: 4, 3: 3, 4: 2, 7: 2}.get(_z, 4)      # C, N, O, S
        if aromatic:
            # 6-ring aromatic N is pyridine-type (no H); 5-ring aromatic N is
            # pyrrole-type and MUST carry one or the ring cannot be aromatic.
            hc[idx] = max(0, _val - 3) + (1 if (_z == 3 and int(size) == 5) else 0)
        else:
            hc[idx] = max(0, _val - 2)
        nxt = ring[(k + 1) % int(size)]
        bd[idx, nxt] = order
        bd[nxt, idx] = order
    # the attachment atom carries one fewer H than its ring-mates: it has two
    # ring bonds plus the bond to the anchor. Forcing 0 made saturated rings
    # come out as radical [C].
    hc[ring[0]] = 0 if aromatic else 1
    bd[anchor, ring[0]] = BOND_SINGLE
    bd[ring[0], anchor] = BOND_SINGLE
    hc[anchor] = max(0, int(hc[anchor]) - 1)
    try:
        return molecular_graph_to_smiles(replace(state, atom_types=at,
                                                 formal_charges=fc,
                                                 implicit_h_counts=hc, bonds=bd))
    except Exception:
        return None


def admissible_anchors(state, seed_smiles: str, delta: float, size: int = 6,
                       aromatic: bool = True, qed_min: float = 0.6,
                       sa_max: float = 4.0, n_anchors: int = 12,
                       gate_fn=None):
    """Anchors whose PREDICTED endpoint satisfies the benchmark constraints.

    Returns [(anchor, predicted_smiles, sim, qed, sa)] for admissible anchors
    only, cheapest-first. R_theta is never called here.
    """
    import os, sys
    import numpy as np
    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, rdFingerprintGenerator, RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    sm = Chem.MolFromSmiles(seed_smiles)
    if sm is None:
        return []
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    sfp = gm.GetFingerprint(sm)
    n = int(getattr(state, "n_real_atoms", state.atom_types.shape[0]))
    out = []
    for a in range(n):
        if int(state.implicit_h_counts[a]) < 1:
            continue
        p = predict_pendant_product(state, a, size=size, aromatic=aromatic)
        if not p:
            continue
        m = Chem.MolFromSmiles(p)
        if m is None:
            continue
        if gate_fn is not None and not gate_fn(p):
            continue
        sim = float(DataStructs.TanimotoSimilarity(sfp, gm.GetFingerprint(m)))
        q = float(QED.qed(m))
        sa = float(sascorer.calculateScore(m))
        if sim < float(delta) or q < qed_min or sa > sa_max:
            continue
        out.append((a, p, round(sim, 3), round(q, 3), round(sa, 2)))
        if len(out) >= int(n_anchors):
            break
    return out


def predict_fused_product(state, edge, size: int = 6, aromatic: bool = False,
                          element: int = 2, hetero=None):
    """SMILES of the product of fusing a `size`-ring onto ring edge `edge`.

    Model-free counterpart of predict_pendant_product. The shared edge supplies
    two of the ring's atoms, so only size-2 new atoms are added:
        u - p0 - p1 - ... - p_{size-3} - v,  with (v,u) the existing bond.

    `aromatic` lays the Kekule parity down during construction rather than
    building saturated and repairing afterwards -- no aromatic-order bond can
    be created by any insertion or closure in this process (measured: orders
    1/2/3 only), so aromaticity can only ever emerge from an alternating
    pattern.
    """
    import numpy as np
    from dataclasses import replace
    from compose_v4.chem.molecular_graph import (molecular_graph_to_smiles,
        BOND_SINGLE, BOND_DOUBLE)
    u, v = int(edge[0]), int(edge[1])
    n_tot = int(state.atom_types.shape[0])
    n = int(getattr(state, "n_real_atoms", n_tot))
    k = int(size) - 2
    if k < 1 or n + k > n_tot:
        return None
    if int(state.implicit_h_counts[u]) < 1 or int(state.implicit_h_counts[v]) < 1:
        return None
    at = np.array(state.atom_types, copy=True)
    fc = np.array(state.formal_charges, copy=True)
    hc = np.array(state.implicit_h_counts, copy=True)
    bd = np.array(state.bonds, copy=True)
    path = list(range(n, n + k))
    chain = [u] + path + [v]
    # STOICHIOMETRIC composition, same generic mechanism as the pendant path:
    # `hetero` maps RING POSITION -> element code over the ring
    # [u, p0, ..., p_{k-1}, v], i.e. positions 0..size-1.
    #
    # Positions 0 and size-1 are the SHARED EDGE and are inherited from the
    # scaffold -- fusing cannot transmute an existing atom. A request naming
    # them is refused rather than silently ignored, so a caller never believes
    # it placed a heteroatom that was actually dropped.
    hetero = dict(hetero or {})
    if any(int(pos) in (0, int(size) - 1) for pos in hetero):
        return None
    for off, idx in enumerate(path):
        _z = int(hetero.get(off + 1, element))
        at[idx] = _z; fc[idx] = 0
        # Hydrogens from VALENCE minus ring bonds, not a carbon-shaped
        # constant. The previous `1 if aromatic else 2` silently assumed
        # valence 4 and produced wrong H counts for every non-carbon element,
        # which is why this path could not carry heteroatoms correctly.
        _val = {2: 4, 3: 3, 4: 2, 7: 2}.get(_z, 4)      # C, N, O, S
        if aromatic:
            hc[idx] = max(0, _val - 3) + (1 if (_z == 3 and int(size) == 5) else 0)
        else:
            hc[idx] = max(0, _val - 2)
    for i in range(len(chain) - 1):
        a, b = chain[i], chain[i + 1]
        # alternate starting single at u so the shared (v,u) bond is untouched
        order = BOND_DOUBLE if (aromatic and i % 2 == 1) else BOND_SINGLE
        bd[a, b] = order; bd[b, a] = order
    hc[u] = max(0, int(hc[u]) - 1)
    hc[v] = max(0, int(hc[v]) - 1)
    try:
        return molecular_graph_to_smiles(replace(state, atom_types=at,
                                                 formal_charges=fc,
                                                 implicit_h_counts=hc, bonds=bd))
    except Exception:
        return None


def admissible_fused_edges(state, seed_smiles: str, delta: float, size: int = 6,
                           aromatic: bool = False, qed_min: float = 0.6,
                           sa_max: float = 4.0, gate_fn=None):
    """Fusable edges whose PREDICTED endpoint satisfies the benchmark gate.

    Same contract as admissible_anchors: constraints define admissibility,
    R_theta ranks what survives. Returns
    [(edge, predicted_smiles, sim, qed, sa)].
    """
    import os, sys
    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, rdFingerprintGenerator, RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    sm = Chem.MolFromSmiles(seed_smiles)
    if sm is None:
        return []
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    sfp = gm.GetFingerprint(sm)
    out = []
    for e in fusable_edges(state):
        p = predict_fused_product(state, e, size=size, aromatic=aromatic)
        if not p:
            continue
        m = Chem.MolFromSmiles(p)
        if m is None:
            continue
        if gate_fn is not None and not gate_fn(p):
            continue
        sim = float(DataStructs.TanimotoSimilarity(sfp, gm.GetFingerprint(m)))
        q = float(QED.qed(m)); sa = float(sascorer.calculateScore(m))
        if sim < float(delta) or q < qed_min or sa > sa_max:
            continue
        out.append((e, p, round(sim, 3), round(q, 3), round(sa, 2)))
    return out


def action_locus(family: str, action) -> set:
    """Atom indices an action acts on, read off the action itself.

    Exact and application-free for the restate families, because each of them
    names its locus: SemanticAtomRestate.v, BondReorder.(a,b), and every
    BondOrderChange inside a RingSystemRestate. Diffing products instead would
    be both slower and ambiguous after canonicalisation.
    """
    if family == "atom_restate_semantic":
        v = getattr(action, "v", None)
        return {int(v)} if v is not None else set()
    if family in ("bond_reorder", "bond_insert"):
        a_, b_ = getattr(action, "a", None), getattr(action, "b", None)
        return {int(a_), int(b_)} if a_ is not None and b_ is not None else set()
    if family == "ring_system_restate":
        s = set()
        for c in (getattr(action, "changes", None) or ()):
            s.add(int(getattr(c, "a", -1)))
            s.add(int(getattr(c, "b", -1)))
        return {int(x) for x in s if x >= 0}
    return set()


def refine_ring(enum_full_fn, apply_fn, to_smiles, gate_fn, state, path,
                length: int, rng, macro: str = "restate",
                temperature: float = 2.0, epsilon: float = 0.15,
                cap: int = GLOBAL_CAP, floor: int = FAMILY_FLOOR):
    """REFINE_RING -- `length` restate steps confined to the ring `path`.

    The scope is R_refine(x,R) = {y in A(x) : touched(x->y) meets R}, with
    R_theta renormalised over that local fiber. The LAW is untouched; only its
    domain is. This is deliberately not a `prefer_fn`: prefer_fn is a predicate
    on the product that relaxes to the bare gate after twelve failed draws, so
    an action R_theta ranks at 30 is never actually reached.

    Measured motivation (diagnostics/restate_gate.json): production restate2 is
    an all-molecule operator, so the two edits that turn a fresh benzene into a
    saturated N heterocycle sit at R_theta ranks 30 and 116 and are effectively
    unreachable in two draws. Restricting the DOMAIN, not the law, is what makes
    them ordinary.

    STATE space throughout, for the same reason build_ring_system_exact is: one
    SMILES round trip renumbers atoms, after which `path` names a different ring
    and step two would refine the wrong one. Canonicalise on exit only.

    Returns (state, trace) or (None, reason).
    """
    import numpy as np

    want = set(MACRO_FAMILIES[macro])
    cur, trace = state, []
    for step in range(int(length)):
        fams, acts, probs = enum_full_fn(cur)
        probs = np.asarray(probs, float)
        support = proposal_support(list(fams), probs, cap=cap, floor=floor)
        clean = np.ones(len(probs), dtype=bool)
        n_local = 0
        for j in range(len(probs)):
            if fams[j] not in want:
                continue
            if action_locus(fams[j], acts[j]) & set(path):
                n_local += 1
            else:
                clean[j] = False
        if n_local == 0:
            return None, f"no_ring_local_{macro}@{step}"
        picked = None
        for _attempt in range(12):
            idx, q = macro_action_distribution(list(fams), probs, support,
                                               macro, clean, temperature, epsilon)
            if idx.size == 0:
                break
            j = int(rng.choice(idx, p=q))
            y = apply_fn(cur, j)
            ys = to_smiles(y) if y is not None else None
            if ys and gate_fn(ys):
                rank = sorted(range(len(probs)), key=lambda k: -probs[k]).index(j)
                picked = (j, y, rank)
                break
            clean[j] = False
        if picked is None:
            # No fallback to unrestricted support. A scope that relaxes under
            # pressure is a preference, and a preference is what already fails.
            if trace:
                return cur, trace + [f"halt@{step}"]
            return None, f"no_clean_ring_local_{macro}@{step}"
        j, y, rank = picked
        trace.append(f"{fams[j]}(r{rank})@ring")
        cur = y
    return cur, trace
