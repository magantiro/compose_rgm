"""T4-v2: a feasibility-conditioned proposal law, and the zero-oracle evidence for it.

WHAT THIS IS
------------
The v1 T4 proposal law is ``q(Z | G)``: a program distribution conditioned on the source
graph alone.  Five delta=0.6 cells return ZERO eligible candidates under it while their
same-target siblings work, so the deficit is source-conditioned rather than
target-conditioned.  This module implements ``q(Z | G, h(G))``: the same structural move
vocabulary, but every decision inside the program is conditioned on the CONTINUOUS
CONSTRAINT HEADROOM of the state it is applied to.

    h(G) = ( QED(G) - QED_MIN,
             SA_MAX - SA(G),
             sim(G, G0) - delta,
             HEAVY_MAX - n(G),
             retained source-bit fraction,
             ring / rotatable-bond / stereocentre context )

Every component is free: QED, SA and Tanimoto need no oracle and no docking call.  That
is the whole point -- the constraint geometry can be learned from tens of thousands of
evaluations at zero oracle cost, and this module does exactly that.

WHICH DECISION THIS TARGETS
---------------------------
Grounded on the measured attribution in ``t4_v2_bottleneck_attribution`` (companion
driver), NOT on intuition.  The prior audit established that the gross marginals --
scale, retained fraction, grow/prune/replace mix -- are indistinguishable between failing
and working cells, so a better scale head cannot be the fix.  The attribution localises
the loss to the CONDITIONAL choice those marginals average over: at a given scale, WHICH
peripheral region the program removes.  Each removable region carries its own
(QED gain, similarity cost) pair; the feasible region is a narrow annulus where a large
QED gain is bought for a small similarity spend, and v1 spends its similarity budget on
regions that do not buy QED.

So the load-bearing conditioning in this module is REGION-LOCAL headroom: candidate moves
are ranked by the realised change they make to the margin VECTOR, with the weight on each
component set by how close that component is to binding.  The global h(G) additionally
selects the structural mode and the scale.

SCALARISATION
-------------
The margin vector is kept as a vector everywhere it is informative: the per-component
normalised margins drive the mode context and the move utility weights, and a Pareto
archive over the raw vector supplies part of the parent stream.  One scalar is used, for
ranking only, and it is the AUGMENTED WEIGHTED TCHEBYCHEFF form

    F(G) = min_c min(z_c, CAP) + LAMBDA * sum_c min(z_c, CAP)

with ``z_c = margin_c / scale_c``.  Justification, in order: (1) ``F >= 0`` iff every
constraint holds, so maximising F is literally maximising the distance to the nearest
violated constraint; (2) the min term makes the binding constraint the only one that
moves the score, which is the behaviour the annulus needs -- slack in a satisfied
constraint must not pay for a violation elsewhere; (3) capping the slack at CAP stops a
runaway-slack direction from dominating; (4) the augmentation term breaks weak-Pareto ties
(this is the standard augmented Tchebycheff device and is why the plain min is not used).

NOT WIRED INTO PRODUCTION.  Nothing under ``src/compose_v4``, ``modal_apps`` or
``configs`` is touched by this file; the gate, the fiber and the executor are imported
and used unmodified.  No formal charge is ever changed by a move, so net charge is an
invariant of every program this law emits.
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field

from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import QED, DataStructs

RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.t4_fiber_campaign import (
    QED_MIN,
    REPRESENTABLE_HEAVY_ATOMS,
    SA_MAX,
    Fiber,
)
from compose_v4.gates.med_chem_gate import is_valid as structurally_valid

SCHEMA_VERSION = "t4_v2_feasibility_controller_v1"

#: Slot capacity every T4 proposal lane pads its source to.  Re-verified per source
#: rather than assumed: a TIGHT graph silently deletes the whole ``atom_insert`` family
#: from legal support and would manufacture a false support gap.
T4_PROPOSAL_SLOTS = 48

#: Normalisers taking each margin into comparable units.  These are deliberately round
#: numbers read off the THRESHOLD scales, not fitted to any cell: QED and Tanimoto live
#: on [0, 1] and their thresholds sit at 0.6, SA lives on ~[1, 10] with a ceiling at 4,
#: and the heavy-atom ceiling is 40.  A sensitivity sweep over them is reported by the
#: driver so the result cannot rest on a knife-edge choice.
Z_SCALE = {"qed": 0.25, "sa": 1.0, "sim": 0.2, "size": 8.0}

#: Slack beyond this many normalised units stops counting.  Without it a molecule with
#: enormous SA slack outranks one that is close to feasible on every axis.
SLACK_CAP = 1.5

#: Augmentation weight for the Tchebycheff tie-break.  Small by construction: it must
#: order ties, never overturn the min term.
TCHEBYCHEFF_LAMBDA = 0.05

#: A fragment smaller than this is not a molecule anyone would return.
MIN_FRAGMENT_HEAVY = 5

#: NARROW chemistry-sanity filter over heteroatom-halogen bonds.
#:
#: This is NOT a med-chem rulebook and it is NOT a halogen ban.  It removes exactly the
#: motifs that are chemically pathological as isolated small-molecule substructures --
#: hypohalites and N-halo/S-halo species (``C(=O)OF`` acyl hypofluorite, ``C(=O)OCl``,
#: ``CCN(F)`` N-fluoroamine, aryl ``OF``) and halogenated imidoyl carbons
#: (``C(=N)F`` / ``C(=N)Cl``).  Every one of them is a heteroatom-HALOGEN bond or an
#: imine-carbon halide; NONE of them is a carbon-halogen bond.
#:
#: Ordinary medicinal-chemistry halogen chemistry is deliberately untouched: C-F, aryl-F,
#: CF3, fluoroethyl, aryl-Cl, aryl-Br, alkyl halides and acyl halides all match nothing
#: here, because every pattern requires the halogen to sit on N, O or S, or on an
#: sp2 carbon that is DOUBLE-BONDED TO NITROGEN.
#:
#: The filter narrows what the PROPOSER emits.  It is not part of the benchmark gate:
#: the published thresholds (sim >= delta, QED >= 0.6, SA <= 4.0) and ``Fiber.check`` are
#: untouched, and every row still records the unfiltered benchmark verdict so the two can
#: never be confused.
PATHOLOGICAL_MOTIF_SMARTS = (
    ("N-F", "[N;!$(N=O)]-[F]"),
    ("O-F", "[O]-[F]"),
    ("S-F", "[S]-[F]"),
    ("N-Cl", "[N]-[Cl]"),
    ("O-Cl", "[O]-[Cl]"),
    ("S-Cl", "[S]-[Cl]"),
    ("N-Br", "[N]-[Br]"),
    ("O-Br", "[O]-[Br]"),
    ("C(=N)-halogen", "[CX3](=[NX2])-[F,Cl,Br,I]"),
)

_PATHOLOGICAL_PATTERNS: tuple[tuple[str, object], ...] = tuple(
    (name, Chem.MolFromSmarts(smarts)) for name, smarts in PATHOLOGICAL_MOTIF_SMARTS
)
for _name, _pattern in _PATHOLOGICAL_PATTERNS:
    if _pattern is None:  # pragma: no cover -- a malformed SMARTS must never be silent
        raise RuntimeError(f"pathological motif SMARTS failed to compile: {_name}")


def pathological_motifs(mol) -> tuple[str, ...]:
    """Names of the pathological heteroatom-halogen motifs present in ``mol``."""

    if mol is None:
        return ()
    return tuple(
        name for name, pattern in _PATHOLOGICAL_PATTERNS if mol.HasSubstructMatch(pattern)
    )


def chemistry_sane(mol) -> bool:
    """True when the molecule carries none of the pathological motifs."""

    return not pathological_motifs(mol)

MODES = ("prune", "replace", "remodel", "grow")

#: Scale buckets, in heavy atoms touched by one program.
SCALE_BUCKETS = ((1, 2), (3, 5), (6, 9), (10, 14), (15, 24))

_SUBSTITUTION_ELEMENTS = (6, 7, 8, 9, 16, 17)
_ADDITION_ELEMENTS = (6, 7, 8, 9)


# ---- Headroom ----


@dataclass(frozen=True)
class Headroom:
    """The continuous constraint headroom of one state, plus its topology context."""

    qed: float
    sa: float
    similarity: float
    heavy: float
    rings: int
    aromatic_rings: int
    rotatable: int
    stereocentres: int
    retained_bit_fraction: float

    m_qed: float
    m_sa: float
    m_sim: float
    m_size: float

    z_qed: float
    z_sa: float
    z_sim: float
    z_size: float

    def z(self) -> dict[str, float]:
        return {"qed": self.z_qed, "sa": self.z_sa, "sim": self.z_sim, "size": self.z_size}

    def vector(self) -> tuple[float, float, float, float]:
        return (self.z_qed, self.z_sa, self.z_sim, self.z_size)

    def binding(self) -> str:
        return min(self.z().items(), key=lambda item: item[1])[0]

    def tchebycheff(self) -> float:
        capped = [min(value, SLACK_CAP) for value in self.vector()]
        return min(capped) + TCHEBYCHEFF_LAMBDA * sum(capped)

    def feasible(self) -> bool:
        return self.m_qed >= 0 and self.m_sa >= 0 and self.m_sim >= 0 and self.m_size >= 0

    def context(self, frozen: tuple[str, ...] = ()) -> str:
        """Discrete headroom context the mode/scale policy is keyed on.

        Three buckets per axis -- short (violated), tight (within one normalised unit),
        slack -- plus a coarse size-pressure flag.  A FROZEN axis is collapsed to a
        constant, which is how the driver measures what each component of h contributes.
        """

        parts = []
        for name in ("qed", "sa", "sim", "size"):
            if name in frozen:
                parts.append(f"{name}=*")
                continue
            value = self.z()[name]
            bucket = "short" if value < 0 else "tight" if value < 1.0 else "slack"
            parts.append(f"{name}={bucket}")
        if "excess" not in frozen:
            parts.append(f"excess={_size_excess_bucket(self.heavy)}")
        return "|".join(parts)


#: QED's MW desirability peaks near 300 Da, i.e. roughly 21 heavy atoms.  Excess size
#: over that is the lever a prune has on QED, and it is a headroom quantity, not a
#: molecule-specific fact.
QED_OPTIMAL_HEAVY = 21.0


def _size_excess_bucket(heavy: float) -> str:
    excess = heavy - QED_OPTIMAL_HEAVY
    if excess <= 0:
        return "under"
    if excess <= 6:
        return "near"
    if excess <= 14:
        return "over"
    return "far"


# ---- Evaluation context ----


class FeasibilityContext:
    """Free (oracle-less) feasibility evaluation of endpoints against one fiber.

    The production ``Fiber`` is used unmodified for the authoritative eligible verdict;
    the stage decomposition recomputed here is measurement only and is asserted against
    ``Fiber.check`` on every row.
    """

    def __init__(
        self,
        source_smiles: str,
        delta: float,
        *,
        support: str = "compose_valid",
        chemistry_filter: bool = True,
    ):
        self.source_smiles = source_smiles
        self.delta = delta
        self.chemistry_filter = chemistry_filter
        self.fiber = Fiber(source_smiles, delta, support=support)
        self.source_mol = Chem.MolFromSmiles(source_smiles)
        if self.source_mol is None:
            raise ValueError(f"source does not parse: {source_smiles!r}")
        self.source_heavy = self.source_mol.GetNumHeavyAtoms()
        self.source_bits = set(self.fiber.generator.GetFingerprint(self.source_mol).GetOnBits())
        self.cache: dict[str, dict] = {}
        self.gate_calls = 0
        self.disagreements = 0
        #: Benchmark-eligible endpoints REMOVED by the chemistry filter.  Counted so the
        #: cost of the filter is always visible next to what it keeps.
        self.filtered_pathological = 0
        self.motif_census: dict[str, int] = {}

    # -- preflight --

    def slot_preflight(self) -> dict:
        tight = smiles_to_molecular_graph(self.source_smiles)
        padded = pad_molecular_graph(
            smiles_to_molecular_graph(self.source_smiles), T4_PROPOSAL_SLOTS
        )
        tight_slots = int(tight.atom_types.shape[0])
        padded_slots = int(padded.atom_types.shape[0])
        evidence = {
            "heavy_atoms": self.source_heavy,
            "tight_graph_slots": tight_slots,
            "padded_graph_slots": padded_slots,
            "free_slots_in_padded_source": padded_slots - self.source_heavy,
            "tight_graph_would_forbid_atom_birth": tight_slots <= self.source_heavy,
            "passed": padded_slots == T4_PROPOSAL_SLOTS and padded_slots > self.source_heavy,
        }
        if not evidence["passed"]:
            raise RuntimeError(f"slot-semantics preflight failed: {evidence}")
        return evidence

    # -- evaluation --

    def headroom(self, mol, similarity: float, quality: float, access: float) -> Headroom:
        bits = set(self.fiber.generator.GetFingerprint(mol).GetOnBits())
        retained = len(self.source_bits & bits) / max(len(self.source_bits), 1)
        heavy = mol.GetNumHeavyAtoms()
        ring_info = mol.GetRingInfo()
        aromatic = sum(
            1
            for ring in ring_info.AtomRings()
            if all(mol.GetAtomWithIdx(index).GetIsAromatic() for index in ring)
        )
        rotatable = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
        stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))
        m_qed = quality - QED_MIN
        m_sa = SA_MAX - access
        m_sim = similarity - self.delta
        m_size = float(REPRESENTABLE_HEAVY_ATOMS - heavy)
        return Headroom(
            qed=quality,
            sa=access,
            similarity=similarity,
            heavy=float(heavy),
            rings=ring_info.NumRings(),
            aromatic_rings=aromatic,
            rotatable=rotatable,
            stereocentres=stereo,
            retained_bit_fraction=retained,
            m_qed=m_qed,
            m_sa=m_sa,
            m_sim=m_sim,
            m_size=m_size,
            z_qed=m_qed / Z_SCALE["qed"],
            z_sa=m_sa / Z_SCALE["sa"],
            z_sim=m_sim / Z_SCALE["sim"],
            z_size=m_size / Z_SCALE["size"],
        )

    def evaluate(self, smiles: str) -> dict | None:
        """One free feasibility evaluation.  Returns None for a repeat (already counted)."""

        if smiles in self.cache:
            return None
        row: dict = {"smiles": smiles}
        self.cache[smiles] = row
        self.gate_calls += 1
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        row["parses"] = mol is not None
        if mol is None or "." in smiles:
            row["stage"] = "chemically_valid"
            row["eligible"] = False
            return row
        heavy = mol.GetNumHeavyAtoms()
        row["heavy"] = heavy
        row["heavy_delta"] = heavy - self.source_heavy
        if heavy > REPRESENTABLE_HEAVY_ATOMS:
            row["stage"] = "capacity_valid"
            row["eligible"] = False
            return row
        similarity = DataStructs.TanimotoSimilarity(
            self.fiber.seed, self.fiber.generator.GetFingerprint(mol)
        )
        quality = QED.qed(mol)
        access = sascorer.calculateScore(mol)
        head = self.headroom(mol, similarity, quality, access)
        row.update(
            {
                "similarity": similarity,
                "qed": quality,
                "sa": access,
                "delta_sim": head.m_sim,
                "delta_qed": head.m_qed,
                "delta_sa": head.m_sa,
                "sim_ok": head.m_sim >= 0,
                "qed_ok": head.m_qed >= 0,
                "sa_ok": head.m_sa >= 0,
                "retained_source_bit_fraction": head.retained_bit_fraction,
                "ring_delta": head.rings - self.source_mol.GetRingInfo().NumRings(),
                "headroom": head,
                "reached_gate": True,
            }
        )
        if not (row["sim_ok"] and row["qed_ok"] and row["sa_ok"]):
            row["stage"] = (
                "similarity" if not row["sim_ok"] else "qed" if not row["qed_ok"] else "sa"
            )
            row["eligible"] = False
            self._cross_check(smiles, False)
            return row
        row["all_three"] = True
        ok = bool(structurally_valid(smiles))
        row["med_chem_ok"] = ok
        # The UNFILTERED verdict.  This is what ``Fiber.check`` reconstructs and what the
        # published benchmark gate means; the chemistry filter never moves it.
        row["benchmark_eligible"] = ok
        self._cross_check(smiles, ok)
        motifs = pathological_motifs(mol)
        row["pathological_motifs"] = list(motifs)
        row["chemistry_sane"] = not motifs
        if ok and motifs:
            self.filtered_pathological += 1
            for name in motifs:
                self.motif_census[name] = self.motif_census.get(name, 0) + 1
        if self.chemistry_filter and motifs:
            row["stage"] = "chemistry_sanity" if ok else "med_chem_valid"
            row["eligible"] = False
            return row
        row["stage"] = "eligible" if ok else "med_chem_valid"
        row["eligible"] = ok
        return row

    def _cross_check(self, smiles: str, verdict: bool) -> None:
        if bool(self.fiber.check(smiles)) != verdict:
            self.disagreements += 1


# ---- Structural move vocabulary ----


def _canonical(editable) -> str | None:
    try:
        molecule = editable.GetMol() if isinstance(editable, Chem.RWMol) else editable
        Chem.SanitizeMol(molecule)
    except (ValueError, RuntimeError):
        return None
    smiles = Chem.MolToSmiles(molecule)
    if not smiles or "." in smiles:
        return None
    return smiles


def _net_charge(mol) -> int:
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


def _bucket_range(bucket: int | None) -> tuple[int, int]:
    if bucket is None:
        return (1, 10**6)
    return SCALE_BUCKETS[bucket]


def prune_moves(
    mol, rng: random.Random, *, bucket: int | None = None, limit: int = 64
) -> list[tuple[str, dict]]:
    """Cut one acyclic single bond; keep either side that is still a molecule.

    Both sides are enumerated on purpose.  WHICH side is kept is exactly the decision the
    attribution identifies as losing the probability mass, so the law must be able to see
    both and the headroom must be what chooses.
    """

    low, high = _bucket_range(bucket)
    out: dict[str, dict] = {}
    bonds = [
        bond
        for bond in mol.GetBonds()
        if not bond.IsInRing() and bond.GetBondType() == Chem.BondType.SINGLE
    ]
    rng.shuffle(bonds)
    parent_heavy = mol.GetNumHeavyAtoms()
    for bond in bonds:
        if len(out) >= limit:
            break
        editable = Chem.RWMol(mol)
        editable.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        try:
            pieces = Chem.GetMolFrags(editable.GetMol(), asMols=True, sanitizeFrags=True)
        except (ValueError, RuntimeError):
            continue
        if len(pieces) != 2:
            continue
        for piece in pieces:
            kept = piece.GetNumHeavyAtoms()
            removed = parent_heavy - kept
            if kept < MIN_FRAGMENT_HEAVY or not low <= removed <= high:
                continue
            smiles = _canonical(piece)
            if smiles is not None:
                out.setdefault(smiles, {"mode": "prune", "atoms_touched": removed})
    return list(out.items())


def replace_moves(
    mol, rng: random.Random, *, bucket: int | None = None, limit: int = 64
) -> list[tuple[str, dict]]:
    """Element substitution and bond-order change; formal charge is never moved.

    RING bonds are included on purpose.  Excluding them was measured to cost the whole
    aromaticity axis: on a saturated fused bicycle the only route to an aromatic (and far
    more accessible) ring system runs through ring bond-order changes, and with them
    excluded the SA floor of the vocabulary sat 0.5 units above the ceiling.
    """

    low, high = _bucket_range(bucket)
    operations: list[tuple] = []
    if low <= 1 <= high:
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:
                continue
            for number in _SUBSTITUTION_ELEMENTS:
                if number != atom.GetAtomicNum():
                    operations.append(("element", atom.GetIdx(), number))
    if low <= 2 <= high:
        for bond in mol.GetBonds():
            if bond.GetIsAromatic():
                continue
            for order in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE):
                if bond.GetBondType() != order:
                    operations.append(
                        ("order", (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()), order)
                    )
    rng.shuffle(operations)
    out: dict[str, dict] = {}
    reference = _net_charge(mol)
    for kind, where, value in operations:
        if len(out) >= limit:
            break
        editable = Chem.RWMol(mol)
        if kind == "element":
            target = editable.GetAtomWithIdx(where)
            target.SetAtomicNum(value)
            target.SetNumExplicitHs(0)
            target.SetNoImplicit(False)
            touched = 1
        else:
            editable.GetBondBetweenAtoms(*where).SetBondType(value)
            touched = 2
        smiles = _canonical(editable)
        if smiles is None:
            continue
        candidate = Chem.MolFromSmiles(smiles)
        if candidate is None or _net_charge(candidate) != reference:
            continue
        out.setdefault(smiles, {"mode": "replace", "atoms_touched": touched})
    return list(out.items())


def remodel_moves(
    mol, rng: random.Random, *, bucket: int | None = None, limit: int = 48
) -> list[tuple[str, dict]]:
    """Topology and complexity edits: drop stereochemistry, open a ring, close a ring.

    This is the mode the SA axis needs.  Ring-system shape is an SA driver that neither a
    prune nor an element swap can reach.  NOTE, measured: clearing a chiral TAG does not
    move SA, because ``sascorer`` counts unassigned centres too -- the move is retained
    because it is cheap and occasionally changes perception, not because it relieves SA.
    """

    low, high = _bucket_range(bucket)
    operations: list[tuple] = []
    centres = [
        index
        for index, _ in Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        )
    ]
    if low <= 1 <= high:
        operations.extend(("stereo", (index,), 1) for index in centres)
    if len(centres) > 1 and low <= len(centres) <= high:
        operations.append(("stereo", tuple(centres), len(centres)))
    if low <= 2 <= high:
        operations.extend(
            ("open", (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()), 2)
            for bond in mol.GetBonds()
            if bond.IsInRing() and not bond.GetIsAromatic()
        )
    free = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetTotalNumHs() > 0 and not atom.GetIsAromatic()
    ]
    if len(free) > 14:
        free = rng.sample(free, 14)
    distances = Chem.GetDistanceMatrix(mol)
    for i, first in enumerate(free):
        for second in free[i + 1 :]:
            span = int(distances[first][second])
            if span in (3, 4) and low <= span + 1 <= high:
                operations.append(("close", (first, second), span + 1))
    rng.shuffle(operations)
    out: dict[str, dict] = {}
    for kind, where, touched in operations:
        if len(out) >= limit:
            break
        editable = Chem.RWMol(mol)
        if kind == "stereo":
            for index in where:
                editable.GetAtomWithIdx(index).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        elif kind == "open":
            editable.RemoveBond(*where)
        else:
            editable.AddBond(where[0], where[1], Chem.BondType.SINGLE)
        smiles = _canonical(editable)
        if smiles is not None:
            out.setdefault(smiles, {"mode": "remodel", "atoms_touched": touched})
    return list(out.items())


#: A small, fixed vocabulary of ordinary medicinal-chemistry caps.  It is applied
#: identically to every source; no entry is chosen for, or derived from, any particular
#: molecule, target or cell.  Multi-atom caps are here for a measured reason: SA's
#: fragment term is an AVERAGE over ECFP fragments, so attaching common fragments dilutes
#: a rare core's penalty in a way no sequence of single-atom additions reaches inside one
#: program.
_GROW_FRAGMENTS = (
    ("C", 1),
    ("N", 1),
    ("O", 1),
    ("F", 1),
    ("CC", 2),
    ("CCC", 3),
    ("C(C)C", 3),
    ("OC", 2),
    ("C(N)=O", 3),
    ("C(=O)O", 3),
    ("c1ccccc1", 6),
    ("C1CCCCC1", 6),
    ("C1CCNCC1", 6),
    ("C(F)(F)F", 4),
)


def grow_moves(
    mol, rng: random.Random, *, bucket: int | None = None, limit: int = 64
) -> list[tuple[str, dict]]:
    """Attach one cap from the fixed fragment vocabulary where a hydrogen is free."""

    low, high = _bucket_range(bucket)
    fragments = [item for item in _GROW_FRAGMENTS if low <= item[1] <= high]
    if not fragments:
        return []
    sites = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetTotalNumHs() > 0]
    operations = [(site, fragment, size) for site in sites for fragment, size in fragments]
    rng.shuffle(operations)
    anchor = mol.GetNumAtoms()
    out: dict[str, dict] = {}
    for site, fragment, size in operations:
        if len(out) >= limit:
            break
        piece = Chem.MolFromSmiles(fragment)
        if piece is None:
            continue
        editable = Chem.RWMol(Chem.CombineMols(mol, piece))
        editable.AddBond(site, anchor, Chem.BondType.SINGLE)
        smiles = _canonical(editable)
        if smiles is not None:
            out.setdefault(smiles, {"mode": "grow", "atoms_touched": size})
    return list(out.items())


_MOVE_TABLE = {
    "prune": prune_moves,
    "replace": replace_moves,
    "remodel": remodel_moves,
    "grow": grow_moves,
}


def enumerate_mode(
    mol, mode: str, rng: random.Random, *, bucket: int | None = None, limit: int = 64
) -> list[tuple[str, dict]]:
    """Candidate successors of one structural mode, pre-filtered to a SCALE bucket.

    The bucket is applied before construction, not after sampling, so ``q(s | h)`` is a
    real restriction of the move set rather than a preference over an already-drawn pool.
    """

    return _MOVE_TABLE[mode](mol, rng, bucket=bucket, limit=limit)


# ---- The proposal law ----


@dataclass
class ArmStat:
    """Running utility of one (context, mode, scale-bucket) action."""

    pulls: int = 0
    total: float = 0.0

    def mean(self, prior: float) -> float:
        if self.pulls == 0:
            return prior
        return self.total / self.pulls


@dataclass
class ProposalPolicy:
    """``q(m, s | h)`` learned online from free feasibility evaluations.

    The policy is a contextual bandit whose CONTEXT IS THE HEADROOM SIGNATURE.  Nothing
    about the target, the cell or any specific molecule enters it: the key is a bucketing
    of the normalised margin vector, so a policy learned on one source transfers verbatim
    to any other source with the same headroom signature.

    ``conditioned=False`` collapses every state to one context.  That arm still LEARNS
    which mode works on average, so comparing it against the conditioned arm isolates the
    value of conditioning on h from the value of learning at all.
    """

    conditioned: bool = True
    frozen: tuple[str, ...] = ()
    optimism: float = 0.35
    temperature: float = 0.30
    stats: dict[tuple[str, str, int], ArmStat] = field(default_factory=dict)

    def context_of(self, head: Headroom) -> str:
        return head.context(self.frozen) if self.conditioned else "*"

    def _stat(self, context: str, mode: str, bucket: int) -> ArmStat:
        return self.stats.setdefault((context, mode, bucket), ArmStat())

    def sample(self, head: Headroom, rng: random.Random) -> tuple[str, int, str]:
        context = self.context_of(head)
        actions = [(mode, index) for mode in MODES for index in range(len(SCALE_BUCKETS))]
        scores = [
            self._stat(context, mode, index).mean(self.optimism)
            + self.optimism / math.sqrt(1 + self._stat(context, mode, index).pulls)
            for mode, index in actions
        ]
        top = max(scores)
        weights = [math.exp((score - top) / self.temperature) for score in scores]
        mode, bucket = rng.choices(actions, weights=weights, k=1)[0]
        return mode, bucket, context

    def update(self, context: str, mode: str, bucket: int, utility: float) -> None:
        stat = self._stat(context, mode, bucket)
        stat.pulls += 1
        stat.total += utility


def _bucket_of(atoms_touched: int) -> int:
    for index, (low, high) in enumerate(SCALE_BUCKETS):
        if low <= atoms_touched <= high:
            return index
    return len(SCALE_BUCKETS) - 1


def move_utility(parent: Headroom, child: Headroom, *, frozen: tuple[str, ...] = ()) -> float:
    """Headroom-conditioned utility of one realised move.

    This is the REGION-LOCAL conditioning and it is the part the attribution says is
    load-bearing.  The score is the augmented-Tchebycheff improvement, so a move is
    rewarded for relieving whichever constraint is currently binding and is NOT rewarded
    for buying slack in a constraint that already holds.  Freezing an axis replaces its
    normalised margin with a neutral constant on BOTH sides of the difference, which is
    how the driver measures that axis's contribution.
    """

    def scalar(head: Headroom) -> float:
        capped = []
        for name in ("qed", "sa", "sim", "size"):
            value = 1.0 if name in frozen else head.z()[name]
            capped.append(min(value, SLACK_CAP))
        return min(capped) + TCHEBYCHEFF_LAMBDA * sum(capped)

    gain = scalar(child) - scalar(parent)
    if child.feasible():
        gain += 1.0
    return gain


@dataclass
class Node:
    smiles: str
    mol: object
    head: Headroom
    depth: int


def _pareto_front(nodes: list[Node]) -> list[Node]:
    front = []
    for node in nodes:
        vector = node.head.vector()
        dominated = False
        for other in nodes:
            if other is node:
                continue
            theirs = other.head.vector()
            if all(t >= v for t, v in zip(theirs, vector)) and any(
                t > v for t, v in zip(theirs, vector)
            ):
                dominated = True
                break
        if not dominated:
            front.append(node)
    return front


def run_search(
    source_smiles: str,
    delta: float,
    *,
    budget: int = 6000,
    seed: int = 0,
    conditioned_proposal: bool = True,
    guided_selection: bool = True,
    frozen: tuple[str, ...] = (),
    population_cap: int = 220,
    fanout: int = 18,
    restart_probability: float = 0.08,
    support: str = "compose_valid",
    max_steps: int = 40000,
    chemistry_filter: bool = True,
    prune_pathological_parents: bool = True,
) -> dict:
    """Budgeted feasible-proposal search from one source.  Zero oracle calls.

    ``budget`` counts DISTINCT endpoint evaluations, which is the same denominator the v1
    audit's funnel uses, so the two are directly comparable.
    """

    rng = random.Random(seed)
    context = FeasibilityContext(
        source_smiles, delta, support=support, chemistry_filter=chemistry_filter
    )
    preflight = context.slot_preflight()

    root_row = context.evaluate(Chem.MolToSmiles(context.source_mol))
    root_head = (
        root_row["headroom"]
        if root_row and "headroom" in root_row
        else None
    )
    if root_head is None:
        raise RuntimeError("source itself does not reach the gate")
    root = Node(root_row["smiles"], context.source_mol, root_head, 0)

    policy = ProposalPolicy(conditioned=conditioned_proposal, frozen=frozen)
    population: list[Node] = [root]
    archive: list[Node] = [root]
    eligible: list[dict] = []
    mode_census: dict[str, int] = {mode: 0 for mode in MODES}
    mode_census_eligible: dict[str, int] = {mode: 0 for mode in MODES}
    steps = 0

    while context.gate_calls < budget and steps < max_steps:
        steps += 1
        if rng.random() < restart_probability or not population:
            parent = root
        elif guided_selection:
            if rng.random() < 0.35:
                front = _pareto_front(population[-population_cap:])
                parent = rng.choice(front) if front else root
            else:
                scores = [node.head.tchebycheff() for node in population]
                top = max(scores)
                weights = [math.exp((score - top) / 0.25) for score in scores]
                parent = rng.choices(population, weights=weights, k=1)[0]
        else:
            parent = rng.choice(population)

        mode, bucket, key = policy.sample(parent.head, rng)
        candidates = enumerate_mode(
            parent.mol, mode, rng, bucket=bucket, limit=fanout * 3
        )
        if not candidates:
            # The bucket is empty for this state; fall back to the unrestricted mode so a
            # dead (mode, scale) pair costs one cheap probe rather than a wasted step.
            candidates = enumerate_mode(parent.mol, mode, rng, limit=fanout * 3)
        if not candidates:
            policy.update(key, mode, bucket, -0.05)
            continue
        pool = candidates if len(candidates) <= fanout else rng.sample(candidates, fanout)

        children: list[tuple[Node, float]] = []
        best_utility = -1.0
        for smiles, meta in pool:
            if context.gate_calls >= budget:
                break
            row = context.evaluate(smiles)
            if row is None:
                continue
            row["mode"] = meta["mode"]
            row["atoms_touched"] = meta["atoms_touched"]
            row["parent_depth"] = parent.depth
            mode_census[meta["mode"]] += 1
            if not row.get("reached_gate"):
                continue
            child_head = row["headroom"]
            utility = move_utility(parent.head, child_head, frozen=frozen)
            best_utility = max(best_utility, utility)
            if row["eligible"]:
                eligible.append(row)
                mode_census_eligible[meta["mode"]] += 1
            if (
                chemistry_filter
                and prune_pathological_parents
                and not row.get("chemistry_sane", True)
            ):
                # Do not grow the search out of a molecule nobody would make.  The gate
                # call is still spent, so the budget accounting is unchanged.
                continue
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is not None:
                children.append((Node(smiles, molecule, child_head, parent.depth + 1), utility))

        policy.update(key, mode, bucket, best_utility if children else -0.05)

        if children:
            if guided_selection:
                utilities = [utility for _, utility in children]
                top = max(utilities)
                weights = [math.exp((utility - top) / 0.20) for utility in utilities]
                chosen = rng.choices(children, weights=weights, k=min(2, len(children)))
            else:
                chosen = rng.sample(children, k=min(2, len(children)))
            for node, _ in chosen:
                population.append(node)
                archive.append(node)
        if len(population) > population_cap:
            if guided_selection:
                keep = _pareto_front(population)
                remainder = [node for node in population if node not in keep]
                remainder.sort(key=lambda node: node.head.tchebycheff(), reverse=True)
                population = (keep + remainder)[:population_cap]
            else:
                population = rng.sample(population, population_cap)

    rows = [row for row in context.cache.values()]
    return {
        "source_smiles": source_smiles,
        "delta": delta,
        "slot_preflight": preflight,
        "arm": {
            "conditioned_proposal": conditioned_proposal,
            "guided_selection": guided_selection,
            "frozen": list(frozen),
            "budget": budget,
            "seed": seed,
            "chemistry_filter": chemistry_filter,
            "prune_pathological_parents": prune_pathological_parents,
        },
        "rows": rows,
        "eligible": eligible,
        "steps": steps,
        "gate_calls": context.gate_calls,
        "gate_reconstruction_disagreements": context.disagreements,
        "benchmark_eligible_filtered_out": context.filtered_pathological,
        "pathological_motif_census": dict(sorted(context.motif_census.items())),
        "mode_census": mode_census,
        "mode_census_eligible": mode_census_eligible,
        "policy_contexts": len({key[0] for key in policy.stats}),
    }


# ---- Bottleneck attribution (zero oracle, independent of both v1 and v2) ----


def removable_arms(source_smiles: str, delta: float) -> list[dict]:
    """Every retained region reachable by cutting ONE acyclic single bond.

    This is the decision space the attribution is about.  Each row is one choice of WHICH
    peripheral region to keep, carrying the similarity it costs and the QED it buys.  The
    enumeration is program-free: it depends on no prior, no learned model and no cell.
    """

    context = FeasibilityContext(source_smiles, delta)
    mol = context.source_mol
    rows = []
    seen: set[str] = set()
    rng = random.Random(0)
    for smiles, meta in prune_moves(mol, rng, limit=400):
        if smiles in seen:
            continue
        seen.add(smiles)
        candidate = Chem.MolFromSmiles(smiles)
        if candidate is None:
            continue
        similarity = DataStructs.TanimotoSimilarity(
            context.fiber.seed, context.fiber.generator.GetFingerprint(candidate)
        )
        quality = QED.qed(candidate)
        access = sascorer.calculateScore(candidate)
        bits = set(context.fiber.generator.GetFingerprint(candidate).GetOnBits())
        cost = 1.0 - similarity
        rows.append(
            {
                "smiles": smiles,
                "removed_heavy": meta["atoms_touched"],
                "similarity": round(similarity, 5),
                "qed": round(quality, 5),
                "sa": round(access, 5),
                "qed_gain": round(quality - QED.qed(mol), 5),
                "similarity_cost": round(cost, 5),
                "qed_gain_per_similarity_cost": round((quality - QED.qed(mol)) / cost, 5)
                if cost > 1e-9
                else 0.0,
                "sim_ok": similarity >= delta,
                "qed_ok": quality >= QED_MIN,
                "sa_ok": access <= SA_MAX,
                "retained_bits": bits,
            }
        )
    rows.sort(key=lambda row: -row["qed_gain_per_similarity_cost"])
    return rows


def region_jaccard(first: set, second: set) -> float:
    if not first and not second:
        return 1.0
    return len(first & second) / max(len(first | second), 1)


def attribute_failure(
    source_smiles: str,
    delta: float,
    v1_endpoints: list[dict],
    *,
    repair_depth: int = 2,
    repair_beam: int = 40,
) -> dict:
    """Attribute a zero-eligible cell to ONE losing decision, with the measurement.

    The candidate attributions, in the order they are ruled out:

      ``not_expressible``       no retained region reaches feasibility even with repairs
      ``scale``                 v1 never proposes anything at the witness's scale
      ``retained_region``       v1 proposes that scale but keeps a DIFFERENT region
      ``repair``                v1 reaches the right region but not the finishing edits
      ``attrition_or_stopping`` the region and scale are both proposed and feasible

    ``v1_endpoints`` are the extremal endpoint rows the v1 audit recorded.  They are the
    BEST v1 did on exactly the axes at issue -- the maximum-QED similarity-passing set and
    the maximum-similarity QED-passing set -- so a witness v1 had proposed would appear at
    the top of one of them.  Absence from that set is therefore evidence v1 never proposed
    it, not merely evidence the artifact is a sample.  Stated as INFERRED, and the limit
    of the sample is reported alongside.
    """

    context = FeasibilityContext(source_smiles, delta)
    arms = removable_arms(source_smiles, delta)

    # -- the witness: best region, then cheap repairs on top of it --
    witness = None
    for row in arms:
        if row["sim_ok"] and row["qed_ok"] and row["sa_ok"] and structurally_valid(row["smiles"]):
            witness = dict(row)
            witness.pop("retained_bits", None)
            witness["repairs"] = 0
            break
    if witness is None:
        frontier = [row for row in arms if row["sim_ok"]][:repair_beam]
        rng = random.Random(0)
        for depth in range(1, repair_depth + 1):
            scored = []
            for row in frontier:
                parent = Chem.MolFromSmiles(row["smiles"])
                if parent is None:
                    continue
                for mode in ("replace", "remodel", "prune", "grow"):
                    for smiles, _ in enumerate_mode(parent, mode, rng):
                        evaluated = context.evaluate(smiles)
                        if evaluated is None or not evaluated.get("reached_gate"):
                            continue
                        if evaluated["eligible"] and witness is None:
                            witness = {
                                "smiles": smiles,
                                "similarity": round(evaluated["similarity"], 5),
                                "qed": round(evaluated["qed"], 5),
                                "sa": round(evaluated["sa"], 5),
                                "removed_heavy": -evaluated["heavy_delta"],
                                "repairs": depth,
                            }
                        if evaluated["sim_ok"]:
                            scored.append(
                                {
                                    "smiles": smiles,
                                    "similarity": evaluated["similarity"],
                                    "qed": evaluated["qed"],
                                    "sa": evaluated["sa"],
                                    "sim_ok": True,
                                }
                            )
                if witness is not None:
                    break
            if witness is not None or not scored:
                break
            scored.sort(key=lambda row: -row["qed"])
            frontier = scored[:repair_beam]

    # -- what v1 did, on the same axes --
    v1_rows = []
    for row in v1_endpoints:
        smiles = row.get("smiles")
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is None:
            continue
        bits = set(context.fiber.generator.GetFingerprint(molecule).GetOnBits())
        v1_rows.append(
            {
                "smiles": smiles,
                "heavy_delta": row.get("heavy_delta"),
                "delta_sim": row.get("delta_sim"),
                "delta_qed": row.get("delta_qed"),
                "bits": bits,
            }
        )

    evidence: dict = {
        "source_smiles": source_smiles,
        "single_cut_regions": len(arms),
        "regions_similarity_passing": sum(1 for row in arms if row["sim_ok"]),
        "regions_qed_passing": sum(1 for row in arms if row["qed_ok"]),
        "regions_feasible_before_repair": sum(
            1 for row in arms if row["sim_ok"] and row["qed_ok"] and row["sa_ok"]
        ),
        "v1_endpoint_sample": len(v1_rows),
        "witness": witness,
    }

    if witness is None:
        evidence["attribution"] = "not_expressible_within_probe"
        evidence["note"] = (
            f"no single-cut region plus <={repair_depth} repairs reached feasibility; "
            "this bounds the probe, not the fiber"
        )
        return evidence

    witness_mol = Chem.MolFromSmiles(witness["smiles"])
    witness_bits = set(context.fiber.generator.GetFingerprint(witness_mol).GetOnBits())
    witness_delta = witness_mol.GetNumHeavyAtoms() - context.source_heavy

    at_scale = [
        row
        for row in v1_rows
        if row["heavy_delta"] is not None and abs(row["heavy_delta"] - witness_delta) <= 2
    ]
    overlaps = [region_jaccard(witness_bits, row["bits"]) for row in v1_rows]
    overlaps_at_scale = [region_jaccard(witness_bits, row["bits"]) for row in at_scale]
    v1_reproduced = any(row["smiles"] == witness["smiles"] for row in v1_rows)

    evidence.update(
        {
            "witness_heavy_delta": witness_delta,
            "v1_endpoints_at_witness_scale": len(at_scale),
            "v1_max_region_overlap": round(max(overlaps), 5) if overlaps else None,
            "v1_max_region_overlap_at_witness_scale": (
                round(max(overlaps_at_scale), 5) if overlaps_at_scale else None
            ),
            "v1_reproduced_witness": v1_reproduced,
            "v1_max_similarity_among_qed_passing": (
                round(
                    max(
                        (
                            row["delta_sim"] + delta
                            for row in v1_rows
                            if row["delta_qed"] is not None
                            and row["delta_sim"] is not None
                            and row["delta_qed"] >= 0
                        ),
                        default=float("nan"),
                    ),
                    5,
                )
            ),
            "witness_similarity": witness["similarity"],
        }
    )

    if v1_reproduced:
        evidence["attribution"] = "attrition_or_stopping"
    elif not at_scale:
        evidence["attribution"] = "scale"
    elif (evidence["v1_max_region_overlap_at_witness_scale"] or 0.0) < 0.55:
        evidence["attribution"] = "retained_region"
    elif witness["repairs"] > 0:
        evidence["attribution"] = "repair"
    else:
        evidence["attribution"] = "retained_region"
    return evidence
