"""
Heavy-atom molecular graph representation for the discrete-diffusion lipid
generator.

Each molecule is represented by four parallel data structures:
  - atom_types[i] ∈ {0, ..., M-1} — heavy element index (M = 12; null + 10
    elements + SCAR, the inert deletion marker)
  - formal_charges[i] ∈ {-2, -1, 0, +1, +2} — formal charge per atom
  - implicit_h_counts[i] ∈ {0, 1, 2, 3, 4} — number of hydrogens bonded to atom i
  - bonds[i, j] ∈ {0, 1, 2, 3, 4} — bond CLASS between heavy atoms (off-diagonal
    only; diagonal is unused — we do NOT track lone-pair electrons here)
    Class meaning:
        0 = no bond
        1 = single (order 1.0)
        2 = double (order 2.0)
        3 = triple (order 3.0)
        4 = aromatic (order 1.5 — averaged over the aromatic system)

Per-atom valence consistency relation (equality, including H count):
    Σ_j bond_class_to_order(bonds[i, j]) + implicit_h_counts[i] - formal_charges[i]
        ∈ ALLOWED_VALENCES[atom_types[i]]

Aromatic edge case (pyrrole-type heteroatoms): the 1.5 averaging assumes the
ring's π electrons are evenly distributed among ring atoms. This is true for
benzene and pyridine (where the heteroatom's lone pair points OUT of the ring)
but breaks for pyrrole/furan/thiophene (where the heteroatom donates its lone
pair INTO the ring). For those atoms the simple sum would be too high by 0.5
per aromatic bond. The valence check uses a tolerance of 1.0 for atoms with
any aromatic bond, and we trust RDKit's aromaticity perception at sanitize
time to catch genuine violations.

This file replaces the earlier `be_matrix.py` from the continuous-flow attempt.
The Ugi BE-matrix formulation (with diagonal lone-pair electrons and electron-flow
accounting) was inherited from FlowER for compatibility with mechanism prediction.
With the pivot to discrete diffusion (memo 12) and FlowER mechanism work deferred
(memo 11), the lone-pair channel adds no information — it is derivable from the
valence equation — so we drop it. The result is a standard molecular-graph
representation matching DiGress, CoCoGraph, and similar discrete graph generators.

Hydrogens are NOT stored as explicit atoms. RDKit's GetTotalNumHs() at parse time
gives us implicit_h_counts; SetNumExplicitHs at decode time recovers them.

KNOWN LIMITATIONS (carried over from be_matrix.py):
  - Stereochemistry is NOT encoded. Cis/trans (E/Z) and tetrahedral (R/S)
    configurations are lost on round-trip. For lipid generation this affects
    polyunsaturated tail patterns (e.g., MC3's linoleyl chains).
  - Atomic isotopes are not encoded.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rdkit import Chem, RDLogger

# Suppress noisy RDKit warnings; we surface our own errors.
RDLogger.DisableLog("rdApp.*")


# ---- Element vocabulary -----------------------------------------------------
#
# Heavy atoms only — hydrogens are stored separately as implicit_h_counts per
# heavy atom. M = 12 (null + 10 heavy elements covering ionizable-lipid
# chemistry and synthesis intermediates + SCAR).
#
# SCAR (scar-restoration, memo scar_restoration_scope.md): a THIRD site-state,
# distinct from `null` (padding) and from every real element. It marks a site
# whose atom was DELETED but whose bonds are kept — an invertible, recoverable
# deletion marker. A scar is INERT in valence (it absorbs any incident bond
# orders) and is contracted out of the graph only at read-out
# (`contract_scars`). Use `is_element` (element identity / valence / aromaticity)
# vs `is_occupied` (connectivity / components) to classify a slot — NEVER a bare
# `!= NULL_IDX`, which silently reclassifies a scar as a real atom.
ELEMENTS: list[str] = [
    "null",  # 0  — padding / inactive slot
    "B",     # 1  — boron, valence 3 (reductants, intermediates)
    "C",     # 2  — carbon, valence 4
    "N",     # 3  — nitrogen, valence 5
    "O",     # 4  — oxygen, valence 6
    "F",     # 5  — fluorine, valence 7
    "P",     # 6  — phosphorus, valence 5
    "S",     # 7  — sulfur, valence 6
    "Cl",    # 8  — chlorine, valence 7
    "Br",    # 9  — bromine, valence 7
    "I",     # 10 — iodine, valence 7
    "SCAR",  # 11 — deletion marker (inert in valence; contracted at read-out)
]
ELEMENT_TO_IDX: dict[str, int] = {e: i for i, e in enumerate(ELEMENTS)}
IDX_TO_ELEMENT: dict[int, str] = {i: e for i, e in enumerate(ELEMENTS)}
NULL_IDX: int = ELEMENT_TO_IDX["null"]
SCAR_IDX: int = ELEMENT_TO_IDX["SCAR"]
M: int = len(ELEMENTS)


def is_element(atom_types: np.ndarray) -> np.ndarray:
    """Boolean mask: slot holds a REAL chemical element (not NULL padding, not
    a SCAR). Use for valence / element-identity / aromaticity decisions — the
    places that ask "is this a real atom I can chemically reason about?".

    Accepts a scalar or an array; returns the same shape as a bool ndarray.
    """
    at = np.asarray(atom_types)
    return (at != NULL_IDX) & (at != SCAR_IDX)


def is_occupied(atom_types: np.ndarray) -> np.ndarray:
    """Boolean mask: slot is OCCUPIED (a real element OR a scar). Use for
    connectivity / component / slot-occupancy decisions — a degree-2 scar DOES
    connect its two neighbors, so it must count as occupied for those.

    Accepts a scalar or an array; returns the same shape as a bool ndarray.
    """
    at = np.asarray(atom_types)
    return at != NULL_IDX


def is_scar(atom_types: np.ndarray) -> np.ndarray:
    """Boolean mask: slot holds a SCAR (the invertible deletion marker). Use for
    the un-scar diff — "this occupied site must be reconstructed into an element."
    Complements is_element/is_occupied (occupied == element | scar).

    Accepts a scalar or an array; returns the same shape as a bool ndarray.
    """
    at = np.asarray(atom_types)
    return at == SCAR_IDX


# Standard chemistry valence per element — the **default** valence (number of
# heavy + H bonds) for the neutral atom. Per-atom valence relation:
#
#     Σ_j bonds[i, j] + implicit_h_counts[i] - formal_charges[i]
#         ∈ ALLOWED_VALENCES[atom_types[i]]
#
# Note: cations (formal_charge > 0) can have MORE bonds (NH4+ has 4 vs NH3's 3),
# anions can have fewer. The minus sign on formal_charge reflects this:
# a +1 charge adds 1 to the available bond count.
#
# Most elements have a single valence. Hypervalent elements (N, P, S, Cl, I)
# have multiple allowed valences. ALLOWED_VALENCES below captures all of them.
STANDARD_VALENCE: dict[str, int] = {
    "null": 0,
    "B": 3,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "P": 5,   # P(V) is most common in lipids (phospholipid headgroups)
    "S": 2,   # S(II) thioether is most common in lipids
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "SCAR": 0,  # inert marker: valence is never enforced on a scar (short-circuited
                # in per_atom_valence_check); 0 only keeps STANDARD_VALENCE_VEC well-formed
}
STANDARD_VALENCE_VEC: np.ndarray = np.array(
    [STANDARD_VALENCE[e] for e in ELEMENTS], dtype=np.int32
)


# Hypervalent elements have multiple allowed valences. The valence-check
# (and downstream constraint enforcement) accepts membership in the set.
ALLOWED_VALENCES: dict[str, list[int]] = {
    # RDKit-EXACT per-atom valence (matches Chem GetValenceList). These are the
    # NEUTRAL allowed valences; charged hypervalence (nitro/ammonium N⁺ valence 4,
    # etc.) is handled by the charge-adjusted valence logic, NOT by widening the
    # neutral list. The old "N":[3,5] + halogen [1,3,5,7] over-permitted neutral
    # N=5 and hypervalent Cl/Br — our check passed them, RDKit rejected them, and
    # samples shattered on SMILES decode (the "fragmentation" was a decode artifact).
    "null": [0],
    "B": [3],
    "C": [4],
    "N": [3],               # neutral N=3; nitro/ammonium are charged N⁺ (valence 4)
    "O": [2],
    "F": [1],
    "P": [3, 5],            # 3 phosphine/phosphite, 5 phosphate/phosphonate
    "S": [2, 4, 6],         # 2 thioether, 4 sulfoxide, 6 sulfone/sulfonate
    "Cl": [1],
    "Br": [1],
    "I": [1, 3, 5],         # hypervalent iodine I(III)/I(V)
    "SCAR": [0],            # inert placeholder; per_atom_valence_check short-circuits
                           # SCAR rows to ok=True regardless of row-sum (never consulted)
}


# ---- (element, valence) atom classes for the atom-type prediction heads ----
#
# The root/insert/restate/grow heads predict over these CLASSES, not bare elements. Each class pins ONE
# valence, so the implicit-H count follows uniquely from the atom's heavy-bond sum and formal charge (the
# head picks a class; H is derived). Registering EVERY allowed valence-state as its own class is what
# keeps atom_restate SYMMETRIC -- at a fixed bond-sum any valence-state is both reachable and leavable, so
# the edit fiber has NO valence sinks. Where restate alone cannot move (a bond-sum with a single feasible
# valence, e.g. P at bond-sum 4 -> only v5), bond_reorder changes the bond-sum and restate then collapses
# the valence; the two together make the whole operator graph strongly connected.
#
# CNOF classes lead, so a 4-wide CNOF checkpoint warm-starts into the first four rows unchanged.
ORGANIC_SUBSET_SYMBOLS: tuple[str, ...] = ("C", "N", "O", "F", "S", "P", "Cl", "Br", "I", "B")
_CNOF_CLASS_SYMBOLS: tuple[str, ...] = ("C", "N", "O", "F")
ATOM_VALENCE_CLASSES: tuple[tuple[int, int], ...] = tuple(
    (ELEMENT_TO_IDX[sym], valence)
    for sym in (
        *_CNOF_CLASS_SYMBOLS,
        *(s for s in ORGANIC_SUBSET_SYMBOLS if s not in _CNOF_CLASS_SYMBOLS),
    )
    for valence in ALLOWED_VALENCES[sym]
)
ATOM_VALENCE_CLASS_TO_INDEX: dict[tuple[int, int], int] = {
    cls: idx for idx, cls in enumerate(ATOM_VALENCE_CLASSES)
}

# Ring-atom element vocabularies for ring_system_atom_head / ring_electronic, which predict a ring atom's
# ELEMENT (aromatic/saturated ROLE is a separate head dimension, so these are ELEMENTS, not valence-
# classes -- the valence follows from element + role + ring bonds). CNOF (4) reproduces the historical
# ring head; ORGANIC adds S and P (the drug-relevant ring heteroatoms -- thiophene/thiazole/phosphole;
# halogens and B are never ring atoms). CNOF-first so a 4-wide ring head warm-starts into the first rows.
CNOF_RING_ELEMENTS: tuple[int, ...] = tuple(ELEMENT_TO_IDX[s] for s in ("C", "N", "O", "F"))
ORGANIC_RING_ELEMENTS: tuple[int, ...] = tuple(ELEMENT_TO_IDX[s] for s in ("C", "N", "O", "F", "S", "P"))


class AtomVocabulary:
    """An ordered set of (element, valence) atom classes the atom-type heads predict over -- the single
    object the model, the candidate fiber, sampling, and teacher-scoring all share, so a checkpoint's head
    width and the fiber it is scored against can never drift apart. ``CNOF_VOCABULARY`` (4 classes)
    reproduces the historical CNOF model byte-for-byte; ``ORGANIC_VOCABULARY`` (15) covers the drug-like
    organic subset. Each class pins one valence, so H is a pure function of bond-sum + charge, and every
    allowed valence-state being its own class is what keeps atom_restate a symmetric (sink-free) operator."""

    def __init__(self, classes: tuple[tuple[int, int], ...]) -> None:
        self.classes: tuple[tuple[int, int], ...] = tuple((int(e), int(v)) for e, v in classes)
        self._to_index: dict[tuple[int, int], int] = {c: i for i, c in enumerate(self.classes)}
        self.element_index: tuple[int, ...] = tuple(e for e, _ in self.classes)

    def __len__(self) -> int:
        return len(self.classes)

    def element_of(self, class_index: int) -> int:
        return self.classes[class_index][0]

    def h_count(self, class_index: int, bond_order_sum: int, formal_charge: int = 0) -> int | None:
        """Implicit-H implied by a class at a site with this heavy-bond sum and charge:
        ``h = valence + charge - bond_sum``; None if outside ``[0, MAX_H_COUNT]`` (class infeasible here).
        Single-valence classes reproduce the old ``CNOF_VALENCE[type] - bond_sum`` derivation exactly."""
        _, valence = self.classes[class_index]
        h = int(valence) + int(formal_charge) - int(bond_order_sum)
        return h if 0 <= h <= MAX_H_COUNT else None

    def class_index(
        self, element: int, bond_order_sum: int, implicit_h_count: int, formal_charge: int = 0
    ) -> int | None:
        """Inverse: the class of an EXISTING atom. The class valence is the NEUTRAL valence
        ``bond_sum + H - charge`` (charged hypervalence keeps the neutral class plus a nonzero charge).
        None if the atom's (element, valence) is not in this vocabulary."""
        valence = int(bond_order_sum) + int(implicit_h_count) - int(formal_charge)
        return self._to_index.get((int(element), valence))


CNOF_VOCABULARY = AtomVocabulary(
    tuple((ELEMENT_TO_IDX[sym], ALLOWED_VALENCES[sym][0]) for sym in _CNOF_CLASS_SYMBOLS)
)
ORGANIC_VOCABULARY = AtomVocabulary(ATOM_VALENCE_CLASSES)


def valence_class_h_count(class_index: int, bond_order_sum: int, formal_charge: int = 0) -> int | None:
    """Module-level convenience over the ORGANIC vocabulary; see ``AtomVocabulary.h_count``."""
    return ORGANIC_VOCABULARY.h_count(class_index, bond_order_sum, formal_charge)


def atom_valence_class_index(
    atom_type_idx: int, bond_order_sum: int, implicit_h_count: int, formal_charge: int = 0
) -> int | None:
    """Module-level convenience over the ORGANIC vocabulary; see ``AtomVocabulary.class_index``."""
    return ORGANIC_VOCABULARY.class_index(atom_type_idx, bond_order_sum, implicit_h_count, formal_charge)


def canonical_h_count(atom_type_idx: int, bond_order_sum: int) -> int | None:
    """Implicit-H for an atom at its natural (smallest-fitting) valence: the smallest allowed valence
    >= bond_order_sum, minus bond_order_sum. Generalizes ``CNOF_VALENCE[type] - bond_sum`` to the
    hypervalent elements (S/P/I pick the smallest valence that accommodates their heavy bonds; a ring
    thioether S is v2, a sulfone S is v6). None if no allowed valence fits (over-bonded) or H is out of
    range. Single-valence elements reproduce the old CNOF derivation exactly."""
    for valence in ALLOWED_VALENCES[IDX_TO_ELEMENT[int(atom_type_idx)]]:  # ascending
        if valence >= bond_order_sum:
            hydrogens = int(valence) - int(bond_order_sum)
            if 0 <= hydrogens <= MAX_H_COUNT:
                return hydrogens
    return None


# Soft cap on bond count per atom — used as a sanity check during data prep.
MAX_BONDS: dict[str, int] = {
    "null": 0,
    "B": 4,
    "C": 4,
    "N": 4,   # neutral 3, +1 N can have 4
    "O": 3,   # neutral 2, +1 O can have 3
    "F": 1,
    "P": 5,
    "S": 6,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "SCAR": 4,  # inert; a scar absorbs any incident orders. Not consulted on the
                # Stage-1 path (soft data-prep cap only); value kept for dict-safety.
}


# Formal charge vocabulary. K = 5 categories.
FORMAL_CHARGES: list[int] = [-2, -1, 0, 1, 2]
CHARGE_TO_IDX: dict[int, int] = {c: i for i, c in enumerate(FORMAL_CHARGES)}
IDX_TO_CHARGE: dict[int, int] = {i: c for i, c in enumerate(FORMAL_CHARGES)}
NEUTRAL_CHARGE_IDX: int = CHARGE_TO_IDX[0]
K: int = len(FORMAL_CHARGES)


# Implicit hydrogen count vocabulary. Each heavy atom carries h_count in
# {0, 1, 2, 3, 4}: e.g., quaternary nitrogen → 0, methane carbon → 4.
H_COUNT_CLASSES: int = 5
MAX_H_COUNT: int = 4


# Bond class vocabulary (categorical for the discrete-diffusion model).
# Classes 1..3 are integer bond orders (Kekulé form); class 4 is aromatic
# (averaged 1.5 order per bond — see module docstring for pyrrole edge case).
BOND_NULL: int = 0
BOND_SINGLE: int = 1
BOND_DOUBLE: int = 2
BOND_TRIPLE: int = 3
BOND_AROMATIC: int = 4
BOND_CLASSES: int = 5
MAX_BOND_ORDER: int = 3  # the highest integer order we model

# Per-class valence contribution. Aromatic = 1.5 (averaged); others are integer.
BOND_CLASS_TO_ORDER: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 1.5)
BOND_CLASS_TO_ORDER_ARR: np.ndarray = np.array(
    BOND_CLASS_TO_ORDER, dtype=np.float32,
)

# Per-class integer H-count change (per endpoint) when a bond of this class is
# inserted/removed. For Kekulé bonds this equals the order. For aromatic we
# use 1 — an approximation that matches the typical case (benzene/pyridine
# carbons lose 1 H per aromatic bond). Pyrrole-style atoms get off-by-one at
# the per-op level; the SMILES round-trip via RDKit aromaticity perception
# canonicalizes implicit_h at sanitize time.
BOND_CLASS_TO_H_CHANGE: tuple[int, ...] = (0, 1, 2, 3, 1)
BOND_CLASS_TO_H_CHANGE_ARR: np.ndarray = np.array(
    BOND_CLASS_TO_H_CHANGE, dtype=np.int32,
)


# ---- Data class -------------------------------------------------------------


@dataclass
class MolecularGraph:
    """A heavy-atom molecular graph.

    All arrays are size N (the heavy-atom count). The bonds matrix is N × N
    symmetric with off-diagonal entries giving heavy-heavy bond orders. The
    diagonal is conventionally zero — we do NOT track lone-pair electrons.

    Attributes:
      atom_types: int32 (N,) — heavy element indices into ELEMENTS
      formal_charges: int32 (N,) — values in FORMAL_CHARGES
      implicit_h_counts: int32 (N,) — values in [0, MAX_H_COUNT]
      bonds: int32 (N, N) — symmetric, entries in [0, MAX_BOND_ORDER], zero diagonal
    """

    atom_types: np.ndarray
    formal_charges: np.ndarray
    implicit_h_counts: np.ndarray
    bonds: np.ndarray

    def __post_init__(self) -> None:
        n = self.atom_types.shape[0]
        assert self.formal_charges.shape == (n,), "formal_charges shape mismatch"
        assert self.implicit_h_counts.shape == (n,), "implicit_h_counts shape mismatch"
        assert self.bonds.shape == (n, n), "bonds shape mismatch"
        assert np.array_equal(self.bonds, self.bonds.T), "bonds must be symmetric"
        assert np.all(np.diagonal(self.bonds) == 0), \
            "bonds diagonal must be zero (no self-bonds; lone pairs are not tracked)"

    @property
    def n_atoms(self) -> int:
        """Total slot count (real + null)."""
        return int(self.atom_types.shape[0])

    @property
    def n_real_atoms(self) -> int:
        """Number of real heavy atoms — ELEMENTS only (excludes NULL padding AND
        SCAR markers). A scar occupies a slot but is not a real atom, so it is not
        counted here (use `is_occupied`/`n_atoms` for slot-occupancy)."""
        return int(np.sum(is_element(self.atom_types)))


# ---- Per-atom valence -------------------------------------------------------


def expected_row_sum(
    atom_type_idx: int, formal_charge: int, implicit_h_count: int,
) -> int:
    """Expected (default) bond row sum for a heavy atom of given element,
    charge, and implicit H count, using the **default** valence per element.

    Per-atom valence relation:
        row_sum = standard_valence(element) + formal_charge - implicit_h_count

    Caveat: hypervalent elements (N, P, S, halogens) have multiple allowed
    valences. This function returns only the default. Use `allowed_row_sums`
    or `per_atom_valence_check` for full multi-valence semantics.
    """
    return int(
        STANDARD_VALENCE_VEC[atom_type_idx]
        + formal_charge
        - implicit_h_count
    )


def h_count_from_valence(
    atom_type_idx: int,
    formal_charge: int,
    row_sum: float,
    *,
    clamp: bool = False,
) -> int:
    """Implicit-H count implied by the per-atom valence equation — the inverse
    of `expected_row_sum`:

        h = standard_valence(element) + formal_charge - row_sum

    `row_sum` is the summed heavy-bond contribution at the atom (integer for
    Kekulé bonds; aromatic-bearing callers pass a float and the full valence
    expression is rounded to the nearest int). With clamp=False the raw value
    is returned and the caller range-checks it; with clamp=True it is clamped
    to [0, MAX_H_COUNT] (the init / perception convention — e.g. clamps P's
    valence 5 down to 4).
    """
    h = int(round(int(STANDARD_VALENCE_VEC[atom_type_idx]) + int(formal_charge) - row_sum))
    if clamp:
        h = max(0, min(MAX_H_COUNT, h))
    return h


def allowed_row_sums(
    atom_type_idx: int, formal_charge: int, implicit_h_count: int,
) -> list[int]:
    """All allowed bond row sums for an atom, accounting for multi-valence
    elements (N, P, S, halogens).

    Returns the list of `v + formal_charge - implicit_h_count` for each
    `v` in ALLOWED_VALENCES[element], filtered to non-negative integers.
    """
    sym = IDX_TO_ELEMENT[int(atom_type_idx)]
    allowed = ALLOWED_VALENCES[sym]
    candidates = [v + formal_charge - implicit_h_count for v in allowed]
    return [c for c in candidates if c >= 0]


def bond_class_to_order(bonds: np.ndarray) -> np.ndarray:
    """Vectorized lookup: bond class index → valence contribution (float).

    Single/double/triple map to 1/2/3; aromatic maps to 1.5.
    """
    return BOND_CLASS_TO_ORDER_ARR[bonds]


def per_atom_valence_check(mg: MolecularGraph) -> np.ndarray:
    """Boolean array (N,) of per-atom valence satisfaction.

    True for each atom i where the row sum (using bond_class_to_order)
    matches an allowed_row_sum for that atom's element + charge + H count.

    Tolerance behavior:
      - Integer bond orders: exact membership check.
      - Aromatic bonds present: |row_sum − allowed| ≤ 1.0. The 1.5-per-bond
        averaging breaks for pyrrole-type atoms (lone-pair donors); the
        tolerance covers this. Final validity is the SMILES round-trip
        sanitize step in molecular_graph_to_smiles, which uses RDKit's
        aromaticity perception.
    """
    contribs = bond_class_to_order(mg.bonds)            # (N, N) float
    row_sums = contribs.sum(axis=1)                     # (N,) float
    has_aromatic = (mg.bonds == BOND_AROMATIC).any(axis=1)
    n = mg.n_atoms
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        # SCAR is inert in valence: it absorbs any incident bond orders, so a
        # scar row is ALWAYS valence-ok regardless of its row-sum (that is the
        # whole point of the deletion marker — bonds are carried through it
        # untouched). Short-circuit before any allowed_row_sums lookup.
        if int(mg.atom_types[i]) == SCAR_IDX:
            ok[i] = True
            continue
        allowed = allowed_row_sums(
            int(mg.atom_types[i]),
            int(mg.formal_charges[i]),
            int(mg.implicit_h_counts[i]),
        )
        row_sum = float(row_sums[i])
        if has_aromatic[i]:
            # Each aromatic (class-4) bond is scored 1.5 but its true Kekulé order
            # is 1 or 2, so it drifts +/-0.5 from integer valence. An atom with k
            # aromatic bonds can legitimately deviate up to 0.5*k; the old flat 1.0
            # only covered k<=2 and false-flagged bridgehead aromatic N (k=3, e.g.
            # imidazo/pyrazolo/triazolo-fused systems) as invalid. A genuine
            # over-valence still exceeds 0.5*k, so real errors are not masked.
            n_arom = int((mg.bonds[i] == BOND_AROMATIC).sum())
            tol = 0.5 * n_arom
            ok[i] = any(abs(row_sum - float(v)) <= tol + 1e-9 for v in allowed)
        else:
            ok[i] = int(round(row_sum)) in allowed
    return ok


# ---- SMILES <-> MolecularGraph ---------------------------------------------


def contract_scars(mg: MolecularGraph) -> Optional[MolecularGraph]:
    """Contract SCAR sites out of a graph, returning the scar-free "active
    molecule". Deterministic read-out post-process (memo §2 point 2 / §3-A step 3).

    A scar carries connectivity through itself while it lives in the trajectory;
    at read-out we finalize the deletion it marks:

      - degree-0 scar (isolated)  -> drop the site (visible result == padding);
      - degree-1 scar (leaf)      -> drop the site + its one bond; the neighbor
                                     regains that bond's order as implicit H;
      - degree-2 scar (internal)  -> contract  u–SCAR–w  into a single  u–w  bond
                                     of order min(o_u, o_w). Valence-safe; mirrors
                                     apply_atom_delete_stitch case 2a exactly, so a
                                     contracted graph == the ground-truth stitched
                                     deletion.
      - degree>=3 scar            -> ILL-DEFINED (cannot merge >=3 neighbors into
                                     one bond); return None (reject the graph).

    Iterates to a fixpoint, so a run of adjacent scars (u–SCAR–SCAR–w) collapses
    to one u–w bond whose order is the min across the run. Bonds/H arithmetic use
    the bond CLASS as the order — correct for classes 1/2/3; Phase-1 guards forbid
    scarring aromatic (class-4) atoms, so a scar never carries a class-4 bond.

    Returns a NEW graph (input untouched); a scar-free input is returned as a
    copy. Returns None on the degree>=3 defensive-reject path.
    """
    atom_types = mg.atom_types.copy()
    formal_charges = mg.formal_charges.copy()
    implicit_h = mg.implicit_h_counts.copy()
    bonds = mg.bonds.copy()
    n = atom_types.shape[0]

    while True:
        scar_sites = np.where(atom_types == SCAR_IDX)[0]
        if scar_sites.size == 0:
            break
        s = int(scar_sites[0])
        nbrs = [(u, int(bonds[s, u])) for u in range(n) if int(bonds[s, u]) > 0]
        deg = len(nbrs)
        if deg >= 3:
            # Ill-defined: cannot collapse 3+ incident bonds into a single edge.
            return None
        # Drop the scar slot itself (becomes NULL padding).
        atom_types[s] = NULL_IDX
        formal_charges[s] = 0
        implicit_h[s] = 0
        if deg == 0:
            continue  # isolated scar -> nothing to stitch
        if deg == 1:
            (u, o_u) = nbrs[0]
            bonds[s, u] = 0
            bonds[u, s] = 0
            implicit_h[u] = int(implicit_h[u]) + o_u  # neighbor regains the bond as H
            continue
        # deg == 2: contract u–SCAR–w -> u–w with order min(o_u, o_w)
        (u, o_u), (w, o_w) = nbrs
        o_new = min(o_u, o_w)
        bonds[s, u] = 0
        bonds[u, s] = 0
        bonds[s, w] = 0
        bonds[w, s] = 0
        implicit_h[u] = int(implicit_h[u]) + o_u - o_new
        implicit_h[w] = int(implicit_h[w]) + o_w - o_new
        bonds[u, w] = o_new
        bonds[w, u] = o_new

    return MolecularGraph(
        atom_types=atom_types,
        formal_charges=formal_charges,
        implicit_h_counts=implicit_h,
        bonds=bonds,
    )


class MolecularGraphError(ValueError):
    """Raised when a SMILES cannot be converted to a MolecularGraph or vice versa."""


def smiles_to_molecular_graph(smiles: str) -> MolecularGraph:
    """Parse a SMILES into a heavy-atom MolecularGraph.

    Hydrogens are NOT included as explicit atoms; instead each heavy atom's
    bonded H count is stored in `implicit_h_counts`. Aromatic inputs are
    Kekulized into alternating single/double bonds so the micro rewrite basis
    can edit them. Charged species use formal charges.

    Raises MolecularGraphError on parse failure, unsupported elements, or H
    counts above MAX_H_COUNT.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise MolecularGraphError(f"RDKit could not parse SMILES: {smiles!r}")

    # Kekulize so aromatic rings use the same editable order-1/2 vocabulary as
    # every other bond. Aromaticity re-emerges during the RDKit round trip.
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception as exc:
        raise MolecularGraphError(
            f"RDKit could not Kekulize SMILES {smiles!r}: {exc}"
        )

    n = mol.GetNumAtoms()  # heavy atoms only (Chem.MolFromSmiles doesn't AddHs)
    atom_types = np.zeros(n, dtype=np.int32)
    formal_charges = np.zeros(n, dtype=np.int32)
    implicit_h_counts = np.zeros(n, dtype=np.int32)
    bonds = np.zeros((n, n), dtype=np.int32)

    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        # RDKit's MolFromSmiles absorbs explicit-H tokens into implicit H counts
        # at parse time, so we never see "H" atoms here even if the SMILES
        # contained `[H]` tokens. No need to check for it.
        if sym not in ELEMENT_TO_IDX:
            raise MolecularGraphError(
                f"Element {sym!r} is not in heavy-atom vocabulary {ELEMENTS}."
            )
        idx = atom.GetIdx()
        atom_types[idx] = ELEMENT_TO_IDX[sym]

        fc = atom.GetFormalCharge()
        if fc not in CHARGE_TO_IDX:
            raise MolecularGraphError(
                f"Formal charge {fc} on atom {sym} is outside vocabulary "
                f"{FORMAL_CHARGES}."
            )
        formal_charges[idx] = fc

        h_count = atom.GetTotalNumHs()
        if h_count < 0 or h_count > MAX_H_COUNT:
            raise MolecularGraphError(
                f"Implicit H count {h_count} on atom {idx} ({sym}) in "
                f"{smiles!r} is outside [0, {MAX_H_COUNT}]."
            )
        implicit_h_counts[idx] = h_count

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        if bt == Chem.BondType.SINGLE:
            cls = BOND_SINGLE
        elif bt == Chem.BondType.DOUBLE:
            cls = BOND_DOUBLE
        elif bt == Chem.BondType.TRIPLE:
            cls = BOND_TRIPLE
        elif bt == Chem.BondType.AROMATIC:
            cls = BOND_AROMATIC
        else:
            raise MolecularGraphError(
                f"Unsupported bond type {bt} between atoms {i} and {j} "
                f"in {smiles!r}."
            )
        bonds[i, j] = cls
        bonds[j, i] = cls

    mg = MolecularGraph(
        atom_types=atom_types, formal_charges=formal_charges,
        implicit_h_counts=implicit_h_counts, bonds=bonds,
    )

    # Sanity check: per-atom valence equation should hold for every atom.
    if not per_atom_valence_check(mg).all():
        bad = np.where(~per_atom_valence_check(mg))[0]
        raise MolecularGraphError(
            f"Per-atom valence check failed for atoms {bad.tolist()} in "
            f"{smiles!r}; check element vocabulary and formal-charge handling."
        )
    return mg


@dataclass
class MolecularSerializationCache:
    """Bounded, explicitly scoped cache of exact-state serialization outcomes.

    Keys snapshot every array, including padding and dtype. Mutating a graph
    therefore cannot retrieve a stale value; canonical aliases are never used
    as input identities. Only strings/None are retained, never mutable graphs.
    """

    max_entries: int
    entries: OrderedDict[tuple[tuple[str, tuple[int, ...], bytes], ...], str | None] = field(
        default_factory=OrderedDict, repr=False
    )
    hits: int = 0
    misses: int = 0


_serialization_cache: ContextVar[MolecularSerializationCache | None] = ContextVar(
    "compose_molecular_serialization_cache", default=None
)


@contextmanager
def molecular_serialization_cache(max_entries: int = 512) -> Iterator[MolecularSerializationCache]:
    """Memoize only inside this context; release entries even after exceptions.

    Context-local storage avoids a process-wide cache shared by unrelated jobs.
    Nested scopes get independent bounded storage and restore their caller.
    """
    if max_entries <= 0:
        raise ValueError("molecular serialization cache needs max_entries > 0")
    cache = MolecularSerializationCache(max_entries)
    token = _serialization_cache.set(cache)
    try:
        yield cache
    finally:
        _serialization_cache.reset(token)
        cache.entries.clear()


def molecular_graph_to_smiles(mg: MolecularGraph) -> str | None:
    """Canonical RDKit serialization, optionally memoized by exact array content."""
    cache = _serialization_cache.get()
    if cache is None:
        return _molecular_graph_to_smiles_uncached(mg)
    arrays = (mg.atom_types, mg.formal_charges, mg.implicit_h_counts, mg.bonds)
    # Object arrays are not the molecular representation; do not cache their
    # pointer bytes or alter the legacy error behavior on malformed inputs.
    if any(a.dtype.hasobject for a in arrays):
        return _molecular_graph_to_smiles_uncached(mg)
    key = tuple((a.dtype.str, a.shape, a.tobytes()) for a in arrays)
    if key in cache.entries:
        cache.hits += 1
        cache.entries.move_to_end(key)
        return cache.entries[key]
    cache.misses += 1
    result = _molecular_graph_to_smiles_uncached(mg)
    cache.entries[key] = result
    if len(cache.entries) > cache.max_entries:
        cache.entries.popitem(last=False)
    return result


def _molecular_graph_to_smiles_uncached(mg: MolecularGraph) -> str | None:
    """Construct an RDKit Mol from a MolecularGraph and return canonical SMILES.

    Null atoms are skipped. Returns None if RDKit's sanitization fails (which
    happens when the molecular graph violates its own implied chemistry — e.g.,
    a kekulization conflict). Does not raise.

    Implicit hydrogens are set explicitly via SetNumExplicitHs so the round-trip
    matches the input molecule exactly.
    """
    # Scar pre-pass (scar-restoration): contract any SCAR sites out of the graph
    # BEFORE the RDKit build so decode only ever sees real elements + NULL. A
    # degree>=3 scar is ill-defined -> contract_scars returns None -> undecodable.
    contracted = contract_scars(mg)
    if contracted is None:
        return None
    mg = contracted

    rw = Chem.RWMol()

    # First pass: figure out which atoms participate in an aromatic bond so
    # we can set their aromatic flag (RDKit requires both atom and bond to
    # be flagged aromatic for sanitize to succeed).
    n = mg.n_atoms
    is_aromatic_atom = (mg.bonds == BOND_AROMATIC).any(axis=1)

    # Map MolecularGraph atom index → RDKit atom index, skipping null atoms.
    rd_idx: dict[int, int] = {}
    for i in range(n):
        elem_idx = int(mg.atom_types[i])
        if elem_idx == NULL_IDX:
            continue
        sym = IDX_TO_ELEMENT[elem_idx]
        atom = Chem.Atom(sym)
        atom.SetFormalCharge(int(mg.formal_charges[i]))
        atom.SetNumExplicitHs(int(mg.implicit_h_counts[i]))
        atom.SetNoImplicit(True)
        if is_aromatic_atom[i]:
            atom.SetIsAromatic(True)
        rd_idx[i] = rw.AddAtom(atom)

    # Visit existing upper-triangle bonds in the same row-major order as the
    # dense scan. Molecular graphs are sparse, while padded slot matrices are
    # not: avoid a Python iteration for every absent edge on every validity call.
    left, right = np.nonzero(np.triu(mg.bonds, k=1))
    for i, j in zip(left.tolist(), right.tolist()):
        if i not in rd_idx or j not in rd_idx:
            continue
        cls = int(mg.bonds[i, j])
        if cls == BOND_NULL:
            continue
        if cls == BOND_SINGLE:
            bt = Chem.BondType.SINGLE
        elif cls == BOND_DOUBLE:
            bt = Chem.BondType.DOUBLE
        elif cls == BOND_TRIPLE:
            bt = Chem.BondType.TRIPLE
        elif cls == BOND_AROMATIC:
            bt = Chem.BondType.AROMATIC
        else:
            return None
        rw.AddBond(rd_idx[i], rd_idx[j], bt)

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
        return None

    return Chem.MolToSmiles(mol)


def is_rdkit_valid(mg: MolecularGraph) -> bool:
    """True iff `mg` round-trips through RDKit: serialize to SMILES, then re-parse.

    The authoritative validity gate — stronger than `per_atom_valence_check`, which
    is a necessary-but-not-sufficient screen for atoms bearing aromatic bonds. Never
    raises; returns False on any serialization/parse failure.
    """
    try:
        smi = molecular_graph_to_smiles(mg)
        return smi is not None and Chem.MolFromSmiles(smi) is not None
    except Exception:
        return False
