"""The region-draw law for bridge-separated substituent excision.

WHAT A REGION DRAW IS
---------------------
A molecular graph's non-ring bonds are its bridges.  Cutting one bridge splits
the molecule into exactly two connected sides, and deleting either side is a
chemically coherent excision: nothing inside the surviving side changes, and the
single retained anchor recovers the hydrogens the severed bond was using.  The
set of such (fragment, anchor) pairs is the *region support* of a state, and a
region-draw law is a probability distribution over that support.

WHY THIS MODULE EXISTS
----------------------
``dynamic_program_synthesis._delete_pendant_fragment`` draws from that support
with ONE law, fixed at the call site: uniform over fragments of at most
``MAX_SEGMENT_LENGTH`` (8) atoms.  Both halves of that law are load-bearing and
both were measured to be wrong for lead optimization on large sources.

* The SIZE CAP.  A coherent 9-15 atom substituent cannot be drawn at all.  It
  has to be expressed as two or three *independent* bounded cuts, so its joint
  proposal mass is a product of per-module terms rather than a single term.
* The UNIFORM WEIGHT.  Every legal cut is equally likely, including the many
  that destroy the similarity the task requires, so the mass that survives the
  free gate is diluted by everything that does not.

Measured on the five exhausted T4 delta=0.6 cells, every clean eligible witness
is a net EXCISION of 7 to 15 heavy atoms -- not one is a growth -- so this is
precisely the move class the cap cannot express in one draw.

NEITHER HALF IS SUFFICIENT ALONE, and that is why this module changes both in
one law.  Raising the cap alone dilutes a working cell's mass across a much
larger uniform support; conditioning alone still forces the large excision
through a product of module terms.

THE LAW
-------
:class:`BridgeRegionLaw` is a pair of decisions:

``maximum``   the largest fragment the law will draw, or ``None`` for "any size"
``margin``    an optional map from a realized child state to a scalar FEASIBILITY
              MARGIN; ``None`` means uniform

Weights are a Boltzmann tilt on the margin with a support floor::

    w(region) = max(floor, exp(margin(child) / temperature))

The floor is what makes this a RE-RANKING rather than a filter: every region
that was drawable keeps strictly positive probability, so the law can only move
mass around the support, never delete part of it.  Combined with ``maximum=None``
widening the support, the new law's support is a strict superset of v1's.

WHAT THE LAW IS ALLOWED TO KNOW
-------------------------------
Structure (the bridge decomposition of the current state) and the task's own
declared free thresholds (similarity delta, QED floor, SA ceiling, heavy-atom
capacity), evaluated on the child the draw would produce.  It reads no cell
identifier, no target name, no seed index, and no oracle value.  Asked "why does
this fire here", the answer is structural: this source has coherent
bridge-separated substituents larger than the cap, and the children that survive
excision differ sharply in free-gate margin.

The law has NO CHARGE RULE.  A cut that removes a charge-bearing fragment and so
changes the child's net formal charge is scored exactly like any other cut, by
the margin of the child it produces.  This is deliberate and load-bearing: one
of the five exhausted cells has a cationic source whose only known witnesses are
neutral children reached by excising the charged fragment.  A charge-preservation
guard here -- of the kind the editing-corpus corruption path legitimately needs --
would silently delete that cell's entire witness set.

WHICH CALLERS CHANGE BEHAVIOUR
------------------------------
None, unless a law is passed explicitly.

* ``_delete_pendant_fragment(source, rng)``, and therefore the
  ``substituent_delete`` and ``segment_replace`` generic modules, keep drawing
  under :data:`UNIFORM_BOUNDED_V1`, which is exactly the previous behaviour:
  same enumeration, same ``rng.permutation`` consumption, same acceptance order.
  Byte-identical endpoints for every existing run.
* ``retained_core_pruning`` and ``dynamic_program_synthesis_v1`` call
  ``_pendant_fragments`` directly with their own ``maximum`` and do not route
  through this module at all; they are untouched.
* Callers that opt in by passing ``region_law=`` to ``compile_generic_module``,
  ``synthesize_dynamic_program`` or ``synthesize_named_module_sequence`` get the
  new law.  That is the only way behaviour changes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from compose_v4.chem.molecular_graph import (
    BOND_CLASS_TO_H_CHANGE,
    MAX_H_COUNT,
    NULL_IDX,
    MolecularGraph,
    is_element,
    molecular_graph_to_smiles,
)
from compose_v4.chem.state import is_valid_state

# ---- Free-gate constants -------------------------------------------------
# These mirror the T4 task declaration in `experiments.t4_fiber_campaign`
# (QED_MIN, SA_MAX, REPRESENTABLE_HEAVY_ATOMS).  They are restated rather than
# imported so this control-layer module does not depend on an experiment module;
# `tests/test_bridge_region_law.py` asserts the two stay equal.
QED_FLOOR = 0.6
SA_CEILING = 4.0
HEAVY_CAPACITY = 40
#: Synthetic accessibility is reported on 1..10; used to put SA slack in the
#: same [0, 1] units as the similarity and QED slacks.
SA_RANGE = 9.0

#: Minimum relative weight any drawable region keeps.  This is what makes the
#: conditioned law a re-ranking instead of a filter.
SUPPORT_FLOOR = 0.05
#: Tilt scale, in normalized gate units.  0.1 means "a tenth of a gate's range
#: is one e-fold of preference".  Sensitivity is reported by the gate harness.
MARGIN_TEMPERATURE = 0.1


def _sascorer():
    """The exact synthetic-accessibility implementation the T4 fiber gate uses.

    `t4_fiber_campaign` scores SA with RDKit's contributed `sascorer`; the margin
    must be computed with the same one or it is not this task's gate.
    """

    import os
    import sys

    from rdkit import RDConfig

    contrib = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if contrib not in sys.path:
        sys.path.append(contrib)
    import sascorer

    return sascorer


class RegionRealizationError(ValueError):
    """The excision this region names does not produce a valid exact state."""


@dataclass(frozen=True)
class BridgeRegion:
    """One bridge-separated substituent of a state, with its retained anchor."""

    fragment: tuple[int, ...]
    anchor: int
    bond_order: int

    @property
    def size(self) -> int:
        return len(self.fragment)


# ---- Support enumeration -------------------------------------------------


def _adjacency(graph: MolecularGraph) -> dict[int, set[int]]:
    real = [int(i) for i in np.flatnonzero(is_element(graph.atom_types))]
    return {
        slot: {int(j) for j in np.flatnonzero(graph.bonds[slot]) if int(j) in set(real)}
        for slot in real
    }


def bridge_separated_regions(
    graph: MolecularGraph, *, maximum: int | None = None
) -> tuple[BridgeRegion, ...]:
    """Every region the law may draw, in a deterministic order.

    A region is one side of a bridge cut: removing the single bond
    ``(anchor, f)`` disconnects the real-atom graph, ``fragment`` is the side not
    containing ``anchor``, and the surviving side is non-empty.  ``maximum=None``
    imposes no size bound, which is the axis v1's ``MAX_SEGMENT_LENGTH`` closes.

    Ordered by ``(size, fragment)`` so two runs enumerate identically; the draw,
    not the enumeration, is where randomness enters.
    """

    adjacency = _adjacency(graph)
    real = set(adjacency)
    found: dict[tuple[tuple[int, ...], int], BridgeRegion] = {}
    for a in sorted(real):
        for b in sorted(adjacency[a]):
            if a >= b:
                continue
            left = _component_without_edge(adjacency, a, frozenset((a, b)))
            if b in left:
                continue  # the bond lies on a cycle: cutting it separates nothing
            right = real - left
            order = int(graph.bonds[a, b])
            for fragment, anchor in ((left, b), (right, a)):
                size = len(fragment)
                if size < 1 or size >= len(real):
                    continue
                if maximum is not None and size > maximum:
                    continue
                key = (tuple(sorted(fragment)), int(anchor))
                found[key] = BridgeRegion(key[0], key[1], order)
    return tuple(found[key] for key in sorted(found, key=lambda row: (len(row[0]), row)))


def _component_without_edge(
    adjacency: dict[int, set[int]], start: int, cut: frozenset[int]
) -> set[int]:
    seen, stack = {start}, [start]
    while stack:
        at = stack.pop()
        for neighbor in adjacency[at]:
            if frozenset((at, neighbor)) == cut or neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return seen


# ---- Exact realization ---------------------------------------------------


def excise_region(graph: MolecularGraph, region: BridgeRegion) -> MolecularGraph:
    """The exact state the region's deletion schedule produces.

    Closed form of the executor's own composition.  ``apply_atom_delete`` returns
    each severed neighbour's hydrogens as it goes, and every bond severed inside
    the fragment returns hydrogens to atoms that are themselves deleted, so the
    only surviving effect is that ``region.anchor`` recovers the hydrogens of the
    single bridge bond.  Because the cut edge is a bridge it is the unique bond
    crossing the boundary, which is what makes the closed form exact.

    ``tests/test_bridge_region_law.py`` drives the LIVE
    ``_delete_pendant_fragment`` over real molecules and requires the endpoints
    to agree; this is not trusted by inspection.
    """

    fragment = {int(slot) for slot in region.fragment}
    anchor = int(region.anchor)
    if anchor in fragment:
        raise RegionRealizationError("region anchor lies inside its own fragment")
    atom_types = graph.atom_types.copy()
    formal_charges = graph.formal_charges.copy()
    implicit_h = graph.implicit_h_counts.copy()
    bonds = graph.bonds.copy()

    crossing = [
        int(j)
        for slot in fragment
        for j in np.flatnonzero(bonds[slot])
        if int(j) not in fragment
    ]
    if sorted(set(crossing)) != [anchor] or len(crossing) != 1:
        raise RegionRealizationError("region is not separated by a single bridge bond")

    implicit_h[anchor] += int(BOND_CLASS_TO_H_CHANGE[int(region.bond_order)])
    if int(implicit_h[anchor]) > MAX_H_COUNT:
        raise RegionRealizationError("excision overfills the retained anchor")
    for slot in fragment:
        bonds[slot, :] = 0
        bonds[:, slot] = 0
        atom_types[slot] = NULL_IDX
        formal_charges[slot] = 0
        implicit_h[slot] = 0
    child = MolecularGraph(atom_types, formal_charges, implicit_h, bonds)
    if not is_valid_state(child):
        raise RegionRealizationError("excision does not produce a valid exact state")
    return child


# ---- The free feasibility gate ------------------------------------------


@dataclass(frozen=True)
class FreeFeasibilityGate:
    """The task constraints that cost nothing to evaluate at proposal time.

    Exactly the scalar gates ``t4_fiber_campaign.Fiber.check`` applies before it
    would ever consult the oracle.  ``margin`` is the signed slack of the
    TIGHTEST of them, normalized so the four are comparable: positive means the
    child clears every free gate, and the magnitude says by how much.
    """

    delta: float
    qed_min: float = QED_FLOOR
    sa_max: float = SA_CEILING
    max_heavy: int = HEAVY_CAPACITY

    def slacks(self, *, similarity: float, qed: float, sa: float, heavy: int) -> dict:
        return {
            "similarity": float(similarity) - float(self.delta),
            "qed": float(qed) - float(self.qed_min),
            "sa": (float(self.sa_max) - float(sa)) / SA_RANGE,
            "heavy": (int(self.max_heavy) - int(heavy)) / float(self.max_heavy),
        }

    def margin(self, *, similarity: float, qed: float, sa: float, heavy: int) -> float:
        return min(self.slacks(similarity=similarity, qed=qed, sa=sa, heavy=heavy).values())

    def passes(self, *, similarity: float, qed: float, sa: float, heavy: int) -> bool:
        return self.margin(similarity=similarity, qed=qed, sa=sa, heavy=heavy) >= 0.0


# ---- The law -------------------------------------------------------------


@dataclass(frozen=True)
class BridgeRegionLaw:
    """A probability law over a state's bridge-separated regions.

    ``maximum``      largest drawable fragment, ``None`` for any size
    ``margin``       child state -> feasibility margin, or ``None`` for uniform
    ``floor``        minimum relative weight, preserving support
    ``temperature``  Boltzmann scale in normalized gate units
    """

    maximum: int | None = None
    margin: Callable[[MolecularGraph], float] | None = None
    floor: float = SUPPORT_FLOOR
    temperature: float = MARGIN_TEMPERATURE

    def __post_init__(self) -> None:
        if self.maximum is not None and self.maximum < 1:
            raise ValueError("region size cap must be at least one atom")
        if not 0.0 < self.floor <= 1.0:
            raise ValueError("support floor must lie in (0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("margin temperature must be positive")

    @property
    def conditioned(self) -> bool:
        return self.margin is not None

    def regions(self, graph: MolecularGraph) -> tuple[BridgeRegion, ...]:
        return bridge_separated_regions(graph, maximum=self.maximum)

    def weights(
        self, graph: MolecularGraph, regions: Sequence[BridgeRegion]
    ) -> np.ndarray:
        """Relative draw weight of each region; strictly positive everywhere."""

        if self.margin is None:
            return np.ones(len(regions), dtype=float)
        out = np.empty(len(regions), dtype=float)
        for position, region in enumerate(regions):
            try:
                child = excise_region(graph, region)
            except RegionRealizationError:
                # The executor would refuse this region too. Keep it at the floor
                # rather than deleting it, so the law never narrows the support.
                out[position] = self.floor
                continue
            out[position] = max(
                self.floor, math.exp(float(self.margin(child)) / self.temperature)
            )
        return out

    def order(self, graph: MolecularGraph, rng) -> list[BridgeRegion]:
        """Regions in weighted draw order, sampled without replacement.

        Efraimidis-Spirakis: with ``u_i`` uniform, sorting by ``-log(u_i)/w_i``
        ascending is exactly successive weighted selection without replacement,
        so the head of the list is one draw from the law and the tail is the law
        conditioned on that draw having been rejected by the executor.  Uniform
        weights reduce to a plain permutation.
        """

        regions = self.regions(graph)
        if not regions:
            return []
        weights = self.weights(graph, regions)
        keys = -np.log(np.clip(rng.random(len(regions)), 1e-300, 1.0)) / weights
        return [regions[int(at)] for at in np.argsort(keys, kind="stable")]


#: The law `_delete_pendant_fragment` has always used. Retained so existing
#: callers stay byte-identical and so counterfactuals can name it.
UNIFORM_BOUNDED_V1 = BridgeRegionLaw(maximum=8, margin=None)


def free_gate_margin_law(
    gate: FreeFeasibilityGate,
    reference_smiles: str,
    *,
    maximum: int | None = None,
    floor: float = SUPPORT_FLOOR,
    temperature: float = MARGIN_TEMPERATURE,
) -> BridgeRegionLaw:
    """The joint repair: one uncapped draw, tilted by the child's free margin.

    ``reference_smiles`` is the similarity reference the task declares -- for
    T4, the ORIGINAL benchmark source, never the current parent, so the law
    cannot drift its own target.  Both halves are set here because each alone was
    measured insufficient: uncapped-uniform dilutes healthy cells, and
    capped-conditioned still splits a large excision across modules.
    """

    from rdkit import Chem
    from rdkit.Chem import QED, DataStructs, rdFingerprintGenerator

    sascorer = _sascorer()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    reference = generator.GetFingerprint(Chem.MolFromSmiles(reference_smiles))
    cache: dict[str, float] = {}

    def margin(child: MolecularGraph) -> float:
        smiles = molecular_graph_to_smiles(child)
        if smiles is None:
            return -1.0
        if smiles not in cache:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or "." in smiles:
                cache[smiles] = -1.0
            else:
                cache[smiles] = gate.margin(
                    similarity=DataStructs.TanimotoSimilarity(
                        reference, generator.GetFingerprint(mol)
                    ),
                    qed=QED.qed(mol),
                    sa=sascorer.calculateScore(mol),
                    heavy=mol.GetNumHeavyAtoms(),
                )
        return cache[smiles]

    return BridgeRegionLaw(
        maximum=maximum, margin=margin, floor=floor, temperature=temperature
    )
