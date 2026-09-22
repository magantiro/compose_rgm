"""Scale and content laws for the completion half of a generic edit module.

WHAT THIS REPAIRS
-----------------
``dynamic_program_synthesis`` completes ``segment_grow`` and the grow half of
``segment_replace`` with :func:`_grow_actions`, which builds a LINEAR
SINGLE-BONDED CHAIN of elements drawn uniformly from ``("C", "N", "O")`` off one
anchor, of length drawn uniformly from ``1 .. min(MAX_SEGMENT_LENGTH, 40 - n)``.
Two properties of that draw are measured defects rather than design choices:

``SCALE``    the hard eight-atom ceiling.  The productive-transition census
             (``diagnostics/pmo_global_delta_census_v1``) measures a median
             largest coherent changed region of 14 atoms and a median parent
             retention of 0.40, against a controller median of 3 and 0.968.
             Only 0.4% of controller proposals reach the 13-atom scale that
             58.3% of productive transitions need.
``CONTENT``  a linear single-bonded C/N/O chain installs no ring, no branch and
             no element outside C/N/O.  80.6% of productive transitions install
             ring content; only 22.2% install something a grow chain could
             build.

This module supplies both axes as explicit, separately switchable laws so the
two can be measured apart rather than confounded:

``ScaleLaw``            a generic octave-balanced size draw bounded only by the
                        state's executable heavy-atom capacity.
``ScaleBalancedRegionLaw``  the same size law applied to the EXCISION half of a
                        replacement, so retention can move at all.
``ComponentBank``       objective-blind attachment-compatible components taken
                        from the task-independent initialization chemistry bank
                        through the production ``pendant_cuts``, installed with
                        Active8 primitives only.

INFORMATION BOUNDARY
--------------------
Nothing here reads a task identity, an oracle value, a target structure, a
teacher route or a winner molecule.  The size law is a parameter-free octave
prior.  The component bank is compiled offline from a fixed objective-blind
molecule list and carries no provenance beyond the component graph itself, so
the whole vocabulary could be frozen before the benchmark is named.

EXECUTION
---------
A component is installed by a breadth-first spanning tree rooted at its
attachment atom: one ``atom_insert`` per atom carrying the tree-parent bond,
then one ``cycle_close`` per non-tree edge.  Hydrogen is RESERVED at insertion
(``h_insert = h_final + total_bond_order - parent_bond_order``) so an atom's
valence is constant and correct at every intermediate state, which is what makes
each intermediate executable.  ``MAX_H_COUNT`` is 4, so a valence-six centre
reached through a single bond -- a sulfone or sulfonamide sulfur -- cannot
reserve its five hydrogens and is refused.  That is a declared scope, measured
by :func:`build_component_bank`, not a silent drop.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    IDX_TO_ELEMENT,
    MolecularGraph,
    is_element,
)
from compose_v4.control.bridge_region_law import BridgeRegionLaw
from compose_v4.control.docking_value import identity
from compose_v4.control.donor_program import pendant_cuts
from compose_v4.experiments.whole_ring_plan import execute_program, fresh_slot
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.operators import AtomInsert, CycleCloseEdge

SCHEMA_VERSION = "pmo_completion_component_law_v1"

#: Octave bands. Equal probability per band, uniform within a band, so the draw
#: is scale-balanced rather than size-uniform and carries no tuned parameter.
OCTAVE_BANDS = ((1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64))

#: The heavy-atom ceiling the PMO executor enforces on a proposal state.
REPRESENTABLE_HEAVY_ATOMS = 40


# ---- Scale ---------------------------------------------------------------


def octave_bands(capacity: int) -> tuple[tuple[int, int], ...]:
    """Octave bands clipped to ``capacity``; empty when no size is drawable."""

    if capacity < 1:
        return ()
    return tuple(
        (low, min(high, capacity)) for low, high in OCTAVE_BANDS if low <= capacity
    )


def scale_balanced_size(rng, capacity: int) -> int:
    """One octave-balanced size in ``1 .. capacity``."""

    bands = octave_bands(capacity)
    if not bands:
        raise ValueError("completion scale draw has no remaining capacity")
    low, high = bands[int(rng.integers(len(bands)))]
    return int(rng.integers(low, high + 1))


def bounded_uniform_size(rng, capacity: int, *, maximum: int) -> int:
    """The v1 size draw: uniform over ``1 .. min(maximum, capacity)``."""

    bound = min(maximum, capacity)
    if bound < 1:
        raise ValueError("completion scale draw has no remaining capacity")
    return int(rng.integers(1, bound + 1))


@dataclass(frozen=True)
class ScaleLaw:
    """Which size a completion or excision asks for.

    ``maximum is None`` is the expanded arm: octave-balanced over the whole
    executable capacity.  An integer reproduces the v1 uniform bounded draw.
    """

    maximum: int | None = None

    def draw(self, rng, capacity: int) -> int:
        if self.maximum is None:
            return scale_balanced_size(rng, capacity)
        return bounded_uniform_size(rng, capacity, maximum=self.maximum)


class ScaleBalancedRegionLaw(BridgeRegionLaw):
    """Uniform region selection WITHIN an octave-balanced size band.

    The region-SELECTION axis (which of the same-size regions is tried) stays
    uniform, exactly as v1.  Only the size distribution moves, which is the axis
    the census attributes 46.7% of the lost mass to.
    """

    def __init__(self) -> None:
        super().__init__(maximum=None, margin=None)

    def weights(self, graph: MolecularGraph, regions) -> np.ndarray:
        if not regions:
            return np.ones(0, dtype=float)
        sizes = np.asarray([region.size for region in regions], dtype=int)
        capacity = int(sizes.max())
        bands = octave_bands(capacity)
        out = np.zeros(len(regions), dtype=float)
        for low, high in bands:
            inside = (sizes >= low) & (sizes <= high)
            count = int(inside.sum())
            if count:
                out[inside] = 1.0 / count
        # A size outside every band cannot occur, but never narrow the support.
        out[out <= 0.0] = 1.0 / max(1, len(regions))
        return out


#: Named so a counterfactual can quote it; v1 never constructs a law object.
UNIFORM_BOUNDED_SCALE_V1 = ScaleLaw(maximum=8)
OCTAVE_BALANCED_SCALE_V1 = ScaleLaw(maximum=None)


# ---- Component content ---------------------------------------------------


@dataclass(frozen=True)
class ComponentSpec:
    """One attachment-rooted pendant component, portable between molecules."""

    elements: tuple[str, ...]
    charges: tuple[int, ...]
    hydrogens: tuple[int, ...]
    bonds: tuple[tuple[int, int, int], ...]
    attachment: int
    has_ring: bool
    branch_points: int
    heteroatoms: int

    @property
    def size(self) -> int:
        return len(self.elements)

    @property
    def key(self) -> str:
        return identity(
            {
                "elements": list(self.elements),
                "charges": list(self.charges),
                "hydrogens": list(self.hydrogens),
                "bonds": [list(bond) for bond in self.bonds],
                "attachment": self.attachment,
            }
        )

    def as_json(self) -> dict:
        return {
            "elements": list(self.elements),
            "charges": list(self.charges),
            "hydrogens": list(self.hydrogens),
            "bonds": [list(bond) for bond in self.bonds],
            "attachment": self.attachment,
            "has_ring": self.has_ring,
            "branch_points": self.branch_points,
            "heteroatoms": self.heteroatoms,
        }

    @classmethod
    def from_json(cls, row: dict) -> ComponentSpec:
        return cls(
            elements=tuple(row["elements"]),
            charges=tuple(int(v) for v in row["charges"]),
            hydrogens=tuple(int(v) for v in row["hydrogens"]),
            bonds=tuple((int(a), int(b), int(o)) for a, b, o in row["bonds"]),
            attachment=int(row["attachment"]),
            has_ring=bool(row["has_ring"]),
            branch_points=int(row["branch_points"]),
            heteroatoms=int(row["heteroatoms"]),
        )


def _adjacency(spec: ComponentSpec) -> dict[int, list[tuple[int, int]]]:
    adjacency: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for a, b, order in spec.bonds:
        adjacency[a].append((b, order))
        adjacency[b].append((a, order))
    return adjacency


def extract_components(graph: MolecularGraph) -> tuple[ComponentSpec, ...]:
    """Every attachment-rooted pendant component of an exact state.

    The cut is the production ``pendant_cuts`` oriented single-bond bridge, so a
    component's stored hydrogen counts already account for one external single
    bond at its attachment atom.  Reinstalling it on any hydrogen-bearing anchor
    therefore reproduces the donor's own local chemistry exactly.
    """

    specs = []
    for cut in pendant_cuts(graph):
        component = tuple(int(slot) for slot in cut.component)
        local = {slot: index for index, slot in enumerate(component)}
        bonds = tuple(
            (local[a], local[b], int(graph.bonds[a, b]))
            for a in component
            for b in component
            if a < b and graph.bonds[a, b]
        )
        degree: collections.Counter[int] = collections.Counter()
        for a, b, _order in bonds:
            degree[a] += 1
            degree[b] += 1
        elements = tuple(
            IDX_TO_ELEMENT[int(graph.atom_types[slot])] for slot in component
        )
        specs.append(
            ComponentSpec(
                elements=elements,
                charges=tuple(int(graph.formal_charges[slot]) for slot in component),
                hydrogens=tuple(
                    int(graph.implicit_h_counts[slot]) for slot in component
                ),
                bonds=bonds,
                attachment=local[int(cut.root)],
                has_ring=len(bonds) >= len(component),
                branch_points=sum(1 for node in degree if degree[node] >= 3),
                heteroatoms=sum(1 for element in elements if element != "C"),
            )
        )
    return tuple(specs)


def install_component_actions(state: MolecularGraph, anchor: int, spec: ComponentSpec):
    """Install ``spec`` at ``anchor`` and return ``(actions, product)``.

    Every action is an Active8 primitive and every intermediate state is
    executed, so the caller receives a product the production executor built.
    """

    adjacency = _adjacency(spec)
    root = spec.attachment
    order, parent, seen = [root], {root: None}, {root}
    for at in order:
        for neighbor, _bond in adjacency[at]:
            if neighbor not in seen:
                seen.add(neighbor)
                parent[neighbor] = at
                order.append(neighbor)
    if len(order) != spec.size:
        raise ValueError("component descriptor is not connected")
    tree_edges = {
        frozenset((node, parent[node])) for node in order if parent[node] is not None
    }
    slot_of: dict[int, int] = {}
    actions: list[dict] = []
    current = state
    for node in order:
        above = parent[node]
        internal = sum(order_ for _neighbor, order_ in adjacency[node])
        if above is None:
            neighbors, reserved = ((int(anchor), 1),), internal
        else:
            bond = next(o for neighbor, o in adjacency[node] if neighbor == above)
            neighbors, reserved = ((slot_of[above], bond),), internal - bond
        action = AtomInsert(
            fresh_slot(current),
            ELEMENT_TO_IDX[spec.elements[node]],
            spec.charges[node],
            spec.hydrogens[node] + reserved,
            neighbors,
        )
        record = encode_action("atom_insert", action)
        current, _ = execute_program(current, [record])
        actions.append(record)
        slot_of[node] = action.slot
    for a, b, bond in spec.bonds:
        if frozenset((a, b)) in tree_edges:
            continue
        low, high = sorted((slot_of[a], slot_of[b]))
        record = encode_action("cycle_close", CycleCloseEdge(low, high, bond))
        current, _ = execute_program(current, [record])
        actions.append(record)
    return actions, current


# ---- The bank ------------------------------------------------------------


@dataclass(frozen=True)
class ComponentBank:
    """Installable components indexed by size."""

    components: tuple[ComponentSpec, ...]
    provenance: dict

    def __post_init__(self) -> None:
        by_size: dict[int, list[int]] = collections.defaultdict(list)
        for index, spec in enumerate(self.components):
            by_size[spec.size].append(index)
        object.__setattr__(self, "_by_size", {k: tuple(v) for k, v in by_size.items()})
        object.__setattr__(self, "_sizes", tuple(sorted(by_size)))

    @property
    def sizes(self) -> tuple[int, ...]:
        return self._sizes  # type: ignore[attr-defined]

    def of_size(self, size: int) -> tuple[ComponentSpec, ...]:
        index = self._by_size.get(size, ())  # type: ignore[attr-defined]
        return tuple(self.components[at] for at in index)

    def order_for(self, rng, *, target: int, capacity: int) -> list[ComponentSpec]:
        """Components in draw order: the target size first, then nearest sizes.

        Only sizes the state can actually hold are offered, so the capacity
        bound is enforced by the support rather than by a post-hoc rejection.
        """

        usable = [size for size in self.sizes if size <= capacity]
        if not usable:
            return []
        ranked = sorted(usable, key=lambda size: (abs(size - target), size))
        out: list[ComponentSpec] = []
        for size in ranked:
            group = self.of_size(size)
            out.extend(group[int(at)] for at in rng.permutation(len(group)))
        return out

    @property
    def identity_sha256(self) -> str:
        return identity(
            {
                "schema_version": SCHEMA_VERSION,
                "components": [spec.key for spec in self.components],
            }
        )

    def as_json(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "provenance": self.provenance,
            "components": [spec.as_json() for spec in self.components],
        }

    @classmethod
    def from_json(cls, payload: dict) -> ComponentBank:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("component bank schema version changed")
        return cls(
            components=tuple(
                ComponentSpec.from_json(row) for row in payload["components"]
            ),
            provenance=payload.get("provenance", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> ComponentBank:
        return cls.from_json(json.loads(Path(path).read_text()))


def _reference_host(slots: int) -> MolecularGraph:
    """A minimal neutral host used only to prove a component installs at all."""

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph

    return pad_molecular_graph(smiles_to_molecular_graph("CC"), slots)


def build_component_bank(
    smiles: list[str],
    *,
    slots: int = 48,
    minimum_size: int = 1,
    maximum_size: int = 32,
    provenance: dict | None = None,
) -> tuple[ComponentBank, dict]:
    """Compile and VALIDATE the bank; return it with its rejection census.

    A component is retained only when it installs through the production
    executor on a reference host, so a runtime draw cannot be refused for a
    reason the bank could have found offline.
    """

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph

    host = _reference_host(slots)
    anchor = int(np.flatnonzero(is_element(host.atom_types))[0])
    census: collections.Counter[str] = collections.Counter()
    keep: dict[str, ComponentSpec] = {}
    for row in smiles:
        graph = smiles_to_molecular_graph(row)
        if graph is None:
            census["unparseable_donor"] += 1
            continue
        for spec in extract_components(pad_molecular_graph(graph, slots)):
            census["enumerated"] += 1
            if not minimum_size <= spec.size <= maximum_size:
                census["outside_size_window"] += 1
                continue
            if any(spec.charges):
                census["charged_component"] += 1
                continue
            if spec.key in keep:
                census["duplicate"] += 1
                continue
            try:
                install_component_actions(host, anchor, spec)
            except Exception as error:  # noqa: BLE001 - census of executor refusals
                census[f"refused:{type(error).__name__}"] += 1
                census["refused_total"] += 1
                continue
            keep[spec.key] = spec
            census["retained"] += 1
    components = tuple(keep[key] for key in sorted(keep))
    bank = ComponentBank(
        components=components,
        provenance={
            **(provenance or {}),
            "slots": slots,
            "minimum_size": minimum_size,
            "maximum_size": maximum_size,
            "donor_molecules": len(smiles),
        },
    )
    return bank, dict(census)


# ---- The completion law --------------------------------------------------


@dataclass(frozen=True)
class CompletionLaw:
    """How a generic module completes: at what SCALE, with what CONTENT.

    ``scale``   the size draw for the grow half and, when ``expand_excision`` is
                set, for the excision half of a replacement.
    ``bank``    ``None`` keeps the v1 linear C/N/O chain; a bank installs a
                structured component.
    """

    scale: ScaleLaw = ScaleLaw(maximum=8)
    bank: ComponentBank | None = None
    expand_excision: bool = False
    name: str = "v1_bounded_linear"

    @property
    def region_law(self) -> ScaleBalancedRegionLaw | None:
        """The excision law this completion implies, or ``None`` for v1."""

        return ScaleBalancedRegionLaw() if self.expand_excision else None

    def capacity(self, state: MolecularGraph) -> int:
        return REPRESENTABLE_HEAVY_ATOMS - int(state.n_real_atoms)

    def complete(self, state: MolecularGraph, rng, *, anchor: int | None = None):
        """Return ``(actions, product, parameters)`` for one completion.

        Raises ``ValueError`` exactly as the v1 grow path does, so the caller's
        family-rotation and failure census are unchanged.
        """

        capacity = self.capacity(state)
        if capacity < 1:
            raise ValueError("completion has no remaining heavy-atom capacity")
        target = self.scale.draw(rng, capacity)
        if self.bank is None:
            from compose_v4.control.dynamic_program_synthesis import _grow_actions

            actions, _at, elements = _grow_actions(
                state, rng, length=target, elements=("C", "N", "O"), anchor=anchor
            )
            product, _ = execute_program(state, actions)
            return (
                actions,
                product,
                {
                    "completion": "linear_chain",
                    "requested_size": target,
                    "inserted_atoms": target,
                    "elements": elements,
                    "has_ring": False,
                    "branch_points": 0,
                },
            )
        anchors = [
            int(slot)
            for slot in np.flatnonzero(is_element(state.atom_types))
            if int(state.implicit_h_counts[slot]) >= 1
        ]
        if anchor is not None:
            anchors = [anchor] if anchor in anchors else []
        if not anchors:
            raise ValueError("completion has no hydrogen-bearing anchor")
        at = anchors[int(rng.integers(len(anchors)))]
        failures: collections.Counter[str] = collections.Counter()
        for spec in self.bank.order_for(rng, target=target, capacity=capacity):
            try:
                actions, product = install_component_actions(state, at, spec)
            except Exception as error:  # noqa: BLE001 - try the next component
                failures[type(error).__name__] += 1
                if sum(failures.values()) >= 32:
                    break
                continue
            return (
                actions,
                product,
                {
                    "completion": "component",
                    "requested_size": target,
                    "inserted_atoms": spec.size,
                    "component_key": spec.key,
                    "has_ring": spec.has_ring,
                    "branch_points": spec.branch_points,
                    "heteroatoms": spec.heteroatoms,
                    "anchor": at,
                },
            )
        raise ValueError(f"no bank component installed: {dict(failures)}")
