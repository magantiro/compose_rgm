"""Carbonyl proposal descriptors over existing primitives, not new operators."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import BOND_SINGLE, NULL_IDX, MolecularGraph, is_element
from compose_v4.control.macro_engine import (
    ELEMENT_CODE,
    match_closure_descriptors,
    match_growth_descriptors,
)

ADD_CARBONYL_OPTION = "add_carbonyl"
INSERT_RING_CARBONYL_OPTION = "insert_ring_carbonyl"
CARBONYL_OPTIONS = (ADD_CARBONYL_OPTION, INSERT_RING_CARBONYL_OPTION)
CORE_PHASES = ("open", "scaffold_extend", "grow", "cyclize")


def carbonyl_indices(graph, families, actions):
    """All neutral =O births at existing carbon, before any ranking cap."""
    result = []
    for i, (family, action) in enumerate(zip(families, actions, strict=True)):
        if family != "atom_insert":
            continue
        neighbors = tuple(action.neighbors)
        if (
            action.atom_type == ELEMENT_CODE["O"]
            and action.formal_charge == 0
            and action.implicit_h_count == 0
            and len(neighbors) == 1
            and neighbors[0][1] == 2
            and 0 <= neighbors[0][0] < graph.n_atoms
            and graph.atom_types[neighbors[0][0]] == ELEMENT_CODE["C"]
        ):
            result.append(i)
    return result


def eligible_core_edges(graph: MolecularGraph, locus) -> tuple[tuple[int, int], ...]:
    """Nonaromatic cycle bonds; the executor still decides chemical legality."""
    if graph.n_real_atoms + 2 > 40 or np.count_nonzero(graph.atom_types == NULL_IDX) < 2:
        return ()
    real = tuple(int(i) for i in np.flatnonzero(is_element(graph.atom_types)))
    network = nx.Graph()
    network.add_nodes_from(real)
    network.add_edges_from((a, b) for a in real for b in real if a < b and graph.bonds[a, b])
    network.remove_edges_from(list(nx.bridges(network)))
    classes = resonance_invariant_bond_classes(graph)
    edges = []
    for a, b in network.edges:
        if a in locus and b in locus and classes[a, b] == BOND_SINGLE:
            edges.extend(((a, b), (b, a)))
    return tuple(sorted(edges))


@dataclass(frozen=True)
class CarbonylProgress:
    edge: tuple[int, int] | None = None
    carbon: int | None = None
    oxygen: int | None = None

    def __post_init__(self):
        if self.edge is not None and (not isinstance(self.edge, tuple) or len(self.edge) != 2):
            raise ValueError("core edge must be an immutable pair")
        slots = (() if self.edge is None else self.edge) + tuple(
            i for i in (self.carbon, self.oxygen) if i is not None
        )
        if any(type(i) is not int or i < 0 for i in slots) or len(slots) != len(set(slots)):
            raise ValueError("carbonyl progress requires distinct nonnegative slots")
        if (self.edge is None and self.carbon is not None) or (
            self.carbon is None and self.oxygen is not None
        ):
            raise ValueError("carbonyl progress is missing a preceding phase")

    def payload(self):
        return {
            "schema_version": "core_carbonyl_progress_v1",
            "edge": None if self.edge is None else list(self.edge),
            "carbon": self.carbon,
            "oxygen": self.oxygen,
        }

    @classmethod
    def from_payload(cls, payload):
        if set(payload) != {"schema_version", "edge", "carbon", "oxygen"} or (
            payload["schema_version"] != "core_carbonyl_progress_v1"
        ):
            raise ValueError("unknown carbonyl progress schema")
        edge = payload["edge"]
        if edge is not None and not isinstance(edge, list):
            raise ValueError("serialized core edge must be an array")
        return cls(None if edge is None else tuple(edge), payload["carbon"], payload["oxygen"])

    def advance(self, edge, slot, step):
        return CarbonylProgress(
            edge, slot if step == 1 else self.carbon, slot if step == 2 else self.oxygen
        )

    def validate(self, graph, origin, locus, step):
        if (
            step not in range(5)
            or (self.edge is None) != (step == 0)
            or ((self.carbon is None) != (step < 2) or (self.oxygen is None) != (step < 3))
        ):
            raise ValueError("core carbonyl phase/progress mismatch")
        old = tuple(int(i) for i in np.flatnonzero(is_element(origin.atom_types)))
        real = {int(i) for i in np.flatnonzero(is_element(graph.atom_types))}
        added = {i for i in (self.carbon, self.oxygen) if i is not None}
        if graph.n_atoms != origin.n_atoms or not set(old) <= real or real - set(old) != added:
            raise ValueError("core program must preserve original atoms and its exact new slots")
        if not added <= set(locus):
            raise ValueError("core program creates atoms outside the mutable region")
        for field in ("atom_types", "formal_charges"):
            if not np.array_equal(
                getattr(graph, field)[list(old)], getattr(origin, field)[list(old)]
            ):
                raise ValueError("core program changed original identities or charges")
        expected = origin.bonds[np.ix_(old, old)].copy()
        if self.edge is not None:
            if self.edge not in eligible_core_edges(origin, locus):
                raise ValueError("core program requires an eligible original cycle edge")
            a, b = (old.index(i) for i in self.edge)
            expected[a, b] = expected[b, a] = 0
        if not np.array_equal(graph.bonds[np.ix_(old, old)], expected):
            raise ValueError("core program changed an unrelated original bond")
        expected_edges = {}
        if self.carbon is not None:
            expected_edges[tuple(sorted((self.edge[0], self.carbon)))] = 1
            if (
                graph.atom_types[self.carbon] != ELEMENT_CODE["C"]
                or graph.formal_charges[self.carbon]
            ):
                raise ValueError("core insertion requires neutral carbon")
        if self.oxygen is not None:
            expected_edges[tuple(sorted((self.carbon, self.oxygen)))] = 2
            if (
                graph.atom_types[self.oxygen] != ELEMENT_CODE["O"]
                or graph.formal_charges[self.oxygen]
            ):
                raise ValueError("core insertion requires neutral oxygen")
        if step == 4:
            expected_edges[tuple(sorted((self.carbon, self.edge[1])))] = 1
        actual = {
            tuple(sorted((a, b))): int(graph.bonds[a, b])
            for a in added
            for b in real
            if graph.bonds[a, b]
        }
        if actual != expected_edges:
            raise ValueError("core carbonyl has a missing or unintended attachment")


def core_descriptor_indices(families, actions, probabilities, progress, edge, step):
    if step == 0:
        return [
            i
            for i, (family, action) in enumerate(zip(families, actions, strict=True))
            if family == "cycle_open" and {action.a, action.b} == set(edge)
        ]
    if step == 1:
        return [
            i
            for i in match_growth_descriptors(
                families,
                actions,
                probabilities,
                None,
                {ELEMENT_CODE["C"]},
                anchors=(edge[0],),
                bond_order=1,
            )
            if actions[i].formal_charge == 0 and actions[i].implicit_h_count == 3
        ]
    if step == 2:
        return [
            i
            for i, (family, action) in enumerate(zip(families, actions, strict=True))
            if family == "atom_insert"
            and action.atom_type == ELEMENT_CODE["O"]
            and action.formal_charge == 0
            and action.implicit_h_count == 0
            and tuple(action.neighbors) == ((progress.carbon, 2),)
        ]
    if step == 3:
        return [
            i
            for i in match_closure_descriptors(families, actions, [(progress.carbon, edge[1])])
            if actions[i].order == 1
        ]
    return []


def completed_core_carbonyl(origin, graph, progress):
    try:
        progress.validate(graph, origin, set(np.flatnonzero(is_element(graph.atom_types))), 4)
    except ValueError:
        return False
    return graph.n_real_atoms == origin.n_real_atoms + 2
