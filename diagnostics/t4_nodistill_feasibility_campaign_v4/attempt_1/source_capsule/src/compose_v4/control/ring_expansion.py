"""Three existing primitives expand a nonaromatic ring edge by one atom.

This opt-in descriptor channel neither invents legal marks nor constructs an
endpoint. The existing option kernel supplies the law and executes every step.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import BOND_SINGLE, NULL_IDX, MolecularGraph, is_element
from compose_v4.control.macro_engine import (
    COMPOSITION_CODES,
    match_closure_descriptors,
    match_growth_descriptors,
)

EXPAND_RING_OPTION = "expand_ring"
EXPANSION_PHASES = ("open", "scaffold_extend", "cyclize")


def eligible_expansion_edges(graph: MolecularGraph, locus) -> tuple[tuple[int, int], ...]:
    """Oriented single cycle edges, excluding aromatic/junction-junction bonds."""
    if graph.n_real_atoms >= 40 or not np.any(graph.atom_types == NULL_IDX):
        return ()
    real = tuple(int(i) for i in np.flatnonzero(is_element(graph.atom_types)))
    network = nx.Graph()
    network.add_nodes_from(real)
    network.add_edges_from((u, v) for u in real for v in real if u < v and graph.bonds[u, v])
    network.remove_edges_from(list(nx.bridges(network)))
    perceived = resonance_invariant_bond_classes(graph)
    mutable = set(locus)
    edges = []
    for u, v in network.edges:
        if (
            u in mutable
            and v in mutable
            and perceived[u, v] == BOND_SINGLE
            and min(network.degree[u], network.degree[v]) == 2
        ):
            edges.extend(((u, v), (v, u)))
    return tuple(sorted(edges))


@dataclass(frozen=True)
class ExpansionProgress:
    edge: tuple[int, int] | None = None
    new_slot: int | None = None

    def __post_init__(self) -> None:
        if self.edge is not None and (not isinstance(self.edge, tuple) or len(self.edge) != 2):
            raise ValueError("expansion edge must be an immutable pair")
        slots = (() if self.edge is None else self.edge) + (
            () if self.new_slot is None else (self.new_slot,)
        )
        if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in slots):
            raise ValueError("expansion slots must be nonnegative integers")
        if len(set(slots)) != len(slots) or (self.edge is None and self.new_slot is not None):
            raise ValueError("expansion slots must be distinct and a new atom requires an edge")

    def payload(self) -> dict:
        return {
            "schema_version": "expansion_progress_v1",
            "edge": None if self.edge is None else list(self.edge),
            "new_slot": self.new_slot,
        }

    @classmethod
    def from_payload(cls, value: dict) -> ExpansionProgress:
        if set(value) != {"schema_version", "edge", "new_slot"} or (
            value["schema_version"] != "expansion_progress_v1"
        ):
            raise ValueError("unknown or malformed expansion progress schema")
        if value["edge"] is not None and not isinstance(value["edge"], list):
            raise ValueError("serialized expansion edge must be an array")
        return cls(None if value["edge"] is None else tuple(value["edge"]), value["new_slot"])

    def validate(self, graph: MolecularGraph, origin: MolecularGraph, locus, step: int) -> None:
        if (
            step not in range(4)
            or (self.edge is None) != (step == 0)
            or ((self.new_slot is None) != (step < 2))
        ):
            raise ValueError("expansion phase/progress mismatch")
        old = tuple(int(i) for i in np.flatnonzero(is_element(origin.atom_types)))
        real = {int(i) for i in np.flatnonzero(is_element(graph.atom_types))}
        new = set() if self.new_slot is None else {self.new_slot}
        if not set(old) <= real or real - set(old) != new or not new <= set(locus):
            raise ValueError("expansion must track exactly one new atom inside the region")
        if graph.n_atoms != origin.n_atoms:
            raise ValueError("expansion requires unchanged persistent-slot dimensions")
        for field in ("atom_types", "formal_charges"):
            if not np.array_equal(
                getattr(graph, field)[list(old)], getattr(origin, field)[list(old)]
            ):
                raise ValueError("expansion must retain original atom identities and charges")
        expected = origin.bonds[np.ix_(old, old)].copy()
        if self.edge is not None:
            if self.edge not in eligible_expansion_edges(origin, locus):
                raise ValueError("expansion requires an eligible original edge inside the region")
            u, v = (old.index(i) for i in self.edge)
            expected[u, v] = expected[v, u] = 0
        if not np.array_equal(graph.bonds[np.ix_(old, old)], expected):
            raise ValueError("expansion changed an unrelated original bond")
        if self.new_slot is not None:
            n = self.new_slot
            if (
                int(graph.atom_types[n]) not in COMPOSITION_CODES["mixed"]
                or graph.formal_charges[n]
            ):
                raise ValueError("expansion grows neutral C/N/O only")
            neighbors = {int(i): int(graph.bonds[n, i]) for i in real if graph.bonds[n, i]}
            expected_neighbors = {self.edge[0]: BOND_SINGLE}
            if step == 3:
                expected_neighbors[self.edge[1]] = BOND_SINGLE
            if neighbors != expected_neighbors:
                raise ValueError("expansion requires precisely the requested path attachments")


def expansion_descriptor_indices(families, actions, probabilities, progress, edge, step):
    """Match only the exact requested edit, with no materialization or ranking cap."""
    if step == 0:
        return [
            i
            for i, (family, action) in enumerate(zip(families, actions))
            if family == "cycle_open"
            and {getattr(action, "a", None), getattr(action, "b", None)} == set(edge)
        ]
    if step == 1:
        return [
            i
            for i in match_growth_descriptors(
                families,
                actions,
                probabilities,
                None,
                COMPOSITION_CODES["mixed"],
                anchors=(edge[0],),
                bond_order=BOND_SINGLE,
            )
            if actions[i].formal_charge == 0 and actions[i].implicit_h_count >= 1
        ]
    if step == 2:
        return [
            i
            for i in match_closure_descriptors(families, actions, [(progress.new_slot, edge[1])])
            if getattr(actions[i], "order", None) == BOND_SINGLE
        ]
    return []


def completed_expansion(
    origin: MolecularGraph, graph: MolecularGraph, progress: ExpansionProgress
) -> bool:
    try:
        progress.validate(graph, origin, np.flatnonzero(is_element(graph.atom_types)), 3)
    except ValueError:
        return False
    # The exact induced-bond/path check establishes edge subdivision, not just
    # matching a ring count or endpoint SMILES.
    return graph.n_real_atoms == origin.n_real_atoms + 1
