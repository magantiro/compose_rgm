"""Descriptors and exact-slot state for the opt-in aromatic C6 fused program.

No executor, model, or endpoint constructor lives here. The program restricts
existing macro marks; the option kernel supplies their probabilities/products.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import BOND_AROMATIC, MolecularGraph, is_element
from compose_v4.control.macro_engine import (
    COMPOSITION_CODES,
    match_closure_descriptors,
    match_growth_descriptors,
)

BUILD_FUSED_RING_OPTION = "build_fused_ring"
FUSED_HORIZON = 5
GROWTH_BOND_ORDERS = (1, 2, 1, 2)


def eligible_fusion_edges(graph: MolecularGraph, locus) -> tuple[tuple[int, int], ...]:
    """Oriented carbon ring edges in the mutable region, including sparse slots.

    An edge belongs to a cycle iff its endpoints remain connected after its
    removal. SCAR/NULL are not elements and never participate in this test.
    This is initial applicability, not a guarantee of a completing suffix.
    """
    real = {int(i) for i in np.flatnonzero(is_element(graph.atom_types))}
    if graph.n_real_atoms + len(GROWTH_BOND_ORDERS) > 40:
        return ()
    # Only NULL slots can receive a birth; SCAR is occupied, not a free atom slot.
    from compose_v4.chem.molecular_graph import NULL_IDX

    if np.count_nonzero(graph.atom_types == NULL_IDX) < len(GROWTH_BOND_ORDERS):
        return ()
    carbon = COMPOSITION_CODES["carbon_rich"]
    anchors = sorted(
        i
        for i in real & set(locus)
        if int(graph.atom_types[i]) in carbon and graph.implicit_h_counts[i] >= 1
    )
    adjacency = {i: {j for j in real if graph.bonds[i, j] != 0} for i in real}
    edges = []
    for u in anchors:
        for v in anchors:
            if u >= v or v not in adjacency[u]:
                continue
            visited, pending = {u}, [u]
            while pending:
                at = pending.pop()
                for nb in adjacency[at]:
                    if {at, nb} == {u, v} or nb in visited:
                        continue
                    visited.add(nb)
                    pending.append(nb)
            if v in visited:
                edges.extend(((u, v), (v, u)))
    return tuple(sorted(edges))


@dataclass(frozen=True)
class FusedProgress:
    edge: tuple[int, int] | None = None
    path: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.path, tuple) or (
            self.edge is not None and (not isinstance(self.edge, tuple) or len(self.edge) != 2)
        ):
            raise ValueError("fused progress requires immutable edge/path tuples")
        slots = (() if self.edge is None else self.edge) + self.path
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in slots):
            raise ValueError("fused edge/path slots must be nonnegative integers")
        if len(slots) != len(set(slots)) or len(self.path) > len(GROWTH_BOND_ORDERS):
            raise ValueError("fused edge/path slots must be distinct and path length at most four")
        if self.edge is None and self.path:
            raise ValueError("fused path requires an oriented edge")

    def payload(self) -> dict:
        return {
            "schema_version": "fused_progress_v1",
            "edge": list(self.edge) if self.edge is not None else None,
            "path": list(self.path),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> FusedProgress:
        if set(payload) != {"schema_version", "edge", "path"} or (
            payload["schema_version"] != "fused_progress_v1"
        ):
            raise ValueError("unknown or malformed fused progress schema")
        if not isinstance(payload["path"], list) or (
            payload["edge"] is not None and not isinstance(payload["edge"], list)
        ):
            raise ValueError("serialized fused edge/path must be arrays")
        edge = None if payload["edge"] is None else tuple(payload["edge"])
        return cls(edge, tuple(payload["path"]))

    def validate(self, graph: MolecularGraph, origin: MolecularGraph, locus, step: int) -> None:
        if not 0 <= step <= FUSED_HORIZON or len(self.path) != min(step, 4):
            raise ValueError("fused phase/path length mismatch")
        if (self.edge is None) != (step == 0):
            raise ValueError("fused edge must be selected at the first physical edit")
        old = {int(i) for i in np.flatnonzero(is_element(origin.atom_types))}
        real = {int(i) for i in np.flatnonzero(is_element(graph.atom_types))}
        if not old <= real or real - old != set(self.path):
            raise ValueError("fused path must identify exactly the newly created atoms")
        if self.edge is not None and self.edge not in eligible_fusion_edges(origin, locus):
            raise ValueError("fused edge is not an eligible original ring edge inside the region")
        if not set(self.path) <= set(locus):
            raise ValueError("fused new atoms must remain in the mutable region")
        old_slots = sorted(old)
        if not np.array_equal(origin.atom_types[old_slots], graph.atom_types[old_slots]):
            raise ValueError("fused option must retain original atom identities")
        if not np.array_equal(origin.formal_charges[old_slots], graph.formal_charges[old_slots]):
            raise ValueError("fused option must retain original charges")
        if not np.array_equal(
            origin.bonds[np.ix_(old_slots, old_slots)] != 0,
            graph.bonds[np.ix_(old_slots, old_slots)] != 0,
        ):
            raise ValueError("fused option must retain original induced connectivity")
        if any(int(graph.atom_types[i]) not in COMPOSITION_CODES["carbon_rich"] for i in self.path):
            raise ValueError("fused option grows carbon only")
        if self.edge is not None:
            chain = (self.edge[0],) + self.path
            expected = {tuple(sorted(pair)) for pair in pairwise(chain)}
            if step == FUSED_HORIZON:
                expected.add(tuple(sorted((self.path[-1], self.edge[1]))))
            actual = {tuple(sorted((i, j))) for i in self.path for j in real if graph.bonds[i, j]}
            if actual != expected:
                raise ValueError("fused path has missing bonds or extra attachments")


def descriptor_indices(families, actions, probabilities, progress: FusedProgress, edge, step):
    """All matching marks, with no greedy ranking cutoff or executor calls."""
    if step < len(GROWTH_BOND_ORDERS):
        return match_growth_descriptors(
            families,
            actions,
            probabilities,
            progress.path[-1] if progress.path else None,
            COMPOSITION_CODES["carbon_rich"],
            anchors=(edge[0],),
            bond_order=GROWTH_BOND_ORDERS[step],
        )
    if step == len(GROWTH_BOND_ORDERS):
        return match_closure_descriptors(families, actions, [(progress.path[-1], edge[1])])
    return []


def completed_fused_cycle(
    origin: MolecularGraph, graph: MolecularGraph, progress: FusedProgress
) -> bool:
    """Independent exact-slot six-cycle/aromatic witness, not an SSSR-count proxy."""
    if progress.edge is None or len(progress.path) != 4:
        return False
    real = np.flatnonzero(is_element(graph.atom_types))
    try:
        progress.validate(graph, origin, frozenset(int(i) for i in real), FUSED_HORIZON)
    except ValueError:
        return False
    cycle = (progress.edge[0],) + progress.path + (progress.edge[1],)
    before_edges = np.count_nonzero(np.triu(origin.bonds != 0, 1))
    after_edges = np.count_nonzero(np.triu(graph.bonds != 0, 1))
    # Connectivity and unchanged old induced graph are checked by the caller/
    # progress validation; four births and five added edges give rank gain one.
    if after_edges - before_edges - (graph.n_real_atoms - origin.n_real_atoms) != 1:
        return False
    aromatic = resonance_invariant_bond_classes(graph)
    return all(aromatic[u, v] == BOND_AROMATIC for u, v in zip(cycle, cycle[1:] + cycle[:1]))
