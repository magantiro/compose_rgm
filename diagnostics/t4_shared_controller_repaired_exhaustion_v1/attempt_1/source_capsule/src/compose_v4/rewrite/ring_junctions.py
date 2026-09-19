"""Chemical topology descriptors for coordinated ring transactions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.rewrite.tracelets import CycleAttach, CycleInsert, RingEarInsert


RING_JUNCTION_KINDS = ("root", "attached", "spiro", "fused", "bridged")
RING_JUNCTION_TO_INDEX = {
    name: index for index, name in enumerate(RING_JUNCTION_KINDS)
}


@dataclass(frozen=True)
class RingJunctionDescriptor:
    """Topology created by one atomic ring transaction.

    ``nominal_ring_size`` is exact for root, attached, spiro, and fused
    additions.  For a bridged ear it is the smallest new cycle containing the
    ear, obtained from the shortest pre-existing path between its anchors.
    """

    kind: str
    new_atoms: int
    nominal_ring_size: int
    anchor_distance: int
    cycle_rank_increment: int = 1


def describe_ring_transaction(
    state: MolecularGraph,
    action: CycleInsert | CycleAttach | RingEarInsert,
) -> RingJunctionDescriptor:
    if isinstance(action, CycleInsert):
        size = len(action.atoms)
        return RingJunctionDescriptor("root", size, size, 0)
    if isinstance(action, CycleAttach):
        size = len(action.atoms)
        return RingJunctionDescriptor("attached", size, size, 1)
    if not isinstance(action, RingEarInsert):
        raise TypeError(f"unsupported ring transaction: {type(action).__name__}")

    span = len(action.atoms)
    a, b = int(action.a), int(action.b)
    if a == b:
        return RingJunctionDescriptor("spiro", span, span + 1, 0)
    graph = _state_graph(state)
    try:
        distance = int(nx.shortest_path_length(graph, a, b))
    except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
        raise ValueError("ear anchors must be connected in the current state") from exc
    kind = "fused" if int(state.bonds[a, b]) != 0 else "bridged"
    return RingJunctionDescriptor(
        kind,
        span,
        span + 1 + distance,
        distance,
    )


def _state_graph(state: MolecularGraph) -> nx.Graph:
    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    graph = nx.Graph()
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (a, b)
        for a, b in combinations(real, 2)
        if int(state.bonds[a, b]) != 0
    )
    return graph


__all__ = [
    "RING_JUNCTION_KINDS",
    "RING_JUNCTION_TO_INDEX",
    "RingJunctionDescriptor",
    "describe_ring_transaction",
]
