"""Complete component-factored Kekule assignment enumeration.

This production helper enumerates every fixed-charge, fixed-hydrogen
single/double lowering of each resonance-invariant aromatic-edge component.
It has no silent assignment cap.  Whole-molecule RDKit or MILP enumeration is
kept outside the runtime as an independent audit oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_SINGLE,
    MolecularGraph,
    is_element,
)
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.kernel import canonical_state_key


@dataclass(frozen=True)
class AromaticComponentAssignments:
    """Complete feasible Kekule assignments for one aromatic-edge component."""

    edges: tuple[tuple[int, int], ...]
    bond_orders: tuple[tuple[int, ...], ...]


def aromatic_edge_components(
    state: MolecularGraph,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return deterministic resonance-invariant aromatic-edge components."""

    perceived = resonance_invariant_bond_classes(state)
    real_slots = tuple(
        int(slot) for slot in np.flatnonzero(is_element(state.atom_types))
    )
    aromatic_edges = tuple(
        (left, right)
        for left_offset, left in enumerate(real_slots)
        for right in real_slots[left_offset + 1 :]
        if int(perceived[left, right]) == BOND_AROMATIC
    )
    graph = nx.Graph()
    graph.add_edges_from(aromatic_edges)
    components: list[tuple[tuple[int, int], ...]] = []
    for nodes in nx.connected_components(graph):
        node_set = frozenset(int(node) for node in nodes)
        components.append(
            tuple(
                edge
                for edge in aromatic_edges
                if edge[0] in node_set and edge[1] in node_set
            )
        )
    return tuple(sorted(components))


def instantiate_component_factored_kekule_alias(
    state: MolecularGraph,
    components: tuple[AromaticComponentAssignments, ...],
    selections: tuple[tuple[int, ...], ...],
) -> MolecularGraph:
    """Build one exact alias from one assignment per aromatic component."""

    if len(components) != len(selections):
        raise ValueError("one bond-order selection is required per aromatic component")
    bonds = state.bonds.copy()
    for component, orders in zip(components, selections, strict=True):
        if len(component.edges) != len(orders):
            raise ValueError("component edge and bond-order lengths differ")
        for (left, right), order in zip(component.edges, orders, strict=True):
            bonds[left, right] = bonds[right, left] = int(order)
    return MolecularGraph(
        state.atom_types.copy(),
        state.formal_charges.copy(),
        state.implicit_h_counts.copy(),
        bonds,
    )


def _degree_constrained_component_orders(
    state: MolecularGraph,
    edges: tuple[tuple[int, int], ...],
    *,
    source_key: str,
) -> AromaticComponentAssignments:
    if not edges:
        raise ValueError("an aromatic component must contain at least one edge")
    source_orders = tuple(int(state.bonds[edge]) for edge in edges)
    if any(order not in {BOND_SINGLE, BOND_DOUBLE} for order in source_orders):
        return AromaticComponentAssignments(edges=edges, bond_orders=())

    vertices = tuple(sorted({vertex for edge in edges for vertex in edge}))
    vertex_offset = {vertex: offset for offset, vertex in enumerate(vertices)}
    demands = [0] * len(vertices)
    for (left, right), order in zip(edges, source_orders, strict=True):
        if order == BOND_DOUBLE:
            demands[vertex_offset[left]] += 1
            demands[vertex_offset[right]] += 1

    remaining_incidence = [[0] * len(vertices) for _ in range(len(edges) + 1)]
    for edge_index in range(len(edges) - 1, -1, -1):
        remaining_incidence[edge_index] = remaining_incidence[edge_index + 1].copy()
        left, right = edges[edge_index]
        remaining_incidence[edge_index][vertex_offset[left]] += 1
        remaining_incidence[edge_index][vertex_offset[right]] += 1

    feasible_orders: list[tuple[int, ...]] = []
    current: list[int] = []

    def visit(edge_index: int, residual: tuple[int, ...]) -> None:
        if any(
            demand < 0 or demand > remaining_incidence[edge_index][offset]
            for offset, demand in enumerate(residual)
        ):
            return
        if edge_index == len(edges):
            if any(residual):
                return
            orders = tuple(current)
            component = AromaticComponentAssignments(edges=edges, bond_orders=(orders,))
            alias = instantiate_component_factored_kekule_alias(
                state,
                (component,),
                (orders,),
            )
            if (
                is_valid_state(alias)
                and is_connected_or_null(alias)
                and canonical_state_key(alias) == source_key
            ):
                feasible_orders.append(orders)
            return

        left, right = edges[edge_index]
        left_offset = vertex_offset[left]
        right_offset = vertex_offset[right]
        for order in (BOND_SINGLE, BOND_DOUBLE):
            next_residual = list(residual)
            if order == BOND_DOUBLE:
                next_residual[left_offset] -= 1
                next_residual[right_offset] -= 1
            current.append(order)
            visit(edge_index + 1, tuple(next_residual))
            current.pop()

    visit(0, tuple(demands))
    return AromaticComponentAssignments(
        edges=edges,
        bond_orders=tuple(sorted(set(feasible_orders))),
    )


def enumerate_component_factored_kekule_assignments(
    state: MolecularGraph,
) -> tuple[AromaticComponentAssignments, ...]:
    """Enumerate every local lowering, independently per aromatic component."""

    if not is_valid_state(state) or not is_connected_or_null(state):
        raise ValueError(
            "component assignment enumeration requires a valid connected source"
        )
    source_key = canonical_state_key(state)
    return tuple(
        _degree_constrained_component_orders(state, edges, source_key=source_key)
        for edges in aromatic_edge_components(state)
    )


__all__ = [
    "AromaticComponentAssignments",
    "aromatic_edge_components",
    "enumerate_component_factored_kekule_assignments",
    "instantiate_component_factored_kekule_alias",
]
