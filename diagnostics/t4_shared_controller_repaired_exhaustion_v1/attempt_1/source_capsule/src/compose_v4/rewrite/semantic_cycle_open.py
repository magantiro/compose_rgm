"""Production resonance-invariant semantics for Editing-V2 cycle opening."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod

import networkx as nx
import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_NULL,
    BOND_SINGLE,
    MolecularGraph,
    is_element,
)
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.rewrite.aromatic_kekule import (
    enumerate_component_factored_kekule_assignments,
    instantiate_component_factored_kekule_alias,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import (
    BondDelete,
    apply_bond_delete,
    is_valid_bond_delete,
)


class SemanticCycleOpenRejectionCode(str, Enum):
    INVALID_SOURCE = "invalid_source"
    INVALID_EDGE = "invalid_edge"
    EDGE_ABSENT = "edge_absent"
    EDGE_IS_BRIDGE = "edge_is_bridge"
    NONAROMATIC_EXECUTOR_REJECTED = "nonaromatic_executor_rejected"
    NO_PRESERVING_KEKULE_ALIAS = "no_charge_h_connectivity_preserving_kekule_alias"
    NO_FORCED_SINGLE_ALIAS = "selected_aromatic_edge_has_no_forced_single_alias"
    SOURCE_ASSIGNMENT_MISSING = "exact_source_assignment_missing"
    COMPONENT_FACTORIZATION_DISAGREEMENT = "component_factorization_disagreement"
    NO_CHARGE_POLICY_PRESERVING_PRODUCT = "no_charge_policy_preserving_product"
    EMPTY_CANONICAL_PRODUCT_GROUP = "empty_canonical_product_group"
    AMBIGUOUS_CANONICAL_PRODUCT = "multiple_canonical_product_groups"


@dataclass(frozen=True)
class SemanticCycleOpenResolution:
    admitted: bool
    rejection_code: SemanticCycleOpenRejectionCode | None
    source_key: str | None
    edge: tuple[int, int]
    semantic_aromatic_edge: bool
    enumerated_alias_count: int
    forced_single_alias_count: int
    executed_product_count: int
    canonical_product_keys: tuple[str, ...]
    successor: MolecularGraph | None
    inverse_bond_order: int | None


def _exact_state_key(state: MolecularGraph) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(int(value) for value in state.atom_types),
        tuple(int(value) for value in state.formal_charges),
        tuple(int(value) for value in state.implicit_h_counts),
        tuple(int(value) for value in state.bonds.reshape(-1)),
    )


def _edge_is_bridge(state: MolecularGraph, edge: tuple[int, int]) -> bool:
    real_slots = tuple(
        int(slot) for slot in np.flatnonzero(is_element(state.atom_types))
    )
    graph = nx.Graph()
    graph.add_nodes_from(real_slots)
    graph.add_edges_from(
        (left, right)
        for offset, left in enumerate(real_slots)
        for right in real_slots[offset + 1 :]
        if int(state.bonds[left, right]) != BOND_NULL
    )
    return frozenset(edge) in {
        frozenset((int(left), int(right))) for left, right in nx.bridges(graph)
    }


def _rejected(
    code: SemanticCycleOpenRejectionCode,
    *,
    edge: tuple[int, int],
    source_key: str | None,
    semantic_aromatic_edge: bool,
    enumerated_alias_count: int = 0,
    forced_single_alias_count: int = 0,
    executed_product_count: int = 0,
    canonical_product_keys: tuple[str, ...] = (),
) -> SemanticCycleOpenResolution:
    return SemanticCycleOpenResolution(
        admitted=False,
        rejection_code=code,
        source_key=source_key,
        edge=edge,
        semantic_aromatic_edge=semantic_aromatic_edge,
        enumerated_alias_count=enumerated_alias_count,
        forced_single_alias_count=forced_single_alias_count,
        executed_product_count=executed_product_count,
        canonical_product_keys=canonical_product_keys,
        successor=None,
        inverse_bond_order=None,
    )


def resolve_semantic_cycle_open(
    state: MolecularGraph,
    action: BondDelete,
) -> SemanticCycleOpenResolution:
    edge = tuple(sorted((int(action.a), int(action.b))))
    if not is_valid_state(state) or not is_connected_or_null(state):
        return _rejected(
            SemanticCycleOpenRejectionCode.INVALID_SOURCE,
            edge=edge,
            source_key=None,
            semantic_aromatic_edge=False,
        )
    source_key = canonical_state_key(state)
    if (
        type(action) is not BondDelete
        or int(action.a) >= int(action.b)
        or edge[0] < 0
        or edge[1] >= state.n_atoms
        or not bool(is_element(state.atom_types[edge[0]]))
        or not bool(is_element(state.atom_types[edge[1]]))
    ):
        return _rejected(
            SemanticCycleOpenRejectionCode.INVALID_EDGE,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=False,
        )
    if int(state.bonds[edge]) == BOND_NULL:
        return _rejected(
            SemanticCycleOpenRejectionCode.EDGE_ABSENT,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=False,
        )
    perceived = resonance_invariant_bond_classes(state)
    semantic_aromatic = int(perceived[edge]) == BOND_AROMATIC
    if _edge_is_bridge(state, edge):
        return _rejected(
            SemanticCycleOpenRejectionCode.EDGE_IS_BRIDGE,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=semantic_aromatic,
        )

    normalized = BondDelete(*edge)
    if not semantic_aromatic:
        if not is_valid_bond_delete(state, normalized):
            return _rejected(
                SemanticCycleOpenRejectionCode.NONAROMATIC_EXECUTOR_REJECTED,
                edge=edge,
                source_key=source_key,
                semantic_aromatic_edge=False,
            )
        successor = apply_bond_delete(state, normalized)
        if not is_connected_or_null(successor) or not charge_policy_preserved(
            state,
            successor,
        ):
            return _rejected(
                SemanticCycleOpenRejectionCode.NONAROMATIC_EXECUTOR_REJECTED,
                edge=edge,
                source_key=source_key,
                semantic_aromatic_edge=False,
            )
        return SemanticCycleOpenResolution(
            admitted=True,
            rejection_code=None,
            source_key=source_key,
            edge=edge,
            semantic_aromatic_edge=False,
            enumerated_alias_count=0,
            forced_single_alias_count=0,
            executed_product_count=1,
            canonical_product_keys=(canonical_state_key(successor),),
            successor=successor,
            inverse_bond_order=int(state.bonds[edge]),
        )

    components = enumerate_component_factored_kekule_assignments(state)
    selected_indices = tuple(
        index for index, component in enumerate(components) if edge in component.edges
    )
    if len(selected_indices) != 1 or any(
        not component.bond_orders for component in components
    ):
        return _rejected(
            (
                SemanticCycleOpenRejectionCode.COMPONENT_FACTORIZATION_DISAGREEMENT
                if len(selected_indices) != 1
                else SemanticCycleOpenRejectionCode.NO_PRESERVING_KEKULE_ALIAS
            ),
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=True,
        )
    source_orders = tuple(
        tuple(int(state.bonds[component_edge]) for component_edge in component.edges)
        for component in components
    )
    if any(
        orders not in component.bond_orders
        for component, orders in zip(components, source_orders, strict=True)
    ):
        return _rejected(
            SemanticCycleOpenRejectionCode.SOURCE_ASSIGNMENT_MISSING,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=True,
        )

    selected_index = selected_indices[0]
    selected_component = components[selected_index]
    selected_edge_offset = selected_component.edges.index(edge)
    selected_forced_single = tuple(
        orders
        for orders in selected_component.bond_orders
        if int(orders[selected_edge_offset]) == BOND_SINGLE
    )
    alias_count = prod(len(component.bond_orders) for component in components)
    irrelevant_multiplier = prod(
        len(component.bond_orders)
        for index, component in enumerate(components)
        if index != selected_index
    )
    forced_single_count = len(selected_forced_single) * irrelevant_multiplier
    if not selected_forced_single:
        return _rejected(
            SemanticCycleOpenRejectionCode.NO_FORCED_SINGLE_ALIAS,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=True,
            enumerated_alias_count=alias_count,
        )

    products: list[MolecularGraph] = []
    for selected_orders in selected_forced_single:
        selections = list(source_orders)
        selections[selected_index] = selected_orders
        alias = instantiate_component_factored_kekule_alias(
            state,
            components,
            tuple(selections),
        )
        if (
            not is_valid_state(alias)
            or not is_connected_or_null(alias)
            or canonical_state_key(alias) != source_key
            or not is_valid_bond_delete(alias, normalized)
        ):
            return _rejected(
                SemanticCycleOpenRejectionCode.COMPONENT_FACTORIZATION_DISAGREEMENT,
                edge=edge,
                source_key=source_key,
                semantic_aromatic_edge=True,
                enumerated_alias_count=alias_count,
                forced_single_alias_count=forced_single_count,
                executed_product_count=len(products),
            )
        successor = apply_bond_delete(alias, normalized)
        if not is_connected_or_null(successor):
            return _rejected(
                SemanticCycleOpenRejectionCode.COMPONENT_FACTORIZATION_DISAGREEMENT,
                edge=edge,
                source_key=source_key,
                semantic_aromatic_edge=True,
                enumerated_alias_count=alias_count,
                forced_single_alias_count=forced_single_count,
                executed_product_count=len(products),
            )
        products.append(successor)

    groups: dict[str, dict[tuple[tuple[int, ...], ...], MolecularGraph]] = {}
    for successor in products:
        if not charge_policy_preserved(state, successor):
            continue
        groups.setdefault(canonical_state_key(successor), {}).setdefault(
            _exact_state_key(successor),
            successor,
        )
    product_keys = tuple(sorted(groups))
    if not product_keys:
        return _rejected(
            (
                SemanticCycleOpenRejectionCode.NO_CHARGE_POLICY_PRESERVING_PRODUCT
                if products
                else SemanticCycleOpenRejectionCode.EMPTY_CANONICAL_PRODUCT_GROUP
            ),
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=True,
            enumerated_alias_count=alias_count,
            forced_single_alias_count=forced_single_count,
        )
    if len(product_keys) != 1:
        return _rejected(
            SemanticCycleOpenRejectionCode.AMBIGUOUS_CANONICAL_PRODUCT,
            edge=edge,
            source_key=source_key,
            semantic_aromatic_edge=True,
            enumerated_alias_count=alias_count,
            forced_single_alias_count=forced_single_count,
            executed_product_count=len(products),
            canonical_product_keys=product_keys,
        )
    representatives = groups[product_keys[0]]
    successor = representatives[min(representatives)]
    return SemanticCycleOpenResolution(
        admitted=True,
        rejection_code=None,
        source_key=source_key,
        edge=edge,
        semantic_aromatic_edge=True,
        enumerated_alias_count=alias_count,
        forced_single_alias_count=forced_single_count,
        executed_product_count=len(products),
        canonical_product_keys=product_keys,
        successor=successor,
        inverse_bond_order=BOND_SINGLE,
    )


__all__ = [
    "SemanticCycleOpenRejectionCode",
    "SemanticCycleOpenResolution",
    "resolve_semantic_cycle_open",
]
