"""Production resonance-invariant semantics for Editing-V2 cycle closing.

The public mark identifies an undirected endpoint pair and a bond order. Every
complete Kekule assignment of an affected aromatic component is executed. The
mark is admitted only when all executions produce one canonical molecule.
Unrelated aromatic components retain their exact stored assignment so a local
rewrite cannot silently mutate a remote persistent-slot state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import prod

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.rewrite.aromatic_kekule import (
    AromaticComponentAssignments,
    enumerate_component_factored_kekule_assignments,
    instantiate_component_factored_kekule_alias,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import (
    BondInsert,
    apply_bond_insert,
    is_valid_bond_insert,
)


class SemanticCycleCloseRejectionCode(str, Enum):
    INVALID_SOURCE = "invalid_source"
    INVALID_ACTION = "invalid_action"
    NO_PRESERVING_KEKULE_ALIAS = "no_charge_h_connectivity_preserving_kekule_alias"
    SOURCE_ASSIGNMENT_MISSING = "exact_source_assignment_missing"
    ALIAS_EXECUTION_DISAGREEMENT = "alias_execution_disagreement"
    NO_CHARGE_POLICY_PRESERVING_PRODUCT = "no_charge_policy_preserving_product"
    EMPTY_CANONICAL_PRODUCT_GROUP = "empty_canonical_product_group"
    AMBIGUOUS_CANONICAL_PRODUCT = "multiple_canonical_product_groups"


@dataclass(frozen=True)
class SemanticCycleCloseResolution:
    admitted: bool
    rejection_code: SemanticCycleCloseRejectionCode | None
    source_key: str | None
    action: BondInsert
    aromatic_component_count: int
    affected_component_count: int
    enumerated_affected_assignment_count: int
    executed_product_count: int
    canonical_product_keys: tuple[str, ...]
    successor: MolecularGraph | None


@dataclass(frozen=True)
class SemanticCycleCloseContext:
    exact_source_identity: tuple[tuple[int, ...], ...]
    source_key: str
    components: tuple[AromaticComponentAssignments, ...]
    source_orders: tuple[tuple[int, ...], ...]


def _exact_state_identity(state: MolecularGraph) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(int(value) for value in state.atom_types),
        tuple(int(value) for value in state.formal_charges),
        tuple(int(value) for value in state.implicit_h_counts),
        tuple(int(value) for value in state.bonds.reshape(-1)),
    )


def prepare_semantic_cycle_close_context(
    state: MolecularGraph,
) -> SemanticCycleCloseContext:
    if not is_valid_state(state) or not is_connected_or_null(state):
        raise ValueError(
            "semantic cycle-close context requires a valid connected source"
        )
    components = enumerate_component_factored_kekule_assignments(state)
    return SemanticCycleCloseContext(
        exact_source_identity=_exact_state_identity(state),
        source_key=canonical_state_key(state),
        components=components,
        source_orders=tuple(
            tuple(int(state.bonds[edge]) for edge in component.edges)
            for component in components
        ),
    )


def _rejected(
    action: BondInsert,
    code: SemanticCycleCloseRejectionCode,
    *,
    source_key: str | None,
    component_count: int = 0,
    affected_count: int = 0,
    assignment_count: int = 0,
    executed_count: int = 0,
    product_keys: tuple[str, ...] = (),
) -> SemanticCycleCloseResolution:
    return SemanticCycleCloseResolution(
        admitted=False,
        rejection_code=code,
        source_key=source_key,
        action=action,
        aromatic_component_count=component_count,
        affected_component_count=affected_count,
        enumerated_affected_assignment_count=assignment_count,
        executed_product_count=executed_count,
        canonical_product_keys=product_keys,
        successor=None,
    )


def resolve_semantic_cycle_close(
    state: MolecularGraph,
    action: BondInsert,
    *,
    context: SemanticCycleCloseContext | None = None,
) -> SemanticCycleCloseResolution:
    normalized = BondInsert(
        min(int(action.a), int(action.b)),
        max(int(action.a), int(action.b)),
        int(action.order),
    )
    if not is_valid_state(state) or not is_connected_or_null(state):
        return _rejected(
            normalized,
            SemanticCycleCloseRejectionCode.INVALID_SOURCE,
            source_key=None,
        )
    source_key = canonical_state_key(state)
    if (
        type(action) is not BondInsert
        or int(action.a) >= int(action.b)
        or int(action.a) < 0
        or int(action.b) >= state.n_atoms
        or not bool(is_element(state.atom_types[int(action.a)]))
        or not bool(is_element(state.atom_types[int(action.b)]))
        or int(state.bonds[int(action.a), int(action.b)]) != 0
        or int(action.order) not in {1, 2, 3}
    ):
        return _rejected(
            normalized,
            SemanticCycleCloseRejectionCode.INVALID_ACTION,
            source_key=source_key,
        )
    if context is None:
        context = prepare_semantic_cycle_close_context(state)
    elif (
        context.exact_source_identity != _exact_state_identity(state)
        or context.source_key != source_key
    ):
        raise ValueError("semantic cycle-close context belongs to another exact source")

    components = context.components
    if any(not component.bond_orders for component in components):
        return _rejected(
            normalized,
            SemanticCycleCloseRejectionCode.NO_PRESERVING_KEKULE_ALIAS,
            source_key=source_key,
            component_count=len(components),
        )
    if any(
        orders not in component.bond_orders
        for component, orders in zip(components, context.source_orders, strict=True)
    ):
        return _rejected(
            normalized,
            SemanticCycleCloseRejectionCode.SOURCE_ASSIGNMENT_MISSING,
            source_key=source_key,
            component_count=len(components),
        )

    endpoints = {int(normalized.a), int(normalized.b)}
    affected = tuple(
        index
        for index, component in enumerate(components)
        if endpoints & {vertex for edge in component.edges for vertex in edge}
    )
    affected_options = tuple(components[index].bond_orders for index in affected)
    assignment_count = prod(len(options) for options in affected_options) or 1
    products: list[MolecularGraph] = []
    for choices in product(*affected_options) if affected_options else ((),):
        selections = list(context.source_orders)
        for component_index, orders in zip(affected, choices, strict=True):
            selections[component_index] = orders
        alias = instantiate_component_factored_kekule_alias(
            state,
            components,
            tuple(selections),
        )
        if (
            not is_valid_state(alias)
            or not is_connected_or_null(alias)
            or canonical_state_key(alias) != source_key
            or not is_valid_bond_insert(alias, normalized)
        ):
            return _rejected(
                normalized,
                SemanticCycleCloseRejectionCode.ALIAS_EXECUTION_DISAGREEMENT,
                source_key=source_key,
                component_count=len(components),
                affected_count=len(affected),
                assignment_count=assignment_count,
                executed_count=len(products),
            )
        successor = apply_bond_insert(alias, normalized)
        if not is_valid_state(successor) or not is_connected_or_null(successor):
            return _rejected(
                normalized,
                SemanticCycleCloseRejectionCode.ALIAS_EXECUTION_DISAGREEMENT,
                source_key=source_key,
                component_count=len(components),
                affected_count=len(affected),
                assignment_count=assignment_count,
                executed_count=len(products),
            )
        products.append(successor)

    groups: dict[str, dict[tuple[tuple[int, ...], ...], MolecularGraph]] = {}
    for successor in products:
        if not charge_policy_preserved(state, successor):
            continue
        groups.setdefault(canonical_state_key(successor), {}).setdefault(
            _exact_state_identity(successor),
            successor,
        )
    product_keys = tuple(sorted(groups))
    if not product_keys:
        return _rejected(
            normalized,
            (
                SemanticCycleCloseRejectionCode.NO_CHARGE_POLICY_PRESERVING_PRODUCT
                if products
                else SemanticCycleCloseRejectionCode.EMPTY_CANONICAL_PRODUCT_GROUP
            ),
            source_key=source_key,
            component_count=len(components),
            affected_count=len(affected),
            assignment_count=assignment_count,
        )
    if len(product_keys) != 1:
        return _rejected(
            normalized,
            SemanticCycleCloseRejectionCode.AMBIGUOUS_CANONICAL_PRODUCT,
            source_key=source_key,
            component_count=len(components),
            affected_count=len(affected),
            assignment_count=assignment_count,
            executed_count=len(products),
            product_keys=product_keys,
        )
    representatives = groups[product_keys[0]]
    successor = representatives[min(representatives)]
    return SemanticCycleCloseResolution(
        admitted=True,
        rejection_code=None,
        source_key=source_key,
        action=normalized,
        aromatic_component_count=len(components),
        affected_component_count=len(affected),
        enumerated_affected_assignment_count=assignment_count,
        executed_product_count=len(products),
        canonical_product_keys=product_keys,
        successor=successor,
    )


__all__ = [
    "SemanticCycleCloseContext",
    "SemanticCycleCloseRejectionCode",
    "SemanticCycleCloseResolution",
    "prepare_semantic_cycle_close_context",
    "resolve_semantic_cycle_close",
]
