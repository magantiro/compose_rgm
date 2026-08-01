"""Non-authorizing resonance-invariant cycle-close semantics prototype.

The Editing-V2 cycle-open integration exposed a separate defect during its
required full-kernel gate: raw ``BondInsert`` cycle closing can produce a
different molecular successor from alternate Kekule encodings of one aromatic
source. This module tests the smallest conservative repair.

For a proposed endpoint pair and bond order, the resolver enumerates every
fixed-charge/hydrogen assignment of each resonance-invariant aromatic
component touching either endpoint. Components unrelated to the action are
fixed to one valid representative because their phase cannot change the local
rewrite. The action is admitted only when every executed affected-component
combination yields one canonical molecular product. Ambiguous actions are
rejected rather than assigned an arbitrary representation-dependent product.

This prototype has no assignment cap and never truncates support silently. It
does not authorize a production action, codec revision, corpus rebuild, Gate0,
T1, or P50.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import prod

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.experiments.aromatic_cycle_open_semantics import (
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

PROTOTYPE_STATUS = "NON_AUTHORIZING_SEMANTIC_CYCLE_CLOSE_PROTOTYPE"


class AromaticCycleCloseRejectionCode(str, Enum):
    INVALID_SOURCE = "invalid_source"
    INVALID_ACTION = "invalid_action"
    NO_PRESERVING_KEKULE_ALIAS = "no_charge_h_connectivity_preserving_kekule_alias"
    ALIAS_EXECUTION_DISAGREEMENT = "alias_execution_disagreement"
    EMPTY_CANONICAL_PRODUCT_GROUP = "empty_canonical_product_group"
    AMBIGUOUS_CANONICAL_PRODUCT = "multiple_canonical_product_groups"


@dataclass(frozen=True)
class AromaticCycleCloseResolution:
    prototype_status: str
    admitted: bool
    rejection_code: AromaticCycleCloseRejectionCode | None
    source_key: str | None
    action: BondInsert
    aromatic_component_count: int
    affected_component_count: int
    enumerated_affected_assignment_count: int
    executed_product_count: int
    canonical_product_keys: tuple[str, ...]
    successor: MolecularGraph | None


@dataclass(frozen=True)
class AromaticCycleCloseContext:
    """Reusable complete component evidence for one exact source state."""

    exact_source_identity: tuple
    source_key: str
    components: tuple[AromaticComponentAssignments, ...]


def _exact_state_identity(state: MolecularGraph) -> tuple:
    return (
        tuple(int(value) for value in state.atom_types),
        tuple(int(value) for value in state.formal_charges),
        tuple(int(value) for value in state.implicit_h_counts),
        tuple(int(value) for value in state.bonds.reshape(-1)),
    )


def prepare_component_factored_cycle_close_context(
    state: MolecularGraph,
) -> AromaticCycleCloseContext:
    if not is_valid_state(state) or not is_connected_or_null(state):
        raise ValueError("cycle-close context requires a valid connected source")
    return AromaticCycleCloseContext(
        exact_source_identity=_exact_state_identity(state),
        source_key=canonical_state_key(state),
        components=enumerate_component_factored_kekule_assignments(state),
    )


def _rejected(
    action: BondInsert,
    code: AromaticCycleCloseRejectionCode,
    *,
    source_key: str | None,
    component_count: int = 0,
    affected_count: int = 0,
    assignment_count: int = 0,
    executed_count: int = 0,
    product_keys: tuple[str, ...] = (),
) -> AromaticCycleCloseResolution:
    return AromaticCycleCloseResolution(
        prototype_status=PROTOTYPE_STATUS,
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


def _affected_component_indices(
    components: tuple[AromaticComponentAssignments, ...],
    action: BondInsert,
) -> tuple[int, ...]:
    endpoints = {int(action.a), int(action.b)}
    return tuple(
        index
        for index, component in enumerate(components)
        if endpoints & {vertex for edge in component.edges for vertex in edge}
    )


def resolve_component_factored_cycle_close(
    state: MolecularGraph,
    action: BondInsert,
    *,
    context: AromaticCycleCloseContext | None = None,
) -> AromaticCycleCloseResolution:
    """Resolve one proposed cycle closure across all affected Kekule phases."""

    normalized = BondInsert(
        min(int(action.a), int(action.b)),
        max(int(action.a), int(action.b)),
        int(action.order),
    )
    if not is_valid_state(state) or not is_connected_or_null(state):
        return _rejected(
            normalized,
            AromaticCycleCloseRejectionCode.INVALID_SOURCE,
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
            AromaticCycleCloseRejectionCode.INVALID_ACTION,
            source_key=source_key,
        )

    if context is None:
        context = prepare_component_factored_cycle_close_context(state)
    elif (
        context.exact_source_identity != _exact_state_identity(state)
        or context.source_key != source_key
    ):
        raise ValueError("cycle-close context belongs to another exact source state")
    components = context.components
    if any(not component.bond_orders for component in components):
        return _rejected(
            normalized,
            AromaticCycleCloseRejectionCode.NO_PRESERVING_KEKULE_ALIAS,
            source_key=source_key,
            component_count=len(components),
        )
    affected = _affected_component_indices(components, normalized)
    selected_orders = [min(component.bond_orders) for component in components]
    affected_options = tuple(components[index].bond_orders for index in affected)
    assignment_count = prod(len(options) for options in affected_options)
    if not affected_options:
        assignment_count = 1

    products: list[MolecularGraph] = []
    for choices in product(*affected_options) if affected_options else ((),):
        selections = selected_orders.copy()
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
                AromaticCycleCloseRejectionCode.ALIAS_EXECUTION_DISAGREEMENT,
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
                AromaticCycleCloseRejectionCode.ALIAS_EXECUTION_DISAGREEMENT,
                source_key=source_key,
                component_count=len(components),
                affected_count=len(affected),
                assignment_count=assignment_count,
                executed_count=len(products),
            )
        products.append(successor)

    groups: dict[str, list[MolecularGraph]] = {}
    for successor in products:
        groups.setdefault(canonical_state_key(successor), []).append(successor)
    product_keys = tuple(sorted(groups))
    if not product_keys:
        return _rejected(
            normalized,
            AromaticCycleCloseRejectionCode.EMPTY_CANONICAL_PRODUCT_GROUP,
            source_key=source_key,
            component_count=len(components),
            affected_count=len(affected),
            assignment_count=assignment_count,
        )
    if len(product_keys) != 1:
        return _rejected(
            normalized,
            AromaticCycleCloseRejectionCode.AMBIGUOUS_CANONICAL_PRODUCT,
            source_key=source_key,
            component_count=len(components),
            affected_count=len(affected),
            assignment_count=assignment_count,
            executed_count=len(products),
            product_keys=product_keys,
        )
    successor = min(
        groups[product_keys[0]],
        key=lambda item: (
            tuple(int(value) for value in item.atom_types),
            tuple(int(value) for value in item.formal_charges),
            tuple(int(value) for value in item.implicit_h_counts),
            tuple(int(value) for value in item.bonds.reshape(-1)),
        ),
    )
    return AromaticCycleCloseResolution(
        prototype_status=PROTOTYPE_STATUS,
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
    "PROTOTYPE_STATUS",
    "AromaticCycleCloseContext",
    "AromaticCycleCloseRejectionCode",
    "AromaticCycleCloseResolution",
    "prepare_component_factored_cycle_close_context",
    "resolve_component_factored_cycle_close",
]
