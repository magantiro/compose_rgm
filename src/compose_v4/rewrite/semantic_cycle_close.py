"""Production resonance-invariant semantics for Editing-V2 cycle closing.

The public mark identifies an undirected endpoint pair and a bond order. Every
complete Kekule assignment of an affected aromatic component is executed. The
mark is admitted only when all executions produce one canonical molecule.
Unrelated aromatic components retain their exact stored assignment so a local
rewrite cannot silently mutate a remote persistent-slot state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from math import prod

import numpy as np

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
    #: Snapshot of the source arrays, so the per-candidate identity guard does
    #: not have to rebuild `_exact_state_identity`. COPIES, because the tuples in
    #: `exact_source_identity` are immutable snapshots and this must behave the
    #: same: a later mutation of the caller's state must not make a stale
    #: context look current.
    source_arrays: tuple[np.ndarray, ...] = ()
    #: selections -> (alias, alias_is_admissible). The enumerator resolves ~960
    #: candidates per state against only ~2.4 DISTINCT Kekule selections, so
    #: without this the alias validity / connectivity / canonical-key triple is
    #: recomputed roughly 400x per distinct alias. Excluded from equality and
    #: repr: a memo of pure functions carries no identity of its own.
    alias_cache: dict = field(default_factory=dict, compare=False, repr=False)


def _context_matches(
    context: SemanticCycleCloseContext,
    state: MolecularGraph,
) -> bool:
    """Exactly `context.exact_source_identity == _exact_state_identity(state)`.

    Same predicate, without materializing four Python tuples per call -- for a
    48-slot state the bond block alone is 2304 elements and the enumerator hits
    this once per candidate. Both forms test VALUE equality, so differing integer
    dtypes compare equal under each and a shape difference is unequal under each.
    Falls back to the original comparison for a context built before this field
    existed, so an unpickled or hand-built context still resolves correctly.
    """

    if not context.source_arrays:
        return context.exact_source_identity == _exact_state_identity(state)
    stored = context.source_arrays
    return bool(
        np.array_equal(stored[0], state.atom_types)
        and np.array_equal(stored[1], state.formal_charges)
        and np.array_equal(stored[2], state.implicit_h_counts)
        and np.array_equal(stored[3], state.bonds)
    )


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
        source_arrays=(
            state.atom_types.copy(),
            state.formal_charges.copy(),
            state.implicit_h_counts.copy(),
            state.bonds.copy(),
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
    # SOURCE-GLOBAL HOIST. `prepare_semantic_cycle_close_context` REFUSES to
    # build a context unless `is_valid_state` and `is_connected_or_null` both
    # hold, so a context that belongs to this exact state already certifies
    # both; and `canonical_state_key` is a pure function of the state, so
    # `context.source_key` IS `canonical_state_key(state)`. Recomputing all
    # three per candidate was 46.4% of `enumerate_cycle_close_edges`.
    #
    # This changes no legality decision. If the context does not belong to this
    # state the original path runs unchanged, and the mismatch still raises
    # below -- AFTER the action gate, so an invalid action against a mismatched
    # context still returns INVALID_ACTION exactly as before.
    matched = context is not None and _context_matches(context, state)
    if matched:
        source_key = context.source_key
    else:
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
    elif not matched or context.source_key != source_key:
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
        # ALIAS MEMO. The alias and the first three checks depend ONLY on the
        # selection tuple -- not on the candidate bond -- so they are shared by
        # every candidate whose endpoints touch the same components. Only
        # `is_valid_bond_insert` is candidate-specific and stays per candidate.
        # The `and` chain reproduces the original `or` short-circuit exactly:
        # each predicate is evaluated only when all earlier ones passed.
        # `apply_bond_insert` and `is_valid_bond_insert` copy rather than
        # mutate, so a shared alias cannot be corrupted by a later candidate.
        selection_key = tuple(selections)
        cached = context.alias_cache.get(selection_key)
        if cached is None:
            alias = instantiate_component_factored_kekule_alias(
                state,
                components,
                selection_key,
            )
            cached = (
                alias,
                bool(
                    is_valid_state(alias)
                    and is_connected_or_null(alias)
                    and canonical_state_key(alias) == source_key
                ),
            )
            context.alias_cache[selection_key] = cached
        alias, alias_admissible = cached
        if not alias_admissible or not is_valid_bond_insert(alias, normalized):
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
