"""Editing-V2 semantic atom-restatement resolution.

The historical :class:`~compose_v4.rewrite.operators.AtomRestate` payload
stores an implicit-hydrogen count derived from one exact Kekule lowering.  On
an aromatic site, executing that payload can therefore produce a different
molecule after changing only the stored resonance phase.  Editing V2 instead
addresses an atom by persistent slot and a broad-organic element-valence
class, resolves every exact lowering of the affected aromatic component, and
admits the action only when all executable lowerings induce one canonical
molecular product.

This module is production domain logic.  Complete global resonance enumeration
remains an independent bounded audit oracle; it is not used by the executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_CLASS_TO_H_CHANGE,
    MAX_H_COUNT,
    MolecularGraph,
    ORGANIC_VOCABULARY,
    is_element,
)
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.rewrite.aromatic_kekule import (
    AromaticComponentAssignments,
    enumerate_component_factored_kekule_assignments,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.operators import (
    AtomRestate,
    apply_atom_restate,
    is_valid_atom_restate,
)


class SemanticAtomRestateRejectionCode(str, Enum):
    """Stable reason codes for actions excluded from the semantic fiber."""

    INVALID_SOURCE = "invalid_source"
    INVALID_ACTION = "invalid_action"
    UNSUPPORTED_FORMAL_CHARGE = "edited_atom_formal_charge_is_not_zero"
    COMPONENT_CONTAINS_NO_KEKULE_ASSIGNMENT = (
        "affected_aromatic_component_has_no_complete_kekule_assignment"
    )
    COMPONENT_ASSIGNMENT_DISAGREEMENT = "component_assignment_disagreement"
    NO_TARGET_COMPATIBLE_ALIAS = "no_target_compatible_source_alias"
    NO_CHARGE_POLICY_PRESERVING_PRODUCT = "no_charge_policy_preserving_product"
    AMBIGUOUS_CANONICAL_PRODUCT = "multiple_canonical_product_groups"


@dataclass(frozen=True)
class SemanticAtomRestateContext:
    """Reusable exact resonance context for one persistent-slot source."""

    source_key: str
    source_exact_state_key: tuple[tuple[int, ...], ...]
    components: tuple[AromaticComponentAssignments, ...]
    component_index_by_vertex: tuple[int, ...]


@dataclass(frozen=True)
class SemanticAtomRestateResolution:
    """One semantic atom-restatement admission decision."""

    admitted: bool
    rejection_code: SemanticAtomRestateRejectionCode | None
    source_key: str | None
    vertex: int
    target_class_index: int
    semantic_aromatic_site: bool
    component_assignment_count: int
    target_compatible_alias_count: int
    charge_policy_preserving_product_count: int
    canonical_product_keys: tuple[str, ...]
    successor: MolecularGraph | None


def _exact_state_key(state: MolecularGraph) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(int(value) for value in state.atom_types),
        tuple(int(value) for value in state.formal_charges),
        tuple(int(value) for value in state.implicit_h_counts),
        tuple(int(value) for value in state.bonds.reshape(-1)),
    )


def _component_vertices(component: AromaticComponentAssignments) -> frozenset[int]:
    return frozenset(vertex for edge in component.edges for vertex in edge)


def _state_with_component_orders(
    state: MolecularGraph,
    component: AromaticComponentAssignments,
    orders: tuple[int, ...],
) -> MolecularGraph:
    if len(component.edges) != len(orders):
        raise ValueError("component edge and bond-order lengths differ")
    bonds = state.bonds.copy()
    for (left, right), order in zip(component.edges, orders, strict=True):
        bonds[left, right] = bonds[right, left] = int(order)
    return MolecularGraph(
        state.atom_types.copy(),
        state.formal_charges.copy(),
        state.implicit_h_counts.copy(),
        bonds,
    )


def prepare_semantic_atom_restate_context(
    state: MolecularGraph,
) -> SemanticAtomRestateContext:
    """Precompute complete affected-component assignments for one source."""

    if not is_valid_state(state) or not is_connected_or_null(state):
        raise ValueError("semantic atom restatement requires a valid connected source")
    source_key = canonical_state_key(state)
    components = enumerate_component_factored_kekule_assignments(state)
    component_index_by_vertex = [-1] * state.n_atoms
    for component_index, component in enumerate(components):
        for vertex in _component_vertices(component):
            if component_index_by_vertex[vertex] >= 0:
                raise ValueError("aromatic-edge components overlap at one vertex")
            component_index_by_vertex[vertex] = component_index
    return SemanticAtomRestateContext(
        source_key=source_key,
        source_exact_state_key=_exact_state_key(state),
        components=components,
        component_index_by_vertex=tuple(component_index_by_vertex),
    )


def _rejected(
    *,
    code: SemanticAtomRestateRejectionCode,
    source_key: str | None,
    vertex: int,
    target_class_index: int,
    semantic_aromatic_site: bool = False,
    component_assignment_count: int = 0,
    target_compatible_alias_count: int = 0,
    charge_policy_preserving_product_count: int = 0,
    canonical_product_keys: tuple[str, ...] = (),
) -> SemanticAtomRestateResolution:
    return SemanticAtomRestateResolution(
        admitted=False,
        rejection_code=code,
        source_key=source_key,
        vertex=vertex,
        target_class_index=target_class_index,
        semantic_aromatic_site=semantic_aromatic_site,
        component_assignment_count=component_assignment_count,
        target_compatible_alias_count=target_compatible_alias_count,
        charge_policy_preserving_product_count=charge_policy_preserving_product_count,
        canonical_product_keys=canonical_product_keys,
        successor=None,
    )


def _target_restate(
    state: MolecularGraph,
    *,
    vertex: int,
    target_class_index: int,
) -> AtomRestate | None:
    bond_valence = sum(
        int(BOND_CLASS_TO_H_CHANGE[int(order)]) for order in state.bonds[vertex]
    )
    hydrogen_count = ORGANIC_VOCABULARY.h_count(
        target_class_index,
        bond_valence,
        formal_charge=0,
    )
    if hydrogen_count is None or not 0 <= int(hydrogen_count) <= MAX_H_COUNT:
        return None
    return AtomRestate(
        v=vertex,
        atom_type=int(ORGANIC_VOCABULARY.element_of(target_class_index)),
        formal_charge=0,
        implicit_h_count=int(hydrogen_count),
    )


def resolve_semantic_atom_restate(
    state: MolecularGraph,
    *,
    vertex: int,
    target_class_index: int,
    context: SemanticAtomRestateContext | None = None,
) -> SemanticAtomRestateResolution:
    """Resolve one `(slot, target class)` action without phase-dependent guessing."""

    vertex = int(vertex)
    target_class_index = int(target_class_index)
    if not is_valid_state(state) or not is_connected_or_null(state):
        return _rejected(
            code=SemanticAtomRestateRejectionCode.INVALID_SOURCE,
            source_key=None,
            vertex=vertex,
            target_class_index=target_class_index,
        )
    source_key = canonical_state_key(state)
    if (
        vertex < 0
        or vertex >= state.n_atoms
        or not bool(is_element(np.asarray(state.atom_types[vertex])))
        or target_class_index < 0
        or target_class_index >= len(ORGANIC_VOCABULARY)
    ):
        return _rejected(
            code=SemanticAtomRestateRejectionCode.INVALID_ACTION,
            source_key=source_key,
            vertex=vertex,
            target_class_index=target_class_index,
        )
    if int(state.formal_charges[vertex]) != 0:
        return _rejected(
            code=SemanticAtomRestateRejectionCode.UNSUPPORTED_FORMAL_CHARGE,
            source_key=source_key,
            vertex=vertex,
            target_class_index=target_class_index,
        )

    if context is None:
        context = prepare_semantic_atom_restate_context(state)
    if (
        context.source_key != source_key
        or context.source_exact_state_key != _exact_state_key(state)
    ):
        raise ValueError("semantic atom-restatement context belongs to another source")
    if len(context.component_index_by_vertex) != state.n_atoms:
        raise ValueError(
            "semantic atom-restatement context has the wrong slot capacity"
        )

    perceived = resonance_invariant_bond_classes(state)
    semantic_aromatic_site = bool(np.any(perceived[vertex] == BOND_AROMATIC))
    component_index = int(context.component_index_by_vertex[vertex])
    if semantic_aromatic_site != (component_index >= 0):
        return _rejected(
            code=SemanticAtomRestateRejectionCode.COMPONENT_ASSIGNMENT_DISAGREEMENT,
            source_key=source_key,
            vertex=vertex,
            target_class_index=target_class_index,
            semantic_aromatic_site=semantic_aromatic_site,
        )

    if component_index < 0:
        aliases = (state,)
    else:
        component = context.components[component_index]
        if not component.bond_orders:
            return _rejected(
                code=(
                    SemanticAtomRestateRejectionCode.COMPONENT_CONTAINS_NO_KEKULE_ASSIGNMENT
                ),
                source_key=source_key,
                vertex=vertex,
                target_class_index=target_class_index,
                semantic_aromatic_site=True,
            )
        aliases = tuple(
            _state_with_component_orders(state, component, orders)
            for orders in component.bond_orders
        )

    product_groups: dict[str, dict[tuple[tuple[int, ...], ...], MolecularGraph]] = {}
    compatible_count = 0
    charge_preserving_count = 0
    for alias in aliases:
        if (
            not is_valid_state(alias)
            or not is_connected_or_null(alias)
            or canonical_state_key(alias) != source_key
        ):
            return _rejected(
                code=SemanticAtomRestateRejectionCode.COMPONENT_ASSIGNMENT_DISAGREEMENT,
                source_key=source_key,
                vertex=vertex,
                target_class_index=target_class_index,
                semantic_aromatic_site=semantic_aromatic_site,
                component_assignment_count=len(aliases),
                target_compatible_alias_count=compatible_count,
            )
        raw_action = _target_restate(
            alias,
            vertex=vertex,
            target_class_index=target_class_index,
        )
        if raw_action is None or not is_valid_atom_restate(alias, raw_action):
            continue
        successor = apply_atom_restate(alias, raw_action)
        if not is_connected_or_null(successor):
            continue
        compatible_count += 1
        if not charge_policy_preserved(state, successor):
            continue
        charge_preserving_count += 1
        key = canonical_state_key(successor)
        product_groups.setdefault(key, {}).setdefault(
            _exact_state_key(successor), successor
        )

    canonical_product_keys = tuple(sorted(product_groups))
    if not canonical_product_keys:
        return _rejected(
            code=(
                SemanticAtomRestateRejectionCode.NO_TARGET_COMPATIBLE_ALIAS
                if compatible_count == 0
                else SemanticAtomRestateRejectionCode.NO_CHARGE_POLICY_PRESERVING_PRODUCT
            ),
            source_key=source_key,
            vertex=vertex,
            target_class_index=target_class_index,
            semantic_aromatic_site=semantic_aromatic_site,
            component_assignment_count=len(aliases),
            target_compatible_alias_count=compatible_count,
        )
    if len(canonical_product_keys) != 1:
        return _rejected(
            code=SemanticAtomRestateRejectionCode.AMBIGUOUS_CANONICAL_PRODUCT,
            source_key=source_key,
            vertex=vertex,
            target_class_index=target_class_index,
            semantic_aromatic_site=semantic_aromatic_site,
            component_assignment_count=len(aliases),
            target_compatible_alias_count=compatible_count,
            charge_policy_preserving_product_count=charge_preserving_count,
            canonical_product_keys=canonical_product_keys,
        )
    representatives = product_groups[canonical_product_keys[0]]
    successor = representatives[min(representatives)]
    return SemanticAtomRestateResolution(
        admitted=True,
        rejection_code=None,
        source_key=source_key,
        vertex=vertex,
        target_class_index=target_class_index,
        semantic_aromatic_site=semantic_aromatic_site,
        component_assignment_count=len(aliases),
        target_compatible_alias_count=compatible_count,
        charge_policy_preserving_product_count=charge_preserving_count,
        canonical_product_keys=canonical_product_keys,
        successor=successor,
    )


__all__ = [
    "SemanticAtomRestateContext",
    "SemanticAtomRestateRejectionCode",
    "SemanticAtomRestateResolution",
    "prepare_semantic_atom_restate_context",
    "resolve_semantic_atom_restate",
]
