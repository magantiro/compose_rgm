"""Chemically factorized neutral C/N/O/F action fiber.

This module removes two sources of brute-force enumeration from the first
scaling gate:

1. padding-slot aliases use one representative slot; and
2. the inserted/restated hydrogen count is derived from neutral valence rather
   than proposed as an independent categorical variable.

The remaining local masks are sufficient for neutral C/N/O/F chemistry. Every
terminal choice is still committed through the production rewrite runtime; a
mask/runtime disagreement is an error rather than a rejected proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    MAX_H_COUNT,
    MolecularGraph,
    NULL_IDX,
    STANDARD_VALENCE,
    is_element,
)
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.fiber import MarkedTransition
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    RewriteSystem,
    canonical_state_key,
    de_novo_rewrite_system,
)
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondDelete,
    BondInsert,
    BondReorder,
    BondReroute,
    MICRO_BOND_CLASSES,
)


CNOF_SYMBOLS = ("C", "N", "O", "F")
CNOF_ATOM_TYPES = tuple(ELEMENT_TO_IDX[symbol] for symbol in CNOF_SYMBOLS)
CNOF_VALENCE = {
    ELEMENT_TO_IDX[symbol]: int(STANDARD_VALENCE[symbol])
    for symbol in CNOF_SYMBOLS
}
RULE_FAMILIES = (
    "atom_insert",
    "atom_delete",
    "atom_restate",
    "bond_insert",
    "bond_delete",
    "bond_reorder",
)
TRANSPORT_RULE_FAMILIES = (*RULE_FAMILIES, "bond_reroute")


class FactorizedFiberError(RuntimeError):
    """Raised when a supposedly exact chemistry mask disagrees with runtime."""


@dataclass(frozen=True)
class FactorizedFiber:
    by_family: dict[str, tuple[MarkedTransition, ...]]
    rule_families: tuple[str, ...] = RULE_FAMILIES

    @property
    def transitions(self) -> tuple[MarkedTransition, ...]:
        return tuple(
            transition
            for family in self.rule_families
            for transition in self.by_family.get(family, ())
        )

    @property
    def enabled_families(self) -> tuple[str, ...]:
        return tuple(
            family for family in self.rule_families if self.by_family.get(family)
        )


def enumerate_factorized_cnof_fiber(
    state: MolecularGraph,
    *,
    system: RewriteSystem | None = None,
    allow_bond_reroute: bool = False,
) -> FactorizedFiber:
    """Enumerate the exact slot-quotiented neutral C/N/O/F action fiber."""

    _assert_supported_state(state)
    runtime = system or de_novo_rewrite_system()
    rule_families = TRANSPORT_RULE_FAMILIES if allow_bond_reroute else RULE_FAMILIES
    by_family: dict[str, list[MarkedTransition]] = {
        family: [] for family in rule_families
    }
    current_key = canonical_state_key(state)
    for rule_name, action in _factorized_candidates(
        state,
        allow_bond_reroute=allow_bond_reroute,
    ):
        try:
            successor = runtime.apply(state, rule_name, action)
        except InvalidRewrite as exc:
            raise FactorizedFiberError(
                f"factorized mask admitted invalid {rule_name}: {action!r}"
            ) from exc
        successor_key = canonical_state_key(successor)
        if successor_key == current_key:
            continue
        by_family[rule_name].append(
            MarkedTransition(
                rule_name=rule_name,
                action=action,
                successor=successor,
                successor_key=successor_key,
            )
        )
    return FactorizedFiber(
        by_family={family: tuple(items) for family, items in by_family.items()},
        rule_families=rule_families,
    )


def _factorized_candidates(
    state: MolecularGraph,
    *,
    allow_bond_reroute: bool,
):
    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    null = tuple(int(v) for v in np.flatnonzero(state.atom_types == NULL_IDX))

    if null:
        slot = null[0]
        if not real:
            for atom_type in CNOF_ATOM_TYPES:
                yield "atom_insert", AtomInsert(
                    slot=slot,
                    atom_type=atom_type,
                    formal_charge=0,
                    implicit_h_count=CNOF_VALENCE[atom_type],
                    neighbors=(),
                )
        else:
            for neighbor in real:
                available_h = int(state.implicit_h_counts[neighbor])
                for order in MICRO_BOND_CLASSES:
                    if order > available_h:
                        continue
                    for atom_type in CNOF_ATOM_TYPES:
                        hydrogen_count = CNOF_VALENCE[atom_type] - order
                        if 0 <= hydrogen_count <= MAX_H_COUNT:
                            yield "atom_insert", AtomInsert(
                                slot=slot,
                                atom_type=atom_type,
                                formal_charge=0,
                                implicit_h_count=hydrogen_count,
                                neighbors=((neighbor, order),),
                            )

    for v in real:
        if _connected_after_atom_deletion(state, v) and all(
            int(state.implicit_h_counts[u]) + int(state.bonds[v, u])
            <= MAX_H_COUNT
            for u in np.flatnonzero(state.bonds[v] != 0)
        ):
            yield "atom_delete", AtomDelete(v)

        row_sum = int(state.bonds[v].sum())
        for atom_type in CNOF_ATOM_TYPES:
            hydrogen_count = CNOF_VALENCE[atom_type] - row_sum
            if not 0 <= hydrogen_count <= MAX_H_COUNT:
                continue
            if (
                atom_type == int(state.atom_types[v])
                and hydrogen_count == int(state.implicit_h_counts[v])
            ):
                continue
            yield "atom_restate", AtomRestate(
                v=v,
                atom_type=atom_type,
                formal_charge=0,
                implicit_h_count=hydrogen_count,
            )

    for a, b in combinations(real, 2):
        old_order = int(state.bonds[a, b])
        if old_order == 0:
            max_order = min(
                int(state.implicit_h_counts[a]),
                int(state.implicit_h_counts[b]),
                max(MICRO_BOND_CLASSES),
            )
            for order in MICRO_BOND_CLASSES:
                if order <= max_order:
                    yield "bond_insert", BondInsert(a, b, order)
            continue

        if (
            _connected_after_bond_deletion(state, a, b)
            and int(state.implicit_h_counts[a]) + old_order <= MAX_H_COUNT
            and int(state.implicit_h_counts[b]) + old_order <= MAX_H_COUNT
        ):
            yield "bond_delete", BondDelete(a, b)

        for new_order in MICRO_BOND_CLASSES:
            if new_order == old_order:
                continue
            delta = new_order - old_order
            new_h_a = int(state.implicit_h_counts[a]) - delta
            new_h_b = int(state.implicit_h_counts[b]) - delta
            if 0 <= new_h_a <= MAX_H_COUNT and 0 <= new_h_b <= MAX_H_COUNT:
                yield "bond_reorder", BondReorder(a, b, new_order)

    if allow_bond_reroute:
        for a, b, left, right in _bridge_partitions(state, real):
            old_order = int(state.bonds[a, b])
            for u in left:
                for v in right:
                    if frozenset((u, v)) == frozenset((a, b)):
                        continue
                    if int(state.bonds[u, v]) != 0:
                        continue
                    for new_order in MICRO_BOND_CLASSES:
                        hydrogen_counts = state.implicit_h_counts.copy()
                        hydrogen_counts[a] += old_order
                        hydrogen_counts[b] += old_order
                        hydrogen_counts[u] -= new_order
                        hydrogen_counts[v] -= new_order
                        if np.any(hydrogen_counts < 0) or np.any(
                            hydrogen_counts > MAX_H_COUNT
                        ):
                            continue
                        yield "bond_reroute", BondReroute(
                            a=a,
                            b=b,
                            u=u,
                            v=v,
                            new_order=new_order,
                        )


def _bridge_partitions(
    state: MolecularGraph,
    real: tuple[int, ...],
) -> tuple[tuple[int, int, tuple[int, ...], tuple[int, ...]], ...]:
    vertices = set(real)
    result = []
    for a, b in combinations(real, 2):
        if int(state.bonds[a, b]) == 0:
            continue
        left = _reachable_without_edge(state, a, frozenset((a, b)))
        if b in left:
            continue
        right = vertices - left
        result.append((a, b, tuple(sorted(left)), tuple(sorted(right))))
    return tuple(result)


def _reachable_without_edge(
    state: MolecularGraph,
    root: int,
    forbidden_edge: frozenset[int],
) -> set[int]:
    seen = {int(root)}
    stack = [int(root)]
    while stack:
        vertex = stack.pop()
        for neighbor in np.flatnonzero(state.bonds[vertex] != 0):
            neighbor = int(neighbor)
            if frozenset((vertex, neighbor)) == forbidden_edge:
                continue
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _connected_after_atom_deletion(state: MolecularGraph, removed: int) -> bool:
    remaining = {
        int(v)
        for v in np.flatnonzero(is_element(state.atom_types))
        if int(v) != removed
    }
    return _connected_subset(state, remaining, forbidden_edge=None)


def _connected_after_bond_deletion(
    state: MolecularGraph,
    a: int,
    b: int,
) -> bool:
    remaining = {int(v) for v in np.flatnonzero(is_element(state.atom_types))}
    return _connected_subset(state, remaining, forbidden_edge=frozenset((a, b)))


def _connected_subset(
    state: MolecularGraph,
    vertices: set[int],
    forbidden_edge: frozenset[int] | None,
) -> bool:
    if not vertices:
        return True
    root = next(iter(vertices))
    seen = {root}
    stack = [root]
    while stack:
        v = stack.pop()
        for u in np.flatnonzero(state.bonds[v] != 0):
            u = int(u)
            if u not in vertices:
                continue
            if forbidden_edge is not None and frozenset((v, u)) == forbidden_edge:
                continue
            if u not in seen:
                seen.add(u)
                stack.append(u)
    return seen == vertices


def _assert_supported_state(state: MolecularGraph) -> None:
    if not is_valid_state(state) or not is_connected_or_null(state):
        raise ValueError("factorized fiber requires a valid connected-or-null state")
    real = is_element(state.atom_types)
    if any(int(atom_type) not in CNOF_ATOM_TYPES for atom_type in state.atom_types[real]):
        raise ValueError("factorized fiber supports only C/N/O/F atoms")
    if np.any(state.formal_charges[real] != 0):
        raise ValueError("factorized fiber supports only neutral atom states")
