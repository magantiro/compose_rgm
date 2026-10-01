"""Finite, exact marked-action fibers for the first learning gate."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

import numpy as np

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    MolecularGraph,
    NULL_IDX,
    is_element,
)
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


@dataclass(frozen=True, order=True)
class AtomState:
    atom_type: int
    formal_charge: int
    implicit_h_count: int


@dataclass(frozen=True)
class MarkedTransition:
    rule_name: str
    action: Any
    successor: MolecularGraph
    successor_key: str


@dataclass(frozen=True)
class ActionFiberSpec:
    """Declared finite chemistry/action language for an exact action fiber."""

    atom_states: tuple[AtomState, ...]
    allow_atom_insert: bool = True
    allow_atom_delete: bool = True
    allow_atom_restate: bool = True
    allow_bond_insert: bool = True
    allow_bond_delete: bool = True
    allow_bond_reorder: bool = True
    allow_bond_reroute: bool = False

    @classmethod
    def neutral_cnof(cls) -> "ActionFiberSpec":
        atom_types = tuple(ELEMENT_TO_IDX[symbol] for symbol in ("C", "N", "O", "F"))
        return cls(
            atom_states=tuple(
                AtomState(atom_type, 0, hydrogen_count)
                for atom_type in atom_types
                for hydrogen_count in range(5)
            )
        )


def enumerate_action_fiber(
    state: MolecularGraph,
    *,
    spec: ActionFiberSpec | None = None,
    system: RewriteSystem | None = None,
) -> tuple[MarkedTransition, ...]:
    """Enumerate every valid marked action in the declared base language.

    Atom insertions attach a single new atom to the connected current molecule;
    non-tree connections are represented by bond insertion. Every candidate is
    passed through the production runtime, and identity transitions are
    removed. Distinct padding-slot marks are retained so successor-level rate
    aggregation can be tested rather than assumed.
    """

    language = spec or ActionFiberSpec.neutral_cnof()
    runtime = system or de_novo_rewrite_system()
    current_key = canonical_state_key(state)
    candidates = _candidate_actions(state, language)
    transitions: list[MarkedTransition] = []
    for rule_name, action in candidates:
        try:
            successor = runtime.apply(state, rule_name, action)
        except InvalidRewrite:
            continue
        key = canonical_state_key(successor)
        if key == current_key:
            continue
        transitions.append(MarkedTransition(rule_name, action, successor, key))
    return tuple(transitions)


def group_transition_indices_by_successor(
    transitions: Iterable[MarkedTransition],
) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {}
    for index, transition in enumerate(transitions):
        grouped.setdefault(transition.successor_key, []).append(index)
    return {key: tuple(indices) for key, indices in grouped.items()}


def _candidate_actions(
    state: MolecularGraph,
    spec: ActionFiberSpec,
) -> Iterable[tuple[str, Any]]:
    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    null = tuple(int(v) for v in np.flatnonzero(state.atom_types == NULL_IDX))

    if spec.allow_atom_insert and null:
        if not real:
            for slot in null:
                for atom_state in spec.atom_states:
                    yield "atom_insert", _atom_insert(slot, atom_state, ())
        else:
            for slot in null:
                for neighbor in real:
                    for order in MICRO_BOND_CLASSES:
                        for atom_state in spec.atom_states:
                            yield "atom_insert", _atom_insert(
                                slot,
                                atom_state,
                                ((neighbor, order),),
                            )

    if spec.allow_atom_delete:
        for v in real:
            yield "atom_delete", AtomDelete(v)

    if spec.allow_atom_restate:
        for v in real:
            for atom_state in spec.atom_states:
                yield "atom_restate", AtomRestate(
                    v,
                    atom_state.atom_type,
                    atom_state.formal_charge,
                    atom_state.implicit_h_count,
                )

    for a, b in combinations(real, 2):
        old_order = int(state.bonds[a, b])
        if old_order == 0 and spec.allow_bond_insert:
            for order in MICRO_BOND_CLASSES:
                yield "bond_insert", BondInsert(a, b, order)
        elif old_order != 0:
            if spec.allow_bond_delete:
                yield "bond_delete", BondDelete(a, b)
            if spec.allow_bond_reorder:
                for order in MICRO_BOND_CLASSES:
                    if order != old_order:
                        yield "bond_reorder", BondReorder(a, b, order)

    if spec.allow_bond_reroute:
        for a, b in combinations(real, 2):
            if int(state.bonds[a, b]) == 0:
                continue
            for u, v in combinations(real, 2):
                if frozenset((a, b)) == frozenset((u, v)):
                    continue
                if int(state.bonds[u, v]) != 0:
                    continue
                for order in MICRO_BOND_CLASSES:
                    yield "bond_reroute", BondReroute(a, b, u, v, order)


def _atom_insert(
    slot: int,
    atom_state: AtomState,
    neighbors: tuple[tuple[int, int], ...],
) -> AtomInsert:
    return AtomInsert(
        slot=slot,
        atom_type=atom_state.atom_type,
        formal_charge=atom_state.formal_charge,
        implicit_h_count=atom_state.implicit_h_count,
        neighbors=neighbors,
    )
