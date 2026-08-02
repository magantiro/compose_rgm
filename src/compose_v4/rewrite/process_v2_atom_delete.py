"""Production Process-V2 semantics for connected-nonleaf atom deletion.

Process V2 expands the learned marked fiber of ``atom_delete`` beyond the
V1 dense mask, which excluded every cyclic atom before any executor
validation.  The expansion is strictly additive:

* the V1 admission rule for root, singleton, and leaf slots is preserved
  bit-for-bit by its original dense mask and is *not* re-decided here;
* this module decides only the *additional* connected-nonleaf candidates,
  meaning real slots whose real-atom degree is at least two.

The two sets are disjoint by construction.  In a connected real-atom graph an
acyclic vertex of degree at least two is always a cut vertex, so the V1 rule
(``atom_topology == 0`` and not an articulation point) admits nothing of degree
at least two.  ``tests/test_process_v2_atom_delete.py`` pins that disjointness
on a bounded panel rather than assuming it.

An additional candidate is admitted only when every declared condition holds:

1. the slot is a real element under the authoritative element predicate;
2. it is non-aromatic under the frozen production representation
   (:func:`resonance_invariant_bond_classes`), because deleting one Kekule
   slot of a perceived aromatic system yields a representation-sensitive
   open-chain successor;
3. it is not a graph articulation point of the real-atom graph;
4. the unchanged production atom-delete executor accepts the operation;
5. the exact persistent-slot successor is connected;
6. the transition satisfies the frozen charge policy; and
7. the successor is within the declared broad-organic, at-most-40-active-atom
   support and is canonicalizable.

The executor is the legality authority.  This module never re-derives a
weaker approximate valence test: ``is_valid_atom_delete`` and
``apply_atom_delete`` are imported unchanged.

On which conditions actually bind, measured rather than asserted.  Executor
validity is deliberately *not* a connectivity predicate, so it cannot be relied
on for connectivity; but condition 3 already supplies it, because removing a
non-cut vertex from a connected graph leaves it connected.  Condition 5 is
therefore implied by condition 3 and is kept as defence in depth, not as an
independent gate.  Over 3,617 candidate slots across 1,592 states, the gates
that ever rejected an otherwise-admissible candidate were aromaticity,
articulation, and the charge policy (7.6%); successor connectivity, executor
validity, declared support, and canonicalizability rejected nothing.  The last
three are nonetheless reachable on constructed states and are pinned in
``tests/test_process_v2_atom_delete_gates.py``, so none of them is a check no
test can fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import networkx as nx
import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_CLASS_TO_H_CHANGE,
    BOND_NULL,
    ORGANIC_VOCABULARY,
    MolecularGraph,
    is_element,
)
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.rewrite.kernel import InvalidRewrite, canonical_state_key
from compose_v4.rewrite.operators import (
    AtomDelete,
    apply_atom_delete,
    is_valid_atom_delete,
)
from compose_v4.rewrite.trace_shard_v3 import MAX_ACTIVE_ATOMS

# A "leaf" is a real slot of real-atom degree at most one; a root or singleton
# slot has degree zero.  Both stay under the preserved V1 rule.
CONNECTED_NONLEAF_MINIMUM_DEGREE = 2


class ProcessV2AtomDeleteRejectionCode(str, Enum):
    INVALID_SOURCE = "invalid_source"
    INVALID_SLOT = "invalid_slot"
    NOT_A_REAL_ELEMENT = "not_a_real_element"
    OUTSIDE_CONNECTED_NONLEAF_EXPANSION = "outside_connected_nonleaf_expansion"
    AROMATIC_ATOM = "aromatic_atom"
    ARTICULATION_POINT = "articulation_point"
    EXECUTOR_REJECTED = "executor_rejected"
    SUCCESSOR_DISCONNECTED = "successor_disconnected"
    CHARGE_POLICY_VIOLATED = "charge_policy_violated"
    SUCCESSOR_OUTSIDE_SUPPORT = "successor_outside_declared_support"
    SUCCESSOR_NOT_CANONICALIZABLE = "successor_not_canonicalizable"


@dataclass(frozen=True)
class ProcessV2AtomDeleteResolution:
    """One complete, reason-coded connected-nonleaf deletion decision."""

    admitted: bool
    rejection_code: ProcessV2AtomDeleteRejectionCode | None
    slot: int
    real_degree: int
    semantic_aromatic_atom: bool
    articulation_point: bool
    source_key: str | None
    successor: MolecularGraph | None
    successor_key: str | None


def _real_atom_graph(state: MolecularGraph) -> nx.Graph:
    real = tuple(int(slot) for slot in np.flatnonzero(is_element(state.atom_types)))
    graph = nx.Graph()
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (left, right)
        for offset, left in enumerate(real)
        for right in real[offset + 1 :]
        if int(state.bonds[left, right]) != BOND_NULL
    )
    return graph


def _within_declared_support(state: MolecularGraph) -> bool:
    """Declared broad-organic, at-most-40-active-atom representability."""

    real = np.flatnonzero(is_element(state.atom_types))
    if int(real.size) > MAX_ACTIVE_ATOMS:
        return False
    for slot in real:
        bond_order_sum = sum(
            int(BOND_CLASS_TO_H_CHANGE[int(order)]) for order in state.bonds[slot]
        )
        if (
            ORGANIC_VOCABULARY.class_index(
                int(state.atom_types[slot]),
                bond_order_sum,
                int(state.implicit_h_counts[slot]),
                int(state.formal_charges[slot]),
            )
            is None
        ):
            return False
    return True


def _rejected(
    code: ProcessV2AtomDeleteRejectionCode,
    *,
    slot: int,
    source_key: str | None,
    real_degree: int = 0,
    semantic_aromatic_atom: bool = False,
    articulation_point: bool = False,
) -> ProcessV2AtomDeleteResolution:
    return ProcessV2AtomDeleteResolution(
        admitted=False,
        rejection_code=code,
        slot=slot,
        real_degree=real_degree,
        semantic_aromatic_atom=semantic_aromatic_atom,
        articulation_point=articulation_point,
        source_key=source_key,
        successor=None,
        successor_key=None,
    )


def resolve_process_v2_connected_nonleaf_atom_delete(
    state: MolecularGraph,
    action: AtomDelete,
) -> ProcessV2AtomDeleteResolution:
    """Decide one additional connected-nonleaf deletion candidate exactly.

    Root, singleton, and leaf slots are reported as
    ``OUTSIDE_CONNECTED_NONLEAF_EXPANSION``: they are governed by the
    unchanged V1 dense rule, not by this expansion.
    """

    slot = int(action.v)
    if not is_valid_state(state) or not is_connected_or_null(state):
        return _rejected(
            ProcessV2AtomDeleteRejectionCode.INVALID_SOURCE,
            slot=slot,
            source_key=None,
        )
    source_key = canonical_state_key(state)
    if type(action) is not AtomDelete or not 0 <= slot < state.n_atoms:
        return _rejected(
            ProcessV2AtomDeleteRejectionCode.INVALID_SLOT,
            slot=slot,
            source_key=source_key,
        )
    if not bool(is_element(state.atom_types[slot])):
        # Covers null slots and SCAR: occupied is not an element.
        return _rejected(
            ProcessV2AtomDeleteRejectionCode.NOT_A_REAL_ELEMENT,
            slot=slot,
            source_key=source_key,
        )

    graph = _real_atom_graph(state)
    real_degree = int(graph.degree[slot])
    if real_degree < CONNECTED_NONLEAF_MINIMUM_DEGREE:
        return _rejected(
            ProcessV2AtomDeleteRejectionCode.OUTSIDE_CONNECTED_NONLEAF_EXPANSION,
            slot=slot,
            source_key=source_key,
            real_degree=real_degree,
        )

    perceived = resonance_invariant_bond_classes(state)
    semantic_aromatic_atom = bool((perceived[slot] == BOND_AROMATIC).any())
    if semantic_aromatic_atom:
        return _rejected(
            ProcessV2AtomDeleteRejectionCode.AROMATIC_ATOM,
            slot=slot,
            source_key=source_key,
            real_degree=real_degree,
            semantic_aromatic_atom=True,
        )

    articulation_point = slot in set(nx.articulation_points(graph))
    if articulation_point:
        return _rejected(
            ProcessV2AtomDeleteRejectionCode.ARTICULATION_POINT,
            slot=slot,
            source_key=source_key,
            real_degree=real_degree,
            articulation_point=True,
        )

    normalized = AtomDelete(slot)
    if not is_valid_atom_delete(state, normalized):
        return _rejected(
            ProcessV2AtomDeleteRejectionCode.EXECUTOR_REJECTED,
            slot=slot,
            source_key=source_key,
            real_degree=real_degree,
        )
    successor = apply_atom_delete(state, normalized)
    if not is_connected_or_null(successor):
        return _rejected(
            ProcessV2AtomDeleteRejectionCode.SUCCESSOR_DISCONNECTED,
            slot=slot,
            source_key=source_key,
            real_degree=real_degree,
        )
    if not charge_policy_preserved(state, successor):
        return _rejected(
            ProcessV2AtomDeleteRejectionCode.CHARGE_POLICY_VIOLATED,
            slot=slot,
            source_key=source_key,
            real_degree=real_degree,
        )
    if not _within_declared_support(successor):
        return _rejected(
            ProcessV2AtomDeleteRejectionCode.SUCCESSOR_OUTSIDE_SUPPORT,
            slot=slot,
            source_key=source_key,
            real_degree=real_degree,
        )
    try:
        successor_key = canonical_state_key(successor)
    except InvalidRewrite:
        return _rejected(
            ProcessV2AtomDeleteRejectionCode.SUCCESSOR_NOT_CANONICALIZABLE,
            slot=slot,
            source_key=source_key,
            real_degree=real_degree,
        )
    return ProcessV2AtomDeleteResolution(
        admitted=True,
        rejection_code=None,
        slot=slot,
        real_degree=real_degree,
        semantic_aromatic_atom=False,
        articulation_point=False,
        source_key=source_key,
        successor=successor,
        successor_key=successor_key,
    )


def enumerate_process_v2_connected_nonleaf_atom_deletes(
    state: MolecularGraph,
) -> tuple[AtomDelete, ...]:
    """Enumerate the additional Process-V2 deletion fiber in slot order.

    The returned actions are exactly the connected-nonleaf expansion.  The
    preserved V1 root, singleton, and leaf candidates are never repeated here.
    """

    if not is_valid_state(state) or not is_connected_or_null(state):
        return ()
    graph = _real_atom_graph(state)
    articulation = set(nx.articulation_points(graph))
    candidates = tuple(
        int(slot)
        for slot in np.flatnonzero(is_element(state.atom_types))
        if int(graph.degree[int(slot)]) >= CONNECTED_NONLEAF_MINIMUM_DEGREE
        and int(slot) not in articulation
    )
    if not candidates:
        return ()
    perceived = resonance_invariant_bond_classes(state)
    return tuple(
        AtomDelete(slot)
        for slot in candidates
        if not bool((perceived[slot] == BOND_AROMATIC).any())
        and resolve_process_v2_connected_nonleaf_atom_delete(
            state,
            AtomDelete(slot),
        ).admitted
    )


def process_v2_connected_nonleaf_atom_delete_mask(state: MolecularGraph) -> np.ndarray:
    """Return the slot-addressed additional Process-V2 deletion admission."""

    mask = np.zeros(state.n_atoms, dtype=np.bool_)
    for action in enumerate_process_v2_connected_nonleaf_atom_deletes(state):
        mask[int(action.v)] = True
    return mask


__all__ = [
    "CONNECTED_NONLEAF_MINIMUM_DEGREE",
    "ProcessV2AtomDeleteRejectionCode",
    "ProcessV2AtomDeleteResolution",
    "enumerate_process_v2_connected_nonleaf_atom_deletes",
    "process_v2_connected_nonleaf_atom_delete_mask",
    "resolve_process_v2_connected_nonleaf_atom_delete",
]
