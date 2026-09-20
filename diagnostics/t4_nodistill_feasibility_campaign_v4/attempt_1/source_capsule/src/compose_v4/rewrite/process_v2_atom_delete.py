"""Production Process-V2 semantics for ``atom_delete``: one admission authority.

Process V2 decides **every** ``atom_delete`` candidate here.  There is no
disjoint union with an inherited dense rule and no candidate is exempt from the
authoritative predicates.  The recorded prospective decision is:

    Apply the authoritative charge policy to every Process V2 atom-delete
    candidate, including inherited root, singleton, and leaf candidates.
    Preserve the general root/singleton/leaf capabilities only when the
    unchanged executor, connectivity, charge, support, and canonicalization
    predicates all pass.  Exclude newly introduced connected-nonleaf candidates
    incident to a SCAR pending a separate SCAR semantic decision.

"Preserve the capability" means root, singleton, and leaf deletion stays
*reachable*.  It does not mean preserving an unfiltered admission set.  The
first Process-V2 implementation read it the other way, exempted the inherited
slots from the charge policy, and thereby preserved a legacy defect: measured
over the 800 Jin-QED leads at 40 production slots, **125 of 3,319** inherited
root/singleton/leaf candidates (3.77%), spread across **105 of 800** molecules,
changed a protected charged centre and were admitted anyway.  On that same panel
the executor, connectivity, declared-support and canonicalizability predicates
excluded **0** further inherited candidates, so the charge policy was the whole
gap.

# ---- Candidate sources ----

The two structural sources stay distinguishable for diagnostics only.  They
select which *additional* gates apply; they never exempt a slot from a common
gate.

* ``inherited_root_singleton_leaf`` -- a real slot of real-atom degree <= 1.
  Deleting a degree-0 slot of a one-atom state yields the null state, which the
  connectivity predicate accepts; the null state is a legal reversible source
  and is not a molecule.
* ``connected_nonleaf`` -- a real slot of real-atom degree >= 2.  This is the
  fiber Process V2 newly reaches; the V1 dense rule excluded every cyclic atom
  before any executor validation.

# ---- Gate order ----

Common gates, applied to BOTH sources, in this order:

1. the slot is a real element under the authoritative element predicate, so a
   null slot and a SCAR slot are never candidates (SCAR is occupied but is not
   an element);
2. the unchanged production executor accepts (:func:`is_valid_atom_delete`);
3. the exact persistent-slot successor is connected or null;
4. the authoritative charge policy is preserved;
5. the successor is within the declared broad-organic, at-most-40-active-atom
   support;
6. the successor is canonicalizable.

Gates applied to ``connected_nonleaf`` only:

7. non-aromatic under :func:`resonance_invariant_bond_classes`.  Deleting one
   Kekule-encoded slot of a perceived aromatic system yields a
   representation-sensitive open-chain successor, so aromatic connected-nonleaf
   deletion stays excluded pending a separate semantic decision and resolver.
8. not a graph articulation point of the real-atom graph;
9. not incident to any SCAR slot, pending a separate SCAR semantic decision.
   ``apply_atom_delete`` writes implicit hydrogen onto a SCAR neighbour, which
   is unsettled.  This gate is deliberately *not* applied to the inherited
   source: a SCAR-adjacent leaf deletion was already reachable, and the recorded
   decision withdraws no inherited capability.

The executor is the legality authority.  This module never re-derives a weaker
approximate valence test: :func:`is_valid_atom_delete` and
:func:`apply_atom_delete` are imported unchanged.

# ---- Which gates bind, measured rather than asserted ----

The gates are conjunctive, so the order changes only the reported
:class:`ProcessV2AtomDeleteRejectionCode`, never the admission decision.  Two
consequences of the frozen order are recorded here rather than left for a reader
to rediscover:

* ``ARTICULATION_POINT`` is **unreachable on a valid connected source**.  On a
  connected real-atom graph, removing a cut vertex disconnects the graph by
  definition, and :func:`is_connected_or_null` is evaluated over exactly that
  graph, so gate 3 always fires first.  The gate and its ``articulation_point``
  diagnostic field are retained: the field is populated for every real slot
  independently of which gate fired, so an articulation exclusion is still
  observable, and the gate remains correct if connectivity semantics ever move.
* ``EXECUTOR_REJECTED`` and ``SUCCESSOR_OUTSIDE_SUPPORT`` never fire on the
  Jin-QED lead panel, because deleting an atom preserves each surviving
  neighbour's class valence.  They are defence in depth, and each has a
  reachable constructed witness in ``tests/test_process_v2_atom_delete_gates.py``
  (``C1CC[SH4]CC1`` and a 42-membered carbocycle) so that neither is a gate no
  test can fail on.
* ``SUCCESSOR_NOT_CANONICALIZABLE`` is **unreachable behind gate 2**, and has no
  witness.  Gate 2's :func:`is_valid_atom_delete` calls ``is_valid_state`` ->
  ``is_rdkit_valid``, which is true only when ``molecular_graph_to_smiles``
  returns a string; gate 6's :func:`canonical_state_key` raises only when that
  same call returns ``None``, and returns ``"<NULL>"`` for the empty successor.
  So a successor that reached gate 6 always has a canonical key.  The gate is
  retained for the same reason as ``ARTICULATION_POINT`` -- it stays correct if
  either predicate is ever reparameterised -- but this file does not claim it is
  tested, because no test can currently fail on it.

Diagnostic fields (``real_degree``, ``candidate_source``,
``semantic_aromatic_atom``, ``articulation_point``, ``scar_incident``) describe
the slot's structure and are filled whenever they are computable, whatever the
rejection code.  ``successor`` and ``successor_key`` are populated only on
admission, so ``admitted`` is equivalent to ``successor is not None``.
"""

from __future__ import annotations

import numbers
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
    SCAR_IDX,
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
# slot has degree zero.  At or above this degree a slot is connected-nonleaf and
# carries the three additional gates.
CONNECTED_NONLEAF_MINIMUM_DEGREE = 2


class ProcessV2AtomDeleteCandidateSource(str, Enum):
    """Structural classification of a slot, for diagnostics and gate selection.

    It is a pure function of real-atom degree and is never an admission claim.
    A null or SCAR slot has degree zero and therefore reports the inherited
    source, while failing the very first common gate.
    """

    INHERITED_ROOT_SINGLETON_LEAF = "inherited_root_singleton_leaf"
    CONNECTED_NONLEAF = "connected_nonleaf"


class ProcessV2AtomDeleteRejectionCode(str, Enum):
    INVALID_SOURCE = "invalid_source"
    INVALID_SLOT = "invalid_slot"
    NOT_A_REAL_ELEMENT = "not_a_real_element"
    AROMATIC_ATOM = "aromatic_atom"
    ARTICULATION_POINT = "articulation_point"
    SCAR_INCIDENT = "scar_incident"
    EXECUTOR_REJECTED = "executor_rejected"
    SUCCESSOR_DISCONNECTED = "successor_disconnected"
    CHARGE_POLICY_VIOLATED = "charge_policy_violated"
    SUCCESSOR_OUTSIDE_SUPPORT = "successor_outside_declared_support"
    SUCCESSOR_NOT_CANONICALIZABLE = "successor_not_canonicalizable"


@dataclass(frozen=True)
class ProcessV2AtomDeleteResolution:
    """One complete, reason-coded Process-V2 atom-deletion decision."""

    admitted: bool
    rejection_code: ProcessV2AtomDeleteRejectionCode | None
    slot: int
    real_degree: int
    candidate_source: ProcessV2AtomDeleteCandidateSource
    semantic_aromatic_atom: bool
    articulation_point: bool
    scar_incident: bool
    source_key: str | None
    successor: MolecularGraph | None
    successor_key: str | None


def _candidate_source(real_degree: int) -> ProcessV2AtomDeleteCandidateSource:
    if real_degree >= CONNECTED_NONLEAF_MINIMUM_DEGREE:
        return ProcessV2AtomDeleteCandidateSource.CONNECTED_NONLEAF
    return ProcessV2AtomDeleteCandidateSource.INHERITED_ROOT_SINGLETON_LEAF


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


@dataclass(frozen=True, eq=False)
class _StateContext:
    """Per-state work hoisted out of the per-slot gate chain.

    The three source-specific gates are state-level computations (an RDKit
    perception pass, one articulation-point pass, one SCAR adjacency scan), so
    resolving every slot of a state recomputes them once instead of once per
    slot.  Nothing here decides anything; it only supplies the gate inputs.

    ``eq=False`` because the fields are numpy arrays and a graph: identity
    comparison is the only meaningful one, and a generated ``__eq__`` would
    raise on an ambiguous array truth value.
    """

    graph: nx.Graph
    articulation: frozenset[int]
    aromatic_atom: np.ndarray
    scar_incident: np.ndarray
    source_key: str

    def real_degree(self, slot: int) -> int:
        return int(self.graph.degree[slot]) if slot in self.graph else 0


def _state_context(state: MolecularGraph) -> _StateContext:
    graph = _real_atom_graph(state)
    aromatic_atom = np.zeros(state.n_atoms, dtype=np.bool_)
    articulation: frozenset[int] = frozenset()
    connected_nonleaf = tuple(
        int(slot)
        for slot in graph.nodes
        if int(graph.degree[slot]) >= CONNECTED_NONLEAF_MINIMUM_DEGREE
    )
    if connected_nonleaf:
        # Only the connected-nonleaf source consults these, so a leaf-only or
        # singleton state skips the RDKit perception pass entirely.
        perceived = resonance_invariant_bond_classes(state)
        for slot in connected_nonleaf:
            aromatic_atom[slot] = bool((perceived[slot] == BOND_AROMATIC).any())
        articulation = frozenset(int(slot) for slot in nx.articulation_points(graph))
    scar = state.atom_types == SCAR_IDX
    scar_incident = ((state.bonds != BOND_NULL) & scar[None, :]).any(axis=1) & is_element(
        state.atom_types
    )
    return _StateContext(
        graph=graph,
        articulation=articulation,
        aromatic_atom=aromatic_atom,
        scar_incident=scar_incident,
        source_key=canonical_state_key(state),
    )


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


def _resolved(
    code: ProcessV2AtomDeleteRejectionCode | None,
    *,
    slot: int,
    context: _StateContext | None,
    source_key: str | None = None,
    real_degree: int = 0,
    successor: MolecularGraph | None = None,
    successor_key: str | None = None,
) -> ProcessV2AtomDeleteResolution:
    """Build one resolution, filling every computable diagnostic field.

    ``context`` is ``None`` exactly when the slot cannot index the state-level
    diagnostic arrays, which is the invalid-source and out-of-range-slot case.
    """

    return ProcessV2AtomDeleteResolution(
        admitted=code is None,
        rejection_code=code,
        slot=slot,
        real_degree=real_degree,
        candidate_source=_candidate_source(real_degree),
        semantic_aromatic_atom=(
            False if context is None else bool(context.aromatic_atom[slot])
        ),
        articulation_point=(False if context is None else slot in context.articulation),
        scar_incident=(False if context is None else bool(context.scar_incident[slot])),
        source_key=source_key if context is None else context.source_key,
        successor=successor,
        successor_key=successor_key,
    )


def _resolve_in_context(
    state: MolecularGraph,
    slot: int,
    context: _StateContext,
) -> ProcessV2AtomDeleteResolution:
    """Run the frozen gate chain for one slot of an already validated state."""

    if not bool(is_element(state.atom_types[slot])):
        # Gate 1.  Covers null slots and SCAR: occupied is not an element.
        return _resolved(
            ProcessV2AtomDeleteRejectionCode.NOT_A_REAL_ELEMENT,
            slot=slot,
            context=context,
        )
    real_degree = context.real_degree(slot)
    source = _candidate_source(real_degree)
    action = AtomDelete(slot)

    def fail(
        code: ProcessV2AtomDeleteRejectionCode,
    ) -> ProcessV2AtomDeleteResolution:
        return _resolved(code, slot=slot, context=context, real_degree=real_degree)

    if not is_valid_atom_delete(state, action):  # Gate 2.
        return fail(ProcessV2AtomDeleteRejectionCode.EXECUTOR_REJECTED)
    successor = apply_atom_delete(state, action)
    if not is_connected_or_null(successor):  # Gate 3.
        return fail(ProcessV2AtomDeleteRejectionCode.SUCCESSOR_DISCONNECTED)
    if not charge_policy_preserved(state, successor):  # Gate 4.
        return fail(ProcessV2AtomDeleteRejectionCode.CHARGE_POLICY_VIOLATED)
    if not _within_declared_support(successor):  # Gate 5.
        return fail(ProcessV2AtomDeleteRejectionCode.SUCCESSOR_OUTSIDE_SUPPORT)
    try:  # Gate 6.
        successor_key = canonical_state_key(successor)
    except InvalidRewrite:
        return fail(ProcessV2AtomDeleteRejectionCode.SUCCESSOR_NOT_CANONICALIZABLE)

    if source is ProcessV2AtomDeleteCandidateSource.CONNECTED_NONLEAF:
        if bool(context.aromatic_atom[slot]):  # Gate 7.
            return fail(ProcessV2AtomDeleteRejectionCode.AROMATIC_ATOM)
        if slot in context.articulation:  # Gate 8.
            return fail(ProcessV2AtomDeleteRejectionCode.ARTICULATION_POINT)
        if bool(context.scar_incident[slot]):  # Gate 9.
            return fail(ProcessV2AtomDeleteRejectionCode.SCAR_INCIDENT)

    return _resolved(
        None,
        slot=slot,
        context=context,
        real_degree=real_degree,
        successor=successor,
        successor_key=successor_key,
    )


def resolve_process_v2_atom_delete(
    state: MolecularGraph,
    action: AtomDelete,
) -> ProcessV2AtomDeleteResolution:
    """Decide one Process-V2 atom-deletion candidate exactly.

    Every candidate is decided here, whatever its structural source.  Root,
    singleton, and leaf slots are gated by the same executor, connectivity,
    charge, support, and canonicalization predicates as connected-nonleaf slots.
    """

    # The action type is settled BEFORE its payload is read.  `int(action.v)`
    # coerces, so a float or bool slot used to be silently truncated to a
    # neighbouring integer and then admitted, and this function is the single
    # admission authority: it must decide the action it was given, not a
    # coerced neighbour of it.  `numbers.Integral` rather than `int` because a
    # slot legitimately arrives as `np.int64` from `np.flatnonzero`; `bool` is
    # excluded explicitly, since it IS integral and `AtomDelete(True)` is a
    # caller error rather than a request to delete slot 1.
    if (
        type(action) is not AtomDelete
        or isinstance(action.v, bool)
        or not isinstance(action.v, numbers.Integral)
    ):
        return _resolved(
            ProcessV2AtomDeleteRejectionCode.INVALID_SOURCE
            if not is_valid_state(state) or not is_connected_or_null(state)
            else ProcessV2AtomDeleteRejectionCode.INVALID_SLOT,
            slot=-1,
            context=None,
        )
    slot = int(action.v)
    if not is_valid_state(state) or not is_connected_or_null(state):
        return _resolved(
            ProcessV2AtomDeleteRejectionCode.INVALID_SOURCE,
            slot=slot,
            context=None,
        )
    context = _state_context(state)
    if not 0 <= slot < state.n_atoms:
        # The slot cannot index the state-level diagnostic arrays, so only the
        # source key is reportable.
        return _resolved(
            ProcessV2AtomDeleteRejectionCode.INVALID_SLOT,
            slot=slot,
            context=None,
            source_key=context.source_key,
        )
    return _resolve_in_context(state, slot, context)


def enumerate_process_v2_atom_deletes(
    state: MolecularGraph,
) -> tuple[AtomDelete, ...]:
    """Enumerate the complete Process-V2 deletion fiber in slot order."""

    if not is_valid_state(state) or not is_connected_or_null(state):
        return ()
    context = _state_context(state)
    return tuple(
        AtomDelete(int(slot))
        for slot in np.flatnonzero(is_element(state.atom_types))
        if _resolve_in_context(state, int(slot), context).admitted
    )


def process_v2_atom_delete_mask(state: MolecularGraph) -> np.ndarray:
    """Return THE effective slot-addressed Process-V2 ``atom_delete`` mask.

    This is the complete effective mask, not an extension of another rule.
    """

    mask = np.zeros(state.n_atoms, dtype=np.bool_)
    for action in enumerate_process_v2_atom_deletes(state):
        mask[int(action.v)] = True
    return mask


__all__ = [
    "CONNECTED_NONLEAF_MINIMUM_DEGREE",
    "ProcessV2AtomDeleteCandidateSource",
    "ProcessV2AtomDeleteRejectionCode",
    "ProcessV2AtomDeleteResolution",
    "enumerate_process_v2_atom_deletes",
    "process_v2_atom_delete_mask",
    "resolve_process_v2_atom_delete",
]
