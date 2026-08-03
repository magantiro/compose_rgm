"""The pure Editing-V2 Active8 action classifier and its complete helper closure.

This is a LEAF.  It imports chemistry and the Action-V4 codec and nothing else --
no corpus contract, no sampling sidecar, no admission policy, no decision index,
no packed store, no process identity.  That is the whole point of the module: the
classifier is the one part of the capability-cell system a Process-V2 consumer
needs in order to name a transition, and it must be importable without acquiring
the registry's dependency cone.  An import added here is an import every
Process-V2 consumer acquires.

WHAT LIVES HERE
---------------
* :class:`SemanticCapabilityCellError` -- the single error type the whole
  capability-cell system raises.  It is defined here, not re-created, because two
  class objects with the same name are not catchable by each other; the registry
  module re-exports THIS object.
* :data:`_KNOWN_CONTEXTS` -- the registered ``family -> contexts`` closure.  The
  registry validates its frozen JSON against this mapping, and
  :func:`classify_action_family_context` refuses anything outside it.
* :class:`CountBin` and :func:`_stratum` -- prospective integer-count strata.
* the pure graph helpers the classifier needs, all of which read a
  :class:`~compose_v4.chem.molecular_graph.MolecularGraph` and compute, never
  load, hash, or authorize anything.

DATA STRUCTURE
--------------
Every function here reads exact padded persistent-slot states.  ``_state_graph``
projects one to the simple graph over its REAL element slots, which is the only
graph notion the contexts are defined over: null and scar slots are not vertices,
and slot indices are stable node labels, so a classification never depends on
canonical SMILES atom order.

INVARIANTS MAINTAINED (and tested)
----------------------------------
* the classifier is a pure function of ``(source, successor, step)``: it reads no
  file, consults no registry, and holds no state;
* every returned ``(family, context)`` pair is registered in
  :data:`_KNOWN_CONTEXTS`; an unregistered pair raises rather than being returned;
* every refusal is a :class:`SemanticCapabilityCellError`, so a caller catching
  that one type catches every classification failure;
* nothing here authorizes anything: no Gate 0, no T1, no P50, no training.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_NULL,
    IDX_TO_ELEMENT,
    ORGANIC_VOCABULARY,
    MolecularGraph,
    is_element,
)
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    BondReorder,
    BondReroute,
    CycleCloseEdge,
    CycleOpenEdge,
    SemanticAtomRestate,
)
from compose_v4.rewrite.ring_restate_semantics import (
    ring_restate_semantic_transition_descriptor,
)
from compose_v4.rewrite.trace import RewriteStep
from compose_v4.rewrite.tracelets import RingSystemRestate

# ---- Errors ----


class SemanticCapabilityCellError(ValueError):
    """Registry or row evidence cannot support an authoritative classification."""


# ---- Registered family contexts ----

_KNOWN_CONTEXTS = {
    "atom_insert": ("root_birth", "one_neighbor_birth"),
    "atom_delete": (
        "singleton_to_null_death",
        "leaf_death",
        "connected_nonleaf_death",
    ),
    "atom_restate": ("element_identity_change", "valence_state_change"),
    "bond_reorder": ("bond_order_increase", "bond_order_decrease"),
    "bond_reroute": (
        "single_atom_pendant_acyclic_source",
        "multi_atom_pendant_acyclic_source",
        "single_atom_pendant_cyclic_source",
        "multi_atom_pendant_cyclic_source",
    ),
    "cycle_insert": (
        "close_to_monocyclic_ring_system",
        "close_to_articulated_polycyclic_ring_system",
        "close_to_nonarticulated_polycyclic_ring_system",
    ),
    "cycle_attach": (
        "open_from_monocyclic_ring_system",
        "open_from_articulated_polycyclic_ring_system",
        "open_from_nonarticulated_polycyclic_ring_system",
    ),
    "ring_system_restate": (
        "aromatization",
        "dearomatization",
        "coordinated_ring_bond_state_change",
    ),
}


# ---- Prospective integer-count strata ----


@dataclass(frozen=True, slots=True)
class CountBin:
    """One prospective integer-count stratum."""

    id: str
    minimum: int
    maximum: int | None

    def contains(self, value: int) -> bool:
        return value >= self.minimum and (self.maximum is None or value <= self.maximum)


def _stratum(value: int, bins: tuple[CountBin, ...], *, field: str) -> str:
    if type(value) is not int or value < 1:
        raise SemanticCapabilityCellError(f"{field} must be a positive integer")
    matched = tuple(item.id for item in bins if item.contains(value))
    if len(matched) != 1:
        raise SemanticCapabilityCellError(
            f"{field} does not resolve to exactly one stratum"
        )
    return matched[0]


# ---- Pure state helpers ----


def _state_graph(state: MolecularGraph) -> nx.Graph:
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


def _cycle_rank(state: MolecularGraph) -> int:
    graph = _state_graph(state)
    if not graph:
        return 0
    return int(
        graph.number_of_edges()
        - graph.number_of_nodes()
        + nx.number_connected_components(graph)
    )


def _real_atom_count(state: MolecularGraph) -> int:
    return int(np.count_nonzero(is_element(state.atom_types)))


def _minimum_cycle_length_for_edge(
    state: MolecularGraph,
    *,
    left: int,
    right: int,
) -> int:
    graph = _state_graph(state)
    if not graph.has_edge(left, right):
        raise SemanticCapabilityCellError("cycle audit edge is absent")
    graph.remove_edge(left, right)
    try:
        alternate = int(nx.shortest_path_length(graph, left, right))
    except (nx.NetworkXNoPath, nx.NodeNotFound) as error:
        raise SemanticCapabilityCellError(
            "cycle audit edge has no alternate path"
        ) from error
    length = alternate + 1
    if length < 3:
        raise SemanticCapabilityCellError("edited cycle length is smaller than three")
    return length


def _action_audit_axes(
    source: MolecularGraph,
    successor: MolecularGraph,
    step: RewriteStep,
    *,
    family: str,
) -> tuple[tuple[tuple[str, str], ...], int | None, bool | None, bool | None]:
    action = step.action
    element_transition: tuple[tuple[str, str], ...] = ()
    cycle_length: int | None = None
    source_aromatic: bool | None = None
    successor_aromatic: bool | None = None
    if family == "atom_insert" and isinstance(action, AtomInsert):
        slot = int(action.slot)
        if (
            is_element(source.atom_types)[slot]
            or not is_element(successor.atom_types)[slot]
        ):
            raise SemanticCapabilityCellError("atom insertion element audit disagrees")
        element_transition = (
            ("source_element", "ABSENT"),
            ("successor_element", IDX_TO_ELEMENT[int(successor.atom_types[slot])]),
        )
    elif family == "atom_delete" and isinstance(action, AtomDelete):
        slot = int(action.v)
        if (
            not is_element(source.atom_types)[slot]
            or is_element(successor.atom_types)[slot]
        ):
            raise SemanticCapabilityCellError("atom deletion element audit disagrees")
        element_transition = (
            ("source_element", IDX_TO_ELEMENT[int(source.atom_types[slot])]),
            ("successor_element", "ABSENT"),
        )
    elif family == "atom_restate" and isinstance(action, SemanticAtomRestate):
        slot = int(action.v)
        if (
            not is_element(source.atom_types)[slot]
            or not is_element(successor.atom_types)[slot]
        ):
            raise SemanticCapabilityCellError(
                "atom restatement element audit disagrees"
            )
        element_transition = (
            ("source_element", IDX_TO_ELEMENT[int(source.atom_types[slot])]),
            ("successor_element", IDX_TO_ELEMENT[int(successor.atom_types[slot])]),
        )
    elif family in {"cycle_insert", "cycle_attach"} and isinstance(
        action,
        (CycleCloseEdge, CycleOpenEdge),
    ):
        left, right = int(action.a), int(action.b)
        selected = successor if family == "cycle_insert" else source
        cycle_length = _minimum_cycle_length_for_edge(
            selected,
            left=left,
            right=right,
        )
        source_has_edge = int(source.bonds[left, right]) != BOND_NULL
        successor_has_edge = int(successor.bonds[left, right]) != BOND_NULL
        source_aromatic = (
            int(source.bonds[left, right]) == BOND_AROMATIC if source_has_edge else None
        )
        successor_aromatic = (
            int(successor.bonds[left, right]) == BOND_AROMATIC
            if successor_has_edge
            else None
        )
    return element_transition, cycle_length, source_aromatic, successor_aromatic


def _ring_system_topology(
    state: MolecularGraph,
    *,
    left: int,
    right: int,
) -> str:
    graph = _state_graph(state)
    edge = frozenset((int(left), int(right)))
    if edge not in {frozenset((int(a), int(b))) for a, b in graph.edges()}:
        raise SemanticCapabilityCellError(
            "cycle topology edge is absent from the selected state"
        )
    bridges = {frozenset((int(a), int(b))) for a, b in nx.bridges(graph)}
    if edge in bridges:
        raise SemanticCapabilityCellError("cycle topology edge is a bridge")
    ring_graph = graph.copy()
    ring_graph.remove_edges_from(tuple(tuple(item) for item in bridges))
    component = next(
        (
            frozenset(int(vertex) for vertex in vertices)
            for vertices in nx.connected_components(ring_graph)
            if int(left) in vertices and int(right) in vertices
        ),
        None,
    )
    if component is None:
        raise SemanticCapabilityCellError("cycle topology edge has no cyclic component")
    subgraph = ring_graph.subgraph(component).copy()
    rank = int(subgraph.number_of_edges() - subgraph.number_of_nodes() + 1)
    if rank == 1:
        return "monocyclic_ring_system"
    if rank < 1:
        raise SemanticCapabilityCellError(
            "cycle topology component has zero graph cycle rank"
        )
    if tuple(nx.articulation_points(subgraph)):
        return "articulated_polycyclic_ring_system"
    return "nonarticulated_polycyclic_ring_system"


def _component_after_cut(
    state: MolecularGraph,
    *,
    root: int,
    left: int,
    right: int,
) -> frozenset[int]:
    graph = _state_graph(state)
    if not graph.has_edge(left, right):
        raise SemanticCapabilityCellError("graft cut edge is absent")
    graph.remove_edge(left, right)
    component = frozenset(
        int(vertex) for vertex in nx.node_connected_component(graph, root)
    )
    if int(right) in component:
        raise SemanticCapabilityCellError("graft cut edge is not a bridge")
    induced = graph.subgraph(component)
    if induced.number_of_edges() != len(component) - 1:
        raise SemanticCapabilityCellError("graft moved component is not a pendant tree")
    return component


# ---- Classification ----


def classify_action_family_context(
    source: MolecularGraph,
    successor: MolecularGraph,
    step: RewriteStep,
) -> tuple[str, str]:
    """Classify one exact Action-V4 transition into a compact family context."""

    try:
        family = action_codec_v4.canonical_family(str(step.rule_name))
    except action_codec_v4.ActionCodecV4Error as error:
        raise SemanticCapabilityCellError(
            "teacher action is outside Action V4"
        ) from error
    action = step.action
    context: str
    if family == "atom_insert":
        if type(action) is not AtomInsert:
            raise SemanticCapabilityCellError("atom_insert payload has another type")
        neighbor_count = len(tuple(action.neighbors))
        if neighbor_count == 0:
            context = "root_birth"
        elif neighbor_count == 1:
            context = "one_neighbor_birth"
        else:
            raise SemanticCapabilityCellError(
                "multi-neighbor birth is outside Editing V2"
            )
    elif family == "atom_delete":
        if type(action) is not AtomDelete:
            raise SemanticCapabilityCellError("atom_delete payload has another type")
        source_graph = _state_graph(source)
        real_count = source_graph.number_of_nodes()
        if int(action.v) not in source_graph:
            raise SemanticCapabilityCellError(
                "atom_delete addresses a non-element slot"
            )
        degree = int(source_graph.degree[int(action.v)])
        if real_count == 1 and degree == 0:
            context = "singleton_to_null_death"
        elif degree == 1:
            context = "leaf_death"
        elif degree >= 2:
            context = "connected_nonleaf_death"
        else:
            raise SemanticCapabilityCellError(
                "non-singleton isolated atom death is invalid"
            )
    elif family == "atom_restate":
        if type(action) is not SemanticAtomRestate:
            raise SemanticCapabilityCellError("atom_restate payload has another type")
        source_element = int(source.atom_types[int(action.v)])
        target_element = int(
            ORGANIC_VOCABULARY.element_of(int(action.target_class_index))
        )
        context = (
            "valence_state_change"
            if source_element == target_element
            else "element_identity_change"
        )
    elif family == "bond_reorder":
        if type(action) is not BondReorder:
            raise SemanticCapabilityCellError("bond_reorder payload has another type")
        old_order = int(source.bonds[int(action.a), int(action.b)])
        new_order = int(action.new_order)
        if new_order > old_order:
            context = "bond_order_increase"
        elif new_order < old_order:
            context = "bond_order_decrease"
        else:
            raise SemanticCapabilityCellError("bond_reorder teacher is not productive")
    elif family == "bond_reroute":
        if type(action) is not BondReroute or int(action.u) != int(action.a):
            raise SemanticCapabilityCellError(
                "Editing-V2 graft context requires the production pendant-tree orientation"
            )
        pendant = _component_after_cut(
            source,
            root=int(action.a),
            left=int(action.a),
            right=int(action.b),
        )
        size = "single_atom" if len(pendant) == 1 else "multi_atom"
        source_kind = "cyclic_source" if _cycle_rank(source) > 0 else "acyclic_source"
        context = f"{size}_pendant_{source_kind}"
    elif family == "cycle_insert":
        if type(action) is not CycleCloseEdge:
            raise SemanticCapabilityCellError("cycle_insert payload has another type")
        topology = _ring_system_topology(
            successor,
            left=int(action.a),
            right=int(action.b),
        )
        context = f"close_to_{topology}"
    elif family == "cycle_attach":
        if type(action) is not CycleOpenEdge:
            raise SemanticCapabilityCellError("cycle_attach payload has another type")
        topology = _ring_system_topology(
            source,
            left=int(action.a),
            right=int(action.b),
        )
        context = f"open_from_{topology}"
    elif family == "ring_system_restate":
        if type(action) is not RingSystemRestate:
            raise SemanticCapabilityCellError(
                "ring_system_restate payload has another type"
            )
        descriptor = ring_restate_semantic_transition_descriptor(source, successor)
        if not descriptor:
            raise SemanticCapabilityCellError(
                "ring restatement has no semantic bond change"
            )
        into_aromatic = tuple(
            old != BOND_AROMATIC and new == BOND_AROMATIC
            for _, _, old, new in descriptor
        )
        out_of_aromatic = tuple(
            old == BOND_AROMATIC and new != BOND_AROMATIC
            for _, _, old, new in descriptor
        )
        if all(into_aromatic):
            context = "aromatization"
        elif all(out_of_aromatic):
            context = "dearomatization"
        else:
            context = "coordinated_ring_bond_state_change"
    else:
        raise SemanticCapabilityCellError(f"unknown Active8 family: {family}")
    allowed = _KNOWN_CONTEXTS.get(family, ())
    if context not in allowed:
        raise SemanticCapabilityCellError(
            f"unregistered family context {family}:{context}"
        )
    return family, context
