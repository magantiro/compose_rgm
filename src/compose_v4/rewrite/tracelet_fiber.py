"""Exact finite C/N/O/F fiber augmented with coordinated ring tracelets."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np

from compose_v4.chem.aromaticity import perceived_aromatic_ring_count
from compose_v4.chem.graph_primitives import compute_topology_features
from compose_v4.chem.molecular_graph import (
    BOND_CLASS_TO_H_CHANGE,
    ELEMENT_TO_IDX,
    MolecularGraph,
    NULL_IDX,
    is_element,
)
from compose_v4.rewrite.factorized_fiber import (
    RULE_FAMILIES as MICRO_RULE_FAMILIES,
    enumerate_factorized_cnof_fiber,
)
from compose_v4.rewrite.fiber import MarkedTransition
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    RewriteSystem,
    canonical_state_key,
    de_novo_rewrite_system,
)
from compose_v4.rewrite.tracelets import (
    AtomPayload,
    BondOrderChange,
    CycleAttach,
    CycleInsert,
    RingEarInsert,
    RingSystemRestate,
)
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomRestate,
    BondDelete,
    BondInsert,
    BondReorder,
)
from compose_v4.rewrite.typed_ring_catalog import (
    TypedRingCatalog,
    EarTemplate,
    oriented_cycle_templates,
    oriented_ear_templates,
)

TRACELET_RULE_FAMILIES = (
    *MICRO_RULE_FAMILIES,
    "cycle_insert",
    "cycle_attach",
    "ring_ear_insert",
    "ring_system_restate",
)
TRACELET_TRANSPORT_RULE_FAMILIES = (*TRACELET_RULE_FAMILIES, "bond_reroute")


@dataclass(frozen=True)
class TraceletFiber:
    by_family: dict[str, tuple[MarkedTransition, ...]]
    rule_families: tuple[str, ...] = TRACELET_RULE_FAMILIES

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


def enumerate_tracelet_cnof_fiber(
    state: MolecularGraph,
    *,
    system: RewriteSystem | None = None,
    ring_catalog: TypedRingCatalog | None = None,
    allow_bond_reroute: bool = False,
) -> TraceletFiber:
    """Enumerate micro actions plus generic carbon ring-scaffold tracelets.

    Padding aliases are quotiented by using the first ``k`` null slots. In
    typed mode, cyclic topology is committed before local decoration: a
    root cycle, a completely closed attached cycle, or an ear whose anchors
    already belong to the same cyclic block.
    """

    runtime = system or de_novo_rewrite_system()
    micro_fiber = enumerate_factorized_cnof_fiber(
        state,
        system=runtime,
        allow_bond_reroute=allow_bond_reroute,
    )
    micro_by_family = micro_fiber.by_family
    if ring_catalog is not None:
        micro_by_family = _block_scoped_micro_fiber(state, micro_by_family)
    rule_families = (
        TRACELET_TRANSPORT_RULE_FAMILIES
        if allow_bond_reroute
        else TRACELET_RULE_FAMILIES
    )
    by_family: dict[str, list[MarkedTransition]] = {
        family: list(micro_by_family.get(family, ())) for family in rule_families
    }
    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    null = tuple(int(v) for v in np.flatnonzero(state.atom_types == NULL_IDX))
    carbon = int(ELEMENT_TO_IDX["C"])

    if ring_catalog is not None:
        _append_typed_catalog_actions(
            by_family,
            state,
            real=real,
            null=null,
            catalog=ring_catalog,
        )
    elif not real:
        for length in range(3, len(null) + 1):
            action = CycleInsert(
                atoms=tuple(AtomPayload(slot, carbon, 0, 2) for slot in null[:length]),
                bond_orders=(1,) * length,
            )
            _append_generic_cycle(by_family, state, action)
    elif null:
        for anchor in real:
            if int(state.implicit_h_counts[anchor]) < 2:
                continue
            for span in range(2, len(null) + 1):
                action = RingEarInsert(
                    a=anchor,
                    b=anchor,
                    atoms=tuple(
                        AtomPayload(slot, carbon, 0, 2) for slot in null[:span]
                    ),
                    bond_orders=(1,) * (span + 1),
                )
                _append_generic_ear(by_family, state, action)
        for a, b in combinations(real, 2):
            if (
                min(
                    int(state.implicit_h_counts[a]),
                    int(state.implicit_h_counts[b]),
                )
                < 1
            ):
                continue
            for span in range(1, len(null) + 1):
                action = RingEarInsert(
                    a=a,
                    b=b,
                    atoms=tuple(
                        AtomPayload(slot, carbon, 0, 2) for slot in null[:span]
                    ),
                    bond_orders=(1,) * (span + 1),
                )
                _append_generic_ear(by_family, state, action)

    by_family["ring_system_restate"].extend(
        enumerate_aromatic_closure_transitions(state, system=runtime)
    )
    return TraceletFiber(
        by_family={family: tuple(items) for family, items in by_family.items()},
        rule_families=rule_families,
    )


def _block_scoped_micro_fiber(
    state: MolecularGraph,
    by_family: dict[str, tuple[MarkedTransition, ...]],
) -> dict[str, tuple[MarkedTransition, ...]]:
    """Protect cyclic-block interiors from incoherent scalar mutations.

    Ring membership is derived from the current chemical graph: no persistent
    lock or fragment identity is added to the state.  Boundary attachment
    remains legal, while scalar cycle closure and atom/bond mutation inside an
    existing cycle must be expressed by coordinated ring-system rules.
    """

    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    graph = nx.Graph()
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (a, b) for a, b in combinations(real, 2) if int(state.bonds[a, b]) != 0
    )
    cycle_edges = {frozenset((int(a), int(b))) for a, b in graph.edges()} - {
        frozenset((int(a), int(b))) for a, b in nx.bridges(graph)
    }
    ring_atoms = {v for edge in cycle_edges for v in edge}

    def permitted(transition: MarkedTransition) -> bool:
        action = transition.action
        if isinstance(action, BondInsert):
            return False
        if isinstance(action, (AtomDelete, AtomRestate)):
            return int(action.v) not in ring_atoms
        if isinstance(action, (BondDelete, BondReorder)):
            return frozenset((int(action.a), int(action.b))) not in cycle_edges
        return True

    return {
        family: tuple(item for item in transitions if permitted(item))
        for family, transitions in by_family.items()
    }


def _append_typed_catalog_actions(
    by_family: dict[str, list[MarkedTransition]],
    state: MolecularGraph,
    *,
    real: tuple[int, ...],
    null: tuple[int, ...],
    catalog: TypedRingCatalog,
) -> None:
    if not real:
        for template in catalog.cycle_templates:
            if template.span > len(null):
                continue
            _append_verified_tracelet(
                by_family,
                state,
                "cycle_insert",
                template.instantiate(null[: template.span]),
            )
        return
    cyclic_blocks = _cyclic_blocks(state)
    ring_atoms = frozenset(v for block in cyclic_blocks for v in block)
    cyclic_pairs = {
        (min(a, b), max(a, b))
        for block in cyclic_blocks
        for a, b in combinations(sorted(block), 2)
    }
    _, closure_topology, _ = compute_topology_features(state)
    allowed_closure_buckets = (
        frozenset(catalog.zero_span_ring_buckets)
        if catalog.zero_span_ring_buckets
        else frozenset(range(1, 6))
    )
    closure_pairs = {
        (int(a), int(b))
        for a, b in combinations(sorted(real), 2)
        if int(state.bonds[a, b]) == 0
        and int(closure_topology[a, b]) in allowed_closure_buckets
    }
    if len(real) == 1:
        anchor = real[0]
        for template in catalog.cycle_templates:
            if template.span - 1 > len(null):
                continue
            for oriented in oriented_cycle_templates(template):
                atom_type, charge, post_h = oriented.atoms[0]
                required = int(BOND_CLASS_TO_H_CHANGE[oriented.bond_orders[0]])
                required += int(BOND_CLASS_TO_H_CHANGE[oriented.bond_orders[-1]])
                if int(state.atom_types[anchor]) != int(atom_type):
                    continue
                if int(state.formal_charges[anchor]) != int(charge):
                    continue
                if int(state.implicit_h_counts[anchor]) != int(post_h) + required:
                    continue
                root_ear = EarTemplate(
                    same_anchor=True,
                    atoms=tuple(oriented.atoms[1:]),
                    bond_orders=tuple(oriented.bond_orders),
                )
                _append_verified_tracelet(
                    by_family,
                    state,
                    "ring_ear_insert",
                    root_ear.instantiate(
                        anchor,
                        anchor,
                        null[: root_ear.span],
                    ),
                )
    for template in catalog.attach_templates:
        if template.span > len(null):
            continue
        required = int(BOND_CLASS_TO_H_CHANGE[template.attachment_order])
        for anchor in real:
            if int(state.implicit_h_counts[anchor]) < required:
                continue
            _append_verified_tracelet(
                by_family,
                state,
                "cycle_attach",
                template.instantiate(anchor, null[: template.span]),
            )
    for template in catalog.ear_templates:
        if template.span > len(null):
            continue
        slots = null[: template.span]
        for oriented in oriented_ear_templates(template):
            if oriented.same_anchor:
                required = int(BOND_CLASS_TO_H_CHANGE[oriented.bond_orders[0]])
                required += int(BOND_CLASS_TO_H_CHANGE[oriented.bond_orders[-1]])
                for anchor in sorted(ring_atoms):
                    if int(state.implicit_h_counts[anchor]) < required:
                        continue
                    _append_verified_tracelet(
                        by_family,
                        state,
                        "ring_ear_insert",
                        oriented.instantiate(anchor, anchor, slots),
                    )
            else:
                # A zero-span ear is the coordinated closure of an existing
                # path.  It must be available on an acyclic target scaffold,
                # where no cyclic block exists yet.  Positive-span ears retain
                # the stricter ring-system extension scope used by the original
                # ring grammar.
                anchors = closure_pairs if oriented.span == 0 else cyclic_pairs
                for a, b in sorted(anchors):
                    left_required = int(BOND_CLASS_TO_H_CHANGE[oriented.bond_orders[0]])
                    right_required = int(
                        BOND_CLASS_TO_H_CHANGE[oriented.bond_orders[-1]]
                    )
                    if int(state.implicit_h_counts[a]) < left_required:
                        continue
                    if int(state.implicit_h_counts[b]) < right_required:
                        continue
                    _append_verified_tracelet(
                        by_family,
                        state,
                        "ring_ear_insert",
                        oriented.instantiate(a, b, slots),
                    )


def _cyclic_blocks(state: MolecularGraph) -> tuple[frozenset[int], ...]:
    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    graph = nx.Graph()
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (a, b) for a, b in combinations(real, 2) if int(state.bonds[a, b]) != 0
    )
    blocks = []
    for vertices in nx.biconnected_components(graph):
        block = frozenset(int(v) for v in vertices)
        if graph.subgraph(block).number_of_edges() >= len(block):
            blocks.append(block)
    return tuple(sorted(blocks, key=lambda item: tuple(sorted(item))))


def _ring_system_restate_candidates(
    state: MolecularGraph,
) -> tuple[RingSystemRestate, ...]:
    real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    graph = nx.Graph()
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (a, b) for a, b in combinations(real, 2) if int(state.bonds[a, b]) != 0
    )
    bridges = {frozenset((a, b)) for a, b in nx.bridges(graph)}
    ring_graph = graph.copy()
    ring_graph.remove_edges_from(tuple(tuple(edge) for edge in bridges))
    actions = []
    for vertices in nx.connected_components(ring_graph):
        component = ring_graph.subgraph(vertices)
        edges = tuple(sorted((min(a, b), max(a, b)) for a, b in component.edges()))
        if len(edges) < 3:
            continue
        maximum_size = len(nx.max_weight_matching(component, maxcardinality=True))
        if maximum_size < 2:
            continue
        doubled_sets = list(_matching_edge_sets(edges, size=maximum_size))
        # The explicit all-single successor supplies the exact inverse family
        # for a coordinated restate without enumerating every partial diene.
        if sum(int(state.bonds[a, b]) != 1 for a, b in edges) >= 2:
            doubled_sets.append(frozenset())
        for doubled in doubled_sets:
            changes = tuple(
                BondOrderChange(a, b, 2 if (a, b) in doubled else 1)
                for a, b in edges
                if int(state.bonds[a, b]) != (2 if (a, b) in doubled else 1)
            )
            if len(changes) >= 2:
                actions.append(RingSystemRestate(changes))
    return tuple(actions)


def _matching_edge_sets(
    edges: tuple[tuple[int, int], ...],
    *,
    size: int,
) -> tuple[frozenset[tuple[int, int]], ...]:
    """Enumerate fixed-size matchings of a small chemical ring-system graph."""

    result: list[frozenset[tuple[int, int]]] = []

    def visit(
        remaining: tuple[tuple[int, int], ...],
        selected: tuple[tuple[int, int], ...],
    ) -> None:
        if len(selected) > size or len(selected) + len(remaining) < size:
            return
        if not remaining:
            if len(selected) == size:
                result.append(frozenset(selected))
            return
        edge = remaining[0]
        visit(remaining[1:], selected)
        occupied = set(edge)
        visit(
            tuple(item for item in remaining[1:] if occupied.isdisjoint(item)),
            (*selected, edge),
        )

    visit(edges, ())
    return tuple(result)


def _append_verified_tracelet(
    by_family: dict[str, list[MarkedTransition]],
    state: MolecularGraph,
    rule_name: str,
    action: CycleInsert | CycleAttach | RingEarInsert,
) -> None:
    """Materialize an already verified typed template in one transaction.

    Catalog templates were compiled from validity-closed micro lowerings.  The
    caller checks their state-dependent application conditions (capacity,
    anchor identity, and available valence), so replaying every micro lowering
    here is redundant and made one macro candidate perform dozens of RDKit
    sanitizations.  We construct the mathematically identical committed state
    directly and canonicalize it once; canonicalization remains the final
    chemistry gate and also supplies the quotient key.
    """

    try:
        successor = _materialize_verified_tracelet(state, action)
        successor_key = canonical_state_key(successor)
    except InvalidRewrite:
        return
    by_family[rule_name].append(
        MarkedTransition(
            rule_name=rule_name,
            action=action,
            successor=successor,
            successor_key=successor_key,
        )
    )


def _materialize_verified_tracelet(
    state: MolecularGraph,
    action: CycleInsert | CycleAttach | RingEarInsert,
) -> MolecularGraph:
    atom_types = state.atom_types.copy()
    charges = state.formal_charges.copy()
    hydrogens = state.implicit_h_counts.copy()
    bonds = state.bonds.copy()

    for atom in action.atoms:
        slot = int(atom.slot)
        atom_types[slot] = int(atom.atom_type)
        charges[slot] = int(atom.formal_charge)
        hydrogens[slot] = int(atom.implicit_h_count)

    slots = tuple(int(atom.slot) for atom in action.atoms)
    if isinstance(action, (CycleInsert, CycleAttach)):
        for index, left in enumerate(slots):
            right = slots[(index + 1) % len(slots)]
            order = int(action.bond_orders[index])
            bonds[left, right] = bonds[right, left] = order
        if isinstance(action, CycleAttach):
            anchor = int(action.anchor)
            first = slots[0]
            order = int(action.attachment_order)
            bonds[anchor, first] = bonds[first, anchor] = order
            hydrogens[anchor] -= int(BOND_CLASS_TO_H_CHANGE[order])
    else:
        a, b = int(action.a), int(action.b)
        path = (a, *slots, b)
        for (left, right), order in zip(
            zip(path, path[1:]),
            action.bond_orders,
        ):
            bonds[left, right] = bonds[right, left] = int(order)
        hydrogens[a] -= int(BOND_CLASS_TO_H_CHANGE[action.bond_orders[0]])
        hydrogens[b] -= int(BOND_CLASS_TO_H_CHANGE[action.bond_orders[-1]])

    return MolecularGraph(atom_types, charges, hydrogens, bonds)


def enumerate_aromatic_closure_transitions(
    state: MolecularGraph,
    *,
    system: RewriteSystem | None = None,
) -> tuple[MarkedTransition, ...]:
    """Enumerate coordinated restates that increase perceived aromatic rings.

    Each executable action remains a validity-checked integer bond-order
    rewrite. Aromaticity is a global application condition on its successor,
    so maximum matchings that merely create nonaromatic conjugation stay in the
    universal micro-rewrite substrate instead of sharing this macro family.
    """

    transitions = []
    for action, successor in _validated_aromatic_restates(state, system=system):
        transitions.append(
            MarkedTransition(
                rule_name="ring_system_restate",
                action=action,
                successor=successor,
                successor_key=canonical_state_key(successor),
            )
        )
    return tuple(transitions)


def enumerate_aromatic_restate_actions(
    state: MolecularGraph,
    *,
    system: RewriteSystem | None = None,
) -> tuple[RingSystemRestate, ...]:
    """Return exact executable aromatic restates without successor quotienting.

    Dense marked-rate training needs the legal rule matches, not canonical
    successor keys. Avoiding that extra RDKit conversion materially reduces
    collation cost while using the same validator and aromaticity condition.
    """

    return tuple(
        action for action, _ in _validated_aromatic_restates(state, system=system)
    )


def enumerate_ring_system_restate_actions(
    state: MolecularGraph,
    *,
    system: RewriteSystem | None = None,
) -> tuple[RingSystemRestate, ...]:
    """Return exact executable ring-system restates that CHANGE perceived aromaticity -- both
    aromatizing (saturated->aromatic) AND de-aromatizing (aromatic->saturated) -- excluding
    aromaticity-neutral Kekule resonance reshuffles (canonical no-ops: a delta of 0 means the
    successor is the same molecule). This wires the ``ring_system_restate`` family at inference so the
    model can flip ring saturation in either direction; B ships aromatize-only
    (``enumerate_aromatic_restate_actions``). Same candidate space + validator; states are Kekule, so
    de-aromatization is a validity-checked double->single lowering that adds implicit H.
    """

    return tuple(
        action
        for action, _ in _validated_restates(
            state, system=system, keep=lambda before, after: after != before
        )
    )


def enumerate_ring_system_restate_transitions(
    state: MolecularGraph,
    *,
    system: RewriteSystem | None = None,
) -> tuple[tuple[RingSystemRestate, MolecularGraph], ...]:
    """Return each executable bidirectional restatement with its exact successor.

    The public action-only enumerator above remains unchanged. Editing-V2 CPU
    collation uses this transition-bearing form so semantic successor groups,
    charge filtering, and descriptors reuse the validator's one executor pass.
    """

    return _validated_restates(
        state,
        system=system,
        keep=lambda before, after: after != before,
    )


def _validated_restates(
    state: MolecularGraph,
    *,
    system: RewriteSystem | None,
    keep,
) -> tuple[tuple[RingSystemRestate, MolecularGraph], ...]:
    """Validate every ``_ring_system_restate_candidates`` action through the rewrite system (the
    validity gate) and keep those whose perceived-aromatic-ring delta satisfies ``keep(before, after)``.
    Every returned successor is therefore valence-safe and connected."""
    runtime = system or de_novo_rewrite_system()
    before = perceived_aromatic_ring_count(state)
    validated = []
    for action in _ring_system_restate_candidates(state):
        try:
            successor = runtime.apply(state, "ring_system_restate", action)
        except InvalidRewrite:
            continue
        if keep(before, perceived_aromatic_ring_count(successor)):
            validated.append((action, successor))
    return tuple(validated)


def _validated_aromatic_restates(
    state: MolecularGraph,
    *,
    system: RewriteSystem | None,
) -> tuple[tuple[RingSystemRestate, MolecularGraph], ...]:
    # Aromatizing only (saturated -> aromatic): the training closure fiber's application condition.
    return _validated_restates(
        state, system=system, keep=lambda before, after: after > before
    )


def _append_generic_cycle(
    by_family: dict[str, list[MarkedTransition]],
    state: MolecularGraph,
    action: CycleInsert,
) -> None:
    """Commit the analytically valid generic C/single-bond cycle carrier.

    These exact preconditions are a specialized compilation of the verified
    ``CycleInsert`` lowering. They avoid repeating RDKit sanitization at every
    micro step for every candidate in a training-time fiber.
    """

    slots = tuple(int(atom.slot) for atom in action.atoms)
    atom_types = state.atom_types.copy()
    charges = state.formal_charges.copy()
    hydrogens = state.implicit_h_counts.copy()
    bonds = state.bonds.copy()
    carbon = int(ELEMENT_TO_IDX["C"])
    for slot in slots:
        atom_types[slot] = carbon
        charges[slot] = 0
        hydrogens[slot] = 2
    for index, left in enumerate(slots):
        right = slots[(index + 1) % len(slots)]
        bonds[left, right] = bonds[right, left] = 1
    successor = MolecularGraph(atom_types, charges, hydrogens, bonds)
    by_family["cycle_insert"].append(
        MarkedTransition(
            rule_name="cycle_insert",
            action=action,
            successor=successor,
            successor_key=canonical_state_key(successor),
        )
    )


def _append_generic_ear(
    by_family: dict[str, list[MarkedTransition]],
    state: MolecularGraph,
    action: RingEarInsert,
) -> None:
    """Commit a generic C/single-bond ear after exact anchor-H masking."""

    a, b = int(action.a), int(action.b)
    slots = tuple(int(atom.slot) for atom in action.atoms)
    atom_types = state.atom_types.copy()
    charges = state.formal_charges.copy()
    hydrogens = state.implicit_h_counts.copy()
    bonds = state.bonds.copy()
    carbon = int(ELEMENT_TO_IDX["C"])
    for slot in slots:
        atom_types[slot] = carbon
        charges[slot] = 0
        hydrogens[slot] = 2
    path = (a, *slots, b)
    for left, right in zip(path, path[1:]):
        bonds[left, right] = bonds[right, left] = 1
    hydrogens[a] -= 1
    hydrogens[b] -= 1
    successor = MolecularGraph(atom_types, charges, hydrogens, bonds)
    by_family["ring_ear_insert"].append(
        MarkedTransition(
            rule_name="ring_ear_insert",
            action=action,
            successor=successor,
            successor_key=canonical_state_key(successor),
        )
    )
