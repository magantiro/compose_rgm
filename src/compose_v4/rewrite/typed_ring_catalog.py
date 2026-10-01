"""Finite proposal support for the typed-ring tracelet diagnostic.

The catalog is deliberately a *proposal* device rather than the final model:
it records typed cycle/attachment/ear marks observed in the training split.
Topology-changing micro closures are deliberately disabled in this mode so
an acyclic atom cannot be retrospectively reinterpreted as a ring atom.  The
catalog lets us test the committed-topology mechanism before implementing an
unrestricted factorized mark decoder.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np

from compose_v4.chem.molecular_graph import (
    BOND_CLASS_TO_H_CHANGE,
    ELEMENT_TO_IDX,
    MolecularGraph,
    is_element,
)

from compose_v4.rewrite.trace import RewriteTrace, execute_trace
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.tracelets import (
    AtomPayload,
    CycleAttach,
    CycleInsert,
    RingEarInsert,
    RingSystemGrow,
)


AtomState = tuple[int, int, int]
RING_SYSTEM_ELECTRONIC_ALIAS_VERSION = 1


@dataclass(frozen=True, order=True)
class CycleTemplate:
    atoms: tuple[AtomState, ...]
    bond_orders: tuple[int, ...]

    @property
    def span(self) -> int:
        return len(self.atoms)

    def instantiate(self, slots: tuple[int, ...]) -> CycleInsert:
        if len(slots) != self.span:
            raise ValueError("cycle template received the wrong number of slots")
        return CycleInsert(
            atoms=tuple(
                AtomPayload(slot, atom_type, charge, hydrogens)
                for slot, (atom_type, charge, hydrogens) in zip(slots, self.atoms)
            ),
            bond_orders=self.bond_orders,
        )


@dataclass(frozen=True, order=True)
class AttachTemplate:
    atoms: tuple[AtomState, ...]
    bond_orders: tuple[int, ...]
    attachment_order: int

    @property
    def span(self) -> int:
        return len(self.atoms)

    def instantiate(self, anchor: int, slots: tuple[int, ...]) -> CycleAttach:
        if len(slots) != self.span:
            raise ValueError("attachment template received the wrong number of slots")
        return CycleAttach(
            anchor=int(anchor),
            atoms=tuple(
                AtomPayload(slot, atom_type, charge, hydrogens)
                for slot, (atom_type, charge, hydrogens) in zip(slots, self.atoms)
            ),
            bond_orders=self.bond_orders,
            attachment_order=int(self.attachment_order),
        )


@dataclass(frozen=True, order=True)
class EarTemplate:
    same_anchor: bool
    atoms: tuple[AtomState, ...]
    bond_orders: tuple[int, ...]

    @property
    def span(self) -> int:
        return len(self.atoms)

    def instantiate(self, a: int, b: int, slots: tuple[int, ...]) -> RingEarInsert:
        if len(slots) != self.span:
            raise ValueError("ear template received the wrong number of slots")
        if self.same_anchor != (int(a) == int(b)):
            raise ValueError("ear template anchor topology does not match")
        return RingEarInsert(
            a=int(a),
            b=int(b),
            atoms=tuple(
                AtomPayload(slot, atom_type, charge, hydrogens)
                for slot, (atom_type, charge, hydrogens) in zip(slots, self.atoms)
            ),
            bond_orders=self.bond_orders,
        )

    def reversed(self) -> "EarTemplate":
        return EarTemplate(
            same_anchor=self.same_anchor,
            atoms=tuple(reversed(self.atoms)),
            bond_orders=tuple(reversed(self.bond_orders)),
        )


@dataclass(frozen=True, order=True)
class RingSystemTemplate:
    """Slot-free complete ring-system rewrite observed in a valid trace."""

    source_atoms: tuple[AtomState, ...]
    target_atoms: tuple[AtomState, ...]
    source_bonds: tuple[tuple[int, int, int], ...]
    target_bonds: tuple[tuple[int, int, int], ...]
    grow_bond_reorders: tuple[tuple[int, int, int], ...]
    grow_atom_payloads: tuple[tuple[int, AtomState], ...]
    inserted_bonds: tuple[tuple[int, int, int], ...]
    deleted_bonds: tuple[tuple[int, int, int], ...]
    delete_atom_payloads: tuple[tuple[int, AtomState], ...]
    delete_bond_reorders: tuple[tuple[int, int, int], ...]
    source_external_bonds: tuple[tuple[int, ...], ...]
    target_aromatic_edges: tuple[tuple[int, int], ...]
    topology_class: str

    @property
    def span(self) -> int:
        return len(self.source_atoms)


@dataclass(frozen=True, order=True)
class RingSystemElectronicAlias:
    """One observed joint atom/electronic assignment paired to a ring pattern."""

    pattern: RingSystemTemplate
    target_atoms: tuple[AtomState, ...]

    def __post_init__(self) -> None:
        if len(self.target_atoms) != self.pattern.span:
            raise ValueError("ring electronic alias has the wrong atom-label span")


class RingSystemElectronicAliasVersionError(RuntimeError):
    """Raised when exact electronic decoding is requested from a legacy catalog."""


@dataclass(frozen=True)
class TypedRingCatalog:
    cycle_templates: tuple[CycleTemplate, ...]
    attach_templates: tuple[AttachTemplate, ...]
    ear_templates: tuple[EarTemplate, ...]
    zero_span_ring_buckets: tuple[int, ...] = ()
    ring_system_templates: tuple[RingSystemTemplate, ...] = ()
    ring_system_template_counts: tuple[int, ...] = ()
    ring_system_electronic_alias_version: int = 0
    ring_system_electronic_aliases: tuple[RingSystemElectronicAlias, ...] = ()
    ring_system_electronic_alias_counts: tuple[int, ...] = ()

    def supports_trace(self, trace: RewriteTrace) -> bool:
        cycles = frozenset(self.cycle_templates)
        attachments = frozenset(self.attach_templates)
        ears = frozenset(self.ear_templates)
        ring_systems = frozenset(self.ring_system_templates)
        _, states = execute_trace(
            trace.source,
            trace.steps,
            return_states=True,
        )
        for index, step in enumerate(trace.steps):
            if isinstance(step.action, CycleInsert):
                if cycle_template(step.action) not in cycles:
                    return False
            elif isinstance(step.action, CycleAttach):
                if attach_template(step.action) not in attachments:
                    return False
            elif isinstance(step.action, RingEarInsert):
                if ear_template(step.action) not in ears:
                    return False
            elif isinstance(step.action, RingSystemGrow):
                if ring_system_template(states[index], step.action) not in ring_systems:
                    return False
        return True

    def statistics(self) -> dict[str, int]:
        return {
            "cycle_templates": len(self.cycle_templates),
            "attach_templates": len(self.attach_templates),
            "ear_templates": len(self.ear_templates),
            "zero_span_ring_buckets": len(self.zero_span_ring_buckets),
            "ring_system_templates": len(self.ring_system_templates),
            "ring_system_observations": sum(
                getattr(self, "ring_system_template_counts", ())
            ),
            "ring_system_electronic_alias_version": int(
                getattr(self, "ring_system_electronic_alias_version", 0)
            ),
            "ring_system_electronic_aliases": len(
                getattr(self, "ring_system_electronic_aliases", ())
            ),
            "ring_system_electronic_observations": sum(
                getattr(self, "ring_system_electronic_alias_counts", ())
            ),
        }


def ring_system_electronic_alias(
    action: RingSystemGrow,
) -> RingSystemElectronicAlias:
    """Remove slots while retaining one observed joint electronic realization."""

    pattern = ring_system_pattern_template(action)
    members = tuple(int(slot) for slot in action.system_atoms)
    payload_by_slot = {int(payload.slot): payload for payload in action.atom_payloads}
    if set(payload_by_slot) != set(members):
        raise ValueError("ring electronic alias requires one payload per member")
    closure_valence = {
        slot: sum(
            int(BOND_CLASS_TO_H_CHANGE[int(bond.order)])
            for bond in action.bond_insertions
            if slot in (int(bond.a), int(bond.b))
        )
        for slot in members
    }
    target_atoms = tuple(
        (
            int(payload_by_slot[slot].atom_type),
            int(payload_by_slot[slot].formal_charge),
            int(payload_by_slot[slot].implicit_h_count) - closure_valence[slot],
        )
        for slot in members
    )
    return RingSystemElectronicAlias(pattern=pattern, target_atoms=target_atoms)


def _ring_electronic_alias_graph(alias: RingSystemElectronicAlias) -> nx.Graph:
    graph = nx.Graph()
    for index, atom_state in enumerate(alias.target_atoms):
        graph.add_node(int(index), color=repr(tuple(int(value) for value in atom_state)))
    pattern = alias.pattern
    source_orders = {
        (min(int(a), int(b)), max(int(a), int(b))): int(order)
        for a, b, order in pattern.source_bonds
    }
    target_orders = {
        (min(int(a), int(b)), max(int(a), int(b))): int(order)
        for a, b, order in pattern.target_bonds
    }
    inserted = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b, _ in pattern.inserted_bonds
    }
    aromatic = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in pattern.target_aromatic_edges
    }
    for edge in sorted(set(source_orders) | set(target_orders)):
        graph.add_edge(
            *edge,
            color=repr(
                (
                    source_orders.get(edge, 0),
                    target_orders.get(edge, 0),
                    edge in inserted,
                    edge in aromatic,
                )
            ),
        )
    return graph


def ring_system_electronic_alias_orbit_bucket_key(
    alias: RingSystemElectronicAlias,
) -> tuple[int, str, str]:
    """Return a coarse WL bucket; exact equivalence still uses isomorphism."""

    graph = _ring_electronic_alias_graph(alias)
    return (
        alias.pattern.span,
        str(alias.pattern.topology_class),
        nx.weisfeiler_lehman_graph_hash(
            graph,
            node_attr="color",
            edge_attr="color",
        ),
    )


def ring_system_electronic_aliases_equivalent(
    left: RingSystemElectronicAlias,
    right: RingSystemElectronicAlias,
) -> bool:
    """Exact colored-graph orbit equivalence for paired electronic aliases."""

    if ring_system_electronic_alias_orbit_bucket_key(
        left
    ) != ring_system_electronic_alias_orbit_bucket_key(right):
        return False
    return nx.is_isomorphic(
        _ring_electronic_alias_graph(left),
        _ring_electronic_alias_graph(right),
        node_match=lambda a, b: a["color"] == b["color"],
        edge_match=lambda a, b: a["color"] == b["color"],
    )


class _RingElectronicAliasAccumulator:
    """Exact orbit counter with WL bucketing for efficient catalog construction."""

    def __init__(self) -> None:
        self._buckets: dict[tuple[int, str, str], list[list[object]]] = {}

    def add(self, alias: RingSystemElectronicAlias) -> None:
        key = ring_system_electronic_alias_orbit_bucket_key(alias)
        graph = _ring_electronic_alias_graph(alias)
        bucket = self._buckets.setdefault(key, [])
        for item in bucket:
            representative = item[0]
            representative_graph = item[1]
            if nx.is_isomorphic(
                graph,
                representative_graph,
                node_match=lambda a, b: a["color"] == b["color"],
                edge_match=lambda a, b: a["color"] == b["color"],
            ):
                item[2] = int(item[2]) + 1
                if alias < representative:
                    item[0] = alias
                    item[1] = graph
                return
        bucket.append([alias, graph, 1])

    def items(self) -> tuple[tuple[RingSystemElectronicAlias, int], ...]:
        items = tuple(
            (item[0], int(item[2]))
            for bucket in self._buckets.values()
            for item in bucket
        )
        return tuple(sorted(items, key=lambda item: (-item[1], item[0])))


def require_ring_system_electronic_aliases(
    catalog: TypedRingCatalog,
    *,
    expected_version: int = RING_SYSTEM_ELECTRONIC_ALIAS_VERSION,
) -> tuple[tuple[RingSystemElectronicAlias, int], ...]:
    """Return a validated v1 alias table or fail loudly for a legacy catalog."""

    version = int(getattr(catalog, "ring_system_electronic_alias_version", 0))
    if version != int(expected_version):
        raise RingSystemElectronicAliasVersionError(
            "exact ring-electronic mode requires catalog alias version "
            f"v{expected_version}; found v{version}. This topology-only catalog "
            "cannot recover discarded atom labels; rebuild the path manifest."
        )
    aliases = tuple(getattr(catalog, "ring_system_electronic_aliases", ()))
    counts = tuple(int(value) for value in getattr(
        catalog,
        "ring_system_electronic_alias_counts",
        (),
    ))
    if len(aliases) != len(counts):
        raise ValueError("ring electronic alias/count tables are misaligned")
    if any(count <= 0 for count in counts):
        raise ValueError("ring electronic alias counts must be positive")
    return tuple(zip(aliases, counts))


def build_typed_ring_catalog(
    traces: Iterable[RewriteTrace],
    *,
    max_cycle_templates: int = 128,
    max_attach_templates: int | None = None,
    max_ear_templates: int = 128,
    max_ring_system_templates: int = 512,
) -> TypedRingCatalog:
    traces = tuple(traces)
    if max_attach_templates is None:
        max_attach_templates = max_cycle_templates
    if (
        max_cycle_templates <= 0
        or max_attach_templates <= 0
        or max_ear_templates <= 0
        or max_ring_system_templates <= 0
    ):
        raise ValueError("typed ring catalog limits must be positive")
    cycle_counts: Counter[CycleTemplate] = Counter()
    attach_counts: Counter[AttachTemplate] = Counter()
    ear_counts: Counter[EarTemplate] = Counter()
    ring_system_counts: Counter[RingSystemTemplate] = Counter()
    electronic_aliases = _RingElectronicAliasAccumulator()
    zero_span_ring_buckets: set[int] = set()
    for trace in traces:
        zero_span_ring_buckets.update(_target_ring_buckets(trace))
        _, states = execute_trace(
            trace.source,
            trace.steps,
            return_states=True,
        )
        for index, step in enumerate(trace.steps):
            if isinstance(step.action, CycleInsert):
                cycle_counts[cycle_template(step.action)] += 1
            elif isinstance(step.action, CycleAttach):
                attach_counts[attach_template(step.action)] += 1
            elif isinstance(step.action, RingEarInsert):
                ear_counts[ear_template(step.action)] += 1
            elif isinstance(step.action, RingSystemGrow):
                ring_system_counts[
                    ring_system_template(states[index], step.action)
                ] += 1
                electronic_aliases.add(ring_system_electronic_alias(step.action))
    return _finalize_typed_ring_catalog(
        cycle_counts=cycle_counts,
        attach_counts=attach_counts,
        ear_counts=ear_counts,
        ring_system_counts=ring_system_counts,
        ring_system_electronic_alias_items=electronic_aliases.items(),
        zero_span_ring_buckets=zero_span_ring_buckets,
        max_cycle_templates=max_cycle_templates,
        max_attach_templates=max_attach_templates,
        max_ear_templates=max_ear_templates,
        max_ring_system_templates=max_ring_system_templates,
    )


def build_typed_ring_catalog_from_paths(
    paths: Iterable[TraceProgressCTMC],
    *,
    max_cycle_templates: int = 128,
    max_attach_templates: int | None = None,
    max_ear_templates: int = 128,
    max_ring_system_templates: int = 512,
) -> TypedRingCatalog:
    """Extract templates from checkpoint-local states without replaying traces."""

    paths = tuple(paths)
    if max_attach_templates is None:
        max_attach_templates = max_cycle_templates
    if (
        max_cycle_templates <= 0
        or max_attach_templates <= 0
        or max_ear_templates <= 0
        or max_ring_system_templates <= 0
    ):
        raise ValueError("typed ring catalog limits must be positive")
    cycle_counts: Counter[CycleTemplate] = Counter()
    attach_counts: Counter[AttachTemplate] = Counter()
    ear_counts: Counter[EarTemplate] = Counter()
    ring_system_counts: Counter[RingSystemTemplate] = Counter()
    electronic_aliases = _RingElectronicAliasAccumulator()
    zero_span_ring_buckets: set[int] = set()
    for path in paths:
        trace = path.trace
        zero_span_ring_buckets.update(_target_ring_buckets(trace))
        for step in trace.steps:
            if isinstance(step.action, CycleInsert):
                cycle_counts[cycle_template(step.action)] += 1
            elif isinstance(step.action, CycleAttach):
                attach_counts[attach_template(step.action)] += 1
            elif isinstance(step.action, RingEarInsert):
                ear_counts[ear_template(step.action)] += 1
            elif isinstance(step.action, RingSystemGrow):
                ring_system_counts[ring_system_pattern_template(step.action)] += 1
                electronic_aliases.add(ring_system_electronic_alias(step.action))
    return _finalize_typed_ring_catalog(
        cycle_counts=cycle_counts,
        attach_counts=attach_counts,
        ear_counts=ear_counts,
        ring_system_counts=ring_system_counts,
        ring_system_electronic_alias_items=electronic_aliases.items(),
        zero_span_ring_buckets=zero_span_ring_buckets,
        max_cycle_templates=max_cycle_templates,
        max_attach_templates=max_attach_templates,
        max_ear_templates=max_ear_templates,
        max_ring_system_templates=max_ring_system_templates,
    )


def _ring_pattern_semantic_graph(template: RingSystemTemplate) -> nx.Graph:
    """Pattern graph used to align electronic aliases with selected semantics."""

    graph = nx.Graph()
    graph.add_nodes_from(range(template.span))
    inserted = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b, _ in template.inserted_bonds
    }
    aromatic = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in template.target_aromatic_edges
    }
    for a, b, order in template.target_bonds:
        edge = (min(int(a), int(b)), max(int(a), int(b)))
        semantic_order = 4 if edge in aromatic else int(order)
        role = "inserted" if edge in inserted else "scaffold"
        graph.add_edge(*edge, color=f"{role}:{semantic_order}")
    return graph


def _ring_pattern_semantic_bucket_key(template: RingSystemTemplate) -> tuple[int, str, str]:
    graph = _ring_pattern_semantic_graph(template)
    return (
        template.span,
        str(template.topology_class),
        nx.weisfeiler_lehman_graph_hash(graph, edge_attr="color"),
    )


def _retain_selected_ring_electronic_aliases(
    items: tuple[tuple[RingSystemElectronicAlias, int], ...],
    selected_templates: tuple[RingSystemTemplate, ...],
) -> tuple[tuple[RingSystemElectronicAlias, int], ...]:
    """Keep aliases whose semantic topology survived the template cap."""

    selected: dict[tuple[int, str, str], list[nx.Graph]] = {}
    for template in selected_templates:
        selected.setdefault(
            _ring_pattern_semantic_bucket_key(template),
            [],
        ).append(_ring_pattern_semantic_graph(template))
    retained = []
    for alias, count in items:
        graph = _ring_pattern_semantic_graph(alias.pattern)
        key = _ring_pattern_semantic_bucket_key(alias.pattern)
        if any(
            nx.is_isomorphic(
                graph,
                candidate,
                edge_match=lambda a, b: a["color"] == b["color"],
            )
            for candidate in selected.get(key, ())
        ):
            retained.append((alias, int(count)))
    return tuple(retained)


def _finalize_typed_ring_catalog(
    *,
    cycle_counts: Counter[CycleTemplate],
    attach_counts: Counter[AttachTemplate],
    ear_counts: Counter[EarTemplate],
    ring_system_counts: Counter[RingSystemTemplate],
    ring_system_electronic_alias_items: tuple[
        tuple[RingSystemElectronicAlias, int], ...
    ],
    zero_span_ring_buckets: set[int],
    max_cycle_templates: int,
    max_attach_templates: int,
    max_ear_templates: int,
    max_ring_system_templates: int,
) -> TypedRingCatalog:
    # Existing-atom ring closures are the universal zero-span part of the ear
    # grammar.  Null-source traces often build isolated rings in one macro and
    # therefore need not expose these marks, whereas a size-matched carbon-tree
    # transport necessarily closes target chords after topology repair.  Keep
    # the primitive closure basis in proposal support and use the remaining
    # budget for corpus-observed typed ears.
    required_ears = tuple(
        EarTemplate(same_anchor=False, atoms=(), bond_orders=(order,))
        for order in (1, 2, 3)
    )[:max_ear_templates]
    remaining_ears = Counter(
        {
            template: count
            for template, count in ear_counts.items()
            if template not in required_ears
        }
    )
    selected_ears = (
        *required_ears,
        *_most_common_stable(
            remaining_ears,
            max(max_ear_templates - len(required_ears), 0),
        ),
    )
    ranked_ring_systems = _most_common_stable_items(
        ring_system_counts,
        max_ring_system_templates,
    )
    selected_ring_systems = tuple(template for template, _ in ranked_ring_systems)
    retained_electronic_aliases = _retain_selected_ring_electronic_aliases(
        ring_system_electronic_alias_items,
        selected_ring_systems,
    )
    return TypedRingCatalog(
        cycle_templates=_most_common_stable(cycle_counts, max_cycle_templates),
        attach_templates=_most_common_stable(attach_counts, max_attach_templates),
        ear_templates=tuple(selected_ears),
        zero_span_ring_buckets=tuple(sorted(zero_span_ring_buckets)),
        ring_system_templates=selected_ring_systems,
        ring_system_template_counts=tuple(
            int(count) for _, count in ranked_ring_systems
        ),
        ring_system_electronic_alias_version=RING_SYSTEM_ELECTRONIC_ALIAS_VERSION,
        ring_system_electronic_aliases=tuple(
            alias for alias, _ in retained_electronic_aliases
        ),
        ring_system_electronic_alias_counts=tuple(
            int(count) for _, count in retained_electronic_aliases
        ),
    )


def _target_ring_buckets(trace: RewriteTrace) -> set[int]:
    """Return corpus-observed smallest-cycle buckets for one target graph."""

    target = trace.target
    real = tuple(int(v) for v in np.flatnonzero(is_element(target.atom_types)))
    graph = nx.Graph()
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (a, b)
        for offset, a in enumerate(real)
        for b in real[offset + 1 :]
        if int(target.bonds[a, b]) != 0
    )
    buckets = set()
    for a, b in tuple(graph.edges()):
        graph.remove_edge(a, b)
        try:
            ring_size = int(nx.shortest_path_length(graph, a, b)) + 1
        except nx.NetworkXNoPath:
            graph.add_edge(a, b)
            continue
        graph.add_edge(a, b)
        if ring_size == 3:
            buckets.add(1)
        elif ring_size == 4:
            buckets.add(2)
        elif ring_size == 5:
            buckets.add(3)
        elif ring_size == 6:
            buckets.add(4)
        elif ring_size >= 7:
            buckets.add(5)
    return buckets


def cycle_template(action: CycleInsert) -> CycleTemplate:
    atoms = tuple(_atom_state(atom) for atom in action.atoms)
    orders = tuple(int(order) for order in action.bond_orders)
    variants = []
    length = len(atoms)
    for start in range(length):
        for direction in (1, -1):
            indices = tuple((start + direction * offset) % length for offset in range(length))
            variant_atoms = tuple(atoms[index] for index in indices)
            variant_orders = tuple(
                orders[index if direction == 1 else (index - 1) % length]
                for index in indices
            )
            variants.append(CycleTemplate(variant_atoms, variant_orders))
    return min(variants)


def attach_template(action: CycleAttach) -> AttachTemplate:
    atoms = tuple(_atom_state(atom) for atom in action.atoms)
    forward = AttachTemplate(
        atoms=atoms,
        bond_orders=tuple(int(order) for order in action.bond_orders),
        attachment_order=int(action.attachment_order),
    )
    reverse = AttachTemplate(
        atoms=(atoms[0], *tuple(reversed(atoms[1:]))),
        bond_orders=tuple(reversed(forward.bond_orders)),
        attachment_order=forward.attachment_order,
    )
    return min(forward, reverse)


def ear_template(action: RingEarInsert) -> EarTemplate:
    forward = EarTemplate(
        same_anchor=int(action.a) == int(action.b),
        atoms=tuple(_atom_state(atom) for atom in action.atoms),
        bond_orders=tuple(int(order) for order in action.bond_orders),
    )
    return min(forward, forward.reversed())


def ring_system_template(
    source: MolecularGraph,
    action: RingSystemGrow,
) -> RingSystemTemplate:
    """Remove absolute slots while retaining the complete before/after patch."""

    if action.interface_atoms or action.atom_insertions:
        raise ValueError(
            "v1 finite ring-system templates require an existing acyclic scaffold"
        )
    members = tuple(int(slot) for slot in action.system_atoms)
    slot_to_index = {slot: index for index, slot in enumerate(members)}
    if len(slot_to_index) != len(members):
        raise ValueError("ring-system template repeats a member")
    source_atoms = tuple(
        (
            int(source.atom_types[slot]),
            int(source.formal_charges[slot]),
            int(source.implicit_h_counts[slot]),
        )
        for slot in members
    )
    source_bonds = tuple(
        (left, right, int(source.bonds[members[left], members[right]]))
        for left in range(len(members))
        for right in range(left + 1, len(members))
        if int(source.bonds[members[left], members[right]]) != 0
    )
    target_order_by_edge = {
        (left, right): int(order) for left, right, order in source_bonds
    }
    for change in action.bond_reorders:
        a = slot_to_index[int(change.a)]
        b = slot_to_index[int(change.b)]
        target_order_by_edge[(min(a, b), max(a, b))] = int(change.new_order)
    for bond in action.bond_insertions:
        a = slot_to_index[int(bond.a)]
        b = slot_to_index[int(bond.b)]
        target_order_by_edge[(min(a, b), max(a, b))] = int(bond.order)
    target_bonds = tuple(
        (a, b, order)
        for (a, b), order in sorted(target_order_by_edge.items())
    )
    payload_by_slot = {
        int(payload.slot): payload for payload in action.atom_payloads
    }
    if set(payload_by_slot) != set(members):
        raise ValueError("ring-system template requires one payload per member")
    insertion_valence = {
        slot: sum(
            int(BOND_CLASS_TO_H_CHANGE[int(bond.order)])
            for bond in action.bond_insertions
            if slot in (int(bond.a), int(bond.b))
        )
        for slot in members
    }
    target_atoms = tuple(
        (
            int(payload_by_slot[slot].atom_type),
            int(payload_by_slot[slot].formal_charge),
            int(payload_by_slot[slot].implicit_h_count) - insertion_valence[slot],
        )
        for slot in members
    )
    reordered_source_h = {
        slot: int(source.implicit_h_counts[slot]) for slot in members
    }
    for change in action.bond_reorders:
        a, b = int(change.a), int(change.b)
        old_order = int(source.bonds[a, b])
        delta = (
            int(BOND_CLASS_TO_H_CHANGE[int(change.new_order)])
            - int(BOND_CLASS_TO_H_CHANGE[old_order])
        )
        reordered_source_h[a] -= delta
        reordered_source_h[b] -= delta

    member_set = set(members)
    external = tuple(
        tuple(
            sorted(
                int(source.bonds[slot, neighbor])
                for neighbor in np.flatnonzero(source.bonds[slot] != 0)
                if int(neighbor) not in member_set
            )
        )
        for slot in members
    )
    return RingSystemTemplate(
        source_atoms=source_atoms,
        target_atoms=target_atoms,
        source_bonds=source_bonds,
        target_bonds=target_bonds,
        grow_bond_reorders=tuple(
            (
                slot_to_index[int(change.a)],
                slot_to_index[int(change.b)],
                int(change.new_order),
            )
            for change in action.bond_reorders
        ),
        grow_atom_payloads=tuple(
            (
                slot_to_index[int(payload.slot)],
                _atom_state(payload),
            )
            for payload in action.atom_payloads
        ),
        inserted_bonds=tuple(
            (
                slot_to_index[int(bond.a)],
                slot_to_index[int(bond.b)],
                int(bond.order),
            )
            for bond in action.bond_insertions
        ),
        deleted_bonds=tuple(
            (
                slot_to_index[int(bond.a)],
                slot_to_index[int(bond.b)],
                int(bond.order),
            )
            for bond in reversed(action.bond_insertions)
        ),
        delete_atom_payloads=tuple(
            (
                slot_to_index[int(payload.slot)],
                (
                    int(source.atom_types[int(payload.slot)]),
                    int(source.formal_charges[int(payload.slot)]),
                    reordered_source_h[int(payload.slot)],
                ),
            )
            for payload in reversed(action.atom_payloads)
        ),
        delete_bond_reorders=tuple(
            (
                slot_to_index[int(change.a)],
                slot_to_index[int(change.b)],
                int(source.bonds[int(change.a), int(change.b)]),
            )
            for change in reversed(action.bond_reorders)
        ),
        source_external_bonds=external,
        target_aromatic_edges=tuple(
            sorted(
                (
                    min(slot_to_index[int(a)], slot_to_index[int(b)]),
                    max(slot_to_index[int(a)], slot_to_index[int(b)]),
                )
                for a, b in action.aromatic_edges
            )
        ),
        topology_class=str(action.topology_class),
    )


def ring_system_pattern_template(action: RingSystemGrow) -> RingSystemTemplate:
    """Build the production topology/bond pattern without replaying its path."""

    if action.interface_atoms or action.atom_insertions:
        raise ValueError(
            "v1 structured ring patterns require an existing acyclic scaffold"
        )
    members = tuple(int(slot) for slot in action.system_atoms)
    slot_to_index = {slot: index for index, slot in enumerate(members)}
    if len(slot_to_index) != len(members):
        raise ValueError("ring-system pattern repeats a member")

    def local_edge(a: int, b: int) -> tuple[int, int]:
        left = slot_to_index[int(a)]
        right = slot_to_index[int(b)]
        return min(left, right), max(left, right)

    source_bonds = tuple(
        sorted(
            (*local_edge(bond.a, bond.b), int(bond.order))
            for bond in action.scaffold_bonds
        )
    )
    target_order_by_edge = {
        (a, b): int(order) for a, b, order in source_bonds
    }
    for change in action.bond_reorders:
        target_order_by_edge[local_edge(change.a, change.b)] = int(
            change.new_order
        )
    inserted_bonds = tuple(
        (*local_edge(bond.a, bond.b), int(bond.order))
        for bond in action.bond_insertions
    )
    for a, b, order in inserted_bonds:
        target_order_by_edge[(a, b)] = int(order)
    placeholder_atoms = tuple((int(ELEMENT_TO_IDX["C"]), 0, 0) for _ in members)
    return RingSystemTemplate(
        source_atoms=placeholder_atoms,
        target_atoms=placeholder_atoms,
        source_bonds=source_bonds,
        target_bonds=tuple(
            (a, b, order)
            for (a, b), order in sorted(target_order_by_edge.items())
        ),
        grow_bond_reorders=(),
        grow_atom_payloads=(),
        inserted_bonds=inserted_bonds,
        deleted_bonds=(),
        delete_atom_payloads=(),
        delete_bond_reorders=(),
        source_external_bonds=tuple(() for _ in members),
        target_aromatic_edges=tuple(
            sorted(local_edge(a, b) for a, b in action.aromatic_edges)
        ),
        topology_class=str(action.topology_class),
    )


def oriented_ear_templates(template: EarTemplate) -> tuple[EarTemplate, ...]:
    reverse = template.reversed()
    return (template,) if reverse == template else (template, reverse)


def oriented_cycle_templates(template: CycleTemplate) -> tuple[CycleTemplate, ...]:
    """Return distinct rotations/reflections for an existing-anchor cycle seed."""

    variants = set()
    length = len(template.atoms)
    for start in range(length):
        for direction in (1, -1):
            indices = tuple(
                (start + direction * offset) % length for offset in range(length)
            )
            variants.add(
                CycleTemplate(
                    atoms=tuple(template.atoms[index] for index in indices),
                    bond_orders=tuple(
                        template.bond_orders[
                            index if direction == 1 else (index - 1) % length
                        ]
                        for index in indices
                    ),
                )
            )
    return tuple(sorted(variants))


def _atom_state(atom: AtomPayload) -> AtomState:
    return (
        int(atom.atom_type),
        int(atom.formal_charge),
        int(atom.implicit_h_count),
    )


def _most_common_stable(counter: Counter, limit: int) -> tuple:
    return tuple(
        template for template, _ in _most_common_stable_items(counter, limit)
    )


def _most_common_stable_items(counter: Counter, limit: int) -> tuple[tuple, ...]:
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return tuple(ranked[:limit])


def ring_catalog_fingerprint(catalog: TypedRingCatalog) -> str:
    """A stable, order-independent content fingerprint of a ring catalog's templates.

    Recorded in the data manifest, training config, and checkpoint metadata so every consumer -- the
    rewrite system, legal-action enumeration, the executor, the data-loader corruption, and sampling --
    can cross-check that they used the SAME versioned ring-system definition (never an ad hoc catalog).
    """
    parts = tuple(
        tuple(sorted(repr(template) for template in getattr(catalog, field, ())))
        for field in ("cycle_templates", "attach_templates", "ear_templates", "ring_system_templates")
    )
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


__all__ = [
    "AttachTemplate",
    "CycleTemplate",
    "EarTemplate",
    "ring_catalog_fingerprint",
    "RING_SYSTEM_ELECTRONIC_ALIAS_VERSION",
    "RingSystemElectronicAlias",
    "RingSystemElectronicAliasVersionError",
    "RingSystemTemplate",
    "TypedRingCatalog",
    "build_typed_ring_catalog",
    "build_typed_ring_catalog_from_paths",
    "attach_template",
    "cycle_template",
    "ear_template",
    "oriented_ear_templates",
    "oriented_cycle_templates",
    "require_ring_system_electronic_aliases",
    "ring_system_electronic_alias",
    "ring_system_electronic_alias_orbit_bucket_key",
    "ring_system_electronic_aliases_equivalent",
    "ring_system_pattern_template",
    "ring_system_template",
]
