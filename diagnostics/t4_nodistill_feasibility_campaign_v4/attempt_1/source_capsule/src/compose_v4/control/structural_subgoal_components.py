"""Address-free components of a structural molecular graph subgoal.

The objects in this module are analysis representations.  They decompose a
complete structural delta without treating role order or a graph hash as exact
identity.  Exact equality is attributed graph isomorphism.
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property

import networkx as nx

from compose_v4.control.structural_subgoal_policy import StructuralDeltaTemplate


def _label(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class AttributedComponent:
    """A deterministic serialization whose equality ignores vertex order."""

    family: str
    node_labels: tuple[str, ...]
    edges: tuple[tuple[int, int, str], ...]
    directed: bool = False

    def __post_init__(self) -> None:
        size = len(self.node_labels)
        if any(
            type(left) is not int
            or type(right) is not int
            or not 0 <= left < size
            or not 0 <= right < size
            or left == right
            or not isinstance(label, str)
            for left, right, label in self.edges
        ):
            raise ValueError("component edge is malformed")
        if not self.directed and any(left > right for left, right, _ in self.edges):
            raise ValueError("undirected component edges must be canonical")
        if len(self.edges) != len(set(self.edges)):
            raise ValueError("component contains duplicate edges")

    @cached_property
    def nx_graph(self) -> nx.Graph:
        graph: nx.Graph = nx.DiGraph() if self.directed else nx.Graph()
        graph.add_nodes_from(
            (index, {"label": label}) for index, label in enumerate(self.node_labels)
        )
        graph.add_edges_from(
            (left, right, {"label": label}) for left, right, label in self.edges
        )
        return graph

    @cached_property
    def bucket_key(self) -> tuple:
        """A necessary, not sufficient, equivalence bucket."""

        if not self.node_labels:
            digest = "empty"
        else:
            digest = nx.weisfeiler_lehman_graph_hash(
                self.nx_graph,
                node_attr="label",
                edge_attr="label",
                iterations=max(3, len(self.node_labels)),
            )
        return (
            self.family,
            self.directed,
            len(self.node_labels),
            len(self.edges),
            digest,
        )


def exact_equivalent(left: AttributedComponent, right: AttributedComponent) -> bool:
    """Require exact attributed graph isomorphism inside a safe hash bucket."""

    if left.bucket_key != right.bucket_key:
        return False
    node_match = nx.algorithms.isomorphism.categorical_node_match("label", None)
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("label", None)
    matcher = (
        nx.algorithms.isomorphism.DiGraphMatcher
        if left.directed
        else nx.algorithms.isomorphism.GraphMatcher
    )
    return bool(
        matcher(
            left.nx_graph,
            right.nx_graph,
            node_match=node_match,
            edge_match=edge_match,
        ).is_isomorphic()
    )


class ExactComponentVocabulary:
    """Small exact-isomorphism vocabulary with deterministic bucket indexing."""

    def __init__(self, components=()):
        self._buckets: dict[tuple, list[AttributedComponent]] = defaultdict(list)
        for component in components:
            self.add(component)

    def add(self, component: AttributedComponent) -> bool:
        rows = self._buckets[component.bucket_key]
        if any(exact_equivalent(component, row) for row in rows):
            return False
        rows.append(component)
        return True

    def contains(self, component: AttributedComponent) -> bool:
        return any(
            exact_equivalent(component, row)
            for row in self._buckets.get(component.bucket_key, ())
        )

    @property
    def size(self) -> int:
        return sum(map(len, self._buckets.values()))

    @property
    def representatives(self) -> tuple[AttributedComponent, ...]:
        return tuple(
            row for key in sorted(self._buckets, key=str) for row in self._buckets[key]
        )


def _component(
    family: str,
    labels: list[object],
    edges: list[tuple[int, int, object]],
    *,
    directed: bool = False,
) -> AttributedComponent:
    by_pair: dict[tuple[int, int], list[str]] = defaultdict(list)
    for left, right, label in edges:
        if not directed and left > right:
            left, right = right, left
        by_pair[left, right].append(_label(label))
    encoded_edges = [
        (left, right, _label(tuple(sorted(labels))))
        for (left, right), labels in by_pair.items()
    ]
    return AttributedComponent(
        family=family,
        node_labels=tuple(_label(row) for row in labels),
        edges=tuple(sorted(encoded_edges)),
        directed=directed,
    )


def _target_roles(template: StructuralDeltaTemplate) -> tuple[int, ...]:
    n_input = len(template.input_atoms)
    return tuple(
        index
        for index in range(n_input + len(template.output_atoms))
        if index >= n_input or template.target_atoms[index] is not None
    )


def _target_atom(template: StructuralDeltaTemplate, role: int) -> tuple[int, ...]:
    n_input = len(template.input_atoms)
    value = (
        template.target_atoms[role]
        if role < n_input
        else template.output_atoms[role - n_input]
    )
    if value is None:
        raise ValueError("deleted input role is absent from the target graph")
    return value


def whole_patch_component(template: StructuralDeltaTemplate) -> AttributedComponent:
    """Joint before/after graph delta, used as the complete-patch control."""

    n_input = len(template.input_atoms)
    labels = [
        ("input", template.input_atoms[index], template.target_atoms[index])
        for index in range(n_input)
    ]
    labels.extend(("output", None, value) for value in template.output_atoms)
    edges = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            before = (
                template.input_bonds[left][right]
                if left < n_input and right < n_input
                else 0
            )
            after = template.target_bonds[left][right]
            if before or after:
                edges.append((left, right, (before, after)))
    return _component("whole_patch", labels, edges)


def source_region_motif(template: StructuralDeltaTemplate) -> AttributedComponent:
    labels = [("source", atom[0], atom[3]) for atom in template.input_atoms]
    edges = [
        (left, right, template.input_bonds[left][right])
        for left in range(len(labels))
        for right in range(left + 1, len(labels))
        if template.input_bonds[left][right]
    ]
    return _component("source_region_motif", labels, edges)


def target_topology(template: StructuralDeltaTemplate) -> AttributedComponent:
    n_input = len(template.input_atoms)
    roles = _target_roles(template)
    local = {role: index for index, role in enumerate(roles)}
    labels = [("input" if role < n_input else "output",) for role in roles]
    edges = [
        (local[left], local[right], ("bond",))
        for index, left in enumerate(roles)
        for right in roles[index + 1 :]
        if template.target_bonds[left][right]
    ]
    return _component("target_topology", labels, edges)


def atom_attributes(template: StructuralDeltaTemplate) -> AttributedComponent:
    labels = [
        ("input_transition", before, after)
        for before, after in zip(
            template.input_atoms, template.target_atoms, strict=True
        )
    ]
    labels.extend(("output", value) for value in template.output_atoms)
    return _component("atom_attributes", labels, [])


def atom_attribute_tokens(
    template: StructuralDeltaTemplate,
) -> tuple[AttributedComponent, ...]:
    labels = [
        ("input_transition", before, after)
        for before, after in zip(
            template.input_atoms, template.target_atoms, strict=True
        )
    ]
    labels.extend(("output", value) for value in template.output_atoms)
    return tuple(_component("atom_attribute_tokens", [label], []) for label in labels)


def bond_attributes(template: StructuralDeltaTemplate) -> AttributedComponent:
    n_input = len(template.input_atoms)
    n_target = len(template.target_bonds)
    labels = []
    for left in range(n_target):
        for right in range(left + 1, n_target):
            before = (
                template.input_bonds[left][right]
                if left < n_input and right < n_input
                else 0
            )
            after = template.target_bonds[left][right]
            if not before and not after:
                continue
            edge_kind = (
                "input_input"
                if right < n_input
                else "input_output" if left < n_input else "output_output"
            )
            labels.append((edge_kind, before, after))
    return _component("bond_attributes", labels, [])


def bond_attribute_tokens(
    template: StructuralDeltaTemplate,
) -> tuple[AttributedComponent, ...]:
    bundled = bond_attributes(template)
    if not bundled.node_labels:
        return (_component("bond_attribute_tokens", [("no_bond_delta",)], []),)
    return tuple(
        AttributedComponent("bond_attribute_tokens", (label,), ())
        for label in bundled.node_labels
    )


def attachment_pattern(template: StructuralDeltaTemplate) -> AttributedComponent:
    n_input = len(template.input_atoms)
    inputs = [
        role
        for role in range(n_input)
        if template.target_atoms[role] is not None
        and any(
            template.target_bonds[role][n_input + output]
            for output in range(len(template.output_atoms))
        )
    ]
    outputs = list(range(n_input, n_input + len(template.output_atoms)))
    roles = [*inputs, *outputs]
    local = {role: index for index, role in enumerate(roles)}
    labels = []
    for role in roles:
        target_degree = sum(bool(value) for value in template.target_bonds[role])
        if role < n_input:
            labels.append(("input", template.input_atoms[role][3], target_degree))
        else:
            labels.append(("output", target_degree))
    edges = [
        (local[left], local[right], template.target_bonds[left][right])
        for left in inputs
        for right in outputs
        if template.target_bonds[left][right]
    ]
    return _component("attachment_pattern", labels, edges)


def attachment_edge_tokens(
    template: StructuralDeltaTemplate,
) -> tuple[AttributedComponent, ...]:
    n_input = len(template.input_atoms)
    rows = []
    for left in range(n_input):
        if template.target_atoms[left] is None:
            continue
        source_degree = template.input_atoms[left][3]
        target_degree = sum(bool(value) for value in template.target_bonds[left])
        for output in range(len(template.output_atoms)):
            right = n_input + output
            order = template.target_bonds[left][right]
            if not order:
                continue
            output_degree = sum(bool(value) for value in template.target_bonds[right])
            rows.append(
                _component(
                    "attachment_edge_tokens",
                    [
                        ("input", source_degree, target_degree),
                        ("output", output_degree),
                    ],
                    [(0, 1, order)],
                )
            )
    return tuple(rows) or (
        _component("attachment_edge_tokens", [("no_created_attachment",)], []),
    )


def dependency_motif(
    actions: tuple[dict, ...],
    component: dict,
    *,
    created_dependency_edges: tuple[dict, ...],
    cycle_dependency_edges: tuple[dict, ...],
) -> AttributedComponent:
    """Created-handle and cycle dependencies, without primitive addresses."""

    primitive_indices = tuple(map(int, component["primitive_indices"]))
    local = {primitive: index for index, primitive in enumerate(primitive_indices)}
    labels = [
        (str(actions[primitive]["executor_rule"]),) for primitive in primitive_indices
    ]
    edges = []
    for row in created_dependency_edges:
        producer, consumer = int(row["producer"]), int(row["consumer"])
        if producer in local and consumer in local:
            edges.append((local[producer], local[consumer], ("created_handle",)))
    for row in cycle_dependency_edges:
        opened, closed = int(row["open"]), int(row["close_or_restate"])
        if opened in local and closed in local:
            edges.append((local[opened], local[closed], ("cycle_open_close",)))
    return _component("dependency_motif", labels, edges, directed=True)


def dependency_tokens(
    dependency: AttributedComponent,
) -> tuple[AttributedComponent, ...]:
    rows = [
        AttributedComponent(
            family="dependency_tokens",
            node_labels=(
                dependency.node_labels[left],
                dependency.node_labels[right],
            ),
            edges=((0, 1, label),),
            directed=True,
        )
        for left, right, label in dependency.edges
    ]
    return tuple(rows) or (
        _component(
            "dependency_tokens",
            [("no_created_or_cycle_dependency",)],
            [],
            directed=True,
        ),
    )


def _target_graph(template: StructuralDeltaTemplate) -> nx.Graph:
    n_input = len(template.input_atoms)
    graph = nx.Graph()
    for role in _target_roles(template):
        graph.add_node(
            role,
            role_type="input" if role < n_input else "output",
            atom=_target_atom(template, role),
        )
    for left, right in itertools.combinations(graph.nodes, 2):
        order = template.target_bonds[left][right]
        if order:
            graph.add_edge(left, right, bond=order)
    return graph


def _source_graph(template: StructuralDeltaTemplate) -> nx.Graph:
    graph = nx.Graph()
    for role, atom in enumerate(template.input_atoms):
        graph.add_node(role, role_type="input", atom=atom)
    for left, right in itertools.combinations(graph.nodes, 2):
        order = template.input_bonds[left][right]
        if order:
            graph.add_edge(left, right, bond=order)
    return graph


def _induced_target_component(
    family: str,
    graph: nx.Graph,
    roles: tuple[int, ...],
    *,
    root: int | None = None,
) -> AttributedComponent:
    local = {role: index for index, role in enumerate(roles)}
    labels = [
        (
            graph.nodes[role]["role_type"],
            graph.nodes[role]["atom"],
            graph.degree[role],
            role == root if root is not None else False,
        )
        for role in roles
    ]
    edges = [
        (local[left], local[right], graph.edges[left, right]["bond"])
        for left, right in graph.subgraph(roles).edges
    ]
    return _component(family, labels, edges)


def _graphlets(
    graph: nx.Graph,
    *,
    family: str,
    include_atom_attributes: bool,
    include_bond_attributes: bool,
) -> tuple[AttributedComponent, ...]:
    fragments = []
    nodes = tuple(sorted(graph.nodes))
    for size in range(1, min(3, len(nodes)) + 1):
        for roles in itertools.combinations(nodes, size):
            if size > 1 and not nx.is_connected(graph.subgraph(roles)):
                continue
            local = {role: index for index, role in enumerate(roles)}
            labels = [
                (
                    graph.nodes[role]["role_type"],
                    graph.nodes[role]["atom"] if include_atom_attributes else None,
                    graph.degree[role],
                )
                for role in roles
            ]
            edges = [
                (
                    local[left],
                    local[right],
                    (
                        graph.edges[left, right]["bond"]
                        if include_bond_attributes
                        else ("bond",)
                    ),
                )
                for left, right in graph.subgraph(roles).edges
            ]
            fragments.append(_component(family, labels, edges))
    return tuple(fragments)


def source_graphlets_up_to_3(
    template: StructuralDeltaTemplate,
) -> tuple[AttributedComponent, ...]:
    return _graphlets(
        _source_graph(template),
        family="source_graphlets_up_to_3",
        include_atom_attributes=True,
        include_bond_attributes=True,
    )


def target_topology_graphlets_up_to_3(
    template: StructuralDeltaTemplate,
) -> tuple[AttributedComponent, ...]:
    return _graphlets(
        _target_graph(template),
        family="target_topology_graphlets_up_to_3",
        include_atom_attributes=False,
        include_bond_attributes=False,
    )


def target_radius_fragments(
    template: StructuralDeltaTemplate, radius: int
) -> tuple[AttributedComponent, ...]:
    if radius not in (1, 2):
        raise ValueError("target fragment radius must be one or two")
    graph = _target_graph(template)
    family = f"target_radius_{radius}_fragments"
    fragments = []
    for root in sorted(graph.nodes):
        lengths = nx.single_source_shortest_path_length(graph, root, cutoff=radius)
        fragments.append(
            _induced_target_component(family, graph, tuple(sorted(lengths)), root=root)
        )
    return tuple(fragments)


def target_graphlets_up_to_3(
    template: StructuralDeltaTemplate,
) -> tuple[AttributedComponent, ...]:
    return _graphlets(
        _target_graph(template),
        family="target_graphlets_up_to_3",
        include_atom_attributes=True,
        include_bond_attributes=True,
    )


def component_families(
    template: StructuralDeltaTemplate,
    dependency: AttributedComponent,
) -> dict[str, tuple[AttributedComponent, ...]]:
    """Return the frozen component decomposition for one structural delta."""

    if dependency.family != "dependency_motif":
        raise ValueError("structural delta requires one dependency motif")
    return {
        "whole_patch": (whole_patch_component(template),),
        "source_region_motif": (source_region_motif(template),),
        "target_topology": (target_topology(template),),
        "atom_attributes": (atom_attributes(template),),
        "atom_attribute_tokens": atom_attribute_tokens(template),
        "bond_attributes": (bond_attributes(template),),
        "bond_attribute_tokens": bond_attribute_tokens(template),
        "attachment_pattern": (attachment_pattern(template),),
        "attachment_edge_tokens": attachment_edge_tokens(template),
        "dependency_motif": (dependency,),
        "dependency_tokens": dependency_tokens(dependency),
        "source_graphlets_up_to_3": source_graphlets_up_to_3(template),
        "target_topology_graphlets_up_to_3": target_topology_graphlets_up_to_3(
            template
        ),
        "target_radius_1_fragments": target_radius_fragments(template, 1),
        "target_radius_2_fragments": target_radius_fragments(template, 2),
        "target_graphlets_up_to_3": target_graphlets_up_to_3(template),
    }


CORE_COMPONENT_FAMILIES = (
    "source_region_motif",
    "target_topology",
    "atom_attributes",
    "bond_attributes",
    "attachment_pattern",
    "dependency_motif",
)

GRANULAR_COMPONENT_FAMILIES = (
    "source_graphlets_up_to_3",
    "target_topology_graphlets_up_to_3",
    "atom_attribute_tokens",
    "bond_attribute_tokens",
    "attachment_edge_tokens",
    "dependency_tokens",
)


__all__ = [
    "CORE_COMPONENT_FAMILIES",
    "GRANULAR_COMPONENT_FAMILIES",
    "AttributedComponent",
    "ExactComponentVocabulary",
    "component_families",
    "dependency_motif",
    "exact_equivalent",
    "whole_patch_component",
]
