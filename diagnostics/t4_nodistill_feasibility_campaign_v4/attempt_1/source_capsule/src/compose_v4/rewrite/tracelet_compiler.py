"""Compile molecules into block-aware, validity-closed ring tracelets."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from compose_v4.chem.aromaticity import perceived_aromatic_ring_count
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_CLASS_TO_H_CHANGE,
    ELEMENT_TO_IDX,
    MAX_H_COUNT,
    MolecularGraph,
    is_element,
)
from compose_v4.chem.state import empty_molecular_graph, is_valid_state
from compose_v4.rewrite.compiler import TraceCompilationError, compile_null_to_target
from compose_v4.rewrite.kernel import InvalidRewrite, RewriteSystem, de_novo_rewrite_system
from compose_v4.rewrite.operators import (
    AtomInsert,
    AtomRestate,
    BondReorder,
)
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.tracelets import (
    AtomPayload,
    BondOrderChange,
    CycleAttach,
    CycleInsert,
    RingEarInsert,
    RingSystemRestate,
    lower_cycle_attach,
    lower_cycle_insert,
    lower_ring_ear_insert,
    lower_ring_system_restate,
)

if TYPE_CHECKING:
    from compose_v4.rewrite.typed_ring_catalog import TypedRingCatalog


def compile_null_to_target_tracelets(
    target: MolecularGraph,
    *,
    system: RewriteSystem | None = None,
    typed_ring_payloads: bool = False,
    ring_catalog: "TypedRingCatalog | None" = None,
) -> RewriteTrace:
    """Compile through ring tracelets, with an optional fully typed diagnostic.

    The legacy path uses generic saturated-carbon carriers.  The typed path
    instead commits the target atom and bond labels inside each cycle/ear
    action.  When a finite proposal catalog does not support all typed marks,
    compilation falls back to the universal typed micro program so training
    and inference retain identical support.
    """

    runtime = system or de_novo_rewrite_system()
    try:
        trace = _compile_null_to_target_tracelets(
            target,
            system=runtime,
            typed_ring_payloads=typed_ring_payloads,
        )
    except (InvalidRewrite, TraceCompilationError):
        if not typed_ring_payloads:
            raise
        real = tuple(int(v) for v in np.flatnonzero(is_element(target.atom_types)))
        graph = _target_graph(target, real)
        if graph.number_of_edges() >= graph.number_of_nodes() and real:
            raise
        return compile_null_to_target(target, system=runtime)
    if typed_ring_payloads and ring_catalog is not None:
        if not ring_catalog.supports_trace(trace):
            raise TraceCompilationError(
                "typed ring catalog does not support the committed-topology trace"
            )
    return trace


def _compile_null_to_target_tracelets(
    target: MolecularGraph,
    *,
    system: RewriteSystem,
    typed_ring_payloads: bool,
) -> RewriteTrace:
    """Compile a connected target using cycles, ears, and electronic restates.

    The topology phase constructs every edge as a single bond. A cyclic block
    starts with a parameterized cycle seed and is extended by open ears. The
    electronic phase then realizes correlated ring-system bond patterns as one
    derived event. Every derived event has a checked micro lowering.
    """

    if not is_valid_state(target):
        raise TraceCompilationError("target state is invalid")
    if np.any(target.bonds == BOND_AROMATIC):
        raise TraceCompilationError("target contains non-Kekulized aromatic bonds")
    real = tuple(int(v) for v in np.flatnonzero(is_element(target.atom_types)))
    source = empty_molecular_graph(target.n_atoms)
    if not real:
        return RewriteTrace(source, target, (), {"compiler": "identity_null"})

    graph = _target_graph(target, real)
    if not nx.is_connected(graph):
        raise TraceCompilationError("tracelet compiler requires a connected target")
    runtime = system
    blocks = [frozenset(component) for component in nx.biconnected_components(graph)]
    block_edges = [
        frozenset(_edge(a, b) for a, b in graph.subgraph(vertices).edges())
        for vertices in blocks
    ]
    cyclic = [
        len(block_edges[index]) >= len(blocks[index])
        for index in range(len(blocks))
    ]
    incident: dict[int, list[int]] = defaultdict(list)
    for block_id, vertices in enumerate(blocks):
        for v in vertices:
            incident[int(v)].append(block_id)
    for values in incident.values():
        values.sort(key=lambda index: (not cyclic[index], tuple(sorted(blocks[index]))))

    cyclic_vertices = {
        int(v)
        for block_id, is_cyclic in enumerate(cyclic)
        if is_cyclic
        for v in blocks[block_id]
    }
    carbon = int(ELEMENT_TO_IDX["C"])

    saturated_h = _saturated_hydrogens(target, graph)
    state = source
    steps: list[RewriteStep] = []
    committed: set[frozenset[int]] = set()
    built: set[int] = set()
    processed_blocks: set[int] = set()
    lowered_steps = 0

    def commit(rule_name: str, action) -> None:
        nonlocal state, lowered_steps
        if rule_name == "cycle_insert":
            lowered_steps += len(lower_cycle_insert(state, action))
        elif rule_name == "cycle_attach":
            lowered_steps += len(lower_cycle_attach(state, action))
        elif rule_name == "ring_ear_insert":
            lowered_steps += len(lower_ring_ear_insert(state, action))
        elif rule_name == "ring_system_restate":
            lowered_steps += len(lower_ring_system_restate(state, action))
        else:
            lowered_steps += 1
        state = runtime.apply(state, rule_name, action)
        steps.append(RewriteStep(rule_name, action))

    def post_topology_h(v: int, committed_after: set[frozenset[int]]) -> int:
        if typed_ring_payloads:
            future_load = sum(
                int(BOND_CLASS_TO_H_CHANGE[int(target.bonds[v, u])])
                for u in graph.neighbors(v)
                if _edge(v, int(u)) not in committed_after
            )
            value = int(target.implicit_h_counts[v]) + int(future_load)
        else:
            future_edges = sum(
                _edge(v, int(u)) not in committed_after
                for u in graph.neighbors(v)
            )
            value = int(saturated_h[v]) + int(future_edges)
        if not 0 <= value <= MAX_H_COUNT:
            raise TraceCompilationError(
                f"slot {v} requires transient H={value} in tracelet topology"
            )
        return value

    def carbon_h(v: int, committed_after: set[frozenset[int]]) -> int:
        committed_degree = sum(
            _edge(v, int(u)) in committed_after for u in graph.neighbors(v)
        )
        value = 4 - int(committed_degree)
        if not 0 <= value <= MAX_H_COUNT:
            raise TraceCompilationError(
                f"slot {v} cannot serve as a saturated carbon ring carrier"
            )
        return value

    def payload(v: int, committed_after: set[frozenset[int]]) -> AtomPayload:
        if typed_ring_payloads:
            return AtomPayload(
                slot=int(v),
                atom_type=int(target.atom_types[v]),
                formal_charge=int(target.formal_charges[v]),
                implicit_h_count=post_topology_h(v, committed_after),
            )
        return AtomPayload(
            slot=int(v),
            atom_type=carbon,
            formal_charge=0,
            implicit_h_count=carbon_h(v, committed_after),
        )

    def topology_order(a: int, b: int) -> int:
        return int(target.bonds[a, b]) if typed_ring_payloads else 1

    def path_orders(path: tuple[int, ...], *, cyclic_path: bool = False) -> tuple[int, ...]:
        if cyclic_path:
            return tuple(
                topology_order(path[index], path[(index + 1) % len(path)])
                for index in range(len(path))
            )
        return tuple(
            topology_order(path[index], path[index + 1])
            for index in range(len(path) - 1)
        )

    def add_ear(path: tuple[int, ...]) -> None:
        if len(path) < 3:
            raise TraceCompilationError("an open ear needs at least one internal atom")
        path_edges = {_edge(path[i], path[i + 1]) for i in range(len(path) - 1)}
        committed_after = committed | path_edges
        action = RingEarInsert(
            a=path[0],
            b=path[-1],
            atoms=tuple(payload(v, committed_after) for v in path[1:-1]),
            bond_orders=path_orders(path),
        )
        commit("ring_ear_insert", action)
        built.update(path[1:-1])
        committed.update(path_edges)

    def finish_cyclic_block(block_id: int) -> None:
        vertices = set(blocks[block_id])
        while vertices - built:
            path = _next_open_ear(graph, vertices, built)
            add_ear(path)
        for edge in sorted(block_edges[block_id], key=_edge_sort_key):
            if edge in committed:
                continue
            a, b = sorted(edge)
            commit(
                "ring_ear_insert",
                RingEarInsert(
                    a=a,
                    b=b,
                    atoms=(),
                    bond_orders=(topology_order(a, b),),
                ),
            )
            committed.add(edge)
        processed_blocks.add(block_id)
        for v in sorted(vertices):
            for child in incident[v]:
                if child not in processed_blocks:
                    process_block(child, v)

    def process_block(block_id: int, entry: int) -> None:
        if block_id in processed_blocks:
            return
        vertices = set(blocks[block_id])
        if entry not in vertices or entry not in built:
            raise TraceCompilationError("block traversal lost its built articulation")
        if cyclic[block_id]:
            overlap = vertices & built
            if overlap != {entry}:
                raise TraceCompilationError(
                    "cyclic child block must meet the prefix at one articulation"
                )
            cycle = _cycle_through_vertex(graph.subgraph(vertices), entry)
            cycle_edges = {
                _edge(cycle[i], cycle[(i + 1) % len(cycle)])
                for i in range(len(cycle))
            }
            committed_after = committed | cycle_edges
            action = RingEarInsert(
                a=entry,
                b=entry,
                atoms=tuple(payload(v, committed_after) for v in cycle[1:]),
                bond_orders=path_orders(cycle, cyclic_path=True),
            )
            commit("ring_ear_insert", action)
            built.update(cycle[1:])
            committed.update(cycle_edges)
            finish_cyclic_block(block_id)
            return

        if len(vertices) != 2 or len(block_edges[block_id]) != 1:
            raise TraceCompilationError("non-cyclic block is not a bridge edge")
        other = next(v for v in vertices if v != entry)
        edge = _edge(entry, other)
        if other in built:
            raise TraceCompilationError("bridge traversal encountered a built child")
        committed_after = committed | {edge}
        cyclic_children = [
            child
            for child in incident[other]
            if child not in processed_blocks and child != block_id and cyclic[child]
        ]
        if cyclic_children:
            # Commit a pendant cyclic block already closed.  In particular,
            # the attachment atom never appears first as an acyclic chain atom
            # whose topological role is reinterpreted by a later closure.
            child = cyclic_children[0]
            cycle = _cycle_through_vertex(graph.subgraph(blocks[child]), other)
            cycle_edges = {
                _edge(cycle[i], cycle[(i + 1) % len(cycle)])
                for i in range(len(cycle))
            }
            committed_after = committed | {edge} | cycle_edges
            commit(
                "cycle_attach",
                CycleAttach(
                    anchor=entry,
                    atoms=tuple(payload(v, committed_after) for v in cycle),
                    bond_orders=path_orders(cycle, cyclic_path=True),
                    attachment_order=topology_order(entry, other),
                ),
            )
            built.update(cycle)
            committed.update(cycle_edges)
            committed.add(edge)
            processed_blocks.add(block_id)
            finish_cyclic_block(child)
            return
        if typed_ring_payloads:
            atom_type = int(target.atom_types[other])
            formal_charge = int(target.formal_charges[other])
            implicit_h_count = post_topology_h(other, committed_after)
        elif other in cyclic_vertices:
            atom_type = carbon
            formal_charge = 0
            implicit_h_count = carbon_h(other, committed_after)
        else:
            atom_type = int(target.atom_types[other])
            formal_charge = int(target.formal_charges[other])
            implicit_h_count = post_topology_h(other, committed_after)
        commit(
            "atom_insert",
            AtomInsert(
                slot=other,
                atom_type=atom_type,
                formal_charge=formal_charge,
                implicit_h_count=implicit_h_count,
                neighbors=((entry, topology_order(entry, other)),),
            ),
        )
        built.add(other)
        committed.add(edge)
        processed_blocks.add(block_id)
        for child in incident[other]:
            if child not in processed_blocks:
                process_block(child, other)

    cyclic_blocks = [index for index, is_cyclic in enumerate(cyclic) if is_cyclic]
    if cyclic_blocks:
        root_block = min(
            cyclic_blocks,
            key=lambda index: (len(blocks[index]), tuple(sorted(blocks[index]))),
        )
        root = min(blocks[root_block])
        cycle = _cycle_through_vertex(graph.subgraph(blocks[root_block]), root)
        cycle_edges = {
            _edge(cycle[i], cycle[(i + 1) % len(cycle)])
            for i in range(len(cycle))
        }
        committed_after = set(cycle_edges)
        commit(
            "cycle_insert",
            CycleInsert(
                atoms=tuple(payload(v, committed_after) for v in cycle),
                bond_orders=path_orders(cycle, cyclic_path=True),
            ),
        )
        built.update(cycle)
        committed.update(cycle_edges)
        finish_cyclic_block(root_block)
    else:
        root = min(real)
        root_h = post_topology_h(root, set())
        commit(
            "atom_insert",
            AtomInsert(
                slot=root,
                atom_type=int(target.atom_types[root]),
                formal_charge=int(target.formal_charges[root]),
                implicit_h_count=root_h,
                neighbors=(),
            ),
        )
        built.add(root)
        for block_id in incident[root]:
            process_block(block_id, root)

    target_edges = {_edge(a, b) for a, b in graph.edges()}
    if built != set(real) or committed != target_edges:
        raise TraceCompilationError("tracelet topology did not cover the target graph")

    if typed_ring_payloads:
        if not _same_state(state, target):
            raise TraceCompilationError(
                "typed tracelet topology did not reconstruct the exact target"
            )
        counts = defaultdict(int)
        for step in steps:
            counts[step.rule_name] += 1
        return RewriteTrace(
            source=source,
            target=target,
            steps=tuple(steps),
            metadata={
                "compiler": "typed_block_ear_tracelets_v1",
                "visible_steps": len(steps),
                "micro_lowered_steps": lowered_steps,
                "cycle_insert_steps": counts["cycle_insert"],
                "cycle_attach_steps": counts["cycle_attach"],
                "ring_ear_insert_steps": counts["ring_ear_insert"],
                "ring_system_restate_steps": 0,
                "ring_atom_restate_steps": 0,
                "micro_fallback_steps": len(steps)
                - counts["cycle_insert"]
                - counts["cycle_attach"]
                - counts["ring_ear_insert"],
            },
        )

    # Ring topology is now explicit. Refine the neutral carbon carrier atoms
    # while every atom-level rewrite can condition on actual ring membership,
    # ring size, and fusion context.
    for v in sorted(cyclic_vertices):
        desired = (
            int(target.atom_types[v]),
            int(target.formal_charges[v]),
            int(saturated_h[v]),
        )
        current = (
            int(state.atom_types[v]),
            int(state.formal_charges[v]),
            int(state.implicit_h_counts[v]),
        )
        if current == desired:
            continue
        commit(
            "atom_restate",
            AtomRestate(
                v=v,
                atom_type=desired[0],
                formal_charge=desired[1],
                implicit_h_count=desired[2],
            ),
        )

    bridges = {_edge(a, b) for a, b in nx.bridges(graph)}
    ring_graph = graph.copy()
    ring_graph.remove_edges_from(tuple(tuple(edge) for edge in bridges))
    electronically_committed: set[frozenset[int]] = set()
    for vertices in sorted(
        nx.connected_components(ring_graph),
        key=lambda component: tuple(sorted(component)),
    ):
        component = ring_graph.subgraph(vertices)
        changes = tuple(
            BondOrderChange(a, b, int(target.bonds[a, b]))
            for a, b in sorted(component.edges())
            if int(target.bonds[a, b]) != 1
        )
        # Reserve a derived electronic event for a genuinely coordinated
        # maximum matching (the common Kekule/conjugated case). Partially
        # unsaturated, cumulated, triple-bonded, and otherwise exotic ring
        # patterns retain exact support through the universal micro fallback.
        if _is_coordinated_ring_pattern(component, changes):
            action = RingSystemRestate(changes)
            try:
                successor = runtime.apply(state, "ring_system_restate", action)
            except InvalidRewrite:
                successor = None
            if (
                successor is not None
                and perceived_aromatic_ring_count(successor)
                > perceived_aromatic_ring_count(state)
            ):
                commit("ring_system_restate", action)
                electronically_committed.update(
                    _edge(item.a, item.b) for item in changes
                )

    for a, b in sorted(graph.edges()):
        edge = _edge(a, b)
        order = int(target.bonds[a, b])
        if order == 1 or edge in electronically_committed:
            continue
        commit("bond_reorder", BondReorder(a, b, order))

    if not _same_state(state, target):
        raise TraceCompilationError("tracelet compiler did not reconstruct exact target")
    counts = defaultdict(int)
    for step in steps:
        counts[step.rule_name] += 1
    return RewriteTrace(
        source=source,
        target=target,
        steps=tuple(steps),
        metadata={
            "compiler": "block_ear_aromatic_closure_tracelets_v2",
            "visible_steps": len(steps),
            "micro_lowered_steps": lowered_steps,
            "cycle_insert_steps": counts["cycle_insert"],
            "cycle_attach_steps": counts["cycle_attach"],
            "ring_ear_insert_steps": counts["ring_ear_insert"],
            "ring_system_restate_steps": counts["ring_system_restate"],
            "ring_atom_restate_steps": counts["atom_restate"],
            "micro_fallback_steps": len(steps)
            - counts["cycle_insert"]
            - counts["cycle_attach"]
            - counts["ring_ear_insert"]
            - counts["ring_system_restate"],
        },
    )


def _target_graph(target: MolecularGraph, real: tuple[int, ...]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (a, b)
        for a in real
        for b in real
        if b > a and int(target.bonds[a, b]) != 0
    )
    return graph


def _saturated_hydrogens(
    target: MolecularGraph,
    graph: nx.Graph,
) -> dict[int, int]:
    result = {}
    for v in graph.nodes():
        value = int(target.implicit_h_counts[v]) + sum(
            int(BOND_CLASS_TO_H_CHANGE[int(target.bonds[v, u])]) - 1
            for u in graph.neighbors(v)
        )
        if not 0 <= value <= MAX_H_COUNT:
            raise TraceCompilationError(
                f"slot {v} cannot be represented by a saturated topology precursor"
            )
        result[int(v)] = value
    return result


def _cycle_through_vertex(graph: nx.Graph, root: int) -> tuple[int, ...]:
    if root not in graph or graph.degree(root) < 2:
        raise TraceCompilationError("cyclic block root does not lie on a cycle")
    without_root = graph.copy()
    neighbors = sorted(without_root.neighbors(root))
    without_root.remove_node(root)
    candidates = []
    for left, right in combinations(neighbors, 2):
        try:
            path = nx.shortest_path(without_root, left, right)
        except nx.NetworkXNoPath:
            continue
        cycle = (int(root), *(int(v) for v in path))
        candidates.append(cycle)
    if not candidates:
        raise TraceCompilationError("could not find an initial cycle through block root")
    return min(candidates, key=lambda cycle: (len(cycle), cycle))


def _next_open_ear(
    graph: nx.Graph,
    block_vertices: set[int],
    built: set[int],
) -> tuple[int, ...]:
    unbuilt = block_vertices - built
    if not unbuilt:
        raise TraceCompilationError("open-ear search was called on a complete block")
    candidates = []
    for component in nx.connected_components(graph.subgraph(unbuilt)):
        component = set(component)
        boundary = sorted(
            {
                int(neighbor)
                for v in component
                for neighbor in graph.neighbors(v)
                if neighbor in built and neighbor in block_vertices
            }
        )
        for left, right in combinations(boundary, 2):
            induced = graph.subgraph(component | {left, right}).copy()
            # The anchors may already share a bond (the usual fused-ring
            # case). Remove it so shortest-path search must traverse the new
            # internal vertices rather than returning the old zero-length ear.
            if induced.has_edge(left, right):
                induced.remove_edge(left, right)
            try:
                path = tuple(int(v) for v in nx.shortest_path(induced, left, right))
            except nx.NetworkXNoPath:
                continue
            if len(path) >= 3 and all(v in component for v in path[1:-1]):
                candidates.append(path)
    if not candidates:
        raise TraceCompilationError("cyclic block admitted no next open ear")
    return min(candidates, key=lambda path: (len(path), path))


def _same_state(left: MolecularGraph, right: MolecularGraph) -> bool:
    return bool(
        np.array_equal(left.atom_types, right.atom_types)
        and np.array_equal(left.formal_charges, right.formal_charges)
        and np.array_equal(left.implicit_h_counts, right.implicit_h_counts)
        and np.array_equal(left.bonds, right.bonds)
    )


def _is_coordinated_ring_pattern(
    component: nx.Graph,
    changes: tuple[BondOrderChange, ...],
) -> bool:
    """Return whether ``changes`` form a maximum double-bond matching.

    This is a structural, representation-level criterion rather than a claim
    that every admitted ring is aromatic. It keeps the macro mark space small
    and delegates non-Kekule electronic patterns to complete micro rewrites.
    """

    if len(changes) < 2 or any(int(change.new_order) != 2 for change in changes):
        return False
    changed_edges = tuple((int(change.a), int(change.b)) for change in changes)
    occupied: set[int] = set()
    for a, b in changed_edges:
        if a in occupied or b in occupied:
            return False
        occupied.update((a, b))
    maximum_size = len(nx.max_weight_matching(component, maxcardinality=True))
    return len(changed_edges) == maximum_size


def _edge(a: int, b: int) -> frozenset[int]:
    return frozenset((int(a), int(b)))


def _edge_sort_key(edge: frozenset[int]) -> tuple[int, int]:
    a, b = sorted(edge)
    return a, b
