"""Validity-closed pilot compiler from alkane-tree noise to data molecules."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import networkx as nx
import numpy as np

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import (
    BOND_AROMATIC,
    BOND_CLASS_TO_H_CHANGE,
    BOND_SINGLE,
    ELEMENT_TO_IDX,
    NULL_IDX,
    MolecularGraph,
    is_element,
)
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.kernel import RewriteSystem, de_novo_rewrite_system
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    AtomRestate,
    BondDelete,
    BondReorder,
    BondReroute,
)
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.tracelet_compiler import (
    TraceCompilationError,
    compile_null_to_target_tracelets,
)
from compose_v4.rewrite.tracelets import (
    AtomPayload,
    BondOrderChange,
    CycleInsert,
    RingBond,
    RingEarInsert,
    RingSystemGrow,
)
from compose_v4.rewrite.typed_ring_catalog import TypedRingCatalog


def compile_carbon_tree_to_target(
    source: MolecularGraph,
    target: MolecularGraph,
    *,
    system: RewriteSystem | None = None,
    use_bond_reroute: bool = False,
    align_source: bool = False,
    flexible_size: bool = False,
    typed_ring_payloads: bool = False,
    ring_catalog: TypedRingCatalog | None = None,
) -> RewriteTrace:
    """Compile a carbon tree into ``target`` through valid connected states.

    The default primitive path peels the random tree to one retained carbon and
    regrows the target.  The equal-size Graft path retains every atom.  Flexible
    Graft transport first grows or shrinks the independently sampled tree to
    the target cardinality, then uses the same atom-retaining topology and ring
    compiler.  Every committed state remains valid and connected.
    """

    runtime = system or de_novo_rewrite_system()
    _validate_transport_pair(
        source,
        target,
        require_aligned_size=use_bond_reroute and not flexible_size,
    )
    if flexible_size and not use_bond_reroute:
        raise ValueError("flexible-size transport requires Graft support")
    if align_source and not use_bond_reroute:
        raise ValueError("source alignment is only defined for Graft transport")
    if flexible_size:
        return _compile_flexible_graft_tree_transport(
            source,
            target,
            runtime=runtime,
            ring_catalog=ring_catalog,
        )
    if use_bond_reroute:
        return _compile_graft_tree_transport(
            source,
            target,
            runtime=runtime,
            ring_catalog=ring_catalog,
        )
    target_trace = compile_null_to_target_tracelets(
        target,
        system=runtime,
        typed_ring_payloads=typed_ring_payloads,
        ring_catalog=ring_catalog,
    )
    return _compile_primitive_tree_transport(
        source,
        target,
        target_trace=target_trace,
        runtime=runtime,
    )


def _compile_flexible_graft_tree_transport(
    source: MolecularGraph,
    target: MolecularGraph,
    *,
    runtime: RewriteSystem,
    ring_catalog: TypedRingCatalog | None,
) -> RewriteTrace:
    """Resize an independent tree, then couple it in the molecular quotient.

    Raw padded slot labels are gauge: the rate model is permutation equivariant
    and the directly sampled carbon-tree prior is exchangeable.  We therefore
    compile the resize path first, let the equal-size quotient compiler choose
    one global source-to-target slot correspondence, and conjugate the visible
    resize actions by that same correspondence.  No hidden mid-trace relabeling
    occurs, and the trace still replays to the array-exact corpus target.
    """

    original_source = source
    state = source
    resize_steps: list[RewriteStep] = []
    source_size = source.n_real_atoms
    target_size = target.n_real_atoms

    def commit(rule_name: str, action) -> None:
        nonlocal state
        state = runtime.apply(state, rule_name, action)
        resize_steps.append(RewriteStep(rule_name, action))

    source_real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    target_real = tuple(int(v) for v in np.flatnonzero(is_element(target.atom_types)))
    if source_size > target_size:
        # Slot labels are aligned only after resizing, so any deterministic
        # leaf-peeling sequence is a valid quotient representative.  Unlike
        # the old path canonicalization, this adds no unobservable Grafts.
        source_tree = _topology_graph(state, source_real)
        while source_tree.number_of_nodes() > target_size:
            leaf = min(int(v) for v, degree in source_tree.degree() if degree == 1)
            commit("atom_delete", AtomDelete(leaf))
            source_tree.remove_node(leaf)
        resize_strategy = "direct_leaf_shrink"
    elif source_size < target_size:
        # Grow a carbon tail from a current leaf.  The newly inserted atom is
        # the next leaf, so all cardinality updates remain local and valid.
        source_tree = _topology_graph(state, source_real)
        if len(source_real) == 1:
            anchor = int(source_real[0])
        else:
            anchor = min(int(v) for v, degree in source_tree.degree() if degree == 1)
        carbon = int(ELEMENT_TO_IDX["C"])
        grow_slots = tuple(
            int(v) for v in np.flatnonzero(state.atom_types == NULL_IDX)
        )[: target_size - source_size]
        for vertex in grow_slots:
            commit(
                "atom_insert",
                AtomInsert(
                    slot=int(vertex),
                    atom_type=carbon,
                    formal_charge=0,
                    implicit_h_count=3,
                    neighbors=((anchor, BOND_SINGLE),),
                ),
            )
            anchor = int(vertex)
        resize_strategy = "leaf_tail_grow"
    else:
        resize_strategy = "identity_size"

    if state.n_real_atoms != target_size:
        raise TraceCompilationError("flexible Graft resize reached the wrong size")

    # The suffix owns the one global quotient alignment and contains only
    # chemically observable topology changes.  Recover that alignment as an
    # isomorphism between the raw resized tree and the suffix source, then use
    # it for the original source and every resize action.
    suffix = _compile_graft_tree_transport(
        state,
        target,
        runtime=runtime,
        ring_catalog=ring_catalog,
    )
    resized_real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    suffix_real = tuple(
        int(v) for v in np.flatnonzero(is_element(suffix.source.atom_types))
    )
    resized_tree = _topology_graph(state, resized_real)
    suffix_tree = _topology_graph(suffix.source, suffix_real)
    quotient_mapping = _lexicographic_isomorphism(resized_tree, suffix_tree)

    source_mapping = {
        int(vertex): int(quotient_mapping[vertex])
        for vertex in source_real
        if vertex in quotient_mapping
    }
    deleted_vertices = tuple(v for v in source_real if v not in source_mapping)
    available_destinations = tuple(
        int(v)
        for v in range(original_source.n_atoms)
        if v not in set(quotient_mapping.values())
    )
    if len(available_destinations) < len(deleted_vertices):
        raise TraceCompilationError("quotient resize alignment ran out of padded slots")
    source_mapping.update(zip(deleted_vertices, available_destinations))
    aligned_source = _relabel_carbon_tree_state(original_source, source_mapping)
    mapped_resize_steps = tuple(
        RewriteStep(step.rule_name, _map_action_slots(step.action, quotient_mapping | source_mapping))
        for step in resize_steps
    )
    resized_endpoint = aligned_source
    for step in mapped_resize_steps:
        resized_endpoint = runtime.apply(resized_endpoint, step.rule_name, step.action)
    if not _same_state(resized_endpoint, suffix.source):
        raise TraceCompilationError("mapped resize prefix did not reach quotient suffix source")

    steps = (*mapped_resize_steps, *suffix.steps)
    endpoint = aligned_source
    for step in steps:
        endpoint = runtime.apply(endpoint, step.rule_name, step.action)
    if not _same_state(endpoint, target):
        raise TraceCompilationError("flexible Graft transport did not reach target")

    counts = Counter(step.rule_name for step in steps)
    trace = RewriteTrace(
        source=aligned_source,
        target=target,
        steps=tuple(steps),
        metadata={
            **suffix.metadata,
            "compiler": "flexible_size_graft_transport_v1",
            "compiler_revision": "quotient_common_topology_v2",
            "transport_strategy": "direct_resize_then_quotient_graft_restate_close",
            "resize_strategy": resize_strategy,
            "source_coupling": "molecular_quotient_slot_alignment",
            "source_heavy_atoms": source_size,
            "target_heavy_atoms": target_size,
            "size_delta": target_size - source_size,
            "visible_steps": len(steps),
            "atom_delete_steps": counts["atom_delete"],
            "atom_insert_steps": counts["atom_insert"],
            "bond_reroute_steps": counts["bond_reroute"],
            "atom_restate_steps": counts["atom_restate"],
            "bond_reorder_steps": counts["bond_reorder"],
            "ring_system_restate_steps": counts["ring_system_restate"],
        },
    )
    if ring_catalog is not None and not ring_catalog.supports_trace(trace):
        raise TraceCompilationError(
            "typed ring catalog does not support flexible Graft trace"
        )
    return trace


def _compile_graft_tree_transport(
    source: MolecularGraph,
    target: MolecularGraph,
    *,
    runtime: RewriteSystem,
    ring_catalog: TypedRingCatalog | None,
) -> RewriteTrace:
    """Compile an equal-size, atom-retaining tree-to-molecule path."""

    original_source = source
    steps: list[RewriteStep] = []

    real = tuple(int(v) for v in np.flatnonzero(is_element(target.atom_types)))
    target_graph = _topology_graph(target, real)
    source, target_tree, graft_actions, graft_planner = _plan_quotient_common_grafts(
        original_source,
        target_graph,
        real=real,
    )
    state = source

    def commit(rule_name: str, action) -> None:
        nonlocal state
        state = runtime.apply(state, rule_name, action)
        steps.append(RewriteStep(rule_name, action))

    target_tree_edges = {_edge(a, b) for a, b in target_tree.edges()}
    for action in graft_actions:
        commit("bond_reroute", action)
    # The target spanning-tree scaffold is obtained by removing only chords,
    # so each deletion preserves connectivity.  Its restored hydrogen counts
    # make it a valid molecule in its own right.
    scaffold = target
    closure_edges = []
    for a, b in sorted(target_graph.edges()):
        if _edge(a, b) in target_tree_edges:
            continue
        scaffold = runtime.apply(scaffold, "bond_delete", BondDelete(int(a), int(b)))
        closure_edges.append((int(a), int(b), int(target.bonds[a, b])))

    ring_systems = _ring_system_components(target_graph)
    ring_system_by_atom = {
        atom: system_id
        for system_id, members in enumerate(ring_systems)
        for atom in members
    }

    # All atoms are still carbon here, so raising retained tree bond orders is
    # valence-safe before installing lower-valence heteroatom identities.
    # Internal cyclic-system bonds are deliberately deferred into the atomic
    # RingSystemGrow event so the model never sees an independently decorated
    # chain that is later reinterpreted by a closure mark.
    for a, b in sorted(target_tree.edges()):
        same_ring_system = (
            a in ring_system_by_atom
            and ring_system_by_atom.get(a) == ring_system_by_atom.get(b)
        )
        if same_ring_system:
            continue
        desired_order = int(scaffold.bonds[a, b])
        if int(state.bonds[a, b]) != desired_order:
            commit("bond_reorder", BondReorder(int(a), int(b), desired_order))
    for v in real:
        if v in ring_system_by_atom:
            continue
        desired = (
            int(scaffold.atom_types[v]),
            int(scaffold.formal_charges[v]),
            int(scaffold.implicit_h_counts[v]),
        )
        present = (
            int(state.atom_types[v]),
            int(state.formal_charges[v]),
            int(state.implicit_h_counts[v]),
        )
        if present != desired:
            commit(
                "atom_restate",
                AtomRestate(
                    v=v,
                    atom_type=desired[0],
                    formal_charge=desired[1],
                    implicit_h_count=desired[2],
                ),
            )

    perceived_target = resonance_invariant_bond_classes(target)
    for system_id, system_atoms in enumerate(ring_systems):
        member_set = set(system_atoms)
        internal_reorders = tuple(
            BondOrderChange(int(a), int(b), int(scaffold.bonds[a, b]))
            for a, b in sorted(target_tree.edges())
            if a in member_set
            and b in member_set
            and int(state.bonds[a, b]) != int(scaffold.bonds[a, b])
        )
        atom_payloads = tuple(
            AtomPayload(
                slot=int(v),
                atom_type=int(scaffold.atom_types[v]),
                formal_charge=int(scaffold.formal_charges[v]),
                implicit_h_count=int(scaffold.implicit_h_counts[v]),
            )
            for v in system_atoms
        )
        insertions = tuple(
            RingBond(int(a), int(b), int(order))
            for a, b, order in closure_edges
            if ring_system_by_atom[a] == system_id
        )
        aromatic_edges = tuple(
            (int(a), int(b))
            for offset, a in enumerate(system_atoms)
            for b in system_atoms[offset + 1 :]
            if int(perceived_target[a, b]) == BOND_AROMATIC
        )
        commit(
            "ring_system_grow",
            RingSystemGrow(
                system_atoms=system_atoms,
                interface_atoms=(),
                scaffold_bonds=tuple(
                    RingBond(int(a), int(b), int(state.bonds[a, b]))
                    for offset, a in enumerate(system_atoms)
                    for b in system_atoms[offset + 1 :]
                    if int(state.bonds[a, b]) != 0
                ),
                bond_reorders=internal_reorders,
                atom_payloads=atom_payloads,
                atom_insertions=(),
                bond_insertions=insertions,
                source_aromatic_edges=(),
                aromatic_edges=aromatic_edges,
                topology_class=_ring_system_topology_class(
                    target_graph.subgraph(system_atoms)
                ),
            ),
        )
    if not _same_state(state, target):
        raise TraceCompilationError("tree transport did not reconstruct target")

    counts = Counter(step.rule_name for step in steps)
    trace = RewriteTrace(
        source=source,
        target=target,
        steps=tuple(steps),
        metadata={
            "compiler": "size_matched_graft_transport_v1",
            "transport_strategy": "graft_then_atomic_ring_system_grow",
            "source_coupling": "molecular_quotient_slot_alignment",
            "source_heavy_atoms": source.n_real_atoms,
            "target_heavy_atoms": target.n_real_atoms,
            "visible_steps": len(steps),
            "atom_delete_steps": counts["atom_delete"],
            "atom_insert_steps": counts["atom_insert"],
            "bond_reroute_steps": counts["bond_reroute"],
            "graft_planner": graft_planner,
            "scalar_closure_steps": 0,
            "zero_span_ring_closure_steps": 0,
            "ring_system_grow_steps": counts["ring_system_grow"],
            "atom_restate_steps": counts["atom_restate"],
            "bond_reorder_steps": counts["bond_reorder"],
            "ring_system_restate_steps": counts["ring_system_restate"],
            "ring_commitment_status": "atomic_full_ring_systems",
        },
    )
    if ring_catalog is not None and not ring_catalog.supports_trace(trace):
        raise TraceCompilationError("typed ring catalog does not support Graft trace")
    return trace


def _ring_system_components(graph: nx.Graph) -> tuple[tuple[int, ...], ...]:
    """Return disjoint cyclic systems, joining fused, bridged, and spiro rings."""

    bridges = {_edge(int(a), int(b)) for a, b in nx.bridges(graph)}
    cyclic_graph = nx.Graph()
    cyclic_graph.add_edges_from(
        (int(a), int(b))
        for a, b in graph.edges()
        if _edge(int(a), int(b)) not in bridges
    )
    return tuple(
        tuple(sorted(int(v) for v in component))
        for component in sorted(
            nx.connected_components(cyclic_graph),
            key=lambda values: tuple(sorted(values)),
        )
    )


def _ring_system_topology_class(graph: nx.Graph) -> str:
    cycle_rank = int(
        graph.number_of_edges()
        - graph.number_of_nodes()
        + nx.number_connected_components(graph)
    )
    if tuple(nx.articulation_points(graph)):
        return "spiro"
    if cycle_rank == 1:
        return "macrocycle" if graph.number_of_nodes() >= 12 else "single_ring"
    degree_three = sum(int(degree >= 3) for _, degree in graph.degree())
    return "bridged_or_fused" if degree_three >= 2 else "polycyclic"


def _plan_quotient_common_grafts(
    source: MolecularGraph,
    target_graph: nx.Graph,
    *,
    real: tuple[int, ...],
) -> tuple[MolecularGraph, nx.Graph, tuple[BondReroute, ...], str]:
    """Align two labeled trees through their earliest common unlabeled shape.

    Padded atom-slot labels are not molecular observables.  The previous
    fallback sorted a path by those labels and consequently emitted long runs
    of chemically self-equivalent Grafts.  Here both trees are reduced only by
    branch-removing Grafts, and compilation stops at the first pair of
    isomorphic intermediate topologies.  One global isomorphism relabels the
    trace source; its mapped source-side steps followed by the inverse
    target-side steps then recover the exact target spanning tree.
    """

    source_real = tuple(int(v) for v in np.flatnonzero(is_element(source.atom_types)))
    if len(source_real) != len(real):
        raise TraceCompilationError("quotient Graft planning requires equal tree sizes")
    source_tree = _topology_graph(source, source_real)
    # Target graph insertion order is deterministic, so this is a stable
    # corpus-side lowering.  No source-slot identity enters the choice.
    target_tree = nx.minimum_spanning_tree(target_graph)
    source_states, source_actions = _branch_reduction_sequence(source_tree)
    target_states, target_actions = _branch_reduction_sequence(target_tree)

    candidates: list[tuple[tuple, dict[int, int], int, int]] = []
    for source_index, source_state in enumerate(source_states):
        for target_index, target_state in enumerate(target_states):
            if sorted(dict(source_state.degree()).values()) != sorted(
                dict(target_state.degree()).values()
            ):
                continue
            try:
                mapping = _lexicographic_isomorphism(source_state, target_state)
            except TraceCompilationError:
                continue
            mapping_key = tuple(mapping[v] for v in sorted(mapping))
            priority = (
                source_index + target_index,
                max(source_index, target_index),
                source_index,
                target_index,
                mapping_key,
            )
            candidates.append((priority, mapping, source_index, target_index))
    if not candidates:
        raise TraceCompilationError("tree reductions found no common quotient topology")
    _, mapping, source_index, target_index = min(candidates, key=lambda item: item[0])

    aligned_source = _relabel_carbon_tree_state(source, mapping)
    mapped_source_actions = tuple(
        _map_action_slots(action, mapping)
        for action in source_actions[:source_index]
    )
    inverse_target_actions = tuple(
        BondReroute(
            a=int(action.u),
            b=int(action.v),
            u=int(action.a),
            v=int(action.b),
            new_order=int(action.new_order),
        )
        for action in reversed(target_actions[:target_index])
    )
    actions = (*mapped_source_actions, *inverse_target_actions)

    check = _topology_graph(
        aligned_source,
        tuple(int(v) for v in np.flatnonzero(is_element(aligned_source.atom_types))),
    )
    for action in actions:
        check.remove_edge(int(action.a), int(action.b))
        check.add_edge(int(action.u), int(action.v))
    if {_edge(a, b) for a, b in check.edges()} != {
        _edge(a, b) for a, b in target_tree.edges()
    }:
        raise TraceCompilationError("quotient Graft plan reached the wrong target tree")
    return aligned_source, target_tree, actions, "quotient_common_topology"


def _branch_reduction_sequence(
    tree: nx.Graph,
) -> tuple[tuple[nx.Graph, ...], tuple[BondReroute, ...]]:
    """Reduce a bounded-degree tree to a path without gauge-only rewrites."""

    current = tree.copy()
    states = [current.copy()]
    actions: list[BondReroute] = []
    while True:
        branch_vertices = sorted(int(v) for v, degree in current.degree() if degree > 2)
        if not branch_vertices:
            break
        branch = branch_vertices[0]
        choices = []
        for moved in sorted(int(v) for v in current.neighbors(branch)):
            trial = current.copy()
            trial.remove_edge(branch, moved)
            moved_component = set(nx.node_connected_component(trial, moved))
            outside_leaves = sorted(
                int(v)
                for v in set(current.nodes()) - moved_component
                if int(trial.degree(v)) == 1
            )
            choices.extend(
                (len(moved_component), moved, leaf) for leaf in outside_leaves
            )
        if not choices:
            raise TraceCompilationError("branch reduction found no valid destination leaf")
        _, moved, leaf = min(choices)
        action = BondReroute(a=moved, b=branch, u=moved, v=leaf)
        before_excess = sum(max(int(degree) - 2, 0) for _, degree in current.degree())
        current.remove_edge(moved, branch)
        current.add_edge(moved, leaf)
        after_excess = sum(max(int(degree) - 2, 0) for _, degree in current.degree())
        if after_excess != before_excess - 1:
            raise TraceCompilationError("branch Graft did not reduce branch excess once")
        actions.append(action)
        states.append(current.copy())
    return tuple(states), tuple(actions)


def _lexicographic_isomorphism(
    source: nx.Graph,
    target: nx.Graph,
) -> dict[int, int]:
    """Return one deterministic total source-to-target graph isomorphism.

    GraphMatcher respects deterministic node insertion order here.  Taking its
    first solution avoids enumerating an exponential number of automorphisms
    for highly symmetric trees; slot identity is gauge, so no lexicographic
    optimum is scientifically meaningful.
    """

    matcher = nx.algorithms.isomorphism.GraphMatcher(source, target)
    candidate = next(matcher.isomorphisms_iter(), None)
    if candidate is None:
        raise TraceCompilationError("requested tree topologies are not isomorphic")
    return {int(a): int(b) for a, b in candidate.items()}


def _relabel_carbon_tree_state(
    source: MolecularGraph,
    mapping: dict[int, int],
) -> MolecularGraph:
    """Apply one visible-trace-wide slot gauge to a carbon-tree source."""

    source_real = tuple(int(v) for v in np.flatnonzero(is_element(source.atom_types)))
    if set(mapping) != set(source_real) or len(set(mapping.values())) != len(source_real):
        raise TraceCompilationError("source relabeling requires a bijection on real atoms")
    if any(not 0 <= int(v) < source.n_atoms for v in mapping.values()):
        raise TraceCompilationError("source relabeling maps outside padded support")
    atom_types = np.full_like(source.atom_types, NULL_IDX)
    charges = np.zeros_like(source.formal_charges)
    hydrogens = np.zeros_like(source.implicit_h_counts)
    bonds = np.zeros_like(source.bonds)
    for old, new in mapping.items():
        atom_types[new] = source.atom_types[old]
        charges[new] = source.formal_charges[old]
        hydrogens[new] = source.implicit_h_counts[old]
    source_tree = _topology_graph(source, source_real)
    for a, b in source_tree.edges():
        left, right = mapping[int(a)], mapping[int(b)]
        bonds[left, right] = bonds[right, left] = int(source.bonds[a, b])
    aligned = MolecularGraph(atom_types, charges, hydrogens, bonds)
    if not is_valid_state(aligned) or not is_connected_or_null(aligned):
        raise TraceCompilationError("quotient source alignment violated tree invariants")
    return aligned


def _map_action_slots(action, mapping: dict[int, int]):
    """Conjugate supported resize/Graft actions by a global slot mapping."""

    if isinstance(action, AtomDelete):
        return AtomDelete(int(mapping[int(action.v)]))
    if isinstance(action, AtomInsert):
        return AtomInsert(
            slot=int(mapping[int(action.slot)]),
            atom_type=int(action.atom_type),
            formal_charge=int(action.formal_charge),
            implicit_h_count=int(action.implicit_h_count),
            neighbors=tuple(
                (int(mapping[int(neighbor)]), int(order))
                for neighbor, order in action.neighbors
            ),
        )
    if isinstance(action, BondReroute):
        return BondReroute(
            a=int(mapping[int(action.a)]),
            b=int(mapping[int(action.b)]),
            u=int(mapping[int(action.u)]),
            v=int(mapping[int(action.v)]),
            new_order=int(action.new_order),
        )
    raise TypeError(f"unsupported quotient-mapped action: {type(action).__name__}")


def _plan_root_free_grafts(
    source: MolecularGraph,
    target_graph: nx.Graph,
    *,
    real: tuple[int, ...],
) -> tuple[nx.Graph, tuple[BondReroute, ...], str]:
    """Construct a deterministic valence-valid root-free Graft plan.

    The fast path greedily exchanges non-target branches for edges of a
    maximum-overlap target spanning tree.  Greedy exchange can encounter a
    saturated carbon even though the two trees remain mutually reachable.  In
    that case we use a constructive, linear-size fallback: reduce both trees
    to the same source-derived path and reverse the target reduction.  Tree-to-path
    reduction only moves a branch onto a leaf, and path canonicalization only
    performs endpoint prefix reversals.  Consequently every intermediate has
    maximum degree at most four, every action is a root-free Graft representable
    by the factorized model, and no combinatorial search is required.
    """

    source_graph = _topology_graph(source, real)
    weighted_target = target_graph.copy()
    for a, b in weighted_target.edges():
        # A zero cost for already present source edges yields a deterministic
        # maximum-overlap target spanning tree.
        weighted_target[a][b]["exchange_cost"] = int(
            not source_graph.has_edge(a, b)
        )

    target_tree = nx.minimum_spanning_tree(
        weighted_target,
        weight="exchange_cost",
    )
    greedy = _greedy_monotone_grafts(source_graph, target_tree)
    if greedy is not None:
        return target_tree, greedy, "bounded_monotone_beam"

    _, source_to_path, source_path_order = _tree_to_canonical_path(
        source_graph,
        real,
        canonicalize=False,
    )
    _, target_to_path, _ = _tree_to_canonical_path(
        target_tree,
        real,
        desired_order=source_path_order,
    )
    path_to_target = tuple(
        BondReroute(
            a=int(action.u),
            b=int(action.v),
            u=int(action.a),
            v=int(action.b),
            new_order=BOND_SINGLE,
        )
        for action in reversed(target_to_path)
    )
    return target_tree, (*source_to_path, *path_to_target), "shared_path_fallback"


def _greedy_monotone_grafts(
    source_tree: nx.Graph,
    target_tree: nx.Graph,
    *,
    beam_width: int = 16,
) -> tuple[BondReroute, ...] | None:
    """Find a short monotone plan with bounded, deterministic beam search."""

    goal = {_edge(a, b) for a, b in target_tree.edges()}
    source_edges = frozenset(_edge(a, b) for a, b in source_tree.edges())
    if source_edges == frozenset(goal):
        return ()
    target_degree = dict(target_tree.degree())
    nodes = tuple(sorted(int(v) for v in source_tree.nodes()))
    beam: dict[
        frozenset[tuple[int, int]],
        tuple[BondReroute, ...],
    ] = {source_edges: ()}

    def graph_from(edges: frozenset[tuple[int, int]]) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

    def expansions(
        edges: frozenset[tuple[int, int]],
    ) -> tuple[tuple[BondReroute, frozenset[tuple[int, int]]], ...]:
        current = graph_from(edges)
        degree = dict(current.degree())
        candidates = []
        for u, v in sorted(goal - set(edges)):
            path = nx.shortest_path(current, u, v)
            for moved, target_parent, removed_neighbor in (
                (int(u), int(v), int(path[1])),
                (int(v), int(u), int(path[-2])),
            ):
                removed = _edge(moved, removed_neighbor)
                if removed in goal or int(degree[target_parent]) >= 4:
                    continue
                action = BondReroute(
                    a=moved,
                    b=removed_neighbor,
                    u=moved,
                    v=target_parent,
                )
                successor = set(edges)
                successor.remove(removed)
                successor.add(_edge(moved, target_parent))
                candidates.append((action, frozenset(successor)))
        return tuple(candidates)

    def priority(edges: frozenset[tuple[int, int]]) -> tuple:
        current = graph_from(edges)
        degree = dict(current.degree())
        fixed_degree = Counter()
        for a, b in goal & set(edges):
            fixed_degree[a] += 1
            fixed_degree[b] += 1
        saturated_missing = sum(
            int(degree[v] >= 4 and fixed_degree[v] < target_degree[v])
            for v in nodes
        )
        degree_error = sum(
            abs(int(degree[v]) - int(target_degree[v])) for v in nodes
        )
        return (
            saturated_missing,
            degree_error,
            tuple(sorted(edges)),
        )

    maximum_depth = len(goal - set(source_edges))
    for _ in range(maximum_depth):
        next_beam: dict[
            frozenset[tuple[int, int]],
            tuple[BondReroute, ...],
        ] = {}
        for edges, plan in beam.items():
            for action, successor in expansions(edges):
                if successor in next_beam:
                    continue
                candidate_plan = (*plan, action)
                if successor == frozenset(goal):
                    return candidate_plan
                next_beam[successor] = candidate_plan
        if not next_beam:
            return None
        ranked = sorted(next_beam, key=priority)[:beam_width]
        beam = {edges: next_beam[edges] for edges in ranked}
    return None


def _tree_to_canonical_path(
    tree: nx.Graph,
    real: tuple[int, ...],
    *,
    canonicalize: bool = True,
    desired_order: tuple[int, ...] | None = None,
) -> tuple[nx.Graph, tuple[BondReroute, ...], tuple[int, ...]]:
    """Reduce a bounded-degree tree to a chosen vertex path without search."""

    current = tree.copy()
    plan: list[BondReroute] = []
    while True:
        branch_vertices = sorted(v for v, degree in current.degree() if degree > 2)
        if not branch_vertices:
            break
        branch = int(branch_vertices[0])
        neighbors = sorted(int(v) for v in current.neighbors(branch))
        # Prefer moving the smallest branch.  Any other component contains a
        # leaf, so the branch can always be reattached without exceeding
        # degree two at the new endpoint.
        choices = []
        for moved in neighbors:
            trial = current.copy()
            trial.remove_edge(branch, moved)
            moved_component = set(nx.node_connected_component(trial, moved))
            outside_leaves = sorted(
                int(v)
                for v in set(real) - moved_component
                if int(trial.degree(v)) == 1
            )
            for leaf in outside_leaves:
                choices.append((len(moved_component), moved, leaf))
        if not choices:
            raise TraceCompilationError("tree-to-path Graft reduction found no leaf")
        _, moved, leaf = min(choices)
        action = BondReroute(
            a=moved,
            b=branch,
            u=moved,
            v=leaf,
        )
        current.remove_edge(moved, branch)
        current.add_edge(moved, leaf)
        plan.append(action)

    endpoints = sorted(int(v) for v, degree in current.degree() if degree == 1)
    if len(real) == 1:
        order = [int(real[0])]
    else:
        if len(endpoints) != 2:
            raise TraceCompilationError("tree-to-path reduction did not produce a path")
        order = list(nx.shortest_path(current, endpoints[0], endpoints[1]))
    if not canonicalize:
        return current, tuple(plan), tuple(int(v) for v in order)
    desired = list(
        desired_order
        if desired_order is not None
        else sorted(int(v) for v in real)
    )
    if sorted(desired) != sorted(int(v) for v in real):
        raise ValueError("desired path order must be a permutation of real vertices")

    # Pancake sorting uses prefix reversals.  Reversing the entire list changes
    # only our orientation of the same undirected path and needs no rewrite.
    for size in range(len(order), 1, -1):
        desired_vertex = desired[size - 1]
        index = order.index(desired_vertex, 0, size)
        if index == size - 1:
            continue
        for prefix_size in ((index + 1,) if index else ()) + (size,):
            if prefix_size <= 1:
                continue
            if prefix_size == len(order):
                order.reverse()
                continue
            moved = int(order[prefix_size])
            removed_neighbor = int(order[prefix_size - 1])
            target = int(order[0])
            action = BondReroute(
                a=moved,
                b=removed_neighbor,
                u=moved,
                v=target,
            )
            current.remove_edge(moved, removed_neighbor)
            current.add_edge(moved, target)
            plan.append(action)
            order[:prefix_size] = reversed(order[:prefix_size])

    if order != desired:
        raise TraceCompilationError("path canonicalization did not sort vertices")
    expected = {_edge(a, b) for a, b in zip(desired, desired[1:])}
    observed = {_edge(a, b) for a, b in current.edges()}
    if observed != expected:
        raise TraceCompilationError("path canonicalization reached wrong edges")
    return current, tuple(plan), tuple(int(v) for v in order)


def align_carbon_tree_source(
    source: MolecularGraph,
    target: MolecularGraph,
) -> MolecularGraph:
    """Relabel an equal-size source tree to shorten its paired Graft path.

    Relabeling does not alter the source's unlabeled molecular state.  The
    operation is training-only and is sound for a permutation-equivariant
    model; ancestral sampling still draws the raw source prior directly.
    """

    _validate_transport_pair(source, target, require_aligned_size=True)
    real = tuple(int(v) for v in np.flatnonzero(is_element(target.atom_types)))
    source_tree = _topology_graph(source, real)
    target_tree = nx.minimum_spanning_tree(_topology_graph(target, real))
    source_degree = dict(source_tree.degree())
    target_degree = dict(target_tree.degree())
    source_root = min(real, key=lambda v: (-source_degree[v], v))
    target_root = min(real, key=lambda v: (-target_degree[v], v))
    mapping = {source_root: target_root}
    used = {target_root}
    queue = [(source_root, target_root)]
    while queue:
        source_parent, target_parent = queue.pop(0)
        source_children = sorted(
            (v for v in source_tree.neighbors(source_parent) if v not in mapping),
            key=lambda v: (-source_degree[v], v),
        )
        target_children = sorted(
            (v for v in target_tree.neighbors(target_parent) if v not in used),
            key=lambda v: (-target_degree[v], v),
        )
        for source_child, target_child in zip(source_children, target_children):
            mapping[source_child] = target_child
            used.add(target_child)
            queue.append((source_child, target_child))
    remaining_source = sorted((v for v in real if v not in mapping), key=lambda v: (-source_degree[v], v))
    remaining_target = set(v for v in real if v not in used)
    for source_vertex in remaining_source:
        target_vertex = min(
            remaining_target,
            key=lambda v: (abs(source_degree[source_vertex] - target_degree[v]), v),
        )
        mapping[source_vertex] = target_vertex
        remaining_target.remove(target_vertex)

    atom_types = source.atom_types.copy()
    charges = source.formal_charges.copy()
    hydrogens = source.implicit_h_counts.copy()
    bonds = np.zeros_like(source.bonds)
    for source_vertex, target_vertex in mapping.items():
        atom_types[target_vertex] = source.atom_types[source_vertex]
        charges[target_vertex] = source.formal_charges[source_vertex]
        hydrogens[target_vertex] = source.implicit_h_counts[source_vertex]
    for a, b in source_tree.edges():
        left, right = mapping[int(a)], mapping[int(b)]
        bonds[left, right] = bonds[right, left] = int(source.bonds[a, b])
    aligned = MolecularGraph(atom_types, charges, hydrogens, bonds)
    if not is_valid_state(aligned) or not is_connected_or_null(aligned):
        raise TraceCompilationError("source alignment violated tree invariants")
    return aligned


def _compile_primitive_tree_transport(
    source: MolecularGraph,
    target: MolecularGraph,
    *,
    target_trace: RewriteTrace,
    runtime: RewriteSystem,
) -> RewriteTrace:
    """Peel to one carbon, then convert the null trace's root transaction."""

    if not target_trace.steps:
        raise TraceCompilationError("primitive tree transport needs a nonempty target")
    first = target_trace.steps[0]
    if isinstance(first.action, AtomInsert):
        target_root = int(first.action.slot)
    elif isinstance(first.action, CycleInsert):
        target_root = int(first.action.atoms[0].slot)
    else:
        raise TraceCompilationError(
            f"unsupported null-trace root instruction: {first.rule_name}"
        )

    state = source
    steps: list[RewriteStep] = []

    def commit(rule_name: str, action) -> None:
        nonlocal state
        state = runtime.apply(state, rule_name, action)
        steps.append(RewriteStep(rule_name, action))

    source_real = tuple(int(v) for v in np.flatnonzero(is_element(state.atom_types)))
    retained_root = source_real[0]
    current_graph = _topology_graph(state, source_real)
    while current_graph.number_of_nodes() > 1:
        leaves = sorted(
            vertex
            for vertex, degree in current_graph.degree()
            if degree == 1 and vertex != retained_root
        )
        if not leaves:
            raise TraceCompilationError("tree peeling lost its retained root")
        leaf = leaves[0]
        commit("atom_delete", AtomDelete(int(leaf)))
        current_graph.remove_node(leaf)

    if retained_root != target_root:
        carbon = int(ELEMENT_TO_IDX["C"])
        commit(
            "atom_insert",
            AtomInsert(
                slot=target_root,
                atom_type=carbon,
                formal_charge=0,
                implicit_h_count=3,
                neighbors=((retained_root, BOND_SINGLE),),
            ),
        )
        commit("atom_delete", AtomDelete(retained_root))

    if isinstance(first.action, AtomInsert):
        root_action = first.action
        desired = (
            int(root_action.atom_type),
            int(root_action.formal_charge),
            int(root_action.implicit_h_count),
        )
        present = (
            int(state.atom_types[target_root]),
            int(state.formal_charges[target_root]),
            int(state.implicit_h_counts[target_root]),
        )
        if present != desired:
            commit(
                "atom_restate",
                AtomRestate(target_root, desired[0], desired[1], desired[2]),
            )
    else:
        cycle = first.action
        root_payload = cycle.atoms[0]
        pre_ring_h = int(root_payload.implicit_h_count) + int(
            BOND_CLASS_TO_H_CHANGE[int(cycle.bond_orders[0])]
        ) + int(BOND_CLASS_TO_H_CHANGE[int(cycle.bond_orders[-1])])
        desired = (
            int(root_payload.atom_type),
            int(root_payload.formal_charge),
            pre_ring_h,
        )
        present = (
            int(state.atom_types[target_root]),
            int(state.formal_charges[target_root]),
            int(state.implicit_h_counts[target_root]),
        )
        if present != desired:
            commit(
                "atom_restate",
                AtomRestate(
                    v=target_root,
                    atom_type=desired[0],
                    formal_charge=desired[1],
                    implicit_h_count=desired[2],
                ),
            )
        commit(
            "ring_ear_insert",
            RingEarInsert(
                a=target_root,
                b=target_root,
                atoms=tuple(cycle.atoms[1:]),
                bond_orders=tuple(int(order) for order in cycle.bond_orders),
            ),
        )

    for step in target_trace.steps[1:]:
        commit(step.rule_name, step.action)
    if not _same_state(state, target):
        raise TraceCompilationError("primitive tree transport did not reconstruct target")

    counts = Counter(step.rule_name for step in steps)
    return RewriteTrace(
        source=source,
        target=target,
        steps=tuple(steps),
        metadata={
            "compiler": "primitive_carbon_tree_transport_tracelets_v2",
            "transport_strategy": "primitive_leaf_delete_tracelet_regrow",
            "source_heavy_atoms": source.n_real_atoms,
            "target_heavy_atoms": target.n_real_atoms,
            "visible_steps": len(steps),
            "atom_delete_steps": counts["atom_delete"],
            "atom_insert_steps": counts["atom_insert"],
            "bond_reroute_steps": 0,
            "scalar_closure_steps": counts["bond_insert"],
            "atom_restate_steps": counts["atom_restate"],
            "bond_reorder_steps": counts["bond_reorder"],
            "ring_system_restate_steps": counts["ring_system_restate"],
            "ring_commitment_status": "topology_committed_tracelets",
        },
    )


def _validate_transport_pair(
    source: MolecularGraph,
    target: MolecularGraph,
    *,
    require_aligned_size: bool,
) -> None:
    if source.n_atoms != target.n_atoms:
        raise TraceCompilationError("source and target need the same padded slot count")
    if not is_valid_state(source) or not is_connected_or_null(source):
        raise TraceCompilationError("source is not a valid connected molecule")
    if not is_valid_state(target) or not is_connected_or_null(target):
        raise TraceCompilationError("target is not a valid connected molecule")
    source_real = tuple(int(v) for v in np.flatnonzero(is_element(source.atom_types)))
    target_real = tuple(int(v) for v in np.flatnonzero(is_element(target.atom_types)))
    if not source_real or not target_real:
        raise TraceCompilationError("tree transport requires nonempty source and target")
    if require_aligned_size and source_real != target_real:
        raise TraceCompilationError(
            "bond-reroute transport requires equal-size aligned slot support"
        )
    carbon = int(ELEMENT_TO_IDX["C"])
    if any(int(source.atom_types[v]) != carbon for v in source_real):
        raise TraceCompilationError("source must contain only carbon atoms")
    if np.any(source.formal_charges[list(source_real)] != 0):
        raise TraceCompilationError("source carbon tree must be neutral")
    graph = _topology_graph(source, source_real)
    if not nx.is_tree(graph):
        raise TraceCompilationError("source topology is not a tree")
    if any(int(source.bonds[a, b]) != BOND_SINGLE for a, b in graph.edges()):
        raise TraceCompilationError("source tree must use single bonds")


def _full_topology_state_index(
    states: tuple[MolecularGraph, ...],
    target: MolecularGraph,
) -> int:
    target_real = is_element(target.atom_types)
    target_topology = target.bonds != 0
    for index, state in enumerate(states):
        if not np.array_equal(is_element(state.atom_types), target_real):
            continue
        if np.array_equal(state.bonds != 0, target_topology):
            return index
    raise TraceCompilationError("target compiler never exposed its complete topology")


def _topology_graph(state: MolecularGraph, real: tuple[int, ...]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(real)
    graph.add_edges_from(
        (a, b)
        for a, b in combinations(real, 2)
        if int(state.bonds[a, b]) != 0
    )
    return graph


def _edge(a: int, b: int) -> frozenset[int]:
    return frozenset((int(a), int(b)))


def _same_state(left: MolecularGraph, right: MolecularGraph) -> bool:
    return bool(
        np.array_equal(left.atom_types, right.atom_types)
        and np.array_equal(left.formal_charges, right.formal_charges)
        and np.array_equal(left.implicit_h_counts, right.implicit_h_counts)
        and np.array_equal(left.bonds, right.bonds)
    )


__all__ = ["compile_carbon_tree_to_target"]
