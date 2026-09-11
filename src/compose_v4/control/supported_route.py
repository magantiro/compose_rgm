"""Answer-known local lowering through an externally supplied production law.

This compiler never adds marks to the law and is not a target-free search policy.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import count

import networkx as nx
import numpy as np

from compose_v4.chem.molecular_graph import NULL_IDX, is_element
from compose_v4.control.docking_value import identity
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.experiments.quotient_invariance import permute_persistent_slots
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.kernel import InvalidRewrite, canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state


def exact_key(graph):
    return identity(encode_state(graph))


def transported_target(original, following, current, permutation):
    """Maintain a bijection when production births use the first empty slot."""
    permutation = list(permutation)
    if exact_key(permute_persistent_slots(original, permutation)) != exact_key(current):
        raise ValueError("route source disagrees with its exact slot correspondence")
    births = np.flatnonzero(~is_element(original.atom_types) & is_element(following.atom_types))
    if len(births) > 1:
        raise ValueError("route requests unsupported multi-atom primitive birth")
    if len(births):
        empty = np.flatnonzero(current.atom_types == NULL_IDX)
        if not len(empty):
            raise ValueError("route birth has no free persistent slot")
        old_position = permutation.index(int(births[0]))
        new_position = int(empty[0])
        permutation[old_position], permutation[new_position] = (
            permutation[new_position],
            permutation[old_position],
        )
        if exact_key(permute_persistent_slots(original, permutation)) != exact_key(current):
            raise ValueError("birth correspondence would change an occupied source slot")
    return permute_persistent_slots(following, permutation), permutation


def distance(graph, target):
    """Exact coordinate mismatch, only a compiler priority, never a task reward."""
    return int(
        np.count_nonzero(graph.atom_types != target.atom_types)
        + np.count_nonzero(graph.formal_charges != target.formal_charges)
        + np.count_nonzero(graph.implicit_h_counts != target.implicit_h_counts)
        + np.count_nonzero(np.triu(graph.bonds != target.bonds, 1))
    )


def _opening_edges(source, target):
    graph = nx.Graph()
    graph.add_nodes_from(int(i) for i in np.flatnonzero(is_element(source.atom_types)))
    graph.add_edges_from((int(a), int(b)) for a, b in zip(*np.nonzero(np.triu(source.bonds, 1))))
    changed = {int(i) for i in np.flatnonzero(source.atom_types != target.atom_types)}
    changed.update(int(i) for i in np.nonzero(source.bonds != target.bonds)[0])
    allowed = set()
    for component in nx.biconnected_components(graph):
        if component & changed:
            allowed.update(tuple(sorted(e)) for e in graph.subgraph(component).edges())
    return allowed


def _relevant(family, action, graph, target, openings):
    if family == "atom_delete":
        return not is_element(target.atom_types[action.v])
    if family == "atom_insert":
        return bool(is_element(target.atom_types[action.slot]))
    if family in ("atom_restate", "atom_restate_semantic"):
        return bool(
            graph.atom_types[action.v] != target.atom_types[action.v]
            or graph.implicit_h_counts[action.v] != target.implicit_h_counts[action.v]
        )
    if family in ("cycle_open", "bond_delete"):
        edge = tuple(sorted((action.a, action.b)))
        return edge in openings or target.bonds[edge] == 0
    if family in ("cycle_close", "bond_insert", "bond_reorder"):
        return bool(target.bonds[action.a, action.b])
    if family == "ring_system_restate":
        return any(tuple(sorted((c.a, c.b))) in openings for c in action.changes)
    return family == "bond_reroute"


@dataclass(frozen=True)
class BridgeConfig:
    max_expansions: int = 16
    max_steps: int = 8

    def __post_init__(self):
        if self.max_expansions < 1 or self.max_steps < 1:
            raise ValueError("bridge bounds must be positive")


def lower_supported(source, target, law, system, config=None):
    """Find a bounded exact-state bridge, certifying every mark against `law`."""
    config = BridgeConfig() if config is None else config
    if not charge_policy_preserved(source, target):
        return {"status": "charge_policy_rejected", "expanded": 0, "steps": []}
    target_key = exact_key(target)
    openings = _opening_edges(source, target)
    serial = count()
    queue = [(distance(source, target), 0, next(serial), source, [])]
    visited = set()
    expanded = 0
    best = distance(source, target)
    while queue and expanded < config.max_expansions:
        _, depth, _, graph, path = heapq.heappop(queue)
        key = exact_key(graph)
        if key == target_key:
            return {"status": "supported", "expanded": expanded, "steps": path}
        if key in visited or depth >= config.max_steps:
            continue
        visited.add(key)
        expanded += 1
        families, actions, probabilities = law(graph)
        origin = canonical_state_key(graph)
        children = {}
        for family, action, probability in zip(families, actions, probabilities, strict=True):
            if probability <= 0 or not _relevant(family, action, graph, target, openings):
                continue
            try:
                child = system.apply(graph, family, action)
            except InvalidRewrite:
                continue
            if not 1 <= child.n_real_atoms <= 40 or not charge_policy_preserved(graph, child):
                continue
            if canonical_state_key(child) == origin:
                continue
            child_key = exact_key(child)
            error = distance(child, target)
            # Do not explore irrelevant chemistry. A ring opening may temporarily
            # worsen the target mismatch; all other moves must make progress.
            if error >= distance(graph, target) and family not in ("cycle_open", "bond_delete"):
                continue
            certificate = {
                "action": encode_action(family, action),
                "source_key": key,
                "state": encode_state(child),
                "selected_mark_probability": float(probability),
            }
            if (
                child_key not in children
                or probability > children[child_key][1]["selected_mark_probability"]
            ):
                children[child_key] = (child, certificate, error)
        for child_key, (child, certificate, error) in sorted(children.items()):
            steps = [*path, certificate]
            if child_key == target_key:
                return {"status": "supported", "expanded": expanded, "steps": steps}
            best = min(best, error)
            if child_key not in visited:
                heapq.heappush(queue, (error, depth + 1, next(serial), child, steps))
    return {"status": "bridge_unresolved", "expanded": expanded, "best_mismatch": best, "steps": []}
