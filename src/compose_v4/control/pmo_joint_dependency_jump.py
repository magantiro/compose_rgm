"""Task-blind joint PMO program latents with exact address-free rebinding.

The checkpoint stores complete sequences of generic action roles.  It never stores
an executable action, persistent address, molecular graph, endpoint, task, or route
identifier.  Runtime binding enumerates exact legal successors from the current
graph and resolves relative created-handle dependencies causally.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any

import networkx as nx
import numpy as np

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.docking_value import identity
from compose_v4.control.pmo_action_roles import action_role_supervision, atom_role
from compose_v4.experiments.pmo_complete_route_dynamic_gate import (
    MAXIMUM_COMPONENTS,
    MAXIMUM_PRIMITIVES,
)
from compose_v4.experiments.pmo_dependency_region_program import (
    DependencyRegionConfig,
    dependency_region_program,
)
from compose_v4.experiments.pmo_legal_action_policy import (
    LegalSuccessor,
    enumerate_rule_successors,
)
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.action_codec_v4 import encode_action
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import CycleCloseEdge
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from compose_v4.rewrite.tracelets import BondOrderChange, RingSystemRestate

SCHEMA = "pmo_joint_dependency_jump_v1"
CHECKPOINT_SCHEMA = "pmo_joint_dependency_jump_checkpoint_v1"

FORBIDDEN_CHECKPOINT_KEYS = {
    "action",
    "actions",
    "endpoint",
    "lineage_identity",
    "member",
    "members",
    "route",
    "route_identity",
    "smiles",
    "source_graph",
    "source_state",
    "state",
    "states",
    "task",
    "task_family",
    "terminal_endpoint",
    "trace_identity",
}


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(map(str, value)) | set().union(
            *(_recursive_keys(child) for child in value.values()), set()
        )
    if isinstance(value, list):
        return set().union(*(_recursive_keys(child) for child in value), set())
    return set()


def _generic_role_sequence(route: dict[str, Any]) -> list[dict[str, Any]]:
    created: dict[int, tuple[int, int]] = {}
    next_ordinal = 0
    roles = []
    for step, (state, record) in enumerate(
        zip(route["states"][:-1], route["actions"], strict=True)
    ):
        role, next_ordinal = action_role_supervision(
            decode_state(state), record, created, step, next_ordinal
        )
        roles.append(role)
    return roles


def _balanced_route_weights(routes: list[dict[str, Any]]) -> dict[str, float]:
    by_family: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for route in routes:
        by_family[str(route["task_family"])][str(route["lineage_identity"])].append(route)
    weights = {}
    for lineages in by_family.values():
        for rows in lineages.values():
            for route in rows:
                weights[str(route["trace_identity"])] = (
                    1.0 / len(by_family) / len(lineages) / len(rows)
                )
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def fit_joint_checkpoint(corpus: dict[str, Any], *, held_fold: int | None) -> dict[str, Any]:
    """Fit complete generic role-sequence latents after an optional fold holdout."""

    selected = [
        route
        for route in corpus["routes"]
        if route["dependency_region_program"].get("complete_representation_supported")
        and (held_fold is None or int(route["test_fold"]) != held_fold)
    ]
    if not selected:
        raise ValueError("joint PMO checkpoint has no complete training routes")
    weights = _balanced_route_weights(selected)
    grouped: dict[str, dict[str, Any]] = {}
    for route in selected:
        roles = _generic_role_sequence(route)
        plan_id = identity({"schema_version": "pmo_joint_role_plan_v1", "roles": roles})
        row = grouped.setdefault(
            plan_id,
            {
                "plan_id": plan_id,
                "roles": roles,
                "primitive_count": len(roles),
                "component_count": int(route["dependency_region_program"]["component_count"]),
                "created_dependency_count": sum(
                    len(role["created_handle_dependencies"]) for role in roles
                ),
                "mass": 0.0,
                "support_count": 0,
            },
        )
        row["mass"] += weights[str(route["trace_identity"])]
        row["support_count"] += 1
    plans = sorted(grouped.values(), key=lambda row: (-row["mass"], row["plan_id"]))
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "fit_scope": "shared_all_routes" if held_fold is None else "held_fold_clean",
        "held_fold": held_fold,
        "maximum_primitives": MAXIMUM_PRIMITIVES,
        "maximum_components": MAXIMUM_COMPONENTS,
        "plan_latents": plans,
        "training_route_count": len(selected),
        "training_plan_count": len(plans),
        "split_identity": corpus["split"]["split_identity"],
        "new_oracle_calls": 0,
    }
    forbidden = FORBIDDEN_CHECKPOINT_KEYS & _recursive_keys(payload)
    if forbidden:
        raise RuntimeError(f"joint checkpoint retained forbidden keys: {sorted(forbidden)}")
    return {"payload": payload, "payload_sha256": identity(payload)}


def _ring_edges(graph) -> tuple[tuple[int, int], ...]:
    real = tuple(int(slot) for slot in np.flatnonzero(is_element(graph.atom_types)))
    network = nx.Graph()
    network.add_nodes_from(real)
    network.add_edges_from(
        (left, right) for left, right in combinations(real, 2) if int(graph.bonds[left, right])
    )
    cyclic = {
        (min(left, right), max(left, right))
        for cycle in nx.cycle_basis(network)
        for left, right in zip(cycle, (*cycle[1:], cycle[0]), strict=True)
    }
    return tuple(sorted(cyclic))


def _expanded_ring_restate_successors(graph, role: dict[str, Any]) -> tuple[LegalSuccessor, ...]:
    """Enumerate generic, executor-verified multi-bond ring restatements.

    The production aromaticity-flip enumerator is retained.  The additional
    support covers aromaticity-neutral coordinated restatements already legal in
    the executor, parameterized only by the role's address-free bond classes.
    """

    runtime = editing_v2_semantic_rewrite_system()
    source_key = canonical_state_key(graph)
    successors = {
        row.successor_key: row for row in enumerate_rule_successors(graph, "ring_system_restate")
    }
    target_orders = tuple(int(value) for value in role["parameters"]["new_bond_classes"])
    if not 2 <= len(target_orders) <= 4:
        return tuple(successors[key] for key in sorted(successors))
    edges = _ring_edges(graph)
    for chosen in combinations(edges, len(target_orders)):
        for orders in sorted(set(permutations(target_orders))):
            if any(int(graph.bonds[a, b]) == order for (a, b), order in zip(chosen, orders)):
                continue
            action = RingSystemRestate(
                tuple(
                    BondOrderChange(a, b, order)
                    for (a, b), order in zip(chosen, orders, strict=True)
                )
            )
            try:
                successor = runtime.apply(graph, "ring_system_restate", action)
            except (InvalidRewrite, ValueError):
                continue
            key = canonical_state_key(successor)
            if key == source_key:
                continue
            successors.setdefault(
                key,
                LegalSuccessor(
                    rule="ring_system_restate",
                    action_record=encode_action("ring_system_restate", action),
                    successor=successor,
                    successor_key=key,
                ),
            )
    return tuple(successors[key] for key in sorted(successors))


def _expanded_cycle_close_successors(
    graph,
    role: dict[str, Any],
    *,
    created: dict[int, tuple[int, int]] | None = None,
    step: int = 0,
) -> tuple[LegalSuccessor, ...]:
    """Enumerate every executor-valid closure with the requested bond class.

    The ordinary production fiber intentionally applies a narrow geometric
    heuristic.  A joint complete-program latent instead specifies the endpoint
    roles and can safely use the full executor-verified closure fiber without
    storing an address or injecting a teacher action.
    """

    runtime = editing_v2_semantic_rewrite_system()
    source_key = canonical_state_key(graph)
    successors = (
        {
            identity(row.action_record): row
            for row in enumerate_rule_successors(graph, "cycle_close")
        }
        if created is None
        else {}
    )
    order = int(role["parameters"]["bond_class"])
    real = tuple(int(slot) for slot in np.flatnonzero(is_element(graph.atom_types)))
    desired = {str(row["role"]): row["descriptor"] for row in role.get("operands", ())}
    for left, right in combinations(real, 2):
        if int(graph.bonds[left, right]):
            continue
        if created is not None and (
            atom_role(graph, left, created, step) != desired.get("endpoint_a")
            or atom_role(graph, right, created, step) != desired.get("endpoint_b")
        ):
            continue
        action = CycleCloseEdge(left, right, order)
        try:
            successor = runtime.apply(graph, "cycle_close", action)
        except (InvalidRewrite, ValueError):
            continue
        key = canonical_state_key(successor)
        if key == source_key:
            continue
        record = encode_action("cycle_close", action)
        successors.setdefault(
            identity(record),
            LegalSuccessor(
                rule="cycle_close",
                action_record=record,
                successor=successor,
                successor_key=key,
            ),
        )
    return tuple(successors[key] for key in sorted(successors))


def enumerate_role_successors(
    graph,
    role: dict[str, Any],
    *,
    created: dict[int, tuple[int, int]] | None = None,
    step: int = 0,
) -> tuple[LegalSuccessor, ...]:
    rule = str(role["executor_rule"])
    if rule == "ring_system_restate":
        return _expanded_ring_restate_successors(graph, role)
    if rule == "cycle_close":
        return _expanded_cycle_close_successors(graph, role, created=created, step=step)
    return enumerate_rule_successors(graph, rule)


@dataclass(frozen=True)
class _BindingPrefix:
    graph: Any
    states: tuple[dict[str, Any], ...]
    actions: tuple[dict[str, Any], ...]
    created: tuple[tuple[int, int, int], ...]
    next_ordinal: int


def _created_dict(prefix: _BindingPrefix) -> dict[int, tuple[int, int]]:
    return {slot: (ordinal, step) for slot, ordinal, step in prefix.created}


def _role_identity(role: dict[str, Any]) -> str:
    return identity(role)


def bind_joint_plan(
    source,
    plan: dict[str, Any],
    *,
    beam_width: int = 8,
) -> list[dict[str, Any]]:
    """Bind one generic joint latent to the current graph without teacher access."""

    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    prefixes = [
        _BindingPrefix(
            graph=source,
            states=(encode_state(source),),
            actions=(),
            created=(),
            next_ordinal=0,
        )
    ]
    for step, desired in enumerate(plan["roles"]):
        desired_identity = _role_identity(desired)
        advanced: dict[tuple[str, str], _BindingPrefix] = {}
        for prefix in prefixes:
            prefix_created = _created_dict(prefix)
            for candidate in enumerate_role_successors(
                prefix.graph, desired, created=prefix_created, step=step
            ):
                created = dict(prefix_created)
                observed, next_ordinal = action_role_supervision(
                    prefix.graph,
                    candidate.action_record,
                    created,
                    step,
                    prefix.next_ordinal,
                )
                if _role_identity(observed) != desired_identity:
                    continue
                child = _BindingPrefix(
                    graph=candidate.successor,
                    states=(*prefix.states, encode_state(candidate.successor)),
                    actions=(*prefix.actions, candidate.action_record),
                    created=tuple(
                        sorted(
                            (slot, ordinal, made_at) for slot, (ordinal, made_at) in created.items()
                        )
                    ),
                    next_ordinal=next_ordinal,
                )
                key = (candidate.successor_key, identity(child.actions))
                advanced.setdefault(key, child)
        ranked = sorted(
            advanced.items(),
            key=lambda row: identity(
                {
                    "plan_id": plan["plan_id"],
                    "step": step,
                    "candidate": row[0],
                    "beam_stream": 20260919,
                }
            ),
        )
        prefixes = [prefix for _, prefix in ranked[:beam_width]]
        if not prefixes:
            return []
    results = []
    for prefix in prefixes:
        program = dependency_region_program(
            list(prefix.states),
            list(prefix.actions),
            config=DependencyRegionConfig(
                runtime_maximum_primitives=MAXIMUM_PRIMITIVES,
                runtime_maximum_components=MAXIMUM_COMPONENTS,
            ),
        )
        if not program["exact_replay"] or not program["complete_representation_supported"]:
            continue
        endpoint, receipt = execute_program(source, list(prefix.actions))
        if receipt["states"] != list(prefix.states):
            raise RuntimeError("joint plan exact replay changed")
        results.append(
            {
                "plan_id": plan["plan_id"],
                "endpoint_key": canonical_state_key(endpoint),
                "endpoint_state": prefix.states[-1],
                "primitive_count": len(prefix.actions),
                "component_count": program["component_count"],
                "created_dependency_edges": len(program["created_dependency_edges"]),
                "actions": list(prefix.actions),
                "states": list(prefix.states),
                "exact_replay": True,
            }
        )
    return results


def generate_joint_candidates(
    source,
    checkpoint: dict[str, Any],
    *,
    candidate_budget: int = 32,
    beam_width: int = 8,
) -> list[dict[str, Any]]:
    """Generate a deterministic, source-bound lock from task-blind joint plans."""

    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("invalid PMO joint checkpoint")
    source_key = canonical_state_key(source)
    by_endpoint: dict[str, dict[str, Any]] = {}
    for plan in checkpoint["plan_latents"]:
        for candidate in bind_joint_plan(source, plan, beam_width=beam_width):
            if candidate["endpoint_key"] == source_key:
                continue
            previous = by_endpoint.get(candidate["endpoint_key"])
            if previous is None or (-float(plan["mass"]), candidate["plan_id"]) < (
                -float(previous["plan_mass"]),
                previous["plan_id"],
            ):
                by_endpoint[candidate["endpoint_key"]] = {
                    **candidate,
                    "plan_mass": float(plan["mass"]),
                    "proposal_lane": "joint_dependency_region_jump",
                }
    ordered = sorted(
        by_endpoint.values(),
        key=lambda row: (
            -int(row["primitive_count"] >= 9),
            -row["plan_mass"],
            row["plan_id"],
            row["endpoint_key"],
        ),
    )
    return ordered[:candidate_budget]


__all__ = [
    "CHECKPOINT_SCHEMA",
    "FORBIDDEN_CHECKPOINT_KEYS",
    "SCHEMA",
    "bind_joint_plan",
    "enumerate_role_successors",
    "fit_joint_checkpoint",
    "generate_joint_candidates",
]
