"""Bounded exact realization of a generated structural graph subgoal.

The realizer is conditioned on a requested target graph. It never receives a
teacher action sequence and never falls back to one. Target-directed actions
are proposed through the existing Active8 executor and every retained prefix
is an exact legal molecular state.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations, count

import networkx as nx
import numpy as np

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY, MolecularGraph, is_element
from compose_v4.control.option_continuation import exact_graph_key
from compose_v4.control.structural_subgoal import StructuralGoal, instantiate_goal
from compose_v4.data.charge_policy import charge_policy_preserved
from compose_v4.rewrite.action_codec_v4 import decode_action, encode_action
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import (
    AtomDelete,
    AtomInsert,
    BondReorder,
    BondReroute,
    CycleCloseEdge,
    CycleOpenEdge,
    SemanticAtomRestate,
)
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from compose_v4.rewrite.tracelets import BondOrderChange, RingSystemRestate

ProgressCallback = Callable[[dict], None]
Role = tuple[str, int, int]
AtomSignature = tuple[int, int, int, int]


@dataclass(frozen=True)
class RealizerConfig:
    maximum_primitives: int = 32
    maximum_active_atoms: int = 40
    maximum_expansions: int = 16_384
    children_per_expansion: int = 12

    def __post_init__(self) -> None:
        for name in (
            "maximum_primitives",
            "maximum_active_atoms",
            "maximum_expansions",
            "children_per_expansion",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_primitives > 32 or self.maximum_active_atoms != 40:
            raise ValueError("realizer must preserve the declared 32-step/40-atom support")


@dataclass(frozen=True)
class _BoundGoal:
    target: MolecularGraph
    target_receipt: dict
    source_targets: tuple[tuple[int, AtomSignature | None], ...]
    source_classes: tuple[tuple[int, int], ...]
    output_roles: tuple[Role, ...]
    output_signatures: tuple[AtomSignature, ...]
    output_classes: tuple[int, ...]
    edge_targets: tuple[tuple[Role, Role, int], ...]
    output_automorphisms: tuple[tuple[int, ...], ...]


def target_distance(graph: MolecularGraph, target: MolecularGraph) -> int:
    """Exact persistent-coordinate mismatch used only as a compiler heuristic."""

    return int(
        np.count_nonzero(graph.atom_types != target.atom_types)
        + np.count_nonzero(graph.formal_charges != target.formal_charges)
        + np.count_nonzero(graph.implicit_h_counts != target.implicit_h_counts)
        + np.count_nonzero(np.triu(graph.bonds != target.bonds, 1))
    )


def _class(graph: MolecularGraph, slot: int) -> int | None:
    return ORGANIC_VOCABULARY.class_index(
        int(graph.atom_types[slot]),
        int(graph.bonds[slot].sum()),
        int(graph.implicit_h_counts[slot]),
        int(graph.formal_charges[slot]),
    )


def _target_actions(graph: MolecularGraph, target: MolecularGraph):
    """Yield generic target-directed Active8 proposals, never teacher actions."""

    real = {int(slot) for slot in np.flatnonzero(is_element(graph.atom_types))}
    wanted = {int(slot) for slot in np.flatnonzero(is_element(target.atom_types))}
    live = sorted(real & wanted)

    for slot in sorted(real - wanted):
        yield "atom_delete", AtomDelete(slot)

    for slot in live:
        target_class = _class(target, slot)
        if target_class is not None and _class(graph, slot) != target_class:
            yield "atom_restate_semantic", SemanticAtomRestate(slot, target_class)

    for slot in sorted(wanted - real):
        target_class = _class(target, slot)
        if target_class is None:
            continue
        for neighbor in live:
            desired_order = int(target.bonds[neighbor, slot])
            if not desired_order:
                continue
            for order in dict.fromkeys((desired_order, 1)):
                hydrogen = ORGANIC_VOCABULARY.h_count(
                    target_class, order, int(target.formal_charges[slot])
                )
                if hydrogen is not None:
                    yield (
                        "atom_insert",
                        AtomInsert(
                            slot,
                            int(target.atom_types[slot]),
                            int(target.formal_charges[slot]),
                            hydrogen,
                            ((neighbor, order),),
                        ),
                    )

    remove, add, reorders = [], [], []
    for left, right in combinations(live, 2):
        old, new = int(graph.bonds[left, right]), int(target.bonds[left, right])
        if old and not new:
            remove.append((left, right))
            yield "cycle_open", CycleOpenEdge(left, right)
        elif new and not old:
            add.append((left, right, new))
            yield "cycle_close", CycleCloseEdge(left, right, new)
        elif old and new and old != new:
            reorders.append(BondOrderChange(left, right, new))
            yield "bond_reorder", BondReorder(left, right, new)
    for left, right in remove:
        for new_left, new_right, order in add:
            yield "bond_reroute", BondReroute(left, right, new_left, new_right, order)
    if len(reorders) > 1:
        yield "ring_system_restate", RingSystemRestate(tuple(reorders))


def _replay_states(source: MolecularGraph, path: tuple[dict, ...]) -> list[dict]:
    system = editing_v2_semantic_rewrite_system()
    states = [encode_state(source)]
    graph = source
    for action_record in path:
        family, action = decode_action(action_record)
        graph = system.apply(graph, family, action)
        states.append(encode_state(graph))
    return states


def _deterministic_target_schedule(
    source: MolecularGraph,
    target: MolecularGraph,
    *,
    config: RealizerConfig,
    progress: ProgressCallback | None,
) -> dict:
    """Schedule the explicit graph delta before invoking bounded state search.

    The target's persistent slots are an internal compiler allocation produced
    after address-free binding. They are not part of the serialized subgoal.
    Each retained action is generated from the current graph/target delta and
    executed by the ordinary exact rewrite system. The role-aware search
    remains the fallback when this deterministic legal schedule reaches a
    local dead end.
    """

    system = editing_v2_semantic_rewrite_system()
    graph = source
    path: tuple[dict, ...] = ()
    visited = {exact_graph_key(source)}
    failures: Counter[str] = Counter()
    expanded = attempted = 0
    best_mismatch = target_distance(source, target)

    while expanded < min(config.maximum_expansions, config.maximum_primitives):
        mismatch = target_distance(graph, target)
        if mismatch == 0:
            return {
                "status": "realized",
                "actions": list(path),
                "states": _replay_states(source, path),
                "expanded": expanded,
                "attempted": attempted,
                "best_mismatch": 0,
                "rejections": dict(sorted(failures.items())),
                "primitive_teacher_actions_used": 0,
                "compiler_strategy": "deterministic_graph_delta_schedule",
            }
        if len(path) >= config.maximum_primitives:
            failures["primitive_limit"] += 1
            break
        expanded += 1
        if progress is not None:
            progress(
                {
                    "expanded": expanded,
                    "attempted": attempted,
                    "frontier": 1,
                    "depth": len(path),
                    "best_mismatch": best_mismatch,
                    "maximum_expansions": config.maximum_expansions,
                    "compiler_strategy": "deterministic_graph_delta_schedule",
                }
            )
        children = []
        current_key = canonical_state_key(graph)
        for family, action in _target_actions(graph, target):
            attempted += 1
            try:
                product = system.apply(graph, family, action)
            except (InvalidRewrite, ValueError) as error:
                failures[f"{family}:{str(error).split(':')[0]}"] += 1
                continue
            if product.n_real_atoms > config.maximum_active_atoms:
                failures["active_atom_limit"] += 1
                continue
            product_key = canonical_state_key(product)
            if product_key == current_key:
                failures["canonical_self_event"] += 1
                continue
            product_exact_key = exact_graph_key(product)
            if product_exact_key in visited:
                continue
            product_mismatch = target_distance(product, target)
            action_record = encode_action(family, action)
            children.append(
                (
                    product_mismatch,
                    family,
                    json.dumps(action_record, sort_keys=True, separators=(",", ":")),
                    product_exact_key,
                    product,
                    action_record,
                )
            )
        if not children:
            break
        (
            product_mismatch,
            _family,
            _action_key,
            product_exact_key,
            graph,
            action_record,
        ) = min(children, key=lambda row: row[:4])
        visited.add(product_exact_key)
        best_mismatch = min(best_mismatch, product_mismatch)
        path += (action_record,)

    return {
        "status": "deterministic_schedule_abstention",
        "actions": [],
        "states": [encode_state(source)],
        "expanded": expanded,
        "attempted": attempted,
        "best_mismatch": best_mismatch,
        "rejections": dict(sorted(failures.items())),
        "primitive_teacher_actions_used": 0,
        "compiler_strategy": "deterministic_graph_delta_schedule",
    }


def _source_role(slot: int) -> Role:
    return ("source", int(slot), -1)


def _output_role(subgoal_index: int, output_index: int) -> Role:
    return ("output", int(subgoal_index), int(output_index))


def _bound_goal(
    source: MolecularGraph,
    goal: StructuralGoal,
    bindings: tuple[tuple[int, ...], ...],
) -> _BoundGoal:
    target, receipt = instantiate_goal(
        source,
        goal,
        bindings,
        prefer_initially_empty_output_slots=True,
    )
    source_targets: dict[int, AtomSignature | None] = {}
    output_signatures: dict[Role, AtomSignature] = {}
    output_classes: dict[Role, int] = {}
    edge_targets: dict[tuple[Role, Role], int] = {}
    for subgoal_index, (subgoal, assignment) in enumerate(
        zip(goal.subgoals, bindings, strict=True)
    ):
        local_roles = tuple(_source_role(slot) for slot in assignment) + tuple(
            _output_role(subgoal_index, index) for index in range(len(subgoal.output_atoms))
        )
        local_targets = (*subgoal.target_atoms, *subgoal.output_atoms)
        for slot, signature in zip(assignment, subgoal.target_atoms, strict=True):
            previous = source_targets.setdefault(int(slot), signature)
            if previous != signature:
                raise ValueError("overlapping subgoals disagree on a source-role target")
        for index, signature in enumerate(subgoal.output_atoms):
            role = _output_role(subgoal_index, index)
            output_signatures[role] = signature
            bond_sum = sum(int(order) for order in subgoal.target_bonds[len(assignment) + index])
            class_index = ORGANIC_VOCABULARY.class_index(
                signature[0], bond_sum, signature[2], signature[1]
            )
            if class_index is None:
                raise ValueError("output role has no supported semantic atom class")
            output_classes[role] = class_index
        for left_index, left in enumerate(local_roles):
            if local_targets[left_index] is None:
                continue
            for right_index in range(left_index + 1, len(local_roles)):
                if local_targets[right_index] is None:
                    continue
                right = local_roles[right_index]
                pair = (left, right) if left < right else (right, left)
                order = int(subgoal.target_bonds[left_index][right_index])
                previous = edge_targets.setdefault(pair, order)
                if previous != order:
                    raise ValueError("overlapping subgoals disagree on a target bond")
    source_classes = []
    for slot, signature in source_targets.items():
        if signature is None:
            continue
        class_index = _class(target, slot)
        if class_index is None:
            raise ValueError("retained source role has no supported semantic atom class")
        source_classes.append((slot, class_index))
    roles = tuple(sorted(output_signatures))
    role_graph = nx.Graph()
    for slot, signature in source_targets.items():
        if signature is not None:
            role_graph.add_node(_source_role(slot), color=("source", int(slot)))
    for role in roles:
        role_graph.add_node(role, color=("output", output_signatures[role]))
    for (left, right), order in edge_targets.items():
        if order:
            role_graph.add_edge(left, right, order=order)
    role_index = {role: index for index, role in enumerate(roles)}
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        role_graph,
        role_graph,
        node_match=lambda left, right: left["color"] == right["color"],
        edge_match=lambda left, right: left["order"] == right["order"],
    )
    output_automorphisms = set()
    for automorphism in matcher.isomorphisms_iter():
        output_automorphisms.add(tuple(role_index[automorphism[role]] for role in roles))
    if not output_automorphisms:
        raise AssertionError("bound structural target has no identity automorphism")
    return _BoundGoal(
        target=target,
        target_receipt=receipt,
        source_targets=tuple(sorted(source_targets.items())),
        source_classes=tuple(sorted(source_classes)),
        output_roles=roles,
        output_signatures=tuple(output_signatures[role] for role in roles),
        output_classes=tuple(output_classes[role] for role in roles),
        edge_targets=tuple(
            (left, right, order) for (left, right), order in sorted(edge_targets.items())
        ),
        output_automorphisms=tuple(sorted(output_automorphisms)),
    )


def _role_slots(bound: _BoundGoal, mapping: tuple[int, ...]) -> dict[Role, int]:
    slots = {
        _source_role(slot): slot
        for slot, signature in bound.source_targets
        if signature is not None
    }
    slots.update(
        {role: slot for role, slot in zip(bound.output_roles, mapping, strict=True) if slot >= 0}
    )
    return slots


def _canonical_mapping(bound: _BoundGoal, mapping: tuple[int, ...]) -> tuple[int, ...]:
    """Quotient logical-role assignments by exact target-patch automorphisms."""

    equivalent = []
    for permutation in bound.output_automorphisms:
        transformed = [-1] * len(mapping)
        for old_index, new_index in enumerate(permutation):
            transformed[new_index] = mapping[old_index]
        equivalent.append(tuple(transformed))
    return min(equivalent)


def _atom_signature(graph: MolecularGraph, slot: int) -> AtomSignature:
    return (
        int(graph.atom_types[slot]),
        int(graph.formal_charges[slot]),
        int(graph.implicit_h_counts[slot]),
        int(np.count_nonzero(graph.bonds[slot])),
    )


def _subgoal_target_matches(
    graph: MolecularGraph,
    goal: StructuralGoal,
    bindings: tuple[tuple[int, ...], ...],
    bound: _BoundGoal,
    mapping: tuple[int, ...],
) -> tuple[bool, ...]:
    """Check each local target inside the realized complete coordinated goal."""

    role_slots = _role_slots(bound, mapping)
    mapped_output_slots = {slot for slot in mapping if slot >= 0}
    matches = []
    for subgoal_index, (subgoal, assignment) in enumerate(
        zip(goal.subgoals, bindings, strict=True)
    ):
        local: list[tuple[int, int]] = []
        matched = True
        for local_index, (slot, signature) in enumerate(
            zip(assignment, subgoal.target_atoms, strict=True)
        ):
            if signature is None:
                matched &= not is_element(graph.atom_types[slot]) or slot in mapped_output_slots
                continue
            matched &= (
                is_element(graph.atom_types[slot]) and _atom_signature(graph, slot) == signature
            )
            local.append((local_index, slot))
        for output_index, signature in enumerate(subgoal.output_atoms):
            role = _output_role(subgoal_index, output_index)
            slot = role_slots.get(role)
            matched &= slot is not None and _atom_signature(graph, slot) == signature
            if slot is not None:
                local.append((len(assignment) + output_index, slot))
        for left_offset, (left_index, left_slot) in enumerate(local):
            for right_index, right_slot in local[left_offset + 1 :]:
                matched &= (
                    int(graph.bonds[left_slot, right_slot])
                    == subgoal.target_bonds[left_index][right_index]
                )
        matches.append(bool(matched))
    return tuple(matches)


def _role_distance(graph: MolecularGraph, bound: _BoundGoal, mapping: tuple[int, ...]) -> int:
    mapped_output_slots = {slot for slot in mapping if slot >= 0}
    mismatch = 0
    for slot, signature in bound.source_targets:
        active = bool(is_element(graph.atom_types[slot]))
        if signature is None:
            if active and slot not in mapped_output_slots:
                mismatch += 1 + int(np.count_nonzero(graph.bonds[slot]))
        elif not active:
            mismatch += 4
        else:
            current = _atom_signature(graph, slot)
            mismatch += sum(left != right for left, right in zip(current, signature, strict=True))
    for slot, signature in zip(mapping, bound.output_signatures, strict=True):
        if slot < 0 or not is_element(graph.atom_types[slot]):
            mismatch += 4
        else:
            current = _atom_signature(graph, slot)
            mismatch += sum(left != right for left, right in zip(current, signature, strict=True))
    slots = _role_slots(bound, mapping)
    for left, right, order in bound.edge_targets:
        if left in slots and right in slots:
            mismatch += int(int(graph.bonds[slots[left], slots[right]]) != order)
        elif order:
            mismatch += 1
    return int(mismatch)


def _role_actions(
    graph: MolecularGraph,
    bound: _BoundGoal,
    mapping: tuple[int, ...],
):
    """Yield legal-action proposals plus any newly assigned output-role index."""

    mapped_output_slots = {slot for slot in mapping if slot >= 0}
    for slot, signature in bound.source_targets:
        if (
            signature is None
            and slot not in mapped_output_slots
            and is_element(graph.atom_types[slot])
        ):
            yield "atom_delete", AtomDelete(slot), None

    for slot, target_class in bound.source_classes:
        if is_element(graph.atom_types[slot]) and _class(graph, slot) != target_class:
            yield "atom_restate_semantic", SemanticAtomRestate(slot, target_class), None

    role_slots = _role_slots(bound, mapping)
    empty = tuple(int(slot) for slot in np.flatnonzero(~is_element(graph.atom_types)))
    if empty:
        output_index = {role: index for index, role in enumerate(bound.output_roles)}
        neighbor_targets: dict[Role, list[tuple[Role, int]]] = {
            role: [] for role in bound.output_roles
        }
        for left, right, order in bound.edge_targets:
            if order and left in neighbor_targets:
                neighbor_targets[left].append((right, order))
            if order and right in neighbor_targets:
                neighbor_targets[right].append((left, order))
        for role, signature, target_class, assigned in zip(
            bound.output_roles,
            bound.output_signatures,
            bound.output_classes,
            mapping,
            strict=True,
        ):
            if assigned >= 0:
                continue
            for neighbor_role, order in sorted(neighbor_targets[role]):
                neighbor_slot = role_slots.get(neighbor_role)
                if neighbor_slot is None:
                    continue
                hydrogen = ORGANIC_VOCABULARY.h_count(target_class, order, signature[1])
                if hydrogen is None:
                    continue
                yield (
                    "atom_insert",
                    AtomInsert(
                        empty[0],
                        signature[0],
                        signature[1],
                        hydrogen,
                        ((neighbor_slot, order),),
                    ),
                    output_index[role],
                )

    remove, add, reorders = [], [], []
    for left, right, order in bound.edge_targets:
        if left not in role_slots or right not in role_slots:
            continue
        left_slot, right_slot = sorted((role_slots[left], role_slots[right]))
        old = int(graph.bonds[left_slot, right_slot])
        if old and not order:
            remove.append((left_slot, right_slot))
            yield "cycle_open", CycleOpenEdge(left_slot, right_slot), None
        elif order and not old:
            add.append((left_slot, right_slot, order))
            yield "cycle_close", CycleCloseEdge(left_slot, right_slot, order), None
        elif old and order and old != order:
            reorders.append(BondOrderChange(left_slot, right_slot, order))
            yield "bond_reorder", BondReorder(left_slot, right_slot, order), None
    for left, right in remove:
        for new_left, new_right, order in add:
            yield (
                "bond_reroute",
                BondReroute(left, right, new_left, new_right, order),
                None,
            )
    if len(reorders) > 1:
        yield "ring_system_restate", RingSystemRestate(tuple(reorders)), None


def realize_target(
    source: MolecularGraph,
    target: MolecularGraph,
    *,
    config: RealizerConfig | None = None,
    progress: ProgressCallback | None = None,
) -> dict:
    """Find a legal exact program for one bound target or return an abstention."""

    config = RealizerConfig() if config is None else config
    if source.n_atoms != target.n_atoms:
        raise ValueError("source and structural target slot counts disagree")
    if not charge_policy_preserved(source, target):
        return {
            "status": "charge_policy_abstention",
            "actions": [],
            "states": [encode_state(source)],
            "expanded": 0,
            "attempted": 0,
            "best_mismatch": target_distance(source, target),
        }

    system = editing_v2_semantic_rewrite_system()
    target_key = canonical_state_key(target)
    serial = count()
    queue = [(target_distance(source, target), 0, next(serial), source, ())]
    visited: set[tuple] = set()
    failures: Counter[str] = Counter()
    expanded = attempted = 0
    best_mismatch = target_distance(source, target)

    while queue and expanded < config.maximum_expansions:
        _mismatch, depth, _, graph, path = heapq.heappop(queue)
        key = exact_graph_key(graph)
        if key in visited:
            continue
        visited.add(key)
        if canonical_state_key(graph) == target_key:
            states = [encode_state(source)]
            replay = source
            for action_record in path:
                family, action = decode_action(action_record)
                replay = system.apply(replay, family, action)
                states.append(encode_state(replay))
            return {
                "status": "realized",
                "actions": list(path),
                "states": states,
                "expanded": expanded,
                "attempted": attempted,
                "best_mismatch": 0,
                "rejections": dict(sorted(failures.items())),
                "primitive_teacher_actions_used": 0,
            }
        if depth >= config.maximum_primitives:
            failures["primitive_limit"] += 1
            continue

        expanded += 1
        if progress is not None:
            progress(
                {
                    "expanded": expanded,
                    "attempted": attempted,
                    "frontier": len(queue),
                    "depth": depth,
                    "best_mismatch": best_mismatch,
                    "maximum_expansions": config.maximum_expansions,
                }
            )
        children = {}
        current_key = canonical_state_key(graph)
        for family, action in _target_actions(graph, target):
            attempted += 1
            try:
                product = system.apply(graph, family, action)
            except (InvalidRewrite, ValueError) as error:
                failures[f"{family}:{str(error).split(':')[0]}"] += 1
                continue
            if product.n_real_atoms > config.maximum_active_atoms:
                failures["active_atom_limit"] += 1
                continue
            product_key = canonical_state_key(product)
            if product_key == current_key:
                failures["canonical_self_event"] += 1
                continue
            product_exact_key = exact_graph_key(product)
            if product_exact_key in visited:
                continue
            product_mismatch = target_distance(product, target)
            best_mismatch = min(best_mismatch, product_mismatch)
            candidate = (
                product_mismatch + 0.02 * (depth + 1),
                next(serial),
                product,
                path + (encode_action(family, action),),
            )
            previous = children.get(product_exact_key)
            if previous is None or candidate[:2] < previous[:2]:
                children[product_exact_key] = candidate
        for score, order, product, actions in sorted(children.values())[
            : config.children_per_expansion
        ]:
            heapq.heappush(queue, (score, depth + 1, order, product, actions))

    return {
        "status": "search_limit_abstention" if queue else "frontier_exhausted_abstention",
        "actions": [],
        "states": [encode_state(source)],
        "expanded": expanded,
        "attempted": attempted,
        "best_mismatch": best_mismatch,
        "rejections": dict(sorted(failures.items())),
        "primitive_teacher_actions_used": 0,
    }


def _realize_bound_goal(
    source: MolecularGraph,
    bound: _BoundGoal,
    *,
    config: RealizerConfig,
    progress: ProgressCallback | None,
) -> dict:
    """Realize logical output roles without fixing their persistent slots in advance."""

    if not charge_policy_preserved(source, bound.target):
        initial_mapping = _canonical_mapping(bound, tuple(-1 for _ in bound.output_roles))
        return {
            "status": "charge_policy_abstention",
            "actions": [],
            "states": [encode_state(source)],
            "expanded": 0,
            "attempted": 0,
            "best_mismatch": _role_distance(source, bound, initial_mapping),
            "rejections": {},
            "primitive_teacher_actions_used": 0,
            "output_role_slots": [],
            "logical_role_automorphisms": len(bound.output_automorphisms),
            "compiler_strategy": "charge_policy_abstention",
            "deterministic_schedule_status": "not_run",
        }

    target_mapping = _canonical_mapping(
        bound,
        tuple(int(slot) for group in bound.target_receipt["output_slots"] for slot in group),
    )
    if len(target_mapping) != len(bound.output_roles):
        raise AssertionError("bound target output allocation disagrees with logical roles")
    scheduled = _deterministic_target_schedule(
        source,
        bound.target,
        config=config,
        progress=progress,
    )
    if scheduled["status"] == "realized":
        endpoint = decode_state(scheduled["states"][-1])
        if _role_distance(endpoint, bound, target_mapping) != 0:
            raise AssertionError("exact target schedule did not satisfy logical role obligations")
        return {
            **scheduled,
            "output_role_slots": list(target_mapping),
            "logical_role_automorphisms": len(bound.output_automorphisms),
            "deterministic_schedule_status": "realized",
        }

    system = editing_v2_semantic_rewrite_system()
    target_key = canonical_state_key(bound.target)
    initial_mapping = _canonical_mapping(bound, tuple(-1 for _ in bound.output_roles))
    serial = count()
    queue = [
        (
            _role_distance(source, bound, initial_mapping),
            0,
            next(serial),
            source,
            initial_mapping,
            (),
        )
    ]
    visited: set[tuple] = set()
    failures: Counter[str] = Counter()
    expanded = int(scheduled["expanded"])
    attempted = int(scheduled["attempted"])
    best_mismatch = min(
        int(scheduled["best_mismatch"]),
        _role_distance(source, bound, initial_mapping),
    )

    while queue and expanded < config.maximum_expansions:
        _mismatch, depth, _, graph, mapping, path = heapq.heappop(queue)
        key = (exact_graph_key(graph), mapping)
        if key in visited:
            continue
        visited.add(key)
        if canonical_state_key(graph) == target_key and _role_distance(graph, bound, mapping) == 0:
            states = [encode_state(source)]
            replay = source
            for action_record in path:
                family, action = decode_action(action_record)
                replay = system.apply(replay, family, action)
                states.append(encode_state(replay))
            return {
                "status": "realized",
                "actions": list(path),
                "states": states,
                "expanded": expanded,
                "attempted": attempted,
                "best_mismatch": 0,
                "rejections": dict(sorted(failures.items())),
                "primitive_teacher_actions_used": 0,
                "output_role_slots": list(mapping),
                "logical_role_automorphisms": len(bound.output_automorphisms),
                "compiler_strategy": "role_aware_bounded_search",
                "deterministic_schedule_status": scheduled["status"],
            }
        if depth >= config.maximum_primitives:
            failures["primitive_limit"] += 1
            continue

        expanded += 1
        if progress is not None:
            progress(
                {
                    "expanded": expanded,
                    "attempted": attempted,
                    "frontier": len(queue),
                    "depth": depth,
                    "best_mismatch": best_mismatch,
                    "maximum_expansions": config.maximum_expansions,
                }
            )
        children = {}
        current_key = canonical_state_key(graph)
        for family, action, output_index in _role_actions(graph, bound, mapping):
            attempted += 1
            try:
                product = system.apply(graph, family, action)
            except (InvalidRewrite, ValueError) as error:
                failures[f"{family}:{str(error).split(':')[0]}"] += 1
                continue
            if product.n_real_atoms > config.maximum_active_atoms:
                failures["active_atom_limit"] += 1
                continue
            product_key = canonical_state_key(product)
            if product_key == current_key:
                failures["canonical_self_event"] += 1
                continue
            next_mapping = mapping
            if output_index is not None:
                if mapping[output_index] >= 0:
                    raise AssertionError("output role was assigned twice")
                mutable_mapping = list(mapping)
                mutable_mapping[output_index] = int(action.slot)
                next_mapping = _canonical_mapping(bound, tuple(mutable_mapping))
            exact_key = (exact_graph_key(product), next_mapping)
            if exact_key in visited:
                continue
            mismatch = _role_distance(product, bound, next_mapping)
            best_mismatch = min(best_mismatch, mismatch)
            candidate = (
                mismatch + 0.02 * (depth + 1),
                next(serial),
                product,
                next_mapping,
                path + (encode_action(family, action),),
            )
            previous = children.get(exact_key)
            if previous is None or candidate[:2] < previous[:2]:
                children[exact_key] = candidate
        for score, order, product, next_mapping, actions in sorted(children.values())[
            : config.children_per_expansion
        ]:
            heapq.heappush(
                queue,
                (score, depth + 1, order, product, next_mapping, actions),
            )

    return {
        "status": "search_limit_abstention" if queue else "frontier_exhausted_abstention",
        "actions": [],
        "states": [encode_state(source)],
        "expanded": expanded,
        "attempted": attempted,
        "best_mismatch": best_mismatch,
        "rejections": dict(sorted(failures.items())),
        "primitive_teacher_actions_used": 0,
        "output_role_slots": [],
        "logical_role_automorphisms": len(bound.output_automorphisms),
        "compiler_strategy": "role_aware_bounded_search",
        "deterministic_schedule_status": scheduled["status"],
    }


def realize_structural_goal(
    source: MolecularGraph,
    goal: StructuralGoal,
    bindings: tuple[tuple[int, ...], ...],
    *,
    config: RealizerConfig | None = None,
    progress: ProgressCallback | None = None,
) -> dict:
    """Instantiate and realize a generated structural goal without teacher actions."""

    bound = _bound_goal(source, goal, bindings)
    config = RealizerConfig() if config is None else config
    result = _realize_bound_goal(source, bound, config=config, progress=progress)
    endpoint = decode_state(result["states"][-1])
    output_role_slots = tuple(result.get("output_role_slots", ()))
    subgoal_matches = (
        _subgoal_target_matches(endpoint, goal, bindings, bound, output_role_slots)
        if result["status"] == "realized"
        else tuple(False for _ in goal.subgoals)
    )
    return {
        **result,
        "goal_id": goal.goal_id,
        "bound_target_receipt": bound.target_receipt,
        "endpoint_matches_bound_target": (
            result["status"] == "realized"
            and canonical_state_key(bound.target) == canonical_state_key(endpoint)
        ),
        "subgoal_targets_match_within_complete_goal": list(subgoal_matches),
    }


__all__ = [
    "RealizerConfig",
    "realize_structural_goal",
    "realize_target",
    "target_distance",
]
