"""Bounded exact realization of a generated structural graph subgoal.

The realizer is conditioned on a requested target graph. It never receives a
teacher action sequence and never falls back to one. Target-directed actions
are proposed through the existing Active8 executor and every retained prefix
is an exact legal molecular state.
"""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations, count

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


def realize_structural_goal(
    source: MolecularGraph,
    goal: StructuralGoal,
    bindings: tuple[tuple[int, ...], ...],
    *,
    config: RealizerConfig | None = None,
    progress: ProgressCallback | None = None,
) -> dict:
    """Instantiate and realize a generated structural goal without teacher actions."""

    target, target_receipt = instantiate_goal(source, goal, bindings)
    result = realize_target(source, target, config=config, progress=progress)
    return {
        **result,
        "goal_id": goal.goal_id,
        "bound_target_receipt": target_receipt,
        "endpoint_matches_bound_target": (
            result["status"] == "realized"
            and canonical_state_key(target)
            == canonical_state_key(decode_state(result["states"][-1]))
        ),
    }


__all__ = [
    "RealizerConfig",
    "realize_structural_goal",
    "realize_target",
    "target_distance",
]
