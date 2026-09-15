"""Macro-free dependency-region programs for exact primitive traces."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.generic_legal_action_policy import _record_parts
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "dependency_region_program_v1"
TRAINING_SCHEMA = "dependency_region_training_corpus_v1"


@dataclass(frozen=True)
class DependencyRegionConfig:
    runtime_maximum_primitives: int = 32
    runtime_maximum_components: int = 8
    join_lifetime_neighbors: bool = True

    def __post_init__(self):
        if min(self.runtime_maximum_primitives, self.runtime_maximum_components) < 1:
            raise ValueError("dependency-region limits must be positive")


class _UnionFind:
    def __init__(self, size: int):
        self.parents = list(range(size))

    def find(self, value: int) -> int:
        parent = self.parents[value]
        if parent != value:
            self.parents[value] = self.find(parent)
        return self.parents[value]

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parents[right_root] = left_root


def _connected_tokens(
    left: set[tuple[str, int]],
    right: set[tuple[str, int]],
    lifetime_bonds: set[frozenset[tuple[str, int]]],
    *,
    join_lifetime_neighbors: bool = True,
) -> bool:
    if left & right:
        return True
    return join_lifetime_neighbors and any(
        frozenset((a, b)) in lifetime_bonds for a in left for b in right
    )


def _trace_structure(states: tuple[dict, ...], actions: tuple[dict, ...]) -> dict:
    graphs = [decode_state(state) for state in states]
    initial_slots = [
        int(slot) for slot in np.flatnonzero(is_element(graphs[0].atom_types))
    ]
    active = {slot: ("source", ordinal) for ordinal, slot in enumerate(initial_slots)}
    initial_token_slots = {
        ("source", ordinal): slot for ordinal, slot in enumerate(initial_slots)
    }
    lifetime_bonds: set[frozenset[tuple[str, int]]] = set()
    footprints = []
    reference_tokens = []
    created_tokens = []
    created_handles = []
    dependency_edges = []
    handle_by_token = {}

    def record_bonds(graph) -> None:
        for left, right in zip(*np.nonzero(np.triu(graph.bonds != 0, 1)), strict=True):
            left_token, right_token = active.get(int(left)), active.get(int(right))
            if left_token is not None and right_token is not None:
                lifetime_bonds.add(frozenset((left_token, right_token)))

    for step, (graph, record) in enumerate(zip(graphs[:-1], actions, strict=True)):
        record_bonds(graph)
        footprint_slots, reference_slots, created_slot, deleted_slot = _record_parts(
            record, graph
        )
        missing = sorted(slot for slot in reference_slots if slot not in active)
        if missing:
            raise ValueError(f"step {step} references inactive slots: {missing}")
        references = {active[slot] for slot in reference_slots}
        footprint = {active[slot] for slot in footprint_slots if slot in active}
        for token in sorted(references):
            if token[0] != "created":
                continue
            handle = handle_by_token[token]
            edge = {
                "producer": handle["producer"],
                "consumer": step,
                "primitive_lag": step - handle["producer"],
                "created_backreference": len(created_handles) - token[1],
            }
            dependency_edges.append(edge)
            handle["consumers"].append(step)
        if deleted_slot is not None:
            active.pop(deleted_slot, None)
        created_token = None
        if created_slot is not None:
            if created_slot in active:
                raise ValueError(f"step {step} overwrites active slot {created_slot}")
            created_token = ("created", len(created_handles))
            active[created_slot] = created_token
            handle = {
                "producer": step,
                "created_ordinal": len(created_handles),
                "slot": created_slot,
                "consumers": [],
            }
            created_handles.append(handle)
            handle_by_token[created_token] = handle
            footprint.add(created_token)
        footprints.append(footprint)
        reference_tokens.append(references)
        created_tokens.append(created_token)
    record_bonds(graphs[-1])
    return {
        "graphs": graphs,
        "footprints": footprints,
        "reference_tokens": reference_tokens,
        "created_tokens": created_tokens,
        "created_handles": created_handles,
        "dependency_edges": dependency_edges,
        "lifetime_bonds": lifetime_bonds,
        "initial_token_slots": initial_token_slots,
        "final_active_slots": dict(active),
    }


def trace_structure(states: tuple[dict, ...], actions: tuple[dict, ...]) -> dict:
    """Expose exact atom-lifetime structure for downstream subgoal extraction."""

    return _trace_structure(states, actions)


def dependency_region_program(
    states: tuple[dict, ...],
    actions: tuple[dict, ...],
    config: DependencyRegionConfig | None = None,
) -> dict:
    """Represent an exact trace as connected components and ordered emissions."""

    if config is None:
        config = DependencyRegionConfig()
    if len(states) != len(actions) + 1 or not actions:
        raise ValueError("a dependency-region trace needs nonempty actions and states")
    structure = _trace_structure(states, actions)
    footprints = structure["footprints"]
    lifetime_bonds = structure["lifetime_bonds"]
    union = _UnionFind(len(actions))
    structural_edges = []
    for left in range(len(actions)):
        for right in range(left + 1, len(actions)):
            if _connected_tokens(
                footprints[left],
                footprints[right],
                lifetime_bonds,
                join_lifetime_neighbors=config.join_lifetime_neighbors,
            ):
                union.union(left, right)
                structural_edges.append({"left": left, "right": right})
    for edge in structure["dependency_edges"]:
        union.union(edge["producer"], edge["consumer"])

    cycle_edges = []
    rules = [str(action["executor_rule"]) for action in actions]
    for start, rule in enumerate(rules):
        if rule != "cycle_open":
            continue
        for stop in range(start + 1, len(actions)):
            if rules[stop] not in {"cycle_close", "ring_system_restate"}:
                continue
            if _connected_tokens(
                footprints[start],
                footprints[stop],
                lifetime_bonds,
                join_lifetime_neighbors=config.join_lifetime_neighbors,
            ):
                union.union(start, stop)
                cycle_edges.append({"open": start, "close_or_restate": stop})
                break

    raw_components = defaultdict(list)
    for step in range(len(actions)):
        raw_components[union.find(step)].append(step)
    ordered_groups = sorted(raw_components.values(), key=lambda group: group[0])
    action_component = {
        step: component_index
        for component_index, group in enumerate(ordered_groups)
        for step in group
    }
    component_microstep = {
        step: microstep
        for group in ordered_groups
        for microstep, step in enumerate(group)
    }
    component_sequence = [action_component[step] for step in range(len(actions))]
    component_runs = []
    for component in component_sequence:
        if not component_runs or component_runs[-1] != component:
            component_runs.append(component)
    seen_components = set()
    component_reentries = 0
    for component in component_runs:
        if component in seen_components:
            component_reentries += 1
        seen_components.add(component)
    components = []
    for component_index, group in enumerate(ordered_groups):
        component_rules = Counter(rules[step] for step in group)
        components.append(
            {
                "component_index": component_index,
                "primitive_indices": group,
                "primitive_count": len(group),
                "rule_counts": dict(sorted(component_rules.items())),
                "changed_atom_lifetime_count": len(
                    set().union(*(footprints[step] for step in group))
                ),
                "created_outputs": sum(
                    structure["created_tokens"][step] is not None for step in group
                ),
                "contains_cycle_open": bool(component_rules["cycle_open"]),
                "contains_cycle_close": bool(component_rules["cycle_close"]),
                "contains_ring_restate": bool(component_rules["ring_system_restate"]),
            }
        )

    emissions = []
    for step, rule in enumerate(rules):
        component = action_component[step]
        microstep = component_microstep[step]
        group = ordered_groups[component]
        before = "open" if microstep == 0 else "continue"
        after = "close" if microstep == len(group) - 1 else None
        created_references = []
        for token in sorted(structure["reference_tokens"][step]):
            if token[0] != "created":
                continue
            producer = structure["created_handles"][token[1]]["producer"]
            created_references.append(
                {
                    "created_backreference": len(
                        [
                            created
                            for created in structure["created_tokens"][:step]
                            if created is not None
                        ]
                    )
                    - token[1],
                    "producer_component_backreference": component
                    - action_component[producer],
                    "producer_primitive_lag": step - producer,
                }
            )
        created_output = structure["created_tokens"][step]
        emissions.append(
            {
                "emission_index": step,
                "component_index": component,
                "component_microstep": microstep,
                "control_before": before,
                "control_after": after,
                "primitive_rule": rule,
                "source_reference_count": sum(
                    token[0] == "source"
                    for token in structure["reference_tokens"][step]
                ),
                "created_handle_references": created_references,
                "creates_output": created_output is not None,
                "created_output_component_ordinal": (
                    sum(
                        structure["created_tokens"][member] is not None
                        for member in group
                        if member <= step
                    )
                    if created_output is not None
                    else None
                ),
            }
        )

    replay_actions = [actions[row["emission_index"]] for row in emissions]
    _, receipt = execute_program(structure["graphs"][0], replay_actions)
    exact_replay = receipt["states"] == list(states)
    if not exact_replay:
        raise ValueError("dependency-region schedule differs from exact trace")
    cross_created = sum(
        action_component[edge["producer"]] != action_component[edge["consumer"]]
        for edge in structure["dependency_edges"]
    )
    cross_cycle = sum(
        action_component[edge["open"]] != action_component[edge["close_or_restate"]]
        for edge in cycle_edges
    )
    runtime_length = len(actions) <= config.runtime_maximum_primitives
    component_budget = len(components) <= config.runtime_maximum_components
    return {
        "components": components,
        "join_lifetime_neighbors": config.join_lifetime_neighbors,
        "emissions": emissions,
        "created_handles": structure["created_handles"],
        "created_dependency_edges": structure["dependency_edges"],
        "cycle_dependency_edges": cycle_edges,
        "structural_connectivity_edges": structural_edges,
        "cross_component_created_dependency_edges": cross_created,
        "cross_component_cycle_dependency_edges": cross_cycle,
        "primitive_transitions": len(actions),
        "component_count": len(components),
        "component_sequence": component_sequence,
        "component_run_sequence": component_runs,
        "component_run_count": len(component_runs),
        "component_reentries": component_reentries,
        "contiguous_component_schedule": component_reentries == 0,
        "singleton_components": sum(len(group) == 1 for group in ordered_groups),
        "exact_replay": exact_replay,
        "runtime_length_supported": runtime_length,
        "component_budget_supported": component_budget,
        "complete_representation_supported": runtime_length and component_budget,
        "abstention_reason": (
            None
            if runtime_length and component_budget
            else (
                "primitive_budget_exceeded"
                if not runtime_length
                else "component_budget_exceeded"
            )
        ),
    }


def dependency_region_summary(routes: list[dict]) -> dict:
    if not routes:
        raise ValueError("dependency-region summary requires routes")
    programs = [route["dependency_region_program"] for route in routes]
    represented = [
        program for program in programs if program["complete_representation_supported"]
    ]
    runtime = [program for program in programs if program["runtime_length_supported"]]
    exact = sum(program["exact_replay"] for program in programs)
    exact_represented = sum(program["exact_replay"] for program in represented)
    component_counts = np.asarray(
        [program["component_count"] for program in programs], dtype=np.int64
    )
    run_counts = np.asarray(
        [program["component_run_count"] for program in programs], dtype=np.int64
    )
    component_sizes = np.asarray(
        [
            component["primitive_count"]
            for program in programs
            for component in program["components"]
        ],
        dtype=np.int64,
    )

    def five_number(values: np.ndarray) -> dict:
        quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
        return {
            key: float(value)
            for key, value in zip(
                ("minimum", "q1", "median", "q3", "maximum"),
                quantiles,
                strict=True,
            )
        }

    return {
        "routes": len(routes),
        "primitive_transitions": sum(
            program["primitive_transitions"] for program in programs
        ),
        "components": sum(program["component_count"] for program in programs),
        "component_count_distribution": dict(
            sorted(
                Counter(
                    str(program["component_count"]) for program in programs
                ).items(),
                key=lambda row: int(row[0]),
            )
        ),
        "component_count_five_number": five_number(component_counts),
        "component_run_count_distribution": dict(
            sorted(
                Counter(
                    str(program["component_run_count"]) for program in programs
                ).items(),
                key=lambda row: int(row[0]),
            )
        ),
        "component_run_count_five_number": five_number(run_counts),
        "component_size_distribution": dict(
            sorted(
                Counter(
                    str(component["primitive_count"])
                    for program in programs
                    for component in program["components"]
                ).items(),
                key=lambda row: int(row[0]),
            )
        ),
        "component_size_five_number": five_number(component_sizes),
        "contiguous_component_schedule_routes": sum(
            program["contiguous_component_schedule"] for program in programs
        ),
        "component_reentries": sum(
            program["component_reentries"] for program in programs
        ),
        "singleton_components": sum(
            program["singleton_components"] for program in programs
        ),
        "created_dependency_edges": sum(
            len(program["created_dependency_edges"]) for program in programs
        ),
        "cycle_dependency_edges": sum(
            len(program["cycle_dependency_edges"]) for program in programs
        ),
        "cross_component_created_dependency_edges": sum(
            program["cross_component_created_dependency_edges"] for program in programs
        ),
        "cross_component_cycle_dependency_edges": sum(
            program["cross_component_cycle_dependency_edges"] for program in programs
        ),
        "exact_replay_routes": exact,
        "exact_replay_coverage": exact / len(programs),
        "exact_replay_precision": exact / len(programs),
        "runtime_length_routes": len(runtime),
        "complete_representation_routes": len(represented),
        "complete_representation_coverage_all_routes": len(represented) / len(programs),
        "complete_representation_coverage_runtime_length": (
            len(represented) / len(runtime) if runtime else None
        ),
        "complete_representation_exact_replay_precision": (
            exact_represented / len(represented) if represented else None
        ),
        "primitive_budget_abstentions": sum(
            program["abstention_reason"] == "primitive_budget_exceeded"
            for program in programs
        ),
        "component_budget_abstentions": sum(
            program["abstention_reason"] == "component_budget_exceeded"
            for program in programs
        ),
    }


__all__ = [
    "SCHEMA",
    "TRAINING_SCHEMA",
    "DependencyRegionConfig",
    "dependency_region_program",
    "dependency_region_summary",
    "trace_structure",
]
