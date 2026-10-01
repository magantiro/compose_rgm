"""Generic structural segmentation of exact PMO primitive programs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "pmo_exact_program_segmentation_v1"
TRAINING_SCHEMA = "pmo_exact_segmented_training_corpus_v1"


@dataclass(frozen=True)
class SegmentationConfig:
    maximum_primitives_per_segment: int = 8
    runtime_maximum_primitives: int = 32
    runtime_maximum_segments: int = 8
    multi_action_label: str = "connected_change_dependency_segment"
    fallback_label: str = "primitive_fallback"

    def __post_init__(self):
        if (
            min(
                self.maximum_primitives_per_segment,
                self.runtime_maximum_primitives,
                self.runtime_maximum_segments,
            )
            < 1
        ):
            raise ValueError("segmentation limits must be positive")
        if not self.multi_action_label or not self.fallback_label:
            raise ValueError("segmentation labels must be explicit")


def _record_parts(
    record: dict, graph
) -> tuple[set[int], set[int], int | None, int | None]:
    """Return footprint, input references, created slot and deleted slot."""

    rule, payload = record["executor_rule"], record["payload"]
    created = deleted = None
    if rule == "atom_insert":
        created = int(payload["slot"])
        references = {int(row[0]) for row in payload["neighbors"]}
        footprint = {*references, created}
    elif rule in {"atom_delete", "atom_restate_semantic"}:
        atom = int(payload["v"])
        references = {atom}
        footprint = {atom}
        if rule == "atom_delete":
            deleted = atom
            footprint.update(int(value) for value in np.flatnonzero(graph.bonds[atom]))
    elif rule in {"cycle_close", "cycle_open", "bond_reorder"}:
        references = {int(payload["a"]), int(payload["b"])}
        footprint = set(references)
    elif rule == "bond_reroute":
        references = {int(payload[name]) for name in ("a", "b", "u", "v")}
        footprint = set(references)
    elif rule == "ring_system_restate":
        references = {
            int(change[name]) for change in payload["changes"] for name in ("a", "b")
        }
        footprint = set(references)
    else:
        raise ValueError(f"unsupported PMO primitive rule: {rule!r}")
    return footprint, references, created, deleted


def _regions_connected(left: set[int], right: set[int], graph) -> bool:
    if left & right:
        return True
    for a in left:
        for b in right:
            if 0 <= a < graph.n_atoms and 0 <= b < graph.n_atoms and graph.bonds[a, b]:
                return True
    return False


def _merge_intervals(intervals: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged = []
    for start, stop in sorted(intervals):
        if start >= stop:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return tuple((int(start), int(stop)) for start, stop in merged)


def segment_exact_trace(
    states: tuple[dict, ...], actions: tuple[dict, ...], config: SegmentationConfig
) -> dict:
    """Partition one exact trace by structural continuity and hard dependencies."""

    if len(states) != len(actions) + 1 or not actions:
        raise ValueError("an exact trace needs one more state than nonempty actions")
    graphs = [decode_state(state) for state in states]
    footprints = []
    rules = []
    handles = []
    dependency_edges = []
    active_handles: dict[int, dict] = {}
    next_ordinal = 0
    for step, (graph, record) in enumerate(zip(graphs[:-1], actions, strict=True)):
        footprint, references, created, deleted = _record_parts(record, graph)
        footprints.append(footprint)
        rules.append(str(record["executor_rule"]))
        for slot in sorted(references):
            handle = active_handles.get(slot)
            if handle is not None:
                handle["consumers"].append(step)
                dependency_edges.append(
                    {
                        "producer": handle["producer"],
                        "consumer": step,
                        "created_ordinal": handle["created_ordinal"],
                        "creation_lag": step - handle["producer"],
                    }
                )
        if deleted is not None:
            active_handles.pop(deleted, None)
        if created is not None:
            if created in active_handles:
                raise ValueError("an atom insert overwrote an active created handle")
            handle = {
                "slot": created,
                "producer": step,
                "created_ordinal": next_ordinal,
                "consumers": [],
            }
            handles.append(handle)
            active_handles[created] = handle
            next_ordinal += 1

    intervals = [
        (handle["producer"], max(handle["consumers"]))
        for handle in handles
        if handle["consumers"]
    ]
    for start, rule in enumerate(rules):
        if rule != "cycle_open":
            continue
        region = set(footprints[start])
        for stop in range(start + 1, len(actions)):
            if not _regions_connected(region, footprints[stop], graphs[stop]):
                break
            region.update(footprints[stop])
            if rules[stop] in {"cycle_close", "ring_system_restate"}:
                intervals.append((start, stop))
                break
    protected = _merge_intervals(intervals)
    fallback_steps = {
        step
        for start, stop in protected
        if stop - start + 1 > config.maximum_primitives_per_segment
        for step in range(start, stop + 1)
    }
    protected_cuts = {
        cut
        for start, stop in protected
        if stop - start + 1 <= config.maximum_primitives_per_segment
        for cut in range(start + 1, stop + 1)
    }
    soft_connected = {
        cut: _regions_connected(footprints[cut - 1], footprints[cut], graphs[cut])
        for cut in range(1, len(actions))
    }
    bounds = []
    start = 0
    while start < len(actions):
        if start in fallback_steps:
            stop = start + 1
        else:
            limit = min(len(actions), start + config.maximum_primitives_per_segment)
            stop = None
            for cut in range(start + 1, limit + 1):
                if cut in protected_cuts:
                    continue
                if cut == len(actions) or not soft_connected.get(cut, False):
                    stop = cut
                    break
            if stop is None:
                candidates = [
                    cut
                    for cut in range(start + 1, limit + 1)
                    if cut not in protected_cuts
                ]
                if not candidates:
                    raise RuntimeError(
                        "bounded segmentation cannot honor protected cuts"
                    )
                stop = max(candidates)
        bounds.append((start, stop))
        start = stop

    action_to_segment = {}
    segments = []
    exact_segments = 0
    for segment_index, (start, stop) in enumerate(bounds):
        for step in range(start, stop):
            action_to_segment[step] = segment_index
        product, receipt = execute_program(graphs[start], list(actions[start:stop]))
        exact = receipt["states"] == list(states[start : stop + 1])
        exact_segments += exact
        if not exact:
            raise ValueError(
                f"segment {segment_index} differs from exact executor replay"
            )
        local_edges = [
            edge
            for edge in dependency_edges
            if start <= edge["producer"] < stop and start <= edge["consumer"] < stop
        ]
        segment_rules = Counter(rules[start:stop])
        segments.append(
            {
                "segment_index": segment_index,
                "start": start,
                "stop": stop,
                "label": (
                    config.fallback_label
                    if stop - start == 1
                    else config.multi_action_label
                ),
                "primitive_count": stop - start,
                "rule_counts": dict(sorted(segment_rules.items())),
                "changed_region_size": len(
                    set().union(*(footprints[step] for step in range(start, stop)))
                ),
                "created_outputs": sum(
                    handle["producer"] in range(start, stop) for handle in handles
                ),
                "within_segment_dependency_edges": local_edges,
                "contains_cycle_open": bool(segment_rules["cycle_open"]),
                "contains_cycle_close": bool(segment_rules["cycle_close"]),
                "contains_ring_restate": bool(segment_rules["ring_system_restate"]),
                "protected_horizon_fallback": start in fallback_steps,
                "exact_replay": exact,
                "endpoint_state": receipt["states"][-1],
            }
        )
        del product
    cross_segment_edges = [
        edge
        for edge in dependency_edges
        if action_to_segment[edge["producer"]] != action_to_segment[edge["consumer"]]
    ]
    runtime_length = len(actions) <= config.runtime_maximum_primitives
    complete_representation = (
        runtime_length and len(segments) <= config.runtime_maximum_segments
    )
    return {
        "segments": segments,
        "created_handles": handles,
        "dependency_edges": dependency_edges,
        "cross_segment_dependency_edges": cross_segment_edges,
        "protected_intervals": [list(row) for row in protected],
        "primitive_transitions": len(actions),
        "module_count": len(segments),
        "multi_action_transitions": sum(
            segment["primitive_count"]
            for segment in segments
            if segment["primitive_count"] > 1
        ),
        "protected_horizon_fallback_transitions": len(fallback_steps),
        "exact_replay_segments": exact_segments,
        "runtime_length_supported": runtime_length,
        "complete_representation_supported": complete_representation,
    }


def segmentation_summary(routes: list[dict]) -> dict:
    if not routes:
        raise ValueError("segmentation summary requires routes")
    segments = [
        segment for route in routes for segment in route["segmentation"]["segments"]
    ]
    actions = sum(route["segmentation"]["primitive_transitions"] for route in routes)
    multi_actions = sum(
        route["segmentation"]["multi_action_transitions"] for route in routes
    )
    exact = sum(segment["exact_replay"] for segment in segments)
    dependencies = [
        edge for route in routes for edge in route["segmentation"]["dependency_edges"]
    ]
    return {
        "routes": len(routes),
        "primitive_transitions": actions,
        "segments": len(segments),
        "multi_action_segments": sum(
            segment["primitive_count"] > 1 for segment in segments
        ),
        "primitive_fallback_segments": sum(
            segment["primitive_count"] == 1 for segment in segments
        ),
        "multi_action_transition_coverage": multi_actions / actions,
        "exact_replay_segments": exact,
        "exact_replay_precision": exact / len(segments),
        "created_handles": sum(
            len(route["segmentation"]["created_handles"]) for route in routes
        ),
        "dependency_edges": len(dependencies),
        "cross_segment_dependency_edges": sum(
            len(route["segmentation"]["cross_segment_dependency_edges"])
            for route in routes
        ),
        "creation_lag_distribution": dict(
            sorted(Counter(str(edge["creation_lag"]) for edge in dependencies).items())
        ),
        "module_count_distribution": dict(
            sorted(
                Counter(
                    str(route["segmentation"]["module_count"]) for route in routes
                ).items(),
                key=lambda row: int(row[0]),
            )
        ),
        "segment_size_distribution": dict(
            sorted(
                Counter(
                    str(segment["primitive_count"]) for segment in segments
                ).items(),
                key=lambda row: int(row[0]),
            )
        ),
        "runtime_length_routes": sum(
            route["segmentation"]["runtime_length_supported"] for route in routes
        ),
        "complete_representation_routes": sum(
            route["segmentation"]["complete_representation_supported"]
            for route in routes
        ),
        "protected_horizon_fallback_transitions": sum(
            route["segmentation"]["protected_horizon_fallback_transitions"]
            for route in routes
        ),
    }


__all__ = [
    "SCHEMA",
    "TRAINING_SCHEMA",
    "SegmentationConfig",
    "segment_exact_trace",
    "segmentation_summary",
]
