from __future__ import annotations

import numpy as np

from compose_v4.experiments.pmo_dependency_region_policy_comparison import (
    POLICIES,
    PolicyConfig,
    event_outcomes,
    fit_fold_checkpoint,
    route_events,
    runtime_checkpoint_leakage,
    score_candidate,
    teacher_forced_metrics,
)


def _state(atom_types, bonds=()):
    size = 48
    implicit = [0] * size
    for slot, atom_type in enumerate(atom_types):
        if atom_type == 2:
            implicit[slot] = 4 - sum(
                order for left, right, order in bonds if slot in {left, right}
            )
    return {
        "n_slots": size,
        "atom_types": [*atom_types, *([0] * (size - len(atom_types)))],
        "formal_charges": [0] * size,
        "implicit_h_counts": implicit,
        "bonds": [list(row) for row in bonds],
    }


def _route(trace, fold, rule="atom_insert"):
    source = _state([2], ())
    if rule == "atom_insert":
        endpoint = _state([2, 2], ((0, 1, 1),))
        action = {
            "executor_rule": "atom_insert",
            "model_family": "atom_insert",
            "payload_type": "AtomInsert",
            "schema": "compose.rewrite.action",
            "schema_version": 4,
            "payload": {
                "slot": 1,
                "atom_type": 2,
                "formal_charge": 0,
                "implicit_h_count": 3,
                "neighbors": [[0, 1]],
            },
        }
    else:
        source = _state([2, 2], ((0, 1, 1),))
        endpoint = _state([2], ())
        action = {
            "executor_rule": "atom_delete",
            "model_family": "atom_delete",
            "payload_type": "AtomDelete",
            "schema": "compose.rewrite.action",
            "schema_version": 4,
            "payload": {"v": 1},
        }
    return {
        "trace_identity": trace,
        "lineage_identity": f"lineage-{trace}",
        "task_family": f"family-{fold}",
        "test_fold": fold,
        "source_state": source,
        "states": [source, endpoint],
        "actions": [action],
    }


def test_route_events_use_generic_targets_and_normalized_features():
    event = route_events(_route("insert", 0), weight=1.0)[0]
    assert event.targets["primitive_rule"] == "atom_insert"
    assert event.targets["region_origin_count"] == "source_one"
    assert event.targets["creates_output"] == "creates_relative_output"
    assert event.targets["component_control"] == "open"
    assert event.targets["component_close"] == "close"
    assert np.isfinite(event.hierarchical_features).all()


def test_fold_fit_and_metrics_cover_all_policies():
    routes = [_route("a", 0), _route("b", 1, "atom_delete"), _route("c", 2)]
    events = tuple(
        event
        for route in routes
        for event in route_events(route, weight=1 / len(routes))
    )
    checkpoint = fit_fold_checkpoint(events, 0, PolicyConfig())
    for policy in POLICIES:
        outcomes = event_outcomes(checkpoint, events, policy)
        metrics = teacher_forced_metrics(outcomes)
        assert metrics["common_factorization"]["coverage"] == 1.0
        assert np.isfinite(score_candidate(checkpoint, routes[0], policy))


def test_runtime_checkpoint_has_no_teacher_payload():
    routes = [_route("a", 0), _route("b", 1), _route("c", 2)]
    events = tuple(
        event
        for route in routes
        for event in route_events(route, weight=1 / len(routes))
    )
    checkpoint = fit_fold_checkpoint(events, 0, PolicyConfig())
    audit = runtime_checkpoint_leakage(checkpoint, {"a", "lineage-a", "family-0"})
    assert audit == {
        "forbidden_key_hits": [],
        "forbidden_teacher_value_hits": [],
    }
