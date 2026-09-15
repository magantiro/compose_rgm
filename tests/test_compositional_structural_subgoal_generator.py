from __future__ import annotations

import json

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.compositional_structural_subgoal_generator import (
    PatchTrainingEvent,
    balanced_event_weights,
    fit_compositional_patch_generator,
    generate_compositional_patches,
    region_features,
)
from compose_v4.control.generic_legal_action_policy import enumerate_rule_successors


def _ethane() -> MolecularGraph:
    atom_types = np.zeros(48, dtype=np.int32)
    charges = np.zeros(48, dtype=np.int32)
    hydrogens = np.zeros(48, dtype=np.int32)
    bonds = np.zeros((48, 48), dtype=np.int32)
    atom_types[:2] = 2
    hydrogens[:2] = 3
    bonds[0, 1] = bonds[1, 0] = 1
    return MolecularGraph(atom_types, charges, hydrogens, bonds)


def _event(
    source: MolecularGraph,
    *,
    domain: str,
    lineage: str,
    route: str,
    component: str,
    candidate_index: int = 0,
) -> PatchTrainingEvent:
    candidate = enumerate_rule_successors(source, "atom_insert")[candidate_index]
    return PatchTrainingEvent(
        domain=domain,
        lineage=lineage,
        route_id=route,
        component_id=component,
        component_events=1,
        depth=0,
        component_source=source,
        current=source,
        candidate=candidate,
        region_slots=(int(candidate.action_record["payload"]["neighbors"][0][0]),),
        mutable_slots=(int(candidate.action_record["payload"]["neighbors"][0][0]),),
        previous_rule=None,
    )


def test_balancing_equalizes_domains_and_nested_lineages() -> None:
    source = _ethane()
    events = [
        _event(source, domain="t4", lineage="a", route="r1", component="c1"),
        _event(
            source,
            domain="t4",
            lineage="a",
            route="r1",
            component="c1",
            candidate_index=1,
        ),
        _event(source, domain="t4", lineage="b", route="r2", component="c2"),
        _event(source, domain="pmo", lineage="p", route="r3", component="c3"),
    ]

    weights = balanced_event_weights(events)

    assert np.isclose(sum(weights), 1)
    assert np.isclose(sum(w for w, row in zip(weights, events) if row.domain == "t4"), 0.5)
    assert np.isclose(sum(w for w, row in zip(weights, events) if row.domain == "pmo"), 0.5)
    assert np.isclose(weights[0], weights[1])


def test_region_features_ignore_role_order() -> None:
    source = _ethane()

    assert np.array_equal(region_features(source, (0, 1)), region_features(source, (1, 0)))


def test_checkpoint_contains_numeric_statistics_not_teacher_objects() -> None:
    source = _ethane()
    events = [
        _event(source, domain="t4", lineage="a", route="r1", component="c1"),
        _event(
            source,
            domain="pmo",
            lineage="p",
            route="r2",
            component="c2",
            candidate_index=1,
        ),
    ]

    model = fit_compositional_patch_generator(events, variance_floor=0.05)
    serialized = json.dumps(model.checkpoint(), sort_keys=True)

    assert model.training_summary["events"] == 2
    for forbidden in (
        "route_id",
        "component_id",
        "source_graph",
        "source_state",
        "endpoint_state",
        "template_id",
        "patch_id",
        "actions",
        "smiles",
    ):
        assert forbidden not in serialized.lower()


def test_joint_generator_emits_only_exactly_realized_novel_patches() -> None:
    source = _ethane()
    events = [
        _event(source, domain="t4", lineage="a", route="r1", component="c1"),
        _event(
            source,
            domain="pmo",
            lineage="p",
            route="r2",
            component="c2",
            candidate_index=1,
        ),
    ]
    model = fit_compositional_patch_generator(events, variance_floor=0.05)

    rows, telemetry = generate_compositional_patches(
        source,
        model,
        pool_size=8,
        first_event_beam=8,
        second_event_beam=1,
        second_event_expansion_per_prefix=2,
        per_rule_first_event_cap=2,
    )

    assert rows
    assert len({row.endpoint_key for row in rows}) == len(rows)
    assert all(row.realization["endpoint_matches_bound_target"] for row in rows)
    assert all(row.realization["primitive_teacher_actions_used"] == 0 for row in rows)
    assert telemetry["exactly_realized"] == len(rows)
    assert all(row.goal.subgoals for row in rows)
