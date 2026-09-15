from __future__ import annotations

import json

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.compositional_structural_subgoal_generator import (
    PatchTrainingEvent,
    balanced_event_weights,
    enumerate_local_legal_successors,
    fit_compositional_patch_generator,
    generate_compositional_patches,
    region_features,
)
from compose_v4.control.generic_legal_action_policy import (
    _record_parts,
    enumerate_legal_successors,
    enumerate_rule_successors,
)
from compose_v4.control.structural_subgoal_components import GRANULAR_COMPONENT_FAMILIES
from tools.t4_compositional_structural_subgoal_generator import _overall_summary


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


def test_local_fiber_is_decision_equivalent_to_filtering_complete_fiber() -> None:
    source = _ethane()
    allowed = (0,)
    expected = {
        candidate.successor_key
        for candidate in enumerate_legal_successors(source)
        if (
            bool(_record_parts(candidate.action_record, source)[1])
            and _record_parts(candidate.action_record, source)[1] <= set(allowed)
        )
    }

    observed = {
        candidate.successor_key for candidate in enumerate_local_legal_successors(source, allowed)
    }

    assert observed == expected


def test_overall_summary_applies_the_frozen_improvement_gate() -> None:
    def coverage(policy: str, cutoff: int) -> dict:
        recovered = int(policy == "balanced_joint_autoregressive" and cutoff == 128)
        families = {
            family: {
                "held_instances": 1,
                "held_recovered": recovered,
                "held_coverage": float(recovered),
            }
            for family in ("whole_patch", *GRANULAR_COMPONENT_FAMILIES)
        }
        return {
            "families": families,
            "unique_patch_yield": cutoff,
            "novel_whole_patch_yield": cutoff,
            "unique_endpoint_yield": cutoff,
            "fixed_k_yield": 1.0,
        }

    policies = {
        policy: {
            "cutoffs": {str(cutoff): coverage(policy, cutoff) for cutoff in (8, 32, 128)},
            "telemetry": {
                "compile_attempts": 128,
                "exactly_realized": 128,
                "candidate_shortfall": 0,
                "legal_patch_compile_coverage_numerator": 128,
                "legal_patch_compile_coverage_denominator": 128,
                "exact_realization_precision_numerator": 128,
                "exact_realization_precision_denominator": 128,
            },
        }
        for policy in ("uniform_joint_grammar", "balanced_joint_autoregressive")
    }
    quality_policy = {
        "per_source": [
            {
                "cutoffs": {
                    str(cutoff): {
                        "exact_recall": 0.0,
                        "transformation_recall": 0.0,
                    }
                    for cutoff in (8, 32, 128)
                }
            }
        ]
    }
    summary = _overall_summary(
        [
            {
                "sources": [{"policies": policies}],
                "endpoint_and_transformation_quality": {
                    "policies": {
                        "uniform_joint_grammar": quality_policy,
                        "balanced_joint_autoregressive": quality_policy,
                    }
                },
            }
        ]
    )

    assert summary["gates"]["passed"] is True
    assert summary["gates"]["learned_improving_cutoffs"] == [128]
    assert (
        summary["policies"]["balanced_joint_autoregressive"]["cutoffs"]["128"]["instance_weighted"][
            "granular_component_coverage"
        ]
        == 1.0
    )
