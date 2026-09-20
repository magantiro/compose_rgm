import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from compose_v4.control.fiber_control import ProgramValue, SearchState
from compose_v4.control.generic_feasibility_headroom_composer_v4 import (
    propose_feasibility_headroom_programs,
)
from compose_v4.control.nodistill_joint_stop_adapter_v1 import (
    propose_joint_stop_candidates,
)
from compose_v4.experiments.t4_integrated_route_fiber import (
    EXPERTS,
    FEASIBILITY_HEADROOM_V4_EXPERT,
    integrated_features,
    merge_expert_pools,
    select_batch,
    validate_expert_vocabulary,
)
from compose_v4.experiments.t4_shared_controller_scored_runtime import (
    feasibility_headroom_v4_records,
)

V4_EXPERTS = (*EXPERTS, FEASIBILITY_HEADROOM_V4_EXPERT)


def _row(smiles: str, expert: str, *, position: int = 0) -> dict:
    return {
        "smiles": smiles,
        "proposal_lane": expert,
        "proposal_experts": [expert],
        "parent": "CCOC(=O)NCC",
        "parent_score": -8.0,
        "similarity": 0.7,
        "delta": 0.6,
        "qed": 0.7,
        "sa": 3.0,
        "regions": 1,
        "created": 1,
        "deleted": 0,
        "families": [expert],
        "program_families": [expert],
        "fingerprint": {10 + position},
    }


def test_v4_expert_is_append_only_and_legacy_vocabulary_is_unchanged():
    assert validate_expert_vocabulary(EXPERTS) == EXPERTS
    assert validate_expert_vocabulary(V4_EXPERTS) == V4_EXPERTS
    with pytest.raises(ValueError, match="unsupported proposal expert vocabulary"):
        validate_expert_vocabulary((FEASIBILITY_HEADROOM_V4_EXPERT, *EXPERTS))


def test_v4_generation_boundary_has_no_identity_or_teacher_input(monkeypatch):
    forbidden = {
        "cell",
        "cell_key",
        "target",
        "route",
        "template",
        "teacher",
        "teacher_endpoint",
        "objective",
        "docking_score",
        "parent_score",
    }
    assert not set(
        inspect.signature(propose_feasibility_headroom_programs).parameters
    ).intersection(forbidden)
    assert not set(
        inspect.signature(propose_joint_stop_candidates).parameters
    ).intersection(forbidden)
    assert not set(
        inspect.signature(feasibility_headroom_v4_records).parameters
    ).intersection(forbidden)

    monkeypatch.setattr(
        "compose_v4.experiments.t4_shared_controller_scored_runtime."
        "propose_feasibility_headroom_programs",
        lambda source, **kwargs: SimpleNamespace(proposals=(), telemetry={}),
    )
    monkeypatch.setattr(
        "compose_v4.experiments.t4_shared_controller_scored_runtime."
        "propose_joint_stop_candidates",
        lambda source, **kwargs: SimpleNamespace(candidates=(), telemetry={}),
    )
    records, telemetry = feasibility_headroom_v4_records(
        parent="CCOC(=O)NCC",
        proposal_seed_value=7,
        similarity_minimum=0.6,
        settings={
            "attempts_per_local_plan": 1,
            "candidate_quota_per_local_plan": 1,
            "maximum_primitives": 32,
            "maximum_blocks": 8,
            "maximum_heavy_atoms": 40,
            "qed_minimum": 0.6,
            "sa_maximum": 4.0,
            "joint_stop": {
                "min_depth": 2,
                "max_depth": 2,
                "beam_width": 4,
                "lock_width": 6,
                "candidates_per_family": 2,
                "expansions_per_parent": 6,
                "max_primitives": 32,
                "max_blocks": 8,
                "max_active_atoms": 40,
            },
        },
    )
    assert records == []
    assert telemetry["endpoint_constraints_applied_during_generation"] is False
    assert telemetry["route_templates_loaded"] == 0
    assert telemetry["runtime_task_cell_or_target_input"] is False
    assert telemetry["runtime_teacher_action_endpoint_or_proximity_input"] is False
    assert telemetry["runtime_objective_or_docking_score_input"] is False


def test_v4_support_budget_drift_fails_before_generation(monkeypatch):
    monkeypatch.setattr(
        "compose_v4.experiments.t4_shared_controller_scored_runtime."
        "propose_feasibility_headroom_programs",
        lambda source, **kwargs: pytest.fail("generation must not run"),
    )
    with pytest.raises(ValueError, match="40 atoms/32 primitives/8 blocks"):
        feasibility_headroom_v4_records(
            parent="CCOC(=O)NCC",
            proposal_seed_value=7,
            similarity_minimum=0.6,
            settings={
                "attempts_per_local_plan": 1,
                "candidate_quota_per_local_plan": 1,
                "maximum_primitives": 33,
            },
        )


def test_v4_macro_plan_provenance_survives_endpoint_union():
    first = _row("C", FEASIBILITY_HEADROOM_V4_EXPERT)
    second = _row("C", FEASIBILITY_HEADROOM_V4_EXPERT, position=1)
    first.update({"macro_plan_identity": "plan-a", "macro_mode": "grow"})
    second.update({"macro_plan_identity": "plan-b", "macro_mode": "replace"})
    result = merge_expert_pools(
        {
            **{expert: [] for expert in EXPERTS},
            FEASIBILITY_HEADROOM_V4_EXPERT: [first, second],
        },
        experts=V4_EXPERTS,
    )
    assert len(result) == 1
    origins = result[0]["proposal_origins"]
    assert {row["macro_plan_identity"] for row in origins} == {"plan-a", "plan-b"}
    assert {row["macro_mode"] for row in origins} == {"grow", "replace"}


def test_batch_eight_can_seed_all_four_experts_with_fresh_feature_width():
    state = SearchState(archive={"P": -8.0})
    candidates = []
    for expert_index, expert in enumerate(V4_EXPERTS):
        for position in range(2):
            row = _row(
                f"{expert_index}-{position}",
                expert,
                position=expert_index * 2 + position,
            )
            row["features"] = integrated_features(row, state, experts=V4_EXPERTS)
            candidates.append(row)
    selected = select_batch(
        candidates,
        ProgramValue(),
        state,
        np.random.default_rng(17),
        round_index=1,
        batch=8,
        exploration=2,
        expert_floor_rounds=2,
        route_scale_floor_rounds=0,
        experts=V4_EXPERTS,
    )
    floors = [row for row in selected if row["selection_kind"] == "expert_floor"]
    assert {row["proposal_lane"] for row in floors} == set(V4_EXPERTS)
    assert len({len(row["features"]) for row in candidates}) == 1
    assert len(candidates[0]["features"]) == 20
    assert len(selected) == 8
