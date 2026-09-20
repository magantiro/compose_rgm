"""Focused checks for the optional joint-STOP production adapter."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest

from compose_v4.control.nodistill_joint_stop_adapter_v1 import (
    JointStopAdapterSettings,
    propose_joint_stop_candidates,
)
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)


def _small_settings() -> JointStopAdapterSettings:
    return JointStopAdapterSettings(
        max_depth=2,
        beam_width=4,
        lock_width=6,
        candidates_per_family=2,
        expansions_per_parent=6,
    )


def _summary(batch) -> list[tuple]:
    return [
        (
            row.endpoint_key,
            row.metadata["candidate_id"],
            row.metadata["joint_stop_rank"],
            row.metadata["primitive_count"],
            row.metadata["rank_features"]["strata"],
            row.metadata["macro_plan"],
        )
        for row in batch.candidates
    ]


def test_adapter_boundary_is_source_seed_settings_only_and_rejects_hidden_fields():
    assert tuple(inspect.signature(propose_joint_stop_candidates).parameters) == (
        "source",
        "seed",
        "settings",
    )
    source = production_state_from_smiles("CCO", max_atoms=48)
    with pytest.raises(ValueError, match="forbidden fields"):
        propose_joint_stop_candidates(
            source,
            seed=7,
            settings={**_small_settings().payload(), "target": "hidden"},
        )
    with pytest.raises(ValueError, match="seed"):
        propose_joint_stop_candidates(source, seed=True, settings=_small_settings())


def test_adapter_returns_ranked_exact_complete_pre_admission_candidates():
    source = production_state_from_smiles("C1CCCCC1", max_atoms=48)
    settings = _small_settings()

    first = propose_joint_stop_candidates(source, seed=23, settings=settings)
    second = propose_joint_stop_candidates(source, seed=23, settings=settings.payload())

    assert first.candidates
    assert _summary(first) == _summary(second)
    assert first.proposals is first.candidates
    assert [row.metadata["joint_stop_rank"] for row in first.candidates] == list(
        range(1, len(first.candidates) + 1)
    )
    assert len({row.endpoint_key for row in first.candidates}) == len(first.candidates)
    assert all(row.actions for row in first.candidates)
    assert all(row.program and row.program_graph for row in first.candidates)
    assert all(len(row.families) == 2 for row in first.candidates)
    assert all(row.actual_changes for row in first.candidates)
    assert all(
        set(row.metadata["macro_plan"]["axes"])
        == {
            "scale",
            "delta_heavy",
            "delta_cycle",
            "retained_interfaces",
            "rewrite_mode",
        }
        for row in first.candidates
    )
    assert all(
        row.metadata["headroom_features"]["primitive_headroom"] == 32 - len(row.actions)
        for row in first.candidates
    )
    assert all(
        row.metadata["exact_execution_verified"] is True
        and row.metadata["endpoint_admission_applied_during_generation"] is False
        and row.metadata["primitive_count"] <= 32
        and row.metadata["block_count"] <= 8
        for row in first.candidates
    )
    runtime = first.telemetry["runtime_inputs"]
    assert runtime == {
        "source_graph": True,
        "seed": True,
        "settings": True,
        "task_or_cell_name": False,
        "target_or_endpoint": False,
        "requested_delta_or_fiber": False,
        "objective_or_docking_score": False,
        "route_or_template_identity": False,
        "teacher_action_or_endpoint": False,
        "absolute_route_address": False,
    }
    assert first.telemetry["endpoint_constraints_applied_during_generation"] is False
    assert first.telemetry["exact_replay_verified"] == len(first.candidates)
    assert first.telemetry["multi_axis_macro_plan_count"] >= 2


def test_adapter_settings_are_immutable_and_support_expansion_fails_closed():
    settings = JointStopAdapterSettings()
    with pytest.raises(FrozenInstanceError):
        settings.max_blocks = 7
    for field, value in (
        ("max_primitives", 33),
        ("max_blocks", 9),
        ("max_active_atoms", 41),
    ):
        with pytest.raises(ValueError, match="may not change"):
            JointStopAdapterSettings(**{field: value})
