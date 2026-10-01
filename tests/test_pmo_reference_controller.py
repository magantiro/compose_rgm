"""The frozen reference changes only the declared PMO exploration slot."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from test_program_reference import (
    checkpoint as checkpoint,  # noqa: PLC0414 -- pytest fixture re-export
)
from test_program_reference import loaded as loaded  # noqa: PLC0414 -- pytest fixture re-export
from test_program_reference import programs

from compose_v4.control import pmo_reference_controller as controller_module
from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.pmo_population_controller import CHANNELS, CHECKPOINT_SCHEMA
from compose_v4.control.reference_guidance import (
    GuidanceConfig,
    ProgramScore,
    ScoredPanel,
)
from compose_v4.control.reference_programs import FrozenProgramReference, ProgramInput

CHECKPOINT = {"schema_version": CHECKPOINT_SCHEMA, "plan_latents": []}
OFF = {"guidance": {"mode": "off"}, "asset": None}


def _controller(seed=13):
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        candidates_per_batch=3,
    )
    return controller_module.PmoReferenceController(
        config,
        source_group="fixture",
        oracle_protocol="synthetic-no-oracle",
        hierarchy=None,
        jump_checkpoint=CHECKPOINT,
        reference_spec=OFF,
    )


def _candidates():
    channel = CHANNELS[0]
    return [
        {
            "candidate_id": key,
            "endpoint": smiles,
            "provenance": {"planner_channel": channel},
        }
        for key, smiles in (("a", "C"), ("b", "CC"), ("c", "CCC"))
    ]


def _base_allocation(self, candidates, *, reserve=0):
    assert reserve == 1
    first = candidates[0]
    channel = first["provenance"]["planner_channel"]
    return [first], {
        "selected_ids": [first["candidate_id"]],
        "selected_by_channel": {name: int(name == channel) for name in CHANNELS},
        "selection_role_by_candidate": {first["candidate_id"]: "online_allocation"},
    }


class FakeReference:
    def score(self, programs):
        values = {"b": -1.0, "c": -0.01}
        return ScoredPanel(
            "a" * 64,
            tuple(
                ProgramScore(row.candidate_id, values[row.candidate_id], "scored")
                for row in programs
            ),
            0.0,
        )


def test_off_and_shadow_keep_the_same_exploration_draw(monkeypatch):
    monkeypatch.setattr(controller_module.PmoPopulationController, "_allocate", _base_allocation)
    monkeypatch.setattr(
        controller_module,
        "pmo_program_input",
        lambda row: ProgramInput(row["candidate_id"], row["endpoint"], None, "fixture"),
    )
    off = _controller()
    shadow = _controller()
    shadow.guidance = GuidanceConfig(mode="shadow")
    shadow.reference = FakeReference()
    off_selected, off_detail = off._allocate(_candidates())
    shadow_selected, shadow_detail = shadow._allocate(_candidates())
    assert [row["candidate_id"] for row in off_selected] == [
        row["candidate_id"] for row in shadow_selected
    ]
    assert off_detail["reference_slot"]["guidance"]["outcome"] == "off"
    assert shadow_detail["reference_slot"]["guidance"]["outcome"] == "shadow"
    assert not shadow_detail["reference_slot"]["guidance"]["probabilities_changed"]


def test_active_reweights_only_remaining_candidates(monkeypatch):
    monkeypatch.setattr(controller_module.PmoPopulationController, "_allocate", _base_allocation)
    monkeypatch.setattr(
        controller_module,
        "pmo_program_input",
        lambda row: ProgramInput(row["candidate_id"], row["endpoint"], None, "fixture"),
    )
    active = _controller()
    active.guidance = GuidanceConfig(mode="active", strength=1.0)
    active.reference = FakeReference()
    selected, detail = active._allocate(_candidates())
    assert selected[0]["candidate_id"] == "a"
    assert selected[1]["candidate_id"] in {"b", "c"}
    receipt = detail["reference_slot"]["guidance"]
    assert receipt["outcome"] == "active"
    assert receipt["probabilities_changed"]
    assert receipt["candidate_ids"] == ("b", "c")
    assert detail["selected_ids"] == [row["candidate_id"] for row in selected]


def test_active_slot_uses_the_frozen_checkpoint_on_exact_programs(monkeypatch, loaded):
    monkeypatch.setattr(controller_module.PmoPopulationController, "_allocate", _base_allocation)
    channel = CHANNELS[0]
    candidates = [_candidates()[0]]
    for program in programs():
        trace = json.loads(program.trace_json)
        candidates.append(
            {
                "candidate_id": program.candidate_id,
                "endpoint": program.endpoint,
                "source_state": trace["states"][0],
                "trace": trace,
                "provenance": {"planner_channel": channel},
            }
        )
    controller = _controller()
    controller.guidance = GuidanceConfig(mode="active", strength=1.0)
    controller.reference = FrozenProgramReference(loaded)
    _, detail = controller._allocate(candidates)
    receipt = detail["reference_slot"]["guidance"]
    assert receipt["outcome"] == "active"
    assert receipt["probabilities_changed"]
    assert receipt["reference"]["score_semantics"].startswith("mean_productive_canonical_successor")


def test_resume_binds_the_reference_specification():
    controller = _controller()
    snapshot = controller.snapshot()
    restored = controller_module.PmoReferenceController.restore(
        snapshot, hierarchy=None, jump_checkpoint=CHECKPOINT, reference_spec=OFF
    )
    assert restored.reference_spec == OFF
    with pytest.raises(ValueError, match="changed on resume"):
        controller_module.PmoReferenceController.restore(
            snapshot,
            hierarchy=None,
            jump_checkpoint=CHECKPOINT,
            reference_spec={"guidance": {"mode": "off"}, "asset": {"path": "wrong"}},
        )


def test_single_query_batch_still_selects_one_candidate():
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=17, score_direction="maximize"),
        candidates_per_batch=1,
    )
    controller = controller_module.PmoReferenceController(
        config,
        source_group="fixture",
        oracle_protocol="synthetic-no-oracle",
        hierarchy=None,
        jump_checkpoint=CHECKPOINT,
        reference_spec=OFF,
    )
    selected, detail = controller._allocate(_candidates()[:1])
    assert [row["candidate_id"] for row in selected] == ["a"]
    assert detail["selected_ids"] == ["a"]
