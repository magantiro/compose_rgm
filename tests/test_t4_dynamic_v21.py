from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.dynamic_program_synthesis_v2 import (
    SHALLOW_CHANNEL,
    STRUCTURED_CHANNEL,
)
from compose_v4.control.dynamic_program_synthesis_v21 import (
    DynamicV21ProgramOptimizer,
    arbitrate_candidates,
    initial_allocator_state,
    initial_dynamic_program_batch_v21,
    validate_allocator_state,
)
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)
from compose_v4.experiments.t4_dynamic_v21 import SELECTED_UNITS, load_contract

ROOT = Path(__file__).resolve().parents[1]


def _config(seed=19):
    return replace(
        ProgramSearchConfig.program_only_recipe(seed=seed),
        attempts_per_batch=24,
        candidates_per_batch=8,
        wall_seconds=20,
    )


def _always_eligible(row):
    return {**row, "oracle_eligible": True}


def _candidate(channel, index):
    return {
        "candidate_id": f"{channel}-{index}",
        "provenance": {"planner_channel": channel},
    }


def test_allocator_falls_back_without_spending_slots_on_empty_channel():
    state = initial_allocator_state(score_direction="minimize")
    candidates = [_candidate(SHALLOW_CHANNEL, index) for index in range(5)]

    selected, receipt = arbitrate_candidates(
        candidates,
        limit=3,
        state=state,
        rng=np.random.default_rng(7),
    )

    assert len(selected) == 3
    assert receipt["available_by_channel"][STRUCTURED_CHANNEL] == 0
    assert receipt["selected_by_channel"] == {
        SHALLOW_CHANNEL: 3,
        STRUCTURED_CHANNEL: 0,
    }


def test_allocator_has_a_floor_only_after_both_channels_are_eligible():
    state = initial_allocator_state(score_direction="minimize")
    candidates = [
        *[_candidate(SHALLOW_CHANNEL, index) for index in range(4)],
        *[_candidate(STRUCTURED_CHANNEL, index) for index in range(4)],
    ]

    selected, receipt = arbitrate_candidates(
        candidates,
        limit=4,
        state=state,
        rng=np.random.default_rng(11),
    )

    assert len(selected) == 4
    assert all(value >= 1 for value in receipt["selected_by_channel"].values())


def test_v21_cold_start_is_route_empty_and_deterministic():
    source = production_state_from_smiles("C1CCCCC1CCO", max_atoms=48)
    kwargs = {
        "source_group": "test-source",
        "oracle_protocol": "test-oracle",
        "eligibility": _always_eligible,
    }

    first = initial_dynamic_program_batch_v21(source, (), _config(), **kwargs)
    second = initial_dynamic_program_batch_v21(source, (), _config(), **kwargs)

    assert first["batch_id"] == second["batch_id"]
    assert first["initial_route_archive"] == []
    assert first["source_library_rows_loaded"] == 0
    assert first["new_oracle_calls"] == 0
    assert first["candidates"]
    validate_allocator_state(first["allocator_state_after_preparation"])


def test_advancing_structured_rng_does_not_change_shallow_rng():
    first = DynamicV21ProgramOptimizer(
        _config(seed=23),
        source_group="test-source",
        oracle_protocol="test-oracle",
        hierarchy=None,
    )
    second = DynamicV21ProgramOptimizer(
        _config(seed=23),
        source_group="test-source",
        oracle_protocol="test-oracle",
        hierarchy=None,
    )
    expected = json.loads(json.dumps(first.shallow_rng.bit_generator.state))

    second.structured_rng.random(100)

    assert first.shallow_rng.bit_generator.state == expected
    assert second.shallow_rng.bit_generator.state == expected


def test_v21_snapshot_restores_all_independent_random_streams():
    optimizer = DynamicV21ProgramOptimizer(
        _config(seed=29),
        source_group="test-source",
        oracle_protocol="test-oracle",
        hierarchy=None,
    )
    optimizer.rng.random(2)
    optimizer.shallow_rng.random(3)
    optimizer.structured_rng.random(5)
    optimizer.allocator_state["channels"][SHALLOW_CHANNEL]["proposals"] = 4

    restored = DynamicV21ProgramOptimizer.restore(optimizer.snapshot(), hierarchy=None)

    assert restored.allocator_state == optimizer.allocator_state
    assert restored.rng.bit_generator.state == optimizer.rng.bit_generator.state
    assert (
        restored.shallow_rng.bit_generator.state
        == optimizer.shallow_rng.bit_generator.state
    )
    assert (
        restored.structured_rng.bit_generator.state
        == optimizer.structured_rng.bit_generator.state
    )


def test_bootstrap_state_and_channel_outcomes_enter_common_archive():
    source = production_state_from_smiles("CCOC(=O)NCC", max_atoms=48)
    config = _config(seed=31)
    batch = initial_dynamic_program_batch_v21(
        source,
        (),
        config,
        source_group="test-source",
        oracle_protocol="test-oracle",
        eligibility=_always_eligible,
    )
    optimizer = DynamicV21ProgramOptimizer(
        config,
        source_group="test-source",
        oracle_protocol="test-oracle",
        hierarchy=None,
    )
    candidate = batch["candidates"][0]

    optimizer.add_measured_program(candidate, receipt_id="receipt", score=-7.0)

    channel = candidate["provenance"]["planner_channel"]
    assert optimizer.allocator_state["channels"][channel]["charged_outcomes"] == 1
    assert optimizer.allocator_state["global_best"] == -7.0
    assert len(optimizer.entries) == 1

    proposed = optimizer.propose_batch(_always_eligible)
    selected_id = proposed["candidates"][0]["candidate_id"]
    locked = optimizer.lock_query_subset(
        proposed["batch_id"],
        [selected_id],
        {
            "policy": "test_final_budget_prefix",
            "selected_ids": [selected_id],
        },
    )
    assert len(locked["candidates"]) == 1
    assert locked["eligible_pool"]["candidates"]


def test_v21_contract_has_five_cells_and_no_runtime_comparison_artifact():
    contract = load_contract(ROOT)

    assert contract["library_programs"] == 0
    assert contract["dynamic_v21_development"]["new_call_ceiling"] == 5000
    assert tuple(contract["dynamic_v21_development"]["development_units"]) == (
        SELECTED_UNITS
    )
    assert not any(
        "comparison" in path or "result" in path for path in contract["inputs"]
    )
