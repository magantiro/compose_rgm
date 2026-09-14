from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v2 import (
    CHANNELS,
    SHALLOW_CHANNEL,
    STRUCTURED_CEILING,
    STRUCTURED_CHANNEL,
    STRUCTURED_FLOOR,
    DynamicV2ProgramOptimizer,
    initial_dynamic_program_batch_v2,
    initial_gate_state,
    structured_probability,
    synthesize_structured_program,
    validate_gate_state,
)
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)
from compose_v4.experiments.t4_dynamic_v2 import (
    REFERENCE,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def _config(seed=19):
    return replace(
        ProgramSearchConfig.program_only_recipe(seed=seed),
        attempts_per_batch=24,
        candidates_per_batch=8,
        wall_seconds=20,
    )


def test_explicit_structured_channel_never_falls_back_to_v0():
    source = production_state_from_smiles("CCOC(=O)NCC", max_atoms=48)
    result = synthesize_structured_program(source, np.random.default_rng(7))

    assert result[4].get("v1_selection") != "preserved_dynamic_v0_channel"


def test_generic_gate_is_bounded_and_increases_after_shallow_stagnation():
    source = production_state_from_smiles("C1CCCCC1CCO", max_atoms=48)
    state = initial_gate_state(score_direction="minimize")
    parent_id = "parent"
    baseline, baseline_features = structured_probability(
        source, state, parent_id=parent_id
    )
    state["parents"][parent_id] = {
        channel: {
            "proposals": 10 if channel == SHALLOW_CHANNEL else 0,
            "exact_executions": 10 if channel == SHALLOW_CHANNEL else 0,
            "eligible": 0,
            "duplicates": 0,
            "ineligible": 10 if channel == SHALLOW_CHANNEL else 0,
            "execution_rejections": 0,
            "charged_outcomes": 4 if channel == SHALLOW_CHANNEL else 0,
            "scored_outcomes": 4 if channel == SHALLOW_CHANNEL else 0,
            "parent_improvements": 0,
            "global_best_improvements": 0,
        }
        for channel in CHANNELS
    }
    state["charged_since_global_best"] = 80
    stagnated, features = structured_probability(source, state, parent_id=parent_id)

    assert STRUCTURED_FLOOR <= baseline < stagnated <= STRUCTURED_CEILING
    assert baseline_features["editable_ring_system_present"] is True
    assert features["adjustments"]["parent_shallow_no_improvement"] == 0.05
    assert features["adjustments"]["global_stagnation"] == 0.05


def test_v2_cold_start_is_route_empty_deterministic_and_uses_both_channels():
    source = production_state_from_smiles("C1CCCCC1CCO", max_atoms=48)
    kwargs = {
        "source_group": "test-source",
        "oracle_protocol": "test-oracle",
        "eligibility": lambda row: {**row, "oracle_eligible": True},
    }
    first = initial_dynamic_program_batch_v2(source, (), _config(), **kwargs)
    second = initial_dynamic_program_batch_v2(source, (), _config(), **kwargs)
    selected = {
        row["planner_channel"] for row in first["attempts"] if "planner_channel" in row
    }

    assert first["batch_id"] == second["batch_id"]
    assert first["initial_route_archive"] == []
    assert first["source_library_rows_loaded"] == 0
    assert first["new_oracle_calls"] == 0
    assert selected == set(CHANNELS)
    assert all(
        row["metadata"]["intermediate_task_evaluations"] == 0
        for row in first["attempts"]
        if row["status"] != "execution_rejected"
    )


def test_v2_snapshot_restore_preserves_gate_state_exactly():
    optimizer = DynamicV2ProgramOptimizer(
        _config(),
        source_group="test-source",
        oracle_protocol="test-oracle",
        hierarchy=None,
    )
    optimizer.dynamic_v2_state["charged_since_global_best"] = 37
    optimizer.dynamic_v2_state["channels"][STRUCTURED_CHANNEL]["proposals"] = 5
    snapshot = optimizer.snapshot()
    restored = DynamicV2ProgramOptimizer.restore(snapshot, hierarchy=None)

    assert isinstance(restored, DynamicV2ProgramOptimizer)
    assert restored.dynamic_v2_state == optimizer.dynamic_v2_state
    assert restored.rng.bit_generator.state == optimizer.rng.bit_generator.state
    validate_gate_state(restored.dynamic_v2_state)


def test_v2_bootstrap_scores_seed_channel_credit_without_parent_reward():
    source = production_state_from_smiles("CCOC(=O)NCC", max_atoms=48)
    config = _config(seed=23)
    initial = initial_dynamic_program_batch_v2(
        source,
        (),
        config,
        source_group="test-source",
        oracle_protocol="test-oracle",
        eligibility=lambda row: {**row, "oracle_eligible": True},
    )
    candidate = initial["candidates"][0]
    channel = candidate["provenance"]["planner_channel"]
    optimizer = DynamicV2ProgramOptimizer(
        config,
        source_group="test-source",
        oracle_protocol="test-oracle",
        hierarchy=None,
    )

    optimizer.add_measured_program(
        candidate,
        receipt_id="bootstrap-receipt",
        score=-7.0,
    )

    counts = optimizer.dynamic_v2_state["channels"][channel]
    assert counts["proposals"] == 1
    assert counts["eligible"] == 1
    assert counts["charged_outcomes"] == 1
    assert counts["scored_outcomes"] == 1
    assert counts["parent_improvements"] == 0
    assert optimizer.dynamic_v2_state["global_best"] == -7.0


def test_v2_contract_keeps_comparison_outcomes_out_of_runtime_inputs():
    contract = load_contract(ROOT)

    assert contract["library_programs"] == 0
    assert contract["dynamic_v2_offline_comparison_reference"] == {
        "path": REFERENCE,
        "sha256": sha256_file(ROOT / REFERENCE),
        "available_to_runtime": False,
    }
    assert REFERENCE not in contract["inputs"]


def test_v2_proposed_candidate_identity_includes_planner_provenance():
    source = production_state_from_smiles("CCOC(=O)NCC", max_atoms=48)
    config = _config(seed=31)
    eligibility = lambda row: {**row, "oracle_eligible": True}
    initial = initial_dynamic_program_batch_v2(
        source,
        (),
        config,
        source_group="test-source",
        oracle_protocol="test-oracle",
        eligibility=eligibility,
    )
    optimizer = DynamicV2ProgramOptimizer(
        config,
        source_group="test-source",
        oracle_protocol="test-oracle",
        hierarchy=None,
    )
    optimizer.add_measured_program(
        initial["candidates"][0], receipt_id="initial-receipt", score=0.0
    )
    batch = optimizer.propose_batch(eligibility)

    assert batch["candidates"]
    for candidate in batch["candidates"]:
        candidate_id = candidate.pop("candidate_id")
        assert candidate_id == identity(candidate)
        assert candidate["provenance"]["planner_channel"] in CHANNELS
