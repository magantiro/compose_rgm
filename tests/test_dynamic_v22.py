from __future__ import annotations

from dataclasses import replace

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v2 import (
    SHALLOW_CHANNEL,
    STRUCTURED_CHANNEL,
)
from compose_v4.control.dynamic_program_synthesis_v22 import (
    DynamicV22PolicyConfig,
    DynamicV22ProgramOptimizer,
    advance_t4_feasibility_frontier,
    allocate_parent_edits,
    initial_dynamic_program_batch_v22,
    initial_frontier_state,
    validate_frontier_state,
)
from compose_v4.control.parent_edit_model import MUTATION_RECIPE, ParentEditModel
from compose_v4.experiments.dynamic_v22_pilot import (
    PMO_TASKS,
    QUERY_BUDGET,
    T4_CELLS,
    DynamicV22PilotAdapter,
    quick_pilot_contract,
    run_t4_pilot_campaign,
)
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    production_state_from_smiles,
)
from compose_v4.rewrite.trace_shard import encode_state


class _Search:
    def __init__(self, direction="minimize"):
        self.config = type("Config", (), {"score_direction": direction})()
        self.entries = {}
        self.observations = {}
        self.oracle_protocol = "fixture-protocol"


def _candidate(channel, index, smiles):
    return {
        "candidate_id": f"{channel}:{index}",
        "endpoint": smiles,
        "source_state": {},
        "program": {"marks": [], "blocks": []},
        "assignment": [],
        "trace": {"states": [{}], "actions": [], "actual_changes": {}},
        "provenance": {
            "planner_channel": channel,
            "metadata": {"modules": [{"family": f"m{index}"}]},
        },
    }


def test_pre_model_allocation_preserves_exact_shallow_spine_and_diversity():
    smiles = ("CC", "CCC", "CCO", "CCN", "CCCC", "CCCO", "CCCN", "CCCl")
    candidates = [
        _candidate(SHALLOW_CHANNEL if index < 4 else STRUCTURED_CHANNEL, index, value)
        for index, value in enumerate(smiles)
    ]
    selected, receipt = allocate_parent_edits(
        _Search(),
        candidates,
        limit=6,
        config=DynamicV22PolicyConfig.for_t4(),
        model=None,
    )
    assert [row["candidate_id"] for row in selected[:3]] == [
        f"{SHALLOW_CHANNEL}:{index}" for index in range(3)
    ]
    assert receipt["policy"] == "pre_model_50_50_shallow_diverse"
    assert receipt["selected_by_role"]["preserved_shallow_pool_order"] == 3
    assert receipt["model_sha256"] is None


def test_learned_allocation_uses_archive_gain_but_retains_half_exploration(monkeypatch):
    smiles = ("CC", "CCC", "CCO", "CCN", "CCCC", "CCCO", "CCCN", "CCCl")
    candidates = [
        _candidate(SHALLOW_CHANNEL if index < 4 else STRUCTURED_CHANNEL, index, value)
        for index, value in enumerate(smiles)
    ]
    values = {smiles[index]: [float(index), 1.0] for index in range(len(smiles))}

    def features(self, candidate, **kwargs):
        return values[candidate["endpoint"]]

    monkeypatch.setattr(
        "compose_v4.control.dynamic_program_synthesis_v22.ParentEditFeatures.with_mutation_context",
        features,
    )
    rows = [
        {
            "features": [float(index), 1.0],
            "utility": float(index),
            "endpoint": f"train:{index}",
            "receipt_id": f"receipt:{index}",
            "oracle_protocol": "fixture-protocol",
        }
        for index in range(8)
    ]
    model = ParentEditModel.fit(
        rows,
        oracle_protocol="fixture-protocol",
        input_sha256="a" * 64,
        recipe=MUTATION_RECIPE,
    )
    selected, receipt = allocate_parent_edits(
        _Search(),
        candidates,
        limit=4,
        config=DynamicV22PolicyConfig.for_t4(),
        model=model,
    )
    assert len(selected) == 4
    assert receipt["selected_by_role"]["preserved_shallow_pool_order"] == 1
    assert receipt["selected_by_role"]["score_blind_structural_diversity"] == 1
    assert receipt["selected_by_role"]["predicted_archive_gain"] == 2
    assert receipt["uncertainty_used"] is False


def test_frontier_state_advances_rng_and_attempt_identity_without_oracle_calls():
    source = production_state_from_smiles("C1CCCCC1CCO", max_atoms=48)
    config = DynamicV22PolicyConfig.for_t4()

    def ineligible(row):
        return {
            **row,
            "qed": 0.55,
            "sa": 2.0,
            "sim": 0.7,
            "v": 0.05,
            "oracle_eligible": False,
        }

    initial = initial_frontier_state(seed=41)
    first = advance_t4_feasibility_frontier(
        source,
        config=config,
        state=initial,
        source_group="fixture",
        oracle_protocol="fixture-protocol",
        eligibility=ineligible,
        delta=0.4,
        attempts_per_channel=8,
    )
    second = advance_t4_feasibility_frontier(
        source,
        config=config,
        state=first["frontier_state"],
        source_group="fixture",
        oracle_protocol="fixture-protocol",
        eligibility=ineligible,
        delta=0.4,
        attempts_per_channel=8,
    )
    assert first["new_oracle_calls"] == second["new_oracle_calls"] == 0
    assert first["frontier_state"]["wave"] == 1
    assert second["frontier_state"]["wave"] == 2
    assert second["frontier_state"]["state_sha256"] != first["frontier_state"]["state_sha256"]
    assert second["frontier_state"]["attempts"] > first["frontier_state"]["attempts"]
    assert set(first["frontier_state"]["attempt_keys"]) <= set(
        second["frontier_state"]["attempt_keys"]
    )
    validate_frontier_state(second["frontier_state"], config=config)


def test_t4_bootstrap_preserves_reconstructable_generation_pool_identity():
    source = production_state_from_smiles("C1CCCCC1CCO", max_atoms=48)
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=7),
        attempts_per_batch=8,
        candidates_per_batch=4,
        wall_seconds=20,
    )

    def eligible(row):
        return {
            **row,
            "qed": 0.7,
            "sa": 2.0,
            "sim": 0.7,
            "oracle_eligible": True,
        }

    batch = initial_dynamic_program_batch_v22(
        source,
        (),
        config,
        source_group="fixture",
        oracle_protocol="fixture-protocol",
        eligibility=eligible,
        policy_config=DynamicV22PolicyConfig.for_t4(),
        delta=0.4,
    )
    generation_ids = batch["proposal_pool"]["generation_candidate_ids"]
    assert batch["proposal_pool"]["pool_id"] == identity(generation_ids)
    assert batch["allocation"]["selected_ids"] == [
        row["candidate_id"] for row in batch["candidates"]
    ]


def test_v22_snapshot_round_trip_preserves_task_model_boundary():
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=43, score_direction="maximize"),
        attempts_per_batch=4,
        candidates_per_batch=2,
        wall_seconds=5,
    )
    optimizer = DynamicV22ProgramOptimizer(
        config,
        source_group="fixture",
        oracle_protocol="pmo:fixture",
        hierarchy=None,
        policy_config=DynamicV22PolicyConfig.for_pmo(),
    )
    optimizer.shallow_rng.random(3)
    optimizer.structured_rng.random(2)
    restored = DynamicV22ProgramOptimizer.restore(optimizer.snapshot(), hierarchy=None)
    assert restored.policy_config == DynamicV22PolicyConfig.for_pmo()
    assert restored.shallow_rng.bit_generator.state == optimizer.shallow_rng.bit_generator.state
    assert (
        restored.structured_rng.bit_generator.state == optimizer.structured_rng.bit_generator.state
    )
    assert restored.parent_edit_model is None


def test_quick_pilot_contract_is_exact_and_excludes_runtime_comparators():
    contract = quick_pilot_contract()
    assert contract["contract_sha256"] == identity(contract["payload"])
    assert tuple(contract["payload"]["task_adapters"]["t4"]["cells"]) == T4_CELLS
    assert tuple(contract["payload"]["task_adapters"]["pmo"]["tasks"]) == PMO_TASKS
    assert contract["payload"]["charged_query_ceiling_per_unit"] == QUERY_BUDGET
    assert contract["payload"]["controller_core"]["runtime_comparator_or_winner_inputs"] == []
    assert contract["payload"]["launch_authorized"] is True
    t4 = DynamicV22PilotAdapter("t4", "t4:fixture", "fixture")
    pmo = DynamicV22PilotAdapter("pmo", "pmo:fixture", "fixture")
    assert t4.policy.archive_k == 1 and pmo.policy.archive_k == 10


def test_t4_runner_persists_frontier_across_zero_candidate_rounds(tmp_path, monkeypatch):
    source = production_state_from_smiles("C1CCCCC1CCO", max_atoms=48)
    config = replace(
        ProgramSearchConfig.program_only_recipe(seed=59),
        attempts_per_batch=2,
        candidates_per_batch=2,
        wall_seconds=10,
    )

    def scorer(_seed, *, delta):
        assert delta == 0.4

        def ineligible(row):
            return {
                **row,
                "qed": 0.55,
                "sa": 2.0,
                "sim": 0.7,
                "oracle_eligible": False,
                "endpoint_exclusion_reasons": ["qed"],
            }

        return ineligible

    monkeypatch.setattr("compose_v4.experiments.dynamic_v22_pilot.strict_endpoint_scorer", scorer)
    first = run_t4_pilot_campaign(
        output=tmp_path,
        source_state=encode_state(source),
        original_seed="C1CCCCC1CCO",
        target="fixture",
        oracle_protocol="t4:fixture",
        config=config,
        evaluate=lambda _smiles: (_ for _ in ()).throw(
            AssertionError("zero-eligible frontier must not call the oracle")
        ),
        rounds=2,
        max_rounds_this_invocation=1,
    )
    assert first["frontier_state"]["wave"] == 1
    result = run_t4_pilot_campaign(
        output=tmp_path,
        source_state=encode_state(source),
        original_seed="C1CCCCC1CCO",
        target="fixture",
        oracle_protocol="t4:fixture",
        config=config,
        evaluate=lambda _smiles: (_ for _ in ()).throw(
            AssertionError("resumed zero-eligible frontier must not call the oracle")
        ),
        rounds=2,
    )
    assert result["charged_oracle_calls"] == 0
    assert result["frontier_state"]["wave"] == 2
    assert len(result["history"]) == 2
    assert all(row["queries_this_round"] == 0 for row in result["history"])
    assert (tmp_path / "round_0000" / "complete.json").exists()
    assert (tmp_path / "round_0001" / "complete.json").exists()
