import json
import shutil
from pathlib import Path

import pytest

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v21 import DynamicV21ProgramOptimizer
from compose_v4.control.program_campaign import ProgramQueryLedger
from compose_v4.control.program_task import ProgramTask
from compose_v4.experiments import pmo_dynamic_v21 as original
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.pmo_dynamic_v21_recovery import (
    FAILED_V21_CALLS_PER_TASK,
    FAILURE_CENSUS,
    MAX_NEW_CALLS_PER_TASK,
    RECOVERY_CONTRACT,
    RECOVERY_PREFLIGHT,
    SOURCE_RUN_ID,
    PMODynamicV21ContinuityAdapter,
    _unit_folder,
    audit_recovery_unit,
    consume_existing_round1_observation,
    load_recovery_contract,
    recover_pending_round,
)
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]


def _locked_rows(task="celecoxib_rediscovery"):
    folder = _unit_folder(ROOT, task)
    complete = json.loads((folder / "campaign/round_0000/complete.json").read_text())
    pending = json.loads((folder / "campaign/round_0001/pending.json").read_text())[
        "batch"
    ]
    query = json.loads((folder / "oracle/query_000032/result.json").read_text())
    return folder, complete, pending, query


def test_campaign_boundary_adapter_preserves_core_pool_mismatch_invariant():
    _, complete, pending, query = _locked_rows()
    candidate = pending["candidates"][0]
    unchanged = DynamicV21ProgramOptimizer.restore(complete["snapshot"])
    with pytest.raises(
        ValueError, match="bootstrap candidates came from different pools"
    ):
        unchanged.add_measured_program(
            candidate, receipt_id=query["receipt_id"], score=query["score"]
        )

    recovered = PMODynamicV21ContinuityAdapter.restore(complete["snapshot"])
    continuity_pool = recovered._bootstrap_pool_id
    generation_pool = candidate["provenance"]["dynamic_v21_bootstrap"]["pool_id"]
    before_selected = sum(
        recovered.allocator_state["channels"][channel]["selected_for_oracle"]
        for channel in recovered.allocator_state["channels"]
    )
    recovered.add_measured_program(
        candidate, receipt_id=query["receipt_id"], score=query["score"]
    )
    after_selected = sum(
        recovered.allocator_state["channels"][channel]["selected_for_oracle"]
        for channel in recovered.allocator_state["channels"]
    )
    assert continuity_pool != generation_pool
    assert recovered._bootstrap_pool_id == continuity_pool
    assert recovered._pmo_generation_pool_ids == [continuity_pool, generation_pool]
    assert after_selected - before_selected == len(pending["candidates"])
    assert recovered.observations[query["receipt_id"]]["score"] == query["score"]
    restored = PMODynamicV21ContinuityAdapter.restore(recovered.snapshot())
    assert restored._bootstrap_pool_id == continuity_pool
    assert restored._pmo_generation_pool_ids == [continuity_pool, generation_pool]

    third_pool = "f" * 64
    first = json.loads(json.dumps(pending["candidates"][0]))
    second = json.loads(json.dumps(pending["candidates"][1]))
    for row in (first, second):
        row["provenance"]["dynamic_v21_bootstrap"]["pool_id"] = third_pool
    selected_before = sum(
        restored.allocator_state["channels"][channel]["selected_for_oracle"]
        for channel in restored.allocator_state["channels"]
    )
    restored.add_measured_program(first, receipt_id="third-pool-0", score=0.1)
    selected_once = sum(
        restored.allocator_state["channels"][channel]["selected_for_oracle"]
        for channel in restored.allocator_state["channels"]
    )
    restored.add_measured_program(second, receipt_id="third-pool-1", score=0.2)
    selected_twice = sum(
        restored.allocator_state["channels"][channel]["selected_for_oracle"]
        for channel in restored.allocator_state["channels"]
    )
    assert selected_once - selected_before == len(pending["candidates"])
    assert selected_twice == selected_once
    assert restored._pmo_generation_pool_ids[-1] == third_pool


def test_zero_oracle_audit_covers_four_contiguous_failed_ledgers_read_only():
    protected = [
        ROOT / original.LAUNCH,
        ROOT / original.LAUNCH_V2,
        ROOT / original.RESULT,
    ]
    protected.extend(
        _unit_folder(ROOT, task) / "failure.json" for task in original.TASKS
    )
    before = {str(path): sha256_file(path) for path in protected}
    rows = [audit_recovery_unit(ROOT, task) for task in original.TASKS]
    after = {str(path): sha256_file(path) for path in protected}
    assert before == after
    assert {row["task"] for row in rows} == set(original.TASKS)
    assert all(row["source_run_id"] == SOURCE_RUN_ID for row in rows)
    assert all(
        row["charged_completed_queries"] == FAILED_V21_CALLS_PER_TASK for row in rows
    )
    assert all(row["unresolved_queries"] == 0 for row in rows)
    assert all(row["round0_complete"] and row["round1_pending"] for row in rows)
    assert all(row["round1_consumed_prefix"] == 1 for row in rows)
    assert all(row["remaining_query_ceiling"] == MAX_NEW_CALLS_PER_TASK for row in rows)
    assert sum(row["new_oracle_calls"] for row in rows) == 0


def test_existing_round1_score_is_consumed_before_any_new_oracle_call(tmp_path):
    folder, complete, pending, _ = _locked_rows()
    copied = tmp_path / "oracle"
    shutil.copytree(folder / "oracle", copied)
    contract = original.load_contract(ROOT)
    protocol = original._runtime_protocol(contract, "celecoxib_rediscovery")
    task = ProgramTask("celecoxib_rediscovery", identity(protocol), "pmo")
    calls = []

    def forbidden(endpoint):
        calls.append(endpoint)
        raise AssertionError("existing observation consumption called the oracle")

    ledger = ProgramQueryLedger(copied, task, forbidden, budget=original.QUERY_BUDGET)
    search = PMODynamicV21ContinuityAdapter.restore(complete["snapshot"])
    before_observations = len(search.observations)
    receipt = consume_existing_round1_observation(search, pending, ledger)
    assert calls == []
    assert receipt["query_index"] == 32
    assert receipt["charged_queries_before"] == receipt["charged_queries_after"] == 33
    assert receipt["new_oracle_calls"] == 0
    assert len(search.observations) == before_observations + 1
    assert search.observations[receipt["receipt_id"]]["score"] == receipt["score"]


def test_pending_round_recovery_keeps_locked_order_after_cached_score(tmp_path):
    folder, _, pending, _ = _locked_rows()
    copied = tmp_path / "unit"
    shutil.copytree(folder, copied)
    contract = original.load_contract(ROOT)
    initialized = original._load_initialization(ROOT, contract)
    protocol = original._runtime_protocol(contract, "celecoxib_rediscovery")
    task = ProgramTask("celecoxib_rediscovery", identity(protocol), "pmo")
    calls = []

    def synthetic(endpoint):
        calls.append(endpoint)
        return 0.25

    ledger = ProgramQueryLedger(
        copied / "oracle", task, synthetic, budget=original.QUERY_BUDGET
    )
    summary = recover_pending_round(ROOT, copied, task, initialized, ledger)
    assert len(calls) == 15
    assert calls[0] == pending["candidates"][1]["endpoint"]
    assert [row["index"] for row in ledger.rows] == list(range(48))
    assert summary["recovery"] == {
        "schema_version": "pmo_dynamic_v21_pending_round_recovery_v1",
        "existing_observation_consumed_before_new_calls": True,
        "existing_query_index": 32,
        "new_oracle_calls_in_round": 15,
    }
    complete = json.loads((copied / "campaign/round_0001/complete.json").read_text())
    restored = PMODynamicV21ContinuityAdapter.restore(complete["snapshot"])
    assert len(restored.observations) == 32
    assert len(restored._pmo_generation_pool_ids) == 2


def test_sealed_recovery_contract_and_failure_census_are_zero_oracle():
    contract = load_recovery_contract(ROOT)
    census = unseal(ROOT / FAILURE_CENSUS)
    preflight = unseal(ROOT / RECOVERY_PREFLIGHT)
    assert contract["source_run_id"] == SOURCE_RUN_ID
    assert contract["max_new_calls_per_task"] == MAX_NEW_CALLS_PER_TASK
    assert contract["automatic_retries"] == 0
    assert contract["initial_task_specific_complete_routes"] == 0
    assert contract["t4_affected"] is False
    assert contract["failure_census_sha256"] == sha256_file(ROOT / FAILURE_CENSUS)
    assert census["source_artifact_count"] == len(census["source_artifact_sha256"])
    assert census["preserved_dynamic_v21_calls"] == 132
    assert census["completed_dynamic_v0_calls"] == 4000
    assert census["new_oracle_calls"] == 0
    assert preflight["contract_sha256"] == sha256_file(ROOT / RECOVERY_CONTRACT)
    assert preflight["passed"] is True
    assert preflight["new_oracle_calls"] == 0
