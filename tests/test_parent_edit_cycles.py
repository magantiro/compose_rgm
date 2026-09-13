"""Launch census, warm learning and query locking; no real oracle calls."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.control.docking_value import identity
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask
from compose_v4.experiments.parent_edit_cycles import (
    configured,
    confirmation_selection,
    fit_measured_edits,
    pmo_oracle,
    warm_archive,
    zero_query_dispatch_failures,
)
from tests.test_adaptive_program_optimizer import optimizer


def test_pmo_dispatch_uses_all_frozen_tasks_without_scoring(monkeypatch):
    import sys
    import types

    root = Path(__file__).resolve().parents[1]
    protocols = json.loads((root / "configs/parent_edit_cycles.json").read_text())["payload"][
        "pmo_protocols"
    ]
    constructed, scored = [], []

    def fake_oracle(*, name):
        constructed.append(name)

        def evaluate(smiles):
            scored.append((name, smiles))
            return 0.25

        return evaluate

    monkeypatch.setitem(sys.modules, "tdc", types.SimpleNamespace(Oracle=fake_oracle))
    adapters = {name: pmo_oracle(name, protocols) for name in protocols}
    assert set(constructed) == {
        "albuterol_similarity",
        "isomers_c7h8n2o2",
        "perindopril_mpo",
        "scaffold_hop",
    }
    assert not scored
    assert adapters["isomers_c7h8n2o2"]("CC") == 0.25
    with pytest.raises(ValueError, match="outside the frozen campaign"):
        pmo_oracle("undeclared", protocols)
    assert len(constructed) == 4


def test_matched_engine_and_real_measured_mutation_training():
    root = Path(__file__).resolve().parents[1]
    if not (root / "diagnostics/parent_edit_cycles/prepared/braf_1_warm.json").exists():
        pytest.skip("requires documented prepared BRAF warm archive")
    base = {"kind": "t4", "seed": 11, "arm": "score_blind"}
    assert configured(base) == configured({**base, "arm": "learned"})
    config = replace(configured(base), require_broad_runtime=False)
    snapshot = warm_archive(root, "braf_1", config, None)
    from compose_v4.control.adaptive_program_optimizer import ProgramOptimizer

    search = ProgramOptimizer.restore(snapshot)
    source = json.loads((root / "configs/parent_edit_cycles.json").read_text())["payload"]
    domain = source["t4_protocols"]["braf_1"]
    task = ProgramTask("braf_1", identity(domain), "t4", domain["original_seed"], 0.4)
    model = fit_measured_edits(search, task)
    assert model.payload["training_endpoint_count"] == 21
    assert len(model.payload["training_receipts"]) == 27


def test_warm_campaign_does_not_recharge_history_and_reports_scored_rounds(tmp_path):
    search = optimizer()
    task = ProgramTask("synthetic", search.oracle_protocol, "pmo")
    # Explicitly synthetic historical reward, not a T4-to-PMO label conversion.
    search.observations["fixture:original"]["score"] = 0.2
    search.config = replace(search.config, score_direction="maximize")
    initialized = {"candidates": [], "role": "synthetic warm history"}
    initialized["lock_sha256"] = identity(initialized)
    events, flushes = [], []
    ledger = ProgramQueryLedger(
        tmp_path / "oracle", task, lambda _: 0.3, budget=4, flush=lambda: flushes.append(1)
    )
    result = run_program_campaign(
        output=tmp_path / "run",
        task=task,
        config=search.config,
        initialization=initialized,
        library=(),
        ledger=ledger,
        rounds=2,
        queries_per_round=2,
        warm_start=search.snapshot(),
        progress=events.append,
        stagnation_rounds=None,
    )
    assert result["oracle_calls"] == 4
    assert all(r["role"] == "candidate" for r in ledger.rows)
    assert len([r for r in events if r["phase"] == "scored"]) == 2
    assert len(flushes) >= 8
    assert all(r["proposal_seconds"] > 0 for r in result["history"])
    assert all(r["seconds"] >= 0 and r["completed_at_utc"] for r in ledger.rows)


def test_confirmation_is_bounded_and_deduplicates_identical_champions():
    contract = {"incumbents": {c: {"smiles": "CC"} for c in ("braf_1", "jak2_1")}}
    rows = [
        {
            "unit": {"name": c, "arm": a},
            "rows": [
                {"endpoint": "CCC", "score": -9, "status": "complete"},
                {"endpoint": "CCCC", "score": -8, "status": "complete"},
            ],
        }
        for c in ("braf_1", "jak2_1")
        for a in ("score_blind", "learned")
    ]
    locked = confirmation_selection(contract, rows)
    assert len(locked) == 8  # Each arm picked the same molecule; physical repeats shared.
    assert {r["seed"] for r in locked} == {1702, 1703}
    assert all(
        set(r["roles"]) == {"learned", "score_blind"} for r in locked if r["endpoint"] == "CCC"
    )


def test_dispatch_recovery_never_retries_charged_or_successful_units():
    from compose_v4.experiments.parent_edit_cycles import CONTRACT

    failed = {
        "unit": {"name": "scaffold_hop", "unit_id": "failed"},
        "oracle_calls": 0,
        "rows": [],
        "error": repr(ValueError("undeclared PMO oracle scaffold_hop")),
    }
    previous = {
        "group": "pmo",
        "task": {"files_sha256": {CONTRACT: "hash"}},
        "results": [
            failed,
            {
                "unit": {"name": "albuterol_similarity", "unit_id": "complete"},
                "oracle_calls": 10,
                "rows": [{"score": 0.2}],
            },
        ],
    }
    assert zero_query_dispatch_failures(previous, "hash") == ["failed"]
    with pytest.raises(ValueError, match="different"):
        zero_query_dispatch_failures(previous, "changed")
    failed["oracle_calls"] = 1
    with pytest.raises(ValueError, match="charged"):
        zero_query_dispatch_failures(previous, "hash")


def test_dispatch_recovery_uses_subset_and_cannot_be_claimed_twice(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from compose_v4.experiments import parent_edit_cycles as cycles
    from compose_v4.experiments.t4_matched_pilot import seal

    unit = {"unit_id": "failed", "kind": "pmo", "name": "scaffold_hop"}
    failed = {
        "unit": unit,
        "oracle_calls": 0,
        "rows": [],
        "error": repr(ValueError("undeclared PMO oracle scaffold_hop")),
    }
    previous = {
        "group": "pmo",
        "task": {"files_sha256": {cycles.CONTRACT: "hash"}},
        "results": [failed],
    }
    prior_id = "a" * 64
    seal(tmp_path / cycles.KIND / prior_id / "pmo/result.json", previous)
    contract = {"units": [unit, {"unit_id": "complete", "kind": "pmo"}]}
    monkeypatch.setattr(cycles, "validate_launch", lambda *args: contract)
    task = {
        "group": "pmo",
        "run_id": "b" * 64,
        "files_sha256": {cycles.CONTRACT: "hash"},
        "recovery": {"run_id": prior_id, "result_sha256": identity(previous), "units": ["failed"]},
    }
    submitted = []

    def parallel(tasks):
        submitted.extend(t["unit_id"] for t in tasks)
        return [failed]

    volume = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    cycles.run_group(task, tmp_path, tmp_path, volume, None, parallel)
    assert submitted == ["failed"]
    with pytest.raises(RuntimeError, match="already has"):
        cycles.run_group({**task, "run_id": "c" * 64}, tmp_path, tmp_path, volume, None, parallel)
    assert submitted == ["failed"]


def test_completed_t4_review_reconciles_calls_and_keeps_repeats_separate(tmp_path):
    from compose_v4.experiments.t4_matched_pilot import unseal
    from tools.parent_edit_cycles import ROOT, review_t4

    if not (ROOT / "diagnostics/parent_edit_cycles/t4_result.json").exists():
        pytest.skip("requires documented completed parent-edit T4 receipts")
    output = tmp_path / "review.json"
    review_t4(output)
    result = unseal(output)
    assert result["first_calls"] == 60
    assert result["confirmation_calls"] == 12
    assert result["total_calls"] == 72
    assert result["oracle_failures"] == 0
    assert result["repeat_only_means"]["jak2_1"]["learned"] == pytest.approx(-10.6)
    assert len(result["inputs_sha256"]) == 9
    assert not result["benchmark_claim"]


def test_started_guard_keeps_ambiguous_charge_without_restarting(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from compose_v4.experiments import parent_edit_cycles as cycles
    from compose_v4.experiments.continuation_profile import publish_json
    from compose_v4.experiments.t4_matched_pilot import seal

    unit = {"unit_id": "blocked", "kind": "pmo", "name": "scaffold_hop"}
    task = {"run_id": "a" * 64, "unit_id": "blocked"}
    folder = tmp_path / cycles.KIND / task["run_id"] / "units" / "blocked"
    seal(folder / "started.json", {"original": True})
    publish_json(folder / "oracle/query_000000/started.json", {"index": 0, "endpoint": "CC"})
    monkeypatch.setattr(cycles, "validate_launch", lambda *args: {"units": [unit]})
    monkeypatch.setattr(cycles, "hierarchy_for", lambda *args: pytest.fail("must not restart"))
    volume = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    result = cycles.run_unit(task, tmp_path, tmp_path, volume, None, None)
    assert result["failure_kind"] == "restart_guard"
    assert result["oracle_calls"] == 1 and result["rows"][0]["status"] == "started"
    assert not result["automatic_retry"] and not result["accounting_complete"]
    assert not (folder / "oracle/query_000000/result.json").exists()


def test_exceptional_worker_does_not_discard_other_results(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from compose_v4.experiments import parent_edit_cycles as cycles

    units = [{"unit_id": x, "kind": "pmo", "name": "scaffold_hop"} for x in ("failed", "ok")]
    monkeypatch.setattr(cycles, "validate_launch", lambda *args: {"units": units})
    task = {"run_id": "a" * 64, "group": "pmo"}
    volume = SimpleNamespace(reload=lambda: None, commit=lambda: None)

    def parallel(tasks):
        yield RuntimeError("fixture worker startup failure")
        yield {"unit": units[1], "rows": [], "oracle_calls": 0, "status": "completed_fixture"}

    result = cycles.run_group(task, tmp_path, tmp_path, volume, None, parallel)
    assert len(result["results"]) == 2
    assert result["results"][0]["failure_kind"] == "worker_exception"
    assert result["results"][1]["status"] == "completed_fixture"
    assert not result["accounting_complete"]
