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
    warm_archive,
)
from tests.test_adaptive_program_optimizer import optimizer


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
