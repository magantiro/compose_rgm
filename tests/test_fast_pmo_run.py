"""The new entry point uses real programs and synthetic labels, never a PMO oracle."""

import json
from dataclasses import asdict, replace

import pytest
from rdkit import Chem

from compose_v4.control.edit_program import extract_program
from compose_v4.control.program_task import initialization_lock
from compose_v4.experiments.continuation_profile import publish_json
from compose_v4.experiments.fast_pmo_run import configuration, execute, load_contract
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.rewrite.trace_shard import encode_state
from tests.test_edit_program import ring_and_carbonyl


def test_direct_entry_completes_without_reference_or_learning(tmp_path):
    source, stages = ring_and_carbonyl()
    program, _ = extract_program(source, stages)
    initialized = initialization_lock(
        [{"state": encode_state(source), "source_id": "fixture"}],
        count=1,
        seed=7,
        source_sha256="a" * 64,
    )
    publish_json(tmp_path / "init.json", initialized)
    publish_json(
        tmp_path / "library.json", [{"program": program.payload(), "source_groups": ["fixture"]}]
    )
    c = {
        "run": {
            "name": "synthetic_fixture",
            "budget": 7,
            "rounds": 3,
            "queries_per_round": 2,
            "initialization_mode": "all_scored_pool",
            "initial_parent_fraction": 0.2,
            "max_seconds": 30,
        },
        "oracle_protocol": {"kind": "synthetic_size_not_task_oracle"},
        "initialization_path": "init.json",
        "library_path": "library.json",
        "configuration": asdict(replace(configuration(), seed=871, candidates_per_batch=2)),
    }
    progress = []
    result = execute(
        c,
        tmp_path,
        tmp_path / "run",
        evaluate=lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms() / 40,
        flush=lambda: None,
        progress=progress.append,
    )
    assert len(result["history"]) == 3
    assert result["oracle_calls"] <= 7 and result["rows"][0]["role"] == "initialization"
    assert result["reference_calls"] == result["model_fits"] == 0
    assert not any(r["model_updated"] for r in result["history"])
    assert result["query_curve"][-1]["queries"] == result["oracle_calls"]
    assert result["stop_reason"] in ("round_limit", "query_budget")
    assert all(
        r["available_scored_parents"] == 1
        for h in result["history"]
        if (r := h["initialization_parent"]) is not None
    )


def test_locked_cap_cannot_be_changed(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    contract = unseal(root / "configs/fast_pmo_run.json")
    contract["run"]["budget"] += 1
    seal(tmp_path / "configs/fast_pmo_run.json", contract)
    with pytest.raises(ValueError, match="bounded approved run"):
        load_contract(tmp_path)
    assert json.loads(json.dumps(asdict(configuration())))["channel_probabilities"][2] == 0


def test_container_import_does_not_read_host_manifest(monkeypatch):
    import runpy
    from pathlib import Path

    import modal

    monkeypatch.setattr(modal, "is_local", lambda: False)
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        if path.name == "fast_pmo_run.json":
            raise AssertionError("container import tried to read host manifest")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    loaded = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "modal_apps/fast_pmo_run_app.py")
    )
    assert "run" in loaded
