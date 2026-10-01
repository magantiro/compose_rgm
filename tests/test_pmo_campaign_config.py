"""PMO campaign inputs are explicit, pinned and separate from charged queries."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.control.pmo_reference_controller import (
    PmoReferenceController,
    initial_pmo_program_batch,
)
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask
from compose_v4.experiments.pmo.__main__ import (
    _initialization,
    _jump_checkpoint,
    _search_config,
    main,
    read_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_bundled_proposal_ablation_configs_change_only_the_arm():
    names = ("example.json", "uniform_chain.json", "created_atom_rebinding.json")
    configurations = [json.loads((ROOT / "experiments/pmo" / name).read_text()) for name in names]
    assert [config.pop("arm") for config in configurations] == [
        "structured",
        "uniform_chain",
        "created_atom_rebinding",
    ]
    assert configurations[0] == configurations[1] == configurations[2]


def _config(tmp_path: Path) -> Path:
    raw = json.loads((ROOT / "experiments/pmo/example.json").read_text())
    raw["oracle"]["python"] = sys.executable
    raw["guidance"] = {
        "mode": "off",
        "strength": 0.0,
        "log_weight_cap": 1.0,
        "on_unsupported": "baseline",
    }
    raw["reference"] = None
    for name in ("jump_plans", "initialization"):
        raw[name]["path"] = str(ROOT / "experiments/pmo" / raw[name]["path"])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    return path


def test_example_assets_load_without_oracle_call(tmp_path):
    config = read_config(_config(tmp_path))
    initial = _initialization(config["initialization"])
    jump = _jump_checkpoint(config["jump_plans"])
    assert len(initial["candidates"]) == 16
    assert initial["new_oracle_calls"] == 0
    assert jump["fit_scope"] == "shared_all_routes"
    assert len(jump["plan_latents"]) > 0


def test_validate_command_makes_no_oracle_worker(tmp_path, monkeypatch, capsys):
    def forbidden_worker(**kwargs):
        raise AssertionError("validate must not start an oracle worker")

    monkeypatch.setattr("compose_v4.experiments.pmo.__main__.PmoOracleClient", forbidden_worker)
    main(["validate", "--config", str(_config(tmp_path))])
    assert json.loads(capsys.readouterr().out)["charged_calls"] == 0


@pytest.mark.parametrize("field", ["initialization", "jump_plans"])
def test_required_campaign_asset_cannot_be_removed(tmp_path, field):
    path = _config(tmp_path)
    raw = json.loads(path.read_text())
    raw[field] = None
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=f"{field} is required"):
        read_config(path)


def test_asset_hash_mismatch_fails_before_oracle_start(tmp_path):
    path = _config(tmp_path)
    raw = json.loads(path.read_text())
    raw["jump_plans"]["sha256"] = "0" * 64
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="hash|SHA|identity"):
        read_config(path)


@pytest.mark.parametrize("field", ["jump_plans", "oracle"])
def test_missing_input_names_the_configuration_field(tmp_path, field):
    path = _config(tmp_path)
    raw = json.loads(path.read_text())
    if field == "oracle":
        raw["oracle"]["python"] = "missing-oracle-python"
        expected = "oracle.python"
    else:
        raw["jump_plans"]["path"] = "missing-jump-plans.json.gz"
        expected = "jump_plans.path"
    path.write_text(json.dumps(raw))
    with pytest.raises(FileNotFoundError, match=expected):
        read_config(path)


def test_asset_backed_task_requires_asset_root(tmp_path):
    path = _config(tmp_path)
    raw = json.loads(path.read_text())
    raw["task"] = "gsk3b"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="requires a pinned PyTDC asset root"):
        read_config(path)


def test_cold_start_receives_declared_region_portfolio(monkeypatch):
    import compose_v4.control.pmo_reference_controller as controller

    captured = {}

    def collect(*args, **kwargs):
        captured.update(kwargs)
        return {"fixture": True}

    monkeypatch.setattr(controller, "initial_dynamic_program_batch_v21", collect)
    result = initial_pmo_program_batch(
        object(),
        (),
        object(),
        source_group="fixture",
        oracle_protocol="synthetic_constant_no_oracle",
        eligibility=lambda row: row,
        proposal_spec={
            "candidate_pool_limit": 128,
            "replacement_option_rate": 0.5,
            "realization_seconds_cap": 100.0,
        },
    )
    assert result == {"fixture": True}
    assert captured["candidate_pool_limit"] == 128
    assert captured["dynamic_synthesis_kwargs"]["replacement_option_rate"] == 0.5
    assert (
        captured["dynamic_synthesis_kwargs"]["replacement_option"].region_law
        is captured["dynamic_synthesis_kwargs"]["region_law"]
    )


def test_one_campaign_round_with_synthetic_scores_only(tmp_path):
    config = read_config(_config(tmp_path))
    initial = _initialization(config["initialization"])
    jump = _jump_checkpoint(config["jump_plans"])
    task = ProgramTask("fixture", "synthetic_constant_no_oracle", "pmo")
    calls = []

    def score(smiles):
        calls.append(smiles)
        return 0.5

    ledger = ProgramQueryLedger(tmp_path / "oracle", task, score, budget=17)
    search = replace(_search_config(11, 1), attempts_per_batch=2, wall_seconds=2.0)
    result = run_program_campaign(
        output=tmp_path / "campaign",
        task=task,
        config=search,
        initialization=initial,
        library=(),
        ledger=ledger,
        rounds=1,
        queries_per_round=1,
        stagnation_rounds=None,
        bootstrap_rounds=1,
        initialization_mode="all_scored_pool",
        optimizer_type=PmoReferenceController,
        optimizer_kwargs={
            "jump_checkpoint": jump,
            "proposal_spec": config["proposal"],
            "reference_spec": {
                "guidance": config["guidance"],
                "asset": None,
            },
        },
        initial_batch_fn=initial_pmo_program_batch,
        initial_batch_kwargs={"proposal_spec": config["proposal"]},
    )
    assert result["oracle_calls"] == len(calls)
    assert 16 <= len(calls) <= 17
    assert (tmp_path / "campaign/round_0000/complete.json").exists()
    previous_calls = len(calls)
    resumed_ledger = ProgramQueryLedger(tmp_path / "oracle", task, score, budget=17)
    resumed = run_program_campaign(
        output=tmp_path / "campaign",
        task=task,
        config=search,
        initialization=initial,
        library=(),
        ledger=resumed_ledger,
        rounds=1,
        queries_per_round=1,
        stagnation_rounds=None,
        bootstrap_rounds=1,
        initialization_mode="all_scored_pool",
        optimizer_type=PmoReferenceController,
        optimizer_kwargs={
            "jump_checkpoint": jump,
            "proposal_spec": config["proposal"],
            "reference_spec": {"guidance": config["guidance"], "asset": None},
        },
        initial_batch_fn=initial_pmo_program_batch,
        initial_batch_kwargs={"proposal_spec": config["proposal"]},
    )
    assert resumed["oracle_calls"] == previous_calls
    assert len(calls) == previous_calls
