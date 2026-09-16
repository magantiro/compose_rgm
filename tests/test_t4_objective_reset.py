"""Zero-oracle orchestration, budget and exact adapter-command regression tests."""

import ast
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.objective_program_search import v0_search_config
from compose_v4.experiments import t4_objective_reset as reset
from compose_v4.experiments.t4_docking_adapter import dock_t4
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.rewrite.trace_shard import encode_state


class Volume:
    def commit(self):
        pass


@pytest.fixture
def unit(monkeypatch):
    monkeypatch.setattr(
        reset,
        "runtime_config",
        lambda seed: replace(
            v0_search_config(seed=seed),
            attempts_per_batch=8,
            candidates_per_batch=2,
            wall_seconds=300,
        ),
    )
    monkeypatch.setattr(
        reset, "strict_endpoint_scorer", lambda *a, **k: lambda c: {"oracle_eligible": True}
    )
    return {
        "cell": "fixture",
        "unit_id": "fixture",
        "target": "fixture",
        "source_idx": 0,
        "original_seed": "CCCC",
        "controller_seed": 31,
        "docking_seed": 1701,
        "source_group": "fixture",
        "oracle_protocol": "fixture:no-oracle",
        "source_state": encode_state(pad_molecular_graph(smiles_to_molecular_graph("CCCC"), 48)),
    }


def test_budget_receipts_and_finished_resume_are_idempotent(unit, tmp_path):
    queries = []

    def fake_dock(smiles, *_args):
        queries.append(smiles)
        return -1.0

    kwargs = {
        "arm": "support_control",
        "run_id": "synthetic-fixture",
        "volume": Volume(),
        "dock": fake_dock,
        "calls": 2,
    }
    result = reset.run_scored_unit(unit, tmp_path, **kwargs)
    assert len(queries) == result["query_count"] == 2
    assert result["termination"] == "query_budget"
    assert sum(s["charged"] for s in result["channel_stats"].values()) == 2
    assert reset.run_scored_unit(unit, tmp_path, **kwargs) == result
    assert len(queries) == 2


def test_failed_docking_still_consumes_budget(unit, tmp_path):
    result = reset.run_scored_unit(
        unit,
        tmp_path,
        arm="objective_search",
        run_id="synthetic-fixture",
        volume=Volume(),
        dock=lambda *a: None,
        calls=2,
    )
    assert result["query_count"] == 2 and result["champion"] is None
    assert all(r["failure"] == "oracle_failed" for r in result["rows"])
    assert sum(s["charged"] for s in result["channel_stats"].values()) == 2


def test_interrupted_query_is_not_automatically_retried(unit, tmp_path):
    attempted = []

    def interrupt(*args):
        attempted.append(args)
        raise InterruptedError("synthetic process loss")

    kwargs = {
        "arm": "support_control",
        "run_id": "synthetic-fixture",
        "volume": Volume(),
        "dock": interrupt,
        "calls": 2,
    }
    with pytest.raises(InterruptedError):
        reset.run_scored_unit(unit, tmp_path, **kwargs)
    with pytest.raises(RuntimeError, match="ambiguous charged query"):
        reset.run_scored_unit(unit, tmp_path, **kwargs)
    assert len(attempted) == 1


def test_empty_rounds_advance_and_persist_without_oracle(unit, tmp_path, monkeypatch):
    monkeypatch.setattr(
        reset, "strict_endpoint_scorer", lambda *a, **k: lambda c: {"oracle_eligible": False}
    )

    def forbidden(*args):
        raise AssertionError("empty pools must never dock")

    result = reset.run_scored_unit(
        unit,
        tmp_path,
        arm="support_control",
        run_id="synthetic-fixture",
        volume=Volume(),
        dock=forbidden,
        calls=2,
        max_rounds=2,
    )
    saved = unseal(tmp_path / "checkpoint.json")
    assert result["query_count"] == 0 and result["termination"] == "round_limit"
    assert saved["bootstrap"]["cursor"] == 16
    assert unseal(tmp_path / "progress.json")["phase"] == "empty_proposal_round"


def test_docking_commands_match_frozen_adapter(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "modal_apps/genmol_t4_opt_app.py").read_text())
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_dock")
    coordinates = ((1.0, 2.0, 3.0), (20, 20, 20))
    namespace = {"BOXES": {"fixture": coordinates}}
    # Execute only the trusted, versioned local function, without importing its
    # Modal image or touching the network. Every subprocess is mocked below.
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "frozen_dock", "exec"), namespace)  # noqa: S102
    observed = []

    def fake_run(command, **kwargs):
        observed.append((command, kwargs))
        if "--out" in command:
            Path(command[command.index("--out") + 1]).write_text("REMARK VINA RESULT: -1.0 0 0\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    tag = str(tmp_path / "pose")
    box = {"coordinates": coordinates, "receptor": "/opt/dock/receptors/fixture.pdbqt"}
    assert dock_t4("CCCC", tag, 1701, box=box) == -1.0
    new = list(observed)
    observed.clear()
    assert namespace["_dock"]("CCCC", "fixture", tag, cpu=1, random_seed=1701) == -1.0
    # The old adapter adds '/tmp/' to an absolute fixture directory. Normalize
    # only that synthetic path; all protocol flags/timeouts must be identical.
    old = [
        ([s.replace("/tmp/" + tag, tag) for s in command], kwargs) for command, kwargs in observed
    ]
    assert new == old
