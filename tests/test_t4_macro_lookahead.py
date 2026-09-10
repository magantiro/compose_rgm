"""Small controller/scheduling fixtures; synthetic scores, no docking or checkpoint."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from test_carbonyl_option import initial, kernel

from compose_v4.control.docking_value import DockingValue
from compose_v4.control.molecular_search_codec import encode_search_state
from compose_v4.control.molecular_task_search import MolecularHierarchy, MolecularSearchState
from compose_v4.experiments import t4_macro_lookahead as policy
from compose_v4.experiments.continuation_profile import ExecutorMeter, sha256_file
from compose_v4.experiments.t4_matched_pilot import seal

ROOT = Path(__file__).resolve().parents[1]


def candidate(smiles, budget=100):
    return {
        "smiles": smiles,
        "attempt_id": smiles,
        "node": {"budget": budget},
        "topology": {
            "cycle_rank": 0,
            "n_ring_systems": 0,
            "n_ring_atoms": 0,
            "n_heavy": len(smiles),
        },
    }


def test_delayed_return_selects_branch_through_ineligible_intermediate():
    a, child, b = candidate("CCC"), candidate("CCCC", 99), candidate("CCN")

    def score(smiles):
        return {
            "terminal_value": {"CCC": 0.0, "CCCC": 1.0, "CCN": 0.05}[smiles],
            "repair_value": 0.5,
        }

    chosen, audit = policy.select_branch([[a, child], [b]], score, np.random.default_rng(0))
    assert chosen == child and audit["score_field"] == "terminal_value"
    assert audit["probabilities"][0] > 0.5
    assert min(audit["probabilities"]) >= 0.05 and audit["kl"] <= 1.0
    _, repair = policy.select_branch(
        [[a], [b]],
        lambda s: {"terminal_value": 0.0, "repair_value": 0.8 if s == "CCC" else 0.01},
        np.random.default_rng(0),
    )
    assert repair["score_field"] == "repair_value" and repair["probabilities"][1] > 0


def test_parent_roles_preserve_repair_and_uniform_without_novelty_only_collapse():
    pool = [candidate("C" * n) for n in range(1, 13)]

    def score(smiles):
        return {
            "oracle_eligible": len(smiles) <= 6,
            "v": max(0, len(smiles) - 6) / 20,
            "planning_docking": -float(len(smiles)),
        }

    parents, audit = policy.parent_selection(
        pool,
        [{"smiles": "CC", "ds": -10.0}],
        pool[0],
        score,
        np.random.default_rng(0),
        expansion_counts={"CCC": 3},
    )
    assert len(parents) == len({r["smiles"] for r in parents}) == 8
    assert [r["role"] for r in audit] == [
        "original_seed",
        "observed_elite",
        "quality",
        "quality",
        "diversity",
        "diversity",
        "repair",
        "uniform",
    ]
    assert all(r["oracle_eligible"] for r in audit[:6])
    assert not audit[6]["oracle_eligible"] and audit[6]["smiles"] == "C" * 7


def test_small_actual_executor_search_and_saved_restart(tmp_path):
    # Synthetic prior labels solely to exercise the frozen-snapshot plumbing.
    rows = [{"smiles": "CCC", "ds": None, "round": 0}]
    rows += [{"smiles": "C" * n, "ds": -float(n), "round": 1} for n in range(1, 19) if n != 3]
    archive_path = tmp_path / "rounds/00/prior/archive.json"
    seal(archive_path, {"archive": rows})
    model = DockingValue.fit(rows, before_round=2, source_sha256=sha256_file(archive_path))
    value_path = tmp_path / "rounds/00/prior/value.json"
    seal(value_path, model.payload)
    seal(tmp_path / "rounds/00/before.json", {"archive": rows})
    node = MolecularSearchState.start(initial("CCC").graph, budget=110, root_id="fixture")
    parent = {
        "smiles": "CCC",
        "attempt_id": "fixture",
        "node": encode_search_state(node),
        "root_node": encode_search_state(node),
        "chain": [],
    }
    records = {}

    def save(name, value):
        records[name] = json.loads(json.dumps(value))

    def execute():
        process = kernel()
        meter = ExecutorMeter(None)
        with meter.instrument():
            result = policy.run_local_search(
                parent,
                MolecularHierarchy(process, lazy_applicability=True, include_carbonyl_options=True),
                seed=7,
                save=save,
                read=records.get,
                meter=meter,
                progress={},
                prefix="worker",
                task={"round": 0, "prior_value_sha256": sha256_file(value_path)},
                episode=tmp_path,
                contract={},
            )
        return result, meter.calls

    first, calls = execute()
    assert 2 <= first["attempts"] <= 6 and calls > 0
    assert first["branch_decision"] == records["task_branch_decision"]
    assert first["branch_decision"]["value_snapshot_sha256"] == model.payload["snapshot_sha256"]
    assert all(
        c["node"]["budget"] < 110 and c["node"]["stage"] == "where" for c in first["candidates"]
    )
    assert any(len(c["chain"]) >= 2 for c in first["candidates"])
    second, calls = execute()
    assert calls == 0
    assert {k: v for k, v in second.items() if k != "proposal_seconds"} == {
        k: v for k, v in first.items() if k != "proposal_seconds"
    }


def test_contract_and_launch_bounds():
    contract = policy.contract_at(ROOT)
    assert contract["compute"]["proposal_containers"] == 16
    tree = ast.parse((ROOT / "modal_apps/genmol_t4_opt_app.py").read_text())
    for name, timeout, containers in (
        ("t4_macro_lookahead", 7200, 1),
        ("t4_macro_lookahead_propose", 1800, 16),
    ):
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        settings = {
            k.arg: ast.literal_eval(k.value)
            for k in function.decorator_list[0].keywords
            if k.arg in ("timeout", "max_containers", "retries")
        }
        assert settings == {"timeout": timeout, "max_containers": containers, "retries": 0}


def test_launcher_uses_deployed_lookahead_and_bounded_receipt(tmp_path, monkeypatch):
    from modal_apps import run_process_v2_p50_app
    from tools import t4_launch

    calls = []

    def lookup(app, function):
        assert (app, function) == ("genmol-t4-opt", policy.KIND)

        def spawn(task):
            calls.append(task)
            return SimpleNamespace(object_id="fixture-call")

        return SimpleNamespace(spawn=spawn)

    (tmp_path / "diagnostics").mkdir()
    monkeypatch.setattr(t4_launch, "ROOT", tmp_path)
    monkeypatch.setattr(t4_launch, "_sha256", lambda path: "b" * 64)
    monkeypatch.setattr(t4_launch.modal.Function, "from_name", lookup)
    monkeypatch.setattr(
        run_process_v2_p50_app,
        "local_image_revision",
        lambda **kw: {"commit": "a" * 40, "image_revision_sha256": "c" * 64},
    )
    t4_launch.launch_continuation_profile({"commit": "a" * 40}, macro_lookahead=True)
    receipt = json.loads((tmp_path / "diagnostics/t4_macro_lookahead_spawn.json").read_text())
    assert len(calls) == 1 and receipt["oracle_call_limit"] == 40
    assert receipt["schema_version"] == "t4_macro_lookahead_spawn_v1"
