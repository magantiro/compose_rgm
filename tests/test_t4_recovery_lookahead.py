"""Focused one-step planning and complete-decision regression checks."""

import ast
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from compose_v4.experiments.t4_macro_beam import BeamConfig
from compose_v4.experiments.t4_recovery_lookahead import (
    best_new_endpoint,
    compare_retention,
    decision_pool,
    enumerator_source_hash,
)


def test_cache_hash_uses_source_not_version_dependent_ast(monkeypatch):
    source = "def enumerate_products(x):\n    return x\n"
    monkeypatch.setattr(
        ast, "dump", lambda *a, **k: pytest.fail("AST serialization is not portable")
    )
    assert enumerator_source_hash(source) == hashlib.sha256(source.rstrip().encode()).hexdigest()
    assert enumerator_source_hash(source + "\nelsewhere = 1\n") == enumerator_source_hash(source)
    assert enumerator_source_hash(source.replace("return x", "return 0")) != enumerator_source_hash(
        source
    )


def test_full_pool_not_only_known_recoverable_state():
    rows = [
        {"attempt_id": f"levels/01/{i}", "smiles": "C" * (i + 1), "node": {"budget": 2}}
        for i in range(9)
    ]
    assert decision_pool(rows[::-1], 9) == rows
    with pytest.raises(ValueError, match="complete"):
        decision_pool(rows[:8], 9)
    rows[0]["node"]["budget"] = 0
    with pytest.raises(ValueError, match="remaining budget"):
        decision_pool(rows, 9)


def test_one_step_value_and_selection_do_not_reward_incumbent_return():
    payload = {"best": -10.0, "scale": 1.0}
    rows = [
        {
            "smiles": "CC",
            "oracle_eligible": True,
            "in_prior_archive": True,
            "predicted_docking": -11.0,
        }
    ]
    assert best_new_endpoint(rows, payload)["one_step_value"] == 0
    rows.append(
        {
            "smiles": "CCC",
            "oracle_eligible": False,
            "in_prior_archive": False,
            "predicted_docking": -12.0,
        }
    )
    assert best_new_endpoint(rows, payload)["endpoint"] is None
    rows.append(
        {
            "smiles": "CCCC",
            "oracle_eligible": True,
            "in_prior_archive": False,
            "predicted_docking": -9.0,
        }
    )
    assert best_new_endpoint(rows, payload)["endpoint"]["smiles"] == "CCCC"
    pool = [
        {"attempt_id": str(i), "smiles": s, "desirability": float(i == 0)}
        for i, s in enumerate(("C", "CC", "CCC"))
    ]
    repairs = {r["smiles"]: rows if r["smiles"] == "CCC" else [] for r in pool}
    incumbent = {"attempt_id": "incumbent", "smiles": "CCO", "node": "exact", "chain": []}
    from dataclasses import asdict

    output = compare_retention(
        pool, repairs, incumbent, payload, asdict(BeamConfig(preserve_root=True))
    )
    before = output["arms"]["immediate"]["recoverable_first_slot_probability"]
    after = output["arms"]["lookahead"]["recoverable_first_slot_probability"]
    assert after > before
    for arm in output["arms"].values():
        d = arm["decision"]["offspring"]
        p, q = np.array(d["first_slot_reference"]), np.array(d["first_slot_probabilities"])
        assert np.all(q >= 0.1 * p - 1e-12) and d["first_slot_kl_against_empirical_pool"] <= 1
    with pytest.raises(ValueError, match="every member"):
        compare_retention(pool, {}, incumbent, payload, asdict(BeamConfig(preserve_root=True)))


def test_launcher_partitions_one_decision_into_three_workers(monkeypatch, tmp_path):
    from modal_apps import run_process_v2_p50_app
    from tools import t4_launch

    calls = []

    def lookup(app, name):
        assert (app, name) == ("genmol-t4-opt", "t4_recovery_lookahead")

        def spawn(task):
            calls.append(task)
            return SimpleNamespace(object_id=f"lookahead-{task['case_index']}")

        return SimpleNamespace(spawn=spawn)

    (tmp_path / "diagnostics").mkdir()
    monkeypatch.setattr(t4_launch, "ROOT", tmp_path)
    monkeypatch.setattr(t4_launch, "_sha256", lambda _: "a" * 64)
    monkeypatch.setattr(t4_launch.modal.Function, "from_name", lookup)
    monkeypatch.setattr(
        run_process_v2_p50_app,
        "local_image_revision",
        lambda **_: {"commit": "b" * 40, "image_revision_sha256": "c" * 64},
    )
    t4_launch.launch_continuation_profile({"commit": "b" * 40}, recovery_lookahead=True)
    receipt = json.loads((tmp_path / "diagnostics/t4_recovery_lookahead_spawn.json").read_text())
    assert [c["case_index"] for c in calls] == [0, 1, 2]
    assert receipt["oracle_calls"] == 0 and len(receipt["cases"]) == 3
