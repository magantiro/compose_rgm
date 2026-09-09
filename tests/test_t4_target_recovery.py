"""Small target-only fixtures; no remote assets or docking are needed."""

import inspect
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.experiments import t4_target_recovery as recovery
from compose_v4.experiments.continuation_profile import verify_file
from compose_v4.experiments.t4_task_search import PreparationConfig
from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import AtomInsert
from compose_v4.rewrite.trace_shard import decode_state, encode_state


def graph(smiles):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)


def test_target_objective_uses_graph_identity_not_smiles_spelling_or_feature_collision(monkeypatch):
    objective = recovery.TargetObjective("OCC")
    assert objective.evaluate(graph("CCO"))["exact_hit"]
    monkeypatch.setattr(recovery, "graph_kernel", lambda *_: np.ones((1, 1)))
    other = objective.evaluate(graph("CCC"))
    assert other["similarity"] == 1 and other["value"] < 1
    assert not other["exact_hit"]
    with pytest.raises(ValueError, match="invalid task-value SMILES"):
        recovery.TargetObjective("not a molecule")


def test_score_cache_does_not_return_another_slot_layout():
    original = graph("CO")
    payload = encode_state(original)
    payload["atom_types"][0], payload["atom_types"][1] = (
        payload["atom_types"][1],
        payload["atom_types"][0],
    )
    for name in ("formal_charges", "implicit_h_counts"):
        payload[name][0], payload[name][1] = payload[name][1], payload[name][0]
    permuted = decode_state(payload)
    objective = recovery.TargetObjective("CCO")
    a, b = objective.evaluate(original), objective.evaluate(permuted)
    assert a["similarity"] == b["similarity"]
    assert b["state"] == encode_state(permuted)
    assert a["state"] != b["state"]


def test_actual_hierarchy_searches_target_without_witness_and_replays():
    def law(current):
        slot = current.n_real_atoms
        return (
            ("atom_insert",) * slot,
            tuple(AtomInsert(slot, ELEMENT_TO_IDX["C"], 0, 3, ((i, 1),)) for i in range(slot)),
            (1.0 / slot,) * slot,
        )

    saved = {}
    result = recovery.run_recovery(
        graph("CC"),
        "CCC",
        law,
        editing_v2_semantic_rewrite_system(),
        config=recovery.RecoveryConfig(primitive_budget=2, max_rollouts=2, initial_rollouts=1),
        save=lambda name, value: saved.__setitem__(name, value),
    )
    assert result["status"] == "target_recovered"
    assert result["committed_edits"] == result["exact_replays"] == 1
    assert result["final"]["exact_hit"] and result["final"]["smiles"] == "CCC"
    assert result["new_oracle_calls"] == 0 and result["automatic_docking"] is False
    assert [e["stage"] for e in result["path"]] == ["where", "what", "how"]
    assert any(k.startswith("laws/") for k in saved)
    assert "search" in saved
    for event in result["path"]:
        d = event["decision"]
        assert d["kl"] <= 1 + 1e-10
        q, floor = np.array(d["probabilities"]), np.array(d["floor"])
        assert np.all(q >= d["epsilon"] * floor - 1e-12)
    json.dumps(result, allow_nan=False)
    assert "witness" not in inspect.signature(recovery.run_recovery).parameters


def test_initial_exact_hit_does_not_enumerate_or_search():
    def forbidden(_):
        raise AssertionError("already exact must not enumerate")

    result = recovery.run_recovery(graph("CC"), "CC", forbidden, None)
    assert result["status"] == "target_recovered"
    assert result["law_enumerations"] == result["committed_edits"] == 0


def test_no_support_is_recorded_as_failure_not_recovery():
    result = recovery.run_recovery(
        graph("C"),
        "CC",
        lambda _: ((), (), ()),
        editing_v2_semantic_rewrite_system(),
        config=recovery.RecoveryConfig(primitive_budget=1, max_rollouts=1, initial_rollouts=1),
    )
    assert result["status"] == "no_admissible_action"
    assert not result["final"]["exact_hit"]


def test_contract_horizon_and_zero_oracle_are_separate_from_benchmark():
    with pytest.raises(ValueError):
        PreparationConfig(primitive_budget=32)
    assert recovery.RecoveryConfig().primitive_budget == 32
    with pytest.raises(ValueError):
        recovery.RecoveryConfig(primitive_budget=33)
    with pytest.raises(ValueError):
        recovery.RecoveryConfig(seed=True)
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "configs/t4_target_recovery.json").read_text())
    digest = contract.pop("contract_sha256")
    assert identity(contract) == digest
    assert contract["compute"]["oracle_call_limit"] == 0
    assert not contract["training_authorized"]
    assert not contract["target"]["witness_actions_supplied"]
    assert recovery.RecoveryConfig(**contract["recovery"]) == recovery.RecoveryConfig()
    assert (root / "tools/t4_launch.py").read_text().count('"--target-recovery"') == 1
    target = contract["target"]
    census_path = root / target["census_path"]
    verify_file(census_path, target["census_sha256"])
    census = json.loads(census_path.read_text())
    cell = next(c for c in census["cells"] if c["cell"] == target["upstream_cell"])
    run = next(r for r in cell["runs"] if r["run_seed"] == target["upstream_run_seed"])
    assert any(w["canonical_smiles"] == target["smiles"] for w in run["winners"])


def test_administrative_stop_is_incomplete_without_fabricated_returns(monkeypatch):
    ticks = iter(range(100))
    monkeypatch.setattr(recovery, "perf_counter", lambda: next(ticks))
    result = recovery.run_recovery(
        graph("CC"),
        "CCC",
        lambda _: pytest.fail("elapsed stop must precede enumeration"),
        editing_v2_semantic_rewrite_system(),
        config=recovery.RecoveryConfig(
            primitive_budget=2, max_rollouts=1, initial_rollouts=1, stop_seconds=1
        ),
    )
    assert result["status"] == "administrative_elapsed_stop"
    assert result["planner"]["rollouts_interrupted"] == 1
    assert result["planner"]["rollouts_completed"] == 0
    assert result["committed_edits"] == 0


def test_launcher_routes_only_zero_oracle_target_function(monkeypatch, tmp_path):
    from tools import t4_launch

    source_root = Path(__file__).resolve().parents[1]
    for name in ("configs/t4_target_recovery.json", "modal_apps/genmol_t4_opt_app.py"):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / name).read_bytes())
    (tmp_path / "diagnostics").mkdir()
    monkeypatch.setattr(t4_launch, "ROOT", tmp_path)
    fake = types.ModuleType("modal_apps.run_process_v2_p50_app")
    fake.local_image_revision = lambda **_: {"image_revision_sha256": "a" * 64}
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    requested = []

    def resolve(app, name):
        requested.append((app, name))
        return types.SimpleNamespace(
            spawn=lambda task: types.SimpleNamespace(object_id="fixture-call")
        )

    monkeypatch.setattr(t4_launch.modal.Function, "from_name", resolve)
    t4_launch.launch_continuation_profile({"commit": "b" * 40}, target_recovery=True)
    assert requested == [("genmol-t4-opt", "t4_target_recovery")]
    receipt = json.loads((tmp_path / "diagnostics/t4_target_recovery_spawn.json").read_text())
    assert receipt["oracle_call_limit"] == 0
