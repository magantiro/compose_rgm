"""Offline profile tests use a declared fixture, never a learned-model substitute."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from compose_v4.control.continuation import ContinuationBudgetExceeded
from compose_v4.control.option_continuation import exact_graph_key
from compose_v4.experiments.continuation_gate import engineering_fixture
from compose_v4.experiments.continuation_profile import (
    ExecutorMeter,
    encode_action,
    initial_state,
    publish_json,
    run_profile,
    sha256_file,
    state_payload,
    verify_file,
)
from compose_v4.rewrite.kernel import RewriteSystem
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]


def fixture_profile(output, budget=64):
    node, kernel = engineering_fixture()
    return run_profile(
        node,
        kernel.enumerate_law,
        kernel.system,
        {"max_executor_applications": budget, "max_expansions": 65, "max_terminal_evaluations": 65},
        output,
        snapshot_id="engineering-fixture-not-Rtheta",
        commit_volume=lambda: None,
        progress={},
    )


def test_profile_persists_complete_rows_and_exact_executor_attempts(tmp_path):
    result = fixture_profile(tmp_path)
    assert result["status"] == "guided"
    assert result["root_cache_parity"] is True
    assert result["kernel_work"]["executor_applications"] == 4
    assert result["oracle_calls"] == 0
    assert result["decision"]["probabilities"] == pytest.approx([0.95, 0.05])
    assert len(result["rows"]) == 3
    for receipt in result["rows"]:
        path = tmp_path / receipt["path"]
        assert sha256_file(path) == receipt["sha256"]
        row = json.loads(path.read_text())
        assert len(row["marks"]) == len(row["probabilities"]) == len(row["successors"])
        assert sum(row["probabilities"]) == pytest.approx(1.0)
    attempts = [json.loads(p.read_text()) for p in (tmp_path / "attempts").glob("*.json")]
    assert sum(len(a["attempts"]) for a in attempts) == 4
    assert not any(a["partial_row"] for a in attempts)
    assert all(a["attempts"][0]["product"]["formal_charges"] for a in attempts)


@pytest.mark.parametrize(
    "budget,status",
    [
        (0, "root_executor_budget_exhausted"),
        (1, "root_executor_budget_exhausted"),
        (2, "budget_abstention"),
        (3, "budget_abstention"),
    ],
)
def test_profile_counts_failed_work_without_publishing_partial_law(tmp_path, budget, status):
    result = fixture_profile(tmp_path, budget)
    assert result["status"] == status
    assert result["kernel_work"]["executor_applications"] == budget
    assert result["chemistry_comparison_authorized"] is False
    if result["decision"]:
        assert result["decision"]["successor_values"] is None
        assert result["decision"]["probabilities"] == pytest.approx([0.5, 0.5])
    attempts = [json.loads(p.read_text()) for p in (tmp_path / "attempts").glob("*.json")]
    assert sum(len(a["attempts"]) for a in attempts) == budget
    if budget == 1:
        assert not result["rows"]
        assert attempts[0]["partial_row"] is True


def test_source_is_freshly_initialized_at_exact_declared_region():
    contract = json.loads((ROOT / "configs/continuation_profile_v1.json").read_text())
    seeds = json.loads((ROOT / contract["source_manifest"]).read_text())
    node = initial_state(contract, seeds)
    assert node.context.locus == frozenset(contract["region_atoms"])
    assert node.remaining == 11
    assert node.option == "build_ring_system"
    assert node.graph.n_real_atoms == 19
    assert exact_graph_key(node.graph) == exact_graph_key(node.origin)
    payload = state_payload(node)
    restored = decode_state(payload["graph"])
    assert exact_graph_key(restored) == exact_graph_key(node.graph)
    with pytest.raises(ValueError, match="exactly one frozen region"):
        initial_state({**contract, "region_atoms": [999]}, seeds)


def test_artifact_round_trip_identity_and_mismatch_are_explicit(tmp_path):
    path = tmp_path / "receipt.json"
    digest = publish_json(path, {"b": 2, "a": [1]})
    assert verify_file(path, digest) == digest
    assert publish_json(path, {"a": [1], "b": 2}) == digest
    with pytest.raises(ValueError, match="input identity mismatch"):
        verify_file(path, "0" * 64)
    assert not path.with_suffix(".json.tmp").exists()


def test_profile_preserves_semantic_and_legacy_action_ontologies():
    from compose_v4.rewrite import action_codec, action_codec_v4
    from compose_v4.rewrite.operators import BondInsert, CycleCloseEdge, SemanticAtomRestate

    for rule, action, codec in (
        ("atom_restate_semantic", SemanticAtomRestate(0, 0), action_codec_v4),
        ("cycle_close", CycleCloseEdge(0, 4, 1), action_codec_v4),
        ("bond_insert", BondInsert(0, 4, 1), action_codec),
    ):
        assert encode_action(rule, action) == codec.encode_action(rule, action)


def test_profile_is_deterministic_except_timings(tmp_path):
    a, b = fixture_profile(tmp_path / "a"), fixture_profile(tmp_path / "b")
    a.pop("timings_seconds")
    b.pop("timings_seconds")
    assert a == b


def test_meter_covers_enumerator_internal_calls_and_restores_executor(tmp_path):
    node, kernel = engineering_fixture()
    original = RewriteSystem.apply

    def reference(graph):
        law = kernel.enumerate_law(graph)
        # Simulates a support check inside the production enumeration path.
        kernel.system.apply(graph, law[0][0], law[1][0])
        return law

    result = run_profile(
        node,
        reference,
        kernel.system,
        {"max_executor_applications": 1, "max_expansions": 2, "max_terminal_evaluations": 2},
        tmp_path,
        snapshot_id="fixture-internal-executor",
        commit_volume=lambda: None,
        progress={},
    )
    assert result["total_public_executor_calls"] == 1
    assert result["executor_calls_by_phase"] == {"law_enumeration": 1, "option_validation": 0}
    assert result["status"] == "root_executor_budget_exhausted"
    assert RewriteSystem.apply is original
    families, actions, _ = kernel.enumerate_law(node.graph)
    expected = kernel.system.apply(node.graph, families[0], actions[0])
    with ExecutorMeter(1).instrument() as meter:
        actual = kernel.system.apply(node.graph, families[0], actions[0])
        assert exact_graph_key(actual) == exact_graph_key(expected)
        with pytest.raises(ContinuationBudgetExceeded), meter.boundary():
            kernel.system.apply(node.graph, families[0], actions[0])
        assert meter.calls == 1
    assert RewriteSystem.apply is original


def _real_ring_restate():
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.rewrite.kernel import de_novo_rewrite_system
    from compose_v4.rewrite.tracelets import BondOrderChange, RingSystemRestate

    graph = pad_molecular_graph(smiles_to_molecular_graph("C1CCCCC1"), 8)
    action = RingSystemRestate(tuple(BondOrderChange(i, (i + 1) % 6, 2) for i in (0, 2, 4)))
    return de_novo_rewrite_system(), graph, action


@pytest.mark.parametrize("limit", [1, 2, 3, 4])
def test_budget_cancellation_escapes_real_ring_restate_lowering(limit):
    system, graph, action = _real_ring_restate()
    original = RewriteSystem.apply
    meter = ExecutorMeter(limit)
    with pytest.raises(ContinuationBudgetExceeded), meter.instrument():
        system.apply(graph, "ring_system_restate", action)
    assert RewriteSystem.apply is original
    assert meter.calls == limit == len(meter.attempts)
    assert sorted(a["call_index"] for a in meter.attempts) == list(range(limit))
    assert meter.attempts[-1]["status"] == "budget_interrupted"
    assert all(a["status"] in {"executed", "budget_interrupted"} for a in meter.attempts)


def test_nested_cancellation_abstains_and_persists_all_entered_calls(tmp_path):
    node, kernel = engineering_fixture()
    system, ring, action = _real_ring_restate()

    def reference(graph):
        if graph.n_real_atoms > 4:
            # Actual legacy lowering inside a law enumeration, as in the failed
            # production profile. It must not turn cancellation into UNSAT.
            system.apply(ring, "ring_system_restate", action)
        return kernel.enumerate_law(graph)

    result = run_profile(
        node,
        reference,
        kernel.system,
        {"max_executor_applications": 3, "max_expansions": 10, "max_terminal_evaluations": 10},
        tmp_path,
        snapshot_id="fixture-real-nested-ring-lowering",
        commit_volume=lambda: None,
        progress={},
    )
    assert result["status"] == "budget_abstention"
    assert result["decision"]["probabilities"] == pytest.approx([0.5, 0.5])
    assert result["decision"]["successor_values"] is None
    assert result["total_public_executor_calls"] == 3
    assert len(result["rows"]) == 1  # interrupted successor row was not published
    receipts = [json.loads(p.read_text()) for p in (tmp_path / "attempts").glob("*.json")]
    attempts = [a for receipt in receipts for a in receipt["attempts"]]
    assert sorted(a["call_index"] for a in attempts) == [0, 1, 2]
    assert sum(a["status"] == "budget_interrupted" for a in attempts) == 1


def test_meter_preserves_real_chemistry_and_propagates_unexpected_errors(monkeypatch):
    system, graph, action = _real_ring_restate()
    expected = system.apply(graph, "ring_system_restate", action)
    with ExecutorMeter(100).instrument() as meter:
        actual = system.apply(graph, "ring_system_restate", action)
    assert exact_graph_key(actual) == exact_graph_key(expected)
    assert meter.calls == len(meter.attempts) > 1
    assert all(a["status"] == "executed" for a in meter.attempts)

    def broken(*args):
        raise ValueError("genuine runtime error")

    monkeypatch.setattr(RewriteSystem, "apply", broken)
    meter = ExecutorMeter(2)
    with pytest.raises(ValueError, match="genuine runtime error"), meter.instrument():
        system.apply(graph, "ring_system_restate", action)
    assert meter.attempts[0]["status"] == "execution_error"


def test_profile_has_no_docking_training_or_fanout_launch_surface():
    source = ROOT / "modal_apps/genmol_t4_opt_app.py"
    tree = ast.parse(source.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "continuation_profile"
    )
    calls = {
        n.func.id
        for n in ast.walk(function)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert not calls & {"_dock", "_dock_many", "t4_population_cell", "optimize"}
    decorator = function.decorator_list[0]
    options = {kw.arg: kw.value for kw in decorator.keywords}
    assert ast.literal_eval(options["max_containers"]) == 1
    assert ast.literal_eval(options["retries"]) == 0
    assert ast.literal_eval(options["timeout"]) == 900
    assert "gpu" not in options
    contract = json.loads((ROOT / "configs/continuation_profile_v1.json").read_text())
    assert contract["oracle_calls"] == 0
    assert contract["winner_inputs"] == []
    assert contract["training_authorized"] is False
    assert np.isclose(
        contract["compute"]["maximum_configured_function_cost_usd"],
        900 * (0.0000131 + 6 * 0.00000222),
    )
