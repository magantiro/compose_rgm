import json
from pathlib import Path

import pytest
from test_carbonyl_option import initial, kernel

from compose_v4.experiments import option_decision_audit as audit
from compose_v4.rewrite.trace_shard import decode_state, encode_state


def test_lock_is_target_free_complete_and_restart_reuses_it(monkeypatch):
    monkeypatch.setattr(audit, "applicable_options", lambda *a, **k: ("generic", "add_carbonyl"))
    records, order = {}, []

    def save(name, payload):
        records[name] = json.loads(json.dumps(payload))
        order.append(name)

    process, graph = kernel(), initial("CCC").graph
    lock = audit.generate_lock(
        graph, "add_carbonyl", process, seed=1000, save=save, read=records.get, progress={}
    )
    assert lock["winner_or_task_value_used"] is False and lock["oracle_calls"] == 0
    assert len(lock["attempts"]) == 4 and order[-1] == "proposal_lock"
    assert all(a["status"] in ("complete", "support_dead_end") for a in lock["attempts"])
    before = process.work.executor_applications
    assert (
        audit.generate_lock(
            graph, "add_carbonyl", process, seed=1000, save=save, read=records.get, progress={}
        )
        == lock
    )
    assert process.work.executor_applications == before


def test_known_prefix_is_separate_and_requires_exact_source():
    path = Path(__file__).resolve().parents[1] / "diagnostics/t4_whole_ring_plan/result.json"
    if not path.exists():
        pytest.skip("documented whole-ring development asset absent")
    stage = json.loads(path.read_text())["attempts"][0]["stages"][-1]
    source = decode_state(stage["states"][0])
    result = audit.witness_support(source, "insert_ring_carbonyl", stage["states"], kernel())
    assert result["supported"] and 0 < result["exact_path_probability"] <= 1
    assert len(result["steps"]) == 4
    with pytest.raises(ValueError, match="exact source"):
        audit.witness_support(initial().graph, "insert_ring_carbonyl", stage["states"], kernel())
    changed = list(stage["states"])
    changed[1] = encode_state(source)
    failure = audit.witness_support(source, "insert_ring_carbonyl", changed, kernel())
    assert not failure["supported"] and failure["exact_path_probability"] == 0


def test_launcher_spawns_exactly_four_zero_oracle_cases(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from modal_apps import run_process_v2_p50_app
    from tools import t4_launch

    calls = []

    def spawn(task):
        calls.append(task)
        return SimpleNamespace(object_id=f"call-{task['case_index']}")

    def from_name(app, function):
        assert (app, function) == ("genmol-t4-opt", "t4_option_decision_audit")
        return SimpleNamespace(spawn=spawn)

    (tmp_path / "diagnostics").mkdir()
    monkeypatch.setattr(t4_launch, "ROOT", tmp_path)
    monkeypatch.setattr(t4_launch, "_sha256", lambda path: "b" * 64)
    monkeypatch.setattr(t4_launch.modal.Function, "from_name", from_name)
    monkeypatch.setattr(
        run_process_v2_p50_app,
        "local_image_revision",
        lambda **kw: {"commit": "a" * 40, "image_revision_sha256": "c" * 64},
    )
    t4_launch.launch_continuation_profile({"commit": "a" * 40}, option_decision=True)
    assert [c["case_index"] for c in calls] == list(range(4))
    receipt = json.loads((tmp_path / "diagnostics/t4_option_decision_spawn.json").read_text())
    assert receipt["oracle_calls"] == 0 and len(receipt["cases"]) == 4
