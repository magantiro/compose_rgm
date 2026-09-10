"""Small actual-executor checks; no learned checkpoint, docking, or network."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_carbonyl_option import initial, kernel

from compose_v4.control.molecular_task_search import MolecularHierarchy, MolecularSearchState
from compose_v4.experiments.continuation_profile import ExecutorMeter
from compose_v4.experiments.t4_macro_beam import (
    BeamConfig,
    exact_archive_graph,
    replay,
    retain,
    run_search,
)
from compose_v4.rewrite.trace_shard import encode_state


def test_archive_alias_matches_without_reconstructing_exact_slots():
    graph = initial("CCC").graph
    row = {"state": encode_state(graph), "smiles": "C(C)C"}
    assert encode_state(exact_archive_graph(row)) == row["state"]
    with pytest.raises(ValueError, match="differs from canonical metadata"):
        exact_archive_graph({**row, "smiles": "CC"})
    with pytest.raises(ValueError, match="invalid SMILES"):
        exact_archive_graph({**row, "smiles": "("})


def forbidden_score(smiles):
    raise AssertionError("post-hoc search called task value during generation")


def test_retention_deduplicates_preserves_floor_and_isolates_post_hoc():
    pool = [
        {"attempt_id": str(i), "smiles": s}
        for i, s in enumerate(("C", "CC", "CCC", "CCCC", "CCCC"))
    ]
    _, baseline = retain(
        pool, BeamConfig(arm="post_hoc"), np.random.default_rng(1), forbidden_score
    )
    assert len(baseline["pool"]) == 4
    assert baseline["first_slot_probabilities"] == baseline["first_slot_reference"]
    chosen, guided = retain(
        pool, BeamConfig(), np.random.default_rng(1), lambda s: {"desirability": float(s == "CCCC")}
    )
    p, q = (
        np.asarray(guided["first_slot_reference"]),
        np.asarray(guided["first_slot_probabilities"]),
    )
    assert q[-1] > p[-1] and np.all(q >= 0.1 * p)
    assert guided["first_slot_kl_against_empirical_pool"] <= 1 + 1e-10
    assert len({r["smiles"] for r in chosen}) == 3
    _, identical = retain(
        pool, BeamConfig(), np.random.default_rng(1), lambda s: {"desirability": 0.5}
    )
    assert np.allclose(identical["first_slot_probabilities"], p)


def execute(records, *, stop_after=None, config=None):
    process = kernel()
    hierarchy = MolecularHierarchy(process, lazy_applicability=True, include_carbonyl_options=True)
    config = config or BeamConfig(arm="post_hoc", depth=2, seed=2011)
    root = MolecularSearchState.start(initial("CCC").graph, budget=44, root_id="fixture")
    saves = 0

    def save(name, value):
        nonlocal saves
        records[name] = json.loads(json.dumps(value))
        saves += 1
        if stop_after == saves:
            raise InterruptedError("checkpoint interruption")

    meter = ExecutorMeter(None)
    with meter.instrument():
        result = run_search(
            root,
            hierarchy,
            config=config,
            score=forbidden_score,
            save=save,
            read=records.get,
            meter=meter,
            progress={},
        )
    return result, meter.calls


def test_complete_options_only_and_interruption_reuses_rng_and_paths():
    records = {}
    full, _ = execute(records)
    assert full["oracle_calls"] == 0 and full["winner_used"] is False
    assert all(
        a["node"]["stage"] == "where" and a["replayed_primitives"] > 0
        for a in full["attempts"]
        if a["status"] == "complete"
    )
    assert all(len(r["chain"]) == 2 for r in full["final_beam"])
    unchanged, calls = execute(records)
    assert unchanged == records["generation_lock"] and calls == 0
    interrupted = {}
    with pytest.raises(InterruptedError):
        execute(interrupted, stop_after=5)
    resumed, _ = execute(interrupted)
    for key in full:
        if key != "proposal_seconds_this_invocation":
            assert json.loads(json.dumps(full[key])) == resumed[key]
    attempt = next(a for a in full["attempts"] if a["status"] == "complete")
    events = copy.deepcopy(attempt["events"])
    event = next(e for e in events if "mark" in e)
    event["product"]["graph"] = event["source"]["graph"]
    with pytest.raises(ValueError, match="replay differs"):
        replay(events, kernel().system)


def test_launcher_spawns_only_four_macro_beam_workers(monkeypatch, tmp_path):
    from modal_apps import run_process_v2_p50_app
    from tools import t4_launch

    calls = []

    def lookup(app, function):
        assert (app, function) == ("genmol-t4-opt", "t4_macro_beam")

        def spawn(task):
            calls.append(task)
            return SimpleNamespace(object_id=f"call-{task['case_index']}")

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
    t4_launch.launch_continuation_profile({"commit": "a" * 40}, macro_beam=True)
    assert [c["case_index"] for c in calls] == [0, 1, 2, 3]
    receipt = json.loads((tmp_path / "diagnostics/t4_macro_beam_spawn.json").read_text())
    assert receipt["oracle_calls"] == 0
    calls.clear()
    t4_launch.launch_continuation_profile(
        {"commit": "a" * 40}, macro_beam=True, macro_beam_cases=[2, 0, 2]
    )
    assert [c["case_index"] for c in calls] == [0, 2]


def test_invalid_search_sizes_fail():
    for fields in ({"width": 4}, {"depth": 5}, {"primitive_budget": 16}, {"arm": "winner"}):
        with pytest.raises(ValueError):
            BeamConfig(**fields)
