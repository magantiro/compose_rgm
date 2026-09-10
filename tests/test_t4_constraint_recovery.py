"""Focused regression checks for intermediate ranking and fixed incumbent reuse."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_t4_macro_beam import execute, forbidden_score

from compose_v4.experiments.t4_macro_beam import BeamConfig, recovery_desirability, retain


def test_recovery_is_graded_without_changing_terminal_eligibility():
    payload = {"best": -10.0, "scale": 1.0}
    scores = [recovery_desirability(v, -9.0, payload, 0.1) for v in (0, 0.01, 0.2)]
    assert 0 < scores[2] < scores[1] < scores[0] < 1
    pool = [{"attempt_id": str(i), "smiles": s} for i, s in enumerate(("C", "CC", "CCC"))]
    incumbent = {"attempt_id": "incumbent", "smiles": "CCCC", "node": "exact", "chain": []}
    lookup = {r["smiles"]: v for r, v in zip(pool, scores)}

    def score(smiles):
        return {"desirability": 0.0, "recovery_desirability": lookup[smiles]}

    config = BeamConfig(preserve_root=True)
    _, hard = retain(pool, config, np.random.default_rng(2), score, incumbent=incumbent)
    _, soft = retain(
        pool,
        BeamConfig(preserve_root=True, retention_score="recovery_desirability"),
        np.random.default_rng(2),
        score,
        incumbent=incumbent,
    )
    assert (
        hard["offspring"]["first_slot_probabilities"] == hard["offspring"]["first_slot_reference"]
    )
    q = soft["offspring"]["first_slot_probabilities"]
    assert q[0] > q[1] > q[2] > 0
    assert soft["offspring"]["first_slot_kl_against_empirical_pool"] <= 1
    for bad in (float("nan"), -1.0):
        with pytest.raises(ValueError):
            recovery_desirability(bad, -9.0, payload, 0.1)
    with pytest.raises(ValueError):
        recovery_desirability(0.1, -9.0, payload, 0.0)


def test_incumbent_is_exact_unique_and_does_not_require_post_hoc_scoring():
    incumbent = {"attempt_id": "incumbent", "smiles": "C", "node": "exact", "chain": []}
    pool = [{"attempt_id": str(i), "smiles": s} for i, s in enumerate(("C", "CC", "CCC", "CCCC"))]
    kept, _ = retain(
        pool,
        BeamConfig(arm="post_hoc", preserve_root=True),
        np.random.default_rng(2),
        forbidden_score,
        incumbent=incumbent,
    )
    assert kept[0] is incumbent and len(kept) == 3
    assert len({r["smiles"] for r in kept}) == 3
    with pytest.raises(ValueError, match="exact root"):
        retain(pool, BeamConfig(preserve_root=True), np.random.default_rng(2), forbidden_score)


def test_incumbent_search_replays_and_resumes_without_changing_paths():
    config = BeamConfig(arm="post_hoc", depth=2, seed=2011, preserve_root=True)
    full, _ = execute({}, config=config)
    for level in full["levels"]:
        assert level["beam"][0]["node"] == full["root"]
        assert level["beam"][0]["chain"] == []
    second = [a for a in full["attempts"] if a["attempt_id"].startswith("levels/01/attempt_00")]
    assert len(second) == 3 and all(a["source"] == full["root"] for a in second)
    interrupted = {}
    with pytest.raises(InterruptedError):
        execute(interrupted, config=config, stop_after=5)
    resumed, _ = execute(interrupted, config=config)
    for key in full:
        if key != "proposal_seconds_this_invocation":
            assert full[key] == resumed[key]


def test_recovery_launcher_spawns_three_zero_oracle_workers(monkeypatch, tmp_path):
    from modal_apps import run_process_v2_p50_app
    from tools import t4_launch

    calls = []

    def lookup(app, function):
        assert (app, function) == ("genmol-t4-opt", "t4_constraint_recovery")

        def spawn(task):
            calls.append(task)
            return SimpleNamespace(object_id=f"call-{task['case_index']}")

        return SimpleNamespace(spawn=spawn)

    (tmp_path / "diagnostics").mkdir()
    monkeypatch.setattr(t4_launch, "ROOT", tmp_path)
    monkeypatch.setattr(t4_launch, "_sha256", lambda _: "b" * 64)
    monkeypatch.setattr(t4_launch.modal.Function, "from_name", lookup)
    monkeypatch.setattr(
        run_process_v2_p50_app,
        "local_image_revision",
        lambda **_: {
            "commit": "a" * 40,
            "image_revision_sha256": "c" * 64,
        },
    )
    t4_launch.launch_continuation_profile({"commit": "a" * 40}, constraint_recovery=True)
    assert [c["case_index"] for c in calls] == [0, 1, 2]
    receipt = json.loads((tmp_path / "diagnostics/t4_constraint_recovery_spawn.json").read_text())
    assert receipt["oracle_calls"] == 0
