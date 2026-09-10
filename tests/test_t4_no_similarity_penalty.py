"""One-variable intermediate ranking ablation, with final gates unchanged."""

import json
from types import SimpleNamespace

import pytest

from compose_v4.experiments.t4_endpoint_selection import acceptable_endpoint
from compose_v4.experiments.t4_macro_beam import BeamConfig, no_similarity_desirability


def test_similarity_does_not_affect_ranking_but_still_blocks_final_eligibility():
    policy = {"qed_min": 0.6, "sa_max": 4.0, "scale": 0.1}
    model = {"best": -10.0, "scale": 1.0}
    base = {"smiles": "CC", "qed": 0.7, "sa": 3.0}
    low = {**base, "sim": 0.2, "v": 0.5}
    high = {**base, "sim": 0.5, "v": 0.0}
    assert no_similarity_desirability(low, -9, model, policy) == no_similarity_desirability(
        high, -9, model, policy
    )
    assert not acceptable_endpoint(low) and acceptable_endpoint(high)
    reference = no_similarity_desirability(base, -9, model, policy)
    for props in ({**base, "qed": 0.3}, {**base, "sa": 6.0}):
        assert 0 < no_similarity_desirability(props, -9, model, policy) < reference
    assert no_similarity_desirability(base, -9.5, model, policy) > reference
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            no_similarity_desirability({**base, "qed": bad}, -9, model, policy)
    with pytest.raises(ValueError):
        no_similarity_desirability(base, -9, model, {**policy, "sa_max": 0})
    assert BeamConfig(retention_score="no_similarity_desirability").arm == "guided"


def test_launcher_spawns_only_new_arm(monkeypatch, tmp_path):
    from modal_apps import run_process_v2_p50_app
    from tools import t4_launch

    calls = []

    def lookup(app, function):
        assert (app, function) == ("genmol-t4-opt", "t4_no_similarity_penalty")

        def spawn(task):
            calls.append(task)
            return SimpleNamespace(object_id="one-new-call")

        return SimpleNamespace(spawn=spawn)

    (tmp_path / "diagnostics").mkdir()
    monkeypatch.setattr(t4_launch, "ROOT", tmp_path)
    monkeypatch.setattr(t4_launch, "_sha256", lambda _: "b" * 64)
    monkeypatch.setattr(t4_launch.modal.Function, "from_name", lookup)
    monkeypatch.setattr(
        run_process_v2_p50_app,
        "local_image_revision",
        lambda **_: {"commit": "a" * 40, "image_revision_sha256": "c" * 64},
    )
    t4_launch.launch_continuation_profile({"commit": "a" * 40}, no_similarity_penalty=True)
    assert len(calls) == 1 and calls[0]["case_index"] == 0
    receipt = json.loads((tmp_path / "diagnostics/t4_no_similarity_penalty_spawn.json").read_text())
    assert receipt["oracle_calls"] == 0 and len(receipt["cases"]) == 1
    assert receipt["cases"][0]["volume_path"].startswith("/t4_no_similarity_penalty/case_0/")
