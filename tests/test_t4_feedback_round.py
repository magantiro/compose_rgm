"""Offline checks for the single 33-evaluation feedback episode."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from test_t4_warm_continuation import synthetic_prepare

from compose_v4.experiments.t4_feedback_round import feedback_archive
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_warm_continuation import run_rounds

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sources(tmp_path):
    contract = json.loads((ROOT / "configs/t4_feedback_round.json").read_text())
    files = {
        "warm": "diagnostics/t4_warm_continuation/attempt_2/warm_start.json",
        "lock": "diagnostics/t4_partial_docking/attempt_1/candidate_lock.json",
        "docking": "diagnostics/t4_partial_docking/attempt_1/docking.json",
    }
    for name, item in contract["source"].items():
        path = tmp_path / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        # Copy real sealed evidence into a fixture namespace, without parsing
        # and reserializing its provenance-bearing bytes.
        path.write_bytes((ROOT / files[name]).read_bytes())
    return tmp_path, contract


def test_feedback_uses_all_scores_and_exact_states_without_execution(sources, monkeypatch):
    from compose_v4.rewrite.kernel import RewriteSystem

    root, contract = sources
    monkeypatch.setattr(RewriteSystem, "apply", lambda *_: pytest.fail("source re-execution"))
    archive = feedback_archive(root, contract)
    assert archive == feedback_archive(root, contract)
    assert archive["oracle_attempts"] == 33 and len(archive["archive"]) == 34
    warm = unseal(root / contract["source"]["warm"]["path"])
    assert archive["archive"][:21] == warm["archive"]
    locked = unseal(root / contract["source"]["lock"]["path"])
    docked = unseal(root / contract["source"]["docking"]["path"])
    for saved, candidate, result in zip(
        archive["archive"][21:], locked["take"], docked["docked"], strict=True
    ):
        assert saved["state"] == candidate["state"]
        assert saved["ds"] == result["ds"]
        assert saved["ancestors"][-1] == candidate["parent"]
    assert archive["rng_state"] != warm["rng_state"]
    origin = archive["feedback_origin"]
    assert origin["event_2"].startswith("partial_seven_parent")
    assert archive["rng_state"] == np.random.default_rng(origin["derived_seed"]).bit_generator.state


def test_feedback_source_hash_fails_closed(sources):
    root, contract = sources
    contract = copy.deepcopy(contract)
    contract["source"]["docking"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="input identity"):
        feedback_archive(root, contract)


def test_shared_runner_can_stop_after_one_locked_round(tmp_path):
    from test_t4_warm_continuation import fixture_source

    from compose_v4.experiments.t4_warm_continuation import initial_archive

    source, _, task = fixture_source(tmp_path)
    warm = initial_archive(unseal(source / "candidate_lock.json"), unseal(source / "docking.json"))
    calls = []

    def dock(smiles, rd):
        assert (tmp_path / "output" / rd / "candidate_lock.json").exists()
        calls.append((smiles, rd))
        return [-9.0] * len(smiles)

    result = run_rounds(
        task,
        warm,
        synthetic_prepare,
        dock,
        tmp_path / "output",
        additional_rounds=1,
        endpoint_policy="legacy_rank_all_v1",
        source_identity={},
    )
    assert len(calls) == 1 and len(result["rounds"]) == 1
    assert result["new_oracle_attempts"] == 1
    assert not (tmp_path / "output/round_3").exists()
