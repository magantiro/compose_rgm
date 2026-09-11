"""Cheap fake-chemistry check of orchestration/accounting, not model qualification."""

import json
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from compose_v4.control.winner_imitation import ImitationRanker
from compose_v4.experiments import pmo_option_particles as experiment
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.pmo_archive_pilot import Store


@pytest.mark.parametrize(
    "arms", [experiment.ARMS, ("reference", "immediate"), ("baseline", "learned")]
)
def test_shared_work_retains_particles_all_scores_and_resume(tmp_path, monkeypatch, arms):
    output = tmp_path / "output"
    model_path = tmp_path / "value.pt"
    torch.save({"model": ImitationRanker(518, hidden=128).state_dict()}, model_path)
    parent = {"id": "root", "smiles": "C", "score": 0.1, "node": {}, "chain": [], "primitives": 0}
    prepared = tmp_path / experiment.PREPARED
    prepared.parent.mkdir(parents=True)
    prepared.write_text(json.dumps({"parents": [parent], "observed": {"C": 0.1}}))
    c = {
        "particles": 4,
        "boundaries": 2,
        "initial_indices": [0] * 4,
        "value_checkpoint": {"path": "value.pt", "sha256": sha256_file(model_path)},
        "task": "perindopril_mpo",
        "new_oracle_limit": 24,
        "seed": 1,
        "beta": 10,
        "arms": list(arms),
    }
    if "future" not in arms:
        del c["value_checkpoint"]  # Actual-score execution must not load a head.
    if "learned" in arms:
        c["proposal_ids"] = {"baseline": None, "learned": "fixture_policy"}
        c["selection_modes"] = {"baseline": "immediate", "learned": "immediate"}
    store = Store(output, lambda: None)

    @contextmanager
    def session(*args):
        yield c, store, {}

    calls = []

    def oracle(smiles):
        calls.append(smiles)
        return 0.9 if len(smiles) >= 5 else 0.1

    monkeypatch.setattr(experiment, "session", session)
    monkeypatch.setattr(experiment, "make_oracle", lambda *args: oracle)
    monkeypatch.setattr(
        "compose_v4.control.trajectory_value.predict",
        lambda model, smiles, budget: np.full(len(smiles), 0.5),
    )
    task_censuses = []

    def parallel(tasks):
        task_censuses.append(tasks)
        for t in reversed(tasks):  # deliberately unordered worker completion
            smiles = "C" * (t["slot"] + t["phase"] + 1)
            candidate = {
                **parent,
                "id": t["worker_id"],
                "smiles": smiles,
                "bundle": {"option": "generic"},
                "node": {"key": smiles},
            }
            yield {
                "worker_id": t["worker_id"],
                "replay_verified": True,
                "attempts": [{"status": "complete"}],
                "candidates": [candidate],
                "proposal_policy_sha256": t.get("proposal_id"),
            }

    task = {"run_id": "test", "image_revision": {"commit": "fixture"}}
    result = experiment.driver_remote(task, tmp_path, tmp_path, None, None, parallel)
    assert len(task_censuses[0]) == (8 if "learned" in arms else 4)
    first = result["rounds"][0]["arms"]
    guided = "learned" if "learned" in arms else "immediate"
    assert first[guided]["resampled"]
    if "reference" in arms:
        assert not first["reference"]["resampled"]
    else:
        assert first["baseline"]["indices"] == first["learned"]["indices"]
    assert first[guided]["indices"] == [3] * 4
    assert len(first[guided]["log_weights"]) == 4
    assert len(calls) == len(set(calls)) == result["new_oracle_calls"]
    assert calls == ["CC", "CCC", "CCCC", "CCCCC", "CCCCCC"]
    assert set(result["arms"]) == set(arms)
    for arm in arms:
        assert set(result["archives"][arm]) == {"C", *calls}
        assert result["arms"][arm]["best"] == 0.9
    # Completed runs return their sealed artifact without new work or queries.
    repeated = experiment.driver_remote(task, tmp_path, tmp_path, None, None, parallel)
    assert repeated == result
    assert len(task_censuses) == 2
