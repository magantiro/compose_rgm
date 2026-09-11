"""Bounded allocation and durable orchestration, without neural laws or docking."""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from compose_v4.experiments import t4_donor_probe as experiment
from compose_v4.experiments.continuation_profile import publish_json
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]


def test_registered_probe_and_canonical_allocation():
    c = experiment.load_contract(ROOT)
    assert len(c["initial_indices"]) * len(c["arms"]) == 32
    assert c["new_oracle_limit"] == 20
    data = json.loads((ROOT / experiment.PREPARED).read_text())
    parent = data["parents"][1]
    row = {**parent, "bundle": {"option": "generic", "bundle_id": "fixture"}}
    workers = [
        {"slot": 0, "proposal_policy_sha256": c["proposal_ids"][arm], "candidates": [row]}
        for arm in c["arms"]
    ]
    lock = experiment.candidate_lock(c, {**data, "prior_smiles": []}, workers)
    assert lock["selected_counts"] == {"baseline": 1, "hybrid": 1}
    assert len(lock["take"]) == 5  # one shared endpoint and four explicit repeats
    assert lock["take"][0]["arms"] == c["arms"]
    assert [r["arm"] for r in lock["take"][0]["origins"]] == c["arms"]
    assert all(r["oracle_eligible"] for r in lock["take"])
    observed = experiment.candidate_lock(c, data, workers)
    assert observed["selected_counts"] == {"baseline": 0, "hybrid": 0}
    assert len(observed["take"]) == 4


@pytest.mark.parametrize("malformed", [False, True])
def test_lock_precedes_docking_and_completed_resume_spends_nothing(
    tmp_path, monkeypatch, malformed
):
    c = experiment.load_contract(ROOT)
    data = json.loads((ROOT / experiment.PREPARED).read_text())
    publish_json(tmp_path / experiment.PREPARED, data)
    app = tmp_path / "modal_apps/genmol_t4_opt_app.py"
    app.parent.mkdir()
    app.write_text("# fixture only\n")
    store = Store(tmp_path / "output", lambda: None)

    @contextmanager
    def session(*args):
        yield c, store, {}

    monkeypatch.setattr(experiment, "session", session)
    calls, batches = [], []

    def parallel(tasks):
        batches.append(len(tasks))
        for task in reversed(tasks):
            if task.get("stage") != "dock":
                yield {
                    "worker_id": task["worker_id"],
                    "slot": task["slot"],
                    "replay_verified": True,
                    "proposal_policy_sha256": task["proposal_id"],
                    "candidates": [],  # no rescue proposals after low coverage
                }
            else:
                assert store.read("proposals") is not None
                lock = unseal(store.output / "candidate_lock.json")
                barrier = json.loads((store.output / "docking_started.json").read_text())
                assert barrier["candidate_lock_sha256"] == task["candidate_lock_sha256"]
                calls.append(task["index"])
                yield {
                    "index": -1 if malformed else task["index"],
                    "smiles": lock["take"][task["index"]]["smiles"],
                    "candidate_lock_sha256": task["candidate_lock_sha256"],
                    "ds": None,  # failed attempts remain charged
                }

    task = {"run_id": "fixture", "image_revision": {"commit": "fixture"}}
    if malformed:
        with pytest.raises(ValueError, match="docking row"):
            experiment.driver_remote(task, tmp_path, tmp_path, None, None, parallel)
        return
    result = experiment.driver_remote(task, tmp_path, tmp_path, None, None, parallel)
    assert batches == [32, 4]
    assert result["new_oracle_calls"] == len(calls) == 4
    assert result["selected_counts"] == {"baseline": 0, "hybrid": 0}
    repeated = experiment.driver_remote(task, tmp_path, tmp_path, None, None, parallel)
    assert repeated == result
    assert batches == [32, 4] and len(calls) == 4
