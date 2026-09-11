"""No-network orchestration/accounting check for the bounded proposal audit."""

import json
from contextlib import contextmanager

from compose_v4.experiments import pmo_region_replacement as experiment
from compose_v4.experiments.pmo_archive_pilot import Store


def test_candidate_lock_single_scoring_and_durable_resume(tmp_path, monkeypatch):
    parent = {"id": "root", "smiles": "C", "score": 0.1, "node": {}, "chain": [], "primitives": 0}
    prepared = tmp_path / experiment.PREPARED
    prepared.parent.mkdir(parents=True)
    prepared.write_text(json.dumps({"parents": [parent], "observed": {"C": 0.1}}))
    store = Store(tmp_path / "output", lambda: None)
    c = {"initial_indices": [0] * 20, "task": "perindopril_mpo", "new_oracle_limit": 20}

    @contextmanager
    def session(*args):
        yield c, store, {}

    calls = []

    def oracle(smiles):
        assert store.read("candidate_lock") is not None
        calls.append(smiles)
        return 0.2

    monkeypatch.setattr(experiment, "session", session)
    monkeypatch.setattr(experiment, "make_oracle", lambda *args: oracle)

    def parallel(tasks):
        for task in reversed(tasks):
            yield {
                "worker_id": task["worker_id"],
                "replay_verified": True,
                "attempts": [{"bundle": {"option": "generic"}}],
                "candidates": [{**parent, "smiles": "CC"}] if task["slot"] else [],
            }

    task = {"run_id": "fixture", "image_revision": {"commit": "fixture"}}
    result = experiment.driver_remote(task, tmp_path, tmp_path, None, None, parallel)
    assert len(result["workers"]) == 20 and len(result["candidates"]) == 19
    assert calls == ["CC"] and result["new_oracle_calls"] == 1
    assert result["archive_metrics"]["best"] == 0.2
    assert result["attempted_options"] == {"generic": 20}
    assert experiment.driver_remote(task, tmp_path, tmp_path, None, None, parallel) == result
    assert calls == ["CC"]
