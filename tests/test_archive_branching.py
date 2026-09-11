import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from compose_v4.control.archive_allocation import sample_archive_parents
from compose_v4.experiments import pmo_option_particles as experiment
from compose_v4.experiments.continuation_profile import publish_json
from compose_v4.experiments.pmo_archive_branching import load_contract
from compose_v4.experiments.pmo_archive_pilot import Store


def test_archive_retains_best_and_gives_every_molecule_exploration():
    archive = {s: {"smiles": s, "score": v} for s, v in (("C", 0.8), ("CC", 0.1), ("CCC", 0.5))}
    left, audit = sample_archive_parents(archive, 16, np.random.default_rng(12))
    right, repeated = sample_archive_parents(
        dict(reversed(list(archive.items()))), 16, np.random.default_rng(12)
    )
    assert left == right and audit == repeated
    assert left[0]["smiles"] == "C"
    assert sum(audit["probabilities"]) == pytest.approx(1)
    assert min(audit["probabilities"]) >= 0.2 / len(archive)
    with pytest.raises(ValueError):
        sample_archive_parents({"O": archive["C"]}, 2, np.random.default_rng(0))


def test_archive_branching_reuses_success_without_cross_arm_imports(tmp_path, monkeypatch):
    parent = {"id": "root", "smiles": "CCC", "score": 0.9, "node": {}, "chain": [], "primitives": 0}
    publish_json(tmp_path / experiment.PREPARED, {"parents": [parent], "observed": {"CCC": 0.9}})
    c = {
        "particles": 2,
        "boundaries": 2,
        "initial_indices": [0, 0],
        "task": "fixture",
        "new_oracle_limit": 8,
        "seed": 123,
        "beta": 10,
        "arms": ["smc", "archive"],
        "parent_selection_modes": {"smc": "smc", "archive": "archive"},
        "selection_modes": {"smc": "immediate"},
        "archive_exploration": 0.2,
    }
    store = Store(tmp_path / "result", lambda: None)

    @contextmanager
    def session(*args):
        yield c, store, {}

    monkeypatch.setattr(experiment, "session", session)
    calls = []

    def oracle(s):
        calls.append(s)
        return 0.95 if s == "CCCC" else 0.1

    monkeypatch.setattr(experiment, "make_oracle", lambda *args: oracle)
    batches = []

    def parallel(tasks):
        batches.append(tasks)
        for t in reversed(tasks):
            p = t["parent"]
            s = "CC" if t["phase"] == 1 else "CCCC" if p["smiles"] == "CCC" else "C"
            child = {
                **p,
                "id": t["worker_id"],
                "smiles": s,
                "chain": p["chain"] + [t["worker_id"]],
                "primitives": p["primitives"] + 1,
                "primitive_count": 1,
                "bundle": {"option": "generic"},
            }
            yield {
                "worker_id": t["worker_id"],
                "replay_verified": True,
                "proposal_policy_sha256": None,
                "candidates": [child],
                "attempts": [{"status": "complete"}],
            }

    task = {"run_id": "fixture", "image_revision": {"commit": "fixture"}}
    result = experiment.driver_remote(task, tmp_path, tmp_path, None, None, parallel)
    assert len(batches[0]) == 2  # shared identical initial proposals
    assert result["arms"]["smc"]["best"] == 0.9
    assert result["arms"]["archive"]["best"] == 0.95
    assert "CCCC" not in result["archives"]["smc"]
    assert result["new_oracle_calls"] == len(calls) == len(set(calls))
    again = experiment.driver_remote(task, tmp_path, tmp_path, None, None, parallel)
    assert again == result and len(batches) == 2
    assert json.loads(json.dumps(result)) == result


def test_registered_champion_comparison_has_matched_proposals():
    c = load_contract(Path(__file__).resolve().parents[1])
    assert c["proposal_ids"]["smc"] == c["proposal_ids"]["archive"]
    assert c["compute"]["max_workers"] + c["compute"]["driver_containers"] == 30
