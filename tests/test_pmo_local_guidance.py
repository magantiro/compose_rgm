import json
from contextlib import contextmanager
from time import perf_counter

import numpy as np
from test_donor_memory import donor, node

from compose_v4.control.donor_memory import proposal_identity
from compose_v4.control.local_endpoint_selector import active, policy_identity, predict, select
from compose_v4.experiments import pmo_local_guidance as experiment
from compose_v4.experiments import pmo_option_particles as driver
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import AtomDelete
from tools.pmo_cross_parent_selection import predict_frozen


def test_local_channel_never_replaces_donor_and_has_identified_policy():
    observed = set()
    for slot in range(100):
        coin = np.random.default_rng(np.random.SeedSequence([20261008, 1, slot, 776])).random()
        chosen = active(20261008, 1, slot)
        assert not chosen or coin >= 0.5
        observed.add("local" if chosen else "donor" if coin < 0.5 else "reference")
    assert observed == {"local", "donor", "reference"}
    assert policy_identity("memory", "model") != proposal_identity("memory")
    assert policy_identity("memory", "model") != policy_identity("memory", "different")


def test_production_prediction_matches_tested_frozen_model():
    from compose_v4.control.docking_value import identity

    model = {"train_indices": [0, 1], "mean": 0.4, "coefficients": [0.2, -0.1]}
    lock = {
        "model": {**model, "snapshot_sha256": identity(model)},
        "all_smiles": ["CC", "CCC"],
        "training_smiles": ["CC", "CCC"],
    }
    candidates = ["CCO", "CCN", "CCCC"]
    np.testing.assert_array_equal(predict(lock, candidates), predict_frozen(lock, candidates))
    for seed in range(10):
        assert select([str(i) for i in range(20)], np.arange(20), np.random.default_rng(seed)) in {
            str(i) for i in range(4, 20)
        }


def test_actual_local_worker_deduplicates_aliases_and_replays(tmp_path, monkeypatch):
    parent = node("CC", 0.2)
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({"observed": {"CC": 0.2}, "endpoint_model": {}}))
    store = Store(tmp_path / "worker", lambda: None)
    task = {
        "parent": parent,
        "worker_id": "fixture",
        "local_exclusions": ["CC"],
        "phase": 1,
        "slot": 0,
        "proposal_id": "policy",
        "image_revision": {"commit": "test"},
    }
    c = {"seed": 20261008, "prepared": {"path": "prepared.json"}, "endpoint_model_sha256": "model"}

    def law(graph):
        return ("atom_delete", "atom_delete"), (AtomDelete(0), AtomDelete(1)), (0.5, 0.5)

    law.counts = {"fresh_laws": 0}
    seen = []

    def prediction(model, pool):
        seen.append(pool)
        return np.array([0.7])

    monkeypatch.setattr(experiment, "predict", prediction)
    runtime = {"system": editing_v2_semantic_rewrite_system()}
    result = experiment.local_worker(task, tmp_path, c, store, {}, runtime, law, perf_counter(), 0)
    assert seen == [["C"]]
    assert result["candidates"][0]["smiles"] == "C"
    assert result["candidates"][0]["structural_change"]["n_deleted"] == 1
    assert len(store.read("local_generation")["products"][0]["witnesses"]) == 2
    assert result["oracle_calls"] == 0 and result["replay_verified"]


def test_driver_keeps_baseline_and_local_tasks_separate_and_locks_round(tmp_path, monkeypatch):
    parent = node("CC", 0.2)
    seed = next(s for s in range(100) if active(s, 1, 0) and not active(s, 1, 1))
    (tmp_path / "prepared.json").write_text(
        json.dumps(
            {"parents": [parent], "observed": {"CC": 0.2}, "initial_donor_memory": [donor(parent)]}
        )
    )
    c = {
        "seed": seed,
        "particles": 2,
        "boundaries": 1,
        "initial_indices": [0, 0],
        "arms": ["baseline", "guided"],
        "prepared": {"path": "prepared.json"},
        "donor_memory_modes": {"baseline": "fixed", "guided": "fixed"},
        "parent_selection_modes": {"baseline": "archive", "guided": "archive"},
        "archive_exploration": 0.2,
        "local_selector_arms": ["guided"],
        "endpoint_model_sha256": "model",
        "task": "perindopril_mpo",
        "new_oracle_limit": 3,
    }
    store = Store(tmp_path / "out", lambda: None)

    @contextmanager
    def session(*args):
        yield c, store, {}

    monkeypatch.setattr(
        driver, "make_oracle", lambda *args: lambda s: {"CCC": 0.4, "CCO": 0.6, "CCN": 0.3}[s]
    )

    def parallel(tasks):
        assert len(tasks) == 3
        assert sum(t.get("local_selector", False) for t in tasks) == 1
        for t in tasks:
            local = t.get("local_selector", False)
            if local:
                assert t["local_exclusions"] == ["CC"]
                assert t["proposal_id"] == policy_identity(t["donor_memory_id"], "model")
            else:
                assert t["proposal_id"] == proposal_identity(t["donor_memory_id"])
            child = {
                **node("CCO" if local else "CCN" if t["slot"] == 1 else "CCC", 0),
                "bundle": {"option": "local_endpoint_selector" if local else "generic"},
            }
            yield {
                "worker_id": t["worker_id"],
                "replay_verified": True,
                "proposal_policy_sha256": t["proposal_id"],
                "attempts": [{}],
                "candidates": [child],
            }

    result = driver.driver_remote(
        {"run_id": "fixture", "image_revision": {}},
        tmp_path,
        tmp_path,
        None,
        None,
        parallel,
        run_session=session,
    )
    assert result["new_oracle_calls"] == 3
    assert result["arms"]["guided"]["best"] == 0.6
    assert "CCO" not in result["archives"]["baseline"]
    assert "CCC" not in result["archives"]["guided"]
    assert "CCN" in result["archives"]["guided"] and "CCN" in result["archives"]["baseline"]
