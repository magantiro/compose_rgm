import copy

import numpy as np
import pytest

from compose_v4.control.branch_policy import BranchPolicy
from compose_v4.experiments.pmo_branch_policy import candidate_pool, worker_identity


def test_frozen_save_accepts_json_equivalence_but_rejects_changed_values(tmp_path):
    from compose_v4.experiments.pmo_archive_pilot import Store
    from compose_v4.experiments.pmo_branch_policy import _frozen_save

    store = Store(tmp_path, lambda: None)
    task = {"parent": {"source_cut": {"component": (2, 4, 8)}}, "seed": 71}
    _frozen_save(store, "identity", task)
    assert store.read("identity") != task  # JSON round-trip changes tuple to list.
    _frozen_save(store, "identity", task)
    changed = copy.deepcopy(task)
    changed["parent"]["source_cut"]["component"] = (2, 4, 9)
    with pytest.raises(ValueError, match="restart changed locked identity"):
        _frozen_save(store, "identity", changed)
    assert store.read("identity")["parent"]["source_cut"]["component"] == [2, 4, 8]


def test_launch_checks_frozen_process_before_remote_allocation(monkeypatch):
    from pathlib import Path

    from compose_v4.experiments.pmo_branch_policy import load_contract
    from compose_v4.rewrite import editing_v2_process_identity as process

    root = Path(__file__).resolve().parents[1]
    assert load_contract(root)["new_oracle_limit"] == 32
    monkeypatch.setattr(
        process, "editing_process_v2_identity", lambda: {"process_identity_sha256": "changed"}
    )
    with pytest.raises(process.EditingV2ProcessIdentityError, match="differs"):
        load_contract(root)


def test_policy_learns_order_and_does_not_read_evaluation_scores():
    observations = [
        {"parent_smiles": "CCO", "parent_score": 0.4, "smiles": "CCN", "score": 0.1},
        {"parent_smiles": "CCO", "parent_score": 0.4, "smiles": "CCC", "score": 0.8},
    ]
    policy = BranchPolicy.fit(observations, source_sha256="a" * 64)
    model_hash = policy.payload["model_sha256"]
    assert policy.utilities(observations)[1] > policy.utilities(observations)[0]
    q, audit = policy.distribution(observations, [0.5, 0.5], guided=True)
    assert q[1] > q[0]
    assert q.sum() == pytest.approx(1)
    assert np.all(q >= 0.05)
    assert audit["kl"] <= 1
    changed = copy.deepcopy(observations)
    changed[0]["score"], changed[1]["score"] = 1.0, 0.0
    assert np.array_equal(policy.utilities(observations), policy.utilities(changed))
    assert policy.payload["model_sha256"] == model_hash
    assert np.array_equal(
        BranchPolicy(copy.deepcopy(policy.payload)).utilities(observations),
        policy.utilities(observations),
    )


def test_pool_aggregates_draws_and_only_excludes_already_observed_queries():
    candidates = [
        {"id": "a", "smiles": "CCO"},
        {"id": "b", "smiles": "CCO"},
        {"id": "c", "smiles": "CCN"},
        {"id": "d", "smiles": "CCC"},
    ]
    pool, reference, excluded = candidate_pool(candidates, {"CCC": 0.5})
    by_smiles = {r["smiles"]: p for r, p in zip(pool, reference, strict=True)}
    assert by_smiles == {"CCN": pytest.approx(1 / 3), "CCO": pytest.approx(2 / 3)}
    assert excluded == [{"id": "d", "reason": "already_in_arm_scored_archive"}]
    assert worker_identity(1, 0, {"node": "a"}) != worker_identity(1, 0, {"node": "b"})
    assert candidate_pool([], {}) == ([], [], [])


@pytest.mark.parametrize("online", [False, True])
@pytest.mark.parametrize("empty_calibration", [False, True])
def test_deployed_loop_locks_before_scoring_freezes_fit_and_resumes_without_charges(
    tmp_path, monkeypatch, online, empty_calibration
):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from compose_v4.control.docking_value import identity
    from compose_v4.experiments import pmo_branch_policy as run
    from compose_v4.experiments.continuation_profile import publish_json
    from compose_v4.experiments.pmo_archive_pilot import Store

    # Synthetic workflow fixture. The production score ledger/driver are real;
    # proposal chemistry and policy fitting are replaced by deterministic stubs.
    root = {"id": "root", "smiles": "C", "score": 0.01, "node": {}, "chain": [], "primitives": 0}
    data = {
        "historical_calls": 2,
        "observed": {"C": 0.01, "CC": 0.02},
        "archive": [root],
        "observations": [
            {"parent_smiles": "C", "parent_score": 0.01, "smiles": "CC", "score": 0.02}
        ],
        "calibration_parents": [] if empty_calibration else [root] * 4,
    }
    publish_json(tmp_path / "prepared.json", data)
    contract = {
        "prepared": {"path": "prepared.json"},
        "task": "perindopril_mpo",
        "new_oracle_limit": 32,
        "seed": 5,
        "evaluation_rounds": 2,
        "contract_sha256": "fixture",
    }
    if online:
        contract.update(online_updates=True, arms=["balanced", "frozen", "learned"])
    store = Store(tmp_path / "output", lambda: None)
    calls, fits = [], []

    @contextmanager
    def session(*args):
        yield contract, store, {}

    class Policy:
        def __init__(self, payload):
            self.payload = payload

        @classmethod
        def fit(cls, observations, source_sha256):
            fits.append(copy.deepcopy(observations))
            return cls(
                {
                    "model_sha256": identity(observations),
                    "observations_sha256": identity(observations),
                    "fit": {"fixture": True},
                }
            )

        def utilities(self, rows):
            return np.arange(len(rows), dtype=float)

        def distribution(self, rows, reference, guided):
            q = np.asarray(reference)
            return q, {"model_sha256": self.payload["model_sha256"]}

    def parallel(tasks):
        assert tasks, "empty calibration must not dispatch a remote map"
        for task in tasks:
            parent = task["parent"]
            candidates = []
            for draw in range(4):
                smiles = "C" * (2 + (task["phase"] * 16 + task["slot"] * 4 + draw) % 38)
                candidates.append(
                    {
                        "id": f"{task['worker_id']}/{draw}",
                        "smiles": smiles,
                        "node": {},
                        "chain": parent["chain"] + [str(draw)],
                        "primitives": 1,
                        "parent_smiles": parent["smiles"],
                        "parent_score": parent["score"],
                        "bundle": {"option": "fixture"},
                    }
                )
            yield {
                "worker_id": task["worker_id"],
                "replay_verified": True,
                "candidates": candidates,
            }

    monkeypatch.setattr(run, "session", session)
    monkeypatch.setattr(run, "BranchPolicy", Policy)
    monkeypatch.setattr(run, "make_oracle", lambda *args: lambda s: calls.append(s) or len(s) / 100)
    task = {"run_id": "fixture", "image_revision": {"commit": "fixture"}}
    volume = SimpleNamespace(reload=lambda: None)
    result = run.driver_remote(task, tmp_path, tmp_path, volume, None, parallel)
    assert result["new_oracle_calls"] == len(calls) <= 32
    assert len(fits) == (3 if online else 2)
    assert max(len(r["smiles"]) for r in fits[1]) <= 17
    assert all(len(a["selections"]) == 8 for a in result["arms"].values())
    for number in (1, 2):
        assert store.read(f"phase/{number}/choices")["training_observations_sha256"] == identity(
            fits[1]
        )
    if online:
        first, second = (store.read(f"phase/{n}/choices") for n in (1, 2))
        assert first["policy_by_arm"]["learned"] == first["policy_by_arm"]["frozen"]
        assert second["policy_by_arm"]["frozen"] == first["policy_by_arm"]["frozen"]
        assert second["policy_by_arm"]["learned"] != first["policy_by_arm"]["learned"]
        own = [
            e["candidate"]["smiles"]
            for e in result["arms"]["learned"]["selections"]
            if e["round"] == 1 and e["status"] == "scored"
        ]
        assert [r["smiles"] for r in fits[2][len(fits[1]) :]] == own
        assert len(result["online_updates"]) == 1
    again = run.driver_remote(task, tmp_path, tmp_path, volume, None, parallel)
    assert again == result
    assert again["new_oracle_calls"] == len(calls)
    assert len(fits) == (3 if online else 2)
