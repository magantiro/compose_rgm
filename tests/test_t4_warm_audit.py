"""Offline audit rejects stale topology and archive bindings."""

import copy
import json

import pytest
from test_t4_warm_continuation import fixture_source, synthetic_prepare

from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.t4_warm_continuation import run_continuation
from tools.t4_warm_audit import candidate_summary, counts, report


def prepared(task, checkpoint, cached, warm):
    lock = synthetic_prepare(task, checkpoint, cached, warm)
    c = lock["take"][0]
    c.update(d_heavy=1, d_cycle_rank=0, d_ring_systems=0, r_release=0.1, r_coherent=0.1)
    for unit in lock["work"]:
        unit["law_enumerations"] = 1
        for row in unit["sampled_transitions"]:
            row.update(sample_probability=1.0, kl=0.0)
    lock.update(
        pool=copy.deepcopy(lock["take"]),
        candidate_diversity=0.0,
        selected_diversity=0.0,
        proposal_seconds=0.0,
        locked_at_utc="2000-01-01T00:00:00+00:00",
    )
    return lock


def test_full_round_audit_and_tampered_warm_identity(tmp_path):
    source, contract, task = fixture_source(tmp_path)
    root = tmp_path / "out"
    run_continuation(task, prepared, lambda *_: [-9.0], root, source=source, contract=contract)
    audit = report(root)
    assert audit["cumulative_oracle_attempts"] == 22
    assert [r["round"] for r in audit["rounds"]] == [2, 3]
    assert (
        audit["rounds"][1]["selected_candidates"][0]["cumulative_delta_from_seed"]["n_heavy"] == 22
    )
    assert [p["cumulative_calls"] for p in audit["best_so_far_curve"]] == [20, 21, 22]
    candidate = audit["rounds"][1]["selected_candidates"][0]
    assert candidate["parent_docking_score"] == -9.0
    assert candidate["ring_descriptors"]["aromatic_rings"] == 0
    ledger_path = root / "round_3/executor_attempts_0.json"
    calls = audit["rounds"][1]["executor_calls"]
    ledger = {
        "prior_calls": 0,
        "attempts": [{"call_index": i, "status": "executed"} for i in range(calls)],
    }
    ledger_path.write_text(json.dumps(ledger))
    assert report(root)["rounds"][1]["executor_receipt_counts"] == (
        {"executed": calls} if calls else {}
    )
    ledger["prior_calls"] = 1
    ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="ledger does not reconcile"):
        report(root)
    ledger["prior_calls"] = 0
    ledger["attempts"].append({"call_index": calls, "status": "executed"})
    ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="ledger does not reconcile"):
        report(root)
    ledger_path.unlink()
    path = root / "round_3/candidate_lock.json"
    lock = unseal(path)
    lock["task"]["warm_start_sha256"] = "wrong"
    seal(path, lock)
    with pytest.raises(ValueError, match="expected complete archive"):
        report(root)


def test_topology_metadata_must_match_exact_state(tmp_path):
    source, contract, task = fixture_source(tmp_path)
    root = tmp_path / "out"
    run_continuation(task, prepared, lambda *_: [-9.0], root, source=source, contract=contract)
    warm, lock = unseal(root / "warm_start.json"), unseal(root / "round_2/candidate_lock.json")
    candidate = lock["take"][0]
    candidate["d_cycle_rank"] = 1
    with pytest.raises(ValueError, match="d_cycle_rank"):
        candidate_summary(
            candidate,
            lock,
            {c["smiles"]: c for c in warm["archive"]},
            counts(warm["archive"][0]["state"]),
        )
