"""No-oracle tests of arm isolation, fresh repeats, failures and restart guards."""

import json

import pytest

from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import compile_program_graph, execute_program_graph
from compose_v4.experiments import t4_second_generation as assay
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from tests.test_edit_program import ring_and_carbonyl


def candidate(name):
    return {
        "candidate_id": name,
        "cell": "fixture",
        "target": "parp1",
        "smiles": name,
        "oracle_protocol": "fixture:no-oracle",
        "docking_seed": 1701,
    }


def fixture():
    candidates = [candidate(name) for name in ("A", "B", "C")]
    lock = {
        "cells": ["fixture"],
        "take": candidates,
        "new_oracle_call_limit": 73,
        "incumbents": {"fixture": {**candidate("I"), "historical_ds": -8}},
        "memberships": [
            {"cell": "fixture", "arm": arm, "query_id": name}
            for arm, name in (("score_rank", "A"), ("score_rank", "C"), ("score_blind", "B"))
        ],
    }
    rows = [
        {**c, "index": i, "ds": value, "failure": None}
        for i, (c, value) in enumerate(zip(candidates, (-10, -9, -20), strict=True))
    ]
    return lock, rows


def test_confirmations_use_each_arms_own_champion_and_repeat_only_comparison():
    lock, rows = fixture()
    nominated = assay.confirmation_candidates(lock, rows)
    blind = [r for r in nominated if any(role.get("arm") == "score_blind" for role in r["roles"])]
    assert {r["smiles"] for r in blind} == {"B"}
    assert len(nominated) == 6
    assert {r["docking_seed"] for r in nominated} == {1702, 1703}
    repeats = [
        {**r, "index": i, "ds": -8 if r["smiles"] == "I" else -10, "failure": None}
        for i, r in enumerate(nominated)
    ]
    result = assay.summarize(lock, rows, repeats)[0]
    assert result["arms"][0]["best"]["ds"] == -20
    assert result["arms"][0]["repeat_mean"] == -10
    assert result["arms"][0]["repeat_improvement_over_incumbent"] == 2


def test_identical_champions_share_physical_confirmations():
    lock, rows = fixture()
    lock["memberships"][-1]["query_id"] = "C"
    nominated = assay.confirmation_candidates(lock, rows)
    assert len(nominated) == 4
    assert all(len(r["roles"]) == 2 for r in nominated if r["smiles"] == "C")


def test_failures_and_budget_limits_are_not_silently_repaired():
    lock, rows = fixture()
    rows[-1].update(ds=None, failure="oracle_failed")
    rows[0].update(ds=None, failure="oracle_failed")
    nominated = assay.confirmation_candidates(lock, rows)
    assert not any(role.get("arm") == "score_rank" for r in nominated for role in r["roles"])
    lock["new_oracle_call_limit"] = 3
    with pytest.raises(ValueError, match="budget"):
        assay.confirmation_candidates(lock, rows)
    rows[1]["oracle_protocol"] = "different-target"
    with pytest.raises(ValueError, match="oracle_protocol"):
        assay.validate_rows(lock["take"], rows)


class Volume:
    def reload(self):
        pass

    def commit(self):
        pass


def test_driver_worker_exact_replay_complete_reuse_and_ambiguous_start(tmp_path, monkeypatch):
    source, stages = ring_and_carbonyl()
    program, binding = extract_program(source, stages)
    _, trace = execute_program_graph(source, compile_program_graph(program), binding)
    row = {**candidate(trace["endpoint"]), "trace": trace, "original_seed": "CC"}
    lock = {
        "take": [row],
        "cells": ["fixture"],
        "incumbents": {"fixture": {**row, "historical_ds": -8}},
        "memberships": [
            {"cell": "fixture", "arm": arm, "query_id": row["candidate_id"]} for arm in assay.ARMS
        ],
        "new_oracle_call_limit": 73,
        "compute": {"fixture": True},
    }
    lock = json.loads(json.dumps(lock))  # Match the production persisted-lock boundary.
    monkeypatch.setattr(assay, "validate_task", lambda *_: lock)
    monkeypatch.setattr(assay, "property_scorer", lambda _: lambda _: {"oracle_eligible": True})
    monkeypatch.setattr(assay.subprocess, "check_output", lambda *_args, **_kwargs: "fixture")
    task = {"run_id": "fixture", "files_sha256": {assay.LOCK: "fixture"}}
    calls = []

    def dock(smiles, target, tag, seed):
        calls.append((smiles, target, seed))
        return -10

    def parallel(tasks):
        return [
            assay.dock_remote(t, tmp_path, tmp_path, Volume(), lambda _: None, dock)
            for t in reversed(tasks)
        ]

    result = assay.run_remote(task, tmp_path, tmp_path, Volume(), lambda _: None, parallel)
    assert result["new_oracle_calls"] == len(calls) == 3
    assert {c[2] for c in calls} == {1701, 1702, 1703}
    assert assay.run_remote(task, tmp_path, tmp_path, Volume(), lambda _: None, parallel) == result
    assert len(calls) == 3
    # A different run with a recorded start but no result must not retry its oracle.
    other = {**task, "run_id": "ambiguous", "stage": "first", "index": 0}
    folder = tmp_path / assay.KIND / "ambiguous" / "first"
    seal(
        folder / "candidate_lock.json",
        {"parent_lock_sha256": "fixture", "stage": "first", "take": [row]},
    )
    other["stage_lock_sha256"] = assay.sha256_file(folder / "candidate_lock.json")
    seal(folder / "rows/00/started.json", {"candidate_id": row["candidate_id"]})
    with pytest.raises(RuntimeError, match="ambiguous"):
        assay.dock_remote(other, tmp_path, tmp_path, Volume(), lambda _: None, dock)
    assert len(calls) == 3
    assert unseal(folder / "rows/00/started.json")["candidate_id"] == row["candidate_id"]
