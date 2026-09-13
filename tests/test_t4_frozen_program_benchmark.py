from __future__ import annotations

from compose_v4.experiments import t4_frozen_program_benchmark as benchmark


def test_strict_endpoint_scorer_preserves_strict_boundaries(monkeypatch):
    scorer = benchmark.strict_endpoint_scorer("CC", delta=0.4)
    monkeypatch.setattr(benchmark.QED, "qed", lambda _molecule: 0.6)
    monkeypatch.setattr(benchmark.sascorer, "calculateScore", lambda _molecule: 4.0)
    monkeypatch.setattr(benchmark.DataStructs, "TanimotoSimilarity", lambda _a, _b: 0.4)

    result = scorer({"smiles": "CCC"})

    assert result["oracle_eligible"] is False
    assert result["endpoint_exclusion_reasons"] == [
        "similarity_not_strictly_above_delta",
        "qed_not_strictly_above_0.6",
        "sa_not_strictly_below_4",
    ]


def test_competitive_plateau_requires_competitive_flat_run():
    policy = {
        "minimum_calls": 500,
        "window_calls": 250,
        "minimum_improvement": 0.3,
        "requires_best_at_or_better_than_ivg_mean": True,
    }
    flat_competitive = [
        {"query": query, "best_score": -10.0 if query <= 250 else -10.2} for query in range(1, 501)
    ]
    improving = [
        {"query": query, "best_score": -10.0 if query <= 250 else -10.3} for query in range(1, 501)
    ]

    stopped = benchmark.competitive_plateau(flat_competitive, -10.1, policy)
    assert stopped["stop"] is True
    assert stopped["reason"] == "competitive_plateau"
    assert benchmark.competitive_plateau(flat_competitive, -10.3, policy)["stop"] is False
    assert benchmark.competitive_plateau(improving, -10.1, policy)["stop"] is False
    assert benchmark.competitive_plateau(flat_competitive[:499], -10.1, policy)["stop"] is False


def test_first_crossing_is_query_ordered_and_lower_is_better():
    curve = [
        {"query": 1, "best_score": None},
        {"query": 2, "best_score": -9.9},
        {"query": 3, "best_score": -10.2},
        {"query": 4, "best_score": -10.2},
    ]
    assert benchmark.first_crossing(curve, -10.1) == 3
    assert benchmark.first_crossing(curve, -10.3) is None


def test_compressed_checkpoints_are_deterministic_and_verified(tmp_path):
    payload = {"z": [3, 2, 1], "a": {"state": "complete"}}
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"

    first_hash = benchmark._write_gzip(first, payload)
    second_hash = benchmark._write_gzip(second, payload)

    assert first_hash == second_hash
    assert first.read_bytes() == second.read_bytes()
    assert benchmark._read_gzip(first) == payload


def test_confirmation_lock_uses_best_first_score_across_replicates():
    contract = {
        "cells": {"braf_1": {}},
        "confirmation_seeds": [9101, 9102],
        "confirmation_call_ceiling": 30,
    }
    results = [
        {
            "status": "complete",
            "unit": {
                "cell": "braf_1",
                "target": "braf",
                "source_idx": 1,
                "replicate": replicate,
                "unit_id": f"braf_1_r{replicate}",
                "oracle_protocol": "fixed",
            },
            "champion": {
                "score": score,
                "endpoint": endpoint,
                "receipt_id": f"receipt-{replicate}",
            },
        }
        for replicate, score, endpoint in (
            (0, -10.0, "CC"),
            (1, -11.0, "CCC"),
            (2, -10.5, "CCCC"),
        )
    ]

    locked = benchmark.confirmation_lock(contract, results)

    assert locked["missing_cells"] == []
    assert len(locked["queries"]) == 2
    assert {row["endpoint"] for row in locked["queries"]} == {"CCC"}
    assert {row["docking_seed"] for row in locked["queries"]} == {9101, 9102}
