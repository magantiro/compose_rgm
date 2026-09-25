"""Frozen-envelope and denominator tests for the published-scale motif run."""

from __future__ import annotations

import pytest
from run_fragment_motif_official_v1 import attempt_samples, benchmark_summary


def _contract() -> dict:
    return {"seeds": [2, 3, 4], "drugs": [f"DRUG_{i}" for i in range(10)]}


def _row(seed: int, drug: str) -> dict:
    return {
        "seed": seed,
        "drug": drug,
        "attempts": 100,
        "outputs": 100,
        "valid_connected_outputs": 100,
        "constraint_fidelity_outputs": 100,
        "metrics": {
            "validity": 100.0,
            "uniqueness": 98.0,
            "quality": float(seed),
            "diversity": 0.6,
        },
    }


def test_summary_requires_all_thirty_full_faithful_rows() -> None:
    contract = _contract()
    rows = [_row(seed, drug) for seed in contract["seeds"] for drug in contract["drugs"]]
    summary = benchmark_summary(rows, contract)
    assert summary["attempts"] == 3000
    assert summary["offered_candidate_draws"] == 24000
    assert summary["official_mean"]["quality"] == 3.0
    assert summary["constraint_fidelity_outputs"] == 3000
    with pytest.raises(ValueError, match="ten prompt rows"):
        benchmark_summary(rows[:-1], contract)
    rows[0]["constraint_fidelity_outputs"] = 99
    with pytest.raises(ValueError, match="nonfaithful"):
        benchmark_summary(rows, contract)


def test_attempt_samples_preserves_failures_in_denominator() -> None:
    attempts = [
        {
            "drug": "A",
            "attempt_index": i,
            "panel": {
                "offered_count": 8,
                "offered": [{"draw": draw} for draw in range(8)],
                "selected_smiles": None if i == 25 else "CC",
            },
        }
        for i in range(100)
    ]
    samples = attempt_samples(attempts, drug="A")
    assert len(samples) == 100
    assert samples[25] != "CC"
    attempts[0]["panel"]["offered"][-1]["draw"] = 99
    with pytest.raises(ValueError, match="eight recorded offers"):
        attempt_samples(attempts, drug="A")
