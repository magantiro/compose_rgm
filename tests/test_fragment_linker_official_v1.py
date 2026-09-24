"""Frozen linker benchmark denominators, independent seeds and lock rules."""

import pytest
from run_fragment_linker_official_v1 import attempt_samples, benchmark_summary, seed_progress


def _attempt(index: int, *, output: str | None = "CC") -> dict:
    return {
        "drug": "fixture",
        "attempt_index": index,
        "panel": {
            "offered_count": 8,
            "offered": [{"draw": draw} for draw in range(8)],
            "selected_smiles": output,
        },
    }


def test_official_linker_population_keeps_all_one_hundred_attempts():
    attempts = [_attempt(i, output=None if i == 0 else "CC") for i in range(100)]
    samples = attempt_samples(attempts, drug="fixture")
    assert len(samples) == 100
    assert samples[0] == ""
    with pytest.raises(ValueError, match="100 ordered"):
        attempt_samples(attempts[:-1], drug="fixture")
    with pytest.raises(ValueError, match="100 ordered"):
        attempt_samples(attempts[::-1], drug="fixture")
    attempts[1]["panel"]["offered"].pop()
    with pytest.raises(ValueError, match="eight recorded draws"):
        attempt_samples(attempts, drug="fixture")


def test_three_fresh_seed_summary_refuses_missing_or_duplicate_prompt_row():
    contract = {
        "seeds": [1, 2, 3],
        "drugs": [str(i) for i in range(10)],
        "attempts_per_prompt_seed": 100,
    }
    rows = [
        {
            "seed": seed,
            "drug": drug,
            "attempts": 100,
            "outputs": 100,
            "connected_valid_outputs": 100,
            "exact_core_path_fidelity_outputs": 100,
            "metrics": {"validity": 100.0, "uniqueness": 90.0, "quality": 30.0, "diversity": 0.55},
        }
        for seed in contract["seeds"]
        for drug in contract["drugs"]
    ]
    summary = benchmark_summary(rows, contract)
    assert summary["official_mean"]["quality"] == 30.0
    assert summary["outputs"] == 3000
    assert summary["per_seed"][0]["seed"] == 1
    with pytest.raises(ValueError, match="exactly ten prompt rows"):
        benchmark_summary(rows[:-1], contract)
    with pytest.raises(ValueError, match="exactly ten prompt rows"):
        benchmark_summary(rows[:-1] + [rows[0]], contract)
    rows[0]["attempts"] = 99
    with pytest.raises(ValueError, match="incomplete 100-attempt"):
        benchmark_summary(rows, contract)
    assert seed_progress(contract, rows[:2], 10) == 210
