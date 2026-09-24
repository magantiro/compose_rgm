"""Behavior checks for the fresh-seed linker signal accounting."""

import pytest

from scripts.analyze_fragment_linker_path_binding_validation import summarize


def _row(*, lengths=(1, 2), seeded=1):
    return {
        "attempts": 3,
        "committed_endpoints": 2,
        "committed_chemically_valid": 2,
        "committed_fragment_preserving": 2,
        "emitted_nonempty": 1,
        "emitted_samples": ["C", "", ""],
        "committed_endpoint_smiles": ["C", "CC"],
        "realized_linker_lengths": list(lengths),
        "seeded_linker_length": seeded,
        "official": {"quality": 25.0, "uniqueness": 50.0, "diversity": 0.6},
    }


def test_summarize_keeps_attempt_and_commit_denominators_distinct():
    first = _row()
    first["attempts"] = 20
    first["emitted_samples"] = ["C"] + [""] * 19
    result = summarize({"X": first})
    assert result["attempts"] == 20
    assert result["commits"] == 2
    assert result["genuine_linkers_longer_than_seed"] == 1
    assert result["genuine_linker_rate_per_attempt"] == 1 / 20
    assert result["emitted_nonempty"] == 1
    assert result["full_task_successes"] == 1
    assert result["committed_exact_chemical_validity"] == 1.0
    assert result["official_quality_fraction"] == 0.25
    assert result["official_uniqueness_fraction"] == 0.5
    assert result["official_diversity"] == 0.6


def test_summarize_refuses_missing_linker_lengths():
    row = _row(lengths=(1,))
    row["attempts"] = 20
    row["emitted_samples"] = ["C"] + [""] * 19
    with pytest.raises(ValueError, match="length/commit mismatch"):
        summarize({"X": row})
