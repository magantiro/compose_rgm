"""Protect the exact upstream winner rule without network access."""

import runpy
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def census():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/ivg_t4_census.py"))


def row(smiles="CCC", score="8", seed="0", **kwargs):
    return {
        "smiles": smiles,
        "docking score": score,
        "seed": seed,
        "qed": "0.7",
        "sa": "3",
        "sim": "0.5",
        **kwargs,
    }


def test_all_co_best_ties_and_duplicate_provenance(census):
    result = census["select_winners"](
        [row(), row(), row("CCO"), row("CCN", "7"), row("CCCC", "9", "1")], 0.4, "fixture.csv"
    )
    first, second = result["runs"]
    assert first["reported_docking_score"] == -8
    assert len(first["winners"]) == 2
    assert first["winners"][0]["source_rows"][0]["line"] == 2
    assert len(first["winners"][0]["source_rows"]) == 2
    assert second["reported_docking_score"] == -9


@pytest.mark.parametrize("boundary", [{"qed": "0.6"}, {"sa": "4"}, {"sim": "0.4"}])
def test_strict_boundaries_keep_empty_run_in_denominator(census, boundary):
    result = census["select_winners"]([row(**boundary)], 0.4, "fixture.csv")
    assert result["runs"][0]["winners"] == []
    assert result["runs"][0]["reported_docking_score"] is None
    assert sum(result["rejection_reason_counts"].values()) == 1


def test_nonfinite_input_fails_with_source_line(census):
    with pytest.raises(ValueError, match="fixture.csv:2"):
        census["select_winners"]([row(score="nan")], 0.4, "fixture.csv")


def test_topology_axes_and_deduplication(census):
    smiles = "c1ccccc1-c1ccccc1"
    d = census["structure"](smiles)
    assert (d["cycle_rank"], d["ring_system_count"]) == (2, 2)
    records = [{"canonical_smiles": smiles, "structure": d}] * 3
    result = census["summarize"](records)
    assert result["unique_structures"] == result["multiple_ring_systems"] == 1
    assert result["taxonomy"]["ring_size_counts"] == {"6": 2}
