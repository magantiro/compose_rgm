"""Denominator and no-censoring guards for the failed held qualification audit."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from analyze_fragment_single_interface_qualification import _prompt


def _row() -> dict:
    smiles = "CCO"
    return {
        "attempts": 3,
        "attempt_records": [
            {"committed_smiles": smiles, "emitted_smiles": smiles},
            {"committed_smiles": smiles, "emitted_smiles": smiles},
            {"committed_smiles": None, "emitted_smiles": None},
        ],
        "committed_endpoint_smiles": [smiles, smiles],
        "emitted_samples": [smiles, smiles, ""],
        "committed_endpoints": 2,
        "emitted_nonempty": 2,
        "committed_chemically_valid": 2,
        "committed_fragment_preserving": 2,
        "committed_interfaces_covered": 2,
        "official": {
            "validity": 200 / 3,
            "uniqueness": 50.0,
            "diversity": 0.0,
            "quality": 100 / 3,
        },
        **{
            key: 0
            for key in (
                "budget_exhausted",
                "constraint_failures",
                "executor_refusals",
                "interface_rejections",
                "lock_rejections",
                "redirections",
                "staging_rejections",
            )
        },
    }


def test_quality_denominators_do_not_drop_failure_or_count_duplicate_twice():
    cache = {"CCO": {"sa": 2.0, "qed": 0.8, "quality_pass": True, "heavy_atoms": 3}}
    result = _prompt(_row(), cache)
    assert result["distinct_qualified_committed"] == 1
    assert result["distinct_qualified_emitted"] == 1
    assert result["quality_committed_per_attempt"] == pytest.approx(1 / 3)
    assert result["quality_committed_per_commit"] == pytest.approx(1 / 2)
    assert result["quality_committed_per_unique_valid"] == pytest.approx(1.0)
    assert result["quality_official_emitted_per_attempt"] == pytest.approx(1 / 3)


def test_attempt_aligned_emissions_cannot_be_censored():
    row = _row()
    row["emitted_samples"] = ["CCO", "CCO"]
    with pytest.raises(AssertionError, match="attempt-aligned emitted list"):
        _prompt(row, {"CCO": {"sa": 2.0, "qed": 0.8, "quality_pass": True, "heavy_atoms": 3}})
