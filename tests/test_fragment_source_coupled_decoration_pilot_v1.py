"""Gate and denominator checks for source-coupled decoration scoring."""

from __future__ import annotations

import pytest
from run_fragment_source_coupled_decoration_pilot_v1 import summarize


def _rows(quality: float, diversity: float) -> list[dict]:
    return [
        {
            "attempts": 20,
            "outputs": 20,
            "valid_connected_outputs": 20,
            "constraint_fidelity_outputs": 20,
            "metrics": {
                "quality": quality,
                "uniqueness": 95.0,
                "diversity": diversity,
                "validity": 100.0,
            },
        }
        for _ in range(10)
    ]


def _contract() -> dict:
    return {
        "attempts_per_prompt_arm": 20,
        "offers_per_attempt": 8,
        "promotion_gate": {
            "minimum_quality_gain_points": 3.0,
            "minimum_diversity": 0.56,
            "minimum_uniqueness_percent": 90.0,
            "minimum_outputs_per_arm": 190,
            "all_committed_valid_and_faithful": True,
        },
    }


def test_quality_diversity_gate_requires_both() -> None:
    rows = {"frozen": _rows(35.0, 0.56), "source_coupled": _rows(39.0, 0.57)}
    assert summarize(rows, _contract())["promotion_gate_pass"] is True
    rows["source_coupled"][0]["metrics"]["diversity"] = 0.4
    result = summarize(rows, _contract())
    assert result["promotion_gate_pass"] is False
    assert result["checks"]["minimum_diversity"] is False


def test_prompt_fidelity_and_population_fail_closed() -> None:
    rows = {"frozen": _rows(35.0, 0.56), "source_coupled": _rows(39.0, 0.57)}
    rows["source_coupled"][0]["constraint_fidelity_outputs"] = 19
    assert summarize(rows, _contract())["promotion_gate_pass"] is False
    with pytest.raises(ValueError, match="incomplete"):
        summarize(
            {"frozen": rows["frozen"], "source_coupled": rows["source_coupled"][:-1]}, _contract()
        )
