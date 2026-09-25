"""Focused checks for the locked scaffold-decoration breadth comparison."""

from __future__ import annotations

import pytest
from run_fragment_content_breadth_pilot_v1 import load_contract, summarize


def _rows(*, quality: float, uniqueness: float, diversity: float) -> list[dict]:
    return [
        {
            "attempts": 20,
            "outputs": 20,
            "valid_connected_outputs": 20,
            "constraint_fidelity_outputs": 20,
            "metrics": {
                "quality": quality,
                "uniqueness": uniqueness,
                "diversity": diversity,
                "validity": 100.0,
            },
        }
        for _ in range(10)
    ]


def test_contract_is_self_hashed_and_frozen() -> None:
    contract, digest = load_contract()
    assert digest == "7e0b76650289280ed763ca81e148bd9676aab465794b72d298f2c82465070526"
    assert contract["arms"] == ["frozen", "uniform_within_cell"]
    assert contract["attempts_per_prompt_arm"] == 20


def test_promotion_requires_all_frozen_metrics() -> None:
    contract, _ = load_contract()
    rows = {
        "frozen": _rows(quality=37.0, uniqueness=97.0, diversity=0.55),
        "uniform_within_cell": _rows(quality=36.0, uniqueness=92.0, diversity=0.565),
    }
    passed = summarize(rows, contract)
    assert passed["promotion_gate_pass"] is True
    assert passed["attempts_per_arm"] == 200
    rows["uniform_within_cell"][0]["outputs"] = 19
    rows["uniform_within_cell"][0]["valid_connected_outputs"] = 19
    rows["uniform_within_cell"][0]["constraint_fidelity_outputs"] = 19
    rows["uniform_within_cell"][0]["metrics"]["validity"] = 95.0
    failed = summarize(rows, contract)
    assert failed["promotion_gate_pass"] is True  # One abstention is permitted.
    rows["uniform_within_cell"][0]["constraint_fidelity_outputs"] = 18
    failed = summarize(rows, contract)
    assert failed["promotion_gate_pass"] is False
    assert failed["checks"]["exact_prompt_fidelity"] is False


def test_incomplete_prompt_population_fails_closed() -> None:
    contract, _ = load_contract()
    rows = {
        "frozen": _rows(quality=37.0, uniqueness=97.0, diversity=0.55),
        "uniform_within_cell": _rows(quality=36.0, uniqueness=92.0, diversity=0.565)[:-1],
    }
    with pytest.raises(ValueError, match="incomplete"):
        summarize(rows, contract)
