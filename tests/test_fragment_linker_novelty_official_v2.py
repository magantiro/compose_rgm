from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from run_fragment_linker_novelty_official_v2 import load_contract, seed_summary


def _row(drug: str, seed: int) -> dict:
    return {
        "drug": drug,
        "seed": seed,
        "attempts": 100,
        "outputs": 100,
        "valid_connected_outputs": 100,
        "exact_core_path_fidelity_outputs": 100,
        "metrics": {
            "quality": 35.0,
            "uniqueness": 82.0,
            "diversity": 0.56,
            "validity": 100.0,
        },
    }


def test_contract_is_self_hashed_and_frozen() -> None:
    contract, digest = load_contract()
    assert len(digest) == 64
    assert contract["seeds"] == [6, 7, 8]
    assert contract["selector"] == "novelty4"


def test_seed_summary_requires_all_faithful_outputs() -> None:
    drugs = ["A", "B"]
    rows = [_row(drug, 6) for drug in drugs]
    result = seed_summary(rows, 6, drugs)
    assert result["outputs"] == 200
    assert result["quality"] == 35.0
    rows[1]["exact_core_path_fidelity_outputs"] = 99
    with pytest.raises(ValueError, match="incomplete or unfaithful"):
        seed_summary(rows, 6, drugs)


def test_seed_summary_rejects_missing_prompt() -> None:
    with pytest.raises(ValueError, match="incomplete or unfaithful"):
        seed_summary([_row("A", 6)], 6, ["A", "B"])
