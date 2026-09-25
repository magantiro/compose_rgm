from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from run_fragment_decoration_official_v2 import (
    load_contract,
    prepared_seed_directory,
    seed_summary,
)


def _row(drug: str, seed: int) -> dict:
    return {
        "drug": drug,
        "seed": seed,
        "attempts": 100,
        "outputs": 100,
        "valid_connected_outputs": 100,
        "constraint_fidelity_outputs": 100,
        "metrics": {
            "quality": 35.0,
            "uniqueness": 98.0,
            "diversity": 0.6,
            "validity": 100.0,
        },
    }


def test_contract_is_self_hashed_and_frozen() -> None:
    contract, digest = load_contract()
    assert len(digest) == 64
    assert contract["seeds"] == [8, 9, 10]
    assert contract["attempts_per_prompt_seed"] == 100


def test_seed_summary_requires_all_valid_faithful_attempts() -> None:
    drugs = ["A", "B"]
    rows = [_row(drug, 8) for drug in drugs]
    result = seed_summary(rows, 8, drugs)
    assert result["outputs"] == 200
    assert result["quality"] == 35.0
    rows[1]["constraint_fidelity_outputs"] = 99
    with pytest.raises(ValueError, match="incomplete prompt population"):
        seed_summary(rows, 8, drugs)


def test_seed_summary_rejects_missing_prompt() -> None:
    with pytest.raises(ValueError, match="incomplete prompt population"):
        seed_summary([_row("A", 8)], 8, ["A", "B"])


def test_seed_namespace_exists_before_first_attempt(tmp_path: Path) -> None:
    path = prepared_seed_directory(tmp_path, 8)
    assert path == tmp_path / "seed8"
    assert path.is_dir()
