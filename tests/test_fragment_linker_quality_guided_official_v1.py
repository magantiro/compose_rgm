"""Contract and accounting checks for the fresh quality-guided linker arm."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_fragment_linker_quality_guided_official_v1 import (
    load_contract,
    material_path,
    quality_scorer_factory,
    seed_summary,
)


def test_frozen_quality_guided_contract_is_self_hashed():
    contract, digest = load_contract()
    assert len(digest) == 64
    assert contract["seeds"] == [9, 10, 11]
    assert contract["candidate_draws_per_attempt"] == 8
    assert (
        contract["comparison_status"]
        == "property_guided_controller_not_unguided_fragment_generator"
    )


def test_material_locator_rejects_unbound_paths():
    with pytest.raises(ValueError, match="invalid linker material locator"):
        material_path("../unbound")


def test_pinned_property_scorer_caches_by_canonical_endpoint():
    cache = {}
    score = quality_scorer_factory(cache)
    first = score("CCO")
    assert score("CCO") == first
    assert len(cache) == 1
    assert 0 <= first[1] <= 1
    with pytest.raises(ValueError, match="invalid endpoint"):
        score("not-a-molecule")


def test_seed_summary_keeps_property_budget_and_validity_accounting():
    row = {
        "drug": "TEST",
        "seed": 9,
        "attempts": 100,
        "outputs": 99,
        "valid_connected_outputs": 99,
        "exact_core_path_fidelity_outputs": 99,
        "property_evaluations": 217,
        "property_candidate_inspections": 650,
        "metrics": {"quality": 45.0, "uniqueness": 90.0, "diversity": 0.55, "validity": 99.0},
    }
    result = seed_summary([row], 9, ["TEST"])
    assert result["property_evaluations"] == 217
    assert result["property_candidate_inspections"] == 650
    assert result["quality"] == 45.0
    with pytest.raises(ValueError, match="incomplete or unfaithful"):
        seed_summary([{**row, "valid_connected_outputs": 98}], 9, ["TEST"])
