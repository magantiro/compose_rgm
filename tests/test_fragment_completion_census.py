"""Retrospective selection diagnostics preserve canonical panel accounting."""

import math

import pytest

from tools.census_fragment_completion_pilot import selection_panel
from tools.summarize_fragment_completion_census import compact_census


def test_selection_tilt_uses_unique_panel_and_saved_supported_index():
    attempt = {
        "committed_smiles": "CCCC",
        "offered": [
            {"draw": 0, "status": "model_support_abstention", "endpoint": "CCN"},
            {"draw": 1, "status": "model_supported", "endpoint": "CC", "mean_log_mark": -2.0},
            {"draw": 2, "status": "model_supported", "endpoint": "CC", "mean_log_mark": -0.1},
            {"draw": 3, "status": "model_supported", "endpoint": "CCCC", "mean_log_mark": -1.0},
        ],
        "selection": {
            "selected_index": 2,
            "unique_model_supported_endpoints": 2,
            "selected_probability": 1 / (1 + math.exp(-1)),
        },
    }
    result = selection_panel(attempt)
    assert result["panel_size"] == 2
    assert result["selected_draw"] == 3
    assert result["heavy_atoms"]["uniform_mean"] == 3
    assert result["heavy_atoms"]["model_minus_uniform"] > 0
    attempt["selection"]["selected_probability"] = 0.5
    with pytest.raises(ValueError, match="probability disagrees"):
        selection_panel(attempt)


def test_empty_model_panel_cannot_claim_output():
    attempt = {"offered": [], "committed_smiles": None}
    assert selection_panel(attempt) == {"panel_size": 0}
    attempt["committed_smiles"] = "CC"
    with pytest.raises(ValueError, match="without model-supported"):
        selection_panel(attempt)


def test_compact_receipt_refuses_inconsistent_attempt_denominator():
    data = {
        "schema": "fragment_completion_pilot_census_v1",
        "attempts_per_arm": 20,
        "prompts": [
            {
                "task": "scaffold_decoration",
                "drug": "fixture",
                "core": {},
                "baseline": {"attempts": 20},
                "new": {"attempts": 20},
            }
        ],
    }
    assert compact_census(data)["attempts_per_arm"] == 20
    data["prompts"][0]["new"]["attempts"] = 19
    with pytest.raises(ValueError, match="denominator disagrees"):
        compact_census(data)
