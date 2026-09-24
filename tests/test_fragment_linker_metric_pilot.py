"""Locked denominators and explicit prefix reuse for the bounded metric pilot."""

import json

import pytest
from run_fragment_training_linker_metric_pilot import (
    immutable_json,
    imported_prefix_attempt,
    pilot_summary,
    prompt_samples,
)


def test_twenty_attempts_including_missing_outputs_reach_official_evaluator():
    rows = [
        {
            "drug": "fixture",
            "attempt_index": i,
            "panel": {"offered_count": 8, "selected_smiles": "CC" if i else None},
        }
        for i in range(20)
    ]
    samples = prompt_samples(rows, drug="fixture")
    assert len(samples) == 20
    assert samples[0] == ""
    with pytest.raises(ValueError, match="all twenty"):
        prompt_samples(rows[1:], drug="fixture")
    with pytest.raises(ValueError, match="all twenty"):
        prompt_samples(list(reversed(rows)), drug="fixture")


def test_summary_preserves_official_macro_means_and_distinct_comparators():
    rows = [
        {
            "drug": str(i),
            "attempts": 20,
            "outputs": 18,
            "exact_core_path_fidelity_outputs": 18,
            "metrics": {
                "validity": 90.0,
                "uniqueness": 80.0,
                "quality": float(i),
                "diversity": 0.5,
            },
        }
        for i in range(10)
    ]
    contract = {
        "required_output_coverage": 0.9,
        "reported_comparators": {
            "ivg": {"validity": 60.37, "uniqueness": 84.76, "quality": 22.33, "diversity": 0.52},
            "genmol_v2": {
                "validity": 81.8,
                "uniqueness": 87.1,
                "quality": 28.6,
                "diversity": 0.566,
            },
        },
    }
    result = pilot_summary(rows, contract)
    assert result["metrics"]["quality"] == 4.5
    assert result["output_coverage"] == 0.9
    assert result["checks"]["output_at_least_90_percent"]
    assert result["support_attempts_reused"] == 20
    assert result["new_candidate_draws"] == 1440
    assert result["reported_comparator_differences"]["ivg"]["quality"] == pytest.approx(4.5 - 22.33)
    assert result["reported_comparator_differences"]["genmol_v2"]["quality"] == pytest.approx(
        4.5 - 28.6
    )
    rows[0]["exact_core_path_fidelity_outputs"] = 17
    assert not pilot_summary(rows, contract)["checks"]["all_selected_exact_core_path_fidelity"]
    with pytest.raises(ValueError, match="all ten"):
        pilot_summary(rows[:-1], contract)


def test_prefix_import_preserves_source_payload_and_records_exact_physical_identity(tmp_path):
    source = {"drug": "fixture", "attempt_index": 0, "panel": {"rng_state_after": {"state": 123}}}
    path = tmp_path / "support.json"
    immutable_json(path, source)
    imported = imported_prefix_attempt(source, path)
    assert {k: v for k, v in imported.items() if k != "origin"} == source
    assert imported["origin"]["role"] == "reused_support_prefix"
    assert len(imported["origin"]["sha256"]) == 64
    assert json.loads(path.read_text()) == source
    immutable_json(path, source)
    with pytest.raises(ValueError, match="immutable pilot artifact"):
        immutable_json(path, {**source, "attempt_index": 1})
