"""Focused contract and reduction checks for the matched linker pilot."""

import json

import pytest
from run_fragment_linker_broaden_pilot_v1 import identity, load_contract, summarize


def contract():
    return {
        "schema": "fragment_linker_broaden_pilot_v1",
        "arms": ["frozen", "sqrt_train_mass"],
        "task": "linker_design",
        "drugs": [f"prompt_{index}" for index in range(10)],
        "seed": 4,
        "attempts_per_prompt_arm": 100,
        "candidate_draws_per_attempt": 8,
        "workers": 1,
        "threads": 1,
        "quality_selection": False,
        "oracle_calls": 0,
        "promotion_gate": {
            "min_quality_gain_points": 3.0,
            "min_uniqueness_gain_points": 8.0,
            "max_diversity_drop": 0.02,
            "min_outputs_per_arm": 1000,
        },
    }


def rows(*, quality_gain=4.0, uniqueness_gain=9.0):
    return [
        {
            "arm": arm,
            "drug": drug,
            "attempts": 100,
            "outputs": 100,
            "exact_core_path_fidelity_outputs": 100,
            "metrics": {
                "validity": 100.0,
                "uniqueness": 72.0 + (uniqueness_gain if arm == "sqrt_train_mass" else 0),
                "quality": 28.0 + (quality_gain if arm == "sqrt_train_mass" else 0),
                "diversity": 0.56,
            },
        }
        for arm in ("frozen", "sqrt_train_mass")
        for drug in contract()["drugs"]
    ]


def test_summary_requires_complete_matched_panel_and_frozen_gate():
    result = summarize(rows(), contract())
    assert result["attempts_per_arm"] == 1000
    assert result["by_arm"]["frozen"]["quality"] == 28.0
    assert result["promotion_gate_passed"]
    assert not summarize(rows(quality_gain=2.0), contract())["promotion_gate_passed"]
    with pytest.raises(ValueError, match="both complete"):
        summarize(rows()[:-1], contract())


def test_contract_hash_and_budget_fail_closed(tmp_path):
    payload = contract()
    path = tmp_path / "contract.json"
    path.write_text(json.dumps({"payload": payload, "payload_sha256": identity(payload)}))
    assert load_contract(path)[0] == payload
    payload["candidate_draws_per_attempt"] = 9
    path.write_text(json.dumps({"payload": payload, "payload_sha256": identity(payload)}))
    with pytest.raises(ValueError, match="matched scientific envelope"):
        load_contract(path)
    path.write_text(json.dumps({"payload": payload, "payload_sha256": "wrong"}))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_contract(path)
