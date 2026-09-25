"""Fail-closed checks for the matched linker novelty pilot."""

import json
import sys

import pytest
import run_fragment_linker_novelty_pilot_v1 as novelty
from run_fragment_linker_broaden_pilot_v1 import identity, summarize


def contract() -> dict:
    return {
        "schema": "fragment_linker_novelty_pilot_v1",
        "arms": ["frozen", "novelty4"],
        "task": "linker_design",
        "drugs": [f"prompt_{index}" for index in range(10)],
        "seed": 5,
        "attempts_per_prompt_arm": 100,
        "candidate_draws_per_attempt": 8,
        "workers": 1,
        "threads": 1,
        "quality_selection": False,
        "oracle_calls": 0,
        "novelty_multiplier": 4.0,
        "archive_scope": "same_prompt_prior_emissions",
        "output_dir": "diagnostics/fragment_linker_novelty_pilot_v1",
        "promotion_gate": {
            "min_quality_gain_points": 3.0,
            "min_uniqueness_gain_points": 8.0,
            "max_diversity_drop": 0.02,
            "min_outputs_per_arm": 1000,
        },
    }


def test_novelty_contract_hash_and_scientific_envelope(tmp_path) -> None:
    payload = contract()
    path = tmp_path / "contract.json"
    path.write_text(json.dumps({"payload": payload, "payload_sha256": identity(payload)}))
    assert novelty.load_contract(path)[0] == payload
    payload["seed"] = 2
    path.write_text(json.dumps({"payload": payload, "payload_sha256": identity(payload)}))
    with pytest.raises(ValueError, match="matched scientific envelope"):
        novelty.load_contract(path)
    path.write_text(json.dumps({"payload": payload, "payload_sha256": "wrong"}))
    with pytest.raises(ValueError, match="hash mismatch"):
        novelty.load_contract(path)


def test_novelty_summary_requires_quality_and_uniqueness_gain() -> None:
    payload = contract()
    rows = [
        {
            "arm": arm,
            "drug": drug,
            "attempts": 100,
            "outputs": 100,
            "exact_core_path_fidelity_outputs": 100,
            "metrics": {
                "validity": 100.0,
                "uniqueness": 70.0 + (9.0 if arm == "novelty4" else 0.0),
                "quality": 28.0 + (4.0 if arm == "novelty4" else 0.0),
                "diversity": 0.56,
            },
        }
        for arm in payload["arms"]
        for drug in payload["drugs"]
    ]
    result = summarize(rows, payload)
    assert result["schema"] == "fragment_linker_novelty_pilot_result_v1"
    assert result["promotion_gate_passed"]
    rows[-1]["metrics"]["quality"] = -100.0
    assert not summarize(rows, payload)["promotion_gate_passed"]


def test_scored_novelty_run_requires_payload_bound_authorization(tmp_path, monkeypatch) -> None:
    payload = contract()
    monkeypatch.setattr(novelty, "ROOT", tmp_path)
    monkeypatch.setattr(novelty, "load_contract", lambda: (payload, identity(payload)))
    monkeypatch.setattr(novelty, "preflight", lambda _: ({}, ()))
    monkeypatch.setattr(novelty, "cell_support_summary", lambda *_: [])
    output = tmp_path / payload["output_dir"]
    monkeypatch.setattr(sys, "argv", ["novelty", "run", "--output-dir", str(output)])
    with pytest.raises(ValueError, match="matching user authorization"):
        novelty.main()
