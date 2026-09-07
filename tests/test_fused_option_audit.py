"""Artifact and stop invariants for the fixed local integration audit."""

import hashlib

import pytest

from compose_v4.experiments.continuation_profile import canonical_bytes
from compose_v4.experiments.fused_option_audit import CONFIG, run_audit


def test_recorded_reference_has_exact_state_hashes_and_complete_receipts():
    result = run_audit()
    assert result["trajectory"]["estimator"] == "reference"
    assert result["work"]["public_executor_applications"] <= 256
    assert len(result["executor_receipts"]) == result["work"]["public_executor_applications"]
    for identity, payload in result["states"].items():
        assert hashlib.sha256(canonical_bytes(payload)).hexdigest() == identity
        assert payload["bundle_id"] == "synthetic-fused-reference-seed0"
    for row in result["rows"].values():
        assert set(row["successors"]) <= result["states"].keys()
        if row["successors"]:
            assert sum(row["probabilities"]) == pytest.approx(1.0)
    if result["trajectory"]["status"] == "complete":
        assert result["independent_topology_witness"]["passed"]
        assert len(result["trajectory"]["trace"]) == 5
    else:
        assert result["trajectory"]["status"] in {
            "no_admissible_action",
            "executor_budget_exhausted",
        }


def test_zero_budget_preserves_incomplete_work_and_no_endpoint(monkeypatch):
    monkeypatch.setitem(CONFIG, "max_executor_applications", 0)
    result = run_audit()
    assert result["trajectory"]["status"] == "executor_budget_exhausted"
    assert result["trajectory"]["endpoint"] is None
    assert result["executor_receipts"] == []
    assert result["rows"] == {}


def test_timeout_is_incomplete_not_unreachability(monkeypatch):
    monkeypatch.setitem(CONFIG, "safety_seconds", 0.0)
    result = run_audit()
    assert result["trajectory"]["status"] == "timeout_incomplete"
    assert result["independent_topology_witness"] is None
