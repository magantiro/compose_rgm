"""Saved-result audit distinguishes scientific equality from wrapper timestamps."""

from pathlib import Path

import pytest

from tools import t4_partial_audit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "diagnostics/t4_warm_continuation/attempt_2"
RUN = ROOT / "diagnostics/t4_partial_docking/attempt_1"


def test_saved_partial_result_all_receipts_bind():
    result = t4_partial_audit.report(RUN, SOURCE)
    assert result["verification"] == "passed"
    assert (
        result["new_oracle_attempts"]
        == result["distinct_canonical"]
        == result["distinct_bundles"]
        == 13
    )
    assert result["maximum_observed_worker_overlap"] <= 4


def test_docking_score_mismatch_still_fails(monkeypatch):
    original = t4_partial_audit.unseal

    def tampered(path):
        value = original(path)
        if path.name == "docking.json":
            value["docked"][0]["ds"] -= 1
        return value

    monkeypatch.setattr(t4_partial_audit, "unseal", tampered)
    with pytest.raises(ValueError, match="differs from sealed docking"):
        t4_partial_audit.report(RUN, SOURCE)
