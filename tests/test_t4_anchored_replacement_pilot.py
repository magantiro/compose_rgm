from __future__ import annotations

import pytest

from compose_v4.experiments.t4_anchored_replacement_pilot import summarize_docking


def row(digest: str, score: float | None) -> dict:
    return {"endpoint_sha256": digest, "score": score}


def test_useful_novel_endpoint_promotes_integration():
    result = summarize_docking(
        [row("a", -9.8), row("b", -10.1)], expected_digests=["a", "b"]
    )
    assert result["promotion"] == "PASS_USEFUL_NOVEL_SUPPORT"
    assert result["best_score"] == -10.1
    assert result["counts_at_or_better"]["minus_10_0"] == 1
    assert not result["strong_signal"]


def test_ivg_level_endpoint_is_a_strong_signal():
    result = summarize_docking([row("a", -10.5)], expected_digests=["a"])
    assert result["strong_signal"]
    assert result["counts_at_or_better"]["ivg_mean_minus_10_4"] == 1


def test_failed_queries_are_charged_and_reported():
    result = summarize_docking([row("a", None)], expected_digests=["a"])
    assert result["charged_calls"] == 1
    assert result["failed_calls"] == 1
    assert result["promotion"] == "FAIL_NO_USEFUL_NOVEL_SUPPORT"


def test_query_order_or_count_mismatch_fails_closed():
    with pytest.raises(ValueError, match="order"):
        summarize_docking([row("b", -10.0)], expected_digests=["a"])
    with pytest.raises(ValueError, match="expected 2"):
        summarize_docking([row("a", -10.0)], expected_digests=["a", "b"])
