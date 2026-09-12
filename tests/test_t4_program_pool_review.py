"""Audit aggregation, never fabricated molecular benchmark evidence."""

import pytest

from tools.review_t4_program_pool import arm_summary


def test_reused_winner_not_counted_as_new_and_failures_keep_denominator():
    pool = {
        "rows": [
            {"status": "complete", "receipt": {"endpoint": s}}
            for s in ("winner", "winner", "variant", "failed_oracle", "ineligible")
        ]
        + [{"status": "rejected", "reason_code": "guard"}],
        "attempts_unstarted": 0,
        "proposal_seconds": 1.0,
    }
    properties = {
        s: {"oracle_eligible": s != "ineligible"}
        for s in ("winner", "variant", "failed_oracle", "ineligible")
    }
    summary = arm_summary(
        pool, properties, {"winner": -13.6, "variant": -12.0, "failed_oracle": None}, "winner"
    )
    assert summary["attempts"] == 6
    assert summary["eligible_attempts"] == 4
    assert summary["known_winner_attempts"] == 2
    assert summary["unique_new_scored"] == 1
    assert summary["unique_failed_dockings"] == 1
    assert summary["best_new_score"] == -12.0
    assert summary["threshold_counts_inclusive"]["-13.0"] == {
        "attempts_including_reused_winner": 2,
        "unique_new_endpoints": 0,
    }
    with pytest.raises(ValueError, match="no docking receipt"):
        arm_summary(pool, properties, {"winner": -13.6}, "winner")
