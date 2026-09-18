from __future__ import annotations

import pytest

from compose_v4.experiments.t4_fiber_lane_support import compare_lanes, summarize_attempts


def _attempt(lane, seed, smiles=()):
    return {
        "lane": lane,
        "seed": seed,
        "elapsed_seconds": 1.0,
        "candidates": [
            {"smiles": value, "program_families": ["functionalize"]} for value in smiles
        ],
    }


def test_lane_summary_deduplicates_completed_endpoints():
    rows = [_attempt("shallow", 1, ["CCO"]), _attempt("shallow", 2, ["CCO", "CCN"])]
    summary = summarize_attempts(rows, expected_attempts=2)
    assert summary["eligible_records"] == 3
    assert summary["unique_eligible_endpoints"] == 2
    assert summary["attempts_with_eligible_endpoint"] == 2
    assert summary["new_oracle_calls"] == 0


def test_lane_summary_refuses_missing_or_reused_attempts():
    with pytest.raises(ValueError):
        summarize_attempts([_attempt("shallow", 1)], expected_attempts=2)
    with pytest.raises(ValueError):
        summarize_attempts(
            [_attempt("structured", 1), _attempt("structured", 1)], expected_attempts=2
        )


def test_promotion_requires_a_strict_eligible_basin_gain():
    shallow = {
        "lane": "shallow",
        "unique_eligible_endpoints": 3,
        "motif_counts_over_unique_eligible": {"amide": 1, "diamine_ring": 0, "basin": 0},
    }
    structured = {
        "lane": "structured",
        "unique_eligible_endpoints": 4,
        "motif_counts_over_unique_eligible": {"amide": 2, "diamine_ring": 1, "basin": 1},
    }
    assert compare_lanes(shallow, structured)["scored_pilot_support_gate"] == "PASS"
    structured["motif_counts_over_unique_eligible"]["basin"] = 0
    assert compare_lanes(shallow, structured)["scored_pilot_support_gate"] == "FAIL"
