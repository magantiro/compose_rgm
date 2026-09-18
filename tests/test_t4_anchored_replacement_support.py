from __future__ import annotations

import pytest

from compose_v4.experiments.t4_anchored_replacement_support import compare_support_lanes


def _arm(lane: str, basin: int, *, failures: int = 0) -> dict:
    return {
        "lane": lane,
        "motif_counts_over_unique_eligible": {"basin": basin},
        "worker_failures": failures,
    }


def test_anchored_lane_must_beat_both_existing_lanes():
    arms = {
        "shallow": _arm("shallow", 0),
        "structured": _arm("structured", 1),
        "anchored_replacement": _arm("anchored_replacement", 2),
    }
    assert compare_support_lanes(arms)["promotion_gate"] == "PASS"
    arms["anchored_replacement"] = _arm("anchored_replacement", 1)
    assert compare_support_lanes(arms)["promotion_gate"] == "FAIL"


def test_anchored_lane_gate_refuses_failures_and_wrong_lane_order():
    arms = {
        "shallow": _arm("shallow", 0),
        "structured": _arm("structured", 0),
        "anchored_replacement": _arm("anchored_replacement", 2, failures=1),
    }
    assert compare_support_lanes(arms)["promotion_gate"] == "FAIL"
    with pytest.raises(ValueError):
        compare_support_lanes(dict(reversed(tuple(arms.items()))))
