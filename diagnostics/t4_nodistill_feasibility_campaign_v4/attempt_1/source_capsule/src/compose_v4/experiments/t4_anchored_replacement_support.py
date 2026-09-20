"""Promotion rule for the zero-oracle anchored-replacement support probe."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "t4_anchored_replacement_support_v1"
REQUIRED_LANES = ("shallow", "structured", "anchored_replacement")


def compare_support_lanes(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Require a strict feasible-basin gain over both existing proposal lanes."""
    if tuple(arms) != REQUIRED_LANES:
        raise ValueError(f"expected proposal lanes in order {REQUIRED_LANES!r}")
    counts = {
        lane: int(arms[lane]["motif_counts_over_unique_eligible"].get("basin", 0))
        for lane in REQUIRED_LANES
    }
    baselines = max(counts["shallow"], counts["structured"])
    gain = counts["anchored_replacement"] - baselines
    failures = sum(int(arms[lane]["worker_failures"]) for lane in REQUIRED_LANES)
    passed = gain > 0 and failures == 0
    return {
        "eligible_basin_counts": counts,
        "anchored_basin_gain_over_best_existing_lane": gain,
        "worker_failures": failures,
        "promotion_gate": "PASS" if passed else "FAIL",
        "gate_definition": (
            "anchored replacement must produce strictly more unique eligible "
            "amide-plus-diamine endpoints than both unchanged shallow and progressive "
            "structured lanes at equal proposal budget, with zero worker failures"
        ),
        "new_oracle_calls": 0,
    }
