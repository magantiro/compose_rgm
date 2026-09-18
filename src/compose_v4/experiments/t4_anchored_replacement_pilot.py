"""Result accounting for the locked anchored-replacement JAK2 pilot."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "t4_anchored_replacement_pilot_result_v1"


def summarize_docking(
    rows: list[dict[str, Any]], *, expected_digests: list[str]
) -> dict[str, Any]:
    """Validate exact query accounting and apply the frozen promotion thresholds."""

    if len(rows) != len(expected_digests):
        raise ValueError(
            f"expected {len(expected_digests)} docking rows, received {len(rows)}"
        )
    observed = [str(row["endpoint_sha256"]) for row in rows]
    if observed != expected_digests:
        raise ValueError("docking rows do not match the locked candidate order")
    if len(observed) != len(set(observed)):
        raise ValueError("a locked endpoint was docked more than once")

    scores = [float(row["score"]) for row in rows if row.get("score") is not None]
    failures = [row for row in rows if row.get("score") is None]
    best = min(scores) if scores else None
    return {
        "charged_calls": len(rows),
        "successful_calls": len(scores),
        "failed_calls": len(failures),
        "best_score": best,
        "counts_at_or_better": {
            "minus_9_8": sum(score <= -9.8 for score in scores),
            "minus_10_0": sum(score <= -10.0 for score in scores),
            "ivg_mean_minus_10_4": sum(score <= -10.4 for score in scores),
            "route_witness_minus_11_0": sum(score <= -11.0 for score in scores),
        },
        "promotion": (
            "PASS_USEFUL_NOVEL_SUPPORT"
            if best is not None and best <= -10.0
            else "FAIL_NO_USEFUL_NOVEL_SUPPORT"
        ),
        "strong_signal": bool(best is not None and best <= -10.4),
        "frozen_rule": (
            "promote anchored replacement into a bounded FiberControl integration if "
            "at least one novel locked endpoint scores at or below -10.0; classify a "
            "score at or below the published IVG mean -10.4 as a strong signal"
        ),
    }
