"""Result accounting for the locked anchored-replacement JAK2 pilot."""

from __future__ import annotations

import statistics
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


def summarize_confirmation(
    rows: list[dict[str, Any]],
    *,
    expected_digests: list[str],
    expected_seeds: list[int],
) -> dict[str, Any]:
    """Reduce the complete candidate-by-seed confirmation grid."""

    expected = {(digest, seed) for digest in expected_digests for seed in expected_seeds}
    observed = {
        (str(row["endpoint_sha256"]), int(row["docking_seed"])) for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"confirmation grid mismatch; missing={missing}, extra={extra}")

    candidates = []
    for digest in expected_digests:
        group = [row for row in rows if row["endpoint_sha256"] == digest]
        scores = [float(row["score"]) for row in group if row.get("score") is not None]
        candidates.append(
            {
                "endpoint_sha256": digest,
                "smiles": group[0]["smiles"],
                "successful_replicates": len(scores),
                "failed_replicates": len(group) - len(scores),
                "mean_score": statistics.fmean(scores) if scores else None,
                "median_score": statistics.median(scores) if scores else None,
                "score_min": min(scores) if scores else None,
                "score_max": max(scores) if scores else None,
                "scores": scores,
            }
        )
    complete = [row for row in candidates if row["successful_replicates"] == len(expected_seeds)]
    best = min(complete, key=lambda row: row["mean_score"]) if complete else None
    return {
        "charged_calls": len(rows),
        "successful_calls": sum(row.get("score") is not None for row in rows),
        "failed_calls": sum(row.get("score") is None for row in rows),
        "candidates": candidates,
        "best_by_mean": best,
        "confirmation": (
            "PASS_REPLICATED_IVG_LEVEL"
            if best is not None and best["mean_score"] <= -10.4
            else "FAIL_NOT_REPLICATED_AT_IVG_LEVEL"
        ),
        "frozen_rule": (
            "confirm if at least one prospectively selected top-three first-pass "
            "endpoint completes every fresh seed and has mean score at or below -10.4"
        ),
    }
