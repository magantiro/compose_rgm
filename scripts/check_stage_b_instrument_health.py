#!/usr/bin/env python3
"""Stage B INSTRUMENT HEALTH ONLY. Deliberately blind to every outcome.

WHY THIS SCRIPT EXISTS
----------------------
Stage B runs as two 12-source batches. Between them, only instrument health may
be inspected. Looking at arm effects and then deciding whether batch 2 runs
would turn the bounded half-panel into optional stopping and spend the panel.

Batch 2 runs regardless. This script exists so that rule is enforced by
construction rather than by intention: it reads the shards and emits ONLY the
fields in `HEALTH_FIELDS`. It never reads `U_P`, any violation count, any arm
landing, any trajectory, or any estimand, and
`tests/test_pathwise_stage_b.py::test_health_check_cannot_leak_an_outcome`
asserts the emitted report contains none of them.

WHAT IT CHECKS
--------------
shard completeness - status and mask-leak flags - per-source runtime and kernel
calls - circuit-breaker margin - mask-empty LOGIC (that the recorded flag
agrees with the recorded counts) - support-tight classification consistency -
panel hash agreement.

Everything here would be identical whichever way the experiment comes out.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: The ONLY per-shard keys this script is permitted to read. Anything
#: outcome-bearing is absent on purpose.
HEALTH_FIELDS = (
    "index", "status", "mask_leak", "kernel_calls", "kernel_call_budget",
    "seconds", "median_retention", "support_tight", "support_tight_threshold",
    "mask_empty_states", "mask_empty_fraction_this_source", "arm_run_order",
    "arms_skipped_on_budget", "horizon", "goal", "corridor", "error",
)

#: Substrings that must never appear in the emitted report.
FORBIDDEN = ("U_P", "potency_gain", "landing", "trajectory", "hidden_path",
             "any_intermediate_violation", "endpoint_in_C", "delta_",
             "terminal_cost", "estimand")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--expect", type=int, default=12,
                        help="how many shards this batch should have produced")
    parser.add_argument("--panel", type=Path,
                        default=REPO / "diagnostics/pathwise_stage_b_panel.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    raw = []
    for path in sorted(args.shards.glob("*.json")):
        shard = json.loads(path.read_text())
        raw.append({k: shard.get(k) for k in HEALTH_FIELDS})

    if not raw:
        raise SystemExit(f"no shards under {args.shards}")

    good = [s for s in raw if s["status"] == "SMOKE_HELD_IN"]
    void = [s for s in raw if s["status"] == "INVALID_INSTRUMENT"]
    leaks = {s["index"]: s["mask_leak"] for s in raw if s.get("mask_leak")}

    calls = [float(s["kernel_calls"]) for s in good if s["kernel_calls"] is not None]
    seconds = [float(s["seconds"]) for s in good if s["seconds"] is not None]
    budget = next((s["kernel_call_budget"] for s in good
                   if s["kernel_call_budget"]), None)

    # Mask-empty LOGIC check: a source reporting empty states must also report a
    # positive empty fraction, and vice versa. Disagreement means the recorder
    # is wrong, independent of what the experiment found.
    logic_errors = []
    for shard in good:
        empties = shard.get("mask_empty_states")
        fraction = shard.get("mask_empty_fraction_this_source")
        if empties is None or fraction is None:
            logic_errors.append({"index": shard["index"], "why": "missing mask-empty fields"})
            continue
        if (empties > 0) != (fraction > 0):
            logic_errors.append({"index": shard["index"],
                                 "why": f"empties={empties} but fraction={fraction}"})
        threshold = shard.get("support_tight_threshold")
        retention = shard.get("median_retention")
        tight = shard.get("support_tight")
        if retention is not None and threshold is not None:
            if tight != (retention < threshold):
                logic_errors.append({
                    "index": shard["index"],
                    "why": f"support_tight={tight} but retention={retention} "
                           f"vs threshold={threshold}"})

    skipped = {s["index"]: s["arms_skipped_on_budget"] for s in good
               if s.get("arms_skipped_on_budget")}
    arm_counts = {s["index"]: len(s["arm_run_order"] or []) for s in good}
    incomplete = {i: n for i, n in arm_counts.items() if n != 5}

    panel_sha = None
    if args.panel.exists():
        panel_sha = json.loads(args.panel.read_text())["panel_sha256"]

    report = {
        "schema": "compose.pathwise.stage_b_instrument_health",
        "scope": "INSTRUMENT HEALTH ONLY -- blind to every arm outcome",
        "batch_2_runs_regardless": True,
        "panel_sha256": panel_sha,
        "shards_found": len(raw),
        "shards_expected": args.expect,
        "shards_complete": len(raw) >= args.expect,
        "status_ok": len(good),
        "status_void": len(void),
        "void_reasons": [s.get("error") for s in void] or None,
        "mask_leaks": leaks or None,
        "arms_run_per_source": arm_counts,
        "sources_with_incomplete_arm_set": incomplete or None,
        "arms_skipped_on_budget": skipped or None,
        "kernel_calls": {
            "min": min(calls) if calls else None,
            "median": statistics.median(calls) if calls else None,
            "max": max(calls) if calls else None,
            "budget": budget,
            "circuit_breaker_margin": (
                round(1 - max(calls) / budget, 4) if calls and budget else None),
            "breaker_hit": bool(calls and budget and max(calls) >= budget),
        },
        "seconds": {
            "min": min(seconds) if seconds else None,
            "median": statistics.median(seconds) if seconds else None,
            "max": max(seconds) if seconds else None,
            "total_container_hours": (
                round(sum(seconds) / 3600, 3) if seconds else None),
        },
        "mask_empty_logic_errors": logic_errors or None,
        "support_tight_count": sum(1 for s in good if s.get("support_tight")),
        "healthy": bool(
            len(raw) >= args.expect and not void and not leaks
            and not logic_errors and not incomplete),
    }

    text = json.dumps(report, indent=2)
    for token in FORBIDDEN:
        if token in text:
            raise SystemExit(f"HEALTH REPORT LEAKED AN OUTCOME FIELD: {token}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    print(f"\nHEALTHY: {report['healthy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
