#!/usr/bin/env python3
"""Stage A2 gate verdicts: cLogP corridor prevalence and viability. LOCAL ONLY.

THE SOURCE IS THE INDEPENDENT UNIT
----------------------------------
12 sources x 6 rollouts is 72 trajectories but **12 observations**. Rollouts
from one molecule are repeated measures. Every interval here is a
**source-clustered bootstrap**: resample SOURCES with replacement, keeping each
source's rollouts together. A trajectory-level interval would be roughly
sqrt(6) too narrow and would make a phenomenon carried by two molecules look
like a population fact.

Trajectory-level incidence is still reported, because the criteria are stated
in those terms -- but it is never given an interval of its own.

HONEST FRAMING, REPEATED HERE SO IT TRAVELS WITH THE NUMBERS
------------------------------------------------------------
Family B was selected for follow-up AFTER the three-family feasibility census
because it alone exhibited the intended reversible-excursion mechanism. Stage
A2 is developmental follow-up, not independent confirmation of the phenomenon.
The stage-A FAIL verdict is not revised by anything in this file.

Status of the emitted artifact: SMOKE_HELD_IN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# ---- criteria, fixed in PROTOCOL.md before the run ------------------------
V3_MIN_EVENTS = 20              # retained verbatim from the census
V4A_MIN_MEDIAN_RETENTION = 0.10
V4B_MAX_EMPTY_FRACTION = 0.05
V5A_MIN_SOURCE_FRACTION = 1 / 3
V5B_MAX_SINGLE_SOURCE_SHARE = 0.50
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260814
# ---------------------------------------------------------------------------


def summarise(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(ordered), 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def cluster_bootstrap(per_source: list[float], draws: int = BOOTSTRAP_DRAWS) -> dict:
    """Resample SOURCES with replacement. The unit of resampling is the unit of
    independence; that is the whole point of doing it this way."""
    if not per_source:
        return {"n_sources": 0}
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(per_source)
    means = []
    for _ in range(draws):
        sample = [per_source[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    return {
        "n_sources": n,
        "point": round(statistics.fmean(per_source), 4),
        "ci95_low": round(means[int(0.025 * draws)], 4),
        "ci95_high": round(means[int(0.975 * draws)], 4),
        "draws": draws,
        "unit": "source (rollouts kept together)",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument(
        "--panel", type=Path, default=REPO / "diagnostics/pathwise_a2_panel.json")
    parser.add_argument(
        "--out", type=Path,
        default=REPO / "diagnostics/pathwise_a2_corridor_prevalence.json")
    args = parser.parse_args()

    shards = [json.loads(p.read_text()) for p in sorted(args.shards.glob("*.json"))]
    if not shards:
        raise SystemExit(f"no shards under {args.shards}")
    void = [s for s in shards if s.get("status") == "INVALID_INSTRUMENT"]
    good = [s for s in shards if s.get("status") == "SMOKE_HELD_IN"]
    if not good:
        raise SystemExit(f"every shard is INVALID_INSTRUMENT ({len(void)})")

    # ---------------- trajectory level (no interval; not independent) -------
    rollouts = [r for s in good for r in s["rollouts"]]
    n_traj = len(rollouts)
    violators = [r for r in rollouts if r["audit"]["any_violation"]]
    events_traj = [r for r in rollouts if r["audit"]["endpoint_valid_path_invalid"]]
    total_events = len(events_traj)
    return_rate = len(events_traj) / len(violators) if violators else 0.0

    # ---------------- source level (the independent unit) ------------------
    per_source_event_rate: list[float] = []
    per_source_violation_rate: list[float] = []
    per_source_has_event: list[float] = []
    events_by_source: dict[int, int] = {}
    for shard in good:
        rolls = shard["rollouts"]
        denom = len(rolls) or 1
        ev = sum(1 for r in rolls if r["audit"]["endpoint_valid_path_invalid"])
        vi = sum(1 for r in rolls if r["audit"]["any_violation"])
        events_by_source[shard["index"]] = ev
        per_source_event_rate.append(ev / denom)
        per_source_violation_rate.append(vi / denom)
        per_source_has_event.append(1.0 if ev else 0.0)

    n_sources = len(good)
    sources_with_event = int(sum(per_source_has_event))
    source_fraction = sources_with_event / n_sources if n_sources else 0.0
    top_share = (max(events_by_source.values()) / total_events
                 if total_events else 0.0)

    # ---------------- excursion geometry -----------------------------------
    depths, durations = [], []
    for r in rollouts:
        for exc in r.get("excursions", []):
            if exc.get("max_depth") is not None:
                depths.append(exc["max_depth"])
            durations.append(float(exc["duration"]))

    # ---------------- V4: what the mask leaves behind ----------------------
    retentions, empties, states = [], 0, 0
    per_source_retention: list[float] = []
    for shard in good:
        local = [c["retention"] for c in shard["mask_census"]
                 if c["retention"] is not None]
        retentions += local
        empties += sum(1 for c in shard["mask_census"] if c["empty_after_mask"])
        states += len(shard["mask_census"])
        if local:
            per_source_retention.append(statistics.median(local))
    median_retention = statistics.median(retentions) if retentions else 0.0
    empty_fraction = empties / states if states else 0.0

    v3 = total_events >= V3_MIN_EVENTS
    v4a = median_retention >= V4A_MIN_MEDIAN_RETENTION
    v4b = empty_fraction <= V4B_MAX_EMPTY_FRACTION
    v5a = source_fraction >= V5A_MIN_SOURCE_FRACTION
    v5b = top_share <= V5B_MAX_SINGLE_SOURCE_SHARE
    verdict = "PASS" if all((v3, v4a, v4b, v5a, v5b)) else "FAIL"

    failures = [name for name, ok in (
        ("V3_event_yield", v3), ("V4a_median_retention", v4a),
        ("V4b_mask_empty_rare", v4b), ("V5a_source_spread", v5a),
        ("V5b_no_single_source_dominates", v5b)) if not ok]

    payload = {
        "schema": "compose.pathwise.a2_prevalence",
        "status": "SMOKE_HELD_IN",
        "held_out_opened": False,
        "framing": (
            "Family B was selected for follow-up AFTER the three-family "
            "feasibility census because it alone exhibited the intended "
            "reversible-excursion mechanism. Stage A2 is developmental "
            "follow-up, not independent confirmation of the phenomenon."
        ),
        "stage_a_verdict_unchanged": "FAIL",
        "panel_sha256": (json.loads(args.panel.read_text())["panel_sha256"]
                         if args.panel.exists() else None),
        "shards": len(shards),
        "shards_void": len(void),
        "void_reasons": [s.get("error") for s in void] or None,
        "independent_unit": "source",
        "trajectory_level": {
            "trajectories": n_traj,
            "violating": len(violators),
            "violation_incidence": round(len(violators) / n_traj, 4) if n_traj else 0.0,
            "return_rate_among_violators": round(return_rate, 4),
            "endpoint_valid_path_invalid_events": total_events,
            "note": ("no interval reported: trajectories from one source are "
                     "repeated measures, not independent observations"),
        },
        "source_level": {
            "sources": n_sources,
            "sources_with_at_least_one_event": sources_with_event,
            "fraction_of_sources_with_event": round(source_fraction, 4),
            "events_by_source": events_by_source,
            "largest_single_source_share_of_events": round(top_share, 4),
            "bootstrap_fraction_of_sources_with_event":
                cluster_bootstrap(per_source_has_event),
            "bootstrap_per_source_event_rate":
                cluster_bootstrap(per_source_event_rate),
            "bootstrap_per_source_violation_rate":
                cluster_bootstrap(per_source_violation_rate),
        },
        "excursion_geometry": {
            "max_depth_logp": summarise(depths),
            "duration_steps": summarise(durations),
            "note": ("duration is the number of consecutive states outside the "
                     "corridor; a multi-step excursion is budget spent inside a "
                     "forbidden region, invisible to endpoint-only filtering"),
        },
        "mask_viability_V4": {
            "states_measured": states,
            "median_retention": round(median_retention, 4),
            "retention_summary": summarise(retentions),
            "per_source_median_retention": summarise(per_source_retention),
            "mask_empty_states": empties,
            "mask_empty_fraction": round(empty_fraction, 4),
        },
        "criteria": {
            "V3_event_yield": {"threshold": V3_MIN_EVENTS,
                               "observed": total_events, "pass": v3},
            "V4a_median_retention": {"threshold": V4A_MIN_MEDIAN_RETENTION,
                                     "observed": round(median_retention, 4),
                                     "pass": v4a},
            "V4b_mask_empty_rare": {"threshold": V4B_MAX_EMPTY_FRACTION,
                                    "observed": round(empty_fraction, 4),
                                    "pass": v4b},
            "V5a_source_spread": {"threshold": round(V5A_MIN_SOURCE_FRACTION, 4),
                                  "observed": round(source_fraction, 4),
                                  "pass": v5a},
            "V5b_no_single_source_dominates": {
                "threshold": V5B_MAX_SINGLE_SOURCE_SHARE,
                "observed": round(top_share, 4), "pass": v5b},
        },
        "verdict": verdict,
        "failed_criteria": failures or None,
        "consequence": (
            "advance to a causal source-level pathwise-control experiment"
            if verdict == "PASS" else
            "pathwise constraints close for good; no fourth predicate is searched"
        ),
        "cost": {
            "kernel_calls": summarise([float(s["kernel_calls"]) for s in good]),
            "seconds": summarise([float(s["seconds"]) for s in good]),
        },
    }
    payload["analysis_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["criteria"], indent=2))
    print(f"\nVERDICT {verdict}"
          + (f"  failed: {', '.join(failures)}" if failures else ""))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
