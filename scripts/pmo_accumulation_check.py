#!/usr/bin/env python
"""Does the controller's small-step proposal distribution ACCUMULATE into the
transformation the productive-basin census requires?

This is the strongest objection to reading the single-proposal delta against a
whole-journey delta: a proposal is one macro option, a productive delta is the
entire remaining transport, so of course one is smaller.  The objection is
testable.  For each blind run, walk the committed parent chain from the
best-scoring molecule back to its lineage root and characterise the
transformation the whole 250-call budget actually accumulated.

ZERO ORACLE CALLS.
"""

from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compose_v4.experiments.global_delta_census import compute_global_delta  # noqa: E402

ATLAS = "diagnostics/pmo_atlas_v1"
OUT = "diagnostics/pmo_global_delta_census_v1"


def lineages() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(ATLAS, "test_c_blind_*.json"))):
        with open(path) as handle:
            payload = json.load(handle)["payload"]
        for run in payload["runs"]:
            trajectory = run["trajectory"]
            parent = {t["endpoint"]: t.get("parent_endpoint") for t in trajectory}
            initialization = {
                t["endpoint"] for t in trajectory if t.get("role") == "initialization"
            }
            best = max(trajectory, key=lambda t: t["score"])
            current, seen, depth = best["endpoint"], set(), 0
            while parent.get(current) and current not in seen:
                seen.add(current)
                current = parent[current]
                depth += 1
            rows.append(
                {
                    "task": run["task"],
                    "lineage_root": current,
                    "best_endpoint": best["endpoint"],
                    "lineage_depth": depth,
                    "root_is_initialization": current in initialization,
                    "best_score": best["score"],
                    "charged_calls": run["charged_calls"],
                }
            )
    return rows


def main() -> None:
    rows = lineages()
    print(f"[strata] blind runs: {len(rows)}  (one lineage per run)")
    records = []
    for row in rows:
        delta = compute_global_delta(row["lineage_root"], row["best_endpoint"])
        records.append({**row, "delta": delta.as_dict()})
        print(
            f"  {row['task']:26s} depth {row['lineage_depth']:2d} of "
            f"{row['charged_calls']:3d} calls   largest_region "
            f"{delta.largest_changed_region:2d}  retained "
            f"{delta.retained_fraction_source:.2f}  score {row['best_score']:.3f}"
        )

    deltas = [r["delta"] for r in records]
    n = len(deltas)
    summary = {
        "n_runs": n,
        "lineage_depth_median": st.median(r["lineage_depth"] for r in rows),
        "lineage_depth_max": max(r["lineage_depth"] for r in rows),
        "runs_whose_best_is_an_initialization_molecule": sum(
            1 for r in rows if r["lineage_depth"] == 0
        ),
        "accumulated_largest_changed_region_median": st.median(
            d["largest_changed_region"] for d in deltas
        ),
        "accumulated_frac_largest_region_ge_6": round(
            sum(1 for d in deltas if d["largest_changed_region"] >= 6) / n, 4
        ),
        "accumulated_frac_largest_region_ge_13": round(
            sum(1 for d in deltas if d["largest_changed_region"] >= 13) / n, 4
        ),
        "accumulated_total_changed_atoms_median": st.median(
            d["total_changed_atoms"] for d in deltas
        ),
        "accumulated_retained_fraction_median": round(
            st.median(d["retained_fraction_source"] for d in deltas), 4
        ),
    }
    print("\n=== ACCUMULATED over the full 250-call budget ===")
    for key, value in summary.items():
        print(f"  {key:52s} {value}")
    print("\n  reference: single proposal   median largest region 3,  retained 0.968")
    print("  reference: productive need   median largest region 14, retained 0.40")

    payload = {
        "schema_version": "pmo_accumulation_check_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "information_regime": "DEVELOPMENT_INFORMED_DIAGNOSTIC",
        "new_oracle_calls": 0,
        "question": (
            "Does the small-step proposal distribution accumulate over 250 charged calls into "
            "the transformation the productive-basin census requires?"
        ),
        "summary": summary,
        "records": records,
    }
    with open(os.path.join(OUT, "accumulation_check_v1.json"), "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
    print(f"\nwrote {OUT}/accumulation_check_v1.json")


if __name__ == "__main__":
    main()
