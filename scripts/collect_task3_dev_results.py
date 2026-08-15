#!/usr/bin/env python
"""Collect development-run summaries into one tracked file.

The run directories themselves are working state -- ledgers and checkpoints,
regenerable in under a minute per seed -- and are not tracked. What must survive
is the comparison: which policy, which seeds, what hypervolume, under what
semantics. Written to diagnostics/ so a later claim can be checked against the
numbers that were actually produced rather than remembered.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

OBJECTIVES = ("qed", "jnk3", "sa", "gsk3b", "drd2")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs/task3_dev"))
    parser.add_argument("--out", type=Path,
                        default=Path("diagnostics/task3_dev_baselines.json"))
    args = parser.parse_args()

    summaries = []
    for path in sorted(args.runs.glob("*/summary.json")):
        summaries.append(json.loads(path.read_text()))
    if not summaries:
        raise SystemExit(f"no summary.json under {args.runs}")

    by_policy: dict[str, list[dict]] = {}
    for summary in summaries:
        by_policy.setdefault(summary["policy"], []).append(summary)

    report: dict = {"semantics": "strict-10k",
                    "note": ("development seeds only; the official five-seed run "
                             "is sealed until the policy is frozen"),
                    "policies": {}}
    print(f"{'policy':<14}{'seeds':<8}{'HV mean':<12}{'HV sd':<10}"
          f"{'spent':<9}best jnk3")
    for policy, runs in sorted(by_policy.items()):
        runs.sort(key=lambda r: r["seed"])
        hvs = [r["hypervolume"] for r in runs]
        jnk3 = [r.get("best_per_objective_normalised", {}).get("jnk3")
                for r in runs]
        jnk3 = [value for value in jnk3 if value is not None]
        entry = {
            "seeds": [r["seed"] for r in runs],
            "hypervolume": hvs,
            "hypervolume_mean": statistics.fmean(hvs),
            "hypervolume_stdev": statistics.stdev(hvs) if len(hvs) > 1 else 0.0,
            "spent": [r["spent"] for r in runs],
            "seconds": [round(r["seconds"], 1) for r in runs],
            "best_jnk3": jnk3,
            "budget": runs[0]["budget"],
        }
        report["policies"][policy] = entry
        print(f"{policy:<14}{len(runs):<8}{entry['hypervolume_mean']:<12.4f}"
              f"{entry['hypervolume_stdev']:<10.4f}"
              f"{max(entry['spent']):<9,}"
              f"{max(jnk3) if jnk3 else float('nan'):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
