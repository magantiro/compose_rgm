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


def best_from_ledger(path: Path) -> dict[str, float]:
    """Per-objective maxima, read back off the durable ledger."""

    if not path.exists():
        return {}
    best = [0.0] * len(OBJECTIVES)
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            values = json.loads(line)["v"]
            best = [max(a, b) for a, b in zip(best, values)]
    return dict(zip(OBJECTIVES, best))


def paired_comparisons(by_policy: dict[str, list[dict]]) -> dict:
    """Compare policies SEED BY SEED, because the seeds are shared.

    Both policies start a given seed from the identical frozen initialization
    set, so the seed-to-seed variation -- which is enormous here, sd 0.2 against
    differences of the same order -- is common to both and cancels in the
    difference. Comparing two unpaired means over eight seeds would be close to
    uninformative; comparing eight paired differences is not.
    """

    names = sorted(by_policy)
    out: dict = {}
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            left = {r["seed"]: r["hypervolume"] for r in by_policy[first]}
            right = {r["seed"]: r["hypervolume"] for r in by_policy[second]}
            seeds = sorted(set(left) & set(right))
            if len(seeds) < 2:
                continue
            differences = [right[s] - left[s] for s in seeds]
            wins = sum(1 for d in differences if d > 0)
            entry = {
                "seeds": seeds,
                "difference": {str(s): right[s] - left[s] for s in seeds},
                "mean_difference": statistics.fmean(differences),
                "stdev_difference": (statistics.stdev(differences)
                                     if len(differences) > 1 else 0.0),
                "wins_for_second": wins, "n": len(seeds),
                "direction": f"{second} minus {first}",
            }
            try:
                from scipy.stats import wilcoxon
                if len(differences) >= 5:
                    entry["wilcoxon_p"] = float(
                        wilcoxon(differences).pvalue)
            except Exception:  # noqa: BLE001 - a missing test is not a failure
                pass
            out[f"{second}_vs_{first}"] = entry
            print(f"\npaired {second} minus {first} over {len(seeds)} seeds: "
                  f"mean {entry['mean_difference']:+.4f} "
                  f"(sd {entry['stdev_difference']:.4f}), "
                  f"{wins}/{len(seeds)} seeds favour {second}"
                  + (f", wilcoxon p={entry['wilcoxon_p']:.3f}"
                     if "wilcoxon_p" in entry else ""))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs/task3_dev"))
    parser.add_argument("--out", type=Path,
                        default=Path("diagnostics/task3_dev_baselines.json"))
    args = parser.parse_args()

    summaries = []
    for path in sorted(args.runs.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        if "best_per_objective_normalised" not in summary:
            # Older runs predate the field. The ledger is the record of what was
            # actually evaluated, so recompute rather than leave a hole.
            summary["best_per_objective_normalised"] = best_from_ledger(
                path.with_name("evaluations.jsonl"))
        summaries.append(summary)
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
        paired = [(r["hypervolume"],
                   r.get("best_per_objective_normalised", {}).get("jnk3"))
                  for r in runs]
        paired = [(h, j) for h, j in paired if j is not None]
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
            # HV on this task is dominated by the JNK3 axis -- the other four
            # objectives are nearly saturated by ordinary drug-like molecules --
            # so this correlation is the single most useful diagnostic here.
            "hv_vs_best_jnk3_correlation":
                (statistics.correlation([h for h, _ in paired],
                                        [j for _, j in paired])
                 if len(paired) > 2 else None),
        }
        report["policies"][policy] = entry
        print(f"{policy:<14}{len(runs):<8}{entry['hypervolume_mean']:<12.4f}"
              f"{entry['hypervolume_stdev']:<10.4f}"
              f"{max(entry['spent']):<9,}"
              f"{max(jnk3) if jnk3 else float('nan'):.3f}")

    report["paired"] = paired_comparisons(by_policy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
