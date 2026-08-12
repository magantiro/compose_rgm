"""Report the verified-hybrid frontier: recovery retained per continuation spent.

Four things, as agreed before the run:

  1. exact recovery per arm;
  2. how many of full rollout's extra recoveries over greedy are retained;
  3. expensive continuation evaluations (and kernel calls);
  4. rescue-retention = (R_hybrid - R_greedy) / (R_full - R_greedy).

Also checks the safety property that motivates the whole construction: because
greedy is always in the shortlist and the expensive continuation makes the final
call under the strict-improvement rule, no hybrid arm should ever lose a pair
that greedy recovers. A violation would be a bug, not a finding, and is flagged
as such.

Continuation counts are reported per pair AND as a fraction of the candidate
universe the arm could have evaluated. Per-pair totals scale with trajectory
length, so quoting them alone would misstate the compute claim; the fraction is
what the shortlist actually controls, and is 1.0 for full rollout by
definition.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ORDER = ("greedy", "full", "sim1", "sim2", "ref1", "ref2", "hphi1", "hphi2")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    files = [f for f in glob.glob(str(args.shards / "**" / "*.json"), recursive=True)
             if "aggregate" not in f]
    rows = [json.loads(Path(f).read_text()) for f in files]
    if not rows:
        raise SystemExit(f"no shards under {args.shards}")
    print(f"{len(rows)} pairs\n")

    greedy = sum(r["arms"]["greedy"]["recovered"] for r in rows)
    full = sum(r["arms"]["full"]["recovered"] for r in rows)
    headroom = full - greedy

    print(f"{'arm':>8} {'recovered':>10} {'retained':>9} {'retention':>10} "
          f"{'cont/pair':>10} {'% of universe':>14} {'lost vs greedy':>15}")
    summary = {}
    for name in ORDER:
        recovered = sum(r["arms"][name]["recovered"] for r in rows)
        evaluations = sum(r["arms"][name]["continuation_evaluations"] for r in rows)
        universe = sum(r["arms"][name]["universe_candidates_seen"] for r in rows)
        per_pair = evaluations / len(rows)
        # The compute claim, stated directly: what FRACTION of the available
        # candidate universe did this arm actually pay a continuation for?
        # 1.0 for `full` by definition. Per-pair totals scale with trajectory
        # length, so quoting them alone would misstate what the shortlist does.
        fraction = (evaluations / universe) if universe else None
        lost = sum(1 for r in rows
                   if r["arms"]["greedy"]["recovered"]
                   and not r["arms"][name]["recovered"])
        retention = ((recovered - greedy) / headroom) if headroom else None
        summary[name] = {
            "recovered": recovered,
            "extra_over_greedy": recovered - greedy,
            "rescue_retention": retention,
            "continuations_total": evaluations,
            "continuations_per_pair": per_pair,
            "universe_candidates_seen": universe,
            "fraction_of_universe_evaluated": fraction,
            "lost_versus_greedy": lost,
            "mean_best_similarity": float(np.mean(
                [r["arms"][name]["best"] for r in rows])),
            "overrides": sum(r["arms"][name]["overrides"] for r in rows),
        }
        print(f"{name:>8} {recovered:>10} {recovered - greedy:>9} "
              f"{'n/a' if retention is None else f'{retention:>9.0%}'} "
              f"{per_pair:>10.1f} "
              f"{'n/a' if fraction is None else f'{fraction:>13.0%}'} {lost:>15}")

    print(f"\nheadroom: full rollout recovers {headroom} more than greedy "
          f"({greedy} -> {full} of {len(rows)})")

    violations = [n for n in ORDER if summary[n]["lost_versus_greedy"] > 0]
    print("\nSAFETY PROPERTY -- greedy is always in the shortlist and the expensive")
    print("continuation makes the final call, so no arm should lose a greedy win.")
    if violations:
        print(f"  VIOLATED by {violations} -- treat as a BUG, not a finding.")
    else:
        print("  HELD for every arm.")

    print("\nmean best similarity")
    for name in ORDER:
        print(f"  {name:>8} {summary[name]['mean_best_similarity']:.4f}")

    print(f"\nkernel calls total {sum(r['kernel_calls'] for r in rows):,}")

    args.out.write_text(json.dumps({
        "schema": "compose.editing_v2.verified_hybrid",
        "status": "DEVELOPMENT_24_SEALED_67_UNTOUCHED",
        "universe": ("Experiment C's exact set: greedy + top-4 similarity + "
                     "top-2 R_theta, deduplicated, no random stratum"),
        "k_meaning": "K counts NON-GREEDY challengers; greedy always included",
        "pairs": len(rows),
        "greedy_recovered": greedy,
        "full_recovered": full,
        "headroom": headroom,
        "safety_property_held": not violations,
        "arms": summary,
        "per_pair": rows,
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
