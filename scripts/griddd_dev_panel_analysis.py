"""Read the 64x5 QED development panel under the frozen judging criterion.

Criterion frozen in `docs/GRIDDD_JIN_PROTOCOL.md` BEFORE any dev number was
read:

  PRIMARY    per-trajectory benchmark event 1[QED >= 0.9 AND Tanimoto >= 0.4]
             over all 320 trajectories, source-clustered.
  SECONDARY  terminal QED and Tanimoto, reported SEPARATELY, so a policy
             cannot "improve" by raising QED while destroying similarity.
  NO combined scalar score. The benchmark event is a conjunction; it stays one.
  Best-of-5 source success is DESCRIPTIVE ONLY and must not drive selection.

Development data. NEVER reported as a benchmark result.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

QED_TARGET, SIM_FLOOR = 0.9, 0.4


def source_clustered_ci(per_source: list[list[int]], draws: int = 4000,
                        seed: int = 0) -> tuple[float, float, float]:
    """Resample SOURCES, not trajectories -- trajectories within a source are
    not independent, so a naive trajectory bootstrap would understate width."""
    rng = np.random.default_rng(seed)
    n = len(per_source)
    means = np.empty(draws)
    for d in range(draws):
        idx = rng.integers(0, n, n)
        vals = [v for i in idx for v in per_source[i]]
        means[d] = float(np.mean(vals)) if vals else 0.0
    flat = [v for s in per_source for v in s]
    return (float(np.mean(flat)),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("/tmp/dev_panel.json.gz"))
    args = ap.parse_args()

    payload = json.loads(gzip.decompress(args.run.read_bytes()).decode())
    results = [r for r in payload["results"] if r.get("status") == "OK"]
    print(f"sources {len(results)}   policy: {payload.get('policy_name')}")
    print(f"{payload.get('not_')}\n")

    qualifies, qed, sim, per_source = [], [], [], []
    dead, zero_denom, kernel, logical = 0, 0, 0, 0
    for r in results:
        row = []
        for t in r["per_replicate"]:
            q = bool(t["qualifies"])
            row.append(int(q))
            qualifies.append(int(q))
            qed.append(float(t["terminal_qed"]))
            sim.append(float(t["similarity"]))
            if t.get("dead_end"):
                dead += 1
        per_source.append(row)
        zero_denom += int(r.get("zero_denominator_events", 0))
        kernel += int(r.get("kernel_calls", 0))
        logical += int(r.get("logical_fiber_requests", 0))

    qed_a, sim_a = np.array(qed), np.array(sim)
    n_traj = len(qualifies)

    m, lo, hi = source_clustered_ci(per_source)
    print("PRIMARY -- per-trajectory benchmark event 1[QED>=0.9 AND sim>=0.4]")
    print(f"  {sum(qualifies)}/{n_traj} trajectories = {m:.4f}"
          f"   source-clustered 95% CI [{lo:.4f}, {hi:.4f}]\n")

    print("SECONDARY -- reported SEPARATELY, never combined")
    for name, a, thr in (("terminal QED", qed_a, QED_TARGET),
                         ("Tanimoto", sim_a, SIM_FLOOR)):
        print(f"  {name:<13} median {np.median(a):.4f}  mean {a.mean():.4f}"
              f"  p10 {np.percentile(a,10):.4f}  p90 {np.percentile(a,90):.4f}"
              f"   >= {thr}: {float((a>=thr).mean()):.4f}")

    both = float(((qed_a >= QED_TARGET) & (sim_a >= SIM_FLOOR)).mean())
    print(f"\n  which constraint binds?  QED>=0.9 alone {float((qed_a>=QED_TARGET).mean()):.4f}"
          f" | sim>=0.4 alone {float((sim_a>=SIM_FLOOR).mean()):.4f} | both {both:.4f}")

    best5 = float(np.mean([1.0 if any(s) else 0.0 for s in per_source]))
    print(f"\nDESCRIPTIVE ONLY -- best-of-5 source success: {best5:.4f} "
          f"({sum(1 for s in per_source if any(s))}/{len(per_source)} sources)")
    print("  (not used for policy selection; 320 paired trajectories are)")

    print(f"\nresources: {kernel:,} kernel calls for {logical:,} logical requests"
          f"  ({1 - kernel/max(1,logical):.1%} saved by memoization)")
    print(f"dead ends {dead}/{n_traj}   zero-denominator fallbacks {zero_denom}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
