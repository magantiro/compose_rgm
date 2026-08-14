"""Read-only 20-region census over an h_phi rollout corpus.

SEPARATE FROM THE GENERATOR BY DESIGN. The runner writes raw trajectories and
computes nothing about event prevalence; this script computes the census
afterwards. That separation is what makes "the goal family was frozen before
the census" true in software rather than only in a document.

Written BEFORE the pilot data existed, so the reading procedure cannot be
shaped by what the numbers turn out to be.

Reports per region, per the frozen protocol:

  * terminal-at-H6 prevalence
  * hit-by-H6 prevalence
  * distribution of first-hit step
  * FRACTION OF HITS SUBSEQUENTLY LOST BY H6

The last is the scientifically interesting one. If many trajectories reach a
region and then fall back out, forced fixed-horizon semantics are actively
wasting COMPOSE's strongest structural feature -- that every intermediate is
already a usable molecule.

THE 20 REGIONS DO NOT MOVE IN RESPONSE TO THIS OUTPUT.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compose_v4.experiments.hphi_rollout import (  # noqa: E402
    BENCHMARK_REGION,
    HORIZON,
    first_hit_step,
    registered_regions,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("/tmp/hphi_pilot.json.gz"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    payload = json.loads(gzip.decompress(args.run.read_bytes()).decode())
    sources = [r for r in payload["results"] if r.get("status") == "OK"]
    trajs = [t for r in sources for t in r["trajectories"]]
    print(f"sources {len(sources)}   trajectories {len(trajs)}   "
          f"corpus {payload.get('corpus_version')}")

    # --- mechanics, before any region statistic -------------------------
    term = Counter(t["termination"]["kind"] for t in trajs)
    edits = [t["termination"]["edits"] for t in trajs]
    kernel = sum(int(r.get("kernel_calls", 0)) for r in sources)
    secs = [float(r.get("seconds", 0.0)) for r in sources]
    print(f"\nMECHANICS")
    print(f"  termination: {dict(term)}")
    print(f"  committed edits: median {np.median(edits):.1f} "
          f"min {min(edits)} max {max(edits)}  (H={HORIZON})")
    print(f"  complete at H{HORIZON}: {term.get('complete',0)}/{len(trajs)} "
          f"= {term.get('complete',0)/max(1,len(trajs)):.4f}")
    print(f"  kernel calls {kernel:,}   per-source seconds median "
          f"{np.median(secs):.1f}")

    # --- the census ------------------------------------------------------
    rows = []
    for region in registered_regions():
        q, s = region
        n_term = n_hit = n_lost = 0
        hit_steps: list[int] = []
        for t in trajs:
            qed, sim = t["qed"], t["similarity_to_source"]
            complete = t["termination"]["kind"] == "complete"
            hit = first_hit_step(qed, sim, region)
            terminal = int(complete and qed[-1] >= q and sim[-1] >= s)
            n_term += terminal
            if hit is not None:
                n_hit += 1
                hit_steps.append(hit)
                if not terminal:
                    n_lost += 1                # reached it, then fell back out
        rows.append({
            "region": f"QED>={q:.2f} & sim>={s:.2f}",
            "qed_threshold": q, "similarity_floor": s,
            "terminal_at_H6": n_term,
            "hit_by_H6": n_hit,
            "hits_lost_by_H6": n_lost,
            "terminal_prevalence": n_term / max(1, len(trajs)),
            "hit_prevalence": n_hit / max(1, len(trajs)),
            "fraction_of_hits_lost": (n_lost / n_hit) if n_hit else None,
            "first_hit_step_median": (float(np.median(hit_steps))
                                      if hit_steps else None),
            "first_hit_step_counts": dict(Counter(hit_steps)),
            "is_benchmark_region": region == BENCHMARK_REGION,
        })

    print(f"\nCENSUS over {len(trajs)} trajectories "
          f"({len(registered_regions())} frozen regions)")
    print(f"{'region':<26}{'terminal':>9}{'hit':>7}{'lost':>7}"
          f"{'hit%':>8}{'lost/hit':>10}{'med step':>9}")
    for r in rows:
        lost = f"{r['fraction_of_hits_lost']:.3f}" if r["fraction_of_hits_lost"] is not None else "—"
        step = f"{r['first_hit_step_median']:.1f}" if r["first_hit_step_median"] is not None else "—"
        mark = " *" if r["is_benchmark_region"] else ""
        print(f"{r['region']:<26}{r['terminal_at_H6']:>9}{r['hit_by_H6']:>7}"
              f"{r['hits_lost_by_H6']:>7}{r['hit_prevalence']:>8.4f}"
              f"{lost:>10}{step:>9}{mark}")

    bench = next(r for r in rows if r["is_benchmark_region"])
    print(f"\n* benchmark region QED>=0.90 & sim>=0.40:")
    print(f"    terminal {bench['terminal_at_H6']}  hit {bench['hit_by_H6']}"
          f"  of {len(trajs)} trajectories")

    out = {"schema": "compose.hphi.region_census",
           "note": "READ-ONLY. The 20 regions do not move in response to this.",
           "n_sources": len(sources), "n_trajectories": len(trajs),
           "mechanics": {"termination": dict(term),
                         "median_edits": float(np.median(edits)),
                         "kernel_calls": kernel},
           "regions": rows}
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
