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
    # Read the horizon from the DATA, not the module constant. An earlier
    # version printed "(H=6)" over an H24 corpus because it used HORIZON.
    H_obs = max(edits) if edits else 0
    print(f"  committed edits: median {np.median(edits):.1f} "
          f"min {min(edits)} max {max(edits)}  (observed H={H_obs})")
    print(f"  complete at H{H_obs}: {term.get('complete',0)}/{len(trajs)} "
          f"= {term.get('complete',0)/max(1,len(trajs)):.4f}")
    print(f"  kernel calls {kernel:,}   per-source seconds median "
          f"{np.median(secs):.1f}")

    # --- the census ------------------------------------------------------
    rows = []
    for region in registered_regions():
        q, s = region
        # THREE DISTINCT EVENTS, never mixed:
        #   qualified@0   the SOURCE already satisfies the region
        #   new_hit       source did NOT qualify; editing first enters at t>=1
        #   lost_after_new_hit   a genuine new hit that no longer qualifies at H
        n_qual0 = n_new_hit = n_retained = n_lost_after = n_term = 0
        hit_steps: list[int] = []
        for t in trajs:
            qed, sim = t["qed"], t["similarity_to_source"]
            complete = t["termination"]["kind"] == "complete"
            terminal = int(complete and qed[-1] >= q and sim[-1] >= s)
            n_term += terminal
            if qed[0] >= q and sim[0] >= s:
                n_qual0 += 1
                continue                    # boundary case, NOT navigation
            hit = first_hit_step(qed, sim, region)
            if hit is not None:
                n_new_hit += 1
                hit_steps.append(hit)       # conditional on a GENUINE new hit
                if terminal:
                    n_retained += 1
                else:
                    n_lost_after += 1
        rows.append({
            "region": f"QED>={q:.2f} & sim>={s:.2f}",
            "qed_threshold": q, "similarity_floor": s,
            "qualified_at_0": n_qual0,
            "new_hit_by_H": n_new_hit,
            "retained_at_H": n_retained,
            "lost_after_new_hit": n_lost_after,
            "terminal_at_H": n_term,
            "new_hit_prevalence": n_new_hit / max(1, len(trajs)),
            # THE LOAD-BEARING QUANTITY for the anytime claim.
            "fraction_of_new_hits_lost": (n_lost_after / n_new_hit)
                                         if n_new_hit else None,
            "first_hit_step_median": (float(np.median(hit_steps))
                                      if hit_steps else None),
            "first_hit_step_counts": dict(Counter(hit_steps)),
            "is_benchmark_region": region == BENCHMARK_REGION,
        })

    print(f"\nCENSUS over {len(trajs)} trajectories "
          f"({len(registered_regions())} frozen regions)")
    print(f"{'region':<26}{'qual@0':>7}{'NEWhit':>7}{'kept':>6}{'LOST':>6}"
          f"{'newhit%':>9}{'lost/new':>9}{'medstep':>8}")
    for r in rows:
        lost = (f"{r['fraction_of_new_hits_lost']:.3f}"
                if r["fraction_of_new_hits_lost"] is not None else "—")
        step = (f"{r['first_hit_step_median']:.1f}"
                if r["first_hit_step_median"] is not None else "—")
        mark = " *" if r["is_benchmark_region"] else ""
        print(f"{r['region']:<26}{r['qualified_at_0']:>7}{r['new_hit_by_H']:>7}"
              f"{r['retained_at_H']:>6}{r['lost_after_new_hit']:>6}"
              f"{r['new_hit_prevalence']:>9.4f}{lost:>9}{step:>8}{mark}")
    print("\n  qual@0  = the SOURCE already satisfied the region; a boundary")
    print("            case for h_hit (= 1 exactly), NOT evidence of navigation")
    print("  NEWhit  = source did NOT qualify and editing found the region")
    print("  LOST    = genuine new hit that no longer qualifies at the horizon")
    print("            <- the load-bearing quantity for the anytime claim")

    bench = next(r for r in rows if r["is_benchmark_region"])
    print(f"\n* benchmark region QED>=0.90 & sim>=0.40: "
          f"qual@0 {bench['qualified_at_0']}  NEW hits {bench['new_hit_by_H']} "
          f"of {len(trajs)}")
    print("  NOTE: on the OFFICIAL benchmark this confound cannot occur -- "
          "sources are QED [0.70,0.80]\n  and success begins at 0.90, so no "
          "benchmark source qualifies at step 0.")

    # --- what TRAINING needs: budget coverage, unique states, gateways -----
    from collections import defaultdict
    by_budget = Counter()
    uniq = set()
    for tj in trajs:
        H = len(tj["path"]) - 1
        for i, s in enumerate(tj["path"]):
            by_budget[H - i] += 1
            uniq.add(s)
    print(f"\nTRAINING SUPPORT")
    print(f"  unique states {len(uniq):,}   prefix examples {sum(by_budget.values()):,}")
    bs = sorted(by_budget)
    print(f"  budgets {bs[0]}..{bs[-1]}; counts at b=0,1,6,12,24: "
          + ", ".join(f"{b}:{by_budget.get(b,0):,}" for b in (0,1,6,12,24)))

    # GATEWAYS: prefixes from which the BENCHMARK region is later reached.
    # These are where conditional continuations should be spent if the tail is
    # starved -- the frozen rule, not a new idea.
    q_b, s_b = BENCHMARK_REGION
    gate = []
    for tj in trajs:
        qed, sim = tj["qed"], tj["similarity_to_source"]
        fh = first_hit_step(qed, sim, BENCHMARK_REGION)
        if fh is None or fh == 0:
            continue
        for i in range(fh):                       # every prefix BEFORE the hit
            gate.append((qed[i], sim[i], fh - i))
    print(f"\nGATEWAYS to {q_b:.2f}/{s_b:.2f}: {len(gate)} prefixes on "
          f"{sum(1 for t in trajs if first_hit_step(t['qed'], t['similarity_to_source'], BENCHMARK_REGION) not in (None,0))} trajectories")
    if gate:
        gq = np.array([g[0] for g in gate]); gs = np.array([g[1] for g in gate])
        gd = np.array([g[2] for g in gate])
        print(f"  QED at gateway  median {np.median(gq):.3f}  "
              f"p10 {np.percentile(gq,10):.3f}  p90 {np.percentile(gq,90):.3f}")
        print(f"  sim at gateway  median {np.median(gs):.3f}")
        print(f"  edits remaining to the hit  median {np.median(gd):.1f}  max {gd.max()}")

    out = {"schema": "compose.hphi.region_census",
           "training_support": {"unique_states": len(uniq),
                                "prefix_examples": sum(by_budget.values()),
                                "by_budget": {str(k): v for k, v in sorted(by_budget.items())}},
           "gateways_to_benchmark": {
               "n_prefixes": len(gate),
               "qed_median": float(np.median([g[0] for g in gate])) if gate else None,
               "sim_median": float(np.median([g[1] for g in gate])) if gate else None,
               "edits_to_hit_median": float(np.median([g[2] for g in gate])) if gate else None},
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
