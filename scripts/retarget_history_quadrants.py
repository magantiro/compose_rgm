"""Does the value of molecular history survive a change of controller class?

Q2 as run compares verified-from-x_3 against verified-from-x_0. That contrast
already had controller parity, so it is valid on its own. This script adds the
greedy-matched partner and asks the strictly stronger question:

    Delta_greedy   = U_B(greedy   from x_3) - U_B(greedy   restart from x_0)
    Delta_verified = U_B(verified from x_3) - U_B(verified restart from x_0)

and classifies every source into four quadrants:

    helps under both              history itself carries value
    helps only under verified     the prefix carries state that MYOPIC control
                                  misvalues; exploiting it needs lookahead
    hurts only under verified     inspect for a systematic bad commitment by
                                  the rollout value, or ordinary noise
    hurts under both              earlier optimisation genuinely moved the
                                  molecule somewhere poorly positioned for the
                                  new requirement

REPORTING ORDER IS DELIBERATE. Quadrant COUNTS come first, individual sources
second. A single discordant source is not a subgroup; the question is whether
the discordance repeats. Naming a source before knowing how many share its
behaviour is how a post-hoc subgroup claim gets born.

This arm was added AFTER the verified Q2 result was seen, so everything here is
development robustness, not confirmation.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

EPS = 1e-9


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", required=True, type=Path)
    parser.add_argument("--greedy-restart", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    arms = {}
    for f in glob.glob(str(args.arms / "*.json")):
        if f.endswith(".partial.json"):
            continue
        r = json.loads(Path(f).read_text())
        arms[(r["history"], r["index"])] = r
    gr = {}
    for f in glob.glob(str(args.greedy_restart / "*.json")):
        r = json.loads(Path(f).read_text())
        gr[(r["history"], r["index"])] = r

    shared = sorted(set(arms) & set(gr))
    print(f"{len(arms)} branch points, {len(gr)} greedy-restart, {len(shared)} joined\n")
    if not shared:
        raise SystemExit("no overlap yet")

    rows = []
    for key in shared:
        a, g = arms[key], gr[key]
        dg = a["arms"]["greedy_retarget"]["b_worst_margin"] - g["b_worst_margin"]
        dv = (a["arms"]["verified_retarget"]["b_worst_margin"]
              - a["arms"]["restart"]["b_worst_margin"])
        quadrant = ("helps under both" if dg > EPS and dv > EPS else
                    "helps only under verified" if dv > EPS else
                    "hurts only under verified" if dg > EPS else
                    "hurts under both")
        rows.append({"history": key[0], "index": key[1], "source": a["source"],
                     "delta_greedy": float(dg), "delta_verified": float(dv),
                     "quadrant": quadrant})

    print("QUADRANT COUNTS -- asked before any individual source is named")
    print(f"{'quadrant':>28} {'P-first':>8} {'D-first':>8} {'total':>7}")
    order = ("helps under both", "helps only under verified",
             "hurts only under verified", "hurts under both")
    counts = {}
    for q in order:
        p = sum(1 for r in rows if r["quadrant"] == q and r["history"] == "P")
        d = sum(1 for r in rows if r["quadrant"] == q and r["history"] == "D")
        counts[q] = {"P": p, "D": d, "total": p + d}
        print(f"{q:>28} {p:>8} {d:>8} {p + d:>7}")

    print("\nPAIRED MEANS by controller class")
    summary = {}
    for h in ("P", "D"):
        sub = [r for r in rows if r["history"] == h]
        if not sub:
            continue
        dg = np.array([r["delta_greedy"] for r in sub])
        dv = np.array([r["delta_verified"] for r in sub])
        summary[h] = {"n": len(sub),
                      "delta_greedy_mean": float(dg.mean()),
                      "delta_greedy_wins": int((dg > EPS).sum()),
                      "delta_verified_mean": float(dv.mean()),
                      "delta_verified_wins": int((dv > EPS).sum())}
        print(f"  {h}-first (n={len(sub)}): greedy {dg.mean():+.3f} "
              f"({int((dg > EPS).sum())}W/{int((dg < -EPS).sum())}L)   "
              f"verified {dv.mean():+.3f} "
              f"({int((dv > EPS).sum())}W/{int((dv < -EPS).sum())}L)")

    # Only now, after the counts are known, look at individual sources.
    print("\nSOURCES WHERE HISTORY HURTS UNDER BOTH CONTROLLERS")
    both = [r for r in rows if r["quadrant"] == "hurts under both"]
    if not both:
        print("  none -- no source is hurt by its history under both controller classes")
    for r in sorted(both, key=lambda r: r["delta_verified"]):
        print(f"  {r['history']}-first src {r['index']:>2}  "
              f"greedy {r['delta_greedy']:+.3f}  verified {r['delta_verified']:+.3f}")

    print("\nDISCORDANT SOURCES (the two controller classes disagree in sign)")
    disc = [r for r in rows if (r["delta_greedy"] > EPS) != (r["delta_verified"] > EPS)]
    print(f"  {len(disc)}/{len(rows)} sources disagree")
    for r in sorted(disc, key=lambda r: abs(r["delta_verified"] - r["delta_greedy"]),
                    reverse=True)[:6]:
        print(f"  {r['history']}-first src {r['index']:>2}  "
              f"greedy {r['delta_greedy']:+.3f}  verified {r['delta_verified']:+.3f}  "
              f"[{r['quadrant']}]")

    args.out.write_text(json.dumps({
        "schema": "compose.retarget.history_quadrants",
        "status": "DEVELOPMENT_ROBUSTNESS_NOT_CONFIRMATION",
        "note": ("greedy_restart was added after the verified Q2 result was "
                 "seen. It does not rescue Q2, which already had controller "
                 "parity; it tests whether the value of history survives a "
                 "change of controller class."),
        "quadrant_counts": counts,
        "by_history": summary,
        "per_source": rows,
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
