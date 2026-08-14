"""Horizon qualification: nested reachability at H in {3, 6, 12, 24}.

Written BEFORE the H24 data landed, so the selection procedure cannot be shaped
by the curve.

THE NESTING IS VERIFIED FIRST. The H24 run used the same sources, the same
sha256 seeds and the same frozen `R_theta` as the H6 pilot, so its first six
committed states MUST reproduce the pilot exactly. If they do not, the runs are
two independent samples rather than a paired continuation and no curve may be
read from them.

THE SELECTION RULE, frozen in advance:

  * choose the SMALLEST horizon at which the NONTRIVIAL reachability curve has
    essentially saturated
  * qualify on the QED>=0.80 and QED>=0.85 tiers, which carry enough events to
    show a curve
  * QED>=0.90 is a REPORTED SECONDARY CHECK ONLY -- a single pilot event cannot
    select a hyperparameter, and choosing H to maximize it would be the
    "increase H until we beat 45.1%" failure
  * `qualified@0` sources are excluded from every count; only genuine
    edit-discovered hits qualify a horizon
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from compose_v4.experiments.hphi_rollout import (  # noqa: E402
    BENCHMARK_REGION,
    first_hit_step,
    registered_regions,
)

NESTED_HORIZONS = (3, 6, 12, 24)
#: Tiers with enough events to qualify a horizon.
QUALIFYING_TIERS = (0.80, 0.85)


def load(path: Path):
    p = json.loads(gzip.decompress(path.read_bytes()).decode())
    return {r["index"]: r for r in p["results"] if r.get("status") == "OK"}


def verify_nesting(pilot: dict, long: dict, prefix: int = 6) -> tuple[bool, str]:
    """The H24 run must reproduce the H6 pilot on its first `prefix` steps."""
    shared = sorted(set(pilot) & set(long))
    if not shared:
        return False, "no shared source indices"
    checked = mismatched = 0
    for i in shared:
        by_rep = {t["replicate"]: t for t in long[i]["trajectories"]}
        for tp in pilot[i]["trajectories"]:
            tl = by_rep.get(tp["replicate"])
            if tl is None:
                return False, f"source {i} replicate {tp['replicate']} missing"
            checked += 1
            if tl["path"][: prefix + 1] != tp["path"][: prefix + 1]:
                mismatched += 1
    ok = mismatched == 0
    return ok, (f"{checked} trajectories checked, {mismatched} mismatched "
                f"on the first {prefix} committed states")


def curve(long: dict, region, horizons=NESTED_HORIZONS):
    """Genuine new-hit prevalence at each nested horizon."""
    q, s = region
    trajs = [t for r in long.values() for t in r["trajectories"]]
    n_eligible = 0
    hits = {h: 0 for h in horizons}
    for t in trajs:
        qed, sim = t["qed"], t["similarity_to_source"]
        if qed[0] >= q and sim[0] >= s:
            continue                       # qualified@0 -- excluded entirely
        n_eligible += 1
        fh = first_hit_step(qed, sim, region)
        if fh is None:
            continue
        for h in horizons:
            if fh <= h:
                hits[h] += 1
    return n_eligible, hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=Path, default=Path("/tmp/hphi_pilot.json.gz"))
    ap.add_argument("--long", type=Path, default=Path("/tmp/hphi_h24.json.gz"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    pilot, long = load(args.pilot), load(args.long)
    print(f"pilot sources {len(pilot)}   H24 sources {len(long)}")

    ok, msg = verify_nesting(pilot, long)
    print(f"\nNESTING CHECK: {msg}")
    if not ok:
        print("  ✗ FAILED -- these are NOT a paired continuation. "
              "No curve may be read.")
        return 1
    print("  ✓ the H24 run reproduces the pilot exactly on its first 6 states")

    print(f"\nGENUINE NEW-HIT PREVALENCE (qualified@0 excluded)")
    hdr = "".join(f"{'H'+str(h):>9}" for h in NESTED_HORIZONS)
    print(f"{'region':<26}{'elig':>6}{hdr}{'  H24/H6':>10}")
    rows = []
    for region in registered_regions():
        n_el, hits = curve(long, region)
        if n_el == 0:
            continue
        ratio = (hits[24] / hits[6]) if hits[6] else None
        cells = "".join(f"{hits[h]:>9}" for h in NESTED_HORIZONS)
        r = f"{ratio:.2f}x" if ratio is not None else "—"
        mark = " *" if region == BENCHMARK_REGION else ""
        print(f"QED>={region[0]:.2f} & sim>={region[1]:.2f}"
              f"{n_el:>7}{cells}{r:>10}{mark}")
        rows.append({"region": list(region), "eligible": n_el,
                     "hits": {str(h): hits[h] for h in NESTED_HORIZONS},
                     "h24_over_h6": ratio,
                     "is_benchmark": region == BENCHMARK_REGION})

    print(f"\nSATURATION on the qualifying tiers {QUALIFYING_TIERS} "
          f"(0.90 is secondary, never selects H)")
    for tier in QUALIFYING_TIERS:
        agg = {h: 0 for h in NESTED_HORIZONS}
        for r in rows:
            if abs(r["region"][0] - tier) < 1e-9:
                for h in NESTED_HORIZONS:
                    agg[h] += r["hits"][str(h)]
        tot = agg[NESTED_HORIZONS[-1]]
        if not tot:
            continue
        frac = {h: agg[h] / tot for h in NESTED_HORIZONS}
        print(f"  QED>={tier:.2f}  " + "  ".join(
            f"H{h}: {agg[h]:>4} ({frac[h]:.2f})" for h in NESTED_HORIZONS))
        # marginal gain from each doubling
        print(f"           marginal  H6->H12 +{agg[12]-agg[6]:<4} "
              f"H12->H24 +{agg[24]-agg[12]:<4}")

    out = {"schema": "compose.hphi.horizon_qualification",
           "nesting_verified": ok, "nesting_detail": msg,
           "horizons": list(NESTED_HORIZONS),
           "qualifying_tiers": list(QUALIFYING_TIERS),
           "selection_rule": ("smallest horizon at which the nontrivial curve "
                              "saturates on the 0.80/0.85 tiers; 0.90 is a "
                              "secondary report and never selects H"),
           "regions": rows}
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
