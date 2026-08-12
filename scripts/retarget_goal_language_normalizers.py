"""Compute the goal-language normalizers from HELD-IN molecules only.

The retargeting protocol requires every property normalizer to be calibrated on
held-in data and frozen before the held-out panel is touched. The two feasibility
censuses were computed on the held-out reserve, which is correct for asking "can
a panel be built" but would be the wrong source for a frozen contract. This
script is the held-in one.

Emits three things:

  1. **Normalizers** `s_j`, taken as the held-in interquartile range. IQR rather
     than standard deviation because the DRD2 log-odds distribution has a long
     upper tail -- a handful of actives would otherwise set the scale for the
     99% of molecules that are inactive.

  2. **A distribution-shift check** against the held-out reserve. If held-in and
     held-out disagree, a normalizer frozen on held-in silently rescales the
     held-out goal and the two panels stop measuring the same thing.

  3. **Position-stratified single-edit gains** for DRD2. This tests the
     assumption underneath every best-of-N reachability estimate: that the
     favourable-gain distribution is the same wherever you start. An activity
     cliff violates it -- the big favourable moves would be available only to
     molecules already near an active, and a controller starting from a typical
     inactive molecule could never find them. Measured, not assumed.

Sets no threshold and selects no goal. Threshold choice is the separate held-in
calibration step, and it needs the reachability numbers this script does not
produce.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

BANDS = ((-np.inf, -6.0), (-6.0, -4.0), (-4.0, -2.0), (-2.0, 0.0), (0.0, np.inf))


def properties(smiles, oracle):
    from rdkit import Chem
    from rdkit.Chem import Crippen, QED

    margin = np.asarray(oracle.margin_many(smiles), dtype=float)
    qed, logp = [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            qed.append(np.nan); logp.append(np.nan); continue
        try:
            qed.append(QED.qed(mol))
        except Exception:
            qed.append(np.nan)
        logp.append(Crippen.MolLogP(mol))
    return margin, np.asarray(qed), np.asarray(logp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reserve", type=Path,
                        default=Path("diagnostics/editing_v2_matched_validation_reserve_ids.json.gz"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json"))
    parser.add_argument("--held-in-sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from rdkit import RDLogger

    from build_analogue_trace_pool import mine_one_cut_pairs
    from compose_v4.drd2_oracle import load_default_oracle

    RDLogger.DisableLog("rdApp.*")
    started = time.perf_counter()

    reserve = json.load(gzip.open(args.reserve, "rt"))
    held_in = sorted(reserve["training_source_keys"])
    random.Random(args.seed).shuffle(held_in)
    held_in = held_in[: args.held_in_sample]
    held_out = sorted(reserve["reserve_source_keys"])
    print(f"held-in sample {len(held_in):,}   held-out {len(held_out):,}")

    oracle = load_default_oracle(str(args.manifest))
    mi, qi, li = properties(held_in, oracle)
    mo, qo, lo = properties(held_out, oracle)
    print(f"[{time.perf_counter()-started:5.1f}s] scored both pools\n")

    names = {"drd2_logodds": (mi, mo), "qed": (qi, qo), "clogp": (li, lo)}
    normalizers, shift = {}, {}
    print(f"{'property':>14} {'held-in p25/p50/p75':>28} {'held-out p25/p50/p75':>28} "
          f"{'IQR ratio':>10}")
    for name, (a, b) in names.items():
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        pa, pb = np.percentile(a, [25, 50, 75]), np.percentile(b, [25, 50, 75])
        iqr_in, iqr_out = pa[2] - pa[0], pb[2] - pb[0]
        normalizers[name] = {"median": float(pa[1]), "iqr": float(iqr_in),
                             "p25": float(pa[0]), "p75": float(pa[2]),
                             "source": "held-in only"}
        shift[name] = {"held_in_median": float(pa[1]), "held_out_median": float(pb[1]),
                       "iqr_ratio_in_over_out": float(iqr_in / iqr_out),
                       "median_difference": float(pa[1] - pb[1])}
        print(f"{name:>14} {pa[0]:>9.3f}/{pa[1]:.3f}/{pa[2]:.3f}     "
              f"{pb[0]:>9.3f}/{pb[1]:.3f}/{pb[2]:.3f}     {iqr_in / iqr_out:>10.3f}")

    print("\nfrozen normalizers s_j = held-in IQR")
    for name, n in normalizers.items():
        print(f"  {name:>14}  centre {n['median']:+.4f}   s_j {n['iqr']:.4f}")

    # Position-stratified gains: is the favourable tail available everywhere?
    print("\nDRD2 single-edit gains stratified by STARTING log-odds")
    print("(tests the position-independence that any best-of-N estimate assumes)")
    pairs = mine_one_cut_pairs(held_out)
    members = sorted({s for p in pairs for s in p[:2]})
    index = {s: i for i, s in enumerate(members)}
    margin = np.asarray(oracle.margin_many(members), dtype=float)
    ia = np.array([index[p[0]] for p in pairs])
    ib = np.array([index[p[1]] for p in pairs])
    start = np.concatenate([margin[ia], margin[ib]])
    gain = np.concatenate([margin[ib] - margin[ia], margin[ia] - margin[ib]])

    print(f"{'start band':>18} {'n':>6} {'P(gain>0)':>10} {'mean+':>8} "
          f"{'p90+':>8} {'p99+':>8} {'max':>8}")
    strata = {}
    for low, high in BANDS:
        sel = (start >= low) & (start < high)
        if sel.sum() < 10:
            continue
        g = gain[sel]
        pos = g[g > 0]
        label = f"[{low:g}, {high:g})"
        strata[label] = {
            "n": int(sel.sum()), "fraction_favourable": float((g > 0).mean()),
            "mean_favourable_gain": float(pos.mean()),
            "p90_favourable": float(np.percentile(pos, 90)),
            "p99_favourable": float(np.percentile(pos, 99)),
            "max": float(g.max()),
        }
        print(f"{label:>18} {sel.sum():>6} {(g > 0).mean():>9.1%} {pos.mean():>8.3f} "
              f"{np.percentile(pos, 90):>8.3f} {np.percentile(pos, 99):>8.3f} "
              f"{g.max():>8.3f}")

    typical = start <= -4.0
    verdict = ("POSITION-INDEPENDENT -- the favourable tail is available from "
               "typical inactive starts, so best-of-N over a wide fiber is not "
               "obviously blocked"
               if strata.get("[-inf, -6)", {}).get("fraction_favourable", 0) > 0.4
               else "POSITION-DEPENDENT -- the tail concentrates near actives")
    print(f"\nfrom a typical held-out start (log-odds <= -4, n={int(typical.sum())}): "
          f"best single edit {gain[typical].max():+.3f}")
    print(f"verdict: {verdict}")

    args.out.write_text(json.dumps({
        "schema": "compose.retarget.goal_language_normalizers",
        "status": "HELD_IN_NORMALIZERS_NO_THRESHOLD_SELECTED",
        "normalizers": normalizers,
        "normalizer_choice": ("IQR, not standard deviation: the DRD2 log-odds "
                              "distribution has a long upper tail and a handful "
                              "of actives would otherwise set the scale for the "
                              "inactive bulk"),
        "distribution_shift": shift,
        "drd2_gain_by_starting_band": strata,
        "position_independence_verdict": verdict,
        "held_in_sample": len(held_in),
        "held_out_total": len(held_out),
        "matched_pairs": len(pairs),
        "seed": args.seed,
        "caveat": ("Stratified gains are measured on held-out matched pairs "
                   "because that is where the panel will live; they set no "
                   "threshold and inform no normalizer. Normalizers come from "
                   "held-in only."),
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
