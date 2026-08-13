"""HV-AUC for the 12-source Pareto smoke, from the committed shards. NO RERUN.

WHY THIS EXISTS. HV-AUC was frozen into the analysis hierarchy as item 2 --
"final HV and HV-AUC" -- and into the reporting rules as "both native and raw
conventions are reported; neither may be substituted for the other". Only final
HV was ever computed, so the smoke is half-delivered on that axis. Everything
needed is already in the shards: `endpoint_z` carries the five committed
endpoints per arm, which is exactly the sequence the curve is built from.

THE CURVE IS THE ONE ALREADY USED FOR N_90. `analyse_pareto_control.py` builds

    trace = [normalized_hypervolume(z[:k+1], reference, utopia) for k in range(len(z))]

i.e. best-so-far hypervolume after k committed trajectories. HV-AUC is the area
under that same trace. Reusing it rather than defining a second curve keeps
N_90 and HV-AUC answering questions about the same object.

TWO CONVENTIONS, BOTH REPORTED, NEITHER SUBSTITUTED
---------------------------------------------------
  raw     mean of the trace in HV units. Comparable across arms WITHIN a source;
          not comparable across sources, because sources differ in how much
          hypervolume is attainable at all.
  native  the same trace divided by HV_star_internal -- the POOLED attainable
          hypervolume across the predeclared internal universe -- before the
          area is taken. This is the "how fast does it approach what is
          reachable here" reading, and it is the one that aggregates across
          sources.

Reporting only `raw` would let a source with a large attainable front dominate
the mean. Reporting only `native` would hide that two arms differ in absolute
achievement. Hence both, as the frozen rule requires.

ORDERING IS A DECLARED CONVENTION, NOT A FACT
---------------------------------------------
HV-AUC depends on the order trajectories are accumulated, and the order here is
the committed preference order w = 0.1 .. 0.9. That is the order the arms
actually ran, so it is the honest one -- but a best-case ordering would score
higher and a worst-case lower, and neither is reported as "the" HV-AUC. The
spread between them is emitted as `ordering_sensitivity` so a reader can see how
much of the number is the convention.

`unguided` CARRIES THE SAME GUARD AS EVERYWHERE ELSE. Its five branches are five
SEEDS, not five preferences, so its curve measures sampling accumulation rather
than preference response. Flagged in the artifact so the number cannot be
misread as a preference-controlled comparison.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: Five branches that are seeds rather than preferences.
PREFERENCE_BLIND = ("unguided",)


def hv_auc(trace: list[float]) -> float:
    """Area under the best-so-far HV curve, normalised to per-trajectory units.

    The trapezoid rule would understate the first trajectory, which is the one
    that matters most for an efficiency claim, so this is the plain mean of the
    best-so-far values -- the discrete area under a step curve over k = 1..K,
    divided by K. Monotone by construction, since the trace is best-so-far.
    """
    return float(np.mean(trace))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path,
                        default=REPO / "local_runtime/pareto_control_smoke")
    parser.add_argument("--census", type=Path,
                        default=REPO / "diagnostics/pareto_tradeoff_census.json")
    parser.add_argument("--out", type=Path,
                        default=REPO / "diagnostics/pareto_smoke_hv_auc.json")
    args = parser.parse_args()

    from compose_v4.experiments.pareto_control import (
        normalized_hypervolume,
        pooled_attainable_hypervolume,
    )

    census = json.loads(args.census.read_text())
    pair = census["adopted_pair"]
    entry = next(p for p in census["pairs"] if p["pair"] == pair)
    key_a, key_b = entry["objective_a"], entry["objective_b"]
    scales = census["frozen_scales"]
    utopia = np.array([scales["utopia_p99"][key_a], scales["utopia_p99"][key_b]])
    reference = np.array([scales["reference_p5"][key_a], scales["reference_p5"][key_b]])

    rows = [json.loads(p.read_text())
            for p in sorted(args.shards.glob("*.json"))
            if "partial" not in p.name]
    print(f"{len(rows)} shards, pair {pair}\n")

    arms = sorted({a for r in rows for a in r["arms"]})
    per_arm: dict[str, dict[str, list[float]]] = {
        a: {"raw": [], "native": [], "best": [], "worst": []} for a in arms}
    per_source: dict[str, dict] = {}

    for row in rows:
        fronts = {a: np.asarray(row["arms"][a]["endpoint_z"], dtype=float)
                  for a in arms if row["arms"].get(a, {}).get("endpoint_z")}
        if len(fronts) < 2:
            continue
        hv_star = pooled_attainable_hypervolume(fronts, reference, utopia)
        entry_src: dict[str, dict] = {"HV_star_internal": hv_star, "arms": {}}
        for arm, z in fronts.items():
            trace = [normalized_hypervolume(z[: k + 1], reference, utopia)
                     for k in range(len(z))]
            raw = hv_auc(trace)
            native = hv_auc([t / hv_star for t in trace]) if hv_star > 0 else float("nan")
            # Ordering sensitivity: the same five endpoints, accumulated in the
            # best and worst possible orders. K = 5, so this is exhaustive.
            aucs = [hv_auc([normalized_hypervolume(z[list(p[: k + 1])], reference, utopia)
                            for k in range(len(z))])
                    for p in itertools.permutations(range(len(z)))]
            per_arm[arm]["raw"].append(raw)
            per_arm[arm]["native"].append(native)
            per_arm[arm]["best"].append(float(max(aucs)))
            per_arm[arm]["worst"].append(float(min(aucs)))
            entry_src["arms"][arm] = {
                "trace": trace, "hv_auc_raw": raw, "hv_auc_native": native,
                "final_hv": trace[-1],
                "ordering_best": float(max(aucs)),
                "ordering_worst": float(min(aucs))}
        per_source[str(row["index"])] = entry_src

    def boot(values: list[float], seed: int = 0) -> dict:
        v = np.asarray(values, dtype=float)
        n = len(v)
        draws = np.array([np.mean(np.random.default_rng(s).choice(v, n, replace=True))
                          for s in range(4000)])
        return {"n": n, "mean": float(v.mean()), "median": float(np.median(v)),
                "ci95": [float(np.percentile(draws, 2.5)),
                         float(np.percentile(draws, 97.5))]}

    summary = {}
    print(f"{'arm':<20}{'HV-AUC raw':>12}{'HV-AUC native':>15}"
          f"{'final HV':>10}   {'ordering [worst,best] raw':>28}")
    for arm in arms:
        if not per_arm[arm]["raw"]:
            continue
        raw, native = boot(per_arm[arm]["raw"]), boot(per_arm[arm]["native"])
        finals = [per_source[s]["arms"][arm]["final_hv"]
                  for s in per_source if arm in per_source[s]["arms"]]
        summary[arm] = {
            "hv_auc_raw": raw, "hv_auc_native": native,
            "final_hv_mean": float(np.mean(finals)),
            "ordering_sensitivity_raw": {
                "worst_mean": float(np.mean(per_arm[arm]["worst"])),
                "best_mean": float(np.mean(per_arm[arm]["best"])),
                "committed_order_mean": raw["mean"]},
            "preference_blind": arm in PREFERENCE_BLIND,
        }
        flag = "  [SEEDS, not preferences]" if arm in PREFERENCE_BLIND else ""
        print(f"{arm:<20}{raw['mean']:>12.4f}{native['mean']:>15.4f}"
              f"{np.mean(finals):>10.4f}   "
              f"[{np.mean(per_arm[arm]['worst']):.4f}, "
              f"{np.mean(per_arm[arm]['best']):.4f}]{flag}")

    # HV-AUC IS A MIXTURE, AND ITS WIN COUNT MUST NOT BE QUOTED AS SET-LEVEL.
    #
    # The curve's early points are close to per-trajectory quality, where
    # verified control holds a near-guaranteed advantage: greedy's action is
    # always in the shortlist and strict improvement never commits a lower V_G.
    # Only the LAST point, k=K, is the set-level object that P2 tests and that
    # was reported 10W/2L. Averaging them produces a statistic that inherits
    # part of the definitional contrast, so the per-k breakdown is emitted and
    # the mixture is stated wherever the aggregate appears.
    per_k = []
    if {"verified_pref", "greedy_pref"} <= set(per_arm):
        K = max(len(v["arms"]["verified_pref"]["trace"]) for v in per_source.values())
        for k in range(K):
            d = np.array([s["arms"]["verified_pref"]["trace"][k]
                          - s["arms"]["greedy_pref"]["trace"][k]
                          for s in per_source.values()])
            per_k.append({"k": k + 1, "mean": float(d.mean()),
                          "wins": int((d > 1e-12).sum()),
                          "losses": int((d < -1e-12).sum()),
                          "is_set_level_contrast": k == K - 1})
        print("\nverified - greedy at each k (k = trajectories accumulated)")
        for e in per_k:
            tag = "  <- THIS is the set-level contrast P2 reports" if e["is_set_level_contrast"] else ""
            print(f"  k={e['k']}  {e['wins']}W/{e['losses']}L  {e['mean']:+.4f}{tag}")

    # The one contrast HV-AUC is frozen to inform: P2, set-level, verified vs
    # greedy. Paired by source, since the source is the independent unit.
    contrast = {}
    if {"verified_pref", "greedy_pref"} <= set(per_arm):
        for conv in ("raw", "native"):
            d = np.array(per_arm["verified_pref"][conv]) - np.array(per_arm["greedy_pref"][conv])
            b = boot(list(d))
            b["wins"] = int((d > 1e-12).sum())
            b["losses"] = int((d < -1e-12).sum())
            contrast[conv] = b
            print(f"\nP2 HV-AUC ({conv}) verified - greedy: {b['mean']:+.4f} "
                  f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  "
                  f"{b['wins']}W/{b['losses']}L")

    args.out.write_text(json.dumps({
        "schema": "compose.pareto.smoke_hv_auc",
        "status": "SMOKE_HELD_IN",
        "held_out_opened": False,
        "provenance": "computed from the COMMITTED 12-source smoke shards; no rerun",
        "pair": pair,
        "conventions": {
            "raw": "mean of best-so-far HV over k=1..K, in HV units",
            "native": "the same trace divided by HV_star_internal before the mean",
            "rule": "both reported; neither may be substituted for the other",
            "curve": "identical to the trace N_90 is computed from",
        },
        "ordering_convention": (
            "trajectories accumulate in the COMMITTED preference order "
            "w = 0.1..0.9, which is the order the arms ran. HV-AUC depends on "
            "this order; the exhaustive best and worst orderings over K=5 are "
            "emitted as ordering_sensitivity so the convention's contribution "
            "is visible rather than assumed away."),
        "preference_blind_guard": (
            "unguided's five branches are five SEEDS, not five preferences, so "
            "its curve measures sampling accumulation and NOT preference "
            "response. It anchors the floor; it is not a controlled arm."),
        "per_arm": summary,
        "P2_hv_auc_verified_minus_greedy": contrast,
        "hv_auc_is_a_mixture": {
            "warning": (
                "HV-AUC's win count MUST NOT be quoted as a set-level result. "
                "The curve's early points approximate per-trajectory quality, "
                "where verified control's advantage is near-definitional; only "
                "k=K is the set-level object P2 tests. HV-AUC blends them."),
            "set_level_contrast_is": "k = K only, reported separately as P2",
            "per_k_verified_minus_greedy": per_k,
        },
        "per_source": per_source,
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
