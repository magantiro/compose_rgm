"""Repaired P3/P4: closed-loop preference control vs FAIRLY FUNDED generate-and-rank.

WRITTEN AND COMMITTED BEFORE ANY OUTCOME WAS READ. The oracle-accounting
correction it depends on was frozen earlier still (`d65327b`, `a27d41b`,
`bd687e2`), and the only top-up number seen before this point was source 000's
blinded feasibility ledger -- a cost fact, explicitly inside the allowed
feasibility set, with no hypervolume computed anywhere in that path.

WHAT WAS BROKEN, AND WHY THE OLD NUMBERS ARE UNUSABLE
------------------------------------------------------
`trajectories_for_kernel_budget` was an OPEN-LOOP estimate: it divided the
COMPOSE arm's kernel budget by an assumed 6 fresh calls per trajectory and hoped
the baseline would spend it. Unguided trajectories all start at the same root,
collide in the enumeration cache, and cannot. Measured attainment on the
committed smoke: **65.5%** for `gen_rank@greedy`, **41.7%** for
`gen_rank@verified`. On source 000 the baseline was handed 61 trajectories
against 368 kernel calls and spent 60 -- **16%**.

The repaired matcher meters instead of estimating, and reached the committed
target on **12/12** sources with 9 kernel calls of total overshoot. Source 000
needed **1,025** trajectories to spend what the broken matcher bought with 61.
So the earlier P3/P4 was not slightly underfunded; it was profoundly unfair, and
its numbers stay withdrawn rather than being compared against these.

NOTHING HERE IS NEW CONVENTION
------------------------------
Reference and utopia are the frozen held-in p5/p99 from the tradeoff census.
Hypervolume, front indices, coverage and diversity come from the frozen
`pareto_control` module. Selection inside the baseline was the frozen
Chebyshev-argmin over the pool. No top-K rule, archive rule or matching
convention is introduced at analysis time.

THREE RESOURCE AXES, NEVER MERGED
---------------------------------
Reported separately, because they disagree by orders of magnitude in OPPOSITE
directions and any single "efficiency" scalar would pick the winner by choosing
an axis:

    kernel calls               matched by construction -- that is the contrast
    algorithmic oracle requests   what the METHOD demanded, implementation-aware
    completed trajectories     whole molecules generated

`benchmark_eval_requests` and evaluator work are reported secondarily.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CONTRASTS = (
    ("P3", "greedy_pref", "greedy_pref"),
    ("P4", "verified_pref", "verified_pref"),
)


def paired(diffs: np.ndarray, draws: int = 8000) -> dict:
    """Source-level paired bootstrap. The source is the independent unit."""
    n = len(diffs)
    boot = np.array([np.mean(np.random.default_rng(s).choice(diffs, n, replace=True))
                     for s in range(draws)])
    return {"n": n, "mean": float(diffs.mean()), "median": float(np.median(diffs)),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wins": int((diffs > 1e-12).sum()), "losses": int((diffs < -1e-12).sum()),
            "ties": int((np.abs(diffs) <= 1e-12).sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", type=Path,
                    default=REPO / "local_runtime/pareto_control_smoke")
    ap.add_argument("--topup", type=Path,
                    default=REPO / "local_runtime/topup/pareto_gen_rank_topup")
    ap.add_argument("--census", type=Path,
                    default=REPO / "diagnostics/pareto_tradeoff_census.json")
    ap.add_argument("--out", type=Path,
                    default=REPO / "diagnostics/pareto_p3_p4_repaired.json")
    args = ap.parse_args()

    from compose_v4.experiments.pareto_control import (
        endpoint_diversity,
        normalized_hypervolume,
        pareto_front_indices,
    )
    from compose_v4.experiments.pareto_oracle_semantics import corrected_cost

    census = json.loads(args.census.read_text())
    pair = census["adopted_pair"]
    entry = next(p for p in census["pairs"] if p["pair"] == pair)
    ka, kb = entry["objective_a"], entry["objective_b"]
    sc = census["frozen_scales"]
    utopia = np.array([sc["utopia_p99"][ka], sc["utopia_p99"][kb]])
    reference = np.array([sc["reference_p5"][ka], sc["reference_p5"][kb]])

    smoke = {json.loads(Path(f).read_text())["index"]: json.loads(Path(f).read_text())
             for f in glob.glob(str(args.smoke / "*.json")) if "partial" not in f}

    results: dict[str, dict] = {}
    for tag, compose_arm, topup_arm in CONTRASTS:
        shards = sorted(glob.glob(str(args.topup / f"*_{topup_arm}.json")))
        if not shards:
            results[tag] = {"status": "NOT_RUN",
                            "reason": f"no top-up shards matched to {topup_arm}"}
            print(f"{tag}: NOT RUN -- no top-up shards for {topup_arm}\n")
            continue

        per_source, hv_d, cov_d, nd_d, div_d = [], [], [], [], []
        cost_rows = []
        unmatched = []
        for path in shards:
            t = json.loads(Path(path).read_text())
            idx = t["index"]
            m = t["matching"]
            if not m["is_scientific"]:
                unmatched.append({"index": idx, "status": m["status"]})
                continue
            s = smoke[idx]
            zc = np.asarray(s["arms"][compose_arm]["endpoint_z"], float)
            zg = np.asarray(t["selected_endpoint_z"], float)

            hv_c = normalized_hypervolume(zc, reference, utopia)
            hv_g = normalized_hypervolume(zg, reference, utopia)
            nd_c = len(pareto_front_indices(zc))
            nd_g = len(pareto_front_indices(zg))

            # Cost, implementation-aware on BOTH sides.
            cc = corrected_cost(compose_arm, s["arms"][compose_arm],
                                len(s["preferences"]))
            gp = {"cost": t["ledger"], "n_trajectories": m["n_trajectories"],
                  "schema": t["schema"]}
            cg = corrected_cost(f"gen_rank@{compose_arm.split('_')[0]}", gp,
                                len(s["preferences"]))

            hv_d.append(hv_c - hv_g)
            nd_d.append(nd_c - nd_g)
            cost_rows.append({
                "index": idx,
                "kernel_compose": cc["kernel_calls"], "kernel_genrank": cg["kernel_calls"],
                "algo_oracle_compose": cc["algorithmic_oracle_requests"],
                "algo_oracle_genrank": cg["algorithmic_oracle_requests"],
                "traj_compose": len(s["preferences"]), "traj_genrank": m["n_trajectories"],
                "bench_compose": cc["benchmark_eval_requests"],
                "bench_genrank": cg["benchmark_eval_requests"],
                "evaluator_compose": cc["native_oracle_calls"],
                "evaluator_genrank": cg["native_oracle_calls"],
            })
            per_source.append({"index": idx, "hv_compose": hv_c, "hv_genrank": hv_g,
                               "hv_diff": hv_c - hv_g,
                               "nondominated_compose": nd_c, "nondominated_genrank": nd_g,
                               "target": m["target_kernel_calls"],
                               "realized": m["realized_kernel_calls"],
                               "genrank_trajectories": m["n_trajectories"]})

        hv_d = np.asarray(hv_d)
        res = {"status": "COMPUTED", "n_sources": len(per_source),
               "excluded_unmatched": unmatched,
               "hypervolume": paired(hv_d),
               "nondominated_set_size": paired(np.asarray(nd_d, float)),
               "per_source": per_source, "resource_rows": cost_rows}

        # Resource frontiers -- three axes, ratios reported per axis, never merged.
        axes = {}
        for name, a, b in (("kernel_calls", "kernel_compose", "kernel_genrank"),
                           ("algorithmic_oracle_requests", "algo_oracle_compose", "algo_oracle_genrank"),
                           ("completed_trajectories", "traj_compose", "traj_genrank"),
                           ("evaluator_calls", "evaluator_compose", "evaluator_genrank")):
            ca = np.array([r[a] for r in cost_rows], float)
            cb = np.array([r[b] for r in cost_rows], float)
            axes[name] = {
                "compose_median": float(np.median(ca)),
                "genrank_median": float(np.median(cb)),
                "ratio_compose_over_genrank_median": (
                    float(np.median(ca / np.where(cb > 0, cb, np.nan)))
                    if (cb > 0).any() else None),
            }
        res["resource_axes"] = axes
        results[tag] = res

        print(f"===== {tag}: {compose_arm} vs FAIRLY FUNDED gen_rank ({len(per_source)} sources) =====")
        h = res["hypervolume"]
        print(f"  hypervolume  COMPOSE - gen_rank : {h['mean']:+.4f} "
              f"[{h['ci95'][0]:+.4f}, {h['ci95'][1]:+.4f}]  {h['wins']}W/{h['losses']}L")
        n = res["nondominated_set_size"]
        print(f"  nondominated set size            : {n['mean']:+.3f} "
              f"[{n['ci95'][0]:+.3f}, {n['ci95'][1]:+.3f}]  {n['wins']}W/{n['losses']}L")
        print(f"  {'axis':<30}{'COMPOSE':>12}{'gen_rank':>12}{'ratio':>10}")
        for name, a in axes.items():
            r = a["ratio_compose_over_genrank_median"]
            print(f"  {name:<30}{a['compose_median']:>12.0f}{a['genrank_median']:>12.0f}"
                  f"{(f'{r:.2f}x' if r else 'n/a'):>10}")
        print()

    args.out.write_text(json.dumps({
        "schema": "compose.pareto.p3_p4_repaired",
        "status": "SMOKE_HELD_IN", "held_out_opened": False,
        "pair": pair,
        "matcher": "closed-loop metered; 12/12 MATCHED, 9 kernel calls total overshoot",
        "superseded": ("the original P3/P4 used an open-loop matcher that funded "
                       "gen_rank to 65.5% (@greedy) and 41.7% (@verified) of the "
                       "intended kernel budget; those numbers stay withdrawn"),
        "conventions": "frozen census scales and frozen pareto_control metrics; nothing new",
        "resource_axes_rule": "reported separately; never merged into one efficiency scalar",
        "results": results,
    }, indent=2, default=float) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
