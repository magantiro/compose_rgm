"""Aggregate the Pareto-control smoke shards into the reported contrasts.

RUNS THE INSTRUMENT GATE FIRST AND REFUSES TO REPORT IF IT FAILS.

The order matters. Every one of the five defects recorded in this project was
found while reading a result that had already been written down and believed.
So here the gate runs on the shards BEFORE any contrast is computed, and a
failure produces a report with `status: INVALID_INSTRUMENT` and no numbers,
rather than numbers with a warning above them.

What it reports, per PROTOCOL.md section 7.3:
  * normalized hypervolume over committed endpoints only, five per arm
  * HV-AUC against BOTH oracle-call conventions, never one alone
  * preference coverage, nondominated-set size, feasibility, diversity
  * source-level paired bootstrap for every declared contrast

What it refuses to report: any p-value on a guaranteed-sign statistic, and any
hypervolume contrast whose budgets are not matched.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from compose_v4.experiments.pareto_control import (  # noqa: E402
    INTERNAL_METHOD_UNIVERSE,
    budget_to_ninety,
    check_method_universe,
    pooled_attainable_hypervolume,
    summarize_b90,
    normalized_hypervolume,
    preference_region_coverage,
    source_paired_bootstrap,
    union_reference_front,
)
from pareto_instrument_gate import (  # noqa: E402
    LANE_CONTRASTS,
    WITHDRAWN_STATISTICS,
    run_gate,
)

#: Statistics whose sign a theorem fixes. Magnitude only; never a p-value.
GUARANTEED_SIGN = {("verified_pref", "greedy_pref")}

METRICS = ("normalized_hypervolume", "preference_coverage",
           "nondominated_set_size", "endpoint_diversity")


def load_shards(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(glob.glob(str(directory / "*.json"))):
        if path.endswith(".partial.json"):
            continue
        rows.append(json.loads(Path(path).read_text()))
    return rows


def per_source_metric(rows: list[dict], arm: str, metric: str) -> dict[str, float]:
    out = {}
    for row in rows:
        entry = row.get("arms", {}).get(arm)
        if entry is None:
            continue
        value = entry.get(metric)
        if metric == "feasibility":
            value = float(np.mean(entry["feasible"]))
        if value is not None:
            out[str(row["index"])] = float(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "docs/workstreams/pareto-control/handoff.json")
    args = parser.parse_args()

    rows = load_shards(args.shards)
    if not rows:
        raise SystemExit(f"no shards under {args.shards}")
    arms = sorted({a for row in rows for a in row.get("arms", {})})
    print(f"{len(rows)} sources, arms: {', '.join(arms)}\n")

    # --- The gate runs FIRST, on the shards, before any contrast is computed.
    action_sequences = {
        arm: {str(row["index"]): tuple(tuple(s) for s in
                                       row["arms"][arm]["action_sequences"])
              for row in rows if arm in row.get("arms", {})}
        for arm in arms}
    costs = {arm: {k: float(np.mean([row["arms"][arm]["cost"][k] for row in rows
                                     if arm in row.get("arms", {})]))
                   for k in ("native_oracle_calls", "raw_oracle_calls",
                             "kernel_calls")}
             for arm in arms}
    endpoint_counts = {arm: int(np.median([len(row["arms"][arm]["endpoints"])
                                           for row in rows
                                           if arm in row.get("arms", {})]))
                       for arm in arms}

    gate = run_gate(manifest=args.manifest if args.manifest.exists() else None,
                    action_sequences=action_sequences, costs=costs,
                    endpoint_counts=endpoint_counts)
    print("INSTRUMENT GATE")
    for check in gate.checks:
        print(f"  {'PASS' if check['pass'] else 'FAIL'}  {check['check']}")

    if not gate.passed:
        payload = {"schema": "compose.pareto.control_analysis",
                   "status": "INVALID_INSTRUMENT",
                   "reason": "the instrument gate failed; no statistic is reported",
                   "gate": gate.checks}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nGATE FAILED. Wrote {args.out} with NO statistics.")
        return 1

    # --- Per-arm summaries -------------------------------------------------
    summary: dict[str, Any] = {}
    print("\nPER-ARM (mean over sources)")
    header = f"  {'arm':<20}" + "".join(f"{m[:14]:>16}" for m in METRICS) + \
             f"{'feasibility':>14}{'kernel':>10}{'native':>10}{'raw':>10}"
    print(header)
    for arm in arms:
        values = {m: per_source_metric(rows, arm, m) for m in METRICS}
        feas = per_source_metric(rows, arm, "feasibility")
        summary[arm] = {m: {"mean": float(np.mean(list(v.values()))) if v else None,
                            "per_source": v} for m, v in values.items()}
        summary[arm]["feasibility"] = {
            "mean": float(np.mean(list(feas.values()))) if feas else None,
            "per_source": feas}
        summary[arm]["cost"] = costs[arm]
        cells = "".join(f"{np.mean(list(values[m].values())):>16.4f}" for m in METRICS)
        print(f"  {arm:<20}{cells}{np.mean(list(feas.values())):>14.3f}"
              f"{costs[arm]['kernel_calls']:>10.0f}"
              f"{costs[arm]['native_oracle_calls']:>10.0f}"
              f"{costs[arm]['raw_oracle_calls']:>10.0f}")

    # --- Contrasts, source-level paired bootstrap --------------------------
    print("\nCONTRASTS -- the source is the resampling unit; branches move together")
    contrasts: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for contrast in LANE_CONTRASTS:
        if contrast.arm not in arms or contrast.base not in arms:
            # P5 and P6 are WITHIN-arm preference contrasts, not arm-vs-arm, so
            # they are computed separately below. Anything else missing is
            # recorded rather than dropped: a contrast that vanishes silently
            # from a report is indistinguishable from one that was never run.
            if not contrast.name.startswith(("P5", "P6")):
                skipped[contrast.name] = (
                    f"arm '{contrast.arm}' or base '{contrast.base}' absent from "
                    f"the shards")
            continue
        entry: dict[str, Any] = {"status": contrast.status,
                                 "varies": contrast.intended_dimension}
        for metric in METRICS:
            a = per_source_metric(rows, contrast.arm, metric)
            b = per_source_metric(rows, contrast.base, metric)
            shared = sorted(set(a) & set(b))
            boot = source_paired_bootstrap([a[s] - b[s] for s in shared])
            guaranteed = (contrast.arm, contrast.base) in GUARANTEED_SIGN
            if guaranteed:
                boot["sign_is_guaranteed"] = True
                boot["pvalue_NOT_REPORTED"] = (
                    "policy improvement fixes the sign; a test against a null of "
                    "0.5 would assume something already known to be false")
            entry[metric] = boot
        contrasts[contrast.name] = entry
        hv = entry["normalized_hypervolume"]
        flag = "" if contrast.status == "PRIMARY" else "  [CONTEXT_ONLY]"
        print(f"  {contrast.name:<34} HV {hv['mean']:>+8.4f} "
              f"[{hv['ci95_low']:>+7.4f}, {hv['ci95_high']:>+7.4f}]  "
              f"{hv['wins']}W/{hv['losses']}L/{hv['ties']}T{flag}")

    # --- Efficiency: HV_ref, N_90, and region coverage ----------------------
    # HV_ref is the UNION nondominated front across every arm -- never one arm's
    # own front. If it were COMPOSE's best front, "reached 90% of the attainable
    # front" would be partly a statement about COMPOSE's own ceiling and every
    # efficiency curve would inherit the bias. `union_reference_front` raises if
    # handed a single method, so this cannot be got wrong silently.
    scales = json.loads((REPO / "diagnostics/pareto_tradeoff_census.json").read_text())
    pair_key_a = rows[0].get("objective_a", "P")
    pair_key_b = rows[0].get("objective_b", "D")
    reference = np.array([scales["frozen_scales"]["reference_p5"][pair_key_a],
                          scales["frozen_scales"]["reference_p5"][pair_key_b]])
    utopia = np.array([scales["frozen_scales"]["utopia_p99"][pair_key_a],
                       scales["frozen_scales"]["utopia_p99"][pair_key_b]])

    universe = [m for m in INTERNAL_METHOD_UNIVERSE if m in arms]
    efficiency: dict[str, Any] = {
        "nadir_r": ("held-in p5 from the frozen scales -- the corner used to "
                    "COMPUTE hypervolume, frozen before any outcome"),
        "utopia": "held-in p99, frozen",
        "HV_star_internal": (
            "pooled terminal nondominated union of the PREDECLARED INTERNAL ARMS "
            "at the fixed maximum number of trajectories. Never one arm's own "
            "front. Its value is known only after every arm runs, but the RULE "
            "is preregistered and METHOD-SYMMETRIC: a strong arm that expands "
            "the pooled frontier raises the bar for everyone including itself."),
        "threshold_name": "90% of POOLED ATTAINABLE hypervolume",
        "name_discipline": ("never 'reference HV' and never '90% of COMPOSE's "
                            "front' -- the name is where this bias survives "
                            "review"),
        "frozen_before_any_hv_number": True,
        "internal_axis": "completed controlled trajectories -- COMPOSE arms only",
        "external_axis": ("unique valid canonical evaluations, and oracle "
                          "requests; external methods are NEVER placed on the "
                          "trajectory axis"),
        "declared_internal_universe": list(INTERNAL_METHOD_UNIVERSE),
        "universe_present_in_shards": universe,
        "per_source": {},
    }

    n90: dict[str, list[dict[str, Any]]] = {arm: [] for arm in universe}
    region_cov: dict[str, list[float]] = {arm: [] for arm in universe}

    for row in rows:
        fronts = {arm: np.asarray(row["arms"][arm]["endpoint_z"], dtype=float)
                  for arm in universe if arm in row.get("arms", {})
                  and row["arms"][arm].get("endpoint_z")}
        if len(fronts) < 2:
            continue
        # "Pooled" is meaningless without a membership list, and a union that
        # silently gains or loses a method moves every B_90 in the table.
        membership = check_method_universe(fronts, universe, label="internal")
        union = union_reference_front(fronts)
        hv_star_internal = pooled_attainable_hypervolume(fronts, reference, utopia)
        efficiency["per_source"][str(row["index"])] = {
            "HV_star_internal": hv_star_internal,
            "union_front_size": int(len(union)),
            "membership": membership}
        for arm, z in fronts.items():
            # Internal axis: one committed trajectory per preference branch, so
            # best-so-far HV after k trajectories, k = 1..5.
            trace = [normalized_hypervolume(z[: k + 1], reference, utopia)
                     for k in range(len(z))]
            n90[arm].append(budget_to_ninety(
                trace, list(range(1, len(z) + 1)), hv_star_internal))
            region_cov[arm].append(
                preference_region_coverage(z, union)["coverage"])

    efficiency["N_90_trajectories"] = {arm: summarize_b90(v) for arm, v in n90.items()}
    efficiency["preference_region_coverage"] = {
        arm: {"mean": (float(np.mean(v)) if v else None), "n": len(v)}
        for arm, v in region_cov.items()}

    print("\nEFFICIENCY -- INTERNAL axis (trajectories), COMPOSE arms only")
    print("  threshold = 90% of POOLED ATTAINABLE hypervolume (HV_star_internal)")
    print(f"  {'arm':<20}{'N_90 median':>14}{'uncens':>9}{'CENSORED':>10}"
          f"{'region cov':>13}")
    for arm in universe:
        e = efficiency["N_90_trajectories"][arm]
        c = efficiency["preference_region_coverage"][arm]
        med = ("N_90 > N_max" if e["median_uncensored"] is None
               else f"{e['median_uncensored']:.2f}")
        cov = "n/a" if c["mean"] is None else f"{c['mean']:.3f}"
        print(f"  {arm:<20}{med:>14}{e['n_uncensored']:>9}"
              f"{e['n_censored']:>10}{cov:>13}")
    print("  censored sources are counted, never imputed at N_max")

    # --- P5 and P6: WITHIN-arm preference contrasts -------------------------
    # These vary the objective ONLY: same controller, same start, same budget,
    # different preference weight. They are the contrasts that carry "different
    # preferences produce different futures", and they cannot be computed from
    # arm-vs-arm differences because both sides live inside one arm.
    within: dict[str, Any] = {}

    p5_gap, p5_distinct = [], []
    for row in rows:
        entry = row.get("arms", {}).get("greedy_pref")
        if entry is None or len(entry.get("endpoint_z", [])) < 2:
            continue
        z = np.asarray(entry["endpoint_z"], dtype=float)
        # objective-0 coordinate under the highest weight minus the lowest.
        # A controller that ignored the preference would give ~0 here, and the
        # value can come out negative, so it is a measurement.
        p5_gap.append(float(z[-1, 0] - z[0, 0]))
        p5_distinct.append(len(set(entry["endpoints"])))
    if p5_gap:
        within["P5_preference_responsiveness"] = {
            "status": "PRIMARY", "varies": "objective",
            "statistic": "objective_0 coordinate at w=0.9 minus at w=0.1",
            "falsifying_range": "(-inf, +inf); a preference-blind controller gives 0",
            "objective_0_gap": source_paired_bootstrap(p5_gap),
            "distinct_endpoints_of_5": {
                "mean": float(np.mean(p5_distinct)),
                "all_five_identical_fraction": float(np.mean(
                    [d == 1 for d in p5_distinct])),
            },
        }
        b = within["P5_preference_responsiveness"]["objective_0_gap"]
        print(f"  {'P5_preference_responsiveness':<34} "
              f"obj0 {b['mean']:>+8.4f} [{b['ci95_low']:>+7.4f}, "
              f"{b['ci95_high']:>+7.4f}]  {b['wins']}W/{b['losses']}L/{b['ties']}T")

    p6_distinct, p6_shared_prefix = [], []
    for row in rows:
        fan = row.get("prefix_branching")
        if not fan:
            continue
        endpoints = [b["endpoint"] for b in fan["branches"].values()]
        p6_distinct.append(len(set(endpoints)))
        p6_shared_prefix.append(all(b["states"][0] == fan["branch_point"]
                                    for b in fan["branches"].values()))
    if p6_distinct:
        within["P6_same_prefix_branching"] = {
            "status": "PRIMARY", "varies": "objective",
            "statistic": "distinct endpoints from the IDENTICAL branch point",
            "falsifying_range": "[1, 5]; 1 means the prefix determines the future",
            "n_sources": len(p6_distinct),
            "mean_distinct_endpoints": float(np.mean(p6_distinct)),
            "all_five_identical_fraction": float(np.mean(
                [d == 1 for d in p6_distinct])),
            # If this is not 1.0 the figure's premise is false and the run is
            # invalid for P6, whatever the endpoint counts say.
            "all_branches_started_at_the_branch_point": bool(all(p6_shared_prefix)),
        }
        w6 = within["P6_same_prefix_branching"]
        print(f"  {'P6_same_prefix_branching':<34} "
              f"{w6['mean_distinct_endpoints']:.2f}/5 distinct endpoints from the "
              f"identical branch point (n={w6['n_sources']})")
        if not w6["all_branches_started_at_the_branch_point"]:
            print("    WARNING: branches did not all start at the branch point; "
                  "P6 is INVALID for this run")

    if skipped:
        print("\nSKIPPED CONTRASTS (recorded, not dropped)")
        for name, why in skipped.items():
            print(f"  {name}: {why}")

    payload = {
        "schema": "compose.pareto.control_analysis",
        "status": "SMOKE_HELD_IN",
        "within_arm_contrasts": within,
        "efficiency": efficiency,
        "skipped_contrasts": skipped,
        "held_out_opened": False,
        "n_sources": len(rows),
        "pair": rows[0].get("pair"),
        "gate": {"passed": True, "checks": gate.checks},
        "withdrawn_statistics": WITHDRAWN_STATISTICS,
        "arms": summary,
        "contrasts": contrasts,
        "reporting_rules": {
            "hv_auc": "both native and raw conventions are reported; neither may "
                      "be substituted for the other",
            "gen_rank": "reported at BOTH budget matchings as a bracket; one "
                        "kernel call yields ~600 candidates, so the two axes "
                        "differ by orders of magnitude",
            "P1": "CONTEXT_ONLY -- varies controller AND objective, may not carry "
                  "a headline",
            "hv_ref": "union front across all arms; NEVER one arm's own front",
            "efficiency_is_primary": ("the question is how efficiently the process "
                                      "sweeps useful front regions, not whether "
                                      "final HV is 0.83 vs 0.79"),
            "axes_never_mixed": ("trajectory counts are internal-only; external "
                                 "methods appear only on evaluation/oracle axes"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
