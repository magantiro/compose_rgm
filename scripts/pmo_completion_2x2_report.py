#!/usr/bin/env python
"""Assemble the 2x2 table and apply the PREDECLARED freeze rule verbatim."""

from __future__ import annotations

import glob
import json
import os

OUT = "diagnostics/pmo_completion_repair_v1"
RULE = os.path.join(OUT, "decision_rule_v1.json")
ORDER = ("v1_current", "content_only", "scale_only", "scale_content")
AXES_CHANGED = {"v1_current": 0, "content_only": 1, "scale_only": 1, "scale_content": 2}

# The productive-transition census, diagnostics/pmo_global_delta_census_v1.
CENSUS = {
    "frac_largest_region_ge_13": 0.5833,
    "median_largest_changed_region": 14.0,
    "frac_installs_ring": 0.8056,
    "median_retained_fraction_source": 0.400,
    "frac_largest_region_ge_6": 0.8889,
}


def main() -> None:
    arms = {}
    for path in glob.glob(os.path.join(OUT, "two_by_two_v1_*.json")):
        arms.update(json.load(open(path))["arms"])
    missing = [name for name in ORDER if name not in arms]
    if missing:
        raise SystemExit(f"arms missing, refusing to emit a verdict: {missing}")

    base = arms["v1_current"]
    rows = []
    print(f"{'arm':15s}{'region_med':>11}{'>=6':>8}{'>=13':>8}{'ret':>8}"
          f"{'ins':>6}{'ring':>7}{'branch':>8}{'chain':>7}{'yield':>7}{'div':>7}")
    for name in ORDER:
        a = arms[name]
        print(f"{name:15s}{a['median_largest_changed_region']:>11}"
              f"{a['frac_largest_region_ge_6']:>8.3f}{a['frac_largest_region_ge_13']:>8.3f}"
              f"{a['median_retained_fraction_source']:>8.3f}"
              f"{a['median_installed_atoms']:>6}{a['frac_installs_ring']:>7.3f}"
              f"{a['frac_installs_branch']:>8.3f}"
              f"{a['frac_installed_expressible_as_grow_chain']:>7.3f}"
              f"{a['executed_yield']:>7.3f}{a['diversity']:>7.3f}")

    print("\nCONDITIONAL on a completion module firing (segment_grow / segment_replace)")
    print(f"{'arm':15s}{'n':>6}{'share':>8}{'region_med':>11}{'>=13':>8}{'ins':>6}"
          f"{'exc':>6}{'ret':>8}{'ring':>7}{'branch':>8}")
    for name in ORDER:
        c = arms[name]["conditional_on_a_completion_module"]
        print(f"{name:15s}{c['n']:>6}{c['share_of_executed']:>8.3f}"
              f"{c['median_largest_changed_region']:>11}{c['frac_largest_region_ge_13']:>8.3f}"
              f"{c['median_installed_atoms']:>6}{c['median_excised_atoms']:>6}"
              f"{c['median_retained_fraction_source']:>8.3f}"
              f"{c['frac_installs_ring']:>7.3f}{c['frac_installs_branch']:>8.3f}")

    # ---- the predeclared rule, applied verbatim
    verdicts = {}
    for name in ORDER:
        a = arms[name]
        yield_ok = a["executed_yield"] >= 0.90 * base["executed_yield"]
        diversity_ok = a["diversity"] >= 0.90 * base["diversity"]
        closures = []
        for key in ("frac_largest_region_ge_13", "median_largest_changed_region",
                    "frac_installs_ring"):
            gap = CENSUS[key] - base[key]
            closures.append(0.0 if gap == 0 else (a[key] - base[key]) / gap)
        verdicts[name] = {
            "eligible": bool(yield_ok and diversity_ok),
            "yield_ok": bool(yield_ok),
            "diversity_ok": bool(diversity_ok),
            "axis_closures": [round(v, 4) for v in closures],
            "mean_gap_closure": round(sum(closures) / len(closures), 4),
            "axes_changed": AXES_CHANGED[name],
        }
    eligible = [n for n in ORDER if n != "v1_current" and verdicts[n]["eligible"]]
    print("\nPREDECLARED FREEZE RULE")
    for name in ORDER:
        v = verdicts[name]
        print(f"  {name:15s} eligible={v['eligible']!s:5s} "
              f"yield_ok={v['yield_ok']!s:5s} div_ok={v['diversity_ok']!s:5s} "
              f"mean_gap_closure={v['mean_gap_closure']:+.4f} closures={v['axis_closures']}")
    if not eligible:
        winner = None
        print("\nVERDICT: ABORT -- no arm passes both eligibility gates.")
    else:
        best = max(verdicts[n]["mean_gap_closure"] for n in eligible)
        tied = [n for n in eligible
                if abs(verdicts[n]["mean_gap_closure"] - best) < 1e-9]
        winner = min(tied, key=lambda n: verdicts[n]["axes_changed"])
        print(f"\nVERDICT: FREEZE {winner} "
              f"(mean gap closure {verdicts[winner]['mean_gap_closure']:+.4f})")

    report = {
        "schema_version": "pmo_completion_2x2_report_v1",
        "new_oracle_calls": 0,
        "decision_rule": RULE,
        "census_reference": CENSUS,
        "arms": arms,
        "rule_application": verdicts,
        "frozen_arm": winner,
    }
    with open(os.path.join(OUT, "two_by_two_report_v1.json"), "w") as handle:
        json.dump(report, handle, sort_keys=True, indent=1)
    print("wrote", os.path.join(OUT, "two_by_two_report_v1.json"))


main()
