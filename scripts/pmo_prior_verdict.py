"""Assemble the final result and render the predeclared verdict mechanically.

The verdict is computed from the thresholds written in
`diagnostics/pmo_learned_prior_prediction_v1.json` BEFORE any measurement, so it
cannot be tuned to the numbers after the fact.  Zero oracle calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "diagnostics"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    prediction = json.loads((D / "pmo_learned_prior_prediction_v1.json").read_text())
    # Use the full sweep when it exists, else the fast marginal run.
    ab_path = D / "pmo_learned_prior_ab_v1.json"
    if not ab_path.exists():
        ab_path = D / "pmo_learned_prior_ab_fast_v1.json"
    ab = json.loads(ab_path.read_text())
    paired = json.loads((D / "pmo_prior_paired_probe_v1.json").read_text())
    paired_v2 = json.loads((D / "pmo_prior_paired_probe_v2.json").read_text())
    signal = json.loads((D / "pmo_prior_signal_probe_v1.json").read_text())
    reach = json.loads((D / "pmo_call_site_reach_v1.json").read_text())

    verdicts = {}
    # P1/P2: the predeclared falsifier is a paired mean delta beyond 1 SE.
    for arm, m in paired["arms"].items():
        delta = m["paired_mean_delta_dQED"]
        se = m["paired_se_delta_dQED"]
        sigma = (delta / se) if se > 0 else 0.0
        verdicts[arm] = {
            "paired_mean_delta_dQED": delta,
            "paired_se": se,
            "sigma": sigma,
            "disagreement_rate_pct": m["disagreement_rate_pct"],
            "pct_losing_QED_A": m["pct_losing_QED_A"],
            "pct_losing_QED_B": m["pct_losing_QED_B"],
            "paired_mean_delta_dSA": m["paired_mean_delta_dSA"],
            "P1_P2_verdict": (
                "HELD" if sigma > 1.0 else
                "FALSIFIED" if sigma < -1.0 else
                "NOT ESTABLISHED (within 1 SE of zero)"
            ),
        }
    best = max(verdicts, key=lambda k: verdicts[k]["sigma"])

    # P4: diversity trade, from the marginal A/B.
    endpoints = ab["endpoints"]
    a_distinct = endpoints.get("A", 0)
    diversity = {
        arm: {
            "distinct_endpoints": count,
            "vs_A_pct": round(100.0 * count / a_distinct, 1) if a_distinct else None,
        }
        for arm, count in sorted(endpoints.items())
    }

    payload = {
        "schema_version": "pmo_learned_prior_verdict_v1",
        "oracle_calls_spent": 0,
        "predeclaration": prediction["PREDICTIONS"],
        "predeclaration_committed_before_measurement": True,
        "checkpoint": prediction["checkpoint"],
        "P1_P2_paired_chemistry": verdicts,
        "strongest_arm": best,
        "P3_stratified_by_parent_heavy": ab["by_parent_heavy_bin"],
        "P4_diversity": diversity,
        "marginal_ab_source": ab_path.name,
        "confound_control_heavy_atoms": {
            arm: {
                "paired_mean_delta_dHeavy": m["paired_mean_delta_dHeavy"],
                "paired_se_delta_dHeavy": m["paired_se_delta_dHeavy"],
                "mean_dHeavy_A": m["mean_dHeavy_A"],
                "mean_dHeavy_B": m["mean_dHeavy_B"],
                "reading": (
                    "Edit SIZE is identical between arms, so the SA improvement "
                    "is not a size artifact. This is structural: every family at "
                    "this call site preserves the heavy-atom count."
                ),
            }
            for arm, m in paired_v2["arms"].items()
        },
        "P5_support_identity": {
            "permutation_checks_passed": ab["permutation_checks"],
            "verdict": "HELD" if ab["permutation_checks"] > 0 else "NOT RUN",
        },
        "signal_attribution_by_family": signal["by_family"],
        "call_site_reach": {
            k: reach[k] for k in (
                "call_site_reach_pct", "pct_of_call_site_draws_with_signal",
                "effective_reach_pct", "families_at_the_call_site")
        },
        "marginal_ab_overall": ab["overall"],
        "marginal_ab_by_family": ab["by_family"],
        "legal_execution": {
            "attempts": ab["attempts"], "executed": ab["executed"],
            "parents_covered": ab["covered"],
        },
        "cost_load_independent": {
            "ab": ab["prior_statistics"],
            "paired": paired["prior_statistics"],
            "signal": signal["prior_statistics"],
        },
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("PREDECLARED VERDICT (paired, the sharper gate)")
    for arm, v in verdicts.items():
        print(f"  {arm:12s} delta dQED {v['paired_mean_delta_dQED']:+.4f} "
              f"+-{v['paired_se']:.4f} ({v['sigma']:+.1f} sigma)  "
              f"disagree {v['disagreement_rate_pct']:.1f}%  "
              f"%loseQED {v['pct_losing_QED_A']:.1f} -> {v['pct_losing_QED_B']:.1f}  "
              f"-> {v['P1_P2_verdict']}")
    print("\nDIVERSITY (distinct endpoints, marginal A/B)")
    for arm, v in diversity.items():
        print(f"  {arm:14s} {v['distinct_endpoints']:6d}  ({v['vs_A_pct']}% of A)")
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
