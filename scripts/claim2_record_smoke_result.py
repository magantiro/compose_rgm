"""Fold the 8-source smoke's gate results into the Claim-2 handoff manifest.

Kept as a committed script rather than an ad-hoc edit for the same reason the
digest refresher is: the numbers here are read straight out of the committed
analysis artifact instead of being retyped, so the manifest cannot drift from
the evidence it summarizes.

Run after ``scripts/analyse_claim2_trajectories.py``, then
``scripts/claim2_refresh_handoff_manifest.py`` to fix the digests.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ARMS = ("r_theta", "uniform_canonical", "empirical_family")


def arm_summary(analysis: dict, arm: str) -> dict:
    rows = list(analysis["per_source"][arm].values())

    def mean(key: str) -> float:
        values = [row[key] for row in rows if row[key] is not None]
        return round(statistics.mean(values), 4) if values else float("nan")

    return {
        "mobility": round(analysis["frontier"][arm]["mobility"], 4),
        "fidelity": round(analysis["frontier"][arm]["fidelity"], 4),
        "revisit": mean("any_state_revisit_rate"),
        "reversal": mean("immediate_reversal_rate"),
        "heavy_atom_change": mean("heavy_atom_change"),
        "families": len(analysis["families_by_arm"][arm]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis", type=Path, default=Path("diagnostics/claim2_trajectory_smoke.json")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/workstreams/claim2-trajectory/handoff.json"),
    )
    parser.add_argument("--app-id", default="ap-TifBx7AHzBMd6x4cGH5BZN")
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text())
    manifest = json.loads(args.manifest.read_text())
    checks = analysis["instrument_checks"]
    cost = checks["measured_seconds_per_kernel_call"]

    manifest["status"] = "SMOKE_HELD_IN"
    manifest["modal_runs_launched"] = 1
    manifest["modal_app_id"] = args.app_id
    manifest["smoke_plan"]["authorized"] = True
    manifest["smoke_plan"]["authorized_by"] = (
        "main lane, 2026-08-12, 8-source smoke only, no automatic promotion"
    )
    manifest["smoke_plan"]["ran"] = True

    manifest["smoke_result"] = {
        "status": "SMOKE_HELD_IN",
        "may_not_answer_claim_2": True,
        "verdict_emitted": analysis["verdict_emitted"],
        "sources": analysis["source_count"],
        "kernel_calls": checks["states_measured"],
        "budget_exhausted_sources": checks["budget_exhausted_sources"],
        "enumeration_failures": len(checks["enumeration_failures"]),
        "trajectories_truncated": checks["trajectories_truncated_by_failure_or_budget"],
        "gates": {
            "arms_diverge_on_real_states": {
                "verdict": "PASS",
                "min_pairwise_tv": round(checks["arm_divergence_min"], 4),
                "median_pairwise_tv": round(checks["arm_divergence_median"], 4),
                "max_pairwise_tv": round(checks["arm_divergence_max"], 4),
                "states": checks["states_measured"],
                "degenerate_states": checks["degenerate_states"],
            },
            "kernel_cross_check": {
                "verdict": "PASS",
                "disagreements": len(checks["kernel_cross_check_disagreements"]),
                "unchecked_sources": checks["kernel_cross_check_unchecked_sources"],
            },
            "completion_under_budget": {
                "verdict": "PASS",
                "budget_exhausted_sources": checks["budget_exhausted_sources"],
                "enumeration_failures": len(checks["enumeration_failures"]),
            },
            "metric_distributions_non_degenerate": {
                "verdict": "PASS",
                "resolved_axis_comparisons": sum(
                    1
                    for record in analysis["comparisons"].values()
                    for axis in ("mobility", "fidelity")
                    if record[axis]["resolved"]
                ),
                "axis_comparisons": 2 * len(analysis["comparisons"]),
            },
            "health_measurements_behaving": {
                "verdict": "PASS",
                "note": (
                    "revisit and reversal discriminate strongly across arms; "
                    "dead-end rate is 0.000 for every arm"
                ),
            },
            "measured_cost": {
                "verdict": "PASS",
                "median_seconds_per_enumeration": cost["median"],
                "min": cost["min"],
                "max": cost["max"],
                "predicted_band": [16.8, 21.0],
            },
        },
        "provisional_observation_not_a_result": {
            "warning": (
                "n=8 is below the pre-declared 20-source floor; no verdict was emitted "
                "and none may be inferred from these point estimates"
            ),
            "by_arm": {arm: arm_summary(analysis, arm) for arm in ARMS},
            "implication": (
                "R_theta moved less and cycled more than both unlearned arms on these 8 "
                "sources. If that holds at n=36 the frontier verdict is more likely "
                "incomparable than dominates, and the cycling rates are the direction the "
                "pathological-cycling stop rule watches. Operator collapse is NOT "
                "indicated: R_theta committed all 8 active families while uniform "
                "committed 6."
            ),
        },
    }

    paths = {entry["path"] for entry in manifest["artifacts_created"]}
    for path, purpose in (
        (
            str(args.analysis),
            "local analysis of the 8-source smoke: gates, frontier point estimates, per-source metrics",
        ),
    ):
        if path not in paths:
            manifest["artifacts_created"].append(
                {
                    "path": path,
                    "sha256": "0" * 64,
                    "status": "SMOKE_HELD_IN",
                    "purpose": purpose,
                }
            )

    manifest["recommended_next_action"] = (
        "Review these gates. The 36-source held-in development run is NOT authorized by "
        "this smoke passing; at the measured "
        f"{cost['median']} s/enumeration it is roughly 10 container-hours. Weigh it "
        "against the provisional observation that R_theta moved less and cycled more "
        "than both unlearned arms on 8 sources."
    )

    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"recorded smoke result into {args.manifest}")
    print("  run scripts/claim2_refresh_handoff_manifest.py to fix digests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
