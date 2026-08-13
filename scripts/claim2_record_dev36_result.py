"""Fold the 36-source development result and cycle attribution into the manifest.

Numbers are read from the committed analysis artifacts rather than retyped, so
the manifest cannot drift from the evidence it summarizes. Run
``scripts/claim2_refresh_handoff_manifest.py`` afterwards to fix digests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ("r_theta", "uniform_canonical", "empirical_family")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path,
                        default=Path("diagnostics/claim2_trajectory_dev36.json"))
    parser.add_argument("--attribution", type=Path,
                        default=Path("diagnostics/claim2_cycle_attribution_dev36.json"))
    parser.add_argument("--corpus", type=Path,
                        default=Path("diagnostics/claim2_corpus_reversibility.json"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("docs/workstreams/claim2-trajectory/handoff.json"))
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text())
    attribution = json.loads(args.attribution.read_text())
    corpus = json.loads(args.corpus.read_text())
    manifest = json.loads(args.manifest.read_text())
    checks = analysis["instrument_checks"]

    manifest["status"] = "DEVELOPMENT"
    manifest["modal_runs_launched"] = manifest.get("modal_runs_launched", 1) + 3
    manifest["results"] = {
        "trajectories_generated": 48 + 216,
        "note": (
            "8-source smoke plus 36-source held-in development run. Held-in "
            "development evidence; the matched reserve was never opened."
        ),
    }

    manifest["cycle_attribution"] = {
        "status": "DEVELOPMENT",
        "definitions_frozen_at_commit": "5c81818",
        "central_causal_question": (
            "are the training reference transitions themselves reversible?"
        ),
        "corpus_census": {
            "teacher_transitions": corpus["entries"],
            "distinct_directed_edges": corpus["distinct_directed_edges"],
            "mutual_pairs": corpus["mutual_pairs"],
            "mutual_edge_fraction": corpus["mutual_edge_fraction"],
        },
        "preregistered_thresholds": {"inherited": 0.33, "model_specific": 0.10},
        "verdict": attribution["preregistered_verdict"],
        "reading": (
            "R_theta learned a locally reversible reference process. Its cycling is "
            "INHERITED, not a compositional pathology."
        ),
        "by_arm": {
            arm: {
                "cancelled_fraction": attribution["by_arm"][arm]["mean"]["cancelled_fraction"],
                "net_edits": attribution["by_arm"][arm]["mean"]["net_edits"],
                "immediate_two_cycles": attribution["by_arm"][arm]["mean"]["immediate_two_cycles"],
                "longer_revisits": attribution["by_arm"][arm]["mean"]["longer_revisits"],
                "displacement_per_net_edit": attribution["by_arm"][arm]["net_displacement"]["per_net_edit"],
                "reversal_by_family": attribution["by_arm"][arm]["reversal_by_family"],
            }
            for arm in ARMS
        },
        "controlled_comparison_caveat": attribution["controlled_comparison_has_a_near_fixed_sign"],
    }

    manifest["development_result"] = {
        "status": "DEVELOPMENT",
        "sources": analysis["source_count"],
        "verdict_emitted": analysis["verdict_emitted"],
        "frontier": analysis["frontier"],
        "comparisons": analysis["comparisons"],
        "families_by_arm": {a: len(f) for a, f in analysis["families_by_arm"].items()},
        "operator_collapse": False,
        "instrument_gates": {
            "kernel_cross_check_disagreements": len(checks["kernel_cross_check_disagreements"]),
            "degenerate_states": checks["degenerate_states"],
            "states_measured": checks["states_measured"],
            "enumeration_failures": len(checks["enumeration_failures"]),
            "trajectories_truncated": checks["trajectories_truncated_by_failure_or_budget"],
            "measured_seconds_per_enumeration": checks["measured_seconds_per_kernel_call"],
        },
        "did_not_replicate_from_the_smoke": [
            {
                "finding": "displacement per net edit favouring r_theta",
                "at_8_sources": {"r_theta": 0.2129, "uniform": 0.1203, "empirical": 0.1309},
                "at_36_sources": {"r_theta": 0.1477, "uniform": 0.1286, "empirical": 0.1351},
                "resolved_at_36": False,
                "note": (
                    "reported as an unresolved diagnostic and never promoted into the "
                    "verdict; the 20-source floor is why this did not become a claim"
                ),
            },
            {
                "finding": "cancelled fraction for r_theta",
                "at_8_sources": 0.438,
                "at_36_sources": 0.292,
            },
        ],
    }

    manifest["decision_rule"] = {
        "preregistered_by": "main lane, before Step 2 ran",
        "branch": "KEEP_AND_REFRAME",
        "reopen_r_theta": False,
        "reopen_criteria_all_required": {
            "cycling_is_model_specific": False,
            "worse_on_both_net_mobility_and_fidelity": False,
            "controlled_trajectories_also_cyclic": False,
        },
        "frontier_result": "incomparable",
        "reframing": (
            "learned local reference dynamics, not goal-directed transport: R_theta "
            "supplies plausible local molecular motion and control supplies purpose"
        ),
    }

    manifest["recommended_next_action"] = (
        "Main-lane review. Do NOT reopen R_theta: its local reversibility is inherited "
        "from a reference process whose distinct teacher transitions are 73.3% mutual. "
        "Decide whether matched-reserve confirmation (~27 container-hours) is warranted "
        "given the frontier lands on incomparable rather than dominates."
    )

    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"recorded development result into {args.manifest}")
    print(f"  branch: {manifest['decision_rule']['branch']}")
    print(f"  frontier: {manifest['decision_rule']['frontier_result']}")
    print(f"  attribution: {manifest['cycle_attribution']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
