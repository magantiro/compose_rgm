"""Generate `handoff.json` with hashes RECOMPUTED FROM DISK.

Every SHA-256 in the manifest is read off the file it names, here, at write
time. None is transcribed by hand.

That is not fastidiousness. A fabricated SHA-256 in a manifest is one of the
five instrument defects already recorded in this project, and it is the worst
kind: it makes an unverifiable artifact look verified, so the reader stops
checking exactly when they should not. `scripts/pareto_instrument_gate.py`
check D4 recomputes these again independently, so a manifest written by hand
would fail the gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FROZEN_INPUTS = (
    "diagnostics/editing_v2_matched_validation_reserve_ids.json.gz",
    "diagnostics/retarget_goal_language_normalizers.json",
    "diagnostics/retarget_calibration_result_3plus3_fixed.json",
    "diagnostics/retarget_calibration_cohort.json",
    "artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json",
    "artifacts/oracles/drd2_svm_v1/drd2_svm_parameters.npz",
    "scripts/build_analogue_trace_pool.py",
)

LANE_ARTIFACTS = (
    "docs/workstreams/pareto-control/PROTOCOL.md",
    "docs/workstreams/pareto-control/STATUS.md",
    "docs/workstreams/pareto-control/DECISION_LOG.md",
    "docs/workstreams/pareto-control/FIGURE_DESIGN.md",
    "scripts/pareto_tradeoff_census.py",
    "scripts/pareto_instrument_gate.py",
    "scripts/pareto_select_cohort.py",
    "scripts/analyse_pareto_control.py",
    "src/compose_v4/experiments/pareto_control.py",
    "tests/test_pareto_control.py",
    "modal_apps/pareto_control_app.py",
    "tests/test_pareto_instrument_gate.py",
    "diagnostics/pareto_tradeoff_census.json",
    "diagnostics/pareto_control_cohort.json",
)


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=REPO / "docs/workstreams/pareto-control/handoff.json")
    args = parser.parse_args()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())

    census_path = REPO / "diagnostics/pareto_tradeoff_census.json"
    census = json.loads(census_path.read_text()) if census_path.exists() else {}

    payload = {
        "schema": "compose.workstream.handoff",
        "workstream": "pareto-control",
        "title": "Target-free Pareto / preference control",
        "status": "DESIGN_ONLY_PLUS_HELD_IN_CENSUS",
        "branch": branch,
        "commit": commit,
        "working_tree_dirty_at_write_time": dirty,
        "base_commit": "a0e680d",
        "held_out_opened": False,
        "held_out_statement": (
            "reserve_source_keys has NOT been read in this lane. Every measurement "
            "uses training_source_keys. No Modal run has been launched."),
        "r_theta": "FROZEN. No retraining. No h_phi exists or was trained in this lane.",
        "frozen_inputs": {p: sha256(REPO / p) for p in FROZEN_INPUTS},
        "lane_artifacts": {p: sha256(REPO / p) for p in LANE_ARTIFACTS},
        "census": {
            "adopted_pair": census.get("adopted_pair"),
            "predeclared_pair_order": census.get("predeclared_pair_order"),
            "gate_thresholds": census.get("predeclared_gate_thresholds"),
            "verdicts": {p["pair"]: p["gate"]["verdict"]
                         for p in census.get("pairs", [])},
            "failed_gates": {p["pair"]: p["gate"]["failed"]
                             for p in census.get("pairs", [])},
            "instruments": census.get("instruments"),
            "frozen_scales": census.get("frozen_scales"),
        },
        "artifact_status_by_file": {
            "docs/workstreams/pareto-control/PROTOCOL.md": "DESIGN_ONLY",
            "docs/workstreams/pareto-control/FIGURE_DESIGN.md": "DESIGN_ONLY",
            "src/compose_v4/experiments/pareto_control.py": "DESIGN_ONLY",
            "tests/test_pareto_control.py": "DESIGN_ONLY",
            "modal_apps/pareto_control_app.py": "DESIGN_ONLY",
            "scripts/pareto_instrument_gate.py": "DESIGN_ONLY",
            "scripts/analyse_pareto_control.py": "DESIGN_ONLY",
            "tests/test_pareto_instrument_gate.py": "DESIGN_ONLY",
            "diagnostics/pareto_control_cohort.json": "DESIGN_ONLY",
            "diagnostics/pareto_tradeoff_census.json": "SMOKE_HELD_IN",
        },
        "reproduction_commands": [
            "python3 -m pytest tests/test_pareto_control.py -q",
            "python3 scripts/pareto_instrument_gate.py",
            "OMP_NUM_THREADS=4 python3 scripts/pareto_tradeoff_census.py "
            "--sources 60 --reach-sources 20 --out diagnostics/pareto_tradeoff_census.json",
            "python3 scripts/pareto_select_cohort.py --size 12",
            "python3 scripts/analyse_pareto_control.py --shards <shard dir> "
            "--out diagnostics/pareto_control_analysis.json",
        ],
        "census_external_inputs": {
            "note": ("The census needs the local Active8 root, a gate-zero DECISION.json "
                     "and the materialized scorer. All three are outside the repo and "
                     "are passed as arguments or environment variables, never hardcoded "
                     "to a session directory."),
            "COMPOSE_ACTIVE8": "<active8 root>",
            "COMPOSE_GATE_ZERO": "<gate_zero DECISION.json>",
            "COMPOSE_MATSCORER": "<materialized_scorer dir>",
            "defaults_point_at": "/Users/rmaganti/compose_trainset_backup/localprep/...",
        },
        "known_defects_and_withdrawn_statistics": {
            "best_candidate_reaches_pooled_p99": (
                "WITHDRAWN before any pair verdict. With a ~600-wide fiber it fires "
                "with probability 0.997 whatever the truth is. Replaced by 6-edit "
                "single-objective greedy rollouts. See DECISION_LOG D-007."),
            "P1_parity": (
                "P1 (greedy_pref vs unguided) varies controller AND objective and is "
                "labelled CONTEXT_ONLY. Caught by the executable gate at design "
                "stage. See DECISION_LOG D-008."),
            "budget_axis_ambiguity": (
                "One kernel call yields ~600 candidates, so native-oracle matching "
                "and kernel matching differ ~600x for gen_rank. P3/P4 are reported "
                "as a bracket. See DECISION_LOG D-009."),
            "census_depth": (
                "The census measures decision states at depths 0 and 1 only. Whether "
                "tradeoff structure persists at depths 2-5 is NOT measured and is "
                "the first thing the held-in smoke will show."),
        },
        "recommended_next_action": (
            "Authorize the 12-source held-in smoke in modal_apps/pareto_control_app.py. "
            "The cohort is already frozen at diagnostics/pareto_control_cohort.json, "
            "disjoint from the census sources. Bounded: held-in only, five arms, "
            "no h_phi, no held-out panel."),
        "not_authorized_without_main_approval": [
            "any Modal run",
            "opening reserve_source_keys",
            "training an h_phi",
            "changing any frozen objective constant",
            "adding a fourth objective pair",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    missing = [k for k, v in {**payload["frozen_inputs"],
                              **payload["lane_artifacts"]}.items() if v is None]
    print(f"wrote {args.out}")
    print(f"  hashed {len(FROZEN_INPUTS) + len(LANE_ARTIFACTS) - len(missing)} files")
    if missing:
        print(f"  MISSING (recorded as null, not fabricated): {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
