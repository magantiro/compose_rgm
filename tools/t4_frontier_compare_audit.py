"""Verify and summarize a completed paired T4 frontier comparison."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from compose_v4.experiments.continuation_profile import (
    canonical_bytes,
    publish_json,
    sha256_file,
)
from compose_v4.experiments.t4_matched_pilot import unseal

ARMS = ("post_hoc", "in_loop")


def _candidate_without_outcome(candidate: dict) -> dict:
    return {k: v for k, v in candidate.items() if k not in ("ds", "round")}


def _three_number(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return [min(values), statistics.median(values), max(values)]


def _strict_rank(value: float, values: list[float]) -> int:
    """One-based lower-is-better rank, with ties sharing the better rank."""
    return 1 + sum(other < value for other in values)


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*.json"))
    }


def report(root: Path) -> dict:
    result_path = root / "result.json"
    result = json.loads(result_path.read_text())
    if (
        result.get("status") != "complete"
        or result.get("automatic_next_round") is not False
    ):
        raise ValueError(
            "comparison is incomplete or authorizes an automatic continuation"
        )
    if result.get("new_oracle_attempts", 0) > 40:
        raise ValueError("comparison exceeded its 40-call authorization")

    barrier = unseal(root / "oracle_barrier.json")
    if barrier != {
        "lock_sha256": barrier["lock_sha256"],
        "maximum_new_attempts": 40,
        "prior_attempts_per_arm": 51,
    }:
        raise ValueError("oracle barrier differs from the authorized comparison")

    arm_reports = {}
    all_docked = []
    warm_best_scores = []
    for arm in ARMS:
        lock_path = root / arm / "candidate_lock.json"
        docking_path = root / arm / "docking.json"
        archive_path = root / arm / "archive.json"
        lock_envelope = json.loads(lock_path.read_text())
        lock = unseal(lock_path)
        docking = unseal(docking_path)
        archive = unseal(archive_path)

        if lock_envelope["payload_sha256"] != barrier["lock_sha256"].get(arm):
            raise ValueError(f"{arm}: oracle barrier does not bind candidate lock")
        if result["arms"][arm] != docking:
            raise ValueError(f"{arm}: root result differs from sealed docking result")
        if archive["oracle_attempts"] != docking["total_oracle_attempts"]:
            raise ValueError(f"{arm}: archive and docking attempt counts differ")
        if lock["prior_oracle_attempts"] != 51:
            raise ValueError(f"{arm}: unexpected warm-start oracle count")

        selected = lock["take"]
        docked = docking["docked"]
        if canonical_bytes(
            [_candidate_without_outcome(c) for c in docked]
        ) != canonical_bytes(selected):
            raise ValueError(f"{arm}: docked candidates differ from the locked batch")
        if docking["new_oracle_attempts"] != len(docked) or docking["failed_dockings"]:
            raise ValueError(f"{arm}: incomplete or failed docking accounting")
        if len({c["smiles"] for c in docked}) != len(docked):
            raise ValueError(f"{arm}: canonical duplicate within docked batch")
        if any(not c["oracle_eligible"] or c["v"] > 0 for c in docked):
            raise ValueError(f"{arm}: ineligible candidate reached the oracle")

        warm = [
            c
            for c in archive["archive"]
            if c.get("round", lock["round"]) < lock["round"]
            and c.get("ds") is not None
            and c.get("v", 1) <= 0
        ]
        if not warm:
            raise ValueError(f"{arm}: no feasible warm-start observations")
        warm_best = min(c["ds"] for c in warm)
        warm_best_scores.append(warm_best)

        predictions = [c["predicted_docking"] for c in docked]
        outcomes = [c["ds"] for c in docked]
        best = min(docked, key=lambda c: c["ds"])
        topology_changes = [
            c for c in docked if c["d_cycle_rank"] != 0 or c["d_ring_systems"] != 0
        ]
        preparation = result["preparation"][arm]
        arm_reports[arm] = {
            "pool_count": docking["audit"]["pool_count"],
            "eligible_count": docking["audit"]["eligible_count"],
            "docked_count": len(docked),
            "unique_canonical_docked": len({c["smiles"] for c in docked}),
            "distinct_docked_bundles": len({c["allocated_bundle_id"] for c in docked}),
            "docked_options": dict(
                sorted(Counter(c["option"] for c in docked).items())
            ),
            "all_completed_options": docking["audit"]["selected_options"],
            "score_min_median_max": _three_number(outcomes),
            "intended_release_min_median_max": _three_number(
                [c["r_release"] for c in docked]
            ),
            "realized_coherent_min_median_max": _three_number(
                [c["r_coherent"] for c in docked]
            ),
            "realized_change_min_median_max": _three_number(
                [c["r_change"] for c in docked]
            ),
            "topology_change_count": len(topology_changes),
            "topology_changes": [
                {
                    key: c[key]
                    for key in (
                        "smiles",
                        "option",
                        "ds",
                        "predicted_docking",
                        "r_release",
                        "r_coherent",
                        "r_change",
                        "d_cycle_rank",
                        "d_ring_systems",
                        "d_heavy",
                        "sim",
                        "qed",
                        "sa",
                        "allocated_bundle_id",
                    )
                }
                for c in topology_changes
            ],
            "best_new": {
                key: best[key]
                for key in (
                    "smiles",
                    "option",
                    "ds",
                    "predicted_docking",
                    "r_release",
                    "r_coherent",
                    "r_change",
                    "d_cycle_rank",
                    "d_ring_systems",
                    "d_heavy",
                    "sim",
                    "qed",
                    "sa",
                    "allocated_bundle_id",
                )
            }
            | {
                "predicted_rank_among_docked": _strict_rank(
                    best["predicted_docking"], predictions
                ),
                "observed_rank_among_docked": _strict_rank(best["ds"], outcomes),
            },
            "warm_best_score": warm_best,
            "improvement_over_warm_best": warm_best - best["ds"],
            "new_oracle_attempts": docking["new_oracle_attempts"],
            "total_oracle_attempts": docking["total_oracle_attempts"],
            "failed_dockings": docking["failed_dockings"],
            "generation_decisions": docking["audit"]["decisions"],
            "proposal_work": {
                key: preparation[key]
                for key in (
                    "parallel_lineages",
                    "seconds",
                    "slowest_lineage_proposal_seconds",
                    "sum_proposal_seconds",
                    "executor_calls",
                    "law_evaluations",
                    "law_cache_hits",
                )
            },
            "candidate_lock_payload_sha256": lock_envelope["payload_sha256"],
            "candidate_lock_file_sha256": sha256_file(lock_path),
            "docking_file_sha256": sha256_file(docking_path),
            "archive_file_sha256": sha256_file(archive_path),
        }
        all_docked.extend((arm, candidate) for candidate in docked)

    if len(set(warm_best_scores)) != 1:
        raise ValueError("arms do not share the same warm-start champion")
    if (
        sum(a["new_oracle_attempts"] for a in arm_reports.values())
        != result["new_oracle_attempts"]
    ):
        raise ValueError("root and per-arm oracle attempt counts differ")

    by_smiles = defaultdict(list)
    for arm, candidate in all_docked:
        by_smiles[candidate["smiles"]].append(
            {"arm": arm, "ds": candidate["ds"], "option": candidate["option"]}
        )
    duplicates = [
        {"smiles": smiles, "observations": observations}
        for smiles, observations in sorted(by_smiles.items())
        if len(observations) > 1
    ]
    inputs = _file_hashes(root)
    return {
        "schema_version": "t4_frontier_compare_review_v1",
        "status": "complete",
        "evidence_role": "one inspected, unreplicated development round",
        "source_run_id": root.name,
        "source_code_revision": result["code_revision"],
        "review_code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "review_script_sha256": sha256_file(Path(__file__)),
        "source_result_sha256": inputs["result.json"],
        "contract_sha256": result["configuration"]["contract_sha256"],
        "input_sha256": inputs,
        "software": {"python": platform.python_version()},
        "configuration": result["configuration"],
        "warm_best_score": warm_best_scores[0],
        "arms": arm_reports,
        "paired_observations": {
            "in_loop_minus_post_hoc_best_score": (
                arm_reports["in_loop"]["best_new"]["ds"]
                - arm_reports["post_hoc"]["best_new"]["ds"]
            ),
            "in_loop_minus_post_hoc_eligible_count": (
                arm_reports["in_loop"]["eligible_count"]
                - arm_reports["post_hoc"]["eligible_count"]
            ),
            "pooled_dockings": len(all_docked),
            "pooled_unique_canonical": len(by_smiles),
            "cross_arm_duplicate_count": len(duplicates),
            "cross_arm_duplicates": duplicates,
        },
        "elapsed_seconds": result["elapsed_seconds"],
        "new_oracle_attempts": result["new_oracle_attempts"],
        "automatic_next_round": result["automatic_next_round"],
        "interpretation": {
            "measured": [
                "In-loop guidance produced eight eligible endpoints versus four post-hoc endpoints.",
                "The best in-loop endpoint scored -10.0 versus -9.9 post-hoc and -9.7 in the shared warm archive.",
                "The best endpoint in each arm was a completed pendant six-member ring construction.",
                "The endpoint predictor ranked each arm's observed best ring endpoint last among that arm's docked endpoints.",
            ],
            "inferred": [
                "Broad executable ring construction is no longer the immediate bottleneck on this development cell.",
                "Endpoint-value ranking remains a bottleneck and should not be relabeled as future value.",
            ],
            "limitations": [
                "One inspected cell and one round do not establish reproducibility, generalization, or benchmark superiority.",
                "The arms match oracle ceilings and primitive horizons, not realized internal compute or number of eligible dockings.",
                "QuickVina is unseeded in this inherited pipeline; the repeated cross-arm molecule differed by 0.1 kcal/mol.",
                "WHERE and WHAT had no task-value contrast in this run; only eight in-loop HOW decisions were task-guided.",
                "No public winner molecule, fragment, route, or target entered proposal generation or task-value fitting.",
            ],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = report(args.root)
    if args.output:
        digest = publish_json(args.output, summary)
        print(json.dumps({"output": str(args.output), "sha256": digest}))
    else:
        print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
