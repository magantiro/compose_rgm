"""Deterministic post-launch review of the sealed strict-delta-0.6 utility panel."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_delta06_utility_launch import (
    RESULT_SCHEMA,
    read_sealed,
    sha256_file,
)

REVIEW_SCHEMA = "t4_delta06_structural_subgoal_utility_review_v1"
SPAWN_SCHEMA = "t4_delta06_structural_subgoal_utility_spawn_v1"
REDUCE_SPAWN_SCHEMA = "t4_delta06_structural_subgoal_utility_reduce_spawn_v1"
ABSTENTION_SCHEMA = "t4_delta06_structural_subgoal_abstention_ledger_v1"

RESULT_PATH = (
    "diagnostics/t4_delta06_structural_subgoal_utility_launch/attempt_1/result.json"
)
SPAWN_PATH = "diagnostics/t4_delta06_structural_subgoal_utility_launch_spawn.json"
REDUCE_SPAWN_PATH = (
    "diagnostics/t4_delta06_structural_subgoal_utility_reduce_spawn.json"
)
ABSTENTION_PATH = (
    "diagnostics/t4_delta06_structural_subgoal_utility_lock/attempt_1/"
    "abstention_ledger.json"
)
BENCHMARK_PATH = "configs/t4_frozen_program_benchmark_v2.json"
SEEDS_PATH = "docs/GENMOL_T4_SEEDS.json"
SCORER_PATH = "modal_apps/genmol_t4_opt_app.py"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _outcome(gain: float) -> str:
    if gain > 0:
        return "better"
    if gain < 0:
        return "worse"
    return "tie"


def _validate_launch_receipts(
    result: dict, spawn: dict, reduce_spawn: dict
) -> tuple[dict[str, str], str]:
    if spawn.get("schema_version") != SPAWN_SCHEMA:
        raise ValueError("unexpected launch spawn receipt schema")
    if reduce_spawn.get("schema_version") != REDUCE_SPAWN_SCHEMA:
        raise ValueError("unexpected reducer spawn receipt schema")
    run_id = result.get("run_id")
    if (
        spawn.get("task", {}).get("run_id") != run_id
        or reduce_spawn.get("run_id") != run_id
    ):
        raise ValueError("launch receipt run identities disagree")
    calls = spawn.get("calls")
    if not isinstance(calls, list) or len(calls) != result.get("request_count"):
        raise ValueError("launch receipt does not contain exactly one call per request")
    call_by_request = {row["request_id"]: row["call_id"] for row in calls}
    if len(call_by_request) != len(calls):
        raise ValueError("launch receipt repeats a request identity")
    worker_call_ids = reduce_spawn.get("worker_call_ids")
    if sorted(call_by_request.values()) != sorted(worker_call_ids):
        raise ValueError("reducer receipt does not bind the exact worker calls")
    return call_by_request, reduce_spawn["call_id"]


def _validate_result(result: dict) -> None:
    requests = result.get("requests")
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("status") != "complete"
        or result.get("request_count") != 19
        or result.get("reservations") != 19
        or result.get("first_score_calls_charged") != 19
        or result.get("successful_scores") != 19
        or result.get("failed_scores") != 0
        or result.get("incomplete") != []
        or result.get("automatic_retries") != 0
        or result.get("replacement_or_backfill") is not False
        or result.get("new_generator_calls") != 0
        or result.get("candidate_trace_available") is not False
        or not isinstance(requests, list)
        or len(requests) != 19
    ):
        raise ValueError(
            "scored result violates the authorized one-shot launch boundary"
        )
    request_ids = [row.get("request_id") for row in requests]
    if request_ids != sorted(request_ids) or len(set(request_ids)) != 19:
        raise ValueError("scored requests are not unique and deterministically ordered")
    if any(
        row.get("status") != "complete"
        or row.get("first_score_call_charged") != 1
        or row.get("score_direction") != "minimize"
        for row in requests
    ):
        raise ValueError("a scored request is not one complete first-score result")


def build_review(repo_root: Path, *, code_revision: str) -> dict:
    """Build a deterministic review; lower docking scores are better."""
    result_path = repo_root / RESULT_PATH
    result = read_sealed(result_path)
    _validate_result(result)

    spawn_path = repo_root / SPAWN_PATH
    reduce_path = repo_root / REDUCE_SPAWN_PATH
    spawn = _load_json(spawn_path)
    reduce_spawn = _load_json(reduce_path)
    call_by_request, reducer_call_id = _validate_launch_receipts(
        result, spawn, reduce_spawn
    )

    abstention_path = repo_root / ABSTENTION_PATH
    abstentions = read_sealed(abstention_path)
    if abstentions.get("schema_version") != ABSTENTION_SCHEMA:
        raise ValueError("unexpected zero-oracle abstention-ledger schema")

    benchmark_path = repo_root / BENCHMARK_PATH
    benchmark = read_sealed(benchmark_path)
    seeds_path = repo_root / SEEDS_PATH
    seeds = _load_json(seeds_path)
    cells = benchmark.get("cells", {})
    ivg = benchmark.get("ivg_reported", {})
    if len(cells) != 15 or len(ivg) != 15 or len(seeds) != 15:
        raise ValueError("frozen T4 comparison universe is not the expected 15 cells")

    score_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    request_scores = []
    membership_count = 0
    for request in result["requests"]:
        request_id = request["request_id"]
        request_scores.append(
            {
                "request_id": request_id,
                "worker_call_id": call_by_request[request_id],
                "target": request["target"],
                "score": request["score"],
                "docking_seconds": request["docking_seconds"],
                "pose_artifact_sha256": request["pose_artifact_sha256"],
                "membership_ids": request["membership_ids"],
            }
        )
        for membership in request["memberships"]:
            membership_count += 1
            score_rows[(membership["cell"], membership["policy_id"])].append(
                {
                    "membership_id": membership["membership_id"],
                    "request_id": request_id,
                    "worker_call_id": call_by_request[request_id],
                    "selection_role": membership["selection_role"],
                    "rank": membership["rank"],
                    "score": request["score"],
                }
            )
    if membership_count != 22:
        raise ValueError("scored membership census changed")

    policy_cells = []
    for cell_row in sorted(abstentions["cells"], key=lambda row: row["cell"]):
        cell = cell_row["cell"]
        if cell not in cells or cell not in ivg:
            raise ValueError(f"abstention ledger contains an unknown cell: {cell}")
        global_index = cells[cell]["global_index"]
        seed = seeds[global_index]
        if seed["target"] != cell_row["target"]:
            raise ValueError(f"seed/reference target mismatch for {cell}")
        reference_score = -float(seed["published_ds"])
        ivg_mean = float(ivg[cell]["mean"])
        for policy_status in sorted(
            cell_row["policy_statuses"], key=lambda row: row["policy_id"]
        ):
            policy_id = policy_status["policy_id"]
            scored = sorted(
                score_rows.get((cell, policy_id), []),
                key=lambda row: (row["score"], row["request_id"], row["membership_id"]),
            )
            expected_count = policy_status["selected_memberships"]
            if len(scored) != expected_count:
                raise ValueError(
                    f"membership/abstention disagreement for {cell}/{policy_id}"
                )
            row = {
                "cell": cell,
                "target": cell_row["target"],
                "policy_id": policy_id,
                "zero_oracle_status": policy_status["status"],
                "selected_memberships": expected_count,
                "scored_memberships": scored,
                "published_source_reference_score": reference_score,
                "ivg_reported_mean": ivg_mean,
                "ivg_reported_run_bests": ivg[cell]["run_bests"],
            }
            if scored:
                best = scored[0]["score"]
                reference_gain = reference_score - best
                ivg_gain = ivg_mean - best
                row.update(
                    {
                        "best_locked_score": best,
                        "best_request_ids": sorted(
                            item["request_id"]
                            for item in scored
                            if item["score"] == best
                        ),
                        "gain_vs_published_source_reference": reference_gain,
                        "published_source_reference_outcome": _outcome(reference_gain),
                        "gain_vs_ivg_reported_mean": ivg_gain,
                        "ivg_reported_mean_outcome": _outcome(ivg_gain),
                    }
                )
            else:
                row.update(
                    {
                        "best_locked_score": None,
                        "best_request_ids": [],
                        "gain_vs_published_source_reference": None,
                        "published_source_reference_outcome": "abstain",
                        "gain_vs_ivg_reported_mean": None,
                        "ivg_reported_mean_outcome": "abstain",
                    }
                )
            policy_cells.append(row)

    if len(policy_cells) != 30 or len(score_rows) != 13:
        raise ValueError("policy-cell comparison census changed")

    def outcomes(field: str, rows: list[dict]) -> dict[str, int]:
        labels = ("better", "tie", "worse", "abstain")
        return {label: sum(row[field] == label for row in rows) for label in labels}

    policy_summary = {}
    for policy_id in sorted({row["policy_id"] for row in policy_cells}):
        rows = [row for row in policy_cells if row["policy_id"] == policy_id]
        policy_summary[policy_id] = {
            "cells": len(rows),
            "scored_cells": sum(row["best_locked_score"] is not None for row in rows),
            "published_source_reference_outcomes": outcomes(
                "published_source_reference_outcome", rows
            ),
            "ivg_reported_mean_outcomes": outcomes("ivg_reported_mean_outcome", rows),
        }

    cell_best_rows = []
    for cell in sorted(cells):
        rows = [
            row
            for row in policy_cells
            if row["cell"] == cell and row["best_locked_score"] is not None
        ]
        if not rows:
            cell_best_rows.append({"cell": cell, "status": "abstain"})
            continue
        best = min(row["best_locked_score"] for row in rows)
        reference_gain = rows[0]["published_source_reference_score"] - best
        ivg_gain = rows[0]["ivg_reported_mean"] - best
        cell_best_rows.append(
            {
                "cell": cell,
                "status": "scored",
                "best_locked_score": best,
                "best_policy_ids": sorted(
                    row["policy_id"] for row in rows if row["best_locked_score"] == best
                ),
                "gain_vs_published_source_reference": reference_gain,
                "published_source_reference_outcome": _outcome(reference_gain),
                "gain_vs_ivg_reported_mean": ivg_gain,
                "ivg_reported_mean_outcome": _outcome(ivg_gain),
            }
        )

    scored_cells = [row for row in cell_best_rows if row["status"] == "scored"]
    payload = {
        "schema_version": REVIEW_SCHEMA,
        "status": "complete_negative_ivg_comparison",
        "code_revision": code_revision,
        "inputs": {
            RESULT_PATH: {
                "physical_sha256": sha256_file(result_path),
                "payload_sha256": identity(result),
            },
            SPAWN_PATH: {"physical_sha256": sha256_file(spawn_path)},
            REDUCE_SPAWN_PATH: {"physical_sha256": sha256_file(reduce_path)},
            ABSTENTION_PATH: {
                "physical_sha256": sha256_file(abstention_path),
                "payload_sha256": identity(abstentions),
            },
            BENCHMARK_PATH: {
                "physical_sha256": sha256_file(benchmark_path),
                "payload_sha256": identity(benchmark),
            },
            SEEDS_PATH: {"physical_sha256": sha256_file(seeds_path)},
            SCORER_PATH: {"physical_sha256": sha256_file(repo_root / SCORER_PATH)},
        },
        "run": {
            "run_id": result["run_id"],
            "modal_app_id": "ap-3EmwfjNd1nfuWzZQsIS5op",
            "reducer_call_id": reducer_call_id,
            "remote_result_path": (
                "/t4_delta06_structural_subgoal_utility_launch/"
                f"{result['run_id']}/result.json"
            ),
            "requests": 19,
            "memberships": 22,
            "successful_scores": 19,
            "failed_scores": 0,
            "automatic_retries": 0,
            "replacements_or_backfill": 0,
        },
        "comparison_definition": {
            "score_direction": "minimize",
            "gain": "reference_score - best_locked_score; positive means locked candidate is better",
            "published_source_reference": (
                "negative of docs/GENMOL_T4_SEEDS.json published_ds, matching the bound "
                "scoring wrapper's seed-score convention"
            ),
            "ivg_reference": (
                "reported three-run mean from the frozen T4 benchmark; it is a published "
                "comparator, not a matched prospective rerun"
            ),
            "cell_policy_best": (
                "post-launch descriptive minimum across the one or two prospectively locked "
                "memberships; it does not select, replace, or backfill a docking request"
            ),
        },
        "request_scores": request_scores,
        "cell_policy_results": policy_cells,
        "cell_best_results": cell_best_rows,
        "summary": {
            "cells_with_scores": len(scored_cells),
            "cell_abstentions": len(cell_best_rows) - len(scored_cells),
            "policy_cells_with_scores": len(score_rows),
            "policy_cell_abstentions": 30 - len(score_rows),
            "policy_cell_published_source_reference_outcomes": outcomes(
                "published_source_reference_outcome", policy_cells
            ),
            "policy_cell_ivg_reported_mean_outcomes": outcomes(
                "ivg_reported_mean_outcome", policy_cells
            ),
            "scored_cell_best_published_source_reference_outcomes": outcomes(
                "published_source_reference_outcome", scored_cells
            ),
            "scored_cell_best_ivg_reported_mean_outcomes": outcomes(
                "ivg_reported_mean_outcome", scored_cells
            ),
            "policy_summary": policy_summary,
        },
        "scientific_interpretation": {
            "measured": (
                "The strict-delta-0.6 locked panel has nonzero endpoint utility relative "
                "to the published source reference in six of eight scored cells."
            ),
            "negative_result": (
                "No scored cell-policy best equals or improves on its frozen IVG reported mean."
            ),
            "limit": result["interpretation_limit"],
        },
    }
    return payload
