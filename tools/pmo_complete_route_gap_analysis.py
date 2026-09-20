#!/usr/bin/env python3
"""Quantify the gap from complete-program support to IVG-competitive PMO search."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.pmo_complete_route_dynamic_gate import (
    SELECTED_TASKS,
    _historical_candidate_rows,
    _member_tasks,
    load_envelope,
)
from compose_v4.experiments.pmo_legal_action_policy import (
    action_features,
    enumerate_rule_successors,
    score_features,
)
from compose_v4.experiments.pmo_route_fiber_production_yield import (
    ACTION_EXPLORATION,
    ACTION_TEMPERATURE,
    CORPUS,
    LEGAL_RUNTIME,
    TASK_FOLDS,
    _fold_checkpoints,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

GATE = "diagnostics/pmo_complete_route_dynamic_gate_v1/attempt_1/result.json"
CHECKPOINT = (
    "diagnostics/pmo_complete_route_dynamic_gate_v1/attempt_1/runtime_fold_checkpoints.json"
)
PARITY = "diagnostics/pmo_ivg_oracle_parity/result.json"
HISTORY = "diagnostics/pmo_online_policy/result_sealed.json"
OUTPUT = "diagnostics/pmo_complete_route_dynamic_gate_v1/attempt_1/gap_analysis.json"


def _write_sealed(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = json.loads(json.dumps(payload, sort_keys=True))
    envelope = {"payload": normalized, "payload_sha256": identity(normalized)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _action_probability(
    graph,
    rule: str,
    successor_key: str,
    legal: dict,
    fold: int,
    cache: dict[tuple[int, str, str, str], float],
) -> float:
    key = (fold, canonical_state_key(graph), rule, successor_key)
    if key in cache:
        return cache[key]
    fiber = tuple(enumerate_rule_successors(graph, rule))
    if not fiber:
        cache[key] = 0.0
        return 0.0
    scores = np.asarray(
        [score_features(legal, action_features(graph, candidate)) for candidate in fiber],
        dtype=float,
    )
    scores = scores / ACTION_TEMPERATURE
    scores -= scores.max()
    learned = np.exp(scores)
    learned /= learned.sum()
    probabilities = ACTION_EXPLORATION / len(fiber) + (1 - ACTION_EXPLORATION) * learned
    probability = sum(
        float(probability)
        for candidate, probability in zip(fiber, probabilities, strict=True)
        if candidate.successor_key == successor_key
    )
    cache[key] = probability
    return probability


def _route_log10_probability(
    route: dict,
    checkpoint: dict,
    legal: dict,
    fold: int,
    cache: dict[tuple[int, str, str, str], float],
) -> float:
    length = len(route["actions"])
    if length > checkpoint["maximum_primitives"]:
        return float("-inf")
    length_probability = dict(
        zip(checkpoint["lengths"], checkpoint["length_probabilities"], strict=True)
    )[length]
    log_probability = math.log10(length_probability)
    for position, (state, action, successor) in enumerate(
        zip(route["states"][:-1], route["actions"], route["states"][1:], strict=True)
    ):
        rule_law = checkpoint["position_rule_probabilities"][position]
        rule_probability = dict(zip(rule_law["classes"], rule_law["probabilities"], strict=True))[
            action["executor_rule"]
        ]
        graph = decode_state(state)
        successor_probability = _action_probability(
            graph,
            action["executor_rule"],
            canonical_state_key(decode_state(successor)),
            legal,
            fold,
            cache,
        )
        if rule_probability <= 0 or successor_probability <= 0:
            return float("-inf")
        log_probability += math.log10(rule_probability * successor_probability)
    return log_probability


def _scale_count(distribution: dict[str, int], minimum: int) -> int:
    return sum(count for length, count in distribution.items() if int(length) >= minimum)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    gate = load_envelope(root / GATE)
    checkpoint = load_envelope(root / CHECKPOINT)
    corpus = load_envelope(root / CORPUS)
    legal = load_envelope(root / LEGAL_RUNTIME)
    parity = load_envelope(root / PARITY)
    history = load_envelope(root / HISTORY)
    checkpoints = {int(row["fold_index"]): row for row in checkpoint["folds"]}
    legal_folds = _fold_checkpoints(legal)

    task_rows = {}
    action_probability_cache: dict[tuple[int, str, str, str], float] = {}
    for task in SELECTED_TASKS:
        fold = TASK_FOLDS[task]
        routes = [row for row in corpus["routes"] if task in _member_tasks(row)]
        complete = [
            row
            for row in routes
            if row["dependency_region_program"].get("complete_representation_supported")
        ]
        log_probabilities = [
            _route_log10_probability(
                row,
                checkpoints[fold],
                legal_folds[fold],
                fold,
                action_probability_cache,
            )
            for row in complete
        ]
        finite = [value for value in log_probabilities if math.isfinite(value)]
        sampled = gate["matched_actual_production_support"][task]
        arm = sampled["arms"]["complete_route"]
        minimum = min(len(row["actions"]) for row in routes)
        ivg = parity["contract"]["tasks"][task]["ivg_no_prescreen_auc_top10"]
        ceiling = parity["results"][task]
        task_rows[task] = {
            "headline_comparator": {
                "role": "IVG_no_prescreen_autonomous_top10_AUC",
                "auc": ivg,
                "oracle_budget": 10000,
                "finish": True,
            },
            "answer_known_ceiling_not_autonomous": {
                "best_score": ceiling["best_score"],
                "auc_top10_official_10k": ceiling["auc_top10_official_10k"],
                "locked_calls": ceiling["oracle_calls"],
            },
            "teacher_route_support": {
                "routes": len(routes),
                "runtime_complete": len(complete),
                "local_only": len(routes) - len(complete),
                "primitive_minimum": minimum,
                "primitive_maximum": max(len(row["actions"]) for row in routes),
            },
            "autonomous_production": {
                "attempts": arm["attempts"],
                "complete": arm["complete"],
                "exact_execution_precision": arm["exact_execution_precision"],
                "teacher_scale_programs": _scale_count(
                    arm["primitive_count_distribution"], minimum
                ),
                "teacher_scale_rate": _scale_count(arm["primitive_count_distribution"], minimum)
                / arm["attempts"],
                "exact_endpoint_support": arm["exact_endpoint_support"],
                "transformation_equivalent_support": arm["transformation_equivalent_support"],
            },
            "teacher_route_probability_without_injection": {
                "evaluated_complete_routes": len(log_probabilities),
                "finite_support": len(finite),
                "best_log10_probability": max(finite) if finite else None,
                "median_log10_probability": (float(np.median(finite)) if finite else None),
                "worst_log10_probability": min(finite) if finite else None,
                "interpretation": (
                    "exact teacher trace probability under the clean factorized "
                    "production law; alternative equal-utility programs are not counted"
                ),
            },
            "productive_continuation_factor_ranks": gate["productive_continuation_rank"][task],
        }

    historical_rows = {
        (str(row["parent_smiles"]), str(row["smiles"])): row
        for row in _historical_candidate_rows(history)
    }
    children = {child for _, child in historical_rows}
    route_scale_children = {
        child
        for (_, child), row in historical_rows.items()
        if int(row.get("primitive_count") or 0) >= 9
    }
    jump_refinement = sum(
        parent in route_scale_children and int(row.get("primitive_count") or 0) <= 8
        for (parent, _), row in historical_rows.items()
    )
    historical_mechanism = {
        "candidate_edges": len(historical_rows),
        "recursive_descendant_edges": sum(parent in children for parent, _ in historical_rows),
        "maximum_primitive_count": max(
            int(row.get("primitive_count") or 0) for row in historical_rows.values()
        ),
        "route_scale_jumps_at_least_9_primitives": len(route_scale_children),
        "route_scale_jump_then_local_refinement_edges": jump_refinement,
        "conclusion": (
            "recursive reward adaptation is supported, but this ledger contains no "
            "route-scale jump and therefore does not validate jump-plus-refinement"
        ),
    }

    payload = {
        "schema_version": "pmo_complete_route_gap_analysis_v1",
        "decision": "NO_SCORED_LAUNCH_PRODUCTION_SUPPORT_INSUFFICIENT",
        "new_oracle_calls": 0,
        "scored_launch_authorized": False,
        "headline_lane": {
            "prescreen": False,
            "oracle_budget": 10000,
            "metric": "top-10 AUC with finish=True",
            "answer_known_teacher_scores_are_comparators": False,
        },
        "inputs": {
            path: sha256_file(root / path)
            for path in (GATE, CHECKPOINT, CORPUS, LEGAL_RUNTIME, PARITY, HISTORY)
        },
        "tasks": task_rows,
        "historical_jump_refinement": historical_mechanism,
        "diagnosis": (
            "Program scale and exact execution are now in support on Celecoxib and "
            "GSK3B, but independent position/rule and WHERE/HOW decisions destroy "
            "whole-program probability. Perindopril additionally requires recursive "
            "composition because every witness exceeds the 32-primitive action horizon."
        ),
        "one_next_architecture": {
            "name": "joint_dependency_region_jump_plus_refinement_controller",
            "change": (
                "replace independent per-position route draws with a source-conditioned "
                "joint dependency-region latent that preserves rule, component-control, "
                "created-handle and WHERE/HOW correlations across a complete <=32-step "
                "jump; keep unchanged v0 exploration, FiberControl and the recursive "
                "archive, then learn short refinement tails from scored descendants"
            ),
            "reused": [
                "sealed 184-route exact corpus and frozen folds",
                "same dependency-region representation and exact compiler",
                "unchanged Dynamic-v0 exploration",
                "FiberControl and shared recursive archive",
            ],
            "deprioritized": [
                "independent position-conditioned complete-route sampler",
                "more tuning of one-step WHERE/HOW ranks",
                "eight-primitive static factorial",
            ],
        },
        "zero_oracle_promotion_gate": {
            "candidate_budget_per_source": 32,
            "requirements": [
                "100% exact execution precision among emitted complete programs",
                "at least 25% teacher-scale proposals on Celecoxib and GSK3B",
                "nonzero exact or transformation-equivalent autonomous support on both Celecoxib and GSK3B",
                "at least one autonomous <=32-step jump followed by an executable descendant refinement on Perindopril",
                "no task/route/endpoint lookup in the runtime checkpoint",
            ],
            "kill": (
                "abandon this joint-latent formulation if a 32-candidate source panel "
                "still has zero exact/equivalent support on either runtime-supported task"
            ),
        },
        "first_meaningful_scored_run_after_gate": {
            "tasks": list(SELECTED_TASKS),
            "development_calls_per_task": 1000,
            "initialization_calls_counted": True,
            "recursive_descendants_required": True,
            "primary_metric": "top-10 AUC with flat tail to the official 10000-call denominator",
            "comparators": {
                task: row["headline_comparator"]["auc"] for task, row in task_rows.items()
            },
            "promotion": (
                "beat IVG no-prescreen AUC on at least two of three tasks and finish "
                "within 0.02 on the third, with at least one winning lineage containing "
                "a route-scale jump and scored descendant refinement"
            ),
            "kill": (
                "stop before broader PMO if the joint lane emits teacher-scale candidates "
                "but FiberControl never continues an improving jump by 250 calls, or if "
                "1000-call flat-tail AUC trails IVG by more than 0.05 on two tasks"
            ),
        },
    }
    _write_sealed(root / OUTPUT, payload)
    print(json.dumps({"output": OUTPUT, "new_oracle_calls": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
