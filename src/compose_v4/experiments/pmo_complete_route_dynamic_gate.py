"""Zero-oracle gate for complete PMO programs plus recursive FiberControl replay.

This module deliberately keeps three evidence roles separate:

* the frozen route corpus fits a task-family/lineage-held-out, sanitized complete-
  program proposal law;
* teacher states and endpoints are used only after generation for support auditing;
* historical PMO outcomes drive a retrospective recursive archive replay and are
  never visible to either proposal law.

The prior eight-primitive factorial remains an immutable negative result.  This gate
does not call an oracle and does not authorize a prospective run.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.fiber_control import (
    ProgramValue,
    SearchState,
    acquisition,
    program_features,
)
from compose_v4.experiments.pmo_dependency_region_program import (
    DependencyRegionConfig,
    dependency_region_program,
)
from compose_v4.experiments.pmo_legal_action_policy import (
    RULES,
    action_features,
    enumerate_rule_successors,
    score_features,
)
from compose_v4.experiments.pmo_route_fiber_production_yield import (
    TASK_FAMILIES,
    TASK_FOLDS,
    _fold_checkpoints,
    _sample_legal_successor,
    sample_route_program,
    sample_v0_program,
)
from compose_v4.experiments.route_proposal_quality import transformation_equivalent
from compose_v4.experiments.whole_ring_plan import execute_program
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SCHEMA = "pmo_complete_route_dynamic_gate_v1"
CHECKPOINT_SCHEMA = "pmo_complete_route_fold_checkpoint_v1"
MAXIMUM_PRIMITIVES = 32
MAXIMUM_COMPONENTS = 8
LENGTH_SMOOTHING = 0.02
RULE_SMOOTHING = 0.02
SEED = 20260919

SELECTED_TASKS = ("celecoxib_rediscovery", "gsk3b", "perindopril_mpo")
ARMS = ("old_eight_primitive", "unchanged_v0", "complete_route")

RUNTIME_FORBIDDEN_KEYS = {
    "action",
    "actions",
    "endpoint",
    "lineage_identity",
    "member",
    "members",
    "route",
    "route_identity",
    "smiles",
    "source_graph",
    "source_state",
    "state",
    "states",
    "task",
    "task_family",
    "terminal_endpoint",
    "trace_identity",
}


def load_envelope(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else path.open
    arguments = (path, "rt") if path.suffix == ".gz" else ("r",)
    with opener(*arguments) as handle:
        envelope = json.load(handle)
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"invalid self-hashed envelope: {path}")
    if identity(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"payload hash mismatch: {path}")
    return envelope["payload"]


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(map(str, value)) | set().union(
            *(_recursive_keys(child) for child in value.values()), set()
        )
    if isinstance(value, list):
        return set().union(*(_recursive_keys(child) for child in value), set())
    return set()


def _member_tasks(route: dict[str, Any]) -> set[str]:
    return {str(member["task"]) for member in route.get("members", ())}


def _validate_corpus(corpus: dict[str, Any]) -> None:
    routes = corpus.get("routes")
    split = corpus.get("split")
    if not isinstance(routes, list) or len(routes) != 184 or not isinstance(split, dict):
        raise ValueError("sealed PMO dependency-region corpus changed")
    if split.get("frozen_before_segmentation_or_panel_generation") is not True:
        raise ValueError("PMO folds were not frozen before representation extraction")
    lineage_fold = split.get("lineage_test_fold")
    if not isinstance(lineage_fold, dict) or len(lineage_fold) != 18:
        raise ValueError("frozen PMO lineage split changed")
    for route in routes:
        if lineage_fold.get(route["lineage_identity"]) != route["test_fold"]:
            raise ValueError("route crossed its frozen shared-lineage fold")
        if len(route["states"]) != len(route["actions"]) + 1:
            raise ValueError("route action/state alignment changed")
        if not route["dependency_region_program"].get("exact_replay"):
            raise ValueError("route lost exact dependency-region replay")


def _weighted_histogram(
    routes: list[dict[str, Any]], value_fn, support: Iterable[Any], smoothing: float
) -> tuple[list[Any], list[float]]:
    by_lineage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in routes:
        by_lineage[str(route["lineage_identity"])].append(route)
    counts = {value: float(smoothing) for value in support}
    for rows in by_lineage.values():
        weight = 1.0 / len(by_lineage) / len(rows)
        for route in rows:
            value = value_fn(route)
            if value not in counts:
                raise ValueError(f"value outside frozen support: {value}")
            counts[value] += weight
    total = sum(counts.values())
    ordered = list(support)
    return ordered, [counts[value] / total for value in ordered]


def fit_complete_route_checkpoints(corpus: dict[str, Any]) -> dict[str, Any]:
    """Fit only generic length and position/rule laws after the frozen split."""

    _validate_corpus(corpus)
    routes = corpus["routes"]
    folds = []
    audits = {}
    for fold in range(3):
        held = [row for row in routes if int(row["test_fold"]) == fold]
        training = [
            row
            for row in routes
            if int(row["test_fold"]) != fold
            and row["dependency_region_program"].get("complete_representation_supported")
        ]
        held_lineages = {str(row["lineage_identity"]) for row in held}
        training_lineages = {str(row["lineage_identity"]) for row in training}
        if not training or training_lineages & held_lineages:
            raise ValueError(f"fold {fold} is empty or leaks a shared lineage")

        lengths, length_probabilities = _weighted_histogram(
            training,
            lambda row: len(row["actions"]),
            range(1, MAXIMUM_PRIMITIVES + 1),
            LENGTH_SMOOTHING,
        )
        position_rules = []
        for position in range(MAXIMUM_PRIMITIVES):
            present = [row for row in training if len(row["actions"]) > position]
            classes, probabilities = _weighted_histogram(
                present or training,
                lambda row, p=position: (
                    row["actions"][p]["executor_rule"]
                    if len(row["actions"]) > p
                    else row["actions"][-1]["executor_rule"]
                ),
                RULES,
                RULE_SMOOTHING,
            )
            position_rules.append(
                {"position": position, "classes": classes, "probabilities": probabilities}
            )
        body = {
            "schema_version": CHECKPOINT_SCHEMA,
            "fold_index": fold,
            "split_identity": corpus["split"]["split_identity"],
            "maximum_primitives": MAXIMUM_PRIMITIVES,
            "maximum_components": MAXIMUM_COMPONENTS,
            "lengths": lengths,
            "length_probabilities": length_probabilities,
            "position_rule_probabilities": position_rules,
            "training_identity": identity(
                {
                    "split_identity": corpus["split"]["split_identity"],
                    "fold": fold,
                    "complete_training_traces": len(training),
                    "training_lineages": len(training_lineages),
                }
            ),
        }
        if RUNTIME_FORBIDDEN_KEYS & _recursive_keys(body):
            raise ValueError("complete-route checkpoint retained teacher content")
        folds.append({**body, "checkpoint_id": identity(body)})
        audits[str(fold)] = {
            "complete_training_traces": len(training),
            "training_lineages": len(training_lineages),
            "held_traces": len(held),
            "held_lineages": len(held_lineages),
            "lineage_overlap": len(training_lineages & held_lineages),
            "held_complete_routes": sum(
                row["dependency_region_program"].get("complete_representation_supported", False)
                for row in held
            ),
            "held_local_only_routes": sum(
                not row["dependency_region_program"].get("complete_representation_supported", False)
                for row in held
            ),
        }
    payload = {
        "schema_version": "pmo_complete_route_fold_checkpoints_v1",
        "split_identity": corpus["split"]["split_identity"],
        "folds": folds,
        "new_oracle_calls": 0,
    }
    if RUNTIME_FORBIDDEN_KEYS & _recursive_keys(payload):
        raise ValueError("runtime payload retained teacher content")
    return {"payload": payload, "payload_sha256": identity(payload), "audit": audits}


def _weighted_order(classes: list[str], probabilities: list[float], rng) -> list[str]:
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(classes),) or np.any(values <= 0):
        raise ValueError("proposal probabilities lost full support")
    values /= values.sum()
    keys = -np.log(np.maximum(rng.random(len(classes)), np.finfo(float).tiny)) / values
    return [classes[index] for index in np.argsort(keys)]


def sample_complete_route_program(
    source, checkpoint: dict[str, Any], legal_checkpoint: dict[str, Any], rng
) -> dict[str, Any]:
    """Sample one exact, dependency-region-validated program up to 32 primitives."""

    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("invalid complete-route checkpoint")
    lengths = np.asarray(checkpoint["lengths"], dtype=int)
    length_probabilities = np.asarray(checkpoint["length_probabilities"], dtype=float)
    target_length = int(rng.choice(lengths, p=length_probabilities))
    graph = source
    source_key = canonical_state_key(source)
    states = [encode_state(source)]
    actions = []
    telemetry = {
        "target_length": target_length,
        "empty_rule_fibers": 0,
        "legal_successors_enumerated": 0,
        "chosen_rule_sequence": [],
    }
    for position in range(target_length):
        law = checkpoint["position_rule_probabilities"][position]
        advanced = False
        for rule in _weighted_order(law["classes"], law["probabilities"], rng):
            candidates = tuple(enumerate_rule_successors(graph, rule))
            telemetry["legal_successors_enumerated"] += len(candidates)
            if not candidates:
                telemetry["empty_rule_fibers"] += 1
                continue
            chosen, _, _ = _sample_legal_successor(graph, candidates, legal_checkpoint, rng)
            graph = chosen.successor
            states.append(encode_state(graph))
            actions.append(chosen.action_record)
            telemetry["chosen_rule_sequence"].append(rule)
            advanced = True
            break
        if not advanced:
            return {"status": "no_legal_successor", "telemetry": telemetry}
    program = dependency_region_program(
        states,
        actions,
        config=DependencyRegionConfig(
            runtime_maximum_primitives=MAXIMUM_PRIMITIVES,
            runtime_maximum_components=MAXIMUM_COMPONENTS,
        ),
    )
    if not program["exact_replay"]:
        raise RuntimeError("complete-route proposal failed exact replay")
    if not program["component_budget_supported"]:
        return {
            "status": "component_budget_abstention",
            "primitive_count": len(actions),
            "component_count": program["component_count"],
            "telemetry": telemetry,
        }
    endpoint = canonical_state_key(graph)
    if endpoint == source_key:
        return {"status": "self_endpoint", "telemetry": telemetry}
    return {
        "status": "complete",
        "endpoint": endpoint,
        "endpoint_state": states[-1],
        "primitive_count": len(actions),
        "component_count": program["component_count"],
        "exact_replay": True,
        "created_dependency_edges": program["created_dependency_edges"],
        "cycle_dependency_edges": program["cycle_dependency_edges"],
        "telemetry": telemetry,
    }


def teacher_forced_support(corpus: dict[str, Any]) -> dict[str, Any]:
    """Measure the runtime contract without changing or sampling the answer."""

    _validate_corpus(corpus)
    tasks = {}
    for task in SELECTED_TASKS:
        routes = [row for row in corpus["routes"] if task in _member_tasks(row)]
        complete = [
            row
            for row in routes
            if row["dependency_region_program"].get("complete_representation_supported")
        ]
        runtime_replays = []
        for row in complete:
            endpoint, receipt = execute_program(decode_state(row["states"][0]), row["actions"])
            runtime_replays.append(
                receipt["states"] == row["states"]
                and canonical_state_key(endpoint)
                == canonical_state_key(decode_state(row["states"][-1]))
            )
        tasks[task] = {
            "routes": len(routes),
            "complete_runtime_routes": len(complete),
            "local_decision_only_routes": len(routes) - len(complete),
            "representability": len(complete) / len(routes) if routes else None,
            "exact_replay_precision": (
                sum(runtime_replays) / len(runtime_replays) if runtime_replays else None
            ),
            "same_runtime_replayed_routes": sum(runtime_replays),
            "primitive_length_min": min(len(row["actions"]) for row in routes),
            "primitive_length_max": max(len(row["actions"]) for row in routes),
            "complete_route_abstention": not bool(complete),
        }
    return tasks


def _first_task_route_state_rows(corpus: dict[str, Any], task: str) -> list[dict]:
    """Collapse duplicate answer-known states without leaking them into fitting."""

    grouped: dict[str, dict[str, Any]] = {}
    routes = [row for row in corpus["routes"] if task in _member_tasks(row)]
    for route in routes:
        # ``terminal_endpoint`` is the canonical key; the lossless graph is the
        # final frozen trace state.
        terminal = decode_state(route["states"][-1])
        terminal_key = canonical_state_key(terminal)
        for position, (state, action, successor_state) in enumerate(
            zip(route["states"][:-1], route["actions"], route["states"][1:], strict=True)
        ):
            graph = decode_state(state)
            key = canonical_state_key(graph)
            row = grouped.setdefault(
                key,
                {
                    "state": state,
                    "source_key": key,
                    "positions": [],
                    "next_successors": set(),
                    "next_rules": defaultdict(set),
                    "terminals": {},
                    "previous_rules": set(),
                },
            )
            row["positions"].append(position)
            successor_key = canonical_state_key(decode_state(successor_state))
            row["next_successors"].add(successor_key)
            row["next_rules"][action["executor_rule"]].add(successor_key)
            row["terminals"][terminal_key] = terminal
            row["previous_rules"].add(
                route["actions"][position - 1]["executor_rule"] if position else "<START>"
            )
    return [grouped[key] for key in sorted(grouped)]


def _candidate_outcome(row: dict, teacher: dict) -> tuple[bool, bool]:
    if row.get("status") != "complete":
        return False, False
    exact = row["endpoint"] in teacher["terminals"]
    candidate = decode_state(row["endpoint_state"]) if row.get("endpoint_state") else None
    equivalent = False
    if candidate is not None:
        source = decode_state(teacher["state"])
        equivalent = any(
            transformation_equivalent(source, candidate, target)
            for target in teacher["terminals"].values()
        )
    return exact, equivalent


def _arm_summary(rows: list[dict]) -> dict[str, Any]:
    complete = [row for row in rows if row["candidate"].get("status") == "complete"]
    endpoints = [row["candidate"]["endpoint"] for row in complete]
    return {
        "attempts": len(rows),
        "complete": len(complete),
        "validity": len(complete) / len(rows) if rows else None,
        "unique_endpoints": len(set(endpoints)),
        "unique_yield": len(set(endpoints)) / len(rows) if rows else None,
        "exact_execution_precision": (
            sum(row["candidate"].get("exact_replay") is True for row in complete) / len(complete)
            if complete
            else None
        ),
        "exact_endpoint_support": sum(row["exact_endpoint"] for row in rows),
        "transformation_equivalent_support": sum(row["transformation_equivalent"] for row in rows),
        "primitive_count_distribution": {
            str(length): count
            for length, count in sorted(
                Counter(row["candidate"]["primitive_count"] for row in complete).items()
            )
        },
        "failure_counts": dict(
            sorted(
                Counter(
                    row["candidate"].get("status", "missing")
                    for row in rows
                    if row["candidate"].get("status") != "complete"
                ).items()
            )
        ),
    }


def run_matched_known_answer_support(
    corpus: dict[str, Any],
    old_runtime: dict[str, Any],
    complete_runtime: dict[str, Any],
    legal_runtime: dict[str, Any],
    *,
    seed: int = SEED,
    limit_states: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """One actual proposal per arm at every unique teacher state."""

    old = {int(row["fold_index"]): row for row in old_runtime["folds"]}
    complete = {int(row["fold_index"]): row for row in complete_runtime["folds"]}
    legal = _fold_checkpoints(legal_runtime)
    if sorted(old) != [0, 1, 2] or sorted(complete) != [0, 1, 2]:
        raise ValueError("matched runtime checkpoint census changed")
    tasks = {}
    for task_index, task in enumerate(SELECTED_TASKS):
        fold = TASK_FOLDS[task]
        states = _first_task_route_state_rows(corpus, task)
        if limit_states is not None:
            states = states[:limit_states]
        arms = {name: [] for name in ARMS}
        arguments = [
            (
                teacher,
                old[fold],
                complete[fold],
                legal[fold],
                seed,
                task_index,
                state_index,
            )
            for state_index, teacher in enumerate(states)
        ]
        if workers > 1 and len(arguments) > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                evaluated = list(pool.map(_evaluate_known_state, arguments))
        else:
            evaluated = list(map(_evaluate_known_state, arguments))
        for candidates in evaluated:
            for arm, outcome in candidates.items():
                arms[arm].append(outcome)
        tasks[task] = {
            "held_fold": fold,
            "held_task_family": TASK_FAMILIES[task],
            "unique_teacher_states": len(states),
            "candidate_budget_per_state_per_arm": 1,
            "arms": {name: _arm_summary(rows) for name, rows in arms.items()},
        }
    return tasks


def _evaluate_known_state(arguments) -> dict[str, dict[str, Any]]:
    teacher, old, complete, legal, seed, task_index, state_index = arguments
    source = decode_state(teacher["state"])
    candidates = {
        "old_eight_primitive": sample_route_program(
            source,
            old,
            legal,
            np.random.default_rng([seed, task_index, state_index, 0]),
        ),
        "unchanged_v0": sample_v0_program(
            source,
            np.random.default_rng([seed, task_index, state_index, 1]),
            0,
        ),
        "complete_route": sample_complete_route_program(
            source,
            complete,
            legal,
            np.random.default_rng([seed, task_index, state_index, 2]),
        ),
    }
    output = {}
    for arm, candidate in candidates.items():
        exact, equivalent = _candidate_outcome(candidate, teacher)
        output[arm] = {
            "candidate": candidate,
            "exact_endpoint": exact,
            "transformation_equivalent": equivalent,
        }
    return output


def productive_continuation_ranks(
    corpus: dict[str, Any],
    complete_runtime: dict[str, Any],
    legal_runtime: dict[str, Any],
    *,
    limit_states: int | None = None,
) -> dict[str, Any]:
    """Rank known next successors after fitting; teachers are never injected."""

    complete = {int(row["fold_index"]): row for row in complete_runtime["folds"]}
    legal = _fold_checkpoints(legal_runtime)
    output = {}
    for task in SELECTED_TASKS:
        fold = TASK_FOLDS[task]
        rule_ranks = []
        where_how_ranks = []
        missing = 0
        teacher_states = _first_task_route_state_rows(corpus, task)
        if limit_states is not None:
            teacher_states = teacher_states[:limit_states]
        for teacher in teacher_states:
            graph = decode_state(teacher["state"])
            position = min(min(teacher["positions"]), MAXIMUM_PRIMITIVES - 1)
            law = complete[fold]["position_rule_probabilities"][position]
            rule_probability = dict(zip(law["classes"], law["probabilities"], strict=True))
            ordered_rules = sorted(RULES, key=lambda rule: (-rule_probability[rule], rule))
            teacher_rule_ranks = [ordered_rules.index(rule) + 1 for rule in teacher["next_rules"]]
            rule_ranks.append(min(teacher_rule_ranks))
            found = []
            # Enumerating every legal rule fiber at every route state is needlessly
            # quadratic.  The factorization is explicit, so report rule rank and
            # conditional WHERE/HOW rank separately.
            for rule, target_successors in teacher["next_rules"].items():
                fiber = tuple(enumerate_rule_successors(graph, rule))
                if not fiber:
                    continue
                values = np.asarray(
                    [score_features(legal[fold], action_features(graph, row)) for row in fiber]
                )
                ranked = sorted(
                    zip(values, (row.successor_key for row in fiber), strict=True),
                    key=lambda row: (-row[0], row[1]),
                )
                found.extend(
                    index for index, (_, key) in enumerate(ranked, 1) if key in target_successors
                )
            if found:
                where_how_ranks.append(min(found))
            else:
                missing += 1
        output[task] = {
            "states": len(where_how_ranks) + missing,
            "runtime_supported_productive_successors": len(where_how_ranks),
            "coverage": (
                len(where_how_ranks) / (len(where_how_ranks) + missing)
                if where_how_ranks or missing
                else None
            ),
            "rule_median_rank": float(np.median(rule_ranks)) if rule_ranks else None,
            "rule_top1_recall": (
                sum(rank <= 1 for rank in rule_ranks) / len(rule_ranks) if rule_ranks else None
            ),
            "where_how_conditional_median_rank": (
                float(np.median(where_how_ranks)) if where_how_ranks else None
            ),
            "where_how_conditional_top1_recall": (
                sum(rank <= 1 for rank in where_how_ranks) / len(where_how_ranks)
                if where_how_ranks
                else None
            ),
            "where_how_conditional_top8_recall": (
                sum(rank <= 8 for rank in where_how_ranks) / len(where_how_ranks)
                if where_how_ranks
                else None
            ),
            "where_how_conditional_top32_recall": (
                sum(rank <= 32 for rank in where_how_ranks) / len(where_how_ranks)
                if where_how_ranks
                else None
            ),
            "joint_rank": "unsupported; factor ranks reported without enumerating unrelated fibers",
            "teacher_injection": False,
        }
    return output


def _historical_candidate_rows(value: Any) -> list[dict[str, Any]]:
    rows = []
    if isinstance(value, dict):
        required = {"parent_smiles", "parent_score", "smiles", "score"}
        if required <= set(value):
            rows.append(value)
        for child in value.values():
            rows.extend(_historical_candidate_rows(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_historical_candidate_rows(child))
    return rows


def _replay_record(row: dict[str, Any], state: SearchState) -> dict[str, Any]:
    change = row.get("structural_change") or {}
    bundle = row.get("bundle") or {}
    option = str(bundle.get("option", "historical"))
    record = {
        "parent_score": -float(row["parent_score"]),
        "similarity": 1.0,
        "delta": 0.0,
        "qed": 1.0,
        "sa": 0.0,
        "regions": int(change.get("n_changed_regions", 1)),
        "created": int(change.get("n_inserted", 0)),
        "deleted": int(change.get("n_deleted", 0)),
        "families": [option],
    }
    return {
        "features": program_features(record, state),
        "parent_score": record["parent_score"],
        "fingerprint": {
            option,
            f"regions:{record['regions']}",
            f"created:{record['created']}",
            f"deleted:{record['deleted']}",
        },
    }


def historical_recursive_fiber_replay(
    historical_payload: dict[str, Any],
    *,
    seed: int = SEED,
    batch: int = 8,
    maximum_calls: int = 32,
    adaptive: bool = True,
) -> dict[str, Any]:
    """Replay only pre-existing outcomes through a recursive shared archive.

    Scores are hidden until after selection.  Negated PMO rewards make the unchanged
    docking-oriented FiberControl minimization convention equivalent to maximization.
    """

    raw = _historical_candidate_rows(historical_payload)
    dedup = {}
    for row in raw:
        key = (str(row["parent_smiles"]), str(row["smiles"]))
        dedup.setdefault(key, row)
    rows = list(dedup.values())
    child_smiles = {str(row["smiles"]) for row in rows}
    roots = sorted({str(row["parent_smiles"]) for row in rows} - child_smiles)
    root_scores = {
        str(row["parent_smiles"]): float(row["parent_score"])
        for row in rows
        if str(row["parent_smiles"]) in roots
    }
    state = SearchState(
        archive={smiles: -score for smiles, score in root_scores.items()},
        budget=min(maximum_calls, len(rows)),
    )
    value = ProgramValue()
    rng = np.random.default_rng(seed)
    observations = []
    selected_edges = set()
    depth = {root: 0 for root in roots}
    rounds = []
    while state.budget > 0:
        available = [
            row
            for row in rows
            if str(row["parent_smiles"]) in state.archive
            and (str(row["parent_smiles"]), str(row["smiles"])) not in selected_edges
        ]
        if not available:
            break
        candidates = [_replay_record(row, state) for row in available]
        room = min(batch, state.budget, len(candidates))
        if adaptive:
            chosen = acquisition(
                candidates,
                value,
                state,
                rng,
                batch=room,
                exploration=min(2, room),
            )
        else:
            chosen = list(map(int, rng.choice(len(candidates), room, replace=False)))
        before = max((-score for score in state.archive.values()), default=float("-inf"))
        for index in chosen:
            row = available[index]
            parent = str(row["parent_smiles"])
            child = str(row["smiles"])
            selected_edges.add((parent, child))
            score = float(row["score"])
            parent_score = float(row["parent_score"])
            state.archive[child] = -score
            depth[child] = max(depth.get(child, 0), depth.get(parent, 0) + 1)
            observations.append((_replay_record(row, state)["features"], score - parent_score))
        if observations:
            value.fit([row[0] for row in observations], [row[1] for row in observations])
        after = max((-score for score in state.archive.values()), default=float("-inf"))
        state.budget -= len(chosen)
        state.rounds += 1
        state.history.append({"improved": after > before})
        rounds.append(
            {
                "round": state.rounds,
                "available": len(available),
                "selected": len(chosen),
                "incumbent_before": before,
                "incumbent_after": after,
            }
        )
        if not chosen:
            break
    improving = sum(
        float(row["score"]) > float(row["parent_score"])
        for row in rows
        if (str(row["parent_smiles"]), str(row["smiles"])) in selected_edges
    )
    return {
        "evidence_role": "retrospective_historical_replay_not_prospective",
        "controller": "fiber_control" if adaptive else "reward_blind",
        "maximum_calls": maximum_calls,
        "historical_candidate_edges": len(rows),
        "root_parents": len(roots),
        "selected_edges": len(selected_edges),
        "selected_improving_edges": improving,
        "maximum_selected_generation": max(depth.values(), default=0),
        "multi_generation_lineage_observed": max(depth.values(), default=0) >= 2,
        "best_reward": max((-score for score in state.archive.values()), default=None),
        "rounds": rounds,
        "scores_hidden_until_selection": True,
        "new_oracle_calls": 0,
    }
