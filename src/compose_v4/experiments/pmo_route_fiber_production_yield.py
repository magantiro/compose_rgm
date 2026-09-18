"""Split-clean PMO route-prior production sampler and matched yield gate.

The route prior is a fitted aggregate transition law over executor-rule sequences.
It contains no teacher route, molecule, endpoint or binding.  At runtime each
sampled rule is bound by the already-qualified held-family legal-action ranker and
executed exactly.  The comparison spends no PMO oracle calls.
"""

from __future__ import annotations

import gzip
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis import (
    CAPACITY_AWARE_THRESHOLD,
    CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS,
    synthesize_dynamic_program,
    synthesize_named_module_sequence,
)
from compose_v4.experiments.continuation_profile import verify_file
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
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

SCHEMA = "pmo_route_fiber_production_yield_v3"
CHECKPOINT_SCHEMA = "pmo_loto_route_transition_checkpoint_v1"
CONTRACT_SCHEMA = "pmo_route_fiber_production_yield_contract_v3"
CONTRACT = "configs/pmo_route_fiber_production_yield_v3.json"
CHECKPOINT = "diagnostics/pmo_route_fiber_pilot/route_transition_checkpoint_v3.json"
RESULT = "diagnostics/pmo_route_fiber_pilot/production_yield_v3.json"

CORPUS = (
    "diagnostics/pmo_dependency_region_program_v2/attempt_1/"
    "training_dependency_region_corpus.json.gz"
)
DEPENDENCY_RUNTIME = (
    "diagnostics/pmo_dependency_region_policy_comparison/attempt_2/"
    "runtime_fold_checkpoints.json.gz"
)
LEGAL_RUNTIME = (
    "diagnostics/pmo_legal_action_policy/attempt_2/"
    "runtime_fold_checkpoints.json.gz"
)
INITIALIZATION = "diagnostics/parent_edit_cycles/prepared/init_20260921.json"

EXPECTED_SHA256 = {
    CORPUS: "9b35529eae0975dcf977bc1fc51f3bd838e0184d51de3a7d2822edd0bc3d8b54",
    DEPENDENCY_RUNTIME: "5e08b8e2c4f765880428ab4b4ae3afd436cb4021aea4e3248bb010d34243d5d6",
    LEGAL_RUNTIME: "e6336cfbcccef71b08434bde4e22a9f2b73cb3d1869f82caa844a87520c2a3a4",
    INITIALIZATION: "a1f7840df1f75567565de633c4df3d013c3165df3e43cbf2ac87b156d5f206ff",
}

TASK_FOLDS = {
    "celecoxib_rediscovery": 1,
    "gsk3b": 0,
    "perindopril_mpo": 2,
}
TASK_FAMILIES = {
    "celecoxib_rediscovery": "rediscovery",
    "gsk3b": "bioactivity",
    "perindopril_mpo": "mpo",
}

START = "<START>"
STOP = "<STOP>"
TOKENS = (*RULES, STOP)
SEED = 20260918
ATTEMPTS_PER_SOURCE_ARM = 2
ROUTE_ATTEMPTS_PER_SOURCE = 1
MAXIMUM_PRIMITIVES = 8
MAXIMUM_COMPONENTS = 8
TRANSITION_SMOOTHING = 0.25
ACTION_EXPLORATION = 0.10
ACTION_TEMPERATURE = 1.0

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


def _load_envelope(path: Path, *, compressed: bool) -> dict[str, Any]:
    opener = gzip.open if compressed else path.open
    arguments = (path, "rt") if compressed else ("r",)
    with opener(*arguments) as handle:
        envelope = json.load(handle)
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"invalid self-hashed envelope: {path}")
    if identity(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"payload hash mismatch: {path}")
    return envelope["payload"]


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(map(str, value))
        for child in value.values():
            keys.update(_recursive_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_recursive_keys(child))
        return keys
    return set()


def _fold_checkpoints(runtime: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = {}
    for item in runtime.get("folds", ()):  # legal and dependency use two wrappers
        checkpoint = item.get("checkpoint", item)
        fold = int(item.get("fold", checkpoint.get("fold_index", -1)))
        if fold in rows:
            raise ValueError(f"duplicate fold checkpoint: {fold}")
        rows[fold] = checkpoint
    if sorted(rows) != [0, 1, 2]:
        raise ValueError("runtime checkpoints must contain folds 0, 1 and 2")
    return rows


def _member_tasks(route: dict[str, Any]) -> set[str]:
    return {str(member["task"]) for member in route.get("members", ())}


def fit_route_transition_checkpoint(corpus: dict[str, Any]) -> dict[str, Any]:
    """Fit source-balanced transition tables after the frozen family split."""

    routes = corpus.get("routes")
    split = corpus.get("split")
    if not isinstance(routes, list) or len(routes) != 184 or not isinstance(split, dict):
        raise ValueError("exact PMO dependency-region corpus changed")
    if split.get("frozen_before_segmentation_or_panel_generation") is not True:
        raise ValueError("PMO split was not frozen before representation work")
    lineage_fold = split.get("lineage_test_fold")
    if not isinstance(lineage_fold, dict) or len(lineage_fold) != 18:
        raise ValueError("PMO lineage split changed")

    for route in routes:
        if lineage_fold.get(route["lineage_identity"]) != route["test_fold"]:
            raise ValueError("PMO route crossed its frozen lineage fold")
        if len(route["states"]) != len(route["actions"]) + 1:
            raise ValueError("PMO exact route lost action/state alignment")
        program = route["dependency_region_program"]
        if not program.get("exact_replay"):
            raise ValueError("PMO dependency-region route is not exact")

    folds = []
    audit = {}
    for fold in range(3):
        training = [route for route in routes if int(route["test_fold"]) != fold]
        held = [route for route in routes if int(route["test_fold"]) == fold]
        by_lineage: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for route in training:
            by_lineage[str(route["lineage_identity"])].append(route)
        if not training or not held or not by_lineage:
            raise ValueError(f"empty PMO training or held fold: {fold}")

        counts = {
            previous: {token: TRANSITION_SMOOTHING for token in TOKENS}
            for previous in (START, *RULES)
        }
        for lineage_routes in by_lineage.values():
            route_weight = 1.0 / len(by_lineage) / len(lineage_routes)
            for route in lineage_routes:
                rules = tuple(action["executor_rule"] for action in route["actions"])
                if not rules or any(rule not in RULES for rule in rules):
                    raise ValueError("PMO route uses an unsupported executor rule")
                previous = START
                for rule in rules:
                    counts[previous][rule] += route_weight
                    previous = rule
                counts[previous][STOP] += route_weight

        transitions = {}
        for previous, values in counts.items():
            total = sum(values.values())
            transitions[previous] = {
                "classes": list(TOKENS),
                "probabilities": [values[token] / total for token in TOKENS],
            }
        body = {
            "schema_version": CHECKPOINT_SCHEMA,
            "fold_index": fold,
            "split_identity": split["split_identity"],
            "maximum_primitives": MAXIMUM_PRIMITIVES,
            "maximum_components": MAXIMUM_COMPONENTS,
            "transition_smoothing": TRANSITION_SMOOTHING,
            "action_exploration": ACTION_EXPLORATION,
            "action_temperature": ACTION_TEMPERATURE,
            "transitions": transitions,
            "training_identity": identity(
                {
                    "input": EXPECTED_SHA256[CORPUS],
                    "split_identity": split["split_identity"],
                    "fold": fold,
                    "training_trace_count": len(training),
                    "training_lineage_count": len(by_lineage),
                }
            ),
        }
        if RUNTIME_FORBIDDEN_KEYS & _recursive_keys(body):
            raise ValueError("sanitized PMO route checkpoint retained teacher fields")
        folds.append({**body, "checkpoint_id": identity(body)})
        audit[str(fold)] = {
            "training_traces": len(training),
            "held_traces": len(held),
            "training_lineages": len(by_lineage),
            "held_lineages": len({route["lineage_identity"] for route in held}),
            "lineage_overlap": len(
                set(by_lineage)
                & {str(route["lineage_identity"]) for route in held}
            ),
            "held_task_families": sorted(
                {str(route["task_family"]) for route in held}
            ),
        }

    for task, fold in TASK_FOLDS.items():
        task_routes = [route for route in routes if task in _member_tasks(route)]
        if not task_routes or any(int(route["test_fold"]) != fold for route in task_routes):
            raise ValueError(f"pilot task is not wholly held out by fold: {task}")
    payload = {
        "schema_version": "pmo_loto_route_transition_checkpoints_v1",
        "split_identity": split["split_identity"],
        "folds": folds,
        "new_oracle_calls": 0,
    }
    if RUNTIME_FORBIDDEN_KEYS & _recursive_keys(payload):
        raise ValueError("PMO route runtime payload retained forbidden teacher keys")
    return {
        "payload": payload,
        "payload_sha256": identity(payload),
        "training_audit": audit,
    }


def _weighted_order(classes: tuple[str, ...], probabilities: np.ndarray, rng) -> list[str]:
    if probabilities.shape != (len(classes),) or np.any(probabilities <= 0):
        raise ValueError("route transition probabilities must have full positive support")
    probabilities = probabilities / probabilities.sum()
    keys = -np.log(np.maximum(rng.random(len(classes)), np.finfo(float).tiny)) / probabilities
    return [classes[index] for index in np.argsort(keys)]


def _sample_legal_successor(graph, candidates, legal_checkpoint, rng):
    scores = np.asarray(
        [score_features(legal_checkpoint, action_features(graph, row)) for row in candidates],
        dtype=float,
    )
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("legal-action scorer returned no finite candidate")
    scaled = scores / ACTION_TEMPERATURE
    scaled -= scaled.max()
    learned = np.exp(scaled)
    learned /= learned.sum()
    probabilities = ACTION_EXPLORATION / len(scores) + (1 - ACTION_EXPLORATION) * learned
    index = int(rng.choice(len(candidates), p=probabilities))
    return candidates[index], float(probabilities[index]), {
        "candidates": len(candidates),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }


def sample_route_program(
    source,
    transition_checkpoint: dict[str, Any],
    legal_checkpoint: dict[str, Any],
    rng,
) -> dict[str, Any]:
    """Sample and exactly execute one route-prior program."""

    if transition_checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("invalid PMO route-transition checkpoint")
    if transition_checkpoint.get("maximum_primitives") != MAXIMUM_PRIMITIVES:
        raise ValueError("PMO route-transition primitive budget changed")
    graph = source
    source_key = canonical_state_key(source)
    states = [encode_state(source)]
    actions = []
    previous = START
    telemetry = {
        "rule_draws": 0,
        "empty_rule_fibers": 0,
        "legal_successors_enumerated": 0,
        "chosen_probabilities": [],
        "chosen_rule_sequence": [],
        "stopped": False,
    }
    began = perf_counter()
    for _ in range(MAXIMUM_PRIMITIVES):
        transition = transition_checkpoint["transitions"][previous]
        classes = tuple(map(str, transition["classes"]))
        probabilities = np.asarray(transition["probabilities"], dtype=float)
        ordered = _weighted_order(classes, probabilities, rng)
        advanced = False
        for token in ordered:
            telemetry["rule_draws"] += 1
            if token == STOP:
                if actions:
                    telemetry["stopped"] = True
                    advanced = True
                    break
                continue
            candidates = tuple(enumerate_rule_successors(graph, token))
            telemetry["legal_successors_enumerated"] += len(candidates)
            if not candidates:
                telemetry["empty_rule_fibers"] += 1
                continue
            selected, probability, fiber = _sample_legal_successor(
                graph, candidates, legal_checkpoint, rng
            )
            graph = selected.successor
            states.append(encode_state(graph))
            actions.append(selected.action_record)
            telemetry["chosen_probabilities"].append(probability)
            telemetry["chosen_rule_sequence"].append(token)
            telemetry.setdefault("fiber_summaries", []).append(fiber)
            previous = token
            advanced = True
            break
        if telemetry["stopped"] or not advanced:
            break
    if not actions:
        return {
            "status": "no_executable_program",
            "seconds": perf_counter() - began,
            "telemetry": telemetry,
        }
    program = dependency_region_program(
        states,
        actions,
        config=DependencyRegionConfig(
            runtime_maximum_primitives=MAXIMUM_PRIMITIVES,
            runtime_maximum_components=MAXIMUM_COMPONENTS,
        ),
    )
    endpoint = canonical_state_key(graph)
    if not program["exact_replay"]:
        raise RuntimeError("PMO route sampler failed exact replay")
    if endpoint == source_key:
        return {
            "status": "self_endpoint",
            "seconds": perf_counter() - began,
            "telemetry": telemetry,
        }
    return {
        "status": "complete",
        "endpoint": endpoint,
        "endpoint_state": states[-1],
        "primitive_count": len(actions),
        "component_count": program["component_count"],
        "created_dependency_edges": program["created_dependency_edges"],
        "cycle_dependency_edges": program["cycle_dependency_edges"],
        "exact_replay": True,
        "seconds": perf_counter() - began,
        "telemetry": telemetry,
    }


def sample_v0_program(source, rng, attempt_index: int) -> dict[str, Any]:
    began = perf_counter()
    try:
        if (
            source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD
            and attempt_index < CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS
        ):
            _, program, _, trace, metadata = synthesize_named_module_sequence(
                source,
                rng,
                ("substituent_delete",),
                max_primitives=32,
                max_blocks=8,
            )
        else:
            _, program, _, trace, metadata = synthesize_dynamic_program(
                source,
                rng,
                max_modules=3,
                max_primitives=32,
                max_blocks=8,
            )
    except ValueError as error:
        return {
            "status": "execution_rejected",
            "reason": str(error),
            "seconds": perf_counter() - began,
        }
    return {
        "status": "complete",
        "endpoint": trace["endpoint"],
        "primitive_count": len(program.marks),
        "component_count": len(metadata["modules"]),
        "exact_replay": True,
        "seconds": perf_counter() - began,
        "module_families": [row["family"] for row in metadata["modules"]],
    }


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row["status"] == "complete"]
    endpoints = [row["endpoint"] for row in complete]
    exact = [row for row in complete if row.get("exact_replay") is True]
    return {
        "attempts": len(rows),
        "complete": len(complete),
        "validity": len(complete) / len(rows) if rows else None,
        "unique_endpoints": len(set(endpoints)),
        "unique_yield": len(set(endpoints)) / len(rows) if rows else None,
        "exact_execution_precision": len(exact) / len(complete) if complete else None,
        "wall_seconds": sum(float(row["seconds"]) for row in rows),
        "primitive_count_distribution": dict(
            sorted(Counter(row["primitive_count"] for row in complete).items())
        ),
        "component_count_distribution": dict(
            sorted(Counter(row["component_count"] for row in complete).items())
        ),
        "failure_counts": dict(
            sorted(Counter(row["status"] for row in rows if row["status"] != "complete").items())
        ),
    }


def _route_source_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _arm_summary(rows)
    complete = [row for row in rows if row["status"] == "complete"]
    rule_counts: Counter[str] = Counter()
    for row in complete:
        rule_counts.update(row["telemetry"]["chosen_rule_sequence"])
    summary.update(
        route_programs_with_created_dependencies=sum(
            bool(row["created_dependency_edges"]) for row in complete
        ),
        route_programs_with_cycle_dependencies=sum(
            bool(row["cycle_dependency_edges"]) for row in complete
        ),
        stopped_before_horizon=sum(row["telemetry"]["stopped"] for row in complete),
        reached_primitive_horizon=sum(
            row["primitive_count"] == MAXIMUM_PRIMITIVES for row in complete
        ),
        legal_successors_enumerated=sum(
            row["telemetry"]["legal_successors_enumerated"] for row in complete
        ),
        empty_rule_fibers=sum(
            row["telemetry"]["empty_rule_fibers"] for row in complete
        ),
        chosen_rule_counts=dict(sorted(rule_counts.items())),
    )
    return summary


def run_matched_yield(
    initialization: dict[str, Any],
    transition_runtime: dict[str, Any],
    legal_runtime: dict[str, Any],
    *,
    progress=None,
) -> dict[str, Any]:
    candidates = initialization.get("candidates")
    if initialization.get("count") != 16 or not isinstance(candidates, list) or len(candidates) != 16:
        raise ValueError("PMO production gate requires the immutable 16-source bank")
    transitions = {
        int(row["fold_index"]): row for row in transition_runtime.get("folds", ())
    }
    legal = _fold_checkpoints(legal_runtime)
    if sorted(transitions) != [0, 1, 2]:
        raise ValueError("PMO route-transition runtime lacks a fold")
    report = progress or (lambda row: None)
    tasks = {}
    for task_index, (task, fold) in enumerate(TASK_FOLDS.items()):
        arm_rows = {"old_v0": [], "additive_route": []}
        source_rows = []
        for source_index, source_row in enumerate(candidates):
            source = decode_state(source_row["state"])
            shared_seed = np.random.SeedSequence([SEED, task_index, source_index, 0])
            shared_baseline = sample_v0_program(
                source, np.random.default_rng(shared_seed), 0
            )
            baseline_second = sample_v0_program(
                source,
                np.random.default_rng(
                    np.random.SeedSequence([SEED, task_index, source_index, 1])
                ),
                1,
            )
            route = sample_route_program(
                source,
                transitions[fold],
                legal[fold],
                np.random.default_rng(
                    np.random.SeedSequence([SEED, task_index, source_index, 2])
                ),
            )
            arm_rows["old_v0"].extend((shared_baseline, baseline_second))
            arm_rows["additive_route"].extend((shared_baseline, route))
            source_result = {
                "source_index": source_index,
                "source_identity": identity(source_row["state"]),
                "shared_v0": shared_baseline,
                "baseline_second_v0": baseline_second,
                "additive_route_draw": route,
            }
            source_rows.append(source_result)
            report(
                {
                    "task": task,
                    "source": source_index + 1,
                    "sources": len(candidates),
                    "route_status": route["status"],
                    "route_seconds": route["seconds"],
                }
            )
        tasks[task] = {
            "held_fold": fold,
            "held_task_family": TASK_FAMILIES[task],
            "sources": source_rows,
            "arms": {name: _arm_summary(rows) for name, rows in arm_rows.items()},
            "proposal_sources": {
                "shared_first_v0": _arm_summary(
                    [row["shared_v0"] for row in source_rows]
                ),
                "baseline_second_v0": _arm_summary(
                    [row["baseline_second_v0"] for row in source_rows]
                ),
                "route_transition_prior": _route_source_summary(
                    [row["additive_route_draw"] for row in source_rows]
                ),
            },
        }
    return tasks


def contract_envelope() -> dict[str, Any]:
    payload = {
        "schema_version": CONTRACT_SCHEMA,
        "authorization": (
            "2026-09-18 user authorized a zero-oracle split-clean PMO production "
            "route proposer and matched actual-sampler yield comparison"
        ),
        "tasks": TASK_FOLDS,
        "held_task_families": TASK_FAMILIES,
        "inputs": EXPECTED_SHA256,
        "arms": {
            "old_v0": ["v0", "v0"],
            "additive_route": ["same_first_v0", "route_transition_prior"],
        },
        "attempts_per_source_arm": ATTEMPTS_PER_SOURCE_ARM,
        "sources": 16,
        "route_maximum_primitives": MAXIMUM_PRIMITIVES,
        "route_maximum_components": MAXIMUM_COMPONENTS,
        "transition_smoothing": TRANSITION_SMOOTHING,
        "action_exploration": ACTION_EXPLORATION,
        "same_first_v0_draw": True,
        "oracle_calls_authorized": 0,
        "scored_launch_authorized": False,
        "outputs": {"checkpoint": CHECKPOINT, "result": RESULT},
    }
    return {"payload": payload, "payload_sha256": identity(payload)}


def build_production_yield(root: Path, *, progress=None) -> tuple[dict, dict]:
    for relative, expected in EXPECTED_SHA256.items():
        verify_file(root / relative, expected)
    corpus = _load_envelope(root / CORPUS, compressed=True)
    legal = _load_envelope(root / LEGAL_RUNTIME, compressed=True)
    dependency = _load_envelope(root / DEPENDENCY_RUNTIME, compressed=True)
    if dependency.get("new_oracle_calls") != 0 or legal.get("new_oracle_calls") != 0:
        raise ValueError("PMO fold checkpoint reports oracle calls")
    checkpoint = fit_route_transition_checkpoint(corpus)
    initialization = json.loads((root / INITIALIZATION).read_text())
    body = {key: value for key, value in initialization.items() if key != "lock_sha256"}
    if identity(body) != initialization.get("lock_sha256"):
        raise ValueError("PMO initialization lock changed")
    tasks = run_matched_yield(
        initialization, checkpoint["payload"], legal, progress=progress
    )
    contract = contract_envelope()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "schema_version": SCHEMA,
        "decision": "ZERO_ORACLE_MATCHED_PRODUCTION_YIELD_COMPLETE",
        "new_oracle_calls": 0,
        "scored_launch_authorized": False,
        "contract": contract["payload"],
        "contract_payload_sha256": contract["payload_sha256"],
        "inputs": EXPECTED_SHA256,
        "code_revision": revision,
        "tasks": tasks,
        "training_audit": checkpoint.pop("training_audit"),
        "checkpoint_payload_sha256": checkpoint["payload_sha256"],
        "claim_boundary": (
            "actual zero-oracle executable proposal yield only; no PMO reward, "
            "optimization advantage or IVG comparison is measured"
        ),
    }
    return checkpoint, {"payload": payload, "payload_sha256": identity(payload)}


__all__ = [
    "CHECKPOINT",
    "CONTRACT",
    "RESULT",
    "build_production_yield",
    "contract_envelope",
    "fit_route_transition_checkpoint",
    "run_matched_yield",
    "sample_route_program",
    "sample_v0_program",
]
