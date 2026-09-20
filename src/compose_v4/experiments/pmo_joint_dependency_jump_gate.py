"""Zero-oracle production gate for joint PMO jumps plus recursive refinement."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis import (
    CAPACITY_AWARE_THRESHOLD,
    CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS,
    synthesize_dynamic_program,
    synthesize_named_module_sequence,
)
from compose_v4.control.pmo_joint_dependency_jump import (
    FORBIDDEN_CHECKPOINT_KEYS,
    bind_joint_plan,
    fit_joint_checkpoint,
)
from compose_v4.experiments.pmo_complete_route_dynamic_gate import (
    SELECTED_TASKS,
    _member_tasks,
)
from compose_v4.experiments.pmo_dependency_region_program import (
    DependencyRegionConfig,
    dependency_region_program,
)
from compose_v4.experiments.route_proposal_quality import transformation_equivalent
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "pmo_joint_dependency_jump_gate_v2"
CONTRACT_SCHEMA = "pmo_joint_dependency_jump_gate_contract_v2"


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(map(str, value)) | set().union(
            *(_recursive_keys(child) for child in value.values()), set()
        )
    if isinstance(value, list):
        return set().union(*(_recursive_keys(child) for child in value), set())
    return set()


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError("PMO joint-jump contract schema changed")
    expected = str(contract.get("contract_sha256", ""))
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if identity(payload) != expected:
        raise ValueError("PMO joint-jump contract self-hash changed")
    if contract.get("oracle_calls_authorized") != 0:
        raise ValueError("zero-oracle PMO joint-jump contract changed")
    if contract.get("modal_launch_authorized") is not False:
        raise ValueError("PMO joint-jump contract unexpectedly authorizes Modal")


def selected_source_state(corpus: dict[str, Any]) -> dict[str, Any]:
    """Return the unique root shared by all three prospectively selected tasks."""

    roots_by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for route in corpus["routes"]:
        source_state = route["states"][0]
        source_key = canonical_state_key(decode_state(source_state))
        for task in SELECTED_TASKS:
            if task in _member_tasks(route):
                roots_by_task[task].setdefault(source_key, source_state)
    common = set.intersection(*(set(roots_by_task[task]) for task in SELECTED_TASKS))
    if len(common) != 1:
        raise ValueError(f"expected one shared selected-task PMO root, found {len(common)}")
    return roots_by_task[SELECTED_TASKS[0]][next(iter(common))]


def bind_plan_receipt(arguments) -> dict[str, Any]:
    """Bind one independently resumable arm/plan work unit."""

    arm, source_state, plan, beam_width, contract_sha256 = arguments
    candidates = bind_joint_plan(decode_state(source_state), plan, beam_width=beam_width)
    return {
        "schema_version": "pmo_joint_dependency_jump_plan_receipt_v1",
        "contract_sha256": contract_sha256,
        "arm": arm,
        "source_id": identity(source_state),
        "plan_id": plan["plan_id"],
        "plan_mass": float(plan["mass"]),
        "primitive_count": plan["primitive_count"],
        "bound_candidates": candidates,
        "new_oracle_calls": 0,
    }


def generate_arm_lock(
    source_state: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    arm: str,
    candidate_budget: int,
    beam_width: int,
    workers: int,
) -> dict[str, Any]:
    """Reference in-memory implementation; production uses durable receipts."""

    del workers
    rows = [
        bind_plan_receipt((arm, source_state, plan, beam_width, "in_memory_test"))
        for plan in checkpoint["plan_latents"]
    ]
    return reduce_arm_receipts(
        source_state,
        checkpoint,
        rows,
        arm=arm,
        candidate_budget=candidate_budget,
    )


def reduce_arm_receipts(
    source_state: dict[str, Any],
    checkpoint: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    arm: str,
    candidate_budget: int,
) -> dict[str, Any]:
    """Deterministically reduce the complete checkpoint plan support."""

    expected = {plan["plan_id"] for plan in checkpoint["plan_latents"]}
    observed = {row["plan_id"] for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError(
            f"{arm} receipt set incomplete: missing={len(expected - observed)}, "
            f"unexpected={len(observed - expected)}"
        )
    by_endpoint: dict[str, dict[str, Any]] = {}
    for row in rows:
        for candidate in row["bound_candidates"]:
            current = by_endpoint.get(candidate["endpoint_key"])
            if current is None or (-float(row["plan_mass"]), candidate["plan_id"]) < (
                -float(current["plan_mass"]),
                current["plan_id"],
            ):
                by_endpoint[candidate["endpoint_key"]] = {
                    **candidate,
                    "plan_mass": float(row["plan_mass"]),
                }
    ordered = sorted(
        by_endpoint.values(),
        key=lambda row: (
            -int(row["primitive_count"] >= 8),
            -row["plan_mass"],
            row["plan_id"],
            row["endpoint_key"],
        ),
    )[:candidate_budget]
    return {
        "arm": arm,
        "source_id": identity(source_state),
        "checkpoint_fit_scope": checkpoint["fit_scope"],
        "checkpoint_held_fold": checkpoint["held_fold"],
        "plan_latents_available": len(checkpoint["plan_latents"]),
        "plan_latents_attempted": len(rows),
        "all_checkpoint_plan_latents_attempted": True,
        "plans_with_bound_candidates": sum(bool(row["bound_candidates"]) for row in rows),
        "bound_before_endpoint_deduplication": sum(len(row["bound_candidates"]) for row in rows),
        "candidates": ordered,
    }


def generate_candidate_lock(
    corpus: dict[str, Any], contract: dict[str, Any], *, workers: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Fit sanitized checkpoints and lock source-bound programs score-blind."""

    runtime = contract["runtime"]
    shared = fit_joint_checkpoint(corpus, held_fold=None)["payload"]
    checkpoints = {"shared_all_routes": shared}
    for selected in contract["selected_tasks"]:
        fold = int(selected["fold"])
        checkpoints[f"held_fold_clean_{fold}"] = fit_joint_checkpoint(corpus, held_fold=fold)[
            "payload"
        ]
    for checkpoint in checkpoints.values():
        forbidden = FORBIDDEN_CHECKPOINT_KEYS & _recursive_keys(checkpoint)
        if forbidden:
            raise RuntimeError(f"checkpoint retained forbidden keys: {sorted(forbidden)}")
    source_state = selected_source_state(corpus)
    arms = {
        name: generate_arm_lock(
            source_state,
            checkpoint,
            arm=name,
            candidate_budget=int(runtime["candidate_budget_per_source_arm"]),
            beam_width=int(runtime["binding_beam_width"]),
            workers=workers,
        )
        for name, checkpoint in checkpoints.items()
    }
    lock = {
        "schema_version": "pmo_joint_dependency_jump_candidate_lock_v1",
        "contract_sha256": contract["contract_sha256"],
        "source_id": identity(source_state),
        "source_state": source_state,
        "scores_present": False,
        "teacher_endpoints_loaded_during_generation": False,
        "all_checkpoint_plan_latents_attempted": True,
        "arms": arms,
        "new_oracle_calls": 0,
    }
    return lock, checkpoints


def evaluate_locked_support(
    lock: dict[str, Any], corpus: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    """Load answer-known targets only after the autonomous candidate lock exists."""

    source = decode_state(lock["source_state"])
    source_key = canonical_state_key(source)
    targets_by_task: dict[str, dict[str, Any]] = defaultdict(dict)
    route_lengths: dict[str, list[int]] = defaultdict(list)
    for route in corpus["routes"]:
        if canonical_state_key(decode_state(route["states"][0])) != source_key:
            continue
        for task in SELECTED_TASKS:
            if task in _member_tasks(route):
                target = decode_state(route["states"][-1])
                targets_by_task[task][canonical_state_key(target)] = target
                route_lengths[task].append(len(route["actions"]))
    output = {}
    for task in SELECTED_TASKS:
        fold = next(int(row["fold"]) for row in contract["selected_tasks"] if row["task"] == task)
        task_arms = {}
        for arm in ("shared_all_routes", f"held_fold_clean_{fold}"):
            candidates = lock["arms"][arm]["candidates"]
            exact = 0
            equivalent = 0
            for candidate in candidates:
                endpoint = decode_state(candidate["endpoint_state"])
                if candidate["endpoint_key"] in targets_by_task[task]:
                    exact += 1
                elif any(
                    transformation_equivalent(source, endpoint, target)
                    for target in targets_by_task[task].values()
                ):
                    equivalent += 1
            task_arms[arm] = {
                "locked_candidates": len(candidates),
                "route_scale_candidates": sum(row["primitive_count"] >= 8 for row in candidates),
                "exact_endpoint_support": exact,
                "transformation_equivalent_support": equivalent,
            }
        output[task] = {
            "held_fold": fold,
            "teacher_route_lengths": sorted(route_lengths[task]),
            "complete_single_program_teacher_support": sum(
                length <= 32 for length in route_lengths[task]
            ),
            "arms": task_arms,
        }
    return output


def sample_recursive_refinements(lock: dict[str, Any], *, seed: int) -> list[dict[str, Any]]:
    """Use locked route-scale endpoints as real parents for generic v0 tails."""

    parents = [
        row
        for row in lock["arms"]["shared_all_routes"]["candidates"]
        if row["primitive_count"] >= 8
    ]
    children = []
    for parent_index, parent in enumerate(parents):
        source = decode_state(parent["endpoint_state"])
        rng = np.random.default_rng([seed, parent_index])
        try:
            if (
                source.n_real_atoms >= CAPACITY_AWARE_THRESHOLD
                and parent_index < CAPACITY_BOOTSTRAP_SINGLE_ATTEMPTS
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
        except (ValueError, RuntimeError, KeyError, IndexError, TypeError) as error:
            children.append(
                {
                    "parent_endpoint": parent["endpoint_key"],
                    "status": "rejected",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        region = dependency_region_program(
            trace["states"],
            trace["actions"],
            config=DependencyRegionConfig(
                runtime_maximum_primitives=32,
                runtime_maximum_components=8,
            ),
        )
        if not region["exact_replay"]:
            raise RuntimeError("generic refinement tail lost exact replay")
        children.append(
            {
                "parent_endpoint": parent["endpoint_key"],
                "parent_generation": 1,
                "parent_primitive_count": parent["primitive_count"],
                "child_generation": 2,
                "child_endpoint": trace["endpoint"],
                "child_endpoint_state": trace["states"][-1],
                "child_primitive_count": len(trace["actions"]),
                "child_component_count": region["component_count"],
                "program_id": program.program_id,
                "module_families": [row["family"] for row in metadata["modules"]],
                "status": "complete",
                "exact_replay": True,
            }
        )
    return children


__all__ = [
    "CONTRACT_SCHEMA",
    "SCHEMA",
    "bind_plan_receipt",
    "evaluate_locked_support",
    "generate_arm_lock",
    "generate_candidate_lock",
    "reduce_arm_receipts",
    "sample_recursive_refinements",
    "selected_source_state",
    "validate_contract",
]
