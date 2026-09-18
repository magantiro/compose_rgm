"""Frozen, zero-oracle contract for the PMO route-prior/FiberControl pilot.

This module prepares and verifies a prospective contract. It deliberately has
no oracle adapter and no launch entry point. A later implementation and sealed
preflight must satisfy this contract before separately authorized scoring.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from compose_v4.control.docking_value import identity

SCHEMA = "pmo_route_fiber_scored_pilot_contract_v1"
CONTRACT = "configs/pmo_route_fiber_scored_pilot_v1.json"

TASKS = {
    "gsk3b": {"fold": 0, "held_task_family": "bioactivity"},
    "perindopril_mpo": {"fold": 2, "held_task_family": "mpo"},
}

ARMS = {
    "old_v0_blind": {"proposal_source": "old_v0", "selection": "blind"},
    "additive_route_blind": {
        "proposal_source": "additive_route",
        "selection": "blind",
    },
    "old_v0_fiber_control": {
        "proposal_source": "old_v0",
        "selection": "fiber_control",
    },
    "additive_route_fiber_control": {
        "proposal_source": "additive_route",
        "selection": "fiber_control",
    },
}

INPUTS = {
    "configs/pmo_dynamic_v21_development_v1.json": (
        "18665d2753eccce200a390f7ba458d7523361506041d4814cb6970022a9fc71d"
    ),
    "configs/pmo_route_fiber_production_yield_v3.json": (
        "efffc3a6daef79dab44314dacc603128e6d800133b0b62b4dc0c5c1cf396641f"
    ),
    "diagnostics/parent_edit_cycles/prepared/init_20260921.json": (
        "a1f7840df1f75567565de633c4df3d013c3165df3e43cbf2ac87b156d5f206ff"
    ),
    "diagnostics/pmo_route_fiber_pilot/production_yield_v3.json": (
        "9637a30db55016cd8952f6713850ab0fc8a24b3b34ac3b1c881f4ad9a0a2d1b1"
    ),
    "diagnostics/pmo_route_fiber_pilot/route_transition_checkpoint_v3.json": (
        "8e9464b2b0e1469b4372029391f1a2328a180b955b0d9512390e603ae8ac966d"
    ),
    "src/compose_v4/control/fiber_control.py": (
        "8aa214ec25a12ea0b5806b64d22f92edba4617514886a2f2d3023322788e01db"
    ),
    "src/compose_v4/experiments/pmo_route_fiber_production_yield.py": (
        "6889b794615cc137b594e51b88b23c9d7c3ef81ee870886462367c5ab8da9a52"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_inputs(root: Path) -> None:
    for relative, expected in INPUTS.items():
        observed = sha256_file(root / relative)
        if observed != expected:
            raise ValueError(
                f"PMO scored-pilot input changed: {relative}: "
                f"expected {expected}, observed {observed}"
            )


def payload() -> dict[str, Any]:
    """Return the deterministic prospective design, with zero launch authority."""
    return {
        "schema_version": SCHEMA,
        "status": "PREPARED_NOT_AUTHORIZED_NOT_LAUNCHABLE",
        "authorization": (
            "2026-09-18 authorization covers preparation of this self-hashed "
            "contract only; it authorizes zero oracle calls and no scored launch"
        ),
        "scientific_question": (
            "Do split-clean route-derived executable proposals and within-task "
            "online FiberControl provide complementary PMO reward gains over "
            "unchanged Dynamic-v0 proposals and reward-blind selection?"
        ),
        "claim_boundary": (
            "A two-task warm development factorial over fixed root-conditioned "
            "candidate pools; it is not a full PMO benchmark, held-out PMO claim, "
            "iterative descendant-refinement result or IVG comparison"
        ),
        "tasks": TASKS,
        "arms": ARMS,
        "budget": {
            "charged_calls_per_task_arm": 48,
            "initialization_calls_per_task_arm": 16,
            "post_initialization_rounds": 4,
            "calls_per_round": 8,
            "task_arm_units": 8,
            "total_charged_call_ceiling": 384,
            "automatic_retries": 0,
            "failed_or_unresolved_reservation_is_charged": True,
            "replacement_or_backfill_after_scoring": False,
        },
        "initialization": {
            "path": "diagnostics/parent_edit_cycles/prepared/init_20260921.json",
            "count": 16,
            "task_independent": True,
            "scores_absent_from_lock": True,
            "each_score_is_charged_inside_each_arm": True,
            "same_order_in_every_arm": True,
            "route_archive_initially_empty": True,
        },
        "split_discipline": {
            "policy": "leave_one_whole_task_family_and_every_shared_lineage_out",
            "runtime_fold_bundle": (
                "extract only the selected sanitized numeric fold checkpoint; "
                "the worker may not load other fold models or exact training traces"
            ),
            "task_identity_used_only_for": [
                "native PMO oracle selection",
                "selection of the preregistered held-family fold",
            ],
            "forbidden_route_runtime_fields": [
                "task",
                "task_family",
                "route_id",
                "lineage_id",
                "source_graph",
                "source_smiles",
                "endpoint",
                "endpoint_smiles",
                "absolute_atom_address",
                "assignment",
                "teacher_action",
                "executable_teacher_trace",
            ],
            "runtime_route_or_endpoint_lookup": False,
        },
        "proposal_factor": {
            "parents": "the same 16 immutable initialization molecules in every round",
            "round_candidate_pools_are_score_blind": True,
            "all_candidate_pools_locked_before_first_oracle_call": True,
            "attempts_per_parent_round": 2,
            "old_v0": ["unchanged_v0_draw_0", "unchanged_v0_draw_1"],
            "additive_route": [
                "the_same_unchanged_v0_draw_0",
                "fold_specific_route_transition_draw",
            ],
            "route_fraction": 0.5,
            "route_maximum_primitives": 8,
            "route_maximum_components": 8,
            "route_action_exploration": 0.1,
            "cross_round_deduplication": (
                "canonical endpoint against initialization and every earlier lock "
                "within that task and proposal source"
            ),
            "candidate_shortfall_rule": (
                "if any locked round contains fewer than eight unique valid exact-"
                "replay endpoints, fail prelaunch before every oracle call"
            ),
            "scope_limit": (
                "proposal pools do not expand descendants; a positive result promotes "
                "the method to a separately contracted Dynamic archive integration"
            ),
        },
        "candidate_locks": {
            "pool_pairs": {
                "old_v0": ["old_v0_blind", "old_v0_fiber_control"],
                "additive_route": [
                    "additive_route_blind",
                    "additive_route_fiber_control",
                ],
            },
            "pool_path_template": (
                "diagnostics/pmo_route_fiber_scored_pilot_v1/locks/"
                "{task}/{proposal_source}/round_{round:02d}/pool.json"
            ),
            "query_path_template": (
                "diagnostics/pmo_route_fiber_scored_pilot_v1/locks/"
                "{task}/{arm}/round_{round:02d}/query.json"
            ),
            "pool_fields": [
                "candidate_id",
                "parent_id",
                "proposal_source",
                "program_identity",
                "canonical_endpoint",
                "endpoint_state",
                "exact_replay",
                "validity",
                "primitive_count",
                "dependency_region_count",
                "created_handle_dependencies",
                "cycle_dependencies",
                "generic_rule_histogram",
                "selection_features",
                "structural_fingerprint",
            ],
            "forbidden_pool_fields": [
                "reward",
                "control_score",
                "winner_identity",
                "comparator_outcome",
            ],
            "rules": [
                "pool lock is self-hashed and immutable before selection",
                "query lock is self-hashed and immutable before reservation",
                "every query candidate must resolve one-to-one into its pool lock",
                "no regeneration, replacement or backfill after any score is observed",
            ],
        },
        "selection_factor": {
            "reward_coordinate": "control_score=-native_PMO_reward",
            "blind": "uniform_without_replacement_from_the_locked_pool",
            "round_1_pairing": (
                "blind and FiberControl arms sharing a proposal source select the "
                "same eight candidate IDs because no transformation reward labels "
                "exist before round 1"
            ),
            "fiber_control": {
                "value_model": "ProgramValue(penalty=1.0)",
                "acquisition": {
                    "beta": 1.0,
                    "batch": 8,
                    "diversity": 0.5,
                    "exploration_quota": 2,
                },
                "training_data": (
                    "only completed candidate transformations and native rewards from "
                    "the same task and arm in earlier rounds"
                ),
                "target": "child_reward-parent_reward",
                "no_outcome_sharing_across_tasks_or_arms": True,
            },
            "feature_schema": [
                "negative_parent_reward",
                "incumbent_reward_minus_parent_reward",
                "route_lane_indicator",
                "primitive_count_over_8",
                "dependency_region_count_over_8",
                "created_atom_count_over_6",
                "deleted_atom_count_over_6",
                "created_dependency_count_over_4",
                "cycle_dependency_count_over_4",
                "eight_generic_executor_rule_counts_over_8",
                "constant_intercept",
            ],
            "task_identity_is_not_a_feature": True,
        },
        "oracle": {
            "adapter": "native PyTDC Oracle(name=task)",
            "direction": "maximize",
            "range": [0.0, 1.0],
            "prescreen": False,
            "environment": {
                "python": "3.11",
                "rdkit": "2023.09.6",
                "pytdc": "1.1.15",
                "numpy": "1.26.4",
                "scikit_learn": "1.2.2",
                "pandas": "2.1.4",
                "scipy": "1.15.0",
                "seaborn": "0.13.2",
                "requests": "2.32.4",
                "setuptools": "75.6.0",
            },
            "gsk3b_asset": {
                "path": (
                    "diagnostics/pmo_ivg_oracle_parity/ivg_oracle_assets/"
                    "oracle/gsk3b_current.pkl"
                ),
                "bytes": 27791877,
                "sha256": (
                    "d3a20701b80e5179c88c3ad4dc3483dd7ab35c50dc055c6773a7f5b63e89b6d5"
                ),
            },
        },
        "metrics": {
            "primary": (
                "native higher-is-better PMO top-ten trapezoid AUC through call 48, "
                "using pmo_top_ten_auc(frequency=1, finish=True)"
            ),
            "secondary": [
                "best native reward through call 48",
                "top-ten mean at calls 16, 24, 32, 40 and 48",
                "best reward at calls 16, 24, 32, 40 and 48",
                "parent improvements by proposal source",
                "route-derived children improving their parent",
                "proposal, exact-execution, oracle and wall time",
                "validity, uniqueness, failures and candidate shortfalls",
            ],
            "postrun_only_comparators": True,
        },
        "promotion": {
            "all_required": [
                "additive_route_fiber_control exceeds old_v0_blind on AUC48 and best reward on both tasks",
                "additive_route_fiber_control exceeds max(additive_route_blind, old_v0_fiber_control) by at least 0.02 AUC48 on at least one task",
                "additive_route_fiber_control is no worse than that max by more than 0.01 AUC48 on the other task",
                "at least one route-derived child improves its own parent on each task",
            ],
            "meaning": (
                "promote to a separately contracted Dynamic archive/refinement "
                "qualification; do not claim full-PMO or IVG superiority"
            ),
        },
        "kill": {
            "scientific": (
                "on both tasks, additive_route_fiber_control fails to exceed "
                "old_v0_fiber_control on AUC48 and additive_route_blind fails to "
                "exceed old_v0_blind on AUC48"
            ),
            "integrity": [
                "any task-family or shared-lineage leakage",
                "any candidate/query lock mutation or unbound oracle receipt",
                "any exact-execution mismatch",
                "any accounting overflow, retry or fabricated score",
                "any candidate-pool shortfall before scoring",
            ],
            "action": (
                "retain the negative result and stop this route-prior/FiberControl "
                "revision; do not widen the lock or add calls post hoc"
            ),
        },
        "logging": {
            "heartbeat_seconds": 30,
            "status_fields": [
                "task",
                "arm",
                "phase",
                "round",
                "charged_calls",
                "remaining_calls",
                "candidate_pool_size",
                "candidate_shortfall",
                "current_best_reward",
                "current_top10_mean",
                "proposal_seconds",
                "oracle_seconds",
                "wall_seconds",
                "estimated_remaining_seconds",
            ],
            "durable_records": [
                "complete candidate pools and attempt statuses",
                "query locks and reservation/result receipts",
                "FiberControl features, predictions, selected indices and fitted state",
                "blind-selection RNG state",
                "best/top-ten curves and AUC inputs",
                "resource and failure accounting",
            ],
        },
        "checkpoint_resume": {
            "checkpoint_after": [
                "initialization",
                "each immutable candidate-pool lock",
                "each immutable query lock",
                "each completed round",
            ],
            "snapshot_fields": [
                "contract and code identities",
                "task, arm, phase and round",
                "ledger and lock identities",
                "SearchState archive, budget, rounds and history",
                "ProgramValue weights, means, scales and observation count",
                "all RNG states",
                "used candidate IDs",
                "metrics and timing accumulators",
            ],
            "resume_policy": (
                "validate every identity and continue from the latest complete phase; "
                "a completed unit is idempotent, while an unresolved charged "
                "reservation blocks automatic continuation"
            ),
            "resume_may_not": [
                "change data, folds, candidates, seeds, features or controller policy",
                "retry an unresolved request",
                "regenerate or backfill a lock",
                "share outcomes across arms or tasks",
            ],
        },
        "prelaunch_gates": [
            "all input and payload hashes verify",
            "the v3 decision remains ZERO_ORACLE_PRODUCTION_SUPPORT_PASSED_YIELD_TIED",
            "fold 0 excludes bioactivity and fold 2 excludes MPO with zero shared-lineage overlap",
            "the pinned oracle environment and GSK3B asset verify without evaluation",
            "pinned-environment v3 decision-equivalence passes on both pilot tasks",
            "all score-blind candidate pools are complete, self-hashed and contain at least eight unique exact endpoints",
            "candidate locks contain no score, task answer, route identity or teacher trace",
            "synthetic query-ledger and interrupted-resume tests pass",
            "implementation, contract, preflight and candidate locks are committed and the relevant worktree paths are clean",
            "an exact scored authorization matching this contract has been recorded",
        ],
        "required_authorization_text": (
            "I authorize the scored PMO route-prior x FiberControl pilot exactly "
            "under configs/pmo_route_fiber_scored_pilot_v1.json, with at most 384 "
            "charged calls, zero retries, and no task or arm outcome sharing."
        ),
        "required_agents_amendment": (
            "Authorize only the self-hashed PMO route-prior x FiberControl v1 pilot: "
            "GSK3B fold 0 and Perindopril MPO fold 2; four frozen arms; 48 charged "
            "calls per task/arm including 16 initialization calls; 384 total; zero "
            "retry; immutable score-blind pool locks; exact receipts and resume; no "
            "T4 change, broader PMO run or post-score candidate replacement."
        ),
        "inputs": INPUTS,
        "planned_outputs": {
            "preflight": "diagnostics/pmo_route_fiber_scored_pilot_v1/preflight.json",
            "candidate_lock_manifest": (
                "diagnostics/pmo_route_fiber_scored_pilot_v1/candidate_locks.json"
            ),
            "progress": "diagnostics/pmo_route_fiber_scored_pilot_v1/progress.json",
            "result": "diagnostics/pmo_route_fiber_scored_pilot_v1/result.json",
        },
        "oracle_calls_authorized": 0,
        "scored_launch_authorized": False,
        "modal_launch_authorized": False,
        "prohibitions": [
            "no oracle or docking call under this preparation contract",
            "no scored or Modal launch",
            "no change to T4 or unrelated dirty files",
            "no comparator available to runtime",
            "no candidate inspection followed by pool regeneration",
            "no extra task, arm, round, call, retry or backfill",
        ],
    }


def envelope(root: Path | None = None) -> dict[str, Any]:
    if root is not None:
        _assert_inputs(root)
    body = payload()
    return {"payload": body, "payload_sha256": identity(body)}


def verify_contract(root: Path, path: Path) -> dict[str, Any]:
    _assert_inputs(root)
    observed = json.loads(path.read_text())
    expected = envelope()
    if observed != expected:
        raise ValueError(
            f"PMO scored-pilot contract differs from frozen policy: {path}"
        )
    return observed
