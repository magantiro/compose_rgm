#!/usr/bin/env python3
"""Materialize the resumable zero-oracle PMO joint dependency-jump gate."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.control.docking_value import identity
from compose_v4.control.pmo_joint_dependency_jump import fit_joint_checkpoint
from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.pmo_complete_route_dynamic_gate import load_envelope
from compose_v4.experiments.pmo_joint_dependency_jump_gate import (
    SCHEMA,
    bind_plan_receipt,
    evaluate_locked_support,
    reduce_arm_receipts,
    sample_recursive_refinements,
    selected_source_state,
    validate_contract,
)

CONTRACT = "configs/pmo_joint_dependency_jump_gate_v2.json"
CORPUS = (
    "diagnostics/pmo_dependency_region_program_v2/attempt_1/"
    "training_dependency_region_corpus.json.gz"
)


def _write_sealed(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = json.loads(json.dumps(payload, sort_keys=True))
    envelope = {"payload": normalized, "payload_sha256": identity(normalized)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _checkpoint_set(corpus: dict, contract: dict) -> dict[str, dict]:
    checkpoints = {"shared_all_routes": fit_joint_checkpoint(corpus, held_fold=None)["payload"]}
    for selected in contract["selected_tasks"]:
        fold = int(selected["fold"])
        checkpoints[f"held_fold_clean_{fold}"] = fit_joint_checkpoint(corpus, held_fold=fold)[
            "payload"
        ]
    return checkpoints


def _receipt_path(root: Path, arm: str, plan_id: str) -> Path:
    return (
        root
        / "diagnostics/pmo_joint_dependency_jump_gate_v1/attempt_2/receipts"
        / arm
        / f"{plan_id}.json"
    )


def _validate_receipt(
    receipt: dict, *, arm: str, source_id: str, plan_id: str, contract_sha256: str
) -> None:
    expected = {
        "schema_version": "pmo_joint_dependency_jump_plan_receipt_v1",
        "contract_sha256": contract_sha256,
        "arm": arm,
        "source_id": source_id,
        "plan_id": plan_id,
        "new_oracle_calls": 0,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"receipt {arm}/{plan_id} changed field {key}")


def _materialize_receipts(
    root: Path,
    contract: dict,
    source_state: dict,
    checkpoints: dict[str, dict],
    *,
    workers: int,
) -> dict[str, list[dict]]:
    source_id = identity(source_state)
    rows: dict[str, list[dict]] = {arm: [] for arm in checkpoints}
    missing = []
    for arm, checkpoint in checkpoints.items():
        for plan in checkpoint["plan_latents"]:
            path = _receipt_path(root, arm, plan["plan_id"])
            if path.exists():
                receipt = load_envelope(path)
                _validate_receipt(
                    receipt,
                    arm=arm,
                    source_id=source_id,
                    plan_id=plan["plan_id"],
                    contract_sha256=contract["contract_sha256"],
                )
                rows[arm].append(receipt)
            else:
                missing.append(
                    (
                        arm,
                        source_state,
                        plan,
                        int(contract["runtime"]["binding_beam_width"]),
                        contract["contract_sha256"],
                    )
                )
    if missing:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_job = {pool.submit(bind_plan_receipt, job): job for job in missing}
            for completed, future in enumerate(as_completed(future_to_job), start=1):
                job = future_to_job[future]
                arm, _, plan, _, _ = job
                receipt = future.result()
                _validate_receipt(
                    receipt,
                    arm=arm,
                    source_id=source_id,
                    plan_id=plan["plan_id"],
                    contract_sha256=contract["contract_sha256"],
                )
                _write_sealed(_receipt_path(root, arm, plan["plan_id"]), receipt)
                rows[arm].append(receipt)
                if completed % 8 == 0 or completed == len(missing):
                    print(
                        json.dumps(
                            {
                                "phase": "plan_receipts",
                                "completed_this_run": completed,
                                "remaining_this_run": len(missing) - completed,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    for arm_rows in rows.values():
        arm_rows.sort(key=lambda row: row["plan_id"])
    return rows


def _shortfall_result(contract: dict, shortfalls: dict[str, int]) -> dict:
    return {
        "schema_version": SCHEMA,
        "decision": "NO_SCORED_LAUNCH_CANDIDATE_SHORTFALL",
        "contract_sha256": contract["contract_sha256"],
        "candidate_shortfalls": shortfalls,
        "teacher_endpoints_loaded": False,
        "new_oracle_calls": 0,
        "modal_launches": 0,
        "scored_launch_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = args.root.resolve()
    contract = json.loads((root / CONTRACT).read_text())
    validate_contract(contract)
    for relative, expected in contract["generation_inputs"].items():
        verify_file(root / relative, expected)
    for relative, expected in contract["implementation_sha256"].items():
        verify_file(root / relative, expected)

    corpus = load_envelope(root / CORPUS)
    checkpoints = _checkpoint_set(corpus, contract)
    source_state = selected_source_state(corpus)
    if identity(source_state) != contract["expected_source_id"]:
        raise ValueError("PMO attempt-2 source identity changed")
    if corpus["split"]["split_identity"] != contract["expected_split_identity"]:
        raise ValueError("PMO attempt-2 split identity changed")
    del corpus
    checkpoint_path = root / contract["outputs"]["checkpoint"]
    lock_path = root / contract["outputs"]["candidate_lock"]
    result_path = root / contract["outputs"]["result"]
    _write_sealed(
        checkpoint_path,
        {
            "schema_version": "pmo_joint_dependency_jump_checkpoints_v2",
            "contract_sha256": contract["contract_sha256"],
            "checkpoints": checkpoints,
            "new_oracle_calls": 0,
        },
    )
    receipts = _materialize_receipts(
        root, contract, source_state, checkpoints, workers=args.workers
    )
    arms = {
        arm: reduce_arm_receipts(
            source_state,
            checkpoint,
            receipts[arm],
            arm=arm,
            candidate_budget=int(contract["runtime"]["candidate_budget_per_source_arm"]),
        )
        for arm, checkpoint in checkpoints.items()
    }
    lock = {
        "schema_version": "pmo_joint_dependency_jump_candidate_lock_v2",
        "contract_sha256": contract["contract_sha256"],
        "source_id": identity(source_state),
        "source_state": source_state,
        "scores_present": False,
        "teacher_endpoints_loaded_during_generation": False,
        "all_checkpoint_plan_latents_attempted": True,
        "arms": arms,
        "new_oracle_calls": 0,
    }
    _write_sealed(lock_path, lock)
    locked = load_envelope(lock_path)
    exact_budget = int(contract["runtime"]["candidate_budget_per_source_arm"])
    shortfalls = {
        arm: exact_budget - len(row["candidates"])
        for arm, row in locked["arms"].items()
        if len(row["candidates"]) != exact_budget
    }
    if shortfalls:
        result = _shortfall_result(contract, shortfalls)
        _write_sealed(result_path, result)
        print(json.dumps(result, sort_keys=True))
        return

    for relative, expected in contract["postlock_diagnostic_inputs"].items():
        verify_file(root / relative, expected)
    corpus = load_envelope(root / CORPUS)
    support = evaluate_locked_support(locked, corpus, contract)
    descendants = sample_recursive_refinements(locked, seed=int(contract["runtime"]["seed"]))
    shared = locked["arms"]["shared_all_routes"]
    route_scale = sum(
        row["primitive_count"] >= contract["runtime"]["minimum_route_scale_primitives"]
        for row in shared["candidates"]
    )
    shared_support = sum(
        row["arms"]["shared_all_routes"]["exact_endpoint_support"]
        + row["arms"]["shared_all_routes"]["transformation_equivalent_support"]
        for task, row in support.items()
        if task in {"celecoxib_rediscovery", "gsk3b"}
    )
    complete_descendants = sum(row["status"] == "complete" for row in descendants)
    tail_count_matches = len(descendants) == route_scale
    perindopril_abstained = (
        support["perindopril_mpo"]["complete_single_program_teacher_support"] == 0
    )
    passed = (
        all(row["exact_replay"] for row in shared["candidates"])
        and shared_support > 0
        and route_scale > 0
        and tail_count_matches
        and complete_descendants > 0
        and perindopril_abstained
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "schema_version": SCHEMA,
        "decision": (
            "ZERO_ORACLE_SUPPORT_GATE_PASSED_FREEZE_SCORED_CONTRACT"
            if passed
            else "NO_SCORED_LAUNCH_SUPPORT_GATE_FAILED"
        ),
        "contract_sha256": contract["contract_sha256"],
        "new_oracle_calls": 0,
        "modal_launches": 0,
        "scored_launch_authorized": False,
        "code_revision_at_run": revision,
        "implementation_sha256": contract["implementation_sha256"],
        "generation_inputs": contract["generation_inputs"],
        "postlock_diagnostic_inputs": contract["postlock_diagnostic_inputs"],
        "artifacts": {
            str(checkpoint_path.relative_to(root)): sha256_file(checkpoint_path),
            str(lock_path.relative_to(root)): sha256_file(lock_path),
        },
        "candidate_lock_payload_sha256": identity(locked),
        "support": support,
        "recursive_refinement": {
            "children": descendants,
            "complete_children": complete_descendants,
            "one_tail_attempted_per_locked_route_scale_jump": tail_count_matches,
            "children_use_locked_jump_endpoint_as_parent": True,
            "frozen_parent_restart": False,
        },
        "gate": {
            "every_arm_locked_exactly_32": all(
                len(row["candidates"]) == exact_budget for row in locked["arms"].values()
            ),
            "all_locked_shared_candidates_exact_replay": all(
                row["exact_replay"] for row in shared["candidates"]
            ),
            "shared_celecoxib_or_gsk_exact_or_equivalent_support": shared_support,
            "route_scale_locked_candidates": route_scale,
            "complete_recursive_descendants": complete_descendants,
            "perindopril_complete_program_abstention": perindopril_abstained,
        },
        "claim_boundary": {
            "shared_checkpoint_uses_all_route_supervision": True,
            "shared_support_is_not_held_task_family_transfer": True,
            "held_fold_clean_support_reported_separately": True,
            "historical_scores_used_for_generation": False,
            "prospective_objective_evidence": False,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
            "machine": platform.machine(),
        },
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_sealed(result_path, result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "result": str(result_path.relative_to(root)),
                "route_scale_candidates": route_scale,
                "shared_support": shared_support,
                "recursive_descendants": complete_descendants,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
