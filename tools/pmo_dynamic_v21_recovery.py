#!/usr/bin/env python3
"""Prepare, audit and run the one-time PMO Dynamic-v2.1 recovery."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments import pmo_dynamic_v21 as original
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.pmo_dynamic_v21_recovery import (
    FAILED_V21_CALLS_PER_TASK,
    FAILURE_CENSUS,
    LAUNCH_V3,
    MAX_NEW_CALLS,
    MAX_NEW_CALLS_PER_TASK,
    RECOVERY_CONTRACT,
    RECOVERY_DOC,
    RECOVERY_PREFLIGHT,
    RESULT_V3,
    SOURCE_CHARGED_CALLS,
    SOURCE_RUN_ID,
    _unit_folder,
    _verify_census_files,
    build_failure_census,
    load_recovery_contract,
    recovery_preflight,
    run_recovery_task,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal

ROOT = Path(__file__).resolve().parents[1]


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _recovery_inputs() -> tuple[str, ...]:
    return (
        original.CONTRACT,
        original.PREFLIGHT,
        original.LAUNCH,
        original.LAUNCH_V2,
        original.PREQUERY_FAILURE,
        original.RESULT,
        FAILURE_CENSUS,
        RECOVERY_DOC,
        "src/compose_v4/experiments/pmo_dynamic_v21_recovery.py",
        "tools/pmo_dynamic_v21_recovery.py",
    )


def prepare(root: Path = ROOT) -> dict:
    """Seal the immutable failure census, recovery contract and zero-call audit."""
    for relative in (FAILURE_CENSUS, RECOVERY_CONTRACT, RECOVERY_PREFLIGHT):
        if (root / relative).exists():
            raise ValueError("PMO v2.1 recovery artifacts already exist; do not relock")
    original_contract = original.load_contract(root)
    census = build_failure_census(root)
    seal(root / FAILURE_CENSUS, census)
    contract = {
        "schema_version": "pmo_dynamic_v21_recovery_contract_v1",
        "scientific_problem": (
            "resume four failed Dynamic-v2.1 PMO units without requerying or relabeling "
            "their 132 completed observations"
        ),
        "primary_output": (
            "four completed query-contiguous Dynamic-v2.1 ledgers and matched offline "
            "Dynamic-v2.1-minus-v0 trajectories"
        ),
        "claim": (
            "a generic campaign-boundary bootstrap-continuity adapter can preserve "
            "physical pool provenance while satisfying the unchanged core invariant"
        ),
        "source_run_id": SOURCE_RUN_ID,
        "source_code_revision": unseal(root / original.LAUNCH_V2)["code_revision"],
        "prepared_code_revision": _git(root, "rev-parse", "HEAD"),
        "source_launch_v2_sha256": sha256_file(root / original.LAUNCH_V2),
        "source_incomplete_result_sha256": sha256_file(root / original.RESULT),
        "source_contract_sha256": sha256_file(root / original.CONTRACT),
        "source_preflight_sha256": sha256_file(root / original.PREFLIGHT),
        "failure_census_sha256": sha256_file(root / FAILURE_CENSUS),
        "tasks": list(original.TASKS),
        "arm": "dynamic_v21",
        "task_seeds": original.TASK_SEEDS,
        "controller": original_contract["controller"],
        "source_calls_per_task": FAILED_V21_CALLS_PER_TASK,
        "preserved_source_calls": len(original.TASKS) * FAILED_V21_CALLS_PER_TASK,
        "max_new_calls_per_task": MAX_NEW_CALLS_PER_TASK,
        "max_new_calls": MAX_NEW_CALLS,
        "final_per_task_ceiling": original.QUERY_BUDGET,
        "final_matched_total_ceiling": original.TOTAL_QUERY_CEILING,
        "workers": original.MAX_WORKERS,
        "cpu_per_worker": 1,
        "gpu": False,
        "automatic_retries": 0,
        "continuity_policy": {
            "boundary": "PMO campaign admission",
            "stable_identity": "first bootstrap pool in the durable archive snapshot",
            "physical_pool_identity": "retained as generation_pool_id",
            "proposal_accounting": "merge score-blind fresh-pool counters once",
            "rng": "advance to each durably locked physical pool post-proposal state",
            "core_mismatch_invariant_changed": False,
        },
        "existing_round1_score_policy": (
            "consume query index 32 from the ledger cache and update the restored "
            "controller before any new oracle call"
        ),
        "preserve": {
            "launch_v1": True,
            "launch_v2": True,
            "incomplete_result": True,
            "original_failures": True,
            "all_source_query_receipts": True,
            "completed_dynamic_v0_units": True,
        },
        "initial_task_specific_complete_routes": 0,
        "task_specific_tuning": False,
        "comparator_available_to_runtime": False,
        "t4_affected": False,
        "inputs": {
            relative: sha256_file(root / relative) for relative in _recovery_inputs()
        },
    }
    seal(root / RECOVERY_CONTRACT, contract)
    load_recovery_contract(root)
    preflight = recovery_preflight(root)
    seal(root / RECOVERY_PREFLIGHT, preflight)
    return {
        "status": "recovery_ready_for_commit",
        "failure_census_sha256": sha256_file(root / FAILURE_CENSUS),
        "recovery_contract_sha256": sha256_file(root / RECOVERY_CONTRACT),
        "recovery_preflight_sha256": sha256_file(root / RECOVERY_PREFLIGHT),
        "preserved_completed_calls": preflight["preserved_completed_calls"],
        "max_new_calls": MAX_NEW_CALLS,
        "new_oracle_calls": 0,
    }


def refresh(root: Path = ROOT) -> dict:
    """Reseal prelaunch implementation identities without changing recovery policy."""
    if (root / LAUNCH_V3).exists() or (root / RESULT_V3).exists():
        raise ValueError("PMO v2.1 recovery already launched; do not refresh")
    if any(
        (_unit_folder(root, task) / "recovery_v1").exists() for task in original.TASKS
    ):
        raise ValueError("PMO v2.1 recovery unit already advanced; do not refresh")
    prior_contract = sha256_file(root / RECOVERY_CONTRACT)
    prior_preflight = sha256_file(root / RECOVERY_PREFLIGHT)
    prior_census = sha256_file(root / FAILURE_CENSUS)
    census = unseal(root / FAILURE_CENSUS)
    _verify_census_files(root, census)
    if build_failure_census(root) != census:
        raise ValueError("PMO v2.1 source failure census changed before refresh")
    contract = unseal(root / RECOVERY_CONTRACT)
    if (
        contract.get("schema_version") != "pmo_dynamic_v21_recovery_contract_v1"
        or contract.get("source_run_id") != SOURCE_RUN_ID
        or contract.get("tasks") != list(original.TASKS)
        or contract.get("max_new_calls_per_task") != MAX_NEW_CALLS_PER_TASK
        or contract.get("automatic_retries") != 0
        or contract.get("t4_affected") is not False
    ):
        raise ValueError("PMO v2.1 recovery policy changed before refresh")
    contract["inputs"] = {
        relative: sha256_file(root / relative) for relative in _recovery_inputs()
    }
    contract.setdefault("prelaunch_repairs", []).append(
        {
            "reason": (
                "permit the 4.3-GB source run tree to remain outside Git only under "
                "the exact committed content-hash census"
            ),
            "prior_contract_sha256": prior_contract,
            "prior_preflight_sha256": prior_preflight,
            "failure_census_sha256": prior_census,
            "scientific_policy_changed": False,
            "new_oracle_calls": 0,
        }
    )
    seal(root / RECOVERY_CONTRACT, contract)
    load_recovery_contract(root)
    preflight = recovery_preflight(root)
    seal(root / RECOVERY_PREFLIGHT, preflight)
    return {
        "status": "recovery_prelaunch_identities_refreshed",
        "prior_contract_sha256": prior_contract,
        "prior_preflight_sha256": prior_preflight,
        "failure_census_sha256": prior_census,
        "recovery_contract_sha256": sha256_file(root / RECOVERY_CONTRACT),
        "recovery_preflight_sha256": sha256_file(root / RECOVERY_PREFLIGHT),
        "new_oracle_calls": 0,
    }


def validate_launch_ready(root: Path, *, workers: int) -> tuple[dict, dict, dict]:
    if type(workers) is not int or not 1 <= workers <= original.MAX_WORKERS:
        raise ValueError("PMO v2.1 recovery permits one to four workers")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("PMO v2.1 recovery requires clean committed tracked files")
    status_rows = _git(
        root, "status", "--porcelain", "--untracked-files=normal"
    ).splitlines()
    allowed_untracked = "?? diagnostics/pmo_dynamic_v21/runs/"
    unexpected = [row for row in status_rows if row != allowed_untracked]
    if unexpected:
        raise ValueError(
            f"PMO v2.1 recovery has unexpected untracked files: {unexpected}"
        )
    tracked = set(_git(root, "ls-files").splitlines())
    required = {
        RECOVERY_CONTRACT,
        FAILURE_CENSUS,
        RECOVERY_PREFLIGHT,
        original.LAUNCH_V2,
        original.RESULT,
        RECOVERY_DOC,
        "src/compose_v4/experiments/pmo_dynamic_v21_recovery.py",
        "tests/test_pmo_dynamic_v21_recovery.py",
        "tools/pmo_dynamic_v21_recovery.py",
    }
    if not required <= tracked:
        raise ValueError(
            f"PMO recovery launch inputs are not committed: {sorted(required - tracked)}"
        )
    if (root / LAUNCH_V3).exists() or (root / RESULT_V3).exists():
        raise ValueError("PMO v2.1 recovery launch already exists; no retry")
    contract = load_recovery_contract(root)
    preflight = unseal(root / RECOVERY_PREFLIGHT)
    census = unseal(root / FAILURE_CENSUS)
    expected_source_paths = set(census["source_artifact_sha256"])
    actual_source_paths = {
        str(path.relative_to(root))
        for path in (root / "diagnostics/pmo_dynamic_v21/runs" / SOURCE_RUN_ID).rglob(
            "*"
        )
        if path.is_file()
    }
    if actual_source_paths != expected_source_paths:
        raise ValueError("PMO v2.1 source run path set differs from its failure census")
    if (
        preflight.get("schema_version") != "pmo_dynamic_v21_recovery_preflight_v1"
        or preflight.get("passed") is not True
        or preflight.get("new_oracle_calls") != 0
        or preflight.get("contract_sha256") != sha256_file(root / RECOVERY_CONTRACT)
        or preflight.get("preserved_completed_calls")
        != len(original.TASKS) * FAILED_V21_CALLS_PER_TASK
        or preflight.get("remaining_query_ceiling") != MAX_NEW_CALLS
        or preflight.get("implementation_sha256") != contract["inputs"]
    ):
        raise ValueError("PMO v2.1 recovery zero-oracle preflight is stale or failed")
    _verify_census_files(root, census)
    for task in original.TASKS:
        folder = _unit_folder(root, task)
        if (folder / "result.json").exists() or (folder / "recovery_v1").exists():
            raise ValueError("PMO v2.1 recovery unit was already advanced")
    original.verify_runtime_environment(root, original.load_contract(root))
    return contract, preflight, census


def _worker(root: str, launch: dict, task_name: str) -> dict:
    return run_recovery_task(Path(root), launch, task_name)


def _completed_v0_results(root: Path) -> list[dict]:
    rows = []
    for task in original.TASKS:
        path = (
            root
            / "diagnostics/pmo_dynamic_v21/runs"
            / SOURCE_RUN_ID
            / "dynamic_v0"
            / task
            / "result.json"
        )
        row = unseal(path)
        if row.get("status") != "complete" or row.get("charged_oracle_calls") != 1000:
            raise ValueError("recovery comparator is not a completed Dynamic-v0 unit")
        rows.append(row)
    return rows


def launch_v3(root: Path = ROOT, *, workers: int = original.MAX_WORKERS) -> dict:
    """Run the one explicitly authorized recovery; never retry it automatically."""
    contract, preflight, census = validate_launch_ready(root, workers=workers)
    body = {
        "schema_version": "pmo_dynamic_v21_recovery_launch_v3",
        "recovery_contract_sha256": sha256_file(root / RECOVERY_CONTRACT),
        "recovery_preflight_sha256": sha256_file(root / RECOVERY_PREFLIGHT),
        "failure_census_sha256": sha256_file(root / FAILURE_CENSUS),
        "code_revision": _git(root, "rev-parse", "HEAD"),
        "source_launch_v2_path": original.LAUNCH_V2,
        "source_launch_v2_sha256": contract["source_launch_v2_sha256"],
        "source_run_id": SOURCE_RUN_ID,
        "source_contract_sha256": contract["source_contract_sha256"],
        "source_incomplete_result_path": original.RESULT,
        "source_incomplete_result_sha256": contract["source_incomplete_result_sha256"],
        "tasks": list(original.TASKS),
        "arm": "dynamic_v21",
        "task_seeds": original.TASK_SEEDS,
        "configuration": contract["controller"],
        "configuration_id": identity(contract["controller"]),
        "source_charged_calls": SOURCE_CHARGED_CALLS,
        "preserved_v21_calls_per_task": FAILED_V21_CALLS_PER_TASK,
        "max_new_calls_per_task": MAX_NEW_CALLS_PER_TASK,
        "max_new_calls": MAX_NEW_CALLS,
        "final_matched_total_ceiling": original.TOTAL_QUERY_CEILING,
        "workers": workers,
        "cpu_per_worker": 1,
        "automatic_retries": 0,
        "continuity_boundary": "PMO campaign admission",
        "preflight_schema": preflight["schema_version"],
    }
    launch = {**body, "recovery_id": identity(body)}
    seal(root / LAUNCH_V3, launch)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_worker, str(root), launch, task): task
            for task in original.TASKS
        }
        for future in as_completed(futures):
            results.append(future.result())
    _verify_census_files(root, census)
    aggregate = original.aggregate_offline(
        root, launch, [*_completed_v0_results(root), *results]
    )
    aggregate.update(
        schema_version="pmo_dynamic_v21_recovery_result_v1",
        recovery_id=launch["recovery_id"],
        recovery_contract_sha256=launch["recovery_contract_sha256"],
        source_run_id=SOURCE_RUN_ID,
        source_incomplete_result_sha256=launch["source_incomplete_result_sha256"],
        preserved_source_charged_calls=SOURCE_CHARGED_CALLS,
        recovery_new_charged_calls=sum(
            max(0, row.get("charged_oracle_calls", 0) - FAILED_V21_CALLS_PER_TASK)
            for row in results
        ),
        recovery_new_charged_call_ceiling=MAX_NEW_CALLS,
        automatic_retries=0,
    )
    seal(root / RESULT_V3, aggregate)
    return aggregate


def status(root: Path = ROOT) -> dict:
    """Read-only recovery status; never constructs an oracle or restores mutable RNG."""
    if not (root / LAUNCH_V3).exists():
        return {
            "status": (
                "prepared" if (root / RECOVERY_PREFLIGHT).exists() else "not_prepared"
            ),
            "new_oracle_calls": 0,
        }
    launch = unseal(root / LAUNCH_V3)
    tasks = {}
    for task in original.TASKS:
        folder = _unit_folder(root, task)
        result_path = folder / "result.json"
        recovery_failure = folder / "recovery_v1/failure.json"
        charged = len(list((folder / "oracle").glob("query_*/started.json")))
        row = {
            "charged_oracle_calls": charged,
            "preserved_charged_calls": FAILED_V21_CALLS_PER_TASK,
            "new_charged_calls": charged - FAILED_V21_CALLS_PER_TASK,
            "remaining_calls": original.QUERY_BUDGET - charged,
        }
        if result_path.exists():
            result = unseal(result_path)
            row.update(status="complete", termination=result["termination"])
        elif recovery_failure.exists():
            failure = unseal(recovery_failure)
            row.update(status="failed", error=failure["error"])
        elif (folder / "recovery_v1/started.json").exists():
            row.update(status="started")
        else:
            row.update(status="pending")
        tasks[task] = row
    return {
        "status": "complete" if (root / RESULT_V3).exists() else "launched",
        "recovery_id": launch["recovery_id"],
        "source_run_id": SOURCE_RUN_ID,
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "refresh", "launch-v3", "status"))
    parser.add_argument("--workers", type=int, default=original.MAX_WORKERS)
    args = parser.parse_args()
    if args.action == "prepare":
        row = prepare()
    elif args.action == "refresh":
        row = refresh()
    elif args.action == "launch-v3":
        row = launch_v3(workers=args.workers)
    else:
        row = status()
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
