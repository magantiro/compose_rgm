"""Audit v3 and launch the exact complete-input T4 rescue repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_frozen_program_rescue import (
    RELAUNCH_V4_LOCK_SCHEMA,
    REPAIRED_INPUT_REASONS,
    raw_sha256,
    validate_prequery_input_identity_error,
    validate_relaunch_query_paths,
    validate_repaired_input_overrides,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from tools.t4_frozen_program_rescue_relaunch_v3 import (
    APP,
    APP_NAME,
    ROOT,
    SOURCE_CONTRACT,
    _paths,
    _query_indices,
    _remote_bytes,
)

V2_LOCK = Path("diagnostics/t4_frozen_program_rescue/relaunch_lock_v2.json")
V3_LOCK = Path("diagnostics/t4_frozen_program_rescue/relaunch_lock_v3.json")
V3_RECEIPT = Path("diagnostics/t4_frozen_program_rescue/relaunch_v3.json")
V4_LOCK = Path("diagnostics/t4_frozen_program_rescue/relaunch_lock_v4.json")
V4_FAILURE = Path("diagnostics/t4_frozen_program_rescue/prequery_failure_0004.json")
V4_RECEIPT = Path("diagnostics/t4_frozen_program_rescue/relaunch_v4.json")


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _input_overrides(contract: dict) -> list[dict]:
    mismatches = {}
    for path, frozen in contract["inputs"].items():
        local = ROOT / path
        if not local.exists():
            raise ValueError(f"frozen T4 input is absent: {path}")
        repaired = hashlib.sha256(local.read_bytes()).hexdigest()
        if repaired != frozen:
            mismatches[path] = repaired
    if set(mismatches) != set(REPAIRED_INPUT_REASONS):
        raise ValueError(f"unexpected frozen-input mismatch set: {sorted(mismatches)}")
    overrides = [
        {
            "path": path,
            "frozen_sha256": contract["inputs"][path],
            "repaired_sha256": mismatches[path],
            "reason": reason,
        }
        for path, reason in sorted(REPAIRED_INPUT_REASONS.items())
    ]
    validate_repaired_input_overrides(
        overrides,
        frozen_sha256={row["path"]: row["frozen_sha256"] for row in overrides},
        repaired_sha256={row["path"]: row["repaired_sha256"] for row in overrides},
    )
    return overrides


def prepare() -> dict:
    import modal

    if (ROOT / V4_LOCK).exists() or (ROOT / V4_FAILURE).exists():
        raise ValueError("T4 rescue v4 audit artifacts already exist")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("prepare T4 rescue v4 from clean committed source")
    lock_v2 = unseal(ROOT / V2_LOCK)
    lock_v3 = unseal(ROOT / V3_LOCK)
    receipt_v3 = unseal(ROOT / V3_RECEIPT)
    if (
        lock_v3.get("schema_version") != "t4_frozen_program_rescue_relaunch_lock_v3"
        or receipt_v3.get("schema_version") != "t4_frozen_program_rescue_relaunch_v3"
        or set(receipt_v3.get("units", {})) != set(lock_v2.get("units", {}))
        or receipt_v3.get("automatic_retries") != 0
    ):
        raise ValueError("T4 rescue v3 lock/receipt lineage changed")

    contract = unseal(ROOT / SOURCE_CONTRACT)
    overrides = _input_overrides(contract)
    failure_override = next(
        row
        for row in overrides
        if row["path"] == "src/compose_v4/control/program_transfer.py"
    )
    volume = modal.Volume.from_name(lock_v2["source_volume"])
    prefix = lock_v2["source_volume_path"]
    failures = {}
    units = {}
    for unit_id, launch in sorted(receipt_v3["units"].items()):
        call = modal.FunctionCall.from_id(launch["call_id"])
        try:
            value = call.get(timeout=1)
        except TimeoutError as error:
            raise ValueError(f"T4 rescue v3 call is still active: {unit_id}") from error
        except ValueError as error:
            message = str(error)
            validate_prequery_input_identity_error(
                message,
                path=failure_override["path"],
                frozen_sha256=failure_override["frozen_sha256"],
                repaired_sha256=failure_override["repaired_sha256"],
            )
            failures[unit_id] = {
                "call_id": launch["call_id"],
                "error_type": type(error).__name__,
                "error": message,
                "dashboard_url": call.get_dashboard_url(),
            }
        else:
            raise ValueError(
                f"T4 rescue v3 unexpectedly returned instead of failing prequery: "
                f"{unit_id}: {value!r}"
            )

        prior = lock_v2["units"][unit_id]
        unit_prefix = f"{prefix}/units/{unit_id}"
        paths = _paths(volume, unit_prefix)
        if f"{unit_prefix}/result.json" in paths:
            raise ValueError(f"T4 rescue v3 unexpectedly completed: {unit_id}")
        started, results = _query_indices(paths)
        if validate_relaunch_query_paths(started, results) != prior["current_query_count"]:
            raise ValueError(f"T4 rescue v3 changed query ledger: {unit_id}")
        if raw_sha256(_remote_bytes(volume, f"{unit_prefix}/failure.json")) != prior[
            "failure_sha256"
        ]:
            raise ValueError(f"T4 rescue v3 changed durable failure: {unit_id}")
        units[unit_id] = prior

    if len(failures) != 19:
        raise ValueError("T4 rescue v3 prequery failure census is not 19")
    audited_at = _stamp()
    failure_body = {
        "schema_version": "t4_frozen_program_rescue_prequery_failure_v4",
        "source_run_id": lock_v2["source_run_id"],
        "relaunch_v3_receipt_sha256": sha256_file(ROOT / V3_RECEIPT),
        "failure_stage": "next_contract_input_identity_before_run_unit",
        "complete_contract_input_overrides": overrides,
        "units": failures,
        "oracle_calls": 0,
        "remote_writes": 0,
        "automatic_retries": 0,
        "audited_at_utc": audited_at,
    }
    seal(ROOT / V4_FAILURE, failure_body)
    implementation = {
        path: sha256_file(ROOT / path)
        for path in (
            APP,
            "src/compose_v4/control/adaptive_program_optimizer.py",
            "src/compose_v4/control/program_transfer.py",
            "src/compose_v4/experiments/t4_frozen_program_rescue.py",
            "tools/t4_frozen_program_rescue_relaunch_v4.py",
            "docs/T4_FROZEN_PROGRAM_RESCUE.md",
        )
    }
    body = {
        "schema_version": RELAUNCH_V4_LOCK_SCHEMA,
        "source_run_id": lock_v2["source_run_id"],
        "source_volume": lock_v2["source_volume"],
        "source_volume_path": prefix,
        "source_task": lock_v2["source_task"],
        "units": units,
        "active_first_relaunch_units": lock_v2["active_first_relaunch_units"],
        "relaunch_v2_lock_sha256": sha256_file(ROOT / V2_LOCK),
        "relaunch_v3_lock_sha256": sha256_file(ROOT / V3_LOCK),
        "relaunch_v3_receipt_sha256": sha256_file(ROOT / V3_RECEIPT),
        "prequery_failure_sha256": sha256_file(ROOT / V4_FAILURE),
        "contract_input_overrides": overrides,
        "new_search_call_ceiling": sum(
            row["remaining_call_ceiling"] for row in units.values()
        ),
        "automatic_retries": 0,
        "controller_changes": False,
        "implementation_sha256": implementation,
        "prepared_code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "prepared_at_utc": audited_at,
        "new_oracle_calls_during_prepare": 0,
    }
    seal(ROOT / V4_LOCK, body)
    return body


def _validate_lock() -> dict:
    lock = unseal(ROOT / V4_LOCK)
    overrides = _input_overrides(unseal(ROOT / SOURCE_CONTRACT))
    if (
        lock.get("schema_version") != RELAUNCH_V4_LOCK_SCHEMA
        or lock.get("contract_input_overrides") != overrides
        or lock.get("automatic_retries") != 0
        or lock.get("controller_changes") is not False
        or sha256_file(ROOT / V4_FAILURE) != lock.get("prequery_failure_sha256")
    ):
        raise ValueError("T4 rescue v4 lock changed scope or lineage")
    for path, digest in lock["implementation_sha256"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"T4 rescue v4 implementation changed: {path}")
    return lock


def launch() -> dict:
    import modal

    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    if (ROOT / V4_RECEIPT).exists():
        raise ValueError("T4 rescue v4 receipt already exists")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("T4 rescue v4 requires clean committed source")
    lock = _validate_lock()
    volume = modal.Volume.from_name(lock["source_volume"])
    prefix = lock["source_volume_path"]
    for unit_id, row in lock["units"].items():
        unit_prefix = f"{prefix}/units/{unit_id}"
        paths = _paths(volume, unit_prefix)
        if f"{unit_prefix}/result.json" in paths:
            raise ValueError(f"T4 unit completed after v4 lock: {unit_id}")
        started, results = _query_indices(paths)
        if validate_relaunch_query_paths(started, results) != row["current_query_count"]:
            raise ValueError(f"remote query ledger changed after v4 lock: {unit_id}")
        if raw_sha256(_remote_bytes(volume, f"{unit_prefix}/failure.json")) != row[
            "failure_sha256"
        ]:
            raise ValueError(f"remote durable failure changed after v4 lock: {unit_id}")

    revision = local_image_revision(expected_commit=assert_synced(strict=True)["commit"])
    files = {
        path: sha256_file(ROOT / path)
        for path in (
            APP,
            str(V4_LOCK),
            "modal_apps/t4_frozen_program_benchmark_app.py",
            "modal_apps/genmol_t4_opt_app.py",
        )
    }
    body = {
        "rescue_image_revision": revision,
        "relaunch_lock_sha256": sha256_file(ROOT / V4_LOCK),
        "files_sha256": files,
    }
    task = {**body, "run_id": identity(body)}
    receipt = {
        "schema_version": "t4_frozen_program_rescue_relaunch_v4",
        "task": task,
        "units": {},
        "automatic_retries": 0,
        "launched_at_utc": _stamp(),
    }
    seal(ROOT / V4_RECEIPT, receipt)
    worker = modal.Function.from_name(APP_NAME, "worker")
    for unit_id in sorted(lock["units"]):
        receipt["units"][unit_id] = {"status": "spawn_reserved", "at": _stamp()}
        seal(ROOT / V4_RECEIPT, receipt)
        call = worker.spawn({**task, "unit_id": unit_id})
        receipt["units"][unit_id] = {
            "status": "spawned",
            "call_id": call.object_id,
            "at": _stamp(),
        }
        seal(ROOT / V4_RECEIPT, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch"))
    args = parser.parse_args()
    result = prepare() if args.action == "prepare" else launch()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
