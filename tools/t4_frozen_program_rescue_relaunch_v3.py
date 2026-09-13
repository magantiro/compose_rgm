"""Audit the prequery identity stop and launch the one-input T4 repair."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_frozen_program_rescue import (
    RELAUNCH_V3_LOCK_SCHEMA,
    REPAIRED_INPUT_PATH,
    REPAIRED_INPUT_REASON,
    raw_sha256,
    validate_relaunch_prequery_identity_error,
    validate_relaunch_query_paths,
    validate_repaired_input_override,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONTRACT = Path("configs/t4_frozen_program_benchmark_v2.json")
V2_LOCK = Path("diagnostics/t4_frozen_program_rescue/relaunch_lock_v2.json")
V2_RECEIPT = Path("diagnostics/t4_frozen_program_rescue/relaunch_v2.json")
V3_LOCK = Path("diagnostics/t4_frozen_program_rescue/relaunch_lock_v3.json")
V3_FAILURE = Path("diagnostics/t4_frozen_program_rescue/prequery_failure_0003.json")
V3_RECEIPT = Path("diagnostics/t4_frozen_program_rescue/relaunch_v3.json")
APP = "modal_apps/t4_frozen_program_rescue_app.py"
APP_NAME = "compose-t4-frozen-program-rescue"
QUERY_PATTERN = re.compile(r"/queries/query_(\d{4})/(started|result)\.json$")


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _remote_bytes(volume, path: str) -> bytes:
    return b"".join(volume.read_file(path))


def _paths(volume, prefix: str) -> set[str]:
    return {entry.path.lstrip("/") for entry in volume.listdir(prefix, recursive=True)}


def _query_indices(paths: set[str]) -> tuple[set[int], set[int]]:
    started, results = set(), set()
    for path in paths:
        match = QUERY_PATTERN.search(path)
        if match:
            (started if match.group(2) == "started" else results).add(
                int(match.group(1))
            )
    return started, results


def _validate_v2() -> tuple[dict, dict]:
    lock = unseal(ROOT / V2_LOCK)
    receipt = unseal(ROOT / V2_RECEIPT)
    if (
        lock.get("schema_version") != "t4_frozen_program_rescue_relaunch_lock_v2"
        or receipt.get("schema_version") != "t4_frozen_program_rescue_relaunch_v2"
        or set(receipt.get("units", {})) != set(lock.get("units", {}))
        or receipt.get("automatic_retries") != 0
    ):
        raise ValueError("T4 rescue v2 lock/receipt lineage changed")
    return lock, receipt


def prepare() -> dict:
    import modal

    if (ROOT / V3_LOCK).exists() or (ROOT / V3_FAILURE).exists():
        raise ValueError("T4 rescue v3 audit artifacts already exist")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("prepare T4 rescue v3 from clean committed source")
    lock_v2, receipt_v2 = _validate_v2()
    source_contract = unseal(ROOT / SOURCE_CONTRACT)
    frozen_sha256 = source_contract["inputs"][REPAIRED_INPUT_PATH]
    repaired_sha256 = sha256_file(ROOT / REPAIRED_INPUT_PATH)
    override = {
        "path": REPAIRED_INPUT_PATH,
        "frozen_sha256": frozen_sha256,
        "repaired_sha256": repaired_sha256,
        "reason": REPAIRED_INPUT_REASON,
    }
    validate_repaired_input_override(
        override,
        frozen_sha256=frozen_sha256,
        repaired_sha256=repaired_sha256,
    )

    volume = modal.Volume.from_name(lock_v2["source_volume"])
    prefix = lock_v2["source_volume_path"]
    failures = {}
    units = {}
    for unit_id, launch in sorted(receipt_v2["units"].items()):
        call = modal.FunctionCall.from_id(launch["call_id"])
        try:
            value = call.get(timeout=1)
        except TimeoutError as error:
            raise ValueError(f"T4 rescue v2 call is still active: {unit_id}") from error
        except ValueError as error:
            message = str(error)
            validate_relaunch_prequery_identity_error(
                message,
                frozen_sha256=frozen_sha256,
                repaired_sha256=repaired_sha256,
            )
            failures[unit_id] = {
                "call_id": launch["call_id"],
                "error_type": type(error).__name__,
                "error": message,
                "dashboard_url": call.get_dashboard_url(),
            }
        else:
            raise ValueError(
                f"T4 rescue v2 unexpectedly returned instead of failing prequery: "
                f"{unit_id}: {value!r}"
            )

        prior = lock_v2["units"][unit_id]
        unit_prefix = f"{prefix}/units/{unit_id}"
        paths = _paths(volume, unit_prefix)
        if f"{unit_prefix}/result.json" in paths:
            raise ValueError(f"T4 rescue v2 unexpectedly completed: {unit_id}")
        started, results = _query_indices(paths)
        query_count = validate_relaunch_query_paths(started, results)
        if query_count != prior["current_query_count"]:
            raise ValueError(f"T4 rescue v2 changed query ledger: {unit_id}")
        remote_failure = _remote_bytes(volume, f"{unit_prefix}/failure.json")
        if raw_sha256(remote_failure) != prior["failure_sha256"]:
            raise ValueError(f"T4 rescue v2 changed durable failure: {unit_id}")
        units[unit_id] = prior

    if len(failures) != 19:
        raise ValueError("T4 rescue v2 prequery failure census is not 19")
    audited_at = _stamp()
    failure_body = {
        "schema_version": "t4_frozen_program_rescue_prequery_failure_v3",
        "source_run_id": lock_v2["source_run_id"],
        "relaunch_v2_receipt_sha256": sha256_file(ROOT / V2_RECEIPT),
        "failure_stage": "contract_input_identity_before_run_unit",
        "contract_input_override": override,
        "units": failures,
        "oracle_calls": 0,
        "remote_writes": 0,
        "automatic_retries": 0,
        "audited_at_utc": audited_at,
    }
    seal(ROOT / V3_FAILURE, failure_body)

    implementation = {
        path: sha256_file(ROOT / path)
        for path in (
            APP,
            "src/compose_v4/control/adaptive_program_optimizer.py",
            "src/compose_v4/experiments/t4_frozen_program_rescue.py",
            "tools/t4_frozen_program_rescue_relaunch_v3.py",
            "docs/T4_FROZEN_PROGRAM_RESCUE.md",
        )
    }
    body = {
        "schema_version": RELAUNCH_V3_LOCK_SCHEMA,
        "source_run_id": lock_v2["source_run_id"],
        "source_volume": lock_v2["source_volume"],
        "source_volume_path": prefix,
        "source_task": lock_v2["source_task"],
        "units": units,
        "active_first_relaunch_units": lock_v2["active_first_relaunch_units"],
        "relaunch_v2_lock_sha256": sha256_file(ROOT / V2_LOCK),
        "relaunch_v2_receipt_sha256": sha256_file(ROOT / V2_RECEIPT),
        "prequery_failure_sha256": sha256_file(ROOT / V3_FAILURE),
        "contract_input_override": override,
        "new_search_call_ceiling": sum(
            row["remaining_call_ceiling"] for row in units.values()
        ),
        "automatic_retries": 0,
        "controller_changes": False,
        "effective_repairs": [
            "accept_exact_charged_missing_observation_status",
            "verify_that_one_repaired_input_against_its_sealed_repair_hash",
        ],
        "implementation_sha256": implementation,
        "prepared_code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "prepared_at_utc": audited_at,
        "new_oracle_calls_during_prepare": 0,
    }
    seal(ROOT / V3_LOCK, body)
    return body


def _validate_lock() -> dict:
    lock = unseal(ROOT / V3_LOCK)
    source_contract = unseal(ROOT / SOURCE_CONTRACT)
    override = lock.get("contract_input_override", {})
    validate_repaired_input_override(
        override,
        frozen_sha256=source_contract["inputs"][REPAIRED_INPUT_PATH],
        repaired_sha256=sha256_file(ROOT / REPAIRED_INPUT_PATH),
    )
    if (
        lock.get("schema_version") != RELAUNCH_V3_LOCK_SCHEMA
        or lock.get("automatic_retries") != 0
        or lock.get("controller_changes") is not False
        or sha256_file(ROOT / V3_FAILURE) != lock.get("prequery_failure_sha256")
    ):
        raise ValueError("T4 rescue v3 lock changed scope or lineage")
    for path, digest in lock["implementation_sha256"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"T4 rescue v3 implementation changed: {path}")
    return lock


def launch() -> dict:
    import modal

    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    if (ROOT / V3_RECEIPT).exists():
        raise ValueError("T4 rescue v3 receipt already exists")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("T4 rescue v3 requires clean committed source")
    lock = _validate_lock()
    volume = modal.Volume.from_name(lock["source_volume"])
    prefix = lock["source_volume_path"]
    for unit_id, row in lock["units"].items():
        unit_prefix = f"{prefix}/units/{unit_id}"
        paths = _paths(volume, unit_prefix)
        if f"{unit_prefix}/result.json" in paths:
            raise ValueError(f"T4 unit completed after v3 lock: {unit_id}")
        started, results = _query_indices(paths)
        if validate_relaunch_query_paths(started, results) != row["current_query_count"]:
            raise ValueError(f"remote query ledger changed after v3 lock: {unit_id}")
        if raw_sha256(_remote_bytes(volume, f"{unit_prefix}/failure.json")) != row[
            "failure_sha256"
        ]:
            raise ValueError(f"remote durable failure changed after v3 lock: {unit_id}")

    revision = local_image_revision(expected_commit=assert_synced(strict=True)["commit"])
    files = {
        path: sha256_file(ROOT / path)
        for path in (
            APP,
            str(V3_LOCK),
            "modal_apps/t4_frozen_program_benchmark_app.py",
            "modal_apps/genmol_t4_opt_app.py",
        )
    }
    body = {
        "rescue_image_revision": revision,
        "relaunch_lock_sha256": sha256_file(ROOT / V3_LOCK),
        "files_sha256": files,
    }
    task = {**body, "run_id": identity(body)}
    receipt = {
        "schema_version": "t4_frozen_program_rescue_relaunch_v3",
        "task": task,
        "units": {},
        "automatic_retries": 0,
        "launched_at_utc": _stamp(),
    }
    seal(ROOT / V3_RECEIPT, receipt)
    worker = modal.Function.from_name(APP_NAME, "worker")
    for unit_id in sorted(lock["units"]):
        receipt["units"][unit_id] = {"status": "spawn_reserved", "at": _stamp()}
        seal(ROOT / V3_RECEIPT, receipt)
        call = worker.spawn({**task, "unit_id": unit_id})
        receipt["units"][unit_id] = {
            "status": "spawned",
            "call_id": call.object_id,
            "at": _stamp(),
        }
        seal(ROOT / V3_RECEIPT, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch"))
    args = parser.parse_args()
    result = prepare() if args.action == "prepare" else launch()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
