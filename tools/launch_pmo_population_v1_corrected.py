#!/usr/bin/env python3
"""Fail-closed deploy/launch/status for the corrected authorized PMO-v1 run.

This launcher is intentionally separate from the terminally failed original
payload.  It binds the corrected worker, source capsule, authorization and
three-task/no-retry budget before any Modal call is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/pmo_population_controller_v1_scored_contract_corrected.json"
AUTH = ROOT / "diagnostics/pmo_population_controller_v1/corrected_scored_authorization_receipt.json"
MANIFEST = ROOT / "diagnostics/pmo_population_controller_v1/source_capsule_manifest_v2.json"
PREFLIGHT = ROOT / "diagnostics/pmo_population_controller_v1/corrected_preflight_receipt.json"
LAUNCH = ROOT / "diagnostics/pmo_population_controller_v1/corrected_scored_launch_receipt.json"
PAYLOAD = "047f6a0fd094ad548553fa1c28641284990a11e221263aa07948aec6eb18200a"
APP = "compose-pmo-population-v1-corrected"
TASKS = ("gsk3b", "perindopril_mpo", "celecoxib_rediscovery")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(path)


def _validate_contract_auth() -> tuple[dict, dict]:
    envelope = json.loads(CONTRACT.read_text())
    if envelope.get("payload_sha256") != PAYLOAD:
        raise RuntimeError("corrected contract payload hash changed")
    payload = envelope["payload"]
    # The contract itself remains fail-closed; user authorization is recorded
    # separately and must be explicit for this corrected payload.
    if payload.get("scored_launch_authorized") is not False or payload.get("modal_launch_authorized") is not False:
        raise RuntimeError("corrected contract fail-closed flags changed")
    auth = json.loads(AUTH.read_text())
    if auth.get("status") != "AUTHORIZED" or auth.get("payload_sha256") != PAYLOAD:
        raise RuntimeError("corrected scored authorization is absent or mismatched")
    if auth.get("scored_launch_authorized") is not True or auth.get("modal_source_upload_authorized") is not True:
        raise RuntimeError("corrected scored or Modal source-upload authorization is absent")
    if auth.get("oracle_calls_authorized") != 750:
        raise RuntimeError("corrected oracle authorization budget changed")
    if tuple(payload.get("tasks", ())) != TASKS:
        raise RuntimeError("task lock changed")
    budget = payload.get("budget", {})
    expected = {
        "charged_calls_per_task": 250,
        "charged_calls_total": 750,
        "initialization_calls_per_task": 16,
        "candidate_calls_per_task": 234,
        "queries_per_round": 16,
        "max_rounds": 64,
        "automatic_retries": 0,
        "backfill": False,
        "cpu_per_worker": 1,
        "gpu": False,
    }
    if budget != expected:
        raise RuntimeError("scored budget/no-retry contract changed")
    manifest = json.loads(MANIFEST.read_text())
    for relative, expected_hash in manifest["source_files"].items():
        path = ROOT / relative
        if not path.exists() or _sha(path) != expected_hash:
            raise RuntimeError(f"source capsule identity mismatch: {relative}")
    return payload, auth


def _validate() -> tuple[dict, dict]:
    payload, auth = _validate_contract_auth()
    if not PREFLIGHT.exists():
        raise RuntimeError("corrected local preflight receipt is missing")
    pre = json.loads(PREFLIGHT.read_text())
    if pre.get("status") not in {
        "PASS_CORRECTED_SOURCE_CONTRACT_AUTHORIZED_PENDING_MODAL_DEPLOY",
        "PASS_CORRECTED_SOURCE_CONTRACT_AUTHORIZED",
    }:
        raise RuntimeError("corrected preflight is not in the authorized deploy state")
    if pre.get("payload_sha256") != PAYLOAD or pre.get("scored_launch_authorized") is not True:
        raise RuntimeError("corrected preflight payload/authorization mismatch")
    return payload, auth


def preflight() -> dict:
    if LAUNCH.exists():
        raise RuntimeError("corrected scored launch receipt already exists; do not duplicate launch")
    payload, auth = _validate_contract_auth()
    receipt = {
        "schema_version": "pmo_population_corrected_scored_preflight_receipt_v1",
        "status": "PASS_CORRECTED_SOURCE_CONTRACT_AUTHORIZED",
        "contract_payload_sha256": PAYLOAD,
        "payload_sha256": PAYLOAD,
        "contract_file_sha256": _sha(CONTRACT),
        "authorization_receipt_sha256": _sha(AUTH),
        "source_capsule_manifest_sha256": _sha(MANIFEST),
        "worker_sha256": _sha(ROOT / "modal_apps/pmo_population_v1_app.py"),
        "tasks": list(TASKS),
        "charged_calls_total": payload["budget"]["charged_calls_total"],
        "automatic_retries": 0,
        "backfill": False,
        "modal_source_upload_authorized": auth["modal_source_upload_authorized"],
        "scored_launch_authorized": True,
        "oracle_calls_before_launch": 0,
        "modal_calls_before_launch": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write(PREFLIGHT, {**receipt, "receipt_sha256": _identity(receipt)})
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def launch() -> dict:
    if LAUNCH.exists():
        raise RuntimeError("corrected scored launch receipt exists; no duplicate launch")
    payload, _ = _validate()
    from modal import Function

    preflight_id = _sha(PREFLIGHT)
    run_id = _identity({
        "schema_version": "pmo_population_corrected_scored_launch_v1",
        "contract_payload_sha256": PAYLOAD,
        "preflight_sha256": preflight_id,
        "source_capsule_manifest_sha256": _sha(MANIFEST),
    })
    receipt = {
        "schema_version": "pmo_population_corrected_scored_launch_v1",
        "run_id": run_id,
        "app": APP,
        "contract_payload_sha256": PAYLOAD,
        "preflight_sha256": preflight_id,
        "tasks": list(TASKS),
        "automatic_retries": 0,
        "backfill": False,
        "calls_per_task": payload["budget"]["charged_calls_per_task"],
        "status": "SPAWNING",
        "calls": {},
    }
    _write(LAUNCH, receipt)
    function = Function.from_name(APP, "run_task")
    for task in TASKS:
        call = function.spawn({"run_id": run_id, "task": task, "contract_payload_sha256": PAYLOAD})
        receipt["calls"][task] = {"call_id": call.object_id, "status": "SPAWNED"}
        _write(LAUNCH, receipt)
        print(task, call.object_id, flush=True)
    receipt["status"] = "SPAWNED_ALL"
    _write(LAUNCH, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def status() -> None:
    if not LAUNCH.exists():
        raise RuntimeError("no corrected launch receipt")
    import modal

    receipt = json.loads(LAUNCH.read_text())
    for task, row in receipt["calls"].items():
        try:
            value = modal.FunctionCall.from_id(row["call_id"]).get(timeout=0)
        except TimeoutError:
            print(task, "PENDING", row["call_id"], flush=True)
        except Exception as error:  # noqa: BLE001
            print(task, "FAILED", type(error).__name__, str(error), flush=True)
        else:
            print(task, json.dumps(value, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "launch", "status"))
    args = parser.parse_args()
    {"preflight": preflight, "launch": launch, "status": status}[args.command]()


if __name__ == "__main__":
    main()
