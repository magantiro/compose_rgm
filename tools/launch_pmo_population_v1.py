#!/usr/bin/env python3
"""Fail-closed preflight, launch, and status for the authorized PMO-v1 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/pmo_population_controller_v1_scored_contract.json"
AUTH = ROOT / "diagnostics/pmo_population_controller_v1/scored_authorization_receipt.json"
MANIFEST = ROOT / "diagnostics/pmo_population_controller_v1/source_capsule_manifest.json"
SUPPORT = ROOT / "diagnostics/pmo_population_live_parent_gate_v1/scheduler_result.json"
PREFLIGHT = ROOT / "diagnostics/pmo_population_controller_v1/scored_preflight_receipt.json"
LAUNCH = ROOT / "diagnostics/pmo_population_controller_v1/scored_launch_receipt.json"
PAYLOAD = "61c1fdf3995f0635b2f46a517e6e5ddd566f3ec9a26ce53ea3dec1c6e5a0b582"
APP = "compose-pmo-population-v1"
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


def preflight() -> dict:
    if PREFLIGHT.exists():
        raise RuntimeError("scored preflight already exists; do not overwrite")
    if LAUNCH.exists():
        raise RuntimeError("scored launch receipt already exists")
    envelope = json.loads(CONTRACT.read_text())
    if envelope.get("payload_sha256") != PAYLOAD:
        raise RuntimeError("scored contract payload hash changed")
    payload = envelope["payload"]
    if payload.get("scored_launch_authorized") is not False:
        raise RuntimeError("frozen scored contract must remain fail-closed on disk")
    if payload.get("modal_launch_authorized") is not False:
        raise RuntimeError("frozen scored contract Modal flag changed")
    auth = json.loads(AUTH.read_text())
    if auth.get("payload_sha256") != PAYLOAD or auth.get("modal_source_upload", {}).get("token_verified") is not True:
        raise RuntimeError("explicit scored or Modal source-upload authorization is absent")
    if tuple(payload.get("tasks", ())) != TASKS:
        raise RuntimeError("task lock changed")
    budget = payload.get("budget", {})
    if budget != {
        "charged_calls_per_task": 1000,
        "charged_calls_total": 3000,
        "initialization_calls_per_task": 16,
        "candidate_calls_per_task": 984,
        "queries_per_round": 16,
        "max_rounds": 64,
        "automatic_retries": 0,
        "backfill": False,
        "cpu_per_worker": 1,
        "gpu": False,
    }:
        raise RuntimeError("scored budget/no-retry contract changed")
    support_envelope = json.loads(SUPPORT.read_text())
    support = support_envelope.get("payload", {})
    if support.get("schema_version") != "pmo_population_live_parent_gate_v1":
        raise RuntimeError("unexpected live-parent support artifact")
    if support.get("support", {}).get("supported_parent_count", 0) < 4:
        raise RuntimeError("live-parent support floor failed")
    manifest = json.loads(MANIFEST.read_text())
    for relative, expected in manifest["source_files"].items():
        path = ROOT / relative
        if not path.exists() or _sha(path) != expected:
            raise RuntimeError(f"source capsule identity mismatch: {relative}")
    if not subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip().startswith("b37900c2"):
        raise RuntimeError("PMO-v1 implementation commit changed")
    receipt = {
        "schema_version": "pmo_population_scored_preflight_receipt_v1",
        "status": "PASS_NO_ORACLE_CALLS",
        "contract_payload_sha256": PAYLOAD,
        "contract_file_sha256": _sha(CONTRACT),
        "authorization_receipt_sha256": _sha(AUTH),
        "source_capsule_manifest_sha256": _sha(MANIFEST),
        "support_scheduler_result_sha256": _sha(SUPPORT),
        "code_revision": "b37900c2",
        "tasks": list(TASKS),
        "charged_calls_total": 3000,
        "automatic_retries": 0,
        "backfill": False,
        "modal_source_upload_authorized": True,
        "oracle_calls_before_launch": 0,
        "modal_calls_before_launch": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write(PREFLIGHT, {**receipt, "receipt_sha256": _identity(receipt)})
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def launch() -> dict:
    if not PREFLIGHT.exists():
        raise RuntimeError("run preflight first")
    if LAUNCH.exists():
        raise RuntimeError("launch receipt exists; no duplicate launch")
    pre = json.loads(PREFLIGHT.read_text())
    if pre.get("status") != "PASS_NO_ORACLE_CALLS":
        raise RuntimeError("preflight did not pass")
    from modal import Function

    run_id = _identity({
        "schema_version": "pmo_population_scored_launch_v1",
        "contract_payload_sha256": PAYLOAD,
        "preflight_sha256": _sha(PREFLIGHT),
        "source_capsule_manifest_sha256": _sha(MANIFEST),
        "code_revision": "b37900c2",
    })
    receipt = {
        "schema_version": "pmo_population_scored_launch_v1",
        "run_id": run_id,
        "app": APP,
        "contract_payload_sha256": PAYLOAD,
        "preflight_sha256": _sha(PREFLIGHT),
        "tasks": list(TASKS),
        "automatic_retries": 0,
        "backfill": False,
        "calls_per_task": 1000,
        "status": "SPAWNING",
        "calls": {},
    }
    _write(LAUNCH, receipt)
    function = Function.from_name(APP, "run_task")
    for task in TASKS:
        call = function.spawn({
            "run_id": run_id,
            "task": task,
            "contract_payload_sha256": PAYLOAD,
        })
        receipt["calls"][task] = {"call_id": call.object_id, "status": "SPAWNED"}
        _write(LAUNCH, receipt)
        print(task, call.object_id, flush=True)
    receipt["status"] = "SPAWNED_ALL"
    _write(LAUNCH, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def status() -> None:
    if not LAUNCH.exists():
        raise RuntimeError("no launch receipt")
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
