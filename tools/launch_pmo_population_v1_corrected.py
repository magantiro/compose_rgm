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
PAYLOAD = "103d9d9273753732fb7fc03207834c88e317c4d1d9bab11b3f3b2f265fc50075"
APP = "compose-pmo-population-v1-corrected"
# The contract's locked task set.  It is what the payload authorizes, and it never
# widens: a launch may spend LESS than its authorization but may never spend more.
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


def _selected(tasks: tuple[str, ...] | None) -> tuple[str, ...]:
    """The subset of the locked task set this launch will actually spawn.

    A re-run forced by a per-task defect should re-run that task, not re-spend the
    budget of the tasks whose readings are still valid.  The subset is recorded in
    both receipts and folded into the run identity, so a one-task run and a
    three-task run under the same payload can never share a run_id.
    """
    if tasks is None:
        return TASKS
    unknown = [task for task in tasks if task not in TASKS]
    if unknown:
        raise RuntimeError(f"tasks outside the locked contract set: {unknown}")
    return tuple(task for task in TASKS if task in set(tasks))


def preflight(tasks: tuple[str, ...] | None = None) -> dict:
    if LAUNCH.exists():
        raise RuntimeError("corrected scored launch receipt already exists; do not duplicate launch")
    payload, auth = _validate_contract_auth()
    selected = _selected(tasks)
    receipt = {
        "schema_version": "pmo_population_corrected_scored_preflight_receipt_v1",
        "status": "PASS_CORRECTED_SOURCE_CONTRACT_AUTHORIZED",
        "contract_payload_sha256": PAYLOAD,
        "payload_sha256": PAYLOAD,
        "contract_file_sha256": _sha(CONTRACT),
        "authorization_receipt_sha256": _sha(AUTH),
        "source_capsule_manifest_sha256": _sha(MANIFEST),
        "worker_sha256": _sha(ROOT / "modal_apps/pmo_population_v1_app.py"),
        "tasks": list(selected),
        "authorized_tasks": list(TASKS),
        "charged_calls_total": (
            len(selected) * payload["budget"]["charged_calls_per_task"]
        ),
        "authorized_charged_calls_total": payload["budget"]["charged_calls_total"],
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


def launch(tasks: tuple[str, ...] | None = None) -> dict:
    if LAUNCH.exists():
        raise RuntimeError("corrected scored launch receipt exists; no duplicate launch")
    payload, _ = _validate()
    selected = _selected(tasks)
    pre = json.loads(PREFLIGHT.read_text())
    if tuple(pre.get("tasks", ())) != selected:
        raise RuntimeError(
            f"preflight covers {pre.get('tasks')}, this launch would spawn "
            f"{list(selected)}; re-run preflight for the tasks being launched"
        )
    from modal import Function

    preflight_id = _sha(PREFLIGHT)
    run_id = _identity({
        "schema_version": "pmo_population_corrected_scored_launch_v1",
        "contract_payload_sha256": PAYLOAD,
        "preflight_sha256": preflight_id,
        "source_capsule_manifest_sha256": _sha(MANIFEST),
        "tasks": list(selected),
    })
    receipt = {
        "schema_version": "pmo_population_corrected_scored_launch_v1",
        "run_id": run_id,
        "app": APP,
        "contract_payload_sha256": PAYLOAD,
        "preflight_sha256": preflight_id,
        "tasks": list(selected),
        "authorized_tasks": list(TASKS),
        "automatic_retries": 0,
        "backfill": False,
        "calls_per_task": payload["budget"]["charged_calls_per_task"],
        "charged_calls_total": (
            len(selected) * payload["budget"]["charged_calls_per_task"]
        ),
        "status": "SPAWNING",
        "calls": {},
    }
    _write(LAUNCH, receipt)
    function = Function.from_name(APP, "run_task")
    for task in selected:
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
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=None,
        help="the subset of the locked task set to launch; the default is all of "
             "them. A subset spends less than the authorization, never more.",
    )
    args = parser.parse_args()
    tasks = tuple(args.tasks) if args.tasks else None
    if args.command == "status":
        status()
    else:
        {"preflight": preflight, "launch": launch}[args.command](tasks)


if __name__ == "__main__":
    main()
