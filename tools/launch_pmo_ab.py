#!/usr/bin/env python3
"""Launch the matched PMO A/B: allocator-repaired baseline vs the same plus online memory.

The two arms are separately sealed payloads whose only top-level difference is
``arm.enable_online_memory``.  The worker resolves its contract from the arm name against
a two-entry whitelist and refuses any payload the A/B receipt does not bind, so this
launcher cannot select a runtime the owner did not authorize.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = "compose-pmo-population-v1-corrected"
RECEIPT = ROOT / "diagnostics/pmo_ab_scored_authorization_v1.json"
LAUNCH = ROOT / "diagnostics/pmo_ab_launch_receipt_v1.json"
ARMS = {
    "A_baseline": "configs/pmo_ab_a_baseline_contract_v1.json",
    "B_memory": "configs/pmo_ab_b_memory_contract_v1.json",
}


def _identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _validate() -> tuple[dict, dict]:
    receipt = json.loads(RECEIPT.read_text())
    if receipt.get("status") != "AUTHORIZED":
        raise RuntimeError("A/B authorization receipt is not AUTHORIZED")
    envelopes = {}
    for arm, rel in ARMS.items():
        env = json.loads((ROOT / rel).read_text())
        if _identity(env["payload"]) != env["payload_sha256"]:
            raise RuntimeError(f"{arm}: payload does not match its own seal")
        if receipt["arms"][arm]["payload_sha256"] != env["payload_sha256"]:
            raise RuntimeError(f"{arm}: receipt does not bind this payload")
        if env["payload"]["arm"]["name"] != arm:
            raise RuntimeError(f"{arm}: payload does not declare its own arm")
        envelopes[arm] = env
    a, b = envelopes["A_baseline"]["payload"], envelopes["B_memory"]["payload"]
    differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if differing != ["arm"]:
        raise RuntimeError(f"arms are NOT matched; they differ in {differing}")
    if a["budget"] != b["budget"] or a["tasks"] != b["tasks"]:
        raise RuntimeError("arms disagree on budget or tasks")
    if a["implementation_sha256"] != b["implementation_sha256"]:
        raise RuntimeError("arms disagree on implementation pins")
    return receipt, envelopes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "launch"))
    args = parser.parse_args()
    _receipt, envelopes = _validate()
    tasks = envelopes["A_baseline"]["payload"]["tasks"]
    per_task = envelopes["A_baseline"]["payload"]["budget"]["charged_calls_per_task"]
    plan = {
        "schema_version": "pmo_ab_launch_receipt_v1",
        "app": APP,
        "arms": {a: envelopes[a]["payload_sha256"] for a in ARMS},
        "tasks": list(tasks),
        "charged_calls_per_task": per_task,
        "charged_calls_total": len(tasks) * per_task * len(ARMS),
        "automatic_retries": 0,
        "matched": "payloads differ in exactly one top-level key: arm",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.command == "preflight":
        print(json.dumps({**plan, "status": "PREFLIGHT_OK_NOTHING_SPAWNED",
                          "oracle_calls_charged": 0}, indent=2, sort_keys=True))
        return
    if LAUNCH.exists():
        raise RuntimeError("A/B launch receipt exists; refusing to duplicate a scored launch")
    from modal import Function

    function = Function.from_name(APP, "run_task")
    plan["calls"] = {}
    for arm, payload in plan["arms"].items():
        for task in tasks:
            run_id = _identity({"arm": arm, "task": task, "payload": payload})
            call = function.spawn(
                {"run_id": run_id, "task": task, "arm": arm,
                 "contract_payload_sha256": payload}
            )
            plan["calls"][f"{arm}/{task}"] = {"call_id": call.object_id, "run_id": run_id}
            print(arm, task, call.object_id, flush=True)
            LAUNCH.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    plan["status"] = "SPAWNED_ALL"
    LAUNCH.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
