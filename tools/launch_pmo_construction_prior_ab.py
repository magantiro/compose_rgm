#!/usr/bin/env python3
"""Launch the matched construction-prior A/B: deployed B, with and without the prior.

Both arms are separately sealed payloads whose only top-level difference is ``arm``, and
within ``arm`` only ``construction_prior`` plus three descriptive strings.  The worker
resolves its contract from the arm name against a whitelist baked into the image and
refuses any payload the authorization receipt does not bind, so this launcher cannot
select a runtime the owner did not authorize.

It also refuses to spawn unless the checkpoint the treated arm declares is present on
the artifact volume with the declared sha256.  A missing model would otherwise surface
inside the container, and although the worker's own preflight would catch it before the
first charged call, catching it here costs nothing and leaves no half-started task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = "compose-pmo-population-v1-corrected"
VOLUME = "compose-v4-artifacts"
RECEIPT = ROOT / "diagnostics/pmo_construction_prior_ab/scored_authorization_v1.json"
LAUNCH = ROOT / "diagnostics/pmo_construction_prior_ab/launch_receipt_v1.json"
VOLUME_CHECKPOINT = "construction_prior/ringcore_a7546e2_best.pt"
ARMS = {
    "B_cp_off": "configs/pmo_ab_b_construction_prior_off_contract_v1.json",
    "B_cp_on": "configs/pmo_ab_b_construction_prior_on_contract_v1.json",
}


def _identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _validate() -> tuple[dict, dict]:
    receipt = json.loads(RECEIPT.read_text())
    if receipt.get("status") != "AUTHORIZED":
        raise RuntimeError("construction-prior A/B authorization receipt is not AUTHORIZED")
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

    off, on = envelopes["B_cp_off"]["payload"], envelopes["B_cp_on"]["payload"]
    differing = sorted(k for k in set(off) | set(on) if off.get(k) != on.get(k))
    if differing != ["arm"]:
        raise RuntimeError(f"arms are NOT matched; they differ in {differing}")
    if off["budget"] != on["budget"] or off["tasks"] != on["tasks"]:
        raise RuntimeError("arms disagree on budget or tasks")
    if off["implementation_sha256"] != on["implementation_sha256"]:
        raise RuntimeError("arms disagree on implementation pins")
    arm_differing = sorted(
        k for k in set(off["arm"]) | set(on["arm"]) if off["arm"].get(k) != on["arm"].get(k)
    )
    if not set(arm_differing) <= {"construction_prior", "name", "comparison", "what_differs"}:
        raise RuntimeError(f"the arms differ beyond the prior: {arm_differing}")
    if off["arm"].get("construction_prior") is not None:
        raise RuntimeError("the control arm declares a construction prior")
    if on["arm"].get("construction_prior") is None:
        raise RuntimeError("the treated arm declares no construction prior")

    # The pins must still match the files on disk, or the run would execute code the
    # authorization does not cover.
    for rel, digest in off["implementation_sha256"].items():
        observed = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if observed != digest:
            raise RuntimeError(f"{rel}: on-disk sha256 {observed} != pinned {digest}")

    total = len(ARMS) * len(off["tasks"]) * off["budget"]["charged_calls_per_task"]
    authorized = receipt["authorized_oracle_calls"]["total"]
    if total != authorized:
        raise RuntimeError(f"plan charges {total} calls; receipt authorizes {authorized}")
    return receipt, envelopes


def _verify_volume_checkpoint(envelopes: dict) -> dict:
    """The treated arm's model must already be on the volume, with the declared bytes."""

    import modal

    declared = envelopes["B_cp_on"]["payload"]["arm"]["construction_prior"]
    volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
    digest, size = hashlib.sha256(), 0
    for chunk in volume.read_file(VOLUME_CHECKPOINT):
        digest.update(chunk)
        size += len(chunk)
    observed = digest.hexdigest()
    if observed != declared["checkpoint_sha256"]:
        raise RuntimeError(
            f"volume checkpoint sha256 {observed} != declared "
            f"{declared['checkpoint_sha256']}"
        )
    return {"path": VOLUME_CHECKPOINT, "bytes": size, "sha256": observed,
            "status": declared["status"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "launch"))
    args = parser.parse_args()
    receipt, envelopes = _validate()
    payload = envelopes["B_cp_off"]["payload"]
    tasks = payload["tasks"]
    per_task = payload["budget"]["charged_calls_per_task"]
    checkpoint = _verify_volume_checkpoint(envelopes)
    plan = {
        "schema_version": "pmo_construction_prior_ab_launch_receipt_v1",
        "app": APP,
        "arms": {arm: envelopes[arm]["payload_sha256"] for arm in ARMS},
        "tasks": list(tasks),
        "charged_calls_per_task": per_task,
        "charged_calls_total": len(tasks) * per_task * len(ARMS),
        "authorized_charged_calls_total": receipt["authorized_oracle_calls"]["total"],
        "automatic_retries": 0,
        "matched": "payloads differ in exactly one top-level key: arm",
        "volume_checkpoint": checkpoint,
        "predeclared_decision_rule_commit": receipt["predeclared_decision_rule"]["commit"],
        "development_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.command == "preflight":
        print(json.dumps({**plan, "status": "PREFLIGHT_OK_NOTHING_SPAWNED",
                          "oracle_calls_charged": 0}, indent=2, sort_keys=True))
        return
    if LAUNCH.exists():
        raise RuntimeError("launch receipt exists; refusing to duplicate a scored launch")
    from modal import Function

    function = Function.from_name(APP, "run_task")
    plan["calls"] = {}
    for arm, digest in plan["arms"].items():
        for task in tasks:
            run_id = _identity({"arm": arm, "task": task, "payload": digest})
            call = function.spawn(
                {"run_id": run_id, "task": task, "arm": arm,
                 "contract_payload_sha256": digest}
            )
            plan["calls"][f"{arm}/{task}"] = {"call_id": call.object_id, "run_id": run_id}
            print(arm, task, call.object_id, flush=True)
            LAUNCH.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    plan["status"] = "SPAWNED_ALL"
    LAUNCH.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
