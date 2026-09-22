"""Launch, resume, preflight and monitor the canonical T4 experiment.

Cells are spawned as SEPARATE calls against the DEPLOYED app, never as one ``.map``
inside a driver. Per-cell independence is the whole point: one cell can fail, be
preempted or be relaunched without any other cell noticing, and ``--mode resume``
re-spawns only the cells with no ``result.json``.

    modal deploy modal_apps/t4_canonical_shared_controller_app.py
    python3 tools/launch_t4_canonical.py --mode verify
    python3 tools/launch_t4_canonical.py --mode preflight
    python3 tools/launch_t4_canonical.py --mode launch --arms A,B,C
    python3 tools/launch_t4_canonical.py --mode status  --run-id <id>
    python3 tools/launch_t4_canonical.py --mode resume  --run-id <id>

`verify` is the gate, and it runs inside `launch` too: it re-hashes every pinned
runtime input, refuses a dirty or untracked pinned file, refuses a contract whose
status is not authorized, and refuses any target-name routing in the runtime closure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.control.docking_value import identity  # noqa: E402
from compose_v4.experiments.continuation_profile import sha256_file  # noqa: E402
from compose_v4.experiments.t4_canonical_controller import (  # noqa: E402
    ARMS,
    assert_no_target_name_routing,
    controller_identity,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal  # noqa: E402

APP_NAME = "compose-t4-canonical-shared-controller"
CONTRACTS = {arm: f"configs/t4_canonical_shared_controller_{arm.lower()}_v1.json" for arm in ARMS}
RECEIPTS = ROOT / "diagnostics/t4_canonical_shared_controller_v1/launches"
AUTHORIZED = "AUTHORIZED_T4_CANONICAL_SHARED_CONTROLLER_15000_CALLS"


def _contracts() -> dict[str, dict]:
    return {arm: unseal(ROOT / path) for arm, path in CONTRACTS.items()}


def verify(*, require_authorized: bool) -> dict:
    """Everything that must be true before a single oracle call is charged."""

    contracts = _contracts()
    findings: list[str] = []

    identities = {arm: controller_identity(payload) for arm, payload in contracts.items()}
    if len(set(identities.values())) != 1:
        findings.append(f"arms disagree on the shared controller: {identities}")

    pinned: set[str] = set()
    for arm, payload in contracts.items():
        if require_authorized and payload["status"] != AUTHORIZED:
            findings.append(f"arm {arm} status is {payload['status']!r}, not authorized")
        for relative, expected in payload["runtime_inputs_sha256"].items():
            pinned.add(relative)
            actual = sha256_file(ROOT / relative)
            if actual != expected:
                findings.append(f"arm {arm}: runtime input {relative} moved -> {actual}")

    paths = sorted(pinned | set(CONTRACTS.values()))
    dirty = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *paths], cwd=ROOT, text=True,
        capture_output=True,
    ).stdout.strip()
    if dirty:
        findings.append(f"pinned runtime inputs are dirty: {dirty.splitlines()}")
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *paths], cwd=ROOT, text=True
    ).strip()
    if untracked:
        findings.append(f"untracked pinned runtime inputs: {untracked.splitlines()}")

    try:
        scanned = assert_no_target_name_routing(
            [ROOT / relative for relative in sorted(pinned) if relative.endswith(".py")]
        )
    except AssertionError as error:
        findings.append(str(error))
        scanned = []

    cells = {}
    for arm, payload in contracts.items():
        for cell in payload["cells"]:
            previous = cells.setdefault(cell["cell"], cell)
            if previous != cell:
                findings.append(f"cell {cell['cell']} differs between arms")

    total = sum(payload["total_charged_call_ceiling"] for payload in contracts.values())
    report = {
        "controller_identity_sha256": identities["A"],
        "arms": {
            arm: {
                "payload_sha256": identity(payload),
                "cells": len(payload["cells"]),
                "experts": payload["experts"],
                "adaptive": payload["adaptive_support_expansion"],
                "ceiling": payload["total_charged_call_ceiling"],
                "status": payload["status"],
            }
            for arm, payload in sorted(contracts.items())
        },
        "total_charged_call_ceiling": total,
        "pinned_runtime_inputs": len(pinned),
        "target_name_routing_scanned_modules": len(scanned),
        "findings": findings,
        "verdict": "PASS" if not findings else "FAIL",
    }
    return report


def _run_id(contracts: dict[str, dict]) -> str:
    body = {
        "schema_version": "t4_canonical_launch_v1",
        "contracts": {arm: identity(payload) for arm, payload in sorted(contracts.items())},
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "automatic_docking_retries": 0,
    }
    return identity(body)


def _tasks(contracts: dict[str, dict], run_id: str, arms: list[str]) -> list[dict]:
    tasks = []
    for arm in arms:
        payload = contracts[arm]
        for cell in payload["cells"]:
            tasks.append(
                {
                    "schema_version": "t4_canonical_cell_task_v1",
                    "run_id": run_id,
                    "arm": arm,
                    "cell": cell["cell"],
                    "contract_payload_sha256": identity(payload),
                }
            )
    return tasks


def _function(name: str):
    import modal

    return modal.Function.from_name(APP_NAME, name)


def _completed(run_id: str) -> set[tuple[str, str]]:
    """Every (arm, cell) whose result.json already exists, read off the volume.

    ONE recursive listing, with backoff. A per-directory walk is ~60 calls and trips
    `VolumeListFiles rate limit exceeded`, which would abort a resume rather than
    merely slow it -- and a resume that aborts halfway is worse than one that waits.
    """

    import time

    import modal

    volume = modal.Volume.from_name(APP_NAME)
    for attempt in range(6):
        try:
            entries = list(volume.listdir(run_id, recursive=True))
            break
        except FileNotFoundError:
            return set()
        except Exception as error:  # noqa: BLE001 - rate limit or transient
            if attempt == 5:
                raise SystemExit(f"could not list the run volume: {error!r}") from error
            time.sleep(15 * (attempt + 1))
    done: set[tuple[str, str]] = set()
    for entry in entries:
        parts = entry.path.split("/")
        if len(parts) >= 4 and parts[-1] == "result.json":
            done.add((parts[-3], parts[-2]))
    return done


def _live(run_id: str) -> set[tuple[str, str]]:
    """Every (arm, cell) whose most recent spawned call has NOT terminated.

    Without this, `--mode resume` would re-spawn a cell that is merely SLOW, because a
    running cell has no `result.json`. Two containers on one cell would write the same
    round locks and the same checkpoint from two different search states -- the exact
    "never resolve a conflict by blind min/max" corruption the round-lock design exists
    to prevent, and it would be invisible until reconciliation.

    Liveness is established by GETTING the call, not by reading a task count: a listing
    showing zero tasks has misled a diagnosis in this repository before. A call that has
    terminated -- returned, raised, or been cancelled -- is re-spawnable; one that is
    still pending or running is not.
    """

    import modal

    latest: dict[tuple[str, str], str] = {}
    for path in sorted(RECEIPTS.glob("*.json")) if RECEIPTS.exists() else []:
        try:
            payload = unseal(path)
        except Exception:  # noqa: BLE001 - a malformed receipt must not block a resume
            continue
        if payload.get("run_id") != run_id:
            continue
        for row in payload.get("spawned") or []:
            latest[(row["arm"], row["cell"])] = row["function_call_id"]

    # MEASURED: `get(timeout=0)` raises `TimeoutError` for a CANCELLED call exactly as it
    # does for a running one, so liveness cannot be read from the call alone. A deliberate
    # cancellation is therefore recorded as a sealed receipt and subtracted here -- an
    # auditable fact rather than an inference. Any other terminal state (returned, raised,
    # retries exhausted) does surface through `get`, because the output or the exception
    # is there to be fetched.
    cancelled: set[str] = set()
    for path in sorted(RECEIPTS.glob("*_cancel_*.json")) if RECEIPTS.exists() else []:
        try:
            payload = unseal(path)
        except Exception:  # noqa: BLE001
            continue
        if payload.get("run_id") != run_id:
            continue
        for row in payload.get("cancelled") or []:
            cancelled.add(row["function_call_id"])

    live: set[tuple[str, str]] = set()
    for key, call_id in latest.items():
        if call_id in cancelled:
            continue
        try:
            modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            live.add(key)
        except Exception:  # noqa: BLE001 - terminated one way or another; re-spawnable
            continue
    return live


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("verify", "preflight", "launch", "resume", "status"), required=True
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--arms", default="A,B,C")
    parser.add_argument("--allow-draft", action="store_true",
                        help="preflight only: run against draft contracts")
    parser.add_argument("--dry-run", action="store_true",
                        help="resume only: print what would be re-spawned and exit")
    args = parser.parse_args()
    arms = [value.strip().upper() for value in args.arms.split(",") if value.strip()]
    contracts = _contracts()

    if args.mode == "verify":
        report = verify(require_authorized=not args.allow_draft)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["verdict"] == "PASS" else 1

    if args.mode == "preflight":
        report = verify(require_authorized=not args.allow_draft)
        if report["verdict"] != "PASS" and not args.allow_draft:
            print(json.dumps(report, indent=2, sort_keys=True))
            return 1
        run_id = args.run_id or _run_id(contracts)
        requests = []
        for arm in arms:
            payload = contracts[arm]
            for cell in payload["cells"]:
                for expert in payload["experts"]:
                    requests.append(
                        {
                            "schema_version": "t4_canonical_cell_task_v1",
                            "run_id": run_id,
                            "arm": arm,
                            "cell": cell["cell"],
                            "expert": expert,
                            "contract_payload_sha256": identity(payload),
                        }
                    )
        print(f"preflight: {len(requests)} (arm, cell, lane) probes, zero oracle calls",
              flush=True)
        started = time.time()
        rows = list(
            _function("preflight_lane").map(requests, order_outputs=False, return_exceptions=True)
        )
        ok = [row for row in rows if not isinstance(row, Exception)]
        failed = [repr(row) for row in rows if isinstance(row, Exception)]
        destination = ROOT / "diagnostics/t4_canonical_shared_controller_v1/preflight_v1.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        seal(
            destination,
            {
                "schema_version": "t4_canonical_preflight_v1",
                "run_id": run_id,
                "verify": report,
                "requested": len(requests),
                "succeeded": len(ok),
                "failed": failed,
                "wall_seconds": round(time.time() - started, 1),
                "rows": sorted(ok, key=lambda row: (row["arm"], row["cell"], row["expert"])),
            },
        )
        print(f"wrote {destination} ({len(ok)}/{len(requests)} ok, "
              f"{round(time.time() - started, 1)}s)")
        return 0 if not failed else 1

    if args.mode == "status":
        if not args.run_id:
            raise SystemExit("--mode status requires --run-id")
        answer = _function("remote_status").remote({"run_id": args.run_id})
        print(json.dumps(answer, indent=2, sort_keys=True))
        return 0

    # ---- launch / resume ----
    report = verify(require_authorized=True)
    if report["verdict"] != "PASS":
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit("verification failed; refusing to charge the oracle")

    run_id = args.run_id or _run_id(contracts)
    if args.mode == "resume" and not args.run_id:
        raise SystemExit("--mode resume requires the prior --run-id")

    tasks = _tasks(contracts, run_id, arms)
    if args.mode == "resume":
        done = _completed(run_id)
        live = _live(run_id) - done
        before = len(tasks)
        tasks = [
            task
            for task in tasks
            if (task["arm"], task["cell"]) not in done
            and (task["arm"], task["cell"]) not in live
        ]
        print(
            f"resume: {len(done)} cells already final, {len(live)} still live and left "
            f"alone, re-spawning {len(tasks)} of {before}"
        )
        if args.dry_run:
            for task in tasks:
                print(f"  would spawn {task['arm']}/{task['cell']}")
            return 0

    run_cell = _function("run_cell")
    receipts = []
    for task in tasks:
        call = run_cell.spawn(task)
        receipts.append({**task, "function_call_id": call.object_id})
        print(f"spawned {task['arm']}/{task['cell']} -> {call.object_id}", flush=True)

    RECEIPTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    seal(
        RECEIPTS / f"{run_id}_{args.mode}_{stamp}.json",
        {
            "schema_version": "t4_canonical_launch_receipt_v1",
            "mode": args.mode,
            "run_id": run_id,
            "arms": arms,
            "spawned": receipts,
            "verify": report,
            "launched_at_utc": stamp,
        },
    )
    print(json.dumps({"run_id": run_id, "spawned": len(receipts), "arms": arms}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
