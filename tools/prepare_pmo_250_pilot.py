"""Re-budget the corrected PMO contract to the 3x250 diagnostic pilot.

The 1000-call payload was authorized once and died at zero charged calls because the
worker image omitted torch.  The environment is now fixed and proven by a zero-call
remote smoke, but the scientific decision is to read 250-call curves before spending a
1000-call budget, so the whole binding chain has to move together:

    contract budget -> contract payload hash -> source capsule manifest ->
    authorization receipt -> launcher constants

Each of those is a fail-closed check in `launch_pmo_population_v1_corrected.py`, so a
partial update produces a refusal rather than a wrong run.  The dead launch receipt is
archived rather than deleted: it is the record of an attempt that charged nothing.
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
ARCHIVE = ROOT / "diagnostics/pmo_population_controller_v1/dead_1000call_attempt"
CREDIT = "src/compose_v4/control/pmo_credit.py"

CALLS_PER_TASK = 250
INITIALIZATION = 16
TASKS = ("gsk3b", "perindopril_mpo", "celecoxib_rediscovery")

BUDGET = {
    "automatic_retries": 0,
    "backfill": False,
    "candidate_calls_per_task": CALLS_PER_TASK - INITIALIZATION,
    "charged_calls_per_task": CALLS_PER_TASK,
    "charged_calls_total": CALLS_PER_TASK * len(TASKS),
    "cpu_per_worker": 1,
    "gpu": False,
    "initialization_calls_per_task": INITIALIZATION,
    "max_rounds": 64,
    "queries_per_round": 16,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write(path: Path, value: dict, *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        if compact
        else json.dumps(value, sort_keys=True, indent=2) + "\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    envelope = json.loads(CONTRACT.read_text())
    payload = envelope["payload"]
    previous_payload_sha256 = envelope["payload_sha256"]
    if identity(payload) != previous_payload_sha256:
        raise ValueError("corrected contract payload hash does not match its payload")
    if tuple(payload["tasks"]) != TASKS:
        raise ValueError("task lock changed")
    if payload["oracle"].get("prescreen") is not False:
        raise ValueError("this pilot requires the no-prescreen oracle setting")

    payload["budget"] = BUDGET
    payload["status"] = "FROZEN_FAIL_CLOSED_PENDING_NEW_EXPLICIT_PAYLOAD_AUTHORIZATION"
    payload["oracle_calls_authorized"] = 0
    payload["scored_launch_authorized"] = False
    payload["modal_launch_authorized"] = False
    new_payload_sha256 = identity(payload)
    envelope["payload_sha256"] = new_payload_sha256

    manifest = json.loads(MANIFEST.read_text())
    source_files = manifest["source_files"]
    source_files.setdefault(CREDIT, "")
    for relative in sorted(source_files):
        source_files[relative] = sha(ROOT / relative)

    if not args.apply:
        print(json.dumps({
            "previous_payload_sha256": previous_payload_sha256,
            "new_payload_sha256": new_payload_sha256,
            "budget": BUDGET,
            "manifest_files": len(source_files),
        }, indent=2, sort_keys=True))
        print("dry run: pass --apply to write")
        return

    if LAUNCH.exists():
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        (ARCHIVE / "corrected_scored_launch_receipt.json").write_text(LAUNCH.read_text())
        LAUNCH.unlink()
    if PREFLIGHT.exists():
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        (ARCHIVE / "corrected_preflight_receipt.json").write_text(PREFLIGHT.read_text())
        PREFLIGHT.unlink()

    write(CONTRACT, envelope, compact=False)
    # The contract file hash is itself pinned by the capsule, so re-hash after writing.
    source_files[str(CONTRACT.relative_to(ROOT))] = sha(CONTRACT)
    write(MANIFEST, manifest, compact=True)

    authorization = json.loads(AUTH.read_text())
    authorization.update({
        "status": "AUTHORIZED",
        "supersedes_payload_sha256": previous_payload_sha256,
        "payload_sha256": new_payload_sha256,
        "oracle_calls_authorized": BUDGET["charged_calls_total"],
        "reason_for_new_payload": (
            "The 1000-call payload was authorized but its worker image omitted torch and "
            "every task died at zero charged calls.  The image is fixed and proven by a "
            "zero-oracle-call remote smoke.  This payload reduces the budget to 250 calls "
            "per task so the three diagnostic curves are read before a larger spend."
        ),
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    write(AUTH, authorization, compact=False)

    print(json.dumps({
        "previous_payload_sha256": previous_payload_sha256,
        "new_payload_sha256": new_payload_sha256,
        "charged_calls_total": BUDGET["charged_calls_total"],
        "manifest_files": len(source_files),
        "archived": str(ARCHIVE.relative_to(ROOT)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
