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
# Superseded receipts are archived under the payload hash they belonged to.  A fixed
# directory name could not survive a second supersession: re-running this tool would
# have overwritten the archived 1000-call receipts with the 250-call ones and erased
# the record of the earlier attempt.
SUPERSEDED = ROOT / "diagnostics/pmo_population_controller_v1/superseded"
CREDIT = "src/compose_v4/control/pmo_credit.py"
CONTINUITY = "src/compose_v4/control/bootstrap_pool_continuity.py"
ORACLE_ASSETS = "src/compose_v4/experiments/pmo_oracle_assets.py"

DEFAULT_REASON = (
    "The 1000-call payload was authorized but its worker image omitted torch and "
    "every task died at zero charged calls.  The image is fixed and proven by a "
    "zero-oracle-call remote smoke.  This payload reduces the budget to 250 calls "
    "per task so the three diagnostic curves are read before a larger spend."
)

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
    parser.add_argument(
        "--reason",
        default=DEFAULT_REASON,
        help="why this payload supersedes the previous one; it is recorded verbatim "
             "in the authorization receipt and is part of the scientific record",
    )
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
    # The capsule recorded the payload it was built for, but nothing updated that
    # field, so it addressed a payload that no longer existed -- a pin resolving to
    # nothing, which is worse than no pin because it reads as verified.
    manifest["contract_payload_sha256"] = new_payload_sha256
    source_files = manifest["source_files"]
    for dependency in (CREDIT, CONTINUITY, ORACLE_ASSETS):
        source_files.setdefault(dependency, "")
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

    # Name the archive after the payload each RECEIPT itself carries, not after the
    # contract's current hash.  Those differ whenever the contract was re-sealed
    # before this tool ran, and filing a receipt under a payload it never described
    # is a pointer that resolves to the wrong thing.
    archives: list[Path] = []
    for receipt, name in ((LAUNCH, "corrected_scored_launch_receipt.json"),
                          (PREFLIGHT, "corrected_preflight_receipt.json")):
        if not receipt.exists():
            continue
        belongs_to = json.loads(receipt.read_text()).get(
            "payload_sha256", previous_payload_sha256)
        archive = SUPERSEDED / f"payload_{belongs_to[:16]}"
        archive.mkdir(parents=True, exist_ok=True)
        destination = archive / name
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite an archived receipt: {destination}"
            )
        destination.write_text(receipt.read_text())
        receipt.unlink()
        archives.append(archive)

    write(CONTRACT, envelope, compact=False)

    authorization = json.loads(AUTH.read_text())
    authorization.update({
        "status": "AUTHORIZED",
        "supersedes_payload_sha256": previous_payload_sha256,
        "payload_sha256": new_payload_sha256,
        "oracle_calls_authorized": BUDGET["charged_calls_total"],
        "reason_for_new_payload": args.reason,
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    write(AUTH, authorization, compact=False)

    # The capsule pins BOTH of the files just rewritten, so it has to be written last
    # and re-hash both.  Writing the authorization receipt after the manifest left the
    # capsule pinning a receipt that no longer existed -- and since `authorized_at_utc`
    # moves on every run, the launcher's capsule check could never pass.
    for pinned in (CONTRACT, AUTH):
        source_files[str(pinned.relative_to(ROOT))] = sha(pinned)
    write(MANIFEST, manifest, compact=True)

    print(json.dumps({
        "previous_payload_sha256": previous_payload_sha256,
        "new_payload_sha256": new_payload_sha256,
        "charged_calls_total": BUDGET["charged_calls_total"],
        "manifest_files": len(source_files),
        "archived": sorted({str(a.relative_to(ROOT)) for a in archives}),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
