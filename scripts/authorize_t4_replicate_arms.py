"""Record the owner's launch authorization and flip the ten replicate arms.

THE PROBLEM THIS SOLVES
-----------------------
`scored_launch_authorized` and `modal_launch_authorized` live INSIDE the payload,
so the act of authorizing necessarily moves the payload hash off the value that
was authorized.  Left implicit, that is indistinguishable from re-sealing a
contract onto a hash nobody approved -- the exact failure this repository has
already paid for.

So each arm records the hash it was authorized AS, plus a reconstruction rule,
and `tests/test_t4_replicate_authorization.py` REBUILDS the pre-authorization
payload and requires it to hash to that value.  "These are the bytes the owner
approved" is then checkable rather than asserted.

AUTHORIZATION PROVENANCE, STATED PLAINLY
----------------------------------------
The authorization reached this agent RELAYED: the coordinating agent quoted the
owner's words and named the ten payload hashes.  It was not a direct owner
signature on this artifact, and the receipt says so rather than implying a
stronger provenance than exists.  Every hash was re-verified against the sealed
files before anything was flipped.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.control.docking_value import identity

#: The ten payload hash prefixes the owner authorized, as relayed.
AUTHORIZED = {
    "5ht1b_d04": "195f8514",
    "5ht1b_d06": "15a615eb",
    "braf_d04": "fea9feb1",
    "braf_d06": "770f40b2",
    "fa7_d04": "a933ff36",
    "fa7_d06": "9d6e8a4e",
    "jak2_d04": "03f74cca",
    "jak2_d06": "a5263844",
    "parp1_d04": "76225d63",
    "parp1_d06": "9210ba34",
}

PENDING_STATUS = "FROZEN_PENDING_OWNER_LAUNCH_AUTHORIZATION"
AUTHORIZED_STATUS = "AUTHORIZED_FOR_SCORED_LAUNCH"
RECEIPT = "diagnostics/t4_replicate_launch_authorization/authorization_v1.json"


def contract_path(arm: str) -> Path:
    return ROOT / f"configs/t4_unified_controller_{arm}_r23_v1.json"


def reconstruct_unauthorized(payload: dict) -> dict:
    """Rebuild the payload as it stood when the owner authorized it.

    Exactly three things move when an arm is authorized, and this undoes all
    three. If a future edit adds a fourth, the reconstruction test fails rather
    than silently certifying a payload the owner never saw.
    """

    rebuilt = {k: v for k, v in payload.items() if k != "authorization"}
    rebuilt["scored_launch_authorized"] = False
    rebuilt["modal_launch_authorized"] = False
    rebuilt["status"] = PENDING_STATUS
    return rebuilt


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()

    # ---- Verify BEFORE mutating. A moved hash is a stop, not a re-seal. ----
    sealed = {}
    for arm, prefix in sorted(AUTHORIZED.items()):
        envelope = json.loads(contract_path(arm).read_text())
        payload, stored = envelope["payload"], envelope["payload_sha256"]
        if identity(payload) != stored:
            print(f"STOP: {arm} self-hash does not verify")
            return 1
        if not stored.startswith(prefix):
            print(f"STOP: {arm} is {stored[:8]}, authorized was {prefix}")
            return 1
        if payload.get("scored_launch_authorized") is not False:
            print(f"STOP: {arm} is already authorized; refusing to re-authorize")
            return 1
        sealed[arm] = (payload, stored)

    receipt = {
        "schema_version": "t4_replicate_launch_authorization_v1",
        "authorized_at_utc": now,
        "budget": {
            "arms": len(sealed),
            "cells": sum(len(p["cells"]) for p, _ in sealed.values()),
            "charged_calls_per_cell": 250,
            "total_charged_call_ceiling": sum(
                p["total_charged_call_ceiling"] for p, _ in sealed.values()
            ),
        },
        "provenance": {
            "owner_words": "yeah i approve but i want the whole suite right please and thank u",
            "reached_this_agent": "RELAYED by the coordinating agent, which quoted the owner and named the ten payload hashes",
            "not": "a direct owner signature on this artifact",
            "verification_before_flip": "all ten payload hashes re-verified against the sealed files, self-hash included",
        },
        "coverage_confirmed_to_the_owner": {
            "question": "is ten arms the whole suite?",
            "answer": True,
            "measured": {
                "targets": 5,
                "starting_molecules_per_target": 3,
                "deltas": [0.4, 0.6],
                "molecule_x_delta_groups": 30,
                "new_replicates_per_group": 2,
                "runs": 60,
                "groups_with_two_distinct_seeds": 30,
                "groups_colliding_with_replicate_1": 0,
                "canonical_cells_covered": "15/15",
                "every_smiles_matches_the_verified_inventory": True,
            },
        },
        "authorized_arms": {
            arm: {
                "contract": str(contract_path(arm).relative_to(ROOT)),
                "authorized_payload_sha256": stored,
                "cells": len(payload["cells"]),
                "delta": payload["delta"],
                "total_charged_call_ceiling": payload["total_charged_call_ceiling"],
            }
            for arm, (payload, stored) in sorted(sealed.items())
        },
    }
    destination = ROOT / RECEIPT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    for arm, (payload, stored) in sorted(sealed.items()):
        payload["scored_launch_authorized"] = True
        payload["modal_launch_authorized"] = True
        payload["status"] = AUTHORIZED_STATUS
        payload["authorization"] = {
            "authorized_payload_sha256": stored,
            "authorized_at_utc": now,
            "receipt": RECEIPT,
            "provenance": "relayed owner authorization; see the receipt",
            "reconstruction_rule": (
                "delete `authorization`, set scored_launch_authorized and "
                "modal_launch_authorized to false, and set status to "
                f"{PENDING_STATUS!r}; the result must hash to "
                "`authorized_payload_sha256`"
            ),
        }
        rebuilt = reconstruct_unauthorized(payload)
        if identity(rebuilt) != stored:
            print(f"STOP: {arm} no longer reconstructs to its authorized hash")
            return 1
        contract_path(arm).write_text(
            json.dumps(
                {"payload": payload, "payload_sha256": identity(payload)},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(
            f"  {arm:12s} authorized_as={stored[:8]} now={identity(payload)[:8]} "
            f"reconstructs=OK"
        )

    print(
        f"\n{len(sealed)} arms authorized, "
        f"{receipt['budget']['total_charged_call_ceiling']} charged-call ceiling"
    )
    print(f"receipt: {RECEIPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
