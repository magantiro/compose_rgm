"""Record the launch authorization for the two revival arms and flip them.

WHY THIS IS A SEPARATE ACT FROM SEALING
---------------------------------------
`scored_launch_authorized` and `modal_launch_authorized` live INSIDE the payload,
so authorizing necessarily moves the payload hash off the value that was
authorized.  Each arm therefore records the hash it was authorized AS plus a
reconstruction rule, so "these are the bytes that were approved" is checkable
rather than asserted.

AUTHORIZATION PROVENANCE, STATED PLAINLY
----------------------------------------
Direct owner instruction in session, quoted verbatim in the receipt: the owner
was shown that these four cells are permanently unable to return a molecule on
the original workspace and asked for the second-workspace expansion to start.
This is a DIRECT instruction, not a relayed one -- unlike the ten-arm replicate
authorization, whose receipt correctly records that it reached the agent relayed.

SCOPE OF WHAT IS BEING AUTHORIZED
---------------------------------
996 charged calls: four cells at 250 each, less the one root call each already
charged on the first workspace.  Nothing else in the panel is touched.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.control.docking_value import identity

#: The pre-authorization payload hash prefixes being authorized.
AUTHORIZED = {
    "fa7_d04": "a2f50528",
    "braf_d04": "ab1257e1",
}

RECEIPT = "diagnostics/t4_revival_launch_authorization/authorization_v1.json"
PENDING_STATUS = "FROZEN_PENDING_OWNER_LAUNCH_AUTHORIZATION"
AUTHORIZED_STATUS = "AUTHORIZED_FOR_SCORED_LAUNCH"
RECONSTRUCTION = (
    "delete `authorization`, set scored_launch_authorized and "
    "modal_launch_authorized to false, and set status to "
    f"'{PENDING_STATUS}'; the result must hash to `authorized_payload_sha256`"
)


def _path(arm: str) -> Path:
    target, tag = arm.rsplit("_", 1)
    return ROOT / f"configs/t4_unified_controller_{target}_{tag}_rev_v1.json"


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    arms = {}
    total = 0
    for arm, prefix in sorted(AUTHORIZED.items()):
        path = _path(arm)
        document = json.loads(path.read_text())
        payload = document["payload"]
        stored = document["payload_sha256"]
        if identity(payload) != stored:
            raise SystemExit(f"{path.name}: stored payload hash does not match its bytes")
        if not stored.startswith(prefix):
            raise SystemExit(f"{path.name}: hash {stored[:8]} is not the authorized {prefix}")
        if payload["status"] != PENDING_STATUS:
            raise SystemExit(f"{path.name}: already {payload['status']}, refusing to re-authorize")
        arms[arm] = {
            "contract": str(path.relative_to(ROOT)),
            "authorized_payload_sha256": stored,
            "cells": [row["cell"] for row in payload["cells"]],
            "total_charged_call_ceiling": payload["total_charged_call_ceiling"],
        }
        total += payload["total_charged_call_ceiling"]

    receipt = {
        "schema_version": "t4_revival_launch_authorization_v1",
        "authorized_at_utc": now,
        "provenance": "direct owner instruction in session",
        "owner_words": "ok yeah why don't u start the nitya expansion for now",
        "context_shown_to_owner": (
            "that fa7_1_r3, fa7_2_r2, fa7_2_r3 and braf_2_r3 died in the round-0 "
            "preemption window and can never return a molecule on rahul-94866, "
            "because _resume_state raises on an unfinished query lock with no "
            "recoverable checkpoint and retries is 0"
        ),
        "workspace": "nitya",
        "arms": arms,
        "budget": {
            "total_charged_call_ceiling": total,
            "cells": sum(len(a["cells"]) for a in arms.values()),
            "per_cell_authorized": 250,
            "per_cell_debited_on_rahul": 1,
            "rule": "lifetime spend per revived cell is 1 (rahul) + 249 (nitya) = 250",
        },
        "reconstruction_rule": RECONSTRUCTION,
        "not_authorized": (
            "the fourteen cells still QUEUED on rahul-94866 are NOT revived; their "
            "run_cell calls are live and a second writer would produce a divergent "
            "duplicate and double-charge the cell"
        ),
    }
    destination = ROOT / RECEIPT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    for arm in sorted(AUTHORIZED):
        path = _path(arm)
        document = json.loads(path.read_text())
        payload = document["payload"]
        stored = document["payload_sha256"]
        payload["scored_launch_authorized"] = True
        payload["modal_launch_authorized"] = True
        payload["status"] = AUTHORIZED_STATUS
        payload["authorization"] = {
            "authorized_payload_sha256": stored,
            "authorized_at_utc": now,
            "receipt": RECEIPT,
            "provenance": "direct owner instruction in session; see the receipt",
            "reconstruction_rule": RECONSTRUCTION,
        }
        path.write_text(
            json.dumps({"payload": payload, "payload_sha256": identity(payload)}, indent=2, sort_keys=True)
            + "\n"
        )
        print(f"{path.name:52s} {stored[:8]} -> {identity(payload)[:8]}  {AUTHORIZED_STATUS}")
    print(f"authorized {total} charged calls across {sum(len(a['cells']) for a in arms.values())} cells")
    print(f"receipt: {RECEIPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
