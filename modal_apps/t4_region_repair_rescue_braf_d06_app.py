"""Region-repair rescue launcher for braf_0 and braf_1 at delta=0.6, 496 calls total.

These cells exhausted their candidate pool in round zero, each after exactly one charged
call, so the rescue ceiling is the arm's authorized 250 per cell less those measured prior
calls: 248 per cell here.

What makes this a rescue rather than a repeat is one executable field.  The contract sets
`proposal.shallow.region_law` to `free_gate_margin_v1`, which threads the repaired
bridge-separated region-draw law to the draw site.  If that field were absent the run
would take `law=None`, which is v1 byte-identical, and would spend the whole budget
reproducing the exhaustion it exists to fix.  `assert_no_unconsumable_region_law_request`
runs on every proposal worker, so the field cannot be silently parked where nothing reads
it, and `assert_region_law_is_consumed` drives the production path until the draw site
consults a probe.

The contract pins the WIRED app and the two modules implementing the repair, and
`_validate_task` re-hashes every entry, so this cannot run against an unwired tree.

App name, volume and output namespace are distinct from the parent arm and from the other
two rescue arms.  `main` spawns, so a launch needs `modal run --detach`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CONTRACT = "configs/t4_region_repair_rescue_braf_d06_v1.json"
AUTHORIZED_STATUS = "AUTHORIZED_BY_OWNER_FOR_REGION_REPAIR_RESCUE"


def _assert_authorized() -> None:
    """Refuse to construct the arm until the owner has authorized the contract.

    The base app's launch gate verifies identity and tree cleanliness but never reads
    `status`, and the base app is hash-pinned so the check cannot be added there.  It
    lives here instead, at import time, which is before `modal run` can spawn anything.
    """

    payload = json.loads((Path(__file__).resolve().parents[1] / CONTRACT).read_text())["payload"]
    status = payload.get("status")
    if status != AUTHORIZED_STATUS:
        raise RuntimeError(
            f"{CONTRACT} is not authorized for launch (status={status!r}); the owner must set "
            f"status to {AUTHORIZED_STATUS!r} and re-seal the contract first"
        )
    lane = (payload.get("proposal") or {}).get("shallow") or {}
    if lane.get("region_law") != "free_gate_margin_v1":
        raise RuntimeError(
            "proposal.shallow.region_law is not free_gate_margin_v1; without it this run "
            "would be v1 verbatim and would waste the whole rescue budget"
        )


_assert_authorized()

os.environ.setdefault("COMPOSE_HELD_CONTRACT", CONTRACT)
os.environ.setdefault(
    "COMPOSE_HELD_CHECKPOINT",
    "diagnostics/t4_held_target_distillation_quality_v1/braf_checkpoint.json",
)
os.environ.setdefault("COMPOSE_HELD_VOLUME", "compose-t4-region-repair-rescue-braf-d06")
os.environ.setdefault("COMPOSE_HELD_OUTPUT", "/region_repair_rescue_braf_d06")
os.environ.setdefault("COMPOSE_HELD_APP", "compose-t4-region-repair-rescue-braf-d06")
os.environ.setdefault("COMPOSE_HELD_RECEPTOR_NAME", "braf")

from modal_apps.t4_integrated_route_fiber_parp1_app import app, main  # noqa: F401
