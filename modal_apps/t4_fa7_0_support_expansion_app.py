"""Support-expansion launcher for fa7_0 at delta=0.6, 248 charged calls, one cell.

fa7_0 is the single blank in the frozen delta=0.6 panel, and it is blank because
the campaign ended it rather than because the search looked and lost: it charged
exactly ONE call -- the seed's own docking score -- and terminated at
`candidate_exhaustion` in round zero with 247 authorized calls never issued.  The
sibling support-expansion arm never reached it at all; it was queued behind fa7_2
and died on resume against a stale query lock.

This arm changes two things, and both are executable rather than declarative.

1.  `proposal.shallow.completion_law`.  The completion half of `segment_replace`
    drew blind -- a length uniform on 1..8 and a uniform C/N/O chain -- while every
    measured eligible completion inserts one or two atoms, so most of the draw mass
    landed where nothing is eligible.  Conditioning it lifted eligible endpoints
    +86% / +50% / +75% on fa7_1 / braf_2 / 5ht1b_0 with no control regressing, and
    on fa7_0 it is the configuration that produced the one generated endpoint which
    RETAINS the amidine, the FA7 S1-pocket pharmacophore the unconditioned path
    deleted.

2.  `support_expansion`.  An empty candidate pool no longer ends the cell.  It runs
    a bounded expansion first -- the zero-support fallback, then an escalating
    parallel fan-out of the primary draw lanes -- and only an expansion that runs to
    its declared end without an eligible endpoint publishes candidate exhaustion.
    Without this the arm is a coin flip on round one, and a lost coin spends the
    whole budget reproducing the termination it exists to fix.

Expansion charges nothing.  It returns proposal records that are locked and docked
through the unchanged round path, so charged-call accounting, the per-query
receipts, the no-retry / no-replacement / no-backfill policy and the 248-call
ceiling are untouched.

App name, volume and output namespace are distinct from every other arm, and the
run id is content-addressed over the contract and the code revision, so this run
cannot inherit the stale query lock that killed its predecessor.  `main` spawns,
so a launch needs `modal run --detach`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CONTRACT = "configs/t4_fa7_0_support_expansion_v1.json"
AUTHORIZED_STATUS = "AUTHORIZED_BY_OWNER_FOR_FA7_0_SUPPORT_EXPANSION"


def _assert_authorized() -> None:
    """Refuse to construct the arm unless both mechanisms are actually declared.

    The base app's launch gate verifies identity and tree cleanliness but never
    reads `status`, and the base app is hash-pinned so the check cannot be added
    there.  It lives here, at import time, which is before `modal run` can spawn
    anything.  Each clause names a field whose absence would make the arm a
    repeat of the failure it exists to fix rather than a fix for it.
    """

    payload = json.loads((Path(__file__).resolve().parents[1] / CONTRACT).read_text())["payload"]
    status = payload.get("status")
    if status != AUTHORIZED_STATUS:
        raise RuntimeError(
            f"{CONTRACT} is not authorized for launch (status={status!r}); the owner must "
            f"set status to {AUTHORIZED_STATUS!r} and re-seal the contract first"
        )
    cells = [row["cell"] for row in payload.get("cells") or ()]
    if cells != ["fa7_0"]:
        raise RuntimeError(
            f"this arm is authorized for fa7_0 alone; the contract declares {cells}"
        )
    lane = (payload.get("proposal") or {}).get("shallow") or {}
    if lane.get("region_law") != "free_gate_margin_v1":
        raise RuntimeError(
            "proposal.shallow.region_law is not free_gate_margin_v1; without it the "
            "region draw is capped at 8 atoms and cannot express the 7-15 atom "
            "excision every measured witness of this cell needs"
        )
    if lane.get("completion_law") != "free_gate_margin_v1":
        raise RuntimeError(
            "proposal.shallow.completion_law is not free_gate_margin_v1; without it "
            "the completion half draws blind and the amidine-retaining endpoint this "
            "arm exists to reach is not the configuration being run"
        )
    expansion = payload.get("support_expansion")
    if not isinstance(expansion, dict) or not expansion.get("draw_ladder"):
        raise RuntimeError(
            "support_expansion declares no draw ladder; the run would terminate on "
            "the first empty candidate pool, which is exactly how this cell went blank"
        )


_assert_authorized()

os.environ.setdefault("COMPOSE_HELD_CONTRACT", CONTRACT)
os.environ.setdefault(
    "COMPOSE_HELD_CHECKPOINT",
    "diagnostics/t4_held_target_distillation_quality_v1/fa7_checkpoint.json",
)
os.environ.setdefault("COMPOSE_HELD_VOLUME", "compose-t4-fa7-0-support-expansion")
os.environ.setdefault("COMPOSE_HELD_OUTPUT", "/fa7_0_support_expansion")
os.environ.setdefault("COMPOSE_HELD_APP", "compose-t4-fa7-0-support-expansion")
os.environ.setdefault("COMPOSE_HELD_RECEPTOR_NAME", "fa7")

from modal_apps.t4_fa7_0_support_expansion_base_app import app, main  # noqa: F401
