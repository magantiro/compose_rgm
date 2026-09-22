"""Why a declared option expired: was its protected bridge ever DRAWN in its window?

The outcome metric -- destination reached -- cannot distinguish a floor that was applied
and still lost from a floor that was never large enough to matter.  This reads the round
snapshots a campaign already published and decomposes every expiry into the step that
actually failed, at zero additional compute and zero oracle cost.

The mechanism has three consecutive requirements per crossing, and each can fail alone:

  1. the window OPENS     -- the bridge is charged, so `note_charged` opens a window
  2. the bridge is DRAWN  -- `_macro_option_candidates` offers the next leg only from a
                             parent the round's schedule drew, which is what makes the
                             parent-mass floor load-bearing rather than decoration
  3. the leg is CHARGED   -- the reserved slot survives allocation into the locked batch

Usage:
  PYTHONPATH=src:scripts python scripts/pmo_macro_option_window_diagnostic.py \
      --work <arms work dir> --out <path.json>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from compose_v4.control.pmo_macro_option_controller import MACRO_OPTION_CHANNEL_TAG


def _arm_rows(folder: Path) -> dict:
    """One arm's campaign directory -> per-option window accounting."""
    rounds = sorted((folder / "campaign").glob("round_*/complete.json"))
    if not rounds:
        return {}
    offered_by_option: Counter = Counter()
    charged_by_option: Counter = Counter()
    for path in sorted((folder / "campaign").glob("round_*/pending.json")):
        batch = json.loads(path.read_text())["batch"]
        # `proposal_pool` is the full generated pool; `candidates` is the LOCKED subset
        # the ledger charges. Both are needed: offered-but-not-charged is a different
        # failure from never offered at all.
        pool = batch.get("proposal_pool", batch)["candidates"]
        for row in pool:
            if row["provenance"].get("entry_channel") == MACRO_OPTION_CHANNEL_TAG:
                offered_by_option[row["provenance"]["macro_option_id"]] += 1
        for row in batch["candidates"]:
            if row["provenance"].get("entry_channel") == MACRO_OPTION_CHANNEL_TAG:
                charged_by_option[row["provenance"]["macro_option_id"]] += 1

    # How often protection actually CHANGED the batch, rather than agreeing with a
    # choice the ordinary allocator had already made. `reserved_already_chosen` is the
    # ordinary policy picking the leg on its own -- protection was inert that round --
    # while `reserved_added` and `displaced` are the reservation doing real work.
    reservation = {"reserved_added": 0, "reserved_already_chosen": 0, "displaced": 0}
    for path in sorted((folder / "campaign").glob("round_*/pending.json")):
        batch = json.loads(path.read_text())["batch"]
        detail = (batch.get("allocation") or {}).get("macro_option_reservation") or {}
        for key in reservation:
            reservation[key] += int(detail.get(key, 0) or 0)

    final = json.loads(rounds[-1].read_text())["snapshot"]["macro_options"]
    out = {}
    for record in final["registry"]["records"]:
        option_id = record["option"]["option_id"]
        stages = len(record["option"]["stages"])
        frontier = record["frontier"]
        status = record["status"]
        if status == "declared":
            # Never charged a first leg at all: the option never started.
            failure = "bridge_never_charged"
        elif status == "reached":
            failure = None
        elif offered_by_option[option_id] > charged_by_option[option_id]:
            # The leg WAS offered inside the window and did not reach the oracle.
            failure = "offered_but_not_charged"
        else:
            # The window opened and the leg was never even offered, which means the
            # bridge was not in the round's drawn parent schedule: the floor was applied
            # and was not enough to get it drawn.
            failure = "bridge_not_drawn_in_window"
        out[option_id] = {
            "status": status,
            "stages": stages,
            "frontier": frontier,
            "legs_charged": frontier + 1,
            "legs_required": stages,
            "legs_offered": offered_by_option[option_id],
            "legs_locked": charged_by_option[option_id],
            "failure": failure,
        }
    return {"options": out, "reservation": reservation}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    seeds = {}
    for seed_dir in sorted(arguments.work.glob("seed_*")):
        arms = {}
        for name in ("protected", "declared_unprotected"):
            folder = seed_dir / name
            if folder.exists():
                arms[name] = _arm_rows(folder)
        if arms:
            seeds[seed_dir.name] = arms

    summary = {}
    reservation_totals: dict[str, dict[str, int]] = {
        "protected": {},
        "declared_unprotected": {},
    }
    for name in ("protected", "declared_unprotected"):
        failures: Counter = Counter()
        legs = {"charged": 0, "required": 0}
        for arms in seeds.values():
            block = arms.get(name) or {}
            for key, value in (block.get("reservation") or {}).items():
                reservation_totals[name][key] = (
                    reservation_totals[name].get(key, 0) + value
                )
            for row in (block.get("options") or {}).values():
                failures[row["failure"] or "reached"] += 1
                legs["charged"] += row["legs_charged"]
                legs["required"] += row["legs_required"]
        summary[name] = {
            "failures": dict(sorted(failures.items())),
            "legs": legs,
            "reservation": reservation_totals[name],
        }

    payload = {
        "schema_version": "pmo_macro_option_window_diagnostic_v1",
        "benchmark_oracle_calls": 0,
        "seeds": seeds,
        "summary": summary,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
