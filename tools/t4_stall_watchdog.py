"""Detect T4 campaign cells that have silently stopped, and say which.

A cell publishes `round_NNN_lock.json` BEFORE docking that round's queries.  The held-target
apps run with `retries: 0`, so a preempted container does NOT restart -- it stops and waits.
The ephemeral app stays alive because its sibling cells are still working, so nothing in
`modal app list` shows the loss.  One cell sat on a root lock for 2.5 hours this way while its
two siblings completed 3 and 4 rounds; it was found by hand, which does not scale to 45 cells.

Two signatures, deliberately kept apart because they need different thresholds:

  STALLED_AT_ROOT   the newest lock is `round_000_lock.json`, it is a root lock (1 query,
                    parent_score null), and there is no checkpoint and no result.  A root dock
                    takes seconds, so anything past a few minutes here is dead, not slow.

  STALLED_MID_ROUND the newest lock index is ahead of the checkpoint's completed rounds AND
                    nothing has changed since the previous snapshot.  Rounds legitimately take
                    ~40-50 minutes, so this needs a generous threshold or it cries wolf: a
                    published lock with an unfinished round is the NORMAL look of work in
                    progress.  Only the absence of change across snapshots distinguishes them.

Read-only.  It never relaunches, stops or writes to any volume -- recovery is a separate,
budget-aware decision, because a published lock counts as a charged call even when no result
artifact exists, and a naive relaunch would double-charge it.

    python3 tools/t4_stall_watchdog.py --state diagnostics/t4_stall_watchdog_state.json

Exits 1 when any cell is stalled, so it can gate a monitor.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

# Reuse the reconciler's arm map rather than duplicating volume names, so a new arm cannot be
# watched by one tool and missed by the other.
from t4_reconcile_ledger import ARMS

ROOT_STALL_MINUTES = 15
MID_ROUND_STALL_MINUTES = 90


def _ls(volume: str, path: str, timeout: int) -> list[str]:
    proc = subprocess.run(
        ["modal", "volume", "ls", volume, path],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def snapshot(timeout: int) -> dict:
    """Per cell: newest lock index, and whether a checkpoint or result exists."""
    cells: dict[str, dict] = {}
    for arm, (volume, _target, _contract) in ARMS.items():
        for run in _ls(volume, "", timeout):
            run_id = run.rsplit("/", 1)[-1]
            if not run_id or "." in run_id:
                continue
            for cell_line in _ls(volume, run_id, timeout):
                cell = cell_line.rsplit("/", 1)[-1]
                if not cell or "." in cell:
                    continue
                names = [n.rsplit("/", 1)[-1] for n in _ls(volume, f"{run_id}/{cell}", timeout)]
                locks = sorted(n for n in names if n.startswith("round_") and n.endswith("_lock.json"))
                cells[f"{arm}/{run_id[:12]}/{cell}"] = {
                    "arm": arm,
                    "cell": cell,
                    "newest_lock": locks[-1] if locks else None,
                    "lock_count": len(locks),
                    "has_checkpoint": "checkpoint.json" in names,
                    "has_result": "result.json" in names,
                }
    return {"taken_at_utc": datetime.now(timezone.utc).isoformat(), "cells": cells}


def classify(now: dict, prev: dict | None) -> list[dict]:
    findings = []
    prev_cells = (prev or {}).get("cells", {})
    prev_at = (prev or {}).get("taken_at_utc")
    minutes = 0.0
    if prev_at:
        delta = datetime.fromisoformat(now["taken_at_utc"]) - datetime.fromisoformat(prev_at)
        minutes = delta.total_seconds() / 60.0

    for key, cur in now["cells"].items():
        if cur["has_result"]:
            continue  # terminal: the cell finished or aborted with a recorded reason
        before = prev_cells.get(key)
        unchanged = before is not None and {
            k: before.get(k) for k in ("newest_lock", "lock_count", "has_checkpoint")
        } == {k: cur.get(k) for k in ("newest_lock", "lock_count", "has_checkpoint")}

        root_only = (
            cur["lock_count"] == 1
            and cur["newest_lock"] == "round_000_lock.json"
            and not cur["has_checkpoint"]
        )
        if root_only and unchanged and minutes >= ROOT_STALL_MINUTES:
            findings.append({**cur, "key": key, "status": "STALLED_AT_ROOT",
                             "unchanged_minutes": round(minutes, 1)})
        elif unchanged and minutes >= MID_ROUND_STALL_MINUTES:
            findings.append({**cur, "key": key, "status": "STALLED_MID_ROUND",
                             "unchanged_minutes": round(minutes, 1)})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=ROOT / "diagnostics/t4_stall_watchdog_state.json")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    prev = json.loads(args.state.read_text()) if args.state.exists() else None
    now = snapshot(args.timeout)
    findings = classify(now, prev)

    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(now, indent=2, sort_keys=True) + "\n")

    watched = len(now["cells"])
    if prev is None:
        print(f"baseline snapshot written: {watched} cells watched; no verdict on a first run")
        return 0

    print(f"{watched} cells watched")
    for f in findings:
        print(f"  {f['status']:18s} {f['key']}  unchanged {f['unchanged_minutes']:.0f} min  "
              f"locks={f['lock_count']} newest={f['newest_lock']} ckpt={f['has_checkpoint']}")
    if not findings:
        print("  no stalled cells")
        return 0
    print(f"\n{len(findings)} stalled cell(s). Recovery is a separate decision: a published lock "
          f"counts as a charged call, so subtract it from that cell's allowance before relaunching.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
