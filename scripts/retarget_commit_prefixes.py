"""Hash the realized phase-A prefixes into a committed artifact.

This runs BETWEEN the two phases and is the point of the whole two-phase
structure: the switch states are fixed and hashed here, before goal B is ever
constructed, so the prefixes provably were not selected or steered with
knowledge of the future requirement.

Phase 2 refuses to run without this file.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = sorted((json.loads(Path(f).read_text())
                   for f in glob.glob(str(args.shards / "*.json"))),
                  key=lambda r: (r["history"], r["index"]))
    if not rows:
        raise SystemExit(f"no prefixes under {args.shards}")

    # The hash covers the realised switch states -- the objects phase 2 branches
    # from. If a prefix were regenerated later, this digest would change.
    digest = hashlib.sha256(json.dumps(
        [[r["history"], r["index"], r["switch_state"]] for r in rows],
        sort_keys=True).encode()).hexdigest()

    by_history: dict[str, int] = {}
    moved = 0
    for r in rows:
        by_history[r["history"]] = by_history.get(r["history"], 0) + 1
        if abs(r["a_worst_switch"] - r["a_worst_start"]) > 1e-9:
            moved += 1
    print(f"{len(rows)} prefixes  {by_history}")
    print(f"phase-A worst margin moved on {moved}/{len(rows)}")
    print(f"prefixes sha256 {digest[:16]}")

    args.out.write_text(json.dumps({
        "schema": "compose.retarget.committed_prefixes",
        "status": "COMMITTED_BEFORE_GOAL_B_WAS_CONSTRUCTED",
        "horizon": 6, "switch_at": 3,
        "goal_language": ("unbounded signed IQR-normalised margins; conjunction "
                          "ranked lexicographically by (worst margin, mean "
                          "margin); success = worst margin >= 0"),
        "prefixes_sha256": digest,
        "count": len(rows), "by_history": by_history,
        "phase_a_moved": moved,
        "prefixes": rows,
    }, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
