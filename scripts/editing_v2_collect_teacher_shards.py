"""Rebuild the teacher aggregate from per-container volume shards.

Recovery path for the case the generation run is designed to survive: the fan-out
completes on Modal but the local client dies before it can write the aggregate.
Each container persists its own shard, so this reassembles them into exactly what
the entrypoint would have produced.

Usage after pulling the shards down:

    modal volume get compose-v4-artifacts \\
        editing_v2/r_theta_run/h_phi_teacher/train <local-dir>
    python scripts/editing_v2_collect_teacher_shards.py \\
        --shards <local-dir> --role train --out diagnostics/<name>.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--role", default="train")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected", type=int, default=0,
                        help="pair count from the frozen cohort; a shortfall is "
                             "reported rather than silently accepted")
    args = parser.parse_args()

    files = sorted(args.shards.rglob("*.json"))
    results = [json.loads(f.read_text()) for f in files]
    if not results:
        raise SystemExit(f"no shards under {args.shards}")

    labels = sum(r["labels"] for r in results)
    calls = sum(r["kernel_calls"] for r in results)
    states = sum(r["decision_states"] for r in results)
    kinds = collections.Counter(row["state_kind"]
                                for r in results for row in r["label_rows"])

    print(f"{len(results)} shards -> {states} decision states, {labels:,} labels, "
          f"{calls:,} kernel calls ({labels/max(calls,1):.2f} per call)")
    total = sum(kinds.values())
    for kind, count in sorted(kinds.items()):
        print(f"  {kind:14} {count:6,} ({count/max(total,1):5.1%})")
    if args.expected and len(results) < args.expected:
        print(f"  SHORTFALL: {len(results)} of {args.expected} pairs present. "
              f"The missing pairs were in flight when the client died; rerun "
              f"them rather than treating this as the full cohort.")

    args.out.write_text(json.dumps({
        "schema": "compose.editing_v2.h_phi_union_collector_check",
        "status": "REASSEMBLED_FROM_VOLUME_SHARDS",
        "role": args.role,
        "shards": len(results),
        "expected_pairs": args.expected or None,
        "label_strata": dict(kinds),
        "per_pair": results,
    }, indent=2) + "\n")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
