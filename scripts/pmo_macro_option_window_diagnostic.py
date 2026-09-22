"""Reduce the per-seed window decompositions the matched arms already recorded.

The outcome metric -- destination reached -- cannot separate a floor that was applied and
still lost from a floor that was never large enough to matter. `run_arm` records the
decomposition inside each seed artifact, because the driver deletes a seed's campaign
directory once that artifact lands; this only aggregates it.

Zero chemistry, zero oracle calls.

Usage:
  PYTHONPATH=src:scripts python scripts/pmo_macro_option_window_diagnostic.py \
      --arms diagnostics/pmo_macro_option_v1/arms \
      --out diagnostics/pmo_macro_option_v1/window_decomposition_v1.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ARMS = ("protected", "declared_unprotected")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    seeds: dict[str, dict] = {}
    summary: dict[str, dict] = {}
    failures = {name: Counter() for name in ARMS}
    legs = {name: {"charged": 0, "required": 0} for name in ARMS}
    reservation = {name: Counter() for name in ARMS}

    for path in sorted(arguments.arms.glob("seed_*.json")):
        row = json.loads(path.read_text())
        if row.get("status") != "complete":
            continue
        block = {}
        for name in ARMS:
            decomposition = row["arms"][name].get("window_decomposition") or {}
            block[name] = decomposition
            for option in (decomposition.get("options") or {}).values():
                failures[name][option["failure"] or "reached"] += 1
                legs[name]["charged"] += option["legs_charged"]
                legs[name]["required"] += option["legs_required"]
            for key, value in (decomposition.get("reservation") or {}).items():
                reservation[name][key] += value
        seeds[str(row["seed"])] = block

    for name in ARMS:
        summary[name] = {
            "failures": dict(sorted(failures[name].items())),
            "legs": legs[name],
            # `reserved_already_chosen` is the ordinary allocator picking the leg on its
            # own -- protection was inert that round. `reserved_added` and `displaced`
            # are the reservation changing the batch.
            "reservation": dict(sorted(reservation[name].items())),
        }

    payload = {
        "schema_version": "pmo_macro_option_window_diagnostic_v2",
        "benchmark_oracle_calls": 0,
        "seeds_read": len(seeds),
        "summary": summary,
        "seeds": seeds,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
