"""Run the matched protected / declared-unprotected macro-option arms.

ZERO charged benchmark oracle calls: every score comes from the declared synthetic
`valley_similarity_v1`.  The arms are matched on seed, initialization, configuration,
budget and the declared option set, and differ in exactly one bit -- protection.

Each seed writes its own artifact as it lands, so a killed driver loses at most the seed
in flight and a relaunch reuses what is already on disk.

Usage:
  PYTHONPATH=src:scripts python scripts/pmo_macro_option_matched_arms.py \
      --seeds 12 --budget 96 --rounds 10 --queries-per-round 8 \
      --out diagnostics/pmo_macro_option_v1/arms
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from time import perf_counter

sys.path.insert(0, "src")

from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_macro_option_arms import run_arm

BASE_SEED = 20260922


def _seed_for(index: int) -> int:
    return BASE_SEED + 1009 * index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--budget", type=int, default=96)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--queries-per-round", type=int, default=8)
    parser.add_argument("--initialization-count", type=int, default=4)
    parser.add_argument("--max-options", type=int, default=4)
    parser.add_argument("--synthesis-attempts", type=int, default=5)
    # MEASURED: declaration costs ~135s once options must prune before installing. A wall
    # that binds would truncate the two arms differently under load and give them
    # different option sets, which VOIDs the pairing outright, so it is set well clear.
    parser.add_argument("--declaration-wall-seconds", type=float, default=600.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    arguments = parser.parse_args()

    out = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    work = arguments.work or (out / "_work")
    settings = {
        "max_options": arguments.max_options,
        "synthesis_attempts_per_origin": arguments.synthesis_attempts,
        "declaration_wall_seconds": arguments.declaration_wall_seconds,
    }
    for index in range(arguments.start, arguments.seeds):
        seed = _seed_for(index)
        destination = out / f"seed_{seed}.json"
        if destination.exists():
            print(f"seed {seed}: already landed", flush=True)
            continue
        began = perf_counter()
        row: dict = {
            "schema_version": "pmo_macro_option_matched_seed_v1",
            "seed": seed,
            "index": index,
            "budget": arguments.budget,
            "rounds": arguments.rounds,
            "queries_per_round": arguments.queries_per_round,
            "initialization_count": arguments.initialization_count,
            "macro_option_settings": settings,
            "benchmark_oracle_calls": 0,
        }
        try:
            arms = {}
            for protection in (True, False):
                name = "protected" if protection else "declared_unprotected"
                folder = work / f"seed_{seed}" / name
                shutil.rmtree(folder, ignore_errors=True)
                folder.mkdir(parents=True, exist_ok=True)
                arms[name] = run_arm(
                    root=Path.cwd(),
                    output=folder,
                    seed=seed,
                    protection=protection,
                    budget=arguments.budget,
                    rounds=arguments.rounds,
                    queries_per_round=arguments.queries_per_round,
                    initialization_count=arguments.initialization_count,
                    macro_option_settings=settings,
                )
            row["arms"] = arms
            row["status"] = "complete"
        except Exception as error:  # noqa: BLE001 - a failed seed must not kill the sweep
            row["status"] = "failed"
            row["error"] = repr(error)
            row["traceback"] = traceback.format_exc()
        row["seconds"] = perf_counter() - began
        row["row_sha256"] = identity({k: v for k, v in row.items() if k != "row_sha256"})
        destination.write_text(json.dumps(row, indent=1, sort_keys=True))
        shutil.rmtree(work / f"seed_{seed}", ignore_errors=True)
        status = row["status"]
        declared = (
            len(row["arms"]["protected"]["options"]) if status == "complete" else 0
        )
        print(
            f"seed {seed}: {status} options={declared} {row['seconds']:.0f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
