"""Generate deep constructive proposals in volume and keep only the eligible ones.

Zero oracle calls. Eligibility is the frozen endpoint gate computed locally, so the
docking budget is untouched: this stage spends CPU to produce a pool that a later,
separately authorized stage may score.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.control.constructive_composition import prospect
from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.t4_construction_scale import builder_families, family_profile
from compose_v4.experiments.t4_frozen_program_benchmark import strict_endpoint_scorer
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/t4_objective_reset_runtime_v1.json"
CORPUS = ROOT / "diagnostics/t4_proposal_prior/dataset_v1/records.jsonl.gz"


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--attempts", type=int, required=True)
    parser.add_argument("--minimum-primitives", type=int, default=15)
    parser.add_argument("--wall-seconds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import gzip

    from compose_v4.control.dynamic_program_synthesis import GENERIC_MODULES

    with gzip.open(CORPUS, "rt") as handle:
        records = [json.loads(line) for line in handle]
    builders = builder_families(family_profile(records), universe=GENERIC_MODULES)

    units = {u["cell"]: u for u in unseal(REGISTRY)["units"]}
    if args.cell not in units:
        parser.error(f"unknown cell {args.cell}; declared cells are {sorted(units)}")
    unit = units[args.cell]

    reported = {"attempts": 0}

    def progress(row):
        if row["attempts"] - reported["attempts"] >= 250:
            reported["attempts"] = row["attempts"]
            print({"cell": args.cell, **row}, flush=True)

    result = prospect(
        decode_state(unit["source_state"]),
        strict_endpoint_scorer(unit["original_seed"], delta=0.4),
        np.random.default_rng(np.random.SeedSequence([args.seed, len(args.cell)])),
        attempts=args.attempts,
        minimum_primitives=args.minimum_primitives,
        builders=builders,
        wall_seconds=args.wall_seconds,
        progress=progress,
    )
    payload = {
        **result,
        "cell": args.cell,
        "seed": args.seed,
        "builders": list(builders),
        "source_registry_sha256": verify_file(REGISTRY, sha256_file(REGISTRY)),
        "code_revision": _revision(),
        "runtime": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "completed_at_utc": _stamp(),
        "interpretation": (
            "an eligible pool, not a result: these endpoints have passed the frozen "
            "similarity, QED and SA gate and have never been docked"
        ),
    }
    seal(args.output, payload)
    print(json.dumps({k: v for k, v in payload.items() if k not in ("pool",)}, indent=2)[:1400])


if __name__ == "__main__":
    main()
