"""Run the 77-route decision-probability audit, shardable across workers.

Zero oracle calls. Teacher routes are diagnostic probes only; nothing produced here
is a runtime controller input.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import _stamp, seal
from compose_v4.experiments.t4_route_decision_audit import SCHEMA_VERSION, aggregate, audit_route

ROOT = Path(__file__).resolve().parents[1]
FORENSICS = ROOT / "diagnostics/t4_known_good_transformation_forensics/attempt_1/result.json"


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--samples", type=int, default=8, help="samples per family per state")
    parser.add_argument("--points", type=int, default=4, help="teacher states probed per route")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("shard must lie inside the declared shard count")

    forensics = json.loads(FORENSICS.read_text())
    forensics = forensics.get("payload", forensics)
    declared = forensics["routes"]
    mine = [r for i, r in enumerate(declared) if i % args.shards == args.shard]

    began, audited, missing = perf_counter(), [], []
    for index, entry in enumerate(mine):
        receipt = ROOT / entry["teacher_receipt"]
        if not receipt.exists():
            missing.append(entry["teacher_receipt"])
            continue
        digest = sha256_file(receipt)
        if digest != entry["teacher_receipt_sha256"]:
            raise ValueError(f"teacher receipt changed: {receipt}")
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, args.shard, index]))
        row = audit_route(receipt, rng=rng, samples=args.samples, points=args.points)
        row.update(cell=entry["cell"], target=entry["target"], probe_id=entry["probe_id"])
        audited.append(row)
        print(
            {
                "shard": args.shard,
                "route": index + 1,
                "of": len(mine),
                "cell": entry["cell"],
                "reproduced": row["any_point_reproduced"],
                "elapsed": round(perf_counter() - began, 1),
            },
            flush=True,
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "shard": args.shard,
        "shards": args.shards,
        "declared_routes": len(declared),
        "routes_in_shard": len(mine),
        "missing_receipts": missing,
        "samples_per_family": args.samples,
        "points_per_route": args.points,
        "routes": audited,
        "aggregate": aggregate(audited),
        "elapsed_seconds": perf_counter() - began,
        "code_revision": _revision(),
        "runtime": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "completed_at_utc": _stamp(),
        "new_oracle_calls": 0,
    }
    seal(args.output, payload)
    print(json.dumps(payload["aggregate"], indent=2)[:1200])


if __name__ == "__main__":
    main()
