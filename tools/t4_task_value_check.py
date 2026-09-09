"""Chronological reuse of counted docking labels; zero new oracle calls."""

from __future__ import annotations

import argparse
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.docking_value import chronological_check
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "diagnostics/t4_ring_program_round/attempt_1/run/round_4/archive.json"
ARCHIVE_SHA256 = "803741324ac0e30bff96f9ad36809c6d08ff8334a76a1cc1927a7296a91daf30"


def run(archive_path: Path, output: Path):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "tools", "configs"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise ValueError("task-value measurement requires a clean committed scientific code tree")
    if sha256_file(archive_path) != ARCHIVE_SHA256:
        raise ValueError(f"saved archive identity mismatch; expected {ARCHIVE_SHA256}")
    warm = unseal(archive_path)
    if (
        warm.get("schema_version") != "t4_exact_archive_v1"
        or warm.get("oracle_attempts") != 51
        or warm.get("round") != 4
        or len(warm["archive"]) != 52
    ):
        raise ValueError("this check requires the complete saved 51-call development archive")
    start = perf_counter()
    result = chronological_check(warm["archive"], source_sha256=sha256_file(archive_path))
    result.update(
        code_revision=revision,
        source_path=str(archive_path),
        input_sha256={
            "archive": sha256_file(archive_path),
            "script": sha256_file(Path(__file__)),
            "predictor": sha256_file(ROOT / "src/compose_v4/control/docking_value.py"),
            "contract": sha256_file(ROOT / "docs/HIERARCHICAL_TASK_SEARCH.md"),
        },
        software={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        hardware={
            "machine": platform.machine(),
            "processor": platform.processor(),
            "precision": "float64",
        },
        seed=None,
        seed_reason="deterministic kernels and linear solve; no sampling",
        split="whole next-round prediction on an already inspected development cell",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        seconds=perf_counter() - start,
    )
    publish_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.archive, args.output)
    print(
        {
            key: result[key]
            for key in (
                "decision",
                "n_scored",
                "n_pairs",
                "mae",
                "baseline_mae",
                "concordance",
                "seconds",
            )
        }
    )


if __name__ == "__main__":
    main()
