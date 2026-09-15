#!/usr/bin/env python3
"""Build the deterministic post-launch strict-delta-0.6 utility review."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_delta06_utility_launch import publish_once
from compose_v4.experiments.t4_delta06_utility_review import build_review

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "diagnostics/t4_delta06_structural_subgoal_utility_launch/attempt_1/review.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--code-revision")
    args = parser.parse_args()
    revision = (
        args.code_revision
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )
    payload = build_review(ROOT, code_revision=revision)
    physical = publish_once(args.output, payload)
    print(f"review={args.output}")
    print(f"physical_sha256={physical}")
    print(f"payload_sha256={identity(payload)}")


if __name__ == "__main__":
    main()
