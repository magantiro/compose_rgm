#!/usr/bin/env python3
"""Fit and lock the autonomous complete-plan endpoint ranker."""

from __future__ import annotations

import argparse
from pathlib import Path

from compose_v4.experiments.pmo_complete_plan_ranker import (
    build,
    evaluate_temporal_holdout,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "holdout"), nargs="?", default="build")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/pmo_delayed_credit_audit.json"
    )
    parser.add_argument("--ranker", type=Path)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--compiled", type=Path)
    parser.add_argument(
        "--audit", type=Path, default=ROOT / "diagnostics/pmo_delayed_credit/audit.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "diagnostics/pmo_delayed_credit/plan_ranker_lock.json",
    )
    args = parser.parse_args()
    if args.command == "build":
        if args.artifact_root is None or args.compiled is None:
            parser.error("build requires --artifact-root and --compiled")
        result = build(
            args.config,
            args.artifact_root,
            args.compiled,
            args.audit,
            args.output,
            ROOT,
        )
        print(args.output)
        print(
            f"{result['candidate_count']} candidates, {result['pool_count']} pools, "
            f"roles={result['selected_role_counts']}"
        )
    else:
        if args.ranker is None or args.holdout is None:
            parser.error("holdout requires --ranker and --holdout")
        result = evaluate_temporal_holdout(args.ranker, args.holdout, args.output, ROOT)
        print(args.output)
        print(result["metrics"])


if __name__ == "__main__":
    main()
