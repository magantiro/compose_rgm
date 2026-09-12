#!/usr/bin/env python3
"""Create the zero-oracle queue lock for the plan-pool prevalence assay."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.pmo_plan_pool_lock import build_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("diagnostics/pmo_plan_policy/prepared.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/pmo_plan_policy/pool_queue_lock.json"),
    )
    parser.add_argument("--max-pools", type=int, default=32)
    args = parser.parse_args()
    value = build_lock(
        args.result, args.artifacts, args.prepared, max_pools=args.max_pools
    )
    value["analysis_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    value["implementation_sha256"] = {
        "src/compose_v4/experiments/pmo_plan_pool_lock.py": sha256_file(
            Path("src/compose_v4/experiments/pmo_plan_pool_lock.py")
        ),
        "tools/pmo_plan_pool_lock.py": sha256_file(Path("tools/pmo_plan_pool_lock.py")),
        "docs/PMO_PLAN_POOL_PREVALENCE.md": sha256_file(
            Path("docs/PMO_PLAN_POOL_PREVALENCE.md")
        ),
    }
    publish_json(args.output, value)
    print(
        f"locked {value['selected_pool_count']} pools and "
        f"{value['queued_unique_products']} unique products; new oracle calls=0"
    )


if __name__ == "__main__":
    main()
