#!/usr/bin/env python3
"""Create the zero-oracle queue lock for the plan-pool prevalence assay."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from compose_v4.experiments.pmo_plan_pool_lock import (
    analyze_compiled_chemistry,
    build_lock,
    compile_locked_queues,
    score_compiled_lock,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock = subparsers.add_parser("lock")
    lock.add_argument("--result", type=Path, required=True)
    lock.add_argument("--artifacts", type=Path, required=True)
    lock.add_argument(
        "--prepared",
        type=Path,
        default=Path("diagnostics/pmo_plan_policy/prepared.json"),
    )
    lock.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/pmo_plan_policy/pool_queue_lock.json"),
    )
    lock.add_argument("--max-pools", type=int, default=32)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--lock", type=Path, required=True)
    compile_parser.add_argument("--artifacts", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--workers", type=int, default=8)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--compiled", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--authorize-new-calls", type=int, required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--compiled", type=Path, required=True)
    analyze_parser.add_argument("--target", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compile":
        value = compile_locked_queues(
            args.lock, args.artifacts, args.output, workers=args.workers
        )
        print(
            f"compiled {value['candidate_count']} unique candidates after "
            f"{value['attempt_count']} attempts; new oracle calls=0"
        )
        return
    if args.command == "analyze":
        value = analyze_compiled_chemistry(args.compiled, args.target, args.output)
        print(
            f"analyzed {value['candidate_count']} locked candidates; "
            f"closest target Tanimoto={value['closest_candidate']['target_tanimoto']:.4f}; "
            "new oracle calls=0"
        )
        return
    if args.command == "score":
        from compose_v4.experiments.pmo_macro_probe import make_oracle

        value = score_compiled_lock(
            args.compiled,
            args.output,
            make_oracle("perindopril_mpo", Path.cwd(), {}),
            authorized_calls=args.authorize_new_calls,
        )
        print(
            f"scored {value['new_oracle_calls']} candidates; "
            f"best={value['best_score']:.9f}; decision={value['decision']}"
        )
        return
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
