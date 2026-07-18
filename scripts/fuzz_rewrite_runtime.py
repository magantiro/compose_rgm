from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from compose_v4.rewrite.fuzz import run_compiled_trace_fuzz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-commits", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run_compiled_trace_fuzz(
        target_commits=args.target_commits,
        seed=args.seed,
    )
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
