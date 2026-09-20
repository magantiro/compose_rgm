"""Run the PMO-v1 zero-oracle live-parent coverage gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from compose_v4.experiments.pmo_population_live_parent_gate import run_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--scheduler", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    result = run_gate(root, output=output, exhaustive=args.exhaustive, scheduler=args.scheduler)
    print(result["payload"]["support"])


if __name__ == "__main__":
    main()
