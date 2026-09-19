#!/usr/bin/env python3
"""Run the sealed zero-oracle nine-cell T4 proposal preflight."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.experiments.t4_nine_cell_support_preflight import run_preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "configs" / "t4_nine_cell_support_preflight_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-scientific", type=Path)
    args = parser.parse_args()
    payload = run_preflight(
        ROOT,
        args.contract.resolve(),
        args.output.resolve(),
        compare_scientific=(
            args.compare_scientific.resolve() if args.compare_scientific else None
        ),
    )
    for cell in payload["cells"]:
        counts = ", ".join(
            f"{row['expert']}={row['eligible_unique']}" for row in cell["experts"]
        )
        print(f"{cell['cell_id']}: {counts}; pooled={cell['pooled_eligible_unique']}")
    return 0 if payload["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
