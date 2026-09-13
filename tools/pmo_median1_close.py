"""Prepare or run the one-query Median1 PMO closeout."""

from __future__ import annotations

import argparse
from pathlib import Path

from compose_v4.experiments.pmo_median1_close import OUTPUT, prepare, run

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=Path(OUTPUT))
    args = parser.parse_args()
    payload = (
        prepare(ROOT, args.output)
        if args.action == "prepare"
        else run(ROOT, args.output)
    )
    print(payload)


if __name__ == "__main__":
    main()
