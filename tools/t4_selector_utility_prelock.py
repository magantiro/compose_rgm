"""Build the prospective zero-oracle T4 selector utility request lock."""

from __future__ import annotations

import argparse
from pathlib import Path

from compose_v4.experiments.t4_selector_utility_prelock import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/t4_selector_utility_prelock_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/t4_selector_utility_prelock/attempt_1"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    physical = run(root, root / args.contract, root / args.output)
    for name, digest in sorted(physical.items()):
        print(f"{name} {digest}")


if __name__ == "__main__":
    main()
