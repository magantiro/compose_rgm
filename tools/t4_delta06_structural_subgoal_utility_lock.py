"""Build the zero-oracle T4 strict-delta-0.6 endpoint utility lock."""

from __future__ import annotations

import argparse
from pathlib import Path

from compose_v4.experiments.t4_delta06_utility_lock import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/t4_delta06_structural_subgoal_utility_lock_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "diagnostics/t4_delta06_structural_subgoal_utility_lock/attempt_1"
        ),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    physical = run(root, root / args.contract, root / args.output)
    for name, digest in physical.items():
        print(f"{name} {digest}")


if __name__ == "__main__":
    main()
