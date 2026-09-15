"""Build the prospective zero-oracle compositional-generator utility lock."""

from __future__ import annotations

import argparse
from pathlib import Path

from compose_v4.experiments.t4_compositional_utility_lock import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prelock",
        type=Path,
        default=Path("configs/t4_compositional_generator_utility_prelock_v1.json"),
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=Path("configs/t4_compositional_generator_utility_inputs_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/t4_compositional_generator_utility_lock/attempt_1"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    physical = run(root, root / args.prelock, root / args.binding, root / args.output)
    for name, digest in physical.items():
        print(f"{name} {digest}")


if __name__ == "__main__":
    main()
