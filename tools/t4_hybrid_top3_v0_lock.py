"""Build the zero-oracle hybrid top-three plus Dynamic-v0 locks."""

from __future__ import annotations

import argparse
from pathlib import Path

from compose_v4.experiments.t4_hybrid_top3_v0_lock import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/t4_hybrid_top3_v0_controller_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostics/t4_hybrid_top3_v0_controller/attempt_1"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    physical = run(root, root / args.contract, root / args.output)
    for name, digest in sorted(physical.items()):
        print(f"{name} {digest}")


if __name__ == "__main__":
    main()
