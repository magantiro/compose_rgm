#!/usr/bin/env python3
"""Audit witnessed delayed credit in prior PMO option genealogies."""

from __future__ import annotations

import argparse
from pathlib import Path

from compose_v4.experiments.pmo_delayed_credit_audit import run

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/pmo_delayed_credit_audit.json"
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "diagnostics/pmo_delayed_credit/audit.json",
    )
    args = parser.parse_args()
    result = run(args.config, args.artifact_root, args.output, ROOT)
    print(args.output)
    print(result["verdict"])
    for model, horizons in result["summary"]["models"].items():
        for horizon, metrics in horizons.items():
            print(
                f"{model} h={horizon}: recoveries={metrics['delayed_recoveries']} "
                f"endpoint_rmse={metrics['endpoint_model_rmse']:.6f} "
                f"future_rmse={metrics['future_model_rmse']:.6f}"
            )


if __name__ == "__main__":
    main()
