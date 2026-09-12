#!/usr/bin/env python3
"""Audit proposal recall and delayed-credit evidence in a completed plan run."""

from __future__ import annotations

import argparse
from pathlib import Path

from compose_v4.experiments.pmo_plan_failure_audit import run

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=ROOT / "diagnostics/pmo_plan_policy/prepared.json")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "diagnostics/pmo_plan_policy/failure_separation.json"
    )
    args = parser.parse_args()
    report = run(args.prepared, args.result, args.output, ROOT)
    print(args.output)
    print(report["summary"])
    print(report["verdict"])


if __name__ == "__main__":
    main()
