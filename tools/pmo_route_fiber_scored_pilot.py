#!/usr/bin/env python3
"""Zero-oracle preparation and read-only status for the frozen PMO pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.experiments.pmo_route_fiber_scored_pilot import (
    build_preflight,
    prepare_candidate_locks,
    status,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("prepare-pools", "preflight", "status"),
        help="No action in this tool calls a PMO oracle.",
    )
    args = parser.parse_args()
    if args.action == "prepare-pools":

        def progress(row: dict) -> None:
            print(json.dumps(row, sort_keys=True), flush=True)

        result = prepare_candidate_locks(ROOT, progress=progress)
    elif args.action == "preflight":
        result = build_preflight(ROOT)
    else:
        result = status(ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
