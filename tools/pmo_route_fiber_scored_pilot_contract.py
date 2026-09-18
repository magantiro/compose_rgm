#!/usr/bin/env python3
"""Prepare or verify the zero-oracle PMO scored-pilot contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.experiments.pmo_route_fiber_scored_pilot_contract import (
    CONTRACT,
    envelope,
    verify_contract,
)


def publish(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--contract", default=CONTRACT)
    args = parser.parse_args()
    path = ROOT / args.contract
    if args.action == "prepare":
        publish(path, envelope(ROOT))
    contract = verify_contract(ROOT, path)
    print(
        json.dumps(
            {
                "contract": str(path),
                "payload_sha256": contract["payload_sha256"],
                "status": contract["payload"]["status"],
                "charged_call_ceiling": contract["payload"]["budget"][
                    "total_charged_call_ceiling"
                ],
                "oracle_calls_authorized": 0,
                "scored_launch_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
