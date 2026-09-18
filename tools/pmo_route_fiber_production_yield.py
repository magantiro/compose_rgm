#!/usr/bin/env python3
"""Run the zero-oracle PMO matched actual-sampler yield comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.experiments.pmo_route_fiber_production_yield import (
    CHECKPOINT,
    CONTRACT,
    RESULT,
    build_production_yield,
    contract_envelope,
)


def publish(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default=CONTRACT)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--output", default=RESULT)
    args = parser.parse_args()
    publish(ROOT / args.contract, contract_envelope())

    def progress(row: dict) -> None:
        print(json.dumps(row, sort_keys=True), flush=True)

    checkpoint, result = build_production_yield(ROOT, progress=progress)
    publish(ROOT / args.checkpoint, checkpoint)
    publish(ROOT / args.output, result)
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "result": args.output,
                "decision": result["payload"]["decision"],
                "new_oracle_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
