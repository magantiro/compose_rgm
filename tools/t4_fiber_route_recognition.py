#!/usr/bin/env python3
"""Run and publish the frozen zero-oracle FiberControl route-recognition diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_fiber_route_recognition import run

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs/t4_fiber_route_recognition_v1.json"
DEFAULT_OUTPUT = ROOT / "diagnostics/t4_fiber_route_recognition_v1/result.json"


def publish(path: Path, payload: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite route-recognition result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    payload = run(ROOT, arguments.contract.resolve())
    publish(arguments.output.resolve(), payload)
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
