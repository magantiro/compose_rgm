#!/usr/bin/env python3
"""Run the deterministic shared-controller production support gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.experiments.t4_shared_controller_completion_contract import (
    payload_identity,
)
from compose_v4.experiments.t4_shared_controller_production_support_gate import (
    run_gate,
    scientific_projection,
)

DEFAULT_CONTRACT = ROOT / "configs/t4_shared_controller_production_support_gate_v1.json"
DEFAULT_OUTPUT = (
    ROOT
    / "diagnostics/t4_shared_controller_production_support_gate_v1/attempt_1/result.json"
)


def _write_once(path: Path, payload: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite production support gate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": payload_identity(payload)}
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    payload = run_gate(ROOT, args.contract)
    if args.compare:
        prior = json.loads(args.compare.read_text())["payload"]
        if scientific_projection(prior) != scientific_projection(payload):
            raise RuntimeError("production support gate scientific projection drift")
    _write_once(args.output, payload)
    print(json.dumps(payload["gate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
