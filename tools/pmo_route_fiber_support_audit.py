#!/usr/bin/env python3
"""Publish the zero-oracle PMO production-proposer support audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compose_v4.experiments.pmo_route_fiber_support_audit import (
    CONTRACT,
    RESULT,
    build_support_audit,
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
    parser.add_argument("--output", default=RESULT)
    args = parser.parse_args()
    contract = contract_envelope()
    publish(ROOT / args.contract, contract)
    result = build_support_audit(ROOT)
    publish(ROOT / args.output, result)
    print(json.dumps({"output": args.output, **result["payload"]}, indent=2))


if __name__ == "__main__":
    main()
