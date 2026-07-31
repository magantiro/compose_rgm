#!/usr/bin/env python3
"""Build one deterministic non-authorizing Editing-V2 four-role assignment."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from compose_v4.data.editing_v2_split_assignment import (
    build_split_assignment,
    load_split_assignment_policy,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _atomic_write(path: Path, value: object) -> None:
    content = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def main() -> int:
    args = _parser().parse_args()
    census = json.loads(args.census.read_bytes())
    policy = load_split_assignment_policy(args.policy)
    assignment = build_split_assignment(census, policy=policy)
    _atomic_write(args.output, assignment)
    print(
        json.dumps(
            {
                "status": assignment["status"],
                "training_authorized": assignment["training_authorized"],
                "gate_results": assignment["gate_results"],
                "assignment_sha256": assignment["assignment_sha256"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
