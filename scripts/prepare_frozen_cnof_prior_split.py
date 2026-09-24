"""Prepare an immutable nested CNOF split without fitting or scoring a model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from compose_v4.data.frozen_cnof_prior_split import prepare_frozen_cnof_split


def _clean_revision() -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError(
            "refusing to publish a scientific split from a dirty source tree; "
            "commit the preparation code first"
        )
    return revision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--source-access-basis", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--max-atoms", type=int, default=40)
    parser.add_argument("--small-train-size", type=int, default=50000)
    parser.add_argument("--validation-size", type=int, default=2000)
    parser.add_argument("--iid-test-size", type=int, default=2000)
    parser.add_argument("--scaffold-test-min", type=int, default=5000)
    parser.add_argument("--scaffold-test-max", type=int, default=7500)
    parser.add_argument("--expected-eligible-unique", type=int, required=True)
    args = parser.parse_args()
    revision = _clean_revision()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen split: {args.output}")
    payload = prepare_frozen_cnof_split(
        args.source,
        expected_source_sha256=args.expected_source_sha256,
        source_access_basis=args.source_access_basis,
        code_revision=revision,
        seed=args.seed,
        max_atoms=args.max_atoms,
        small_train_size=args.small_train_size,
        validation_size=args.validation_size,
        iid_test_size=args.iid_test_size,
        scaffold_test_min=args.scaffold_test_min,
        scaffold_test_max=args.scaffold_test_max,
        expected_eligible_unique=args.expected_eligible_unique,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite frozen split: {args.output}")
        # Atomic no-clobber publication, including if another process creates
        # the destination between the explicit check and this operation.
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "payload_sha256": payload["payload_sha256"],
                "census": payload["census"],
                "partition_counts": {
                    name: len(rows) for name, rows in payload["partitions"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
