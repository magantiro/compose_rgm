#!/usr/bin/env python3
"""Audit exact packed editing paths against the protected-charge contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compose_v4.data.packed_charge_policy_audit import (  # noqa: E402
    audit_packed_charge_policy,
    resolve_unified_manifest_shards,
)


def _atomic_json_write(path: Path, payload: object) -> None:
    content = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            os.fchmod(handle.fileno(), 0o644)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream every exact consecutive state in a unified packed editing "
            "corpus and report charge-policy violations."
        )
    )
    parser.add_argument("--unified-manifest", type=Path, required=True)
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=None,
        help="Override manifest roots.audit_layers (useful after artifact transfer).",
    )
    parser.add_argument(
        "--mmp-root",
        type=Path,
        default=None,
        help="Override manifest roots.mmp_layer (useful after artifact transfer).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclusions-output",
        type=Path,
        default=None,
        help=(
            "Optionally write the report's hashed exact-entry exclusion payload "
            "as a standalone JSON artifact."
        ),
    )
    parser.add_argument("--max-examples", type=int, default=32)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest, shards = resolve_unified_manifest_shards(
        args.unified_manifest,
        audit_root=args.audit_root,
        mmp_root=args.mmp_root,
    )
    completed = 0

    def progress(shard: dict[str, object]) -> None:
        nonlocal completed
        completed += 1
        counts = shard["counts"]
        print(
            f"[{completed:02d}/{len(shards):02d}] "
            f"{shard['manifest_layer']}/{shard['partition']}/"
            f"{shard['relative_path']}: "
            f"{counts['entries']} traces, "
            f"{counts['transitions']} transitions, "
            f"{counts['transitions_with_violations']} violating transitions",
            flush=True,
        )

    report = audit_packed_charge_policy(
        shards,
        max_examples=args.max_examples,
        source_manifest_path=args.unified_manifest,
        source_manifest=manifest,
        progress_callback=progress,
    )
    _atomic_json_write(args.output, report)
    if args.exclusions_output is not None:
        _atomic_json_write(args.exclusions_output, report["exclusion_payload"])
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": report["counts"],
                "violating_transitions_by_type": report[
                    "violating_transitions_by_type"
                ],
                "output": str(args.output),
                "exclusions_output": (
                    None
                    if args.exclusions_output is None
                    else str(args.exclusions_output)
                ),
                "exclusion_payload_sha256": report["exclusion_payload"][
                    "payload_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
