#!/usr/bin/env python3
"""Audit fast-path assumptions against an authenticated exhaustive T1 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rdkit
import torch

from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
    validate_process_v2_t1_prepared_inputs,
)
from compose_v4.experiments.editing_v2_semantic_t1_prepared_inputs import (
    compiled_successor_map_from_payload,
)
from compose_v4.experiments.factorized_successor_training import (
    teacher_successor_fiber_from_exact_digest,
)

_SCHEMA = "compose.editing_v2.successor_fiber_fastpath_oracle_audit"
_SCHEMA_VERSION = 1


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_revision(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 40 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("source revision must be a full lowercase Git commit")
    return normalized


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def audit(input_path: Path, *, source_revision: str) -> dict[str, object]:
    raw = input_path.read_bytes()
    implementation_path = Path(__file__).resolve()
    implementation_bytes = implementation_path.read_bytes()
    value = json.loads(raw)
    prepared = validate_process_v2_t1_prepared_inputs(value)
    family_entries: Counter[str] = Counter()
    family_teacher_aliases: Counter[str] = Counter()
    cross_family_entries: list[str] = []
    virtual_entries: list[str] = []
    virtual_by_family: Counter[str] = Counter()
    raw_mark_counts: list[int] = []
    canonical_successor_counts: list[int] = []
    teacher_alias_counts: list[int] = []
    for entry in prepared["entries"]:
        family = str(entry["model_family"])
        partition = compiled_successor_map_from_payload(entry["successor_partition"])
        teacher = teacher_successor_fiber_from_exact_digest(
            partition, str(entry["target_state_sha256"])
        )
        family_entries[family] += 1
        family_teacher_aliases[family] += len(teacher.aliases)
        raw_mark_counts.append(int(entry["raw_mark_count"]))
        canonical_successor_counts.append(int(entry["canonical_successor_count"]))
        teacher_alias_counts.append(len(teacher.aliases))
        if any(alias.family_name != family for alias in teacher.aliases):
            cross_family_entries.append(str(entry["panel_entry_sha256"]))
        if partition.state_support.virtual_aliases:
            virtual_entries.append(str(entry["panel_entry_sha256"]))
            virtual_by_family.update(
                alias.family_name for alias in partition.state_support.virtual_aliases
            )
    n = len(prepared["entries"])
    body: dict[str, object] = {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "status": "MEASURED_EXHAUSTIVE_PANEL_ORACLE_AUDIT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": _require_revision(source_revision),
        "implementation": {
            "path": str(implementation_path),
            "file_sha256": _sha256(implementation_bytes),
        },
        "input": {
            "path": str(input_path.resolve()),
            "file_sha256": _sha256(raw),
            "bytes": len(raw),
            "artifact_sha256": prepared["artifact_sha256"],
            "process_identity_sha256": prepared["process_identity_sha256"],
            "panel_sha256": prepared["panel_sha256"],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "rdkit": rdkit.__version__,
        },
        "sample": {
            "entry_count": n,
            "family_entry_counts": dict(sorted(family_entries.items())),
            "raw_mark_count_min": min(raw_mark_counts),
            "raw_mark_count_max": max(raw_mark_counts),
            "raw_mark_count_mean": sum(raw_mark_counts) / n,
            "canonical_successor_count_min": min(canonical_successor_counts),
            "canonical_successor_count_max": max(canonical_successor_counts),
            "canonical_successor_count_mean": sum(canonical_successor_counts) / n,
        },
        "teacher_fibers": {
            "alias_count_min": min(teacher_alias_counts),
            "alias_count_max": max(teacher_alias_counts),
            "alias_count_mean": sum(teacher_alias_counts) / n,
            "aliases_by_declared_family": dict(sorted(family_teacher_aliases.items())),
            "cross_family_entry_count": len(cross_family_entries),
            "cross_family_entry_sha256s": sorted(cross_family_entries),
        },
        "canonical_self_events": {
            "entry_count": len(virtual_entries),
            "entry_sha256s": sorted(virtual_entries),
            "alias_count_by_family": dict(sorted(virtual_by_family.items())),
        },
        "claim_boundary": {
            "measured": (
                "Every complete successor partition in this frozen 512-state T1 panel was "
                "revalidated before the alias audit."
            ),
            "not_claimed": (
                "Zero observed cross-family or self aliases on this bounded panel is not a "
                "universal proof for every molecule in declared support."
            ),
        },
    }
    return {**body, "audit_sha256": _sha256(json.dumps(body, sort_keys=True).encode())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    arguments = parser.parse_args()
    result = audit(arguments.input, source_revision=arguments.source_revision)
    _atomic_write(arguments.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
