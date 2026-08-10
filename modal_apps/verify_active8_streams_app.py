"""Compare the volume's Active8 transition streams against the local copy.

WHY THIS EXISTS
---------------
The gate-zero decisions disagree on ``source_index_sha256``: the local Active8
computes 2dd60909 while the volume's Active8 -- the one the corpus was compiled
under -- matches the v6 decision. Both report 270 eligible shards, so it is not
a difference in how many tasks exist.

That matters beyond gate-zero. The evaluation panel's role gate, which caught 3
sources belonging to final_test, read ``partition_role`` from
``tasks/*/transitions.jsonl.gz`` in the LOCAL copy. If those streams differ
between local and volume, the panel's held-out claim rests on the wrong
assignment.

``source_index_sha256`` covers decision shards, not transition streams, so the
two can legitimately differ while the roles agree -- but that is an argument,
and the panel's provenance should not rest on one. This computes the same
digest over the volume's streams so the question is answered by comparison.

CPU only. It opens no molecular cache and needs no accelerator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
app = modal.App("compose-v4-verify-active8-streams")


@app.function(
    image=image,
    cpu=2.0,
    memory=8 * 1024,
    timeout=20 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def digest_active8_streams(
    active8_root: str = (
        "/artifacts/editing_v2/process_v2_active8/"
        "8ecc0e5e825a15200560c58960d4f662c9ca23be785c825c9c24aa73308144bb"
    ),
) -> dict[str, Any]:
    """Digest every transition stream, and the per-role source sets."""

    import collections
    import gzip

    artifact_volume.reload()
    root = Path(active8_root)
    streams = sorted((root / "tasks").rglob("transitions.jsonl.gz"))

    aggregate = hashlib.sha256()
    total_bytes = 0
    by_role: dict[str, set[str]] = collections.defaultdict(set)
    for path in streams:
        payload = path.read_bytes()
        total_bytes += len(payload)
        aggregate.update(path.name.encode())
        aggregate.update(hashlib.sha256(payload).digest())
        with gzip.open(path, "rt") as handle:
            for line in handle:
                record = json.loads(line)
                evidence = record["candidate_evidence"]
                if evidence.get("exclusion_reason") is not None:
                    continue
                by_role[str(record.get("partition_role", "?"))].add(
                    str(evidence["source_canonical_key"])
                )

    # A digest per role as well as the byte digest: if the bytes differ for an
    # innocuous reason (ordering, compression level) the roles can still be
    # shown identical, which is the property the panel actually depends on.
    role_digests = {
        role: hashlib.sha256(
            json.dumps(sorted(sources), sort_keys=True).encode()
        ).hexdigest()
        for role, sources in sorted(by_role.items())
    }
    return {
        "schema": "compose.editing_v2.active8_stream_digest",
        "status": "STREAM_DIGEST_EVIDENCE_ONLY_NO_AUTHORITY",
        "active8_root": str(root),
        "stream_count": len(streams),
        "stream_bytes": total_bytes,
        "aggregate_sha256": aggregate.hexdigest(),
        "sources_per_role": {role: len(v) for role, v in sorted(by_role.items())},
        "role_source_digests": role_digests,
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(digest_active8_streams.remote(), indent=2, sort_keys=True))
