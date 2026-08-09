"""Census the COMPILED Process-V2 corpus where it lives, on the volume.

WHY REMOTE
----------
The corpus has two halves: a local directory and the Modal volume. The volume
half is ~21.8k entries whose ``BATCH.pt`` tensors are ~92 kB each -- about 2 GB
that a census does not need and should not move. ``modal volume get`` also
declines to recurse a directory (it writes nothing and leaves empty dirs
behind), so pulling it would mean one CLI call per file across ~1,450 slices.

Scanning in place reads only ``ENTRIES.json`` and returns aggregates plus the
canonical (source, successor) digests needed to deduplicate ACROSS the two
halves. Digests rather than keys: 16 bytes each instead of two SMILES strings,
which is the difference between a ~0.7 MB return and a ~30 MB one.

WHY IT REPORTS partition_role
-----------------------------
Six held-out tasks were once queued as train. They failed closed, but a census
that cannot see role would not have been able to say so. Every entry's role is
counted, so "train only" is an observation here rather than an assumption.
"""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path, PurePosixPath

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = PurePosixPath("/root/compose")
ARTIFACT_ROOT = PurePosixPath("/artifacts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONUNBUFFERED": "1"})
)

app = modal.App("compose-v4-scan-process-v2-corpus")
artifact_volume = modal.Volume.from_name("compose-v4-artifacts", create_if_missing=False)


@app.function(
    image=image,
    cpu=2.0,
    memory=8 * 1024,
    timeout=30 * 60,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def scan_corpus(corpus_root: str) -> dict:
    """Aggregate every published slice under ``corpus_root/chunks``."""

    artifact_volume.reload()
    chunks = Path(corpus_root) / "chunks"
    families: collections.Counter = collections.Counter()
    cells: collections.Counter = collections.Counter()
    roles: collections.Counter = collections.Counter()
    per_task: collections.Counter = collections.Counter()
    pair_digests: set[str] = set()
    source_digests: set[str] = set()
    entries = 0
    slices = 0
    missing_receipt = 0

    for task_dir in sorted(chunks.iterdir()):
        if not task_dir.is_dir():
            continue
        for slice_dir in sorted(task_dir.iterdir()):
            entries_path = slice_dir / "ENTRIES.json"
            if not (slice_dir / "RECEIPT.json").exists():
                # An unreceipted slice is an interrupted write, not corpus.
                missing_receipt += 1
                continue
            if not entries_path.exists():
                missing_receipt += 1
                continue
            slices += 1
            payload = json.loads(entries_path.read_text())
            for entry in payload["entries"]:
                entries += 1
                families[str(entry["model_family"])] += 1
                cells[str(entry.get("capability_cell_id", "?"))] += 1
                roles[str(entry.get("partition_role", "?"))] += 1
                per_task[task_dir.name] += 1
                source = str(entry["source_state_sha256"])
                successor = str(entry["successor_canonical_key"])
                pair_digests.add(
                    hashlib.blake2b(
                        f"{source}\t{successor}".encode(), digest_size=16
                    ).hexdigest()
                )
                source_digests.add(source)

    return {
        "corpus_root": str(corpus_root),
        "slices": slices,
        "slices_without_receipt_or_entries": missing_receipt,
        "entries": entries,
        "families": dict(families.most_common()),
        "capability_cells": dict(cells.most_common()),
        "partition_roles": dict(roles),
        "tasks": len(per_task),
        "distinct_pair_digests": sorted(pair_digests),
        "distinct_source_digests": len(source_digests),
    }


@app.function(
    image=image,
    cpu=2.0,
    memory=8 * 1024,
    timeout=60 * 60,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def archive_corpus(corpus_root: str, archive_path: str) -> dict:
    """Tar a corpus into ONE file on the volume so it can be pulled down.

    ``modal volume get`` does not recurse a directory -- it writes nothing and
    leaves empty directories behind -- so backing up ~1,450 slices would
    otherwise mean ~4,350 CLI calls. One archive is one call.

    The uncompressed tar is deliberate: ``BATCH.pt`` tensors dominate the bytes
    and do not compress usefully, so gzip would cost minutes of CPU to save
    little, and a corrupt gzip loses everything while a truncated tar loses only
    its tail.
    """

    import hashlib
    import tarfile

    artifact_volume.reload()
    root = Path(corpus_root)
    target = Path(archive_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    files = 0
    with tarfile.open(target, "w") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(root)))
                files += 1
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    artifact_volume.commit()
    return {
        "archive_path": str(target),
        "files": files,
        "bytes": target.stat().st_size,
        "sha256": digest.hexdigest(),
    }


@app.local_entrypoint()
def archive(
    corpus_root: str = "/artifacts/editing_v2/process_v2_v2_corpus",
    archive_path: str = "/artifacts/editing_v2/backups/process_v2_v2_corpus.tar",
) -> None:
    result = archive_corpus.remote(corpus_root, archive_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nnow: modal volume get compose-v4-artifacts {archive_path.lstrip('/')} <dest>")


@app.local_entrypoint()
def main(
    corpus_root: str = "/artifacts/editing_v2/process_v2_v2_corpus",
    out: str = "",
) -> None:
    result = scan_corpus.remote(corpus_root)
    digests = result.pop("distinct_pair_digests")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"distinct canonical pairs: {len(digests):,}")
    if out:
        Path(out).write_text(
            json.dumps({**result, "distinct_pair_digests": digests}, sort_keys=True)
        )
        print(f"wrote {out}")
