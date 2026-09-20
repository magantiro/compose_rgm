"""Reconstruct a launch source capsule from its manifest and the git object store.

A source capsule is a verbatim copy of every file a launch read, written beside the
run's diagnostics so the exact bytes are recoverable years later.  The copy is large
(12-17 MB each, and a campaign accumulates one per attempt), and it is redundant:
the manifest records `git_blob_oid` and `sha256` for every entry, so as long as each
blob is reachable from a ref the tree can be rebuilt exactly.

The manifests are therefore the committed artifact and the directories are not.  This
script is the other half of that decision -- without it the manifests would be a
promise nobody can cash.

    python3 tools/rebuild_source_capsule.py <manifest.json> --out <dir>
    python3 tools/rebuild_source_capsule.py <manifest.json> --verify <existing-capsule>

`--verify` checks an on-disk capsule against its manifest without writing anything,
which is how the no-commit decision was justified in the first place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _entries(manifest: Path) -> dict[str, dict]:
    payload = json.loads(manifest.read_text())
    payload = payload.get("payload", payload)
    return payload["files"]


def _blob(oid: str) -> bytes | None:
    got = subprocess.run(["git", "cat-file", "blob", oid], capture_output=True, check=False)
    return got.stdout if got.returncode == 0 else None


def rebuild(manifest: Path, out: Path) -> tuple[int, list[str]]:
    failures: list[str] = []
    written = 0
    for rel, meta in _entries(manifest).items():
        data = _blob(meta["git_blob_oid"])
        if data is None:
            failures.append(f"{rel}: blob {meta['git_blob_oid']} not in object store")
            continue
        if hashlib.sha256(data).hexdigest() != meta["sha256"]:
            failures.append(f"{rel}: blob content does not match recorded sha256")
            continue
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written += 1
    return written, failures


def verify(manifest: Path, capsule: Path) -> tuple[int, list[str]]:
    failures: list[str] = []
    checked = 0
    for rel, meta in _entries(manifest).items():
        path = capsule / rel
        if not path.exists():
            failures.append(f"{rel}: missing from capsule on disk")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            failures.append(f"{rel}: on-disk bytes differ from manifest sha256")
            continue
        checked += 1
    return checked, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, help="write the reconstructed capsule here")
    parser.add_argument("--verify", type=Path, help="check an existing capsule instead")
    args = parser.parse_args()

    if bool(args.out) == bool(args.verify):
        parser.error("pass exactly one of --out or --verify")

    if args.verify:
        count, failures = verify(args.manifest, args.verify)
        label = "verified"
    else:
        count, failures = rebuild(args.manifest, args.out)
        label = "written"

    print(f"{label}: {count} files")
    for line in failures[:20]:
        print(f"  FAIL {line}")
    if failures:
        print(f"  {len(failures)} failures")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
