"""Recompute every SHA-256 in the Claim-2 handoff manifest from the real files.

This exists because a digest in ``handoff.json`` was once recorded with a real
16-character prefix and an invented remainder. Handoff acceptance gates 2 and 3
are "handoff.json validates" and "all frozen-input hashes match", so a wrong
digest is worse than a missing one: it looks verified.

Transcribing hashes by hand is the failure mode, so it is removed rather than
warned about. ``tests/test_claim2_handoff_manifest.py`` then proves the manifest
matches the tree; this script is how you make it match after editing a tracked
artifact.

    python3 scripts/claim2_refresh_handoff_manifest.py            # rewrite
    python3 scripts/claim2_refresh_handoff_manifest.py --check    # CI mode

``--check`` writes nothing and exits non-zero if any digest is stale, so it can
gate a handoff without being able to paper over one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MANIFEST = Path("docs/workstreams/claim2-trajectory/handoff.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale digests and exit non-zero; write nothing",
    )
    args = parser.parse_args()

    path = args.repo_root / args.manifest
    manifest = json.loads(path.read_text())

    stale: list[tuple[str, str, str]] = []
    missing: list[str] = []
    for section in ("artifacts_created", "frozen_inputs"):
        for entry in manifest.get(section, []):
            relative = entry.get("path")
            if not relative:
                continue
            target = args.repo_root / relative
            if not target.exists():
                missing.append(relative)
                continue
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != entry.get("sha256"):
                stale.append((relative, str(entry.get("sha256")), digest))
                entry["sha256"] = digest

    for relative in missing:
        print(f"MISSING  {relative} is declared in the manifest but not on disk")
    for relative, old, new in stale:
        print(f"STALE    {relative}\n         {old[:16]}... -> {new[:16]}...")

    if missing:
        print(f"\n{len(missing)} declared artifact(s) missing; not writing.")
        return 2
    if args.check:
        print(f"\n{len(stale)} stale digest(s)." if stale else "\nall digests current.")
        return 1 if stale else 0
    if stale:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nrewrote {args.manifest} with {len(stale)} corrected digest(s)")
    else:
        print("\nall digests current; nothing to do")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
