"""Verify the submitted PDF and pinned result-artifact identities.

This is an identity audit, not a table reducer or a scientific-run replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = "compose_submitted_paper_lineage_v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")


class LineageError(ValueError):
    """An identity manifest or one of its pinned sources is invalid."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_manifest(manifest: object) -> dict:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise LineageError(f"manifest.schema_version must be {SCHEMA_VERSION!r}")
    paper = manifest.get("paper")
    if not isinstance(paper, dict) or not SHA256.fullmatch(str(paper.get("sha256", ""))):
        raise LineageError("manifest.paper.sha256 must be a lowercase SHA-256 digest")
    if not isinstance(paper.get("pages"), int) or paper["pages"] <= 0:
        raise LineageError("manifest.paper.pages must be a positive integer")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise LineageError("manifest.sources must be a nonempty list")
    seen: set[str] = set()
    for index, source in enumerate(sources):
        field = f"manifest.sources[{index}]"
        if not isinstance(source, dict):
            raise LineageError(f"{field} must be an object")
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id or source_id in seen:
            raise LineageError(f"{field}.id must be a unique nonempty string")
        seen.add(source_id)
        if not REVISION.fullmatch(str(source.get("artifact_revision", ""))):
            raise LineageError(f"{field}.artifact_revision must be a full lowercase Git SHA")
        if not SHA256.fullmatch(str(source.get("artifact_sha256", ""))):
            raise LineageError(f"{field}.artifact_sha256 must be a lowercase SHA-256 digest")
        raw_path = source.get("artifact_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise LineageError(f"{field}.artifact_path must be a nonempty relative path")
        path = PurePosixPath(raw_path)
        if path.is_absolute() or ".." in path.parts or str(path) != raw_path:
            raise LineageError(f"{field}.artifact_path is not a safe normalized relative path")
    return manifest


def _git_source(root: Path, source: dict) -> bytes:
    object_name = f"{source['artifact_revision']}:{source['artifact_path']}"
    result = subprocess.run(
        ["git", "-C", str(root), "show", object_name],
        check=False,
        capture_output=True,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise LineageError(f"{object_name}: {detail or 'Git object unavailable'}")
    return result.stdout


def verify_manifest(
    manifest: object,
    paper_bytes: bytes,
    source_reader: Callable[[dict], bytes],
) -> dict:
    """Return one explicit identity result per input; never substitute a source."""
    document = _validate_manifest(manifest)
    paper_expected = document["paper"]["sha256"]
    paper_actual = _sha256(paper_bytes)
    source_checks = []
    for source in document["sources"]:
        try:
            actual = _sha256(source_reader(source))
        except LineageError as exc:
            source_checks.append({"id": source["id"], "status": "MISSING", "detail": str(exc)})
            continue
        source_checks.append(
            {
                "id": source["id"],
                "status": "MATCH" if actual == source["artifact_sha256"] else "MISMATCH",
                "expected_sha256": source["artifact_sha256"],
                "actual_sha256": actual,
            }
        )
    paper_status = "MATCH" if paper_actual == paper_expected else "MISMATCH"
    return {
        "schema_version": "compose_submitted_paper_identity_check_v1",
        "scope": "PDF and historical Git artifact byte identity only; no metric or experiment replay",
        "paper": {
            "status": paper_status,
            "expected_sha256": paper_expected,
            "actual_sha256": paper_actual,
        },
        "sources": source_checks,
        "status": "PASS"
        if paper_status == "MATCH" and all(row["status"] == "MATCH" for row in source_checks)
        else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, required=True, help="Exact submitted PDF to check")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root"
    )
    args = parser.parse_args()
    manifest_path = args.root / "experiments/paper/manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paper_bytes = args.paper.read_bytes()
        report = verify_manifest(manifest, paper_bytes, lambda row: _git_source(args.root, row))
    except (OSError, json.JSONDecodeError, LineageError) as exc:
        parser.exit(2, f"submission-lineage input error: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
