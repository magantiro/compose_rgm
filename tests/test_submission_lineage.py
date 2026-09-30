"""Identity checks for the submitted-paper evidence map."""

from __future__ import annotations

import hashlib

import pytest

from tools.verify_submission_lineage import LineageError, verify_manifest


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest() -> dict:
    return {
        "schema_version": "compose_submitted_paper_lineage_v1",
        "paper": {"sha256": _digest(b"paper"), "pages": 1},
        "sources": [
            {
                "id": "table",
                "artifact_revision": "a" * 40,
                "artifact_path": "results/table.json",
                "artifact_sha256": _digest(b"rows"),
            }
        ],
    }


def test_matches_only_exact_paper_and_artifact_bytes() -> None:
    manifest = _manifest()
    report = verify_manifest(manifest, b"paper", lambda _: b"rows")
    assert report["status"] == "PASS"
    assert report["paper"]["status"] == "MATCH"
    assert report["sources"][0]["status"] == "MATCH"

    changed_paper = verify_manifest(manifest, b"other paper", lambda _: b"rows")
    assert changed_paper["status"] == "FAIL"
    assert changed_paper["paper"]["status"] == "MISMATCH"

    changed_rows = verify_manifest(manifest, b"paper", lambda _: b"other rows")
    assert changed_rows["status"] == "FAIL"
    assert changed_rows["sources"][0]["status"] == "MISMATCH"


def test_missing_artifact_is_not_replaced() -> None:
    def missing(_: dict) -> bytes:
        raise LineageError("pinned Git object absent")

    report = verify_manifest(_manifest(), b"paper", missing)
    assert report["status"] == "FAIL"
    assert report["sources"][0]["status"] == "MISSING"


@pytest.mark.parametrize("path", ["/absolute.json", "../escaped.json", "a/../table.json"])
def test_rejects_unsafe_artifact_paths(path: str) -> None:
    manifest = _manifest()
    manifest["sources"][0]["artifact_path"] = path
    with pytest.raises(LineageError, match="artifact_path"):
        verify_manifest(manifest, b"paper", lambda _: b"rows")


def test_rejects_duplicate_ids() -> None:
    manifest = _manifest()
    manifest["sources"].append(dict(manifest["sources"][0]))
    with pytest.raises(LineageError, match="unique"):
        verify_manifest(manifest, b"paper", lambda _: b"rows")
