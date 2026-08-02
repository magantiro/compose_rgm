"""Cross-producer source identity must use the nested canonical namespace."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from modal_apps import (
    build_editing_v2_semantic_p50_successor_cache_app as p50_cache_app,
)
from modal_apps import run_semantic_active8_decisions_app as active8_app

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
TREE = "b" * 40


def _clean_git(_root: Path, *arguments: str) -> str:
    return {
        ("rev-parse", "HEAD"): COMMIT,
        ("rev-parse", "HEAD^{tree}"): TREE,
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
        ("ls-files",): "\n".join(
            path.relative_to(_root).as_posix()
            for path in _root.rglob("*")
            if path.is_file() and ".git" not in path.parts
        ),
    }[arguments]


def _reseal(revision: dict[str, object], sha_function) -> None:
    body = dict(revision)
    body.pop("source_revision_sha256", None)
    revision["source_revision_sha256"] = sha_function(body)


def test_real_producer_attestations_share_only_nested_execution_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(active8_app, "_git", _clean_git)
    monkeypatch.setattr(p50_cache_app, "_git", _clean_git)

    active8 = active8_app.local_source_revision(
        expected_commit=COMMIT,
        repo_root=ROOT,
    )
    cache = p50_cache_app.local_source_revision(
        expected_commit=COMMIT,
        repo_root=ROOT,
    )

    assert active8["source_revision_sha256"] != cache["source_revision_sha256"]
    assert active8["execution_source_revision"] == cache["execution_source_revision"]
    assert active8["execution_source_revision"]["commit"] == COMMIT
    assert active8["execution_source_revision"]["tree"] == TREE

    active8_app._validate_remote_source_revision(
        active8,
        loaded=active8_app._imports(ROOT),
        remote_root=ROOT,
    )
    assert p50_cache_app._validate_source_revision(cache, remote_root=ROOT) == cache


def test_v1_mixed_namespace_and_serialized_byte_forgery_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(active8_app, "_git", _clean_git)
    monkeypatch.setattr(p50_cache_app, "_git", _clean_git)
    active8 = active8_app.local_source_revision(
        expected_commit=COMMIT,
        repo_root=ROOT,
    )
    cache = p50_cache_app.local_source_revision(
        expected_commit=COMMIT,
        repo_root=ROOT,
    )

    legacy = copy.deepcopy(cache)
    legacy.pop("execution_source_revision")
    legacy["schema_version"] = 1
    _reseal(legacy, p50_cache_app._sha)
    with pytest.raises(RuntimeError, match="execution-source|serialized source"):
        p50_cache_app._validate_source_revision(legacy, remote_root=ROOT)

    mixed = copy.deepcopy(cache)
    mixed["execution_source_revision"] = active8
    _reseal(mixed, p50_cache_app._sha)
    with pytest.raises(RuntimeError, match="execution-source"):
        p50_cache_app._validate_source_revision(mixed, remote_root=ROOT)

    forged = copy.deepcopy(cache)
    first = next(iter(forged["serialized_source_hashes"]))
    forged["serialized_source_hashes"][first] = "f" * 64
    forged["serialized_source_hashes_sha256"] = p50_cache_app._sha(
        forged["serialized_source_hashes"]
    )
    _reseal(forged, p50_cache_app._sha)
    with pytest.raises(RuntimeError, match="serialized source"):
        p50_cache_app._validate_source_revision(forged, remote_root=ROOT)
