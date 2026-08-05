from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from modal_apps import run_process_v2_t1_app as launcher


def _fixture_tree(root: Path) -> tuple[str, ...]:
    paths = (
        launcher.LAUNCHER_SOURCE,
        "src/compose_v4/module.py",
        "configs/policy.json",
    )
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n", encoding="utf-8")
    return paths


def _revision(root: Path, *, commit: str = "1" * 40, tree: str = "2" * 40) -> dict:
    sources = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in launcher._serialized_source_paths(root)
    }
    body = {
        "schema": launcher.REVISION_SCHEMA,
        "schema_version": launcher.REVISION_SCHEMA_VERSION,
        "commit": commit,
        "tree": tree,
        "worktree_clean": True,
        "serialized_sources": sources,
    }
    return {**body, "image_revision_sha256": launcher._sha256(body)}


def test_local_revision_binds_clean_commit_and_every_serialized_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracked = _fixture_tree(tmp_path)
    answers = {
        ("rev-parse", "HEAD"): "1" * 40,
        ("rev-parse", "HEAD^{tree}"): "2" * 40,
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
        ("ls-files",): "\n".join(tracked),
    }
    monkeypatch.setattr(launcher, "_git", lambda _root, *args: answers[args])

    revision = launcher.local_image_revision(
        expected_commit="1" * 40,
        repo_root=tmp_path,
    )

    assert revision["commit"] == "1" * 40
    assert revision["tree"] == "2" * 40
    assert set(revision["serialized_sources"]) == set(tracked)
    assert revision["image_revision_sha256"] == launcher._sha256(
        {key: value for key, value in revision.items() if key != "image_revision_sha256"}
    )


@pytest.mark.parametrize(
    ("expected_commit", "observed_commit", "dirty"),
    [
        ("x" * 40, "1" * 40, ""),
        ("1" * 40, "2" * 40, ""),
        ("1" * 40, "1" * 40, " M src/changed.py"),
    ],
)
def test_local_revision_refuses_malformed_wrong_or_dirty_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected_commit: str,
    observed_commit: str,
    dirty: str,
) -> None:
    tracked = _fixture_tree(tmp_path)
    answers = {
        ("rev-parse", "HEAD"): observed_commit,
        ("rev-parse", "HEAD^{tree}"): "2" * 40,
        ("status", "--porcelain=v1", "--untracked-files=all"): dirty,
        ("ls-files",): "\n".join(tracked),
    }
    monkeypatch.setattr(launcher, "_git", lambda _root, *args: answers[args])

    with pytest.raises(RuntimeError):
        launcher.local_image_revision(
            expected_commit=expected_commit,
            repo_root=tmp_path,
        )


def test_local_revision_refuses_untracked_serialized_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracked = _fixture_tree(tmp_path)
    answers = {
        ("rev-parse", "HEAD"): "1" * 40,
        ("rev-parse", "HEAD^{tree}"): "2" * 40,
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
        ("ls-files",): "\n".join(tracked[:-1]),
    }
    monkeypatch.setattr(launcher, "_git", lambda _root, *args: answers[args])

    with pytest.raises(RuntimeError, match="Git-tracked"):
        launcher.local_image_revision(expected_commit="1" * 40, repo_root=tmp_path)


def test_remote_revision_rehashes_every_serialized_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture_tree(tmp_path)
    revision = _revision(tmp_path)
    monkeypatch.setattr(launcher, "_VALIDATED_REVISION_SHA256", None)
    launcher._validate_remote_revision(revision, remote_root=tmp_path)

    changed = tmp_path / "src/compose_v4/module.py"
    changed.write_text("changed\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "_VALIDATED_REVISION_SHA256", None)
    with pytest.raises(RuntimeError, match="source differs"):
        launcher._validate_remote_revision(revision, remote_root=tmp_path)


def test_remote_revision_refuses_changed_inventory_with_stale_self_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture_tree(tmp_path)
    revision = _revision(tmp_path)
    revision["serialized_sources"].pop("configs/policy.json")
    monkeypatch.setattr(launcher, "_VALIDATED_REVISION_SHA256", None)

    with pytest.raises(RuntimeError, match="image revision disagrees"):
        launcher._validate_remote_revision(revision, remote_root=tmp_path)


@pytest.mark.parametrize(
    "value",
    [
        "relative/path",
        "/tmp/outside",
        "/artifacts/../escape",
    ],
)
def test_artifact_paths_must_be_normalized_and_below_volume(value: str) -> None:
    with pytest.raises(ValueError):
        launcher._require_artifact_path(value, field="artifact")
