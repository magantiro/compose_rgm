from __future__ import annotations

import hashlib
import json
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


def test_panel_surface_delegates_to_native_interface_and_reopens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    active8 = artifact_root / "active8" / ("a" * 64)
    gate_zero = artifact_root / "gate0" / ("b" * 64) / "DECISION.json"
    prefix = artifact_root / "t1_panel"
    active8.mkdir(parents=True)
    gate_zero.parent.mkdir(parents=True)
    gate_zero.write_text("{}\n", encoding="utf-8")
    panel_sha = "c" * 64
    panel = {
        "panel_sha256": panel_sha,
        "family_counts": {"atom_insert": 64},
        "capability_cell_counts": {"cell": 64},
        "entries": [{"entry": 1}],
    }
    source_token = object()
    events: list[str] = []

    def open_source(active8_root: Path, **kwargs: object) -> object:
        assert active8_root == active8
        assert kwargs == {
            "gate_zero_decision_path": gate_zero,
            "artifact_root": artifact_root,
            "repo_root": tmp_path,
        }
        events.append("open")
        return source_token

    def build_panel(observed: object) -> dict:
        assert observed is source_token
        events.append("build")
        return panel

    def write_panel(value: dict, *, output_root: Path, source: object) -> Path:
        assert value == panel
        assert source is source_token
        events.append("write")
        path = output_root / "PROCESS_V2_T1_PANEL.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def load_panel(path: Path, *, source: object) -> dict:
        assert source is source_token
        assert path.is_file()
        events.append("load")
        return panel

    loaded = {
        "PANEL_FILENAME": "PROCESS_V2_T1_PANEL.json",
        "canonical_bytes": lambda value: json.dumps(value, sort_keys=True).encode(),
        "open_source": open_source,
        "build_panel": build_panel,
        "write_panel": write_panel,
        "load_panel": load_panel,
    }
    revision = {"image_revision_sha256": "d" * 64}
    monkeypatch.setattr(launcher, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(launcher, "_validate_remote_revision", lambda *_args, **_kwargs: None)

    result = launcher._materialize_panel(
        image_revision=revision,
        active8_run_root=str(active8),
        gate_zero_decision=str(gate_zero),
        output_prefix=str(prefix),
        remote_root=tmp_path,
        artifact_root=artifact_root,
        reload_volume=lambda: events.append("reload"),
        commit_volume=lambda: events.append("commit"),
        loaded=loaded,
    )

    assert events == ["reload", "open", "build", "write", "commit", "reload", "load"]
    assert result["panel_sha256"] == panel_sha
    assert result["entry_count"] == 1
    assert result["training_authorized"] is False
    assert result["t1_authorized"] is False
    assert result["bounded_p50_authorized"] is False


def test_panel_surface_refuses_writer_path_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    active8 = artifact_root / "active8"
    gate_zero = artifact_root / "gate0" / "DECISION.json"
    active8.mkdir(parents=True)
    gate_zero.parent.mkdir(parents=True)
    gate_zero.write_text("{}\n", encoding="utf-8")
    panel = {
        "panel_sha256": "c" * 64,
        "family_counts": {},
        "capability_cell_counts": {},
        "entries": [],
    }
    monkeypatch.setattr(launcher, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(launcher, "_validate_remote_revision", lambda *_args, **_kwargs: None)
    loaded = {
        "PANEL_FILENAME": "PROCESS_V2_T1_PANEL.json",
        "canonical_bytes": lambda value: b"same",
        "open_source": lambda *_args, **_kwargs: object(),
        "build_panel": lambda _source: panel,
        "write_panel": lambda *_args, **_kwargs: artifact_root / "substituted.json",
        "load_panel": lambda *_args, **_kwargs: panel,
    }

    with pytest.raises(RuntimeError, match="another path"):
        launcher._materialize_panel(
            image_revision={"image_revision_sha256": "d" * 64},
            active8_run_root=str(active8),
            gate_zero_decision=str(gate_zero),
            output_prefix=str(artifact_root / "panel"),
            remote_root=tmp_path,
            artifact_root=artifact_root,
            reload_volume=lambda: None,
            commit_volume=lambda: None,
            loaded=loaded,
        )


def test_panel_modal_surface_has_bounded_cpu_geometry() -> None:
    source = (launcher.ROOT / launcher.LAUNCHER_SOURCE).read_text(encoding="utf-8")
    decorator = source[
        source.index("@app.function(", source.index("def _materialize_panel")) : source.index(
            "def materialize_process_v2_t1_panel"
        )
    ]
    assert "cpu=PANEL_CPU" in decorator
    assert "memory=PANEL_MEMORY_MB" in decorator
    assert "timeout=PANEL_TIMEOUT_SECONDS" in decorator
    assert "max_containers=1" in decorator
    assert "gpu=" not in decorator
