"""Offline input gates, generation configuration and output preservation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from compose_v4.experiments.fragments.assets import (
    child_path,
    install_asset,
    load_registry,
    verify_assets,
)
from compose_v4.experiments.fragments.errors import EvidenceError
from compose_v4.experiments.fragments.generation import (
    load_settings,
    run_generation,
    source_revision,
)
from compose_v4.experiments.fragments.generation_worker import environment, seed_for

ROOT = Path(__file__).resolve().parents[1]
TASKS = list(load_registry(ROOT)["task_assets"])


@pytest.fixture
def small_registry():
    return {
        "assets": {
            "checkpoint": {
                "path": "checkpoint.pt",
                "sha256": hashlib.sha256(b"weights").hexdigest(),
            }
        },
        "task_assets": {"motif_extension": ["checkpoint"]},
    }


def test_asset_copy_preserves_source_and_is_idempotent(tmp_path, small_registry):
    source = tmp_path / "original.pt"
    source.write_bytes(b"weights")
    target = install_asset(small_registry, tmp_path / "assets", "checkpoint", source)
    assert source.read_bytes() == target.read_bytes() == b"weights"
    assert source.stat().st_ino != target.stat().st_ino
    assert install_asset(small_registry, tmp_path / "assets", "checkpoint", source) == target
    assert verify_assets(small_registry, tmp_path / "assets")["checkpoint"] == target


def test_wrong_asset_cannot_replace_existing_file(tmp_path, small_registry):
    source, assets = tmp_path / "source", tmp_path / "assets"
    source.write_bytes(b"weights")
    assets.mkdir()
    (assets / "checkpoint.pt").write_bytes(b"user data")
    with pytest.raises(FileExistsError):
        install_asset(small_registry, assets, "checkpoint", source)
    assert (assets / "checkpoint.pt").read_bytes() == b"user data"
    source.write_bytes(b"wrong weights")
    with pytest.raises(EvidenceError, match="wrong source hash"):
        install_asset(small_registry, tmp_path / "other", "checkpoint", source)
    assert not (tmp_path / "other").exists()


def test_missing_asset_names_the_expected_identity(tmp_path, small_registry):
    with pytest.raises(EvidenceError, match=small_registry["assets"]["checkpoint"]["sha256"]):
        verify_assets(small_registry, tmp_path)
    with pytest.raises(EvidenceError, match="unknown fragment task"):
        verify_assets(small_registry, tmp_path, "development")


@pytest.mark.parametrize("name", ["../escape", "/absolute", "x/../../escape", ""])
def test_relative_paths_fail_closed(tmp_path, name):
    with pytest.raises(EvidenceError, match="unsafe"):
        child_path(tmp_path, name)


def test_symlink_escape_fails(tmp_path):
    inside = tmp_path / "inside"
    inside.mkdir()
    (inside / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(EvidenceError, match="symlink"):
        child_path(inside, "link/file")


@pytest.mark.parametrize("task", TASKS)
def test_generation_settings_cover_declared_tasks(task):
    settings = load_settings(ROOT, task)
    assert settings["tasks"][task]["seeds"]
    if task.startswith("superstructure_"):
        for field in ("sampler_config", "attachment_control"):
            assert isinstance(settings["tasks"][task][field], dict)
            assert settings["tasks"][task][field]


@pytest.mark.parametrize(
    "field,value",
    [("schema", "unknown"), ("max_attempts_per_prompt_seed", True), ("prompts", ["X", "X"])],
)
def test_invalid_generation_settings_fail(tmp_path, field, value):
    target = tmp_path / "experiments/fragments"
    target.mkdir(parents=True)
    settings = load_settings(ROOT, "motif_extension")
    settings[field] = value
    (target / "generation.json").write_text(json.dumps(settings))
    with pytest.raises(EvidenceError, match="invalid"):
        load_settings(tmp_path, "motif_extension")


def test_generation_refuses_existing_output_before_sampling(tmp_path, monkeypatch):
    root = tmp_path / "source"
    target = root / "experiments/fragments"
    target.mkdir(parents=True)
    for name in ("assets.json", "generation.json"):
        shutil.copyfile(ROOT / "experiments/fragments" / name, target / name)
    monkeypatch.setattr(
        "compose_v4.experiments.fragments.generation.verify_assets", lambda *args, **kwargs: {}
    )
    existing = tmp_path / "result"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        run_generation(
            root=root,
            assets=tmp_path,
            task="motif_extension",
            output=existing,
            python=Path(sys.executable),
            attempts=1,
        )


def test_generation_without_metrics_needs_no_evaluator(tmp_path, small_registry):
    small_registry["assets"]["evaluator_metrics"] = {
        "path": "missing.py",
        "sha256": "0" * 64,
    }
    (tmp_path / "checkpoint.pt").write_bytes(b"weights")
    paths = verify_assets(small_registry, tmp_path, "motif_extension", include_evaluator=False)
    assert set(paths) == {"checkpoint"}
    with pytest.raises(EvidenceError, match="missing evaluator_metrics"):
        verify_assets(small_registry, tmp_path, "motif_extension")


def test_environment_drift_fails_before_model_import(monkeypatch):
    monkeypatch.setattr(
        "compose_v4.experiments.fragments.generation_worker.version", lambda _: "wrong"
    )
    with pytest.raises(ValueError, match="wrong rdkit version"):
        environment({"rdkit": "2024.3.5"})


def test_seed_derivation_is_stable():
    assert seed_for("BARICITINIB", "superstructure_generation", 0) == 722072583


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_generation_refuses_existing_output(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(
        "compose_v4.experiments.fragments.generation.verify_assets", lambda *args, **kwargs: {}
    )
    target = tmp_path / "result"
    if kind == "file":
        target.write_text("user-owned")
    elif kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        run_generation(
            root=ROOT,
            assets=tmp_path,
            task="motif_extension",
            output=target,
            python=Path(sys.executable),
            attempts=1,
        )
    if kind == "file":
        assert target.read_text() == "user-owned"


def test_worker_failure_keeps_diagnostics_but_does_not_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "compose_v4.experiments.fragments.generation.verify_assets", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        "compose_v4.experiments.fragments.generation.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 9, "", ""),
    )
    target = tmp_path / "result"
    with pytest.raises(EvidenceError, match="worker exited 9"):
        run_generation(
            root=ROOT,
            assets=tmp_path,
            task="motif_extension",
            output=target,
            python=Path(sys.executable),
            attempts=1,
        )
    assert not target.exists()
    pending = list(tmp_path.glob(".result.pending-*"))
    assert len(pending) == 1
    assert json.loads((pending[0] / "provenance.json").read_text())["status"] == "failed"
    assert (pending[0] / "worker.log").exists()
    assert not (tmp_path / ".result.lock").exists()


def test_plain_source_export_cannot_inherit_enclosing_git_revision(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("must not search enclosing repositories for a plain source export")

    monkeypatch.setattr("compose_v4.experiments.fragments.generation.subprocess.run", forbidden)
    assert source_revision(tmp_path) is None
