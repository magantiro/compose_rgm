"""Offline runtime packaging, input gates, and preservation invariants."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from compose_v4.experiments.fragments.assets import (
    child_path,
    install_asset,
    load_registry,
    sha256,
    verify_assets,
)
from compose_v4.experiments.fragments.evidence import EvidenceError
from compose_v4.experiments.fragments.generation import (
    check_parity,
    fixtures,
    run_generation,
    source_revision,
    verify_runtime,
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
def test_preserved_runtime_and_fixture_hashes(task):
    archive, record = verify_runtime(ROOT, task)
    assert archive.stat().st_size < 1_000_000
    assert record["files"] and len(record["source_revision"]) == 40
    assert fixtures(ROOT, task)
    if task == "motif_extension":
        assert "src/compose_v4/benchmark/joint_completion_prior.py" in record["files"]


def test_archive_drift_rejected(tmp_path):
    task = "superstructure_generation"
    target = tmp_path / "experiments/fragments/runtime" / task
    shutil.copytree(ROOT / "experiments/fragments/runtime" / task, target)
    with (target / "source.zip").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(EvidenceError, match="archive hash mismatch"):
        verify_runtime(tmp_path, task)


def test_malicious_archive_path_rejected_even_with_consistent_hashes(tmp_path):
    task = "motif_extension"
    target = tmp_path / "experiments/fragments/runtime" / task
    target.mkdir(parents=True)
    with zipfile.ZipFile(target / "source.zip", "w") as bundle:
        bundle.writestr("../escape.py", b"bad")
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "compose_fragment_runtime_source_v1",
                "task": task,
                "archive_sha256": sha256(target / "source.zip"),
                "files": {"../escape.py": hashlib.sha256(b"bad").hexdigest()},
            }
        )
    )
    with pytest.raises(EvidenceError, match="unsafe relative path"):
        verify_runtime(tmp_path, task)


def test_parity_retains_negative_result():
    records = fixtures(ROOT, "motif_extension")
    saved = json.loads((ROOT / "experiments/fragments/fixtures" / records[1]["path"]).read_text())
    result = {"cells": [{"attempts": [{"panel": copy.deepcopy(saved["panel"])}]}]}
    assert check_parity(ROOT, "motif_extension", result, records)["passed"]
    result["cells"][0]["attempts"][0]["panel"]["selected_smiles"] = "wrong"
    checked = check_parity(ROOT, "motif_extension", result, records)
    assert not checked["passed"]
    assert checked["differences"][0]["different_top_level_fields"] == ["selected_smiles"]


def test_environment_drift_fails_before_model_import(monkeypatch):
    monkeypatch.setattr(
        "compose_v4.experiments.fragments.generation_worker.version", lambda _: "wrong"
    )
    with pytest.raises(ValueError, match="wrong rdkit version"):
        environment({"rdkit": "2024.3.5"})


def test_seed_derivation_matches_saved_superstructure():
    record = fixtures(ROOT, "superstructure_generation")[1]
    saved = json.loads((ROOT / "experiments/fragments/fixtures" / record["path"]).read_text())
    assert (
        seed_for("BARICITINIB", "superstructure_generation", 0)
        == saved["results"]["superstructure_generation"]["per_drug"]["BARICITINIB"][0]["rng_seed"]
    )


def test_source_capture_is_deterministic_and_handles_namespace_packages(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "capture_fragment_runtime", ROOT / "tools/capture_fragment_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    package = source / "src/compose_v4/example"
    package.mkdir(parents=True)
    (package / "a.py").write_text("from . import b\n")
    (package / "b.py").write_text("VALUE = 1\n")
    captured = module.closure(source, ["compose_v4.example.a"])
    assert set(captured) == {"src/compose_v4/example/a.py", "src/compose_v4/example/b.py"}
    for name in ("first.zip", "second.zip"):
        module.write_archive(tmp_path / name, captured)
    assert (tmp_path / "first.zip").read_bytes() == (tmp_path / "second.zip").read_bytes()
    with pytest.raises(FileExistsError):
        module.write_archive(tmp_path / "first.zip", captured)


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_generation_refuses_existing_output(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(
        "compose_v4.experiments.fragments.generation.verify_assets", lambda *args: {}
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
        "compose_v4.experiments.fragments.generation.verify_assets", lambda *args: {}
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
