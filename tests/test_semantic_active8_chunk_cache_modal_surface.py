"""Static safety contract for the semantic Active8 chunk-cache Modal app."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "modal_apps" / "build_semantic_active8_chunk_cache_app.py"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _load_launcher():
    spec = importlib.util.spec_from_file_location("semantic_chunk_cache_launcher", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modal_surface_is_cpu_only_and_volume_v1_writer_safe() -> None:
    source = APP_PATH.read_text()
    tree = ast.parse(source)
    assert "MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS = 5" in source
    assert "MAX_MAP_CONTAINERS = MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS" in source
    launcher = _load_launcher()
    assert launcher.MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS == 5
    assert launcher.MAX_MAP_CONTAINERS == 5
    assert "EXPECTED_SOURCE_TASKS = 20" in source
    assert "gpu=" not in source
    worker = _function(tree, "build_one_source_cache")
    worker_source = ast.get_source_segment(source, worker)
    assert worker_source is not None
    assert "_validate_remote_source_revision" in worker_source
    assert "execute_semantic_active8_chunk_cache_task" in worker_source
    assert "artifact_volume.reload()" in worker_source
    assert "artifact_volume.commit()" in worker_source
    worker_decorator = ast.get_source_segment(
        source, _function(tree, "build_one_source_cache").decorator_list[0]
    )
    assert worker_decorator is not None
    assert "max_containers=MAX_MAP_CONTAINERS" in worker_decorator

    reducer_source = ast.get_source_segment(source, _function(tree, "reduce_global_cache"))
    assert reducer_source is not None
    assert reducer_source.index("artifact_volume.reload()") < reducer_source.index(
        "reduce_semantic_active8_chunk_caches"
    )
    assert "os.link(" not in source
    assert ".link_to(" not in source


def test_driver_validates_caller_completion_once_then_maps_and_reduces() -> None:
    source = APP_PATH.read_text()
    tree = ast.parse(source)
    driver = _function(tree, "driver")
    driver_source = ast.get_source_segment(source, driver)
    assert driver_source is not None
    assert driver_source.count("resolve_editing_v2_semantic_active8_sources") == 1
    assert "semantic_migration_completion" in driver_source
    assert "write_semantic_active8_chunk_cache_plan" in driver_source
    assert "completed_semantic_active8_chunk_cache_task_ids" in driver_source
    assert "build_one_source_cache.starmap" in driver_source
    assert "reduce_global_cache.remote(plan)" in driver_source


def test_local_entrypoint_requires_clean_exact_revision_and_waits() -> None:
    source = APP_PATH.read_text()
    tree = ast.parse(source)
    main = _function(tree, "main")
    main_source = ast.get_source_segment(source, main)
    assert main_source is not None
    assert "semantic_migration_completion" in main_source
    assert "expected_commit" in main_source
    assert "local_source_revision(expected_commit=expected_commit)" in main_source
    assert "driver.remote(" in main_source
    assert "driver.spawn(" not in main_source


def test_launcher_contains_no_later_stage_or_live_hash_authority() -> None:
    source = APP_PATH.read_text()
    assert 'training_launched": False' in source
    assert 'active8_admission_run": False' in source
    assert "P50" in source
    assert "expected_commit: str" in source
    assert "EXPECTED_" not in "\n".join(
        line for line in source.splitlines() if "sha256" in line.lower()
    )


def test_source_revision_requires_exact_clean_commit_and_hashes_serialized_tree(
    monkeypatch,
) -> None:
    launcher = _load_launcher()

    def clean_git(_root, *arguments):
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return "b" * 40
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(launcher, "_git", clean_git)
    revision = launcher.local_source_revision(
        expected_commit="a" * 40,
        repo_root=ROOT,
    )
    assert revision["commit"] == "a" * 40
    assert revision["tree"] == "b" * 40
    assert revision["worktree_clean"] is True
    assert set(revision["serialized_sources"]) == set(launcher._serialized_source_paths(ROOT))
    assert (
        "src/compose_v4/data/semantic_active8_chunk_cache_mapreduce.py"
        in revision["serialized_sources"]
    )
    launcher._validate_remote_source_revision(
        revision,
        loaded=launcher._imports(ROOT),
        remote_root=ROOT,
    )

    with pytest.raises(RuntimeError, match="exact clean committed worktree"):
        launcher.local_source_revision(
            expected_commit="c" * 40,
            repo_root=ROOT,
        )

    def dirty_git(root, *arguments):
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return " M dirty.py"
        return clean_git(root, *arguments)

    monkeypatch.setattr(launcher, "_git", dirty_git)
    with pytest.raises(RuntimeError, match="exact clean committed worktree"):
        launcher.local_source_revision(
            expected_commit="a" * 40,
            repo_root=ROOT,
        )
