"""Static safety contract for the semantic migration Modal entrypoint."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "modal_apps" / "materialize_editing_v2_semantic_corpus_app.py"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_modal_surface_is_cpu_only_and_maps_one_frozen_task_identity() -> None:
    source = APP_PATH.read_text()
    tree = ast.parse(source)
    assert "MAX_MAP_CONTAINERS = 20" in source
    assert "gpu=" not in source
    worker = _function(tree, "migrate_one_shard")
    worker_source = ast.get_source_segment(source, worker)
    assert worker_source is not None
    assert "execute_semantic_trace_migration_task" in worker_source
    assert "artifact_volume.reload()" in worker_source
    assert "artifact_volume.commit()" in worker_source


def test_driver_persists_plan_then_maps_only_missing_tasks_and_reduces() -> None:
    source = APP_PATH.read_text()
    tree = ast.parse(source)
    driver = _function(tree, "driver")
    driver_source = ast.get_source_segment(source, driver)
    assert driver_source is not None
    assert "resolve_editing_v2_active8_sources" in driver_source
    assert "write_semantic_trace_migration_plan" in driver_source
    assert "completed_semantic_trace_migration_task_ids" in driver_source
    assert "migrate_one_shard.starmap" in driver_source
    assert "reduce_complete.remote(plan)" in driver_source


def test_local_entrypoint_requires_clean_revision_and_waits_for_driver() -> None:
    source = APP_PATH.read_text()
    tree = ast.parse(source)
    main = _function(tree, "main")
    main_source = ast.get_source_segment(source, main)
    assert main_source is not None
    assert "_local_source_revision()" in main_source
    assert "driver.remote(" in main_source
    assert "driver.spawn(" not in main_source
