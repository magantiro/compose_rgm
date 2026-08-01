"""Modal surface contracts for reusable semantic T1 CPU coordinates."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from modal_apps import build_editing_v2_semantic_t1_successor_cache_app as app


def test_local_source_revision_rejects_dirty_or_wrong_commit() -> None:
    with (
        patch.object(app, "_git", side_effect=["1" * 40, "2" * 40, ""]),
        patch.object(
            app,
            "_serialized_source_hashes",
            return_value={"source.py": "a" * 64},
        ),
    ):
        revision = app.local_source_revision(
            expected_commit="1" * 40,
            repo_root=Path("/fixture"),
        )
    assert revision["worktree_clean"] is True

    with (
        patch.object(
            app,
            "_git",
            side_effect=["1" * 40, "2" * 40, "?? dirty.py"],
        ),
        pytest.raises(RuntimeError, match="clean committed"),
    ):
        app.local_source_revision(
            expected_commit="1" * 40,
            repo_root=Path("/fixture"),
        )


def test_modal_surface_fans_out_bounded_cpu_only_reusable_work() -> None:
    source = Path(app.__file__).read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    worker = functions["compile_one_semantic_t1_cache_leaf"]
    worker_decorator = ast.get_source_segment(source, worker.decorator_list[0])
    assert worker_decorator is not None
    assert "gpu=" not in worker_decorator
    assert "cpu=4.0" in worker_decorator
    assert "max_containers=MAX_MAP_CONTAINERS" in worker_decorator
    assert app.MAX_MAP_CONTAINERS == 5
    assert ".starmap(" in source
    assert "artifact_volume.reload()" in source
    assert "artifact_volume.commit()" in source
    assert "read_semantic_packed_artifact" in source
    assert "build_semantic_scratch_runtime" in source
    assert "compile_semantic_t1_successor_cache_leaf" in source
    assert "compile_successor_fiber_trace_union" not in source
    assert "T1SuccessorCacheIdentity" not in source


def test_modal_surface_contains_no_gpu_training_or_repeated_state_stage() -> None:
    source = Path(app.__file__).read_text()
    assert '"device": "cpu"' in source
    assert '"dtype": "torch.float32"' in source
    assert '"repeated_state_empirical_law_compiled": False' in source
    assert '"training_authorized": False' in source
    assert '"bounded_p50_authorized": False' in source
    assert "optimizer" in source.splitlines()[9]
    assert "torch.optim" not in source
    assert "cuda()" not in source
    assert '.to("cuda")' not in source


def test_partial_or_restarted_runs_cannot_publish_completion_early() -> None:
    source = Path(app.__file__).read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    driver_source = ast.get_source_segment(source, functions["driver"])
    reducer_source = ast.get_source_segment(
        source,
        functions["reduce_semantic_t1_successor_cache"],
    )
    worker_source = ast.get_source_segment(
        source,
        functions["compile_one_semantic_t1_cache_leaf"],
    )
    assert driver_source is not None
    assert reducer_source is not None
    assert worker_source is not None
    assert driver_source.index("validate_semantic_t1_successor_cache_leaf_for_plan") < (
        driver_source.index(".starmap(")
    )
    assert driver_source.index(".starmap(") < driver_source.index(
        "reduce_semantic_t1_successor_cache.remote"
    )
    assert reducer_source.index("build_semantic_t1_successor_cache_manifest") < (
        reducer_source.index("build_semantic_t1_successor_cache_completion")
    )
    assert reducer_source.index("build_semantic_t1_successor_cache_completion") < (
        reducer_source.index("artifact_volume.commit()")
    )
    assert worker_source.index("write_semantic_t1_successor_cache_leaf") < (
        worker_source.index("artifact_volume.commit()")
    )
