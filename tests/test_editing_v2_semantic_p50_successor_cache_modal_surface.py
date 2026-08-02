"""Modal surface contracts for the reusable semantic P50 successor cache."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from modal_apps import build_editing_v2_semantic_p50_successor_cache_app as app
from compose_v4.experiments.editing_v2_execution_source_revision import (
    build_execution_source_revision,
)


def _function_sources() -> tuple[str, dict[str, str]]:
    source = Path(app.__file__).read_text()
    tree = ast.parse(source)
    functions = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert all(value is not None for value in functions.values())
    return source, functions  # type: ignore[return-value]


def test_image_copy_inventory_matches_the_hashed_source_inventory() -> None:
    root = Path(app.__file__).resolve().parents[1]
    expected = {app.LAUNCHER_SOURCE}
    for source_directory in app.IMAGE_SOURCE_DIRECTORIES:
        expected.update(
            path.relative_to(root).as_posix()
            for path in (root / source_directory).rglob("*")
            if path.is_file() and path.suffix != ".pyc" and "__pycache__" not in path.parts
        )
    assert app.IMAGE_SOURCE_DIRECTORIES == ("src", "configs")
    assert set(app._serialized_source_paths(root)) == expected


def test_local_source_revision_rejects_dirty_or_wrong_commit() -> None:
    with (
        patch.object(
            app,
            "_git",
            side_effect=["1" * 40, "2" * 40, "", "source.py"],
        ),
        patch.object(
            app,
            "_serialized_source_hashes",
            return_value={"source.py": "a" * 64},
        ),
        patch.object(
            app,
            "_imports",
            return_value={"build_execution_source_revision": build_execution_source_revision},
        ),
    ):
        revision = app.local_source_revision(
            expected_commit="1" * 40,
            repo_root=Path("/fixture"),
        )
    assert revision["worktree_clean"] is True
    assert revision["execution_source_revision"]["commit"] == "1" * 40

    with (
        patch.object(
            app,
            "_git",
            side_effect=["1" * 40, "2" * 40, "", "tracked.py"],
        ),
        patch.object(
            app,
            "_serialized_source_hashes",
            return_value={"ignored.py": "a" * 64},
        ),
        pytest.raises(RuntimeError, match="serialized source/config must be Git-tracked"),
    ):
        app.local_source_revision(
            expected_commit="1" * 40,
            repo_root=Path("/fixture"),
        )

    with (
        patch.object(app, "_git", side_effect=["1" * 40, "2" * 40, "?? dirty.py"]),
        pytest.raises(RuntimeError, match="clean committed"),
    ):
        app.local_source_revision(
            expected_commit="1" * 40,
            repo_root=Path("/fixture"),
        )

    with (
        patch.object(app, "_git", side_effect=["1" * 40, "2" * 40, ""]),
        pytest.raises(RuntimeError, match="clean committed"),
    ):
        app.local_source_revision(
            expected_commit="3" * 40,
            repo_root=Path("/fixture"),
        )


def test_modal_surface_uses_bounded_cpu_only_restart_safe_workers() -> None:
    source, functions = _function_sources()
    tree = ast.parse(source)
    worker_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "compile_semantic_p50_cache_task_batch"
    )
    worker_decorator = ast.get_source_segment(source, worker_node.decorator_list[0])
    assert worker_decorator is not None
    assert "gpu=" not in worker_decorator
    assert "cpu=4.0" in worker_decorator
    assert "max_containers=MAX_VOLUME_WRITERS" in worker_decorator
    assert app.MAX_VOLUME_WRITERS == 5

    worker = functions["compile_semantic_p50_cache_task_batch"]
    driver = functions["driver"]
    assert "compile_trace_union" in worker
    assert "build_semantic_p50_successor_cache_leaf" in worker
    assert worker.index("write_semantic_p50_successor_cache_artifact") < worker.index(
        "artifact_volume.commit()"
    )
    assert "for task_identity in task_identities" in worker
    assert driver.index("validate_semantic_p50_successor_cache_leaf_for_plan") < (
        driver.index(".starmap(")
    )
    assert driver.index(".starmap(") < driver.index("reduce_semantic_p50_successor_cache.remote")
    assert '"leaf_reused": reused' in worker


def test_task_batches_are_deterministic_balanced_and_writer_bounded() -> None:
    tasks = [
        {"task_identity_sha256": f"task-{index}", "closure_record_count": count}
        for index, count in enumerate((100, 90, 80, 70, 60, 50, 40, 30, 20))
    ]
    batches = app._task_batches(tasks, maximum_batches=5)
    assert batches == app._task_batches(tuple(reversed(tasks)), maximum_batches=5)
    assert len(batches) == 5
    assert {identity for batch in batches for identity in batch} == {
        task["task_identity_sha256"] for task in tasks
    }
    assert sum(len(batch) for batch in batches) == len(tasks)

    task_by_identity = {task["task_identity_sha256"]: task for task in tasks}
    loads = [
        sum(task_by_identity[identity]["closure_record_count"] for identity in batch)
        for batch in batches
    ]
    assert max(loads) - min(loads) <= max(task["closure_record_count"] for task in tasks)


def test_reducer_strictly_reopens_once_before_publishing_completion() -> None:
    source, functions = _function_sources()
    reducer = functions["reduce_semantic_p50_successor_cache"]
    assert source.count("open_semantic_p50_successor_cache(") == 1
    assert reducer.index("build_semantic_p50_successor_cache_manifest") < reducer.index(
        "build_semantic_p50_successor_cache_completion"
    )
    assert reducer.index("build_semantic_p50_successor_cache_completion") < reducer.index(
        "open_semantic_p50_successor_cache"
    )
    assert reducer.index("open_semantic_p50_successor_cache") < reducer.index(
        "artifact_volume.commit()"
    )
    assert 'scratch_runtime=opened["scratch"]' in reducer
    assert "opened_cache.train_requested_records" in reducer
    assert "opened_cache.validation_requested_records" in reducer
    assert "opened_cache.records" in reducer
    assert reducer.strip().endswith(
        "return _artifact_address(completion_path, artifact_root=ARTIFACT_ROOT)"
    )


def test_launcher_reopens_exact_prerequisites_and_has_no_training_stage() -> None:
    source, functions = _function_sources()
    opener = functions["_open_exact_prerequisites"]
    for required in (
        "load_semantic_p50_prepared_recipe",
        'loaded["load_source"]',
        'loaded["validate_gate_zero"]',
        'loaded["validate_t1"]',
        'loaded["validate_prerequisites"]',
        "_scratch_runtime(",
        "expected_semantic_p50_successor_cache_identity_from_reopened_prerequisites",
    ):
        assert required in opener

    assert '"device": "cpu"' in source
    assert "gpu=" not in source
    assert "torch.optim" not in source
    assert "cuda()" not in source
    assert '.to("cuda")' not in source
    assert "editing_v2_semantic_p50_validation_baseline" not in source
    assert "model_scores_or_probabilities_stored" not in source
    assert "hazard_coordinates_included" not in source


def test_input_addresses_are_exact_normalized_artifact_paths() -> None:
    addresses = {name: f"/artifacts/{name}.json" for name in app._INPUT_FIELDS}
    assert (
        app._validate_input_addresses(
            addresses,
            artifact_root=Path("/tmp/artifacts"),
        )
        == addresses
    )

    with pytest.raises(TypeError, match="fields disagree"):
        app._validate_input_addresses(
            {name: value for name, value in addresses.items() if name != "t1_decision"},
            artifact_root=Path("/tmp/artifacts"),
        )
    with pytest.raises(ValueError, match="normalized /artifacts path"):
        app._validate_input_addresses(
            {**addresses, "t1_decision": "/artifacts/../escape.json"},
            artifact_root=Path("/tmp/artifacts"),
        )


def test_serialized_revision_covers_launcher_core_sources_and_configs() -> None:
    paths = app._serialized_source_paths(Path(app.ROOT))
    assert app.LAUNCHER_SOURCE in paths
    assert "src/compose_v4/experiments/editing_v2_semantic_p50_successor_cache.py" in paths
    assert app.SEMANTIC_MODEL_PROCESS_SOURCE in paths
    assert len(paths) == len(set(paths))
