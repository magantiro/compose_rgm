"""Static and helper-level checks for the restart-safe Modal launcher."""

from __future__ import annotations

import ast
from pathlib import Path

from modal_apps import (
    materialize_editing_v2_refined_role_lane_mapreduce_app as mapreduce_app,
)


def test_launcher_freezes_cpu_concurrency_and_exact_source_shard_count() -> None:
    assert mapreduce_app.EXPECTED_SOURCE_SHARDS == 62
    assert mapreduce_app.MAX_MAP_CONTAINERS == 5
    assert mapreduce_app.MAX_CELL_CONTAINERS == 5
    source = Path(mapreduce_app.__file__).read_text()
    tree = ast.parse(source)
    assert "gpu=" not in source
    assert "map_role_lane_source_shard" in source
    assert "reduce_role_lane_cell" in source
    assert "reduce_role_lane_mapreduce" in source
    assert "validate_continuation_completion_header" in source
    assert '"training_authorized": True' not in source
    assert any(
        isinstance(node, ast.Try)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "commit"
            for statement in node.finalbody
            for call in ast.walk(statement)
        )
        for node in ast.walk(tree)
    )


def test_source_revision_binds_reference_and_mapreduce_launchers() -> None:
    files = set(mapreduce_app.continuation.SERIALIZED_SOURCE_FILES)
    assert "modal_apps/materialize_editing_v2_refined_role_lane_app.py" in files
    assert (
        "modal_apps/materialize_editing_v2_refined_role_lane_mapreduce_app.py" in files
    )
    assert "src/compose_v4/data/editing_v2_role_lane_packed_materializer.py" in files
    assert "src/compose_v4/data/editing_v2_role_lane_mapreduce.py" in files


def test_driver_reloads_child_volume_snapshot_before_final_validation() -> None:
    source = Path(mapreduce_app.__file__).read_text()
    assemble = source.index("manifest = assemble_final.remote(str(plan_path))")
    reload_snapshot = source.index("artifact_volume.reload()", assemble)
    validate = source.index("continuation.validate_role_lane_output(", reload_snapshot)

    assert assemble < reload_snapshot < validate
