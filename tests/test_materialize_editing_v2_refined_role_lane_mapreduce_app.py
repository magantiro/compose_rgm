"""Static and helper-level checks for the restart-safe Modal launcher."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

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


def test_reusable_plan_pointer_reopens_exact_task_inputs_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    plan_path = artifact_root / "execution" / "ROLE_LANE_MAPREDUCE_PLAN.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_bytes(b"frozen plan bytes\n")
    monkeypatch.setattr(mapreduce_app, "ARTIFACT_ROOT", artifact_root)

    request = {
        "run_identity_sha256": "1" * 64,
        "source_revision": {"commit": "a" * 40},
    }
    candidate_identity = {"manifest_sha256": "2" * 64}
    source_stream = {"source_stream_sha256": "3" * 64}
    split_identity = {"assignment_sha256": "4" * 64}
    receipt_sha256 = "5" * 64
    plan = {
        "plan_sha256": "6" * 64,
        "run_identity_sha256": "7" * 64,
        "code_revision": request["source_revision"]["commit"],
        "upstream_validation_receipt_sha256": receipt_sha256,
        "reference_materialization": {
            "run_identity_body": {
                "output_artifact_prefix": "/artifacts/final",
                "candidate_materialization": candidate_identity,
                "candidate_provenance_source_stream": source_stream,
                "split_assignment": split_identity,
            }
        },
    }
    pointer = mapreduce_app._build_plan_pointer(
        request=request,
        plan_path=plan_path,
        plan=plan,
        upstream_validation_receipt_sha256=receipt_sha256,
    )
    pointer_path = artifact_root / "run" / mapreduce_app.PLAN_POINTER_FILENAME
    pointer_path.parent.mkdir()
    pointer_path.write_bytes(mapreduce_app._canonical_bytes(pointer))
    observed_calls: list[tuple[Path, Path, bool]] = []

    def validate(path, *, artifact_root, validate_task_inputs):
        observed_calls.append((Path(path), Path(artifact_root), validate_task_inputs))
        return plan

    monkeypatch.setattr(
        mapreduce_app.mapreduce,
        "validate_role_lane_mapreduce_plan",
        validate,
    )
    reopened = mapreduce_app._load_reusable_plan(
        pointer_path=pointer_path,
        request=request,
        upstream_validation_receipt_sha256=receipt_sha256,
        expected_final_output_prefix="/artifacts/final",
        expected_candidate_identity=candidate_identity,
        expected_source_stream=source_stream,
        expected_split_identity=split_identity,
    )
    assert reopened == (plan, plan_path)
    assert observed_calls == [(plan_path, artifact_root, True)]

    tampered = json.loads(pointer_path.read_bytes())
    tampered["training_authorized"] = True
    pointer_path.write_bytes(mapreduce_app._canonical_bytes(tampered))
    with pytest.raises(RuntimeError, match="identity or authority disagrees"):
        mapreduce_app._load_reusable_plan(
            pointer_path=pointer_path,
            request=request,
            upstream_validation_receipt_sha256=receipt_sha256,
            expected_final_output_prefix="/artifacts/final",
            expected_candidate_identity=candidate_identity,
            expected_source_stream=source_stream,
            expected_split_identity=split_identity,
        )
