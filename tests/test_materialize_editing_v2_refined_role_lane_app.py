from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_apps import materialize_editing_v2_refined_role_lane_app as refined_app
from modal_apps import materialize_editing_v2_semantic_v4_migration_app as migration_app


def test_immutable_stage_reuses_exact_bytes_and_rejects_collision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stage.json"
    value = {"status": "NO_AUTHORITY", "training_authorized": False}
    assert refined_app._write_immutable_json(path, value) is True
    original = path.read_bytes()
    assert refined_app._write_immutable_json(path, value) is False
    assert path.read_bytes() == original
    with pytest.raises(RuntimeError, match="immutable refined-role/lane collision"):
        refined_app._write_immutable_json(path, {**value, "status": "changed"})
    assert path.read_bytes() == original


def test_migration_dispatches_only_the_explicit_refined_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    completion_path = artifact_root / "refined" / "REFINED_ROLE_LANE_COMPLETE.json"
    completion_path.parent.mkdir(parents=True)
    completion_path.write_text(
        json.dumps({"schema": migration_app.REFINED_STRUCTURAL_COMPLETION_SCHEMA})
    )
    expected = ({"completion": True}, {"source_shard_count": 20}, object())
    observed: dict[str, object] = {}

    def validate(path: str, **kwargs: object):
        observed["path"] = path
        observed.update(kwargs)
        return expected

    address = "/artifacts/refined/REFINED_ROLE_LANE_COMPLETE.json"
    assert (
        migration_app._validate_structural_completion(
            address,
            artifact_root=artifact_root,
            remote_root=tmp_path / "repo",
            max_row_bytes=1024,
            loaded={
                "refined": SimpleNamespace(validate_for_semantic_v4_migration=validate),
                "candidates": object(),
                "bridge": object(),
            },
        )
        == expected
    )
    assert observed["path"] == address


def test_modal_surface_is_cpu_only_and_uses_production_materializer() -> None:
    source = Path(refined_app.__file__).read_text()
    assert "gpu=" not in source
    assert "materialize_editing_v2_role_lane_packed" in source
    assert "load_validated_refinement_inputs" in source
    assert "validate_continuation_completion_header" in source
    assert '"training_authorized": True' not in source
    assert '"bounded_p50_authorized": True' not in source
