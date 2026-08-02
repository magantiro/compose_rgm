"""Cross-stage Modal provenance and mounted-path closure regressions."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from compose_v4.data import semantic_trace_migration_mapreduce as migration_core
from modal_apps import build_editing_v2_semantic_t1_successor_cache_app as cache_app
from modal_apps import build_semantic_active8_chunk_cache_app as chunk_app
from modal_apps import materialize_editing_v2_semantic_t1_panel_cache_app as panel_app
from modal_apps import run_editing_v2_semantic_gate_zero_app as gate_app
from modal_apps import run_semantic_active8_decisions_app as decision_app


@pytest.mark.parametrize(
    "launcher",
    (chunk_app, decision_app, gate_app, panel_app, cache_app),
)
def test_downstream_images_mount_semantic_migration_provenance_closure(
    launcher: object,
) -> None:
    """Every migration validator source must exist in each downstream image."""

    mounted_directories = set(launcher.IMAGE_SOURCE_DIRECTORIES)
    mounted_files = set(getattr(launcher, "IMAGE_SOURCE_FILES", ()))
    mounted_files.add(getattr(launcher, "MIGRATION_LAUNCHER_SOURCE", ""))
    assert "src" in mounted_directories
    for relative in migration_core._SOURCE_FILES:
        assert (
            Path(relative).parts[0] in mounted_directories
            or relative in mounted_files
        )
        assert (launcher.ROOT / relative).is_file()


def test_decision_fanout_preserves_lexical_artifact_address(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resolved Volume path must cross RPC as /artifacts, not its backend path."""

    backend = tmp_path / "backend"
    backend.mkdir()
    mount = tmp_path / "artifacts"
    mount.symlink_to(backend, target_is_directory=True)
    plan = backend / "decisions" / "PLAN.json"
    plan.parent.mkdir()
    plan.write_bytes(b"{}\n")
    monkeypatch.setattr(decision_app, "ARTIFACT_ROOT", mount)

    address = decision_app._artifact_address(plan)
    assert address == "/artifacts/decisions/PLAN.json"
    assert (
        decision_app._mounted_artifact_path(
            address,
            field="plan_artifact_path",
        )
        == plan.resolve()
    )

    source = Path(decision_app.__file__).read_text()
    tree = ast.parse(source)
    driver = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "driver"
    )
    driver_source = ast.get_source_segment(source, driver)
    assert driver_source is not None
    assert "plan_artifact_path = _artifact_address(plan_path)" in driver_source
    assert "(str(plan_path), task_identity, source_revision)" not in driver_source
