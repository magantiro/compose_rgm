"""Static and local gates for the Process-V2 Active8 fast finalizer."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import modal_apps.run_process_v2_active8_fast_finalize_app as launcher

ROOT = Path(__file__).resolve().parents[1]


def test_fast_finalizer_is_one_bounded_hash_only_container() -> None:
    source = (ROOT / launcher.LAUNCHER_SOURCE).read_text()
    assert launcher.TIMEOUT_SECONDS == 2 * 60 * 60
    assert launcher.HASH_WORKERS == 16
    assert "cpu=4.0" in source
    assert "memory=8 * 1024" in source
    assert "max_containers=1" in source
    assert ".starmap(" not in source
    assert ".map(" not in source


def test_fast_launcher_invokes_only_the_fast_production_publisher() -> None:
    source = (ROOT / launcher.LAUNCHER_SOURCE).read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("candidate" in name or "model" in name for name in imported)
    assert "fast_finalize_process_v2_active8" in source
    assert "artifact_volume.reload()" in source
    assert source.index("artifact_volume.reload()") < source.index(
        "fast_finalize_process_v2_active8("
    )
    assert source.index("fast_finalize_process_v2_active8(") < source.index(
        "artifact_volume.commit()"
    )


def test_serialized_image_contains_fast_finalizer_and_bound_inputs() -> None:
    sources = set(launcher._serialized_source_paths(ROOT))
    assert launcher.LAUNCHER_SOURCE in sources
    assert (
        "src/compose_v4/data/editing_v2_process_v2_active8_fast_finalize.py"
        in sources
    )
    assert "configs/editing_v2_process_v2_active8_decision_runtime.json" in sources


def test_local_revision_refuses_a_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = {
        ("rev-parse", "HEAD"): "1" * 40,
        ("rev-parse", "HEAD^{tree}"): "2" * 40,
        ("status", "--porcelain=v1", "--untracked-files=all"): " M tracked.py",
    }
    monkeypatch.setattr(
        launcher,
        "_git",
        lambda _root, *arguments: answers[arguments],
    )
    with pytest.raises(RuntimeError, match="exact clean commit"):
        launcher.local_image_revision(expected_commit="1" * 40, repo_root=ROOT)


@pytest.mark.parametrize(
    "value",
    ["relative/path", "/tmp/outside", "/artifacts/../outside"],
)
def test_artifact_paths_cannot_escape_the_mounted_volume(value: str) -> None:
    with pytest.raises(ValueError):
        launcher._require_artifact_path(value, field="test_path")


def test_remote_surface_is_one_run_root_plus_exact_revision() -> None:
    parameters = inspect.signature(launcher.fast_finalize_remote.get_raw_f()).parameters
    assert tuple(parameters) == ("run_root", "revision")


def test_launcher_never_grants_downstream_authority() -> None:
    source = (ROOT / launcher.LAUNCHER_SOURCE).read_text()
    for field in (
        "training_authorized",
        "bounded_p50_authorized",
    ):
        assert f'"{field}": False' in source
