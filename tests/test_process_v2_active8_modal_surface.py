"""Static and local gates for the Process-V2 Active8 Modal orchestration."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

import modal_apps.run_process_v2_active8_app as launcher
from compose_v4.data.editing_v2_process_v2_active8_map import (
    TEACHER_SUPPORT_BATCH_SIZE,
)
from compose_v4.data.editing_v2_process_v2_active8_admission import (
    ProductionProcessV2BatchedTeacherSupportChecker,
)

ROOT = Path(__file__).resolve().parents[1]


def _plan(size: int = 12) -> dict[str, object]:
    return {
        "run_artifact_root": "/artifacts/editing_v2/process_v2_active8/" + "a" * 64,
        "tasks": [
            {"task_identity_sha256": f"{index:064x}"} for index in range(size)
        ],
    }


def test_two_partitions_are_disjoint_complete_and_restart_safe() -> None:
    plan = _plan()
    completed = {f"{2:064x}", f"{7:064x}"}
    left = launcher._partition_task_ids(
        plan, partition_count=2, partition_index=0, completed=completed
    )
    right = launcher._partition_task_ids(
        plan, partition_count=2, partition_index=1, completed=completed
    )
    expected = {
        str(task["task_identity_sha256"])
        for task in plan["tasks"]
        if str(task["task_identity_sha256"]) not in completed
    }
    assert set(left).isdisjoint(right)
    assert set(left) | set(right) == expected
    assert left == sorted(left, key=lambda item: int(item, 16))
    assert right == sorted(right, key=lambda item: int(item, 16))


@pytest.mark.parametrize(
    ("count", "index"),
    [(0, 0), (2, -1), (2, 2), (True, 0)],
)
def test_invalid_partition_geometry_is_refused(count: int, index: int) -> None:
    with pytest.raises(ValueError):
        launcher._partition_task_ids(
            _plan(), partition_count=count, partition_index=index, completed=set()
        )


def test_launcher_uses_the_measured_low_memory_operating_point() -> None:
    runtime = json.loads((ROOT / launcher.RUNTIME_CONTRACT_SOURCE).read_bytes())
    assert TEACHER_SUPPORT_BATCH_SIZE == 8
    assert launcher.MAP_CPU == 1.0
    assert launcher.MAP_MEMORY_MB == 8192
    assert launcher.MAX_MAP_CONTAINERS == 40
    assert runtime["model"] == {
        "atom_vocabulary_class_count": 15,
        "candidate_cache_size": 4096,
        "candidate_time": 0.5,
        "catalog_fingerprint": "639ff6078c32d43c",
        "dtype": "torch.float32",
        "hidden_dim": 256,
        "initialization_seed": 20260730,
        "mark_dim": 32,
        "max_atoms": 40,
        "message_passing_steps": 6,
        "operator_capability_fingerprint": "d79ffe8ef65f3fb3",
    }
    parameters = inspect.signature(
        ProductionProcessV2BatchedTeacherSupportChecker
    ).parameters
    assert parameters["time"].default == runtime["model"]["candidate_time"]
    assert (
        parameters["chemistry_feature_cache_size"].default
        == runtime["model"]["candidate_cache_size"]
    )


def test_the_serialized_image_contains_every_active8_answer_module() -> None:
    sources = set(launcher._serialized_source_paths(ROOT))
    assert launcher.LAUNCHER_SOURCE in sources
    for name in (
        "editing_v2_process_v2_active8_map.py",
        "editing_v2_process_v2_active8_plan.py",
        "editing_v2_process_v2_active8_reduce.py",
        "editing_v2_process_v2_active8_sentinel.py",
    ):
        assert f"src/compose_v4/data/{name}" in sources


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


def test_launcher_has_no_legacy_active8_or_training_import() -> None:
    tree = ast.parse((ROOT / launcher.LAUNCHER_SOURCE).read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("semantic_active8_decision_mapreduce" in name for name in imported)
    assert not any("training" in name for name in imported)


def test_plan_path_is_the_exact_mounted_content_address() -> None:
    plan = _plan()
    assert str(launcher._active8_plan_path(plan)) == (
        str(plan["run_artifact_root"]) + "/PROCESS_V2_ACTIVE8_PLAN.json"
    )
