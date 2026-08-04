"""Static and local gates for the Process-V2 Active8 Modal orchestration."""

from __future__ import annotations

import ast
import inspect
import json
from concurrent.futures import Future
from pathlib import Path

import pytest

import modal_apps.run_process_v2_active8_app as launcher
from compose_v4.data.editing_v2_process_v2_active8_admission import (
    ProductionProcessV2BatchedTeacherSupportChecker,
)
from compose_v4.data.editing_v2_process_v2_active8_map import (
    TEACHER_SUPPORT_BATCH_SIZE,
)

ROOT = Path(__file__).resolve().parents[1]


def _plan(size: int = 12) -> dict[str, object]:
    return {
        "run_artifact_root": "/artifacts/editing_v2/process_v2_active8/" + "a" * 64,
        "tasks": [
            {
                "task_identity_sha256": f"{index:064x}",
                "chunk_row_count": index + 1,
            }
            for index in range(size)
        ],
    }


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


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
    assert launcher.MAP_CPU == 16.0
    assert launcher.TASK_PROCESSES_PER_CONTAINER == 16
    assert launcher.MAP_MEMORY_MB == 64 * 1024
    assert launcher.MAX_MAP_CONTAINERS == 5
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
    parameters = inspect.signature(ProductionProcessV2BatchedTeacherSupportChecker).parameters
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


def test_groups_are_bounded_complete_and_keep_plan_order() -> None:
    plan = _plan()
    identities = [str(task["task_identity_sha256"]) for task in plan["tasks"]]
    groups = launcher._task_groups(plan, identities, group_size=4)

    assert [int(group[0], 16) for group in groups] == [0, 4, 8]
    assert all(1 <= len(group) <= 4 for group in groups)
    assert {identity for group in groups for identity in group} == set(identities)
    assert sum(len(group) for group in groups) == len(identities)


def test_completed_tasks_are_removed_before_restart_groups_are_built() -> None:
    plan = _plan()
    completed = {f"{2:064x}", f"{7:064x}"}
    missing = launcher._partition_task_ids(
        plan, partition_count=1, partition_index=0, completed=completed
    )
    groups = launcher._task_groups(plan, missing, group_size=3)

    submitted = {identity for group in groups for identity in group}
    assert submitted.isdisjoint(completed)
    assert submitted | completed == {str(task["task_identity_sha256"]) for task in plan["tasks"]}


@pytest.mark.parametrize("group_size", [0, 17, True])
def test_invalid_process_group_size_is_refused(group_size: int) -> None:
    with pytest.raises(ValueError, match="group_size"):
        launcher._task_groups(_plan(), [], group_size=group_size)


def test_a_child_failure_propagates_out_of_group_collection() -> None:
    left: Future[dict[str, str]] = Future()
    right: Future[dict[str, str]] = Future()
    left.set_result({"task_identity_sha256": "a" * 64, "receipt_sha256": "1" * 64})
    right.set_exception(RuntimeError("child failed"))

    with pytest.raises(RuntimeError, match="child failed"):
        launcher._collect_process_results({left: "a" * 64, right: "b" * 64})


def test_group_worker_has_five_writers_and_children_never_commit() -> None:
    source = (ROOT / launcher.LAUNCHER_SOURCE).read_text()
    tree = ast.parse(source)
    child = ast.get_source_segment(source, _function(tree, "_execute_one_chunk_process"))
    group = ast.get_source_segment(source, _function(tree, "decide_task_group"))
    decorator = ast.get_source_segment(
        source, _function(tree, "decide_task_group").decorator_list[0]
    )
    assert child is not None and group is not None and decorator is not None
    assert "artifact_volume.commit" not in child
    assert "artifact_volume.reload" not in child
    assert 'multiprocessing.get_context("spawn")' in group
    assert group.count("artifact_volume.commit()") == 1
    assert (
        group.index("_collect_process_results")
        < group.index("validate_process_v2_active8_task_result")
        < group.index("artifact_volume.commit()")
    )
    assert "max_containers=MAX_MAP_CONTAINERS" in decorator
    assert launcher.MAX_MAP_CONTAINERS == 5


def test_first_launch_defaults_to_the_exact_single_train_smoke_task() -> None:
    parameters = inspect.signature(launcher.main.info.raw_f).parameters
    assert parameters["group_size"].default == 1
    assert parameters["group_limit"].default == 1
    assert (
        parameters["smoke_task_identity"].default
        == launcher.DEFAULT_SMOKE_TASK_IDENTITY
        == "c3740f704521edb8d1e686c8e39c80d10331919ada12bfd7cb5ed492b8fb21bd"
    )


def test_exact_smoke_selection_and_explicit_full_grouping() -> None:
    smoke = launcher.DEFAULT_SMOKE_TASK_IDENTITY
    plan = _plan(size=34)
    plan["tasks"][17]["task_identity_sha256"] = smoke
    identities = [str(task["task_identity_sha256"]) for task in plan["tasks"]]

    assert launcher._submission_groups(
        plan,
        identities,
        completed=set(),
        group_size=1,
        group_limit=1,
        smoke_task_identity=smoke,
    ) == [[smoke]]
    assert (
        launcher._submission_groups(
            plan,
            identities,
            completed={smoke},
            group_size=1,
            group_limit=1,
            smoke_task_identity=smoke,
        )
        == []
    )

    full = launcher._submission_groups(
        plan,
        identities,
        completed=set(),
        group_size=16,
        group_limit=0,
        smoke_task_identity="",
    )
    assert [len(group) for group in full] == [16, 16, 2]
    assert [identity for group in full for identity in group] == identities


def test_every_remote_plan_phase_requires_the_execution_commit() -> None:
    plan = {"binding": {"execution_commit": "1" * 40}}
    launcher._require_plan_revision(plan, {"commit": "1" * 40})
    with pytest.raises(RuntimeError, match="execution commit"):
        launcher._require_plan_revision(plan, {"commit": "2" * 40})


def test_bounded_reducers_request_explicit_scratch_disk() -> None:
    source = (ROOT / launcher.LAUNCHER_SOURCE).read_text()
    tree = ast.parse(source)
    for name in ("prepare_sentinel", "finalize"):
        decorator = ast.get_source_segment(source, _function(tree, name).decorator_list[0])
        assert decorator is not None
        assert "ephemeral_disk=REDUCTION_EPHEMERAL_DISK_MB" in decorator
    assert launcher.REDUCTION_EPHEMERAL_DISK_MB == 8192
