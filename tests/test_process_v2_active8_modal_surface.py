"""Static and local gates for the Process-V2 Active8 Modal orchestration."""

from __future__ import annotations

import ast
import inspect
import json
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
        "plan_sha256": "b" * 64,
        "binding": {"rebind_completion_sha256": "c" * 64},
        "run_artifact_root": "/artifacts/editing_v2/process_v2_active8/" + "a" * 64,
        "tasks": [
            {
                "task_identity_sha256": f"{index:064x}",
                "rebind_task_identity_sha256": f"{index + 10_000:064x}",
                "chunk_row_count": index + 1,
            }
            for index in range(size)
        ],
    }


def _pilot_plan() -> dict[str, object]:
    plan = _plan(size=100)
    for index, task in enumerate(plan["tasks"]):
        task["split"] = "train" if index < 90 else "validation"
        task["data_lane"] = "observed_local_analogue"
    plan["tasks"][80].update(launcher.DEFAULT_SYNTHETIC_CANARY_SELECTOR)
    plan["tasks"][-1].update(launcher.DEFAULT_SMOKE_TASK_SELECTOR)
    return plan


def _task_workloads(
    plan: dict[str, object], transitions: list[int] | None = None
) -> dict[str, object]:
    values = transitions or [index + 1 for index in range(len(plan["tasks"]))]
    assert len(values) == len(plan["tasks"])
    completion = {
        "completion_sha256": plan["binding"]["rebind_completion_sha256"],
        "counts": {"admitted_transitions": sum(values)},
        "result_inventory": [
            {
                "task_identity_sha256": task["rebind_task_identity_sha256"],
                "counts": {"admitted_transitions": value},
            }
            for task, value in zip(plan["tasks"], values, strict=True)
        ],
    }
    return launcher._build_task_workloads(plan, completion)


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
    assert launcher.MAP_CPU == 1.0
    assert launcher.FULL_MAP_TASKS_PER_CONTAINER == 1
    assert launcher.MAP_MEMORY_MB == 4 * 1024
    assert launcher.MAX_MAP_CONTAINERS == 80
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
    groups = launcher._task_groups(plan, identities, group_size=1)

    assert [int(group[0], 16) for group in groups] == list(range(12))
    assert all(len(group) == 1 for group in groups)
    assert {identity for group in groups for identity in group} == set(identities)
    assert sum(len(group) for group in groups) == len(identities)


def test_task_workloads_are_an_exact_rebind_to_active8_join() -> None:
    plan = _plan(size=5)
    observed = _task_workloads(plan, [9, 1, 7, 3, 5])

    assert observed == {
        str(task["task_identity_sha256"]): value
        for task, value in zip(plan["tasks"], [9, 1, 7, 3, 5], strict=True)
    }
    assert sum(observed.values()) == 25


@pytest.mark.parametrize(
    "mutation",
    ["wrong_completion", "missing_result", "duplicate_result", "boolean_count", "wrong_total"],
)
def test_task_workloads_refuse_incomplete_or_untyped_evidence(mutation: str) -> None:
    plan = _plan(size=3)
    completion = {
        "completion_sha256": plan["binding"]["rebind_completion_sha256"],
        "counts": {"admitted_transitions": 6},
        "result_inventory": [
            {
                "task_identity_sha256": task["rebind_task_identity_sha256"],
                "counts": {"admitted_transitions": index + 1},
            }
            for index, task in enumerate(plan["tasks"])
        ],
    }
    if mutation == "wrong_completion":
        completion["completion_sha256"] = "d" * 64
    elif mutation == "missing_result":
        completion["result_inventory"].pop()
        completion["counts"]["admitted_transitions"] = 3
    elif mutation == "duplicate_result":
        completion["result_inventory"][1]["task_identity_sha256"] = completion["result_inventory"][
            0
        ]["task_identity_sha256"]
    elif mutation == "boolean_count":
        completion["result_inventory"][0]["counts"]["admitted_transitions"] = True
    else:
        completion["counts"]["admitted_transitions"] = 7

    with pytest.raises(RuntimeError, match="workload"):
        launcher._build_task_workloads(plan, completion)


def test_workload_ordering_submits_heaviest_tasks_first_and_is_restart_safe() -> None:
    plan = _plan(size=12)
    transitions = [1, 100, 2, 99, 3, 98, 4, 97, 5, 96, 6, 95]
    workloads = _task_workloads(plan, transitions)
    identities = [str(task["task_identity_sha256"]) for task in plan["tasks"]]
    complete = {identities[3]}
    missing = [identity for identity in identities if identity not in complete]

    groups = launcher._workload_ordered_task_groups(
        plan,
        missing,
        group_size=1,
        task_workloads=workloads,
    )
    work = launcher._validate_task_workloads(plan, workloads)

    assert [len(group) for group in groups] == [1] * 11
    assert {identity for group in groups for identity in group} == set(missing)
    assert not ({identity for group in groups for identity in group} & complete)
    flattened_work = [work[identity] for group in groups for identity in group]
    assert flattened_work == sorted((work[identity] for identity in missing), reverse=True)


def test_completed_tasks_are_removed_before_restart_groups_are_built() -> None:
    plan = _plan()
    completed = {f"{2:064x}", f"{7:064x}"}
    missing = launcher._partition_task_ids(
        plan, partition_count=1, partition_index=0, completed=completed
    )
    groups = launcher._task_groups(plan, missing, group_size=1)

    submitted = {identity for group in groups for identity in group}
    assert submitted.isdisjoint(completed)
    assert submitted | completed == {str(task["task_identity_sha256"]) for task in plan["tasks"]}


@pytest.mark.parametrize("group_size", [0, 2, True])
def test_invalid_process_group_size_is_refused(group_size: int) -> None:
    with pytest.raises(ValueError, match="group_size"):
        launcher._task_groups(_plan(), [], group_size=group_size)


def test_tasks_are_independent_right_sized_and_measured() -> None:
    source = (ROOT / launcher.LAUNCHER_SOURCE).read_text()
    tree = ast.parse(source)
    node = _function(tree, "decide_task")
    body = ast.get_source_segment(source, node)
    decorator = ast.get_source_segment(source, node.decorator_list[0])
    assert body is not None and decorator is not None
    assert "ProcessPoolExecutor" not in body
    assert body.count("artifact_volume.commit()") == 1
    assert body.index("validate_process_v2_active8_task_result") < body.index(
        "artifact_volume.commit()"
    )
    for field in ("wall_seconds", "cpu_seconds", "process_peak_rss_mb"):
        assert field in body
    assert "cpu=MAP_CPU" in decorator
    assert "memory=MAP_MEMORY_MB" in decorator
    assert "max_containers=MAX_MAP_CONTAINERS" in decorator


def test_first_launch_defaults_to_three_independent_train_canaries() -> None:
    parameters = inspect.signature(launcher.main.info.raw_f).parameters
    assert launcher.PILOT_GROUP_COUNT == 3
    assert launcher.PILOT_GROUP_SIZE == 1
    assert launcher.PILOT_TASK_COUNT == 3
    assert launcher.MAP_CPU == 1.0
    assert launcher.MAP_MEMORY_MB == 4 * 1024
    assert launcher.MAX_MAP_CONTAINERS == 80
    assert parameters["group_size"].default == launcher.PILOT_GROUP_SIZE
    assert parameters["group_limit"].default == launcher.PILOT_GROUP_COUNT
    assert parameters["max_map_containers"].default == launcher.MAX_MAP_CONTAINERS
    assert parameters["pilot"].default is True
    assert parameters["full_map"].default is False
    assert parameters["reduce"].default is False
    assert launcher.DEFAULT_SMOKE_TASK_SELECTOR == {
        "split": "train",
        "data_lane": "real_endpoint_multistep_path",
        "chunk_index": 24,
        "entry_start": 49_152,
        "entry_stop": 51_200,
        "chunk_row_count": 2_048,
    }


def test_exact_pilot_selection_and_explicit_full_grouping() -> None:
    plan = _pilot_plan()
    identities = [str(task["task_identity_sha256"]) for task in plan["tasks"]]
    canary = identities[-1]

    pilot = launcher._submission_groups(
        plan,
        identities,
        completed=set(),
        group_size=1,
        group_limit=3,
        pilot=True,
        full_map=False,
    )
    pilot_ids = [identity for group in pilot for identity in group]
    assert [len(group) for group in pilot] == [1, 1, 1]
    assert pilot_ids == [identities[0], identities[80], canary]
    assert canary in pilot_ids
    task_by_identity = {str(task["task_identity_sha256"]): task for task in plan["tasks"]}
    assert {task_by_identity[identity]["split"] for identity in pilot_ids} == {"train"}

    full = launcher._submission_groups(
        plan,
        identities,
        completed=set(),
        group_size=1,
        group_limit=0,
        pilot=False,
        full_map=True,
        task_workloads=_task_workloads(plan),
    )
    assert [len(group) for group in full] == [1] * 100
    assert [identity for group in full for identity in group] == list(reversed(identities))


def test_pilot_partial_and_full_restarts_submit_only_unfinished_cohort_tasks() -> None:
    plan = _pilot_plan()
    cohort = launcher._pilot_task_ids(plan)
    completed = {cohort[1]}
    missing = launcher._partition_task_ids(
        plan, partition_count=1, partition_index=0, completed=completed
    )

    groups = launcher._submission_groups(
        plan,
        missing,
        completed=completed,
        group_size=1,
        group_limit=3,
        pilot=True,
        full_map=False,
    )
    submitted = [identity for group in groups for identity in group]
    assert [len(group) for group in groups] == [1, 1]
    assert submitted == [identity for identity in cohort if identity not in completed]
    assert set(submitted).isdisjoint(completed)

    all_cohort_complete = set(cohort)
    missing_after_full_restart = launcher._partition_task_ids(
        plan,
        partition_count=1,
        partition_index=0,
        completed=all_cohort_complete,
    )
    assert (
        launcher._submission_groups(
            plan,
            missing_after_full_restart,
            completed=all_cohort_complete,
            group_size=1,
            group_limit=3,
            pilot=True,
            full_map=False,
        )
        == []
    )


def test_pilot_locator_survives_binding_derived_task_identity_changes() -> None:
    left = _pilot_plan()
    right = _pilot_plan()
    left["tasks"][-1]["task_identity_sha256"] = "a" * 64
    right["tasks"][-1]["task_identity_sha256"] = "b" * 64

    assert launcher._pilot_task_ids(left) == [
        str(left["tasks"][0]["task_identity_sha256"]),
        str(left["tasks"][80]["task_identity_sha256"]),
        "a" * 64,
    ]
    assert launcher._pilot_task_ids(right) == [
        str(right["tasks"][0]["task_identity_sha256"]),
        str(right["tasks"][80]["task_identity_sha256"]),
        "b" * 64,
    ]


@pytest.mark.parametrize(("pilot", "full_map"), [(False, False), (True, True)])
def test_pilot_and_full_map_are_mutually_exclusive(pilot: bool, full_map: bool) -> None:
    plan = _pilot_plan()
    identities = [str(task["task_identity_sha256"]) for task in plan["tasks"]]
    with pytest.raises(ValueError, match="exactly one"):
        launcher._submission_groups(
            plan,
            identities,
            completed=set(),
            group_size=1,
            group_limit=3,
            pilot=pilot,
            full_map=full_map,
        )


@pytest.mark.parametrize(
    ("pilot", "full_map", "reduce"),
    [
        (False, False, False),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_remote_driver_requires_exactly_one_phase(
    monkeypatch: pytest.MonkeyPatch,
    pilot: bool,
    full_map: bool,
    reduce: bool,
) -> None:
    monkeypatch.setattr(launcher, "_validate_remote_revision", lambda _revision: None)
    with pytest.raises(ValueError, match="exactly one Active8 phase"):
        launcher.driver.info.raw_f(
            "/cache",
            "/rebind",
            launcher.OUTPUT_ARTIFACT_PREFIX,
            1,
            0,
            16,
            5,
            launcher.MAX_MAP_CONTAINERS,
            pilot,
            full_map,
            reduce,
            launcher.DEFAULT_SENTINEL_PAIRS_PER_PARTITION,
            {},
        )


@pytest.mark.parametrize("max_map_containers", [0, 81, True])
def test_remote_driver_refuses_invalid_runtime_container_bound(
    monkeypatch: pytest.MonkeyPatch, max_map_containers: int
) -> None:
    monkeypatch.setattr(launcher, "_validate_remote_revision", lambda _revision: None)
    with pytest.raises(ValueError, match="max_map_containers"):
        launcher.driver.info.raw_f(
            "/cache",
            "/rebind",
            launcher.OUTPUT_ARTIFACT_PREFIX,
            1,
            0,
            launcher.PILOT_GROUP_SIZE,
            launcher.PILOT_GROUP_COUNT,
            max_map_containers,
            True,
            False,
            False,
            launcher.DEFAULT_SENTINEL_PAIRS_PER_PARTITION,
            {},
        )


def test_local_main_only_spawns_the_remote_driver() -> None:
    source = (ROOT / launcher.LAUNCHER_SOURCE).read_text()
    tree = ast.parse(source)
    main = ast.get_source_segment(source, _function(tree, "main"))
    assert main is not None
    assert main.count("driver.spawn(") == 1
    assert ".remote(" not in main
    assert ".starmap(" not in main


def test_remote_driver_owns_orchestration_without_a_volume_mount_or_commit() -> None:
    source = (ROOT / launcher.LAUNCHER_SOURCE).read_text()
    tree = ast.parse(source)
    driver_node = _function(tree, "driver")
    driver = ast.get_source_segment(source, driver_node)
    decorator = ast.get_source_segment(source, driver_node.decorator_list[0])
    assert driver is not None and decorator is not None
    for call in (
        "prepare_plan.remote(",
        "scan_completed.remote(",
        "decide_task.starmap(",
        "prepare_sentinel.remote(",
        "run_sentinel_partition.starmap(",
        "finalize.remote(",
    ):
        assert call in driver
    assert "decide_task.update_autoscaler(max_containers=max_map_containers)" in driver
    assert "volumes=" not in decorator
    assert "artifact_volume" not in driver
    assert ".commit(" not in driver
    assert ".reload(" not in driver


def test_every_remote_plan_phase_requires_the_execution_commit() -> None:
    plan = {"binding": {"execution_commit": "1" * 40}}
    launcher._require_plan_revision(plan, {"commit": "1" * 40})
    with pytest.raises(RuntimeError, match="execution commit"):
        launcher._require_plan_revision(plan, {"commit": "2" * 40})


def test_bounded_reducers_use_the_default_modal_scratch_disk() -> None:
    source = (ROOT / launcher.LAUNCHER_SOURCE).read_text()
    tree = ast.parse(source)
    for name in ("prepare_sentinel", "finalize"):
        decorator = ast.get_source_segment(source, _function(tree, name).decorator_list[0])
        assert decorator is not None
        assert "ephemeral_disk=" not in decorator
