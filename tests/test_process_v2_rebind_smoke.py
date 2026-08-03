"""The reusable one-chunk Process-V2 rebind smoke is bounded and nonauthorizing."""

from __future__ import annotations

import ast
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "modal_apps" / "run_process_v2_rebind_app.py"


def _load_app():
    spec = importlib.util.spec_from_file_location("process_v2_rebind_smoke_app", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app():
    return _load_app()


def _task(app, **updates):
    value = {
        "v1_task_identity_sha256": app.SMOKE_V1_TASK_IDENTITY_SHA256,
        "data_lane": app.SMOKE_DATA_LANE,
        "split": app.SMOKE_SPLIT,
        "chunk_index": app.SMOKE_CHUNK_INDEX,
        "entry_start": app.SMOKE_ENTRY_START,
        "entry_stop": app.SMOKE_ENTRY_STOP,
        "chunk_row_count": app.SMOKE_EXPECTED_ROWS,
        "task_identity_sha256": "b" * 64,
    }
    value.update(updates)
    return value


def _plan(app, *, tasks=None):
    return {
        "tasks": [_task(app)] if tasks is None else tasks,
        "plan_sha256": "a" * 64,
        "run_identity_sha256": "c" * 64,
        "run_artifact_root": "/artifacts/editing_v2/process_v2_rebind/" + "c" * 64,
        "source_revision": {"source_revision_sha256": "e" * 64},
        "cache_binding": {
            "cache_run_artifact_root": (
                "/artifacts/editing_v2/process_v2_chunk_cache/" + "f" * 64
            )
        },
    }


def _record(app, plan, task, **updates):
    body = {
        "schema": app.SMOKE_SCHEMA,
        "schema_version": app.SMOKE_SCHEMA_VERSION,
        "status": app.SMOKE_STATUS,
        **app._smoke_authority_envelope(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": "2" * 40,
        "code_tree": "3" * 40,
        "image_revision_sha256": "d" * 64,
        "source_revision_sha256": "e" * 64,
        "cache_run_artifact_root": (
            "/artifacts/editing_v2/process_v2_chunk_cache/" + "f" * 64
        ),
        "plan_sha256": plan["plan_sha256"],
        "run_identity_sha256": plan["run_identity_sha256"],
        "run_artifact_root": plan["run_artifact_root"],
        "task_identity_sha256": task["task_identity_sha256"],
        "task_address": {
            "v1_task_identity_sha256": app.SMOKE_V1_TASK_IDENTITY_SHA256,
            "data_lane": app.SMOKE_DATA_LANE,
            "split": app.SMOKE_SPLIT,
            "chunk_index": app.SMOKE_CHUNK_INDEX,
            "entry_start": app.SMOKE_ENTRY_START,
            "entry_stop": app.SMOKE_ENTRY_STOP,
            "row_count": app.SMOKE_EXPECTED_ROWS,
        },
        "receipt_sha256": "1" * 64,
        "counts": {
            "source_entries": app.SMOKE_EXPECTED_ROWS,
            "admitted_entries": app.SMOKE_EXPECTED_ROWS - 1,
            "rejected_entries": 1,
        },
        "resource_request": {
            "cpu": app.MAP_CPU,
            "memory_mib": app.MAP_MEMORY_MB,
            "timeout_seconds": app.MAP_TIMEOUT_SECONDS,
            "worker_count": 1,
        },
        "thresholds": {
            "max_wall_seconds": app.SMOKE_MAX_WALL_SECONDS,
            "max_peak_rss_mib": app.SMOKE_MAX_PEAK_RSS_MIB,
        },
        "measurements": {
            "wall_seconds": 1.5,
            "user_cpu_seconds": 1.0,
            "system_cpu_seconds": 0.5,
            "peak_rss_mib": 512.0,
        },
        "environment": {
            "python": "3.11",
            "platform": "Linux",
            "processor": "x86_64",
            "logical_cpu_count": 2,
            "torch": "2.4.0",
            "numpy": "1.26.4",
            "rdkit": "2024.3.5",
            "modal": "1.3.5",
            "precision": "not_applicable_cpu_rebind",
        },
        "smoke_result": "PASS",
    }
    body.update(updates)
    return {**body, "diagnostic_sha256": app._sha256(body)}


def _function_source(name: str) -> str:
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name
    )
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def test_the_smoke_selects_one_exact_full_chunk(app) -> None:
    plan = _plan(app)
    assert app._select_frozen_smoke_task(plan) == plan["tasks"][0]

    for field, value in (
        ("data_lane", "another_lane"),
        ("split", "validation"),
        ("entry_start", app.SMOKE_ENTRY_START + 1),
        ("entry_stop", app.SMOKE_ENTRY_STOP - 1),
        ("chunk_row_count", app.SMOKE_EXPECTED_ROWS - 1),
    ):
        with pytest.raises(RuntimeError, match="smoke chunk moved"):
            app._select_frozen_smoke_task(_plan(app, tasks=[_task(app, **{field: value})]))

    with pytest.raises(RuntimeError, match="exactly one"):
        app._select_frozen_smoke_task(_plan(app, tasks=[_task(app), _task(app)]))


def test_the_smoke_record_is_self_hashed_reconciled_and_nonauthorizing(app) -> None:
    plan = _plan(app)
    task = plan["tasks"][0]
    record = _record(app, plan, task)
    assert app._validate_smoke_record(record, plan=plan, task=task) == record

    for field in app._smoke_authority_envelope():
        mutated = dict(record)
        mutated[field] = True
        body = {key: item for key, item in mutated.items() if key != "diagnostic_sha256"}
        mutated["diagnostic_sha256"] = app._sha256(body)
        with pytest.raises(RuntimeError, match="stale or self-inconsistent"):
            app._validate_smoke_record(mutated, plan=plan, task=task)

    mutated = dict(record)
    mutated["counts"] = {
        "source_entries": app.SMOKE_EXPECTED_ROWS,
        "admitted_entries": app.SMOKE_EXPECTED_ROWS,
        "rejected_entries": 1,
    }
    body = {key: item for key, item in mutated.items() if key != "diagnostic_sha256"}
    mutated["diagnostic_sha256"] = app._sha256(body)
    with pytest.raises(RuntimeError, match="census does not reconcile"):
        app._validate_smoke_record(mutated, plan=plan, task=task)


def test_a_failed_threshold_is_recorded_rather_than_relabelled_pass(app) -> None:
    plan = _plan(app)
    task = plan["tasks"][0]
    record = _record(
        app,
        plan,
        task,
        measurements={
            "wall_seconds": app.SMOKE_MAX_WALL_SECONDS + 1.0,
            "user_cpu_seconds": 1.0,
            "system_cpu_seconds": 0.5,
            "peak_rss_mib": 512.0,
        },
    )
    with pytest.raises(RuntimeError, match="decision disagrees"):
        app._validate_smoke_record(record, plan=plan, task=task)

    failed = dict(record)
    failed["smoke_result"] = "FAIL"
    body = {key: item for key, item in failed.items() if key != "diagnostic_sha256"}
    failed["diagnostic_sha256"] = app._sha256(body)
    assert app._validate_smoke_record(failed, plan=plan, task=task)["smoke_result"] == "FAIL"


def test_the_smoke_executes_normal_work_skips_reduction_and_commits_once() -> None:
    remote = _function_source("smoke_one_chunk_driver")
    assert "build_cache_fed_plan(" in remote
    assert "write_process_v2_rebind_plan(" in remote
    assert "completed_process_v2_rebind_task_ids(" in remote
    assert "execute_process_v2_rebind_task(" in remote
    assert "reduce_process_v2_rebind" not in remote
    assert "reduce_rebind.remote" not in remote
    assert "artifact_volume.commit()" in remote
    assert "_publish_smoke_record(" in remote

    local = _function_source("smoke_one_chunk")
    assert "local_image_revision(" in local
    assert "repository_process_v2_rebind_source_revision(" in local
    assert "smoke_one_chunk_driver.remote(" in local
    assert "SMOKE_V1_TASK_IDENTITY_SHA256" not in local


def test_the_diagnostic_namespace_is_outside_the_production_run(app) -> None:
    plan = _plan(app)
    task = plan["tasks"][0]
    path = app._smoke_artifact_path(plan, task)
    assert str(path).startswith("/artifacts/editing_v2/process_v2_rebind_smoke/")
    assert not str(path).startswith(str(plan["run_artifact_root"]))
    assert path.name == app.SMOKE_FILENAME


def test_record_serialization_is_stable_json(app) -> None:
    plan = _plan(app)
    task = plan["tasks"][0]
    record = _record(app, plan, task)
    assert json.loads(app._canonical_bytes(record)) == record
