"""Safety contract for single-task semantic-migration recovery."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from modal_apps import recover_editing_v2_semantic_migration_task_app as recovery


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _fixture_target(root: Path) -> tuple[dict[str, object], dict[str, object], Path]:
    run_root = "/artifacts/editing_v2/semantic_migration/run-a"
    task_id = "a" * 64
    output = f"{run_root}/tasks/{task_id}"
    task = {
        "task_identity_sha256": task_id,
        "data_lane": "observed_local_analogue",
        "split": "train",
        "source_artifact_path": "/artifacts/source.jsonl",
        "source_shard_sha256": "b" * 64,
        "output_artifact_path": output,
    }
    plan = {
        "training_authorized": False,
        "plan_sha256": "c" * 64,
        "run_identity_sha256": "d" * 64,
        "run_artifact_root": run_root,
        "source_revision": {"source_revision_sha256": "e" * 64},
        "implementation_revision": "f" * 40,
        "process_identity": {"process_identity_sha256": "1" * 64},
        "builder_identity": {"identity_sha256": "2" * 64},
        "task_inventory_sha256": "3" * 64,
        "tasks": [task],
    }
    plan_path = root / "editing_v2/semantic_migration/run-a/SEMANTIC_MIGRATION_PLAN.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_bytes(recovery._canonical_bytes(plan) + b"\n")
    request = {
        "plan_artifact_path": (f"{run_root}/SEMANTIC_MIGRATION_PLAN.json"),
        "expected_plan_file_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "expected_plan_sha256": plan["plan_sha256"],
        "expected_run_identity_sha256": plan["run_identity_sha256"],
        "expected_core_source_revision_sha256": plan["source_revision"]["source_revision_sha256"],
        "task_identity_sha256": task_id,
        "expected_task_output_artifact_path": output,
    }
    return plan, request, plan_path


def test_image_mount_and_nonpreemptible_worker_are_narrow() -> None:
    source = Path(recovery.__file__).read_text()
    tree = ast.parse(source)
    assert recovery.IMAGE_SOURCE_DIRECTORIES == ("src", "configs")
    assert recovery.IMAGE_SOURCE_FILES == (
        recovery.LAUNCHER_SOURCE,
        recovery.MIGRATION_LAUNCHER_SOURCE,
    )
    assert "for source_directory in IMAGE_SOURCE_DIRECTORIES" in source
    assert "for source_file in IMAGE_SOURCE_FILES" in source
    assert "str(REMOTE_ROOT / source_directory)" in source
    assert "str(REMOTE_ROOT / source_file)" in source

    worker = _function(tree, "recover_one_task")
    decorator = next(
        item
        for item in worker.decorator_list
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == "function"
    )
    keywords = {keyword.arg: keyword.value for keyword in decorator.keywords}
    assert isinstance(keywords["nonpreemptible"], ast.Constant)
    assert keywords["nonpreemptible"].value is True
    assert "gpu" not in keywords
    assert isinstance(keywords["cpu"], ast.Constant)
    assert keywords["cpu"].value == 2.0
    worker_source = ast.get_source_segment(source, worker)
    assert worker_source is not None
    assert 'loaded["execute_semantic_trace_migration_task"]' in worker_source
    assert "artifact_volume.reload()" in worker_source
    assert "artifact_volume.commit()" in worker_source
    assert "reduce_semantic_trace_migration" not in worker_source


def test_serialized_source_inventory_exactly_matches_mounted_closure() -> None:
    paths = recovery._serialized_source_paths(recovery.ROOT)
    assert tuple(sorted(paths)) == paths
    assert len(paths) == len(set(paths))
    assert set(recovery.IMAGE_SOURCE_FILES).issubset(paths)
    for relative in paths:
        path = recovery.ROOT / relative
        assert path.is_file()
        assert "__pycache__" not in path.parts
        assert path.suffix != ".pyc"
        assert relative in recovery.IMAGE_SOURCE_FILES or relative.startswith(
            recovery.IMAGE_SOURCE_DIRECTORIES
        )


def test_exact_plan_and_single_task_are_authenticated(tmp_path: Path) -> None:
    plan, request, plan_path = _fixture_target(tmp_path)
    loaded = {
        "PLAN_FILENAME": plan_path.name,
        "load_semantic_trace_migration_plan": lambda path, *, repo_root: plan,
    }
    observed_plan, task, observed_path = recovery._load_exact_target(
        request,
        artifact_root=tmp_path,
        repo_root=tmp_path,
        loaded=loaded,
    )
    assert observed_plan is plan
    assert task["task_identity_sha256"] == request["task_identity_sha256"]
    assert observed_path == plan_path


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("plan_file_sha", "plan file SHA-256"),
        ("noncanonical_plan", "plan bytes are not canonical"),
        ("plan_identity", "plan identity or authority"),
        ("missing_task", "absent or duplicated"),
        ("duplicate_task", "absent or duplicated"),
        ("output_address", "task output address"),
    ],
)
def test_plan_and_task_validation_fail_closed(tmp_path: Path, mutation: str, match: str) -> None:
    plan, request, plan_path = _fixture_target(tmp_path)
    if mutation == "plan_file_sha":
        request["expected_plan_file_sha256"] = "0" * 64
    elif mutation == "noncanonical_plan":
        plan_path.write_bytes(recovery._canonical_bytes(plan, pretty=True))
        request["expected_plan_file_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    elif mutation == "plan_identity":
        request["expected_plan_sha256"] = "0" * 64
    elif mutation == "missing_task":
        request["task_identity_sha256"] = "0" * 64
    elif mutation == "duplicate_task":
        plan["tasks"] = [plan["tasks"][0], dict(plan["tasks"][0])]
        plan_path.write_bytes(recovery._canonical_bytes(plan) + b"\n")
        request["expected_plan_file_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    elif mutation == "output_address":
        request["expected_task_output_artifact_path"] = "/artifacts/wrong/output"
    loaded = {
        "PLAN_FILENAME": plan_path.name,
        "load_semantic_trace_migration_plan": lambda path, *, repo_root: plan,
    }
    with pytest.raises(RuntimeError, match=match):
        recovery._load_exact_target(
            request,
            artifact_root=tmp_path,
            repo_root=tmp_path,
            loaded=loaded,
        )


def test_recovery_receipt_grants_no_downstream_authority(tmp_path: Path) -> None:
    plan, request, plan_path = _fixture_target(tmp_path)
    request["request_sha256"] = "4" * 64
    request["source_revision"] = {"source_revision_sha256": "5" * 64}
    task = plan["tasks"][0]
    result = {
        "task_identity_sha256": task["task_identity_sha256"],
        "output_artifact_path": task["output_artifact_path"],
        "receipt_sha256": "6" * 64,
        "counts": {"source": 1, "admitted": 1, "rejected": 0},
        "reused": False,
    }
    receipt = recovery._build_receipt(
        request=request,
        plan=plan,
        task=task,
        result=result,
        plan_file_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    )
    assert receipt["status"] == recovery.RECEIPT_STATUS
    for authority in recovery._AUTHORITY_FIELDS:
        assert receipt[authority] is False
    assert receipt["training_launched"] is False
    assert receipt["downstream_stage_launched"] is False
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert receipt["receipt_sha256"] == recovery._sha(body)
    assert (
        receipt["upstream_plan"]["plan_file_sha256"]
        == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    )
