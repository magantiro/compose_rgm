"""Focused contract tests for the new-only semantic V4 Modal migration job."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modal_apps import materialize_editing_v2_semantic_v4_migration_app as migration_app

SHA = "a" * 64


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _self_hashed(body: dict[str, object], field: str) -> dict[str, object]:
    return {**body, field: migration_app._canonical_sha256(body)}


def _source_revision() -> dict[str, object]:
    core = {
        "schema": "compose.data.semantic_trace_migration_source_revision",
        "schema_version": 1,
        "commit": "1" * 40,
        "tree": "2" * 40,
        "worktree_clean": True,
        "implementation_files": {"core.py": SHA},
        "implementation_files_sha256": SHA,
        "process_identity_sha256": SHA,
        "builder_identity_sha256": SHA,
        "source_revision_sha256": SHA,
    }
    body: dict[str, object] = {
        "schema": migration_app.SOURCE_REVISION_SCHEMA,
        "schema_version": migration_app.SOURCE_REVISION_SCHEMA_VERSION,
        "commit": "1" * 40,
        "tree": "2" * 40,
        "worktree_clean": True,
        "launcher": {
            "relative_path": migration_app.LAUNCHER_SOURCE,
            "file_sha256": SHA,
        },
        "semantic_core_revision": core,
    }
    return {**body, "source_revision_sha256": migration_app._canonical_sha256(body)}


def _structural_identity() -> dict[str, object]:
    return {
        "completion_artifact_path": "/artifacts/split/SPLIT_PIPELINE_COMPLETE.json",
        "completion_file_sha256": "1" * 64,
        "completion_sha256": "2" * 64,
        "structural_run_identity_sha256": "3" * 64,
        "candidate_manifest_sha256": "4" * 64,
        "bridge_manifest_sha256": "5" * 64,
        "candidate_source_stream_sha256": "6" * 64,
        "split_assignment_sha256": "7" * 64,
        "lane_registry_sha256": "8" * 64,
        "membership_receipt_sha256": "9" * 64,
        "source_shard_count": 20,
    }


def test_run_request_is_content_addressed_and_cannot_authorize_later_stages() -> None:
    first = migration_app.build_run_request(
        source_revision=_source_revision(),
        structural_identity=_structural_identity(),
        python_runtime={"implementation": "CPython", "version": "3.11.9"},
        max_row_bytes=1024,
    )
    second = migration_app.build_run_request(
        source_revision=_source_revision(),
        structural_identity=_structural_identity(),
        python_runtime={"implementation": "CPython", "version": "3.11.9"},
        max_row_bytes=1024,
    )

    assert first == second
    assert first["run_identity_sha256"] == migration_app._canonical_sha256(
        {key: value for key, value in first.items() if key != "run_identity_sha256"}
    )
    assert all(first[field] is False for field in migration_app._AUTHORITY_FIELDS)
    assert first["structural_input"]["source_shard_count"] == 20

    too_few = {**_structural_identity(), "source_shard_count": 19}
    with pytest.raises(RuntimeError, match="exactly 20"):
        migration_app.build_run_request(
            source_revision=_source_revision(),
            structural_identity=too_few,
            python_runtime={"implementation": "CPython", "version": "3.11.9"},
            max_row_bytes=1024,
        )


def test_local_source_revision_requires_exact_clean_commit_and_binds_core() -> None:
    core = {"commit": "1" * 40, "tree": "2" * 40, "source_revision_sha256": SHA}
    with (
        patch.object(migration_app, "_git", side_effect=["1" * 40, "2" * 40, ""]),
        patch.object(migration_app, "_file_sha256", return_value=SHA),
        patch(
            "compose_v4.data.semantic_trace_migration_mapreduce."
            "build_semantic_migration_source_revision",
            return_value=core,
        ),
    ):
        revision = migration_app.local_source_revision(
            expected_commit="1" * 40,
            repo_root=Path("/fixture"),
        )
    assert revision["commit"] == "1" * 40
    assert revision["tree"] == "2" * 40
    assert revision["semantic_core_revision"] == core
    assert revision["launcher"]["file_sha256"] == SHA

    with (
        patch.object(
            migration_app,
            "_git",
            side_effect=["1" * 40, "2" * 40, " M src/dirty.py"],
        ),
        pytest.raises(RuntimeError, match="clean committed"),
    ):
        migration_app.local_source_revision(
            expected_commit="1" * 40,
            repo_root=Path("/fixture"),
        )


def test_remote_source_revision_reopens_launcher_and_core_hashes(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    launcher_path = remote_root / migration_app.LAUNCHER_SOURCE
    launcher_path.parent.mkdir(parents=True)
    launcher_path.write_text("# exact serialized launcher\n")
    core = {"commit": "1" * 40, "tree": "2" * 40}
    body: dict[str, object] = {
        "schema": migration_app.SOURCE_REVISION_SCHEMA,
        "schema_version": migration_app.SOURCE_REVISION_SCHEMA_VERSION,
        "commit": "1" * 40,
        "tree": "2" * 40,
        "worktree_clean": True,
        "launcher": {
            "relative_path": migration_app.LAUNCHER_SOURCE,
            "file_sha256": migration_app._file_sha256(launcher_path),
        },
        "semantic_core_revision": core,
    }
    revision = {
        **body,
        "source_revision_sha256": migration_app._canonical_sha256(body),
    }
    loaded = {
        "validate_semantic_migration_source_revision": (
            lambda value, *, repo_root: {**value, "repo_root": str(repo_root)}
        )
    }

    validated = migration_app._validate_source_revision(
        revision,
        remote_root=remote_root,
        loaded=loaded,
    )
    assert validated == revision

    launcher_path.write_text("# changed after serialization\n")
    with pytest.raises(RuntimeError, match="serialized source revision"):
        migration_app._validate_source_revision(
            revision,
            remote_root=remote_root,
            loaded=loaded,
        )


def test_missing_task_selection_is_ordered_restart_safe_and_strict() -> None:
    task_ids = [f"{index:064x}" for index in range(20)]
    plan = {"tasks": [{"task_identity_sha256": task_id} for task_id in task_ids]}

    missing = migration_app._missing_task_ids(plan, {task_ids[1], task_ids[7]})
    assert missing == [task_id for task_id in task_ids if task_id not in {task_ids[1], task_ids[7]}]

    with pytest.raises(RuntimeError, match="unexpected"):
        migration_app._missing_task_ids(plan, {"f" * 64})

    duplicate_plan = {"tasks": [{"task_identity_sha256": task_ids[0]}] * 20}
    with pytest.raises(RuntimeError, match="20 unique"):
        migration_app._missing_task_ids(duplicate_plan, set())


def test_immutable_stage_reuses_exact_bytes_and_rejects_collision(tmp_path: Path) -> None:
    target = tmp_path / "stage.json"
    assert migration_app._write_immutable_json(target, {"value": 1}) is True
    assert migration_app._write_immutable_json(target, {"value": 1}) is False
    with pytest.raises(RuntimeError, match="immutable"):
        migration_app._write_immutable_json(target, {"value": 2})


class _FakeCandidates:
    MATERIALIZATION_FILENAME = "PACKED_CANDIDATE_MATERIALIZATION.json"

    def __init__(self, manifest: dict[str, object]) -> None:
        self.manifest = manifest

    def validate_packed_candidate_materialization(self, *args: object, **kwargs: object) -> dict:
        del args, kwargs
        return self.manifest


class _FakeBridge:
    BRIDGE_MANIFEST_FILENAME = "CANDIDATE_PROVENANCE_BRIDGE.json"

    def __init__(self, manifest: dict[str, object]) -> None:
        self.manifest = manifest

    def validate_candidate_provenance_bridge(self, *args: object, **kwargs: object) -> dict:
        del args, kwargs
        return self.manifest


class _FakeRoleLane:
    ROLE_LANE_MATERIALIZATION_FILENAME = "ROLE_LANE_PACKED_MATERIALIZATION.json"
    LANE_REGISTRY_FILENAME = "EDITING_V2_LANE_REGISTRY.json"
    RESOLVED_MEMBERSHIP_FILENAME = "RESOLVED_PACKED_MEMBERSHIP.json"


def _structural_fixture(tmp_path: Path) -> tuple[str, dict[str, object], dict[str, object]]:
    artifact_root = tmp_path / "artifacts"
    remote_root = tmp_path / "remote"
    candidate_root = artifact_root / "candidate"
    bridge_root = artifact_root / "bridge"
    registry_path = artifact_root / "registry.json"
    contract_path = remote_root / "configs/editing_corpus_v2_contract.json"
    for directory in (candidate_root, bridge_root, contract_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    source_stream = {
        "source_stream_sha256": "7" * 64,
        "candidate_materialization": {"manifest_sha256": "2" * 64},
    }
    candidate_manifest = {
        "manifest_sha256": "2" * 64,
        "rows": {
            "file_sha256": "3" * 64,
            "semantic_sha256": "4" * 64,
            "address_stream_sha256": "5" * 64,
        },
    }
    _write_json(candidate_root / _FakeCandidates.MATERIALIZATION_FILENAME, candidate_manifest)
    candidate_record = {
        "path": "/artifacts/candidate",
        "manifest_file_sha256": migration_app._file_sha256(
            candidate_root / _FakeCandidates.MATERIALIZATION_FILENAME
        ),
        "manifest_sha256": "2" * 64,
        "rows_file_sha256": "3" * 64,
        "rows_semantic_sha256": "4" * 64,
        "address_stream_sha256": "5" * 64,
    }

    bridge_manifest = {
        "manifest_sha256": "6" * 64,
        "source_stream": source_stream,
        "outputs": {
            "candidate_audit_ledger": {
                "relative_path": "ledger.json",
                "file_sha256": "8" * 64,
                "semantic_sha256": "9" * 64,
            },
            "candidate_audit_rows": {
                "relative_path": "rows.jsonl",
                "file_sha256": "a" * 64,
                "semantic_sha256": "b" * 64,
            },
            "split_candidates": {
                "relative_path": "split.jsonl",
                "file_sha256": "c" * 64,
                "semantic_sha256": "d" * 64,
            },
        },
    }
    _write_json(bridge_root / _FakeBridge.BRIDGE_MANIFEST_FILENAME, bridge_manifest)
    bridge_record = {
        "path": "/artifacts/bridge",
        "manifest_file_sha256": migration_app._file_sha256(
            bridge_root / _FakeBridge.BRIDGE_MANIFEST_FILENAME
        ),
        "manifest_sha256": "6" * 64,
        "source_stream_sha256": "7" * 64,
        "outputs": bridge_manifest["outputs"],
    }
    _write_json(registry_path, {"registry_sha256": "e" * 64})
    contract = {
        "contract_id": "editing-v2",
        "schema": "contract",
        "schema_version": 2,
    }
    _write_json(contract_path, contract)

    request_body = {
        "schema": "compose.editing_v2_structural_split_modal_run",
        "schema_version": 1,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "bounded_p50_authorized": False,
        "active8_authorized": False,
        "final_test_selection_authorized": False,
        "source_revision": {"fixture": True},
        "python_runtime": {"implementation": "CPython", "version": "3.11"},
        "candidate": candidate_record,
        "candidate_provenance_bridge": bridge_record,
        "candidate_provenance_registry": {
            "path": "/artifacts/registry.json",
            "file_sha256": migration_app._file_sha256(registry_path),
            "registry_sha256": "e" * 64,
        },
        "split_census_policy": {"fixture": True},
        "split_assignment_policy": {"fixture": True},
        "editing_corpus_contract": {
            "path": str(contract_path),
            "file_sha256": migration_app._file_sha256(contract_path),
            "semantic_sha256": migration_app._canonical_sha256(contract),
            "contract_id": "editing-v2",
            "schema": "contract",
            "schema_version": 2,
        },
        "max_row_bytes": 1024,
        "max_source_row_bytes": 2048,
        "output_prefix": "/artifacts/split_runs",
    }
    request = {
        **request_body,
        "run_identity_sha256": migration_app._canonical_sha256(request_body),
    }
    run_address = f"/artifacts/split_runs/{request['run_identity_sha256']}"
    run_root = artifact_root / "split_runs" / request["run_identity_sha256"]
    role_root = run_root / "role_lane"
    role_root.mkdir(parents=True)

    census = _self_hashed(
        {
            "census_structural_complete": True,
            "invalid_rows": [],
        },
        "census_sha256",
    )
    census_path = run_root / "EDITING_V2_SPLIT_CENSUS_V4.json"
    _write_json(census_path, census)
    census_record = {
        "artifact_path": f"{run_address}/EDITING_V2_SPLIT_CENSUS_V4.json",
        "file_sha256": migration_app._file_sha256(census_path),
        "census_sha256": census["census_sha256"],
        "status": "BLOCKED",
        "census_structural_complete": True,
        "blockers": ["external-authority"],
    }

    assignment = _self_hashed(
        {
            "training_authorized": False,
            "gate_results": {"all_roles": True, "all_lanes": True},
            "source_stream": source_stream,
        },
        "assignment_sha256",
    )
    assignment_path = run_root / "EDITING_V2_SPLIT_ASSIGNMENT.json"
    _write_json(assignment_path, assignment)
    assignment_record = {
        "artifact_path": f"{run_address}/EDITING_V2_SPLIT_ASSIGNMENT.json",
        "file_sha256": migration_app._file_sha256(assignment_path),
        "assignment_sha256": assignment["assignment_sha256"],
        "policy_schema_version": 1,
        "artifact_schema_version": 2,
        "gate_results": assignment["gate_results"],
        "blockers": ["active8.not_run"],
    }

    lane_registry_path = role_root / _FakeRoleLane.LANE_REGISTRY_FILENAME
    membership_path = role_root / _FakeRoleLane.RESOLVED_MEMBERSHIP_FILENAME
    _write_json(lane_registry_path, {"registry_sha256": "f" * 64})
    _write_json(membership_path, {"receipt_sha256": "0" * 64})
    role_manifest_body = {
        "training_authorized": False,
        "active8_admission_status": "NOT_RUN",
        "run_artifact_root": f"{run_address}/role_lane",
    }
    role_manifest = _self_hashed(role_manifest_body, "manifest_sha256")
    role_manifest_path = role_root / _FakeRoleLane.ROLE_LANE_MATERIALIZATION_FILENAME
    _write_json(role_manifest_path, role_manifest)
    role_record = {
        "artifact_root": f"{run_address}/role_lane",
        "manifest_file_sha256": migration_app._file_sha256(role_manifest_path),
        "manifest_sha256": role_manifest["manifest_sha256"],
        "lane_registry_file_sha256": migration_app._file_sha256(lane_registry_path),
        "lane_registry_sha256": "f" * 64,
        "membership_file_sha256": migration_app._file_sha256(membership_path),
        "membership_receipt_sha256": "0" * 64,
        "output_shards": 20,
        "active8_admission_status": "NOT_RUN",
    }

    completion_body = {
        "schema": migration_app.STRUCTURAL_COMPLETION_SCHEMA,
        "schema_version": migration_app.STRUCTURAL_COMPLETION_SCHEMA_VERSION,
        "status": migration_app.STRUCTURAL_COMPLETION_STATUS,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "bounded_p50_authorized": False,
        "active8_admission_status": "NOT_RUN",
        "final_test_selection_use": "forbidden_not_performed",
        "request": request,
        "run_root": run_address,
        "census": census_record,
        "assignment": assignment_record,
        "role_lane_packed": role_record,
        "blockers": ["active8.not_run", "training.not_authorized"],
    }
    completion = {
        **completion_body,
        "completion_sha256": migration_app._canonical_sha256(completion_body),
    }
    completion_path = run_root / migration_app.STRUCTURAL_COMPLETION_FILENAME
    _write_json(completion_path, completion)
    context = {
        "artifact_root": artifact_root,
        "remote_root": remote_root,
        "candidate_manifest": candidate_manifest,
        "bridge_manifest": bridge_manifest,
        "assignment": assignment,
        "role_record": role_record,
    }
    return f"{run_address}/SPLIT_PIPELINE_COMPLETE.json", context, completion


def _fake_loaded(context: dict[str, object]) -> tuple[dict[str, object], SimpleNamespace]:
    role_record = context["role_record"]
    assignment = context["assignment"]
    bridge = context["bridge_manifest"]
    resolved = SimpleNamespace(
        bindings=tuple(range(20)),
        membership_receipt_file_sha256=role_record["membership_file_sha256"],
        membership_receipt_sha256=role_record["membership_receipt_sha256"],
        candidate_materialization_manifest_sha256="2" * 64,
        candidate_provenance_source_stream=bridge["source_stream"],
        split_assignment_sha256=assignment["assignment_sha256"],
        lane_registry_sha256=role_record["lane_registry_sha256"],
    )
    loaded = {
        "candidates": _FakeCandidates(context["candidate_manifest"]),
        "bridge": _FakeBridge(context["bridge_manifest"]),
        "role_lane": _FakeRoleLane,
        "validate_split_component_census": lambda value: value,
        "resolve_editing_v2_active8_sources": lambda **kwargs: resolved,
    }
    return loaded, resolved


def test_structural_completion_reopens_exact_lineage_and_resolves_20_shards(
    tmp_path: Path,
) -> None:
    completion_path, context, _ = _structural_fixture(tmp_path)
    loaded, resolved = _fake_loaded(context)

    _, identity, observed = migration_app._validate_structural_completion(
        completion_path,
        artifact_root=context["artifact_root"],
        remote_root=context["remote_root"],
        max_row_bytes=1024,
        loaded=loaded,
    )

    assert observed is resolved
    assert identity["source_shard_count"] == 20
    assert identity["candidate_manifest_sha256"] == "2" * 64
    assert identity["membership_receipt_sha256"] == "0" * 64


def test_structural_completion_rejects_tampered_role_grid_before_migration(
    tmp_path: Path,
) -> None:
    completion_path, context, completion = _structural_fixture(tmp_path)
    loaded, _ = _fake_loaded(context)
    tampered = copy.deepcopy(completion)
    tampered["role_lane_packed"]["output_shards"] = 19
    body = {key: value for key, value in tampered.items() if key != "completion_sha256"}
    tampered["completion_sha256"] = migration_app._canonical_sha256(body)
    physical = migration_app._artifact_path(
        completion_path,
        artifact_root=context["artifact_root"],
        field="fixture completion",
    )
    _write_json(physical, tampered)

    with pytest.raises(RuntimeError, match="role/lane output"):
        migration_app._validate_structural_completion(
            completion_path,
            artifact_root=context["artifact_root"],
            remote_root=context["remote_root"],
            max_row_bytes=1024,
            loaded=loaded,
        )


def test_wrapper_completion_preserves_every_authority_boundary() -> None:
    request = migration_app.build_run_request(
        source_revision=_source_revision(),
        structural_identity=_structural_identity(),
        python_runtime={"implementation": "CPython", "version": "3.11.9"},
        max_row_bytes=1024,
    )
    completion = migration_app._wrapper_completion(
        request=request,
        core_identity={"task_count": 20},
    )

    assert completion["status"] == migration_app.COMPLETION_STATUS
    assert all(completion[field] is False for field in migration_app._AUTHORITY_FIELDS)
    assert completion["active8_admission_status"] == "NOT_RUN"
    assert completion["gate_zero_status"] == "NOT_RUN"
    assert completion["t1_status"] == "NOT_RUN"
    assert completion["bounded_p50_status"] == "NOT_RUN"
    assert completion["training_status"] == "NOT_RUN"
    assert completion["completion_sha256"] == migration_app._canonical_sha256(
        {key: value for key, value in completion.items() if key != "completion_sha256"}
    )


def test_modal_surface_is_cpu_only_new_only_and_maps_durable_missing_tasks() -> None:
    app_path = Path(migration_app.__file__)
    source = app_path.read_text()
    tree = ast.parse(source)
    driver = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "driver"
    )
    driver_source = ast.get_source_segment(source, driver)
    assert driver_source is not None

    assert "gpu=" not in source
    assert "MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS = 5" in source
    assert "MAX_MAP_CONTAINERS = MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS" in source
    assert migration_app.MODAL_VOLUME_V1_MAX_CONCURRENT_WRITERS == 5
    assert migration_app.MAX_MAP_CONTAINERS == 5
    assert "editing_v2_semantic_active8_source_adapter" not in source
    assert "semantic_active8_chunk_cache" not in source
    assert "_validate_structural_completion" in driver_source
    assert "completed_semantic_trace_migration_task_ids" in driver_source
    assert "migrate_one_shard.starmap" in driver_source
    assert "reduce_complete.remote(plan)" in driver_source
    assert driver_source.count("artifact_volume.commit()") >= 3

    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "migrate_one_shard"
    )
    worker_decorator = ast.get_source_segment(source, worker.decorator_list[0])
    assert worker_decorator is not None
    assert "max_containers=MAX_MAP_CONTAINERS" in worker_decorator
    worker_source = ast.get_source_segment(source, worker)
    assert worker_source is not None
    assert worker_source.index("artifact_volume.reload()") < worker_source.index(
        "execute_semantic_trace_migration_task"
    )
    assert "artifact_volume.commit()" in worker_source

    reducer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "reduce_complete"
    )
    reducer_source = ast.get_source_segment(source, reducer)
    assert reducer_source is not None
    assert reducer_source.index("artifact_volume.reload()") < reducer_source.index(
        "reduce_semantic_trace_migration"
    )
    assert "os.link(" not in source
    assert ".link_to(" not in source

    argument_names = [argument.arg for argument in driver.args.kwonlyargs]
    assert argument_names == [
        "source_revision",
        "structural_completion_path",
        "max_row_bytes",
        "output_prefix",
    ]
