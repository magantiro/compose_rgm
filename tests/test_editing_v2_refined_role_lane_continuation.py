from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

from compose_v4.data import editing_v2_refined_role_lane_continuation as continuation
from compose_v4.data.editing_v2_packed_candidate_materializer import canonical_sha256

SHA = "a" * 64
COMMIT = "b" * 40


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _self_hashed(body: dict, field: str) -> dict:
    return {**body, field: canonical_sha256(body)}


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    artifact_root = tmp_path / "artifacts"
    run_root = artifact_root / "split" / SHA
    census_body = {
        "census_structural_complete": True,
        "invalid_rows": [],
        "status": "BLOCKED",
        "blockers": ["authority.missing"],
        "source_stream": {"source_stream_sha256": "1" * 64},
        "hard_component_census": {"component_inventory_sha256": "2" * 64},
    }
    census = _self_hashed(census_body, "census_sha256")
    census_path = run_root / "EDITING_V2_SPLIT_CENSUS_V4.json"
    _write(census_path, census)
    parent_assignment_body = {
        "source_census_sha256": census["census_sha256"],
        "source_component_inventory_sha256": "2" * 64,
        "source_stream": census["source_stream"],
        "gate_results": {"per_lane": False},
        "blockers": ["physical_lane_shards.not_built", "split_gate.per_lane"],
        "schema_version": 2,
        "policy": {"schema_version": 1},
        "candidate_resolution_stream_sha256": "3" * 64,
    }
    parent_assignment = _self_hashed(parent_assignment_body, "assignment_sha256")
    assignment_path = run_root / "EDITING_V2_SPLIT_ASSIGNMENT.json"
    _write(assignment_path, parent_assignment)
    request_body = {
        "schema": "compose.editing_v2_structural_split_modal_run",
        "schema_version": 1,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "bounded_p50_authorized": False,
        "active8_authorized": False,
        "final_test_selection_authorized": False,
        "source_revision": {},
        "python_runtime": {"implementation": "CPython", "version": "3.11"},
        "candidate": {"manifest_sha256": "4" * 64},
        "candidate_provenance_bridge": {"source_stream_sha256": "1" * 64},
        "candidate_provenance_registry": {},
        "split_census_policy": {},
        "split_assignment_policy": {},
        "editing_corpus_contract": {},
        "max_row_bytes": 1,
        "max_source_row_bytes": 1,
        "output_prefix": "/artifacts/split",
    }
    request = {**request_body, "run_identity_sha256": canonical_sha256(request_body)}
    census_record = {
        "artifact_path": str(census_path).replace(str(artifact_root), "/artifacts"),
        "file_sha256": continuation.file_sha256(census_path),
        "census_sha256": census["census_sha256"],
        "status": "BLOCKED",
        "census_structural_complete": True,
        "blockers": ["authority.missing"],
    }
    assignment_record = {
        "artifact_path": str(assignment_path).replace(str(artifact_root), "/artifacts"),
        "file_sha256": continuation.file_sha256(assignment_path),
        "assignment_sha256": parent_assignment["assignment_sha256"],
        "policy_schema_version": 1,
        "artifact_schema_version": 2,
        "gate_results": {"per_lane": False},
        "blockers": parent_assignment["blockers"],
    }
    parent_body = {
        "schema": continuation.PARENT_COMPLETION_SCHEMA,
        "schema_version": continuation.PARENT_COMPLETION_SCHEMA_VERSION,
        "status": continuation.PARENT_BLOCKED_STATUS,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "bounded_p50_authorized": False,
        "active8_admission_status": "NOT_RUN",
        "final_test_selection_use": "forbidden_not_performed",
        "request": request,
        "run_root": f"/artifacts/split/{request['run_identity_sha256']}",
        "census": census_record,
        "assignment": assignment_record,
        "role_lane_packed": None,
        "blockers": parent_assignment["blockers"],
    }
    # The physical parent directory is itself request-addressed.
    actual_run_root = artifact_root / "split" / request["run_identity_sha256"]
    actual_run_root.mkdir(parents=True)
    actual_census = actual_run_root / census_path.name
    actual_assignment = actual_run_root / assignment_path.name
    census_path.replace(actual_census)
    assignment_path.replace(actual_assignment)
    census_record["artifact_path"] = str(actual_census).replace(
        str(artifact_root), "/artifacts"
    )
    census_record["file_sha256"] = continuation.file_sha256(actual_census)
    assignment_record["artifact_path"] = str(actual_assignment).replace(
        str(artifact_root), "/artifacts"
    )
    assignment_record["file_sha256"] = continuation.file_sha256(actual_assignment)
    parent_body["census"] = census_record
    parent_body["assignment"] = assignment_record
    parent = _self_hashed(parent_body, "completion_sha256")
    parent_path = actual_run_root / continuation.PARENT_COMPLETION_FILENAME
    _write(parent_path, parent)

    refined_assignment = {
        **parent_assignment,
        "gate_results": {"per_lane": True},
        "blockers": ["physical_lane_shards.not_built"],
    }
    refined_assignment = _self_hashed(
        {
            key: value
            for key, value in refined_assignment.items()
            if key != "assignment_sha256"
        },
        "assignment_sha256",
    )
    refinement = _self_hashed(
        {
            "refined_assignment": refined_assignment,
            "training_authorized": False,
        },
        "refinement_sha256",
    )
    refinement_path = artifact_root / "refinement" / "REFINEMENT.json"
    _write(refinement_path, refinement)

    monkeypatch.setattr(
        continuation, "validate_split_component_census", lambda value: value
    )

    def validate_assignment(value, *, require_failed=False, require_passing=False):
        if require_failed and all(value["gate_results"].values()):
            raise AssertionError("expected failed fixture")
        if require_passing and not all(value["gate_results"].values()):
            raise AssertionError("expected passing fixture")
        return dict(value)

    monkeypatch.setattr(
        continuation, "validate_split_assignment_v2", validate_assignment
    )

    def validate_refinement(value, **kwargs):
        assert kwargs["parent_assignment"] == parent_assignment
        assert kwargs["parent_assignment_file_sha256"] == continuation.file_sha256(
            actual_assignment
        )
        return dict(value)

    monkeypatch.setattr(
        continuation, "validate_refinement_artifact", validate_refinement
    )
    parent_address = str(parent_path).replace(str(artifact_root), "/artifacts")
    refinement_address = str(refinement_path).replace(str(artifact_root), "/artifacts")
    return artifact_root, parent_address, refinement_address, refined_assignment


def test_blocked_parent_and_refinement_reopen_with_no_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root, parent_path, refinement_path, refined_assignment = _fixture(
        tmp_path, monkeypatch
    )
    inputs = continuation.load_validated_refinement_inputs(
        parent_completion_artifact_path=parent_path,
        refinement_artifact_path=refinement_path,
        artifact_root=artifact_root,
    )
    assert inputs.refined_assignment == refined_assignment
    request = continuation.build_continuation_request(
        source_revision={"commit": COMMIT, "tree": COMMIT, "worktree_clean": True},
        python_runtime={"implementation": "CPython", "version": "3.11"},
        inputs=inputs,
        artifact_root=artifact_root,
        max_source_row_bytes=1024,
    )
    assert request["training_authorized"] is False
    assert request["gate_zero_authorized"] is False
    assert request["t1_authorized"] is False
    assert request["bounded_p50_authorized"] is False
    assert request["parent_blocked_completion"]["parent_assignment_sha256"] == (
        inputs.parent_assignment["assignment_sha256"]
    )
    assert request["split_assignment_refinement"]["refined_assignment_sha256"] == (
        refined_assignment["assignment_sha256"]
    )
    run_root = (
        artifact_root
        / "editing_v2"
        / "refined_role_lane"
        / request["run_identity_sha256"]
    )
    assignment_path = run_root / continuation.REFINED_ASSIGNMENT_FILENAME
    _write(assignment_path, refined_assignment)
    role_record = {
        "artifact_root": f"/artifacts/role/{SHA}",
        "manifest_file_sha256": "5" * 64,
        "manifest_sha256": "6" * 64,
        "lane_registry_file_sha256": "7" * 64,
        "lane_registry_sha256": "8" * 64,
        "membership_file_sha256": "9" * 64,
        "membership_receipt_sha256": "c" * 64,
        "output_shards": 20,
        "active8_admission_status": "NOT_RUN",
    }
    completion = continuation.build_continuation_completion(
        request=request,
        inputs=inputs,
        artifact_root=artifact_root,
        run_root=f"{continuation.OUTPUT_PREFIX}/{request['run_identity_sha256']}",
        assignment_record=continuation.refined_assignment_record(
            assignment_path,
            refined_assignment,
            artifact_root=artifact_root,
        ),
        role_lane_record=role_record,
    )
    completion_path = run_root / continuation.CONTINUATION_COMPLETION_FILENAME
    _write(completion_path, completion)
    completion_address = str(completion_path).replace(str(artifact_root), "/artifacts")
    observed, _, observed_assignment_path, observed_assignment = (
        continuation.validate_continuation_completion_header(
            completion_address,
            artifact_root=artifact_root,
        )
    )
    assert observed == completion
    assert observed_assignment_path == assignment_path
    assert observed_assignment == refined_assignment


def test_parent_status_tampering_fails_even_after_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root, parent_address, refinement_address, _ = _fixture(
        tmp_path, monkeypatch
    )
    parent_path = artifact_root / Path(
        *PurePosixPath(parent_address).relative_to("/artifacts").parts
    )
    parent = json.loads(parent_path.read_text())
    parent["status"] = "COMPLETE_PRE_ACTIVE8_NO_TRAINING_AUTHORITY"
    parent["completion_sha256"] = canonical_sha256(
        {key: value for key, value in parent.items() if key != "completion_sha256"}
    )
    _write(parent_path, parent)
    with pytest.raises(
        continuation.EditingV2RefinedRoleLaneContinuationError,
        match="identity, status, or authority",
    ):
        continuation.load_validated_refinement_inputs(
            parent_completion_artifact_path=parent_address,
            refinement_artifact_path=refinement_address,
            artifact_root=artifact_root,
        )
