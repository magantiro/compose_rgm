"""Contract tests for the Editing V2 split-refinement Modal bridge."""

from __future__ import annotations

import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from modal_apps import refine_editing_v2_split_assignment_app as refine_app

SHA = "a" * 64
PARENT_ROOT = "/artifacts/editing_v2/split_pipeline/" + "b" * 64
PARENT_COMPLETION_PATH = f"{PARENT_ROOT}/SPLIT_PIPELINE_COMPLETE.json"
PARENT_ASSIGNMENT_PATH = f"{PARENT_ROOT}/EDITING_V2_SPLIT_ASSIGNMENT.json"


def _source_revision() -> dict[str, object]:
    hashes = {"source.py": SHA}
    body: dict[str, object] = {
        "schema": refine_app.SOURCE_REVISION_SCHEMA,
        "schema_version": refine_app.SOURCE_REVISION_SCHEMA_VERSION,
        "commit": "1" * 40,
        "tree": "2" * 40,
        "worktree_clean": True,
        "serialized_source_hashes": hashes,
        "serialized_source_hashes_sha256": refine_app._canonical_sha256(hashes),
    }
    return {**body, "source_revision_sha256": refine_app._canonical_sha256(body)}


def _request() -> dict[str, object]:
    return refine_app.build_run_request(
        source_revision=_source_revision(),
        python_runtime={"implementation": "CPython", "version": "3.11.9"},
        parent_completion_path=PARENT_COMPLETION_PATH,
        expected_parent_completion_file_sha256="c" * 64,
        expected_parent_completion_sha256="d" * 64,
        parent_assignment_path=PARENT_ASSIGNMENT_PATH,
        expected_parent_assignment_file_sha256="e" * 64,
        expected_parent_assignment_sha256="f" * 64,
    )


def _parent_completion() -> dict[str, object]:
    request_body = {
        "schema": "parent-request",
        "schema_version": 1,
        "training_authorized": False,
    }
    request = {
        **request_body,
        "run_identity_sha256": refine_app._canonical_sha256(request_body),
    }
    body: dict[str, object] = {
        "schema": refine_app.PARENT_COMPLETION_SCHEMA,
        "schema_version": refine_app.PARENT_COMPLETION_SCHEMA_VERSION,
        "status": refine_app.PARENT_BLOCKED_STATUS,
        "training_authorized": False,
        "gate_zero_authorized": False,
        "bounded_p50_authorized": False,
        "active8_admission_status": "NOT_RUN",
        "final_test_selection_use": "forbidden_not_performed",
        "request": request,
        "run_root": PARENT_ROOT,
        "census": {"file_sha256": SHA, "census_sha256": "1" * 64},
        "assignment": {
            "artifact_path": f"/__modal/volumes/exact/{PARENT_ASSIGNMENT_PATH[11:]}",
            "artifact_schema_version": 2,
            "file_sha256": "e" * 64,
            "assignment_sha256": "f" * 64,
            "blockers": [
                "physical_lane_shards.not_built",
                "split_gate.per_lane_mass_ratio_within_limit",
            ],
            "gate_results": {
                "all_observed_lanes_present_in_every_role": True,
                "all_roles_nonempty": True,
                "overall_mass_ratio_within_limit": True,
                "per_lane_mass_ratio_within_limit": False,
            },
            "policy_schema_version": 1,
        },
        "role_lane_packed": None,
        "blockers": [
            "role_lane_packed.not_built",
            "split_gate.per_lane_mass_ratio_within_limit",
        ],
    }
    return {**body, "completion_sha256": refine_app._canonical_sha256(body)}


def _record(path: str, semantic_field: str, semantic_sha256: str) -> dict[str, object]:
    return {
        "artifact_path": path,
        "file_sha256": SHA,
        semantic_field: semantic_sha256,
    }


def _completion(request: dict[str, object]) -> dict[str, object]:
    run_root = f"{request['output_prefix']}/{request['run_identity_sha256']}"
    return refine_app.build_completion(
        request=request,
        run_root=run_root,
        parent_completion=request["parent_completion"],
        parent_assignment=request["parent_assignment"],
        refinement=_record(
            f"{run_root}/{refine_app.REFINEMENT_FILENAME}",
            "refinement_sha256",
            "3" * 64,
        ),
        refined_assignment={
            **_record(
                f"{run_root}/{refine_app.REFINED_ASSIGNMENT_FILENAME}",
                "assignment_sha256",
                "4" * 64,
            ),
            "candidate_resolution_stream_sha256": "5" * 64,
            "all_frozen_split_gates_pass": True,
        },
    )


def test_request_is_deterministic_content_addressed_and_nonauthorizing() -> None:
    first = _request()
    second = _request()

    assert first == second
    assert refine_app.validate_run_request(first) == first
    assert first["schema"] == refine_app.RUN_SCHEMA
    assert first["parent_completion"]["file_sha256"] == "c" * 64
    assert first["parent_assignment"]["assignment_sha256"] == "f" * 64
    assert all(first[field] is False for field in refine_app._NO_AUTHORITY)

    upgraded = copy.deepcopy(first)
    upgraded["training_authorized"] = True
    upgraded_body = {
        key: value for key, value in upgraded.items() if key != "run_identity_sha256"
    }
    upgraded["run_identity_sha256"] = refine_app._canonical_sha256(upgraded_body)
    with pytest.raises(ValueError, match="authority"):
        refine_app.validate_run_request(upgraded)


@pytest.mark.parametrize(
    "field,value",
    [
        ("parent_completion_path", "/tmp/parent.json"),
        ("parent_assignment_path", "/artifacts/../tmp/assignment.json"),
        ("expected_parent_completion_file_sha256", "bad"),
        ("output_prefix", "/tmp/output"),
    ],
)
def test_request_rejects_escaping_or_unpinned_inputs(field: str, value: str) -> None:
    arguments = {
        "source_revision": _source_revision(),
        "python_runtime": {"implementation": "CPython", "version": "3.11.9"},
        "parent_completion_path": PARENT_COMPLETION_PATH,
        "expected_parent_completion_file_sha256": "c" * 64,
        "expected_parent_completion_sha256": "d" * 64,
        "parent_assignment_path": PARENT_ASSIGNMENT_PATH,
        "expected_parent_assignment_file_sha256": "e" * 64,
        "expected_parent_assignment_sha256": "f" * 64,
        "output_prefix": refine_app.OUTPUT_PREFIX,
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        refine_app.build_run_request(**arguments)


def test_parent_completion_requires_exact_blocked_receipt_and_assignment_pointer() -> (
    None
):
    completion = _parent_completion()
    validated = refine_app.validate_parent_completion(
        completion,
        expected_completion_sha256=completion["completion_sha256"],
        parent_completion_path=PARENT_COMPLETION_PATH,
        parent_assignment_path=PARENT_ASSIGNMENT_PATH,
        expected_parent_assignment_file_sha256="e" * 64,
        expected_parent_assignment_sha256="f" * 64,
    )
    assert validated == completion

    upgraded = copy.deepcopy(completion)
    upgraded["status"] = "COMPLETE_PRE_ACTIVE8_NO_TRAINING_AUTHORITY"
    body = {key: value for key, value in upgraded.items() if key != "completion_sha256"}
    upgraded["completion_sha256"] = refine_app._canonical_sha256(body)
    with pytest.raises(RuntimeError, match="identity"):
        refine_app.validate_parent_completion(
            upgraded,
            expected_completion_sha256=upgraded["completion_sha256"],
            parent_completion_path=PARENT_COMPLETION_PATH,
            parent_assignment_path=PARENT_ASSIGNMENT_PATH,
            expected_parent_assignment_file_sha256="e" * 64,
            expected_parent_assignment_sha256="f" * 64,
        )


def test_completion_is_self_hashed_replay_bound_and_strictly_nonauthorizing() -> None:
    request = _request()
    completion = _completion(request)
    assert (
        refine_app.validate_completion(completion, expected_request=request)
        == completion
    )
    assert completion["role_lane_packed"] is None
    assert all(completion[field] is False for field in refine_app._NO_AUTHORITY)

    tampered = copy.deepcopy(completion)
    tampered["refined_assignment"]["all_frozen_split_gates_pass"] = False
    body = {key: value for key, value in tampered.items() if key != "completion_sha256"}
    tampered["completion_sha256"] = refine_app._canonical_sha256(body)
    with pytest.raises(RuntimeError, match="identity"):
        refine_app.validate_completion(tampered, expected_request=request)


def test_local_source_revision_requires_the_exact_clean_commit() -> None:
    with (
        patch.object(refine_app, "_git", side_effect=["1" * 40, "2" * 40, ""]),
        patch.object(
            refine_app, "_serialized_source_hashes", return_value={"source.py": SHA}
        ),
    ):
        revision = refine_app.local_source_revision(
            expected_commit="1" * 40,
            repo_root=Path("/fixture"),
        )
    assert revision["commit"] == "1" * 40
    assert revision["worktree_clean"] is True

    with (
        patch.object(
            refine_app,
            "_git",
            side_effect=["1" * 40, "2" * 40, "?? untracked.py"],
        ),
        pytest.raises(RuntimeError, match="clean committed"),
    ):
        refine_app.local_source_revision(
            expected_commit="1" * 40,
            repo_root=Path("/fixture"),
        )


def test_immutable_publish_reuses_exact_bytes_and_rejects_collision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    value = {"schema": "fixture", "training_authorized": False}

    assert refine_app._write_immutable_json(path, value) is True
    assert refine_app._write_immutable_json(path, value) is False

    path.write_bytes(b"{}\n")
    with pytest.raises(RuntimeError, match="collision"):
        refine_app._write_immutable_json(path, value)
