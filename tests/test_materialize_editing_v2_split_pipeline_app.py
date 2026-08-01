"""Contract tests for the new-only Editing V2 structural split Modal job."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modal_apps import materialize_editing_v2_split_pipeline_app as split_app

SHA = "a" * 64


def _source_revision() -> dict[str, object]:
    return {
        "schema": split_app.SOURCE_REVISION_SCHEMA,
        "schema_version": split_app.SOURCE_REVISION_SCHEMA_VERSION,
        "commit": "1" * 40,
        "tree": "2" * 40,
        "worktree_clean": True,
        "serialized_source_hashes": {"source.py": SHA},
        "serialized_source_hashes_sha256": SHA,
        "source_revision_sha256": SHA,
    }


def _request(**updates: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "source_revision": _source_revision(),
        "python_runtime": {"implementation": "CPython", "version": "3.11.9"},
        "candidate_root": "/artifacts/editing_v2/candidates/exact",
        "candidate_identity": {
            "manifest_file_sha256": SHA,
            "manifest_sha256": SHA,
            "rows_file_sha256": SHA,
            "rows_semantic_sha256": SHA,
            "address_stream_sha256": SHA,
        },
        "bridge_root": "/artifacts/editing_v2/provenance/exact/bridge",
        "bridge_identity": {
            "manifest_file_sha256": SHA,
            "manifest_sha256": SHA,
            "source_stream_sha256": SHA,
            "outputs": {
                "candidate_audit_ledger": {
                    "relative_path": "candidate_audit_ledger.json",
                    "file_sha256": SHA,
                    "semantic_sha256": SHA,
                },
                "candidate_audit_rows": {
                    "relative_path": "candidate_audit_rows.jsonl",
                    "file_sha256": SHA,
                    "semantic_sha256": SHA,
                },
                "split_candidates": {
                    "relative_path": "split_candidates.jsonl",
                    "file_sha256": SHA,
                    "semantic_sha256": SHA,
                },
            },
        },
        "registry_path": "/artifacts/editing_v2/provenance/exact/registry.json",
        "registry_identity": {"file_sha256": SHA, "registry_sha256": SHA},
        "census_policy_path": split_app.DEFAULT_CENSUS_POLICY_PATH,
        "census_policy_identity": {
            "file_sha256": SHA,
            "semantic_sha256": SHA,
            "policy_id": "census-v4",
            "schema": "census-policy",
            "schema_version": 4,
        },
        "assignment_policy_path": split_app.DEFAULT_ASSIGNMENT_POLICY_PATH,
        "assignment_policy_identity": {
            "file_sha256": SHA,
            "semantic_sha256": SHA,
            "policy_id": "assignment-v1",
            "schema": "assignment-policy",
            "schema_version": 1,
        },
        "corpus_contract_path": split_app.DEFAULT_CORPUS_CONTRACT_PATH,
        "corpus_contract_identity": {
            "file_sha256": SHA,
            "semantic_sha256": SHA,
            "contract_id": "editing-v2",
            "schema": "editing-contract",
            "schema_version": 2,
        },
        "max_row_bytes": 1024,
        "max_source_row_bytes": 2048,
        "output_prefix": split_app.OUTPUT_PREFIX,
    }
    arguments.update(updates)
    return split_app.build_run_request(**arguments)


def _census(*, structural_complete: bool) -> dict[str, object]:
    blockers = list(_FakeSplitCensus.REQUIRED_AUTHORITY_BLOCKERS)
    if not structural_complete:
        blockers.insert(0, "stream.invalid_rows")
    return {
        "status": "BLOCKED",
        "status_scope": "CENSUS_STRUCTURAL_COMPLETION_ONLY",
        "census_computation_complete": True,
        "census_structural_complete": structural_complete,
        "corpus_ready": False,
        "split_ready": False,
        "training_ready": False,
        "authority": {
            "corpus": "NO_CORPUS_READINESS_AUTHORITY",
            "split": "NO_SPLIT_AUTHORITY",
            "training": "NO_TRAINING_AUTHORITY",
        },
        "blockers": blockers,
        "invalid_rows": [] if structural_complete else [{"reason": "fixture"}],
    }


class _FakeSplitCensus:
    REQUIRED_AUTHORITY_BLOCKERS = (
        "external_authority.observed_group_membership:UNRESOLVED",
        "split_assignment.not_performed_by_component_census",
    )

    @staticmethod
    def validate_split_component_census(census: object) -> None:
        assert isinstance(census, dict)


def test_run_request_is_deterministic_content_addressed_and_nonauthorizing() -> None:
    first = _request()
    second = _request()

    assert first == second
    assert first["schema"] == split_app.RUN_SCHEMA
    assert first["training_authorized"] is False
    assert first["gate_zero_authorized"] is False
    assert first["bounded_p50_authorized"] is False
    assert first["active8_authorized"] is False
    assert first["final_test_selection_authorized"] is False
    assert len(first["run_identity_sha256"]) == 64
    assert first["split_census_policy"]["schema_version"] == 4
    assert first["split_assignment_policy"]["schema_version"] == 1
    assert first["candidate"]["rows_file_sha256"] == SHA
    assert first["candidate_provenance_bridge"]["source_stream_sha256"] == SHA
    assert set(first["candidate_provenance_bridge"]["outputs"]) == {
        "candidate_audit_ledger",
        "candidate_audit_rows",
        "split_candidates",
    }


def test_local_source_revision_requires_exact_clean_commit_and_binds_tree() -> None:
    with (
        patch.object(
            split_app,
            "_git",
            side_effect=["1" * 40, "2" * 40, ""],
        ),
        patch.object(split_app, "_serialized_source_hashes", return_value={"source.py": SHA}),
    ):
        revision = split_app.local_source_revision(
            expected_commit="1" * 40,
            repo_root=Path("/fixture"),
        )
    assert revision["commit"] == "1" * 40
    assert revision["tree"] == "2" * 40
    assert revision["worktree_clean"] is True
    assert len(revision["source_revision_sha256"]) == 64

    with (
        patch.object(
            split_app,
            "_git",
            side_effect=["1" * 40, "2" * 40, "?? untracked.py"],
        ),
        pytest.raises(RuntimeError, match="clean committed"),
    ):
        split_app.local_source_revision(
            expected_commit="1" * 40,
            repo_root=Path("/fixture"),
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"candidate_root": "/tmp/candidates"},
        {"bridge_root": "/artifacts/../tmp/bridge"},
        {"census_policy_path": "/tmp/census.json"},
        {"max_row_bytes": 0},
        {"candidate_identity": {"manifest_sha256": "bad"}},
        {
            "bridge_identity": {
                "manifest_file_sha256": SHA,
                "manifest_sha256": SHA,
                "source_stream_sha256": "bad",
                "outputs": {},
            }
        },
        {"source_revision": {**_source_revision(), "worktree_clean": False}},
    ],
)
def test_run_request_rejects_escaping_incomplete_or_dirty_inputs(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _request(**updates)


def test_structural_census_stays_blocked_but_allows_assignment_only_when_complete() -> None:
    complete = _census(structural_complete=True)
    incomplete = _census(structural_complete=False)

    assert split_app._census_allows_assignment(complete, _FakeSplitCensus) is True
    assert split_app._census_allows_assignment(incomplete, _FakeSplitCensus) is False
    assert complete["status"] == "BLOCKED"
    assert complete["training_ready"] is False
    assert all(
        blocker in complete["blockers"] for blocker in _FakeSplitCensus.REQUIRED_AUTHORITY_BLOCKERS
    )

    upgraded = copy.deepcopy(complete)
    upgraded["status"] = "PASS"
    with pytest.raises(RuntimeError, match="authority boundary"):
        split_app._census_allows_assignment(upgraded, _FakeSplitCensus)


def test_assignment_requires_every_gate_and_never_training_authority() -> None:
    passing = {
        "training_authorized": False,
        "gate_results": {
            "all_roles_nonempty": True,
            "all_observed_lanes_present_in_every_role": True,
            "overall_mass_ratio_within_limit": True,
            "per_lane_mass_ratio_within_limit": True,
        },
    }
    assert split_app._assignment_allows_packing(passing) is True

    failing = copy.deepcopy(passing)
    failing["gate_results"]["per_lane_mass_ratio_within_limit"] = False
    assert split_app._assignment_allows_packing(failing) is False

    unauthorized = copy.deepcopy(passing)
    unauthorized["training_authorized"] = True
    with pytest.raises(RuntimeError, match="training authority"):
        split_app._assignment_allows_packing(unauthorized)


def test_role_lane_stage_is_restart_safe_and_calls_production_materializer_once(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    prefix = "/artifacts/test/run/role_lane"
    published_root = artifact_root / "test/run/role_lane" / ("b" * 64)
    calls: list[dict[str, object]] = []

    def materialize(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        published_root.mkdir(parents=True)
        return {"run_artifact_root": f"{prefix}/{'b' * 64}"}

    role_lane = SimpleNamespace(
        materialize_editing_v2_role_lane_packed=materialize,
    )
    validated_manifest = {"run_artifact_root": f"{prefix}/{'b' * 64}"}
    validated_record = {"manifest_sha256": SHA}
    arguments = {
        "output_prefix": prefix,
        "candidate_root": artifact_root / "candidate",
        "bridge_root": artifact_root / "bridge",
        "registry_path": artifact_root / "registry.json",
        "assignment_path": artifact_root / "assignment.json",
        "contract_path": tmp_path / "contract.json",
        "artifact_root": artifact_root,
        "code_revision": "1" * 40,
        "max_source_row_bytes": 2048,
        "candidate_identity": {"manifest_sha256": SHA},
        "source_stream": {"source_stream_sha256": SHA},
        "loaded": {"role_lane": role_lane},
    }
    with patch.object(
        split_app,
        "_validate_role_lane_output",
        return_value=(validated_manifest, validated_record),
    ):
        _, _, first_reused = split_app._materialize_or_reuse_role_lane(**arguments)
        _, _, second_reused = split_app._materialize_or_reuse_role_lane(**arguments)

    assert first_reused is False
    assert second_reused is True
    assert len(calls) == 1
    assert calls[0]["max_source_row_bytes"] == 2048
    assert calls[0]["output_artifact_prefix"] == prefix


def test_each_json_stage_reuses_exact_bytes_and_rejects_collisions(tmp_path: Path) -> None:
    stage = tmp_path / "stage.json"

    assert split_app._write_immutable_json(stage, {"value": 1}) is True
    assert split_app._write_immutable_json(stage, {"value": 1}) is False
    with pytest.raises(RuntimeError, match="immutable split-pipeline collision"):
        split_app._write_immutable_json(stage, {"value": 2})


def test_role_lane_reopen_binds_split_contract_source_and_physical_receipt(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    assignment_path = artifact_root / "assignment.json"
    contract_path = tmp_path / "contract.json"
    role_source = tmp_path / "role_lane.py"
    source_stream = {"source_stream_sha256": SHA}
    assignment = {
        "assignment_sha256": "b" * 64,
        "candidate_resolution_stream_sha256": "c" * 64,
        "source_stream": source_stream,
    }
    assignment_path.write_text(json.dumps(assignment))
    contract = {"contract_id": "editing-v2"}
    contract_path.write_text(json.dumps(contract))
    role_source.write_text("# exact fixture source\n")
    expected_split = {
        "file_sha256": split_app._file_sha256(assignment_path),
        "assignment_sha256": "b" * 64,
        "candidate_resolution_stream_sha256": "c" * 64,
        "source_stream_sha256": SHA,
    }
    expected_contract = {
        "contract_id": "editing-v2",
        "file_sha256": split_app._file_sha256(contract_path),
    }
    candidate_identity = {"manifest_sha256": "d" * 64}
    output_prefix = "/artifacts/run/role"
    role_run_identity_body = {
        "schema": "role-schema",
        "schema_version": 2,
        "code_revision": "1" * 40,
        "candidate_materialization": candidate_identity,
        "candidate_provenance_source_stream": source_stream,
        "split_assignment": expected_split,
        "editing_corpus_contract": expected_contract,
        "materializer_implementation_sha256": split_app._file_sha256(role_source),
        "output_artifact_prefix": output_prefix,
        "lane_order": ["lane"],
        "role_order": ["train", "validation", "controller_validation", "final_test"],
        "envelope_rewrite_contract": "fixture-envelope",
        "sampler_contract": {"fixture": True},
    }
    role_run_identity = split_app._canonical_sha256(role_run_identity_body)
    output_root = artifact_root / "run/role" / role_run_identity
    output_root.mkdir(parents=True)
    lane_registry_path = output_root / "LANE.json"
    membership_path = output_root / "MEMBERSHIP.json"
    manifest_path = output_root / "ROLE.json"
    lane_registry_path.write_text("{}")
    membership_path.write_text("{}")
    output_address = split_app._artifact_address(output_root, artifact_root=artifact_root)
    body = {
        **role_run_identity_body,
        "status": "role-complete",
        "training_authorized": False,
        "active8_admission_status": "NOT_RUN",
        "run_identity_sha256": role_run_identity,
        "run_artifact_root": output_address,
        "lane_registry": {"registry_sha256": "e" * 64},
        "resolved_packed_membership": {"receipt_sha256": "f" * 64},
        "totals": {"output_shards": 20},
    }
    manifest = {**body, "manifest_sha256": split_app._canonical_sha256(body)}
    manifest_path.write_text(json.dumps(manifest))
    resolve_calls: list[dict[str, object]] = []

    def resolve(**kwargs: object) -> object:
        resolve_calls.append(kwargs)
        return SimpleNamespace(candidate_provenance_source_stream=source_stream)

    loaded = {
        "role_lane": SimpleNamespace(
            __file__=str(role_source),
            ROLE_LANE_MATERIALIZATION_FILENAME=manifest_path.name,
            ROLE_LANE_MATERIALIZATION_SCHEMA="role-schema",
            ROLE_LANE_MATERIALIZATION_SCHEMA_VERSION=2,
            ROLE_LANE_MATERIALIZATION_STATUS="role-complete",
            LANE_REGISTRY_FILENAME=lane_registry_path.name,
            RESOLVED_MEMBERSHIP_FILENAME=membership_path.name,
        ),
        "active8_adapter": SimpleNamespace(resolve_editing_v2_active8_sources=resolve),
        "lane_registry": SimpleNamespace(
            editing_corpus_contract_identity=lambda value, *, contract_file_sha256: {
                "contract_id": value["contract_id"],
                "file_sha256": contract_file_sha256,
            }
        ),
        "load_editing_corpus_contract": lambda path: contract,
    }
    _, record = split_app._validate_role_lane_output(
        output_root,
        candidate_root=artifact_root / "candidate",
        split_assignment_path=assignment_path,
        contract_path=contract_path,
        artifact_root=artifact_root,
        expected_output_prefix=output_prefix,
        expected_code_revision="1" * 40,
        expected_candidate_identity=candidate_identity,
        expected_source_stream=source_stream,
        loaded=loaded,
    )

    assert record["output_shards"] == 20
    assert record["membership_receipt_sha256"] == "f" * 64
    assert resolve_calls[0]["expected_receipt_sha256"] == "f" * 64

    tampered_body = {**body, "split_assignment": {**expected_split, "file_sha256": SHA}}
    tampered = {
        **tampered_body,
        "manifest_sha256": split_app._canonical_sha256(tampered_body),
    }
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="identity or authority"):
        split_app._validate_role_lane_output(
            output_root,
            candidate_root=artifact_root / "candidate",
            split_assignment_path=assignment_path,
            contract_path=contract_path,
            artifact_root=artifact_root,
            expected_output_prefix=output_prefix,
            expected_code_revision="1" * 40,
            expected_candidate_identity=candidate_identity,
            expected_source_stream=source_stream,
            loaded=loaded,
        )


def test_modal_surface_is_cpu_only_and_calls_the_production_role_lane_materializer() -> None:
    source = Path(split_app.__file__).read_text()

    assert "gpu=" not in source
    assert "cpu=8.0" in source
    assert "memory=65536" in source
    assert "materialize_editing_v2_role_lane_packed" in source
    assert "validate_candidate_provenance_bridge" in source
    assert '"training_launched": False' in source
