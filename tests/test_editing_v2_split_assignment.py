from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from compose_v4.data.editing_corpus_contract import load_editing_corpus_contract
from compose_v4.data.editing_v2_candidate_provenance_bridge import (
    SOURCE_STREAM_SCHEMA,
    SOURCE_STREAM_SCHEMA_VERSION,
)
from compose_v4.data.editing_v2_split_assignment import (
    EditingV2SplitAssignmentError,
    build_split_assignment,
    load_split_assignment_policy,
)
from compose_v4.data.editing_v2_split_census import (
    CANDIDATE_ROW_SCHEMA,
    CANDIDATE_ROW_SCHEMA_VERSION,
    IDENTITY_DEFINITION_CONTRACT_SCHEMA,
    IDENTITY_DEFINITION_CONTRACT_SCHEMA_VERSION,
    build_split_component_census,
    canonical_sha256,
    default_split_census_policy,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "editing_corpus_v2_contract.json"
POLICY_PATH = ROOT / "configs" / "editing_v2_split_assignment_policy_v1.json"
CONTRACT = load_editing_corpus_contract(CONTRACT_PATH)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _identity_definitions() -> dict:
    return {
        "schema": IDENTITY_DEFINITION_CONTRACT_SCHEMA,
        "schema_version": IDENTITY_DEFINITION_CONTRACT_SCHEMA_VERSION,
        "exact_molecule": {
            "definition_id": "fixture-exact",
            "implementation_sha256": SHA_A,
            "identity_namespace": "fixture-molecule",
        },
        "partition_scaffold": {
            "definition_id": "fixture-scaffold",
            "implementation_sha256": SHA_B,
            "identity_namespace": "fixture-scaffold",
        },
        "source_group": {
            "definition_id": "fixture-source-group",
            "implementation_sha256": SHA_C,
            "identity_namespace": "fixture-source-group",
        },
    }


def _profile_components(profile_id: str) -> dict[str, str]:
    profile = next(
        row
        for row in CONTRACT["evidence_component_contract"]["profiles"]
        if row["id"] == profile_id
    )
    return dict(profile["components"])


def _row(index: int, lane: str, mass: int) -> dict:
    candidate_id = f"candidate-{index:03d}"
    profile = (
        "executor_generated_walk"
        if lane == "reversible_synthetic_walk"
        else "inferred_relation_compiled_path"
    )
    return {
        "schema": CANDIDATE_ROW_SCHEMA,
        "schema_version": CANDIDATE_ROW_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "candidate_ledger_row_sha256": canonical_sha256(
            {"candidate_id": candidate_id, "ledger": True}
        ),
        "candidate_envelope_sha256": canonical_sha256(
            {"candidate_id": candidate_id, "envelope": True}
        ),
        "data_lane": lane,
        "evidence_profile_id": profile,
        "evidence_components": _profile_components(profile),
        "mass_units": mass,
        "molecule_ids": [f"molecule-{index:03d}"],
        "partition_scaffold_ids": [f"scaffold-{index:03d}"],
        "source_group_ids": [f"source-group-{index:03d}"],
        "source_group_namespace": {
            "schema": "compose.editing_v2_relationship_namespace",
            "schema_version": 2,
            "relationship_type": "source_group",
            "namespace_id": "fixture-source-groups",
            "source_asset_id": "fixture-source-asset",
            "source_asset_sha256": SHA_C,
            "definition_id": "fixture-source-group-definition",
            "implementation_sha256": SHA_C,
            "cross_lane_sharing_authorized": True,
        },
        "identity_definitions": _identity_definitions(),
        "shared_prefix_branch_group_id": None,
        "alternative_route_group_id": None,
        "inverse_pair_group_id": None,
        "correction_group_id": None,
        "constraint_compatible_alternative_route_group_id": None,
        "relationship_namespaces": {
            "shared_prefix_branch": None,
            "alternative_route": None,
            "inverse_pair": None,
            "correction": None,
            "constraint_compatible_alternative_route": None,
        },
        "document_group_id": None,
        "document_provenance": None,
        "series_group_id": None,
        "series_provenance": None,
        "transformation_signature": None,
        "transformation_definition": None,
        "metadata": {"fixture": True},
    }


def _provenance() -> dict:
    source = ROOT / "src" / "compose_v4" / "data" / "editing_v2_split_census.py"
    policy = ROOT / "configs" / "editing_corpus_v2_contract.json"
    return {
        "schema": "compose.editing_v2_split_census_provenance",
        "schema_version": 1,
        "code_revision": {"commit_sha": "a" * 40, "dirty": False},
        "python_runtime": {"implementation": "CPython", "version": "3.11.0"},
        "source_files": [
            {
                "path": str(source.relative_to(ROOT)),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "bytes": source.stat().st_size,
            }
        ],
        "policy_file": {
            "path": str(policy.relative_to(ROOT)),
            "sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
            "bytes": policy.stat().st_size,
        },
        "editing_corpus_contract_file": {
            "path": str(CONTRACT_PATH.relative_to(ROOT)),
            "sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
            "bytes": CONTRACT_PATH.stat().st_size,
        },
    }


def _census() -> dict:
    lanes = [row["id"] for row in CONTRACT["data_lanes"]]
    rows = [_row(index, lanes[index % len(lanes)], 1 + (index % 7)) for index in range(160)]
    source_stream_body = {
        "schema": SOURCE_STREAM_SCHEMA,
        "schema_version": SOURCE_STREAM_SCHEMA_VERSION,
        "nonempty_jsonl_rows": len(rows),
        "candidate_materialization": {
            "manifest_file_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "rows_file_sha256": "3" * 64,
            "rows_semantic_sha256": "4" * 64,
            "address_stream_sha256": "5" * 64,
        },
        "provenance_registry": {
            "file_sha256": "6" * 64,
            "registry_sha256": "7" * 64,
        },
        "candidate_audit_ledger": {
            "file_sha256": "8" * 64,
            "semantic_sha256": "9" * 64,
            "rows_file_sha256": "0" * 64,
            "rows_sha256": "a" * 64,
        },
        "split_candidates": {
            "file_sha256": "b" * 64,
            "semantic_sha256": "c" * 64,
        },
    }
    source_stream = {
        **source_stream_body,
        "source_stream_sha256": canonical_sha256(source_stream_body),
    }
    return build_split_component_census(
        rows,
        policy=default_split_census_policy(
            CONTRACT,
            identity_definitions=_identity_definitions(),
        ),
        editing_corpus_contract=CONTRACT,
        implementation_provenance=_provenance(),
        source_stream=source_stream,
    )


def test_assignment_is_deterministic_component_complete_and_sealed() -> None:
    census = _census()
    policy = load_split_assignment_policy(POLICY_PATH)
    first = build_split_assignment(census, policy=policy)
    second = build_split_assignment(census, policy=policy)
    assert first == second
    assert first["training_authorized"] is False
    assert first["split_assignment_selected"] is True
    assert first["split_assignment_authorized"] is False
    assert first["schema_version"] == 2
    assert first["source_stream"] == census["source_stream"]
    assert "must not be inspected" in first["sealed_final_test_policy"]
    resolutions = first["candidate_resolutions"]
    assert len(resolutions) == census["input_summary"]["valid_vertices"]
    assert len({row["candidate_id"] for row in resolutions}) == len(resolutions)
    assert all(
        row["source_endpoint_role"] == row["target_endpoint_role"] == row["assigned_role"]
        for row in resolutions
    )
    assert set(first["role_summaries"]) == {
        "train",
        "validation",
        "controller_validation",
        "final_test",
    }


def test_assignment_balances_every_lane_on_fixture() -> None:
    assignment = build_split_assignment(
        _census(),
        policy=load_split_assignment_policy(POLICY_PATH),
    )
    assert all(assignment["gate_results"].values())
    for summary in assignment["role_summaries"].values():
        assert all(value > 0 for value in summary["mass_units_by_lane"].values())


def test_policy_and_census_tampering_fail_closed() -> None:
    policy = load_split_assignment_policy(POLICY_PATH)
    weakened = copy.deepcopy(policy)
    weakened["gates"]["all_roles_nonempty"] = False
    with pytest.raises(EditingV2SplitAssignmentError, match="cannot be weakened"):
        build_split_assignment(_census(), policy=weakened)

    malformed = copy.deepcopy(_census())
    malformed["census_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="census SHA-256|identity"):
        build_split_assignment(malformed, policy=policy)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda stream: stream.pop("provenance_registry"),
        lambda stream: stream["candidate_materialization"].update(
            {"rows_file_sha256": "not-a-sha256"}
        ),
        lambda stream: stream["provenance_registry"].update({"registry_sha256": "0" * 64}),
        lambda stream: stream["candidate_audit_ledger"].update({"rows_sha256": "0" * 64}),
        lambda stream: stream["split_candidates"].update({"semantic_sha256": "0" * 64}),
        lambda stream: stream.update({"source_stream_sha256": "0" * 64}),
    ],
)
def test_assignment_rejects_missing_or_mismatched_candidate_source_identity(
    mutation,
) -> None:
    census = _census()
    mutation(census["source_stream"])
    census_body = {key: value for key, value in census.items() if key != "census_sha256"}
    census["census_sha256"] = canonical_sha256(census_body)
    with pytest.raises(
        EditingV2SplitAssignmentError,
        match="candidate provenance source-stream",
    ):
        build_split_assignment(
            census,
            policy=load_split_assignment_policy(POLICY_PATH),
        )


def test_policy_json_is_stable_and_ordered() -> None:
    policy = load_split_assignment_policy(POLICY_PATH)
    assert set(policy["target_mass_ratios"]["numerators"]) == {
        "train",
        "validation",
        "controller_validation",
        "final_test",
    }
    assert tuple(policy["partition_roles"]) == (
        "train",
        "validation",
        "controller_validation",
        "final_test",
    )
    assert json.loads(json.dumps(policy, sort_keys=True)) == policy
