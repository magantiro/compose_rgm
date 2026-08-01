from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import build_editing_v2_split_census as split_census_cli
from compose_v4.data.editing_corpus_contract import load_editing_corpus_contract
from compose_v4.data.editing_v2_split_census import (
    CANDIDATE_ROW_SCHEMA,
    CANDIDATE_ROW_SCHEMA_VERSION,
    EditingV2SplitCensusError,
    IDENTITY_DEFINITION_CONTRACT_SCHEMA,
    IDENTITY_DEFINITION_CONTRACT_SCHEMA_VERSION,
    IMPLEMENTATION_PROVENANCE_SCHEMA,
    IMPLEMENTATION_PROVENANCE_SCHEMA_VERSION,
    OBSERVED_GROUP_PROVENANCE_SCHEMA,
    OBSERVED_GROUP_PROVENANCE_SCHEMA_VERSION,
    RELATIONSHIP_NAMESPACE_SCHEMA,
    RELATIONSHIP_NAMESPACE_SCHEMA_VERSION,
    TRANSFORMATION_DEFINITION_SCHEMA,
    TRANSFORMATION_DEFINITION_SCHEMA_VERSION,
    build_split_component_census,
    canonical_sha256,
    default_split_census_policy,
    validate_split_component_census,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "editing_corpus_v2_contract.json"
EDITING_CORPUS_CONTRACT = load_editing_corpus_contract(CONTRACT_PATH)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _identity_definitions() -> dict:
    return {
        "schema": IDENTITY_DEFINITION_CONTRACT_SCHEMA,
        "schema_version": IDENTITY_DEFINITION_CONTRACT_SCHEMA_VERSION,
        "exact_molecule": {
            "definition_id": "fixture-exact-molecule-v1",
            "implementation_sha256": SHA_A,
            "identity_namespace": "fixture-canonical-molecule",
        },
        "partition_scaffold": {
            "definition_id": "fixture-partition-scaffold-v1",
            "implementation_sha256": SHA_B,
            "identity_namespace": "fixture-partition-scaffold",
        },
        "source_group": {
            "definition_id": "fixture-source-group-v1",
            "implementation_sha256": SHA_C,
            "identity_namespace": "fixture-source-group",
        },
    }


def _relationship_namespace(
    relationship_type: str,
    *,
    source_asset_id: str = "fixture-source-a",
    source_asset_sha256: str = SHA_C,
    namespace_id: str = "fixture-local-groups",
    cross_lane: bool = False,
) -> dict:
    return {
        "schema": RELATIONSHIP_NAMESPACE_SCHEMA,
        "schema_version": RELATIONSHIP_NAMESPACE_SCHEMA_VERSION,
        "relationship_type": relationship_type,
        "namespace_id": namespace_id,
        "source_asset_id": source_asset_id,
        "source_asset_sha256": source_asset_sha256,
        "definition_id": f"fixture-{relationship_type}-definition-v1",
        "implementation_sha256": SHA_D,
        "cross_lane_sharing_authorized": cross_lane,
    }


def _implementation_provenance() -> dict:
    return {
        "schema": IMPLEMENTATION_PROVENANCE_SCHEMA,
        "schema_version": IMPLEMENTATION_PROVENANCE_SCHEMA_VERSION,
        "code_revision": {
            "commit_sha": "1" * 40,
            "dirty": False,
        },
        "python_runtime": {
            "implementation": "CPython",
            "version": "3.11.0",
        },
        "source_files": [
            {
                "path": "src/compose_v4/data/editing_v2_split_census.py",
                "sha256": SHA_A,
                "bytes": 1,
            }
        ],
        "policy_file": {
            "path": "fixture-policy.json",
            "sha256": SHA_B,
            "bytes": 2,
        },
        "editing_corpus_contract_file": {
            "path": "configs/editing_corpus_v2_contract.json",
            "sha256": SHA_C,
            "bytes": 3,
        },
    }


def _policy() -> dict:
    return default_split_census_policy(
        EDITING_CORPUS_CONTRACT,
        identity_definitions=_identity_definitions(),
    )


def _candidate_source_stream(input_path: Path, *, file_sha256: str | None = None) -> dict:
    stream_body = {
        "schema": "compose.editing_v2_split_candidate_source_stream",
        "schema_version": 2,
        "nonempty_jsonl_rows": len(_base_rows()),
        "candidate_materialization": {
            "manifest_file_sha256": SHA_A,
            "manifest_sha256": SHA_B,
            "rows_file_sha256": SHA_C,
            "rows_semantic_sha256": SHA_D,
            "address_stream_sha256": SHA_A,
        },
        "provenance_registry": {
            "file_sha256": SHA_B,
            "registry_sha256": SHA_C,
        },
        "candidate_audit_ledger": {
            "file_sha256": SHA_C,
            "semantic_sha256": SHA_D,
            "rows_file_sha256": "e" * 64,
            "rows_sha256": SHA_A,
        },
        "split_candidates": {
            "file_sha256": file_sha256 or split_census_cli._file_sha256(input_path),
            "semantic_sha256": SHA_B,
        },
    }
    return {
        **stream_body,
        "source_stream_sha256": canonical_sha256(stream_body),
    }


def _build(
    rows: Iterable[dict],
    *,
    policy: dict | None = None,
    editing_corpus_contract: dict | None = None,
    implementation_provenance: dict | None = None,
) -> dict:
    return build_split_component_census(
        rows,
        policy=deepcopy(policy) if policy is not None else _policy(),
        editing_corpus_contract=(
            deepcopy(editing_corpus_contract)
            if editing_corpus_contract is not None
            else deepcopy(EDITING_CORPUS_CONTRACT)
        ),
        implementation_provenance=(
            deepcopy(implementation_provenance)
            if implementation_provenance is not None
            else _implementation_provenance()
        ),
    )


def _observed_provenance(
    *,
    namespace: str = "fixture-series",
    row_id: str = "row-1",
) -> dict:
    return {
        "schema": OBSERVED_GROUP_PROVENANCE_SCHEMA,
        "schema_version": OBSERVED_GROUP_PROVENANCE_SCHEMA_VERSION,
        "provenance_kind": "observed_source_group",
        "group_namespace": namespace,
        "source_asset_id": "fixture-asset",
        "source_asset_sha256": SHA_A,
        "source_version": "fixture-v1",
        "access_basis": "test fixture",
        "source_row_ids": [row_id],
    }


def _transformation_definition() -> dict:
    return {
        "schema": TRANSFORMATION_DEFINITION_SCHEMA,
        "schema_version": TRANSFORMATION_DEFINITION_SCHEMA_VERSION,
        "definition_id": "fixture-mcs-variable-fragments-v1",
        "implementation_sha256": SHA_B,
        "derivation_kind": "structurally_inferred_definition",
    }


def _profile_components(profile_id: str) -> dict[str, str]:
    profile = next(
        profile
        for profile in EDITING_CORPUS_CONTRACT["evidence_component_contract"]["profiles"]
        if profile["id"] == profile_id
    )
    return dict(profile["components"])


def _row(
    candidate_id: str,
    *,
    lane: str = "observed_local_analogue",
    evidence_profile_id: str | None = None,
    evidence_components: dict[str, str] | None = None,
    mass: int = 1,
    molecules: tuple[str, ...] | None = None,
    scaffolds: tuple[str, ...] | None = None,
    identity_definitions: dict | None = None,
    source_group: str | None = None,
    source_group_source_asset_id: str = "fixture-source-group-asset",
    source_group_source_asset_sha256: str = SHA_D,
    source_group_namespace_id: str = "fixture-source-groups",
    source_group_cross_lane: bool = True,
    shared_prefix: str | None = None,
    alternative_route: str | None = None,
    inverse_pair: str | None = None,
    correction: str | None = None,
    constrained_alternative_route: str | None = None,
    relationship_source_asset_id: str = "fixture-source-a",
    relationship_source_asset_sha256: str = SHA_C,
    relationship_namespace_id: str = "fixture-local-groups",
    cross_lane_relationships: bool = False,
    relationship_namespace_overrides: dict | None = None,
    document: str | None = None,
    document_provenance: dict | None = None,
    series: str | None = None,
    series_provenance: dict | None = None,
    transformation: str | None = None,
) -> dict:
    groups = {
        "shared_prefix_branch": shared_prefix,
        "alternative_route": alternative_route,
        "inverse_pair": inverse_pair,
        "correction": correction,
        "constraint_compatible_alternative_route": (constrained_alternative_route),
    }
    selected_profile = evidence_profile_id or (
        "executor_generated_walk"
        if lane == "reversible_synthetic_walk"
        else "observed_pair_compiled_path"
    )
    relationship_namespaces = (
        deepcopy(relationship_namespace_overrides)
        if relationship_namespace_overrides is not None
        else {
            relationship_type: (
                _relationship_namespace(
                    relationship_type,
                    source_asset_id=relationship_source_asset_id,
                    source_asset_sha256=relationship_source_asset_sha256,
                    namespace_id=relationship_namespace_id,
                    cross_lane=cross_lane_relationships,
                )
                if group_id is not None
                else None
            )
            for relationship_type, group_id in groups.items()
        }
    )
    return {
        "schema": CANDIDATE_ROW_SCHEMA,
        "schema_version": CANDIDATE_ROW_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "candidate_ledger_row_sha256": canonical_sha256(
            {"candidate_id": candidate_id, "binding": "ledger-row"}
        ),
        "candidate_envelope_sha256": canonical_sha256(
            {"candidate_id": candidate_id, "binding": "candidate-envelope"}
        ),
        "data_lane": lane,
        "evidence_profile_id": selected_profile,
        "evidence_components": (
            _profile_components(selected_profile)
            if evidence_components is None
            else deepcopy(evidence_components)
        ),
        "mass_units": mass,
        "molecule_ids": list(molecules or (f"mol-{candidate_id}",)),
        "partition_scaffold_ids": list(scaffolds or (f"scaffold-{candidate_id}",)),
        "source_group_ids": [source_group or f"source-group-{candidate_id}"],
        "source_group_namespace": _relationship_namespace(
            "source_group",
            source_asset_id=source_group_source_asset_id,
            source_asset_sha256=source_group_source_asset_sha256,
            namespace_id=source_group_namespace_id,
            cross_lane=source_group_cross_lane,
        ),
        "identity_definitions": (
            _identity_definitions()
            if identity_definitions is None
            else deepcopy(identity_definitions)
        ),
        "shared_prefix_branch_group_id": shared_prefix,
        "alternative_route_group_id": alternative_route,
        "inverse_pair_group_id": inverse_pair,
        "correction_group_id": correction,
        "constraint_compatible_alternative_route_group_id": (constrained_alternative_route),
        "relationship_namespaces": relationship_namespaces,
        "document_group_id": document,
        "document_provenance": document_provenance,
        "series_group_id": series,
        "series_provenance": series_provenance,
        "transformation_signature": transformation,
        "transformation_definition": (
            _transformation_definition() if transformation is not None else None
        ),
        "metadata": {"fixture": True},
    }


def _base_rows() -> list[dict]:
    return [
        _row(
            "a",
            lane="observed_local_analogue",
            mass=2,
            molecules=("shared-molecule",),
            scaffolds=("shared-scaffold",),
        ),
        _row(
            "b",
            lane="operator_aware_real_endpoint",
            mass=3,
            molecules=("shared-molecule",),
            scaffolds=("shared-scaffold",),
        ),
        _row("c", mass=5, transformation="common-swap"),
        _row(
            "d",
            lane="reversible_synthetic_walk",
            mass=7,
            transformation="common-swap",
        ),
        _row("e", mass=11),
        _row("f", mass=13),
        _row("g", mass=17),
        _row("h", mass=19),
    ]


def test_census_keeps_transformations_diagnostic_and_measures_bridge_effect() -> None:
    census = _build(_base_rows())

    assert census["status"] == "BLOCKED"
    assert census["status_scope"] == "CENSUS_STRUCTURAL_COMPLETION_ONLY"
    assert census["census_structural_complete"] is True
    assert census["corpus_ready"] is False
    assert census["split_ready"] is False
    assert census["training_ready"] is False
    assert census["authority"] == {
        "corpus": "NO_CORPUS_READINESS_AUTHORITY",
        "split": "NO_SPLIT_AUTHORITY",
        "training": "NO_TRAINING_AUTHORITY",
    }
    assert census["split_ratios_selected"] is False
    assert census["split_assignment_selected"] is False
    assert {
        "external_authority."
        "authoritative_trace_derived_record_membership_classification:"
        "unresolved_no_go_pending_trace_derived_predicate_receipts",
        "external_authority.physical_relationship_receipt_resolution:"
        "unresolved_no_go_pending_physical_receipt_resolver",
        "split_assignment.not_performed_by_component_census",
    }.issubset(census["blockers"])
    assert census["hard_component_census"]["component_count"] == 7
    assert census["identities"]["candidate_source_binding_inventory_sha256"] == canonical_sha256(
        census["candidate_source_binding_inventory"]
    )
    assert {
        "evidence_class",
        "declared_partition_role",
    }.isdisjoint(census["vertex_inventory"][0])

    components = census["hard_component_census"]["components"]
    shared = next(component for component in components if component["candidate_count"] == 2)
    assert shared["candidate_ids"] == ["a", "b"]
    assert shared["data_lanes"] == [
        "observed_local_analogue",
        "operator_aware_real_endpoint",
    ]
    assert shared["candidate_count_by_lane"] == {
        "observed_local_analogue": 1,
        "operator_aware_real_endpoint": 1,
    }
    assert shared["mass_units_by_lane"] == {
        "observed_local_analogue": 2,
        "operator_aware_real_endpoint": 3,
    }
    assert census["cross_lane_overlap"]["components_with_multiple_lanes"] == 1

    transformation = next(
        effect
        for effect in census["incremental_edge_type_effects"]
        if effect["edge_type"] == "transformation_signature"
    )
    assert transformation["mode"] == "diagnostic"
    assert transformation["groups_bridging_hard_components"] == 1
    assert transformation["potential_component_unions"] == 1
    assert transformation["hard_baseline"]["component_count"] == 7
    assert transformation["hypothetical_after_this_type_only"]["component_count"] == 6

    ranges = census["attainable_four_role_ranges"]
    assert ranges["four_nonempty_roles_feasible"] is True
    assert ranges["ratios_selected"] is False
    assert ranges["per_role_symmetric_marginal_bounds"]["minimum_mass_units"] == 5
    assert ranges["per_role_symmetric_marginal_bounds"]["maximum_mass_units"] == 60

    validate_split_component_census(census)


def test_output_is_invariant_to_semantically_irrelevant_row_order() -> None:
    rows = _base_rows()
    forward = _build(rows)
    reverse = _build(list(reversed(rows)))

    assert forward == reverse
    assert forward["census_sha256"] == reverse["census_sha256"]


def test_builder_consumes_a_one_shot_candidate_stream_once() -> None:
    consumed: list[str] = []

    def stream():
        for row in _base_rows():
            consumed.append(row["candidate_id"])
            yield row

    census = _build(stream())

    assert consumed == [row["candidate_id"] for row in _base_rows()]
    assert census["input_summary"]["source_rows"] == len(consumed)
    assert census["census_structural_complete"] is True


@pytest.mark.parametrize(
    ("bad_row", "reason_code"),
    [
        (
            _row("bad", lane="invented_lane"),
            "data_lane.not_in_contract",
        ),
        (
            _row(
                "bad",
                lane="observed_local_analogue",
                evidence_profile_id="executor_generated_walk",
            ),
            "evidence_assignment.invalid",
        ),
    ],
)
def test_noncontract_lane_or_evidence_profile_blocks_structural_pass(
    bad_row: dict,
    reason_code: str,
) -> None:
    census = _build([bad_row, _row("a"), _row("b"), _row("c"), _row("d")])

    assert census["status"] == "BLOCKED"
    assert census["census_structural_complete"] is False
    assert "stream.invalid_rows" in census["blockers"]
    assert reason_code in census["invalid_rows"][0]["reason_codes"]


def test_policy_lane_identity_must_match_validated_editing_contract() -> None:
    policy = _policy()
    policy["lane_contract_identity"]["ordered_lanes"][0]["admissible_evidence_profiles"] = [
        "executor_generated_walk"
    ]

    with pytest.raises(
        EditingV2SplitCensusError,
        match="lane_contract_identity disagrees",
    ):
        _build(_base_rows(), policy=policy)


def test_identity_definition_mismatch_is_reason_coded_and_blocks() -> None:
    definitions = _identity_definitions()
    definitions["exact_molecule"]["implementation_sha256"] = "e" * 64
    bad = _row("bad", identity_definitions=definitions)

    census = _build([bad, _row("a"), _row("b"), _row("c"), _row("d")])

    assert census["status"] == "BLOCKED"
    assert "identity_definitions.contract_mismatch" in census["invalid_rows"][0]["reason_codes"]


def test_source_group_is_hard_across_data_lanes() -> None:
    rows = [
        _row(
            "a",
            lane="observed_local_analogue",
            source_group="shared-source",
        ),
        _row(
            "b",
            lane="operator_aware_real_endpoint",
            source_group="shared-source",
        ),
        _row("c"),
        _row("d"),
    ]

    census = _build(rows)

    components = [
        component["candidate_ids"] for component in census["hard_component_census"]["components"]
    ]
    assert ["a", "b"] in components
    assert census["hard_component_census"]["component_count"] == 3
    source_group = next(
        group
        for group in census["relationship_inventory"]
        if group["edge_type"] == "source_group" and group["member_count"] == 2
    )
    assert source_group["mode"] == "hard"
    assert source_group["cross_lane"] is True
    assert census["relationship_summary"]["source_group"]["mode"] == "hard"


def test_equal_source_group_labels_from_distinct_source_assets_do_not_merge() -> None:
    rows = [
        _row(
            "a",
            source_group="source-7",
            source_group_source_asset_id="source-asset-a",
            source_group_source_asset_sha256=SHA_A,
        ),
        _row(
            "b",
            source_group="source-7",
            source_group_source_asset_id="source-asset-b",
            source_group_source_asset_sha256=SHA_B,
        ),
        _row("c"),
        _row("d"),
    ]

    census = _build(rows)

    assert census["hard_component_census"]["component_count"] == 4
    assert census["relationship_summary"]["source_group"]["multi_member_groups"] == 0


@pytest.mark.parametrize(
    ("mutate", "reason_code"),
    [
        (
            lambda row: row.__setitem__("source_group_ids", []),
            "source_group_ids.empty",
        ),
        (
            lambda row: row.__setitem__("source_group_namespace", None),
            "source_group_namespace.missing",
        ),
        (
            lambda row: row["source_group_namespace"].__setitem__(
                "cross_lane_sharing_authorized", False
            ),
            "source_group_namespace.cross_lane_sharing_required",
        ),
    ],
)
def test_missing_or_invalid_source_group_binding_is_reason_coded(
    mutate,
    reason_code: str,
) -> None:
    bad = _row("bad")
    mutate(bad)

    census = _build([bad, _row("a"), _row("b"), _row("c"), _row("d")])

    assert census["census_structural_complete"] is False
    assert reason_code in census["invalid_rows"][0]["reason_codes"]
    assert "bad" not in {vertex["candidate_id"] for vertex in census["vertex_inventory"]}


def test_source_group_binding_does_not_synthesize_document_or_series_evidence() -> None:
    census = _build([_row("a", source_group="source-a"), _row("b")])

    row = next(vertex for vertex in census["vertex_inventory"] if vertex["candidate_id"] == "a")
    assert row["document_group_id"] is None
    assert row["document_provenance"] is None
    assert row["series_group_id"] is None
    assert row["series_provenance"] is None
    assert census["relationship_summary"]["document_group"]["groups"] == 0
    assert census["relationship_summary"]["series_group"]["groups"] == 0


def test_observed_document_and_series_groups_are_hard_and_source_scoped() -> None:
    doc_a = _observed_provenance(namespace="documents", row_id="doc-row-a")
    doc_b = _observed_provenance(namespace="documents", row_id="doc-row-b")
    series_a = _observed_provenance(namespace="series", row_id="series-row-a")
    series_b = _observed_provenance(namespace="series", row_id="series-row-b")
    rows = [
        _row("a", document="document-7", document_provenance=doc_a),
        _row("b", document="document-7", document_provenance=doc_b),
        _row("c", series="series-9", series_provenance=series_a),
        _row("d", series="series-9", series_provenance=series_b),
        _row("e"),
        _row("f"),
    ]

    census = _build(rows)

    components = [
        component["candidate_ids"] for component in census["hard_component_census"]["components"]
    ]
    assert ["a", "b"] in components
    assert ["c", "d"] in components
    assert census["relationship_summary"]["document_group"]["mode"] == "hard"
    assert census["relationship_summary"]["series_group"]["mode"] == "hard"


def test_equal_document_labels_from_distinct_source_assets_do_not_merge() -> None:
    provenance_a = _observed_provenance(namespace="local-document-ids", row_id="a")
    provenance_b = _observed_provenance(namespace="local-document-ids", row_id="b")
    provenance_b["source_asset_id"] = "different-fixture-asset"
    provenance_b["source_asset_sha256"] = "c" * 64
    rows = [
        _row("a", document="document-1", document_provenance=provenance_a),
        _row("b", document="document-1", document_provenance=provenance_b),
        _row("c"),
        _row("d"),
    ]

    census = _build(rows)

    assert census["hard_component_census"]["component_count"] == 4
    assert census["relationship_summary"]["document_group"]["multi_member_groups"] == 0


def test_equal_simple_group_labels_from_distinct_sources_do_not_merge() -> None:
    rows = [
        _row(
            "a",
            inverse_pair="inverse-pair-1",
            relationship_source_asset_id="source-a",
            relationship_source_asset_sha256=SHA_A,
        ),
        _row(
            "b",
            inverse_pair="inverse-pair-1",
            relationship_source_asset_id="source-b",
            relationship_source_asset_sha256=SHA_B,
        ),
        _row("c"),
        _row("d"),
    ]

    census = _build(rows)

    assert census["status"] == "BLOCKED"
    assert census["hard_component_census"]["component_count"] == 4
    assert census["relationship_summary"]["inverse_pair"]["multi_member_groups"] == 0


def test_intentional_cross_lane_simple_group_uses_shared_declared_namespace() -> None:
    rows = [
        _row(
            "a",
            lane="observed_local_analogue",
            inverse_pair="inverse-pair-1",
            cross_lane_relationships=True,
        ),
        _row(
            "b",
            lane="operator_aware_real_endpoint",
            inverse_pair="inverse-pair-1",
            cross_lane_relationships=True,
        ),
        _row("c"),
        _row("d"),
        _row("e"),
    ]

    census = _build(rows)

    assert census["status"] == "BLOCKED"
    assert census["hard_component_census"]["component_count"] == 5
    inverse_group = next(
        group
        for group in census["relationship_inventory"]
        if group["edge_type"] == "inverse_pair" and group["member_count"] == 2
    )
    assert inverse_group["cross_lane"] is True
    assert inverse_group["mode"] == "diagnostic"
    assert inverse_group["relationship_key"]["data_lane_scope"] is None


@pytest.mark.parametrize(
    ("row_keyword", "edge_type"),
    [
        ("shared_prefix", "shared_prefix_branch"),
        ("alternative_route", "alternative_route"),
        ("inverse_pair", "inverse_pair"),
        ("correction", "correction"),
        (
            "constrained_alternative_route",
            "constraint_compatible_alternative_route",
        ),
    ],
)
def test_unresolved_relationship_groups_remain_diagnostic(
    row_keyword: str,
    edge_type: str,
) -> None:
    relationship = {row_keyword: "shared-group"}
    rows = [
        _row("a", **relationship),
        _row("b", **relationship),
        _row("c"),
        _row("d"),
        _row("e"),
    ]

    census = _build(rows)

    assert census["status"] == "BLOCKED"
    assert census["hard_component_census"]["component_count"] == 5
    effect = next(
        value
        for value in census["incremental_edge_type_effects"]
        if value["edge_type"] == edge_type
    )
    assert effect["mode"] == "diagnostic"
    assert effect["potential_component_unions"] == 1


def test_unresolved_relationship_receipts_cannot_create_hard_split_edges() -> None:
    policy = _policy()
    policy["edge_modes"]["inverse_pair"] = "hard"

    with pytest.raises(
        EditingV2SplitCensusError,
        match="before physical receipt resolution",
    ):
        _build(_base_rows(), policy=policy)


@pytest.mark.parametrize(
    ("field", "provenance_field", "reason_code"),
    [
        ("document_group_id", "document_provenance", "document_provenance.missing"),
        ("series_group_id", "series_provenance", "series_provenance.missing"),
    ],
)
def test_missing_observed_group_provenance_is_reason_coded_and_blocks(
    field: str,
    provenance_field: str,
    reason_code: str,
) -> None:
    bad = _row("bad")
    bad[field] = "claimed-observed-group"
    bad[provenance_field] = None
    rows = [bad, _row("a"), _row("b"), _row("c"), _row("d")]

    census = _build(rows)

    assert census["status"] == "BLOCKED"
    assert "stream.invalid_rows" in census["blockers"]
    assert census["input_summary"]["valid_vertices"] == 4
    assert census["input_summary"]["invalid_rows"] == 1
    assert reason_code in census["invalid_rows"][0]["reason_codes"]
    assert "bad" not in {vertex["candidate_id"] for vertex in census["vertex_inventory"]}
    validate_split_component_census(census)


def test_inferred_series_provenance_cannot_masquerade_as_observed() -> None:
    provenance = _observed_provenance(namespace="series")
    provenance["provenance_kind"] = "structurally_inferred_group"
    bad = _row(
        "bad",
        series="series-1",
        series_provenance=provenance,
    )

    census = _build([bad, _row("a"), _row("b"), _row("c"), _row("d")])

    assert "series_provenance.not_observed" in census["invalid_rows"][0]["reason_codes"]
    assert census["status"] == "BLOCKED"


def test_candidate_census_rejects_caller_supplied_partition_claims() -> None:
    bad = _row("bad")
    bad["declared_partition_role"] = "train"
    census = _build([bad, _row("a"), _row("b"), _row("c"), _row("d")])

    assert census["census_structural_complete"] is False
    assert "row.fields_mismatch" in census["invalid_rows"][0]["reason_codes"]
    assert "declared_role_audit" not in census


def test_duplicate_candidate_ids_exclude_every_occurrence() -> None:
    rows = [_row("duplicate"), _row("duplicate"), _row("a"), _row("b")]

    census = _build(rows)

    assert census["input_summary"]["valid_vertices"] == 2
    assert census["input_summary"]["invalid_rows"] == 2
    assert all("candidate_id.duplicate" in row["reason_codes"] for row in census["invalid_rows"])


def test_transformation_can_be_hard_only_with_explicit_authorization() -> None:
    rows = [
        _row("a", transformation="same"),
        _row("b", transformation="same"),
        _row("c"),
        _row("d"),
        _row("e"),
    ]
    policy = _policy()
    policy["edge_modes"]["transformation_signature"] = "hard"

    with pytest.raises(
        EditingV2SplitCensusError,
        match="transformation_hard_authorized",
    ):
        _build(rows, policy=policy)

    policy["transformation_hard_authorized"] = True
    census = _build(rows, policy=policy)
    assert census["hard_component_census"]["component_count"] == 4
    assert census["relationship_summary"]["transformation_signature"]["mode"] == "hard"


@pytest.mark.parametrize("edge_type", ["partition_scaffold", "source_group"])
def test_mandatory_hard_identity_policy_cannot_be_weakened(edge_type: str) -> None:
    policy = _policy()
    policy["edge_modes"][edge_type] = "diagnostic"

    with pytest.raises(EditingV2SplitCensusError, match="may not be weakened"):
        _build(_base_rows(), policy=policy)


def test_validator_rejects_tampered_output() -> None:
    census = _build(_base_rows())
    tampered = deepcopy(census)
    tampered["input_summary"]["valid_vertices"] += 1

    with pytest.raises(EditingV2SplitCensusError, match="SHA-256 mismatch"):
        validate_split_component_census(tampered)


def test_validator_rejects_provenance_tampering_even_with_rehashed_outer_body() -> None:
    census = _build(_base_rows())
    tampered = deepcopy(census)
    tampered["implementation_provenance"]["python_runtime"]["version"] = "0.0.0"
    body = dict(tampered)
    body.pop("census_sha256")
    tampered["census_sha256"] = canonical_sha256(body)

    with pytest.raises(
        EditingV2SplitCensusError,
        match="implementation_provenance_sha256",
    ):
        validate_split_component_census(tampered)


def test_validator_rejects_rehashed_authority_or_stale_vertex_tampering() -> None:
    census = _build(_base_rows())
    tampered = deepcopy(census)
    tampered["blockers"] = [
        blocker for blocker in tampered["blockers"] if not blocker.startswith("external_authority.")
    ]
    body = dict(tampered)
    body.pop("census_sha256")
    tampered["census_sha256"] = canonical_sha256(body)
    with pytest.raises(
        EditingV2SplitCensusError,
        match="mandatory external-authority blocker",
    ):
        validate_split_component_census(tampered)

    tampered = deepcopy(census)
    tampered["vertex_inventory"][0]["evidence_class"] = "observed"
    tampered["identities"]["vertex_inventory_sha256"] = canonical_sha256(
        tampered["vertex_inventory"]
    )
    body = dict(tampered)
    body.pop("census_sha256")
    tampered["census_sha256"] = canonical_sha256(body)
    with pytest.raises(EditingV2SplitCensusError, match="vertex\\[0\\] fields"):
        validate_split_component_census(tampered)


def test_thin_cli_writes_a_valid_immutable_bridge_bound_census(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "census.json"
    policy_path = tmp_path / "policy.json"
    input_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in _base_rows()))
    policy_path.write_text(json.dumps(_policy(), indent=2, sort_keys=True) + "\n")
    source_stream = _candidate_source_stream(input_path)
    monkeypatch.setattr(
        split_census_cli,
        "validate_candidate_provenance_bridge",
        lambda *args, **kwargs: {"source_stream": source_stream},
    )
    command = [
        "--input-jsonl",
        str(input_path),
        "--output-json",
        str(output_path),
        "--candidate-materialization-dir",
        str(tmp_path / "candidate_materialization"),
        "--candidate-provenance-bridge-dir",
        str(tmp_path / "candidate_bridge"),
        "--candidate-provenance-registry",
        str(tmp_path / "candidate_registry.json"),
        "--policy-json",
        str(policy_path),
        "--editing-corpus-contract",
        str(CONTRACT_PATH),
    ]

    first = split_census_cli.main(command)
    assert first == 2
    census = json.loads(output_path.read_text())
    validate_split_component_census(census)
    assert census["status_scope"] == "CENSUS_STRUCTURAL_COMPLETION_ONLY"
    assert census["authority"]["split"] == "NO_SPLIT_AUTHORITY"
    assert census["authority"]["training"] == "NO_TRAINING_AUTHORITY"
    assert census["census_structural_complete"] is True
    assert census["source_stream"] == source_stream
    provenance = census["implementation_provenance"]
    assert len(provenance["code_revision"]["commit_sha"]) >= 40
    assert isinstance(provenance["code_revision"]["dirty"], bool)
    assert provenance["python_runtime"]["version"]
    assert provenance["policy_file"]["sha256"]
    assert provenance["editing_corpus_contract_file"]["sha256"]
    assert len(provenance["source_files"]) == 5
    first_bytes = output_path.read_bytes()

    second = split_census_cli.main(command)
    assert second == 2
    assert output_path.read_bytes() == first_bytes


def test_thin_cli_rejects_split_input_not_bound_by_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "candidates.jsonl"
    input_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in _base_rows()))
    source_stream = _candidate_source_stream(input_path, file_sha256="0" * 64)
    monkeypatch.setattr(
        split_census_cli,
        "validate_candidate_provenance_bridge",
        lambda *args, **kwargs: {"source_stream": source_stream},
    )

    with pytest.raises(
        EditingV2SplitCensusError,
        match="physical SHA-256 disagrees",
    ):
        split_census_cli._validated_candidate_source_stream(
            input_jsonl=input_path,
            candidate_materialization_dir=tmp_path / "candidate_materialization",
            candidate_provenance_bridge_dir=tmp_path / "candidate_bridge",
            candidate_provenance_registry=tmp_path / "candidate_registry.json",
            editing_corpus_contract_path=CONTRACT_PATH,
        )


def test_thin_cli_rejects_legacy_generic_source_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "candidates.jsonl"
    input_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in _base_rows()))
    monkeypatch.setattr(
        split_census_cli,
        "validate_candidate_provenance_bridge",
        lambda *args, **kwargs: {
            "source_stream": {
                "path": str(input_path),
                "sha256": split_census_cli._file_sha256(input_path),
                "bytes": input_path.stat().st_size,
                "nonempty_jsonl_rows": len(_base_rows()),
            }
        },
    )

    with pytest.raises(
        EditingV2SplitCensusError,
        match="candidate provenance bridge validation failed",
    ):
        split_census_cli._validated_candidate_source_stream(
            input_jsonl=input_path,
            candidate_materialization_dir=tmp_path / "candidate_materialization",
            candidate_provenance_bridge_dir=tmp_path / "candidate_bridge",
            candidate_provenance_registry=tmp_path / "candidate_registry.json",
            editing_corpus_contract_path=CONTRACT_PATH,
        )
