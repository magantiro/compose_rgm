from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.data.editing_candidate_audit_ledger import (
    EVIDENCE_COMPONENTS,
    CandidateAuditAttempt,
    CandidateAuditLedgerAccumulator,
    CandidateCompilerIdentity,
    CandidateEvidence,
    CandidateEvidenceComponent,
    CandidateLaneResolution,
    CandidatePolicyBindings,
    CandidateSourceIdentity,
    build_candidate_audit_ledger,
    candidate_attempt_stream_sha256,
    canonical_json_bytes,
)
from compose_v4.data.editing_corpus_contract import (
    REQUIRED_DATA_LANES,
    load_editing_corpus_contract,
)
from compose_v4.data.editing_semantic_sampling_sidecar import (
    SEMANTIC_SAMPLING_SIDECAR_STATUS,
    Active8SidecarIdentity,
    EditingSemanticSamplingSidecarError,
    EndpointDescriptor,
    LaneRegistryIdentity,
    MappingDescriptor,
    PackedCorpusIdentity,
    PathDescriptor,
    SamplingCoefficients,
    SemanticGroups,
    SemanticSamplingProgress,
    SplitAssignmentIdentity,
    active8_admitted_trace_keys_sha256,
    admitted_record_envelope_inventory_sha256,
    build_semantic_sampling_sidecar,
    split_resolution_stream_sha256,
    validate_semantic_sampling_sidecar,
)
from compose_v4.data.packed_trace_store import PackedTraceAddress

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "editing_corpus_v2_contract.json"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
TRACE_ID = "trace-observed-local-0"
CANDIDATE_LEDGER_FILE_SHA256 = "0" * 64
SPLIT_MANIFEST_SHA256 = "9" * 64
ADMITTED_ENVELOPE_SHA256 = "8" * 64


def _contract() -> dict:
    return load_editing_corpus_contract(CONTRACT_PATH)


def _contract_file_sha256() -> str:
    return hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()


def _component(
    kind: str,
    reference_ids: tuple[str, ...],
    digest: str,
) -> CandidateEvidenceComponent:
    return CandidateEvidenceComponent(
        kind=kind,
        reference_ids=reference_ids,
        evidence_sha256=digest,
    )


def _candidate_attempt() -> CandidateAuditAttempt:
    profile = next(
        profile
        for profile in _contract()["evidence_component_contract"]["profiles"]
        if profile["id"] == "observed_pair_compiled_path"
    )
    evidence = {
        component: _component(
            profile["components"][component],
            (f"{component}-reference-1",),
            chr(ord("a") + index) * 64,
        )
        for index, component in enumerate(EVIDENCE_COMPONENTS)
    }
    return CandidateAuditAttempt(
        attempt_index=0,
        candidate_id="candidate-observed-local-0",
        candidate_payload_sha256="1" * 64,
        source_record_ids=("source-row-1", "source-row-2"),
        lane_resolution=CandidateLaneResolution(
            data_lane="operator_aware_real_endpoint",
            evidence_profile_id="observed_pair_compiled_path",
            resolver_identity_sha256=SHA_A,
            routing_policy_sha256=SHA_B,
            receipt_sha256=SHA_C,
        ),
        evidence=CandidateEvidence(**evidence),
        policy_bindings=CandidatePolicyBindings(
            mapping_policy_sha256=SHA_A,
            metric_policy_sha256=SHA_B,
        ),
        accepted_trace_id=TRACE_ID,
    )


def _candidate_ledger() -> dict:
    attempts = (_candidate_attempt(),)
    return build_candidate_audit_ledger(
        contract=_contract(),
        contract_file_sha256=_contract_file_sha256(),
        source_identity=CandidateSourceIdentity(
            manifest_file_sha256=SHA_C,
            manifest_sha256=SHA_D,
        ),
        compiler_identity=CandidateCompilerIdentity(
            implementation_sha256=SHA_E,
            config_sha256=SHA_F,
            operator_contract_sha256=SHA_A,
            canonicalizer_sha256=SHA_B,
        ),
        attempts=attempts,
        expected_attempt_count=1,
        expected_attempt_stream_sha256=(candidate_attempt_stream_sha256(attempts)),
    )


def _streamed_candidate_ledger(tmp_path: Path) -> tuple[dict, Path]:
    accumulator = CandidateAuditLedgerAccumulator(
        contract=_contract(),
        contract_file_sha256=_contract_file_sha256(),
        source_identity=CandidateSourceIdentity(
            manifest_file_sha256=SHA_C,
            manifest_sha256=SHA_D,
        ),
        compiler_identity=CandidateCompilerIdentity(
            implementation_sha256=SHA_E,
            config_sha256=SHA_F,
            operator_contract_sha256=SHA_A,
            canonicalizer_sha256=SHA_B,
        ),
    )
    row = accumulator.append(_candidate_attempt())
    rows_path = tmp_path / "candidate_audit_rows.jsonl"
    rows_path.write_bytes(canonical_json_bytes(row) + b"\n")
    ledger = accumulator.finalize(
        rows_relative_path=rows_path.name,
        rows_file_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        expected_attempt_count=1,
        expected_attempt_stream_sha256=candidate_attempt_stream_sha256((_candidate_attempt(),)),
    )
    return ledger, rows_path


def _lane_registry_identity() -> LaneRegistryIdentity:
    lane_ids = REQUIRED_DATA_LANES
    return LaneRegistryIdentity(
        registry_file_sha256=SHA_A,
        registry_sha256=SHA_B,
        lane_completion_sha256=tuple(
            (lane, str(index + 1) * 64) for index, lane in enumerate(lane_ids)
        ),
        lane_shard_inventory_sha256=tuple(
            (lane, chr(ord("a") + index) * 64) for index, lane in enumerate(lane_ids)
        ),
    )


def _packed_identity(
    progress_rows: tuple[SemanticSamplingProgress, ...],
) -> PackedCorpusIdentity:
    return PackedCorpusIdentity(
        source_manifest_file_sha256=SHA_C,
        source_manifest_sha256=SHA_D,
        unified_packed_manifest_file_sha256=SHA_E,
        unified_packed_manifest_sha256=SHA_F,
        packing_implementation_sha256=SHA_A,
        admitted_record_envelope_inventory_sha256=(
            admitted_record_envelope_inventory_sha256(progress_rows)
        ),
    )


def _address(
    *,
    layer: str = "operator_aware_real_endpoint",
    partition: str = "train",
) -> PackedTraceAddress:
    return PackedTraceAddress(
        packed_shard_content_sha256=SHA_A,
        packed_shard_name="shard_0000.jsonl.gz",
        entry_index=7,
        trace_id=TRACE_ID,
        layer=layer,
        partition=partition,
        source_key="CC",
        target_key="C",
        path_length=1,
    )


def _admitted_keys(
    address: PackedTraceAddress | None = None,
) -> tuple[tuple[str, int, str], ...]:
    selected = address or _address()
    return (
        (
            selected.packed_shard_content_sha256,
            selected.entry_index,
            selected.trace_id,
        ),
    )


def _active8_identity(
    keys: tuple[tuple[str, int, str], ...] | None = None,
) -> Active8SidecarIdentity:
    selected = keys or _admitted_keys()
    return Active8SidecarIdentity(
        inventory_manifest_file_sha256=SHA_B,
        inventory_sha256=SHA_C,
        effective_source_corpus_cache_sha256=SHA_D,
        unified_packed_manifest_sha256=SHA_F,
        support_contract_sha256=SHA_E,
        admitted_trace_count=len(selected),
        admitted_trace_keys_sha256=(active8_admitted_trace_keys_sha256(selected)),
    )


def _partition_resolution(
    *,
    role: str = "train",
    manifest_sha256: str = SPLIT_MANIFEST_SHA256,
) -> dict[str, object]:
    return {
        "status": "assigned",
        "assigned_role": role,
        "source_endpoint_role": role,
        "target_endpoint_role": role,
        "split_component_id": "split-component-1",
        "split_assignment_manifest_sha256": manifest_sha256,
    }


def _progress(
    progress_index: int,
    *,
    address: PackedTraceAddress | None = None,
    candidate_row_sha256: str | None = None,
) -> SemanticSamplingProgress:
    ledger = _candidate_ledger()
    return SemanticSamplingProgress(
        address=address or _address(),
        progress_index=progress_index,
        candidate_ledger_row_sha256=(candidate_row_sha256 or ledger["rows"][0]["row_sha256"]),
        partition_resolution=_partition_resolution(
            role=(address or _address()).partition,
        ),
        admitted_record_envelope_sha256=ADMITTED_ENVELOPE_SHA256,
        groups=SemanticGroups(
            source_group_id="source-group-1",
            scaffold_group_id="scaffold-group-1",
            series_group_id="series-group-1",
            document_or_source_group_id="document-group-1",
            transformation_signature="terminal-carbon-deletion",
        ),
        # Membership is hash-bound metadata. The contract intentionally keeps
        # chemistry-derived classifier authority as an explicit NO_GO blocker.
        record_membership_cells=("cardinality_shrinkage",),
        record_membership_classifier_identity_sha256=SHA_C,
        endpoint_descriptor=EndpointDescriptor(
            source_canonical_key="CC",
            target_canonical_key="C",
            atom_count_delta=-1,
            graph_cycle_rank_delta=0,
            source_endpoint_evidence_kind="observed_real_endpoint",
            target_endpoint_evidence_kind="observed_real_endpoint",
            pair_relationship_evidence_kind="observed_real_endpoint_pair",
        ),
        path_descriptor=PathDescriptor(
            path_length=1,
            operator_family_set=("atom_delete",),
            path_evidence_kind="compiled_path_between_real_endpoints",
            intermediate_evidence_kind="executor_generated_intermediates",
            action_sequence_evidence_kind="compiler_generated_action_sequence",
            maximum_intermediate_source_distance=1.0,
            immediate_reversal_count=0,
            repeated_state_count=0,
            metric_policy_sha256=SHA_B,
        ),
        relationship_group_ids=("inverse-pair-group-1",),
        mappings=MappingDescriptor(
            constant_core_mapping=((0, 0),),
            protected_mapping=None,
            mapping_policy_sha256=SHA_A,
        ),
        sampling_coefficients=SamplingCoefficients(
            path_sampling_coefficient=0.5,
            progress_sampling_coefficient=0.75,
            effective_teacher_coefficient=0.125 + progress_index * 0.025,
            coefficient_policy_sha256=SHA_D,
        ),
    )


def _split_identity(
    progress_rows: tuple[SemanticSamplingProgress, ...],
) -> SplitAssignmentIdentity:
    return SplitAssignmentIdentity(
        manifest_file_sha256=SHA_B,
        manifest_sha256=SPLIT_MANIFEST_SHA256,
        resolution_stream_sha256=split_resolution_stream_sha256(progress_rows),
    )


def _build(
    progress_rows: tuple[SemanticSamplingProgress, ...] | None = None,
    *,
    address: PackedTraceAddress | None = None,
    active8_identity: Active8SidecarIdentity | None = None,
    admitted_keys: tuple[tuple[str, int, str], ...] | None = None,
    packed_identity: PackedCorpusIdentity | None = None,
    candidate_ledger: dict | None = None,
    candidate_ledger_rows_path: Path | None = None,
) -> dict:
    selected_address = address or _address()
    keys = admitted_keys or _admitted_keys(selected_address)
    rows = progress_rows or (
        _progress(0, address=selected_address),
        _progress(1, address=selected_address),
    )
    selected_packed_identity = packed_identity or _packed_identity(rows)
    return build_semantic_sampling_sidecar(
        contract=_contract(),
        contract_file_sha256=_contract_file_sha256(),
        lane_registry_identity=_lane_registry_identity(),
        packed_corpus_identity=selected_packed_identity,
        active8_identity=active8_identity or _active8_identity(keys),
        split_assignment_identity=_split_identity(rows),
        active8_admitted_trace_keys=keys,
        candidate_ledger=candidate_ledger or _candidate_ledger(),
        candidate_ledger_manifest_file_sha256=(CANDIDATE_LEDGER_FILE_SHA256),
        candidate_ledger_rows_path=candidate_ledger_rows_path,
        progress_rows=rows,
    )


def _validate(
    manifest: dict,
    *,
    expected_rows: tuple[SemanticSamplingProgress, ...] | None = None,
    candidate_ledger: dict | None = None,
    candidate_ledger_rows_path: Path | None = None,
) -> None:
    rows = expected_rows or (_progress(0), _progress(1))
    validate_semantic_sampling_sidecar(
        manifest,
        contract=_contract(),
        contract_file_sha256=_contract_file_sha256(),
        expected_lane_registry_identity=_lane_registry_identity(),
        expected_packed_corpus_identity=_packed_identity(rows),
        expected_active8_identity=_active8_identity(),
        expected_split_assignment_identity=_split_identity(rows),
        expected_active8_admitted_trace_keys=_admitted_keys(),
        candidate_ledger=candidate_ledger or _candidate_ledger(),
        expected_candidate_ledger_manifest_file_sha256=(CANDIDATE_LEDGER_FILE_SHA256),
        candidate_ledger_rows_path=candidate_ledger_rows_path,
    )


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for child in value.values() for key in _all_keys(child)}
    if isinstance(value, list):
        return {key for child in value for key in _all_keys(child)}
    return set()


def test_sidecar_is_deterministic_exactly_addressed_and_metadata_only() -> None:
    first = _build()
    second = _build()

    assert first == second
    assert first["status"] == SEMANTIC_SAMPLING_SIDECAR_STATUS
    assert first["training_authorized"] is False
    assert first["counts"]["records"] == 1
    assert first["counts"]["progress_rows"] == 2
    assert [row["progress_key"]["progress_index"] for row in first["rows"]] == [0, 1]
    assert first["rows"][0]["packed_address"]["packed_shard_content_sha256"] == SHA_A
    assert (
        first["rows"][0]["candidate_ledger_binding"]["candidate_row_sha256"]
        == _candidate_ledger()["rows"][0]["row_sha256"]
    )
    assert first["rows"][0]["groups"]["series_scaffold_or_source_group_id"] == "series-group-1"
    assert first["rows"][0]["relationship_group_ids"] == ["inverse-pair-group-1"]
    assert first["rows"][0]["partition_resolution"]["assigned_role"] == "train"
    assert first["split_assignment_identity"][
        "resolution_stream_sha256"
    ] == split_resolution_stream_sha256((_progress(0), _progress(1)))
    assert first["packed_corpus_identity"][
        "admitted_record_envelope_inventory_sha256"
    ] == admitted_record_envelope_inventory_sha256((_progress(0), _progress(1)))
    assert {
        "source_state_exact",
        "successor_states_exact",
        "target_state_exact",
        "executable_actions",
        "inverse_record_id",
        "semantic_capability_ids",
        "capability_membership_policy_sha256",
        "relationship_groups",
        "evidence_class",
    }.isdisjoint(_all_keys(first))
    _validate(first)
    _validate(json.loads(json.dumps(first, sort_keys=True)))


def test_sidecar_consumes_schema_v3_candidate_rows_without_rehydrating_ledger(
    tmp_path: Path,
) -> None:
    ledger, rows_path = _streamed_candidate_ledger(tmp_path)

    manifest = _build(
        candidate_ledger=ledger,
        candidate_ledger_rows_path=rows_path,
    )
    assert manifest["candidate_ledger_identity"]["ledger_sha256"] == ledger["ledger_sha256"]
    assert manifest["candidate_ledger_identity"]["rows_sha256"] == ledger["rows_sha256"]
    _validate(
        manifest,
        candidate_ledger=ledger,
        candidate_ledger_rows_path=rows_path,
    )

    with pytest.raises(
        EditingSemanticSamplingSidecarError,
        match="requires candidate_ledger_rows_path",
    ):
        _build(candidate_ledger=ledger)


def test_component_evidence_is_preserved_without_observed_boolean_upgrade() -> None:
    manifest = _build()
    row = manifest["rows"][0]
    assert row["endpoint_descriptor"]["source_endpoint_evidence_kind"] == ("observed_real_endpoint")
    assert row["path_descriptor"]["path_evidence_kind"] == ("compiled_path_between_real_endpoints")
    assert row["path_descriptor"]["action_sequence_evidence_kind"] == (
        "compiler_generated_action_sequence"
    )
    assert "path_observed" not in _all_keys(row)
    assert row["record_membership_cells"] == ["cardinality_shrinkage"]
    assert row["record_membership_classifier_identity_sha256"] == SHA_C
    assert {
        "external_authority."
        "authoritative_trace_derived_record_membership_classification:"
        "unresolved_no_go_pending_trace_derived_predicate_receipts",
        "external_authority.physical_relationship_receipt_resolution:"
        "unresolved_no_go_pending_physical_receipt_resolver",
    }.issubset(manifest["blockers"])


def test_sidecar_requires_complete_progress_coverage_for_every_active8_trace() -> None:
    with pytest.raises(
        EditingSemanticSamplingSidecarError,
        match="every progress position",
    ):
        _build((_progress(0),))

    with pytest.raises(
        EditingSemanticSamplingSidecarError,
        match="deterministic packed/progress order",
    ):
        _build((_progress(1), _progress(0)))


def test_split_resolution_and_admitted_envelope_identities_fail_closed() -> None:
    wrong_resolution = replace(
        _progress(0),
        partition_resolution=_partition_resolution(manifest_sha256=SHA_A),
    )
    with pytest.raises(
        EditingSemanticSamplingSidecarError,
        match="bound split resolution or manifest",
    ):
        _build(
            (
                wrong_resolution,
                replace(_progress(1), partition_resolution=wrong_resolution.partition_resolution),
            )
        )

    rows = (_progress(0), _progress(1))
    wrong_envelope_inventory = replace(
        _packed_identity(rows),
        admitted_record_envelope_inventory_sha256=SHA_A,
    )
    with pytest.raises(
        EditingSemanticSamplingSidecarError,
        match="admitted-envelope inventory",
    ):
        _build(rows, packed_identity=wrong_envelope_inventory)


def test_unresolved_record_membership_cell_is_not_promoted_to_authority() -> None:
    unresolved = replace(
        _progress(0),
        record_membership_cells=("constraint_feasible_path",),
    )
    with pytest.raises(
        EditingSemanticSamplingSidecarError,
        match="unresolved membership cells",
    ):
        _build(
            (
                unresolved,
                replace(
                    _progress(1),
                    record_membership_cells=("constraint_feasible_path",),
                ),
            )
        )


def test_sidecar_rejects_non_active8_or_wrong_candidate_trace() -> None:
    other_address = replace(_address(), entry_index=8)
    with pytest.raises(
        EditingSemanticSamplingSidecarError,
        match="trace set disagrees",
    ):
        _build(
            (
                _progress(0, address=other_address),
                _progress(1, address=other_address),
            ),
            address=other_address,
            admitted_keys=_admitted_keys(),
            active8_identity=_active8_identity(_admitted_keys()),
        )

    with pytest.raises(
        EditingSemanticSamplingSidecarError,
        match="does not resolve",
    ):
        _build(
            (
                _progress(0, candidate_row_sha256=SHA_F),
                _progress(1, candidate_row_sha256=SHA_F),
            )
        )


def test_unresolved_mapping_metric_or_coefficient_policy_blocks_sidecar() -> None:
    unresolved_mapping = replace(
        _progress(0),
        mappings=MappingDescriptor(
            constant_core_mapping=((0, 0),),
            protected_mapping=None,
            mapping_policy_sha256=None,
            unresolved_mapping_policies=("constant_core_mapping",),
        ),
    )
    with pytest.raises(
        EditingSemanticSamplingSidecarError,
        match="unresolved mapping policy",
    ):
        _build((unresolved_mapping, _progress(1)))

    unresolved_metric = replace(
        _progress(0),
        path_descriptor=replace(
            _progress(0).path_descriptor,
            metric_policy_sha256=None,
            unresolved_metric_policies=("intermediate_distance",),
        ),
    )
    with pytest.raises(
        EditingSemanticSamplingSidecarError,
        match="unresolved metric policy",
    ):
        _build((unresolved_metric, _progress(1)))

    unresolved_coefficient = replace(
        _progress(0),
        sampling_coefficients=replace(
            _progress(0).sampling_coefficients,
            unresolved_coefficient_policies=("progress_weight",),
        ),
    )
    with pytest.raises(
        EditingSemanticSamplingSidecarError,
        match="unresolved coefficient policy",
    ):
        _build((unresolved_coefficient, _progress(1)))


@pytest.mark.parametrize(
    "address",
    [
        _address(partition="development"),
        _address(layer="legacy_mmp"),
    ],
)
def test_sidecar_enforces_exact_partition_roles_and_lanes(
    address: PackedTraceAddress,
) -> None:
    with pytest.raises(
        EditingSemanticSamplingSidecarError,
        match="outside the v2 contract",
    ):
        _build(
            (
                _progress(0, address=address),
                _progress(1, address=address),
            ),
            address=address,
        )


def test_active8_and_source_manifest_bindings_are_fail_closed() -> None:
    keys = _admitted_keys()
    wrong_active8 = replace(
        _active8_identity(keys),
        admitted_trace_keys_sha256=SHA_A,
    )
    with pytest.raises(
        EditingSemanticSamplingSidecarError,
        match="Active8 identity disagrees",
    ):
        _build(active8_identity=wrong_active8)

    rows = (_progress(0), _progress(1))
    wrong_source = replace(
        _packed_identity(rows),
        source_manifest_sha256=SHA_E,
    )
    with pytest.raises(
        EditingSemanticSamplingSidecarError,
        match="different source manifests",
    ):
        _build(packed_identity=wrong_source)


def test_sidecar_row_and_manifest_tampering_fail_closed() -> None:
    manifest = _build()
    tampered = copy.deepcopy(manifest)
    tampered["rows"][0]["sampling_coefficients"]["effective_teacher_coefficient"] = 99.0
    with pytest.raises(
        EditingSemanticSamplingSidecarError,
        match="not canonical|hash",
    ):
        _validate(tampered)

    tampered = copy.deepcopy(manifest)
    tampered["counts"]["records"] = 2
    with pytest.raises(
        EditingSemanticSamplingSidecarError,
        match="counts disagree",
    ):
        _validate(tampered)
