from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from compose_v4.data.editing_candidate_audit_ledger import (
    CANDIDATE_AUDIT_LEDGER_STATUS,
    EVIDENCE_COMPONENTS,
    CandidateAuditAttempt,
    CandidateCompilerIdentity,
    CandidateEvidence,
    CandidateEvidenceComponent,
    CandidateLaneResolution,
    CandidatePolicyBindings,
    CandidateSourceIdentity,
    EditingCandidateAuditLedgerError,
    build_candidate_audit_ledger,
    candidate_attempt_stream_sha256,
    canonical_sha256,
    validate_candidate_audit_ledger,
)
from compose_v4.data.editing_corpus_contract import load_editing_corpus_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "editing_corpus_v2_contract.json"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _contract() -> dict:
    return load_editing_corpus_contract(CONTRACT_PATH)


def _contract_file_sha256() -> str:
    return hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()


def _source_identity() -> CandidateSourceIdentity:
    return CandidateSourceIdentity(
        manifest_file_sha256=SHA_A,
        manifest_sha256=SHA_B,
    )


def _compiler_identity() -> CandidateCompilerIdentity:
    return CandidateCompilerIdentity(
        implementation_sha256=SHA_C,
        config_sha256=SHA_D,
        operator_contract_sha256=SHA_E,
        canonicalizer_sha256=SHA_A,
    )


def _component(kind: str, index: int) -> CandidateEvidenceComponent:
    return CandidateEvidenceComponent(
        kind=kind,
        reference_ids=(f"evidence-reference-{index}",),
        evidence_sha256=chr(ord("a") + index) * 64,
    )


def _evidence(profile_id: str) -> CandidateEvidence:
    profile = next(
        profile
        for profile in _contract()["evidence_component_contract"]["profiles"]
        if profile["id"] == profile_id
    )
    return CandidateEvidence(
        **{
            component: _component(profile["components"][component], index)
            for index, component in enumerate(EVIDENCE_COMPONENTS)
        }
    )


def _lane_resolution(
    *,
    lane: str = "operator_aware_real_endpoint",
    profile_id: str = "observed_pair_compiled_path",
) -> CandidateLaneResolution:
    return CandidateLaneResolution(
        data_lane=lane,
        evidence_profile_id=profile_id,
        resolver_identity_sha256=SHA_A,
        routing_policy_sha256=SHA_B,
        receipt_sha256=SHA_C,
    )


def _resolved_policies() -> CandidatePolicyBindings:
    return CandidatePolicyBindings(
        mapping_policy_sha256=SHA_A,
        metric_policy_sha256=SHA_B,
    )


def _compiled_attempt(
    *,
    attempt_index: int = 0,
    candidate_id: str = "candidate-0",
    trace_id: str = "trace-0",
    lane: str = "operator_aware_real_endpoint",
    profile_id: str = "observed_pair_compiled_path",
) -> CandidateAuditAttempt:
    return CandidateAuditAttempt(
        attempt_index=attempt_index,
        candidate_id=candidate_id,
        candidate_payload_sha256=chr(ord("1") + attempt_index) * 64,
        source_record_ids=("source-row-1", "source-row-2"),
        policy_bindings=_resolved_policies(),
        lane_resolution=_lane_resolution(lane=lane, profile_id=profile_id),
        evidence=_evidence(profile_id),
        accepted_trace_id=trace_id,
    )


def _rejected_attempt() -> CandidateAuditAttempt:
    return CandidateAuditAttempt(
        attempt_index=1,
        candidate_id="candidate-1",
        candidate_payload_sha256="2" * 64,
        source_record_ids=("source-row-3",),
        policy_bindings=CandidatePolicyBindings(
            mapping_policy_sha256=None,
            metric_policy_sha256=SHA_B,
            unresolved_mapping_policies=("constant_core_mapping_policy",),
        ),
        rejection_code="policy.unresolved_mapping",
        rejection_detail="constant-core mapping policy is not frozen",
    )


def _ledger(attempts: tuple[CandidateAuditAttempt, ...]) -> dict:
    return build_candidate_audit_ledger(
        contract=_contract(),
        contract_file_sha256=_contract_file_sha256(),
        source_identity=_source_identity(),
        compiler_identity=_compiler_identity(),
        attempts=attempts,
        expected_attempt_count=len(attempts),
        expected_attempt_stream_sha256=candidate_attempt_stream_sha256(attempts),
    )


def _validate(ledger: dict, attempts: tuple[CandidateAuditAttempt, ...]) -> None:
    validate_candidate_audit_ledger(
        ledger,
        contract=_contract(),
        contract_file_sha256=_contract_file_sha256(),
        expected_source_identity=_source_identity(),
        expected_compiler_identity=_compiler_identity(),
        expected_attempt_count=len(attempts),
        expected_attempt_stream_sha256=candidate_attempt_stream_sha256(attempts),
    )


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for child in value.values() for key in _all_keys(child)}
    if isinstance(value, list):
        return {key for child in value for key in _all_keys(child)}
    return set()


def test_candidate_ledger_preserves_compiled_and_rejected_attempts() -> None:
    attempts = (_compiled_attempt(), _rejected_attempt())
    first = _ledger(attempts)
    second = _ledger(attempts)

    assert first == second
    assert first["status"] == CANDIDATE_AUDIT_LEDGER_STATUS
    assert first["training_authorized"] is False
    assert first["counts"] == {
        "attempted": 2,
        "compiled_candidates": 1,
        "rejected_candidates": 1,
        "without_lane_resolution": 1,
        "by_lane": {
            lane["id"]: int(lane["id"] == "operator_aware_real_endpoint")
            for lane in _contract()["data_lanes"]
        },
        "by_rejection_code": {"policy.unresolved_mapping": 1},
    }
    assert first["rows"][0]["disposition"] == "compiled_candidate"
    assert first["rows"][1]["disposition"] == "rejected_candidate"
    assert first["rows"][1]["lane_resolution"] is None
    assert first["rows"][1]["evidence"] is None
    assert first["rows_sha256"] == canonical_sha256(first["rows"])
    assert "split_assignment.unresolved_at_candidate_stage" in first["blockers"]
    assert {
        "proposed_partition",
        "proposed_lane",
        "partition",
        "evidence_class",
        "source_state_exact",
        "successor_states_exact",
        "executable_actions",
    }.isdisjoint(_all_keys(first))
    _validate(first, attempts)
    _validate(json.loads(json.dumps(first, sort_keys=True)), attempts)


def test_compiled_candidate_binds_exact_v2_profile_and_six_components() -> None:
    row = _ledger((_compiled_attempt(),))["rows"][0]
    assert row["data_lane"] == "operator_aware_real_endpoint"
    assert row["evidence_profile_id"] == "observed_pair_compiled_path"
    assert tuple(row["evidence"]) == EVIDENCE_COMPONENTS
    assert {
        component: row["evidence"][component]["kind"] for component in EVIDENCE_COMPONENTS
    } == _contract()["evidence_component_contract"]["profiles"][1]["components"]
    assert row["lane_resolution"]["receipt_sha256"] == SHA_C


def test_synthetic_walk_uses_only_its_admissible_lane_profile() -> None:
    synthetic = _compiled_attempt(
        lane="reversible_synthetic_walk",
        profile_id="executor_generated_walk",
    )
    row = _ledger((synthetic,))["rows"][0]
    assert row["data_lane"] == "reversible_synthetic_walk"
    assert row["evidence"]["action_sequence"]["kind"] == ("executor_generated_action_sequence")

    wrong_lane = replace(
        synthetic,
        lane_resolution=_lane_resolution(
            lane="observed_local_analogue",
            profile_id="executor_generated_walk",
        ),
    )
    with pytest.raises(
        EditingCandidateAuditLedgerError,
        match="lane/profile mismatch",
    ):
        _ledger((wrong_lane,))


def test_reserved_or_unavailable_profile_never_compiles() -> None:
    attempt = _compiled_attempt()
    unavailable_profile = "observed_pair_observed_actions"
    attempt = replace(
        attempt,
        lane_resolution=_lane_resolution(
            lane="operator_aware_real_endpoint",
            profile_id=unavailable_profile,
        ),
        evidence=_evidence(unavailable_profile),
    )
    with pytest.raises(EditingCandidateAuditLedgerError, match="reserved"):
        _ledger((attempt,))


def test_unresolved_policy_or_incomplete_evidence_never_compiles() -> None:
    unresolved = replace(
        _compiled_attempt(),
        policy_bindings=CandidatePolicyBindings(
            mapping_policy_sha256=SHA_A,
            metric_policy_sha256=None,
            unresolved_metric_policies=("intermediate_distance_definition",),
        ),
    )
    with pytest.raises(
        EditingCandidateAuditLedgerError,
        match="cannot grant compilation",
    ):
        _ledger((unresolved,))

    no_evidence = replace(_compiled_attempt(), evidence=None)
    with pytest.raises(
        EditingCandidateAuditLedgerError,
        match="six-component evidence",
    ):
        _ledger((no_evidence,))


def test_rejected_candidate_can_remain_unrouted_and_incomplete() -> None:
    attempt = replace(
        _rejected_attempt(),
        attempt_index=0,
        candidate_id="candidate-0",
        candidate_payload_sha256="1" * 64,
    )
    ledger = _ledger((attempt,))
    row = ledger["rows"][0]
    assert row["data_lane"] is None
    assert row["evidence_profile_id"] is None
    assert row["accepted_trace_id"] is None
    assert row["rejection"]["code"] == "policy.unresolved_mapping"


def test_scalar_evidence_class_has_no_compatibility_path() -> None:
    attempts = (_compiled_attempt(),)
    ledger = _ledger(attempts)
    tampered = copy.deepcopy(ledger)
    tampered["rows"][0]["evidence_class"] = "real"
    with pytest.raises(EditingCandidateAuditLedgerError, match="fields disagree"):
        _validate(tampered, attempts)


def test_attempt_stream_completeness_and_order_are_fail_closed() -> None:
    second = replace(_rejected_attempt(), attempt_index=2)
    with pytest.raises(EditingCandidateAuditLedgerError, match="contiguous"):
        candidate_attempt_stream_sha256((_compiled_attempt(), second))

    attempts = (_compiled_attempt(), _rejected_attempt())
    with pytest.raises(EditingCandidateAuditLedgerError, match="row count disagrees"):
        build_candidate_audit_ledger(
            contract=_contract(),
            contract_file_sha256=_contract_file_sha256(),
            source_identity=_source_identity(),
            compiler_identity=_compiler_identity(),
            attempts=attempts,
            expected_attempt_count=3,
            expected_attempt_stream_sha256=candidate_attempt_stream_sha256(attempts),
        )


def test_row_hash_provenance_and_blockers_fail_closed() -> None:
    attempts = (_compiled_attempt(), _rejected_attempt())
    ledger = _ledger(attempts)

    tampered = copy.deepcopy(ledger)
    tampered["rows"][0]["lane_resolution"]["receipt_sha256"] = SHA_E
    with pytest.raises(EditingCandidateAuditLedgerError, match="not canonical|hash"):
        _validate(tampered, attempts)

    tampered = copy.deepcopy(ledger)
    tampered["blockers"] = []
    with pytest.raises(
        EditingCandidateAuditLedgerError,
        match="blocker identity",
    ):
        _validate(tampered, attempts)

    with pytest.raises(
        EditingCandidateAuditLedgerError,
        match="provenance",
    ):
        validate_candidate_audit_ledger(
            ledger,
            contract=_contract(),
            contract_file_sha256=_contract_file_sha256(),
            expected_source_identity=CandidateSourceIdentity(
                manifest_file_sha256=SHA_D,
                manifest_sha256=SHA_B,
            ),
            expected_compiler_identity=_compiler_identity(),
            expected_attempt_count=2,
            expected_attempt_stream_sha256=candidate_attempt_stream_sha256(attempts),
        )
