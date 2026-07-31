"""Deterministic, non-authorizing audit ledger for editing-V2 candidates.

The ledger is upstream of split assignment and packed admitted records.  It
preserves every attempted candidate, including attempts that cannot be routed,
compiled, or admitted.  A compiled trace is not an admitted training record.

Schema version 2 removes caller-declared partitions and scalar evidence
classes.  A compiled candidate binds one hash-addressed lane-resolution receipt,
one admissible evidence profile, and the exact six-component evidence mapping
declared by the editing-corpus contract.  The validator checks only this
envelope.  It does not verify lane-routing semantics, chemistry-derived corpus
membership, or relationship receipts, and the ledger records those limitations
as explicit launch blockers.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from compose_v4.data.editing_corpus_contract import (
    REQUIRED_DATA_LANES,
    validate_editing_corpus_contract,
    validate_evidence_assignment_envelope,
)

CANDIDATE_AUDIT_ROW_SCHEMA = "compose.editing_v2_candidate_audit_row"
CANDIDATE_AUDIT_ROW_SCHEMA_VERSION = 2
CANDIDATE_AUDIT_ROW_STATUS = "CANDIDATE_ATTEMPT_NO_TRAINING_AUTHORITY"
CANDIDATE_AUDIT_LEDGER_SCHEMA = "compose.editing_v2_candidate_audit_ledger"
CANDIDATE_AUDIT_LEDGER_SCHEMA_VERSION = 2
CANDIDATE_AUDIT_LEDGER_STATUS = "COMPLETE_NO_TRAINING_AUTHORITY"

EVIDENCE_COMPONENTS = (
    "source_endpoint",
    "target_endpoint",
    "pair_relationship",
    "path",
    "intermediates",
    "action_sequence",
)

_HEX64 = frozenset("0123456789abcdef")
_REJECTION_CODE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_ROW_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "attempt_index",
    "candidate_id",
    "candidate_payload_sha256",
    "contract_identity",
    "source_identity",
    "compiler_identity",
    "source_record_ids",
    "data_lane",
    "evidence_profile_id",
    "evidence",
    "lane_resolution",
    "policy_bindings",
    "disposition",
    "accepted_trace_id",
    "rejection",
    "row_sha256",
}
_LEDGER_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "blockers",
    "contract_identity",
    "source_identity",
    "compiler_identity",
    "lane_ids",
    "expected_attempt_count",
    "expected_attempt_stream_sha256",
    "counts",
    "rows",
    "rows_sha256",
    "ledger_sha256",
}
_CONTRACT_IDENTITY_FIELDS = {
    "schema",
    "schema_version",
    "contract_id",
    "file_sha256",
    "semantic_sha256",
}
_SOURCE_IDENTITY_FIELDS = {
    "manifest_file_sha256",
    "manifest_sha256",
}
_COMPILER_IDENTITY_FIELDS = {
    "implementation_sha256",
    "config_sha256",
    "operator_contract_sha256",
    "canonicalizer_sha256",
}
_EVIDENCE_COMPONENT_FIELDS = {
    "kind",
    "reference_ids",
    "evidence_sha256",
}
_LANE_RESOLUTION_FIELDS = {
    "data_lane",
    "evidence_profile_id",
    "resolver_identity_sha256",
    "routing_policy_sha256",
    "receipt_sha256",
}
_POLICY_FIELDS = {
    "mapping_policy_sha256",
    "metric_policy_sha256",
    "unresolved_mapping_policies",
    "unresolved_metric_policies",
}
_REJECTION_FIELDS = {"code", "detail"}


class EditingCandidateAuditLedgerError(ValueError):
    """The candidate ledger is malformed, incomplete, or scientifically unsafe."""


@dataclass(frozen=True)
class CandidateSourceIdentity:
    """Physical and semantic identity of the candidate-source manifest."""

    manifest_file_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class CandidateCompilerIdentity:
    """Exact implementation/configuration identities used for compilation."""

    implementation_sha256: str
    config_sha256: str
    operator_contract_sha256: str
    canonicalizer_sha256: str


@dataclass(frozen=True)
class CandidateEvidenceComponent:
    """One v2 evidence component plus provenance references."""

    kind: str
    reference_ids: tuple[str, ...]
    evidence_sha256: str


@dataclass(frozen=True)
class CandidateEvidence:
    """The exact six evidence components required by schema version 2."""

    source_endpoint: CandidateEvidenceComponent
    target_endpoint: CandidateEvidenceComponent
    pair_relationship: CandidateEvidenceComponent
    path: CandidateEvidenceComponent
    intermediates: CandidateEvidenceComponent
    action_sequence: CandidateEvidenceComponent


@dataclass(frozen=True)
class CandidateLaneResolution:
    """Hash-bound lane-routing output, never a caller proposal."""

    data_lane: str
    evidence_profile_id: str
    resolver_identity_sha256: str
    routing_policy_sha256: str
    receipt_sha256: str


@dataclass(frozen=True)
class CandidatePolicyBindings:
    """Frozen mapping/metric policies or explicit unresolved policy names."""

    mapping_policy_sha256: str | None
    metric_policy_sha256: str | None
    unresolved_mapping_policies: tuple[str, ...] = ()
    unresolved_metric_policies: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateAuditAttempt:
    """One attempted candidate and its compiler disposition.

    Rejected attempts may have no lane resolution or evidence bundle.  Compiled
    attempts require both, but remain candidate-only artifacts until split and
    whole-trace admission are resolved downstream.
    """

    attempt_index: int
    candidate_id: str
    candidate_payload_sha256: str
    source_record_ids: tuple[str, ...]
    policy_bindings: CandidatePolicyBindings
    lane_resolution: CandidateLaneResolution | None = None
    evidence: CandidateEvidence | None = None
    accepted_trace_id: str | None = None
    rejection_code: str | None = None
    rejection_detail: str | None = None


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic JSON bytes with non-finite values forbidden."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EditingCandidateAuditLedgerError(
            f"value is not deterministic JSON: {error}"
        ) from error


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 identity of deterministic JSON content."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_copy(value: Any, *, field: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except EditingCandidateAuditLedgerError as error:
        raise EditingCandidateAuditLedgerError(f"{field}: {error}") from error


def _require_exact_fields(
    value: object,
    expected: set[str],
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EditingCandidateAuditLedgerError(f"{field} must be an object")
    actual = set(value)
    if actual != expected:
        raise EditingCandidateAuditLedgerError(
            f"{field} fields disagree; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return value


def _require_sha256(value: object, *, field: str) -> str:
    digest = value if isinstance(value, str) else ""
    if len(digest) != 64 or any(character not in _HEX64 for character in digest):
        raise EditingCandidateAuditLedgerError(f"{field} must be a full lowercase SHA-256")
    return digest


def _require_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise EditingCandidateAuditLedgerError(f"{field} must be nonempty normalized text")
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise EditingCandidateAuditLedgerError(f"{field} must be a nonnegative integer")
    return value


def _normalized_unique_text(
    values: object,
    *,
    field: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise EditingCandidateAuditLedgerError(f"{field} must be a sequence of strings")
    normalized = tuple(
        _require_text(value, field=f"{field}[{index}]") for index, value in enumerate(values)
    )
    if len(normalized) != len(set(normalized)):
        raise EditingCandidateAuditLedgerError(f"{field} contains duplicates")
    if not allow_empty and not normalized:
        raise EditingCandidateAuditLedgerError(f"{field} must not be empty")
    if normalized != tuple(sorted(normalized)):
        raise EditingCandidateAuditLedgerError(f"{field} must be deterministically sorted")
    return normalized


def _contract_context(
    contract: Mapping[str, Any],
    *,
    contract_file_sha256: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    try:
        validate_editing_corpus_contract(dict(contract))
    except (TypeError, ValueError) as error:
        raise EditingCandidateAuditLedgerError(
            f"editing-corpus contract is invalid: {error}"
        ) from error
    lane_ids = tuple(str(lane["id"]) for lane in contract["data_lanes"])
    if lane_ids != REQUIRED_DATA_LANES:
        raise EditingCandidateAuditLedgerError(
            "candidate ledger requires the exact ordered five-lane v2 contract"
        )
    identity = {
        "schema": contract["schema"],
        "schema_version": contract["schema_version"],
        "contract_id": _require_text(
            contract["contract_id"],
            field="editing-corpus contract_id",
        ),
        "file_sha256": _require_sha256(
            contract_file_sha256,
            field="editing-corpus contract file SHA-256",
        ),
        "semantic_sha256": canonical_sha256(contract),
    }
    return identity, lane_ids


def _launch_blockers(contract: Mapping[str, Any]) -> list[str]:
    blockers = ["split_assignment.unresolved_at_candidate_stage"]
    blockers.extend(
        f"external_authority.{requirement['id']}:{requirement['status']}"
        for requirement in contract["admitted_record_contract"]["external_authority_requirements"]
    )
    return blockers


def _source_identity(
    value: CandidateSourceIdentity | Mapping[str, Any],
) -> dict[str, str]:
    payload = (
        {
            "manifest_file_sha256": value.manifest_file_sha256,
            "manifest_sha256": value.manifest_sha256,
        }
        if isinstance(value, CandidateSourceIdentity)
        else dict(
            _require_exact_fields(
                value,
                _SOURCE_IDENTITY_FIELDS,
                field="candidate source identity",
            )
        )
    )
    return {
        field: _require_sha256(payload[field], field=f"source_identity.{field}")
        for field in sorted(_SOURCE_IDENTITY_FIELDS)
    }


def _compiler_identity(
    value: CandidateCompilerIdentity | Mapping[str, Any],
) -> dict[str, str]:
    payload = (
        {
            "implementation_sha256": value.implementation_sha256,
            "config_sha256": value.config_sha256,
            "operator_contract_sha256": value.operator_contract_sha256,
            "canonicalizer_sha256": value.canonicalizer_sha256,
        }
        if isinstance(value, CandidateCompilerIdentity)
        else dict(
            _require_exact_fields(
                value,
                _COMPILER_IDENTITY_FIELDS,
                field="candidate compiler identity",
            )
        )
    )
    return {
        field: _require_sha256(payload[field], field=f"compiler_identity.{field}")
        for field in sorted(_COMPILER_IDENTITY_FIELDS)
    }


def _evidence_component(
    value: CandidateEvidenceComponent | Mapping[str, Any],
    *,
    component: str,
) -> dict[str, Any]:
    payload = (
        {
            "kind": value.kind,
            "reference_ids": list(value.reference_ids),
            "evidence_sha256": value.evidence_sha256,
        }
        if isinstance(value, CandidateEvidenceComponent)
        else dict(
            _require_exact_fields(
                value,
                _EVIDENCE_COMPONENT_FIELDS,
                field=f"evidence.{component}",
            )
        )
    )
    return {
        "kind": _require_text(payload["kind"], field=f"evidence.{component}.kind"),
        "reference_ids": list(
            _normalized_unique_text(
                payload["reference_ids"],
                field=f"evidence.{component}.reference_ids",
                allow_empty=True,
            )
        ),
        "evidence_sha256": _require_sha256(
            payload["evidence_sha256"],
            field=f"evidence.{component}.evidence_sha256",
        ),
    }


def _evidence_bundle(
    value: CandidateEvidence | Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]] | None:
    if value is None:
        return None
    if isinstance(value, CandidateEvidence):
        components: Mapping[str, Any] = {
            component: getattr(value, component) for component in EVIDENCE_COMPONENTS
        }
    else:
        components = _require_exact_fields(
            value,
            set(EVIDENCE_COMPONENTS),
            field="candidate evidence",
        )
    return {
        component: _evidence_component(
            components[component],
            component=component,
        )
        for component in EVIDENCE_COMPONENTS
    }


def _lane_resolution(
    value: CandidateLaneResolution | Mapping[str, Any] | None,
    *,
    lane_ids: tuple[str, ...],
) -> dict[str, str] | None:
    if value is None:
        return None
    payload = (
        {
            "data_lane": value.data_lane,
            "evidence_profile_id": value.evidence_profile_id,
            "resolver_identity_sha256": value.resolver_identity_sha256,
            "routing_policy_sha256": value.routing_policy_sha256,
            "receipt_sha256": value.receipt_sha256,
        }
        if isinstance(value, CandidateLaneResolution)
        else dict(
            _require_exact_fields(
                value,
                _LANE_RESOLUTION_FIELDS,
                field="lane_resolution",
            )
        )
    )
    lane = _require_text(payload["data_lane"], field="lane_resolution.data_lane")
    if lane not in lane_ids:
        raise EditingCandidateAuditLedgerError(
            f"lane_resolution.data_lane is outside the exact five-lane contract: {lane!r}"
        )
    return {
        "data_lane": lane,
        "evidence_profile_id": _require_text(
            payload["evidence_profile_id"],
            field="lane_resolution.evidence_profile_id",
        ),
        "resolver_identity_sha256": _require_sha256(
            payload["resolver_identity_sha256"],
            field="lane_resolution.resolver_identity_sha256",
        ),
        "routing_policy_sha256": _require_sha256(
            payload["routing_policy_sha256"],
            field="lane_resolution.routing_policy_sha256",
        ),
        "receipt_sha256": _require_sha256(
            payload["receipt_sha256"],
            field="lane_resolution.receipt_sha256",
        ),
    }


def _policy_bindings(
    value: CandidatePolicyBindings | Mapping[str, Any],
) -> dict[str, Any]:
    payload = (
        {
            "mapping_policy_sha256": value.mapping_policy_sha256,
            "metric_policy_sha256": value.metric_policy_sha256,
            "unresolved_mapping_policies": list(value.unresolved_mapping_policies),
            "unresolved_metric_policies": list(value.unresolved_metric_policies),
        }
        if isinstance(value, CandidatePolicyBindings)
        else dict(
            _require_exact_fields(
                value,
                _POLICY_FIELDS,
                field="candidate policy bindings",
            )
        )
    )
    unresolved_mapping = _normalized_unique_text(
        payload["unresolved_mapping_policies"],
        field="policy_bindings.unresolved_mapping_policies",
        allow_empty=True,
    )
    unresolved_metric = _normalized_unique_text(
        payload["unresolved_metric_policies"],
        field="policy_bindings.unresolved_metric_policies",
        allow_empty=True,
    )
    mapping_sha = payload["mapping_policy_sha256"]
    metric_sha = payload["metric_policy_sha256"]
    if mapping_sha is not None:
        mapping_sha = _require_sha256(
            mapping_sha,
            field="policy_bindings.mapping_policy_sha256",
        )
    if metric_sha is not None:
        metric_sha = _require_sha256(
            metric_sha,
            field="policy_bindings.metric_policy_sha256",
        )
    if mapping_sha is None and not unresolved_mapping:
        raise EditingCandidateAuditLedgerError(
            "missing mapping policy must be recorded as unresolved"
        )
    if metric_sha is None and not unresolved_metric:
        raise EditingCandidateAuditLedgerError(
            "missing metric policy must be recorded as unresolved"
        )
    return {
        "mapping_policy_sha256": mapping_sha,
        "metric_policy_sha256": metric_sha,
        "unresolved_mapping_policies": list(unresolved_mapping),
        "unresolved_metric_policies": list(unresolved_metric),
    }


def _policies_resolved(value: Mapping[str, Any]) -> bool:
    return (
        value["mapping_policy_sha256"] is not None
        and value["metric_policy_sha256"] is not None
        and not value["unresolved_mapping_policies"]
        and not value["unresolved_metric_policies"]
    )


def candidate_attempt_stream_sha256(
    attempts: Sequence[CandidateAuditAttempt],
) -> str:
    """Hash ordered upstream attempt identities, independent of outcomes."""

    stream: list[dict[str, Any]] = []
    for expected_index, attempt in enumerate(attempts):
        if not isinstance(attempt, CandidateAuditAttempt):
            raise EditingCandidateAuditLedgerError(
                "attempt stream must contain CandidateAuditAttempt values"
            )
        if attempt.attempt_index != expected_index:
            raise EditingCandidateAuditLedgerError(
                "candidate attempt indexes must be contiguous from zero"
            )
        stream.append(
            {
                "attempt_index": expected_index,
                "candidate_id": _require_text(
                    attempt.candidate_id,
                    field="candidate_id",
                ),
                "candidate_payload_sha256": _require_sha256(
                    attempt.candidate_payload_sha256,
                    field="candidate_payload_sha256",
                ),
            }
        )
    return canonical_sha256(stream)


def _build_candidate_row(
    attempt: CandidateAuditAttempt,
    *,
    contract: Mapping[str, Any],
    contract_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    compiler_identity: Mapping[str, Any],
    lane_ids: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(attempt, CandidateAuditAttempt):
        raise EditingCandidateAuditLedgerError(
            "candidate attempts must be typed CandidateAuditAttempt values"
        )
    attempt_index = _require_nonnegative_int(
        attempt.attempt_index,
        field="attempt_index",
    )
    candidate_id = _require_text(attempt.candidate_id, field="candidate_id")
    source_record_ids = _normalized_unique_text(
        attempt.source_record_ids,
        field="source_record_ids",
        allow_empty=False,
    )
    evidence = _evidence_bundle(attempt.evidence)
    resolution = _lane_resolution(attempt.lane_resolution, lane_ids=lane_ids)
    policies = _policy_bindings(attempt.policy_bindings)

    compiled = attempt.accepted_trace_id is not None
    rejected = attempt.rejection_code is not None or attempt.rejection_detail is not None
    if compiled == rejected:
        raise EditingCandidateAuditLedgerError(
            "each candidate must have exactly one compiled-trace or rejection disposition"
        )
    if compiled:
        accepted_trace_id = _require_text(
            attempt.accepted_trace_id,
            field="accepted_trace_id",
        )
        if evidence is None or resolution is None:
            raise EditingCandidateAuditLedgerError(
                "compiled candidate requires a lane resolution and six-component evidence"
            )
        if not _policies_resolved(policies):
            raise EditingCandidateAuditLedgerError(
                "unresolved mapping or metric policy cannot grant compilation disposition"
            )
        missing_references = [
            component
            for component, component_payload in evidence.items()
            if not component_payload["reference_ids"]
        ]
        if missing_references:
            raise EditingCandidateAuditLedgerError(
                "compiled candidate evidence components require provenance references: "
                f"{missing_references}"
            )
        evidence_components = {
            component: evidence[component]["kind"] for component in EVIDENCE_COMPONENTS
        }
        try:
            validate_evidence_assignment_envelope(
                contract,
                data_lane=resolution["data_lane"],
                evidence_profile_id=resolution["evidence_profile_id"],
                evidence_components=evidence_components,
            )
        except ValueError as error:
            raise EditingCandidateAuditLedgerError(
                f"compiled candidate evidence assignment is invalid: {error}"
            ) from error
        data_lane = resolution["data_lane"]
        evidence_profile_id = resolution["evidence_profile_id"]
        disposition = "compiled_candidate"
        rejection = None
    else:
        accepted_trace_id = None
        code = _require_text(attempt.rejection_code, field="rejection_code")
        if _REJECTION_CODE.fullmatch(code) is None:
            raise EditingCandidateAuditLedgerError(
                "rejection_code must be a namespaced lowercase reason"
            )
        detail = _require_text(
            attempt.rejection_detail,
            field="rejection_detail",
        )
        unresolved = (
            policies["unresolved_mapping_policies"] or policies["unresolved_metric_policies"]
        )
        if unresolved and not code.startswith("policy."):
            raise EditingCandidateAuditLedgerError(
                "unresolved policy rejection must use a policy.* reason code"
            )
        data_lane = None if resolution is None else resolution["data_lane"]
        evidence_profile_id = None if resolution is None else resolution["evidence_profile_id"]
        disposition = "rejected_candidate"
        rejection = {"code": code, "detail": detail}

    body = {
        "schema": CANDIDATE_AUDIT_ROW_SCHEMA,
        "schema_version": CANDIDATE_AUDIT_ROW_SCHEMA_VERSION,
        "status": CANDIDATE_AUDIT_ROW_STATUS,
        "training_authorized": False,
        "attempt_index": attempt_index,
        "candidate_id": candidate_id,
        "candidate_payload_sha256": _require_sha256(
            attempt.candidate_payload_sha256,
            field="candidate_payload_sha256",
        ),
        "contract_identity": _json_copy(
            contract_identity,
            field="contract_identity",
        ),
        "source_identity": _json_copy(
            source_identity,
            field="source_identity",
        ),
        "compiler_identity": _json_copy(
            compiler_identity,
            field="compiler_identity",
        ),
        "source_record_ids": list(source_record_ids),
        "data_lane": data_lane,
        "evidence_profile_id": evidence_profile_id,
        "evidence": evidence,
        "lane_resolution": resolution,
        "policy_bindings": policies,
        "disposition": disposition,
        "accepted_trace_id": accepted_trace_id,
        "rejection": rejection,
    }
    return {**body, "row_sha256": canonical_sha256(body)}


def _attempt_from_row(payload: Mapping[str, Any]) -> CandidateAuditAttempt:
    rejection = payload["rejection"]
    resolution = payload["lane_resolution"]
    evidence = payload["evidence"]
    return CandidateAuditAttempt(
        attempt_index=payload["attempt_index"],
        candidate_id=payload["candidate_id"],
        candidate_payload_sha256=payload["candidate_payload_sha256"],
        source_record_ids=tuple(payload["source_record_ids"]),
        policy_bindings=CandidatePolicyBindings(
            mapping_policy_sha256=payload["policy_bindings"]["mapping_policy_sha256"],
            metric_policy_sha256=payload["policy_bindings"]["metric_policy_sha256"],
            unresolved_mapping_policies=tuple(
                payload["policy_bindings"]["unresolved_mapping_policies"]
            ),
            unresolved_metric_policies=tuple(
                payload["policy_bindings"]["unresolved_metric_policies"]
            ),
        ),
        lane_resolution=(None if resolution is None else CandidateLaneResolution(**resolution)),
        evidence=(
            None
            if evidence is None
            else CandidateEvidence(
                **{
                    component: CandidateEvidenceComponent(
                        kind=evidence[component]["kind"],
                        reference_ids=tuple(evidence[component]["reference_ids"]),
                        evidence_sha256=evidence[component]["evidence_sha256"],
                    )
                    for component in EVIDENCE_COMPONENTS
                }
            )
        ),
        accepted_trace_id=payload["accepted_trace_id"],
        rejection_code=None if rejection is None else rejection["code"],
        rejection_detail=None if rejection is None else rejection["detail"],
    )


def _validate_candidate_row(
    row: object,
    *,
    expected_index: int,
    contract: Mapping[str, Any],
    contract_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    compiler_identity: Mapping[str, Any],
    lane_ids: tuple[str, ...],
) -> dict[str, Any]:
    payload = _require_exact_fields(
        row,
        _ROW_FIELDS,
        field=f"candidate ledger row[{expected_index}]",
    )
    if (
        payload["schema"] != CANDIDATE_AUDIT_ROW_SCHEMA
        or payload["schema_version"] != CANDIDATE_AUDIT_ROW_SCHEMA_VERSION
        or payload["status"] != CANDIDATE_AUDIT_ROW_STATUS
        or payload["training_authorized"] is not False
        or payload["attempt_index"] != expected_index
    ):
        raise EditingCandidateAuditLedgerError(
            f"candidate ledger row[{expected_index}] identity or authority is invalid"
        )
    if (
        payload["contract_identity"] != contract_identity
        or payload["source_identity"] != source_identity
        or payload["compiler_identity"] != compiler_identity
    ):
        raise EditingCandidateAuditLedgerError(
            f"candidate ledger row[{expected_index}] provenance disagrees"
        )
    if payload["disposition"] == "compiled_candidate":
        if payload["rejection"] is not None:
            raise EditingCandidateAuditLedgerError(
                "compiled candidate row cannot carry a rejection"
            )
    elif payload["disposition"] == "rejected_candidate":
        _require_exact_fields(
            payload["rejection"],
            _REJECTION_FIELDS,
            field=f"candidate ledger row[{expected_index}].rejection",
        )
    else:
        raise EditingCandidateAuditLedgerError(
            f"candidate ledger row[{expected_index}] has invalid disposition"
        )
    normalized = _build_candidate_row(
        _attempt_from_row(payload),
        contract=contract,
        contract_identity=contract_identity,
        source_identity=source_identity,
        compiler_identity=compiler_identity,
        lane_ids=lane_ids,
    )
    if dict(payload) != normalized:
        raise EditingCandidateAuditLedgerError(
            f"candidate ledger row[{expected_index}] is not canonical or its hash disagrees"
        )
    return normalized


def _ledger_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    lane_ids: tuple[str, ...],
) -> dict[str, Any]:
    dispositions = Counter(str(row["disposition"]) for row in rows)
    rejection_codes = Counter(
        str(row["rejection"]["code"]) for row in rows if row["rejection"] is not None
    )
    return {
        "attempted": len(rows),
        "compiled_candidates": dispositions["compiled_candidate"],
        "rejected_candidates": dispositions["rejected_candidate"],
        "without_lane_resolution": sum(row["lane_resolution"] is None for row in rows),
        "by_lane": {lane: sum(row["data_lane"] == lane for row in rows) for lane in lane_ids},
        "by_rejection_code": {code: rejection_codes[code] for code in sorted(rejection_codes)},
    }


def build_candidate_audit_ledger(
    *,
    contract: Mapping[str, Any],
    contract_file_sha256: str,
    source_identity: CandidateSourceIdentity,
    compiler_identity: CandidateCompilerIdentity,
    attempts: Sequence[CandidateAuditAttempt],
    expected_attempt_count: int,
    expected_attempt_stream_sha256: str,
) -> dict[str, Any]:
    """Build a complete, deterministic, non-authorizing candidate ledger."""

    contract_identity, lane_ids = _contract_context(
        contract,
        contract_file_sha256=contract_file_sha256,
    )
    normalized_source = _source_identity(source_identity)
    normalized_compiler = _compiler_identity(compiler_identity)
    expected_count = _require_nonnegative_int(
        expected_attempt_count,
        field="expected_attempt_count",
    )
    if expected_count <= 0:
        raise EditingCandidateAuditLedgerError(
            "candidate ledger must preserve at least one attempted candidate"
        )
    if len(attempts) != expected_count:
        raise EditingCandidateAuditLedgerError(
            "candidate ledger row count disagrees with the source attempt census"
        )
    observed_stream = candidate_attempt_stream_sha256(attempts)
    expected_stream = _require_sha256(
        expected_attempt_stream_sha256,
        field="expected_attempt_stream_sha256",
    )
    if observed_stream != expected_stream:
        raise EditingCandidateAuditLedgerError(
            "candidate ledger attempt stream disagrees with the frozen source stream"
        )
    rows = [
        _build_candidate_row(
            attempt,
            contract=contract,
            contract_identity=contract_identity,
            source_identity=normalized_source,
            compiler_identity=normalized_compiler,
            lane_ids=lane_ids,
        )
        for attempt in attempts
    ]
    candidate_ids = [str(row["candidate_id"]) for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise EditingCandidateAuditLedgerError("candidate ledger repeats a candidate_id")
    accepted_trace_ids = [
        str(row["accepted_trace_id"]) for row in rows if row["accepted_trace_id"] is not None
    ]
    if len(accepted_trace_ids) != len(set(accepted_trace_ids)):
        raise EditingCandidateAuditLedgerError("candidate ledger repeats a compiled trace_id")
    rows_sha256 = canonical_sha256(rows)
    body = {
        "schema": CANDIDATE_AUDIT_LEDGER_SCHEMA,
        "schema_version": CANDIDATE_AUDIT_LEDGER_SCHEMA_VERSION,
        "status": CANDIDATE_AUDIT_LEDGER_STATUS,
        "training_authorized": False,
        "blockers": _launch_blockers(contract),
        "contract_identity": contract_identity,
        "source_identity": normalized_source,
        "compiler_identity": normalized_compiler,
        "lane_ids": list(lane_ids),
        "expected_attempt_count": expected_count,
        "expected_attempt_stream_sha256": expected_stream,
        "counts": _ledger_counts(rows, lane_ids=lane_ids),
        "rows": rows,
        "rows_sha256": rows_sha256,
    }
    ledger = {**body, "ledger_sha256": canonical_sha256(body)}
    validate_candidate_audit_ledger(
        ledger,
        contract=contract,
        contract_file_sha256=contract_file_sha256,
        expected_source_identity=source_identity,
        expected_compiler_identity=compiler_identity,
        expected_attempt_count=expected_count,
        expected_attempt_stream_sha256=expected_stream,
    )
    return ledger


def validate_candidate_audit_ledger(
    ledger: object,
    *,
    contract: Mapping[str, Any],
    contract_file_sha256: str,
    expected_source_identity: CandidateSourceIdentity | Mapping[str, Any],
    expected_compiler_identity: CandidateCompilerIdentity | Mapping[str, Any],
    expected_attempt_count: int,
    expected_attempt_stream_sha256: str,
) -> Mapping[str, Any]:
    """Validate completeness, provenance, disposition, and content hashes."""

    payload = _require_exact_fields(
        ledger,
        _LEDGER_FIELDS,
        field="candidate audit ledger",
    )
    contract_identity, lane_ids = _contract_context(
        contract,
        contract_file_sha256=contract_file_sha256,
    )
    source_identity = _source_identity(expected_source_identity)
    compiler_identity = _compiler_identity(expected_compiler_identity)
    expected_count = _require_nonnegative_int(
        expected_attempt_count,
        field="expected_attempt_count",
    )
    expected_stream = _require_sha256(
        expected_attempt_stream_sha256,
        field="expected_attempt_stream_sha256",
    )
    if (
        payload["schema"] != CANDIDATE_AUDIT_LEDGER_SCHEMA
        or payload["schema_version"] != CANDIDATE_AUDIT_LEDGER_SCHEMA_VERSION
        or payload["status"] != CANDIDATE_AUDIT_LEDGER_STATUS
        or payload["training_authorized"] is not False
    ):
        raise EditingCandidateAuditLedgerError(
            "candidate audit ledger identity or authority is invalid"
        )
    if (
        payload["blockers"] != _launch_blockers(contract)
        or payload["contract_identity"] != contract_identity
        or payload["source_identity"] != source_identity
        or payload["compiler_identity"] != compiler_identity
        or tuple(payload["lane_ids"]) != lane_ids
        or payload["expected_attempt_count"] != expected_count
        or payload["expected_attempt_stream_sha256"] != expected_stream
    ):
        raise EditingCandidateAuditLedgerError(
            "candidate audit ledger provenance or blocker identity disagrees"
        )
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, list) or len(raw_rows) != expected_count:
        raise EditingCandidateAuditLedgerError(
            "candidate audit ledger does not preserve every attempted candidate"
        )
    rows = [
        _validate_candidate_row(
            row,
            expected_index=index,
            contract=contract,
            contract_identity=contract_identity,
            source_identity=source_identity,
            compiler_identity=compiler_identity,
            lane_ids=lane_ids,
        )
        for index, row in enumerate(raw_rows)
    ]
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise EditingCandidateAuditLedgerError("candidate audit ledger repeats a candidate_id")
    accepted_ids = [
        row["accepted_trace_id"] for row in rows if row["accepted_trace_id"] is not None
    ]
    if len(accepted_ids) != len(set(accepted_ids)):
        raise EditingCandidateAuditLedgerError("candidate audit ledger repeats a compiled trace_id")
    observed_stream = canonical_sha256(
        [
            {
                "attempt_index": row["attempt_index"],
                "candidate_id": row["candidate_id"],
                "candidate_payload_sha256": row["candidate_payload_sha256"],
            }
            for row in rows
        ]
    )
    if observed_stream != expected_stream:
        raise EditingCandidateAuditLedgerError(
            "candidate audit ledger rows disagree with the frozen attempt stream"
        )
    if payload["rows_sha256"] != canonical_sha256(rows):
        raise EditingCandidateAuditLedgerError("candidate audit ledger row-content hash disagrees")
    if payload["counts"] != _ledger_counts(rows, lane_ids=lane_ids):
        raise EditingCandidateAuditLedgerError(
            "candidate audit ledger counts disagree with its rows"
        )
    body = {key: value for key, value in payload.items() if key != "ledger_sha256"}
    if payload["ledger_sha256"] != canonical_sha256(body):
        raise EditingCandidateAuditLedgerError("candidate audit ledger self-hash disagrees")
    return MappingProxyType(dict(payload))


__all__ = [
    "CANDIDATE_AUDIT_LEDGER_SCHEMA",
    "CANDIDATE_AUDIT_LEDGER_SCHEMA_VERSION",
    "CANDIDATE_AUDIT_LEDGER_STATUS",
    "CANDIDATE_AUDIT_ROW_SCHEMA",
    "CANDIDATE_AUDIT_ROW_SCHEMA_VERSION",
    "CANDIDATE_AUDIT_ROW_STATUS",
    "EVIDENCE_COMPONENTS",
    "CandidateAuditAttempt",
    "CandidateCompilerIdentity",
    "CandidateEvidence",
    "CandidateEvidenceComponent",
    "CandidateLaneResolution",
    "CandidatePolicyBindings",
    "CandidateSourceIdentity",
    "EditingCandidateAuditLedgerError",
    "build_candidate_audit_ledger",
    "candidate_attempt_stream_sha256",
    "canonical_json_bytes",
    "canonical_sha256",
    "validate_candidate_audit_ledger",
]
