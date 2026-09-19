"""Post-Active8 semantic and sampling metadata for editing schema version 2.

The sidecar is metadata-only.  It binds each progress row to a compiled
candidate, one exact assigned split resolution, one exact admitted-record
envelope identity, and the physical/semantic parent inventories.  It never
stores molecular states or actions.

The sidecar remains non-authorizing.  Record-membership labels are structurally
validated but not recomputed from chemistry, and relationship-group identifiers
are not treated as physically resolved receipts.  Both limitations are copied
from the corpus contract into explicit NO_GO blockers.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from compose_v4.data.editing_candidate_audit_ledger import (
    CANDIDATE_AUDIT_LEDGER_ARTIFACT_SCHEMA_VERSION,
    CandidateCompilerIdentity,
    CandidateSourceIdentity,
    EditingCandidateAuditLedgerError,
    canonical_sha256,
    iter_candidate_audit_rows,
    validate_candidate_audit_ledger,
    validate_candidate_audit_ledger_artifact,
)
from compose_v4.data.editing_corpus_contract import (
    ACTIVE8_FAMILIES,
    REQUIRED_DATA_LANES,
    REQUIRED_PARTITION_ROLES,
    validate_editing_corpus_contract,
    validate_partition_resolution,
    validate_record_membership_assignment,
)
from compose_v4.data.editing_v2_lane_registry import (
    LANE_REGISTRY_SCHEMA,
    LANE_REGISTRY_SCHEMA_VERSION,
    LANE_REGISTRY_STATUS,
)
from compose_v4.data.packed_trace_store import PackedTraceAddress

SEMANTIC_SAMPLING_ROW_SCHEMA = "compose.editing_v2_semantic_sampling_progress"
SEMANTIC_SAMPLING_ROW_SCHEMA_VERSION = 2
SEMANTIC_SAMPLING_ROW_STATUS = "POST_ACTIVE8_METADATA_NO_TRAINING_AUTHORITY"
SEMANTIC_SAMPLING_SIDECAR_SCHEMA = "compose.editing_v2_semantic_sampling_sidecar"
SEMANTIC_SAMPLING_SIDECAR_SCHEMA_VERSION = 2
SEMANTIC_SAMPLING_SIDECAR_STATUS = "COMPLETE_NO_TRAINING_AUTHORITY"

_HEX64 = frozenset("0123456789abcdef")
_ROW_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "contract_identity",
    "lane_registry_binding",
    "packed_corpus_identity",
    "active8_identity",
    "split_assignment_identity",
    "candidate_ledger_binding",
    "packed_address",
    "record_key_sha256",
    "partition_resolution",
    "admitted_record_envelope_sha256",
    "progress_key",
    "progress_key_sha256",
    "groups",
    "record_membership_cells",
    "record_membership_classifier_identity_sha256",
    "endpoint_descriptor",
    "path_descriptor",
    "relationship_group_ids",
    "mappings",
    "sampling_coefficients",
    "row_sha256",
}
_MANIFEST_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "blockers",
    "contract_identity",
    "lane_registry_identity",
    "packed_corpus_identity",
    "active8_identity",
    "split_assignment_identity",
    "candidate_ledger_identity",
    "partition_roles",
    "lane_ids",
    "counts",
    "rows",
    "rows_sha256",
    "manifest_sha256",
}
_CONTRACT_IDENTITY_FIELDS = {
    "schema",
    "schema_version",
    "contract_id",
    "file_sha256",
    "semantic_sha256",
}
_LANE_REGISTRY_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "registry_file_sha256",
    "registry_sha256",
    "lane_completion_sha256",
    "lane_shard_inventory_sha256",
}
_PACKED_CORPUS_FIELDS = {
    "source_manifest_file_sha256",
    "source_manifest_sha256",
    "unified_packed_manifest_file_sha256",
    "unified_packed_manifest_sha256",
    "packing_implementation_sha256",
    "admitted_record_envelope_inventory_sha256",
}
_ACTIVE8_FIELDS = {
    "inventory_manifest_file_sha256",
    "inventory_sha256",
    "effective_source_corpus_cache_sha256",
    "unified_packed_manifest_sha256",
    "support_contract_sha256",
    "admitted_trace_count",
    "admitted_trace_keys_sha256",
}
_SPLIT_ASSIGNMENT_FIELDS = {
    "manifest_file_sha256",
    "manifest_sha256",
    "resolution_stream_sha256",
}
_CANDIDATE_LEDGER_IDENTITY_FIELDS = {
    "manifest_file_sha256",
    "ledger_sha256",
    "rows_sha256",
}
_CANDIDATE_LEDGER_ROW_BINDING_FIELDS = {
    *_CANDIDATE_LEDGER_IDENTITY_FIELDS,
    "candidate_id",
    "candidate_row_sha256",
}
_PACKED_ADDRESS_FIELDS = {
    "packed_shard_content_sha256",
    "packed_shard_name",
    "entry_index",
    "trace_id",
    "lane",
    "partition",
    "source_key",
    "target_key",
    "path_length",
}
_PROGRESS_KEY_FIELDS = {
    "packed_shard_content_sha256",
    "entry_index",
    "trace_id",
    "progress_index",
}
_GROUP_FIELDS = {
    "source_group_id",
    "scaffold_group_id",
    "series_group_id",
    "document_or_source_group_id",
    "transformation_signature",
    "series_scaffold_or_source_group_id",
}
_ENDPOINT_DESCRIPTOR_FIELDS = {
    "source_canonical_key",
    "target_canonical_key",
    "atom_count_delta",
    "graph_cycle_rank_delta",
    "source_endpoint_evidence_kind",
    "target_endpoint_evidence_kind",
    "pair_relationship_evidence_kind",
}
_PATH_DESCRIPTOR_FIELDS = {
    "path_length",
    "operator_family_set",
    "path_evidence_kind",
    "intermediate_evidence_kind",
    "action_sequence_evidence_kind",
    "maximum_intermediate_source_distance",
    "immediate_reversal_count",
    "repeated_state_count",
    "metric_policy_sha256",
    "unresolved_metric_policies",
}
_MAPPING_FIELDS = {
    "constant_core_mapping",
    "protected_mapping",
    "mapping_policy_sha256",
    "unresolved_mapping_policies",
}
_COEFFICIENT_FIELDS = {
    "path_sampling_coefficient",
    "progress_sampling_coefficient",
    "effective_teacher_coefficient",
    "coefficient_policy_sha256",
    "unresolved_coefficient_policies",
}


class EditingSemanticSamplingSidecarError(ValueError):
    """The semantic/sampling sidecar is incomplete or provenance-inconsistent."""


@dataclass(frozen=True)
class LaneRegistryIdentity:
    """Exact five-lane registry identities resolved before sidecar creation."""

    registry_file_sha256: str
    registry_sha256: str
    lane_completion_sha256: tuple[tuple[str, str], ...]
    lane_shard_inventory_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PackedCorpusIdentity:
    """Physical/semantic packed-corpus and admitted-envelope identities."""

    source_manifest_file_sha256: str
    source_manifest_sha256: str
    unified_packed_manifest_file_sha256: str
    unified_packed_manifest_sha256: str
    packing_implementation_sha256: str
    admitted_record_envelope_inventory_sha256: str


@dataclass(frozen=True)
class Active8SidecarIdentity:
    """Active8 inventory and exact admitted-trace stream identity."""

    inventory_manifest_file_sha256: str
    inventory_sha256: str
    effective_source_corpus_cache_sha256: str
    unified_packed_manifest_sha256: str
    support_contract_sha256: str
    admitted_trace_count: int
    admitted_trace_keys_sha256: str


@dataclass(frozen=True)
class SplitAssignmentIdentity:
    """Physical, semantic, and complete resolution-stream split identities."""

    manifest_file_sha256: str
    manifest_sha256: str
    resolution_stream_sha256: str


@dataclass(frozen=True)
class SemanticGroups:
    """Leakage and hierarchical-sampling group identities."""

    source_group_id: str
    scaffold_group_id: str
    series_group_id: str | None
    document_or_source_group_id: str
    transformation_signature: str | None


@dataclass(frozen=True)
class EndpointDescriptor:
    """Endpoint identities and graph-level deltas, never exact state tensors."""

    source_canonical_key: str
    target_canonical_key: str
    atom_count_delta: int
    graph_cycle_rank_delta: int
    source_endpoint_evidence_kind: str
    target_endpoint_evidence_kind: str
    pair_relationship_evidence_kind: str


@dataclass(frozen=True)
class PathDescriptor:
    """Path-level semantic descriptors under one frozen metric policy."""

    path_length: int
    operator_family_set: tuple[str, ...]
    path_evidence_kind: str
    intermediate_evidence_kind: str
    action_sequence_evidence_kind: str
    maximum_intermediate_source_distance: float
    immediate_reversal_count: int
    repeated_state_count: int
    metric_policy_sha256: str | None
    unresolved_metric_policies: tuple[str, ...] = ()


@dataclass(frozen=True)
class MappingDescriptor:
    """Slot mappings interpreted under one frozen mapping policy."""

    constant_core_mapping: tuple[tuple[int, int], ...] | None
    protected_mapping: tuple[tuple[int, int], ...] | None
    mapping_policy_sha256: str | None
    unresolved_mapping_policies: tuple[str, ...] = ()


@dataclass(frozen=True)
class SamplingCoefficients:
    """Explicit path/progress coefficients and effective teacher mass."""

    path_sampling_coefficient: float
    progress_sampling_coefficient: float
    effective_teacher_coefficient: float
    coefficient_policy_sha256: str
    unresolved_coefficient_policies: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticSamplingProgress:
    """One post-Active8 progress-addressed semantic/sampling row."""

    address: PackedTraceAddress
    progress_index: int
    candidate_ledger_row_sha256: str
    partition_resolution: Mapping[str, Any]
    admitted_record_envelope_sha256: str
    groups: SemanticGroups
    record_membership_cells: tuple[str, ...]
    record_membership_classifier_identity_sha256: str
    endpoint_descriptor: EndpointDescriptor
    path_descriptor: PathDescriptor
    relationship_group_ids: tuple[str, ...]
    mappings: MappingDescriptor
    sampling_coefficients: SamplingCoefficients


def _require_exact_fields(
    value: object,
    expected: set[str],
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EditingSemanticSamplingSidecarError(f"{field} must be an object")
    actual = set(value)
    if actual != expected:
        raise EditingSemanticSamplingSidecarError(
            f"{field} fields disagree; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return value


def _require_sha256(value: object, *, field: str) -> str:
    digest = value if isinstance(value, str) else ""
    if len(digest) != 64 or any(character not in _HEX64 for character in digest):
        raise EditingSemanticSamplingSidecarError(f"{field} must be a full lowercase SHA-256")
    return digest


def _require_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise EditingSemanticSamplingSidecarError(f"{field} must be nonempty normalized text")
    return value


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field=field)


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise EditingSemanticSamplingSidecarError(f"{field} must be a nonnegative integer")
    return value


def _require_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise EditingSemanticSamplingSidecarError(f"{field} must be an integer")
    return value


def _require_nonnegative_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EditingSemanticSamplingSidecarError(f"{field} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise EditingSemanticSamplingSidecarError(f"{field} must be a finite nonnegative number")
    return result


def _require_positive_float(value: object, *, field: str) -> float:
    result = _require_nonnegative_float(value, field=field)
    if result <= 0.0:
        raise EditingSemanticSamplingSidecarError(f"{field} must be strictly positive")
    return result


def _unique_sorted_text(
    values: object,
    *,
    field: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise EditingSemanticSamplingSidecarError(f"{field} must be a sequence of strings")
    normalized = tuple(
        _require_text(value, field=f"{field}[{index}]") for index, value in enumerate(values)
    )
    if len(normalized) != len(set(normalized)):
        raise EditingSemanticSamplingSidecarError(f"{field} contains duplicates")
    if not allow_empty and not normalized:
        raise EditingSemanticSamplingSidecarError(f"{field} must not be empty")
    if normalized != tuple(sorted(normalized)):
        raise EditingSemanticSamplingSidecarError(f"{field} must be deterministically sorted")
    return normalized


def _contract_context(
    contract: Mapping[str, Any],
    *,
    contract_file_sha256: str,
) -> tuple[
    dict[str, Any],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    try:
        validate_editing_corpus_contract(dict(contract))
    except (TypeError, ValueError) as error:
        raise EditingSemanticSamplingSidecarError(
            f"editing-corpus contract is invalid: {error}"
        ) from error
    partition_roles = tuple(contract["split_contract"]["partition_roles"])
    lane_ids = tuple(str(lane["id"]) for lane in contract["data_lanes"])
    operator_order = tuple(contract["operator_basis"]["active8"])
    membership_order = tuple(cell["id"] for cell in contract["record_membership_contract"]["cells"])
    if partition_roles != REQUIRED_PARTITION_ROLES:
        raise EditingSemanticSamplingSidecarError(
            "sidecar requires the exact four editing-V2 partition roles"
        )
    if lane_ids != REQUIRED_DATA_LANES:
        raise EditingSemanticSamplingSidecarError(
            "sidecar requires the exact five editing-V2 lanes"
        )
    if operator_order != ACTIVE8_FAMILIES:
        raise EditingSemanticSamplingSidecarError(
            "sidecar requires the exact Active8 operator order"
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
    return identity, partition_roles, lane_ids, operator_order, membership_order


def _launch_blockers(contract: Mapping[str, Any]) -> list[str]:
    return [
        f"external_authority.{requirement['id']}:{requirement['status']}"
        for requirement in contract["admitted_record_contract"]["external_authority_requirements"]
    ]


def _ordered_sha_mapping(
    values: object,
    *,
    lane_ids: tuple[str, ...],
    field: str,
) -> dict[str, str]:
    if isinstance(values, Mapping):
        payload = dict(values)
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        try:
            payload = dict(values)
        except (TypeError, ValueError) as error:
            raise EditingSemanticSamplingSidecarError(
                f"{field} must contain lane/hash pairs"
            ) from error
    else:
        raise EditingSemanticSamplingSidecarError(f"{field} must contain lane/hash pairs")
    if set(payload) != set(lane_ids):
        raise EditingSemanticSamplingSidecarError(
            f"{field} must exactly cover the five-lane contract"
        )
    return {lane: _require_sha256(payload[lane], field=f"{field}.{lane}") for lane in lane_ids}


def _lane_registry_identity(
    value: LaneRegistryIdentity | Mapping[str, Any],
    *,
    lane_ids: tuple[str, ...],
) -> dict[str, Any]:
    if isinstance(value, LaneRegistryIdentity):
        payload: Mapping[str, Any] = {
            "schema": LANE_REGISTRY_SCHEMA,
            "schema_version": LANE_REGISTRY_SCHEMA_VERSION,
            "status": LANE_REGISTRY_STATUS,
            "training_authorized": False,
            "registry_file_sha256": value.registry_file_sha256,
            "registry_sha256": value.registry_sha256,
            "lane_completion_sha256": dict(value.lane_completion_sha256),
            "lane_shard_inventory_sha256": dict(value.lane_shard_inventory_sha256),
        }
    else:
        payload = _require_exact_fields(
            value,
            _LANE_REGISTRY_FIELDS,
            field="lane registry identity",
        )
    if (
        payload["schema"] != LANE_REGISTRY_SCHEMA
        or payload["schema_version"] != LANE_REGISTRY_SCHEMA_VERSION
        or payload["status"] != LANE_REGISTRY_STATUS
        or payload["training_authorized"] is not False
    ):
        raise EditingSemanticSamplingSidecarError(
            "lane registry identity is not a frozen non-authorizing registry"
        )
    return {
        "schema": LANE_REGISTRY_SCHEMA,
        "schema_version": LANE_REGISTRY_SCHEMA_VERSION,
        "status": LANE_REGISTRY_STATUS,
        "training_authorized": False,
        "registry_file_sha256": _require_sha256(
            payload["registry_file_sha256"],
            field="lane_registry.registry_file_sha256",
        ),
        "registry_sha256": _require_sha256(
            payload["registry_sha256"],
            field="lane_registry.registry_sha256",
        ),
        "lane_completion_sha256": _ordered_sha_mapping(
            payload["lane_completion_sha256"],
            lane_ids=lane_ids,
            field="lane_registry.lane_completion_sha256",
        ),
        "lane_shard_inventory_sha256": _ordered_sha_mapping(
            payload["lane_shard_inventory_sha256"],
            lane_ids=lane_ids,
            field="lane_registry.lane_shard_inventory_sha256",
        ),
    }


def _packed_corpus_identity(
    value: PackedCorpusIdentity | Mapping[str, Any],
) -> dict[str, str]:
    payload = (
        {field: getattr(value, field) for field in _PACKED_CORPUS_FIELDS}
        if isinstance(value, PackedCorpusIdentity)
        else dict(
            _require_exact_fields(
                value,
                _PACKED_CORPUS_FIELDS,
                field="packed corpus identity",
            )
        )
    )
    return {
        field: _require_sha256(payload[field], field=f"packed_corpus.{field}")
        for field in sorted(_PACKED_CORPUS_FIELDS)
    }


def _active8_identity(
    value: Active8SidecarIdentity | Mapping[str, Any],
) -> dict[str, Any]:
    payload = (
        {field: getattr(value, field) for field in _ACTIVE8_FIELDS}
        if isinstance(value, Active8SidecarIdentity)
        else dict(
            _require_exact_fields(
                value,
                _ACTIVE8_FIELDS,
                field="Active8 sidecar identity",
            )
        )
    )
    return {
        **{
            field: _require_sha256(payload[field], field=f"active8.{field}")
            for field in sorted(_ACTIVE8_FIELDS - {"admitted_trace_count"})
        },
        "admitted_trace_count": _require_nonnegative_int(
            payload["admitted_trace_count"],
            field="active8.admitted_trace_count",
        ),
    }


def _split_assignment_identity(
    value: SplitAssignmentIdentity | Mapping[str, Any],
) -> dict[str, str]:
    payload = (
        {field: getattr(value, field) for field in _SPLIT_ASSIGNMENT_FIELDS}
        if isinstance(value, SplitAssignmentIdentity)
        else dict(
            _require_exact_fields(
                value,
                _SPLIT_ASSIGNMENT_FIELDS,
                field="split assignment identity",
            )
        )
    )
    return {
        field: _require_sha256(
            payload[field],
            field=f"split_assignment.{field}",
        )
        for field in sorted(_SPLIT_ASSIGNMENT_FIELDS)
    }


def _candidate_ledger_identity(
    ledger: Mapping[str, Any],
    *,
    manifest_file_sha256: str,
) -> dict[str, str]:
    return {
        "manifest_file_sha256": _require_sha256(
            manifest_file_sha256,
            field="candidate ledger manifest file SHA-256",
        ),
        "ledger_sha256": _require_sha256(
            ledger["ledger_sha256"],
            field="candidate ledger semantic SHA-256",
        ),
        "rows_sha256": _require_sha256(
            ledger["rows_sha256"],
            field="candidate ledger rows SHA-256",
        ),
    }


def _trace_key_tuple(value: object) -> tuple[str, int, str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise EditingSemanticSamplingSidecarError(
            "Active8 admitted trace key must be (shard_sha256, entry_index, trace_id)"
        )
    return (
        _require_sha256(value[0], field="admitted trace shard SHA-256"),
        _require_nonnegative_int(
            value[1],
            field="admitted trace entry_index",
        ),
        _require_text(value[2], field="admitted trace_id"),
    )


def normalize_admitted_trace_keys(
    keys: Sequence[tuple[str, int, str]],
) -> tuple[tuple[str, int, str], ...]:
    """Return unique exact Active8 trace keys in canonical order."""

    normalized = tuple(sorted(_trace_key_tuple(key) for key in keys))
    if len(normalized) != len(set(normalized)):
        raise EditingSemanticSamplingSidecarError(
            "Active8 admitted trace stream repeats an exact trace key"
        )
    return normalized


def active8_admitted_trace_keys_sha256(
    keys: Sequence[tuple[str, int, str]],
) -> str:
    """Hash the exact post-Active8 trace set consumed by this sidecar."""

    return canonical_sha256([list(key) for key in normalize_admitted_trace_keys(keys)])


def _packed_address(address: PackedTraceAddress) -> dict[str, Any]:
    if not isinstance(address, PackedTraceAddress):
        raise EditingSemanticSamplingSidecarError(
            "sidecar rows require a typed immutable PackedTraceAddress"
        )
    return {
        "packed_shard_content_sha256": _require_sha256(
            address.packed_shard_content_sha256,
            field="packed address shard SHA-256",
        ),
        "packed_shard_name": _require_text(
            address.packed_shard_name,
            field="packed_shard_name",
        ),
        "entry_index": _require_nonnegative_int(
            address.entry_index,
            field="entry_index",
        ),
        "trace_id": _require_text(address.trace_id, field="trace_id"),
        "lane": _require_text(address.layer, field="lane"),
        "partition": _require_text(address.partition, field="partition"),
        "source_key": _require_text(address.source_key, field="source_key"),
        "target_key": _require_text(address.target_key, field="target_key"),
        "path_length": _require_nonnegative_int(
            address.path_length,
            field="path_length",
        ),
    }


def _record_key(address: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "packed_shard_content_sha256": address["packed_shard_content_sha256"],
        "entry_index": address["entry_index"],
        "trace_id": address["trace_id"],
    }


def _record_bindings(
    progress_rows: Sequence[SemanticSamplingProgress],
) -> list[dict[str, Any]]:
    bindings: dict[tuple[str, int, str], dict[str, Any]] = {}
    for progress in progress_rows:
        if not isinstance(progress, SemanticSamplingProgress):
            raise EditingSemanticSamplingSidecarError(
                "record-binding helpers require SemanticSamplingProgress values"
            )
        address = _packed_address(progress.address)
        key = (
            address["packed_shard_content_sha256"],
            address["entry_index"],
            address["trace_id"],
        )
        binding = {
            "record_key": _record_key(address),
            "partition_resolution": dict(progress.partition_resolution),
            "admitted_record_envelope_sha256": _require_sha256(
                progress.admitted_record_envelope_sha256,
                field="admitted_record_envelope_sha256",
            ),
        }
        prior = bindings.setdefault(key, binding)
        if prior != binding:
            raise EditingSemanticSamplingSidecarError(
                f"trace {key!r} changes split or envelope identity across progress"
            )
    return [bindings[key] for key in sorted(bindings)]


def split_resolution_stream_sha256(
    progress_rows: Sequence[SemanticSamplingProgress],
) -> str:
    """Hash every record key and assigned split resolution exactly once."""

    return canonical_sha256(
        [
            {
                "record_key": binding["record_key"],
                "partition_resolution": binding["partition_resolution"],
            }
            for binding in _record_bindings(progress_rows)
        ]
    )


def admitted_record_envelope_inventory_sha256(
    progress_rows: Sequence[SemanticSamplingProgress],
) -> str:
    """Hash every record key and admitted-envelope identity exactly once."""

    return canonical_sha256(
        [
            {
                "record_key": binding["record_key"],
                "admitted_record_envelope_sha256": binding["admitted_record_envelope_sha256"],
            }
            for binding in _record_bindings(progress_rows)
        ]
    )


def _groups(value: SemanticGroups | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, SemanticGroups):
        payload: Mapping[str, Any] = {
            "source_group_id": value.source_group_id,
            "scaffold_group_id": value.scaffold_group_id,
            "series_group_id": value.series_group_id,
            "document_or_source_group_id": value.document_or_source_group_id,
            "transformation_signature": value.transformation_signature,
            "series_scaffold_or_source_group_id": (
                value.series_group_id or value.scaffold_group_id or value.source_group_id
            ),
        }
    else:
        payload = _require_exact_fields(
            value,
            _GROUP_FIELDS,
            field="semantic groups",
        )
    source = _require_text(payload["source_group_id"], field="source_group_id")
    scaffold = _require_text(
        payload["scaffold_group_id"],
        field="scaffold_group_id",
    )
    series = _optional_text(payload["series_group_id"], field="series_group_id")
    expected_hierarchy = series or scaffold or source
    if payload["series_scaffold_or_source_group_id"] != expected_hierarchy:
        raise EditingSemanticSamplingSidecarError(
            "series/scaffold/source sampling group does not follow declared precedence"
        )
    return {
        "source_group_id": source,
        "scaffold_group_id": scaffold,
        "series_group_id": series,
        "document_or_source_group_id": _require_text(
            payload["document_or_source_group_id"],
            field="document_or_source_group_id",
        ),
        "transformation_signature": _optional_text(
            payload["transformation_signature"],
            field="transformation_signature",
        ),
        "series_scaffold_or_source_group_id": expected_hierarchy,
    }


def _endpoint_descriptor(
    value: EndpointDescriptor | Mapping[str, Any],
    *,
    address: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    payload = (
        {field: getattr(value, field) for field in _ENDPOINT_DESCRIPTOR_FIELDS}
        if isinstance(value, EndpointDescriptor)
        else dict(
            _require_exact_fields(
                value,
                _ENDPOINT_DESCRIPTOR_FIELDS,
                field="endpoint descriptor",
            )
        )
    )
    normalized = {
        "source_canonical_key": _require_text(
            payload["source_canonical_key"],
            field="endpoint.source_canonical_key",
        ),
        "target_canonical_key": _require_text(
            payload["target_canonical_key"],
            field="endpoint.target_canonical_key",
        ),
        "atom_count_delta": _require_int(
            payload["atom_count_delta"],
            field="endpoint.atom_count_delta",
        ),
        "graph_cycle_rank_delta": _require_int(
            payload["graph_cycle_rank_delta"],
            field="endpoint.graph_cycle_rank_delta",
        ),
        "source_endpoint_evidence_kind": _require_text(
            payload["source_endpoint_evidence_kind"],
            field="endpoint.source_endpoint_evidence_kind",
        ),
        "target_endpoint_evidence_kind": _require_text(
            payload["target_endpoint_evidence_kind"],
            field="endpoint.target_endpoint_evidence_kind",
        ),
        "pair_relationship_evidence_kind": _require_text(
            payload["pair_relationship_evidence_kind"],
            field="endpoint.pair_relationship_evidence_kind",
        ),
    }
    expected = {
        "source_canonical_key": address["source_key"],
        "target_canonical_key": address["target_key"],
        "source_endpoint_evidence_kind": evidence["source_endpoint"]["kind"],
        "target_endpoint_evidence_kind": evidence["target_endpoint"]["kind"],
        "pair_relationship_evidence_kind": evidence["pair_relationship"]["kind"],
    }
    if any(normalized[field] != expected[field] for field in expected):
        raise EditingSemanticSamplingSidecarError(
            "endpoint descriptor disagrees with packed or candidate-ledger identity"
        )
    return normalized


def _ordered_operator_families(
    values: object,
    *,
    operator_order: tuple[str, ...],
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise EditingSemanticSamplingSidecarError("operator_family_set must be a sequence")
    observed = tuple(values)
    if (
        not observed
        or len(observed) != len(set(observed))
        or any(family not in operator_order for family in observed)
        or observed != tuple(family for family in operator_order if family in observed)
    ):
        raise EditingSemanticSamplingSidecarError(
            "operator_family_set must be a nonempty ordered subset of Active8"
        )
    return observed


def _path_descriptor(
    value: PathDescriptor | Mapping[str, Any],
    *,
    address: Mapping[str, Any],
    evidence: Mapping[str, Any],
    operator_order: tuple[str, ...],
    expected_metric_policy_sha256: str,
) -> dict[str, Any]:
    payload = (
        {
            field: (
                list(getattr(value, field))
                if field in {"operator_family_set", "unresolved_metric_policies"}
                else getattr(value, field)
            )
            for field in _PATH_DESCRIPTOR_FIELDS
        }
        if isinstance(value, PathDescriptor)
        else dict(
            _require_exact_fields(
                value,
                _PATH_DESCRIPTOR_FIELDS,
                field="path descriptor",
            )
        )
    )
    unresolved = _unique_sorted_text(
        payload["unresolved_metric_policies"],
        field="path.unresolved_metric_policies",
        allow_empty=True,
    )
    if unresolved or payload["metric_policy_sha256"] is None:
        raise EditingSemanticSamplingSidecarError(
            "unresolved metric policy cannot create a semantic/sampling sidecar"
        )
    metric_sha = _require_sha256(
        payload["metric_policy_sha256"],
        field="path.metric_policy_sha256",
    )
    if metric_sha != expected_metric_policy_sha256:
        raise EditingSemanticSamplingSidecarError(
            "path metric policy disagrees with the compiled candidate row"
        )
    normalized = {
        "path_length": _require_nonnegative_int(
            payload["path_length"],
            field="path.path_length",
        ),
        "operator_family_set": list(
            _ordered_operator_families(
                payload["operator_family_set"],
                operator_order=operator_order,
            )
        ),
        "path_evidence_kind": _require_text(
            payload["path_evidence_kind"],
            field="path.path_evidence_kind",
        ),
        "intermediate_evidence_kind": _require_text(
            payload["intermediate_evidence_kind"],
            field="path.intermediate_evidence_kind",
        ),
        "action_sequence_evidence_kind": _require_text(
            payload["action_sequence_evidence_kind"],
            field="path.action_sequence_evidence_kind",
        ),
        "maximum_intermediate_source_distance": _require_nonnegative_float(
            payload["maximum_intermediate_source_distance"],
            field="path.maximum_intermediate_source_distance",
        ),
        "immediate_reversal_count": _require_nonnegative_int(
            payload["immediate_reversal_count"],
            field="path.immediate_reversal_count",
        ),
        "repeated_state_count": _require_nonnegative_int(
            payload["repeated_state_count"],
            field="path.repeated_state_count",
        ),
        "metric_policy_sha256": metric_sha,
        "unresolved_metric_policies": [],
    }
    expected = {
        "path_length": address["path_length"],
        "path_evidence_kind": evidence["path"]["kind"],
        "intermediate_evidence_kind": evidence["intermediates"]["kind"],
        "action_sequence_evidence_kind": evidence["action_sequence"]["kind"],
    }
    if any(normalized[field] != expected[field] for field in expected):
        raise EditingSemanticSamplingSidecarError(
            "path descriptor disagrees with packed or candidate-ledger identity"
        )
    return normalized


def _slot_mapping(value: object, *, field: str) -> list[list[int]] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EditingSemanticSamplingSidecarError(f"{field} must be a sequence of slot pairs")
    pairs: list[tuple[int, int]] = []
    for index, pair in enumerate(value):
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise EditingSemanticSamplingSidecarError(f"{field}[{index}] must be a two-slot pair")
        pairs.append(
            (
                _require_nonnegative_int(
                    pair[0],
                    field=f"{field}[{index}].source_slot",
                ),
                _require_nonnegative_int(
                    pair[1],
                    field=f"{field}[{index}].target_slot",
                ),
            )
        )
    if len({source for source, _target in pairs}) != len(pairs) or len(
        {target for _source, target in pairs}
    ) != len(pairs):
        raise EditingSemanticSamplingSidecarError(f"{field} must be one-to-one")
    return [[source, target] for source, target in sorted(pairs)]


def _mappings(
    value: MappingDescriptor | Mapping[str, Any],
    *,
    expected_mapping_policy_sha256: str,
) -> dict[str, Any]:
    payload = (
        {
            field: (
                list(getattr(value, field))
                if field == "unresolved_mapping_policies"
                else getattr(value, field)
            )
            for field in _MAPPING_FIELDS
        }
        if isinstance(value, MappingDescriptor)
        else dict(
            _require_exact_fields(
                value,
                _MAPPING_FIELDS,
                field="mapping descriptor",
            )
        )
    )
    unresolved = _unique_sorted_text(
        payload["unresolved_mapping_policies"],
        field="mappings.unresolved_mapping_policies",
        allow_empty=True,
    )
    if unresolved or payload["mapping_policy_sha256"] is None:
        raise EditingSemanticSamplingSidecarError(
            "unresolved mapping policy cannot create a semantic/sampling sidecar"
        )
    mapping_sha = _require_sha256(
        payload["mapping_policy_sha256"],
        field="mappings.mapping_policy_sha256",
    )
    if mapping_sha != expected_mapping_policy_sha256:
        raise EditingSemanticSamplingSidecarError(
            "mapping policy disagrees with the compiled candidate row"
        )
    return {
        "constant_core_mapping": _slot_mapping(
            payload["constant_core_mapping"],
            field="constant_core_mapping",
        ),
        "protected_mapping": _slot_mapping(
            payload["protected_mapping"],
            field="protected_mapping",
        ),
        "mapping_policy_sha256": mapping_sha,
        "unresolved_mapping_policies": [],
    }


def _coefficients(
    value: SamplingCoefficients | Mapping[str, Any],
) -> dict[str, Any]:
    payload = (
        {
            field: (
                list(getattr(value, field))
                if field == "unresolved_coefficient_policies"
                else getattr(value, field)
            )
            for field in _COEFFICIENT_FIELDS
        }
        if isinstance(value, SamplingCoefficients)
        else dict(
            _require_exact_fields(
                value,
                _COEFFICIENT_FIELDS,
                field="sampling coefficients",
            )
        )
    )
    unresolved = _unique_sorted_text(
        payload["unresolved_coefficient_policies"],
        field="coefficients.unresolved_coefficient_policies",
        allow_empty=True,
    )
    if unresolved:
        raise EditingSemanticSamplingSidecarError(
            "unresolved coefficient policy cannot create a sampling sidecar"
        )
    return {
        "path_sampling_coefficient": _require_positive_float(
            payload["path_sampling_coefficient"],
            field="coefficients.path_sampling_coefficient",
        ),
        "progress_sampling_coefficient": _require_positive_float(
            payload["progress_sampling_coefficient"],
            field="coefficients.progress_sampling_coefficient",
        ),
        "effective_teacher_coefficient": _require_positive_float(
            payload["effective_teacher_coefficient"],
            field="coefficients.effective_teacher_coefficient",
        ),
        "coefficient_policy_sha256": _require_sha256(
            payload["coefficient_policy_sha256"],
            field="coefficients.coefficient_policy_sha256",
        ),
        "unresolved_coefficient_policies": [],
    }


def _ordered_membership_cells(
    contract: Mapping[str, Any],
    values: object,
    *,
    membership_order: tuple[str, ...],
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise EditingSemanticSamplingSidecarError("record_membership_cells must be a sequence")
    observed = tuple(values)
    try:
        validate_record_membership_assignment(contract, observed)
    except ValueError as error:
        raise EditingSemanticSamplingSidecarError(
            f"record membership assignment is invalid: {error}"
        ) from error
    if observed != tuple(cell for cell in membership_order if cell in observed):
        raise EditingSemanticSamplingSidecarError(
            "record_membership_cells must follow contract order"
        )
    status_by_cell = {
        cell["id"]: cell["admission_status"]
        for cell in contract["record_membership_contract"]["cells"]
    }
    unresolved = {
        cell: status_by_cell[cell] for cell in observed if status_by_cell[cell] != "admissible"
    }
    if unresolved:
        raise EditingSemanticSamplingSidecarError(
            f"sidecar cannot include unresolved membership cells: {unresolved}"
        )
    return observed


def _candidate_rows_by_sha(
    candidate_ledger: Mapping[str, Any],
    *,
    required_row_sha256s: set[str],
    candidate_ledger_rows_path: str | Path | None,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    if candidate_ledger.get("schema_version") == CANDIDATE_AUDIT_LEDGER_ARTIFACT_SCHEMA_VERSION:
        if candidate_ledger_rows_path is None:
            raise EditingSemanticSamplingSidecarError(
                "streamed candidate ledger requires candidate_ledger_rows_path"
            )
        rows = (row for row, _ in iter_candidate_audit_rows(candidate_ledger_rows_path))
    else:
        rows = iter(candidate_ledger["rows"])
    try:
        for row in rows:
            digest = str(row["row_sha256"])
            if digest not in required_row_sha256s:
                continue
            if digest in result:
                raise EditingSemanticSamplingSidecarError("candidate ledger repeats a row SHA-256")
            result[digest] = row
    except EditingCandidateAuditLedgerError as error:
        raise EditingSemanticSamplingSidecarError(
            f"candidate ledger rows are invalid: {error}"
        ) from error
    return result


def _candidate_binding(
    *,
    candidate_row: Mapping[str, Any],
    candidate_ledger_identity: Mapping[str, str],
) -> dict[str, str]:
    return {
        **candidate_ledger_identity,
        "candidate_id": str(candidate_row["candidate_id"]),
        "candidate_row_sha256": str(candidate_row["row_sha256"]),
    }


def _lane_row_binding(
    lane_registry: Mapping[str, Any],
    *,
    lane: str,
) -> dict[str, Any]:
    return {
        "registry_file_sha256": lane_registry["registry_file_sha256"],
        "registry_sha256": lane_registry["registry_sha256"],
        "lane_id": lane,
        "lane_completion_sha256": lane_registry["lane_completion_sha256"][lane],
        "lane_shard_inventory_sha256": lane_registry["lane_shard_inventory_sha256"][lane],
    }


def _build_progress_row(
    progress: SemanticSamplingProgress,
    *,
    contract: Mapping[str, Any],
    contract_identity: Mapping[str, Any],
    lane_registry: Mapping[str, Any],
    packed_corpus: Mapping[str, Any],
    active8: Mapping[str, Any],
    split_assignment: Mapping[str, Any],
    candidate_ledger_identity: Mapping[str, str],
    candidate_rows: Mapping[str, Mapping[str, Any]],
    partition_roles: tuple[str, ...],
    lane_ids: tuple[str, ...],
    operator_order: tuple[str, ...],
    membership_order: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(progress, SemanticSamplingProgress):
        raise EditingSemanticSamplingSidecarError(
            "sidecar rows must be typed SemanticSamplingProgress values"
        )
    address = _packed_address(progress.address)
    lane = address["lane"]
    partition = address["partition"]
    if lane not in lane_ids or partition not in partition_roles:
        raise EditingSemanticSamplingSidecarError(
            "packed address lane or partition is outside the v2 contract"
        )
    candidate_digest = _require_sha256(
        progress.candidate_ledger_row_sha256,
        field="candidate_ledger_row_sha256",
    )
    try:
        candidate_row = candidate_rows[candidate_digest]
    except KeyError:
        raise EditingSemanticSamplingSidecarError(
            "sidecar row does not resolve to a candidate-ledger row"
        ) from None
    if (
        candidate_row["disposition"] != "compiled_candidate"
        or candidate_row["accepted_trace_id"] != address["trace_id"]
        or candidate_row["data_lane"] != lane
    ):
        raise EditingSemanticSamplingSidecarError(
            "sidecar address disagrees with its compiled candidate-ledger row"
        )

    resolution = dict(progress.partition_resolution)
    try:
        validate_partition_resolution(
            contract,
            resolution,
            admitted_record=True,
        )
    except (TypeError, ValueError) as error:
        raise EditingSemanticSamplingSidecarError(
            f"sidecar partition resolution is invalid: {error}"
        ) from error
    if (
        resolution["assigned_role"] != partition
        or resolution["split_assignment_manifest_sha256"] != split_assignment["manifest_sha256"]
    ):
        raise EditingSemanticSamplingSidecarError(
            "packed partition disagrees with the bound split resolution or manifest"
        )

    evidence = candidate_row["evidence"]
    expected_mapping_sha = candidate_row["policy_bindings"]["mapping_policy_sha256"]
    expected_metric_sha = candidate_row["policy_bindings"]["metric_policy_sha256"]
    if not isinstance(expected_mapping_sha, str) or not isinstance(
        expected_metric_sha,
        str,
    ):
        raise EditingSemanticSamplingSidecarError("compiled candidate row lacks resolved policies")
    endpoint = _endpoint_descriptor(
        progress.endpoint_descriptor,
        address=address,
        evidence=evidence,
    )
    path = _path_descriptor(
        progress.path_descriptor,
        address=address,
        evidence=evidence,
        operator_order=operator_order,
        expected_metric_policy_sha256=expected_metric_sha,
    )
    mappings = _mappings(
        progress.mappings,
        expected_mapping_policy_sha256=expected_mapping_sha,
    )
    coefficients = _coefficients(progress.sampling_coefficients)
    membership_cells = _ordered_membership_cells(
        contract,
        progress.record_membership_cells,
        membership_order=membership_order,
    )
    membership_classifier_identity = _require_sha256(
        progress.record_membership_classifier_identity_sha256,
        field="record_membership_classifier_identity_sha256",
    )
    relationship_group_ids = _unique_sorted_text(
        progress.relationship_group_ids,
        field="relationship_group_ids",
        allow_empty=True,
    )
    envelope_sha256 = _require_sha256(
        progress.admitted_record_envelope_sha256,
        field="admitted_record_envelope_sha256",
    )
    progress_index = _require_nonnegative_int(
        progress.progress_index,
        field="progress_index",
    )
    if progress_index > address["path_length"]:
        raise EditingSemanticSamplingSidecarError(
            "progress_index lies beyond the packed trace path"
        )
    record_key = _record_key(address)
    progress_key = {**record_key, "progress_index": progress_index}
    body = {
        "schema": SEMANTIC_SAMPLING_ROW_SCHEMA,
        "schema_version": SEMANTIC_SAMPLING_ROW_SCHEMA_VERSION,
        "status": SEMANTIC_SAMPLING_ROW_STATUS,
        "training_authorized": False,
        "contract_identity": dict(contract_identity),
        "lane_registry_binding": _lane_row_binding(
            lane_registry,
            lane=lane,
        ),
        "packed_corpus_identity": dict(packed_corpus),
        "active8_identity": dict(active8),
        "split_assignment_identity": dict(split_assignment),
        "candidate_ledger_binding": _candidate_binding(
            candidate_row=candidate_row,
            candidate_ledger_identity=candidate_ledger_identity,
        ),
        "packed_address": address,
        "record_key_sha256": canonical_sha256(record_key),
        "partition_resolution": resolution,
        "admitted_record_envelope_sha256": envelope_sha256,
        "progress_key": progress_key,
        "progress_key_sha256": canonical_sha256(progress_key),
        "groups": _groups(progress.groups),
        "record_membership_cells": list(membership_cells),
        "record_membership_classifier_identity_sha256": (membership_classifier_identity),
        "endpoint_descriptor": endpoint,
        "path_descriptor": path,
        "relationship_group_ids": list(relationship_group_ids),
        "mappings": mappings,
        "sampling_coefficients": coefficients,
    }
    return {**body, "row_sha256": canonical_sha256(body)}


def _row_sort_key(
    row: Mapping[str, Any],
    *,
    partition_roles: tuple[str, ...],
    lane_ids: tuple[str, ...],
) -> tuple[Any, ...]:
    address = row["packed_address"]
    return (
        partition_roles.index(address["partition"]),
        lane_ids.index(address["lane"]),
        address["packed_shard_content_sha256"],
        address["entry_index"],
        address["trace_id"],
        row["progress_key"]["progress_index"],
    )


def _validate_complete_progress_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    admitted_trace_keys: tuple[tuple[str, int, str], ...],
) -> None:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        address = row["packed_address"]
        key = (
            address["packed_shard_content_sha256"],
            address["entry_index"],
            address["trace_id"],
        )
        grouped[key].append(row)
    if tuple(sorted(grouped)) != admitted_trace_keys:
        raise EditingSemanticSamplingSidecarError(
            "sidecar trace set disagrees with the exact post-Active8 admitted stream"
        )
    invariant_fields = (
        "contract_identity",
        "lane_registry_binding",
        "packed_corpus_identity",
        "active8_identity",
        "split_assignment_identity",
        "candidate_ledger_binding",
        "packed_address",
        "record_key_sha256",
        "partition_resolution",
        "admitted_record_envelope_sha256",
        "groups",
        "record_membership_cells",
        "record_membership_classifier_identity_sha256",
        "endpoint_descriptor",
        "path_descriptor",
        "relationship_group_ids",
        "mappings",
    )
    for key, record_rows in grouped.items():
        path_length = record_rows[0]["packed_address"]["path_length"]
        progress_indexes = [row["progress_key"]["progress_index"] for row in record_rows]
        if progress_indexes != list(range(path_length + 1)):
            raise EditingSemanticSamplingSidecarError(
                f"sidecar trace {key!r} does not cover every progress position exactly once"
            )
        first = record_rows[0]
        for row in record_rows[1:]:
            if any(row[field] != first[field] for field in invariant_fields):
                raise EditingSemanticSamplingSidecarError(
                    f"sidecar trace {key!r} changes record metadata across progress"
                )
            if (
                row["sampling_coefficients"]["path_sampling_coefficient"]
                != first["sampling_coefficients"]["path_sampling_coefficient"]
                or row["sampling_coefficients"]["coefficient_policy_sha256"]
                != first["sampling_coefficients"]["coefficient_policy_sha256"]
            ):
                raise EditingSemanticSamplingSidecarError(
                    f"sidecar trace {key!r} changes its path coefficient or policy"
                )


def _sidecar_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    partition_roles: tuple[str, ...],
    lane_ids: tuple[str, ...],
) -> dict[str, Any]:
    record_rows = [row for row in rows if row["progress_key"]["progress_index"] == 0]
    membership_counts = Counter(
        membership for row in record_rows for membership in row["record_membership_cells"]
    )
    return {
        "records": len(record_rows),
        "progress_rows": len(rows),
        "by_partition": {
            partition: sum(row["packed_address"]["partition"] == partition for row in record_rows)
            for partition in partition_roles
        },
        "by_lane": {
            lane: sum(row["packed_address"]["lane"] == lane for row in record_rows)
            for lane in lane_ids
        },
        "records_by_membership_cell": {
            membership: membership_counts[membership] for membership in sorted(membership_counts)
        },
    }


def _validated_candidate_ledger(
    candidate_ledger: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    contract_file_sha256: str,
    candidate_ledger_rows_path: str | Path | None,
) -> Mapping[str, Any]:
    try:
        source = CandidateSourceIdentity(**candidate_ledger["source_identity"])
        compiler = CandidateCompilerIdentity(**candidate_ledger["compiler_identity"])
        if candidate_ledger.get("schema_version") == CANDIDATE_AUDIT_LEDGER_ARTIFACT_SCHEMA_VERSION:
            if candidate_ledger_rows_path is None:
                raise EditingSemanticSamplingSidecarError(
                    "streamed candidate ledger requires candidate_ledger_rows_path"
                )
            return validate_candidate_audit_ledger_artifact(
                candidate_ledger,
                rows_path=candidate_ledger_rows_path,
                contract=contract,
                contract_file_sha256=contract_file_sha256,
                expected_source_identity=source,
                expected_compiler_identity=compiler,
                expected_attempt_count=candidate_ledger["expected_attempt_count"],
                expected_attempt_stream_sha256=candidate_ledger["expected_attempt_stream_sha256"],
            )
        return validate_candidate_audit_ledger(
            candidate_ledger,
            contract=contract,
            contract_file_sha256=contract_file_sha256,
            expected_source_identity=source,
            expected_compiler_identity=compiler,
            expected_attempt_count=candidate_ledger["expected_attempt_count"],
            expected_attempt_stream_sha256=candidate_ledger["expected_attempt_stream_sha256"],
        )
    except (
        KeyError,
        TypeError,
        EditingCandidateAuditLedgerError,
    ) as error:
        raise EditingSemanticSamplingSidecarError(
            f"candidate ledger is invalid: {error}"
        ) from error


def _assert_stream_identities(
    progress_rows: Sequence[SemanticSamplingProgress],
    *,
    packed_corpus: Mapping[str, Any],
    split_assignment: Mapping[str, Any],
) -> None:
    if (
        split_resolution_stream_sha256(progress_rows)
        != split_assignment["resolution_stream_sha256"]
    ):
        raise EditingSemanticSamplingSidecarError(
            "split assignment identity disagrees with the complete resolution stream"
        )
    if (
        admitted_record_envelope_inventory_sha256(progress_rows)
        != packed_corpus["admitted_record_envelope_inventory_sha256"]
    ):
        raise EditingSemanticSamplingSidecarError(
            "packed corpus identity disagrees with the admitted-envelope inventory"
        )


def build_semantic_sampling_sidecar(
    *,
    contract: Mapping[str, Any],
    contract_file_sha256: str,
    lane_registry_identity: LaneRegistryIdentity,
    packed_corpus_identity: PackedCorpusIdentity,
    active8_identity: Active8SidecarIdentity,
    split_assignment_identity: SplitAssignmentIdentity,
    active8_admitted_trace_keys: Sequence[tuple[str, int, str]],
    candidate_ledger: Mapping[str, Any],
    candidate_ledger_manifest_file_sha256: str,
    progress_rows: Sequence[SemanticSamplingProgress],
    candidate_ledger_rows_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic sidecar over the exact post-Active8 trace stream."""

    (
        contract_identity,
        partition_roles,
        lane_ids,
        operator_order,
        membership_order,
    ) = _contract_context(
        contract,
        contract_file_sha256=contract_file_sha256,
    )
    lane_registry = _lane_registry_identity(
        lane_registry_identity,
        lane_ids=lane_ids,
    )
    packed_corpus = _packed_corpus_identity(packed_corpus_identity)
    active8 = _active8_identity(active8_identity)
    split_assignment = _split_assignment_identity(split_assignment_identity)
    admitted_keys = normalize_admitted_trace_keys(active8_admitted_trace_keys)
    if active8["admitted_trace_count"] != len(admitted_keys) or active8[
        "admitted_trace_keys_sha256"
    ] != active8_admitted_trace_keys_sha256(admitted_keys):
        raise EditingSemanticSamplingSidecarError(
            "Active8 identity disagrees with the exact admitted trace stream"
        )
    if active8["unified_packed_manifest_sha256"] != packed_corpus["unified_packed_manifest_sha256"]:
        raise EditingSemanticSamplingSidecarError(
            "Active8 and sidecar name different unified packed manifests"
        )
    _assert_stream_identities(
        progress_rows,
        packed_corpus=packed_corpus,
        split_assignment=split_assignment,
    )
    validated_ledger = _validated_candidate_ledger(
        candidate_ledger,
        contract=contract,
        contract_file_sha256=contract_file_sha256,
        candidate_ledger_rows_path=candidate_ledger_rows_path,
    )
    if (
        validated_ledger["source_identity"]["manifest_file_sha256"]
        != packed_corpus["source_manifest_file_sha256"]
        or validated_ledger["source_identity"]["manifest_sha256"]
        != packed_corpus["source_manifest_sha256"]
    ):
        raise EditingSemanticSamplingSidecarError(
            "candidate ledger and packed corpus name different source manifests"
        )
    ledger_identity = _candidate_ledger_identity(
        validated_ledger,
        manifest_file_sha256=candidate_ledger_manifest_file_sha256,
    )
    candidate_rows = _candidate_rows_by_sha(
        validated_ledger,
        required_row_sha256s={progress.candidate_ledger_row_sha256 for progress in progress_rows},
        candidate_ledger_rows_path=candidate_ledger_rows_path,
    )
    rows = [
        _build_progress_row(
            progress,
            contract=contract,
            contract_identity=contract_identity,
            lane_registry=lane_registry,
            packed_corpus=packed_corpus,
            active8=active8,
            split_assignment=split_assignment,
            candidate_ledger_identity=ledger_identity,
            candidate_rows=candidate_rows,
            partition_roles=partition_roles,
            lane_ids=lane_ids,
            operator_order=operator_order,
            membership_order=membership_order,
        )
        for progress in progress_rows
    ]
    if not rows:
        raise EditingSemanticSamplingSidecarError("semantic/sampling sidecar cannot be empty")
    expected_order = sorted(
        rows,
        key=lambda row: _row_sort_key(
            row,
            partition_roles=partition_roles,
            lane_ids=lane_ids,
        ),
    )
    if rows != expected_order:
        raise EditingSemanticSamplingSidecarError(
            "semantic/sampling rows must be in deterministic packed/progress order"
        )
    if len({row["progress_key_sha256"] for row in rows}) != len(rows):
        raise EditingSemanticSamplingSidecarError(
            "semantic/sampling sidecar repeats a progress key"
        )
    _validate_complete_progress_rows(rows, admitted_trace_keys=admitted_keys)
    body = {
        "schema": SEMANTIC_SAMPLING_SIDECAR_SCHEMA,
        "schema_version": SEMANTIC_SAMPLING_SIDECAR_SCHEMA_VERSION,
        "status": SEMANTIC_SAMPLING_SIDECAR_STATUS,
        "training_authorized": False,
        "blockers": _launch_blockers(contract),
        "contract_identity": contract_identity,
        "lane_registry_identity": lane_registry,
        "packed_corpus_identity": packed_corpus,
        "active8_identity": active8,
        "split_assignment_identity": split_assignment,
        "candidate_ledger_identity": ledger_identity,
        "partition_roles": list(partition_roles),
        "lane_ids": list(lane_ids),
        "counts": _sidecar_counts(
            rows,
            partition_roles=partition_roles,
            lane_ids=lane_ids,
        ),
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
    }
    manifest = {**body, "manifest_sha256": canonical_sha256(body)}
    validate_semantic_sampling_sidecar(
        manifest,
        contract=contract,
        contract_file_sha256=contract_file_sha256,
        expected_lane_registry_identity=lane_registry_identity,
        expected_packed_corpus_identity=packed_corpus_identity,
        expected_active8_identity=active8_identity,
        expected_split_assignment_identity=split_assignment_identity,
        expected_active8_admitted_trace_keys=admitted_keys,
        candidate_ledger=validated_ledger,
        expected_candidate_ledger_manifest_file_sha256=(candidate_ledger_manifest_file_sha256),
        candidate_ledger_rows_path=candidate_ledger_rows_path,
    )
    return manifest


def _progress_from_row(row: Mapping[str, Any]) -> SemanticSamplingProgress:
    address = row["packed_address"]
    return SemanticSamplingProgress(
        address=PackedTraceAddress(
            packed_shard_content_sha256=address["packed_shard_content_sha256"],
            packed_shard_name=address["packed_shard_name"],
            entry_index=address["entry_index"],
            trace_id=address["trace_id"],
            layer=address["lane"],
            partition=address["partition"],
            source_key=address["source_key"],
            target_key=address["target_key"],
            path_length=address["path_length"],
        ),
        progress_index=row["progress_key"]["progress_index"],
        candidate_ledger_row_sha256=row["candidate_ledger_binding"]["candidate_row_sha256"],
        partition_resolution=dict(row["partition_resolution"]),
        admitted_record_envelope_sha256=row["admitted_record_envelope_sha256"],
        groups=SemanticGroups(
            source_group_id=row["groups"]["source_group_id"],
            scaffold_group_id=row["groups"]["scaffold_group_id"],
            series_group_id=row["groups"]["series_group_id"],
            document_or_source_group_id=row["groups"]["document_or_source_group_id"],
            transformation_signature=row["groups"]["transformation_signature"],
        ),
        record_membership_cells=tuple(row["record_membership_cells"]),
        record_membership_classifier_identity_sha256=row[
            "record_membership_classifier_identity_sha256"
        ],
        endpoint_descriptor=EndpointDescriptor(**row["endpoint_descriptor"]),
        path_descriptor=PathDescriptor(
            **{
                **row["path_descriptor"],
                "operator_family_set": tuple(row["path_descriptor"]["operator_family_set"]),
                "unresolved_metric_policies": tuple(
                    row["path_descriptor"]["unresolved_metric_policies"]
                ),
            }
        ),
        relationship_group_ids=tuple(row["relationship_group_ids"]),
        mappings=MappingDescriptor(
            constant_core_mapping=(
                None
                if row["mappings"]["constant_core_mapping"] is None
                else tuple(tuple(pair) for pair in row["mappings"]["constant_core_mapping"])
            ),
            protected_mapping=(
                None
                if row["mappings"]["protected_mapping"] is None
                else tuple(tuple(pair) for pair in row["mappings"]["protected_mapping"])
            ),
            mapping_policy_sha256=row["mappings"]["mapping_policy_sha256"],
            unresolved_mapping_policies=tuple(row["mappings"]["unresolved_mapping_policies"]),
        ),
        sampling_coefficients=SamplingCoefficients(
            path_sampling_coefficient=row["sampling_coefficients"]["path_sampling_coefficient"],
            progress_sampling_coefficient=row["sampling_coefficients"][
                "progress_sampling_coefficient"
            ],
            effective_teacher_coefficient=row["sampling_coefficients"][
                "effective_teacher_coefficient"
            ],
            coefficient_policy_sha256=row["sampling_coefficients"]["coefficient_policy_sha256"],
            unresolved_coefficient_policies=tuple(
                row["sampling_coefficients"]["unresolved_coefficient_policies"]
            ),
        ),
    )


def validate_semantic_sampling_sidecar(
    manifest: object,
    *,
    contract: Mapping[str, Any],
    contract_file_sha256: str,
    expected_lane_registry_identity: LaneRegistryIdentity | Mapping[str, Any],
    expected_packed_corpus_identity: PackedCorpusIdentity | Mapping[str, Any],
    expected_active8_identity: Active8SidecarIdentity | Mapping[str, Any],
    expected_split_assignment_identity: SplitAssignmentIdentity | Mapping[str, Any],
    expected_active8_admitted_trace_keys: Sequence[tuple[str, int, str]],
    candidate_ledger: Mapping[str, Any],
    expected_candidate_ledger_manifest_file_sha256: str,
    candidate_ledger_rows_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Re-derive every sidecar row and validate all parent/content identities."""

    payload = _require_exact_fields(
        manifest,
        _MANIFEST_FIELDS,
        field="semantic/sampling sidecar",
    )
    (
        contract_identity,
        partition_roles,
        lane_ids,
        operator_order,
        membership_order,
    ) = _contract_context(
        contract,
        contract_file_sha256=contract_file_sha256,
    )
    lane_registry = _lane_registry_identity(
        expected_lane_registry_identity,
        lane_ids=lane_ids,
    )
    packed_corpus = _packed_corpus_identity(expected_packed_corpus_identity)
    active8 = _active8_identity(expected_active8_identity)
    split_assignment = _split_assignment_identity(expected_split_assignment_identity)
    admitted_keys = normalize_admitted_trace_keys(expected_active8_admitted_trace_keys)
    if (
        active8["admitted_trace_count"] != len(admitted_keys)
        or active8["admitted_trace_keys_sha256"]
        != active8_admitted_trace_keys_sha256(admitted_keys)
        or active8["unified_packed_manifest_sha256"]
        != packed_corpus["unified_packed_manifest_sha256"]
    ):
        raise EditingSemanticSamplingSidecarError(
            "expected Active8 or packed-corpus identity is inconsistent"
        )
    validated_ledger = _validated_candidate_ledger(
        candidate_ledger,
        contract=contract,
        contract_file_sha256=contract_file_sha256,
        candidate_ledger_rows_path=candidate_ledger_rows_path,
    )
    ledger_identity = _candidate_ledger_identity(
        validated_ledger,
        manifest_file_sha256=(expected_candidate_ledger_manifest_file_sha256),
    )
    if (
        validated_ledger["source_identity"]["manifest_file_sha256"]
        != packed_corpus["source_manifest_file_sha256"]
        or validated_ledger["source_identity"]["manifest_sha256"]
        != packed_corpus["source_manifest_sha256"]
    ):
        raise EditingSemanticSamplingSidecarError(
            "candidate ledger and packed source provenance disagree"
        )
    if (
        payload["schema"] != SEMANTIC_SAMPLING_SIDECAR_SCHEMA
        or payload["schema_version"] != SEMANTIC_SAMPLING_SIDECAR_SCHEMA_VERSION
        or payload["status"] != SEMANTIC_SAMPLING_SIDECAR_STATUS
        or payload["training_authorized"] is not False
    ):
        raise EditingSemanticSamplingSidecarError(
            "semantic/sampling sidecar identity or authority is invalid"
        )
    if (
        payload["blockers"] != _launch_blockers(contract)
        or payload["contract_identity"] != contract_identity
        or payload["lane_registry_identity"] != lane_registry
        or payload["packed_corpus_identity"] != packed_corpus
        or payload["active8_identity"] != active8
        or payload["split_assignment_identity"] != split_assignment
        or payload["candidate_ledger_identity"] != ledger_identity
        or tuple(payload["partition_roles"]) != partition_roles
        or tuple(payload["lane_ids"]) != lane_ids
    ):
        raise EditingSemanticSamplingSidecarError(
            "semantic/sampling sidecar parent or blocker identity disagrees"
        )
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise EditingSemanticSamplingSidecarError("semantic/sampling sidecar rows must be nonempty")
    required_candidate_rows = {
        row["candidate_ledger_binding"]["candidate_row_sha256"]
        for row in raw_rows
        if isinstance(row, Mapping)
        and isinstance(row.get("candidate_ledger_binding"), Mapping)
        and isinstance(
            row["candidate_ledger_binding"].get("candidate_row_sha256"),
            str,
        )
    }
    candidate_rows = _candidate_rows_by_sha(
        validated_ledger,
        required_row_sha256s=required_candidate_rows,
        candidate_ledger_rows_path=candidate_ledger_rows_path,
    )
    normalized_rows: list[dict[str, Any]] = []
    typed_progress: list[SemanticSamplingProgress] = []
    for index, raw_row in enumerate(raw_rows):
        row = _require_exact_fields(
            raw_row,
            _ROW_FIELDS,
            field=f"semantic/sampling row[{index}]",
        )
        progress = _progress_from_row(row)
        normalized = _build_progress_row(
            progress,
            contract=contract,
            contract_identity=contract_identity,
            lane_registry=lane_registry,
            packed_corpus=packed_corpus,
            active8=active8,
            split_assignment=split_assignment,
            candidate_ledger_identity=ledger_identity,
            candidate_rows=candidate_rows,
            partition_roles=partition_roles,
            lane_ids=lane_ids,
            operator_order=operator_order,
            membership_order=membership_order,
        )
        if dict(row) != normalized:
            raise EditingSemanticSamplingSidecarError(
                f"semantic/sampling row[{index}] is not canonical or its hash disagrees"
            )
        normalized_rows.append(normalized)
        typed_progress.append(progress)
    _assert_stream_identities(
        typed_progress,
        packed_corpus=packed_corpus,
        split_assignment=split_assignment,
    )
    expected_order = sorted(
        normalized_rows,
        key=lambda row: _row_sort_key(
            row,
            partition_roles=partition_roles,
            lane_ids=lane_ids,
        ),
    )
    if normalized_rows != expected_order:
        raise EditingSemanticSamplingSidecarError(
            "semantic/sampling rows are not in deterministic order"
        )
    if len({row["progress_key_sha256"] for row in normalized_rows}) != len(normalized_rows):
        raise EditingSemanticSamplingSidecarError(
            "semantic/sampling sidecar repeats a progress key"
        )
    _validate_complete_progress_rows(
        normalized_rows,
        admitted_trace_keys=admitted_keys,
    )
    if payload["rows_sha256"] != canonical_sha256(normalized_rows):
        raise EditingSemanticSamplingSidecarError("semantic/sampling row-content hash disagrees")
    counts = _sidecar_counts(
        normalized_rows,
        partition_roles=partition_roles,
        lane_ids=lane_ids,
    )
    if payload["counts"] != counts:
        raise EditingSemanticSamplingSidecarError("semantic/sampling counts disagree with rows")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload["manifest_sha256"] != canonical_sha256(body):
        raise EditingSemanticSamplingSidecarError("semantic/sampling sidecar self-hash disagrees")
    return MappingProxyType(dict(payload))


__all__ = [
    "SEMANTIC_SAMPLING_ROW_SCHEMA",
    "SEMANTIC_SAMPLING_ROW_SCHEMA_VERSION",
    "SEMANTIC_SAMPLING_ROW_STATUS",
    "SEMANTIC_SAMPLING_SIDECAR_SCHEMA",
    "SEMANTIC_SAMPLING_SIDECAR_SCHEMA_VERSION",
    "SEMANTIC_SAMPLING_SIDECAR_STATUS",
    "Active8SidecarIdentity",
    "EditingSemanticSamplingSidecarError",
    "EndpointDescriptor",
    "LaneRegistryIdentity",
    "MappingDescriptor",
    "PackedCorpusIdentity",
    "PathDescriptor",
    "SamplingCoefficients",
    "SemanticGroups",
    "SemanticSamplingProgress",
    "SplitAssignmentIdentity",
    "active8_admitted_trace_keys_sha256",
    "admitted_record_envelope_inventory_sha256",
    "build_semantic_sampling_sidecar",
    "normalize_admitted_trace_keys",
    "split_resolution_stream_sha256",
    "validate_semantic_sampling_sidecar",
]
