"""Freeze candidate provenance and bridge packed headers to split-census rows.

The packed-candidate materializer deliberately stops before the formal
candidate audit ledger and split-component census.  This module closes that
boundary without upgrading evidence.  It requires an explicit, self-hashed
registry that binds:

* the exact candidate materialization and editing-corpus contract;
* every source asset and its source-group namespace;
* compiler, mapping, metric, evidence-reference, and identity definitions.

Every candidate header becomes one formal audit-ledger row.  Rejected headers
remain rejected ledger attempts and never enter the split-candidate stream.
Only routed headers produce schema-V4 split rows, always with ``mass_units=1``.
Packed headers contain no document or genuine-series receipts, so this bridge
always emits null document and series groups and provenance.  It never derives
those claims from filenames, source-group labels, or endpoint identities.

The adapter output is a same-parent atomic directory publication.  Its
``source_stream`` object is intended to be copied verbatim into the split
census.  It binds physical and semantic hashes for the candidate
materialization, frozen registry, formal ledger, and normalized split rows.
None of these artifacts grants split, Active8, Gate-0, or training authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from compose_v4.data.editing_candidate_audit_ledger import (
    EVIDENCE_COMPONENTS,
    CandidateAuditAttempt,
    CandidateCompilerIdentity,
    CandidateEvidence,
    CandidateEvidenceComponent,
    CandidateLaneResolution,
    CandidatePolicyBindings,
    CandidateSourceIdentity,
    build_candidate_audit_ledger,
    candidate_attempt_stream_sha256,
    validate_candidate_audit_ledger,
)
from compose_v4.data.editing_corpus_contract import load_editing_corpus_contract
from compose_v4.data.editing_v2_packed_candidate_materializer import (
    CANDIDATE_HEADER_SCHEMA,
    CANDIDATE_HEADER_SCHEMA_VERSION,
    CANDIDATE_HEADER_STATUS,
    CANDIDATE_ROWS_FILENAME,
    MATERIALIZATION_FILENAME,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    validate_packed_candidate_materialization,
)
from compose_v4.data.editing_v2_split_census import (
    CANDIDATE_ROW_SCHEMA,
    CANDIDATE_ROW_SCHEMA_VERSION,
    IDENTITY_DEFINITION_CONTRACT_SCHEMA,
    IDENTITY_DEFINITION_CONTRACT_SCHEMA_VERSION,
    RELATIONSHIP_NAMESPACE_SCHEMA,
    RELATIONSHIP_NAMESPACE_SCHEMA_VERSION,
    default_split_census_policy,
)

PROVENANCE_REGISTRY_SCHEMA = "compose.editing_v2_candidate_provenance_registry"
PROVENANCE_REGISTRY_SCHEMA_VERSION = 1
PROVENANCE_REGISTRY_STATUS = "FROZEN_CANDIDATE_PROVENANCE_NO_TRAINING_AUTHORITY"

PROVENANCE_BRIDGE_SCHEMA = "compose.editing_v2_candidate_provenance_bridge"
PROVENANCE_BRIDGE_SCHEMA_VERSION = 1
PROVENANCE_BRIDGE_STATUS = "COMPLETE_CANDIDATE_PROVENANCE_NO_TRAINING_AUTHORITY"

SOURCE_STREAM_SCHEMA = "compose.editing_v2_split_candidate_source_stream"
SOURCE_STREAM_SCHEMA_VERSION = 1

REGISTRY_FILENAME = "CANDIDATE_PROVENANCE_REGISTRY.json"
AUDIT_LEDGER_FILENAME = "candidate_audit_ledger.json"
SPLIT_ROWS_FILENAME = "split_candidates.jsonl"
BRIDGE_MANIFEST_FILENAME = "CANDIDATE_PROVENANCE_BRIDGE.json"

REQUIRED_BLOCKERS = (
    "split_component_census.not_run",
    "split_assignment.not_run",
    "active8_whole_trace_admission.not_run",
    "gate_zero.not_run",
    "training.not_authorized",
)

_SHA256_HEX = frozenset("0123456789abcdef")
_REGISTRY_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "candidate_materialization",
    "editing_corpus_contract",
    "compiler_identity",
    "policy_bindings",
    "evidence_identity",
    "identity_definitions",
    "source_assets",
    "registry_sha256",
}
_CANDIDATE_MATERIALIZATION_FIELDS = {
    "manifest_file_sha256",
    "manifest_sha256",
    "rows_file_sha256",
    "rows_semantic_sha256",
    "address_stream_sha256",
}
_CONTRACT_FIELDS = {"contract_id", "file_sha256", "semantic_sha256"}
_COMPILER_FIELDS = {
    "implementation_sha256",
    "config_sha256",
    "operator_contract_sha256",
    "canonicalizer_sha256",
}
_POLICY_FIELDS = {"mapping_policy_sha256", "metric_policy_sha256"}
_EVIDENCE_IDENTITY_FIELDS = {
    "definition_id",
    "implementation_sha256",
    "source_record_id_fields",
    "component_reference_fields",
    "evidence_identity_sha256",
}
_SOURCE_ASSET_FIELDS = {
    "source_asset_id",
    "source_asset_path",
    "source_asset_sha256",
    "source_version",
    "access_basis",
    "source_group_namespace",
}
_NAMESPACE_FIELDS = {
    "schema",
    "schema_version",
    "relationship_type",
    "namespace_id",
    "source_asset_id",
    "source_asset_sha256",
    "definition_id",
    "implementation_sha256",
    "cross_lane_sharing_authorized",
}
_IDENTITY_DEFINITION_FIELDS = {
    "schema",
    "schema_version",
    "exact_molecule",
    "partition_scaffold",
    "source_group",
}
_ONE_IDENTITY_DEFINITION_FIELDS = {
    "definition_id",
    "implementation_sha256",
    "identity_namespace",
}
_SOURCE_STREAM_FIELDS = {
    "schema",
    "schema_version",
    "nonempty_jsonl_rows",
    "candidate_materialization",
    "provenance_registry",
    "candidate_audit_ledger",
    "split_candidates",
    "source_stream_sha256",
}
_SOURCE_STREAM_REGISTRY_FIELDS = {"file_sha256", "registry_sha256"}
_SOURCE_STREAM_LEDGER_FIELDS = {
    "file_sha256",
    "semantic_sha256",
    "rows_sha256",
}
_SOURCE_STREAM_ROWS_FIELDS = {"file_sha256", "semantic_sha256"}
_BRIDGE_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "blockers",
    "source_stream",
    "counts",
    "outputs",
    "manifest_sha256",
}
_BRIDGE_COUNT_FIELDS = {"attempted", "routed", "rejected", "split_rows"}
_BRIDGE_OUTPUT_FIELDS = {"candidate_audit_ledger", "split_candidates"}
_OUTPUT_IDENTITY_FIELDS = {
    "relative_path",
    "file_sha256",
    "semantic_sha256",
}
_RELATIONSHIP_TYPES = (
    "shared_prefix_branch",
    "alternative_route",
    "inverse_pair",
    "correction",
    "constraint_compatible_alternative_route",
)


class EditingV2CandidateProvenanceBridgeError(ValueError):
    """A frozen identity or candidate provenance boundary is incomplete."""


def _require_mapping(
    value: object, *, fields: set[str], field: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EditingV2CandidateProvenanceBridgeError(f"{field} must be an object")
    actual = set(value)
    if actual != fields:
        raise EditingV2CandidateProvenanceBridgeError(
            f"{field} fields disagree; missing={sorted(fields - actual)}, "
            f"unexpected={sorted(actual - fields)}"
        )
    return value


def _require_sha256(value: object, *, field: str) -> str:
    digest = value if isinstance(value, str) else ""
    if len(digest) != 64 or any(character not in _SHA256_HEX for character in digest):
        raise EditingV2CandidateProvenanceBridgeError(
            f"{field} must be a full lowercase SHA-256"
        )
    return digest


def _require_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise EditingV2CandidateProvenanceBridgeError(
            f"{field} must be nonempty normalized text"
        )
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise EditingV2CandidateProvenanceBridgeError(
            f"{field} must be a nonnegative integer"
        )
    return value


def _require_sorted_texts(
    value: object,
    *,
    field: str,
    nonempty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EditingV2CandidateProvenanceBridgeError(
            f"{field} must be a sequence of strings"
        )
    values = tuple(
        _require_text(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    )
    if values != tuple(sorted(set(values))):
        raise EditingV2CandidateProvenanceBridgeError(
            f"{field} must be unique and deterministically sorted"
        )
    if nonempty and not values:
        raise EditingV2CandidateProvenanceBridgeError(f"{field} must not be empty")
    return values


def _load_json(path: Path, *, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingV2CandidateProvenanceBridgeError(
            f"cannot load {field}: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise EditingV2CandidateProvenanceBridgeError(f"{field} must be an object")
    return value


def _materialization_identity(
    candidate_root: Path,
    materialization: Mapping[str, Any],
) -> dict[str, str]:
    rows = materialization.get("rows")
    if not isinstance(rows, Mapping):
        raise EditingV2CandidateProvenanceBridgeError(
            "candidate materialization rows identity is absent"
        )
    return {
        "manifest_file_sha256": file_sha256(candidate_root / MATERIALIZATION_FILENAME),
        "manifest_sha256": _require_sha256(
            materialization.get("manifest_sha256"),
            field="candidate materialization manifest_sha256",
        ),
        "rows_file_sha256": _require_sha256(
            rows.get("file_sha256"),
            field="candidate materialization rows.file_sha256",
        ),
        "rows_semantic_sha256": _require_sha256(
            rows.get("semantic_sha256"),
            field="candidate materialization rows.semantic_sha256",
        ),
        "address_stream_sha256": _require_sha256(
            rows.get("address_stream_sha256"),
            field="candidate materialization rows.address_stream_sha256",
        ),
    }


def _normalize_identity_definitions(value: object) -> dict[str, Any]:
    payload = _require_mapping(
        value,
        fields=_IDENTITY_DEFINITION_FIELDS,
        field="registry.identity_definitions",
    )
    if (
        payload["schema"] != IDENTITY_DEFINITION_CONTRACT_SCHEMA
        or payload["schema_version"] != IDENTITY_DEFINITION_CONTRACT_SCHEMA_VERSION
    ):
        raise EditingV2CandidateProvenanceBridgeError(
            "registry.identity_definitions schema is unsupported"
        )
    normalized: dict[str, Any] = {
        "schema": IDENTITY_DEFINITION_CONTRACT_SCHEMA,
        "schema_version": IDENTITY_DEFINITION_CONTRACT_SCHEMA_VERSION,
    }
    for identity_type in ("exact_molecule", "partition_scaffold", "source_group"):
        definition = _require_mapping(
            payload[identity_type],
            fields=_ONE_IDENTITY_DEFINITION_FIELDS,
            field=f"registry.identity_definitions.{identity_type}",
        )
        normalized[identity_type] = {
            "definition_id": _require_text(
                definition["definition_id"],
                field=f"registry.identity_definitions.{identity_type}.definition_id",
            ),
            "implementation_sha256": _require_sha256(
                definition["implementation_sha256"],
                field=(
                    f"registry.identity_definitions.{identity_type}.implementation_sha256"
                ),
            ),
            "identity_namespace": _require_text(
                definition["identity_namespace"],
                field=f"registry.identity_definitions.{identity_type}.identity_namespace",
            ),
        }
    return normalized


def _normalize_namespace(
    value: object,
    *,
    source_asset_id: str,
    source_asset_sha256: str,
) -> dict[str, Any]:
    payload = _require_mapping(
        value,
        fields=_NAMESPACE_FIELDS,
        field=f"registry source asset {source_asset_id!r}.source_group_namespace",
    )
    if (
        payload["schema"] != RELATIONSHIP_NAMESPACE_SCHEMA
        or payload["schema_version"] != RELATIONSHIP_NAMESPACE_SCHEMA_VERSION
        or payload["relationship_type"] != "source_group"
        or payload["source_asset_id"] != source_asset_id
        or payload["source_asset_sha256"] != source_asset_sha256
        or payload["cross_lane_sharing_authorized"] is not True
    ):
        raise EditingV2CandidateProvenanceBridgeError(
            f"registry source asset {source_asset_id!r}.source_group_namespace "
            "must be the exact cross-lane source-group namespace for that asset"
        )
    return {
        "schema": RELATIONSHIP_NAMESPACE_SCHEMA,
        "schema_version": RELATIONSHIP_NAMESPACE_SCHEMA_VERSION,
        "relationship_type": "source_group",
        "namespace_id": _require_text(
            payload["namespace_id"],
            field=f"registry source asset {source_asset_id!r}.namespace_id",
        ),
        "source_asset_id": source_asset_id,
        "source_asset_sha256": source_asset_sha256,
        "definition_id": _require_text(
            payload["definition_id"],
            field=f"registry source asset {source_asset_id!r}.definition_id",
        ),
        "implementation_sha256": _require_sha256(
            payload["implementation_sha256"],
            field=f"registry source asset {source_asset_id!r}.implementation_sha256",
        ),
        "cross_lane_sharing_authorized": True,
    }


def _normalize_source_assets(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise EditingV2CandidateProvenanceBridgeError(
            "registry.source_assets must be a nonempty list"
        )
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        payload = _require_mapping(
            item,
            fields=_SOURCE_ASSET_FIELDS,
            field=f"registry.source_assets[{index}]",
        )
        source_id = _require_text(
            payload["source_asset_id"],
            field=f"registry.source_assets[{index}].source_asset_id",
        )
        source_sha = _require_sha256(
            payload["source_asset_sha256"],
            field=f"registry.source_assets[{index}].source_asset_sha256",
        )
        normalized.append(
            {
                "source_asset_id": source_id,
                "source_asset_path": _require_text(
                    payload["source_asset_path"],
                    field=f"registry.source_assets[{index}].source_asset_path",
                ),
                "source_asset_sha256": source_sha,
                "source_version": _require_text(
                    payload["source_version"],
                    field=f"registry.source_assets[{index}].source_version",
                ),
                "access_basis": _require_text(
                    payload["access_basis"],
                    field=f"registry.source_assets[{index}].access_basis",
                ),
                "source_group_namespace": _normalize_namespace(
                    payload["source_group_namespace"],
                    source_asset_id=source_id,
                    source_asset_sha256=source_sha,
                ),
            }
        )
    normalized.sort(key=lambda item: item["source_asset_id"])
    ids = [item["source_asset_id"] for item in normalized]
    if len(ids) != len(set(ids)):
        raise EditingV2CandidateProvenanceBridgeError(
            "registry.source_assets repeats a source_asset_id"
        )
    return normalized


def _normalize_evidence_identity(value: object) -> dict[str, Any]:
    payload = _require_mapping(
        value,
        fields=_EVIDENCE_IDENTITY_FIELDS,
        field="registry.evidence_identity",
    )
    source_fields = _require_sorted_texts(
        payload["source_record_id_fields"],
        field="registry.evidence_identity.source_record_id_fields",
        nonempty=True,
    )
    raw_component_fields = _require_mapping(
        payload["component_reference_fields"],
        fields=set(EVIDENCE_COMPONENTS),
        field="registry.evidence_identity.component_reference_fields",
    )
    component_fields = {
        component: list(
            _require_sorted_texts(
                raw_component_fields[component],
                field=(
                    f"registry.evidence_identity.component_reference_fields.{component}"
                ),
                nonempty=True,
            )
        )
        for component in EVIDENCE_COMPONENTS
    }
    body = {
        "definition_id": _require_text(
            payload["definition_id"],
            field="registry.evidence_identity.definition_id",
        ),
        "implementation_sha256": _require_sha256(
            payload["implementation_sha256"],
            field="registry.evidence_identity.implementation_sha256",
        ),
        "source_record_id_fields": list(source_fields),
        "component_reference_fields": component_fields,
    }
    supplied = _require_sha256(
        payload["evidence_identity_sha256"],
        field="registry.evidence_identity.evidence_identity_sha256",
    )
    if supplied != canonical_sha256(body):
        raise EditingV2CandidateProvenanceBridgeError(
            "registry.evidence_identity semantic SHA-256 disagrees"
        )
    return {**body, "evidence_identity_sha256": supplied}


def _normalize_registry(
    registry: object,
    *,
    candidate_materialization: Mapping[str, Any],
    contract_identity: Mapping[str, Any],
    materialization: Mapping[str, Any],
    editing_corpus_contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _require_mapping(
        registry,
        fields=_REGISTRY_FIELDS,
        field="candidate provenance registry",
    )
    if (
        payload["schema"] != PROVENANCE_REGISTRY_SCHEMA
        or payload["schema_version"] != PROVENANCE_REGISTRY_SCHEMA_VERSION
        or payload["status"] != PROVENANCE_REGISTRY_STATUS
        or payload["training_authorized"] is not False
    ):
        raise EditingV2CandidateProvenanceBridgeError(
            "candidate provenance registry identity or authority is invalid"
        )
    registered_materialization = dict(
        _require_mapping(
            payload["candidate_materialization"],
            fields=_CANDIDATE_MATERIALIZATION_FIELDS,
            field="registry.candidate_materialization",
        )
    )
    for name in _CANDIDATE_MATERIALIZATION_FIELDS:
        _require_sha256(
            registered_materialization[name],
            field=f"registry.candidate_materialization.{name}",
        )
    if registered_materialization != dict(candidate_materialization):
        raise EditingV2CandidateProvenanceBridgeError(
            "registry.candidate_materialization disagrees with the exact candidate materialization"
        )
    registered_contract = dict(
        _require_mapping(
            payload["editing_corpus_contract"],
            fields=_CONTRACT_FIELDS,
            field="registry.editing_corpus_contract",
        )
    )
    for name in ("file_sha256", "semantic_sha256"):
        _require_sha256(
            registered_contract[name],
            field=f"registry.editing_corpus_contract.{name}",
        )
    _require_text(
        registered_contract["contract_id"],
        field="registry.editing_corpus_contract.contract_id",
    )
    if registered_contract != dict(contract_identity):
        raise EditingV2CandidateProvenanceBridgeError(
            "registry.editing_corpus_contract disagrees with the exact contract"
        )

    compiler = dict(
        _require_mapping(
            payload["compiler_identity"],
            fields=_COMPILER_FIELDS,
            field="registry.compiler_identity",
        )
    )
    for name in _COMPILER_FIELDS:
        compiler[name] = _require_sha256(
            compiler[name], field=f"registry.compiler_identity.{name}"
        )
    policies = dict(
        _require_mapping(
            payload["policy_bindings"],
            fields=_POLICY_FIELDS,
            field="registry.policy_bindings",
        )
    )
    for name in _POLICY_FIELDS:
        policies[name] = _require_sha256(
            policies[name], field=f"registry.policy_bindings.{name}"
        )
    evidence_identity = _normalize_evidence_identity(payload["evidence_identity"])
    identity_definitions = _normalize_identity_definitions(
        payload["identity_definitions"]
    )
    # Reuse the production census validator so the bridge cannot drift from the
    # exact identity-definition contract expected by schema-V4 rows.
    default_split_census_policy(
        editing_corpus_contract,
        identity_definitions=identity_definitions,
    )
    sources = _normalize_source_assets(payload["source_assets"])
    materialized_sources = sorted(
        (
            {
                "source_asset_id": source.get("source_asset_id"),
                "source_asset_path": source.get("source_asset_path"),
                "source_asset_sha256": source.get("source_asset_sha256"),
            }
            for source in materialization.get("sources", [])
            if isinstance(source, Mapping)
        ),
        key=lambda item: str(item["source_asset_id"]),
    )
    registered_sources = [
        {
            "source_asset_id": source["source_asset_id"],
            "source_asset_path": source["source_asset_path"],
            "source_asset_sha256": source["source_asset_sha256"],
        }
        for source in sources
    ]
    if not materialized_sources or materialized_sources != registered_sources:
        raise EditingV2CandidateProvenanceBridgeError(
            "registry.source_assets must bind every materialized source asset exactly; "
            "required fields are source_asset_id, source_asset_path, source_asset_sha256, "
            "source_version, access_basis, and source_group_namespace"
        )

    body = {
        "schema": PROVENANCE_REGISTRY_SCHEMA,
        "schema_version": PROVENANCE_REGISTRY_SCHEMA_VERSION,
        "status": PROVENANCE_REGISTRY_STATUS,
        "training_authorized": False,
        "candidate_materialization": registered_materialization,
        "editing_corpus_contract": registered_contract,
        "compiler_identity": compiler,
        "policy_bindings": policies,
        "evidence_identity": evidence_identity,
        "identity_definitions": identity_definitions,
        "source_assets": sources,
    }
    supplied = _require_sha256(
        payload["registry_sha256"], field="registry.registry_sha256"
    )
    if supplied != canonical_sha256(body):
        raise EditingV2CandidateProvenanceBridgeError(
            "candidate provenance registry semantic SHA-256 disagrees"
        )
    return {**body, "registry_sha256": supplied}


def build_candidate_provenance_registry(
    *,
    candidate_root: str | Path,
    editing_corpus_contract_path: str | Path,
    compiler_identity: Mapping[str, Any],
    policy_bindings: Mapping[str, Any],
    evidence_identity: Mapping[str, Any],
    identity_definitions: Mapping[str, Any],
    source_assets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a complete self-hashed registry from explicit frozen identities.

    This function does not infer source versions, access bases, compiler
    identities, policy hashes, evidence selectors, or source-group namespaces.
    Omitting any such field fails with its exact registry path.
    """

    candidate_path = Path(candidate_root)
    contract_path = Path(editing_corpus_contract_path)
    materialization = validate_packed_candidate_materialization(candidate_path)
    contract = load_editing_corpus_contract(contract_path)
    candidate_identity = _materialization_identity(candidate_path, materialization)
    contract_identity = {
        "contract_id": contract["contract_id"],
        "file_sha256": file_sha256(contract_path),
        "semantic_sha256": canonical_sha256(contract),
    }
    evidence_payload = dict(evidence_identity)
    if "evidence_identity_sha256" not in evidence_payload:
        evidence_payload["evidence_identity_sha256"] = canonical_sha256(
            evidence_payload
        )
    body = {
        "schema": PROVENANCE_REGISTRY_SCHEMA,
        "schema_version": PROVENANCE_REGISTRY_SCHEMA_VERSION,
        "status": PROVENANCE_REGISTRY_STATUS,
        "training_authorized": False,
        "candidate_materialization": candidate_identity,
        "editing_corpus_contract": contract_identity,
        "compiler_identity": dict(compiler_identity),
        "policy_bindings": dict(policy_bindings),
        "evidence_identity": evidence_payload,
        "identity_definitions": dict(identity_definitions),
        "source_assets": [dict(source) for source in source_assets],
    }
    candidate = {**body, "registry_sha256": canonical_sha256(body)}
    return _normalize_registry(
        candidate,
        candidate_materialization=candidate_identity,
        contract_identity=contract_identity,
        materialization=materialization,
        editing_corpus_contract=contract,
    )


def write_candidate_provenance_registry(
    registry: Mapping[str, Any],
    path: str | Path,
) -> None:
    """Atomically persist an already validated canonical registry value."""

    target = Path(path)
    if target.exists():
        raise EditingV2CandidateProvenanceBridgeError(
            f"registry output already exists; refusing overwrite: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".staging",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(registry, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _lookup_path(
    row: Mapping[str, Any], selector: str, *, field: str
) -> tuple[str, ...]:
    value: Any = row
    for component in selector.split("."):
        if not component or not isinstance(value, Mapping) or component not in value:
            raise EditingV2CandidateProvenanceBridgeError(
                f"{field} selector {selector!r} is absent from the candidate header"
            )
        value = value[component]
    if isinstance(value, str):
        return (_require_text(value, field=f"{field} selector {selector!r}"),)
    if isinstance(value, list):
        return _require_sorted_texts(
            value,
            field=f"{field} selector {selector!r}",
            nonempty=True,
        )
    raise EditingV2CandidateProvenanceBridgeError(
        f"{field} selector {selector!r} must resolve to text or a sorted text list"
    )


def _resolve_references(
    row: Mapping[str, Any],
    selectors: Sequence[str],
    *,
    field: str,
) -> tuple[str, ...]:
    values = {
        item
        for selector in selectors
        for item in _lookup_path(row, selector, field=field)
    }
    if not values:
        raise EditingV2CandidateProvenanceBridgeError(
            f"{field} resolved no provenance references"
        )
    return tuple(sorted(values))


def _validate_header(
    row: object,
    *,
    expected_index: int,
    registry_sources: Mapping[str, Mapping[str, Any]],
    materializer_implementation_sha256: str,
    routing_policy_sha256: str,
) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise EditingV2CandidateProvenanceBridgeError(
            f"candidate header {expected_index} must be an object"
        )
    if (
        row.get("schema") != CANDIDATE_HEADER_SCHEMA
        or row.get("schema_version") != CANDIDATE_HEADER_SCHEMA_VERSION
        or row.get("status") != CANDIDATE_HEADER_STATUS
        or row.get("training_authorized") is not False
        or row.get("attempt_index") != expected_index
    ):
        raise EditingV2CandidateProvenanceBridgeError(
            f"candidate header {expected_index} identity, authority, or order disagrees"
        )
    supplied = _require_sha256(
        row.get("row_sha256"), field=f"candidate header {expected_index}.row_sha256"
    )
    body = {key: value for key, value in row.items() if key != "row_sha256"}
    if supplied != canonical_sha256(body):
        raise EditingV2CandidateProvenanceBridgeError(
            f"candidate header {expected_index} self-hash disagrees"
        )
    source = row.get("source_asset")
    if not isinstance(source, Mapping):
        raise EditingV2CandidateProvenanceBridgeError(
            f"candidate header {expected_index}.source_asset must be an object"
        )
    source_id = _require_text(
        source.get("source_asset_id"),
        field=f"candidate header {expected_index}.source_asset_id",
    )
    registered = registry_sources.get(source_id)
    if registered is None or any(
        source.get(name) != registered[name]
        for name in ("source_asset_id", "source_asset_path", "source_asset_sha256")
    ):
        raise EditingV2CandidateProvenanceBridgeError(
            f"candidate header {expected_index} source asset is absent or disagrees "
            "with the frozen registry"
        )
    disposition = row.get("disposition")
    resolution = row.get("lane_resolution")
    if disposition == "routed_candidate":
        if not isinstance(resolution, Mapping) or row.get("rejection") is not None:
            raise EditingV2CandidateProvenanceBridgeError(
                f"routed candidate header {expected_index} lacks its exact lane receipt"
            )
        if (
            resolution.get("data_lane") != row.get("data_lane")
            or resolution.get("evidence_profile_id") != row.get("evidence_profile_id")
            or resolution.get("resolver_identity_sha256")
            != materializer_implementation_sha256
            or resolution.get("routing_policy_sha256") != routing_policy_sha256
            or row.get("routing_policy_sha256") != routing_policy_sha256
        ):
            raise EditingV2CandidateProvenanceBridgeError(
                f"routed candidate header {expected_index} lane or resolver identity disagrees"
            )
    elif disposition == "rejected_candidate":
        if resolution is not None or row.get("data_lane") is not None:
            raise EditingV2CandidateProvenanceBridgeError(
                f"rejected candidate header {expected_index} cannot carry a routed lane"
            )
        rejection = row.get("rejection")
        if not isinstance(rejection, Mapping):
            raise EditingV2CandidateProvenanceBridgeError(
                f"rejected candidate header {expected_index} lacks a rejection receipt"
            )
    else:
        raise EditingV2CandidateProvenanceBridgeError(
            f"candidate header {expected_index} has an unsupported disposition"
        )
    return row


def _iter_headers(
    rows_path: Path,
    *,
    registry_sources: Mapping[str, Mapping[str, Any]],
    materializer_implementation_sha256: str,
    routing_policy_sha256: str,
    max_row_bytes: int,
) -> Iterable[tuple[Mapping[str, Any], bytes]]:
    with rows_path.open("rb") as handle:
        expected_index = 0
        while True:
            raw_line = handle.readline(max_row_bytes + 1)
            if not raw_line:
                break
            if len(raw_line) > max_row_bytes or not raw_line.endswith(b"\n"):
                raise EditingV2CandidateProvenanceBridgeError(
                    f"candidate header {expected_index} exceeds the row bound or lacks newline"
                )
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EditingV2CandidateProvenanceBridgeError(
                    f"candidate header {expected_index} is invalid JSON"
                ) from error
            yield (
                _validate_header(
                    row,
                    expected_index=expected_index,
                    registry_sources=registry_sources,
                    materializer_implementation_sha256=materializer_implementation_sha256,
                    routing_policy_sha256=routing_policy_sha256,
                ),
                raw_line,
            )
            expected_index += 1


def _evidence_for_header(
    row: Mapping[str, Any],
    *,
    evidence_identity: Mapping[str, Any],
) -> CandidateEvidence:
    component_kinds = _require_mapping(
        row.get("evidence_components"),
        fields=set(EVIDENCE_COMPONENTS),
        field=f"candidate {row.get('candidate_id')!r}.evidence_components",
    )
    selectors = evidence_identity["component_reference_fields"]
    components: dict[str, CandidateEvidenceComponent] = {}
    for component in EVIDENCE_COMPONENTS:
        kind = _require_text(
            component_kinds[component],
            field=f"candidate {row.get('candidate_id')!r}.evidence_components.{component}",
        )
        references = _resolve_references(
            row,
            selectors[component],
            field=f"candidate {row.get('candidate_id')!r}.evidence.{component}",
        )
        evidence_sha256 = canonical_sha256(
            {
                "evidence_identity_sha256": evidence_identity[
                    "evidence_identity_sha256"
                ],
                "candidate_header_row_sha256": row["row_sha256"],
                "component": component,
                "kind": kind,
                "reference_ids": list(references),
            }
        )
        components[component] = CandidateEvidenceComponent(
            kind=kind,
            reference_ids=references,
            evidence_sha256=evidence_sha256,
        )
    return CandidateEvidence(**components)


def _attempt_for_header(
    row: Mapping[str, Any],
    *,
    policies: CandidatePolicyBindings,
    evidence_identity: Mapping[str, Any],
) -> CandidateAuditAttempt:
    source_record_ids = _resolve_references(
        row,
        evidence_identity["source_record_id_fields"],
        field=f"candidate {row.get('candidate_id')!r}.source_record_ids",
    )
    evidence = _evidence_for_header(row, evidence_identity=evidence_identity)
    if row["disposition"] == "routed_candidate":
        resolution = row["lane_resolution"]
        return CandidateAuditAttempt(
            attempt_index=row["attempt_index"],
            candidate_id=row["candidate_id"],
            candidate_payload_sha256=row["candidate_payload_sha256"],
            source_record_ids=source_record_ids,
            policy_bindings=policies,
            lane_resolution=CandidateLaneResolution(
                data_lane=resolution["data_lane"],
                evidence_profile_id=resolution["evidence_profile_id"],
                resolver_identity_sha256=resolution["resolver_identity_sha256"],
                routing_policy_sha256=resolution["routing_policy_sha256"],
                receipt_sha256=resolution["receipt_sha256"],
            ),
            evidence=evidence,
            accepted_trace_id=row["packed_address"]["trace_id"],
        )
    rejection = row["rejection"]
    return CandidateAuditAttempt(
        attempt_index=row["attempt_index"],
        candidate_id=row["candidate_id"],
        candidate_payload_sha256=row["candidate_payload_sha256"],
        source_record_ids=source_record_ids,
        policy_bindings=policies,
        evidence=evidence,
        rejection_code=rejection["code"],
        rejection_detail=json.dumps(
            rejection,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
    )


def _split_row(
    header: Mapping[str, Any],
    ledger_row: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    source_registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source_id = header["source_asset"]["source_asset_id"]
    source = source_registry[source_id]
    groups = header["groups"]
    return {
        "schema": CANDIDATE_ROW_SCHEMA,
        "schema_version": CANDIDATE_ROW_SCHEMA_VERSION,
        "candidate_id": header["candidate_id"],
        "candidate_ledger_row_sha256": ledger_row["row_sha256"],
        "candidate_envelope_sha256": header["candidate_payload_sha256"],
        "data_lane": header["data_lane"],
        "evidence_profile_id": header["evidence_profile_id"],
        "evidence_components": {
            component: header["evidence_components"][component]
            for component in sorted(EVIDENCE_COMPONENTS)
        },
        "mass_units": 1,
        "molecule_ids": sorted(groups["molecule_ids"]),
        "partition_scaffold_ids": sorted(groups["scaffold_ids"]),
        "source_group_ids": sorted(groups["source_group_ids"]),
        "source_group_namespace": source["source_group_namespace"],
        "identity_definitions": registry["identity_definitions"],
        "shared_prefix_branch_group_id": None,
        "alternative_route_group_id": None,
        "inverse_pair_group_id": None,
        "correction_group_id": None,
        "constraint_compatible_alternative_route_group_id": None,
        "relationship_namespaces": {
            relationship: None for relationship in _RELATIONSHIP_TYPES
        },
        "document_group_id": None,
        "document_provenance": None,
        "series_group_id": None,
        "series_provenance": None,
        "transformation_signature": None,
        "transformation_definition": None,
        "metadata": {
            "attempt_index": header["attempt_index"],
            "source_kind": header["source_kind"],
            "source_asset_id": source_id,
            "source_asset_sha256": source["source_asset_sha256"],
            "packed_address_sha256": header["packed_address"]["address_sha256"],
            "trace_id": header["packed_address"]["trace_id"],
            "candidate_header_row_sha256": header["row_sha256"],
            "candidate_payload_sha256": header["candidate_payload_sha256"],
            "candidate_audit_ledger_row_sha256": ledger_row["row_sha256"],
            "provenance_registry_sha256": registry["registry_sha256"],
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("wb") as handle:
        handle.write(
            json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _source_stream(
    *,
    candidate_identity: Mapping[str, Any],
    registry_file_sha256: str,
    registry_sha256: str,
    ledger_file_sha256: str,
    ledger_sha256: str,
    ledger_rows_sha256: str,
    split_file_sha256: str,
    split_semantic_sha256: str,
    split_rows: int,
) -> dict[str, Any]:
    body = {
        "schema": SOURCE_STREAM_SCHEMA,
        "schema_version": SOURCE_STREAM_SCHEMA_VERSION,
        "nonempty_jsonl_rows": split_rows,
        "candidate_materialization": dict(candidate_identity),
        "provenance_registry": {
            "file_sha256": registry_file_sha256,
            "registry_sha256": registry_sha256,
        },
        "candidate_audit_ledger": {
            "file_sha256": ledger_file_sha256,
            "semantic_sha256": ledger_sha256,
            "rows_sha256": ledger_rows_sha256,
        },
        "split_candidates": {
            "file_sha256": split_file_sha256,
            "semantic_sha256": split_semantic_sha256,
        },
    }
    return {**body, "source_stream_sha256": canonical_sha256(body)}


def materialize_candidate_provenance_bridge(
    *,
    candidate_root: str | Path,
    provenance_registry_path: str | Path,
    editing_corpus_contract_path: str | Path,
    output_dir: str | Path,
    max_row_bytes: int = 2 * 1024 * 1024,
) -> dict[str, Any]:
    """Atomically emit a complete formal ledger and routed-only split rows."""

    if type(max_row_bytes) is not int or max_row_bytes <= 0:
        raise EditingV2CandidateProvenanceBridgeError(
            "max_row_bytes must be a positive integer"
        )
    candidate_path = Path(candidate_root)
    registry_path = Path(provenance_registry_path)
    contract_path = Path(editing_corpus_contract_path)
    materialization = validate_packed_candidate_materialization(
        candidate_path,
        max_row_bytes=max_row_bytes,
    )
    contract = load_editing_corpus_contract(contract_path)
    candidate_identity = _materialization_identity(candidate_path, materialization)
    contract_identity = {
        "contract_id": contract["contract_id"],
        "file_sha256": file_sha256(contract_path),
        "semantic_sha256": canonical_sha256(contract),
    }
    registry = _normalize_registry(
        _load_json(registry_path, field="candidate provenance registry"),
        candidate_materialization=candidate_identity,
        contract_identity=contract_identity,
        materialization=materialization,
        editing_corpus_contract=contract,
    )
    source_registry = {
        source["source_asset_id"]: source for source in registry["source_assets"]
    }
    implementation = materialization.get("implementation")
    inputs = materialization.get("inputs")
    if not isinstance(implementation, Mapping) or not isinstance(inputs, Mapping):
        raise EditingV2CandidateProvenanceBridgeError(
            "candidate materialization implementation or inputs identity is absent"
        )
    routing_input = inputs.get("routing_policy")
    source_input = inputs.get("source_manifest")
    if not isinstance(routing_input, Mapping) or not isinstance(source_input, Mapping):
        raise EditingV2CandidateProvenanceBridgeError(
            "candidate materialization routing-policy or source-manifest identity is absent"
        )
    materializer_sha = _require_sha256(
        implementation.get("file_sha256"),
        field="candidate materialization implementation.file_sha256",
    )
    routing_sha = _require_sha256(
        routing_input.get("semantic_sha256"),
        field="candidate materialization routing_policy.semantic_sha256",
    )
    source_identity = CandidateSourceIdentity(
        manifest_file_sha256=_require_sha256(
            source_input.get("file_sha256"),
            field="candidate materialization source_manifest.file_sha256",
        ),
        manifest_sha256=_require_sha256(
            source_input.get("manifest_sha256"),
            field="candidate materialization source_manifest.manifest_sha256",
        ),
    )
    compiler_identity = CandidateCompilerIdentity(**registry["compiler_identity"])
    policies = CandidatePolicyBindings(**registry["policy_bindings"])

    attempts: list[CandidateAuditAttempt] = []
    header_file_digest = hashlib.sha256()
    header_semantic_digest = hashlib.sha256()
    address_stream_digest = hashlib.sha256()
    rows_path = candidate_path / CANDIDATE_ROWS_FILENAME
    for header, raw_line in _iter_headers(
        rows_path,
        registry_sources=source_registry,
        materializer_implementation_sha256=materializer_sha,
        routing_policy_sha256=routing_sha,
        max_row_bytes=max_row_bytes,
    ):
        header_file_digest.update(raw_line)
        header_semantic_digest.update(header["row_sha256"].encode("ascii"))
        header_semantic_digest.update(b"\n")
        address_stream_digest.update(
            header["packed_address"]["address_sha256"].encode("ascii")
        )
        address_stream_digest.update(b"\n")
        attempts.append(
            _attempt_for_header(
                header,
                policies=policies,
                evidence_identity=registry["evidence_identity"],
            )
        )
    if (
        header_file_digest.hexdigest() != candidate_identity["rows_file_sha256"]
        or header_semantic_digest.hexdigest()
        != candidate_identity["rows_semantic_sha256"]
        or address_stream_digest.hexdigest()
        != candidate_identity["address_stream_sha256"]
    ):
        raise EditingV2CandidateProvenanceBridgeError(
            "candidate headers changed after materialization validation"
        )
    expected_attempts = materialization.get("rows", {}).get("totals", {}).get("rows")
    if type(expected_attempts) is not int or len(attempts) != expected_attempts:
        raise EditingV2CandidateProvenanceBridgeError(
            "candidate header count disagrees with the materialization census"
        )
    attempt_stream_sha256 = candidate_attempt_stream_sha256(attempts)
    ledger = build_candidate_audit_ledger(
        contract=contract,
        contract_file_sha256=contract_identity["file_sha256"],
        source_identity=source_identity,
        compiler_identity=compiler_identity,
        attempts=attempts,
        expected_attempt_count=len(attempts),
        expected_attempt_stream_sha256=attempt_stream_sha256,
    )

    target = Path(output_dir)
    if target.exists():
        raise EditingV2CandidateProvenanceBridgeError(
            f"bridge output directory already exists; refusing overwrite: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".staging",
        )
    )
    published = False
    try:
        ledger_path = staging / AUDIT_LEDGER_FILENAME
        split_path = staging / SPLIT_ROWS_FILENAME
        _write_json(ledger_path, ledger)
        split_file_digest = hashlib.sha256()
        split_semantic_digest = hashlib.sha256()
        split_rows = 0
        ledger_rows = iter(ledger["rows"])
        with split_path.open("wb") as split_handle:
            for header, _ in _iter_headers(
                rows_path,
                registry_sources=source_registry,
                materializer_implementation_sha256=materializer_sha,
                routing_policy_sha256=routing_sha,
                max_row_bytes=max_row_bytes,
            ):
                audit_row = next(ledger_rows)
                if audit_row["candidate_id"] != header["candidate_id"]:
                    raise EditingV2CandidateProvenanceBridgeError(
                        "formal ledger order disagrees with the candidate-header stream"
                    )
                if audit_row["disposition"] != "compiled_candidate":
                    continue
                split_row = _split_row(
                    header,
                    audit_row,
                    registry=registry,
                    source_registry=source_registry,
                )
                encoded = canonical_json_bytes(split_row) + b"\n"
                split_handle.write(encoded)
                split_file_digest.update(encoded)
                split_semantic_digest.update(
                    canonical_sha256(split_row).encode("ascii")
                )
                split_semantic_digest.update(b"\n")
                split_rows += 1
            try:
                next(ledger_rows)
            except StopIteration:
                pass
            else:
                raise EditingV2CandidateProvenanceBridgeError(
                    "formal ledger contains rows absent from the candidate-header stream"
                )
            split_handle.flush()
            os.fsync(split_handle.fileno())
        if split_rows == 0:
            raise EditingV2CandidateProvenanceBridgeError(
                "candidate bridge produced no routed split candidates"
            )
        ledger_file_sha = file_sha256(ledger_path)
        split_file_sha = file_sha256(split_path)
        if split_file_sha != split_file_digest.hexdigest():
            raise EditingV2CandidateProvenanceBridgeError(
                "split-candidate bytes changed during materialization"
            )
        stream = _source_stream(
            candidate_identity=candidate_identity,
            registry_file_sha256=file_sha256(registry_path),
            registry_sha256=registry["registry_sha256"],
            ledger_file_sha256=ledger_file_sha,
            ledger_sha256=ledger["ledger_sha256"],
            ledger_rows_sha256=ledger["rows_sha256"],
            split_file_sha256=split_file_sha,
            split_semantic_sha256=split_semantic_digest.hexdigest(),
            split_rows=split_rows,
        )
        outputs = {
            "candidate_audit_ledger": {
                "relative_path": AUDIT_LEDGER_FILENAME,
                "file_sha256": ledger_file_sha,
                "semantic_sha256": ledger["ledger_sha256"],
            },
            "split_candidates": {
                "relative_path": SPLIT_ROWS_FILENAME,
                "file_sha256": split_file_sha,
                "semantic_sha256": split_semantic_digest.hexdigest(),
            },
        }
        body = {
            "schema": PROVENANCE_BRIDGE_SCHEMA,
            "schema_version": PROVENANCE_BRIDGE_SCHEMA_VERSION,
            "status": PROVENANCE_BRIDGE_STATUS,
            "training_authorized": False,
            "blockers": list(REQUIRED_BLOCKERS),
            "source_stream": stream,
            "counts": {
                "attempted": len(attempts),
                "routed": ledger["counts"]["compiled_candidates"],
                "rejected": ledger["counts"]["rejected_candidates"],
                "split_rows": split_rows,
            },
            "outputs": outputs,
        }
        manifest = {**body, "manifest_sha256": canonical_sha256(body)}
        _write_json(staging / BRIDGE_MANIFEST_FILENAME, manifest)
        os.replace(staging, target)
        published = True
        return manifest
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def validate_candidate_provenance_bridge(
    output_dir: str | Path,
    *,
    candidate_root: str | Path,
    provenance_registry_path: str | Path,
    editing_corpus_contract_path: str | Path,
) -> Mapping[str, Any]:
    """Reopen a published bridge and validate all physical/semantic bindings."""

    root = Path(output_dir)
    manifest = _require_mapping(
        _load_json(
            root / BRIDGE_MANIFEST_FILENAME, field="candidate provenance bridge"
        ),
        fields=_BRIDGE_FIELDS,
        field="candidate provenance bridge",
    )
    if (
        manifest["schema"] != PROVENANCE_BRIDGE_SCHEMA
        or manifest["schema_version"] != PROVENANCE_BRIDGE_SCHEMA_VERSION
        or manifest["status"] != PROVENANCE_BRIDGE_STATUS
        or manifest["training_authorized"] is not False
        or tuple(manifest["blockers"]) != REQUIRED_BLOCKERS
    ):
        raise EditingV2CandidateProvenanceBridgeError(
            "candidate provenance bridge identity, authority, or blockers disagree"
        )
    supplied = _require_sha256(
        manifest["manifest_sha256"], field="bridge.manifest_sha256"
    )
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if supplied != canonical_sha256(body):
        raise EditingV2CandidateProvenanceBridgeError(
            "candidate provenance bridge semantic SHA-256 disagrees"
        )
    counts = _require_mapping(
        manifest["counts"], fields=_BRIDGE_COUNT_FIELDS, field="bridge.counts"
    )
    for name in _BRIDGE_COUNT_FIELDS:
        _require_nonnegative_int(counts[name], field=f"bridge.counts.{name}")
    if (
        counts["attempted"] != counts["routed"] + counts["rejected"]
        or counts["routed"] != counts["split_rows"]
        or counts["attempted"] <= 0
    ):
        raise EditingV2CandidateProvenanceBridgeError("bridge counts are inconsistent")
    outputs = _require_mapping(
        manifest["outputs"],
        fields=_BRIDGE_OUTPUT_FIELDS,
        field="bridge.outputs",
    )
    for name in _BRIDGE_OUTPUT_FIELDS:
        output = _require_mapping(
            outputs[name],
            fields=_OUTPUT_IDENTITY_FIELDS,
            field=f"bridge.outputs.{name}",
        )
        expected_name = (
            AUDIT_LEDGER_FILENAME
            if name == "candidate_audit_ledger"
            else SPLIT_ROWS_FILENAME
        )
        if output["relative_path"] != expected_name:
            raise EditingV2CandidateProvenanceBridgeError(
                f"bridge.outputs.{name}.relative_path disagrees"
            )
        if file_sha256(root / expected_name) != output["file_sha256"]:
            raise EditingV2CandidateProvenanceBridgeError(
                f"bridge.outputs.{name} physical SHA-256 disagrees"
            )

    candidate_path = Path(candidate_root)
    contract_path = Path(editing_corpus_contract_path)
    registry_path = Path(provenance_registry_path)
    materialization = validate_packed_candidate_materialization(candidate_path)
    contract = load_editing_corpus_contract(contract_path)
    candidate_identity = _materialization_identity(candidate_path, materialization)
    contract_identity = {
        "contract_id": contract["contract_id"],
        "file_sha256": file_sha256(contract_path),
        "semantic_sha256": canonical_sha256(contract),
    }
    registry = _normalize_registry(
        _load_json(registry_path, field="candidate provenance registry"),
        candidate_materialization=candidate_identity,
        contract_identity=contract_identity,
        materialization=materialization,
        editing_corpus_contract=contract,
    )
    stream = _require_mapping(
        manifest["source_stream"],
        fields=_SOURCE_STREAM_FIELDS,
        field="bridge.source_stream",
    )
    stream_body = {
        key: value for key, value in stream.items() if key != "source_stream_sha256"
    }
    if (
        stream["schema"] != SOURCE_STREAM_SCHEMA
        or stream["schema_version"] != SOURCE_STREAM_SCHEMA_VERSION
        or stream["candidate_materialization"] != candidate_identity
        or stream["nonempty_jsonl_rows"] != counts["split_rows"]
        or stream["source_stream_sha256"] != canonical_sha256(stream_body)
    ):
        raise EditingV2CandidateProvenanceBridgeError(
            "bridge.source_stream identity or candidate materialization disagrees"
        )
    stream_registry = _require_mapping(
        stream["provenance_registry"],
        fields=_SOURCE_STREAM_REGISTRY_FIELDS,
        field="bridge.source_stream.provenance_registry",
    )
    if stream_registry != {
        "file_sha256": file_sha256(registry_path),
        "registry_sha256": registry["registry_sha256"],
    }:
        raise EditingV2CandidateProvenanceBridgeError(
            "bridge source-stream registry identity disagrees"
        )
    ledger = _load_json(root / AUDIT_LEDGER_FILENAME, field="candidate audit ledger")
    stream_ledger = _require_mapping(
        stream["candidate_audit_ledger"],
        fields=_SOURCE_STREAM_LEDGER_FIELDS,
        field="bridge.source_stream.candidate_audit_ledger",
    )
    if stream_ledger != {
        "file_sha256": outputs["candidate_audit_ledger"]["file_sha256"],
        "semantic_sha256": ledger.get("ledger_sha256"),
        "rows_sha256": ledger.get("rows_sha256"),
    }:
        raise EditingV2CandidateProvenanceBridgeError(
            "bridge source-stream formal-ledger identity disagrees"
        )
    source_input = materialization["inputs"]["source_manifest"]
    attempts = [
        _attempt_for_header(
            header,
            policies=CandidatePolicyBindings(**registry["policy_bindings"]),
            evidence_identity=registry["evidence_identity"],
        )
        for header, _ in _iter_headers(
            candidate_path / CANDIDATE_ROWS_FILENAME,
            registry_sources={
                source["source_asset_id"]: source
                for source in registry["source_assets"]
            },
            materializer_implementation_sha256=materialization["implementation"][
                "file_sha256"
            ],
            routing_policy_sha256=materialization["inputs"]["routing_policy"][
                "semantic_sha256"
            ],
            max_row_bytes=2 * 1024 * 1024,
        )
    ]
    attempt_sha = candidate_attempt_stream_sha256(attempts)
    validate_candidate_audit_ledger(
        ledger,
        contract=contract,
        contract_file_sha256=contract_identity["file_sha256"],
        expected_source_identity=CandidateSourceIdentity(
            manifest_file_sha256=source_input["file_sha256"],
            manifest_sha256=source_input["manifest_sha256"],
        ),
        expected_compiler_identity=CandidateCompilerIdentity(
            **registry["compiler_identity"]
        ),
        expected_attempt_count=len(attempts),
        expected_attempt_stream_sha256=attempt_sha,
    )
    split_digest = hashlib.sha256()
    split_count = 0
    source_registry = {
        source["source_asset_id"]: source for source in registry["source_assets"]
    }
    with (root / SPLIT_ROWS_FILENAME).open("rb") as split_handle:
        for header, _ in _iter_headers(
            candidate_path / CANDIDATE_ROWS_FILENAME,
            registry_sources=source_registry,
            materializer_implementation_sha256=materialization["implementation"][
                "file_sha256"
            ],
            routing_policy_sha256=materialization["inputs"]["routing_policy"][
                "semantic_sha256"
            ],
            max_row_bytes=2 * 1024 * 1024,
        ):
            audit_row = ledger["rows"][header["attempt_index"]]
            if audit_row["disposition"] != "compiled_candidate":
                continue
            raw_line = split_handle.readline(2 * 1024 * 1024 + 1)
            if not raw_line or len(raw_line) > 2 * 1024 * 1024:
                raise EditingV2CandidateProvenanceBridgeError(
                    "split-candidate stream is truncated or exceeds its row bound"
                )
            if not raw_line.endswith(b"\n"):
                raise EditingV2CandidateProvenanceBridgeError(
                    "split-candidate stream lacks a terminating newline"
                )
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EditingV2CandidateProvenanceBridgeError(
                    f"split candidate {split_count} is invalid JSON"
                ) from error
            expected_row = _split_row(
                header,
                audit_row,
                registry=registry,
                source_registry=source_registry,
            )
            if not isinstance(row, Mapping) or dict(row) != expected_row:
                raise EditingV2CandidateProvenanceBridgeError(
                    f"split candidate {split_count} disagrees with its exact routed "
                    "header, formal ledger row, or frozen registry"
                )
            split_digest.update(canonical_sha256(row).encode("ascii"))
            split_digest.update(b"\n")
            split_count += 1
        if split_handle.read(1):
            raise EditingV2CandidateProvenanceBridgeError(
                "split-candidate stream contains a row without a routed header"
            )
    stream_rows = _require_mapping(
        stream["split_candidates"],
        fields=_SOURCE_STREAM_ROWS_FIELDS,
        field="bridge.source_stream.split_candidates",
    )
    if (
        split_count != counts["split_rows"]
        or stream_rows["file_sha256"] != outputs["split_candidates"]["file_sha256"]
        or stream_rows["semantic_sha256"] != split_digest.hexdigest()
        or outputs["split_candidates"]["semantic_sha256"] != split_digest.hexdigest()
    ):
        raise EditingV2CandidateProvenanceBridgeError(
            "bridge split-candidate count or semantic identity disagrees"
        )
    return manifest


__all__ = [
    "AUDIT_LEDGER_FILENAME",
    "BRIDGE_MANIFEST_FILENAME",
    "PROVENANCE_BRIDGE_SCHEMA",
    "PROVENANCE_BRIDGE_SCHEMA_VERSION",
    "PROVENANCE_REGISTRY_SCHEMA",
    "PROVENANCE_REGISTRY_SCHEMA_VERSION",
    "REGISTRY_FILENAME",
    "SOURCE_STREAM_SCHEMA",
    "SOURCE_STREAM_SCHEMA_VERSION",
    "SPLIT_ROWS_FILENAME",
    "EditingV2CandidateProvenanceBridgeError",
    "build_candidate_provenance_registry",
    "materialize_candidate_provenance_bridge",
    "validate_candidate_provenance_bridge",
    "write_candidate_provenance_registry",
]
