"""Deterministic split-component census for the editing-V2 candidate universe.

This module is intentionally upstream-agnostic.  It consumes a normalized stream
of candidate identity rows, not packed traces, chemistry objects, or one
particular mining ledger.  Each candidate is one indivisible vertex.  Shared
identities create typed relationship edges, and only edge types declared
``hard`` by the hash-bound policy participate in connected components.

The census does not choose split ratios or assign a partition.  It measures the
indivisible component structure that a later, separately frozen split policy
must respect.  ``census_structural_complete`` describes that bounded
computation only and carries no corpus-readiness, split, or training authority.

Schema version 4 binds each vertex to an upstream candidate-ledger row and
candidate envelope, validates the lane/profile/six-component evidence envelope,
and makes source-group isolation an explicit cross-lane hard relationship.
Caller-supplied partition claims remain forbidden. Post-Active8 relationship
types remain diagnostic until a physical receipt resolver exists. The builder
consumes its source iterable once and never materializes the raw input stream.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from compose_v4.data.editing_corpus_contract import (
    EditingCorpusContractError,
    validate_editing_corpus_contract,
    validate_evidence_assignment_envelope,
)

CANDIDATE_ROW_SCHEMA = "compose.editing_v2_split_candidate"
CANDIDATE_ROW_SCHEMA_VERSION = 4
OBSERVED_GROUP_PROVENANCE_SCHEMA = "compose.observed_group_provenance"
OBSERVED_GROUP_PROVENANCE_SCHEMA_VERSION = 2
TRANSFORMATION_DEFINITION_SCHEMA = "compose.transformation_signature_definition"
TRANSFORMATION_DEFINITION_SCHEMA_VERSION = 2
LANE_CONTRACT_IDENTITY_SCHEMA = "compose.editing_v2_lane_contract_identity"
LANE_CONTRACT_IDENTITY_SCHEMA_VERSION = 2
IDENTITY_DEFINITION_CONTRACT_SCHEMA = "compose.editing_v2_identity_definition_contract"
IDENTITY_DEFINITION_CONTRACT_SCHEMA_VERSION = 2
RELATIONSHIP_NAMESPACE_SCHEMA = "compose.editing_v2_relationship_namespace"
RELATIONSHIP_NAMESPACE_SCHEMA_VERSION = 2
IMPLEMENTATION_PROVENANCE_SCHEMA = "compose.editing_v2_split_census_provenance"
IMPLEMENTATION_PROVENANCE_SCHEMA_VERSION = 1
SPLIT_CENSUS_POLICY_SCHEMA = "compose.editing_v2_split_census_policy"
SPLIT_CENSUS_POLICY_SCHEMA_VERSION = 4
SPLIT_CENSUS_SCHEMA = "compose.editing_v2_split_component_census"
SPLIT_CENSUS_SCHEMA_VERSION = 4

PARTITION_ROLES = (
    "train",
    "validation",
    "controller_validation",
    "final_test",
)

EDGE_TYPES = (
    "exact_molecule",
    "partition_scaffold",
    "source_group",
    "document_group",
    "series_group",
    "shared_prefix_branch",
    "alternative_route",
    "inverse_pair",
    "correction",
    "constraint_compatible_alternative_route",
    "transformation_signature",
)
MANDATORY_HARD_EDGE_TYPES = frozenset(
    {
        "exact_molecule",
        "partition_scaffold",
        "source_group",
        "document_group",
        "series_group",
    }
)
RECEIPT_GATED_RELATIONSHIP_EDGE_TYPES = frozenset(
    {
        "shared_prefix_branch",
        "alternative_route",
        "inverse_pair",
        "correction",
        "constraint_compatible_alternative_route",
    }
)
REQUIRED_AUTHORITY_BLOCKERS = (
    "external_authority."
    "authoritative_trace_derived_record_membership_classification:"
    "unresolved_no_go_pending_trace_derived_predicate_receipts",
    "external_authority.physical_relationship_receipt_resolution:"
    "unresolved_no_go_pending_physical_receipt_resolver",
    "split_assignment.not_performed_by_component_census",
)
EDGE_MODES = frozenset({"hard", "diagnostic"})
IDENTITY_TYPES = ("exact_molecule", "partition_scaffold", "source_group")
SIMPLE_RELATIONSHIP_FIELDS = {
    "shared_prefix_branch": "shared_prefix_branch_group_id",
    "alternative_route": "alternative_route_group_id",
    "inverse_pair": "inverse_pair_group_id",
    "correction": "correction_group_id",
    "constraint_compatible_alternative_route": (
        "constraint_compatible_alternative_route_group_id"
    ),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_ROW_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "candidate_id",
        "candidate_ledger_row_sha256",
        "candidate_envelope_sha256",
        "data_lane",
        "evidence_profile_id",
        "evidence_components",
        "mass_units",
        "molecule_ids",
        "partition_scaffold_ids",
        "source_group_ids",
        "source_group_namespace",
        "identity_definitions",
        "shared_prefix_branch_group_id",
        "alternative_route_group_id",
        "inverse_pair_group_id",
        "correction_group_id",
        "constraint_compatible_alternative_route_group_id",
        "relationship_namespaces",
        "document_group_id",
        "document_provenance",
        "series_group_id",
        "series_provenance",
        "transformation_signature",
        "transformation_definition",
        "metadata",
    }
)
_OBSERVED_PROVENANCE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "provenance_kind",
        "group_namespace",
        "source_asset_id",
        "source_asset_sha256",
        "source_version",
        "access_basis",
        "source_row_ids",
    }
)
_TRANSFORMATION_DEFINITION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "definition_id",
        "implementation_sha256",
        "derivation_kind",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "policy_id",
        "partition_roles",
        "lane_contract_identity",
        "identity_definitions",
        "edge_modes",
        "transformation_hard_authorized",
    }
)
_LANE_CONTRACT_IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "editing_corpus_contract_id",
        "editing_corpus_contract_sha256",
        "ordered_lanes",
        "identity_sha256",
    }
)
_IDENTITY_DEFINITION_CONTRACT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "exact_molecule",
        "partition_scaffold",
        "source_group",
    }
)
_IDENTITY_DEFINITION_FIELDS = frozenset(
    {
        "definition_id",
        "implementation_sha256",
        "identity_namespace",
    }
)
_RELATIONSHIP_NAMESPACE_FIELDS = frozenset(
    {
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
)
_IMPLEMENTATION_PROVENANCE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "code_revision",
        "python_runtime",
        "source_files",
        "policy_file",
        "editing_corpus_contract_file",
    }
)
_CODE_REVISION_FIELDS = frozenset({"commit_sha", "dirty"})
_PYTHON_RUNTIME_FIELDS = frozenset({"implementation", "version"})
_FILE_IDENTITY_FIELDS = frozenset({"path", "sha256", "bytes"})


class EditingV2SplitCensusError(ValueError):
    """The census input, policy, or persisted output is malformed."""


def canonical_json(value: Any) -> str:
    """Return the deterministic JSON representation used for every identity."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EditingV2SplitCensusError(
            f"value is not deterministic JSON: {exc}"
        ) from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def editing_lane_contract_identity(
    editing_corpus_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the ordered lane identity from the validated corpus contract."""

    if not isinstance(editing_corpus_contract, Mapping):
        raise EditingV2SplitCensusError("editing-corpus contract must be an object")
    contract = dict(editing_corpus_contract)
    try:
        validate_editing_corpus_contract(contract)
    except EditingCorpusContractError as exc:
        raise EditingV2SplitCensusError(
            f"editing-corpus contract is invalid: {exc}"
        ) from exc
    ordered_lanes = [
        {
            "id": str(lane["id"]),
            "admissible_evidence_profiles": list(lane["admissible_evidence_profiles"]),
            "reserved_evidence_profiles": list(lane["reserved_evidence_profiles"]),
        }
        for lane in contract["data_lanes"]
    ]
    body = {
        "schema": LANE_CONTRACT_IDENTITY_SCHEMA,
        "schema_version": LANE_CONTRACT_IDENTITY_SCHEMA_VERSION,
        "editing_corpus_contract_id": _require_string(
            contract.get("contract_id"),
            field="editing_corpus_contract.contract_id",
        ),
        "editing_corpus_contract_sha256": canonical_sha256(contract),
        "ordered_lanes": ordered_lanes,
    }
    return {**body, "identity_sha256": canonical_sha256(body)}


def default_split_census_policy(
    editing_corpus_contract: Mapping[str, Any],
    *,
    identity_definitions: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the explicit default scientific policy.

    Transformation signatures are diagnostic.  Making them hard requires both
    changing their mode and setting ``transformation_hard_authorized`` to true,
    so a generic transformation label can never silently merge components.

    Identity definitions are deliberately supplied by the caller.  This module
    has no authority to invent production canonical-molecule or partition-
    scaffold implementations.
    """

    normalized_identity_definitions = _normalize_identity_definition_contract_or_raise(
        identity_definitions,
        field="policy.identity_definitions",
    )
    return {
        "schema": SPLIT_CENSUS_POLICY_SCHEMA,
        "schema_version": SPLIT_CENSUS_POLICY_SCHEMA_VERSION,
        "policy_id": "compose-editing-v2-split-census-default-v4",
        "partition_roles": list(PARTITION_ROLES),
        "lane_contract_identity": editing_lane_contract_identity(
            editing_corpus_contract
        ),
        "identity_definitions": normalized_identity_definitions,
        "edge_modes": {
            edge_type: (
                "diagnostic"
                if edge_type
                in RECEIPT_GATED_RELATIONSHIP_EDGE_TYPES | {"transformation_signature"}
                else "hard"
            )
            for edge_type in EDGE_TYPES
        },
        "transformation_hard_authorized": False,
    }


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EditingV2SplitCensusError(f"{field} must be a nonempty string")
    return value


def _normalize_policy(
    policy: Mapping[str, Any],
    *,
    expected_lane_contract_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(policy, Mapping):
        raise EditingV2SplitCensusError("split-census policy must be an object")
    payload = dict(policy)
    if set(payload) != _POLICY_FIELDS:
        raise EditingV2SplitCensusError(
            "split-census policy fields disagree: "
            f"missing={sorted(_POLICY_FIELDS - set(payload))}, "
            f"extra={sorted(set(payload) - _POLICY_FIELDS)}"
        )
    if (
        payload["schema"] != SPLIT_CENSUS_POLICY_SCHEMA
        or payload["schema_version"] != SPLIT_CENSUS_POLICY_SCHEMA_VERSION
    ):
        raise EditingV2SplitCensusError("unsupported split-census policy schema")
    _require_string(payload["policy_id"], field="policy.policy_id")
    if tuple(payload["partition_roles"]) != PARTITION_ROLES:
        raise EditingV2SplitCensusError(
            "policy.partition_roles must equal the ordered editing-V2 four-role contract"
        )
    lane_contract_identity = payload["lane_contract_identity"]
    if not isinstance(lane_contract_identity, Mapping):
        raise EditingV2SplitCensusError(
            "policy.lane_contract_identity must be an object"
        )
    if dict(lane_contract_identity) != dict(expected_lane_contract_identity):
        raise EditingV2SplitCensusError(
            "policy.lane_contract_identity disagrees with the validated editing-corpus contract"
        )
    identity_definitions = _normalize_identity_definition_contract_or_raise(
        payload["identity_definitions"],
        field="policy.identity_definitions",
    )
    edge_modes = payload["edge_modes"]
    if not isinstance(edge_modes, Mapping) or set(edge_modes) != set(EDGE_TYPES):
        raise EditingV2SplitCensusError(
            "policy.edge_modes must define every split relationship exactly once"
        )
    normalized_modes: dict[str, str] = {}
    for edge_type in EDGE_TYPES:
        mode = edge_modes[edge_type]
        if mode not in EDGE_MODES:
            raise EditingV2SplitCensusError(
                f"policy.edge_modes.{edge_type} must be one of {sorted(EDGE_MODES)}"
            )
        normalized_modes[edge_type] = str(mode)
    weakened = sorted(
        edge_type
        for edge_type in MANDATORY_HARD_EDGE_TYPES
        if normalized_modes[edge_type] != "hard"
    )
    if weakened:
        raise EditingV2SplitCensusError(
            f"mandatory split relationships may not be weakened: {weakened}"
        )
    receipt_gated_hard = sorted(
        edge_type
        for edge_type in RECEIPT_GATED_RELATIONSHIP_EDGE_TYPES
        if normalized_modes[edge_type] == "hard"
    )
    if receipt_gated_hard:
        raise EditingV2SplitCensusError(
            "post-Active8 relationship groups may not become hard split edges "
            "before physical receipt resolution: "
            f"{receipt_gated_hard}"
        )
    authorization = payload["transformation_hard_authorized"]
    if not isinstance(authorization, bool):
        raise EditingV2SplitCensusError(
            "policy.transformation_hard_authorized must be boolean"
        )
    if (
        normalized_modes["transformation_signature"] == "hard"
        and authorization is not True
    ):
        raise EditingV2SplitCensusError(
            "hard transformation signatures require explicit transformation_hard_authorized=true"
        )
    normalized = {
        **payload,
        "partition_roles": list(PARTITION_ROLES),
        "lane_contract_identity": dict(expected_lane_contract_identity),
        "identity_definitions": identity_definitions,
        "edge_modes": normalized_modes,
    }
    canonical_json(normalized)
    return normalized


def _error(errors: list[dict[str, str]], code: str, detail: str) -> None:
    item = {"code": code, "detail": detail}
    if item not in errors:
        errors.append(item)


def _normalize_identity_definition_contract(
    value: Any,
    *,
    field: str,
    errors: list[dict[str, str]],
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        _error(errors, f"{field}.invalid", f"{field} must be an object")
        return None
    payload = dict(value)
    if set(payload) != _IDENTITY_DEFINITION_CONTRACT_FIELDS:
        _error(
            errors,
            f"{field}.fields_mismatch",
            f"{field} fields disagree: "
            f"missing={sorted(_IDENTITY_DEFINITION_CONTRACT_FIELDS - set(payload))}, "
            f"extra={sorted(set(payload) - _IDENTITY_DEFINITION_CONTRACT_FIELDS)}",
        )
        return None
    if (
        payload["schema"] != IDENTITY_DEFINITION_CONTRACT_SCHEMA
        or payload["schema_version"] != IDENTITY_DEFINITION_CONTRACT_SCHEMA_VERSION
    ):
        _error(errors, f"{field}.schema_mismatch", f"{field} has an unsupported schema")

    normalized = {
        "schema": IDENTITY_DEFINITION_CONTRACT_SCHEMA,
        "schema_version": IDENTITY_DEFINITION_CONTRACT_SCHEMA_VERSION,
    }
    for identity_type in IDENTITY_TYPES:
        definition_field = f"{field}.{identity_type}"
        raw_definition = payload[identity_type]
        if not isinstance(raw_definition, Mapping):
            _error(
                errors,
                f"{definition_field}.invalid",
                f"{definition_field} must be an object",
            )
            continue
        definition = dict(raw_definition)
        if set(definition) != _IDENTITY_DEFINITION_FIELDS:
            _error(
                errors,
                f"{definition_field}.fields_mismatch",
                f"{definition_field} fields disagree: "
                f"missing={sorted(_IDENTITY_DEFINITION_FIELDS - set(definition))}, "
                f"extra={sorted(set(definition) - _IDENTITY_DEFINITION_FIELDS)}",
            )
            continue
        for string_field in ("definition_id", "identity_namespace"):
            if (
                not isinstance(definition[string_field], str)
                or not definition[string_field].strip()
            ):
                _error(
                    errors,
                    f"{definition_field}.{string_field}_invalid",
                    f"{definition_field}.{string_field} must be a nonempty string",
                )
        if (
            not isinstance(definition["implementation_sha256"], str)
            or _SHA256_RE.fullmatch(definition["implementation_sha256"]) is None
        ):
            _error(
                errors,
                f"{definition_field}.implementation_sha256_invalid",
                f"{definition_field}.implementation_sha256 must be a full lowercase SHA-256",
            )
        normalized[identity_type] = definition
    if any(identity_type not in normalized for identity_type in IDENTITY_TYPES):
        return None
    return normalized


def _normalize_identity_definition_contract_or_raise(
    value: Any,
    *,
    field: str,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    normalized = _normalize_identity_definition_contract(
        value,
        field=field,
        errors=errors,
    )
    if normalized is None or errors:
        details = "; ".join(item["detail"] for item in errors)
        raise EditingV2SplitCensusError(
            f"{field} is invalid" + (f": {details}" if details else "")
        )
    return normalized


def _normalize_relationship_namespace(
    value: Any,
    *,
    relationship_type: str,
    field: str,
    errors: list[dict[str, str]],
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        _error(errors, f"{field}.missing", f"{field} must be a namespace object")
        return None
    payload = dict(value)
    if set(payload) != _RELATIONSHIP_NAMESPACE_FIELDS:
        _error(
            errors,
            f"{field}.fields_mismatch",
            f"{field} fields disagree: "
            f"missing={sorted(_RELATIONSHIP_NAMESPACE_FIELDS - set(payload))}, "
            f"extra={sorted(set(payload) - _RELATIONSHIP_NAMESPACE_FIELDS)}",
        )
        return None
    if (
        payload["schema"] != RELATIONSHIP_NAMESPACE_SCHEMA
        or payload["schema_version"] != RELATIONSHIP_NAMESPACE_SCHEMA_VERSION
    ):
        _error(errors, f"{field}.schema_mismatch", f"{field} has an unsupported schema")
    if payload["relationship_type"] != relationship_type:
        _error(
            errors,
            f"{field}.relationship_type_mismatch",
            f"{field}.relationship_type must equal {relationship_type!r}",
        )
    for string_field in (
        "namespace_id",
        "source_asset_id",
        "definition_id",
    ):
        if (
            not isinstance(payload[string_field], str)
            or not payload[string_field].strip()
        ):
            _error(
                errors,
                f"{field}.{string_field}_invalid",
                f"{field}.{string_field} must be a nonempty string",
            )
    for sha_field in ("source_asset_sha256", "implementation_sha256"):
        if (
            not isinstance(payload[sha_field], str)
            or _SHA256_RE.fullmatch(payload[sha_field]) is None
        ):
            _error(
                errors,
                f"{field}.{sha_field}_invalid",
                f"{field}.{sha_field} must be a full lowercase SHA-256",
            )
    if not isinstance(payload["cross_lane_sharing_authorized"], bool):
        _error(
            errors,
            f"{field}.cross_lane_sharing_authorized_invalid",
            f"{field}.cross_lane_sharing_authorized must be boolean",
        )
    return payload


def _normalize_relationship_namespaces(
    value: Any,
    *,
    group_values: Mapping[str, str | None],
    errors: list[dict[str, str]],
) -> dict[str, dict[str, Any] | None]:
    field = "relationship_namespaces"
    if not isinstance(value, Mapping):
        _error(errors, f"{field}.invalid", f"{field} must be an object")
        return {
            relationship_type: None for relationship_type in SIMPLE_RELATIONSHIP_FIELDS
        }
    payload = dict(value)
    expected_fields = set(SIMPLE_RELATIONSHIP_FIELDS)
    if set(payload) != expected_fields:
        _error(
            errors,
            f"{field}.fields_mismatch",
            f"{field} fields disagree: missing={sorted(expected_fields - set(payload))}, "
            f"extra={sorted(set(payload) - expected_fields)}",
        )
    normalized: dict[str, dict[str, Any] | None] = {}
    for relationship_type, group_field in SIMPLE_RELATIONSHIP_FIELDS.items():
        namespace_value = payload.get(relationship_type)
        if group_values[group_field] is None:
            if namespace_value is not None:
                _error(
                    errors,
                    f"{field}.{relationship_type}.orphan",
                    f"{field}.{relationship_type} is forbidden without {group_field}",
                )
            normalized[relationship_type] = None
            continue
        normalized[relationship_type] = _normalize_relationship_namespace(
            namespace_value,
            relationship_type=relationship_type,
            field=f"{field}.{relationship_type}",
            errors=errors,
        )
    return normalized


def _normalize_file_identity(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EditingV2SplitCensusError(f"{field} must be an object")
    payload = dict(value)
    if set(payload) != _FILE_IDENTITY_FIELDS:
        raise EditingV2SplitCensusError(
            f"{field} fields disagree: "
            f"missing={sorted(_FILE_IDENTITY_FIELDS - set(payload))}, "
            f"extra={sorted(set(payload) - _FILE_IDENTITY_FIELDS)}"
        )
    _require_string(payload["path"], field=f"{field}.path")
    if (
        not isinstance(payload["sha256"], str)
        or _SHA256_RE.fullmatch(payload["sha256"]) is None
    ):
        raise EditingV2SplitCensusError(
            f"{field}.sha256 must be a full lowercase SHA-256"
        )
    if (
        isinstance(payload["bytes"], bool)
        or not isinstance(payload["bytes"], int)
        or payload["bytes"] < 0
    ):
        raise EditingV2SplitCensusError(f"{field}.bytes must be a nonnegative integer")
    return payload


def _normalize_implementation_provenance(value: Any) -> dict[str, Any]:
    field = "implementation_provenance"
    if not isinstance(value, Mapping):
        raise EditingV2SplitCensusError(f"{field} must be an object")
    payload = dict(value)
    if set(payload) != _IMPLEMENTATION_PROVENANCE_FIELDS:
        raise EditingV2SplitCensusError(
            f"{field} fields disagree: "
            f"missing={sorted(_IMPLEMENTATION_PROVENANCE_FIELDS - set(payload))}, "
            f"extra={sorted(set(payload) - _IMPLEMENTATION_PROVENANCE_FIELDS)}"
        )
    if (
        payload["schema"] != IMPLEMENTATION_PROVENANCE_SCHEMA
        or payload["schema_version"] != IMPLEMENTATION_PROVENANCE_SCHEMA_VERSION
    ):
        raise EditingV2SplitCensusError(f"{field} has an unsupported schema")

    code_revision = payload["code_revision"]
    if (
        not isinstance(code_revision, Mapping)
        or set(code_revision) != _CODE_REVISION_FIELDS
    ):
        raise EditingV2SplitCensusError(
            f"{field}.code_revision must contain commit_sha and dirty exactly"
        )
    if (
        not isinstance(code_revision["commit_sha"], str)
        or _GIT_COMMIT_RE.fullmatch(code_revision["commit_sha"]) is None
    ):
        raise EditingV2SplitCensusError(
            f"{field}.code_revision.commit_sha must be a full lowercase Git commit"
        )
    if not isinstance(code_revision["dirty"], bool):
        raise EditingV2SplitCensusError(f"{field}.code_revision.dirty must be boolean")

    python_runtime = payload["python_runtime"]
    if (
        not isinstance(python_runtime, Mapping)
        or set(python_runtime) != _PYTHON_RUNTIME_FIELDS
    ):
        raise EditingV2SplitCensusError(
            f"{field}.python_runtime must contain implementation and version exactly"
        )
    for runtime_field in _PYTHON_RUNTIME_FIELDS:
        _require_string(
            python_runtime[runtime_field],
            field=f"{field}.python_runtime.{runtime_field}",
        )

    source_files = payload["source_files"]
    if not isinstance(source_files, list) or not source_files:
        raise EditingV2SplitCensusError(f"{field}.source_files must be a nonempty list")
    normalized_source_files = [
        _normalize_file_identity(item, field=f"{field}.source_files[{index}]")
        for index, item in enumerate(source_files)
    ]
    paths = [item["path"] for item in normalized_source_files]
    if len(paths) != len(set(paths)):
        raise EditingV2SplitCensusError(
            f"{field}.source_files contains duplicate paths"
        )
    normalized_source_files.sort(key=lambda item: item["path"])

    normalized = {
        "schema": IMPLEMENTATION_PROVENANCE_SCHEMA,
        "schema_version": IMPLEMENTATION_PROVENANCE_SCHEMA_VERSION,
        "code_revision": dict(code_revision),
        "python_runtime": dict(python_runtime),
        "source_files": normalized_source_files,
        "policy_file": _normalize_file_identity(
            payload["policy_file"],
            field=f"{field}.policy_file",
        ),
        "editing_corpus_contract_file": _normalize_file_identity(
            payload["editing_corpus_contract_file"],
            field=f"{field}.editing_corpus_contract_file",
        ),
    }
    canonical_json(normalized)
    return normalized


def _optional_group_id(
    value: Any,
    *,
    field: str,
    errors: list[dict[str, str]],
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        _error(errors, f"{field}.invalid", f"{field} must be null or a nonempty string")
        return None
    return value


def _string_list(
    value: Any,
    *,
    field: str,
    nonempty: bool,
    errors: list[dict[str, str]],
) -> list[str]:
    if not isinstance(value, list):
        _error(errors, f"{field}.invalid", f"{field} must be a list of strings")
        return []
    if any(not isinstance(item, str) or not item.strip() for item in value):
        _error(
            errors,
            f"{field}.invalid",
            f"{field} entries must be nonempty strings",
        )
        return []
    if len(value) != len(set(value)):
        _error(errors, f"{field}.duplicate", f"{field} contains duplicate entries")
    normalized = sorted(set(value))
    if nonempty and not normalized:
        _error(errors, f"{field}.empty", f"{field} cannot be empty")
    return normalized


def _normalize_observed_provenance(
    value: Any,
    *,
    relationship: str,
    errors: list[dict[str, str]],
) -> dict[str, Any] | None:
    prefix = f"{relationship}_provenance"
    if not isinstance(value, Mapping):
        _error(
            errors,
            f"{prefix}.missing",
            f"{prefix} is required for a declared {relationship} group",
        )
        return None
    payload = dict(value)
    if set(payload) != _OBSERVED_PROVENANCE_FIELDS:
        _error(
            errors,
            f"{prefix}.fields_mismatch",
            f"{prefix} fields disagree: "
            f"missing={sorted(_OBSERVED_PROVENANCE_FIELDS - set(payload))}, "
            f"extra={sorted(set(payload) - _OBSERVED_PROVENANCE_FIELDS)}",
        )
        return None
    if (
        payload["schema"] != OBSERVED_GROUP_PROVENANCE_SCHEMA
        or payload["schema_version"] != OBSERVED_GROUP_PROVENANCE_SCHEMA_VERSION
    ):
        _error(
            errors,
            f"{prefix}.schema_mismatch",
            f"{prefix} has an unsupported schema",
        )
    if payload["provenance_kind"] != "observed_source_group":
        _error(
            errors,
            f"{prefix}.not_observed",
            f"{prefix}.provenance_kind must be 'observed_source_group'",
        )
    for field in (
        "group_namespace",
        "source_asset_id",
        "source_version",
        "access_basis",
    ):
        if not isinstance(payload[field], str) or not payload[field].strip():
            _error(
                errors,
                f"{prefix}.{field}_invalid",
                f"{prefix}.{field} must be a nonempty string",
            )
    if (
        not isinstance(payload["source_asset_sha256"], str)
        or _SHA256_RE.fullmatch(payload["source_asset_sha256"]) is None
    ):
        _error(
            errors,
            f"{prefix}.source_asset_sha256_invalid",
            f"{prefix}.source_asset_sha256 must be a full lowercase SHA-256",
        )
    source_rows = _string_list(
        payload["source_row_ids"],
        field=f"{prefix}.source_row_ids",
        nonempty=True,
        errors=errors,
    )
    normalized = {**payload, "source_row_ids": source_rows}
    return normalized


def _normalize_transformation_definition(
    value: Any,
    *,
    errors: list[dict[str, str]],
) -> dict[str, Any] | None:
    field = "transformation_definition"
    if not isinstance(value, Mapping):
        _error(
            errors,
            f"{field}.missing",
            "transformation_definition is required for a declared signature",
        )
        return None
    payload = dict(value)
    if set(payload) != _TRANSFORMATION_DEFINITION_FIELDS:
        _error(
            errors,
            f"{field}.fields_mismatch",
            f"{field} fields disagree: "
            f"missing={sorted(_TRANSFORMATION_DEFINITION_FIELDS - set(payload))}, "
            f"extra={sorted(set(payload) - _TRANSFORMATION_DEFINITION_FIELDS)}",
        )
        return None
    if (
        payload["schema"] != TRANSFORMATION_DEFINITION_SCHEMA
        or payload["schema_version"] != TRANSFORMATION_DEFINITION_SCHEMA_VERSION
    ):
        _error(errors, f"{field}.schema_mismatch", f"{field} has an unsupported schema")
    if (
        not isinstance(payload["definition_id"], str)
        or not payload["definition_id"].strip()
    ):
        _error(
            errors,
            f"{field}.definition_id_invalid",
            f"{field}.definition_id must be a nonempty string",
        )
    if (
        not isinstance(payload["implementation_sha256"], str)
        or _SHA256_RE.fullmatch(payload["implementation_sha256"]) is None
    ):
        _error(
            errors,
            f"{field}.implementation_sha256_invalid",
            f"{field}.implementation_sha256 must be a full lowercase SHA-256",
        )
    if payload["derivation_kind"] not in {
        "observed_source_definition",
        "structurally_inferred_definition",
        "synthetic_definition",
    }:
        _error(
            errors,
            f"{field}.derivation_kind_invalid",
            f"{field}.derivation_kind is unknown",
        )
    return payload


@dataclass(frozen=True)
class _ParsedRow:
    source_index: int
    source_row_sha256: str
    normalized: dict[str, Any] | None
    errors: tuple[dict[str, str], ...]
    raw_row: Any


def _parse_row(
    raw_row: Any,
    *,
    source_index: int,
    editing_corpus_contract: Mapping[str, Any],
    expected_identity_definitions: Mapping[str, Any],
) -> _ParsedRow:
    source_row_sha256 = canonical_sha256(raw_row)
    if not isinstance(raw_row, Mapping):
        return _ParsedRow(
            source_index=source_index,
            source_row_sha256=source_row_sha256,
            normalized=None,
            errors=(
                {
                    "code": "row.not_object",
                    "detail": "candidate row must be a JSON object",
                },
            ),
            raw_row=raw_row,
        )

    row = dict(raw_row)
    errors: list[dict[str, str]] = []
    if set(row) != _ROW_FIELDS:
        _error(
            errors,
            "row.fields_mismatch",
            f"candidate row fields disagree: missing={sorted(_ROW_FIELDS - set(row))}, "
            f"extra={sorted(set(row) - _ROW_FIELDS)}",
        )
    if row.get("schema") != CANDIDATE_ROW_SCHEMA:
        _error(errors, "row.schema_mismatch", "candidate row schema is unsupported")
    if row.get("schema_version") != CANDIDATE_ROW_SCHEMA_VERSION:
        _error(
            errors,
            "row.schema_version_mismatch",
            "candidate row schema version is unsupported",
        )

    candidate_id = row.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        _error(
            errors,
            "candidate_id.invalid",
            "candidate_id must be a nonempty string",
        )
        candidate_id = None

    identity_hashes: dict[str, str | None] = {}
    for field in ("candidate_ledger_row_sha256", "candidate_envelope_sha256"):
        value = row.get(field)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            _error(
                errors,
                f"{field}.invalid",
                f"{field} must be a full lowercase SHA-256",
            )
            identity_hashes[field] = None
        else:
            identity_hashes[field] = value

    lane_ids = {str(lane["id"]) for lane in editing_corpus_contract["data_lanes"]}
    data_lane = row.get("data_lane")
    if not isinstance(data_lane, str) or not data_lane.strip():
        _error(errors, "data_lane.invalid", "data_lane must be a nonempty string")
        data_lane = None
    elif data_lane not in lane_ids:
        _error(
            errors,
            "data_lane.not_in_contract",
            f"data_lane {data_lane!r} is not declared by the editing-corpus contract",
        )
    evidence_profile_id = row.get("evidence_profile_id")
    if not isinstance(evidence_profile_id, str) or not evidence_profile_id.strip():
        _error(
            errors,
            "evidence_profile_id.invalid",
            "evidence_profile_id must be a nonempty string",
        )
        evidence_profile_id = None
    evidence_components_raw = row.get("evidence_components")
    evidence_components = (
        dict(evidence_components_raw)
        if isinstance(evidence_components_raw, Mapping)
        else None
    )
    if evidence_components is None:
        _error(
            errors,
            "evidence_components.invalid",
            "evidence_components must be the exact six-component object",
        )
    elif data_lane is not None and evidence_profile_id is not None:
        try:
            validate_evidence_assignment_envelope(
                editing_corpus_contract,
                data_lane=data_lane,
                evidence_profile_id=evidence_profile_id,
                evidence_components=evidence_components,
            )
        except EditingCorpusContractError as exc:
            _error(
                errors,
                "evidence_assignment.invalid",
                f"evidence assignment is invalid: {exc}",
            )
        evidence_components = {
            key: evidence_components[key] for key in sorted(evidence_components)
        }
    mass_units = row.get("mass_units")
    if (
        isinstance(mass_units, bool)
        or not isinstance(mass_units, int)
        or mass_units <= 0
    ):
        _error(
            errors,
            "mass_units.invalid",
            "mass_units must be a positive integer",
        )
        mass_units = None

    molecule_ids = _string_list(
        row.get("molecule_ids"),
        field="molecule_ids",
        nonempty=True,
        errors=errors,
    )
    scaffold_ids = _string_list(
        row.get("partition_scaffold_ids"),
        field="partition_scaffold_ids",
        nonempty=True,
        errors=errors,
    )
    source_group_ids = _string_list(
        row.get("source_group_ids"),
        field="source_group_ids",
        nonempty=True,
        errors=errors,
    )
    source_group_namespace = _normalize_relationship_namespace(
        row.get("source_group_namespace"),
        relationship_type="source_group",
        field="source_group_namespace",
        errors=errors,
    )
    if (
        source_group_namespace is not None
        and source_group_namespace.get("cross_lane_sharing_authorized") is not True
    ):
        _error(
            errors,
            "source_group_namespace.cross_lane_sharing_required",
            "source_group_namespace must authorize cross-lane sharing",
        )
    identity_definitions = _normalize_identity_definition_contract(
        row.get("identity_definitions"),
        field="identity_definitions",
        errors=errors,
    )
    if identity_definitions is not None and dict(identity_definitions) != dict(
        expected_identity_definitions
    ):
        _error(
            errors,
            "identity_definitions.contract_mismatch",
            "identity_definitions disagree with the hash-bound split-census policy",
        )

    group_values = {
        field: _optional_group_id(row.get(field), field=field, errors=errors)
        for field in (
            "shared_prefix_branch_group_id",
            "alternative_route_group_id",
            "inverse_pair_group_id",
            "correction_group_id",
            "constraint_compatible_alternative_route_group_id",
            "document_group_id",
            "series_group_id",
            "transformation_signature",
        )
    }
    relationship_namespaces = _normalize_relationship_namespaces(
        row.get("relationship_namespaces"),
        group_values=group_values,
        errors=errors,
    )

    document_provenance = None
    if group_values["document_group_id"] is not None:
        document_provenance = _normalize_observed_provenance(
            row.get("document_provenance"),
            relationship="document",
            errors=errors,
        )
    elif row.get("document_provenance") is not None:
        _error(
            errors,
            "document_provenance.orphan",
            "document_provenance is forbidden without document_group_id",
        )

    series_provenance = None
    if group_values["series_group_id"] is not None:
        series_provenance = _normalize_observed_provenance(
            row.get("series_provenance"),
            relationship="series",
            errors=errors,
        )
    elif row.get("series_provenance") is not None:
        _error(
            errors,
            "series_provenance.orphan",
            "series_provenance is forbidden without series_group_id",
        )

    transformation_definition = None
    if group_values["transformation_signature"] is not None:
        transformation_definition = _normalize_transformation_definition(
            row.get("transformation_definition"),
            errors=errors,
        )
    elif row.get("transformation_definition") is not None:
        _error(
            errors,
            "transformation_definition.orphan",
            "transformation_definition is forbidden without transformation_signature",
        )

    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        _error(errors, "metadata.invalid", "metadata must be a JSON object")
        metadata = {}
    else:
        metadata = dict(metadata)

    normalized = {
        "schema": CANDIDATE_ROW_SCHEMA,
        "schema_version": CANDIDATE_ROW_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        **identity_hashes,
        "data_lane": data_lane,
        "evidence_profile_id": evidence_profile_id,
        "evidence_components": evidence_components,
        "mass_units": mass_units,
        "molecule_ids": molecule_ids,
        "partition_scaffold_ids": scaffold_ids,
        "source_group_ids": source_group_ids,
        "source_group_namespace": source_group_namespace,
        "identity_definitions": identity_definitions,
        **group_values,
        "relationship_namespaces": relationship_namespaces,
        "document_provenance": document_provenance,
        "series_provenance": series_provenance,
        "transformation_definition": transformation_definition,
        "metadata": metadata,
    }
    return _ParsedRow(
        source_index=source_index,
        source_row_sha256=source_row_sha256,
        normalized=normalized,
        errors=tuple(sorted(errors, key=lambda item: (item["code"], item["detail"]))),
        raw_row=row if errors else None,
    )


def _relationship_key(row: Mapping[str, Any], edge_type: str) -> list[dict[str, Any]]:
    if edge_type == "exact_molecule":
        return [
            {
                "identity_definition": row["identity_definitions"]["exact_molecule"],
                "molecule_id": value,
            }
            for value in row["molecule_ids"]
        ]
    if edge_type == "partition_scaffold":
        return [
            {
                "identity_definition": row["identity_definitions"][
                    "partition_scaffold"
                ],
                "partition_scaffold_id": value,
            }
            for value in row["partition_scaffold_ids"]
        ]
    if edge_type == "source_group":
        return [
            {
                "identity_definition": row["identity_definitions"]["source_group"],
                "namespace": row["source_group_namespace"],
                "source_group_id": value,
            }
            for value in row["source_group_ids"]
        ]
    if edge_type in SIMPLE_RELATIONSHIP_FIELDS:
        value = row[SIMPLE_RELATIONSHIP_FIELDS[edge_type]]
        namespace = row["relationship_namespaces"][edge_type]
        if value is None:
            return []
        lane_scope = (
            None if namespace["cross_lane_sharing_authorized"] else row["data_lane"]
        )
        return [
            {
                "namespace": namespace,
                "data_lane_scope": lane_scope,
                "group_id": value,
            }
        ]
    if edge_type in {"document_group", "series_group"}:
        relationship = edge_type.removesuffix("_group")
        group_id = row[f"{relationship}_group_id"]
        provenance = row[f"{relationship}_provenance"]
        if group_id is None:
            return []
        return [
            {
                "group_namespace": provenance["group_namespace"],
                "source_asset_id": provenance["source_asset_id"],
                "source_asset_sha256": provenance["source_asset_sha256"],
                "group_id": group_id,
            }
        ]
    if edge_type == "transformation_signature":
        signature = row["transformation_signature"]
        definition = row["transformation_definition"]
        if signature is None:
            return []
        return [
            {
                "definition_id": definition["definition_id"],
                "implementation_sha256": definition["implementation_sha256"],
                "derivation_kind": definition["derivation_kind"],
                "signature": signature,
            }
        ]
    raise AssertionError(f"unknown relationship edge type {edge_type!r}")


class _UnionFind:
    def __init__(self, vertices: Iterable[str]) -> None:
        ordered = tuple(sorted(vertices))
        self.parent = {vertex: vertex for vertex in ordered}
        self.size = {vertex: 1 for vertex in ordered}

    def find(self, vertex: str) -> str:
        parent = self.parent[vertex]
        if parent != vertex:
            self.parent[vertex] = self.find(parent)
        return self.parent[vertex]

    def union(self, left: str, right: str) -> bool:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return False
        if (self.size[root_left], root_left) < (self.size[root_right], root_right):
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]
        return True

    def components(self) -> list[list[str]]:
        by_root: dict[str, list[str]] = defaultdict(list)
        for vertex in sorted(self.parent):
            by_root[self.find(vertex)].append(vertex)
        return sorted(
            (sorted(members) for members in by_root.values()),
            key=lambda members: (members[0], members),
        )

    def copy(self) -> "_UnionFind":
        clone = _UnionFind(())
        clone.parent = dict(self.parent)
        clone.size = dict(self.size)
        return clone


def _nearest_rank(values: Sequence[int], probability: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return int(ordered[rank - 1])


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "total": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean_numerator": 0,
            "mean_denominator": 0,
        }
    return {
        "count": len(values),
        "total": int(sum(values)),
        "min": int(min(values)),
        "p50": _nearest_rank(values, 0.50),
        "p90": _nearest_rank(values, 0.90),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "max": int(max(values)),
        "mean_numerator": int(sum(values)),
        "mean_denominator": len(values),
    }


def _component_state(
    union_find: _UnionFind,
    *,
    mass_by_candidate: Mapping[str, int],
) -> dict[str, int]:
    components = union_find.components()
    masses = [
        sum(mass_by_candidate[item] for item in members) for members in components
    ]
    return {
        "component_count": len(components),
        "largest_component_candidates": max(
            (len(item) for item in components), default=0
        ),
        "largest_component_mass_units": max(masses, default=0),
    }


def _attainable_ranges(components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    component_count = len(components)
    total_mass = sum(int(component["mass_units"]) for component in components)
    total_candidates = sum(
        int(component["candidate_count"]) for component in components
    )
    feasible = component_count >= len(PARTITION_ROLES)
    result: dict[str, Any] = {
        "partition_roles": list(PARTITION_ROLES),
        "ratios_selected": False,
        "role_assignment_selected": False,
        "component_count": component_count,
        "four_nonempty_roles_feasible": feasible,
        "total_mass_units": total_mass,
        "total_candidates": total_candidates,
        "bounds_are_marginal_not_joint_guarantees": True,
        "per_role_symmetric_marginal_bounds": None,
    }
    if not feasible:
        return result
    by_mass = sorted(
        components,
        key=lambda component: (
            int(component["mass_units"]),
            str(component["component_id"]),
        ),
    )
    by_candidates = sorted(
        components,
        key=lambda component: (
            int(component["candidate_count"]),
            str(component["component_id"]),
        ),
    )
    reserved_mass = by_mass[: len(PARTITION_ROLES) - 1]
    reserved_candidates = by_candidates[: len(PARTITION_ROLES) - 1]
    result["per_role_symmetric_marginal_bounds"] = {
        "minimum_mass_units": int(by_mass[0]["mass_units"]),
        "maximum_mass_units": total_mass
        - sum(int(component["mass_units"]) for component in reserved_mass),
        "minimum_candidates": int(by_candidates[0]["candidate_count"]),
        "maximum_candidates": total_candidates
        - sum(int(component["candidate_count"]) for component in reserved_candidates),
        "maximum_mass_reserved_component_ids": [
            str(component["component_id"]) for component in reserved_mass
        ],
        "maximum_candidates_reserved_component_ids": [
            str(component["component_id"]) for component in reserved_candidates
        ],
        "interpretation": (
            "Each role is symmetric. Bounds range over assignments in which all four "
            "roles receive at least one whole hard component; no target ratio or "
            "particular assignment is implied."
        ),
    }
    return result


def _invalid_entry(
    parsed: _ParsedRow, errors: Sequence[dict[str, str]]
) -> dict[str, Any]:
    candidate_id = (
        parsed.normalized.get("candidate_id")
        if isinstance(parsed.normalized, Mapping)
        else None
    )
    return {
        "source_index": parsed.source_index,
        "candidate_id": candidate_id,
        "source_row_sha256": parsed.source_row_sha256,
        "reason_codes": sorted({item["code"] for item in errors}),
        "reasons": sorted(errors, key=lambda item: (item["code"], item["detail"])),
        "raw_row": (
            parsed.raw_row if parsed.raw_row is not None else parsed.normalized
        ),
    }


def build_split_component_census(
    rows: Iterable[Any],
    *,
    policy: Mapping[str, Any],
    editing_corpus_contract: Mapping[str, Any],
    implementation_provenance: Mapping[str, Any],
    source_stream: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the deterministic census without choosing ratios or assignments."""

    lane_contract_identity = editing_lane_contract_identity(editing_corpus_contract)
    normalized_policy = _normalize_policy(
        policy,
        expected_lane_contract_identity=lane_contract_identity,
    )
    normalized_provenance = _normalize_implementation_provenance(
        implementation_provenance
    )
    parsed_rows: list[_ParsedRow] = []
    source_row_sha256s: list[str] = []
    source_row_count = 0
    for index, row in enumerate(rows):
        parsed = _parse_row(
            row,
            source_index=index,
            editing_corpus_contract=editing_corpus_contract,
            expected_identity_definitions=normalized_policy["identity_definitions"],
        )
        parsed_rows.append(parsed)
        source_row_sha256s.append(parsed.source_row_sha256)
        source_row_count += 1

    duplicate_counts = Counter(
        parsed.normalized["candidate_id"]
        for parsed in parsed_rows
        if parsed.normalized is not None
        and parsed.normalized["candidate_id"] is not None
    )
    invalid_rows: list[dict[str, Any]] = []
    vertices: list[dict[str, Any]] = []
    for parsed in parsed_rows:
        errors = list(parsed.errors)
        if (
            parsed.normalized is not None
            and parsed.normalized["candidate_id"] is not None
            and duplicate_counts[parsed.normalized["candidate_id"]] > 1
        ):
            _error(
                errors,
                "candidate_id.duplicate",
                "every occurrence of a duplicate candidate_id is excluded",
            )
        if errors or parsed.normalized is None:
            invalid_rows.append(_invalid_entry(parsed, errors))
            continue
        normalized = dict(parsed.normalized)
        row_sha256 = canonical_sha256(normalized)
        vertices.append(
            {
                **normalized,
                "source_row_sha256": parsed.source_row_sha256,
                "row_sha256": row_sha256,
            }
        )
    vertices.sort(key=lambda row: row["candidate_id"])
    invalid_rows.sort(
        key=lambda row: (
            row["candidate_id"] or "",
            row["source_row_sha256"],
            row["source_index"],
        )
    )

    source_stream_payload: dict[str, Any] | None
    if source_stream is None:
        source_stream_payload = None
    elif not isinstance(source_stream, Mapping):
        raise EditingV2SplitCensusError("source_stream must be a JSON object")
    else:
        source_stream_payload = dict(source_stream)
        canonical_json(source_stream_payload)
        declared_rows = source_stream_payload.get("nonempty_jsonl_rows")
        if (
            isinstance(declared_rows, bool)
            or not isinstance(declared_rows, int)
            or declared_rows < 0
        ):
            raise EditingV2SplitCensusError(
                "source_stream.nonempty_jsonl_rows must be a nonnegative integer"
            )
        if declared_rows != source_row_count:
            raise EditingV2SplitCensusError(
                "source_stream.nonempty_jsonl_rows disagrees with streamed rows"
            )

    row_stream_sha256 = canonical_sha256(sorted(source_row_sha256s))
    policy_sha256 = canonical_sha256(normalized_policy)
    lane_contract_identity_sha256 = canonical_sha256(lane_contract_identity)
    identity_definition_contract_sha256 = canonical_sha256(
        normalized_policy["identity_definitions"]
    )
    implementation_provenance_sha256 = canonical_sha256(normalized_provenance)
    vertex_inventory_sha256 = canonical_sha256(vertices)
    invalid_rows_sha256 = canonical_sha256(invalid_rows)
    candidate_source_binding_inventory = [
        {
            "candidate_id": row["candidate_id"],
            "candidate_ledger_row_sha256": row["candidate_ledger_row_sha256"],
            "candidate_envelope_sha256": row["candidate_envelope_sha256"],
        }
        for row in vertices
    ]
    candidate_source_binding_inventory_sha256 = canonical_sha256(
        candidate_source_binding_inventory
    )

    by_id = {str(row["candidate_id"]): row for row in vertices}
    mass_by_candidate = {
        candidate_id: int(row["mass_units"]) for candidate_id, row in by_id.items()
    }
    relationship_groups: dict[str, dict[str, dict[str, Any]]] = {
        edge_type: {} for edge_type in EDGE_TYPES
    }
    for row in vertices:
        candidate_id = str(row["candidate_id"])
        for edge_type in EDGE_TYPES:
            for relationship_key in _relationship_key(row, edge_type):
                serialized_key = canonical_json(relationship_key)
                group = relationship_groups[edge_type].setdefault(
                    serialized_key,
                    {
                        "relationship_key": relationship_key,
                        "members": set(),
                        "evidence_sha256s": set(),
                    },
                )
                group["members"].add(candidate_id)
                if edge_type in {"document_group", "series_group"}:
                    relationship = edge_type.removesuffix("_group")
                    group["evidence_sha256s"].add(
                        canonical_sha256(row[f"{relationship}_provenance"])
                    )
                if edge_type == "transformation_signature":
                    group["evidence_sha256s"].add(
                        canonical_sha256(row["transformation_definition"])
                    )

    relationship_inventory: list[dict[str, Any]] = []
    edge_ledger: list[dict[str, Any]] = []
    edges_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge_type in EDGE_TYPES:
        mode = normalized_policy["edge_modes"][edge_type]
        for serialized_key, group in sorted(relationship_groups[edge_type].items()):
            members = sorted(group["members"])
            relationship_key_sha256 = canonical_sha256(group["relationship_key"])
            lanes = sorted({str(by_id[member]["data_lane"]) for member in members})
            inventory_entry = {
                "edge_type": edge_type,
                "mode": mode,
                "relationship_key": group["relationship_key"],
                "relationship_key_sha256": relationship_key_sha256,
                "members": members,
                "member_count": len(members),
                "mass_units": sum(mass_by_candidate[member] for member in members),
                "data_lanes": lanes,
                "cross_lane": len(lanes) > 1,
                "evidence_sha256s": sorted(group["evidence_sha256s"]),
            }
            relationship_inventory.append(inventory_entry)
            if len(members) < 2:
                continue
            anchor = members[0]
            for member in members[1:]:
                edge_body = {
                    "edge_type": edge_type,
                    "mode": mode,
                    "relationship_key": group["relationship_key"],
                    "relationship_key_sha256": relationship_key_sha256,
                    "left_candidate_id": anchor,
                    "right_candidate_id": member,
                    "evidence_sha256s": sorted(group["evidence_sha256s"]),
                }
                edge = {
                    **edge_body,
                    "edge_id": f"split-edge-{canonical_sha256(edge_body)}",
                }
                edge_ledger.append(edge)
                edges_by_type[edge_type].append(edge)
    relationship_inventory.sort(
        key=lambda item: (
            EDGE_TYPES.index(item["edge_type"]),
            canonical_json(item["relationship_key"]),
        )
    )
    edge_ledger.sort(
        key=lambda edge: (
            EDGE_TYPES.index(edge["edge_type"]),
            canonical_json(edge["relationship_key"]),
            edge["left_candidate_id"],
            edge["right_candidate_id"],
        )
    )

    union_find = _UnionFind(by_id)
    incremental_effects: list[dict[str, Any]] = []
    edge_application: dict[str, bool] = {}
    for edge_type in EDGE_TYPES:
        if normalized_policy["edge_modes"][edge_type] != "hard":
            continue
        before = _component_state(union_find, mass_by_candidate=mass_by_candidate)
        groups_bridging = 0
        for group in relationship_inventory:
            if group["edge_type"] != edge_type or group["member_count"] < 2:
                continue
            if len({union_find.find(member) for member in group["members"]}) > 1:
                groups_bridging += 1
        successful = 0
        redundant = 0
        for edge in edges_by_type[edge_type]:
            joined = union_find.union(
                edge["left_candidate_id"], edge["right_candidate_id"]
            )
            edge_application[edge["edge_id"]] = joined
            successful += int(joined)
            redundant += int(not joined)
        after = _component_state(union_find, mass_by_candidate=mass_by_candidate)
        groups = [
            group for group in relationship_inventory if group["edge_type"] == edge_type
        ]
        incremental_effects.append(
            {
                "edge_type": edge_type,
                "mode": "hard",
                "relationship_groups": len(groups),
                "multi_member_groups": sum(
                    group["member_count"] > 1 for group in groups
                ),
                "ledger_edges": len(edges_by_type[edge_type]),
                "groups_bridging_prior_components": groups_bridging,
                "successful_unions": successful,
                "redundant_edges": redundant,
                "before": before,
                "after": after,
            }
        )

    hard_components = union_find.components()
    final_hard_state = _component_state(union_find, mass_by_candidate=mass_by_candidate)
    for edge_type in EDGE_TYPES:
        if normalized_policy["edge_modes"][edge_type] != "diagnostic":
            continue
        hypothetical = union_find.copy()
        groups = [
            group for group in relationship_inventory if group["edge_type"] == edge_type
        ]
        groups_bridging = sum(
            len({union_find.find(member) for member in group["members"]}) > 1
            for group in groups
            if group["member_count"] > 1
        )
        potential_unions = 0
        already_connected = 0
        for edge in edges_by_type[edge_type]:
            bridges = union_find.find(edge["left_candidate_id"]) != union_find.find(
                edge["right_candidate_id"]
            )
            edge_application[edge["edge_id"]] = bridges
            potential_unions += int(
                hypothetical.union(
                    edge["left_candidate_id"], edge["right_candidate_id"]
                )
            )
            already_connected += int(not bridges)
        incremental_effects.append(
            {
                "edge_type": edge_type,
                "mode": "diagnostic",
                "relationship_groups": len(groups),
                "multi_member_groups": sum(
                    group["member_count"] > 1 for group in groups
                ),
                "ledger_edges": len(edges_by_type[edge_type]),
                "groups_bridging_hard_components": groups_bridging,
                "potential_component_unions": potential_unions,
                "edges_already_connected_by_hard_policy": already_connected,
                "hard_baseline": final_hard_state,
                "hypothetical_after_this_type_only": _component_state(
                    hypothetical, mass_by_candidate=mass_by_candidate
                ),
            }
        )
    incremental_effects.sort(key=lambda item: EDGE_TYPES.index(item["edge_type"]))

    edge_ledger = [
        {
            **edge,
            (
                "merged_at_hard_application"
                if edge["mode"] == "hard"
                else "bridges_final_hard_components"
            ): bool(edge_application.get(edge["edge_id"], False)),
        }
        for edge in edge_ledger
    ]
    edge_ledger_sha256 = canonical_sha256(edge_ledger)

    component_inventory: list[dict[str, Any]] = []
    for members in hard_components:
        component_body = {"candidate_ids": members}
        component_id = f"split-component-{canonical_sha256(component_body)}"
        molecule_ids = sorted(
            {
                molecule_id
                for member in members
                for molecule_id in by_id[member]["molecule_ids"]
            }
        )
        scaffold_ids = sorted(
            {
                scaffold_id
                for member in members
                for scaffold_id in by_id[member]["partition_scaffold_ids"]
            }
        )
        source_group_ids = sorted(
            {
                source_group_id
                for member in members
                for source_group_id in by_id[member]["source_group_ids"]
            }
        )
        lanes = sorted({str(by_id[member]["data_lane"]) for member in members})
        candidate_count_by_lane = {
            lane: sum(str(by_id[member]["data_lane"]) == lane for member in members)
            for lane in lanes
        }
        mass_units_by_lane = {
            lane: sum(
                mass_by_candidate[member]
                for member in members
                if str(by_id[member]["data_lane"]) == lane
            )
            for lane in lanes
        }
        component_inventory.append(
            {
                "component_id": component_id,
                "candidate_ids": members,
                "candidate_count": len(members),
                "mass_units": sum(mass_by_candidate[member] for member in members),
                "molecule_ids": molecule_ids,
                "molecule_count": len(molecule_ids),
                "partition_scaffold_ids": scaffold_ids,
                "partition_scaffold_count": len(scaffold_ids),
                "source_group_ids": source_group_ids,
                "source_group_count": len(source_group_ids),
                "data_lanes": lanes,
                "candidate_count_by_lane": candidate_count_by_lane,
                "mass_units_by_lane": mass_units_by_lane,
                "cross_lane": len(lanes) > 1,
            }
        )
    component_inventory.sort(key=lambda component: component["component_id"])
    component_inventory_sha256 = canonical_sha256(component_inventory)

    component_sizes = [
        int(component["candidate_count"]) for component in component_inventory
    ]
    component_masses = [
        int(component["mass_units"]) for component in component_inventory
    ]
    hard_component_census = {
        "component_count": len(component_inventory),
        "candidate_size_distribution": _distribution(component_sizes),
        "mass_units_distribution": _distribution(component_masses),
        "largest_component_candidate_fraction": {
            "numerator": max(component_sizes, default=0),
            "denominator": len(vertices),
        },
        "largest_component_mass_fraction": {
            "numerator": max(component_masses, default=0),
            "denominator": sum(component_masses),
        },
        "components": component_inventory,
        "component_inventory_sha256": component_inventory_sha256,
    }

    cross_lane_components = [
        {
            "component_id": component["component_id"],
            "data_lanes": component["data_lanes"],
            "candidate_count": component["candidate_count"],
            "mass_units": component["mass_units"],
        }
        for component in component_inventory
        if component["cross_lane"]
    ]
    lane_pair_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"component_count": 0, "candidate_count": 0, "mass_units": 0}
    )
    for component in component_inventory:
        for left, right in combinations(component["data_lanes"], 2):
            counts = lane_pair_counts[(left, right)]
            counts["component_count"] += 1
            counts["candidate_count"] += int(component["candidate_count"])
            counts["mass_units"] += int(component["mass_units"])
    cross_lane_overlap = {
        "components_with_multiple_lanes": len(cross_lane_components),
        "candidate_count_in_cross_lane_components": sum(
            int(component["candidate_count"]) for component in cross_lane_components
        ),
        "mass_units_in_cross_lane_components": sum(
            int(component["mass_units"]) for component in cross_lane_components
        ),
        "components": cross_lane_components,
        "lane_pairs": [
            {
                "left_lane": left,
                "right_lane": right,
                **lane_pair_counts[(left, right)],
            }
            for left, right in sorted(lane_pair_counts)
        ],
        "multi_lane_relationship_groups_by_edge_type": {
            edge_type: sum(
                group["cross_lane"]
                for group in relationship_inventory
                if group["edge_type"] == edge_type
            )
            for edge_type in EDGE_TYPES
        },
    }

    structural_blockers: list[str] = []
    if source_row_count == 0:
        structural_blockers.append("stream.empty")
    if invalid_rows:
        structural_blockers.append("stream.invalid_rows")
    if len(component_inventory) < len(PARTITION_ROLES):
        structural_blockers.append("components.fewer_than_four_nonempty_roles")
    authority_blockers = [
        f"external_authority.{requirement['id']}:{requirement['status']}"
        for requirement in editing_corpus_contract["admitted_record_contract"][
            "external_authority_requirements"
        ]
    ]
    authority_blockers.append("split_assignment.not_performed_by_component_census")
    if tuple(authority_blockers) != REQUIRED_AUTHORITY_BLOCKERS:
        raise EditingV2SplitCensusError(
            "editing-corpus external authority blockers have drifted from "
            "split-census schema version 3"
        )
    blockers = structural_blockers + authority_blockers

    relationship_summary = {
        edge_type: {
            "mode": normalized_policy["edge_modes"][edge_type],
            "groups": sum(
                group["edge_type"] == edge_type for group in relationship_inventory
            ),
            "multi_member_groups": sum(
                group["edge_type"] == edge_type and group["member_count"] > 1
                for group in relationship_inventory
            ),
            "ledger_edges": len(edges_by_type[edge_type]),
        }
        for edge_type in EDGE_TYPES
    }

    identities = {
        "policy_sha256": policy_sha256,
        "lane_contract_identity_sha256": lane_contract_identity_sha256,
        "identity_definition_contract_sha256": identity_definition_contract_sha256,
        "implementation_provenance_sha256": implementation_provenance_sha256,
        "row_stream_sha256": row_stream_sha256,
        "vertex_inventory_sha256": vertex_inventory_sha256,
        "invalid_rows_sha256": invalid_rows_sha256,
        "candidate_source_binding_inventory_sha256": (
            candidate_source_binding_inventory_sha256
        ),
        "relationship_inventory_sha256": canonical_sha256(relationship_inventory),
        "edge_ledger_sha256": edge_ledger_sha256,
        "component_inventory_sha256": component_inventory_sha256,
    }
    body = {
        "schema": SPLIT_CENSUS_SCHEMA,
        "schema_version": SPLIT_CENSUS_SCHEMA_VERSION,
        "status": "PASS" if not blockers else "BLOCKED",
        "status_scope": "CENSUS_STRUCTURAL_COMPLETION_ONLY",
        "census_computation_complete": True,
        "census_structural_complete": not structural_blockers,
        "corpus_ready": False,
        "split_ready": False,
        "training_ready": False,
        "split_ratios_selected": False,
        "split_assignment_selected": False,
        "split_assignment_authorized": False,
        "authority": {
            "corpus": "NO_CORPUS_READINESS_AUTHORITY",
            "split": "NO_SPLIT_AUTHORITY",
            "training": "NO_TRAINING_AUTHORITY",
        },
        "blockers": blockers,
        "partition_roles": list(PARTITION_ROLES),
        "policy": normalized_policy,
        "lane_contract_identity": lane_contract_identity,
        "identity_definitions": normalized_policy["identity_definitions"],
        "implementation_provenance": normalized_provenance,
        "source_stream": source_stream_payload,
        "identities": identities,
        "input_summary": {
            "source_rows": source_row_count,
            "valid_vertices": len(vertices),
            "invalid_rows": len(invalid_rows),
            "total_valid_mass_units": sum(
                int(vertex["mass_units"]) for vertex in vertices
            ),
            "data_lanes": sorted({str(vertex["data_lane"]) for vertex in vertices}),
            "ordered_contract_data_lanes": [
                str(lane["id"]) for lane in lane_contract_identity["ordered_lanes"]
            ],
        },
        "invalid_rows": invalid_rows,
        "candidate_source_binding_inventory": candidate_source_binding_inventory,
        "vertex_inventory": vertices,
        "relationship_summary": relationship_summary,
        "relationship_inventory": relationship_inventory,
        "edge_ledger": edge_ledger,
        "incremental_edge_type_effects": incremental_effects,
        "hard_component_census": hard_component_census,
        "cross_lane_overlap": cross_lane_overlap,
        "attainable_four_role_ranges": _attainable_ranges(component_inventory),
        "interpretation": {
            "hard_components_are_indivisible": True,
            "transformation_default": "diagnostic",
            "document_and_series_requirement": (
                "A declared document or series group is admitted only with "
                "source-scoped observed provenance."
            ),
            "source_group_requirement": (
                "Every candidate declares a source-scoped namespace and at least one "
                "source-group identity. Source-group edges are hard across data lanes."
            ),
            "no_ratio_or_assignment_claim": (
                "This artifact measures component constraints only. It neither "
                "chooses target ratios nor assigns a component to a role."
            ),
            "caller_partition_claims": (
                "Candidate rows contain no partition assignment. Exact partition "
                "resolution is a later, separately hash-bound artifact."
            ),
            "pass_scope": (
                "census_structural_complete means only that this component-census "
                "computation completed without structural blockers under the bound "
                "policy. It does not establish corpus readiness, split readiness, "
                "or training authority."
            ),
        },
    }
    return {**body, "census_sha256": canonical_sha256(body)}


def validate_split_component_census(census: Mapping[str, Any]) -> None:
    """Verify the outer identity and every material sub-artifact hash."""

    if not isinstance(census, Mapping):
        raise EditingV2SplitCensusError("split census must be an object")
    payload = dict(census)
    if (
        payload.get("schema") != SPLIT_CENSUS_SCHEMA
        or payload.get("schema_version") != SPLIT_CENSUS_SCHEMA_VERSION
    ):
        raise EditingV2SplitCensusError("unsupported split-census schema")
    supplied = payload.pop("census_sha256", None)
    if not isinstance(supplied, str) or supplied != canonical_sha256(payload):
        raise EditingV2SplitCensusError("split census SHA-256 mismatch")
    identities = payload.get("identities")
    if not isinstance(identities, Mapping):
        raise EditingV2SplitCensusError("split census lacks identities")
    checks = {
        "policy_sha256": payload.get("policy"),
        "lane_contract_identity_sha256": payload.get("lane_contract_identity"),
        "identity_definition_contract_sha256": payload.get("identity_definitions"),
        "implementation_provenance_sha256": payload.get("implementation_provenance"),
        "vertex_inventory_sha256": payload.get("vertex_inventory"),
        "invalid_rows_sha256": payload.get("invalid_rows"),
        "candidate_source_binding_inventory_sha256": payload.get(
            "candidate_source_binding_inventory"
        ),
        "relationship_inventory_sha256": payload.get("relationship_inventory"),
        "edge_ledger_sha256": payload.get("edge_ledger"),
        "component_inventory_sha256": payload.get("hard_component_census", {}).get(
            "components"
        ),
    }
    for field, value in checks.items():
        if identities.get(field) != canonical_sha256(value):
            raise EditingV2SplitCensusError(
                f"split census {field} does not match its payload"
            )
    lane_identity = payload.get("lane_contract_identity")
    if (
        not isinstance(lane_identity, Mapping)
        or set(lane_identity) != _LANE_CONTRACT_IDENTITY_FIELDS
    ):
        raise EditingV2SplitCensusError(
            "split census lane-contract identity fields disagree"
        )
    lane_identity_body = dict(lane_identity)
    lane_identity_sha256 = lane_identity_body.pop("identity_sha256")
    if lane_identity_sha256 != canonical_sha256(lane_identity_body):
        raise EditingV2SplitCensusError(
            "split census lane-contract internal identity does not match its payload"
        )
    if (
        lane_identity_body.get("schema") != LANE_CONTRACT_IDENTITY_SCHEMA
        or lane_identity_body.get("schema_version")
        != LANE_CONTRACT_IDENTITY_SCHEMA_VERSION
    ):
        raise EditingV2SplitCensusError(
            "split census lane-contract identity schema is unsupported"
        )
    normalized_policy = _normalize_policy(
        payload.get("policy"),
        expected_lane_contract_identity=lane_identity,
    )
    if normalized_policy != payload.get("policy"):
        raise EditingV2SplitCensusError("split census policy is not canonical")
    normalized_identity_definitions = _normalize_identity_definition_contract_or_raise(
        payload.get("identity_definitions"),
        field="split_census.identity_definitions",
    )
    if normalized_identity_definitions != payload.get("identity_definitions"):
        raise EditingV2SplitCensusError(
            "split census identity definitions are not canonical"
        )
    normalized_provenance = _normalize_implementation_provenance(
        payload.get("implementation_provenance")
    )
    if normalized_provenance != payload.get("implementation_provenance"):
        raise EditingV2SplitCensusError(
            "split census implementation provenance is not canonical"
        )
    if payload.get("policy", {}).get("lane_contract_identity") != payload.get(
        "lane_contract_identity"
    ):
        raise EditingV2SplitCensusError(
            "split census policy lane identity does not match its payload"
        )
    if payload.get("policy", {}).get("identity_definitions") != payload.get(
        "identity_definitions"
    ):
        raise EditingV2SplitCensusError(
            "split census policy identity definitions do not match its payload"
        )

    vertices = payload.get("vertex_inventory")
    if not isinstance(vertices, list):
        raise EditingV2SplitCensusError("split census vertex inventory must be a list")
    expected_vertex_fields = set(_ROW_FIELDS) | {
        "source_row_sha256",
        "row_sha256",
    }
    for index, vertex in enumerate(vertices):
        if not isinstance(vertex, Mapping) or set(vertex) != expected_vertex_fields:
            raise EditingV2SplitCensusError(
                f"split census vertex[{index}] fields disagree"
            )
        if _SHA256_RE.fullmatch(str(vertex["source_row_sha256"])) is None:
            raise EditingV2SplitCensusError(
                f"split census vertex[{index}] source-row identity is invalid"
            )
        row_body = {field: vertex[field] for field in _ROW_FIELDS}
        if vertex["row_sha256"] != canonical_sha256(row_body):
            raise EditingV2SplitCensusError(
                f"split census vertex[{index}] row hash disagrees"
            )
    expected_candidate_bindings = [
        {
            "candidate_id": row["candidate_id"],
            "candidate_ledger_row_sha256": row["candidate_ledger_row_sha256"],
            "candidate_envelope_sha256": row["candidate_envelope_sha256"],
        }
        for row in vertices
    ]
    if payload.get("candidate_source_binding_inventory") != expected_candidate_bindings:
        raise EditingV2SplitCensusError(
            "split census candidate source bindings disagree with vertices"
        )

    if payload.get("status_scope") != "CENSUS_STRUCTURAL_COMPLETION_ONLY":
        raise EditingV2SplitCensusError(
            "split census status scope is not structural-only"
        )
    if payload.get("status") != "BLOCKED":
        raise EditingV2SplitCensusError(
            "split census must remain blocked until external authorities resolve"
        )
    blockers = payload.get("blockers")
    if not isinstance(blockers, list) or any(
        blocker not in blockers for blocker in REQUIRED_AUTHORITY_BLOCKERS
    ):
        raise EditingV2SplitCensusError(
            "split census omits a mandatory external-authority blocker"
        )
    if payload.get("census_computation_complete") is not True:
        raise EditingV2SplitCensusError(
            "split census must record computation completion explicitly"
        )
    if payload.get("corpus_ready") is not False:
        raise EditingV2SplitCensusError("split census may not assert corpus readiness")
    if payload.get("split_ready") is not False:
        raise EditingV2SplitCensusError("split census may not assert split readiness")
    if payload.get("training_ready") is not False:
        raise EditingV2SplitCensusError(
            "split census may not assert training readiness"
        )
    for field in (
        "split_ratios_selected",
        "split_assignment_selected",
        "split_assignment_authorized",
    ):
        if payload.get(field) is not False:
            raise EditingV2SplitCensusError(f"split census may not assert {field}")
    expected_authority = {
        "corpus": "NO_CORPUS_READINESS_AUTHORITY",
        "split": "NO_SPLIT_AUTHORITY",
        "training": "NO_TRAINING_AUTHORITY",
    }
    if payload.get("authority") != expected_authority:
        raise EditingV2SplitCensusError(
            "split census must remain NO_SPLIT and NO_TRAINING authority"
        )


__all__ = [
    "CANDIDATE_ROW_SCHEMA",
    "CANDIDATE_ROW_SCHEMA_VERSION",
    "EDGE_TYPES",
    "EditingV2SplitCensusError",
    "IDENTITY_DEFINITION_CONTRACT_SCHEMA",
    "IDENTITY_DEFINITION_CONTRACT_SCHEMA_VERSION",
    "IMPLEMENTATION_PROVENANCE_SCHEMA",
    "IMPLEMENTATION_PROVENANCE_SCHEMA_VERSION",
    "LANE_CONTRACT_IDENTITY_SCHEMA",
    "LANE_CONTRACT_IDENTITY_SCHEMA_VERSION",
    "OBSERVED_GROUP_PROVENANCE_SCHEMA",
    "OBSERVED_GROUP_PROVENANCE_SCHEMA_VERSION",
    "PARTITION_ROLES",
    "RELATIONSHIP_NAMESPACE_SCHEMA",
    "RELATIONSHIP_NAMESPACE_SCHEMA_VERSION",
    "SPLIT_CENSUS_POLICY_SCHEMA",
    "SPLIT_CENSUS_POLICY_SCHEMA_VERSION",
    "SPLIT_CENSUS_SCHEMA",
    "SPLIT_CENSUS_SCHEMA_VERSION",
    "TRANSFORMATION_DEFINITION_SCHEMA",
    "TRANSFORMATION_DEFINITION_SCHEMA_VERSION",
    "build_split_component_census",
    "canonical_json",
    "canonical_sha256",
    "default_split_census_policy",
    "editing_lane_contract_identity",
    "validate_split_component_census",
]
