"""Resolve Editing-V2 lane shards into exact, non-authorizing Active8 sources.

Candidate materialization proves candidate identity and five-lane routing.  Split
assignment chooses one of four leakage-safe roles.  Neither artifact proves that
the copied/repartitioned packed traces belong to the current executable Active8
fiber.  That last decision remains the job of the whole-trace Active8 inventory.

This adapter closes only the physical boundary between those stages.  It
validates a separate ``resolved_packed_membership`` receipt which binds:

* the exact candidate materialization, split assignment, and five-lane registry;
* every new lane/role packed shard, packed manifest, and optional overlay byte;
* every output entry index to one selected candidate, its original immutable
  packed address and row hash, and its rewritten row hash and packed address.

The resolver rejects missing, duplicated, rejected, unassigned, or extra
candidate addresses.  It also reconciles trace/state/transition counts against
both the packed manifests and the lane registry.  The returned
``Active8SourceShard`` objects are inputs to Active8 admission, not evidence that
admission passed and never training authority.

Receipt schema version 3 intentionally rejects earlier versions. Version 2
bound rewritten rows and addresses but did not bind the candidate-provenance
source stream carried by split assignment.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from compose_v4.data.active8_trace_inventory import Active8SourceShard
from compose_v4.data.editing_v2_lane_registry import (
    ResolvedEditingV2LaneRegistry,
    ResolvedEditingV2Shard,
    resolve_editing_v2_lane_registry,
)
from compose_v4.data.editing_v2_packed_candidate_materializer import (
    CANDIDATE_ROWS_FILENAME,
    MATERIALIZATION_FILENAME,
    canonical_sha256,
    file_sha256,
    validate_packed_candidate_materialization,
)
from compose_v4.data.editing_v2_split_assignment import (
    SPLIT_ASSIGNMENT_SCHEMA,
    SPLIT_ASSIGNMENT_STATUS,
    SPLIT_ASSIGNMENT_VERSION,
    EditingV2SplitAssignmentError,
    validate_candidate_source_stream,
)
from compose_v4.data.editing_v2_split_census import PARTITION_ROLES
from compose_v4.data.packed_trace_store import (
    PACKED_STORE_SCHEMA,
    PACKED_STORE_SCHEMA_VERSION,
    manifest_path_for,
)
from compose_v4.data.provenance_overlay import overlay_path_for

RESOLVED_PACKED_MEMBERSHIP_SCHEMA = "compose.editing_v2_resolved_packed_membership"
RESOLVED_PACKED_MEMBERSHIP_SCHEMA_VERSION = 3
RESOLVED_PACKED_MEMBERSHIP_STATUS = (
    "COMPLETE_PHYSICAL_MEMBERSHIP_ACTIVE8_NOT_RUN_NO_TRAINING_AUTHORITY"
)
ACTIVE8_ADMISSION_STATUS = "NOT_RUN"
CANDIDATE_AUTHORITY = "IDENTITY_AND_LANE_ROUTING_ONLY"
REQUIRED_BLOCKERS = (
    "active8_whole_trace_admission.not_run",
    "gate_zero.not_run",
    "training.not_authorized",
)

_SHA256_HEX = frozenset("0123456789abcdef")
_RECEIPT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "candidate_materialization_authority",
    "active8_admission_status",
    "membership_complete",
    "blockers",
    "inputs",
    "shards",
    "receipt_sha256",
}
_INPUT_FIELDS = {
    "candidate_materialization",
    "split_assignment",
    "lane_registry",
}
_CANDIDATE_INPUT_FIELDS = {
    "manifest_file_sha256",
    "manifest_sha256",
    "rows_file_sha256",
    "rows_semantic_sha256",
    "address_stream_sha256",
}
_SPLIT_INPUT_FIELDS = {
    "file_sha256",
    "assignment_sha256",
    "candidate_resolution_stream_sha256",
    "source_stream_sha256",
}
_LANE_REGISTRY_INPUT_FIELDS = {
    "registry_file_sha256",
    "registry_sha256",
    "editing_corpus_contract_file_sha256",
    "editing_corpus_contract_id",
    "lane_completion_manifest_file_sha256",
    "lane_completion_sha256",
    "lane_shard_inventory_sha256",
}
_SHARD_FIELDS = {
    "data_lane",
    "partition_role",
    "relative_path",
    "artifact_path",
    "file_sha256",
    "packed_manifest_artifact_path",
    "packed_manifest_file_sha256",
    "packed_provenance_overlay_artifact_path",
    "packed_provenance_overlay_file_sha256",
    "entry_memberships",
}
_MEMBERSHIP_FIELDS = {
    "output_entry_index",
    "candidate_id",
    "original_packed_address",
    "original_packed_row_sha256",
    "output_packed_row_sha256",
    "output_packed_address",
}
_ORIGINAL_ADDRESS_FIELDS = {
    "source_asset",
    "packed_address",
}
_SOURCE_ASSET_FIELDS = {
    "source_asset_id",
    "source_asset_path",
    "source_asset_sha256",
}
_PACKED_ADDRESS_FIELDS = {
    "relative_path",
    "packed_shard_file_sha256",
    "packed_shard_manifest_file_sha256",
    "historical_provenance_overlay_file_sha256",
    "entry_index",
    "trace_id",
    "layer",
    "cache_partition",
    "path_length",
    "address_sha256",
}
_OUTPUT_PACKED_ADDRESS_FIELDS = {
    "packed_shard_file_sha256",
    "packed_shard_manifest_file_sha256",
    "packed_provenance_overlay_file_sha256",
    "packed_shard_name",
    "entry_index",
    "trace_id",
    "layer",
    "partition",
    "source_key",
    "target_key",
    "path_length",
    "address_sha256",
}
_RESOLUTION_FIELDS = {
    "candidate_id",
    "component_id",
    "status",
    "assigned_role",
    "source_endpoint_role",
    "target_endpoint_role",
}


class EditingV2Active8SourceAdapterError(ValueError):
    """The selected Editing-V2 packed membership is incomplete or inconsistent."""


@dataclass(frozen=True)
class ResolvedActive8SourceBinding:
    """One exact physical shard and its pre-Active8 byte identities."""

    source: Active8SourceShard
    artifact_path: str
    packed_shard_file_sha256: str
    packed_manifest_artifact_path: str
    packed_manifest_file_sha256: str
    packed_provenance_overlay_artifact_path: str | None
    packed_provenance_overlay_file_sha256: str | None
    candidate_ids: tuple[str, ...]
    original_address_sha256s: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedEditingV2Active8Sources:
    """Complete five-lane/four-role inputs awaiting whole-trace admission."""

    membership_receipt_path: Path
    membership_receipt_file_sha256: str
    membership_receipt_sha256: str
    candidate_materialization_manifest_sha256: str
    candidate_provenance_source_stream: Mapping[str, Any]
    split_assignment_sha256: str
    lane_registry_sha256: str
    bindings: tuple[ResolvedActive8SourceBinding, ...]
    source_manifest: Mapping[str, Any]

    @property
    def shards(self) -> tuple[Active8SourceShard, ...]:
        """Return the exact physical sources expected by Active8 inventory."""

        return tuple(binding.source for binding in self.bindings)


def _require_mapping(
    value: object,
    *,
    fields: set[str],
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EditingV2Active8SourceAdapterError(f"{field} must be an object")
    actual = set(value)
    if actual != fields:
        raise EditingV2Active8SourceAdapterError(
            f"{field} fields disagree; "
            f"missing={sorted(fields - actual)}, "
            f"unexpected={sorted(actual - fields)}"
        )
    return value


def _require_sha256(value: object, *, field: str) -> str:
    digest = value if isinstance(value, str) else ""
    if len(digest) != 64 or any(character not in _SHA256_HEX for character in digest):
        raise EditingV2Active8SourceAdapterError(f"{field} must be a full lowercase SHA-256")
    return digest


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise EditingV2Active8SourceAdapterError(
            f"{field} must be nonempty whitespace-normalized text"
        )
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise EditingV2Active8SourceAdapterError(f"{field} must be a nonnegative integer")
    return value


def _load_json(path: Path, *, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingV2Active8SourceAdapterError(f"cannot load {field}: {path}") from error
    if not isinstance(value, Mapping):
        raise EditingV2Active8SourceAdapterError(f"{field} must be an object")
    return value


def _semantic_hash(
    value: Mapping[str, Any],
    *,
    hash_field: str,
    field: str,
) -> str:
    supplied = _require_sha256(value.get(hash_field), field=f"{field}.{hash_field}")
    body = {key: item for key, item in value.items() if key != hash_field}
    if canonical_sha256(body) != supplied:
        raise EditingV2Active8SourceAdapterError(f"{field} semantic SHA-256 disagrees")
    return supplied


def _candidate_input_identity(
    candidate_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    rows = manifest.get("rows")
    if not isinstance(rows, Mapping):
        raise EditingV2Active8SourceAdapterError(
            "candidate materialization rows identity is absent"
        )
    return {
        "manifest_file_sha256": file_sha256(candidate_root / MATERIALIZATION_FILENAME),
        "manifest_sha256": _require_sha256(
            manifest.get("manifest_sha256"),
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


def _load_candidate_rows(
    candidate_root: Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    rows_path = candidate_root / CANDIDATE_ROWS_FILENAME
    candidates: dict[str, Mapping[str, Any]] = {}
    address_owner: dict[str, str] = {}
    with rows_path.open("rb") as handle:
        for index, raw_line in enumerate(handle):
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EditingV2Active8SourceAdapterError(
                    f"candidate header {index} is invalid JSON"
                ) from error
            if not isinstance(row, Mapping):
                raise EditingV2Active8SourceAdapterError(
                    f"candidate header {index} must be an object"
                )
            candidate_id = _require_text(
                row.get("candidate_id"),
                field=f"candidate header {index}.candidate_id",
            )
            if candidate_id in candidates:
                raise EditingV2Active8SourceAdapterError(
                    f"duplicate candidate_id in materialization: {candidate_id!r}"
                )
            address = row.get("packed_address")
            if not isinstance(address, Mapping):
                raise EditingV2Active8SourceAdapterError(
                    f"candidate {candidate_id!r} lacks its original packed address"
                )
            address_sha256 = _require_sha256(
                address.get("address_sha256"),
                field=f"candidate {candidate_id!r}.packed_address.address_sha256",
            )
            previous = address_owner.get(address_sha256)
            if previous is not None:
                raise EditingV2Active8SourceAdapterError(
                    "duplicate original packed address in candidate materialization: "
                    f"{address_sha256} belongs to {previous!r} and {candidate_id!r}"
                )
            candidates[candidate_id] = row
            address_owner[address_sha256] = candidate_id
    if not candidates:
        raise EditingV2Active8SourceAdapterError(
            "candidate materialization contains no candidate rows"
        )
    return candidates, address_owner


def _validate_split_assignment(
    path: Path,
) -> tuple[Mapping[str, Any], dict[str, str]]:
    assignment = _load_json(path, field="split assignment")
    _semantic_hash(
        assignment,
        hash_field="assignment_sha256",
        field="split assignment",
    )
    if (
        assignment.get("schema") != SPLIT_ASSIGNMENT_SCHEMA
        or assignment.get("schema_version") != SPLIT_ASSIGNMENT_VERSION
        or assignment.get("status") != SPLIT_ASSIGNMENT_STATUS
        or assignment.get("training_authorized") is not False
        or assignment.get("split_assignment_selected") is not True
        or assignment.get("split_assignment_authorized") is not False
        or tuple(assignment.get("partition_roles") or ()) != PARTITION_ROLES
    ):
        raise EditingV2Active8SourceAdapterError(
            "split assignment identity, selection status, roles, or authority disagrees"
        )
    gates = assignment.get("gate_results")
    if (
        not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
    ):
        raise EditingV2Active8SourceAdapterError(
            "split assignment gates must all pass before packed materialization"
        )
    raw_resolutions = assignment.get("candidate_resolutions")
    if not isinstance(raw_resolutions, list) or not raw_resolutions:
        raise EditingV2Active8SourceAdapterError(
            "split assignment candidate resolutions must be nonempty"
        )
    expected_stream_sha256 = canonical_sha256(raw_resolutions)
    stream_sha256 = _require_sha256(
        assignment.get("candidate_resolution_stream_sha256"),
        field="split assignment candidate_resolution_stream_sha256",
    )
    if stream_sha256 != expected_stream_sha256:
        raise EditingV2Active8SourceAdapterError(
            "split assignment candidate resolution stream SHA-256 disagrees"
        )
    roles: dict[str, str] = {}
    ordered_ids: list[str] = []
    for index, raw_resolution in enumerate(raw_resolutions):
        resolution = _require_mapping(
            raw_resolution,
            fields=_RESOLUTION_FIELDS,
            field=f"split assignment candidate_resolutions[{index}]",
        )
        candidate_id = _require_text(
            resolution["candidate_id"],
            field=f"split assignment candidate_resolutions[{index}].candidate_id",
        )
        role = _require_text(
            resolution["assigned_role"],
            field=f"split assignment candidate_resolutions[{index}].assigned_role",
        )
        if (
            candidate_id in roles
            or role not in PARTITION_ROLES
            or resolution["status"] != "assigned"
            or resolution["source_endpoint_role"] != role
            or resolution["target_endpoint_role"] != role
        ):
            raise EditingV2Active8SourceAdapterError(
                f"split resolution for candidate {candidate_id!r} is duplicate or inconsistent"
            )
        roles[candidate_id] = role
        _require_text(
            resolution["component_id"],
            field=f"split resolution {candidate_id!r}.component_id",
        )
        ordered_ids.append(candidate_id)
    if ordered_ids != sorted(ordered_ids):
        raise EditingV2Active8SourceAdapterError(
            "split candidate resolutions must be sorted by candidate_id"
        )
    try:
        validate_candidate_source_stream(
            assignment.get("source_stream"),
            expected_nonempty_rows=len(roles),
        )
    except EditingV2SplitAssignmentError as error:
        raise EditingV2Active8SourceAdapterError(
            "split assignment candidate provenance source-stream identity disagrees"
        ) from error
    return assignment, roles


def _split_input_identity(
    path: Path,
    assignment: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "file_sha256": file_sha256(path),
        "assignment_sha256": _require_sha256(
            assignment.get("assignment_sha256"),
            field="split assignment assignment_sha256",
        ),
        "candidate_resolution_stream_sha256": _require_sha256(
            assignment.get("candidate_resolution_stream_sha256"),
            field="split assignment candidate_resolution_stream_sha256",
        ),
        "source_stream_sha256": _require_sha256(
            assignment.get("source_stream", {}).get("source_stream_sha256"),
            field="split assignment candidate provenance source_stream_sha256",
        ),
    }


def _lane_sha_mapping(
    registry: ResolvedEditingV2LaneRegistry,
    attribute: str,
) -> dict[str, str]:
    return {
        lane.lane_id: _require_sha256(
            getattr(lane, attribute),
            field=f"lane {lane.lane_id}.{attribute}",
        )
        for lane in registry.lanes
    }


def _lane_registry_input_identity(
    registry: ResolvedEditingV2LaneRegistry,
) -> dict[str, Any]:
    return {
        "registry_file_sha256": registry.registry_file_sha256,
        "registry_sha256": registry.registry_sha256,
        "editing_corpus_contract_file_sha256": (registry.editing_corpus_contract_file_sha256),
        "editing_corpus_contract_id": registry.editing_corpus_contract_id,
        "lane_completion_manifest_file_sha256": _lane_sha_mapping(
            registry,
            "completion_manifest_file_sha256",
        ),
        "lane_completion_sha256": _lane_sha_mapping(
            registry,
            "completion_sha256",
        ),
        "lane_shard_inventory_sha256": _lane_sha_mapping(
            registry,
            "shard_inventory_sha256",
        ),
    }


def _validate_receipt_inputs(
    receipt_inputs: object,
    *,
    candidate_identity: Mapping[str, Any],
    split_identity: Mapping[str, Any],
    registry_identity: Mapping[str, Any],
) -> None:
    inputs = _require_mapping(
        receipt_inputs,
        fields=_INPUT_FIELDS,
        field="membership receipt inputs",
    )
    candidate = _require_mapping(
        inputs["candidate_materialization"],
        fields=_CANDIDATE_INPUT_FIELDS,
        field="membership receipt inputs.candidate_materialization",
    )
    split = _require_mapping(
        inputs["split_assignment"],
        fields=_SPLIT_INPUT_FIELDS,
        field="membership receipt inputs.split_assignment",
    )
    registry = _require_mapping(
        inputs["lane_registry"],
        fields=_LANE_REGISTRY_INPUT_FIELDS,
        field="membership receipt inputs.lane_registry",
    )
    if (
        dict(candidate) != dict(candidate_identity)
        or dict(split) != dict(split_identity)
        or dict(registry) != dict(registry_identity)
    ):
        raise EditingV2Active8SourceAdapterError(
            "membership receipt input identities disagree with the exact "
            "candidate materialization, split assignment, or lane registry"
        )


def _expected_original_address(row: Mapping[str, Any]) -> dict[str, Any]:
    source = _require_mapping(
        row.get("source_asset"),
        fields=_SOURCE_ASSET_FIELDS,
        field=f"candidate {row.get('candidate_id')!r}.source_asset",
    )
    address = _require_mapping(
        row.get("packed_address"),
        fields=_PACKED_ADDRESS_FIELDS,
        field=f"candidate {row.get('candidate_id')!r}.packed_address",
    )
    _require_sha256(
        source["source_asset_sha256"],
        field=f"candidate {row.get('candidate_id')!r}.source_asset_sha256",
    )
    for name in (
        "packed_shard_file_sha256",
        "packed_shard_manifest_file_sha256",
        "address_sha256",
    ):
        _require_sha256(
            address[name],
            field=f"candidate {row.get('candidate_id')!r}.packed_address.{name}",
        )
    historical_overlay = address["historical_provenance_overlay_file_sha256"]
    if historical_overlay is not None:
        _require_sha256(
            historical_overlay,
            field=(
                f"candidate {row.get('candidate_id')!r}.packed_address."
                "historical_provenance_overlay_file_sha256"
            ),
        )
    return {
        "source_asset": dict(source),
        "packed_address": dict(address),
    }


def _packed_manifest_identity(
    shard: ResolvedEditingV2Shard,
    *,
    receipt_shard: Mapping[str, Any],
) -> tuple[str, str, str | None, str | None, Mapping[str, Any]]:
    manifest_path = manifest_path_for(shard.local_path)
    expected_manifest_artifact = str(
        PurePosixPath(shard.artifact_path).with_suffix(".manifest.json")
    )
    if receipt_shard["packed_manifest_artifact_path"] != expected_manifest_artifact:
        raise EditingV2Active8SourceAdapterError(
            f"packed manifest artifact path disagrees for {shard.artifact_path}"
        )
    if not manifest_path.is_file():
        raise EditingV2Active8SourceAdapterError(f"packed manifest is absent: {manifest_path}")
    manifest_sha256 = file_sha256(manifest_path)
    if manifest_sha256 != _require_sha256(
        receipt_shard["packed_manifest_file_sha256"],
        field=f"packed manifest SHA-256 for {shard.artifact_path}",
    ):
        raise EditingV2Active8SourceAdapterError(
            f"packed manifest SHA-256 disagrees for {shard.artifact_path}"
        )
    manifest = _load_json(manifest_path, field="packed manifest")
    if (
        manifest.get("schema") != PACKED_STORE_SCHEMA
        or manifest.get("schema_version") != PACKED_STORE_SCHEMA_VERSION
    ):
        raise EditingV2Active8SourceAdapterError(
            f"packed manifest schema is unsupported for {shard.artifact_path}"
        )

    overlay_path = overlay_path_for(shard.local_path)
    expected_overlay_artifact = f"{shard.artifact_path}.provenance.json"
    declared_overlay_artifact = receipt_shard["packed_provenance_overlay_artifact_path"]
    declared_overlay_sha256 = receipt_shard["packed_provenance_overlay_file_sha256"]
    if declared_overlay_artifact is None or declared_overlay_sha256 is None:
        if declared_overlay_artifact is not None or declared_overlay_sha256 is not None:
            raise EditingV2Active8SourceAdapterError(
                f"packed overlay path/hash nullability disagrees for {shard.artifact_path}"
            )
        if overlay_path.exists():
            raise EditingV2Active8SourceAdapterError(
                f"packed overlay exists but is unbound for {shard.artifact_path}"
            )
        return (
            expected_manifest_artifact,
            manifest_sha256,
            None,
            None,
            manifest,
        )
    if declared_overlay_artifact != expected_overlay_artifact:
        raise EditingV2Active8SourceAdapterError(
            f"packed overlay artifact path disagrees for {shard.artifact_path}"
        )
    overlay_sha256 = _require_sha256(
        declared_overlay_sha256,
        field=f"packed overlay SHA-256 for {shard.artifact_path}",
    )
    if not overlay_path.is_file() or file_sha256(overlay_path) != overlay_sha256:
        raise EditingV2Active8SourceAdapterError(
            f"packed overlay is absent or its SHA-256 disagrees for {shard.artifact_path}"
        )
    return (
        expected_manifest_artifact,
        manifest_sha256,
        expected_overlay_artifact,
        overlay_sha256,
        manifest,
    )


def _iter_output_rows(
    path: Path,
    *,
    max_row_bytes: int = 16 * 1024 * 1024,
):
    if type(max_row_bytes) is not int or max_row_bytes <= 0:
        raise EditingV2Active8SourceAdapterError("output row byte bound must be positive")
    with gzip.open(path, "rb") as handle:
        output_index = 0
        while True:
            raw_line = handle.readline(max_row_bytes + 1)
            if not raw_line:
                break
            if len(raw_line) > max_row_bytes or not raw_line.endswith(b"\n"):
                raise EditingV2Active8SourceAdapterError(
                    f"packed output row exceeds its byte bound or lacks newline: {path}"
                )
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EditingV2Active8SourceAdapterError(
                    f"packed output row {output_index} is invalid JSON: {path}"
                ) from error
            if not isinstance(row, Mapping) or set(row) != {"trace", "states"}:
                raise EditingV2Active8SourceAdapterError(
                    f"packed output row {output_index} must contain trace and states"
                )
            yield output_index, row
            output_index += 1


def _output_packed_address(
    *,
    row: Mapping[str, Any],
    shard: ResolvedEditingV2Shard,
    manifest_sha256: str,
    overlay_sha256: str | None,
    output_entry_index: int,
) -> dict[str, Any]:
    trace = row.get("trace")
    if not isinstance(trace, Mapping):
        raise EditingV2Active8SourceAdapterError(
            f"packed output row {output_entry_index} lacks a trace envelope"
        )
    body = {
        "packed_shard_file_sha256": shard.sha256,
        "packed_shard_manifest_file_sha256": manifest_sha256,
        "packed_provenance_overlay_file_sha256": overlay_sha256,
        "packed_shard_name": shard.local_path.name,
        "entry_index": output_entry_index,
        "trace_id": _require_text(
            trace.get("trace_id"),
            field=f"packed output row {output_entry_index}.trace_id",
        ),
        "layer": _require_text(
            trace.get("layer"),
            field=f"packed output row {output_entry_index}.layer",
        ),
        "partition": _require_text(
            trace.get("partition"),
            field=f"packed output row {output_entry_index}.partition",
        ),
        "source_key": _require_text(
            trace.get("source_key"),
            field=f"packed output row {output_entry_index}.source_key",
        ),
        "target_key": _require_text(
            trace.get("target_key"),
            field=f"packed output row {output_entry_index}.target_key",
        ),
        "path_length": _require_nonnegative_int(
            trace.get("path_length"),
            field=f"packed output row {output_entry_index}.path_length",
        ),
    }
    return {**body, "address_sha256": canonical_sha256(body)}


def _registry_shards(
    registry: ResolvedEditingV2LaneRegistry,
) -> tuple[ResolvedEditingV2Shard, ...]:
    role_order = {role: index for index, role in enumerate(PARTITION_ROLES)}
    lane_order = {lane.lane_id: index for index, lane in enumerate(registry.lanes)}
    lane_roots = {lane.lane_id: PurePosixPath(lane.declared_root) for lane in registry.lanes}
    shards = tuple(shard for lane in registry.lanes for shard in lane.shards)
    return tuple(
        sorted(
            shards,
            key=lambda shard: (
                lane_order[shard.lane_id],
                role_order[shard.partition],
                str(PurePosixPath(shard.artifact_path).relative_to(lane_roots[shard.lane_id])),
            ),
        )
    )


def _registry_relative_path(
    registry: ResolvedEditingV2LaneRegistry,
    shard: ResolvedEditingV2Shard,
) -> str:
    lane = next(lane for lane in registry.lanes if lane.lane_id == shard.lane_id)
    return str(PurePosixPath(shard.artifact_path).relative_to(PurePosixPath(lane.declared_root)))


def _validate_complete_lane_role_coverage(
    registry: ResolvedEditingV2LaneRegistry,
    shards: Sequence[ResolvedEditingV2Shard],
) -> None:
    expected = {(lane.lane_id, role) for lane in registry.lanes for role in PARTITION_ROLES}
    observed = {(shard.lane_id, shard.partition) for shard in shards}
    if observed != expected:
        raise EditingV2Active8SourceAdapterError(
            "resolved packed sources do not cover all five-lane/four-role cells; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def resolve_editing_v2_active8_sources(
    *,
    candidate_materialization_dir: str | Path,
    split_assignment_path: str | Path,
    editing_corpus_contract_path: str | Path,
    lane_registry_path: str | Path,
    artifact_root: str | Path,
    membership_receipt_path: str | Path,
    expected_receipt_sha256: str | None = None,
) -> ResolvedEditingV2Active8Sources:
    """Validate exact copied membership and return sources awaiting Active8."""

    candidate_root = Path(candidate_materialization_dir)
    candidate_manifest = validate_packed_candidate_materialization(candidate_root)
    candidates, _ = _load_candidate_rows(candidate_root)
    candidate_identity = _candidate_input_identity(
        candidate_root,
        candidate_manifest,
    )

    split_path = Path(split_assignment_path)
    assignment, assigned_roles = _validate_split_assignment(split_path)
    split_identity = _split_input_identity(split_path, assignment)

    routed_candidates = {
        candidate_id
        for candidate_id, row in candidates.items()
        if row.get("disposition") == "routed_candidate"
    }
    rejected_candidates = {
        candidate_id
        for candidate_id, row in candidates.items()
        if row.get("disposition") == "rejected_candidate"
    }
    if routed_candidates | rejected_candidates != set(candidates):
        raise EditingV2Active8SourceAdapterError(
            "candidate materialization contains an unknown disposition"
        )
    selected = set(assigned_roles)
    if selected != routed_candidates:
        raise EditingV2Active8SourceAdapterError(
            "split assignment must select every routed candidate exactly once and no "
            "rejected candidate; "
            f"missing={sorted(routed_candidates - selected)}, "
            f"unselected_or_rejected={sorted(selected - routed_candidates)}"
        )

    registry = resolve_editing_v2_lane_registry(
        editing_corpus_contract_path=editing_corpus_contract_path,
        registry_path=lane_registry_path,
        artifact_root=artifact_root,
    )
    if tuple(registry.partition_roles) != PARTITION_ROLES or len(registry.lanes) != 5:
        raise EditingV2Active8SourceAdapterError(
            "lane registry must resolve exactly five lanes and four roles"
        )
    registry_identity = _lane_registry_input_identity(registry)
    resolved_registry_shards = _registry_shards(registry)
    _validate_complete_lane_role_coverage(registry, resolved_registry_shards)

    receipt_path = Path(membership_receipt_path)
    receipt = _require_mapping(
        _load_json(receipt_path, field="resolved packed membership receipt"),
        fields=_RECEIPT_FIELDS,
        field="resolved packed membership receipt",
    )
    receipt_sha256 = _semantic_hash(
        receipt,
        hash_field="receipt_sha256",
        field="resolved packed membership receipt",
    )
    if expected_receipt_sha256 is not None and receipt_sha256 != _require_sha256(
        expected_receipt_sha256,
        field="expected_receipt_sha256",
    ):
        raise EditingV2Active8SourceAdapterError(
            "resolved packed membership receipt identity differs from the expected identity"
        )
    blockers = receipt["blockers"]
    if (
        receipt["schema"] != RESOLVED_PACKED_MEMBERSHIP_SCHEMA
        or receipt["schema_version"] != RESOLVED_PACKED_MEMBERSHIP_SCHEMA_VERSION
        or receipt["status"] != RESOLVED_PACKED_MEMBERSHIP_STATUS
        or receipt["training_authorized"] is not False
        or receipt["candidate_materialization_authority"] != CANDIDATE_AUTHORITY
        or receipt["active8_admission_status"] != ACTIVE8_ADMISSION_STATUS
        or receipt["membership_complete"] is not True
        or not isinstance(blockers, list)
        or tuple(blockers) != REQUIRED_BLOCKERS
    ):
        raise EditingV2Active8SourceAdapterError(
            "membership receipt schema, authority boundary, blockers, or status disagrees"
        )
    _validate_receipt_inputs(
        receipt["inputs"],
        candidate_identity=candidate_identity,
        split_identity=split_identity,
        registry_identity=registry_identity,
    )

    raw_receipt_shards = receipt["shards"]
    if not isinstance(raw_receipt_shards, list):
        raise EditingV2Active8SourceAdapterError("membership receipt shards must be a list")
    if len(raw_receipt_shards) != len(resolved_registry_shards):
        raise EditingV2Active8SourceAdapterError(
            "membership receipt must bind every lane-registry shard exactly once"
        )

    seen_candidates: set[str] = set()
    seen_addresses: set[str] = set()
    seen_artifacts: set[str] = set()
    seen_source_digests: set[str] = set()
    bindings: list[ResolvedActive8SourceBinding] = []
    for shard_index, (raw_receipt_shard, registry_shard) in enumerate(
        zip(raw_receipt_shards, resolved_registry_shards, strict=True)
    ):
        registry_relative_path = _registry_relative_path(
            registry,
            registry_shard,
        )
        receipt_shard = _require_mapping(
            raw_receipt_shard,
            fields=_SHARD_FIELDS,
            field=f"membership receipt shards[{shard_index}]",
        )
        if (
            receipt_shard["data_lane"] != registry_shard.lane_id
            or receipt_shard["partition_role"] != registry_shard.partition
            or receipt_shard["relative_path"] != registry_relative_path
            or receipt_shard["artifact_path"] != registry_shard.artifact_path
            or receipt_shard["file_sha256"] != registry_shard.sha256
        ):
            raise EditingV2Active8SourceAdapterError(
                f"membership receipt shard {shard_index} disagrees with lane registry order or identity"
            )
        if registry_shard.artifact_path in seen_artifacts:
            raise EditingV2Active8SourceAdapterError(
                f"duplicate output packed shard: {registry_shard.artifact_path}"
            )
        seen_artifacts.add(registry_shard.artifact_path)
        if registry_shard.sha256 in seen_source_digests:
            raise EditingV2Active8SourceAdapterError(
                "one physical packed source digest is declared more than once: "
                f"{registry_shard.sha256}"
            )
        seen_source_digests.add(registry_shard.sha256)
        (
            manifest_artifact_path,
            manifest_sha256,
            overlay_artifact_path,
            overlay_sha256,
            packed_manifest,
        ) = _packed_manifest_identity(
            registry_shard,
            receipt_shard=receipt_shard,
        )
        memberships = receipt_shard["entry_memberships"]
        if not isinstance(memberships, list) or not memberships:
            raise EditingV2Active8SourceAdapterError(
                f"output shard {registry_shard.artifact_path} has no entry memberships"
            )
        candidate_ids: list[str] = []
        address_sha256s: list[str] = []
        state_count = 0
        transition_count = 0
        physical_output_rows = iter(_iter_output_rows(registry_shard.local_path))
        for output_index, raw_membership in enumerate(memberships):
            try:
                physical_output_index, output_row = next(physical_output_rows)
            except StopIteration as error:
                raise EditingV2Active8SourceAdapterError(
                    f"physical output row count disagrees for {registry_shard.artifact_path}"
                ) from error
            if physical_output_index != output_index:
                raise EditingV2Active8SourceAdapterError(
                    f"physical output indices are not contiguous for {registry_shard.artifact_path}"
                )
            membership = _require_mapping(
                raw_membership,
                fields=_MEMBERSHIP_FIELDS,
                field=(
                    f"membership receipt shards[{shard_index}].entry_memberships[{output_index}]"
                ),
            )
            if membership["output_entry_index"] != output_index:
                raise EditingV2Active8SourceAdapterError(
                    f"output entry indices must be contiguous for {registry_shard.artifact_path}"
                )
            candidate_id = _require_text(
                membership["candidate_id"],
                field=f"membership shard {shard_index} candidate_id",
            )
            row = candidates.get(candidate_id)
            if (
                row is None
                or candidate_id not in assigned_roles
                or row.get("data_lane") != registry_shard.lane_id
                or assigned_roles[candidate_id] != registry_shard.partition
            ):
                raise EditingV2Active8SourceAdapterError(
                    f"candidate {candidate_id!r} is absent, unselected, or placed in the wrong lane/role"
                )
            if candidate_id in seen_candidates:
                raise EditingV2Active8SourceAdapterError(
                    f"selected candidate appears more than once: {candidate_id!r}"
                )
            original = _require_mapping(
                membership["original_packed_address"],
                fields=_ORIGINAL_ADDRESS_FIELDS,
                field=f"candidate {candidate_id!r}.original_packed_address",
            )
            expected_original = _expected_original_address(row)
            if dict(original) != expected_original:
                raise EditingV2Active8SourceAdapterError(
                    f"candidate {candidate_id!r} original packed address disagrees"
                )
            _require_sha256(
                membership["original_packed_row_sha256"],
                field=f"candidate {candidate_id!r}.original_packed_row_sha256",
            )
            output_row_sha256 = canonical_sha256(output_row)
            if membership["output_packed_row_sha256"] != output_row_sha256:
                raise EditingV2Active8SourceAdapterError(
                    f"candidate {candidate_id!r} output packed row SHA-256 disagrees"
                )
            declared_output_address = _require_mapping(
                membership["output_packed_address"],
                fields=_OUTPUT_PACKED_ADDRESS_FIELDS,
                field=f"candidate {candidate_id!r}.output_packed_address",
            )
            expected_output_address = _output_packed_address(
                row=output_row,
                shard=registry_shard,
                manifest_sha256=manifest_sha256,
                overlay_sha256=overlay_sha256,
                output_entry_index=output_index,
            )
            if dict(declared_output_address) != expected_output_address:
                raise EditingV2Active8SourceAdapterError(
                    f"candidate {candidate_id!r} output packed address disagrees"
                )
            if (
                expected_output_address["layer"] != registry_shard.lane_id
                or expected_output_address["partition"] != registry_shard.partition
            ):
                raise EditingV2Active8SourceAdapterError(
                    f"candidate {candidate_id!r} output trace envelope disagrees with lane/role"
                )
            address_sha256 = expected_original["packed_address"]["address_sha256"]
            if address_sha256 in seen_addresses:
                raise EditingV2Active8SourceAdapterError(
                    f"original packed address appears more than once: {address_sha256}"
                )
            seen_candidates.add(candidate_id)
            seen_addresses.add(address_sha256)
            candidate_ids.append(candidate_id)
            address_sha256s.append(address_sha256)
            state_count += _require_nonnegative_int(
                row["exact_states"]["state_count"],
                field=f"candidate {candidate_id!r}.exact_states.state_count",
            )
            transition_count += _require_nonnegative_int(
                row["packed_address"]["path_length"],
                field=f"candidate {candidate_id!r}.packed_address.path_length",
            )
        try:
            next(physical_output_rows)
        except StopIteration:
            pass
        else:
            raise EditingV2Active8SourceAdapterError(
                f"physical output row count disagrees for {registry_shard.artifact_path}"
            )

        manifest_entries = _require_nonnegative_int(
            packed_manifest.get("entries"),
            field=f"packed manifest {manifest_artifact_path}.entries",
        )
        manifest_states = _require_nonnegative_int(
            packed_manifest.get("states"),
            field=f"packed manifest {manifest_artifact_path}.states",
        )
        if (
            len(memberships) != registry_shard.records
            or manifest_entries != registry_shard.records
            or state_count != registry_shard.states
            or manifest_states != registry_shard.states
            or transition_count != registry_shard.transitions
        ):
            raise EditingV2Active8SourceAdapterError(
                f"packed membership census disagrees for {registry_shard.artifact_path}"
            )
        source = Active8SourceShard(
            manifest_layer=registry_shard.lane_id,
            envelope_layer=registry_shard.lane_id,
            partition=registry_shard.partition,
            relative_path=registry_relative_path,
            path=registry_shard.local_path,
        )
        bindings.append(
            ResolvedActive8SourceBinding(
                source=source,
                artifact_path=registry_shard.artifact_path,
                packed_shard_file_sha256=registry_shard.sha256,
                packed_manifest_artifact_path=manifest_artifact_path,
                packed_manifest_file_sha256=manifest_sha256,
                packed_provenance_overlay_artifact_path=(overlay_artifact_path),
                packed_provenance_overlay_file_sha256=overlay_sha256,
                candidate_ids=tuple(candidate_ids),
                original_address_sha256s=tuple(address_sha256s),
            )
        )

    if seen_candidates != selected:
        raise EditingV2Active8SourceAdapterError(
            "resolved packed membership is incomplete; "
            f"missing selected candidates={sorted(selected - seen_candidates)}"
        )
    return ResolvedEditingV2Active8Sources(
        membership_receipt_path=receipt_path.resolve(),
        membership_receipt_file_sha256=file_sha256(receipt_path),
        membership_receipt_sha256=receipt_sha256,
        candidate_materialization_manifest_sha256=(candidate_identity["manifest_sha256"]),
        candidate_provenance_source_stream=dict(assignment["source_stream"]),
        split_assignment_sha256=split_identity["assignment_sha256"],
        lane_registry_sha256=registry_identity["registry_sha256"],
        bindings=tuple(bindings),
        source_manifest=dict(receipt),
    )


__all__ = [
    "ACTIVE8_ADMISSION_STATUS",
    "CANDIDATE_AUTHORITY",
    "REQUIRED_BLOCKERS",
    "RESOLVED_PACKED_MEMBERSHIP_SCHEMA",
    "RESOLVED_PACKED_MEMBERSHIP_SCHEMA_VERSION",
    "RESOLVED_PACKED_MEMBERSHIP_STATUS",
    "EditingV2Active8SourceAdapterError",
    "ResolvedActive8SourceBinding",
    "ResolvedEditingV2Active8Sources",
    "resolve_editing_v2_active8_sources",
]
