"""Fail-closed registry and resolver for editing-V2 evidence lanes.

This module deliberately does not replace the legacy three-layer production
corpus.  It defines the independent, versioned envelope needed to bind the
five evidence lanes declared by the authoritative editing-corpus contract.
The resolver is read-only: it validates physical completion manifests and
their content-addressed shard inventories but never creates scientific
artifacts or grants training authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from compose_v4.data.editing_corpus_contract import (
    EXPECTED_SCHEMA as EDITING_CORPUS_CONTRACT_SCHEMA,
)
from compose_v4.data.editing_corpus_contract import (
    EXPECTED_SCHEMA_VERSION as EDITING_CORPUS_CONTRACT_SCHEMA_VERSION,
)
from compose_v4.data.editing_corpus_contract import (
    load_editing_corpus_contract,
)

LANE_REGISTRY_SCHEMA = "compose.editing_v2_lane_registry"
LANE_REGISTRY_SCHEMA_VERSION = 2
LANE_REGISTRY_STATUS = "FROZEN_COMPLETE_NO_TRAINING_AUTHORITY"
LANE_COMPLETION_SCHEMA = "compose.editing_v2_lane_completion"
LANE_COMPLETION_SCHEMA_VERSION = 2
LANE_COMPLETION_STATUS = "COMPLETE_NO_TRAINING_AUTHORITY"
LANE_COMPLETION_FILENAME = "LANE_COMPLETE.json"
EXPECTED_LANE_COUNT = 5

_HEX64 = frozenset("0123456789abcdef")
_REGISTRY_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "editing_corpus_contract",
    "partition_roles",
    "lanes",
    "registry_sha256",
}
_CONTRACT_IDENTITY_FIELDS = {
    "schema",
    "schema_version",
    "contract_id",
    "file_sha256",
}
_REGISTRY_LANE_FIELDS = {
    "lane_id",
    "admissible_evidence_profiles",
    "reserved_evidence_profiles",
    "declared_root",
    "completion_manifest_path",
    "completion_manifest_sha256",
}
_COMPLETION_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "editing_corpus_contract",
    "lane_id",
    "admissible_evidence_profiles",
    "reserved_evidence_profiles",
    "declared_root",
    "partition_roles",
    "counts",
    "shards",
    "shard_inventory_sha256",
    "completion_sha256",
}
_COUNTS_FIELDS = {
    "records",
    "states",
    "transitions",
    "shards",
    "by_partition",
}
_PARTITION_COUNTS_FIELDS = {
    "records",
    "states",
    "transitions",
    "shards",
}
_SHARD_FIELDS = {
    "partition",
    "relative_path",
    "sha256",
    "bytes",
    "records",
    "states",
    "transitions",
}


class EditingV2LaneRegistryError(ValueError):
    """The editing-V2 lane registry or one of its physical inputs is invalid."""


@dataclass(frozen=True)
class EditingV2LaneDefinition:
    """One lane identity read from the authoritative editing-corpus contract."""

    lane_id: str
    admissible_evidence_profiles: tuple[str, ...]
    reserved_evidence_profiles: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedEditingV2Shard:
    """One physically verified shard in a content-addressed lane inventory."""

    lane_id: str
    partition: str
    artifact_path: str
    local_path: Path
    sha256: str
    bytes: int
    records: int
    states: int
    transitions: int


@dataclass(frozen=True)
class ResolvedEditingV2Lane:
    """One complete lane and all of its physically verified shards."""

    lane_id: str
    admissible_evidence_profiles: tuple[str, ...]
    reserved_evidence_profiles: tuple[str, ...]
    declared_root: str
    completion_manifest_path: str
    completion_manifest_file_sha256: str
    completion_sha256: str
    shard_inventory_sha256: str
    counts: Mapping[str, Any]
    shards: tuple[ResolvedEditingV2Shard, ...]


@dataclass(frozen=True)
class ResolvedEditingV2LaneRegistry:
    """The complete five-lane registry resolved against one contract and mount."""

    registry_path: Path
    registry_file_sha256: str
    registry_sha256: str
    editing_corpus_contract_path: Path
    editing_corpus_contract_file_sha256: str
    editing_corpus_contract_id: str
    partition_roles: tuple[str, ...]
    lanes: tuple[ResolvedEditingV2Lane, ...]


def canonical_json_bytes(payload: Any) -> bytes:
    """Return deterministic JSON bytes used for semantic identities."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def canonical_sha256(payload: Any) -> str:
    """Return the SHA-256 of deterministic JSON serialization."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Stream one file into SHA-256 without interpreting its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_fields(
    value: object,
    expected: set[str],
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EditingV2LaneRegistryError(f"{field} must be an object")
    actual = set(value)
    if actual != expected:
        raise EditingV2LaneRegistryError(
            f"{field} fields disagree; "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return value


def _require_sha256(value: object, *, field: str) -> str:
    digest = value if isinstance(value, str) else ""
    if len(digest) != 64 or any(character not in _HEX64 for character in digest):
        raise EditingV2LaneRegistryError(f"{field} must be a full lowercase SHA-256")
    return digest


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EditingV2LaneRegistryError(f"{field} must be a nonnegative integer")
    return value


def _require_nonempty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise EditingV2LaneRegistryError(
            f"{field} must be a nonempty, whitespace-normalized string"
        )
    return value


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return sorted(duplicate)


def _require_profile_list(
    value: object,
    *,
    field: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise EditingV2LaneRegistryError(f"{field} must be a list")
    profiles = tuple(
        _require_identifier(
            profile_id,
            field=f"{field}[{index}]",
        )
        for index, profile_id in enumerate(value)
    )
    if not allow_empty and not profiles:
        raise EditingV2LaneRegistryError(f"{field} must be nonempty")
    duplicate_profiles = _duplicates(profiles)
    if duplicate_profiles:
        raise EditingV2LaneRegistryError(f"{field} repeats evidence profiles: {duplicate_profiles}")
    return profiles


def _require_identifier(value: object, *, field: str) -> str:
    identifier = _require_nonempty_text(value, field=field)
    if not identifier[0].islower() or any(
        not (character.islower() or character.isdigit() or character == "_")
        for character in identifier
    ):
        raise EditingV2LaneRegistryError(
            f"{field} must contain only lowercase letters, digits, and underscores"
        )
    return identifier


def _require_artifact_path(
    value: object,
    *,
    field: str,
    directory: bool,
) -> str:
    raw = value if isinstance(value, str) else ""
    candidate = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or not candidate.is_absolute()
        or len(candidate.parts) < 3
        or candidate.parts[1] != "artifacts"
        or ".." in candidate.parts
        or str(candidate) != raw
    ):
        kind = "directory" if directory else "file"
        raise EditingV2LaneRegistryError(f"{field} must be a normalized {kind} below /artifacts")
    if directory and raw.endswith("/"):
        raise EditingV2LaneRegistryError(f"{field} must not contain a trailing slash")
    if not directory and raw.endswith("/"):
        raise EditingV2LaneRegistryError(f"{field} must name a file")
    return raw


def _require_relative_shard_path(value: object, *, field: str) -> str:
    raw = value if isinstance(value, str) else ""
    candidate = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or candidate.is_absolute()
        or candidate in (PurePosixPath("."), PurePosixPath(".."))
        or ".." in candidate.parts
        or str(candidate) != raw
    ):
        raise EditingV2LaneRegistryError(
            f"{field} must be a normalized relative path without traversal"
        )
    return raw


def _mounted_artifact_path(
    artifact_path: str,
    *,
    artifact_root: Path,
    field: str,
) -> Path:
    normalized = _require_artifact_path(
        artifact_path,
        field=field,
        directory=False,
    )
    root = artifact_root.resolve()
    mounted = root / PurePosixPath(normalized).relative_to("/artifacts")
    resolved = mounted.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise EditingV2LaneRegistryError(f"{field} resolves outside the artifact mount") from error
    return resolved


def _load_json_file(path: Path, *, field: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise EditingV2LaneRegistryError(f"{field} is absent: {path}")
    try:
        payload = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingV2LaneRegistryError(f"{field} is invalid JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise EditingV2LaneRegistryError(f"{field} must contain a JSON object")
    return payload


def editing_v2_lane_definitions(
    contract: Mapping[str, Any],
) -> tuple[EditingV2LaneDefinition, ...]:
    """Read ordered lane/profile identities from the authoritative contract."""

    raw_lanes = contract.get("data_lanes")
    if not isinstance(raw_lanes, list) or not raw_lanes:
        raise EditingV2LaneRegistryError(
            "editing-corpus contract data_lanes must be a nonempty list"
        )
    lanes: list[EditingV2LaneDefinition] = []
    for index, raw_lane in enumerate(raw_lanes):
        if not isinstance(raw_lane, Mapping):
            raise EditingV2LaneRegistryError(
                f"editing-corpus contract data_lanes[{index}] must be an object"
            )
        lane_id = _require_identifier(
            raw_lane.get("id"),
            field=f"editing-corpus contract data_lanes[{index}].id",
        )
        admissible = _require_profile_list(
            raw_lane.get("admissible_evidence_profiles"),
            field=(f"editing-corpus contract data_lanes[{index}].admissible_evidence_profiles"),
            allow_empty=False,
        )
        reserved = _require_profile_list(
            raw_lane.get("reserved_evidence_profiles"),
            field=(f"editing-corpus contract data_lanes[{index}].reserved_evidence_profiles"),
            allow_empty=True,
        )
        duplicate_profiles = _duplicates((*admissible, *reserved))
        if duplicate_profiles:
            raise EditingV2LaneRegistryError(
                f"editing-corpus contract data_lanes[{index}] repeats evidence profiles: "
                f"{duplicate_profiles}"
            )
        lanes.append(
            EditingV2LaneDefinition(
                lane_id=lane_id,
                admissible_evidence_profiles=admissible,
                reserved_evidence_profiles=reserved,
            )
        )
    duplicate_lanes = _duplicates([lane.lane_id for lane in lanes])
    if duplicate_lanes:
        raise EditingV2LaneRegistryError(
            f"editing-corpus contract has duplicate lanes: {duplicate_lanes}"
        )
    if len(lanes) != EXPECTED_LANE_COUNT:
        raise EditingV2LaneRegistryError(
            "editing-corpus contract schema version 2 must declare exactly "
            f"{EXPECTED_LANE_COUNT} evidence lanes"
        )
    return tuple(lanes)


def editing_v2_partition_roles(contract: Mapping[str, Any]) -> tuple[str, ...]:
    """Read ordered split roles without introducing repository-local defaults."""

    split_contract = contract.get("split_contract")
    if not isinstance(split_contract, Mapping):
        raise EditingV2LaneRegistryError("editing-corpus contract split_contract must be an object")
    raw_roles = split_contract.get("partition_roles")
    if not isinstance(raw_roles, list) or not raw_roles:
        raise EditingV2LaneRegistryError(
            "editing-corpus contract split_contract.partition_roles must be a nonempty list"
        )
    roles = tuple(
        _require_identifier(
            role,
            field=f"editing-corpus contract partition_roles[{index}]",
        )
        for index, role in enumerate(raw_roles)
    )
    duplicate_roles = _duplicates(roles)
    if duplicate_roles:
        raise EditingV2LaneRegistryError(
            f"editing-corpus contract has duplicate partition roles: {duplicate_roles}"
        )
    return roles


def editing_corpus_contract_identity(
    contract: Mapping[str, Any],
    *,
    contract_file_sha256: str,
) -> dict[str, Any]:
    """Build the exact physical/semantic contract reference used by every lane."""

    schema = contract.get("schema")
    schema_version = contract.get("schema_version")
    contract_id = _require_nonempty_text(
        contract.get("contract_id"),
        field="editing-corpus contract_id",
    )
    if (
        schema != EDITING_CORPUS_CONTRACT_SCHEMA
        or schema_version != EDITING_CORPUS_CONTRACT_SCHEMA_VERSION
    ):
        raise EditingV2LaneRegistryError("editing-corpus contract schema identity is unsupported")
    return {
        "schema": schema,
        "schema_version": schema_version,
        "contract_id": contract_id,
        "file_sha256": _require_sha256(
            contract_file_sha256,
            field="editing-corpus contract file SHA-256",
        ),
    }


def _validate_contract_identity(
    value: object,
    *,
    expected: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    payload = _require_exact_fields(value, _CONTRACT_IDENTITY_FIELDS, field=field)
    _require_sha256(payload["file_sha256"], field=f"{field}.file_sha256")
    if dict(payload) != dict(expected):
        raise EditingV2LaneRegistryError(
            f"{field} disagrees with the authoritative editing-corpus contract"
        )
    return dict(payload)


def _semantic_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    field: str,
) -> str:
    supplied = _require_sha256(payload.get(hash_field), field=f"{field}.{hash_field}")
    body = {key: value for key, value in payload.items() if key != hash_field}
    expected = canonical_sha256(body)
    if supplied != expected:
        raise EditingV2LaneRegistryError(
            f"{field} {hash_field} disagrees with its canonical payload"
        )
    return supplied


def _validate_partition_counts(
    counts: object,
    *,
    partition_roles: tuple[str, ...],
    field: str,
) -> dict[str, Any]:
    payload = _require_exact_fields(counts, _COUNTS_FIELDS, field=field)
    totals = {
        name: _require_nonnegative_int(payload[name], field=f"{field}.{name}")
        for name in ("records", "states", "transitions", "shards")
    }
    by_partition = payload["by_partition"]
    if not isinstance(by_partition, Mapping):
        raise EditingV2LaneRegistryError(f"{field}.by_partition must be an object")
    if set(by_partition) != set(partition_roles):
        raise EditingV2LaneRegistryError(
            f"{field}.by_partition roles disagree; "
            f"missing={sorted(set(partition_roles) - set(by_partition))}, "
            f"unexpected={sorted(set(by_partition) - set(partition_roles))}"
        )
    normalized_partitions: dict[str, dict[str, int]] = {}
    for partition in partition_roles:
        partition_payload = _require_exact_fields(
            by_partition[partition],
            _PARTITION_COUNTS_FIELDS,
            field=f"{field}.by_partition[{partition!r}]",
        )
        normalized_partitions[partition] = {
            name: _require_nonnegative_int(
                partition_payload[name],
                field=f"{field}.by_partition[{partition!r}].{name}",
            )
            for name in ("records", "states", "transitions", "shards")
        }
    for name, total in totals.items():
        observed = sum(values[name] for values in normalized_partitions.values())
        if observed != total:
            raise EditingV2LaneRegistryError(
                f"{field}.{name}={total} disagrees with partition sum {observed}"
            )
    if totals["records"] <= 0 or totals["shards"] <= 0:
        raise EditingV2LaneRegistryError(f"{field} must declare at least one record and one shard")
    return {
        **totals,
        "by_partition": normalized_partitions,
    }


def _validate_shards(
    shards: object,
    *,
    lane_id: str,
    declared_root: str,
    partition_roles: tuple[str, ...],
    counts: Mapping[str, Any],
    artifact_root: Path,
) -> tuple[tuple[ResolvedEditingV2Shard, ...], list[dict[str, Any]]]:
    if not isinstance(shards, list) or not shards:
        raise EditingV2LaneRegistryError(f"lane {lane_id!r} shards must be a nonempty list")
    role_order = {role: index for index, role in enumerate(partition_roles)}
    normalized: list[dict[str, Any]] = []
    resolved: list[ResolvedEditingV2Shard] = []
    seen_relative_paths: set[str] = set()
    for index, raw_shard in enumerate(shards):
        field = f"lane {lane_id!r} shards[{index}]"
        shard = _require_exact_fields(raw_shard, _SHARD_FIELDS, field=field)
        partition = _require_identifier(
            shard["partition"],
            field=f"{field}.partition",
        )
        if partition not in role_order:
            raise EditingV2LaneRegistryError(
                f"{field}.partition is not allowed by the editing-corpus contract: {partition!r}"
            )
        relative_path = _require_relative_shard_path(
            shard["relative_path"],
            field=f"{field}.relative_path",
        )
        if relative_path in seen_relative_paths:
            raise EditingV2LaneRegistryError(
                f"lane {lane_id!r} has duplicate shard path: {relative_path!r}"
            )
        seen_relative_paths.add(relative_path)
        if PurePosixPath(relative_path).parts[0] != partition:
            raise EditingV2LaneRegistryError(
                f"{field}.relative_path must be rooted in its partition directory"
            )
        digest = _require_sha256(shard["sha256"], field=f"{field}.sha256")
        byte_count = _require_nonnegative_int(
            shard["bytes"],
            field=f"{field}.bytes",
        )
        record_count = _require_nonnegative_int(
            shard["records"],
            field=f"{field}.records",
        )
        state_count = _require_nonnegative_int(
            shard["states"],
            field=f"{field}.states",
        )
        transition_count = _require_nonnegative_int(
            shard["transitions"],
            field=f"{field}.transitions",
        )
        if byte_count <= 0 or record_count <= 0:
            raise EditingV2LaneRegistryError(f"{field} must bind a nonempty physical shard")
        artifact_path = str(PurePosixPath(declared_root) / relative_path)
        local_path = _mounted_artifact_path(
            artifact_path,
            artifact_root=artifact_root,
            field=f"{field}.artifact_path",
        )
        if not local_path.is_file():
            raise EditingV2LaneRegistryError(f"{field} is absent: {local_path}")
        if local_path.stat().st_size != byte_count:
            raise EditingV2LaneRegistryError(f"{field}.bytes disagrees with the physical shard")
        physical_sha256 = file_sha256(local_path)
        if physical_sha256 != digest:
            raise EditingV2LaneRegistryError(f"{field}.sha256 disagrees with the physical shard")
        normalized_shard = {
            "partition": partition,
            "relative_path": relative_path,
            "sha256": digest,
            "bytes": byte_count,
            "records": record_count,
            "states": state_count,
            "transitions": transition_count,
        }
        normalized.append(normalized_shard)
        resolved.append(
            ResolvedEditingV2Shard(
                lane_id=lane_id,
                partition=partition,
                artifact_path=artifact_path,
                local_path=local_path,
                sha256=digest,
                bytes=byte_count,
                records=record_count,
                states=state_count,
                transitions=transition_count,
            )
        )

    expected_order = sorted(
        normalized,
        key=lambda item: (
            role_order[item["partition"]],
            item["relative_path"],
        ),
    )
    if normalized != expected_order:
        raise EditingV2LaneRegistryError(
            f"lane {lane_id!r} shard inventory is not deterministically ordered"
        )
    if len(normalized) != counts["shards"]:
        raise EditingV2LaneRegistryError(f"lane {lane_id!r} shard count disagrees with its counts")
    for metric in ("records", "states", "transitions", "shards"):
        if metric == "shards":
            observed_total = len(normalized)
        else:
            observed_total = sum(int(shard[metric]) for shard in normalized)
        if observed_total != counts[metric]:
            raise EditingV2LaneRegistryError(
                f"lane {lane_id!r} {metric} disagrees with its shard inventory"
            )
        for partition in partition_roles:
            partition_shards = [shard for shard in normalized if shard["partition"] == partition]
            if metric == "shards":
                observed_partition = len(partition_shards)
            else:
                observed_partition = sum(int(shard[metric]) for shard in partition_shards)
            expected_partition = counts["by_partition"][partition][metric]
            if observed_partition != expected_partition:
                raise EditingV2LaneRegistryError(
                    f"lane {lane_id!r} partition {partition!r} {metric} "
                    "disagrees with its shard inventory"
                )
    return tuple(resolved), normalized


def _resolve_lane(
    lane_entry: Mapping[str, Any],
    *,
    expected_lane: EditingV2LaneDefinition,
    expected_contract_identity: Mapping[str, Any],
    partition_roles: tuple[str, ...],
    artifact_root: Path,
) -> ResolvedEditingV2Lane:
    lane_id = expected_lane.lane_id
    declared_root = _require_artifact_path(
        lane_entry["declared_root"],
        field=f"registry lane {lane_id!r}.declared_root",
        directory=True,
    )
    completion_artifact_path = _require_artifact_path(
        lane_entry["completion_manifest_path"],
        field=f"registry lane {lane_id!r}.completion_manifest_path",
        directory=False,
    )
    expected_completion_path = str(PurePosixPath(declared_root) / LANE_COMPLETION_FILENAME)
    if completion_artifact_path != expected_completion_path:
        raise EditingV2LaneRegistryError(
            f"registry lane {lane_id!r} completion manifest is outside its "
            "declared root or has the wrong name"
        )
    completion_path = _mounted_artifact_path(
        completion_artifact_path,
        artifact_root=artifact_root,
        field=f"registry lane {lane_id!r}.completion_manifest_path",
    )
    completion_file_sha256 = _require_sha256(
        lane_entry["completion_manifest_sha256"],
        field=f"registry lane {lane_id!r}.completion_manifest_sha256",
    )
    if not completion_path.is_file():
        raise EditingV2LaneRegistryError(
            f"registry lane {lane_id!r} completion manifest is absent: {completion_path}"
        )
    if file_sha256(completion_path) != completion_file_sha256:
        raise EditingV2LaneRegistryError(
            f"registry lane {lane_id!r} completion file SHA-256 disagrees"
        )
    completion = _require_exact_fields(
        _load_json_file(
            completion_path,
            field=f"lane {lane_id!r} completion manifest",
        ),
        _COMPLETION_FIELDS,
        field=f"lane {lane_id!r} completion manifest",
    )
    if (
        completion["schema"] != LANE_COMPLETION_SCHEMA
        or completion["schema_version"] != LANE_COMPLETION_SCHEMA_VERSION
        or completion["status"] != LANE_COMPLETION_STATUS
        or completion["training_authorized"] is not False
    ):
        raise EditingV2LaneRegistryError(
            f"lane {lane_id!r} completion schema, status, or authority is invalid"
        )
    _validate_contract_identity(
        completion["editing_corpus_contract"],
        expected=expected_contract_identity,
        field=f"lane {lane_id!r} completion editing_corpus_contract",
    )
    completion_admissible_profiles = _require_profile_list(
        completion["admissible_evidence_profiles"],
        field=f"lane {lane_id!r} completion admissible_evidence_profiles",
        allow_empty=False,
    )
    completion_reserved_profiles = _require_profile_list(
        completion["reserved_evidence_profiles"],
        field=f"lane {lane_id!r} completion reserved_evidence_profiles",
        allow_empty=True,
    )
    if (
        completion["lane_id"] != lane_id
        or completion_admissible_profiles != expected_lane.admissible_evidence_profiles
        or completion_reserved_profiles != expected_lane.reserved_evidence_profiles
        or completion["declared_root"] != declared_root
    ):
        raise EditingV2LaneRegistryError(
            f"lane {lane_id!r} completion identity disagrees with its registry "
            "entry or the editing-corpus contract"
        )
    if tuple(completion["partition_roles"]) != partition_roles:
        raise EditingV2LaneRegistryError(
            f"lane {lane_id!r} completion partition roles disagree with the editing-corpus contract"
        )
    counts = _validate_partition_counts(
        completion["counts"],
        partition_roles=partition_roles,
        field=f"lane {lane_id!r} counts",
    )
    resolved_shards, normalized_shards = _validate_shards(
        completion["shards"],
        lane_id=lane_id,
        declared_root=declared_root,
        partition_roles=partition_roles,
        counts=counts,
        artifact_root=artifact_root,
    )
    shard_inventory_sha256 = _require_sha256(
        completion["shard_inventory_sha256"],
        field=f"lane {lane_id!r}.shard_inventory_sha256",
    )
    expected_inventory_sha256 = canonical_sha256(normalized_shards)
    if shard_inventory_sha256 != expected_inventory_sha256:
        raise EditingV2LaneRegistryError(f"lane {lane_id!r} shard inventory SHA-256 disagrees")
    if PurePosixPath(declared_root).name != shard_inventory_sha256:
        raise EditingV2LaneRegistryError(
            f"lane {lane_id!r} declared root is not content-addressed by its shard inventory"
        )
    completion_sha256 = _semantic_hash(
        completion,
        hash_field="completion_sha256",
        field=f"lane {lane_id!r} completion manifest",
    )
    return ResolvedEditingV2Lane(
        lane_id=lane_id,
        admissible_evidence_profiles=expected_lane.admissible_evidence_profiles,
        reserved_evidence_profiles=expected_lane.reserved_evidence_profiles,
        declared_root=declared_root,
        completion_manifest_path=completion_artifact_path,
        completion_manifest_file_sha256=completion_file_sha256,
        completion_sha256=completion_sha256,
        shard_inventory_sha256=shard_inventory_sha256,
        counts=counts,
        shards=resolved_shards,
    )


def resolve_editing_v2_lane_registry(
    *,
    editing_corpus_contract_path: str | Path,
    registry_path: str | Path,
    artifact_root: str | Path,
) -> ResolvedEditingV2LaneRegistry:
    """Resolve every declared lane and reject any partial or mismatched corpus."""

    contract_path = Path(editing_corpus_contract_path)
    if not contract_path.is_file():
        raise EditingV2LaneRegistryError(f"editing-corpus contract is absent: {contract_path}")
    contract = load_editing_corpus_contract(contract_path)
    contract_file_sha256 = file_sha256(contract_path)
    contract_identity = editing_corpus_contract_identity(
        contract,
        contract_file_sha256=contract_file_sha256,
    )
    lane_definitions = editing_v2_lane_definitions(contract)
    partition_roles = editing_v2_partition_roles(contract)

    mount_root = Path(artifact_root)
    if not mount_root.is_dir():
        raise EditingV2LaneRegistryError(
            f"artifact root is absent or not a directory: {mount_root}"
        )
    registry_source = Path(registry_path)
    registry = _require_exact_fields(
        _load_json_file(registry_source, field="editing-V2 lane registry"),
        _REGISTRY_FIELDS,
        field="editing-V2 lane registry",
    )
    if (
        registry["schema"] != LANE_REGISTRY_SCHEMA
        or registry["schema_version"] != LANE_REGISTRY_SCHEMA_VERSION
        or registry["status"] != LANE_REGISTRY_STATUS
        or registry["training_authorized"] is not False
    ):
        raise EditingV2LaneRegistryError(
            "editing-V2 lane registry schema, status, or authority is invalid"
        )
    _validate_contract_identity(
        registry["editing_corpus_contract"],
        expected=contract_identity,
        field="editing-V2 lane registry editing_corpus_contract",
    )
    if tuple(registry["partition_roles"]) != partition_roles:
        raise EditingV2LaneRegistryError(
            "editing-V2 lane registry partition roles disagree with the editing-corpus contract"
        )
    raw_lane_entries = registry["lanes"]
    if not isinstance(raw_lane_entries, list):
        raise EditingV2LaneRegistryError("editing-V2 lane registry lanes must be a list")
    entries_by_id: dict[str, Mapping[str, Any]] = {}
    ordered_entry_ids: list[str] = []
    for index, raw_entry in enumerate(raw_lane_entries):
        entry = _require_exact_fields(
            raw_entry,
            _REGISTRY_LANE_FIELDS,
            field=f"editing-V2 lane registry lanes[{index}]",
        )
        lane_id = _require_identifier(
            entry["lane_id"],
            field=f"editing-V2 lane registry lanes[{index}].lane_id",
        )
        ordered_entry_ids.append(lane_id)
        if lane_id in entries_by_id:
            raise EditingV2LaneRegistryError(
                f"editing-V2 lane registry has duplicate lane: {lane_id!r}"
            )
        entries_by_id[lane_id] = entry
    expected_lane_ids = [lane.lane_id for lane in lane_definitions]
    if set(entries_by_id) != set(expected_lane_ids):
        raise EditingV2LaneRegistryError(
            "editing-V2 lane registry lanes disagree with the contract; "
            f"missing={sorted(set(expected_lane_ids) - set(entries_by_id))}, "
            f"unexpected={sorted(set(entries_by_id) - set(expected_lane_ids))}"
        )
    if ordered_entry_ids != expected_lane_ids:
        raise EditingV2LaneRegistryError(
            "editing-V2 lane registry lanes are not in authoritative contract order"
        )
    resolved_lanes: list[ResolvedEditingV2Lane] = []
    seen_roots: set[str] = set()
    seen_completion_paths: set[str] = set()
    seen_shard_paths: set[str] = set()
    for lane_definition in lane_definitions:
        entry = entries_by_id[lane_definition.lane_id]
        registry_admissible_profiles = _require_profile_list(
            entry["admissible_evidence_profiles"],
            field=(f"registry lane {lane_definition.lane_id!r} admissible_evidence_profiles"),
            allow_empty=False,
        )
        registry_reserved_profiles = _require_profile_list(
            entry["reserved_evidence_profiles"],
            field=(f"registry lane {lane_definition.lane_id!r} reserved_evidence_profiles"),
            allow_empty=True,
        )
        if (
            registry_admissible_profiles != lane_definition.admissible_evidence_profiles
            or registry_reserved_profiles != lane_definition.reserved_evidence_profiles
        ):
            raise EditingV2LaneRegistryError(
                f"registry lane {lane_definition.lane_id!r} evidence-profile set "
                "disagrees with the editing-corpus contract"
            )
        resolved_lane = _resolve_lane(
            entry,
            expected_lane=lane_definition,
            expected_contract_identity=contract_identity,
            partition_roles=partition_roles,
            artifact_root=mount_root,
        )
        if resolved_lane.declared_root in seen_roots:
            raise EditingV2LaneRegistryError("editing-V2 lanes must not share a declared root")
        seen_roots.add(resolved_lane.declared_root)
        if resolved_lane.completion_manifest_path in seen_completion_paths:
            raise EditingV2LaneRegistryError(
                "editing-V2 lanes must not share a completion manifest"
            )
        seen_completion_paths.add(resolved_lane.completion_manifest_path)
        for shard in resolved_lane.shards:
            if shard.artifact_path in seen_shard_paths:
                raise EditingV2LaneRegistryError("editing-V2 lanes must not share a physical shard")
            seen_shard_paths.add(shard.artifact_path)
        resolved_lanes.append(resolved_lane)

    registry_sha256 = _semantic_hash(
        registry,
        hash_field="registry_sha256",
        field="editing-V2 lane registry",
    )
    return ResolvedEditingV2LaneRegistry(
        registry_path=registry_source.resolve(),
        registry_file_sha256=file_sha256(registry_source),
        registry_sha256=registry_sha256,
        editing_corpus_contract_path=contract_path.resolve(),
        editing_corpus_contract_file_sha256=contract_file_sha256,
        editing_corpus_contract_id=str(contract["contract_id"]),
        partition_roles=partition_roles,
        lanes=tuple(resolved_lanes),
    )


__all__ = [
    "EditingV2LaneDefinition",
    "EditingV2LaneRegistryError",
    "EXPECTED_LANE_COUNT",
    "LANE_COMPLETION_FILENAME",
    "LANE_COMPLETION_SCHEMA",
    "LANE_COMPLETION_SCHEMA_VERSION",
    "LANE_COMPLETION_STATUS",
    "LANE_REGISTRY_SCHEMA",
    "LANE_REGISTRY_SCHEMA_VERSION",
    "LANE_REGISTRY_STATUS",
    "ResolvedEditingV2Lane",
    "ResolvedEditingV2LaneRegistry",
    "ResolvedEditingV2Shard",
    "canonical_sha256",
    "editing_corpus_contract_identity",
    "editing_v2_lane_definitions",
    "editing_v2_partition_roles",
    "file_sha256",
    "resolve_editing_v2_lane_registry",
]
