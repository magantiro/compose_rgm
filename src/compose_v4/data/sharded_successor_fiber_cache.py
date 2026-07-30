"""Exact lazy lookup over provenance-bound successor-fiber cache shards.

The canonical-JSON backend remains the bounded correctness oracle. The indexed
SQLite backend resolves one row at a time without materializing a shard-sized
Python object graph. Both remain development-only until a complete corpus
inventory and the preregistered training gates explicitly authorize a run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import tempfile
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.chem.persistent_state_identity import (
    PERSISTENT_STATE_DIGEST_SCHEMA,
    PERSISTENT_STATE_DIGEST_VERSION,
    persistent_slot_state_sha256,
)
from compose_v4.data.indexed_successor_fiber_cache import (
    INDEXED_SUCCESSOR_FIBER_STORAGE_BACKEND,
    IndexedSuccessorFiberCache,
    IndexedSuccessorFiberCacheError,
    open_indexed_successor_fiber_cache,
)
from compose_v4.data.packed_trace_store import PackedTraceAddress
from compose_v4.data.successor_fiber_cache import (
    DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
    SUCCESSOR_FIBER_CACHE_SCHEMA,
    SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION,
    SuccessorFiberCache,
    SuccessorFiberCacheError,
    SuccessorFiberCacheLimits,
    SuccessorFiberCacheProvenance,
    SuccessorFiberCacheRecord,
    read_successor_fiber_cache,
)

SUCCESSOR_FIBER_INVENTORY_SCHEMA = (
    "compose.data.sharded_successor_fiber_cache_inventory"
)
SUCCESSOR_FIBER_INVENTORY_SCHEMA_VERSION = 1
SUCCESSOR_FIBER_INVENTORY_STATUS = "BOUNDED_DEVELOPMENT_ONLY"
SUCCESSOR_FIBER_STORAGE_BACKEND = "canonical_json_lru_development_v1"
SUCCESSOR_FIBER_INDEXED_STORAGE_BACKEND = (
    INDEXED_SUCCESSOR_FIBER_STORAGE_BACKEND
)
SUPPORTED_SUCCESSOR_FIBER_STORAGE_BACKENDS = frozenset(
    (
        SUCCESSOR_FIBER_STORAGE_BACKEND,
        SUCCESSOR_FIBER_INDEXED_STORAGE_BACKEND,
    )
)
MAX_INVENTORY_BYTES = 16 << 20

_ROOT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "storage_backend",
    "full_corpus_training_authorized",
    "source_corpus_inventory_sha256",
    "compatibility_payload",
    "compatibility_sha256",
    "unified_packed_manifest_sha256",
    "representability_overlay_sha256",
    "entry_count",
    "entries",
    "inventory_sha256",
}
_ENTRY_FIELDS = {
    "packed_shard_content_sha256",
    "packed_shard_name",
    "layer",
    "partition",
    "packed_manifest_sha256",
    "packed_provenance_overlay_sha256",
    "cache_relative_path",
    "cache_content_sha256",
    "cache_file_sha256",
    "cache_file_bytes",
    "packed_entry_count",
    "active_trace_count",
    "excluded_entry_indices",
    "progress_record_count",
    "jump_record_count",
    "terminal_record_count",
    "teacher_alias_count",
    "virtual_alias_count",
    "maximum_alias_count",
    "provenance",
}
_COMPATIBILITY_FIELDS = {
    "successor_cache_schema",
    "successor_cache_schema_version",
    "support_signature",
    "ordered_family_vocabulary",
    "ordered_action_table_vocabulary",
    "coordinate_schema_version",
    "operator_registry_hash",
    "capability_hash",
    "canonicalization_version",
    "canonicalizer_contract_sha256",
    "rdkit_version",
    "executor_implementation_hash",
    "action_enumerator_implementation_hash",
    "fiber_compiler_implementation_hash",
    "persistent_state_digest_schema",
    "persistent_state_digest_schema_version",
    "packed_corpus_schema",
    "packed_corpus_schema_version",
    "trace_codec_schema",
    "trace_codec_schema_version",
    "tensorization_implementation_hash",
}


class ShardedSuccessorFiberCacheError(SuccessorFiberCacheError):
    """The inventory, shard binding, or exact lookup is unusable."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ShardedSuccessorFiberCacheError(
            "successor-fiber inventory is not finite canonical JSON"
        ) from error


def _inventory_self_hash(payload: Mapping[str, Any]) -> str:
    unhashed = dict(payload)
    unhashed.pop("inventory_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(unhashed)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_mapping(
    value: object,
    *,
    fields: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShardedSuccessorFiberCacheError(f"{name} must be an object")
    observed = set(value)
    if observed != fields:
        raise ShardedSuccessorFiberCacheError(
            f"{name} schema mismatch: missing={sorted(fields - observed)}, "
            f"unexpected={sorted(observed - fields)}"
        )
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ShardedSuccessorFiberCacheError(
            f"{field} must be a nonnegative integer"
        )
    return value


def _nonempty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ShardedSuccessorFiberCacheError(f"{field} must be nonempty text")
    return value


def _safe_relative_path(value: object) -> str:
    text = _nonempty_text(value, field="cache_relative_path")
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or pure == PurePosixPath(".")
        or "\\" in text
        or any(character in text for character in "*?[]")
    ):
        raise ShardedSuccessorFiberCacheError(
            "cache_relative_path must be a safe POSIX-relative path"
        )
    return pure.as_posix()


def _validate_compatibility_payload(value: object) -> dict[str, Any]:
    payload = dict(
        _require_exact_mapping(
            value,
            fields=_COMPATIBILITY_FIELDS,
            name="successor-cache compatibility payload",
        )
    )
    if payload["successor_cache_schema"] != SUCCESSOR_FIBER_CACHE_SCHEMA:
        raise ShardedSuccessorFiberCacheError(
            "compatibility payload names another successor-cache schema"
        )
    if (
        payload["successor_cache_schema_version"]
        != SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION
    ):
        raise ShardedSuccessorFiberCacheError(
            "compatibility payload names another successor-cache version"
        )
    if payload["persistent_state_digest_schema"] != PERSISTENT_STATE_DIGEST_SCHEMA:
        raise ShardedSuccessorFiberCacheError(
            "compatibility payload names another persistent-state schema"
        )
    if (
        payload["persistent_state_digest_schema_version"]
        != PERSISTENT_STATE_DIGEST_VERSION
    ):
        raise ShardedSuccessorFiberCacheError(
            "compatibility payload names another persistent-state version"
        )
    if not isinstance(payload["support_signature"], Mapping) or not payload[
        "support_signature"
    ]:
        raise ShardedSuccessorFiberCacheError(
            "compatibility payload requires the full support-signature object"
        )
    for field in (
        "ordered_family_vocabulary",
        "ordered_action_table_vocabulary",
    ):
        values = payload[field]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(item, str) or not item for item in values)
            or len(values) != len(set(values))
        ):
            raise ShardedSuccessorFiberCacheError(
                f"{field} must be a nonempty unique ordered string array"
            )
    for field in (
        "successor_cache_schema_version",
        "coordinate_schema_version",
        "persistent_state_digest_schema_version",
        "packed_corpus_schema_version",
        "trace_codec_schema_version",
    ):
        value = payload[field]
        if type(value) is not int or value < 0:
            raise ShardedSuccessorFiberCacheError(
                f"{field} must be a nonnegative integer"
            )
    for field in _COMPATIBILITY_FIELDS - {
        "support_signature",
        "ordered_family_vocabulary",
        "ordered_action_table_vocabulary",
        "successor_cache_schema_version",
        "coordinate_schema_version",
        "persistent_state_digest_schema_version",
        "packed_corpus_schema_version",
        "trace_codec_schema_version",
    }:
        if not isinstance(payload[field], str) or not payload[field]:
            raise ShardedSuccessorFiberCacheError(
                f"{field} must be a nonempty string"
            )
    # Round-trip now so nested tuples or nonfinite values cannot make two
    # launchers disagree about the compatibility identity.
    return json.loads(_canonical_json_bytes(payload))


@dataclass(frozen=True)
class SuccessorFiberCacheCompatibility:
    """Canonical support-only identity supplied independently at launch."""

    canonical_payload: bytes
    compatibility_sha256: str

    def __post_init__(self) -> None:
        try:
            decoded = json.loads(self.canonical_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("compatibility payload is not canonical JSON") from error
        try:
            payload = _validate_compatibility_payload(decoded)
        except ShardedSuccessorFiberCacheError as error:
            raise ValueError(str(error)) from error
        canonical = _canonical_json_bytes(payload)
        if canonical != self.canonical_payload:
            raise ValueError("compatibility payload bytes are not canonical")
        observed = hashlib.sha256(canonical).hexdigest()
        if observed != self.compatibility_sha256:
            raise ValueError("compatibility payload SHA-256 does not match")

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> SuccessorFiberCacheCompatibility:
        canonical = _canonical_json_bytes(
            _validate_compatibility_payload(payload)
        )
        return cls(
            canonical_payload=canonical,
            compatibility_sha256=hashlib.sha256(canonical).hexdigest(),
        )

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_payload)


@dataclass(frozen=True, order=True)
class SuccessorFiberShardSpec:
    """Frozen identity and census of one packed-shard derivative."""

    packed_shard_content_sha256: str
    packed_shard_name: str
    layer: str
    partition: str
    packed_manifest_sha256: str
    packed_provenance_overlay_sha256: str | None
    cache_relative_path: str
    cache_content_sha256: str
    cache_file_sha256: str
    cache_file_bytes: int
    packed_entry_count: int
    active_trace_count: int
    excluded_entry_indices: tuple[int, ...]
    progress_record_count: int
    jump_record_count: int
    terminal_record_count: int
    teacher_alias_count: int
    virtual_alias_count: int
    maximum_alias_count: int
    provenance: SuccessorFiberCacheProvenance

    def __post_init__(self) -> None:
        if not _is_sha256(self.packed_shard_content_sha256):
            raise ValueError(
                "packed_shard_content_sha256 must be a lowercase SHA-256"
            )
        if (
            not isinstance(self.packed_shard_name, str)
            or not self.packed_shard_name
            or Path(self.packed_shard_name).name != self.packed_shard_name
        ):
            raise ValueError("packed_shard_name must be one nonempty basename")
        for field in ("layer", "partition"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be nonempty")
        try:
            normalized_path = _safe_relative_path(self.cache_relative_path)
        except ShardedSuccessorFiberCacheError as error:
            raise ValueError(str(error)) from error
        if normalized_path != self.cache_relative_path:
            raise ValueError("cache_relative_path is not normalized")
        for field in (
            "packed_manifest_sha256",
            "cache_content_sha256",
            "cache_file_sha256",
        ):
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"{field} must be a lowercase SHA-256")
        if self.packed_provenance_overlay_sha256 is not None and not _is_sha256(
            self.packed_provenance_overlay_sha256
        ):
            raise ValueError(
                "packed_provenance_overlay_sha256 must be null or a SHA-256"
            )
        for field in (
            "cache_file_bytes",
            "packed_entry_count",
            "active_trace_count",
            "progress_record_count",
            "jump_record_count",
            "terminal_record_count",
            "teacher_alias_count",
            "virtual_alias_count",
            "maximum_alias_count",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.cache_file_bytes <= 0:
            raise ValueError("cache_file_bytes must be positive")
        if self.packed_entry_count <= 0:
            raise ValueError("packed_entry_count must be positive")
        if self.active_trace_count <= 0:
            raise ValueError("active_trace_count must be positive")
        if (
            not isinstance(self.excluded_entry_indices, tuple)
            or any(
                type(index) is not int
                or not 0 <= index < self.packed_entry_count
                for index in self.excluded_entry_indices
            )
            or self.excluded_entry_indices
            != tuple(sorted(set(self.excluded_entry_indices)))
        ):
            raise ValueError(
                "excluded_entry_indices must be sorted unique in-range integers"
            )
        if (
            self.active_trace_count + len(self.excluded_entry_indices)
            != self.packed_entry_count
        ):
            raise ValueError(
                "active plus excluded trace count must equal packed entry count"
            )
        if self.terminal_record_count != self.active_trace_count:
            raise ValueError("each complete cached trace must have one terminal row")
        if self.progress_record_count < self.active_trace_count:
            raise ValueError("progress record count is smaller than trace count")
        if (
            self.jump_record_count + self.terminal_record_count
            != self.progress_record_count
        ):
            raise ValueError("jump plus terminal rows must equal progress rows")
        if self.teacher_alias_count < self.jump_record_count:
            # Every jump has at least one teacher-successor alias.
            raise ValueError("teacher alias count is smaller than jump count")
        total_alias_count = self.teacher_alias_count + self.virtual_alias_count
        if self.maximum_alias_count > total_alias_count:
            raise ValueError("maximum alias count exceeds the shard total")
        if (
            self.provenance.packed_shard_content_sha256
            != self.packed_shard_content_sha256
        ):
            raise ValueError("spec and cache provenance use different packed shards")
        if self.provenance.packed_manifest_sha256 != self.packed_manifest_sha256:
            raise ValueError("spec and cache provenance use different manifests")
        if (
            self.provenance.packed_provenance_overlay_sha256
            != self.packed_provenance_overlay_sha256
        ):
            raise ValueError(
                "spec and cache provenance use different provenance overlays"
            )


@dataclass(frozen=True)
class SuccessorFiberCacheInventory:
    """Validated immutable inventory for exact shard selection."""

    source_corpus_inventory_sha256: str
    compatibility: SuccessorFiberCacheCompatibility
    unified_packed_manifest_sha256: str
    representability_overlay_sha256: str
    entries: tuple[SuccessorFiberShardSpec, ...]
    inventory_sha256: str
    status: str = SUCCESSOR_FIBER_INVENTORY_STATUS
    storage_backend: str = SUCCESSOR_FIBER_STORAGE_BACKEND
    full_corpus_training_authorized: bool = False

    def __post_init__(self) -> None:
        if not _is_sha256(self.source_corpus_inventory_sha256):
            raise ValueError(
                "source_corpus_inventory_sha256 must be a lowercase SHA-256"
            )
        if not isinstance(
            self.compatibility,
            SuccessorFiberCacheCompatibility,
        ):
            raise TypeError("compatibility must be a frozen compatibility identity")
        for field in (
            "unified_packed_manifest_sha256",
            "representability_overlay_sha256",
        ):
            if not _is_sha256(getattr(self, field)):
                raise ValueError(f"{field} must be a lowercase SHA-256")
        if not self.entries:
            raise ValueError("successor-fiber inventory cannot be empty")
        if self.entries != tuple(
            sorted(
                self.entries,
                key=lambda item: item.packed_shard_content_sha256,
            )
        ):
            raise ValueError("successor-fiber inventory entries are not canonical")
        digests = tuple(
            entry.packed_shard_content_sha256 for entry in self.entries
        )
        paths = tuple(entry.cache_relative_path for entry in self.entries)
        if len(digests) != len(set(digests)):
            raise ValueError("successor-fiber inventory repeats a packed shard")
        if len(paths) != len(set(paths)):
            raise ValueError("successor-fiber inventory repeats a cache path")
        if not _is_sha256(self.inventory_sha256):
            raise ValueError("inventory_sha256 must be a lowercase SHA-256")
        if self.status != SUCCESSOR_FIBER_INVENTORY_STATUS:
            raise ValueError("inventory status is not bounded-development")
        if self.storage_backend not in SUPPORTED_SUCCESSOR_FIBER_STORAGE_BACKENDS:
            raise ValueError("inventory storage backend is unsupported")
        if type(self.full_corpus_training_authorized) is not bool:
            raise TypeError("full_corpus_training_authorized must be Boolean")
        if self.full_corpus_training_authorized:
            raise ValueError(
                "a bounded-development successor cache cannot authorize "
                "full-corpus training"
            )
        for entry in self.entries:
            provenance = entry.provenance
            if (
                provenance.unified_packed_manifest_sha256
                != self.unified_packed_manifest_sha256
                or provenance.representability_overlay_sha256
                != self.representability_overlay_sha256
            ):
                raise ValueError(
                    "cache shard provenance disagrees with inventory-wide corpus "
                    "identities"
                )

    @property
    def by_packed_digest(self) -> Mapping[str, SuccessorFiberShardSpec]:
        return {
            entry.packed_shard_content_sha256: entry
            for entry in self.entries
        }


def _spec_payload(spec: SuccessorFiberShardSpec) -> dict[str, Any]:
    return {
        "packed_shard_content_sha256": spec.packed_shard_content_sha256,
        "packed_shard_name": spec.packed_shard_name,
        "layer": spec.layer,
        "partition": spec.partition,
        "packed_manifest_sha256": spec.packed_manifest_sha256,
        "packed_provenance_overlay_sha256": (
            spec.packed_provenance_overlay_sha256
        ),
        "cache_relative_path": spec.cache_relative_path,
        "cache_content_sha256": spec.cache_content_sha256,
        "cache_file_sha256": spec.cache_file_sha256,
        "cache_file_bytes": spec.cache_file_bytes,
        "packed_entry_count": spec.packed_entry_count,
        "active_trace_count": spec.active_trace_count,
        "excluded_entry_indices": list(spec.excluded_entry_indices),
        "progress_record_count": spec.progress_record_count,
        "jump_record_count": spec.jump_record_count,
        "terminal_record_count": spec.terminal_record_count,
        "teacher_alias_count": spec.teacher_alias_count,
        "virtual_alias_count": spec.virtual_alias_count,
        "maximum_alias_count": spec.maximum_alias_count,
        "provenance": asdict(spec.provenance),
    }


def seal_successor_fiber_cache_inventory(
    entries: Iterable[SuccessorFiberShardSpec],
    *,
    source_corpus_inventory_sha256: str,
    compatibility: SuccessorFiberCacheCompatibility,
    unified_packed_manifest_sha256: str,
    representability_overlay_sha256: str,
    storage_backend: str = SUCCESSOR_FIBER_STORAGE_BACKEND,
) -> tuple[bytes, SuccessorFiberCacheInventory]:
    """Create canonical, self-hashed bounded-development inventory bytes."""

    ordered = tuple(
        sorted(
            entries,
            key=lambda item: item.packed_shard_content_sha256,
        )
    )
    provisional = {
        "schema": SUCCESSOR_FIBER_INVENTORY_SCHEMA,
        "schema_version": SUCCESSOR_FIBER_INVENTORY_SCHEMA_VERSION,
        "status": SUCCESSOR_FIBER_INVENTORY_STATUS,
        "storage_backend": storage_backend,
        "full_corpus_training_authorized": False,
        "source_corpus_inventory_sha256": source_corpus_inventory_sha256,
        "compatibility_payload": compatibility.payload,
        "compatibility_sha256": compatibility.compatibility_sha256,
        "unified_packed_manifest_sha256": unified_packed_manifest_sha256,
        "representability_overlay_sha256": (
            representability_overlay_sha256
        ),
        "entry_count": len(ordered),
        "entries": [_spec_payload(entry) for entry in ordered],
    }
    provisional["inventory_sha256"] = _inventory_self_hash(provisional)
    encoded = _canonical_json_bytes(provisional) + b"\n"
    if len(encoded) > MAX_INVENTORY_BYTES:
        raise ShardedSuccessorFiberCacheError("cache inventory exceeds size bound")
    inventory = deserialize_successor_fiber_cache_inventory(
        encoded,
        expected_inventory_sha256=provisional["inventory_sha256"],
        expected_compatibility_sha256=compatibility.compatibility_sha256,
    )
    return encoded, inventory


def _decode_provenance(value: object) -> SuccessorFiberCacheProvenance:
    if not isinstance(value, Mapping):
        raise ShardedSuccessorFiberCacheError(
            "cache-inventory provenance must be an object"
        )
    try:
        return SuccessorFiberCacheProvenance(**dict(value))
    except (TypeError, ValueError) as error:
        raise ShardedSuccessorFiberCacheError(
            "cache-inventory provenance is invalid"
        ) from error


def _decode_spec(value: object) -> SuccessorFiberShardSpec:
    payload = _require_exact_mapping(
        value,
        fields=_ENTRY_FIELDS,
        name="cache inventory entry",
    )
    excluded = payload["excluded_entry_indices"]
    if not isinstance(excluded, list):
        raise ShardedSuccessorFiberCacheError(
            "excluded_entry_indices must be an array"
        )
    try:
        return SuccessorFiberShardSpec(
            packed_shard_content_sha256=payload[
                "packed_shard_content_sha256"
            ],
            packed_shard_name=payload["packed_shard_name"],
            layer=payload["layer"],
            partition=payload["partition"],
            packed_manifest_sha256=payload["packed_manifest_sha256"],
            packed_provenance_overlay_sha256=payload[
                "packed_provenance_overlay_sha256"
            ],
            cache_relative_path=payload["cache_relative_path"],
            cache_content_sha256=payload["cache_content_sha256"],
            cache_file_sha256=payload["cache_file_sha256"],
            cache_file_bytes=payload["cache_file_bytes"],
            packed_entry_count=payload["packed_entry_count"],
            active_trace_count=payload["active_trace_count"],
            excluded_entry_indices=tuple(excluded),
            progress_record_count=payload["progress_record_count"],
            jump_record_count=payload["jump_record_count"],
            terminal_record_count=payload["terminal_record_count"],
            teacher_alias_count=payload["teacher_alias_count"],
            virtual_alias_count=payload["virtual_alias_count"],
            maximum_alias_count=payload["maximum_alias_count"],
            provenance=_decode_provenance(payload["provenance"]),
        )
    except (TypeError, ValueError) as error:
        raise ShardedSuccessorFiberCacheError(
            "cache inventory entry is invalid"
        ) from error


def deserialize_successor_fiber_cache_inventory(
    encoded: bytes,
    *,
    expected_inventory_sha256: str,
    expected_compatibility_sha256: str,
) -> SuccessorFiberCacheInventory:
    """Decode an exact self-hashed inventory; no best-effort recovery."""

    if len(encoded) > MAX_INVENTORY_BYTES:
        raise ShardedSuccessorFiberCacheError("cache inventory exceeds size bound")
    try:
        value = json.loads(
            encoded,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ShardedSuccessorFiberCacheError(
            "cache inventory is not valid finite JSON"
        ) from error
    payload = _require_exact_mapping(
        value,
        fields=_ROOT_FIELDS,
        name="cache inventory root",
    )
    if payload["schema"] != SUCCESSOR_FIBER_INVENTORY_SCHEMA:
        raise ShardedSuccessorFiberCacheError("cache inventory schema drifted")
    if (
        payload["schema_version"]
        != SUCCESSOR_FIBER_INVENTORY_SCHEMA_VERSION
    ):
        raise ShardedSuccessorFiberCacheError(
            "cache inventory schema version drifted"
        )
    if payload["status"] != SUCCESSOR_FIBER_INVENTORY_STATUS:
        raise ShardedSuccessorFiberCacheError("cache inventory status drifted")
    if (
        payload["storage_backend"]
        not in SUPPORTED_SUCCESSOR_FIBER_STORAGE_BACKENDS
    ):
        raise ShardedSuccessorFiberCacheError(
            "cache inventory backend drifted"
        )
    if payload["full_corpus_training_authorized"] is not False:
        raise ShardedSuccessorFiberCacheError(
            "development cache inventory claims full training authorization"
        )
    recorded_hash = payload["inventory_sha256"]
    if not _is_sha256(recorded_hash):
        raise ShardedSuccessorFiberCacheError(
            "cache inventory lacks a SHA-256"
        )
    if not _is_sha256(expected_inventory_sha256):
        raise ShardedSuccessorFiberCacheError(
            "expected inventory hash is invalid"
        )
    observed_hash = _inventory_self_hash(payload)
    if observed_hash != recorded_hash:
        raise ShardedSuccessorFiberCacheError(
            "cache inventory self-hash does not match"
        )
    if recorded_hash != expected_inventory_sha256:
        raise ShardedSuccessorFiberCacheError(
            "cache inventory does not match the frozen expected hash"
        )
    try:
        compatibility = SuccessorFiberCacheCompatibility.from_payload(
            payload["compatibility_payload"]
        )
    except (ShardedSuccessorFiberCacheError, TypeError, ValueError) as error:
        raise ShardedSuccessorFiberCacheError(
            "cache inventory compatibility payload is invalid"
        ) from error
    recorded_compatibility = payload["compatibility_sha256"]
    if not _is_sha256(recorded_compatibility):
        raise ShardedSuccessorFiberCacheError(
            "cache inventory compatibility hash is invalid"
        )
    if compatibility.compatibility_sha256 != recorded_compatibility:
        raise ShardedSuccessorFiberCacheError(
            "cache inventory compatibility payload/hash disagree"
        )
    if not _is_sha256(expected_compatibility_sha256):
        raise ShardedSuccessorFiberCacheError(
            "expected compatibility hash is invalid"
        )
    if recorded_compatibility != expected_compatibility_sha256:
        raise ShardedSuccessorFiberCacheError(
            "cache inventory does not match the launch compatibility identity"
        )
    entries_payload = payload["entries"]
    if not isinstance(entries_payload, list):
        raise ShardedSuccessorFiberCacheError(
            "cache inventory entries must be an array"
        )
    if _nonnegative_int(payload["entry_count"], field="entry_count") != len(
        entries_payload
    ):
        raise ShardedSuccessorFiberCacheError(
            "cache inventory entry count does not match"
        )
    entries = tuple(_decode_spec(entry) for entry in entries_payload)
    try:
        inventory = SuccessorFiberCacheInventory(
            source_corpus_inventory_sha256=payload[
                "source_corpus_inventory_sha256"
            ],
            compatibility=compatibility,
            unified_packed_manifest_sha256=payload[
                "unified_packed_manifest_sha256"
            ],
            representability_overlay_sha256=payload[
                "representability_overlay_sha256"
            ],
            entries=entries,
            inventory_sha256=recorded_hash,
            status=payload["status"],
            storage_backend=payload["storage_backend"],
            full_corpus_training_authorized=payload[
                "full_corpus_training_authorized"
            ],
        )
    except (TypeError, ValueError) as error:
        raise ShardedSuccessorFiberCacheError(
            f"cache inventory contract is invalid: {error}"
        ) from error
    return inventory


def write_successor_fiber_cache_inventory(
    path: str | Path,
    encoded: bytes,
    *,
    expected_inventory_sha256: str,
    expected_compatibility_sha256: str,
) -> Path:
    """Validate and atomically freeze inventory bytes without overwriting."""

    deserialize_successor_fiber_cache_inventory(
        encoded,
        expected_inventory_sha256=expected_inventory_sha256,
        expected_compatibility_sha256=expected_compatibility_sha256,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            if destination.read_bytes() != encoded:
                raise FileExistsError(
                    "successor-fiber inventory already exists with different "
                    f"content: {destination}"
                ) from None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return destination


def read_successor_fiber_cache_inventory(
    path: str | Path,
    *,
    expected_inventory_sha256: str,
    expected_compatibility_sha256: str,
) -> SuccessorFiberCacheInventory:
    source = Path(path)
    if not source.is_file():
        raise ShardedSuccessorFiberCacheError(
            f"cache inventory does not exist: {source}"
        )
    if source.stat().st_size > MAX_INVENTORY_BYTES:
        raise ShardedSuccessorFiberCacheError("cache inventory exceeds size bound")
    return deserialize_successor_fiber_cache_inventory(
        source.read_bytes(),
        expected_inventory_sha256=expected_inventory_sha256,
        expected_compatibility_sha256=expected_compatibility_sha256,
    )


def summarize_successor_fiber_cache(
    cache: SuccessorFiberCache,
) -> dict[str, int]:
    records = cache.records
    trace_keys = {record.address.trace_key for record in records}
    terminal_count = sum(record.address.is_terminal for record in records)
    teacher_alias_counts = tuple(
        0 if record.teacher_fiber is None else len(record.teacher_fiber.aliases)
        for record in records
    )
    virtual_alias_counts = tuple(
        len(record.state_support.virtual_aliases) for record in records
    )
    alias_counts = tuple(
        teacher + virtual
        for teacher, virtual in zip(
            teacher_alias_counts,
            virtual_alias_counts,
            strict=True,
        )
    )
    return {
        "active_trace_count": len(trace_keys),
        "progress_record_count": len(records),
        "jump_record_count": len(records) - terminal_count,
        "terminal_record_count": terminal_count,
        "teacher_alias_count": sum(teacher_alias_counts),
        "virtual_alias_count": sum(virtual_alias_counts),
        "maximum_alias_count": max(alias_counts, default=0),
    }


class ShardedSuccessorFiberCache:
    """Lazy exact-address cache over one frozen backend per inventory."""

    def __init__(
        self,
        cache_root: str | Path,
        inventory: SuccessorFiberCacheInventory,
        *,
        expected_inventory_sha256: str,
        expected_compatibility_sha256: str,
        max_open_shards: int = 2,
        limits: SuccessorFiberCacheLimits = DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
    ) -> None:
        if type(max_open_shards) is not int or max_open_shards <= 0:
            raise ValueError("max_open_shards must be a positive integer")
        if inventory.inventory_sha256 != expected_inventory_sha256:
            raise ShardedSuccessorFiberCacheError(
                "cache store inventory does not match the independently frozen hash"
            )
        if (
            inventory.compatibility.compatibility_sha256
            != expected_compatibility_sha256
        ):
            raise ShardedSuccessorFiberCacheError(
                "cache store compatibility does not match the launch identity"
            )
        self.cache_root = Path(cache_root)
        self.inventory = inventory
        self.max_open_shards = max_open_shards
        self.limits = limits
        self._specs = dict(inventory.by_packed_digest)
        self._loaded: OrderedDict[
            str,
            SuccessorFiberCache | IndexedSuccessorFiberCache,
        ] = OrderedDict()
        compatibility = inventory.compatibility.payload
        support_signature_hash = hashlib.sha256(
            _canonical_json_bytes(compatibility["support_signature"])
        ).hexdigest()
        for spec in inventory.entries:
            provenance = spec.provenance
            expected_fields = {
                "operator_registry_hash": compatibility[
                    "operator_registry_hash"
                ],
                "capability_hash": compatibility["capability_hash"],
                "support_signature_sha256": support_signature_hash,
                "canonicalization_version": compatibility[
                    "canonicalization_version"
                ],
                "canonicalizer_contract_sha256": compatibility[
                    "canonicalizer_contract_sha256"
                ],
                "packed_corpus_schema": compatibility[
                    "packed_corpus_schema"
                ],
                "packed_corpus_schema_version": compatibility[
                    "packed_corpus_schema_version"
                ],
                "coordinate_schema_version": compatibility[
                    "coordinate_schema_version"
                ],
                "tensorization_implementation_hash": compatibility[
                    "tensorization_implementation_hash"
                ],
                "fiber_compiler_implementation_hash": compatibility[
                    "fiber_compiler_implementation_hash"
                ],
            }
            disagreements = {
                field: {
                    "cache": getattr(provenance, field),
                    "compatibility": expected,
                }
                for field, expected in expected_fields.items()
                if getattr(provenance, field) != expected
            }
            if disagreements:
                raise ShardedSuccessorFiberCacheError(
                    "cache provenance disagrees with compatibility payload: "
                    f"{disagreements}"
                )

    @property
    def loaded_shard_digests(self) -> tuple[str, ...]:
        return tuple(self._loaded)

    def _path_for(self, spec: SuccessorFiberShardSpec) -> Path:
        root = self.cache_root.resolve()
        candidate = (root / spec.cache_relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ShardedSuccessorFiberCacheError(
                "cache inventory path escapes its declared root"
            ) from error
        return candidate

    def _verify_loaded(
        self,
        cache: SuccessorFiberCache | IndexedSuccessorFiberCache,
        spec: SuccessorFiberShardSpec,
    ) -> None:
        if isinstance(cache, IndexedSuccessorFiberCache):
            metadata = cache.metadata
            observed = {
                "active_trace_count": metadata.active_trace_count,
                "progress_record_count": metadata.record_count,
                "jump_record_count": metadata.jump_record_count,
                "terminal_record_count": metadata.terminal_record_count,
                "teacher_alias_count": metadata.teacher_alias_count,
                "virtual_alias_count": metadata.virtual_alias_count,
                "maximum_alias_count": metadata.maximum_alias_count,
            }
        else:
            observed = summarize_successor_fiber_cache(cache)
        expected = {
            "active_trace_count": spec.active_trace_count,
            "progress_record_count": spec.progress_record_count,
            "jump_record_count": spec.jump_record_count,
            "terminal_record_count": spec.terminal_record_count,
            "teacher_alias_count": spec.teacher_alias_count,
            "virtual_alias_count": spec.virtual_alias_count,
            "maximum_alias_count": spec.maximum_alias_count,
        }
        if observed != expected:
            raise ShardedSuccessorFiberCacheError(
                f"cache shard census mismatch: expected={expected}, "
                f"observed={observed}"
            )
        if isinstance(cache, IndexedSuccessorFiberCache):
            if (
                cache.metadata.packed_shard_name != spec.packed_shard_name
                or cache.metadata.layer != spec.layer
                or cache.metadata.partition != spec.partition
            ):
                raise ShardedSuccessorFiberCacheError(
                    "indexed cache envelope disagrees with its inventory entry"
                )
            cache.require_entry_census(
                packed_entry_count=spec.packed_entry_count,
                excluded_entry_indices=spec.excluded_entry_indices,
            )
            return
        for record in cache.records:
            address = record.address
            if (
                address.packed_shard_content_sha256
                != spec.packed_shard_content_sha256
                or address.packed_shard_name != spec.packed_shard_name
                or address.layer != spec.layer
                or address.partition != spec.partition
            ):
                raise ShardedSuccessorFiberCacheError(
                    "cache record envelope disagrees with its inventory entry"
                )
        observed_entry_indices = {
            record.address.entry_index for record in cache.records
        }
        expected_entry_indices = set(range(spec.packed_entry_count)) - set(
            spec.excluded_entry_indices
        )
        if observed_entry_indices != expected_entry_indices:
            missing = sorted(expected_entry_indices - observed_entry_indices)
            unexpected = sorted(observed_entry_indices - expected_entry_indices)
            raise ShardedSuccessorFiberCacheError(
                "cache shard omits or invents whole packed traces: "
                f"missing={missing[:20]}, unexpected={unexpected[:20]}"
            )

    def _load(
        self,
        spec: SuccessorFiberShardSpec,
    ) -> SuccessorFiberCache | IndexedSuccessorFiberCache:
        digest = spec.packed_shard_content_sha256
        cached = self._loaded.get(digest)
        if cached is not None:
            self._loaded.move_to_end(digest)
            return cached
        path = self._path_for(spec)
        if not path.is_file():
            raise ShardedSuccessorFiberCacheError(
                f"declared cache shard is absent: {path}"
            )
        if self.inventory.storage_backend == SUCCESSOR_FIBER_INDEXED_STORAGE_BACKEND:
            try:
                cache = open_indexed_successor_fiber_cache(
                    path,
                    expected_provenance=spec.provenance,
                    expected_content_sha256=spec.cache_content_sha256,
                    expected_file_sha256=spec.cache_file_sha256,
                    expected_file_bytes=spec.cache_file_bytes,
                    limits=self.limits,
                )
            except IndexedSuccessorFiberCacheError as error:
                raise ShardedSuccessorFiberCacheError(
                    "indexed successor-cache shard failed verification"
                ) from error
        else:
            if path.stat().st_size != spec.cache_file_bytes:
                raise ShardedSuccessorFiberCacheError(
                    "cache shard byte count disagrees with inventory"
                )
            if _sha256_file(path) != spec.cache_file_sha256:
                raise ShardedSuccessorFiberCacheError(
                    "cache shard byte SHA-256 disagrees with inventory"
                )
            cache = read_successor_fiber_cache(
                path,
                expected_provenance=spec.provenance,
                expected_content_sha256=spec.cache_content_sha256,
                limits=self.limits,
            )
        try:
            self._verify_loaded(cache, spec)
        except IndexedSuccessorFiberCacheError as error:
            raise ShardedSuccessorFiberCacheError(
                "indexed successor-cache shard failed its inventory census"
            ) from error
        self._loaded[digest] = cache
        self._loaded.move_to_end(digest)
        while len(self._loaded) > self.max_open_shards:
            _, evicted = self._loaded.popitem(last=False)
            if isinstance(evicted, IndexedSuccessorFiberCache):
                evicted.close()
        return cache

    def require(
        self,
        address: PackedTraceAddress,
        *,
        progress_index: int,
        source_state: MolecularGraph,
    ) -> SuccessorFiberCacheRecord:
        """Resolve one exact row and validate the sampled persistent state."""

        if type(progress_index) is not int or not 0 <= progress_index <= address.path_length:
            raise ValueError("progress_index lies outside the packed trace")
        try:
            spec = self._specs[address.packed_shard_content_sha256]
        except KeyError:
            raise ShardedSuccessorFiberCacheError(
                "packed shard is absent from the frozen successor-cache inventory"
            ) from None
        if (
            address.packed_shard_name != spec.packed_shard_name
            or address.layer != spec.layer
            or address.partition != spec.partition
        ):
            raise ShardedSuccessorFiberCacheError(
                "sampled packed address disagrees with the inventory lane"
            )
        cache = self._load(spec)
        try:
            record = cache.require_packed_address(
                address,
                progress_index=progress_index,
            )
        except (KeyError, IndexedSuccessorFiberCacheError) as error:
            raise ShardedSuccessorFiberCacheError(
                "successor cache has no exact sampled trace-progress row"
            ) from error
        observed_state_sha256 = persistent_slot_state_sha256(source_state)
        if observed_state_sha256 != record.source_state_sha256:
            raise ShardedSuccessorFiberCacheError(
                "sampled exact state disagrees with its successor-cache row"
            )
        return record

    def __getstate__(self) -> dict[str, Any]:
        state = copy.copy(self.__dict__)
        state["_loaded"] = OrderedDict()
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(state)

    def assert_full_corpus_training_authorized(self) -> None:
        raise ShardedSuccessorFiberCacheError(
            "this successor-cache inventory is bounded-development only; "
            "complete corpus and training-gate evidence has not been frozen"
        )


def assert_store_is_pickle_safe(store: ShardedSuccessorFiberCache) -> None:
    """Diagnostic guard used before DataLoader multiprocessing."""

    restored = pickle.loads(pickle.dumps(store))
    if restored.loaded_shard_digests:
        raise ShardedSuccessorFiberCacheError(
            "pickled successor cache retained worker-unsafe loaded shards"
        )


__all__ = [
    "MAX_INVENTORY_BYTES",
    "SUCCESSOR_FIBER_INDEXED_STORAGE_BACKEND",
    "SUCCESSOR_FIBER_INVENTORY_SCHEMA",
    "SUCCESSOR_FIBER_INVENTORY_SCHEMA_VERSION",
    "SUCCESSOR_FIBER_INVENTORY_STATUS",
    "SUCCESSOR_FIBER_STORAGE_BACKEND",
    "SUPPORTED_SUCCESSOR_FIBER_STORAGE_BACKENDS",
    "ShardedSuccessorFiberCache",
    "ShardedSuccessorFiberCacheError",
    "SuccessorFiberCacheCompatibility",
    "SuccessorFiberCacheInventory",
    "SuccessorFiberShardSpec",
    "assert_store_is_pickle_safe",
    "deserialize_successor_fiber_cache_inventory",
    "read_successor_fiber_cache_inventory",
    "seal_successor_fiber_cache_inventory",
    "summarize_successor_fiber_cache",
    "write_successor_fiber_cache_inventory",
]
