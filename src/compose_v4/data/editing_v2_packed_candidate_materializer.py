"""Stream immutable packed traces into compact Editing-V2 candidate headers.

This module is deliberately upstream of split assignment, whole-trace Active8
admission, and training.  It does not decode molecular states, decode actions,
invoke RDKit, or replay the executor.  Instead it verifies and hashes the exact
wire objects already frozen in a packed trace cache.

The input source manifest binds every logical source asset, physical packed
shard, original packed manifest, and frozen provenance-overlay sidecar by
SHA-256. Historical sidecars are verified only as immutable byte-binding
receipts. Their recorded implementation fields are never treated as current
training provenance. Each packed JSONL entry is consumed once and reduced to a
compact header containing:

* its immutable physical shard and entry address;
* exact encoded-state count and stream identities;
* endpoint, molecule, scaffold, and source-group identities;
* operator-family counts and atom-count/cycle-rank deltas;
* the committed five-lane routing receipt and six-component evidence profile;
* or an explicit rejected disposition for disabled/off-basis operations.

Outputs are written into a same-parent staging directory and published by one
directory rename.  A failed scan therefore leaves no partial public artifact.
The implementation retains only one packed entry plus bounded counters in
memory.  Its output is candidate evidence only and carries no split, admission,
Gate-0, T1, P50, checkpoint-selection, or training authority.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from compose_v4.data.editing_corpus_contract import (
    ACTIVE8_FAMILIES,
    DISABLED_FAMILIES,
    load_editing_corpus_contract,
    validate_evidence_assignment_envelope,
)
from compose_v4.data.editing_v2_candidate_router import (
    EXECUTOR_GENERATED_WALK_FROM_REAL_ENDPOINT,
    INFERRED_REAL_ENDPOINT_PAIR,
    load_routing_policy,
    route_editing_v2_candidate,
    route_receipt_payload,
    routing_policy_sha256,
)

SOURCE_MANIFEST_SCHEMA = "compose.editing_v2_packed_candidate_sources"
SOURCE_MANIFEST_SCHEMA_VERSION = 2
SOURCE_MANIFEST_STATUS = "FROZEN_PACKED_CACHE_SOURCES_NO_TRAINING_AUTHORITY"

SOURCE_BINDING_REGISTRY_SCHEMA = "compose.editing_v2_overlay_source_bindings"
SOURCE_BINDING_REGISTRY_SCHEMA_VERSION = 1
SOURCE_BINDING_REGISTRY_STATUS = "FROZEN_LAYER_SOURCE_BINDINGS_NO_TRAINING_AUTHORITY"

UPSTREAM_OVERLAY_COMPLETION_SCHEMA = "compose.editing_v2_upstream_overlay_completion"
UPSTREAM_OVERLAY_COMPLETION_SCHEMA_VERSION = 2
UPSTREAM_OVERLAY_COMPLETION_STATUS = "COMPLETE_IMMUTABLE_BYTE_BINDINGS_NO_TRAINING_AUTHORITY"

CANDIDATE_HEADER_SCHEMA = "compose.editing_v2_packed_candidate_header"
CANDIDATE_HEADER_SCHEMA_VERSION = 1
CANDIDATE_HEADER_STATUS = "CANDIDATE_HEADER_NO_SPLIT_OR_TRAINING_AUTHORITY"

MATERIALIZATION_SCHEMA = "compose.editing_v2_packed_candidate_materialization"
MATERIALIZATION_SCHEMA_VERSION = 1
MATERIALIZATION_STATUS = "COMPLETE_CANDIDATE_HEADERS_NO_TRAINING_AUTHORITY"
DERIVATIVE_PROVENANCE_STATUS = "FRESH_EDITING_V2_DERIVATIVE_PROVENANCE"

PACKED_STORE_SCHEMA = "compose.data.packed_trace"
PACKED_STORE_SCHEMA_VERSION = 1
HISTORICAL_OVERLAY_SCHEMA = "compose.data.provenance_overlay"
HISTORICAL_OVERLAY_SCHEMA_VERSION = 1
HISTORICAL_OVERLAY_ROLE = "HISTORICAL_BYTE_BINDING_RECEIPT_ONLY_NOT_CURRENT_TRAINING_PROVENANCE"
TRACE_SCHEMA = "compose.rewrite.trace"
TRACE_SCHEMA_VERSION = 2
ACTION_SCHEMA = "compose.rewrite.action"
ACTION_SCHEMA_VERSION = 2
PARTITION_ISOLATION_SCHEMA = "compose.data.partition_isolation"
PARTITION_ISOLATION_SCHEMA_VERSION = 1

CANDIDATE_ROWS_FILENAME = "candidate_headers.jsonl"
MATERIALIZATION_FILENAME = "PACKED_CANDIDATE_MATERIALIZATION.json"

DEFAULT_MAX_PACKED_ENTRY_BYTES = 16 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHORT_IMPLEMENTATION_HASH_RE = re.compile(r"^[0-9a-f]{16}$")
_ACTIVE8_SET = frozenset(ACTIVE8_FAMILIES)
_DISABLED_SET = frozenset(DISABLED_FAMILIES)
_SOURCE_KINDS = frozenset(
    {
        INFERRED_REAL_ENDPOINT_PAIR,
        EXECUTOR_GENERATED_WALK_FROM_REAL_ENDPOINT,
    }
)
_REAL_ENDPOINT_LAYERS = frozenset({"mmp_analogue"})
_SYNTHETIC_LAYERS = frozenset({"corruption", "cycle_ops"})

_SOURCE_FIELDS = {
    "source_asset_id",
    "source_asset_path",
    "source_asset_sha256",
    "source_kind",
    "allowed_layers",
    "shards",
}
_SOURCE_BINDING_FIELDS = _SOURCE_FIELDS - {"shards"}
_SHARD_FIELDS = {
    "relative_path",
    "file_sha256",
    "manifest_relative_path",
    "manifest_file_sha256",
    "historical_provenance_overlay_relative_path",
    "historical_provenance_overlay_file_sha256",
    "historical_provenance_overlay_role",
}
_SOURCE_MANIFEST_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "upstream_overlay_completion",
    "source_binding_registry",
    "sources",
    "manifest_sha256",
}
_SOURCE_BINDING_REGISTRY_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "bindings",
    "registry_sha256",
}
_OVERLAY_COMPLETION_IDENTITY_FIELDS = {
    "artifact_path",
    "file_sha256",
    "semantic_sha256",
    "candidate_source_shards",
    "candidate_source_inventory_sha256",
}
_SOURCE_BINDING_REGISTRY_IDENTITY_FIELDS = {
    "artifact_path",
    "file_sha256",
    "semantic_sha256",
}
_UPSTREAM_COMPLETION_FIELDS = {
    "path",
    "file_sha256",
    "completion_flag",
    "inventory_origin",
    "expected_shards",
    "shard_inventory_sha256",
}
_OVERLAY_COMPLETION_RECEIPT_FIELDS = {
    "layer",
    "partition",
    "shard_path",
    "shard_file_sha256",
    "manifest_path",
    "manifest_file_sha256",
    "entries",
    "states",
    "overlay_path",
    "overlay_file_sha256",
    "overlay_fields_sha256",
    "overlay_semantic_sha256",
    "overlay_validation_role",
}
_OVERLAY_COMPLETION_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "commit",
    "launcher_source_sha256",
    "packed_root",
    "mmp_root",
    "upstream_completions",
    "fields_for_new_overlays",
    "sentinel_entries_per_new_overlay",
    "shards",
    "completion_sha256",
}
_OVERLAY_VALIDATION_ROLES = frozenset(
    {
        "current_live_overlay_at_completion",
        "historical_immutable_byte_receipt",
    }
)
_COMPLETION_LAYERS = frozenset({"corruption", "cycle_ops", "mmp_analogue"})
_COMPLETION_PARTITIONS = frozenset({"train", "validation", "test"})
_PACKED_MANIFEST_FIELDS = {
    "schema",
    "schema_version",
    "sampler_contract",
    "entries",
    "states",
    "provenance",
}
_HISTORICAL_OVERLAY_FIELDS = {
    "schema",
    "schema_version",
    "shard",
    "packed_shard_content_sha256",
    "original_manifest_sha256",
    "fields",
    "packer_commit",
    "upgrade_implementation_hash",
    "certification",
}
_PARTITION_ISOLATION_FIELDS = {
    "schema",
    "schema_version",
    "molecule_ids",
    "scaffold_ids",
    "source_group_id",
}
_ENCODED_STATE_FIELDS = {
    "n_slots",
    "atom_types",
    "formal_charges",
    "implicit_h_counts",
    "bonds",
}
_ACTION_FIELDS = {
    "schema",
    "schema_version",
    "executor_rule",
    "model_family",
    "payload_type",
    "payload",
}
_EXECUTOR_TO_FAMILY = {
    "atom_insert": "atom_insert",
    "atom_delete": "atom_delete",
    "atom_restate": "atom_restate",
    "bond_reorder": "bond_reorder",
    "bond_reroute": "bond_reroute",
    "bond_insert": "cycle_insert",
    "bond_delete": "cycle_attach",
    "ring_system_delete": "ring_system_delete",
    "ring_system_restate": "ring_system_restate",
    # Historical caches may contain an attempted macro row even though the
    # current action codec refuses to persist new ring-system-growth actions.
    "ring_system_grow": "ring_system_grow",
}


class EditingV2PackedCandidateMaterializationError(ValueError):
    """A packed source or candidate materialization is incomplete or unsafe."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one deterministic JSON encoding used by all identities."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EditingV2PackedCandidateMaterializationError(
            f"value is not finite deterministic JSON: {error}"
        ) from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_fields(
    value: object,
    fields: set[str],
    *,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EditingV2PackedCandidateMaterializationError(f"{field} must be an object")
    if set(value) != fields:
        raise EditingV2PackedCandidateMaterializationError(
            f"{field} fields disagree; "
            f"missing={sorted(fields - set(value))}, "
            f"unexpected={sorted(set(value) - fields)}"
        )
    return value


def _require_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise EditingV2PackedCandidateMaterializationError(
            f"{field} must be nonempty normalized text"
        )
    return value


def _require_sha256(value: object, *, field: str) -> str:
    digest = value if isinstance(value, str) else ""
    if _SHA256_RE.fullmatch(digest) is None:
        raise EditingV2PackedCandidateMaterializationError(
            f"{field} must be a full lowercase SHA-256"
        )
    return digest


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise EditingV2PackedCandidateMaterializationError(f"{field} must be a nonnegative integer")
    return value


def _require_positive_int(value: object, *, field: str) -> int:
    number = _require_nonnegative_int(value, field=field)
    if number == 0:
        raise EditingV2PackedCandidateMaterializationError(f"{field} must be a positive integer")
    return number


def _require_sorted_unique_text(
    value: object,
    *,
    field: str,
    allowed: frozenset[str] | None = None,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise EditingV2PackedCandidateMaterializationError(f"{field} must be a nonempty list")
    items = tuple(
        _require_text(item, field=f"{field}[{index}]") for index, item in enumerate(value)
    )
    if len(items) != len(set(items)) or items != tuple(sorted(items)):
        raise EditingV2PackedCandidateMaterializationError(
            f"{field} must contain unique deterministically sorted text"
        )
    if allowed is not None and not set(items).issubset(allowed):
        raise EditingV2PackedCandidateMaterializationError(
            f"{field} contains unsupported values: {sorted(set(items) - allowed)}"
        )
    return items


def _normalize_unique_text_collection(
    value: object,
    *,
    field: str,
) -> tuple[str, ...]:
    """Validate an unordered wire collection and return canonical text order."""

    if not isinstance(value, list) or not value:
        raise EditingV2PackedCandidateMaterializationError(f"{field} must be a nonempty list")
    items = tuple(
        _require_text(item, field=f"{field}[{index}]") for index, item in enumerate(value)
    )
    if len(items) != len(set(items)):
        raise EditingV2PackedCandidateMaterializationError(
            f"{field} must not contain duplicate identities"
        )
    return tuple(sorted(items))


def _safe_relative_path(value: object, *, field: str) -> str:
    text = _require_text(value, field=field)
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or pure == PurePosixPath(".")
        or ".." in pure.parts
        or "\\" in text
        or any(character in text for character in "*?[]")
    ):
        raise EditingV2PackedCandidateMaterializationError(
            f"{field} must be a safe normalized relative path"
        )
    if str(pure) != text:
        raise EditingV2PackedCandidateMaterializationError(f"{field} must be normalized")
    return text


def _resolve_relative(root: Path, relative_path: str, *, field: str) -> Path:
    resolved_root = Path(root).resolve()
    candidate = (resolved_root / Path(relative_path)).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise EditingV2PackedCandidateMaterializationError(f"{field} escapes the artifact root")
    return candidate


def _normalized_source_binding(binding: object, *, index: int) -> dict[str, Any]:
    payload = _require_exact_fields(
        binding,
        _SOURCE_BINDING_FIELDS,
        field=f"bindings[{index}]",
    )
    source_kind = _require_text(
        payload["source_kind"],
        field=f"bindings[{index}].source_kind",
    )
    if source_kind not in _SOURCE_KINDS:
        raise EditingV2PackedCandidateMaterializationError(
            f"bindings[{index}].source_kind is unsupported: {source_kind!r}"
        )
    allowed_layers = _require_sorted_unique_text(
        payload["allowed_layers"],
        field=f"bindings[{index}].allowed_layers",
        allowed=_COMPLETION_LAYERS,
    )
    expected_layers = (
        _REAL_ENDPOINT_LAYERS if source_kind == INFERRED_REAL_ENDPOINT_PAIR else _SYNTHETIC_LAYERS
    )
    if not set(allowed_layers).issubset(expected_layers):
        raise EditingV2PackedCandidateMaterializationError(
            f"bindings[{index}].allowed_layers disagrees with source_kind"
        )
    return {
        "source_asset_id": _require_text(
            payload["source_asset_id"], field=f"bindings[{index}].source_asset_id"
        ),
        "source_asset_path": _require_text(
            payload["source_asset_path"], field=f"bindings[{index}].source_asset_path"
        ),
        "source_asset_sha256": _require_sha256(
            payload["source_asset_sha256"],
            field=f"bindings[{index}].source_asset_sha256",
        ),
        "source_kind": source_kind,
        "allowed_layers": list(allowed_layers),
    }


def build_overlay_source_binding_registry(
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze explicit scientific source ownership for every completion layer."""

    if isinstance(bindings, (str, bytes)) or not bindings:
        raise EditingV2PackedCandidateMaterializationError(
            "overlay source bindings must be a nonempty sequence"
        )
    normalized = [
        _normalized_source_binding(binding, index=index) for index, binding in enumerate(bindings)
    ]
    ordered = sorted(
        normalized,
        key=lambda item: (item["source_kind"], item["source_asset_id"]),
    )
    if normalized != ordered:
        raise EditingV2PackedCandidateMaterializationError(
            "overlay source bindings must be deterministically sorted"
        )
    source_ids = [item["source_asset_id"] for item in normalized]
    if len(source_ids) != len(set(source_ids)):
        raise EditingV2PackedCandidateMaterializationError(
            "overlay source bindings repeat a source_asset_id"
        )
    owners: dict[str, str] = {}
    for binding in normalized:
        for layer in binding["allowed_layers"]:
            if layer in owners:
                raise EditingV2PackedCandidateMaterializationError(
                    f"overlay source layer {layer!r} has multiple source owners"
                )
            owners[layer] = binding["source_asset_id"]
    if set(owners) != _COMPLETION_LAYERS:
        raise EditingV2PackedCandidateMaterializationError(
            "overlay source bindings must own corruption, cycle_ops, and mmp_analogue exactly once"
        )
    body = {
        "schema": SOURCE_BINDING_REGISTRY_SCHEMA,
        "schema_version": SOURCE_BINDING_REGISTRY_SCHEMA_VERSION,
        "status": SOURCE_BINDING_REGISTRY_STATUS,
        "training_authorized": False,
        "bindings": normalized,
    }
    return {**body, "registry_sha256": canonical_sha256(body)}


def validate_overlay_source_binding_registry(registry: object) -> dict[str, Any]:
    payload = _require_exact_fields(
        registry,
        _SOURCE_BINDING_REGISTRY_FIELDS,
        field="overlay source binding registry",
    )
    rebuilt = build_overlay_source_binding_registry(payload["bindings"])
    if dict(payload) != rebuilt:
        raise EditingV2PackedCandidateMaterializationError(
            "overlay source binding registry identity or self-hash disagrees"
        )
    return rebuilt


def load_overlay_source_binding_registry(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingV2PackedCandidateMaterializationError(
            f"cannot load overlay source binding registry {path}"
        ) from error
    return validate_overlay_source_binding_registry(payload)


def _source_inventory(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "source_asset_id": source["source_asset_id"],
                **dict(shard),
            }
            for source in sources
            for shard in source["shards"]
        ),
        key=lambda item: (item["source_asset_id"], item["relative_path"]),
    )


def _normalized_bound_identity(
    value: object,
    *,
    fields: set[str],
    field: str,
) -> dict[str, Any]:
    payload = _require_exact_fields(value, fields, field=field)
    artifact_path = _require_text(payload["artifact_path"], field=f"{field}.artifact_path")
    if Path(artifact_path).name == "" or ".." in PurePosixPath(artifact_path).parts:
        raise EditingV2PackedCandidateMaterializationError(
            f"{field}.artifact_path must be a normalized non-escaping path"
        )
    normalized = {
        "artifact_path": artifact_path,
        "file_sha256": _require_sha256(payload["file_sha256"], field=f"{field}.file_sha256"),
        "semantic_sha256": _require_sha256(
            payload["semantic_sha256"], field=f"{field}.semantic_sha256"
        ),
    }
    return normalized


def _normalized_source(source: object, *, index: int) -> dict[str, Any]:
    payload = _require_exact_fields(
        source,
        _SOURCE_FIELDS,
        field=f"sources[{index}]",
    )
    source_kind = _require_text(
        payload["source_kind"],
        field=f"sources[{index}].source_kind",
    )
    if source_kind not in _SOURCE_KINDS:
        raise EditingV2PackedCandidateMaterializationError(
            f"sources[{index}].source_kind is unsupported: {source_kind!r}"
        )
    allowed_layers = _require_sorted_unique_text(
        payload["allowed_layers"],
        field=f"sources[{index}].allowed_layers",
    )
    expected_layers = (
        _REAL_ENDPOINT_LAYERS if source_kind == INFERRED_REAL_ENDPOINT_PAIR else _SYNTHETIC_LAYERS
    )
    if not set(allowed_layers).issubset(expected_layers):
        raise EditingV2PackedCandidateMaterializationError(
            f"sources[{index}].allowed_layers disagrees with source_kind"
        )
    raw_shards = payload["shards"]
    if not isinstance(raw_shards, list) or not raw_shards:
        raise EditingV2PackedCandidateMaterializationError(
            f"sources[{index}].shards must be nonempty"
        )
    shards: list[dict[str, str]] = []
    for shard_index, raw_shard in enumerate(raw_shards):
        shard = _require_exact_fields(
            raw_shard,
            _SHARD_FIELDS,
            field=f"sources[{index}].shards[{shard_index}]",
        )
        relative_path = _safe_relative_path(
            shard["relative_path"],
            field=f"sources[{index}].shards[{shard_index}].relative_path",
        )
        manifest_relative_path = _safe_relative_path(
            shard["manifest_relative_path"],
            field=(f"sources[{index}].shards[{shard_index}].manifest_relative_path"),
        )
        expected_manifest = str(PurePosixPath(relative_path).with_suffix(".manifest.json"))
        if manifest_relative_path != expected_manifest:
            raise EditingV2PackedCandidateMaterializationError(
                f"sources[{index}].shards[{shard_index}] manifest path must equal "
                f"{expected_manifest!r}"
            )
        overlay_relative_path = _safe_relative_path(
            shard["historical_provenance_overlay_relative_path"],
            field=(
                f"sources[{index}].shards[{shard_index}]"
                ".historical_provenance_overlay_relative_path"
            ),
        )
        expected_overlay = f"{relative_path}.provenance.json"
        if overlay_relative_path != expected_overlay:
            raise EditingV2PackedCandidateMaterializationError(
                f"sources[{index}].shards[{shard_index}] historical overlay path "
                f"must equal {expected_overlay!r}"
            )
        overlay_role = _require_text(
            shard["historical_provenance_overlay_role"],
            field=(f"sources[{index}].shards[{shard_index}].historical_provenance_overlay_role"),
        )
        if overlay_role != HISTORICAL_OVERLAY_ROLE:
            raise EditingV2PackedCandidateMaterializationError(
                f"sources[{index}].shards[{shard_index}] historical overlay role "
                "must deny current training-provenance authority"
            )
        shards.append(
            {
                "relative_path": relative_path,
                "file_sha256": _require_sha256(
                    shard["file_sha256"],
                    field=f"sources[{index}].shards[{shard_index}].file_sha256",
                ),
                "manifest_relative_path": manifest_relative_path,
                "manifest_file_sha256": _require_sha256(
                    shard["manifest_file_sha256"],
                    field=(f"sources[{index}].shards[{shard_index}].manifest_file_sha256"),
                ),
                "historical_provenance_overlay_relative_path": (overlay_relative_path),
                "historical_provenance_overlay_file_sha256": _require_sha256(
                    shard["historical_provenance_overlay_file_sha256"],
                    field=(
                        f"sources[{index}].shards[{shard_index}]"
                        ".historical_provenance_overlay_file_sha256"
                    ),
                ),
                "historical_provenance_overlay_role": overlay_role,
            }
        )
    if shards != sorted(shards, key=lambda item: item["relative_path"]):
        raise EditingV2PackedCandidateMaterializationError(
            f"sources[{index}].shards must be sorted by relative_path"
        )
    if len({item["relative_path"] for item in shards}) != len(shards):
        raise EditingV2PackedCandidateMaterializationError(
            f"sources[{index}].shards contains duplicate physical paths"
        )
    return {
        "source_asset_id": _require_text(
            payload["source_asset_id"],
            field=f"sources[{index}].source_asset_id",
        ),
        "source_asset_path": _require_text(
            payload["source_asset_path"],
            field=f"sources[{index}].source_asset_path",
        ),
        "source_asset_sha256": _require_sha256(
            payload["source_asset_sha256"],
            field=f"sources[{index}].source_asset_sha256",
        ),
        "source_kind": source_kind,
        "allowed_layers": list(allowed_layers),
        "shards": shards,
    }


def build_packed_candidate_source_manifest(
    sources: Sequence[Mapping[str, Any]],
    *,
    upstream_overlay_completion: Mapping[str, Any],
    source_binding_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic, self-hashed packed-cache source manifest."""

    if isinstance(sources, (str, bytes)) or not sources:
        raise EditingV2PackedCandidateMaterializationError(
            "packed candidate sources must be a nonempty sequence"
        )
    normalized = [_normalized_source(source, index=index) for index, source in enumerate(sources)]
    ordered = sorted(
        normalized,
        key=lambda item: (item["source_kind"], item["source_asset_id"]),
    )
    if normalized != ordered:
        raise EditingV2PackedCandidateMaterializationError(
            "packed candidate sources must be deterministically sorted"
        )
    source_ids = [item["source_asset_id"] for item in normalized]
    if len(source_ids) != len(set(source_ids)):
        raise EditingV2PackedCandidateMaterializationError("source_asset_id must be unique")
    shard_paths = [shard["relative_path"] for source in normalized for shard in source["shards"]]
    if len(shard_paths) != len(set(shard_paths)):
        raise EditingV2PackedCandidateMaterializationError(
            "physical shard relative paths must be unique across sources"
        )
    completion_identity_base = _normalized_bound_identity(
        upstream_overlay_completion,
        fields=_OVERLAY_COMPLETION_IDENTITY_FIELDS,
        field="upstream_overlay_completion",
    )
    candidate_source_shards = _require_positive_int(
        upstream_overlay_completion["candidate_source_shards"],
        field="upstream_overlay_completion.candidate_source_shards",
    )
    candidate_source_inventory_sha256 = _require_sha256(
        upstream_overlay_completion["candidate_source_inventory_sha256"],
        field="upstream_overlay_completion.candidate_source_inventory_sha256",
    )
    source_inventory = _source_inventory(normalized)
    if candidate_source_shards != len(source_inventory) or (
        candidate_source_inventory_sha256 != canonical_sha256(source_inventory)
    ):
        raise EditingV2PackedCandidateMaterializationError(
            "packed candidate sources do not exactly match the upstream overlay completion inventory"
        )
    completion_identity = {
        **completion_identity_base,
        "candidate_source_shards": candidate_source_shards,
        "candidate_source_inventory_sha256": candidate_source_inventory_sha256,
    }
    registry_identity = _normalized_bound_identity(
        source_binding_registry,
        fields=_SOURCE_BINDING_REGISTRY_IDENTITY_FIELDS,
        field="source_binding_registry",
    )
    body = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "status": SOURCE_MANIFEST_STATUS,
        "training_authorized": False,
        "upstream_overlay_completion": completion_identity,
        "source_binding_registry": registry_identity,
        "sources": normalized,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def validate_packed_candidate_source_manifest(
    manifest: object,
) -> dict[str, Any]:
    payload = _require_exact_fields(
        manifest,
        _SOURCE_MANIFEST_FIELDS,
        field="packed candidate source manifest",
    )
    rebuilt = build_packed_candidate_source_manifest(
        payload["sources"],
        upstream_overlay_completion=payload["upstream_overlay_completion"],
        source_binding_registry=payload["source_binding_registry"],
    )
    if dict(payload) != rebuilt:
        raise EditingV2PackedCandidateMaterializationError(
            "packed candidate source manifest identity or self-hash disagrees"
        )
    return rebuilt


def load_packed_candidate_source_manifest(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingV2PackedCandidateMaterializationError(
            f"cannot load packed candidate source manifest {path}"
        ) from error
    return validate_packed_candidate_source_manifest(payload)


def write_packed_candidate_source_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> None:
    """Atomically write one validated source manifest without partial bytes."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            validate_packed_candidate_source_manifest(manifest),
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _absolute_normalized_posix_path(value: object, *, field: str) -> PurePosixPath:
    text = _require_text(value, field=field)
    path = PurePosixPath(text)
    if not path.is_absolute() or str(path) != text or ".." in path.parts:
        raise EditingV2PackedCandidateMaterializationError(
            f"{field} must be a normalized absolute POSIX path"
        )
    return path


def _load_overlay_completion(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> tuple[dict[str, Any], PurePosixPath, list[dict[str, Any]]]:
    completion_path = Path(path)
    expected_file_sha256 = _require_sha256(
        expected_file_sha256,
        field="expected overlay completion file SHA-256",
    )
    if not completion_path.is_file() or file_sha256(completion_path) != expected_file_sha256:
        raise EditingV2PackedCandidateMaterializationError(
            "upstream overlay completion is absent or its physical SHA-256 disagrees"
        )
    try:
        raw = json.loads(completion_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingV2PackedCandidateMaterializationError(
            "upstream overlay completion is invalid JSON"
        ) from error
    payload = _require_exact_fields(
        raw,
        _OVERLAY_COMPLETION_FIELDS,
        field="upstream overlay completion",
    )
    completion_body = {key: value for key, value in payload.items() if key != "completion_sha256"}
    semantic_sha256 = _require_sha256(
        payload["completion_sha256"],
        field="upstream overlay completion.completion_sha256",
    )
    if (
        payload["schema"] != UPSTREAM_OVERLAY_COMPLETION_SCHEMA
        or payload["schema_version"] != UPSTREAM_OVERLAY_COMPLETION_SCHEMA_VERSION
        or payload["status"] != UPSTREAM_OVERLAY_COMPLETION_STATUS
        or payload["training_authorized"] is not False
        or semantic_sha256 != canonical_sha256(completion_body)
    ):
        raise EditingV2PackedCandidateMaterializationError(
            "upstream overlay completion schema, authority, or self-hash disagrees"
        )
    if _COMMIT_RE.fullmatch(str(payload["commit"])) is None:
        raise EditingV2PackedCandidateMaterializationError(
            "upstream overlay completion.commit must be a full Git SHA"
        )
    _require_sha256(
        payload["launcher_source_sha256"],
        field="upstream overlay completion.launcher_source_sha256",
    )
    packed_root = _absolute_normalized_posix_path(
        payload["packed_root"], field="upstream overlay completion.packed_root"
    )
    mmp_root = _absolute_normalized_posix_path(
        payload["mmp_root"], field="upstream overlay completion.mmp_root"
    )
    common_root_text = os.path.commonpath((str(packed_root), str(mmp_root)))
    common_root = PurePosixPath(common_root_text)
    if common_root == PurePosixPath("/") or common_root in {packed_root, mmp_root}:
        raise EditingV2PackedCandidateMaterializationError(
            "upstream overlay completion roots require one non-root artifact namespace"
        )
    upstream = _require_exact_fields(
        payload["upstream_completions"],
        {"audit_pack", "mmp_pack"},
        field="upstream overlay completion.upstream_completions",
    )
    normalized_upstream: dict[str, dict[str, Any]] = {}
    for name in ("audit_pack", "mmp_pack"):
        identity = _require_exact_fields(
            upstream[name],
            _UPSTREAM_COMPLETION_FIELDS,
            field=f"upstream overlay completion.upstream_completions.{name}",
        )
        normalized_upstream[name] = {
            "path": str(
                _absolute_normalized_posix_path(
                    identity["path"],
                    field=f"upstream overlay completion.upstream_completions.{name}.path",
                )
            ),
            "file_sha256": _require_sha256(
                identity["file_sha256"],
                field=f"upstream overlay completion.upstream_completions.{name}.file_sha256",
            ),
            "completion_flag": _require_text(
                identity["completion_flag"],
                field=f"upstream overlay completion.upstream_completions.{name}.completion_flag",
            ),
            "inventory_origin": _require_text(
                identity["inventory_origin"],
                field=f"upstream overlay completion.upstream_completions.{name}.inventory_origin",
            ),
            "expected_shards": _require_positive_int(
                identity["expected_shards"],
                field=f"upstream overlay completion.upstream_completions.{name}.expected_shards",
            ),
            "shard_inventory_sha256": _require_sha256(
                identity["shard_inventory_sha256"],
                field=(
                    f"upstream overlay completion.upstream_completions.{name}"
                    ".shard_inventory_sha256"
                ),
            ),
        }
    expected_upstream = {
        "audit_pack": (packed_root / "PACK_COMPLETE.json", "PACK_COMPLETE"),
        "mmp_pack": (mmp_root / "MMP_PACK_COMPLETE.json", "MMP_PACK_COMPLETE"),
    }
    for name, (expected_path, expected_flag) in expected_upstream.items():
        if (
            normalized_upstream[name]["path"] != str(expected_path)
            or normalized_upstream[name]["completion_flag"] != expected_flag
        ):
            raise EditingV2PackedCandidateMaterializationError(
                f"upstream overlay completion.{name} address or completion flag disagrees"
            )
    if not isinstance(payload["fields_for_new_overlays"], Mapping):
        raise EditingV2PackedCandidateMaterializationError(
            "upstream overlay completion.fields_for_new_overlays must be an object"
        )
    _require_positive_int(
        payload["sentinel_entries_per_new_overlay"],
        field="upstream overlay completion.sentinel_entries_per_new_overlay",
    )
    raw_receipts = payload["shards"]
    if not isinstance(raw_receipts, list) or not raw_receipts:
        raise EditingV2PackedCandidateMaterializationError(
            "upstream overlay completion.shards must be nonempty"
        )
    normalized_receipts: list[dict[str, Any]] = []
    candidate_shards: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw_receipt in enumerate(raw_receipts):
        receipt = _require_exact_fields(
            raw_receipt,
            _OVERLAY_COMPLETION_RECEIPT_FIELDS,
            field=f"upstream overlay completion.shards[{index}]",
        )
        layer = _require_text(receipt["layer"], field=f"completion shard {index}.layer")
        partition = _require_text(receipt["partition"], field=f"completion shard {index}.partition")
        if layer not in _COMPLETION_LAYERS or partition not in _COMPLETION_PARTITIONS:
            raise EditingV2PackedCandidateMaterializationError(
                f"upstream overlay completion.shards[{index}] cell is unsupported"
            )
        shard_path = _absolute_normalized_posix_path(
            receipt["shard_path"], field=f"completion shard {index}.shard_path"
        )
        shard_name = shard_path.name
        if not shard_name.endswith(".jsonl.gz"):
            raise EditingV2PackedCandidateMaterializationError(
                f"upstream overlay completion.shards[{index}] is not a packed shard"
            )
        expected_shard_path = (
            mmp_root / partition / shard_name
            if layer == "mmp_analogue"
            else packed_root / layer / partition / shard_name
        )
        manifest_path = _absolute_normalized_posix_path(
            receipt["manifest_path"], field=f"completion shard {index}.manifest_path"
        )
        overlay_path = _absolute_normalized_posix_path(
            receipt["overlay_path"], field=f"completion shard {index}.overlay_path"
        )
        if (
            shard_path != expected_shard_path
            or manifest_path != PurePosixPath(str(shard_path)).with_suffix(".manifest.json")
            or overlay_path != PurePosixPath(f"{shard_path}.provenance.json")
        ):
            raise EditingV2PackedCandidateMaterializationError(
                f"upstream overlay completion.shards[{index}] escapes its exact source cell"
            )
        if str(shard_path) in seen_paths:
            raise EditingV2PackedCandidateMaterializationError(
                "upstream overlay completion repeats a physical shard"
            )
        seen_paths.add(str(shard_path))
        role = _require_text(
            receipt["overlay_validation_role"],
            field=f"completion shard {index}.overlay_validation_role",
        )
        if role not in _OVERLAY_VALIDATION_ROLES:
            raise EditingV2PackedCandidateMaterializationError(
                f"upstream overlay completion.shards[{index}] validation role is unsupported"
            )
        packed_sha256 = _require_sha256(
            receipt["shard_file_sha256"], field=f"completion shard {index}.shard_file_sha256"
        )
        manifest_sha256 = _require_sha256(
            receipt["manifest_file_sha256"],
            field=f"completion shard {index}.manifest_file_sha256",
        )
        overlay_file_sha256 = _require_sha256(
            receipt["overlay_file_sha256"],
            field=f"completion shard {index}.overlay_file_sha256",
        )
        _require_sha256(
            receipt["overlay_fields_sha256"],
            field=f"completion shard {index}.overlay_fields_sha256",
        )
        _require_sha256(
            receipt["overlay_semantic_sha256"],
            field=f"completion shard {index}.overlay_semantic_sha256",
        )
        entries = _require_positive_int(
            receipt["entries"], field=f"completion shard {index}.entries"
        )
        states = _require_positive_int(receipt["states"], field=f"completion shard {index}.states")
        normalized_receipts.append(
            {
                "layer": layer,
                "partition": partition,
                "shard": shard_name,
                "packed_sha256": packed_sha256,
                "manifest_sha256": manifest_sha256,
                "entries": entries,
                "states": states,
            }
        )
        candidate_shards.append(
            {
                "layer": layer,
                "relative_path": str(shard_path.relative_to(common_root)),
                "file_sha256": packed_sha256,
                "manifest_relative_path": str(manifest_path.relative_to(common_root)),
                "manifest_file_sha256": manifest_sha256,
                "historical_provenance_overlay_relative_path": str(
                    overlay_path.relative_to(common_root)
                ),
                "historical_provenance_overlay_file_sha256": overlay_file_sha256,
                "historical_provenance_overlay_role": HISTORICAL_OVERLAY_ROLE,
            }
        )
    expected_total = sum(identity["expected_shards"] for identity in normalized_upstream.values())
    if len(normalized_receipts) != expected_total:
        raise EditingV2PackedCandidateMaterializationError(
            "upstream overlay completion shard count disagrees with upstream inventories"
        )
    observed_cells = {(item["layer"], item["partition"]) for item in normalized_receipts}
    expected_cells = {
        (layer, partition) for layer in _COMPLETION_LAYERS for partition in _COMPLETION_PARTITIONS
    }
    if observed_cells != expected_cells:
        raise EditingV2PackedCandidateMaterializationError(
            "upstream overlay completion does not cover every required layer/partition cell"
        )
    for name, layers in (
        ("audit_pack", _SYNTHETIC_LAYERS),
        ("mmp_pack", _REAL_ENDPOINT_LAYERS),
    ):
        inventory = sorted(
            (item for item in normalized_receipts if item["layer"] in layers),
            key=lambda item: (item["layer"], item["partition"], item["shard"]),
        )
        if len(inventory) != normalized_upstream[name]["expected_shards"] or (
            canonical_sha256(inventory) != normalized_upstream[name]["shard_inventory_sha256"]
        ):
            raise EditingV2PackedCandidateMaterializationError(
                f"upstream overlay completion {name} inventory identity disagrees"
            )
    candidate_shards.sort(key=lambda item: item["relative_path"])
    return dict(payload), common_root, candidate_shards


def build_packed_candidate_source_manifest_from_overlay_completion(
    *,
    overlay_completion_path: str | Path,
    expected_overlay_completion_file_sha256: str,
    source_binding_registry_path: str | Path,
    expected_source_binding_registry_file_sha256: str,
) -> dict[str, Any]:
    """Consume one exact completion and explicit source registry without inference."""

    completion, _, candidate_shards = _load_overlay_completion(
        overlay_completion_path,
        expected_file_sha256=expected_overlay_completion_file_sha256,
    )
    registry_path = Path(source_binding_registry_path)
    expected_registry_sha256 = _require_sha256(
        expected_source_binding_registry_file_sha256,
        field="expected source binding registry file SHA-256",
    )
    if not registry_path.is_file() or file_sha256(registry_path) != expected_registry_sha256:
        raise EditingV2PackedCandidateMaterializationError(
            "source binding registry is absent or its physical SHA-256 disagrees"
        )
    registry = load_overlay_source_binding_registry(registry_path)
    owner_by_layer = {
        layer: binding for binding in registry["bindings"] for layer in binding["allowed_layers"]
    }
    shards_by_source: dict[str, list[dict[str, Any]]] = {
        binding["source_asset_id"]: [] for binding in registry["bindings"]
    }
    for candidate_shard in candidate_shards:
        layer = candidate_shard["layer"]
        owner = owner_by_layer[layer]
        shards_by_source[owner["source_asset_id"]].append(
            {key: value for key, value in candidate_shard.items() if key != "layer"}
        )
    sources = [
        {
            **binding,
            "shards": sorted(
                shards_by_source[binding["source_asset_id"]],
                key=lambda item: item["relative_path"],
            ),
        }
        for binding in registry["bindings"]
    ]
    source_inventory = _source_inventory(sources)
    completion_identity = {
        "artifact_path": str(Path(overlay_completion_path)),
        "file_sha256": file_sha256(Path(overlay_completion_path)),
        "semantic_sha256": completion["completion_sha256"],
        "candidate_source_shards": len(source_inventory),
        "candidate_source_inventory_sha256": canonical_sha256(source_inventory),
    }
    registry_identity = {
        "artifact_path": str(registry_path),
        "file_sha256": file_sha256(registry_path),
        "semantic_sha256": registry["registry_sha256"],
    }
    return build_packed_candidate_source_manifest(
        sources,
        upstream_overlay_completion=completion_identity,
        source_binding_registry=registry_identity,
    )


def _load_packed_manifest(
    path: Path,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise EditingV2PackedCandidateMaterializationError(
            f"packed shard manifest is absent: {path}"
        )
    observed_sha256 = file_sha256(path)
    if observed_sha256 != expected_file_sha256:
        raise EditingV2PackedCandidateMaterializationError(
            f"packed shard manifest SHA-256 mismatch for {path}: "
            f"expected {expected_file_sha256}, observed {observed_sha256}"
        )
    try:
        payload = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingV2PackedCandidateMaterializationError(
            f"packed shard manifest is invalid JSON: {path}"
        ) from error
    manifest = _require_exact_fields(
        payload,
        _PACKED_MANIFEST_FIELDS,
        field=f"packed shard manifest {path}",
    )
    if (
        manifest["schema"] != PACKED_STORE_SCHEMA
        or manifest["schema_version"] != PACKED_STORE_SCHEMA_VERSION
    ):
        raise EditingV2PackedCandidateMaterializationError(
            f"packed shard manifest schema is unsupported: {path}"
        )
    _require_nonnegative_int(manifest["entries"], field=f"{path}.entries")
    _require_nonnegative_int(manifest["states"], field=f"{path}.states")
    if not isinstance(manifest["provenance"], Mapping):
        raise EditingV2PackedCandidateMaterializationError(f"{path}.provenance must be an object")
    return dict(manifest)


def _load_historical_overlay_receipt(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_shard_name: str,
    expected_packed_shard_sha256: str,
    expected_original_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify a frozen legacy sidecar as a byte receipt, never as live provenance.

    The production ``provenance_overlay.load_overlay`` function intentionally
    compares the sidecar's recorded implementation hash with the current source
    code.  That is correct for loading a cache under the old scientific
    training contract, but it is not the right migration rule here: these
    sidecars describe historical cache bytes and are expected to carry old
    implementation and tensorization hashes.  This parser therefore verifies
    the sidecar's own bytes and its immutable shard/manifest bindings while
    recording, rather than endorsing, its historical semantic claims.
    """

    if not path.is_file():
        raise EditingV2PackedCandidateMaterializationError(
            f"historical packed provenance overlay is absent: {path}"
        )
    observed_file_sha256 = file_sha256(path)
    if observed_file_sha256 != expected_file_sha256:
        raise EditingV2PackedCandidateMaterializationError(
            f"historical packed provenance overlay SHA-256 mismatch for {path}: "
            f"expected {expected_file_sha256}, observed {observed_file_sha256}"
        )
    try:
        payload = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingV2PackedCandidateMaterializationError(
            f"historical packed provenance overlay is invalid JSON: {path}"
        ) from error
    overlay = _require_exact_fields(
        payload,
        _HISTORICAL_OVERLAY_FIELDS,
        field=f"historical packed provenance overlay {path}",
    )
    if (
        overlay["schema"] != HISTORICAL_OVERLAY_SCHEMA
        or overlay["schema_version"] != HISTORICAL_OVERLAY_SCHEMA_VERSION
    ):
        raise EditingV2PackedCandidateMaterializationError(
            f"historical packed provenance overlay schema is unsupported: {path}"
        )
    if overlay["shard"] != expected_shard_name:
        raise EditingV2PackedCandidateMaterializationError(
            f"historical packed provenance overlay names a different shard: {path}"
        )
    overlay_shard_sha256 = _require_sha256(
        overlay["packed_shard_content_sha256"],
        field=f"{path}.packed_shard_content_sha256",
    )
    if overlay_shard_sha256 != expected_packed_shard_sha256:
        raise EditingV2PackedCandidateMaterializationError(
            f"historical packed provenance overlay does not bind the observed shard: {path}"
        )
    overlay_manifest_sha256 = _require_sha256(
        overlay["original_manifest_sha256"],
        field=f"{path}.original_manifest_sha256",
    )
    if overlay_manifest_sha256 != expected_original_manifest_sha256:
        raise EditingV2PackedCandidateMaterializationError(
            f"historical packed provenance overlay does not bind the observed "
            f"original manifest: {path}"
        )
    fields = overlay["fields"]
    if not isinstance(fields, Mapping):
        raise EditingV2PackedCandidateMaterializationError(f"{path}.fields must be an object")
    required_recorded_fields = {
        "codec_implementation_hash",
        "trace_schema_version",
        "packed_store_schema_version",
        "tensorization_implementation_hash",
    }
    missing_recorded_fields = required_recorded_fields - set(fields)
    if missing_recorded_fields:
        raise EditingV2PackedCandidateMaterializationError(
            f"historical packed provenance overlay lacks its original scientific "
            f"receipt fields: {sorted(missing_recorded_fields)}"
        )
    recorded_tensorization_hash = fields["tensorization_implementation_hash"]
    if (
        not isinstance(recorded_tensorization_hash, str)
        or _SHORT_IMPLEMENTATION_HASH_RE.fullmatch(recorded_tensorization_hash) is None
    ):
        raise EditingV2PackedCandidateMaterializationError(
            f"{path}.fields.tensorization_implementation_hash must be a "
            "historical 16-character lowercase hash"
        )
    recorded_codec_hash = fields["codec_implementation_hash"]
    if (
        not isinstance(recorded_codec_hash, str)
        or _SHORT_IMPLEMENTATION_HASH_RE.fullmatch(recorded_codec_hash) is None
    ):
        raise EditingV2PackedCandidateMaterializationError(
            f"{path}.fields.codec_implementation_hash must be a historical "
            "16-character lowercase hash"
        )
    recorded_trace_schema_version = _require_positive_int(
        fields["trace_schema_version"],
        field=f"{path}.fields.trace_schema_version",
    )
    recorded_packed_store_schema_version = _require_positive_int(
        fields["packed_store_schema_version"],
        field=f"{path}.fields.packed_store_schema_version",
    )
    recorded_upgrade_hash = overlay["upgrade_implementation_hash"]
    if (
        not isinstance(recorded_upgrade_hash, str)
        or _SHORT_IMPLEMENTATION_HASH_RE.fullmatch(recorded_upgrade_hash) is None
    ):
        raise EditingV2PackedCandidateMaterializationError(
            f"{path}.upgrade_implementation_hash must be a historical 16-character lowercase hash"
        )
    packer_commit = _require_text(
        overlay["packer_commit"],
        field=f"{path}.packer_commit",
    )
    certification = overlay["certification"]
    if not isinstance(certification, Mapping):
        raise EditingV2PackedCandidateMaterializationError(
            f"{path}.certification must be an object"
        )
    # Prove every retained historical field is finite deterministic JSON.  The
    # semantic hash binds fields not copied into the compact summary.
    semantic_sha256 = canonical_sha256(dict(overlay))
    certification_sha256 = canonical_sha256(dict(certification))
    return {
        "role": HISTORICAL_OVERLAY_ROLE,
        "training_provenance_authorized": False,
        "live_overlay_semantic_validation_applied": False,
        "file_sha256": observed_file_sha256,
        "semantic_sha256": semantic_sha256,
        "schema": HISTORICAL_OVERLAY_SCHEMA,
        "schema_version": HISTORICAL_OVERLAY_SCHEMA_VERSION,
        "bindings": {
            "packed_shard_content_sha256": overlay_shard_sha256,
            "original_manifest_sha256": overlay_manifest_sha256,
        },
        "recorded_upgrade_implementation_hash": recorded_upgrade_hash,
        "recorded_tensorization_implementation_hash": (recorded_tensorization_hash),
        "recorded_codec_implementation_hash": recorded_codec_hash,
        "recorded_trace_schema_version": recorded_trace_schema_version,
        "recorded_packed_store_schema_version": (recorded_packed_store_schema_version),
        "packer_commit": packer_commit,
        "certification_sha256": certification_sha256,
    }


def validate_historical_provenance_overlay(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_shard_name: str,
    expected_packed_shard_sha256: str,
    expected_original_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate an immutable legacy sidecar strictly as a historical byte receipt."""

    return _load_historical_overlay_receipt(
        Path(path),
        expected_file_sha256=expected_file_sha256,
        expected_shard_name=expected_shard_name,
        expected_packed_shard_sha256=expected_packed_shard_sha256,
        expected_original_manifest_sha256=expected_original_manifest_sha256,
    )


def _partition_groups(trace: Mapping[str, Any], *, address: str) -> dict[str, Any]:
    metadata = trace.get("metadata")
    isolation = metadata.get("partition_isolation") if isinstance(metadata, Mapping) else None
    payload = _require_exact_fields(
        isolation,
        _PARTITION_ISOLATION_FIELDS,
        field=f"{address}.metadata.partition_isolation",
    )
    if (
        payload["schema"] != PARTITION_ISOLATION_SCHEMA
        or payload["schema_version"] != PARTITION_ISOLATION_SCHEMA_VERSION
    ):
        raise EditingV2PackedCandidateMaterializationError(
            f"{address} has an unsupported partition-isolation schema"
        )
    molecule_ids = _normalize_unique_text_collection(
        payload["molecule_ids"],
        field=f"{address}.partition_isolation.molecule_ids",
    )
    scaffold_ids = _normalize_unique_text_collection(
        payload["scaffold_ids"],
        field=f"{address}.partition_isolation.scaffold_ids",
    )
    source_group_id = _require_text(
        payload["source_group_id"],
        field=f"{address}.partition_isolation.source_group_id",
    )
    return {
        "molecule_ids": list(molecule_ids),
        "scaffold_ids": list(scaffold_ids),
        "source_group_ids": [source_group_id],
    }


def _validate_encoded_state(
    state: object,
    *,
    address: str,
    progress_index: int,
    expected_n_slots: int,
) -> bytes:
    payload = _require_exact_fields(
        state,
        _ENCODED_STATE_FIELDS,
        field=f"{address}.states[{progress_index}]",
    )
    n_slots = _require_positive_int(
        payload["n_slots"],
        field=f"{address}.states[{progress_index}].n_slots",
    )
    if n_slots != expected_n_slots:
        raise EditingV2PackedCandidateMaterializationError(
            f"{address}.states[{progress_index}] n_slots disagrees with trace envelope"
        )
    for field in ("atom_types", "formal_charges", "implicit_h_counts"):
        values = payload[field]
        if (
            not isinstance(values, list)
            or len(values) != n_slots
            or any(type(value) is not int for value in values)
        ):
            raise EditingV2PackedCandidateMaterializationError(
                f"{address}.states[{progress_index}].{field} must contain "
                f"exactly n_slots integer values"
            )
    bonds = payload["bonds"]
    if not isinstance(bonds, list):
        raise EditingV2PackedCandidateMaterializationError(
            f"{address}.states[{progress_index}].bonds must be a list"
        )
    for bond_index, bond in enumerate(bonds):
        if (
            not isinstance(bond, list)
            or len(bond) != 3
            or any(type(value) is not int for value in bond)
        ):
            raise EditingV2PackedCandidateMaterializationError(
                f"{address}.states[{progress_index}].bonds[{bond_index}] must be an integer triple"
            )
        left, right, order = bond
        if not (0 <= left < right < n_slots) or order <= 0:
            raise EditingV2PackedCandidateMaterializationError(
                f"{address}.states[{progress_index}].bonds[{bond_index}] "
                "has an invalid slot address or order"
            )
    return canonical_json_bytes(payload)


def _exact_state_identity(
    entry: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    address: str,
    packed_shard_sha256: str,
    entry_index: int,
    path_length: int,
) -> dict[str, Any]:
    states = entry.get("states")
    if not isinstance(states, list) or len(states) != path_length + 1:
        raise EditingV2PackedCandidateMaterializationError(
            f"{address}.states must contain path_length + 1 exact states"
        )
    n_slots = _require_positive_int(trace.get("n_slots"), field=f"{address}.n_slots")
    state_stream = hashlib.sha256()
    progress_addresses = hashlib.sha256()
    first_state_bytes: bytes | None = None
    for progress_index, state in enumerate(states):
        encoded = _validate_encoded_state(
            state,
            address=address,
            progress_index=progress_index,
            expected_n_slots=n_slots,
        )
        if progress_index == 0:
            first_state_bytes = encoded
        state_sha256 = hashlib.sha256(encoded).hexdigest()
        state_stream.update(encoded)
        state_stream.update(b"\n")
        progress_address = {
            "packed_shard_file_sha256": packed_shard_sha256,
            "entry_index": entry_index,
            "progress_index": progress_index,
            "encoded_state_sha256": state_sha256,
        }
        progress_addresses.update(canonical_json_bytes(progress_address))
        progress_addresses.update(b"\n")
    source_state = trace.get("source_state")
    if first_state_bytes is None or canonical_json_bytes(source_state) != first_state_bytes:
        raise EditingV2PackedCandidateMaterializationError(
            f"{address}.trace.source_state disagrees with states[0]"
        )
    return {
        "encoding": "compose.rewrite.trace.encoded_state_v2",
        "n_slots": n_slots,
        "state_count": len(states),
        "encoded_state_stream_sha256": state_stream.hexdigest(),
        "progress_address_stream_sha256": progress_addresses.hexdigest(),
    }


def _integer_histogram(value: object, *, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise EditingV2PackedCandidateMaterializationError(f"{field} must be an object")
    normalized: dict[str, int] = {}
    for key, count in value.items():
        name = _require_text(key, field=f"{field}.key")
        number = _require_positive_int(count, field=f"{field}.{name}")
        normalized[name] = number
    return dict(sorted(normalized.items()))


def _operator_summary(
    trace: Mapping[str, Any],
    *,
    address: str,
    path_length: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    steps = trace.get("steps")
    if not isinstance(steps, list) or len(steps) != path_length:
        raise EditingV2PackedCandidateMaterializationError(
            f"{address}.steps must contain exactly path_length entries"
        )
    family_histogram: Counter[str] = Counter()
    executor_histogram: Counter[str] = Counter()
    atom_delta_sum = 0
    cycle_delta_sum = 0
    action_stream = hashlib.sha256()
    step_delta_stream = hashlib.sha256()
    rejections: list[dict[str, Any]] = []
    for step_index, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping):
            raise EditingV2PackedCandidateMaterializationError(
                f"{address}.steps[{step_index}] must be an object"
            )
        action = _require_exact_fields(
            raw_step.get("action"),
            _ACTION_FIELDS,
            field=f"{address}.steps[{step_index}].action",
        )
        if action["schema"] != ACTION_SCHEMA or action["schema_version"] != ACTION_SCHEMA_VERSION:
            raise EditingV2PackedCandidateMaterializationError(
                f"{address}.steps[{step_index}] has an unsupported action schema"
            )
        executor_rule = _require_text(
            action["executor_rule"],
            field=f"{address}.steps[{step_index}].action.executor_rule",
        )
        family = _require_text(
            action["model_family"],
            field=f"{address}.steps[{step_index}].action.model_family",
        )
        _require_text(
            action["payload_type"],
            field=f"{address}.steps[{step_index}].action.payload_type",
        )
        if not isinstance(action["payload"], Mapping):
            raise EditingV2PackedCandidateMaterializationError(
                f"{address}.steps[{step_index}].action.payload must be an object"
            )
        family_histogram[family] += 1
        executor_histogram[executor_rule] += 1
        atom_delta = raw_step.get("atom_count_delta")
        cycle_delta = raw_step.get("cycle_rank_delta")
        if type(atom_delta) is not int or type(cycle_delta) is not int:
            raise EditingV2PackedCandidateMaterializationError(
                f"{address}.steps[{step_index}] deltas must be integers"
            )
        atom_delta_sum += atom_delta
        cycle_delta_sum += cycle_delta
        step_identity = {
            "action": dict(action),
            "successor_key": _require_text(
                raw_step.get("successor_key"),
                field=f"{address}.steps[{step_index}].successor_key",
            ),
            "atom_count_delta": atom_delta,
            "graph_cycle_rank_delta": cycle_delta,
        }
        action_stream.update(canonical_json_bytes(step_identity))
        action_stream.update(b"\n")
        step_delta_stream.update(
            canonical_json_bytes(
                {
                    "step_index": step_index,
                    "atom_count_delta": atom_delta,
                    "graph_cycle_rank_delta": cycle_delta,
                }
            )
        )
        step_delta_stream.update(b"\n")
        expected_family = _EXECUTOR_TO_FAMILY.get(executor_rule)
        if expected_family is None:
            rejections.append(
                {
                    "step_index": step_index,
                    "executor_rule": executor_rule,
                    "family": family,
                    "reason": "unknown_executor_rule",
                }
            )
        elif family != expected_family:
            rejections.append(
                {
                    "step_index": step_index,
                    "executor_rule": executor_rule,
                    "family": family,
                    "reason": "executor_family_ontology_mismatch",
                }
            )
        elif executor_rule in _DISABLED_SET or family in _DISABLED_SET:
            rejections.append(
                {
                    "step_index": step_index,
                    "executor_rule": executor_rule,
                    "family": family,
                    "reason": "disabled_family",
                }
            )
        elif family not in _ACTIVE8_SET:
            rejections.append(
                {
                    "step_index": step_index,
                    "executor_rule": executor_rule,
                    "family": family,
                    "reason": "outside_active8",
                }
            )
        elif executor_rule == "atom_insert":
            neighbors = action["payload"].get("neighbors")
            if not isinstance(neighbors, list):
                raise EditingV2PackedCandidateMaterializationError(
                    f"{address}.steps[{step_index}] atom_insert neighbors must be a list"
                )
            if len(neighbors) > 1:
                rejections.append(
                    {
                        "step_index": step_index,
                        "executor_rule": executor_rule,
                        "family": family,
                        "reason": "unsupported_multi_neighbor_atom_insert",
                    }
                )
    declared_histogram = _integer_histogram(
        trace.get("family_histogram"),
        field=f"{address}.family_histogram",
    )
    observed_histogram = dict(sorted(family_histogram.items()))
    if observed_histogram != declared_histogram:
        raise EditingV2PackedCandidateMaterializationError(
            f"{address}.family_histogram disagrees with encoded steps"
        )
    declared_executor_histogram = _integer_histogram(
        trace.get("operator_histogram"),
        field=f"{address}.operator_histogram",
    )
    observed_executor_histogram = dict(sorted(executor_histogram.items()))
    if observed_executor_histogram != declared_executor_histogram:
        raise EditingV2PackedCandidateMaterializationError(
            f"{address}.operator_histogram disagrees with encoded steps"
        )
    declared_atom_delta = trace.get("atom_count_delta")
    declared_cycle_delta = trace.get("cycle_rank_delta")
    if (
        type(declared_atom_delta) is not int
        or type(declared_cycle_delta) is not int
        or declared_atom_delta != atom_delta_sum
        or declared_cycle_delta != cycle_delta_sum
    ):
        raise EditingV2PackedCandidateMaterializationError(
            f"{address} trace deltas disagree with encoded step deltas"
        )
    return (
        {
            "operator_families": sorted(family_histogram),
            "family_histogram": observed_histogram,
            "executor_histogram": observed_executor_histogram,
            "action_stream_sha256": action_stream.hexdigest(),
            "atom_count_delta": atom_delta_sum,
            "graph_cycle_rank_delta": cycle_delta_sum,
            "step_delta_stream_sha256": step_delta_stream.hexdigest(),
        },
        sorted(
            rejections,
            key=lambda item: (item["step_index"], item["reason"], item["family"]),
        ),
    )


def _evidence_components(
    contract: Mapping[str, Any],
    *,
    profile_id: str,
) -> dict[str, str]:
    for profile in contract["evidence_component_contract"]["profiles"]:
        if profile["id"] == profile_id:
            if profile["admission_status"] != "admissible":
                raise EditingV2PackedCandidateMaterializationError(
                    f"evidence profile {profile_id!r} is not admissible"
                )
            return dict(profile["components"])
    raise EditingV2PackedCandidateMaterializationError(
        f"evidence profile {profile_id!r} is absent from the corpus contract"
    )


def _profile_for_source_kind(source_kind: str) -> str:
    return (
        "inferred_relation_compiled_path"
        if source_kind == INFERRED_REAL_ENDPOINT_PAIR
        else "executor_generated_walk"
    )


def _candidate_header(
    entry: object,
    *,
    attempt_index: int,
    source: Mapping[str, Any],
    shard_summary: Mapping[str, Any],
    entry_index: int,
    routing_policy: Mapping[str, Any],
    corpus_contract: Mapping[str, Any],
    implementation_sha256: str,
) -> dict[str, Any]:
    if not isinstance(entry, Mapping) or set(entry) != {"trace", "states"}:
        raise EditingV2PackedCandidateMaterializationError(
            f"{shard_summary['relative_path']} entry {entry_index} must contain "
            "exactly trace and states"
        )
    trace = entry["trace"]
    if not isinstance(trace, Mapping):
        raise EditingV2PackedCandidateMaterializationError(
            f"{shard_summary['relative_path']} entry {entry_index}.trace must be an object"
        )
    address_label = f"{shard_summary['relative_path']} entry {entry_index}"
    if trace.get("schema") != TRACE_SCHEMA or trace.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise EditingV2PackedCandidateMaterializationError(
            f"{address_label} has an unsupported trace schema"
        )
    trace_id = _require_text(trace.get("trace_id"), field=f"{address_label}.trace_id")
    layer = _require_text(trace.get("layer"), field=f"{address_label}.layer")
    if layer not in source["allowed_layers"]:
        raise EditingV2PackedCandidateMaterializationError(
            f"{address_label}.layer is outside its source manifest allowance"
        )
    cache_partition = _require_text(
        trace.get("partition"),
        field=f"{address_label}.partition",
    )
    source_key = _require_text(
        trace.get("source_key"),
        field=f"{address_label}.source_key",
    )
    target_key = _require_text(
        trace.get("target_key"),
        field=f"{address_label}.target_key",
    )
    path_length = _require_positive_int(
        trace.get("path_length"),
        field=f"{address_label}.path_length",
    )
    exact_states = _exact_state_identity(
        entry,
        trace,
        address=address_label,
        packed_shard_sha256=shard_summary["file_sha256"],
        entry_index=entry_index,
        path_length=path_length,
    )
    operator_summary, operator_rejections = _operator_summary(
        trace,
        address=address_label,
        path_length=path_length,
    )
    groups = _partition_groups(trace, address=address_label)
    address_body = {
        "source_asset_sha256": source["source_asset_sha256"],
        "packed_shard_file_sha256": shard_summary["file_sha256"],
        "historical_provenance_overlay_file_sha256": shard_summary["historical_provenance_overlay"][
            "file_sha256"
        ],
        "entry_index": entry_index,
        "trace_id": trace_id,
        "layer": layer,
        "cache_partition": cache_partition,
        "path_length": path_length,
        "state_count": exact_states["state_count"],
        "encoded_state_stream_sha256": exact_states["encoded_state_stream_sha256"],
    }
    address_sha256 = canonical_sha256(address_body)
    candidate_id = f"packed-{address_sha256}"
    source_kind = source["source_kind"]
    profile_id = _profile_for_source_kind(source_kind)
    evidence_components = _evidence_components(
        corpus_contract,
        profile_id=profile_id,
    )
    evidence_identity = {
        "evidence_profile_id": profile_id,
        "components": evidence_components,
        "source_asset_sha256": source["source_asset_sha256"],
        "address_sha256": address_sha256,
    }
    metadata = trace.get("metadata")
    if not isinstance(metadata, Mapping):
        raise EditingV2PackedCandidateMaterializationError(
            f"{address_label}.metadata must be an object"
        )
    compiler_path_class: str | None = None
    if source_kind == INFERRED_REAL_ENDPOINT_PAIR:
        compiler_path_class = _require_text(
            metadata.get("compiler_path_class"),
            field=f"{address_label}.metadata.compiler_path_class",
        )

    if operator_rejections:
        data_lane = None
        lane_resolution = None
        disposition = "rejected_candidate"
        rejection = {
            "code": "operator.not_in_active8_support",
            "attempts": operator_rejections,
        }
    else:
        route = route_editing_v2_candidate(
            source_kind=source_kind,
            compiler_path_class=compiler_path_class,
            path_length=path_length,
            operator_families=operator_summary["operator_families"],
            policy=dict(routing_policy),
        )
        route_payload = route_receipt_payload(route)
        validate_evidence_assignment_envelope(
            corpus_contract,
            data_lane=route.data_lane,
            evidence_profile_id=route.evidence_profile_id,
            evidence_components=evidence_components,
        )
        receipt_body = {
            "candidate_id": candidate_id,
            "address_sha256": address_sha256,
            "materializer_implementation_sha256": implementation_sha256,
            "route": route_payload,
        }
        data_lane = route.data_lane
        lane_resolution = {
            **route_payload,
            "resolver_identity_sha256": implementation_sha256,
            "receipt_sha256": canonical_sha256(receipt_body),
        }
        disposition = "routed_candidate"
        rejection = None

    trace_envelope_sha256 = canonical_sha256(trace)
    candidate_payload_sha256 = canonical_sha256(
        {
            "trace_envelope_sha256": trace_envelope_sha256,
            "encoded_state_stream_sha256": exact_states["encoded_state_stream_sha256"],
        }
    )
    body = {
        "schema": CANDIDATE_HEADER_SCHEMA,
        "schema_version": CANDIDATE_HEADER_SCHEMA_VERSION,
        "status": CANDIDATE_HEADER_STATUS,
        "training_authorized": False,
        "attempt_index": attempt_index,
        "candidate_id": candidate_id,
        "candidate_payload_sha256": candidate_payload_sha256,
        "disposition": disposition,
        "rejection": rejection,
        "source_kind": source_kind,
        "source_asset": {
            "source_asset_id": source["source_asset_id"],
            "source_asset_path": source["source_asset_path"],
            "source_asset_sha256": source["source_asset_sha256"],
        },
        "packed_address": {
            "relative_path": shard_summary["relative_path"],
            "packed_shard_file_sha256": shard_summary["file_sha256"],
            "packed_shard_manifest_file_sha256": shard_summary["manifest_file_sha256"],
            "historical_provenance_overlay_file_sha256": shard_summary[
                "historical_provenance_overlay"
            ]["file_sha256"],
            "entry_index": entry_index,
            "trace_id": trace_id,
            "layer": layer,
            "cache_partition": cache_partition,
            "path_length": path_length,
            "address_sha256": address_sha256,
        },
        "endpoint_identity": {
            "source_key": source_key,
            "target_key": target_key,
        },
        "groups": groups,
        "compiler_path_class": compiler_path_class,
        "data_lane": data_lane,
        "lane_resolution": lane_resolution,
        "routing_policy_sha256": routing_policy_sha256(dict(routing_policy)),
        "evidence_profile_id": profile_id,
        "evidence_components": evidence_components,
        "evidence_sha256": canonical_sha256(evidence_identity),
        "operator_summary": operator_summary,
        "exact_states": exact_states,
        "trace_envelope_sha256": trace_envelope_sha256,
    }
    return {**body, "row_sha256": canonical_sha256(body)}


def _iter_bounded_gzip_jsonl(
    path: Path,
    *,
    max_entry_bytes: int,
) -> Iterable[tuple[int, dict[str, Any]]]:
    if type(max_entry_bytes) is not int or max_entry_bytes <= 0:
        raise EditingV2PackedCandidateMaterializationError(
            "max_entry_bytes must be a positive integer"
        )
    entry_index = 0
    try:
        with gzip.open(path, "rb") as handle:
            while True:
                raw_line = handle.readline(max_entry_bytes + 1)
                if not raw_line:
                    break
                if len(raw_line) > max_entry_bytes or not raw_line.endswith(b"\n"):
                    raise EditingV2PackedCandidateMaterializationError(
                        f"packed entry exceeds {max_entry_bytes} bytes or lacks a terminating newline: "
                        f"{path} physical line after entry {entry_index - 1}"
                    )
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise EditingV2PackedCandidateMaterializationError(
                        f"packed shard contains invalid JSON at {path} entry {entry_index}"
                    ) from error
                if not isinstance(payload, dict):
                    raise EditingV2PackedCandidateMaterializationError(
                        f"packed shard entry must be an object: {path} entry {entry_index}"
                    )
                yield entry_index, payload
                entry_index += 1
    except EditingV2PackedCandidateMaterializationError:
        raise
    except (EOFError, OSError) as error:
        raise EditingV2PackedCandidateMaterializationError(
            f"packed shard is not a complete readable gzip stream: {path}"
        ) from error


def _counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _empty_totals() -> dict[str, Any]:
    return {
        "rows": 0,
        "routed_rows": 0,
        "rejected_rows": 0,
        "states": 0,
        "transitions": 0,
        "atom_count_delta_sum": 0,
        "graph_cycle_rank_delta_sum": 0,
        "family_histogram": Counter(),
        "routed_family_histogram": Counter(),
        "rejected_family_histogram": Counter(),
        "lane_histogram": Counter(),
        "rejection_reason_histogram": Counter(),
    }


def _accumulate_totals(totals: dict[str, Any], row: Mapping[str, Any]) -> None:
    totals["rows"] += 1
    state_count = int(row["exact_states"]["state_count"])
    path_length = int(row["packed_address"]["path_length"])
    totals["states"] += state_count
    totals["transitions"] += path_length
    operator = row["operator_summary"]
    totals["atom_count_delta_sum"] += int(operator["atom_count_delta"])
    totals["graph_cycle_rank_delta_sum"] += int(operator["graph_cycle_rank_delta"])
    totals["family_histogram"].update(operator["family_histogram"])
    if row["disposition"] == "routed_candidate":
        totals["routed_rows"] += 1
        totals["routed_family_histogram"].update(operator["family_histogram"])
        totals["lane_histogram"][row["data_lane"]] += 1
    else:
        totals["rejected_rows"] += 1
        totals["rejected_family_histogram"].update(operator["family_histogram"])
        for attempt in row["rejection"]["attempts"]:
            totals["rejection_reason_histogram"][attempt["reason"]] += 1


def _final_totals(totals: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rows": int(totals["rows"]),
        "routed_rows": int(totals["routed_rows"]),
        "rejected_rows": int(totals["rejected_rows"]),
        "states": int(totals["states"]),
        "transitions": int(totals["transitions"]),
        "atom_count_delta_sum": int(totals["atom_count_delta_sum"]),
        "graph_cycle_rank_delta_sum": int(totals["graph_cycle_rank_delta_sum"]),
        "family_histogram": _counter_payload(totals["family_histogram"]),
        "routed_family_histogram": _counter_payload(totals["routed_family_histogram"]),
        "rejected_family_histogram": _counter_payload(totals["rejected_family_histogram"]),
        "lane_histogram": _counter_payload(totals["lane_histogram"]),
        "rejection_reason_histogram": _counter_payload(totals["rejection_reason_histogram"]),
    }


def materialize_packed_candidate_headers(
    *,
    source_manifest_path: str | Path,
    artifact_root: str | Path,
    routing_policy_path: str | Path,
    editing_corpus_contract_path: str | Path,
    output_dir: str | Path,
    code_revision: str,
    max_entry_bytes: int = DEFAULT_MAX_PACKED_ENTRY_BYTES,
) -> dict[str, Any]:
    """Stream all declared packed rows and atomically publish candidate headers."""

    if _COMMIT_RE.fullmatch(code_revision) is None:
        raise EditingV2PackedCandidateMaterializationError(
            "code_revision must be a full lowercase 40-character Git SHA"
        )
    source_manifest_source = Path(source_manifest_path)
    routing_policy_source = Path(routing_policy_path)
    corpus_contract_source = Path(editing_corpus_contract_path)
    source_manifest = load_packed_candidate_source_manifest(source_manifest_source)
    routing_policy = load_routing_policy(routing_policy_source)
    corpus_contract = load_editing_corpus_contract(corpus_contract_source)
    implementation_source = Path(__file__)
    implementation_sha256 = file_sha256(implementation_source)
    target = Path(output_dir)
    if target.exists():
        raise EditingV2PackedCandidateMaterializationError(
            f"output directory already exists; refusing overwrite: {target}"
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
        rows_path = staging / CANDIDATE_ROWS_FILENAME
        rows_file_digest = hashlib.sha256()
        rows_semantic_digest = hashlib.sha256()
        address_stream_digest = hashlib.sha256()
        global_totals = _empty_totals()
        source_summaries: list[dict[str, Any]] = []
        attempt_index = 0
        with rows_path.open("wb") as rows_handle:
            for source in source_manifest["sources"]:
                source_totals = _empty_totals()
                shard_summaries: list[dict[str, Any]] = []
                for shard in source["shards"]:
                    shard_path = _resolve_relative(
                        Path(artifact_root),
                        shard["relative_path"],
                        field="packed shard relative_path",
                    )
                    manifest_path = _resolve_relative(
                        Path(artifact_root),
                        shard["manifest_relative_path"],
                        field="packed shard manifest_relative_path",
                    )
                    overlay_path = _resolve_relative(
                        Path(artifact_root),
                        shard["historical_provenance_overlay_relative_path"],
                        field=("packed shard historical_provenance_overlay_relative_path"),
                    )
                    if not shard_path.is_file():
                        raise EditingV2PackedCandidateMaterializationError(
                            f"packed shard is absent: {shard_path}"
                        )
                    observed_shard_sha256 = file_sha256(shard_path)
                    if observed_shard_sha256 != shard["file_sha256"]:
                        raise EditingV2PackedCandidateMaterializationError(
                            f"packed shard SHA-256 mismatch for {shard_path}: "
                            f"expected {shard['file_sha256']}, "
                            f"observed {observed_shard_sha256}"
                        )
                    packed_manifest = _load_packed_manifest(
                        manifest_path,
                        expected_file_sha256=shard["manifest_file_sha256"],
                    )
                    historical_overlay_receipt = _load_historical_overlay_receipt(
                        overlay_path,
                        expected_file_sha256=shard["historical_provenance_overlay_file_sha256"],
                        expected_shard_name=shard_path.name,
                        expected_packed_shard_sha256=shard["file_sha256"],
                        expected_original_manifest_sha256=shard["manifest_file_sha256"],
                    )
                    shard_totals = _empty_totals()
                    shard_summary = {
                        "relative_path": shard["relative_path"],
                        "file_sha256": shard["file_sha256"],
                        "bytes": shard_path.stat().st_size,
                        "manifest_relative_path": shard["manifest_relative_path"],
                        "manifest_file_sha256": shard["manifest_file_sha256"],
                        "historical_provenance_overlay_relative_path": shard[
                            "historical_provenance_overlay_relative_path"
                        ],
                        "historical_provenance_overlay": (historical_overlay_receipt),
                    }
                    for entry_index, entry in _iter_bounded_gzip_jsonl(
                        shard_path,
                        max_entry_bytes=max_entry_bytes,
                    ):
                        row = _candidate_header(
                            entry,
                            attempt_index=attempt_index,
                            source=source,
                            shard_summary=shard_summary,
                            entry_index=entry_index,
                            routing_policy=routing_policy,
                            corpus_contract=corpus_contract,
                            implementation_sha256=implementation_sha256,
                        )
                        encoded_row = canonical_json_bytes(row) + b"\n"
                        rows_handle.write(encoded_row)
                        rows_file_digest.update(encoded_row)
                        rows_semantic_digest.update(row["row_sha256"].encode("ascii"))
                        rows_semantic_digest.update(b"\n")
                        address_stream_digest.update(
                            row["packed_address"]["address_sha256"].encode("ascii")
                        )
                        address_stream_digest.update(b"\n")
                        _accumulate_totals(global_totals, row)
                        _accumulate_totals(source_totals, row)
                        _accumulate_totals(shard_totals, row)
                        attempt_index += 1
                    final_shard_totals = _final_totals(shard_totals)
                    if (
                        final_shard_totals["rows"] != packed_manifest["entries"]
                        or final_shard_totals["states"] != packed_manifest["states"]
                    ):
                        raise EditingV2PackedCandidateMaterializationError(
                            f"physical packed shard census disagrees with its manifest: "
                            f"{shard_path}"
                        )
                    shard_summaries.append(
                        {
                            **shard_summary,
                            "packed_manifest_entries": packed_manifest["entries"],
                            "packed_manifest_states": packed_manifest["states"],
                            "totals": final_shard_totals,
                        }
                    )
                source_summaries.append(
                    {
                        "source_asset_id": source["source_asset_id"],
                        "source_asset_path": source["source_asset_path"],
                        "source_asset_sha256": source["source_asset_sha256"],
                        "source_kind": source["source_kind"],
                        "allowed_layers": source["allowed_layers"],
                        "shards": shard_summaries,
                        "totals": _final_totals(source_totals),
                    }
                )
            rows_handle.flush()
            os.fsync(rows_handle.fileno())

        observed_rows_file_sha256 = file_sha256(rows_path)
        if observed_rows_file_sha256 != rows_file_digest.hexdigest():
            raise EditingV2PackedCandidateMaterializationError(
                "candidate row bytes changed between streaming write and publication"
            )
        totals = _final_totals(global_totals)
        if totals["rows"] == 0:
            raise EditingV2PackedCandidateMaterializationError(
                "packed sources produced no candidate attempts"
            )
        manifest_body = {
            "schema": MATERIALIZATION_SCHEMA,
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "status": MATERIALIZATION_STATUS,
            "training_authorized": False,
            "blockers": [
                "active8_exact_candidate_admission.not_run",
                "chemistry_derived_membership_receipts.not_resolved",
                "four_role_split_assignment.not_applied",
                "sampler_stream.not_frozen",
            ],
            "code_revision": code_revision,
            "provenance_boundary": {
                "status": DERIVATIVE_PROVENANCE_STATUS,
                "current_derivative_authority": (
                    "materializer_implementation_plus_routing_policy_plus_editing_corpus_contract"
                ),
                "historical_cache_sidecar_role": HISTORICAL_OVERLAY_ROLE,
                "historical_sidecar_fields_used_as_current_training_provenance": False,
            },
            "implementation": {
                "relative_path": (
                    "src/compose_v4/data/editing_v2_packed_candidate_materializer.py"
                ),
                "file_sha256": implementation_sha256,
                "uses_rdkit_or_executor_replay": False,
                "uses_live_packed_overlay_semantic_validation": False,
                "historical_overlay_interpretation": HISTORICAL_OVERLAY_ROLE,
                "streaming_max_packed_entry_bytes": max_entry_bytes,
                "atomic_publication": "same_parent_staging_directory_rename",
            },
            "inputs": {
                "source_manifest": {
                    "file_sha256": file_sha256(source_manifest_source),
                    "manifest_sha256": source_manifest["manifest_sha256"],
                },
                "routing_policy": {
                    "file_sha256": file_sha256(routing_policy_source),
                    "semantic_sha256": routing_policy_sha256(routing_policy),
                },
                "editing_corpus_contract": {
                    "file_sha256": file_sha256(corpus_contract_source),
                    "semantic_sha256": canonical_sha256(corpus_contract),
                    "contract_id": corpus_contract["contract_id"],
                },
            },
            "sources": source_summaries,
            "rows": {
                "relative_path": CANDIDATE_ROWS_FILENAME,
                "file_sha256": observed_rows_file_sha256,
                "semantic_sha256": rows_semantic_digest.hexdigest(),
                "address_stream_sha256": address_stream_digest.hexdigest(),
                "totals": totals,
            },
        }
        materialization = {
            **manifest_body,
            "manifest_sha256": canonical_sha256(manifest_body),
        }
        manifest_path = staging / MATERIALIZATION_FILENAME
        with manifest_path.open("wb") as manifest_handle:
            manifest_handle.write(
                json.dumps(materialization, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            )
            manifest_handle.flush()
            os.fsync(manifest_handle.fileno())
        os.replace(staging, target)
        published = True
        return materialization
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def validate_packed_candidate_materialization(
    output_dir: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    max_row_bytes: int = 2 * 1024 * 1024,
) -> dict[str, Any]:
    """Reopen and validate a published row stream and its self-hashed manifest."""

    root = Path(output_dir)
    manifest_path = root / MATERIALIZATION_FILENAME
    rows_path = root / CANDIDATE_ROWS_FILENAME
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EditingV2PackedCandidateMaterializationError(
            f"cannot load candidate materialization manifest: {manifest_path}"
        ) from error
    if not isinstance(manifest, Mapping):
        raise EditingV2PackedCandidateMaterializationError(
            "candidate materialization manifest must be an object"
        )
    supplied_sha256 = _require_sha256(
        manifest.get("manifest_sha256"),
        field="materialization.manifest_sha256",
    )
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("schema") != MATERIALIZATION_SCHEMA
        or manifest.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION
        or manifest.get("status") != MATERIALIZATION_STATUS
        or manifest.get("training_authorized") is not False
        or canonical_sha256(body) != supplied_sha256
    ):
        raise EditingV2PackedCandidateMaterializationError(
            "candidate materialization identity, authority, or self-hash disagrees"
        )
    if expected_manifest_sha256 is not None:
        if (
            _require_sha256(
                expected_manifest_sha256,
                field="expected_manifest_sha256",
            )
            != supplied_sha256
        ):
            raise EditingV2PackedCandidateMaterializationError(
                "candidate materialization differs from the expected semantic identity"
            )
    rows = manifest.get("rows")
    if (
        not isinstance(rows, Mapping)
        or rows.get("relative_path") != CANDIDATE_ROWS_FILENAME
        or not rows_path.is_file()
        or file_sha256(rows_path) != rows.get("file_sha256")
    ):
        raise EditingV2PackedCandidateMaterializationError(
            "candidate row file is absent or its physical SHA-256 disagrees"
        )
    semantic_digest = hashlib.sha256()
    address_digest = hashlib.sha256()
    totals = _empty_totals()
    with rows_path.open("rb") as handle:
        expected_index = 0
        while True:
            raw_line = handle.readline(max_row_bytes + 1)
            if not raw_line:
                break
            if len(raw_line) > max_row_bytes or not raw_line.endswith(b"\n"):
                raise EditingV2PackedCandidateMaterializationError(
                    "candidate header exceeds the validation row bound"
                )
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EditingV2PackedCandidateMaterializationError(
                    f"candidate header {expected_index} is invalid JSON"
                ) from error
            if (
                not isinstance(row, Mapping)
                or row.get("schema") != CANDIDATE_HEADER_SCHEMA
                or row.get("schema_version") != CANDIDATE_HEADER_SCHEMA_VERSION
                or row.get("status") != CANDIDATE_HEADER_STATUS
                or row.get("training_authorized") is not False
                or row.get("attempt_index") != expected_index
            ):
                raise EditingV2PackedCandidateMaterializationError(
                    f"candidate header {expected_index} identity or order disagrees"
                )
            row_sha256 = _require_sha256(
                row.get("row_sha256"),
                field=f"candidate header {expected_index}.row_sha256",
            )
            row_body = {key: value for key, value in row.items() if key != "row_sha256"}
            if canonical_sha256(row_body) != row_sha256:
                raise EditingV2PackedCandidateMaterializationError(
                    f"candidate header {expected_index} self-hash disagrees"
                )
            semantic_digest.update(row_sha256.encode("ascii"))
            semantic_digest.update(b"\n")
            address_sha256 = _require_sha256(
                row["packed_address"]["address_sha256"],
                field=f"candidate header {expected_index}.address_sha256",
            )
            address_digest.update(address_sha256.encode("ascii"))
            address_digest.update(b"\n")
            _accumulate_totals(totals, row)
            expected_index += 1
    if (
        semantic_digest.hexdigest() != rows.get("semantic_sha256")
        or address_digest.hexdigest() != rows.get("address_stream_sha256")
        or _final_totals(totals) != rows.get("totals")
    ):
        raise EditingV2PackedCandidateMaterializationError(
            "candidate row semantic stream or aggregate census disagrees"
        )
    return dict(manifest)


__all__ = [
    "CANDIDATE_HEADER_SCHEMA",
    "CANDIDATE_ROWS_FILENAME",
    "DEFAULT_MAX_PACKED_ENTRY_BYTES",
    "EditingV2PackedCandidateMaterializationError",
    "HISTORICAL_OVERLAY_ROLE",
    "MATERIALIZATION_FILENAME",
    "MATERIALIZATION_SCHEMA",
    "SOURCE_BINDING_REGISTRY_SCHEMA",
    "SOURCE_MANIFEST_SCHEMA",
    "build_overlay_source_binding_registry",
    "build_packed_candidate_source_manifest",
    "build_packed_candidate_source_manifest_from_overlay_completion",
    "canonical_json_bytes",
    "canonical_sha256",
    "file_sha256",
    "load_overlay_source_binding_registry",
    "load_packed_candidate_source_manifest",
    "materialize_packed_candidate_headers",
    "validate_historical_provenance_overlay",
    "validate_packed_candidate_materialization",
    "validate_packed_candidate_source_manifest",
    "validate_overlay_source_binding_registry",
    "write_packed_candidate_source_manifest",
]
