"""Mechanical, content-addressed chunk cache for semantic Active8 admission.

The semantic migration artifact remains immutable.  This module validates one
complete semantic packed source, scans its canonical row stream once, and
publishes deterministic contiguous gzip chunks.  Chunk receipts preserve the
original source-shard identity and the exact mapping from a chunk-local row to
its global source entry index.  No chemistry decision, Active8 admission, or
training authority is produced here.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from compose_v4.data.packed_trace_store import (
    AddressedPackedTrace,
    PackedTraceAddress,
    PackedTraceProgress,
)
from compose_v4.data.semantic_packed_trace_store import (
    ENCODING,
    SHARD_FILENAME,
    load_semantic_packed_manifest,
    validate_semantic_packed_entry_ranges,
)
from compose_v4.rewrite.trace_shard import decode_state
from compose_v4.rewrite.trace_shard_v3 import (
    SemanticTraceShardError,
    decode_semantic_trace_record,
)

CACHE_SCHEMA = "compose.data.semantic_active8_chunk_cache"
CACHE_SCHEMA_VERSION = 1
CACHE_STATUS = "COMPLETE_MECHANICAL_CACHE_NO_ACTIVE8_DECISIONS"
CHUNK_RECEIPT_SCHEMA = "compose.data.semantic_active8_chunk_receipt"
CHUNK_RECEIPT_SCHEMA_VERSION = 1
CHUNK_RECEIPT_STATUS = "COMPLETE_MECHANICAL_CHUNK_NO_ACTIVE8_DECISION"
BUILDER_SCHEMA = "compose.data.semantic_active8_chunk_cache_builder"
BUILDER_SCHEMA_VERSION = 1
ADDRESS_CONTRACT_SCHEMA = "compose.data.semantic_active8_original_address"
ADDRESS_CONTRACT_SCHEMA_VERSION = 1
CHUNK_POLICY_ALGORITHM = "contiguous_global_entry_ranges_v1"
DEFAULT_TARGET_ROWS_PER_CHUNK = 500

_CACHE_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "active8_decisions_run",
    "source_identity",
    "chunk_policy",
    "builder_identity",
    "run_identity_sha256",
    "chunk_count",
    "chunk_inventory",
    "chunk_inventory_sha256",
    "observed_source_census",
    "completion_sha256",
}
_RECEIPT_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "active8_decision",
    "source_identity",
    "chunk_policy_sha256",
    "chunk_index",
    "entry_start",
    "entry_stop",
    "row_count",
    "raw_row_stream_sha256",
    "chunk_object_path",
    "chunk_physical_sha256",
    "encoding",
    "address_contract",
    "receipt_sha256",
}
_SOURCE_IDENTITY_FIELDS = {
    "semantic_shard_sha256",
    "semantic_manifest_physical_sha256",
    "semantic_manifest_sha256",
    "record_stream_sha256",
    "process_identity_sha256",
    "process_contract_sha256",
    "action_codec_schema_version",
    "action_codec_implementation_hash",
    "trace_schema",
    "trace_schema_version",
    "entries",
    "states",
    "actions",
    "family_histogram",
    "data_lane",
    "split",
    "source_binding_sha256",
    "decision_binding_sha256",
    "source_identity_sha256",
}
_CHUNK_POLICY_FIELDS = {
    "algorithm",
    "target_rows_per_chunk",
    "encoding",
    "empty_source_policy",
    "chunk_policy_sha256",
}
_CHUNK_INVENTORY_FIELDS = {
    "chunk_index",
    "entry_start",
    "entry_stop",
    "row_count",
    "raw_row_stream_sha256",
    "chunk_object_path",
    "chunk_physical_sha256",
    "receipt_object_path",
    "receipt_file_sha256",
    "receipt_sha256",
}


class SemanticActive8ChunkCacheError(RuntimeError):
    """The mechanical semantic chunk cache is incomplete or inconsistent."""


def _canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SemanticActive8ChunkCacheError(f"{field} must be a lowercase SHA-256")
    return value


def _publish_bytes_immutable(path: Path, content: bytes) -> bool:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_sha256 = hashlib.sha256(content).hexdigest()
    if target.exists():
        if (
            not target.is_file()
            or _file_sha256(target) != expected_sha256
            or target.read_bytes() != content
        ):
            raise SemanticActive8ChunkCacheError(
                f"immutable semantic chunk-cache collision at {target}"
            )
        return True
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
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
        if _file_sha256(target) != expected_sha256 or target.read_bytes() != content:
            raise SemanticActive8ChunkCacheError(
                f"immutable semantic chunk-cache collision at {target}"
            )
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return False


def _safe_object_path(root: Path, relative_path: object, *, field: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise SemanticActive8ChunkCacheError(f"{field} must be nonempty text")
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or pure == PurePosixPath(".")
        or ".." in pure.parts
        or "\\" in relative_path
        or any(character in relative_path for character in "*?[]")
    ):
        raise SemanticActive8ChunkCacheError(f"{field} is unsafe")
    trusted_root = Path(root).resolve()
    candidate = (trusted_root / Path(*pure.parts)).resolve()
    if not candidate.is_relative_to(trusted_root):
        raise SemanticActive8ChunkCacheError(f"{field} escapes the cache root")
    return candidate


def semantic_active8_chunk_cache_builder_identity() -> dict[str, object]:
    """Bind cache artifacts to the exact mechanical implementation bytes."""

    body: dict[str, object] = {
        "schema": BUILDER_SCHEMA,
        "schema_version": BUILDER_SCHEMA_VERSION,
        "algorithm": "single_source_scan_deterministic_contiguous_chunks_v1",
        "implementation_file_sha256": _file_sha256(Path(__file__)),
    }
    return {**body, "identity_sha256": _canonical_sha256(body)}


def _source_identity(
    manifest: Mapping[str, Any],
    *,
    manifest_physical_sha256: str,
) -> dict[str, object]:
    identity = {
        "semantic_shard_sha256": _require_sha256(
            manifest["shard_sha256"],
            field="manifest.shard_sha256",
        ),
        "semantic_manifest_physical_sha256": _require_sha256(
            manifest_physical_sha256,
            field="manifest_physical_sha256",
        ),
        "semantic_manifest_sha256": _require_sha256(
            manifest["manifest_sha256"],
            field="manifest.manifest_sha256",
        ),
        "record_stream_sha256": _require_sha256(
            manifest["record_stream_sha256"],
            field="manifest.record_stream_sha256",
        ),
        "process_identity_sha256": _require_sha256(
            manifest["process_identity_sha256"],
            field="manifest.process_identity_sha256",
        ),
        "process_contract_sha256": _require_sha256(
            manifest["process_contract_sha256"],
            field="manifest.process_contract_sha256",
        ),
        "action_codec_schema_version": manifest["action_codec_schema_version"],
        "action_codec_implementation_hash": manifest[
            "action_codec_implementation_hash"
        ],
        "trace_schema": manifest["trace_schema"],
        "trace_schema_version": manifest["trace_schema_version"],
        "entries": manifest["entries"],
        "states": manifest["states"],
        "actions": manifest["actions"],
        "family_histogram": manifest["family_histogram"],
        "data_lane": manifest["data_lane"],
        "split": manifest["split"],
        "source_binding_sha256": _canonical_sha256(manifest["source_binding"]),
        "decision_binding_sha256": _canonical_sha256(manifest["decision_binding"]),
    }
    identity["source_identity_sha256"] = _canonical_sha256(identity)
    return identity


def _validate_source_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _SOURCE_IDENTITY_FIELDS:
        raise SemanticActive8ChunkCacheError("chunk-cache source identity is malformed")
    body = {key: item for key, item in value.items() if key != "source_identity_sha256"}
    if value.get("source_identity_sha256") != _canonical_sha256(body):
        raise SemanticActive8ChunkCacheError(
            "chunk-cache source identity self-hash disagrees"
        )
    for field in (
        "semantic_shard_sha256",
        "semantic_manifest_physical_sha256",
        "semantic_manifest_sha256",
        "record_stream_sha256",
        "process_identity_sha256",
        "process_contract_sha256",
        "source_binding_sha256",
        "decision_binding_sha256",
        "source_identity_sha256",
    ):
        _require_sha256(value.get(field), field=f"source_identity.{field}")
    for field in ("entries", "states", "actions"):
        if type(value.get(field)) is not int or value[field] < 0:
            raise SemanticActive8ChunkCacheError(
                f"source_identity.{field} must be nonnegative"
            )
    if not isinstance(value.get("family_histogram"), dict) or any(
        not isinstance(family, str) or type(count) is not int or count < 0
        for family, count in value["family_histogram"].items()
    ):
        raise SemanticActive8ChunkCacheError(
            "source_identity.family_histogram is malformed"
        )
    for field in (
        "action_codec_implementation_hash",
        "trace_schema",
        "data_lane",
        "split",
    ):
        if not isinstance(value.get(field), str) or not value[field]:
            raise SemanticActive8ChunkCacheError(
                f"source_identity.{field} must be nonempty text"
            )
    for field in ("action_codec_schema_version", "trace_schema_version"):
        if type(value.get(field)) is not int or value[field] <= 0:
            raise SemanticActive8ChunkCacheError(
                f"source_identity.{field} must be positive"
            )
    return value


def _chunk_policy(target_rows_per_chunk: int) -> dict[str, object]:
    if type(target_rows_per_chunk) is not int or target_rows_per_chunk <= 0:
        raise ValueError("target_rows_per_chunk must be a positive integer")
    body: dict[str, object] = {
        "algorithm": CHUNK_POLICY_ALGORITHM,
        "target_rows_per_chunk": target_rows_per_chunk,
        "encoding": ENCODING,
        "empty_source_policy": "one_safe_empty_chunk_0_0",
    }
    return {**body, "chunk_policy_sha256": _canonical_sha256(body)}


def _validate_chunk_policy(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _CHUNK_POLICY_FIELDS:
        raise SemanticActive8ChunkCacheError("semantic chunk-cache policy is malformed")
    try:
        expected = _chunk_policy(value["target_rows_per_chunk"])
    except ValueError as error:
        raise SemanticActive8ChunkCacheError(
            "semantic chunk-cache policy is malformed"
        ) from error
    if value != expected:
        raise SemanticActive8ChunkCacheError("semantic chunk-cache policy disagrees")
    return value


def _address_contract(
    source_identity: Mapping[str, object],
    *,
    entry_start: int,
) -> dict[str, object]:
    return {
        "schema": ADDRESS_CONTRACT_SCHEMA,
        "schema_version": ADDRESS_CONTRACT_SCHEMA_VERSION,
        "global_entry_index_rule": "entry_start_plus_chunk_local_row_index",
        "entry_start": entry_start,
        "packed_shard_content_sha256": source_identity["semantic_shard_sha256"],
        "packed_shard_name": SHARD_FILENAME,
        "layer": source_identity["data_lane"],
        "partition": source_identity["split"],
    }


def _decode_canonical_row(
    raw_line: bytes,
    *,
    global_entry_index: int,
    source_identity: Mapping[str, object],
    validate_replay: bool,
) -> tuple[dict[str, Any], Any, tuple[Any, ...]]:
    if not raw_line.strip():
        raise SemanticActive8ChunkCacheError(
            f"semantic source row {global_entry_index} is blank"
        )
    try:
        record = json.loads(raw_line)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SemanticActive8ChunkCacheError(
            f"semantic source row {global_entry_index} is not JSON"
        ) from error
    if (
        not isinstance(record, dict)
        or _canonical_json_bytes(
            record,
            newline=True,
        )
        != raw_line
    ):
        raise SemanticActive8ChunkCacheError(
            f"semantic source row {global_entry_index} is not canonical JSONL"
        )
    try:
        trace = decode_semantic_trace_record(record, validate_replay=validate_replay)
        states = tuple(decode_state(payload) for payload in record["states"])
    except (KeyError, TypeError, SemanticTraceShardError) as error:
        raise SemanticActive8ChunkCacheError(
            f"semantic source row {global_entry_index} is malformed"
        ) from error
    if (
        record["data_lane"] != source_identity["data_lane"]
        or record["split"] != source_identity["split"]
    ):
        raise SemanticActive8ChunkCacheError(
            f"semantic source row {global_entry_index} leaves its declared lane or split"
        )
    return record, trace, states


def _deterministic_gzip(rows: Iterable[bytes]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as handle:
        for row in rows:
            handle.write(row)
    return output.getvalue()


def _chunk_object_relative_path(chunk_sha256: str) -> str:
    return f"objects/chunks/{chunk_sha256[:2]}/{chunk_sha256}.jsonl.gz"


def _receipt_object_relative_path(receipt_file_sha256: str) -> str:
    return f"objects/receipts/{receipt_file_sha256[:2]}/{receipt_file_sha256}.json"


def _publish_chunk(
    output_root: Path,
    *,
    source_identity: Mapping[str, object],
    chunk_policy: Mapping[str, object],
    chunk_index: int,
    entry_start: int,
    rows: tuple[bytes, ...],
) -> dict[str, object]:
    entry_stop = entry_start + len(rows)
    raw_stream = hashlib.sha256()
    for row in rows:
        raw_stream.update(row)
    chunk_bytes = _deterministic_gzip(rows)
    chunk_sha256 = hashlib.sha256(chunk_bytes).hexdigest()
    chunk_relative = _chunk_object_relative_path(chunk_sha256)
    _publish_bytes_immutable(Path(output_root) / chunk_relative, chunk_bytes)
    receipt_body: dict[str, object] = {
        "schema": CHUNK_RECEIPT_SCHEMA,
        "schema_version": CHUNK_RECEIPT_SCHEMA_VERSION,
        "status": CHUNK_RECEIPT_STATUS,
        "training_authorized": False,
        "active8_decision": None,
        "source_identity": dict(source_identity),
        "chunk_policy_sha256": chunk_policy["chunk_policy_sha256"],
        "chunk_index": chunk_index,
        "entry_start": entry_start,
        "entry_stop": entry_stop,
        "row_count": len(rows),
        "raw_row_stream_sha256": raw_stream.hexdigest(),
        "chunk_object_path": chunk_relative,
        "chunk_physical_sha256": chunk_sha256,
        "encoding": ENCODING,
        "address_contract": _address_contract(
            source_identity,
            entry_start=entry_start,
        ),
    }
    receipt = {**receipt_body, "receipt_sha256": _canonical_sha256(receipt_body)}
    receipt_bytes = _canonical_json_bytes(receipt, newline=True)
    receipt_file_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    receipt_relative = _receipt_object_relative_path(receipt_file_sha256)
    _publish_bytes_immutable(Path(output_root) / receipt_relative, receipt_bytes)
    return {
        "chunk_index": chunk_index,
        "entry_start": entry_start,
        "entry_stop": entry_stop,
        "row_count": len(rows),
        "raw_row_stream_sha256": raw_stream.hexdigest(),
        "chunk_object_path": chunk_relative,
        "chunk_physical_sha256": chunk_sha256,
        "receipt_object_path": receipt_relative,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
    }


def _receipt_from_inventory(
    output_root: Path,
    inventory: Mapping[str, object],
    *,
    source_identity: Mapping[str, object],
    chunk_policy: Mapping[str, object],
) -> dict[str, Any]:
    receipt_path = _safe_object_path(
        output_root,
        inventory.get("receipt_object_path"),
        field="chunk_inventory.receipt_object_path",
    )
    if not receipt_path.is_file():
        raise SemanticActive8ChunkCacheError(
            f"semantic chunk receipt is absent: {receipt_path}"
        )
    receipt_file_sha256 = _file_sha256(receipt_path)
    if receipt_file_sha256 != _require_sha256(
        inventory.get("receipt_file_sha256"),
        field="chunk_inventory.receipt_file_sha256",
    ):
        raise SemanticActive8ChunkCacheError("semantic chunk receipt bytes disagree")
    receipt_bytes = receipt_path.read_bytes()
    try:
        receipt = json.loads(receipt_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticActive8ChunkCacheError(
            f"semantic chunk receipt is unreadable: {receipt_path}"
        ) from error
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_FIELDS:
        raise SemanticActive8ChunkCacheError("semantic chunk receipt fields disagree")
    if _canonical_json_bytes(receipt, newline=True) != receipt_bytes:
        raise SemanticActive8ChunkCacheError(
            "semantic chunk receipt is not canonical JSON"
        )
    if (
        receipt.get("schema") != CHUNK_RECEIPT_SCHEMA
        or receipt.get("schema_version") != CHUNK_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != CHUNK_RECEIPT_STATUS
        or receipt.get("training_authorized") is not False
        or receipt.get("active8_decision") is not None
    ):
        raise SemanticActive8ChunkCacheError(
            "semantic chunk receipt crosses its mechanical boundary"
        )
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if receipt.get("receipt_sha256") != _canonical_sha256(receipt_body):
        raise SemanticActive8ChunkCacheError(
            "semantic chunk receipt self-hash disagrees"
        )
    if receipt["source_identity"] != source_identity:
        raise SemanticActive8ChunkCacheError("semantic chunk source identity disagrees")
    if receipt["chunk_policy_sha256"] != chunk_policy["chunk_policy_sha256"]:
        raise SemanticActive8ChunkCacheError("semantic chunk policy identity disagrees")
    _validate_receipt_range(receipt, source_identity=source_identity)
    inventory_projection = {
        "chunk_index": receipt["chunk_index"],
        "entry_start": receipt["entry_start"],
        "entry_stop": receipt["entry_stop"],
        "row_count": receipt["row_count"],
        "raw_row_stream_sha256": receipt["raw_row_stream_sha256"],
        "chunk_object_path": receipt["chunk_object_path"],
        "chunk_physical_sha256": receipt["chunk_physical_sha256"],
        "receipt_object_path": inventory["receipt_object_path"],
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
    }
    if dict(inventory) != inventory_projection:
        raise SemanticActive8ChunkCacheError(
            "semantic chunk inventory and receipt disagree"
        )
    return receipt


def _validate_receipt_range(
    receipt: Mapping[str, Any],
    *,
    source_identity: Mapping[str, object],
) -> None:
    for field in ("chunk_index", "entry_start", "entry_stop", "row_count"):
        if type(receipt.get(field)) is not int or receipt[field] < 0:
            raise SemanticActive8ChunkCacheError(
                f"semantic chunk receipt {field} must be nonnegative"
            )
    if (
        receipt["entry_start"] > receipt["entry_stop"]
        or receipt["entry_stop"] > source_identity["entries"]
        or receipt["row_count"] != receipt["entry_stop"] - receipt["entry_start"]
    ):
        raise SemanticActive8ChunkCacheError(
            "semantic chunk receipt range lies outside its source census"
        )
    if receipt["row_count"] == 0 and not (
        source_identity["entries"] == 0
        and receipt["entry_start"] == 0
        and receipt["entry_stop"] == 0
    ):
        raise SemanticActive8ChunkCacheError(
            "an empty semantic chunk is valid only for an empty source"
        )
    for field in (
        "raw_row_stream_sha256",
        "chunk_physical_sha256",
        "chunk_policy_sha256",
        "receipt_sha256",
    ):
        _require_sha256(receipt.get(field), field=f"receipt.{field}")
    if receipt.get("encoding") != ENCODING:
        raise SemanticActive8ChunkCacheError("semantic chunk encoding disagrees")
    if receipt.get("address_contract") != _address_contract(
        source_identity,
        entry_start=receipt["entry_start"],
    ):
        raise SemanticActive8ChunkCacheError(
            "semantic chunk original-address contract disagrees"
        )


def _iter_validated_chunk_rows(
    output_root: Path,
    receipt: Mapping[str, Any],
    *,
    sentinel_replay_entries: int,
) -> Iterable[tuple[bytes, dict[str, Any], Any, tuple[Any, ...]]]:
    chunk_path = _safe_object_path(
        output_root,
        receipt["chunk_object_path"],
        field="receipt.chunk_object_path",
    )
    if (
        not chunk_path.is_file()
        or _file_sha256(chunk_path) != receipt["chunk_physical_sha256"]
    ):
        raise SemanticActive8ChunkCacheError("semantic chunk object bytes disagree")
    raw_stream = hashlib.sha256()
    observed = 0
    seen_trace_ids: set[str] = set()
    canonical_rows: list[bytes] = []
    with gzip.open(chunk_path, "rb") as handle:
        for local_index, raw_line in enumerate(handle):
            global_index = receipt["entry_start"] + local_index
            record, trace, states = _decode_canonical_row(
                raw_line,
                global_entry_index=global_index,
                source_identity=receipt["source_identity"],
                validate_replay=global_index < sentinel_replay_entries,
            )
            trace_id = str(record["trace_id"])
            if trace_id in seen_trace_ids:
                raise SemanticActive8ChunkCacheError(
                    f"duplicate trace_id {trace_id!r} in semantic chunk"
                )
            seen_trace_ids.add(trace_id)
            raw_stream.update(raw_line)
            canonical_rows.append(raw_line)
            observed += 1
            yield raw_line, record, trace, states
    if observed != receipt["row_count"]:
        raise SemanticActive8ChunkCacheError(
            "semantic chunk row count disagrees with its receipt"
        )
    if raw_stream.hexdigest() != receipt["raw_row_stream_sha256"]:
        raise SemanticActive8ChunkCacheError(
            "semantic chunk raw row stream disagrees with its receipt"
        )
    if _deterministic_gzip(canonical_rows) != chunk_path.read_bytes():
        raise SemanticActive8ChunkCacheError(
            "semantic chunk bytes are not in deterministic gzip form"
        )


def build_semantic_active8_chunk_cache(
    artifact_dir: Path,
    *,
    expected_shard_sha256: str,
    expected_manifest_sha256: str,
    output_root: Path,
    target_rows_per_chunk: int = DEFAULT_TARGET_ROWS_PER_CHUNK,
    expected_source_binding: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate and mechanically chunk one semantic source in one stream pass."""

    policy = _chunk_policy(target_rows_per_chunk)
    artifact_root = Path(artifact_dir)
    output_root = Path(output_root)
    manifest = load_semantic_packed_manifest(
        artifact_root,
        expected_shard_sha256=expected_shard_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_source_binding=expected_source_binding,
    )
    source_identity = _source_identity(
        manifest,
        manifest_physical_sha256=expected_manifest_sha256,
    )
    builder_identity = semantic_active8_chunk_cache_builder_identity()
    run_body = {
        "schema": CACHE_SCHEMA,
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_identity": source_identity,
        "chunk_policy": policy,
        "builder_identity": builder_identity,
    }
    run_identity_sha256 = _canonical_sha256(run_body)
    completion_path = output_root / "runs" / run_identity_sha256 / "COMPLETE.json"
    if completion_path.is_file():
        return load_semantic_active8_chunk_cache(
            output_root,
            run_identity_sha256=run_identity_sha256,
            expected_source_shard_sha256=expected_shard_sha256,
            expected_source_manifest_sha256=expected_manifest_sha256,
        )

    shard_path = artifact_root / SHARD_FILENAME
    source_stream = hashlib.sha256()
    counts: Counter[str] = Counter()
    families: Counter[str] = Counter()
    seen_trace_ids: set[str] = set()
    chunk_rows: list[bytes] = []
    chunk_inventory: list[dict[str, object]] = []
    chunk_start = 0

    def flush_chunk() -> None:
        nonlocal chunk_rows, chunk_start
        chunk_inventory.append(
            _publish_chunk(
                output_root,
                source_identity=source_identity,
                chunk_policy=policy,
                chunk_index=len(chunk_inventory),
                entry_start=chunk_start,
                rows=tuple(chunk_rows),
            )
        )
        chunk_start += len(chunk_rows)
        chunk_rows = []

    with gzip.open(shard_path, "rb") as handle:
        for entry_index, raw_line in enumerate(handle):
            record, trace, _ = _decode_canonical_row(
                raw_line,
                global_entry_index=entry_index,
                source_identity=source_identity,
                validate_replay=False,
            )
            trace_id = str(record["trace_id"])
            if trace_id in seen_trace_ids:
                raise SemanticActive8ChunkCacheError(
                    f"duplicate trace_id {trace_id!r} in semantic source"
                )
            seen_trace_ids.add(trace_id)
            source_stream.update(raw_line)
            counts["entries"] += 1
            counts["states"] += len(record["states"])
            counts["actions"] += len(trace.steps)
            families.update(record["family_histogram"])
            chunk_rows.append(raw_line)
            if len(chunk_rows) == target_rows_per_chunk:
                flush_chunk()
    if chunk_rows or counts["entries"] == 0:
        flush_chunk()

    observed_census = {
        "entries": counts["entries"],
        "states": counts["states"],
        "actions": counts["actions"],
        "family_histogram": dict(sorted(families.items())),
        "record_stream_sha256": source_stream.hexdigest(),
    }
    expected_census = {
        "entries": source_identity["entries"],
        "states": source_identity["states"],
        "actions": source_identity["actions"],
        "family_histogram": source_identity["family_histogram"],
        "record_stream_sha256": source_identity["record_stream_sha256"],
    }
    if observed_census != expected_census:
        raise SemanticActive8ChunkCacheError(
            "semantic source row stream or census disagrees with its manifest"
        )
    validate_semantic_packed_entry_ranges(
        tuple(
            (int(chunk["entry_start"]), int(chunk["entry_stop"]))
            for chunk in chunk_inventory
        ),
        entries=counts["entries"],
    )
    completion_body: dict[str, object] = {
        "schema": CACHE_SCHEMA,
        "schema_version": CACHE_SCHEMA_VERSION,
        "status": CACHE_STATUS,
        "training_authorized": False,
        "active8_decisions_run": False,
        "source_identity": source_identity,
        "chunk_policy": policy,
        "builder_identity": builder_identity,
        "run_identity_sha256": run_identity_sha256,
        "chunk_count": len(chunk_inventory),
        "chunk_inventory": chunk_inventory,
        "chunk_inventory_sha256": _canonical_sha256(chunk_inventory),
        "observed_source_census": observed_census,
    }
    completion = {
        **completion_body,
        "completion_sha256": _canonical_sha256(completion_body),
    }
    _publish_bytes_immutable(
        completion_path,
        _canonical_json_bytes(completion, newline=True),
    )
    return completion


def load_semantic_active8_chunk_cache(
    output_root: Path,
    *,
    run_identity_sha256: str,
    expected_source_shard_sha256: str,
    expected_source_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify one completed cache, every chunk, and full source reconciliation."""

    run_identity_sha256 = _require_sha256(
        run_identity_sha256,
        field="run_identity_sha256",
    )
    completion_path = Path(output_root) / "runs" / run_identity_sha256 / "COMPLETE.json"
    if not completion_path.is_file():
        raise SemanticActive8ChunkCacheError(
            f"semantic chunk-cache completion is absent: {completion_path}"
        )
    completion_bytes = completion_path.read_bytes()
    try:
        completion = json.loads(completion_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticActive8ChunkCacheError(
            f"semantic chunk-cache completion is unreadable: {completion_path}"
        ) from error
    if not isinstance(completion, dict) or set(completion) != _CACHE_FIELDS:
        raise SemanticActive8ChunkCacheError(
            "semantic chunk-cache completion fields disagree"
        )
    if _canonical_json_bytes(completion, newline=True) != completion_bytes:
        raise SemanticActive8ChunkCacheError(
            "semantic chunk-cache completion is not canonical JSON"
        )
    if (
        completion.get("schema") != CACHE_SCHEMA
        or completion.get("schema_version") != CACHE_SCHEMA_VERSION
        or completion.get("status") != CACHE_STATUS
        or completion.get("training_authorized") is not False
        or completion.get("active8_decisions_run") is not False
    ):
        raise SemanticActive8ChunkCacheError(
            "semantic chunk cache crosses its mechanical boundary"
        )
    body = {
        key: value for key, value in completion.items() if key != "completion_sha256"
    }
    if completion.get("completion_sha256") != _canonical_sha256(body):
        raise SemanticActive8ChunkCacheError(
            "semantic chunk-cache completion self-hash disagrees"
        )
    source_identity = _validate_source_identity(completion["source_identity"])
    if source_identity["semantic_shard_sha256"] != _require_sha256(
        expected_source_shard_sha256,
        field="expected_source_shard_sha256",
    ):
        raise SemanticActive8ChunkCacheError(
            "semantic chunk-cache source shard disagrees"
        )
    if source_identity["semantic_manifest_physical_sha256"] != _require_sha256(
        expected_source_manifest_sha256,
        field="expected_source_manifest_sha256",
    ):
        raise SemanticActive8ChunkCacheError(
            "semantic chunk-cache source manifest disagrees"
        )
    policy = _validate_chunk_policy(completion["chunk_policy"])
    if (
        completion["builder_identity"]
        != semantic_active8_chunk_cache_builder_identity()
    ):
        raise SemanticActive8ChunkCacheError(
            "semantic chunk-cache builder identity is stale"
        )
    expected_run_body = {
        "schema": CACHE_SCHEMA,
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_identity": source_identity,
        "chunk_policy": policy,
        "builder_identity": completion["builder_identity"],
    }
    if (
        completion["run_identity_sha256"] != run_identity_sha256
        or _canonical_sha256(expected_run_body) != run_identity_sha256
    ):
        raise SemanticActive8ChunkCacheError(
            "semantic chunk-cache run identity disagrees"
        )
    inventory = completion["chunk_inventory"]
    if (
        not isinstance(inventory, list)
        or not inventory
        or type(completion["chunk_count"]) is not int
        or completion["chunk_count"] != len(inventory)
        or completion["chunk_inventory_sha256"] != _canonical_sha256(inventory)
    ):
        raise SemanticActive8ChunkCacheError(
            "semantic chunk-cache inventory is malformed"
        )
    if any(
        not isinstance(item, dict) or set(item) != _CHUNK_INVENTORY_FIELDS
        for item in inventory
    ):
        raise SemanticActive8ChunkCacheError(
            "semantic chunk-cache inventory row is malformed"
        )
    try:
        ranges = validate_semantic_packed_entry_ranges(
            tuple((item["entry_start"], item["entry_stop"]) for item in inventory),
            entries=source_identity["entries"],
        )
    except ValueError as error:
        raise SemanticActive8ChunkCacheError(
            "semantic chunk-cache ranges do not partition the source"
        ) from error
    source_stream = hashlib.sha256()
    counts: Counter[str] = Counter()
    families: Counter[str] = Counter()
    seen_trace_ids: set[str] = set()
    for expected_chunk_index, (chunk_summary, expected_range) in enumerate(
        zip(inventory, ranges, strict=True)
    ):
        receipt = _receipt_from_inventory(
            Path(output_root),
            chunk_summary,
            source_identity=source_identity,
            chunk_policy=policy,
        )
        if (
            receipt["chunk_index"] != expected_chunk_index
            or (receipt["entry_start"], receipt["entry_stop"]) != expected_range
            or receipt["row_count"] != receipt["entry_stop"] - receipt["entry_start"]
        ):
            raise SemanticActive8ChunkCacheError(
                "semantic chunk receipt range identity disagrees"
            )
        for raw_line, record, trace, _ in _iter_validated_chunk_rows(
            Path(output_root),
            receipt,
            sentinel_replay_entries=0,
        ):
            trace_id = str(record["trace_id"])
            if trace_id in seen_trace_ids:
                raise SemanticActive8ChunkCacheError(
                    f"duplicate trace_id {trace_id!r} across semantic chunks"
                )
            seen_trace_ids.add(trace_id)
            source_stream.update(raw_line)
            counts["entries"] += 1
            counts["states"] += len(record["states"])
            counts["actions"] += len(trace.steps)
            families.update(record["family_histogram"])
    observed_census = {
        "entries": counts["entries"],
        "states": counts["states"],
        "actions": counts["actions"],
        "family_histogram": dict(sorted(families.items())),
        "record_stream_sha256": source_stream.hexdigest(),
    }
    expected_census = {
        "entries": source_identity["entries"],
        "states": source_identity["states"],
        "actions": source_identity["actions"],
        "family_histogram": source_identity["family_histogram"],
        "record_stream_sha256": source_identity["record_stream_sha256"],
    }
    if (
        observed_census != expected_census
        or completion["observed_source_census"] != expected_census
    ):
        raise SemanticActive8ChunkCacheError(
            "semantic chunk-cache rows do not reconstruct the exact source stream"
        )
    return completion


def read_semantic_active8_chunk(
    output_root: Path,
    *,
    receipt_object_path: str,
    expected_receipt_file_sha256: str,
    expected_source_shard_sha256: str,
    expected_source_manifest_sha256: str,
    sentinel_replay_entries: int = 0,
) -> Iterable[AddressedPackedTrace]:
    """Read one independently bound chunk with original global addresses."""

    if type(sentinel_replay_entries) is not int or sentinel_replay_entries < 0:
        raise ValueError("sentinel_replay_entries must be a nonnegative integer")
    output_root = Path(output_root)
    receipt_path = _safe_object_path(
        output_root,
        receipt_object_path,
        field="receipt_object_path",
    )
    if not receipt_path.is_file() or _file_sha256(receipt_path) != _require_sha256(
        expected_receipt_file_sha256,
        field="expected_receipt_file_sha256",
    ):
        raise SemanticActive8ChunkCacheError("semantic chunk receipt bytes disagree")
    receipt_bytes = receipt_path.read_bytes()
    try:
        receipt = json.loads(receipt_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticActive8ChunkCacheError(
            f"semantic chunk receipt is unreadable: {receipt_path}"
        ) from error
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_FIELDS:
        raise SemanticActive8ChunkCacheError("semantic chunk receipt fields disagree")
    if _canonical_json_bytes(receipt, newline=True) != receipt_bytes:
        raise SemanticActive8ChunkCacheError(
            "semantic chunk receipt is not canonical JSON"
        )
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if (
        receipt.get("schema") != CHUNK_RECEIPT_SCHEMA
        or receipt.get("schema_version") != CHUNK_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != CHUNK_RECEIPT_STATUS
        or receipt.get("training_authorized") is not False
        or receipt.get("active8_decision") is not None
        or receipt.get("receipt_sha256") != _canonical_sha256(receipt_body)
    ):
        raise SemanticActive8ChunkCacheError(
            "semantic chunk receipt crosses its mechanical boundary"
        )
    source_identity = _validate_source_identity(receipt["source_identity"])
    _validate_receipt_range(receipt, source_identity=source_identity)
    if source_identity["semantic_shard_sha256"] != _require_sha256(
        expected_source_shard_sha256,
        field="expected_source_shard_sha256",
    ):
        raise SemanticActive8ChunkCacheError("semantic chunk source shard disagrees")
    if source_identity["semantic_manifest_physical_sha256"] != _require_sha256(
        expected_source_manifest_sha256,
        field="expected_source_manifest_sha256",
    ):
        raise SemanticActive8ChunkCacheError("semantic chunk source manifest disagrees")

    def rows() -> Iterable[AddressedPackedTrace]:
        for local_index, (_, record, trace, states) in enumerate(
            _iter_validated_chunk_rows(
                output_root,
                receipt,
                sentinel_replay_entries=sentinel_replay_entries,
            )
        ):
            global_index = receipt["entry_start"] + local_index
            canonical_keys = record["canonical_state_keys"]
            yield AddressedPackedTrace(
                address=PackedTraceAddress(
                    packed_shard_content_sha256=source_identity[
                        "semantic_shard_sha256"
                    ],
                    packed_shard_name=SHARD_FILENAME,
                    entry_index=global_index,
                    trace_id=str(record["trace_id"]),
                    layer=str(source_identity["data_lane"]),
                    partition=str(source_identity["split"]),
                    source_key=str(canonical_keys[0]),
                    target_key=str(canonical_keys[-1]),
                    path_length=int(record["path_length"]),
                ),
                trace=trace,
                path=PackedTraceProgress(trace, states),
            )

    return rows()


__all__ = [
    "CACHE_SCHEMA",
    "CACHE_SCHEMA_VERSION",
    "CACHE_STATUS",
    "CHUNK_RECEIPT_SCHEMA",
    "CHUNK_RECEIPT_SCHEMA_VERSION",
    "DEFAULT_TARGET_ROWS_PER_CHUNK",
    "SemanticActive8ChunkCacheError",
    "build_semantic_active8_chunk_cache",
    "load_semantic_active8_chunk_cache",
    "read_semantic_active8_chunk",
    "semantic_active8_chunk_cache_builder_identity",
]
