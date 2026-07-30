"""Immutable indexed successor-fiber storage without whole-shard materialization.

The canonical-JSON cache remains the correctness oracle. This backend stores
the same validated support coordinates in one read-only SQLite file per packed
shard and resolves rows by ``(entry_index, progress_index)``. Opening a shard
verifies its frozen byte SHA-256, semantic content identity, exact provenance,
schema, and census. A fully validated parent may transfer an identity-bound
receipt so trusted workers recheck immutable schema and metadata without
repeating the byte or row scans. A lookup decodes one compact row and never
invokes RDKit, the executor, or the support compiler.

This module does not authorize full-corpus training. That decision belongs to
the frozen inventory after representative throughput/RSS benchmarks pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
import tempfile
from collections import OrderedDict
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.data.packed_trace_store import PackedTraceAddress
from compose_v4.data.successor_fiber_cache import (
    DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheError,
    SuccessorFiberCacheLimits,
    SuccessorFiberCacheProvenance,
    SuccessorFiberCacheRecord,
    canonical_successor_fiber_records,
    validate_successor_fiber_cache_record,
)
from compose_v4.experiments.factorized_successor_training import (
    StateProductiveSupport,
    TeacherSuccessorAlias,
    TeacherSuccessorFiber,
)

INDEXED_SUCCESSOR_FIBER_CACHE_SCHEMA = (
    "compose.data.indexed_successor_fiber_cache"
)
INDEXED_SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION = 2
INDEXED_SUCCESSOR_FIBER_STORAGE_BACKEND = "sqlite_indexed_immutable_v1"
INDEXED_SUCCESSOR_FIBER_APPLICATION_ID = 0x434D5053

_METADATA_FIELDS = {
    "schema",
    "schema_version",
    "storage_backend",
    "content_sha256",
    "provenance",
    "packed_shard_name",
    "layer",
    "partition",
    "record_count",
    "active_trace_count",
    "jump_record_count",
    "terminal_record_count",
    "teacher_alias_count",
    "virtual_alias_count",
    "maximum_alias_count",
}
_TABLES = {"metadata", "records"}
_METADATA_COLUMNS = ("singleton", "payload")
_RECORD_COLUMNS = (
    "entry_index",
    "progress_index",
    "path_length",
    "trace_id",
    "trace_source_key",
    "trace_target_key",
    "source_key",
    "source_state_sha256",
    "target_key",
    "target_state_sha256",
    "aliases",
    "virtual_aliases",
)
_METADATA_TABLE_SQL = """
CREATE TABLE metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    payload BLOB NOT NULL
) WITHOUT ROWID
"""
_RECORDS_TABLE_SQL = """
CREATE TABLE records (
    entry_index INTEGER NOT NULL,
    progress_index INTEGER NOT NULL,
    path_length INTEGER NOT NULL,
    trace_id TEXT NOT NULL,
    trace_source_key TEXT NOT NULL,
    trace_target_key TEXT NOT NULL,
    source_key TEXT NOT NULL,
    source_state_sha256 TEXT NOT NULL,
    target_key TEXT,
    target_state_sha256 TEXT,
    aliases BLOB NOT NULL,
    virtual_aliases BLOB NOT NULL,
    PRIMARY KEY (entry_index, progress_index)
) WITHOUT ROWID
"""


class IndexedSuccessorFiberCacheError(SuccessorFiberCacheError):
    """An indexed cache is malformed, stale, tampered with, or incomplete."""


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
        raise IndexedSuccessorFiberCacheError(
            "indexed cache payload is not finite canonical JSON"
        ) from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _alias_rows(
    aliases: tuple[TeacherSuccessorAlias, ...],
) -> list[list[object]]:
    return [
        [alias.family_name, alias.table_name, list(alias.coordinate)]
        for alias in aliases
    ]


def _encode_aliases(aliases: tuple[TeacherSuccessorAlias, ...]) -> bytes:
    return _canonical_json_bytes(_alias_rows(aliases))


def _decode_aliases(
    encoded: bytes,
    *,
    name: str,
) -> tuple[TeacherSuccessorAlias, ...]:
    try:
        payload = json.loads(
            encoded,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise IndexedSuccessorFiberCacheError(
            f"{name} is not valid finite JSON"
        ) from error
    if not isinstance(payload, list):
        raise IndexedSuccessorFiberCacheError(f"{name} must be an array")
    aliases: list[TeacherSuccessorAlias] = []
    for item in payload:
        if (
            not isinstance(item, list)
            or len(item) != 3
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
            or not isinstance(item[2], list)
        ):
            raise IndexedSuccessorFiberCacheError(
                f"{name} contains a malformed coordinate"
            )
        try:
            aliases.append(
                TeacherSuccessorAlias(
                    family_name=item[0],
                    table_name=item[1],
                    coordinate=tuple(item[2]),
                )
            )
        except (TypeError, ValueError) as error:
            raise IndexedSuccessorFiberCacheError(
                f"{name} contains an invalid alias"
            ) from error
    result = tuple(aliases)
    if result != tuple(sorted(set(result))):
        raise IndexedSuccessorFiberCacheError(
            f"{name} is not sorted and duplicate-free"
        )
    return result


def _record_row(record: SuccessorFiberCacheRecord) -> tuple[object, ...]:
    teacher = record.teacher_fiber
    aliases = () if teacher is None else teacher.aliases
    return (
        record.address.entry_index,
        record.address.progress_index,
        record.address.path_length,
        record.address.trace_id,
        record.address.trace_source_key,
        record.address.trace_target_key,
        record.source_key,
        record.source_state_sha256,
        record.target_key,
        record.target_state_sha256,
        _encode_aliases(aliases),
        _encode_aliases(record.state_support.virtual_aliases),
    )


def _semantic_hash_header(
    *,
    provenance: SuccessorFiberCacheProvenance,
    packed_shard_name: str,
    layer: str,
    partition: str,
) -> dict[str, object]:
    return {
        "schema": INDEXED_SUCCESSOR_FIBER_CACHE_SCHEMA,
        "schema_version": INDEXED_SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION,
        "storage_backend": INDEXED_SUCCESSOR_FIBER_STORAGE_BACKEND,
        "provenance": asdict(provenance),
        "packed_shard_name": packed_shard_name,
        "layer": layer,
        "partition": partition,
    }


def _update_semantic_hash(digest: Any, value: object) -> None:
    encoded = _canonical_json_bytes(value)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _connect_writable(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA page_size=4096")
    connection.execute("PRAGMA auto_vacuum=NONE")
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        f"PRAGMA application_id={INDEXED_SUCCESSOR_FIBER_APPLICATION_ID}"
    )
    connection.execute(
        f"PRAGMA user_version={INDEXED_SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION}"
    )
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(_METADATA_TABLE_SQL)
    connection.execute(_RECORDS_TABLE_SQL)


def _metadata_for_records(
    records: tuple[SuccessorFiberCacheRecord, ...],
    *,
    provenance: SuccessorFiberCacheProvenance,
    content_sha256: str,
) -> dict[str, object]:
    first = records[0].address
    entry_indices = {record.address.entry_index for record in records}
    jump_count = sum(record.teacher_fiber is not None for record in records)
    terminal_count = len(records) - jump_count
    teacher_alias_count = sum(
        0 if record.teacher_fiber is None else len(record.teacher_fiber.aliases)
        for record in records
    )
    virtual_alias_count = sum(
        len(record.state_support.virtual_aliases) for record in records
    )
    maximum_alias_count = max(
        (
            len(record.state_support.virtual_aliases)
            + (
                0
                if record.teacher_fiber is None
                else len(record.teacher_fiber.aliases)
            )
        )
        for record in records
    )
    return {
        **_semantic_hash_header(
            provenance=provenance,
            packed_shard_name=first.packed_shard_name,
            layer=first.layer,
            partition=first.partition,
        ),
        "content_sha256": content_sha256,
        "record_count": len(records),
        "active_trace_count": len(entry_indices),
        "jump_record_count": jump_count,
        "terminal_record_count": terminal_count,
        "teacher_alias_count": teacher_alias_count,
        "virtual_alias_count": virtual_alias_count,
        "maximum_alias_count": maximum_alias_count,
    }


@dataclass(frozen=True)
class IndexedSuccessorFiberCacheMetadata:
    """Frozen identity and census stored inside one indexed shard."""

    content_sha256: str
    file_sha256: str
    file_bytes: int
    provenance: SuccessorFiberCacheProvenance
    packed_shard_name: str
    layer: str
    partition: str
    record_count: int
    active_trace_count: int
    jump_record_count: int
    terminal_record_count: int
    teacher_alias_count: int
    virtual_alias_count: int
    maximum_alias_count: int

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, object],
        *,
        file_sha256: str,
        file_bytes: int,
    ) -> IndexedSuccessorFiberCacheMetadata:
        if set(payload) != _METADATA_FIELDS:
            raise IndexedSuccessorFiberCacheError(
                "indexed cache metadata fields disagree with the schema"
            )
        if payload["schema"] != INDEXED_SUCCESSOR_FIBER_CACHE_SCHEMA:
            raise IndexedSuccessorFiberCacheError(
                "indexed cache uses another schema"
            )
        if (
            payload["schema_version"]
            != INDEXED_SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION
        ):
            raise IndexedSuccessorFiberCacheError(
                "indexed cache uses another schema version"
            )
        if payload["storage_backend"] != INDEXED_SUCCESSOR_FIBER_STORAGE_BACKEND:
            raise IndexedSuccessorFiberCacheError(
                "indexed cache uses another storage backend"
            )
        for name, value in (
            ("content_sha256", payload["content_sha256"]),
            ("file_sha256", file_sha256),
        ):
            if not _is_sha256(value):
                raise IndexedSuccessorFiberCacheError(
                    f"{name} must be a lowercase SHA-256"
                )
        if type(file_bytes) is not int or file_bytes <= 0:
            raise IndexedSuccessorFiberCacheError(
                "indexed cache byte count must be positive"
            )
        provenance_payload = payload["provenance"]
        if not isinstance(provenance_payload, dict):
            raise IndexedSuccessorFiberCacheError(
                "indexed cache provenance must be an object"
            )
        try:
            provenance = SuccessorFiberCacheProvenance(**provenance_payload)
        except (TypeError, ValueError) as error:
            raise IndexedSuccessorFiberCacheError(
                "indexed cache provenance is invalid"
            ) from error
        counts: dict[str, int] = {}
        for name in (
            "record_count",
            "active_trace_count",
            "jump_record_count",
            "terminal_record_count",
            "teacher_alias_count",
            "virtual_alias_count",
            "maximum_alias_count",
        ):
            value = payload[name]
            if type(value) is not int or value < 0:
                raise IndexedSuccessorFiberCacheError(
                    f"indexed cache {name} must be nonnegative"
                )
            counts[name] = value
        if counts["record_count"] <= 0 or counts["active_trace_count"] <= 0:
            raise IndexedSuccessorFiberCacheError(
                "indexed cache census cannot be empty"
            )
        if (
            counts["jump_record_count"] + counts["terminal_record_count"]
            != counts["record_count"]
            or counts["terminal_record_count"]
            != counts["active_trace_count"]
        ):
            raise IndexedSuccessorFiberCacheError(
                "indexed cache jump/terminal census is inconsistent"
            )
        for name in ("packed_shard_name", "layer", "partition"):
            value = payload[name]
            if not isinstance(value, str) or not value:
                raise IndexedSuccessorFiberCacheError(
                    f"indexed cache {name} must be nonempty"
                )
        packed_shard_name = str(payload["packed_shard_name"])
        if Path(packed_shard_name).name != packed_shard_name:
            raise IndexedSuccessorFiberCacheError(
                "indexed cache packed_shard_name must be a basename"
            )
        return cls(
            content_sha256=str(payload["content_sha256"]),
            file_sha256=file_sha256,
            file_bytes=file_bytes,
            provenance=provenance,
            packed_shard_name=packed_shard_name,
            layer=str(payload["layer"]),
            partition=str(payload["partition"]),
            **counts,
        )


def _decode_record_row(
    row: tuple[object, ...],
    *,
    metadata: IndexedSuccessorFiberCacheMetadata,
) -> SuccessorFiberCacheRecord:
    (
        entry_index,
        progress_index,
        path_length,
        trace_id,
        trace_source_key,
        trace_target_key,
        source_key,
        source_state_sha256,
        target_key,
        target_state_sha256,
        aliases_blob,
        virtual_aliases_blob,
    ) = row
    try:
        address = SuccessorFiberCacheAddress(
            packed_shard_content_sha256=(
                metadata.provenance.packed_shard_content_sha256
            ),
            packed_shard_name=metadata.packed_shard_name,
            entry_index=entry_index,
            layer=metadata.layer,
            partition=metadata.partition,
            trace_id=trace_id,
            trace_source_key=trace_source_key,
            trace_target_key=trace_target_key,
            progress_index=progress_index,
            path_length=path_length,
        )
        virtual_aliases = _decode_aliases(
            virtual_aliases_blob,
            name="indexed virtual aliases",
        )
        support = StateProductiveSupport(
            source_key=source_key,
            source_state_sha256=source_state_sha256,
            virtual_aliases=virtual_aliases,
        )
        aliases = _decode_aliases(
            aliases_blob,
            name="indexed teacher aliases",
        )
        if target_key is None:
            if target_state_sha256 is not None or aliases:
                raise IndexedSuccessorFiberCacheError(
                    "indexed terminal row carries teacher successor data"
                )
            teacher_fiber = None
        else:
            if not isinstance(target_state_sha256, str):
                raise IndexedSuccessorFiberCacheError(
                    "indexed jump row lacks its target state digest"
                )
            teacher_fiber = TeacherSuccessorFiber(
                source_key=source_key,
                target_key=target_key,
                target_state_sha256=target_state_sha256,
                aliases=aliases,
                state_support=support,
            )
        return SuccessorFiberCacheRecord(
            address=address,
            state_support=support,
            teacher_fiber=teacher_fiber,
        )
    except IndexedSuccessorFiberCacheError:
        raise
    except (TypeError, ValueError) as error:
        raise IndexedSuccessorFiberCacheError(
            "indexed cache row is invalid"
        ) from error


def write_indexed_successor_fiber_cache(
    path: Path,
    records: tuple[SuccessorFiberCacheRecord, ...],
    *,
    provenance: SuccessorFiberCacheProvenance,
    limits: SuccessorFiberCacheLimits = DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
) -> IndexedSuccessorFiberCacheMetadata:
    """Atomically write one deterministic, complete packed-shard derivative."""

    ordered = canonical_successor_fiber_records(
        records,
        provenance=provenance,
        limits=limits,
    )
    if not ordered:
        raise IndexedSuccessorFiberCacheError(
            "indexed successor cache cannot be empty"
        )
    envelopes = {
        (
            record.address.packed_shard_name,
            record.address.layer,
            record.address.partition,
        )
        for record in ordered
    }
    if len(envelopes) != 1:
        raise IndexedSuccessorFiberCacheError(
            "one indexed cache must derive from one packed-shard lane"
        )
    packed_shard_name, layer, partition = next(iter(envelopes))
    semantic_digest = hashlib.sha256()
    _update_semantic_hash(
        semantic_digest,
        _semantic_hash_header(
            provenance=provenance,
            packed_shard_name=packed_shard_name,
            layer=layer,
            partition=partition,
        ),
    )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    metadata: dict[str, object] | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".sqlite.tmp",
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)
        connection = _connect_writable(temporary)
        try:
            _create_schema(connection)
            for record in ordered:
                row = _record_row(record)
                _update_semantic_hash(
                    semantic_digest,
                    [
                        *row[:10],
                        json.loads(row[10]),
                        json.loads(row[11]),
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO records
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )
            content_sha256 = semantic_digest.hexdigest()
            metadata = _metadata_for_records(
                ordered,
                provenance=provenance,
                content_sha256=content_sha256,
            )
            connection.execute(
                "INSERT INTO metadata VALUES (1, ?)",
                (_canonical_json_bytes(metadata),),
            )
            connection.commit()
        finally:
            connection.close()
        if temporary.stat().st_size > limits.max_file_bytes:
            raise IndexedSuccessorFiberCacheError(
                "indexed successor cache exceeds the file bound"
            )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if _sha256_file(temporary) != _sha256_file(destination):
                raise FileExistsError(
                    "indexed successor cache already exists with different "
                    f"bytes: {destination}"
                ) from None
        if metadata is None:
            raise RuntimeError("indexed cache metadata was not constructed")
        file_sha256 = _sha256_file(destination)
        return IndexedSuccessorFiberCacheMetadata.from_payload(
            metadata,
            file_sha256=file_sha256,
            file_bytes=destination.stat().st_size,
        )
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _read_metadata(connection: sqlite3.Connection) -> dict[str, object]:
    row = connection.execute(
        "SELECT payload FROM metadata WHERE singleton = 1"
    ).fetchone()
    if row is None:
        raise IndexedSuccessorFiberCacheError(
            "indexed cache lacks its singleton metadata row"
        )
    try:
        payload = json.loads(
            row[0],
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise IndexedSuccessorFiberCacheError(
            "indexed cache metadata is not valid finite JSON"
        ) from error
    if not isinstance(payload, dict):
        raise IndexedSuccessorFiberCacheError(
            "indexed cache metadata must be an object"
        )
    return payload


def _validate_sqlite_schema(
    connection: sqlite3.Connection,
    *,
    run_integrity_check: bool = True,
) -> None:
    application_id = int(
        connection.execute("PRAGMA application_id").fetchone()[0]
    )
    user_version = int(
        connection.execute("PRAGMA user_version").fetchone()[0]
    )
    if application_id != INDEXED_SUCCESSOR_FIBER_APPLICATION_ID:
        raise IndexedSuccessorFiberCacheError(
            "SQLite application_id does not identify a COMPOSE cache"
        )
    if user_version != INDEXED_SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION:
        raise IndexedSuccessorFiberCacheError(
            "SQLite user_version does not match the indexed-cache schema"
        )
    if run_integrity_check:
        quick_check = connection.execute("PRAGMA quick_check(1)").fetchone()
        if quick_check != ("ok",):
            raise IndexedSuccessorFiberCacheError(
                "indexed cache fails SQLite structural integrity"
            )
    objects = tuple(
        (str(row[0]), str(row[1]), row[2])
        for row in connection.execute(
            """
            SELECT type, name, sql FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        )
    )
    if {(kind, name) for kind, name, _ in objects} != {
        ("table", name) for name in _TABLES
    }:
        raise IndexedSuccessorFiberCacheError(
            "indexed cache has missing or unexpected schema objects"
        )
    expected_sql = {
        "metadata": " ".join(_METADATA_TABLE_SQL.split()),
        "records": " ".join(_RECORDS_TABLE_SQL.split()),
    }
    for kind, name, raw_sql in objects:
        if (
            kind != "table"
            or not isinstance(raw_sql, str)
            or " ".join(raw_sql.split()) != expected_sql[name]
        ):
            raise IndexedSuccessorFiberCacheError(
                f"indexed cache table {name!r} has another exact DDL"
            )
    for table, expected in (
        ("metadata", _METADATA_COLUMNS),
        ("records", _RECORD_COLUMNS),
    ):
        observed = tuple(
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if observed != expected:
            raise IndexedSuccessorFiberCacheError(
                f"indexed cache table {table!r} has another column schema"
            )


def _audit_indexed_records(
    connection: sqlite3.Connection,
    *,
    metadata: IndexedSuccessorFiberCacheMetadata,
    limits: SuccessorFiberCacheLimits,
) -> tuple[str, dict[str, int]]:
    """Recompute semantic identity, census, and complete trace chains."""

    semantic_digest = hashlib.sha256()
    _update_semantic_hash(
        semantic_digest,
        _semantic_hash_header(
            provenance=metadata.provenance,
            packed_shard_name=metadata.packed_shard_name,
            layer=metadata.layer,
            partition=metadata.partition,
        ),
    )
    record_count = 0
    active_trace_count = 0
    jump_record_count = 0
    terminal_record_count = 0
    teacher_alias_count = 0
    virtual_alias_count = 0
    maximum_alias_count = 0
    previous: SuccessorFiberCacheRecord | None = None
    cursor = connection.execute(
        """
        SELECT entry_index, progress_index, path_length, trace_id,
               trace_source_key, trace_target_key, source_key,
               source_state_sha256, target_key, target_state_sha256,
               aliases, virtual_aliases
        FROM records
        ORDER BY entry_index, progress_index
        """
    )
    for raw_row in cursor:
        row = tuple(raw_row)
        record = _decode_record_row(row, metadata=metadata)
        try:
            validate_successor_fiber_cache_record(
                record,
                provenance=metadata.provenance,
                limits=limits,
            )
        except SuccessorFiberCacheError as error:
            raise IndexedSuccessorFiberCacheError(
                "indexed cache row exceeds the shared validation contract: "
                f"{error}"
            ) from error
        canonical_row = _record_row(record)
        _update_semantic_hash(
            semantic_digest,
            [
                *canonical_row[:10],
                json.loads(canonical_row[10]),
                json.loads(canonical_row[11]),
            ],
        )
        address = record.address
        if previous is None or (
            address.entry_index != previous.address.entry_index
        ):
            if previous is not None and not previous.address.is_terminal:
                raise IndexedSuccessorFiberCacheError(
                    "indexed cache trace ends before its terminal progress"
                )
            if address.progress_index != 0:
                raise IndexedSuccessorFiberCacheError(
                    "indexed cache trace does not begin at progress zero"
                )
            active_trace_count += 1
        else:
            if (
                address.trace_key != previous.address.trace_key
                or address.progress_index
                != previous.address.progress_index + 1
            ):
                raise IndexedSuccessorFiberCacheError(
                    "indexed cache trace address/progress chain is inconsistent"
                )
            if (
                previous.target_key != record.source_key
                or previous.target_state_sha256
                != record.source_state_sha256
            ):
                raise IndexedSuccessorFiberCacheError(
                    "indexed cache target/source state chain is inconsistent"
                )
        if (
            address.progress_index == 0
            and record.source_key != address.trace_source_key
        ):
            raise IndexedSuccessorFiberCacheError(
                "indexed cache first state disagrees with its packed source"
            )
        if (
            address.is_terminal
            and record.source_key != address.trace_target_key
        ):
            raise IndexedSuccessorFiberCacheError(
                "indexed cache terminal state disagrees with its packed target"
            )
        teacher_count = (
            0
            if record.teacher_fiber is None
            else len(record.teacher_fiber.aliases)
        )
        virtual_count = len(record.state_support.virtual_aliases)
        record_count += 1
        jump_record_count += int(record.teacher_fiber is not None)
        terminal_record_count += int(record.teacher_fiber is None)
        teacher_alias_count += teacher_count
        virtual_alias_count += virtual_count
        maximum_alias_count = max(
            maximum_alias_count,
            teacher_count + virtual_count,
        )
        previous = record
    if previous is None or not previous.address.is_terminal:
        raise IndexedSuccessorFiberCacheError(
            "indexed cache is empty or lacks its final terminal progress"
        )
    census = {
        "record_count": record_count,
        "active_trace_count": active_trace_count,
        "jump_record_count": jump_record_count,
        "terminal_record_count": terminal_record_count,
        "teacher_alias_count": teacher_alias_count,
        "virtual_alias_count": virtual_alias_count,
        "maximum_alias_count": maximum_alias_count,
    }
    return semantic_digest.hexdigest(), census


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri_path = quote(str(path.resolve()), safe="/")
    connection = sqlite3.connect(
        f"file:{uri_path}?mode=ro&immutable=1",
        uri=True,
    )
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA trusted_schema=OFF")
    return connection


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def inspect(cls, path: Path) -> _FileIdentity:
        stat = path.stat()
        return cls(
            device=stat.st_dev,
            inode=stat.st_ino,
            size=stat.st_size,
            modified_ns=stat.st_mtime_ns,
            changed_ns=stat.st_ctime_ns,
        )


def _require_file_identity(path: Path, expected: _FileIdentity) -> None:
    try:
        observed = _FileIdentity.inspect(path)
    except OSError as error:
        raise IndexedSuccessorFiberCacheError(
            "indexed cache file disappeared after verification"
        ) from error
    if observed != expected:
        raise IndexedSuccessorFiberCacheError(
            "indexed cache file identity changed after verification"
        )


@dataclass(frozen=True)
class IndexedSuccessorFiberCacheOpenReceipt:
    """Trusted intra-launch proof that a parent fully validated one shard.

    The receipt is deliberately stronger than a path plus hashes: it binds the
    resolved absolute path, the exact filesystem object observed around the
    parent's byte and semantic audit, the expected scientific identities, the
    validated SQLite metadata payload, and the resource limits under which all
    rows were checked. It is a process-transfer capability for workers in the
    same trusted launch, not a persistent substitute for full validation.
    """

    absolute_path: str
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int
    expected_file_sha256: str
    expected_content_sha256: str
    expected_provenance: SuccessorFiberCacheProvenance
    validated_metadata: IndexedSuccessorFiberCacheMetadata
    validated_metadata_payload: bytes
    validated_limits: SuccessorFiberCacheLimits
    validated_packed_entry_count: int | None = None
    validated_excluded_entry_indices: tuple[int, ...] = ()

    @property
    def file_identity(self) -> _FileIdentity:
        return _FileIdentity(
            device=self.device,
            inode=self.inode,
            size=self.size,
            modified_ns=self.modified_ns,
            changed_ns=self.changed_ns,
        )


def _validate_open_receipt_contract(
    receipt: IndexedSuccessorFiberCacheOpenReceipt,
    *,
    path: Path,
    expected_provenance: SuccessorFiberCacheProvenance,
    expected_content_sha256: str,
    expected_file_sha256: str,
    expected_file_bytes: int,
    limits: SuccessorFiberCacheLimits,
) -> tuple[Path, _FileIdentity]:
    if not isinstance(receipt, IndexedSuccessorFiberCacheOpenReceipt):
        raise TypeError(
            "parent-validated receipt must be an indexed-cache open receipt"
        )
    receipt_path = Path(receipt.absolute_path)
    if not receipt_path.is_absolute():
        raise IndexedSuccessorFiberCacheError(
            "parent-validated receipt path is not absolute"
        )
    try:
        source = Path(path).resolve(strict=True)
    except OSError as error:
        raise IndexedSuccessorFiberCacheError(
            "indexed cache file disappeared before worker open"
        ) from error
    if source != receipt_path:
        raise IndexedSuccessorFiberCacheError(
            "parent-validated receipt names another absolute cache path"
        )
    if (
        receipt.expected_provenance != expected_provenance
        or receipt.expected_content_sha256 != expected_content_sha256
        or receipt.expected_file_sha256 != expected_file_sha256
        or receipt.size != expected_file_bytes
        or receipt.validated_limits != limits
    ):
        raise IndexedSuccessorFiberCacheError(
            "parent-validated receipt disagrees with worker open expectations"
        )
    metadata = receipt.validated_metadata
    if (
        metadata.provenance != receipt.expected_provenance
        or metadata.content_sha256 != receipt.expected_content_sha256
        or metadata.file_sha256 != receipt.expected_file_sha256
        or metadata.file_bytes != receipt.size
    ):
        raise IndexedSuccessorFiberCacheError(
            "parent-validated receipt metadata disagrees with its identities"
        )
    try:
        decoded_payload = json.loads(
            receipt.validated_metadata_payload,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise IndexedSuccessorFiberCacheError(
            "parent-validated receipt metadata is not finite JSON"
        ) from error
    if not isinstance(decoded_payload, dict):
        raise IndexedSuccessorFiberCacheError(
            "parent-validated receipt metadata must be an object"
        )
    reconstructed = IndexedSuccessorFiberCacheMetadata.from_payload(
        decoded_payload,
        file_sha256=receipt.expected_file_sha256,
        file_bytes=receipt.size,
    )
    if reconstructed != metadata:
        raise IndexedSuccessorFiberCacheError(
            "parent-validated receipt metadata payload is inconsistent"
        )
    if receipt.validated_packed_entry_count is None:
        if receipt.validated_excluded_entry_indices:
            raise IndexedSuccessorFiberCacheError(
                "parent-validated receipt has exclusions without an entry census"
            )
    else:
        packed_entry_count = receipt.validated_packed_entry_count
        excluded = receipt.validated_excluded_entry_indices
        if (
            type(packed_entry_count) is not int
            or packed_entry_count <= 0
            or excluded != tuple(sorted(set(excluded)))
            or any(
                type(index) is not int
                or not 0 <= index < packed_entry_count
                for index in excluded
            )
        ):
            raise IndexedSuccessorFiberCacheError(
                "parent-validated receipt has an invalid entry census"
            )
    identity = receipt.file_identity
    if (
        any(
            type(value) is not int or value < 0
            for value in (
                identity.device,
                identity.inode,
                identity.size,
                identity.modified_ns,
                identity.changed_ns,
            )
        )
        or identity.size <= 0
    ):
        raise IndexedSuccessorFiberCacheError(
            "parent-validated receipt has an invalid file identity"
        )
    return source, identity


def _connect_from_parent_validated_receipt(
    receipt: IndexedSuccessorFiberCacheOpenReceipt,
    *,
    path: Path,
    expected_provenance: SuccessorFiberCacheProvenance,
    expected_content_sha256: str,
    expected_file_sha256: str,
    expected_file_bytes: int,
    limits: SuccessorFiberCacheLimits,
) -> tuple[
    sqlite3.Connection,
    IndexedSuccessorFiberCacheMetadata,
    Path,
    _FileIdentity,
]:
    """Lightweight worker open; never hashes or audits the records table."""

    source, identity = _validate_open_receipt_contract(
        receipt,
        path=path,
        expected_provenance=expected_provenance,
        expected_content_sha256=expected_content_sha256,
        expected_file_sha256=expected_file_sha256,
        expected_file_bytes=expected_file_bytes,
        limits=limits,
    )
    _require_file_identity(source, identity)
    try:
        connection = _connect_read_only(source)
        try:
            _require_file_identity(source, identity)
            if int(connection.execute("PRAGMA query_only").fetchone()[0]) != 1:
                raise IndexedSuccessorFiberCacheError(
                    "worker SQLite connection is not query-only"
                )
            database_path = connection.execute("PRAGMA database_list").fetchone()
            if (
                database_path is None
                or len(database_path) < 3
                or Path(str(database_path[2])).resolve(strict=True) != source
            ):
                raise IndexedSuccessorFiberCacheError(
                    "worker SQLite connection opened another cache path"
                )
            _validate_sqlite_schema(
                connection,
                run_integrity_check=False,
            )
            payload = _read_metadata(connection)
            if (
                _canonical_json_bytes(payload)
                != receipt.validated_metadata_payload
            ):
                raise IndexedSuccessorFiberCacheError(
                    "worker SQLite metadata differs from the parent validation"
                )
            metadata = IndexedSuccessorFiberCacheMetadata.from_payload(
                payload,
                file_sha256=expected_file_sha256,
                file_bytes=expected_file_bytes,
            )
            if metadata != receipt.validated_metadata:
                raise IndexedSuccessorFiberCacheError(
                    "worker SQLite metadata object differs from its receipt"
                )
            _require_file_identity(source, identity)
        except Exception:
            connection.close()
            raise
    except IndexedSuccessorFiberCacheError:
        raise
    except (OSError, sqlite3.DatabaseError) as error:
        raise IndexedSuccessorFiberCacheError(
            "parent-validated indexed cache failed lightweight worker open"
        ) from error
    return connection, metadata, source, identity


class IndexedSuccessorFiberCache:
    """Lazy per-process read-only handle with a bounded decoded-row LRU."""

    def __init__(
        self,
        path: Path,
        *,
        expected_provenance: SuccessorFiberCacheProvenance,
        expected_content_sha256: str,
        expected_file_sha256: str,
        expected_file_bytes: int,
        decoded_row_cache_size: int = 4096,
        limits: SuccessorFiberCacheLimits = DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
        parent_validated_open_receipt: (
            IndexedSuccessorFiberCacheOpenReceipt | None
        ) = None,
    ) -> None:
        try:
            source = Path(path).resolve(strict=True)
        except OSError as error:
            raise IndexedSuccessorFiberCacheError(
                f"indexed cache file does not exist: {path}"
            ) from error
        if not source.is_file():
            raise IndexedSuccessorFiberCacheError(
                f"indexed cache path is not a file: {source}"
            )
        if not _is_sha256(expected_content_sha256) or not _is_sha256(
            expected_file_sha256
        ):
            raise ValueError(
                "expected indexed-cache identities must be SHA-256 digests"
            )
        if (
            type(expected_file_bytes) is not int
            or expected_file_bytes <= 0
            or expected_file_bytes > limits.max_file_bytes
        ):
            raise ValueError("expected indexed-cache byte count is invalid")
        if (
            type(decoded_row_cache_size) is not int
            or decoded_row_cache_size <= 0
        ):
            raise ValueError("decoded-row cache size must be positive")
        if parent_validated_open_receipt is not None:
            connection, metadata, source, verified_identity = (
                _connect_from_parent_validated_receipt(
                    parent_validated_open_receipt,
                    path=source,
                    expected_provenance=expected_provenance,
                    expected_content_sha256=expected_content_sha256,
                    expected_file_sha256=expected_file_sha256,
                    expected_file_bytes=expected_file_bytes,
                    limits=limits,
                )
            )
            self.path = source
            self.metadata = metadata
            self.decoded_row_cache_size = decoded_row_cache_size
            self._limits = limits
            self._file_identity = verified_identity
            self._open_receipt = parent_validated_open_receipt
            self._connection = connection
            self._connection_pid = os.getpid()
            self._runtime_pid = os.getpid()
            self._decoded = OrderedDict()
            return
        verified_identity = _FileIdentity.inspect(source)
        if verified_identity.size != expected_file_bytes:
            raise IndexedSuccessorFiberCacheError(
                "indexed cache byte count disagrees with the frozen inventory"
            )
        observed_file_sha256 = _sha256_file(source)
        _require_file_identity(source, verified_identity)
        if observed_file_sha256 != expected_file_sha256:
            raise IndexedSuccessorFiberCacheError(
                "indexed cache byte SHA-256 disagrees with the frozen inventory"
            )
        try:
            connection = _connect_read_only(source)
            try:
                _require_file_identity(source, verified_identity)
                _validate_sqlite_schema(connection)
                payload = _read_metadata(connection)
                metadata_payload = _canonical_json_bytes(payload)
                metadata = IndexedSuccessorFiberCacheMetadata.from_payload(
                    payload,
                    file_sha256=observed_file_sha256,
                    file_bytes=expected_file_bytes,
                )
                if metadata.provenance != expected_provenance:
                    raise IndexedSuccessorFiberCacheError(
                        "indexed cache provenance disagrees with the frozen inventory"
                    )
                if metadata.content_sha256 != expected_content_sha256:
                    raise IndexedSuccessorFiberCacheError(
                        "indexed cache semantic content hash disagrees with the inventory"
                    )
                if (
                    metadata.record_count > limits.max_records
                    or metadata.teacher_alias_count
                    + metadata.virtual_alias_count
                    > limits.max_total_aliases
                    or metadata.maximum_alias_count
                    > limits.max_aliases_per_record
                ):
                    raise IndexedSuccessorFiberCacheError(
                        "indexed cache census exceeds configured resource limits"
                    )
                audited_content_sha256, audited_census = (
                    _audit_indexed_records(
                        connection,
                        metadata=metadata,
                        limits=limits,
                    )
                )
                _require_file_identity(source, verified_identity)
            finally:
                connection.close()
        except sqlite3.DatabaseError as error:
            raise IndexedSuccessorFiberCacheError(
                "indexed cache is not a readable SQLite artifact"
            ) from error
        if audited_content_sha256 != metadata.content_sha256:
            raise IndexedSuccessorFiberCacheError(
                "indexed cache recomputed semantic hash disagrees with metadata"
            )
        expected_census = {
            "record_count": metadata.record_count,
            "active_trace_count": metadata.active_trace_count,
            "jump_record_count": metadata.jump_record_count,
            "terminal_record_count": metadata.terminal_record_count,
            "teacher_alias_count": metadata.teacher_alias_count,
            "virtual_alias_count": metadata.virtual_alias_count,
            "maximum_alias_count": metadata.maximum_alias_count,
        }
        if audited_census != expected_census:
            raise IndexedSuccessorFiberCacheError(
                "indexed cache recomputed census disagrees with metadata"
            )
        self.path = source
        self.metadata = metadata
        self.decoded_row_cache_size = decoded_row_cache_size
        self._limits = limits
        self._file_identity = verified_identity
        self._open_receipt = IndexedSuccessorFiberCacheOpenReceipt(
            absolute_path=str(source),
            device=verified_identity.device,
            inode=verified_identity.inode,
            size=verified_identity.size,
            modified_ns=verified_identity.modified_ns,
            changed_ns=verified_identity.changed_ns,
            expected_file_sha256=expected_file_sha256,
            expected_content_sha256=expected_content_sha256,
            expected_provenance=expected_provenance,
            validated_metadata=metadata,
            validated_metadata_payload=metadata_payload,
            validated_limits=limits,
        )
        self._connection: sqlite3.Connection | None = None
        self._connection_pid: int | None = None
        self._runtime_pid: int | None = os.getpid()
        self._decoded: OrderedDict[
            tuple[int, int],
            SuccessorFiberCacheRecord,
        ] = OrderedDict()

    @property
    def content_sha256(self) -> str:
        return self.metadata.content_sha256

    @property
    def parent_validated_open_receipt(
        self,
    ) -> IndexedSuccessorFiberCacheOpenReceipt:
        """Return the trusted same-launch worker-open capability."""

        return self._open_receipt

    def require_entry_census(
        self,
        *,
        packed_entry_count: int,
        excluded_entry_indices: tuple[int, ...] = (),
    ) -> None:
        """Require the exact packed-entry set declared by a frozen inventory."""

        if type(packed_entry_count) is not int or packed_entry_count <= 0:
            raise ValueError("packed_entry_count must be positive")
        if (
            not isinstance(excluded_entry_indices, tuple)
            or excluded_entry_indices
            != tuple(sorted(set(excluded_entry_indices)))
            or any(
                type(index) is not int
                or not 0 <= index < packed_entry_count
                for index in excluded_entry_indices
            )
        ):
            raise ValueError(
                "excluded_entry_indices must be sorted unique in-range integers"
            )
        expected_active_count = (
            packed_entry_count - len(excluded_entry_indices)
        )
        if self.metadata.active_trace_count != expected_active_count:
            raise IndexedSuccessorFiberCacheError(
                "indexed cache active-trace count disagrees with the packed census"
            )
        receipt = self._open_receipt
        if receipt.validated_packed_entry_count is not None:
            if (
                receipt.validated_packed_entry_count != packed_entry_count
                or receipt.validated_excluded_entry_indices
                != excluded_entry_indices
            ):
                raise IndexedSuccessorFiberCacheError(
                    "requested packed-entry census omits or invents indices "
                    "relative to the parent-validated receipt"
                )
            return
        connection = self._connection_for_process()
        try:
            row = connection.execute(
                """
                SELECT COUNT(DISTINCT entry_index),
                       MIN(entry_index),
                       MAX(entry_index)
                FROM records
                """
            ).fetchone()
            if row is None:
                raise IndexedSuccessorFiberCacheError(
                    "indexed cache cannot inspect its entry census"
                )
            distinct_count, minimum, maximum = row
            excluded_present = tuple(
                index
                for index in excluded_entry_indices
                if connection.execute(
                    """
                    SELECT 1 FROM records
                    WHERE entry_index = ? LIMIT 1
                    """,
                    (index,),
                ).fetchone()
                is not None
            )
        except sqlite3.DatabaseError as error:
            raise IndexedSuccessorFiberCacheError(
                "indexed-cache entry census failed"
            ) from error
        if (
            int(distinct_count) != expected_active_count
            or minimum is None
            or maximum is None
            or int(minimum) < 0
            or int(maximum) >= packed_entry_count
            or excluded_present
        ):
            raise IndexedSuccessorFiberCacheError(
                "indexed cache omits or invents packed entry indices"
            )
        self._open_receipt = replace(
            receipt,
            validated_packed_entry_count=packed_entry_count,
            validated_excluded_entry_indices=excluded_entry_indices,
        )

    def _connection_for_process(self) -> sqlite3.Connection:
        process_id = os.getpid()
        if self._runtime_pid != process_id:
            self.close()
            self._decoded = OrderedDict()
            self._runtime_pid = process_id
        if (
            self._connection is not None
            and self._connection_pid == process_id
        ):
            return self._connection
        self.close()
        connection, metadata, source, identity = (
            _connect_from_parent_validated_receipt(
                self._open_receipt,
                path=self.path,
                expected_provenance=self.metadata.provenance,
                expected_content_sha256=self.metadata.content_sha256,
                expected_file_sha256=self.metadata.file_sha256,
                expected_file_bytes=self.metadata.file_bytes,
                limits=self._limits,
            )
        )
        if (
            source != self.path
            or identity != self._file_identity
            or metadata != self.metadata
        ):
            connection.close()
            raise IndexedSuccessorFiberCacheError(
                "worker cache open disagrees with the validated cache object"
            )
        self._connection = connection
        self._connection_pid = process_id
        return self._connection

    def _decode_record(
        self,
        row: tuple[object, ...],
    ) -> SuccessorFiberCacheRecord:
        return _decode_record_row(row, metadata=self.metadata)

    def require_packed_address(
        self,
        address: PackedTraceAddress,
        *,
        progress_index: int,
    ) -> SuccessorFiberCacheRecord:
        """Return one exact row and require its complete packed identity."""

        if type(progress_index) is not int or not (
            0 <= progress_index <= address.path_length
        ):
            raise ValueError("progress_index lies outside the packed trace")
        if self._runtime_pid != os.getpid():
            # Forked processes must discard inherited decoded rows and prove
            # the immutable SQLite identity/metadata before serving anything.
            self._connection_for_process()
        _require_file_identity(self.path, self._file_identity)
        metadata = self.metadata
        if (
            address.packed_shard_content_sha256
            != metadata.provenance.packed_shard_content_sha256
            or address.packed_shard_name != metadata.packed_shard_name
            or address.layer != metadata.layer
            or address.partition != metadata.partition
        ):
            raise IndexedSuccessorFiberCacheError(
                "packed address disagrees with the indexed-cache shard"
            )
        key = (address.entry_index, progress_index)
        cached = self._decoded.get(key)
        if cached is not None:
            self._decoded.move_to_end(key)
            record = cached
        else:
            connection = self._connection_for_process()
            try:
                row = connection.execute(
                    """
                    SELECT entry_index, progress_index, path_length, trace_id,
                           trace_source_key, trace_target_key, source_key,
                           source_state_sha256, target_key, target_state_sha256,
                           aliases, virtual_aliases
                    FROM records
                    WHERE entry_index = ? AND progress_index = ?
                    """,
                    key,
                ).fetchone()
            except sqlite3.DatabaseError as error:
                raise IndexedSuccessorFiberCacheError(
                    "indexed-cache lookup failed"
                ) from error
            if row is None:
                raise IndexedSuccessorFiberCacheError(
                    "indexed cache has no exact trace-progress row"
                )
            record = self._decode_record(row)
            self._decoded[key] = record
            self._decoded.move_to_end(key)
            while len(self._decoded) > self.decoded_row_cache_size:
                self._decoded.popitem(last=False)
        expected = SuccessorFiberCacheAddress.from_packed_trace(
            address,
            progress_index=progress_index,
        )
        if record.address != expected:
            raise IndexedSuccessorFiberCacheError(
                "indexed cache record disagrees with the full packed address"
            )
        _require_file_identity(self.path, self._file_identity)
        return record

    def require(
        self,
        address: PackedTraceAddress,
        *,
        progress_index: int,
        source_state: MolecularGraph,
    ) -> SuccessorFiberCacheRecord:
        record = self.require_packed_address(
            address,
            progress_index=progress_index,
        )
        if (
            persistent_slot_state_sha256(source_state)
            != record.source_state_sha256
        ):
            raise IndexedSuccessorFiberCacheError(
                "sampled exact state disagrees with its indexed-cache row"
            )
        return record

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
        self._connection = None
        self._connection_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_connection"] = None
        state["_connection_pid"] = None
        state["_runtime_pid"] = None
        state["_decoded"] = OrderedDict()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __del__(self) -> None:
        with suppress(sqlite3.Error):
            self.close()


def open_indexed_successor_fiber_cache(
    path: Path,
    *,
    expected_provenance: SuccessorFiberCacheProvenance,
    expected_content_sha256: str,
    expected_file_sha256: str,
    expected_file_bytes: int,
    decoded_row_cache_size: int = 4096,
    limits: SuccessorFiberCacheLimits = DEFAULT_SUCCESSOR_FIBER_CACHE_LIMITS,
    parent_validated_open_receipt: (
        IndexedSuccessorFiberCacheOpenReceipt | None
    ) = None,
) -> IndexedSuccessorFiberCache:
    return IndexedSuccessorFiberCache(
        path,
        expected_provenance=expected_provenance,
        expected_content_sha256=expected_content_sha256,
        expected_file_sha256=expected_file_sha256,
        expected_file_bytes=expected_file_bytes,
        decoded_row_cache_size=decoded_row_cache_size,
        limits=limits,
        parent_validated_open_receipt=parent_validated_open_receipt,
    )


def assert_indexed_cache_is_pickle_safe(
    cache: IndexedSuccessorFiberCache,
) -> None:
    restored = pickle.loads(pickle.dumps(cache))
    if restored._connection is not None or restored._decoded:
        raise IndexedSuccessorFiberCacheError(
            "pickled indexed cache retained process-local state"
        )
    if (
        restored.parent_validated_open_receipt
        != cache.parent_validated_open_receipt
    ):
        raise IndexedSuccessorFiberCacheError(
            "pickled indexed cache lost its parent validation receipt"
        )


__all__ = [
    "INDEXED_SUCCESSOR_FIBER_APPLICATION_ID",
    "INDEXED_SUCCESSOR_FIBER_CACHE_SCHEMA",
    "INDEXED_SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION",
    "INDEXED_SUCCESSOR_FIBER_STORAGE_BACKEND",
    "IndexedSuccessorFiberCache",
    "IndexedSuccessorFiberCacheError",
    "IndexedSuccessorFiberCacheMetadata",
    "IndexedSuccessorFiberCacheOpenReceipt",
    "assert_indexed_cache_is_pickle_safe",
    "open_indexed_successor_fiber_cache",
    "write_indexed_successor_fiber_cache",
]
