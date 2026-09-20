"""Materialize the frozen Editing-V2 split into 20 physical packed shards.

This module is the deterministic physical bridge between candidate routing and
whole-trace Active8 admission.  It mechanically applies the frozen four-role
assignment to every routed candidate, reads the candidate's immutable original
packed address, and writes exactly one packed shard for every five-lane by
four-role cell.

No chemistry object is decoded and no executor is called.  The only permitted
row mutation is the addressing envelope required by the downstream Active8
reader:

* ``trace.layer`` becomes the Editing-V2 data lane;
* ``trace.partition`` becomes the assigned split role.

The states array, steps and actions, source/target identities, trace identity,
path length, slot capacity, seed, metadata, and every other trace field are
canonical-hash compared before and after the rewrite.  Any other drift aborts
publication.

Outputs use deterministic gzip and JSON serialization, standard packed
manifests, content-addressed lane roots, the generic five-lane registry, and
the version-2 resolved-membership receipt consumed by
``editing_v2_active8_source_adapter``.  Publication is one same-parent
directory rename.  Every output remains explicitly pre-Active8 and carries no
training or final-test-selection authority.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from compose_v4.data.editing_corpus_contract import load_editing_corpus_contract
from compose_v4.data.editing_v2_active8_source_adapter import (
    ACTIVE8_ADMISSION_STATUS,
    CANDIDATE_AUTHORITY,
    REQUIRED_BLOCKERS,
    RESOLVED_PACKED_MEMBERSHIP_SCHEMA,
    RESOLVED_PACKED_MEMBERSHIP_SCHEMA_VERSION,
    RESOLVED_PACKED_MEMBERSHIP_STATUS,
    EditingV2Active8SourceAdapterError,
    _candidate_input_identity,
    _expected_original_address,
    _split_input_identity,
    _validate_split_assignment,
)
from compose_v4.data.editing_v2_candidate_provenance_bridge import (
    EditingV2CandidateProvenanceBridgeError,
    validate_candidate_provenance_bridge,
)
from compose_v4.data.editing_v2_lane_registry import (
    LANE_COMPLETION_FILENAME,
    LANE_COMPLETION_SCHEMA,
    LANE_COMPLETION_SCHEMA_VERSION,
    LANE_COMPLETION_STATUS,
    LANE_REGISTRY_SCHEMA,
    LANE_REGISTRY_SCHEMA_VERSION,
    LANE_REGISTRY_STATUS,
    editing_corpus_contract_identity,
    editing_v2_lane_definitions,
    file_sha256,
)
from compose_v4.data.editing_v2_packed_candidate_materializer import (
    CANDIDATE_ROWS_FILENAME,
    canonical_json_bytes,
    canonical_sha256,
    validate_packed_candidate_materialization,
)
from compose_v4.data.editing_v2_split_assignment import (
    EditingV2SplitAssignmentError,
    validate_candidate_source_stream,
)
from compose_v4.data.editing_v2_split_census import PARTITION_ROLES
from compose_v4.data.packed_trace_store import (
    PACKED_STORE_SCHEMA,
    PACKED_STORE_SCHEMA_VERSION,
    manifest_path_for,
    sampler_contract,
)
from compose_v4.data.provenance_overlay import overlay_path_for

ROLE_LANE_MATERIALIZATION_SCHEMA = "compose.editing_v2_role_lane_packed_materialization"
ROLE_LANE_MATERIALIZATION_SCHEMA_VERSION = 2
ROLE_LANE_MATERIALIZATION_STATUS = (
    "COMPLETE_PHYSICAL_DERIVATIVE_ACTIVE8_NOT_RUN_NO_TRAINING_AUTHORITY"
)
ROLE_LANE_MATERIALIZATION_FILENAME = "ROLE_LANE_PACKED_MATERIALIZATION.json"
LANE_REGISTRY_FILENAME = "EDITING_V2_LANE_REGISTRY.json"
RESOLVED_MEMBERSHIP_FILENAME = "RESOLVED_PACKED_MEMBERSHIP.json"
OUTPUT_NAMESPACE = "/artifacts/editing_v2/role_lane_packed"
ENVELOPE_REWRITE_CONTRACT = "only_trace_layer_and_trace_partition_may_change_v1"
PACKED_SHARD_PROVENANCE_SCHEMA = "compose.editing_v2_role_lane_packed_shard_provenance"
PACKED_SHARD_PROVENANCE_SCHEMA_VERSION = 2
DEFAULT_MAX_SOURCE_ROW_BYTES = 16 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CELL_SHARD_NAME = "shard_0000.jsonl.gz"
_FROZEN_TRACE_FIELDS = (
    "steps",
    "source_key",
    "target_key",
    "trace_id",
    "path_length",
    "n_slots",
    "seed",
    "metadata",
)


class EditingV2RoleLaneMaterializationError(ValueError):
    """The physical role/lane derivative cannot be proved complete."""


@dataclass
class _CellWriter:
    lane: str
    role: str
    provisional_path: Path
    raw_handle: BinaryIO
    gzip_handle: gzip.GzipFile
    records: int = 0
    states: int = 0
    transitions: int = 0
    original_address_digest: Any = None
    output_row_digest: Any = None

    def __post_init__(self) -> None:
        self.original_address_digest = hashlib.sha256()
        self.output_row_digest = hashlib.sha256()


def _require_sha256(value: object, *, field: str) -> str:
    digest = value if isinstance(value, str) else ""
    if _SHA256_RE.fullmatch(digest) is None:
        raise EditingV2RoleLaneMaterializationError(
            f"{field} must be a full lowercase SHA-256"
        )
    return digest


def _safe_relative_path(value: object, *, field: str) -> str:
    raw = value if isinstance(value, str) else ""
    candidate = PurePosixPath(raw)
    if (
        not raw
        or candidate.is_absolute()
        or candidate in (PurePosixPath("."), PurePosixPath(".."))
        or ".." in candidate.parts
        or "\\" in raw
        or any(character in raw for character in "*?[]")
        or str(candidate) != raw
    ):
        raise EditingV2RoleLaneMaterializationError(
            f"{field} must be a safe normalized relative path"
        )
    return raw


def _artifact_prefix(value: str) -> str:
    candidate = PurePosixPath(value)
    if (
        not candidate.is_absolute()
        or len(candidate.parts) < 3
        or candidate.parts[1] != "artifacts"
        or ".." in candidate.parts
        or str(candidate) != value
    ):
        raise EditingV2RoleLaneMaterializationError(
            "output artifact prefix must be normalized below /artifacts"
        )
    return value


def _resolve_source(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / relative_path).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise EditingV2RoleLaneMaterializationError(
            "original packed source escapes the artifact root"
        )
    return candidate


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _stream_sha256(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(canonical_json_bytes(value))
        digest.update(b"\n")
    return digest.hexdigest()


def _state_stream_sha256(states: object) -> str:
    if not isinstance(states, list) or not states:
        raise EditingV2RoleLaneMaterializationError(
            "packed source states must be a nonempty list"
        )
    return _stream_sha256(states)


def _field_sha256(value: Any) -> str:
    return canonical_sha256(value)


def _rewrite_envelope_only(
    entry: Mapping[str, Any],
    *,
    lane: str,
    role: str,
) -> dict[str, Any]:
    """Return one JSON-only derivative with the two permitted envelope edits."""

    if set(entry) != {"trace", "states"}:
        raise EditingV2RoleLaneMaterializationError(
            "packed source row must contain exactly trace and states"
        )
    trace = entry["trace"]
    states = entry["states"]
    if not isinstance(trace, Mapping) or not isinstance(states, list):
        raise EditingV2RoleLaneMaterializationError(
            "packed source row trace/states types are invalid"
        )
    output_trace = dict(trace)
    output_trace["layer"] = lane
    output_trace["partition"] = role
    return {"trace": output_trace, "states": states}


def _assert_envelope_only(
    original: Mapping[str, Any],
    rewritten: Mapping[str, Any],
    *,
    lane: str,
    role: str,
) -> dict[str, str]:
    """Prove that every semantic payload outside the address envelope is frozen."""

    if set(original) != {"trace", "states"} or set(rewritten) != {
        "trace",
        "states",
    }:
        raise EditingV2RoleLaneMaterializationError(
            "original and rewritten rows must contain exactly trace and states"
        )
    original_trace = original["trace"]
    rewritten_trace = rewritten["trace"]
    if not isinstance(original_trace, Mapping) or not isinstance(
        rewritten_trace,
        Mapping,
    ):
        raise EditingV2RoleLaneMaterializationError(
            "original and rewritten trace envelopes must be objects"
        )
    frozen_hashes = {
        "states_sha256": _field_sha256(original["states"]),
    }
    if frozen_hashes["states_sha256"] != _field_sha256(rewritten["states"]):
        raise EditingV2RoleLaneMaterializationError(
            "envelope rewrite mutated the exact states array"
        )
    for field in _FROZEN_TRACE_FIELDS:
        before = _field_sha256(original_trace.get(field))
        after = _field_sha256(rewritten_trace.get(field))
        frozen_hashes[f"{field}_sha256"] = before
        if before != after:
            raise EditingV2RoleLaneMaterializationError(
                f"envelope rewrite mutated frozen trace field {field!r}"
            )
    restored = dict(rewritten_trace)
    restored["layer"] = original_trace.get("layer")
    restored["partition"] = original_trace.get("partition")
    if restored != dict(original_trace):
        raise EditingV2RoleLaneMaterializationError(
            "envelope rewrite changed a trace field outside layer/partition"
        )
    if rewritten_trace.get("layer") != lane or rewritten_trace.get("partition") != role:
        raise EditingV2RoleLaneMaterializationError(
            "rewritten trace envelope does not equal its assigned lane/role"
        )
    return frozen_hashes


def _iter_bounded_gzip_jsonl(
    path: Path,
    *,
    max_row_bytes: int,
):
    if type(max_row_bytes) is not int or max_row_bytes <= 0:
        raise EditingV2RoleLaneMaterializationError(
            "max_source_row_bytes must be positive"
        )
    entry_index = 0
    with gzip.open(path, "rb") as handle:
        while True:
            raw_line = handle.readline(max_row_bytes + 1)
            if not raw_line:
                break
            if len(raw_line) > max_row_bytes or not raw_line.endswith(b"\n"):
                raise EditingV2RoleLaneMaterializationError(
                    f"source row exceeds byte bound or lacks newline: {path}"
                )
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EditingV2RoleLaneMaterializationError(
                    f"source packed row {entry_index} is invalid JSON: {path}"
                ) from error
            if not isinstance(value, Mapping):
                raise EditingV2RoleLaneMaterializationError(
                    f"source packed row {entry_index} must be an object: {path}"
                )
            yield entry_index, value
            entry_index += 1


def _verify_frozen_source(
    *,
    artifact_root: Path,
    relative_path: str,
    shard_sha256: str,
    manifest_sha256: str,
    overlay_sha256: str | None,
) -> Path:
    source = _resolve_source(artifact_root, relative_path)
    if not source.is_file() or file_sha256(source) != shard_sha256:
        raise EditingV2RoleLaneMaterializationError(
            f"original packed shard is absent or its SHA-256 disagrees: {source}"
        )
    manifest = manifest_path_for(source)
    if not manifest.is_file() or file_sha256(manifest) != manifest_sha256:
        raise EditingV2RoleLaneMaterializationError(
            f"original packed manifest is absent or its SHA-256 disagrees: {manifest}"
        )
    overlay = overlay_path_for(source)
    if overlay_sha256 is None:
        if overlay.exists():
            raise EditingV2RoleLaneMaterializationError(
                f"original packed overlay exists but is unbound: {overlay}"
            )
    elif not overlay.is_file() or file_sha256(overlay) != overlay_sha256:
        raise EditingV2RoleLaneMaterializationError(
            f"original packed overlay is absent or its SHA-256 disagrees: {overlay}"
        )
    return source


def _database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript("""
        PRAGMA journal_mode = DELETE;
        PRAGMA synchronous = FULL;
        CREATE TABLE selected (
            candidate_id TEXT PRIMARY KEY,
            data_lane TEXT NOT NULL,
            assigned_role TEXT NOT NULL,
            source_relative_path TEXT NOT NULL,
            source_shard_sha256 TEXT NOT NULL,
            source_manifest_sha256 TEXT NOT NULL,
            source_overlay_sha256 TEXT,
            source_entry_index INTEGER NOT NULL,
            source_address_sha256 TEXT NOT NULL UNIQUE,
            candidate_json TEXT NOT NULL,
            UNIQUE(source_shard_sha256, source_entry_index)
        );
        CREATE TABLE candidate_seen (
            candidate_id TEXT PRIMARY KEY,
            source_address_sha256 TEXT NOT NULL UNIQUE
        );
        CREATE TABLE memberships (
            data_lane TEXT NOT NULL,
            assigned_role TEXT NOT NULL,
            output_entry_index INTEGER NOT NULL,
            candidate_id TEXT NOT NULL UNIQUE,
            original_address_json TEXT NOT NULL,
            original_packed_row_sha256 TEXT NOT NULL,
            output_packed_row_sha256 TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            source_key TEXT NOT NULL,
            target_key TEXT NOT NULL,
            path_length INTEGER NOT NULL,
            PRIMARY KEY(data_lane, assigned_role, output_entry_index)
        );
        """)
    return connection


def _index_selected_candidates(
    connection: sqlite3.Connection,
    *,
    candidate_root: Path,
    assigned_roles: Mapping[str, str],
    lane_ids: tuple[str, ...],
    expected_rows_file_sha256: str | None = None,
) -> int:
    if expected_rows_file_sha256 is not None:
        _require_sha256(
            expected_rows_file_sha256,
            field="candidate rows expected physical SHA-256",
        )
    rows_path = candidate_root / CANDIDATE_ROWS_FILENAME
    selected_candidate_ids = set(assigned_roles)
    observed_selected: set[str] = set()
    routed_count = 0
    rows_digest = hashlib.sha256()
    with rows_path.open("rb") as handle:
        for raw_line in handle:
            rows_digest.update(raw_line)
            row = json.loads(raw_line)
            candidate_id = str(row.get("candidate_id"))
            disposition = row.get("disposition")
            original = _expected_original_address(row)
            address = original["packed_address"]
            overlay_sha256 = address["historical_provenance_overlay_file_sha256"]
            if overlay_sha256 is not None:
                _require_sha256(
                    overlay_sha256,
                    field=f"candidate {candidate_id}.historical overlay",
                )
            try:
                connection.execute(
                    "INSERT INTO candidate_seen VALUES (?, ?)",
                    (
                        candidate_id,
                        _require_sha256(
                            address["address_sha256"],
                            field=f"candidate {candidate_id}.address SHA-256",
                        ),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise EditingV2RoleLaneMaterializationError(
                    f"duplicate candidate or original address: {candidate_id!r}"
                ) from error
            if disposition == "rejected_candidate":
                if candidate_id in selected_candidate_ids:
                    raise EditingV2RoleLaneMaterializationError(
                        f"split assignment selected rejected candidate {candidate_id!r}"
                    )
                continue
            if disposition != "routed_candidate":
                raise EditingV2RoleLaneMaterializationError(
                    f"candidate {candidate_id!r} has an unknown disposition"
                )
            routed_count += 1
            role = assigned_roles.get(candidate_id)
            lane = row.get("data_lane")
            if role is None or lane not in lane_ids:
                raise EditingV2RoleLaneMaterializationError(
                    f"routed candidate {candidate_id!r} is unselected or has an unknown lane"
                )
            if candidate_id in observed_selected:
                raise EditingV2RoleLaneMaterializationError(
                    f"selected candidate is duplicated: {candidate_id!r}"
                )
            observed_selected.add(candidate_id)
            source_entry_index = address["entry_index"]
            if type(source_entry_index) is not int or source_entry_index < 0:
                raise EditingV2RoleLaneMaterializationError(
                    f"candidate {candidate_id!r}.entry_index must be a nonnegative integer"
                )
            try:
                connection.execute(
                    """
                    INSERT INTO selected VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate_id,
                        lane,
                        role,
                        _safe_relative_path(
                            address["relative_path"],
                            field=f"candidate {candidate_id}.relative_path",
                        ),
                        _require_sha256(
                            address["packed_shard_file_sha256"],
                            field=f"candidate {candidate_id}.shard SHA-256",
                        ),
                        _require_sha256(
                            address["packed_shard_manifest_file_sha256"],
                            field=f"candidate {candidate_id}.manifest SHA-256",
                        ),
                        overlay_sha256,
                        source_entry_index,
                        address["address_sha256"],
                        canonical_json_bytes(row).decode("utf-8"),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise EditingV2RoleLaneMaterializationError(
                    f"duplicate selected candidate or original address: {candidate_id!r}"
                ) from error
    if (
        expected_rows_file_sha256 is not None
        and rows_digest.hexdigest() != expected_rows_file_sha256
    ):
        raise EditingV2RoleLaneMaterializationError(
            "candidate rows physical SHA-256 disagrees during indexing"
        )
    if observed_selected != selected_candidate_ids or routed_count != len(
        selected_candidate_ids
    ):
        raise EditingV2RoleLaneMaterializationError(
            "selected/routed candidate sets are incomplete; "
            f"missing={sorted(selected_candidate_ids - observed_selected)}, "
            f"unexpected={sorted(observed_selected - selected_candidate_ids)}"
        )
    connection.commit()
    return routed_count


def _open_cell_writers(
    staging: Path,
    *,
    lane_ids: tuple[str, ...],
) -> dict[tuple[str, str], _CellWriter]:
    writers: dict[tuple[str, str], _CellWriter] = {}
    for lane in lane_ids:
        for role in PARTITION_ROLES:
            path = staging / "_cells" / lane / role / _CELL_SHARD_NAME
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = path.open("wb")
            compressed = gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                mtime=0,
            )
            writers[(lane, role)] = _CellWriter(
                lane=lane,
                role=role,
                provisional_path=path,
                raw_handle=raw,
                gzip_handle=compressed,
            )
    return writers


def _close_cell_writers(writers: Mapping[tuple[str, str], _CellWriter]) -> None:
    for writer in writers.values():
        try:
            writer.gzip_handle.close()
        finally:
            writer.raw_handle.flush()
            os.fsync(writer.raw_handle.fileno())
            writer.raw_handle.close()


def _validate_original_candidate(
    entry: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    trace = entry.get("trace")
    states = entry.get("states")
    address = candidate["packed_address"]
    if not isinstance(trace, Mapping) or not isinstance(states, list):
        raise EditingV2RoleLaneMaterializationError(
            "original packed row lacks trace/states"
        )
    for field, expected in (
        ("trace_id", address["trace_id"]),
        ("layer", address["layer"]),
        ("partition", address["cache_partition"]),
        ("path_length", address["path_length"]),
    ):
        if trace.get(field) != expected:
            raise EditingV2RoleLaneMaterializationError(
                f"original packed row {field} disagrees with candidate address"
            )
    trace_sha256 = canonical_sha256(trace)
    states_sha256 = _state_stream_sha256(states)
    if (
        trace_sha256 != candidate["trace_envelope_sha256"]
        or states_sha256 != candidate["exact_states"]["encoded_state_stream_sha256"]
        or len(states) != candidate["exact_states"]["state_count"]
    ):
        raise EditingV2RoleLaneMaterializationError(
            "original packed row payload disagrees with candidate materialization"
        )
    payload_sha256 = canonical_sha256(
        {
            "trace_envelope_sha256": trace_sha256,
            "encoded_state_stream_sha256": states_sha256,
        }
    )
    if payload_sha256 != candidate["candidate_payload_sha256"]:
        raise EditingV2RoleLaneMaterializationError(
            "original packed row candidate payload SHA-256 disagrees"
        )


def _materialize_selected_rows(
    connection: sqlite3.Connection,
    *,
    artifact_root: Path,
    writers: Mapping[tuple[str, str], _CellWriter],
    max_source_row_bytes: int,
) -> None:
    source_groups = connection.execute("""
        SELECT DISTINCT source_relative_path, source_shard_sha256,
                        source_manifest_sha256, source_overlay_sha256
        FROM selected
        ORDER BY source_relative_path, source_shard_sha256
        """).fetchall()
    output_indices = {cell: 0 for cell in writers}
    for (
        relative_path,
        shard_sha256,
        manifest_sha256,
        overlay_sha256,
    ) in source_groups:
        source_path = _verify_frozen_source(
            artifact_root=artifact_root,
            relative_path=relative_path,
            shard_sha256=shard_sha256,
            manifest_sha256=manifest_sha256,
            overlay_sha256=overlay_sha256,
        )
        selected_rows = iter(
            connection.execute(
                """
                SELECT source_entry_index, candidate_json
                FROM selected
                WHERE source_relative_path = ?
                  AND source_shard_sha256 = ?
                  AND source_manifest_sha256 = ?
                  AND source_overlay_sha256 IS ?
                ORDER BY source_entry_index
                """,
                (
                    relative_path,
                    shard_sha256,
                    manifest_sha256,
                    overlay_sha256,
                ),
            )
        )
        try:
            wanted_index, candidate_json = next(selected_rows)
        except StopIteration:
            continue
        for source_entry_index, entry in _iter_bounded_gzip_jsonl(
            source_path,
            max_row_bytes=max_source_row_bytes,
        ):
            if source_entry_index < wanted_index:
                continue
            if source_entry_index > wanted_index:
                raise EditingV2RoleLaneMaterializationError(
                    f"selected original address is missing: {relative_path} entry {wanted_index}"
                )
            candidate = json.loads(candidate_json)
            _validate_original_candidate(entry, candidate)
            candidate_id = candidate["candidate_id"]
            lane = candidate["data_lane"]
            role_row = connection.execute(
                "SELECT assigned_role FROM selected WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if role_row is None:
                raise EditingV2RoleLaneMaterializationError(
                    f"selected candidate disappeared from index: {candidate_id!r}"
                )
            role = role_row[0]
            rewritten = _rewrite_envelope_only(
                entry,
                lane=lane,
                role=role,
            )
            _assert_envelope_only(
                entry,
                rewritten,
                lane=lane,
                role=role,
            )
            writer = writers[(lane, role)]
            output_index = output_indices[(lane, role)]
            encoded = canonical_json_bytes(rewritten)
            writer.gzip_handle.write(encoded + b"\n")
            original_row_sha256 = canonical_sha256(entry)
            output_row_sha256 = hashlib.sha256(encoded).hexdigest()
            original_address = _expected_original_address(candidate)
            writer.original_address_digest.update(
                original_address["packed_address"]["address_sha256"].encode("ascii")
            )
            writer.original_address_digest.update(b"\n")
            writer.output_row_digest.update(output_row_sha256.encode("ascii"))
            writer.output_row_digest.update(b"\n")
            state_count = int(candidate["exact_states"]["state_count"])
            path_length = int(candidate["packed_address"]["path_length"])
            writer.records += 1
            writer.states += state_count
            writer.transitions += path_length
            trace = rewritten["trace"]
            connection.execute(
                """
                INSERT INTO memberships VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lane,
                    role,
                    output_index,
                    candidate_id,
                    canonical_json_bytes(original_address).decode("utf-8"),
                    original_row_sha256,
                    output_row_sha256,
                    trace["trace_id"],
                    trace["source_key"],
                    trace["target_key"],
                    trace["path_length"],
                ),
            )
            output_indices[(lane, role)] += 1
            try:
                wanted_index, candidate_json = next(selected_rows)
            except StopIteration:
                wanted_index = None
                break
        if wanted_index is not None:
            raise EditingV2RoleLaneMaterializationError(
                f"selected original address is missing: {relative_path} entry {wanted_index}"
            )
    connection.commit()
    selected_count = connection.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
    membership_count = connection.execute(
        "SELECT COUNT(*) FROM memberships"
    ).fetchone()[0]
    if selected_count != membership_count:
        raise EditingV2RoleLaneMaterializationError(
            "physical materialization did not consume every selected candidate"
        )


def _cell_manifest(
    writer: _CellWriter,
    *,
    code_revision: str,
    implementation_sha256: str,
    candidate_identity: Mapping[str, Any],
    candidate_provenance_source_stream: Mapping[str, Any],
    split_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if writer.records <= 0:
        raise EditingV2RoleLaneMaterializationError(
            f"lane/role cell is empty: {writer.lane}/{writer.role}"
        )
    return {
        "schema": PACKED_STORE_SCHEMA,
        "schema_version": PACKED_STORE_SCHEMA_VERSION,
        "sampler_contract": sampler_contract(),
        "entries": writer.records,
        "states": writer.states,
        "provenance": {
            "schema": PACKED_SHARD_PROVENANCE_SCHEMA,
            "schema_version": PACKED_SHARD_PROVENANCE_SCHEMA_VERSION,
            "status": (
                "CURRENT_PHYSICAL_DERIVATIVE_ACTIVE8_NOT_RUN_NO_TRAINING_AUTHORITY"
            ),
            "training_authorized": False,
            "code_revision": code_revision,
            "materializer_implementation_sha256": implementation_sha256,
            "candidate_materialization": dict(candidate_identity),
            "candidate_provenance_source_stream": dict(
                candidate_provenance_source_stream
            ),
            "split_assignment": dict(split_identity),
            "data_lane": writer.lane,
            "partition_role": writer.role,
            "envelope_rewrite_contract": ENVELOPE_REWRITE_CONTRACT,
            "original_address_stream_sha256": (
                writer.original_address_digest.hexdigest()
            ),
            "output_row_stream_sha256": writer.output_row_digest.hexdigest(),
            "final_test_selection_use": "forbidden_not_performed",
        },
    }


def _publish_lane_files(
    staging: Path,
    *,
    run_artifact_root: str,
    lane_definitions,
    writers: Mapping[tuple[str, str], _CellWriter],
    code_revision: str,
    implementation_sha256: str,
    candidate_identity: Mapping[str, Any],
    candidate_provenance_source_stream: Mapping[str, Any],
    split_identity: Mapping[str, Any],
    contract_identity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    registry_lanes: list[dict[str, Any]] = []
    published_cells: dict[tuple[str, str], dict[str, Any]] = {}
    for lane_definition in lane_definitions:
        lane = lane_definition.lane_id
        provisional_rows: list[dict[str, Any]] = []
        for role in PARTITION_ROLES:
            writer = writers[(lane, role)]
            manifest = _cell_manifest(
                writer,
                code_revision=code_revision,
                implementation_sha256=implementation_sha256,
                candidate_identity=candidate_identity,
                candidate_provenance_source_stream=(candidate_provenance_source_stream),
                split_identity=split_identity,
            )
            manifest_path = manifest_path_for(writer.provisional_path)
            _write_json(manifest_path, manifest)
            provisional_rows.append(
                {
                    "partition": role,
                    "relative_path": f"{role}/{_CELL_SHARD_NAME}",
                    "sha256": file_sha256(writer.provisional_path),
                    "bytes": writer.provisional_path.stat().st_size,
                    "records": writer.records,
                    "states": writer.states,
                    "transitions": writer.transitions,
                    "_manifest_sha256": file_sha256(manifest_path),
                    "_provisional_path": writer.provisional_path,
                    "_provisional_manifest_path": manifest_path,
                }
            )
        inventory = [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in provisional_rows
        ]
        inventory_sha256 = canonical_sha256(inventory)
        declared_root = f"{run_artifact_root}/lanes/{lane}/{inventory_sha256}"
        local_lane_root = staging / "lanes" / lane / inventory_sha256
        for row in provisional_rows:
            destination = local_lane_root / row["relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(row["_provisional_path"], destination)
            destination_manifest = manifest_path_for(destination)
            os.replace(
                row["_provisional_manifest_path"],
                destination_manifest,
            )
            artifact_path = f"{declared_root}/{row['relative_path']}"
            published_cells[(lane, row["partition"])] = {
                **{key: value for key, value in row.items() if not key.startswith("_")},
                "artifact_path": artifact_path,
                "local_path": destination,
                "manifest_artifact_path": str(
                    PurePosixPath(artifact_path).with_suffix(".manifest.json")
                ),
                "manifest_file_sha256": row["_manifest_sha256"],
            }
        by_partition = {
            role: {
                "records": published_cells[(lane, role)]["records"],
                "states": published_cells[(lane, role)]["states"],
                "transitions": published_cells[(lane, role)]["transitions"],
                "shards": 1,
            }
            for role in PARTITION_ROLES
        }
        completion_body = {
            "schema": LANE_COMPLETION_SCHEMA,
            "schema_version": LANE_COMPLETION_SCHEMA_VERSION,
            "status": LANE_COMPLETION_STATUS,
            "training_authorized": False,
            "editing_corpus_contract": dict(contract_identity),
            "lane_id": lane,
            "admissible_evidence_profiles": list(
                lane_definition.admissible_evidence_profiles
            ),
            "reserved_evidence_profiles": list(
                lane_definition.reserved_evidence_profiles
            ),
            "declared_root": declared_root,
            "partition_roles": list(PARTITION_ROLES),
            "counts": {
                "records": sum(row["records"] for row in inventory),
                "states": sum(row["states"] for row in inventory),
                "transitions": sum(row["transitions"] for row in inventory),
                "shards": len(inventory),
                "by_partition": by_partition,
            },
            "shards": inventory,
            "shard_inventory_sha256": inventory_sha256,
        }
        completion = {
            **completion_body,
            "completion_sha256": canonical_sha256(completion_body),
        }
        completion_path = local_lane_root / LANE_COMPLETION_FILENAME
        _write_json(completion_path, completion)
        registry_lanes.append(
            {
                "lane_id": lane,
                "admissible_evidence_profiles": list(
                    lane_definition.admissible_evidence_profiles
                ),
                "reserved_evidence_profiles": list(
                    lane_definition.reserved_evidence_profiles
                ),
                "declared_root": declared_root,
                "completion_manifest_path": (
                    f"{declared_root}/{LANE_COMPLETION_FILENAME}"
                ),
                "completion_manifest_sha256": file_sha256(completion_path),
            }
        )
    return registry_lanes, published_cells


def _output_address(
    *,
    cell: Mapping[str, Any],
    output_entry_index: int,
    trace_id: str,
    lane: str,
    role: str,
    source_key: str,
    target_key: str,
    path_length: int,
) -> dict[str, Any]:
    body = {
        "packed_shard_file_sha256": cell["sha256"],
        "packed_shard_manifest_file_sha256": cell["manifest_file_sha256"],
        "packed_provenance_overlay_file_sha256": None,
        "packed_shard_name": _CELL_SHARD_NAME,
        "entry_index": output_entry_index,
        "trace_id": trace_id,
        "layer": lane,
        "partition": role,
        "source_key": source_key,
        "target_key": target_key,
        "path_length": path_length,
    }
    return {**body, "address_sha256": canonical_sha256(body)}


class _HashSink:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()

    def write(self, content: bytes) -> None:
        self.digest.update(content)


def _emit_receipt_body(
    sink,
    *,
    fixed_fields: Mapping[str, Any],
    lane_order: tuple[str, ...],
    published_cells: Mapping[tuple[str, str], Mapping[str, Any]],
    connection: sqlite3.Connection,
    receipt_sha256: str | None,
) -> None:
    fields = dict(fixed_fields)
    fields["shards"] = None
    if receipt_sha256 is not None:
        fields["receipt_sha256"] = receipt_sha256
    sink.write(b"{")
    first_field = True
    for key in sorted(fields):
        if not first_field:
            sink.write(b",")
        first_field = False
        sink.write(canonical_json_bytes(key))
        sink.write(b":")
        if key != "shards":
            sink.write(canonical_json_bytes(fields[key]))
            continue
        sink.write(b"[")
        first_shard = True
        for lane, role in (
            (lane, role) for lane in lane_order for role in PARTITION_ROLES
        ):
            if not first_shard:
                sink.write(b",")
            first_shard = False
            cell = published_cells[(lane, role)]
            shard_fields = {
                "data_lane": lane,
                "partition_role": role,
                "relative_path": cell["relative_path"],
                "artifact_path": cell["artifact_path"],
                "file_sha256": cell["sha256"],
                "packed_manifest_artifact_path": cell["manifest_artifact_path"],
                "packed_manifest_file_sha256": cell["manifest_file_sha256"],
                "packed_provenance_overlay_artifact_path": None,
                "packed_provenance_overlay_file_sha256": None,
                "entry_memberships": None,
            }
            sink.write(b"{")
            first_shard_field = True
            for shard_key in sorted(shard_fields):
                if not first_shard_field:
                    sink.write(b",")
                first_shard_field = False
                sink.write(canonical_json_bytes(shard_key))
                sink.write(b":")
                if shard_key != "entry_memberships":
                    sink.write(canonical_json_bytes(shard_fields[shard_key]))
                    continue
                sink.write(b"[")
                first_membership = True
                for (
                    output_index,
                    candidate_id,
                    original_address_json,
                    original_row_sha256,
                    output_row_sha256,
                    trace_id,
                    source_key,
                    target_key,
                    path_length,
                ) in connection.execute(
                    """
                    SELECT output_entry_index, candidate_id,
                           original_address_json,
                           original_packed_row_sha256,
                           output_packed_row_sha256,
                           trace_id, source_key, target_key, path_length
                    FROM memberships
                    WHERE data_lane = ? AND assigned_role = ?
                    ORDER BY output_entry_index
                    """,
                    (lane, role),
                ):
                    if not first_membership:
                        sink.write(b",")
                    first_membership = False
                    membership = {
                        "output_entry_index": output_index,
                        "candidate_id": candidate_id,
                        "original_packed_address": json.loads(original_address_json),
                        "original_packed_row_sha256": original_row_sha256,
                        "output_packed_row_sha256": output_row_sha256,
                        "output_packed_address": _output_address(
                            cell=cell,
                            output_entry_index=output_index,
                            trace_id=trace_id,
                            lane=lane,
                            role=role,
                            source_key=source_key,
                            target_key=target_key,
                            path_length=path_length,
                        ),
                    }
                    sink.write(canonical_json_bytes(membership))
                sink.write(b"]")
            sink.write(b"}")
        sink.write(b"]")
    sink.write(b"}")


def _write_membership_receipt(
    path: Path,
    *,
    fixed_fields: Mapping[str, Any],
    lane_order: tuple[str, ...],
    published_cells: Mapping[tuple[str, str], Mapping[str, Any]],
    connection: sqlite3.Connection,
) -> tuple[str, str]:
    semantic_fields = dict(fixed_fields)
    hash_sink = _HashSink()
    _emit_receipt_body(
        hash_sink,
        fixed_fields=semantic_fields,
        lane_order=lane_order,
        published_cells=published_cells,
        connection=connection,
        receipt_sha256=None,
    )
    receipt_sha256 = hash_sink.digest.hexdigest()
    # The emitter must not serialize its private ordering instruction.
    with path.open("wb") as handle:
        _emit_receipt_body(
            handle,
            fixed_fields=semantic_fields,
            lane_order=lane_order,
            published_cells=published_cells,
            connection=connection,
            receipt_sha256=receipt_sha256,
        )
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return receipt_sha256, file_sha256(path)


def materialize_editing_v2_role_lane_packed(
    *,
    candidate_materialization_dir: str | Path,
    candidate_provenance_bridge_dir: str | Path,
    candidate_provenance_registry_path: str | Path,
    split_assignment_path: str | Path,
    editing_corpus_contract_path: str | Path,
    artifact_root: str | Path,
    code_revision: str,
    output_artifact_prefix: str = OUTPUT_NAMESPACE,
    max_source_row_bytes: int = DEFAULT_MAX_SOURCE_ROW_BYTES,
) -> dict[str, Any]:
    """Build and atomically publish the exact 20-cell pre-Active8 derivative."""

    if _COMMIT_RE.fullmatch(code_revision) is None:
        raise EditingV2RoleLaneMaterializationError(
            "code_revision must be a full lowercase 40-character Git SHA"
        )
    artifact_root = Path(artifact_root)
    if not artifact_root.is_dir():
        raise EditingV2RoleLaneMaterializationError(
            f"artifact root is absent: {artifact_root}"
        )
    output_artifact_prefix = _artifact_prefix(output_artifact_prefix)
    candidate_root = Path(candidate_materialization_dir)
    candidate_manifest = validate_packed_candidate_materialization(candidate_root)
    candidate_identity = _candidate_input_identity(
        candidate_root,
        candidate_manifest,
    )
    split_path = Path(split_assignment_path)
    try:
        assignment, assigned_roles = _validate_split_assignment(split_path)
        split_identity = _split_input_identity(split_path, assignment)
    except EditingV2Active8SourceAdapterError as error:
        raise EditingV2RoleLaneMaterializationError(
            "split assignment validation failed before publication"
        ) from error
    contract_path = Path(editing_corpus_contract_path)
    contract = load_editing_corpus_contract(contract_path)
    try:
        bridge = validate_candidate_provenance_bridge(
            candidate_provenance_bridge_dir,
            candidate_root=candidate_root,
            provenance_registry_path=candidate_provenance_registry_path,
            editing_corpus_contract_path=contract_path,
        )
        candidate_provenance_source_stream = validate_candidate_source_stream(
            bridge.get("source_stream"),
            expected_candidate_materialization=candidate_identity,
            expected_nonempty_rows=len(assigned_roles),
        )
    except (
        EditingV2CandidateProvenanceBridgeError,
        EditingV2SplitAssignmentError,
        OSError,
    ) as error:
        raise EditingV2RoleLaneMaterializationError(
            "candidate provenance bridge validation failed before publication"
        ) from error
    if candidate_provenance_source_stream != assignment["source_stream"]:
        raise EditingV2RoleLaneMaterializationError(
            "split assignment candidate provenance source-stream identity is stale or mismatched"
        )
    contract_identity = editing_corpus_contract_identity(
        contract,
        contract_file_sha256=file_sha256(contract_path),
    )
    lane_definitions = editing_v2_lane_definitions(contract)
    lane_ids = tuple(lane.lane_id for lane in lane_definitions)
    implementation_sha256 = file_sha256(Path(__file__))
    run_identity_body = {
        "schema": ROLE_LANE_MATERIALIZATION_SCHEMA,
        "schema_version": ROLE_LANE_MATERIALIZATION_SCHEMA_VERSION,
        "code_revision": code_revision,
        "materializer_implementation_sha256": implementation_sha256,
        "candidate_materialization": candidate_identity,
        "candidate_provenance_source_stream": candidate_provenance_source_stream,
        "split_assignment": split_identity,
        "editing_corpus_contract": contract_identity,
        "lane_order": list(lane_ids),
        "role_order": list(PARTITION_ROLES),
        "envelope_rewrite_contract": ENVELOPE_REWRITE_CONTRACT,
        "sampler_contract": sampler_contract(),
        "output_artifact_prefix": output_artifact_prefix,
    }
    run_identity_sha256 = canonical_sha256(run_identity_body)
    run_artifact_root = f"{output_artifact_prefix}/{run_identity_sha256}"
    target = artifact_root / PurePosixPath(run_artifact_root).relative_to("/artifacts")
    if target.exists():
        raise EditingV2RoleLaneMaterializationError(
            f"immutable output already exists: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=target.parent,
            prefix=f".{run_identity_sha256}.",
            suffix=".staging",
        )
    )
    database_descriptor, database_name = tempfile.mkstemp(
        prefix="compose-editing-v2-role-lane-",
        suffix=".sqlite3",
    )
    os.close(database_descriptor)
    database_path = Path(database_name)
    connection = _database(database_path)
    published = False
    writers: dict[tuple[str, str], _CellWriter] = {}
    try:
        # Validation above proves row-stream integrity. Indexing below is a
        # second streaming pass that never retains packed states in memory.
        routed_count = _index_selected_candidates(
            connection,
            candidate_root=candidate_root,
            assigned_roles=assigned_roles,
            lane_ids=lane_ids,
        )
        writers = _open_cell_writers(staging, lane_ids=lane_ids)
        try:
            _materialize_selected_rows(
                connection,
                artifact_root=artifact_root,
                writers=writers,
                max_source_row_bytes=max_source_row_bytes,
            )
        finally:
            _close_cell_writers(writers)
        registry_lanes, published_cells = _publish_lane_files(
            staging,
            run_artifact_root=run_artifact_root,
            lane_definitions=lane_definitions,
            writers=writers,
            code_revision=code_revision,
            implementation_sha256=implementation_sha256,
            candidate_identity=candidate_identity,
            candidate_provenance_source_stream=(candidate_provenance_source_stream),
            split_identity=split_identity,
            contract_identity=contract_identity,
        )
        registry_body = {
            "schema": LANE_REGISTRY_SCHEMA,
            "schema_version": LANE_REGISTRY_SCHEMA_VERSION,
            "status": LANE_REGISTRY_STATUS,
            "training_authorized": False,
            "editing_corpus_contract": contract_identity,
            "partition_roles": list(PARTITION_ROLES),
            "lanes": registry_lanes,
        }
        registry = {
            **registry_body,
            "registry_sha256": canonical_sha256(registry_body),
        }
        registry_path = staging / LANE_REGISTRY_FILENAME
        _write_json(registry_path, registry)
        completion_file_sha256 = {}
        completion_sha256 = {}
        inventory_sha256 = {}
        for lane in registry_lanes:
            local_completion = staging / PurePosixPath(
                lane["completion_manifest_path"]
            ).relative_to(PurePosixPath(run_artifact_root))
            completion = json.loads(local_completion.read_bytes())
            lane_id = lane["lane_id"]
            completion_file_sha256[lane_id] = file_sha256(local_completion)
            completion_sha256[lane_id] = completion["completion_sha256"]
            inventory_sha256[lane_id] = completion["shard_inventory_sha256"]
        receipt_fixed = {
            "schema": RESOLVED_PACKED_MEMBERSHIP_SCHEMA,
            "schema_version": RESOLVED_PACKED_MEMBERSHIP_SCHEMA_VERSION,
            "status": RESOLVED_PACKED_MEMBERSHIP_STATUS,
            "training_authorized": False,
            "candidate_materialization_authority": CANDIDATE_AUTHORITY,
            "active8_admission_status": ACTIVE8_ADMISSION_STATUS,
            "membership_complete": True,
            "blockers": list(REQUIRED_BLOCKERS),
            "inputs": {
                "candidate_materialization": candidate_identity,
                "split_assignment": split_identity,
                "lane_registry": {
                    "registry_file_sha256": file_sha256(registry_path),
                    "registry_sha256": registry["registry_sha256"],
                    "editing_corpus_contract_file_sha256": (
                        contract_identity["file_sha256"]
                    ),
                    "editing_corpus_contract_id": contract_identity["contract_id"],
                    "lane_completion_manifest_file_sha256": (completion_file_sha256),
                    "lane_completion_sha256": completion_sha256,
                    "lane_shard_inventory_sha256": inventory_sha256,
                },
            },
        }
        receipt_path = staging / RESOLVED_MEMBERSHIP_FILENAME
        receipt_sha256, receipt_file_sha256 = _write_membership_receipt(
            receipt_path,
            fixed_fields=receipt_fixed,
            lane_order=lane_ids,
            published_cells=published_cells,
            connection=connection,
        )
        totals = {
            "routed_candidates": routed_count,
            "output_shards": len(published_cells),
            "output_records": sum(
                int(cell["records"]) for cell in published_cells.values()
            ),
            "output_states": sum(
                int(cell["states"]) for cell in published_cells.values()
            ),
            "output_transitions": sum(
                int(cell["transitions"]) for cell in published_cells.values()
            ),
        }
        materialization_body = {
            **run_identity_body,
            "status": ROLE_LANE_MATERIALIZATION_STATUS,
            "training_authorized": False,
            "active8_admission_status": ACTIVE8_ADMISSION_STATUS,
            "run_identity_sha256": run_identity_sha256,
            "run_artifact_root": run_artifact_root,
            "selection_policy": (
                "mechanical_application_of_frozen_split_no_metric_selection"
            ),
            "final_test_selection_use": "forbidden_not_performed",
            "lane_registry": {
                "relative_path": LANE_REGISTRY_FILENAME,
                "file_sha256": file_sha256(registry_path),
                "registry_sha256": registry["registry_sha256"],
            },
            "resolved_packed_membership": {
                "relative_path": RESOLVED_MEMBERSHIP_FILENAME,
                "file_sha256": receipt_file_sha256,
                "receipt_sha256": receipt_sha256,
                "schema_version": (RESOLVED_PACKED_MEMBERSHIP_SCHEMA_VERSION),
            },
            "totals": totals,
            "atomic_publication": "same_parent_directory_rename",
        }
        materialization = {
            **materialization_body,
            "manifest_sha256": canonical_sha256(materialization_body),
        }
        _write_json(
            staging / ROLE_LANE_MATERIALIZATION_FILENAME,
            materialization,
        )
        shutil.rmtree(staging / "_cells")
        os.replace(staging, target)
        published = True
        return materialization
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
        if not published:
            for writer in writers.values():
                if not writer.gzip_handle.closed:
                    writer.gzip_handle.close()
                if not writer.raw_handle.closed:
                    writer.raw_handle.close()
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "DEFAULT_MAX_SOURCE_ROW_BYTES",
    "ENVELOPE_REWRITE_CONTRACT",
    "LANE_REGISTRY_FILENAME",
    "OUTPUT_NAMESPACE",
    "RESOLVED_MEMBERSHIP_FILENAME",
    "ROLE_LANE_MATERIALIZATION_FILENAME",
    "ROLE_LANE_MATERIALIZATION_SCHEMA",
    "ROLE_LANE_MATERIALIZATION_SCHEMA_VERSION",
    "ROLE_LANE_MATERIALIZATION_STATUS",
    "EditingV2RoleLaneMaterializationError",
    "materialize_editing_v2_role_lane_packed",
]
