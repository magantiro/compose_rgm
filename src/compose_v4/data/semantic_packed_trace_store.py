"""Deterministic exact-state packed storage for semantic Editing V2 traces.

Historical packed shards are immutable inputs.  They are migrated offline into
this separate artifact; the production reader never translates a legacy action
at load time.  Publication is an atomic directory rename, and the completion
receipt is part of the required artifact boundary.

Reading a *superseded* artifact
-------------------------------

``load_semantic_packed_manifest`` and the two readers accept optional
``expected_process_identity`` and ``expected_builder_identity`` objects.  When
both are omitted (the default) every validation is exactly what it is today and
the artifact must match the *live* semantic-process identity.

Supplying a pinned identity is required when the live implementation identity
has been superseded, which happens by design whenever the semantic process is
versioned: ``editing_v2_process_identity()`` hashes its implementation sources,
so an already published artifact can never match the live value again.  The
alternative behaviours are worse: silently accepting any identity would erase
the binding, and refusing outright would make an immutable historical artifact
permanently unreadable, which is the same as deleting scientific data.

The pinned path is therefore not a weakened gate:

* the caller must *name* the superseded identity explicitly, so the artifact it
  is willing to read is recorded in its own provenance rather than inferred;
* the pinned object is validated for internal self-consistency by
  :func:`validate_frozen_process_identity`.  That check proves the object is
  well formed under a declared identity schema; it does **not** prove the
  object was ever produced by this repository, because anyone can recompute a
  self-hash.  What actually closes the gap is the next requirement;
* the manifest must still match that identity exactly, together with its
  contract hash, process-semantics label, schema, physical hashes, completion
  receipt, censuses, and source binding, all unchanged;
* ``expected_builder_identity`` only *adds* an exact-match requirement to a
  field that is otherwise checked for self-consistency alone.

A reader that pins an identity is asserting compatibility with historical
bytes.  It is the caller's obligation, not this module's, to re-prove every
chemical claim the artifact carries.

Recoverable row reads
---------------------

:func:`read_semantic_packed_artifact_range` raises on the first unreadable row,
which is the right behaviour for a consumer that needs every row.  A proof
scan instead has to *account* for an unreadable row: it must name the exact
physical entry, attach a reason code, and refuse to publish, rather than die
with an untyped exception.  :func:`read_semantic_packed_artifact_range_rows`
exposes the same validated read with an explicit ``recover_row_errors`` switch
that turns a per-row decode failure into a :class:`SemanticPackedRowRead`
carrying its exact reason.  Artifact-level failures (inventory, manifest,
hashes, completion receipt, identity, range bounds, blank rows, row census)
are never recoverable and still raise.  The default is ``False``, which is
byte-for-byte today's behaviour.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from compose_v4.data.packed_trace_store import (
    AddressedPackedTrace,
    PackedTraceAddress,
    PackedTraceProgress,
    sampler_contract,
)
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite import editing_v2_process_identity as process_identity_module
from compose_v4.rewrite.editing_v2_process_identity import (
    PROCESS_SEMANTICS,
    editing_v2_process_identity,
    require_editing_v2_process_identity,
)
from compose_v4.rewrite.trace_shard import decode_state
from compose_v4.rewrite.trace_shard_v3 import (
    TRACE_SCHEMA,
    TRACE_SCHEMA_VERSION,
    SemanticTraceShardError,
    decode_semantic_trace_record,
)

STORE_SCHEMA = "compose.data.semantic_packed_trace"
STORE_SCHEMA_VERSION = 1
STORE_STATUS = "EXACT_SEMANTIC_DERIVATIVE_GATE0_NOT_RUN"
COMPLETION_SCHEMA = "compose.data.semantic_packed_trace_completion"
COMPLETION_SCHEMA_VERSION = 1
SHARD_FILENAME = "traces.jsonl.gz"
MANIFEST_FILENAME = "manifest.json"
COMPLETION_FILENAME = "COMPLETE.json"
ENCODING = "deterministic_gzip_jsonl_utf8_sort_keys_mtime_0"
DEFAULT_SENTINEL_REPLAY_ENTRIES = 8
_MANIFEST_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "training_authorized",
    "encoding",
    "shard_filename",
    "shard_sha256",
    "record_stream_sha256",
    "entries",
    "states",
    "actions",
    "family_histogram",
    "data_lane",
    "split",
    "source_binding",
    "decision_binding",
    "process_semantics",
    "process_identity_sha256",
    "process_contract_sha256",
    "action_codec_schema_version",
    "action_codec_implementation_hash",
    "trace_schema",
    "trace_schema_version",
    "sampler_contract",
    "builder_identity",
    "manifest_sha256",
}
_COMPLETION_FIELDS = {
    "schema",
    "schema_version",
    "status",
    "manifest_physical_sha256",
    "shard_sha256",
    "process_identity_sha256",
    "completion_sha256",
}


class SemanticPackedStoreError(RuntimeError):
    """A semantic packed artifact is incomplete, stale, or inconsistent."""


@dataclass(frozen=True)
class SemanticPackedRowRead:
    """One physical shard row: either decoded, or reason-carrying, never both.

    ``record`` is the exact persisted JSON object when the line parsed at all,
    so a caller can still address a malformed row by its own provenance.
    ``addressed`` is populated only when the complete row validation passed,
    and ``error`` is populated only when it did not.
    """

    entry_index: int
    record: dict[str, Any] | None
    addressed: AddressedPackedTrace | None
    error: str | None


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


def _required_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SemanticPackedStoreError(f"{field} must be a lowercase SHA-256")
    return value


def _frozen_process_identity_validator():
    """Return the frozen-identity validator or name the exact missing symbol."""

    validator = getattr(
        process_identity_module,
        "validate_frozen_process_identity",
        None,
    )
    if validator is None:
        raise SemanticPackedStoreError(
            "reading a semantic packed artifact under a pinned historical "
            "identity requires compose_v4.rewrite.editing_v2_process_identity."
            "validate_frozen_process_identity, which is absent"
        )
    return validator


def _validate_pinned_process_identity(value: object) -> dict[str, Any]:
    """Accept only a self-consistent, explicitly named historical identity."""

    if not isinstance(value, Mapping):
        raise SemanticPackedStoreError("expected_process_identity must be an object")
    validate = _frozen_process_identity_validator()
    try:
        validated = validate(dict(value))
    except ValueError as error:
        raise SemanticPackedStoreError(
            "pinned semantic process identity is not internally self-consistent"
        ) from error
    if not isinstance(validated, Mapping):
        raise SemanticPackedStoreError(
            "pinned semantic process identity validation did not return an object"
        )
    return dict(validated)


def _validate_source_binding(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SemanticPackedStoreError("source_binding must be an object")
    binding = dict(value)
    required = {
        "source_entry_count",
        "source_unified_manifest_sha256",
        "source_shard_name",
        "source_shard_sha256",
        "source_manifest_sha256",
        "source_overlay_sha256",
    }
    if set(binding) != required:
        raise SemanticPackedStoreError(
            "source_binding fields disagree with the semantic packed schema"
        )
    name = binding["source_shard_name"]
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise SemanticPackedStoreError(
            "source_binding.source_shard_name must be one nonempty basename"
        )
    for field in ("source_shard_sha256", "source_manifest_sha256"):
        _required_sha256(binding[field], field=f"source_binding.{field}")
    overlay = binding["source_overlay_sha256"]
    if overlay is not None:
        _required_sha256(
            overlay,
            field="source_binding.source_overlay_sha256",
        )
    entry_count = binding["source_entry_count"]
    if type(entry_count) is not int or entry_count < 0:
        raise SemanticPackedStoreError(
            "source_binding.source_entry_count must be nonnegative"
        )
    _required_sha256(
        binding["source_unified_manifest_sha256"],
        field="source_binding.source_unified_manifest_sha256",
    )
    return binding


def _validate_decision_binding(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SemanticPackedStoreError("decision_binding must be an object")
    binding = dict(value)
    required = {
        "decision_ledger_sha256",
        "source_count",
        "admitted_count",
        "rejected_count",
    }
    if set(binding) != required:
        raise SemanticPackedStoreError(
            "decision_binding fields disagree with the semantic packed schema"
        )
    _required_sha256(
        binding["decision_ledger_sha256"],
        field="decision_binding.decision_ledger_sha256",
    )
    for field in ("source_count", "admitted_count", "rejected_count"):
        count = binding[field]
        if type(count) is not int or count < 0:
            raise SemanticPackedStoreError(
                f"decision_binding.{field} must be nonnegative"
            )
    if binding["source_count"] != (
        binding["admitted_count"] + binding["rejected_count"]
    ):
        raise SemanticPackedStoreError(
            "decision binding does not account for every source trace"
        )
    return binding


@lru_cache(maxsize=1)
def semantic_packed_builder_identity() -> dict[str, object]:
    """Record the exact migration and storage implementation used to build rows."""

    root = Path(__file__).resolve().parents[3]
    relative_paths = (
        "src/compose_v4/data/semantic_packed_trace_store.py",
        "src/compose_v4/data/semantic_trace_migration_materializer.py",
        "src/compose_v4/rewrite/semantic_trace_migration.py",
        "src/compose_v4/rewrite/trace_shard_v3.py",
    )
    sources = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in relative_paths
    }
    body: dict[str, object] = {
        "schema": "compose.data.semantic_packed_builder_identity",
        "schema_version": 1,
        "implementation_source_sha256": sources,
    }
    return {**body, "identity_sha256": _canonical_sha256(body)}


def _manifest_self_hash(manifest: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )


def _completion_self_hash(completion: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {key: value for key, value in completion.items() if key != "completion_sha256"}
    )


def _artifact_files(artifact_dir: Path) -> tuple[Path, Path, Path]:
    root = Path(artifact_dir)
    return (
        root / SHARD_FILENAME,
        root / MANIFEST_FILENAME,
        root / COMPLETION_FILENAME,
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticPackedStoreError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise SemanticPackedStoreError(f"{label} must be an object")
    return value


def load_semantic_packed_manifest(
    artifact_dir: Path,
    *,
    expected_shard_sha256: str,
    expected_manifest_sha256: str,
    expected_source_binding: Mapping[str, object] | None = None,
    expected_process_identity: Mapping[str, object] | None = None,
    expected_builder_identity: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Validate the complete physical artifact and return its manifest.

    ``expected_process_identity`` and ``expected_builder_identity`` default to
    ``None``, which keeps the live-identity requirement and every other check
    exactly as it is today.  See the module docstring for why a pinned
    superseded identity is the only honest way to read historical bytes.
    """

    root = Path(artifact_dir)
    shard_path, manifest_path, completion_path = _artifact_files(root)
    if not root.is_dir():
        raise SemanticPackedStoreError(f"semantic packed artifact is absent: {root}")
    actual_files = {path.name for path in root.iterdir()}
    expected_files = {SHARD_FILENAME, MANIFEST_FILENAME, COMPLETION_FILENAME}
    if actual_files != expected_files:
        raise SemanticPackedStoreError(
            "semantic packed artifact file inventory is incomplete or unexpected"
        )
    manifest = _load_json_object(manifest_path, label="semantic packed manifest")
    completion = _load_json_object(
        completion_path,
        label="semantic packed completion receipt",
    )
    if set(manifest) != _MANIFEST_FIELDS:
        raise SemanticPackedStoreError("semantic packed manifest fields disagree")
    if set(completion) != _COMPLETION_FIELDS:
        raise SemanticPackedStoreError("semantic packed completion fields disagree")
    if manifest.get("schema") != STORE_SCHEMA:
        raise SemanticPackedStoreError("semantic packed manifest schema disagrees")
    if manifest.get("schema_version") != STORE_SCHEMA_VERSION:
        raise SemanticPackedStoreError("semantic packed manifest version disagrees")
    if manifest.get("status") != STORE_STATUS:
        raise SemanticPackedStoreError("semantic packed artifact status disagrees")
    if manifest.get("training_authorized") is not False:
        raise SemanticPackedStoreError(
            "semantic packed artifact cannot grant training authority"
        )
    if manifest.get("encoding") != ENCODING:
        raise SemanticPackedStoreError("semantic packed encoding disagrees")
    if manifest.get("shard_filename") != SHARD_FILENAME:
        raise SemanticPackedStoreError("semantic packed shard filename disagrees")
    if manifest.get("manifest_sha256") != _manifest_self_hash(manifest):
        raise SemanticPackedStoreError("semantic packed manifest self-hash disagrees")
    if completion.get("schema") != COMPLETION_SCHEMA:
        raise SemanticPackedStoreError("semantic packed completion schema disagrees")
    if completion.get("schema_version") != COMPLETION_SCHEMA_VERSION:
        raise SemanticPackedStoreError("semantic packed completion version disagrees")
    if completion.get("status") != "COMPLETE":
        raise SemanticPackedStoreError("semantic packed completion status disagrees")
    if completion.get("completion_sha256") != _completion_self_hash(completion):
        raise SemanticPackedStoreError("semantic packed completion self-hash disagrees")
    manifest_physical_sha256 = _file_sha256(manifest_path)
    if manifest_physical_sha256 != _required_sha256(
        expected_manifest_sha256,
        field="expected_manifest_sha256",
    ):
        raise SemanticPackedStoreError(
            "semantic packed manifest physical SHA-256 disagrees"
        )
    if completion.get("manifest_physical_sha256") != manifest_physical_sha256:
        raise SemanticPackedStoreError(
            "completion receipt does not bind the manifest bytes"
        )
    shard_sha256 = _file_sha256(shard_path)
    if shard_sha256 != _required_sha256(
        expected_shard_sha256,
        field="expected_shard_sha256",
    ):
        raise SemanticPackedStoreError(
            "semantic packed shard expected SHA-256 disagrees"
        )
    if manifest.get("shard_sha256") != shard_sha256:
        raise SemanticPackedStoreError("semantic packed shard SHA-256 disagrees")
    if completion.get("shard_sha256") != shard_sha256:
        raise SemanticPackedStoreError(
            "completion receipt does not bind the shard bytes"
        )
    process_sha256 = _required_sha256(
        manifest.get("process_identity_sha256"),
        field="process_identity_sha256",
    )
    if expected_process_identity is None:
        try:
            process_identity = require_editing_v2_process_identity(process_sha256)
        except ValueError as error:
            raise SemanticPackedStoreError(
                "semantic packed artifact process identity is stale"
            ) from error
    else:
        process_identity = _validate_pinned_process_identity(expected_process_identity)
        if process_identity.get("process_identity_sha256") != process_sha256:
            raise SemanticPackedStoreError(
                "semantic packed artifact pinned process identity disagrees"
            )
    if manifest.get("process_contract_sha256") != process_identity["contract_sha256"]:
        raise SemanticPackedStoreError("semantic packed process contract disagrees")
    if manifest.get("process_semantics") != PROCESS_SEMANTICS:
        raise SemanticPackedStoreError("semantic packed process semantics disagree")
    if completion.get("process_identity_sha256") != process_sha256:
        raise SemanticPackedStoreError(
            "semantic packed completion process identity disagrees"
        )
    if manifest.get("action_codec_schema_version") != action_codec_v4.SCHEMA_VERSION:
        raise SemanticPackedStoreError("semantic packed action codec version disagrees")
    if (
        manifest.get("action_codec_implementation_hash")
        != action_codec_v4.codec_implementation_hash()
    ):
        raise SemanticPackedStoreError(
            "semantic packed action codec identity disagrees"
        )
    if manifest.get("trace_schema") != TRACE_SCHEMA:
        raise SemanticPackedStoreError("semantic packed trace schema disagrees")
    if manifest.get("trace_schema_version") != TRACE_SCHEMA_VERSION:
        raise SemanticPackedStoreError("semantic packed trace version disagrees")
    if manifest.get("sampler_contract") != sampler_contract():
        raise SemanticPackedStoreError("semantic packed sampler contract disagrees")
    for field in ("data_lane", "split"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value:
            raise SemanticPackedStoreError(
                f"semantic packed manifest {field} must be nonempty"
            )
    builder = manifest.get("builder_identity")
    if not isinstance(builder, dict) or builder.get(
        "identity_sha256"
    ) != _canonical_sha256(
        {key: value for key, value in builder.items() if key != "identity_sha256"}
    ):
        raise SemanticPackedStoreError("semantic packed builder identity disagrees")
    if expected_builder_identity is not None:
        if not isinstance(expected_builder_identity, Mapping):
            raise SemanticPackedStoreError(
                "expected_builder_identity must be an object"
            )
        if builder != dict(expected_builder_identity):
            raise SemanticPackedStoreError(
                "semantic packed pinned builder identity disagrees"
            )
    source_binding = _validate_source_binding(manifest.get("source_binding"))
    decision_binding = _validate_decision_binding(manifest.get("decision_binding"))
    if source_binding["source_entry_count"] != decision_binding["source_count"]:
        raise SemanticPackedStoreError(
            "semantic packed source and decision censuses disagree"
        )
    if manifest.get("entries") != decision_binding["admitted_count"]:
        raise SemanticPackedStoreError(
            "semantic packed admitted census disagrees with its decisions"
        )
    if expected_source_binding is not None:
        expected = _validate_source_binding(expected_source_binding)
        if source_binding != expected:
            raise SemanticPackedStoreError("semantic packed source binding disagrees")
    for field in ("entries", "states", "actions"):
        value = manifest.get(field)
        if type(value) is not int or value < 0:
            raise SemanticPackedStoreError(
                f"semantic packed manifest {field} must be nonnegative"
            )
    return manifest


def _write_staging_artifact(
    staging: Path,
    records: Iterable[Mapping[str, object]],
    *,
    data_lane: str,
    split: str,
    source_binding: Mapping[str, object],
    decision_binding: Mapping[str, object],
) -> dict[str, Any]:
    shard_path, manifest_path, completion_path = _artifact_files(staging)
    binding = _validate_source_binding(source_binding)
    decisions = _validate_decision_binding(decision_binding)
    if not data_lane or not split:
        raise SemanticPackedStoreError("data_lane and split must be nonempty")
    process_identity = editing_v2_process_identity()
    builder_identity = semantic_packed_builder_identity()
    row_stream = hashlib.sha256()
    counts: Counter[str] = Counter()
    families: Counter[str] = Counter()
    seen_trace_ids: set[str] = set()

    raw_handle = shard_path.open("wb")
    try:
        compressed = gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        )
        try:
            for index, raw_record in enumerate(records):
                record = dict(raw_record)
                try:
                    trace = decode_semantic_trace_record(record, validate_replay=True)
                except SemanticTraceShardError as error:
                    raise SemanticPackedStoreError(
                        f"semantic packed row {index} failed exact replay"
                    ) from error
                trace_id = record["trace_id"]
                if trace_id in seen_trace_ids:
                    raise SemanticPackedStoreError(
                        f"duplicate trace_id {trace_id!r} in semantic packed shard"
                    )
                seen_trace_ids.add(str(trace_id))
                source_address = record.get("source_address")
                if not isinstance(source_address, Mapping) or not source_address:
                    raise SemanticPackedStoreError(
                        f"semantic packed row {index} lacks its source address"
                    )
                if record["data_lane"] != data_lane or record["split"] != split:
                    raise SemanticPackedStoreError(
                        f"semantic packed row {index} leaves its declared lane or split"
                    )
                line = _canonical_json_bytes(record, newline=True)
                row_stream.update(line)
                compressed.write(line)
                counts["entries"] += 1
                counts["states"] += len(record["states"])
                counts["actions"] += len(trace.steps)
                families.update(record["family_histogram"])
        finally:
            compressed.close()
        raw_handle.flush()
        os.fsync(raw_handle.fileno())
    finally:
        raw_handle.close()

    manifest_body: dict[str, Any] = {
        "schema": STORE_SCHEMA,
        "schema_version": STORE_SCHEMA_VERSION,
        "status": STORE_STATUS,
        "training_authorized": False,
        "encoding": ENCODING,
        "shard_filename": SHARD_FILENAME,
        "shard_sha256": _file_sha256(shard_path),
        "record_stream_sha256": row_stream.hexdigest(),
        "entries": counts["entries"],
        "states": counts["states"],
        "actions": counts["actions"],
        "family_histogram": dict(sorted(families.items())),
        "data_lane": data_lane,
        "split": split,
        "source_binding": binding,
        "decision_binding": decisions,
        "process_semantics": PROCESS_SEMANTICS,
        "process_identity_sha256": process_identity["process_identity_sha256"],
        "process_contract_sha256": process_identity["contract_sha256"],
        "action_codec_schema_version": action_codec_v4.SCHEMA_VERSION,
        "action_codec_implementation_hash": action_codec_v4.codec_implementation_hash(),
        "trace_schema": TRACE_SCHEMA,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "sampler_contract": sampler_contract(),
        "builder_identity": builder_identity,
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": _canonical_sha256(manifest_body),
    }
    if decisions["admitted_count"] != counts["entries"]:
        raise SemanticPackedStoreError(
            "decision ledger admitted count does not equal the semantic row count"
        )
    if decisions["source_count"] != binding["source_entry_count"]:
        raise SemanticPackedStoreError(
            "decision ledger source count does not equal the frozen source census"
        )
    manifest_bytes = _canonical_json_bytes(manifest, newline=True)
    manifest_path.write_bytes(manifest_bytes)
    completion_body: dict[str, Any] = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": "COMPLETE",
        "manifest_physical_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "shard_sha256": manifest["shard_sha256"],
        "process_identity_sha256": manifest["process_identity_sha256"],
    }
    completion = {
        **completion_body,
        "completion_sha256": _canonical_sha256(completion_body),
    }
    completion_path.write_bytes(_canonical_json_bytes(completion, newline=True))
    return manifest


def write_semantic_packed_artifact(
    artifact_dir: Path,
    records: Iterable[Mapping[str, object]],
    *,
    data_lane: str,
    split: str,
    source_binding: Mapping[str, object],
    decision_binding: Mapping[str, object],
) -> dict[str, Any]:
    """Replay, validate, and atomically publish one complete semantic shard."""

    target = Path(artifact_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".staging",
        )
    )
    try:
        manifest = _write_staging_artifact(
            staging,
            records,
            data_lane=data_lane,
            split=split,
            source_binding=source_binding,
            decision_binding=decision_binding,
        )
        staging_shard, staging_manifest, _ = _artifact_files(staging)
        staging_shard_sha256 = _file_sha256(staging_shard)
        staging_manifest_sha256 = _file_sha256(staging_manifest)
        # Prove the staged artifact complete before it becomes visible.
        load_semantic_packed_manifest(
            staging,
            expected_shard_sha256=staging_shard_sha256,
            expected_manifest_sha256=staging_manifest_sha256,
            expected_source_binding=source_binding,
        )
        if target.exists():
            existing = load_semantic_packed_manifest(
                target,
                expected_shard_sha256=staging_shard_sha256,
                expected_manifest_sha256=staging_manifest_sha256,
                expected_source_binding=source_binding,
            )
            if existing != manifest:
                raise SemanticPackedStoreError(
                    f"immutable semantic packed artifact collision at {target}"
                )
            return existing
        try:
            os.rename(staging, target)
        except OSError:
            if not target.exists():
                raise
            existing = load_semantic_packed_manifest(
                target,
                expected_shard_sha256=staging_shard_sha256,
                expected_manifest_sha256=staging_manifest_sha256,
                expected_source_binding=source_binding,
            )
            if existing != manifest:
                raise SemanticPackedStoreError(
                    f"immutable semantic packed artifact collision at {target}"
                )
            return existing
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def read_semantic_packed_artifact(
    artifact_dir: Path,
    *,
    expected_shard_sha256: str,
    expected_manifest_sha256: str,
    expected_source_binding: Mapping[str, object] | None = None,
    expected_process_identity: Mapping[str, object] | None = None,
    expected_builder_identity: Mapping[str, object] | None = None,
    sentinel_replay_entries: int = DEFAULT_SENTINEL_REPLAY_ENTRIES,
) -> Iterable[AddressedPackedTrace]:
    """Yield exact addressed paths with deterministic first-N replay.

    The optional pinned identities are forwarded unchanged to
    :func:`load_semantic_packed_manifest`; omitting them keeps today's exact
    live-identity behaviour.
    """

    if type(sentinel_replay_entries) is not int or sentinel_replay_entries < 0:
        raise ValueError("sentinel_replay_entries must be a nonnegative integer")
    root = Path(artifact_dir)
    shard_path, _, _ = _artifact_files(root)
    manifest = load_semantic_packed_manifest(
        root,
        expected_shard_sha256=expected_shard_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_source_binding=expected_source_binding,
        expected_process_identity=expected_process_identity,
        expected_builder_identity=expected_builder_identity,
    )
    shard_sha256 = str(manifest["shard_sha256"])
    seen_trace_ids: set[str] = set()
    row_stream = hashlib.sha256()
    observed: Counter[str] = Counter()
    families: Counter[str] = Counter()
    with gzip.open(shard_path, "rb") as handle:
        for entry_index, raw_line in enumerate(handle):
            if not raw_line.strip():
                continue
            row_stream.update(raw_line)
            try:
                record = json.loads(raw_line)
                trace = decode_semantic_trace_record(
                    record,
                    validate_replay=entry_index < sentinel_replay_entries,
                    expected_process_identity=expected_process_identity,
                )
                states = tuple(decode_state(payload) for payload in record["states"])
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                SemanticTraceShardError,
            ) as error:
                raise SemanticPackedStoreError(
                    f"semantic packed row {entry_index} is malformed"
                ) from error
            trace_id = str(record["trace_id"])
            if trace_id in seen_trace_ids:
                raise SemanticPackedStoreError(
                    f"duplicate trace_id {trace_id!r} in semantic packed shard"
                )
            seen_trace_ids.add(trace_id)
            path = PackedTraceProgress(trace, states)
            canonical_keys = record["canonical_state_keys"]
            address = PackedTraceAddress(
                packed_shard_content_sha256=shard_sha256,
                packed_shard_name=SHARD_FILENAME,
                entry_index=entry_index,
                trace_id=trace_id,
                layer=str(record["data_lane"]),
                partition=str(record["split"]),
                source_key=str(canonical_keys[0]),
                target_key=str(canonical_keys[-1]),
                path_length=int(record["path_length"]),
            )
            observed["entries"] += 1
            observed["states"] += len(states)
            observed["actions"] += len(trace.steps)
            families.update(record["family_histogram"])
            yield AddressedPackedTrace(address=address, trace=trace, path=path)
    for field in ("entries", "states", "actions"):
        if observed[field] != manifest[field]:
            raise SemanticPackedStoreError(
                f"semantic packed {field} census disagrees with its manifest"
            )
    if row_stream.hexdigest() != manifest["record_stream_sha256"]:
        raise SemanticPackedStoreError(
            "semantic packed record stream SHA-256 disagrees"
        )
    if dict(sorted(families.items())) != manifest["family_histogram"]:
        raise SemanticPackedStoreError("semantic packed family census disagrees")


def _validate_semantic_entry_range(
    *,
    entry_start: int,
    entry_stop: int,
    entries: int,
) -> tuple[int, int]:
    if type(entry_start) is not int or type(entry_stop) is not int:
        raise ValueError("semantic packed entry-range bounds must be integers")
    if type(entries) is not int or entries < 0:
        raise ValueError("semantic packed entry census must be a nonnegative integer")
    if entry_start < 0 or entry_stop < 0 or entry_start > entry_stop:
        raise ValueError(
            "semantic packed entry range must be a nonnegative half-open interval"
        )
    if entry_stop > entries:
        raise ValueError(
            f"semantic packed entry range [{entry_start}, {entry_stop}) is outside "
            f"the manifest census [0, {entries})"
        )
    if entry_start == entry_stop and not (entries == 0 and entry_start == 0):
        raise ValueError(
            "an empty semantic packed entry range is valid only for an empty shard"
        )
    return entry_start, entry_stop


def validate_semantic_packed_entry_ranges(
    entry_ranges: Iterable[tuple[int, int]],
    *,
    entries: int,
) -> tuple[tuple[int, int], ...]:
    """Require ordered half-open ranges to partition one exact shard census.

    This is the range-level completeness contract used by planners and
    reducers.  Individual range readers deliberately do not claim that they
    observed the complete shard; this function proves that an ordered set of
    independently readable ranges has neither a gap nor an overlap.
    """

    if type(entries) is not int or entries < 0:
        raise ValueError("semantic packed entry census must be a nonnegative integer")
    try:
        raw_ranges = tuple(entry_ranges)
    except TypeError as error:
        raise ValueError("semantic packed entry ranges must be iterable") from error
    if not raw_ranges:
        raise ValueError("semantic packed entry-range partition cannot be empty")

    normalized: list[tuple[int, int]] = []
    cursor = 0
    for index, raw_range in enumerate(raw_ranges):
        if not isinstance(raw_range, (tuple, list)) or len(raw_range) != 2:
            raise ValueError(
                f"semantic packed entry range {index} must contain two bounds"
            )
        entry_start, entry_stop = _validate_semantic_entry_range(
            entry_start=raw_range[0],
            entry_stop=raw_range[1],
            entries=entries,
        )
        if entry_start > cursor:
            raise ValueError(
                f"semantic packed entry-range partition has a gap before {entry_start}"
            )
        if entry_start < cursor:
            raise ValueError(
                f"semantic packed entry-range partition overlaps before {cursor}"
            )
        normalized.append((entry_start, entry_stop))
        cursor = entry_stop
    if cursor != entries:
        raise ValueError(
            "semantic packed entry-range partition does not cover the manifest census: "
            f"covered [0, {cursor}), expected [0, {entries})"
        )
    return tuple(normalized)


def read_semantic_packed_artifact_range_rows(
    artifact_dir: Path,
    *,
    expected_shard_sha256: str,
    expected_manifest_sha256: str,
    entry_start: int,
    entry_stop: int,
    expected_source_binding: Mapping[str, object] | None = None,
    expected_process_identity: Mapping[str, object] | None = None,
    expected_builder_identity: Mapping[str, object] | None = None,
    sentinel_replay_entries: int = DEFAULT_SENTINEL_REPLAY_ENTRIES,
    recover_row_errors: bool = False,
) -> Iterator[SemanticPackedRowRead]:
    """Yield one validated half-open range as addressable row reads.

    This is the single implementation behind
    :func:`read_semantic_packed_artifact_range`; see that function for the
    range and validation contract.  ``recover_row_errors=False`` reproduces it
    exactly, raising :class:`SemanticPackedStoreError` on the first unreadable
    row.  ``recover_row_errors=True`` instead yields a
    :class:`SemanticPackedRowRead` whose ``error`` names the exact failure, so
    a proof scan can account for every physical entry it read.  Only per-row
    decode, lane/split and duplicate-identity failures are recoverable; every
    artifact-level failure still raises.
    """

    if type(sentinel_replay_entries) is not int or sentinel_replay_entries < 0:
        raise ValueError("sentinel_replay_entries must be a nonnegative integer")
    if type(recover_row_errors) is not bool:
        raise ValueError("recover_row_errors must be a boolean")
    root = Path(artifact_dir)
    shard_path, _, _ = _artifact_files(root)
    manifest = load_semantic_packed_manifest(
        root,
        expected_shard_sha256=expected_shard_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_source_binding=expected_source_binding,
        expected_process_identity=expected_process_identity,
        expected_builder_identity=expected_builder_identity,
    )
    entry_start, entry_stop = _validate_semantic_entry_range(
        entry_start=entry_start,
        entry_stop=entry_stop,
        entries=manifest["entries"],
    )
    shard_sha256 = str(manifest["shard_sha256"])

    def rows() -> Iterator[SemanticPackedRowRead]:
        observed_entries = 0
        seen_trace_ids: set[str] = set()
        with gzip.open(shard_path, "rb") as handle:
            for entry_index, raw_line in enumerate(handle):
                if not raw_line.strip():
                    raise SemanticPackedStoreError(
                        f"semantic packed row {entry_index} is blank"
                    )
                if entry_index < entry_start:
                    continue
                if entry_index >= entry_stop:
                    break
                observed_entries += 1
                record: object = None
                try:
                    record = json.loads(raw_line)
                    trace = decode_semantic_trace_record(
                        record,
                        validate_replay=entry_index < sentinel_replay_entries,
                        expected_process_identity=expected_process_identity,
                    )
                    states = tuple(
                        decode_state(payload) for payload in record["states"]
                    )
                except (
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                    SemanticTraceShardError,
                ) as error:
                    if not recover_row_errors:
                        raise SemanticPackedStoreError(
                            f"semantic packed row {entry_index} is malformed"
                        ) from error
                    yield SemanticPackedRowRead(
                        entry_index=entry_index,
                        record=record if isinstance(record, dict) else None,
                        addressed=None,
                        error=f"semantic packed row {entry_index} is malformed: {error}",
                    )
                    continue
                if (
                    record["data_lane"] != manifest["data_lane"]
                    or record["split"] != manifest["split"]
                ):
                    detail = (
                        f"semantic packed row {entry_index} leaves its declared lane or split"
                    )
                    if not recover_row_errors:
                        raise SemanticPackedStoreError(detail)
                    yield SemanticPackedRowRead(
                        entry_index=entry_index,
                        record=dict(record),
                        addressed=None,
                        error=detail,
                    )
                    continue
                trace_id = str(record["trace_id"])
                if trace_id in seen_trace_ids:
                    detail = (
                        f"duplicate trace_id {trace_id!r} in semantic packed entry range"
                    )
                    if not recover_row_errors:
                        raise SemanticPackedStoreError(detail)
                    yield SemanticPackedRowRead(
                        entry_index=entry_index,
                        record=dict(record),
                        addressed=None,
                        error=detail,
                    )
                    continue
                seen_trace_ids.add(trace_id)
                path = PackedTraceProgress(trace, states)
                canonical_keys = record["canonical_state_keys"]
                address = PackedTraceAddress(
                    packed_shard_content_sha256=shard_sha256,
                    packed_shard_name=SHARD_FILENAME,
                    entry_index=entry_index,
                    trace_id=trace_id,
                    layer=str(record["data_lane"]),
                    partition=str(record["split"]),
                    source_key=str(canonical_keys[0]),
                    target_key=str(canonical_keys[-1]),
                    path_length=int(record["path_length"]),
                )
                yield SemanticPackedRowRead(
                    entry_index=entry_index,
                    record=dict(record),
                    addressed=AddressedPackedTrace(
                        address=address,
                        trace=trace,
                        path=path,
                    ),
                    error=None,
                )

        expected_entries = entry_stop - entry_start
        if observed_entries != expected_entries:
            raise SemanticPackedStoreError(
                f"semantic packed entry range [{entry_start}, {entry_stop}) expected "
                f"{expected_entries} rows but observed {observed_entries}"
            )

    return rows()


def read_semantic_packed_artifact_range(
    artifact_dir: Path,
    *,
    expected_shard_sha256: str,
    expected_manifest_sha256: str,
    entry_start: int,
    entry_stop: int,
    expected_source_binding: Mapping[str, object] | None = None,
    expected_process_identity: Mapping[str, object] | None = None,
    expected_builder_identity: Mapping[str, object] | None = None,
    sentinel_replay_entries: int = DEFAULT_SENTINEL_REPLAY_ENTRIES,
) -> Iterable[AddressedPackedTrace]:
    """Yield one validated half-open range with original shard addresses.

    The complete artifact inventory, manifest bytes, shard bytes, completion
    receipt, process identity, and optional source binding are validated before
    a range is exposed.  Rows in ``[entry_start, entry_stop)`` retain their
    original physical entry indices and undergo the same trace-v3 validation
    as the full reader.  Sentinel replay remains global to the shard: only
    selected rows whose original index is below ``sentinel_replay_entries``
    execute chemistry replay.

    Range exhaustion proves exactly the requested row count, not a full-shard
    semantic census.  Call :func:`validate_semantic_packed_entry_ranges` on the
    complete planned range set before treating independent results as an exact
    shard partition.  This reader is a bounded correctness/reference primitive,
    not the distributed Active8 hot path: repeated ranges over one gzip source
    would repeatedly hash the artifact and decompress its prefix.  Distributed
    admission should consume the separately validated semantic Active8 chunk
    cache instead.

    The optional pinned identities are forwarded unchanged to
    :func:`load_semantic_packed_manifest`; omitting them keeps today's exact
    live-identity behaviour.
    """

    reads = read_semantic_packed_artifact_range_rows(
        artifact_dir,
        expected_shard_sha256=expected_shard_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        entry_start=entry_start,
        entry_stop=entry_stop,
        expected_source_binding=expected_source_binding,
        expected_process_identity=expected_process_identity,
        expected_builder_identity=expected_builder_identity,
        sentinel_replay_entries=sentinel_replay_entries,
        recover_row_errors=False,
    )

    def addressed_rows() -> Iterable[AddressedPackedTrace]:
        for read in reads:
            if read.addressed is None:
                raise SemanticPackedStoreError(
                    f"semantic packed row {read.entry_index} is malformed"
                )
            yield read.addressed

    return addressed_rows()


__all__ = [
    "COMPLETION_FILENAME",
    "DEFAULT_SENTINEL_REPLAY_ENTRIES",
    "MANIFEST_FILENAME",
    "SHARD_FILENAME",
    "STORE_SCHEMA",
    "STORE_SCHEMA_VERSION",
    "STORE_STATUS",
    "SemanticPackedRowRead",
    "SemanticPackedStoreError",
    "load_semantic_packed_manifest",
    "read_semantic_packed_artifact",
    "read_semantic_packed_artifact_range",
    "read_semantic_packed_artifact_range_rows",
    "semantic_packed_builder_identity",
    "validate_semantic_packed_entry_ranges",
    "write_semantic_packed_artifact",
]
