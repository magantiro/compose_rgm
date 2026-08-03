"""A mechanical, content-addressed chunk cache over the exact V1 payload.

Why this exists
---------------

Every Process-V2 consumer of the migrated payload currently addresses it by
*entry range*, and the only range reader available
(``semantic_packed_trace_store.read_semantic_packed_artifact_range_rows``)
hashes the whole compressed shard and then decompresses from byte zero to reach
its range start.  Its own docstring says so.  With the rebind's
``DEFAULT_ENTRIES_PER_TASK = 64`` over a ~32k-entry shard that is ~505 range
tasks per shard, each paying one full-file SHA-256 and a decompression prefix,
so the decompressed work grows as ``entries x ranges / 2``.  The cost is
triangular in the number of ranges and nothing about it is scientific.

This module removes the rescan instead of tuning it.  Each of the twenty source
shards is read **exactly once**, in a single fused pass that simultaneously

* hashes the raw compressed bytes,
* decompresses them,
* hashes the decompressed record stream,
* validates every row's identity, and
* writes deterministic bounded chunk files,

after which every downstream worker reads complete chunk files and never opens
``traces.jsonl.gz`` again.

What it is, and is not
----------------------

This is **mechanical infrastructure, not evidence**.  It preserves every source
record byte for byte: a chunk is the verbatim concatenation of its source lines,
so nothing is re-encoded, no state is reconstructed from SMILES, and the record
identities the cache publishes are the ones the payload already carried.  It
decides nothing, admits nothing and authorizes nothing.

Two identities, deliberately separate
-------------------------------------

``cache_semantic_identity_sha256`` addresses *what was cached*: the completion,
the pinned identities, the codec, and the ordered record identity of each of the
twenty sources.  It is invariant to ``records_per_chunk``, to how many
containers ran, and to the order in which they ran.

``cache_physical_identity_sha256`` addresses *how it was produced*: the semantic
identity plus the source revision, the output prefix and the chunk size.  It is
the run address, so changing the chunk size relocates the artifact without
changing what the artifact means.  A reviewer comparing two schedules compares
the first and expects equality; comparing the second and expecting equality
would be a category error.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from compose_v4.data.editing_process_v2_rebind import (
    ProcessV2RebindError,
    mounted_process_v2_artifact_path,
)
from compose_v4.data.editing_v2_process_v2_completion_binder import (
    ProcessV2ExactCompletionBinding,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    CHUNK_CACHE_COMPLETION_SCHEMA,
    CHUNK_CACHE_COMPLETION_SCHEMA_VERSION,
    CHUNK_MANIFEST_SCHEMA,
    CHUNK_MANIFEST_SCHEMA_VERSION,
    authority_false_block,
    canonical_bytes,
    canonical_sha256,
    require_authority_false,
)
from compose_v4.data.semantic_packed_trace_store import (
    MANIFEST_FILENAME as SEMANTIC_MANIFEST_FILENAME,
)
from compose_v4.data.semantic_packed_trace_store import (
    SHARD_FILENAME as SEMANTIC_SHARD_FILENAME,
)
from compose_v4.data.semantic_trace_migration_materializer import (
    SEMANTIC_ARTIFACT_DIRNAME,
)
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite.trace_shard_v3 import TRACE_SCHEMA, TRACE_SCHEMA_VERSION

# ---- Frozen artifact identity ------------------------------------------------

PLAN_SCHEMA = "compose.data.editing_v2_process_v2_chunk_cache_plan"
PLAN_SCHEMA_VERSION = 1
PLAN_STATUS = "FROZEN_MECHANICAL_CHUNK_PLAN_NO_DOWNSTREAM_AUTHORITY"
MANIFEST_SCHEMA = CHUNK_MANIFEST_SCHEMA
MANIFEST_SCHEMA_VERSION = CHUNK_MANIFEST_SCHEMA_VERSION
MANIFEST_STATUS = "COMPLETE_MECHANICAL_SOURCE_CHUNKS_NO_DOWNSTREAM_AUTHORITY"
COMPLETION_SCHEMA = CHUNK_CACHE_COMPLETION_SCHEMA
COMPLETION_SCHEMA_VERSION = CHUNK_CACHE_COMPLETION_SCHEMA_VERSION
COMPLETION_STATUS = "COMPLETE_MECHANICAL_CHUNK_CACHE_NO_DOWNSTREAM_AUTHORITY"
REFUSAL_SCHEMA = "compose.data.editing_v2_process_v2_chunk_cache_refusal"
REFUSAL_SCHEMA_VERSION = 1
REFUSAL_STATUS = "INTEGRITY_REFUSAL_NOTHING_PUBLISHED_UNDER_THE_RUN_NAMESPACE"

PLAN_FILENAME = "PROCESS_V2_CHUNK_CACHE_PLAN.json"
MANIFEST_FILENAME = "CHUNK_MANIFEST.json"
COMPLETION_FILENAME = "PROCESS_V2_CHUNK_CACHE_COMPLETE.json"
SOURCE_DIRNAME = "sources"
CHUNK_ENCODING = "verbatim_source_lines_deterministic_gzip_mtime_0"
CHUNK_ADDRESS_RULE = "entry_start_plus_chunk_local_row_index"

DEFAULT_OUTPUT_ARTIFACT_PREFIX = "/artifacts/editing_v2/process_v2_chunk_cache"
# One chunk is the unit a downstream worker takes. Small enough that 40 workers
# get real parallelism over 20 shards, large enough that a shard does not emit
# tens of thousands of tiny volume objects.
DEFAULT_RECORDS_PER_CHUNK = 2048
MAX_RECORDS_PER_CHUNK = 65_536
# The fused pass reads the raw gzip in blocks of this size; peak memory is one
# read block, one bounded decompression block and one chunk, independent of
# shard size. The decompression bound is what keeps a highly compressible shard
# from expanding a single read block into the whole corpus.
READ_BLOCK_BYTES = 1 << 20
DECOMPRESS_BLOCK_BYTES = 1 << 18

_CHUNK_RECORD_FIELDS: frozenset[str] = frozenset(
    {
        "chunk_index",
        "entry_start",
        "entry_stop",
        "row_count",
        "uncompressed_sha256",
        "uncompressed_bytes",
        "chunk_filename",
        "chunk_file_sha256",
    }
)
_REQUIRED_ROW_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "schema_version",
        "trace_id",
        "data_lane",
        "split",
        "path_length",
        "record_sha256",
        "process_identity_sha256",
        "process_contract_sha256",
        "action_codec_schema_version",
        "action_codec_implementation_hash",
    }
)


# ---- Bounded execution fan-out -------------------------------------------------
#
# Both Process-V2 map/reduce launchers import these, so the platform container
# ceiling has exactly one home and cannot drift between two apps. They are pure
# and Modal-free on purpose: a launcher's concurrency behaviour has to be
# testable without a Modal client.

# The owner's hard ceiling. A request above it is refused, never clamped:
# silently clamping is what made the previous `--max-map-containers` cosmetic.
PLATFORM_MAX_MAP_CONTAINERS = 40
# Cache construction is one reader per source shard and there are twenty source
# shards, so twenty is the whole job.
DEFAULT_CACHE_MAP_CONTAINERS = 20
# CPU-bound chunk work after the cache exists may use the full ceiling.
DEFAULT_CHUNK_MAP_CONTAINERS = 40


class ProcessV2ConcurrencyError(RuntimeError):
    """A requested map fan-out is outside the platform bound."""


def validate_map_container_bound(
    value: object, *, platform_max: int = PLATFORM_MAX_MAP_CONTAINERS
) -> int:
    """Refuse, never clamp, a request outside ``[1, platform_max]``."""

    if type(value) is not int or not 1 <= value <= platform_max:
        raise ProcessV2ConcurrencyError(
            f"max_map_containers must be an integer in [1, {platform_max}]; the map "
            "fan-out bounds concurrent volume writers and is a platform limit, not a hint"
        )
    return value


def plan_submission_waves(
    task_ids: Sequence[str], max_map_containers: int
) -> tuple[tuple[str, ...], ...]:
    """Partition tasks into deterministic sequential waves of at most N.

    Sorted, so the wave a task lands in depends on the task set and the bound
    alone, never on the order a caller happened to discover missing work.
    """

    bound = validate_map_container_bound(max_map_containers)
    ordered = sorted(str(task_id) for task_id in task_ids)
    if len(set(ordered)) != len(ordered):
        raise ProcessV2ConcurrencyError("the bounded map was given a repeated task identity")
    return tuple(tuple(ordered[start : start + bound]) for start in range(0, len(ordered), bound))


def run_bounded_map(
    task_ids: Sequence[str],
    *,
    max_map_containers: int,
    submit: Callable[[tuple[str, ...]], Iterable[Any]],
    set_autoscaler: Callable[..., Any] | None = None,
) -> list[Any]:
    """Run every task in bounded waves, blocking between them.

    ``submit`` is called once per wave and must not return until that wave is
    finished, which is what bounds *submitted* work.  ``set_autoscaler`` lowers
    the function's own container ceiling to the same number; on Modal 1.3.5 that
    is ``Function.update_autoscaler(max_containers=...)``, the only supported
    surface (``Function.with_options`` does not exist in this version).

    Neither mechanism guarantees N simultaneously live containers: Modal may run
    fewer.  The guarantee is an upper bound, which is the direction that matters
    for a volume writer.
    """

    bound = validate_map_container_bound(max_map_containers)
    waves = plan_submission_waves(task_ids, bound)
    if set_autoscaler is not None:
        set_autoscaler(max_containers=bound)
    results: list[Any] = []
    for index, wave in enumerate(waves):
        wave_results = list(submit(wave))
        if len(wave_results) != len(wave):
            raise ProcessV2ConcurrencyError(
                f"bounded map wave {index} lost or repeated a task result"
            )
        results.extend(wave_results)
    if len(results) != len({str(task_id) for task_id in task_ids}):
        raise ProcessV2ConcurrencyError("the bounded map did not cover every task exactly once")
    return results


class ProcessV2ChunkCacheError(RuntimeError):
    """The chunk cache cannot be planned, built, or reduced."""


class ProcessV2ChunkCacheIncomplete(ProcessV2ChunkCacheError):
    """A planned source cache is absent."""


class ProcessV2ChunkCacheIntegrityError(ProcessV2ChunkCacheError):
    """A source shard disagrees with the identity the payload published for it.

    Raised before any staging directory is created, so an integrity refusal
    publishes nothing at all under the run namespace.
    """


# ---- Deterministic serialization ---------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(READ_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _self_hash(value: Mapping[str, Any], *, field: str) -> str:
    return canonical_sha256({key: item for key, item in value.items() if key != field})


def _publish_json_atomically(target: Path, payload: Mapping[str, Any], *, label: str) -> Path:
    content = canonical_bytes(payload) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != content:
            raise ProcessV2ChunkCacheError(f"immutable {label} collision at {target}")
        return target
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return target


# ---- The one fused pass -------------------------------------------------------


@dataclass(frozen=True)
class FusedSourceScan:
    """What one pass over one raw gzip shard observed."""

    shard_file_sha256: str
    record_stream_sha256: str
    record_identity_sha256: str
    entries: int
    compressed_bytes: int
    decompressed_bytes: int
    chunks: tuple[dict[str, Any], ...]


class _FusedGzipReader:
    """One ``open``, one sequential read, one digest, one decompression.

    ``gzip.open`` decompresses without exposing the compressed bytes, so the
    physical shard hash would need a second full read of the same file.  Driving
    ``zlib.decompressobj`` directly fuses the two, which is the whole point of
    this module: the raw digest, the record-stream digest and the chunk writer
    all consume the same single pass.
    """

    def __init__(self, path: Path, *, opener: Callable[..., BinaryIO] = open) -> None:
        self._path = Path(path)
        self._opener = opener
        self._raw_digest = hashlib.sha256()
        self.compressed_bytes = 0
        self.decompressed_bytes = 0

    @property
    def shard_file_sha256(self) -> str:
        return self._raw_digest.hexdigest()

    def lines(self) -> Iterator[bytes]:
        decompressor = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
        pending = bytearray()
        read_any = False
        with self._opener(self._path, "rb") as handle:
            while block := handle.read(READ_BLOCK_BYTES):
                read_any = True
                self._raw_digest.update(block)
                self.compressed_bytes += len(block)
                feed: bytes = block
                # `decompress` is bounded by `max_length` and the unconsumed
                # input is carried in `unconsumed_tail`. Without that bound one
                # compressed block of highly repetitive rows expands to the
                # whole shard, and peak memory becomes the decompressed size.
                while feed or decompressor.unconsumed_tail:
                    if decompressor.unconsumed_tail:
                        source = decompressor.unconsumed_tail
                    else:
                        source, feed = feed, b""
                    pending += decompressor.decompress(source, DECOMPRESS_BLOCK_BYTES)
                    cut = pending.rfind(b"\n")
                    if cut >= 0:
                        yield from self._emit(bytes(pending[: cut + 1]))
                        del pending[: cut + 1]
                    if decompressor.eof:
                        leftover = decompressor.unused_data
                        if not leftover:
                            break
                        # A concatenated gzip member starts in the leftover.
                        decompressor = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
                        feed = leftover
        pending += decompressor.flush()
        if read_any and not decompressor.eof:
            raise ProcessV2ChunkCacheIntegrityError(
                f"the semantic shard gzip stream is truncated: {self._path}"
            )
        yield from self._emit(bytes(pending))

    def _emit(self, payload: bytes) -> Iterator[bytes]:
        # One line at a time. ``splitlines`` would materialise every line of the
        # block simultaneously, which puts a second copy of the block on the
        # heap for no benefit.
        start = 0
        while (index := payload.find(b"\n", start)) >= 0:
            line = payload[start : index + 1]
            self.decompressed_bytes += len(line)
            yield line
            start = index + 1
        if start < len(payload):
            tail = payload[start:]
            self.decompressed_bytes += len(tail)
            yield tail


def scan_source_shard_once(
    semantic_dir: Path,
    *,
    data_lane: str,
    split: str,
    expected_shard_sha256: str,
    expected_manifest_sha256: str,
    pinned_process_identity_sha256: str,
    records_per_chunk: int,
    chunk_sink: Callable[[int, int, int, bytes], dict[str, Any]] | None = None,
    opener: Callable[..., BinaryIO] = open,
) -> FusedSourceScan:
    """Validate and chunk one source shard in exactly one raw read.

    ``chunk_sink`` receives ``(chunk_index, entry_start, entry_stop, payload)``
    and returns the published chunk record; passing ``None`` scans without
    writing, which is what the audit path uses.
    """

    manifest_path = Path(semantic_dir) / SEMANTIC_MANIFEST_FILENAME
    shard_path = Path(semantic_dir) / SEMANTIC_SHARD_FILENAME
    if not manifest_path.is_file() or not shard_path.is_file():
        raise ProcessV2ChunkCacheIncomplete(f"the semantic packed artifact is absent: {semantic_dir}")
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
        raise ProcessV2ChunkCacheIntegrityError(
            f"the semantic manifest physical hash disagrees: {manifest_path}"
        )
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, Mapping):
        raise ProcessV2ChunkCacheIntegrityError(f"the semantic manifest must be an object: {manifest_path}")
    if manifest.get("manifest_sha256") != _self_hash(manifest, field="manifest_sha256"):
        raise ProcessV2ChunkCacheIntegrityError(
            f"the semantic manifest self-hash disagrees: {manifest_path}"
        )
    if (
        manifest.get("shard_sha256") != expected_shard_sha256
        or manifest.get("shard_filename") != SEMANTIC_SHARD_FILENAME
        or manifest.get("data_lane") != data_lane
        or manifest.get("split") != split
        or manifest.get("process_identity_sha256") != pinned_process_identity_sha256
        or manifest.get("trace_schema") != TRACE_SCHEMA
        or manifest.get("trace_schema_version") != TRACE_SCHEMA_VERSION
    ):
        raise ProcessV2ChunkCacheIntegrityError(
            f"the semantic manifest does not describe the planned source: {manifest_path}"
        )
    expected_entries = int(manifest["entries"])
    expected_stream = str(manifest["record_stream_sha256"])

    stream_digest = hashlib.sha256()
    identity_digest = hashlib.sha256()
    chunks: list[dict[str, Any]] = []
    pending_rows: list[bytes] = []
    chunk_start = 0
    entries = 0
    reader = _FusedGzipReader(shard_path, opener=opener)

    def flush(entry_stop: int) -> None:
        payload = b"".join(pending_rows)
        record = {
            "chunk_index": len(chunks),
            "entry_start": chunk_start,
            "entry_stop": entry_stop,
            "row_count": entry_stop - chunk_start,
            "uncompressed_sha256": hashlib.sha256(payload).hexdigest(),
            "uncompressed_bytes": len(payload),
        }
        if chunk_sink is not None:
            record = {
                **record,
                **chunk_sink(len(chunks), chunk_start, entry_stop, payload),
            }
        chunks.append(record)
        pending_rows.clear()

    for line in reader.lines():
        if not line.strip():
            raise ProcessV2ChunkCacheIntegrityError(
                f"the semantic shard carries a blank row at entry {entries}: {shard_path}"
            )
        stream_digest.update(line)
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProcessV2ChunkCacheIntegrityError(
                f"semantic shard row {entries} is not JSON: {shard_path}"
            ) from error
        if not isinstance(row, Mapping) or not _REQUIRED_ROW_FIELDS.issubset(row):
            raise ProcessV2ChunkCacheIntegrityError(
                f"semantic shard row {entries} omits its identity fields: {shard_path}"
            )
        if (
            row["schema"] != TRACE_SCHEMA
            or row["schema_version"] != TRACE_SCHEMA_VERSION
            or row["data_lane"] != data_lane
            or row["split"] != split
            or row["process_identity_sha256"] != pinned_process_identity_sha256
            or row["action_codec_schema_version"] != action_codec_v4.SCHEMA_VERSION
            or row["action_codec_implementation_hash"] != manifest["action_codec_implementation_hash"]
        ):
            raise ProcessV2ChunkCacheIntegrityError(
                f"semantic shard row {entries} leaves its declared identity: {shard_path}"
            )
        identity_digest.update(
            canonical_bytes([entries, row["trace_id"], row["record_sha256"], row["path_length"]])
        )
        pending_rows.append(line)
        entries += 1
        if len(pending_rows) == records_per_chunk:
            flush(entries)
            chunk_start = entries
    if pending_rows or not chunks:
        # An empty source publishes exactly one empty chunk, so the address
        # space of a zero-entry lane/role cell is still explicit.
        flush(entries)

    shard_file_sha256 = reader.shard_file_sha256
    if shard_file_sha256 != expected_shard_sha256:
        raise ProcessV2ChunkCacheIntegrityError(
            f"the semantic shard physical hash disagrees: {shard_path}"
        )
    if entries != expected_entries:
        raise ProcessV2ChunkCacheIntegrityError(
            f"the semantic shard holds {entries} rows, its manifest declares {expected_entries}"
        )
    if stream_digest.hexdigest() != expected_stream:
        raise ProcessV2ChunkCacheIntegrityError(
            f"the semantic record stream hash disagrees: {shard_path}"
        )
    return FusedSourceScan(
        shard_file_sha256=shard_file_sha256,
        record_stream_sha256=stream_digest.hexdigest(),
        record_identity_sha256=identity_digest.hexdigest(),
        entries=entries,
        compressed_bytes=reader.compressed_bytes,
        decompressed_bytes=reader.decompressed_bytes,
        chunks=tuple(chunks),
    )


# ---- Plan ---------------------------------------------------------------------


def _source_task_body(entry: Mapping[str, Any], *, payload_root: str) -> dict[str, Any]:
    identity = str(entry["task_identity_sha256"])
    return {
        "v1_task_identity_sha256": identity,
        "v1_task_artifact_path": f"{payload_root}/tasks/{identity}",
        "data_lane": str(entry["data_lane"]),
        "split": str(entry["split"]),
        "semantic_shard_sha256": str(entry["semantic_shard_sha256"]),
        "semantic_manifest_sha256": str(entry["semantic_manifest_sha256"]),
        "entries": int(entry["counts"]["admitted"]),
    }


MIGRATION_IDENTITY_FIELDS: tuple[str, ...] = (
    "completion_sha256",
    "run_identity_sha256",
    "result_inventory_sha256",
    "process_identity_sha256",
    "builder_identity_sha256",
)


def migration_identity_from_completion(completion: Mapping[str, Any]) -> dict[str, str]:
    """The five completion fields the cache identity binds, and nothing else."""

    return {field: str(completion[field]) for field in MIGRATION_IDENTITY_FIELDS}


def cache_semantic_identity(
    *,
    migration_identity: Mapping[str, str],
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The identity of *what* is cached, invariant to how it was scheduled.

    Chunk size, task partitioning and container count are all absent by
    construction: only the completion, the pinned identities, the codec and the
    ordered per-source record identity appear.

    There are two forms of this identity and they are deliberately different
    values.  The **planned** form is built from sources that carry only what the
    exact completion already declares, because a plan cannot know a stream digest
    it has not measured yet.  The **measured** form adds each source's
    ``record_stream_sha256`` and ``record_identity_sha256``, which twenty
    independent fused passes produced.  The reducer publishes the measured form
    as ``cache_semantic_identity_sha256`` and keeps the planned one beside it as
    ``planned_semantic_identity_sha256``, after proving the measured sources
    reduce to the planned identity when the measurements are removed.
    """

    if set(migration_identity) != set(MIGRATION_IDENTITY_FIELDS):
        raise ProcessV2ChunkCacheError(
            f"the cache identity requires exactly {sorted(MIGRATION_IDENTITY_FIELDS)}"
        )
    ordered = sorted(
        (
            {
                "data_lane": str(source["data_lane"]),
                "split": str(source["split"]),
                "v1_task_identity_sha256": str(source["v1_task_identity_sha256"]),
                "semantic_shard_sha256": str(source["semantic_shard_sha256"]),
                "semantic_manifest_sha256": str(source["semantic_manifest_sha256"]),
                "entries": int(source["entries"]),
                **(
                    {
                        "record_stream_sha256": str(source["record_stream_sha256"]),
                        "record_identity_sha256": str(source["record_identity_sha256"]),
                    }
                    if "record_stream_sha256" in source
                    else {}
                ),
            }
            for source in sources
        ),
        key=lambda source: (source["data_lane"], source["split"]),
    )
    body = {
        "schema": f"{COMPLETION_SCHEMA}.semantic_identity",
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "migration_completion_sha256": migration_identity["completion_sha256"],
        "migration_run_identity_sha256": migration_identity["run_identity_sha256"],
        "migration_result_inventory_sha256": migration_identity["result_inventory_sha256"],
        "pinned_process_identity_sha256": migration_identity["process_identity_sha256"],
        "pinned_builder_identity_sha256": migration_identity["builder_identity_sha256"],
        "action_codec_schema_version": action_codec_v4.SCHEMA_VERSION,
        "action_codec_implementation_hash": action_codec_v4.codec_implementation_hash(),
        "trace_schema": TRACE_SCHEMA,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "sources": ordered,
        "entries": sum(int(source["entries"]) for source in ordered),
    }
    return {**body, "semantic_identity_sha256": canonical_sha256(body)}


def plan_process_v2_chunk_cache(
    binding: ProcessV2ExactCompletionBinding,
    *,
    source_revision: Mapping[str, Any],
    output_artifact_prefix: str = DEFAULT_OUTPUT_ARTIFACT_PREFIX,
    records_per_chunk: int = DEFAULT_RECORDS_PER_CHUNK,
) -> dict[str, Any]:
    """Freeze one task per source shard, addressed by the physical identity."""

    if type(records_per_chunk) is not int or not 1 <= records_per_chunk <= MAX_RECORDS_PER_CHUNK:
        raise ProcessV2ChunkCacheError(
            f"records_per_chunk must lie in [1, {MAX_RECORDS_PER_CHUNK}]"
        )
    if not isinstance(source_revision, Mapping) or "source_revision_sha256" not in source_revision:
        raise ProcessV2ChunkCacheError("the chunk-cache plan requires a bound source revision")
    payload_root = binding.payload_root_artifact_path
    sources = [
        _source_task_body(entry, payload_root=payload_root) for entry in binding.source_inventory
    ]
    semantic_identity = cache_semantic_identity(
        migration_identity=migration_identity_from_completion(binding.completion),
        sources=sources,
    )
    physical_body = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "semantic_identity_sha256": semantic_identity["semantic_identity_sha256"],
        "source_revision_sha256": str(source_revision["source_revision_sha256"]),
        "output_artifact_prefix": output_artifact_prefix,
        "records_per_chunk": records_per_chunk,
    }
    physical_identity_sha256 = canonical_sha256(physical_body)
    run_artifact_root = f"{output_artifact_prefix}/{physical_identity_sha256}"
    tasks: list[dict[str, Any]] = []
    for source in sources:
        task_identity_sha256 = canonical_sha256(
            {"cache_physical_identity_sha256": physical_identity_sha256, "source": source}
        )
        tasks.append(
            {
                **source,
                "task_identity_sha256": task_identity_sha256,
                "output_artifact_path": (
                    f"{run_artifact_root}/{SOURCE_DIRNAME}/{task_identity_sha256}"
                ),
            }
        )
    body: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": PLAN_STATUS,
        **authority_false_block(),
        "source_revision": dict(source_revision),
        "completion_binding": binding.as_payload(),
        "cache_semantic_identity": semantic_identity,
        "cache_semantic_identity_sha256": semantic_identity["semantic_identity_sha256"],
        "cache_physical_identity_sha256": physical_identity_sha256,
        "run_artifact_root": run_artifact_root,
        "output_artifact_prefix": output_artifact_prefix,
        "records_per_chunk": records_per_chunk,
        "expected_task_count": len(tasks),
        "expected_entry_count": int(binding.rebind_source_entries),
        "tasks": tasks,
        "task_inventory_sha256": canonical_sha256(tasks),
    }
    plan = {**body, "plan_sha256": canonical_sha256(body)}
    return validate_process_v2_chunk_cache_plan(plan)


def validate_process_v2_chunk_cache_plan(value: object) -> dict[str, Any]:
    """Validate plan identity, task addressing and census. Reads no artifact."""

    if not isinstance(value, Mapping):
        raise ProcessV2ChunkCacheError("the chunk-cache plan must be an object")
    plan = dict(value)
    require_authority_false(plan, label="the chunk-cache plan")
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != PLAN_STATUS
        or plan.get("plan_sha256") != _self_hash(plan, field="plan_sha256")
    ):
        raise ProcessV2ChunkCacheError("chunk-cache plan schema, authority or self-hash disagrees")
    semantic = plan.get("cache_semantic_identity")
    if (
        not isinstance(semantic, Mapping)
        or semantic.get("semantic_identity_sha256")
        != _self_hash(semantic, field="semantic_identity_sha256")
        or plan.get("cache_semantic_identity_sha256") != semantic.get("semantic_identity_sha256")
    ):
        raise ProcessV2ChunkCacheError("the chunk-cache semantic identity disagrees with itself")
    physical_identity_sha256 = canonical_sha256(
        {
            "schema": PLAN_SCHEMA,
            "schema_version": PLAN_SCHEMA_VERSION,
            "semantic_identity_sha256": semantic["semantic_identity_sha256"],
            "source_revision_sha256": str(plan["source_revision"]["source_revision_sha256"]),
            "output_artifact_prefix": plan["output_artifact_prefix"],
            "records_per_chunk": plan["records_per_chunk"],
        }
    )
    run_artifact_root = f"{plan['output_artifact_prefix']}/{physical_identity_sha256}"
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ProcessV2ChunkCacheError("the chunk-cache plan must contain at least one task")
    seen: set[str] = set()
    for index, task in enumerate(tasks):
        source = {
            key: task[key]
            for key in (
                "v1_task_identity_sha256",
                "v1_task_artifact_path",
                "data_lane",
                "split",
                "semantic_shard_sha256",
                "semantic_manifest_sha256",
                "entries",
            )
        }
        identity = canonical_sha256(
            {"cache_physical_identity_sha256": physical_identity_sha256, "source": source}
        )
        if identity in seen:
            raise ProcessV2ChunkCacheError("the chunk-cache plan repeats a source task")
        seen.add(identity)
        if (
            task.get("task_identity_sha256") != identity
            or task.get("output_artifact_path")
            != f"{run_artifact_root}/{SOURCE_DIRNAME}/{identity}"
        ):
            raise ProcessV2ChunkCacheError(f"chunk-cache task {index} identity or path disagrees")
    if (
        plan["cache_physical_identity_sha256"] != physical_identity_sha256
        or plan["run_artifact_root"] != run_artifact_root
        or plan["expected_task_count"] != len(tasks)
        or plan["expected_entry_count"] != sum(int(task["entries"]) for task in tasks)
        or plan["task_inventory_sha256"] != canonical_sha256(tasks)
    ):
        raise ProcessV2ChunkCacheError("chunk-cache plan addressing or census disagrees")
    return plan


def write_process_v2_chunk_cache_plan(plan: Mapping[str, Any], *, artifact_root: Path) -> Path:
    validated = validate_process_v2_chunk_cache_plan(plan)
    run_root = _mounted(validated["run_artifact_root"], artifact_root, "plan.run_artifact_root")
    run_root.mkdir(parents=True, exist_ok=True)
    return _publish_json_atomically(
        run_root / PLAN_FILENAME, validated, label="Process-V2 chunk-cache plan"
    )


def _mounted(artifact_path: str, artifact_root: Path, field: str) -> Path:
    try:
        return mounted_process_v2_artifact_path(
            artifact_path, artifact_root=artifact_root, field=field
        )
    except ProcessV2RebindError as error:
        raise ProcessV2ChunkCacheError(str(error)) from error


# ---- Source task execution ----------------------------------------------------


def _task_by_identity(plan: Mapping[str, Any], task_identity_sha256: str) -> dict[str, Any]:
    matches = [
        dict(task)
        for task in plan["tasks"]
        if task["task_identity_sha256"] == task_identity_sha256
    ]
    if len(matches) != 1:
        raise ProcessV2ChunkCacheError(
            f"the chunk-cache plan names {len(matches)} tasks {task_identity_sha256}"
        )
    return matches[0]


def _chunk_filename(chunk_index: int) -> str:
    return f"chunk-{chunk_index:06d}.jsonl.gz"


def _write_chunk(path: Path, payload: bytes) -> str:
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            compressed.write(payload)
    return _sha256_file(path)


def validate_process_v2_chunk_cache_source(
    output: Path,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one published source cache by its manifest and chunk hashes."""

    manifest_path = Path(output) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ProcessV2ChunkCacheIncomplete(f"the source chunk manifest is absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, Mapping):
        raise ProcessV2ChunkCacheError(f"the source chunk manifest must be an object: {manifest_path}")
    require_authority_false(manifest, label="the source chunk manifest")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != MANIFEST_STATUS
        or manifest.get("manifest_sha256") != _self_hash(manifest, field="manifest_sha256")
    ):
        raise ProcessV2ChunkCacheError(f"the source chunk manifest identity disagrees: {manifest_path}")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ProcessV2ChunkCacheError("a source chunk manifest must list at least one chunk")
    expected_names = {MANIFEST_FILENAME}
    entry_cursor = 0
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping) or set(chunk) != _CHUNK_RECORD_FIELDS:
            raise ProcessV2ChunkCacheError(f"source chunk {index} fields disagree")
        if (
            chunk["chunk_index"] != index
            or chunk["entry_start"] != entry_cursor
            or chunk["entry_stop"] != entry_cursor + int(chunk["row_count"])
        ):
            raise ProcessV2ChunkCacheError(
                f"source chunk {index} does not continue the entry address space"
            )
        entry_cursor = int(chunk["entry_stop"])
        name = str(chunk["chunk_filename"])
        if name != _chunk_filename(index):
            raise ProcessV2ChunkCacheError(f"source chunk {index} filename disagrees")
        expected_names.add(name)
        chunk_path = Path(output) / name
        if not chunk_path.is_file():
            raise ProcessV2ChunkCacheIncomplete(f"a published chunk is absent: {chunk_path}")
        if _sha256_file(chunk_path) != chunk["chunk_file_sha256"]:
            raise ProcessV2ChunkCacheError(f"a published chunk's physical hash disagrees: {chunk_path}")
    if entry_cursor != int(manifest["entries"]):
        raise ProcessV2ChunkCacheError("the source chunk manifest census does not cover its entries")
    observed = {path.name for path in Path(output).iterdir()}
    if observed != expected_names:
        raise ProcessV2ChunkCacheError(
            "a published source cache carries unexpected or missing objects; "
            f"missing={sorted(expected_names - observed)}, extras={sorted(observed - expected_names)}"
        )
    if expected_manifest is not None and dict(manifest) != dict(expected_manifest):
        raise ProcessV2ChunkCacheError(f"the published source chunk manifest disagrees: {manifest_path}")
    return dict(manifest)


def execute_process_v2_chunk_cache_task(
    plan: Mapping[str, Any],
    task_identity_sha256: str,
    *,
    artifact_root: Path,
    opener: Callable[..., BinaryIO] = open,
) -> dict[str, Any]:
    """Build or exactly reuse one source cache in a single fused pass."""

    validated = validate_process_v2_chunk_cache_plan(plan)
    task = _task_by_identity(validated, task_identity_sha256)
    output = _mounted(task["output_artifact_path"], artifact_root, "task.output_artifact_path")
    if output.exists():
        manifest = validate_process_v2_chunk_cache_source(output)
        if manifest["task_identity_sha256"] != task_identity_sha256:
            raise ProcessV2ChunkCacheError(f"immutable Process-V2 chunk-cache collision at {output}")
        return {
            "task_identity_sha256": task_identity_sha256,
            "output_artifact_path": task["output_artifact_path"],
            "manifest_sha256": manifest["manifest_sha256"],
            "entries": int(manifest["entries"]),
            "chunk_count": len(manifest["chunks"]),
            "reused": True,
        }

    semantic_dir = (
        _mounted(task["v1_task_artifact_path"], artifact_root, "task.v1_task_artifact_path")
        / SEMANTIC_ARTIFACT_DIRNAME
    )
    pinned = str(
        validated["completion_binding"]["pinned_process_identity_sha256"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.", suffix=".staging")
    )
    published = False
    try:

        def sink(chunk_index: int, _start: int, _stop: int, payload: bytes) -> dict[str, Any]:
            name = _chunk_filename(chunk_index)
            return {
                "chunk_filename": name,
                "chunk_file_sha256": _write_chunk(staging / name, payload),
            }

        scan = scan_source_shard_once(
            semantic_dir,
            data_lane=str(task["data_lane"]),
            split=str(task["split"]),
            expected_shard_sha256=str(task["semantic_shard_sha256"]),
            expected_manifest_sha256=str(task["semantic_manifest_sha256"]),
            pinned_process_identity_sha256=pinned,
            records_per_chunk=int(validated["records_per_chunk"]),
            chunk_sink=sink,
            opener=opener,
        )
        if scan.entries != int(task["entries"]):
            raise ProcessV2ChunkCacheIntegrityError(
                f"source shard {task['v1_task_identity_sha256']} holds {scan.entries} rows, the "
                f"exact completion declares {task['entries']}"
            )
        manifest_body: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": MANIFEST_STATUS,
            **authority_false_block(),
            "cache_semantic_identity_sha256": validated["cache_semantic_identity_sha256"],
            "cache_physical_identity_sha256": validated["cache_physical_identity_sha256"],
            "task_identity_sha256": task_identity_sha256,
            "v1_task_identity_sha256": task["v1_task_identity_sha256"],
            "v1_task_artifact_path": task["v1_task_artifact_path"],
            "data_lane": task["data_lane"],
            "split": task["split"],
            "pinned_process_identity_sha256": pinned,
            "semantic_shard_sha256": scan.shard_file_sha256,
            "semantic_manifest_sha256": task["semantic_manifest_sha256"],
            "record_stream_sha256": scan.record_stream_sha256,
            "record_identity_sha256": scan.record_identity_sha256,
            "entries": scan.entries,
            "records_per_chunk": int(validated["records_per_chunk"]),
            "encoding": CHUNK_ENCODING,
            "address_rule": CHUNK_ADDRESS_RULE,
            "chunk_count": len(scan.chunks),
            "chunks": [dict(chunk) for chunk in scan.chunks],
            "chunk_inventory_sha256": canonical_sha256([dict(c) for c in scan.chunks]),
        }
        manifest = {**manifest_body, "manifest_sha256": canonical_sha256(manifest_body)}
        _publish_json_atomically(
            staging / MANIFEST_FILENAME, manifest, label="Process-V2 source chunk manifest"
        )
        validate_process_v2_chunk_cache_source(staging, expected_manifest=manifest)
        if output.exists():
            validate_process_v2_chunk_cache_source(output, expected_manifest=manifest)
        else:
            try:
                os.rename(staging, output)
                published = True
            except OSError:
                if not output.exists():
                    raise
                validate_process_v2_chunk_cache_source(output, expected_manifest=manifest)
    finally:
        if not published and staging.exists():
            for entry in sorted(staging.iterdir()):
                entry.unlink()
            staging.rmdir()
    return {
        "task_identity_sha256": task_identity_sha256,
        "output_artifact_path": task["output_artifact_path"],
        "manifest_sha256": manifest["manifest_sha256"],
        "entries": scan.entries,
        "chunk_count": len(scan.chunks),
        "compressed_bytes": scan.compressed_bytes,
        "decompressed_bytes": scan.decompressed_bytes,
        "reused": False,
    }


def completed_process_v2_chunk_cache_task_ids(
    plan: Mapping[str, Any], *, artifact_root: Path
) -> set[str]:
    """Exact reusable source identities; any other object is a hard failure."""

    validated = validate_process_v2_chunk_cache_plan(plan)
    source_root = _mounted(
        f"{validated['run_artifact_root']}/{SOURCE_DIRNAME}", artifact_root, "plan source root"
    )
    expected = {str(task["task_identity_sha256"]) for task in validated["tasks"]}
    if not source_root.exists():
        return set()
    observed = {path.name for path in source_root.iterdir()}
    staging = {
        name
        for name in observed
        if any(name.startswith(f".{task_id}.") and name.endswith(".staging") for task_id in expected)
    }
    unexpected = (observed - staging) - expected
    if unexpected:
        raise ProcessV2ChunkCacheError(
            f"the chunk-cache source namespace contains unexpected objects: {sorted(unexpected)}"
        )
    complete: set[str] = set()
    for task in validated["tasks"]:
        identity = str(task["task_identity_sha256"])
        if identity not in observed:
            continue
        manifest = validate_process_v2_chunk_cache_source(source_root / identity)
        if (
            manifest["task_identity_sha256"] != identity
            or manifest["v1_task_identity_sha256"] != task["v1_task_identity_sha256"]
            or manifest["semantic_shard_sha256"] != task["semantic_shard_sha256"]
            or int(manifest["entries"]) != int(task["entries"])
            or manifest["cache_semantic_identity_sha256"]
            != validated["cache_semantic_identity_sha256"]
        ):
            raise ProcessV2ChunkCacheError(
                f"a published source cache does not belong to this plan: {identity}"
            )
        complete.add(identity)
    return complete


# ---- Deterministic global reduction -------------------------------------------


def reduce_process_v2_chunk_cache(
    plan: Mapping[str, Any], *, artifact_root: Path
) -> dict[str, Any]:
    """Publish the completion after proving every planned source is present."""

    validated = validate_process_v2_chunk_cache_plan(plan)
    run_root = _mounted(validated["run_artifact_root"], artifact_root, "plan.run_artifact_root")
    published_plan = run_root / PLAN_FILENAME
    if (
        not published_plan.is_file()
        or published_plan.read_bytes() != canonical_bytes(validated) + b"\n"
    ):
        raise ProcessV2ChunkCacheError(
            "the chunk-cache reduction requires the exact published plan bytes"
        )
    complete = completed_process_v2_chunk_cache_task_ids(validated, artifact_root=artifact_root)
    expected = {str(task["task_identity_sha256"]) for task in validated["tasks"]}
    missing = expected - complete
    if missing:
        raise ProcessV2ChunkCacheIncomplete(
            f"the chunk cache is missing {len(missing)} of {len(expected)} planned source caches"
        )
    source_root = run_root / SOURCE_DIRNAME
    observed_sources: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    entries = 0
    chunk_count = 0
    for task in validated["tasks"]:
        identity = str(task["task_identity_sha256"])
        manifest = validate_process_v2_chunk_cache_source(source_root / identity)
        observed_sources.append(
            {
                "data_lane": manifest["data_lane"],
                "split": manifest["split"],
                "v1_task_identity_sha256": manifest["v1_task_identity_sha256"],
                "semantic_shard_sha256": manifest["semantic_shard_sha256"],
                "semantic_manifest_sha256": manifest["semantic_manifest_sha256"],
                "entries": int(manifest["entries"]),
                "record_stream_sha256": manifest["record_stream_sha256"],
                "record_identity_sha256": manifest["record_identity_sha256"],
            }
        )
        inventory.append(
            {
                "task_identity_sha256": identity,
                "data_lane": manifest["data_lane"],
                "split": manifest["split"],
                "v1_task_identity_sha256": manifest["v1_task_identity_sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
                "chunk_inventory_sha256": manifest["chunk_inventory_sha256"],
                "record_identity_sha256": manifest["record_identity_sha256"],
                "entries": int(manifest["entries"]),
                "chunk_count": int(manifest["chunk_count"]),
            }
        )
        entries += int(manifest["entries"])
        chunk_count += int(manifest["chunk_count"])
    if entries != int(validated["expected_entry_count"]):
        raise ProcessV2ChunkCacheError(
            f"the chunk cache holds {entries} records, the exact completion declares "
            f"{validated['expected_entry_count']}"
        )
    # The observed identity is recomputed from what the workers measured, and
    # must equal the identity the plan froze from the completion.
    planned = validated["cache_semantic_identity"]
    observed_identity = cache_semantic_identity(
        migration_identity={
            "completion_sha256": planned["migration_completion_sha256"],
            "run_identity_sha256": planned["migration_run_identity_sha256"],
            "result_inventory_sha256": planned["migration_result_inventory_sha256"],
            "process_identity_sha256": planned["pinned_process_identity_sha256"],
            "builder_identity_sha256": planned["pinned_builder_identity_sha256"],
        },
        sources=observed_sources,
    )
    # The planned identity is derived from the exact completion; the observed
    # one is derived from what twenty independent fused passes measured. They
    # differ only in the per-source stream and record digests, which the plan
    # cannot know, so equality here is the cache's content proof.
    planned_without_measurements = canonical_sha256(
        {
            **{key: item for key, item in planned.items() if key != "semantic_identity_sha256"},
            "sources": [
                {
                    key: item
                    for key, item in source.items()
                    if key not in {"record_stream_sha256", "record_identity_sha256"}
                }
                for source in observed_identity["sources"]
            ],
        }
    )
    if planned_without_measurements != planned["semantic_identity_sha256"]:
        raise ProcessV2ChunkCacheError(
            "the measured chunk-cache source identity does not reduce to the planned one"
        )
    body: dict[str, Any] = {
        "schema": COMPLETION_SCHEMA,
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": COMPLETION_STATUS,
        **authority_false_block(),
        "cache_semantic_identity": observed_identity,
        "cache_semantic_identity_sha256": observed_identity["semantic_identity_sha256"],
        "planned_semantic_identity_sha256": validated["cache_semantic_identity_sha256"],
        "cache_physical_identity_sha256": validated["cache_physical_identity_sha256"],
        "plan_sha256": validated["plan_sha256"],
        "run_artifact_root": validated["run_artifact_root"],
        "records_per_chunk": int(validated["records_per_chunk"]),
        "source_count": len(inventory),
        "entries": entries,
        "chunk_count": chunk_count,
        "source_inventory": inventory,
        "source_inventory_sha256": canonical_sha256(inventory),
    }
    completion = {**body, "completion_sha256": canonical_sha256(body)}
    _publish_json_atomically(
        run_root / COMPLETION_FILENAME, completion, label="Process-V2 chunk-cache completion"
    )
    return completion


def load_process_v2_chunk_cache_completion(
    plan: Mapping[str, Any], *, artifact_root: Path
) -> dict[str, Any]:
    validated = validate_process_v2_chunk_cache_plan(plan)
    run_root = _mounted(validated["run_artifact_root"], artifact_root, "plan.run_artifact_root")
    path = run_root / COMPLETION_FILENAME
    if not path.is_file():
        raise ProcessV2ChunkCacheIncomplete(f"the chunk-cache completion is absent: {path}")
    completion = json.loads(path.read_bytes())
    require_authority_false(completion, label="the chunk-cache completion")
    if (
        completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("schema_version") != COMPLETION_SCHEMA_VERSION
        or completion.get("status") != COMPLETION_STATUS
        or completion.get("completion_sha256") != _self_hash(completion, field="completion_sha256")
        or completion.get("plan_sha256") != validated["plan_sha256"]
    ):
        raise ProcessV2ChunkCacheError(f"the chunk-cache completion is malformed or stale: {path}")
    return completion


# ---- Reading the cache --------------------------------------------------------


def read_process_v2_chunk_records(
    source_output: Path,
    *,
    chunk_index: int,
    expected_chunk_file_sha256: str | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(entry_index, record)`` for one chunk, verified by its own hash.

    A downstream worker reads complete chunk files through this function and
    never reopens the original packed gzip shard.
    """

    manifest = validate_process_v2_chunk_cache_source(Path(source_output))
    chunks = manifest["chunks"]
    if not 0 <= chunk_index < len(chunks):
        raise ProcessV2ChunkCacheError(
            f"chunk {chunk_index} is outside the {len(chunks)}-chunk source cache"
        )
    chunk = chunks[chunk_index]
    if (
        expected_chunk_file_sha256 is not None
        and chunk["chunk_file_sha256"] != expected_chunk_file_sha256
    ):
        raise ProcessV2ChunkCacheError(f"chunk {chunk_index} does not carry the expected hash")
    path = Path(source_output) / str(chunk["chunk_filename"])
    entry_index = int(chunk["entry_start"])
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            digest.update(line)
            yield entry_index, json.loads(line)
            entry_index += 1
    if entry_index != int(chunk["entry_stop"]):
        raise ProcessV2ChunkCacheError(f"chunk {chunk_index} does not hold its declared rows")
    if digest.hexdigest() != chunk["uncompressed_sha256"]:
        raise ProcessV2ChunkCacheError(f"chunk {chunk_index} record bytes disagree with its hash")


def iter_process_v2_chunk_cache_records(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
    task_identity_sha256: str,
) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield every ``(entry_index, record)`` of one cached source, in order."""

    validated = validate_process_v2_chunk_cache_plan(plan)
    task = _task_by_identity(validated, task_identity_sha256)
    output = _mounted(task["output_artifact_path"], artifact_root, "task.output_artifact_path")
    manifest = validate_process_v2_chunk_cache_source(output)
    for chunk in manifest["chunks"]:
        yield from read_process_v2_chunk_records(output, chunk_index=int(chunk["chunk_index"]))


# ---- Refusal reporting ---------------------------------------------------------


def write_chunk_cache_refusal_report(
    *,
    diagnostic_root: Path,
    plan: Mapping[str, Any] | None,
    stage: str,
    error: BaseException,
    detail: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a structured refusal in a separate content-addressed namespace.

    Nothing is published under the scientific run namespace on refusal; the
    report lives beside it so an operator can read why without a partial
    artifact acquiring authority by proximity.
    """

    body: dict[str, Any] = {
        "schema": REFUSAL_SCHEMA,
        "schema_version": REFUSAL_SCHEMA_VERSION,
        "status": REFUSAL_STATUS,
        **authority_false_block(),
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "cache_semantic_identity_sha256": (
            str(plan["cache_semantic_identity_sha256"]) if plan is not None else None
        ),
        "cache_physical_identity_sha256": (
            str(plan["cache_physical_identity_sha256"]) if plan is not None else None
        ),
        "detail": dict(detail or {}),
    }
    report = {**body, "refusal_sha256": canonical_sha256(body)}
    target = Path(diagnostic_root) / f"{report['refusal_sha256']}.json"
    return _publish_json_atomically(target, report, label="Process-V2 chunk-cache refusal")


__all__ = [
    "CHUNK_ADDRESS_RULE",
    "CHUNK_ENCODING",
    "COMPLETION_FILENAME",
    "COMPLETION_SCHEMA",
    "COMPLETION_SCHEMA_VERSION",
    "COMPLETION_STATUS",
    "DECOMPRESS_BLOCK_BYTES",
    "DEFAULT_CACHE_MAP_CONTAINERS",
    "DEFAULT_CHUNK_MAP_CONTAINERS",
    "DEFAULT_OUTPUT_ARTIFACT_PREFIX",
    "DEFAULT_RECORDS_PER_CHUNK",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_STATUS",
    "MAX_RECORDS_PER_CHUNK",
    "MIGRATION_IDENTITY_FIELDS",
    "PLAN_FILENAME",
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_VERSION",
    "PLAN_STATUS",
    "PLATFORM_MAX_MAP_CONTAINERS",
    "READ_BLOCK_BYTES",
    "REFUSAL_SCHEMA",
    "SOURCE_DIRNAME",
    "FusedSourceScan",
    "ProcessV2ChunkCacheError",
    "ProcessV2ChunkCacheIncomplete",
    "ProcessV2ChunkCacheIntegrityError",
    "ProcessV2ConcurrencyError",
    "cache_semantic_identity",
    "completed_process_v2_chunk_cache_task_ids",
    "execute_process_v2_chunk_cache_task",
    "iter_process_v2_chunk_cache_records",
    "load_process_v2_chunk_cache_completion",
    "migration_identity_from_completion",
    "plan_process_v2_chunk_cache",
    "plan_submission_waves",
    "read_process_v2_chunk_records",
    "reduce_process_v2_chunk_cache",
    "run_bounded_map",
    "scan_source_shard_once",
    "validate_map_container_bound",
    "validate_process_v2_chunk_cache_plan",
    "validate_process_v2_chunk_cache_source",
    "write_chunk_cache_refusal_report",
    "write_process_v2_chunk_cache_plan",
]
