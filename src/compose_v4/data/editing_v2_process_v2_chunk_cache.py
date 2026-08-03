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
identity plus the **narrow cache implementation revision**, the output prefix
and the chunk size.  It is the run address, so changing the chunk size relocates
the artifact without changing what the artifact means.  A reviewer comparing two
schedules compares the first and expects equality; comparing the second and
expecting equality would be a category error.

Two revisions, also deliberately separate
-----------------------------------------

``cache_implementation_sha256`` is the *narrow* revision.  It hashes exactly the
modules whose contents change what this cache stores, reads or validates, listed
in :data:`CACHE_IMPLEMENTATION_FILES`.  It is **owner-computed**: every boundary
re-derives it from the files that are actually present and requires equality, so
a caller-supplied revision is provenance at most and never authority.  It
deliberately carries no Git commit: a commit that does not touch a behaviour-
affecting module must not relocate the artifact.

"Reads" is load-bearing and was once only aspirational.  The list named the
modules that *write* a chunk and stopped there, so two trees differing only in
``rewrite/progress.py`` read the same committed cache, published the identical
narrow revision, both validated clean, and handed their callers different
decoded rows.  The list now covers the whole decode closure -- the modules that
run while a cached line becomes a ``SemanticPackedRowRead``, the modules that
define that row's constituent types, and the modules defining the payload
classes the frozen V4 persistence surface can construct.  ``codec_implementation
_hash()`` does not substitute for any of this: it hashes a *declared surface*,
so a codec whose declared rules are unchanged but whose decoding differs leaves
it fixed.

The launcher's *broad* image revision -- every file under ``src`` and ``configs``
plus the launcher and the plan driver -- is execution-environment provenance.  It
proves a container is running the tree it claims, and it is deliberately absent
from every scientific identity: hashing it into the cache address meant that
editing any file anywhere under ``src`` or ``configs`` relocated the whole cache.

Layering
--------

This module imports no other Process-V2 layer.  It owns the ``/artifacts`` path
primitives the rebind re-exports, so the rebind can consume the cache without an
import cycle; the earlier direction (a low-level cache importing the high-level
rebind) is what would have made that cycle.
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
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, BinaryIO

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
from compose_v4.data.packed_trace_store import (
    AddressedPackedTrace,
    PackedTraceAddress,
    PackedTraceProgress,
)
from compose_v4.data.semantic_packed_trace_store import (
    MANIFEST_FILENAME as SEMANTIC_MANIFEST_FILENAME,
)
from compose_v4.data.semantic_packed_trace_store import (
    SHARD_FILENAME as SEMANTIC_SHARD_FILENAME,
)
from compose_v4.data.semantic_packed_trace_store import (
    SemanticPackedRowRead,
)
from compose_v4.data.semantic_trace_migration_materializer import (
    SEMANTIC_ARTIFACT_DIRNAME,
)
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite.trace_shard import decode_state
from compose_v4.rewrite.trace_shard_v3 import (
    TRACE_SCHEMA,
    TRACE_SCHEMA_VERSION,
    SemanticTraceShardError,
    decode_semantic_trace_record,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never an import edge
    from compose_v4.data.editing_v2_process_v2_completion_binder import (
        ProcessV2ExactCompletionBinding,
    )

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
CACHE_IMPLEMENTATION_REVISION_SCHEMA = (
    "compose.data.editing_v2_process_v2_chunk_cache_implementation_revision"
)
CACHE_IMPLEMENTATION_REVISION_SCHEMA_VERSION = 1

# The narrow revision: exactly the modules whose *contents* decide what this
# cache stores, how it is addressed, how it is validated and how it decodes.
#
# The read half of that sentence is not a hand-list.  It is the *decode closure*
# of :func:`read_process_v2_chunk_target` -- the modules that actually execute
# while a cached line becomes a ``SemanticPackedRowRead``, the modules that
# define the types that row is built from, and the modules that define the
# payload classes the frozen V4 persistence surface can construct.
# ``tests/test_process_v2_chunk_cache_identity.py`` re-derives that closure by
# running the real decode under an execution tracer and requires this tuple to
# cover it, and proves per module that editing it moves
# ``cache_implementation_sha256``.  Neither check reads this tuple to build its
# expectation, so shortening the tuple fails them.
#
# `editing_process_v2_rebind.py` is deliberately absent -- the rebind consumes
# the cache and cannot change what the cache holds, so including it would
# relocate every cached byte for an unrelated edit, which is the conflation this
# revision exists to remove.  `provenance_overlay.py` is absent for the same
# reason and by measurement: `packed_trace_store` imports it only for the raw
# addressed-shard readers, which this cache never calls.
CACHE_IMPLEMENTATION_FILES: tuple[str, ...] = (
    "src/compose_v4/chem/molecular_graph.py",
    "src/compose_v4/chem/state.py",
    "src/compose_v4/data/editing_v2_process_v2_chunk_cache.py",
    "src/compose_v4/data/editing_v2_process_v2_completion_binder.py",
    "src/compose_v4/data/editing_v2_process_v2_schema.py",
    "src/compose_v4/data/packed_trace_store.py",
    "src/compose_v4/data/semantic_packed_trace_store.py",
    "src/compose_v4/data/semantic_trace_migration_materializer.py",
    "src/compose_v4/rewrite/action_codec.py",
    "src/compose_v4/rewrite/action_codec_v3.py",
    "src/compose_v4/rewrite/action_codec_v4.py",
    "src/compose_v4/rewrite/editing_v2_process_identity.py",
    "src/compose_v4/rewrite/operators.py",
    "src/compose_v4/rewrite/progress.py",
    "src/compose_v4/rewrite/trace.py",
    "src/compose_v4/rewrite/trace_shard.py",
    "src/compose_v4/rewrite/trace_shard_v3.py",
    "src/compose_v4/rewrite/tracelets.py",
)
# `.../src/compose_v4/data/<this file>` -> the repository root. The Modal image
# lays the tree out identically under `/root/compose`, so a container resolves
# its own root with no configuration.
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
_CACHE_IMPLEMENTATION_REVISION_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "schema_version",
        "implementation_files",
        "implementation_files_sha256",
        "cache_implementation_sha256",
    }
)

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
# The exact key set a published source chunk manifest carries. An exact set is
# what makes a *resealed* manifest fail: a body that has been re-hashed after an
# edit is self-consistent by construction, so only re-deriving every declared
# value from the artifact itself can refuse it.
_SOURCE_MANIFEST_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "cache_implementation_sha256",
        "cache_semantic_identity_sha256",
        "cache_physical_identity_sha256",
        "task_identity_sha256",
        "v1_task_identity_sha256",
        "v1_task_artifact_path",
        "data_lane",
        "split",
        "pinned_process_identity_sha256",
        "semantic_shard_sha256",
        "semantic_manifest_sha256",
        "record_stream_sha256",
        "record_identity_sha256",
        "entries",
        "records_per_chunk",
        "encoding",
        "address_rule",
        "chunk_count",
        "chunks",
        "chunk_inventory_sha256",
        "manifest_sha256",
    }
)
_COMPLETION_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        *authority_false_block(),
        "cache_implementation_sha256",
        "cache_semantic_identity",
        "cache_semantic_identity_sha256",
        "planned_semantic_identity_sha256",
        "cache_physical_identity_sha256",
        "plan_sha256",
        "run_artifact_root",
        "records_per_chunk",
        "source_count",
        "entries",
        "chunk_count",
        "source_inventory",
        "source_inventory_sha256",
        "completion_sha256",
    }
)
_COMPLETION_SOURCE_FIELDS: frozenset[str] = frozenset(
    {
        "task_identity_sha256",
        "data_lane",
        "split",
        "v1_task_identity_sha256",
        "manifest_sha256",
        "chunk_inventory_sha256",
        "record_identity_sha256",
        "entries",
        "chunk_count",
    }
)


# ---- `/artifacts` path primitives ----------------------------------------------
#
# Owned here rather than in the rebind, which re-exports them. The rebind now
# consumes this module, so the previous direction was an import cycle waiting to
# happen; and a *reader* of these artifacts must resolve them through the same
# normalization and containment check the writer used, not a second weaker one.


class ProcessV2ArtifactPathError(RuntimeError):
    """An ``/artifacts`` path is not normalized, or escapes its mounted root."""


def require_process_v2_artifact_path(value: object, *, field: str) -> str:
    """Accept only a normalized absolute path strictly below ``/artifacts``."""

    raw = value if isinstance(value, str) else ""
    path = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or not path.is_absolute()
        or len(path.parts) < 3
        or path.parts[1] != "artifacts"
        or ".." in path.parts
        or str(path) != raw
        or raw.endswith("/")
    ):
        raise ProcessV2ArtifactPathError(f"{field} must be a normalized path below /artifacts")
    return raw


def mount_process_v2_artifact_path(
    artifact_path: str, *, artifact_root: Path, field: str
) -> Path:
    """Resolve one ``/artifacts`` path under a mounted root, refusing escapes."""

    normalized = require_process_v2_artifact_path(artifact_path, field=field)
    root = Path(artifact_root).resolve()
    resolved = (root / PurePosixPath(normalized).relative_to("/artifacts")).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ProcessV2ArtifactPathError(f"{field} resolves outside the artifact root") from error
    return resolved


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


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProcessV2ChunkCacheError(f"{field} must be a lowercase SHA-256")
    return value


def _require_nonempty_str(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProcessV2ChunkCacheError(f"{field} must be a nonempty string")
    return value


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ProcessV2ChunkCacheError(f"{field} must be a nonnegative integer")
    return value


# ---- The narrow cache implementation revision ---------------------------------


def build_cache_implementation_revision(*, repo_root: Path | None = None) -> dict[str, Any]:
    """Hash exactly the modules that decide what this cache stores and reads.

    No Git object appears here.  The Git commit is launch discipline and belongs
    to the launcher's broad image revision; folding it in would relocate every
    cached byte whenever an unrelated file was committed, which is the same
    conflation defect one layer up.
    """

    root = Path(repo_root) if repo_root is not None else DEFAULT_REPO_ROOT
    implementation_files: dict[str, str] = {}
    for relative in CACHE_IMPLEMENTATION_FILES:
        source = root / relative
        if not source.is_file():
            raise ProcessV2ChunkCacheError(
                f"a chunk-cache implementation source is absent: {source}"
            )
        implementation_files[relative] = _sha256_file(source)
    body: dict[str, Any] = {
        "schema": CACHE_IMPLEMENTATION_REVISION_SCHEMA,
        "schema_version": CACHE_IMPLEMENTATION_REVISION_SCHEMA_VERSION,
        "implementation_files": implementation_files,
        "implementation_files_sha256": canonical_sha256(implementation_files),
    }
    return {**body, "cache_implementation_sha256": canonical_sha256(body)}


def validate_cache_implementation_revision(
    value: object, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Recompute the revision from the live files and require exact equality.

    A caller-supplied revision is provenance, never authority: this is the only
    function that decides what the narrow revision *is*, and every plan, worker,
    manifest, completion, reduction and loader boundary routes through it.
    """

    if not isinstance(value, Mapping):
        raise ProcessV2ChunkCacheError("the cache implementation revision must be an object")
    payload = dict(value)
    if set(payload) != _CACHE_IMPLEMENTATION_REVISION_FIELDS:
        raise ProcessV2ChunkCacheError(
            "cache implementation revision fields disagree; missing="
            f"{sorted(_CACHE_IMPLEMENTATION_REVISION_FIELDS - set(payload))}, "
            f"extras={sorted(set(payload) - _CACHE_IMPLEMENTATION_REVISION_FIELDS)}"
        )
    live = build_cache_implementation_revision(repo_root=repo_root)
    if payload != live:
        raise ProcessV2ChunkCacheError(
            "the cache implementation revision disagrees with the modules that are "
            f"actually present: {payload.get('cache_implementation_sha256')} declared, "
            f"{live['cache_implementation_sha256']} live"
        )
    return payload


def cache_implementation_sha256(*, repo_root: Path | None = None) -> str:
    """The live narrow revision digest, recomputed from the files each time."""

    return str(build_cache_implementation_revision(repo_root=repo_root)[
        "cache_implementation_sha256"
    ])


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
    repo_root: Path | None = None,
    cache_implementation_revision: Mapping[str, Any] | None = None,
    output_artifact_prefix: str = DEFAULT_OUTPUT_ARTIFACT_PREFIX,
    records_per_chunk: int = DEFAULT_RECORDS_PER_CHUNK,
) -> dict[str, Any]:
    """Freeze one task per source shard, addressed by the physical identity.

    ``cache_implementation_revision`` is optional and never trusted: supplied or
    not, the revision the plan carries is the one recomputed from the modules
    that are actually present.  Supplying one only adds the requirement that the
    caller's belief matches.
    """

    if type(records_per_chunk) is not int or not 1 <= records_per_chunk <= MAX_RECORDS_PER_CHUNK:
        raise ProcessV2ChunkCacheError(
            f"records_per_chunk must lie in [1, {MAX_RECORDS_PER_CHUNK}]"
        )
    if cache_implementation_revision is None:
        revision = build_cache_implementation_revision(repo_root=repo_root)
    else:
        revision = validate_cache_implementation_revision(
            cache_implementation_revision, repo_root=repo_root
        )
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
        "cache_implementation_sha256": str(revision["cache_implementation_sha256"]),
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
        "cache_implementation_revision": dict(revision),
        "cache_implementation_sha256": str(revision["cache_implementation_sha256"]),
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
    return validate_process_v2_chunk_cache_plan(plan, repo_root=repo_root)


def validate_process_v2_chunk_cache_plan(
    value: object, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Validate plan identity, revision, task addressing and census.

    Reads no volume artifact.  It does re-derive the narrow implementation
    revision from the repository, which is the point: a plan whose nested
    revision was edited and re-sealed is self-consistent and must still fail.
    """

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
    revision = validate_cache_implementation_revision(
        plan.get("cache_implementation_revision"), repo_root=repo_root
    )
    if plan.get("cache_implementation_sha256") != revision["cache_implementation_sha256"]:
        raise ProcessV2ChunkCacheError(
            "the chunk-cache plan implementation digest does not address its own revision"
        )
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
            "cache_implementation_sha256": revision["cache_implementation_sha256"],
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


def write_process_v2_chunk_cache_plan(
    plan: Mapping[str, Any], *, artifact_root: Path, repo_root: Path | None = None
) -> Path:
    validated = validate_process_v2_chunk_cache_plan(plan, repo_root=repo_root)
    run_root = _mounted(validated["run_artifact_root"], artifact_root, "plan.run_artifact_root")
    run_root.mkdir(parents=True, exist_ok=True)
    return _publish_json_atomically(
        run_root / PLAN_FILENAME, validated, label="Process-V2 chunk-cache plan"
    )


def _mounted(artifact_path: str, artifact_root: Path, field: str) -> Path:
    try:
        return mount_process_v2_artifact_path(
            artifact_path, artifact_root=artifact_root, field=field
        )
    except ProcessV2ArtifactPathError as error:
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


def chunk_object_filename(chunk_index: int) -> str:
    """The one chunk-naming rule, so no consumer rebuilds a second one."""

    return f"chunk-{chunk_index:06d}.jsonl.gz"


_chunk_filename = chunk_object_filename


def _write_chunk(path: Path, payload: bytes) -> str:
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            compressed.write(payload)
    return _sha256_file(path)


def validate_process_v2_chunk_cache_manifest(
    value: object, *, repo_root: Path | None = None, label: str = "the source chunk manifest"
) -> dict[str, Any]:
    """Validate one source chunk manifest **without opening any chunk file**.

    Every declared value is re-derived: the exact field set, the schema, the
    authority block, the self-hash, the live implementation revision, the chunk
    address space, the chunking rule, the chunk count and the chunk inventory
    hash.  That last pair is what refuses a *resealed* manifest -- one whose
    ``chunk_count`` was edited and whose ``manifest_sha256`` was recomputed is
    internally consistent, and only re-deriving it from ``chunks`` can say no.

    This is the structural half of the contract.  It is deliberately separate
    from :func:`validate_process_v2_chunk_cache_source`, which additionally
    hashes every published chunk: whole-generation validation belongs at a
    publication or opening boundary, never inside a worker reading one chunk.
    """

    if not isinstance(value, Mapping):
        raise ProcessV2ChunkCacheError(f"{label} must be an object")
    manifest = dict(value)
    if set(manifest) != _SOURCE_MANIFEST_FIELDS:
        raise ProcessV2ChunkCacheError(
            f"{label} fields disagree; missing="
            f"{sorted(_SOURCE_MANIFEST_FIELDS - set(manifest))}, "
            f"extras={sorted(set(manifest) - _SOURCE_MANIFEST_FIELDS)}"
        )
    require_authority_false(manifest, label=label)
    if (
        manifest["schema"] != MANIFEST_SCHEMA
        or manifest["schema_version"] != MANIFEST_SCHEMA_VERSION
        or manifest["status"] != MANIFEST_STATUS
        or manifest["manifest_sha256"] != _self_hash(manifest, field="manifest_sha256")
    ):
        raise ProcessV2ChunkCacheError(f"{label} identity disagrees")
    if manifest["cache_implementation_sha256"] != cache_implementation_sha256(repo_root=repo_root):
        raise ProcessV2ChunkCacheError(
            f"{label} was built by a different chunk-cache implementation revision"
        )
    for field in (
        "cache_semantic_identity_sha256",
        "cache_physical_identity_sha256",
        "task_identity_sha256",
        "v1_task_identity_sha256",
        "pinned_process_identity_sha256",
        "semantic_shard_sha256",
        "semantic_manifest_sha256",
        "record_stream_sha256",
        "record_identity_sha256",
        "chunk_inventory_sha256",
    ):
        _require_sha256(manifest[field], field=f"{label}.{field}")
    for field in ("data_lane", "split"):
        _require_nonempty_str(manifest[field], field=f"{label}.{field}")
    require_process_v2_artifact_path(
        manifest["v1_task_artifact_path"], field=f"{label}.v1_task_artifact_path"
    )
    if manifest["encoding"] != CHUNK_ENCODING or manifest["address_rule"] != CHUNK_ADDRESS_RULE:
        raise ProcessV2ChunkCacheError(f"{label} encoding or address rule disagrees")
    records_per_chunk = manifest["records_per_chunk"]
    if (
        type(records_per_chunk) is not int
        or not 1 <= records_per_chunk <= MAX_RECORDS_PER_CHUNK
    ):
        raise ProcessV2ChunkCacheError(f"{label} records_per_chunk is outside its bound")
    chunks = manifest["chunks"]
    if not isinstance(chunks, list) or not chunks:
        raise ProcessV2ChunkCacheError(f"{label} must list at least one chunk")
    entry_cursor = 0
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping) or set(chunk) != _CHUNK_RECORD_FIELDS:
            raise ProcessV2ChunkCacheError(f"source chunk {index} fields disagree")
        row_count = _require_nonnegative_int(
            chunk["row_count"], field=f"source chunk {index} row_count"
        )
        _require_nonnegative_int(
            chunk["uncompressed_bytes"], field=f"source chunk {index} uncompressed_bytes"
        )
        _require_sha256(chunk["uncompressed_sha256"], field=f"source chunk {index} hash")
        _require_sha256(chunk["chunk_file_sha256"], field=f"source chunk {index} file hash")
        if (
            chunk["chunk_index"] != index
            or chunk["entry_start"] != entry_cursor
            or chunk["entry_stop"] != entry_cursor + row_count
        ):
            raise ProcessV2ChunkCacheError(
                f"source chunk {index} does not continue the entry address space"
            )
        # The chunking rule itself, not merely a contiguous cover: every chunk
        # but the last holds exactly `records_per_chunk` rows. Without this a
        # resealed manifest could declare any partition it liked and still read
        # as a valid address space.
        if index < len(chunks) - 1 and row_count != records_per_chunk:
            raise ProcessV2ChunkCacheError(
                f"source chunk {index} holds {row_count} rows, not the declared "
                f"{records_per_chunk} the chunking rule requires of every non-final chunk"
            )
        if index == len(chunks) - 1 and not 0 <= row_count <= records_per_chunk:
            raise ProcessV2ChunkCacheError(f"the final source chunk holds {row_count} rows")
        entry_cursor = int(chunk["entry_stop"])
        if str(chunk["chunk_filename"]) != _chunk_filename(index):
            raise ProcessV2ChunkCacheError(f"source chunk {index} filename disagrees")
    if entry_cursor != _require_nonnegative_int(manifest["entries"], field=f"{label}.entries"):
        raise ProcessV2ChunkCacheError(f"{label} census does not cover its entries")
    if manifest["chunk_count"] != len(chunks):
        raise ProcessV2ChunkCacheError(
            f"{label} declares {manifest['chunk_count']} chunks and lists {len(chunks)}"
        )
    if manifest["chunk_inventory_sha256"] != canonical_sha256([dict(c) for c in chunks]):
        raise ProcessV2ChunkCacheError(f"{label} inventory hash does not address its own chunks")
    return manifest


def validate_process_v2_chunk_cache_source(
    output: Path,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate one published source cache **whole**: manifest and every chunk.

    This is the publication and opening boundary.  It hashes every chunk file in
    the source, which is exactly what a target-chunk read must not do.
    """

    manifest_path = Path(output) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ProcessV2ChunkCacheIncomplete(f"the source chunk manifest is absent: {manifest_path}")
    try:
        raw_manifest = json.loads(manifest_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessV2ChunkCacheError(
            f"the source chunk manifest is not JSON: {manifest_path}"
        ) from error
    manifest = validate_process_v2_chunk_cache_manifest(
        raw_manifest, repo_root=repo_root, label=f"the source chunk manifest {manifest_path}"
    )
    expected_names = {MANIFEST_FILENAME}
    for chunk in manifest["chunks"]:
        name = str(chunk["chunk_filename"])
        expected_names.add(name)
        chunk_path = Path(output) / name
        if not chunk_path.is_file():
            raise ProcessV2ChunkCacheIncomplete(f"a published chunk is absent: {chunk_path}")
        if _sha256_file(chunk_path) != chunk["chunk_file_sha256"]:
            raise ProcessV2ChunkCacheError(f"a published chunk's physical hash disagrees: {chunk_path}")
    observed = {path.name for path in Path(output).iterdir()}
    if observed != expected_names:
        raise ProcessV2ChunkCacheError(
            "a published source cache carries unexpected or missing objects; "
            f"missing={sorted(expected_names - observed)}, extras={sorted(observed - expected_names)}"
        )
    if expected_manifest is not None and manifest != dict(expected_manifest):
        raise ProcessV2ChunkCacheError(f"the published source chunk manifest disagrees: {manifest_path}")
    return manifest


def execute_process_v2_chunk_cache_task(
    plan: Mapping[str, Any],
    task_identity_sha256: str,
    *,
    artifact_root: Path,
    repo_root: Path | None = None,
    opener: Callable[..., BinaryIO] = open,
) -> dict[str, Any]:
    """Build or exactly reuse one source cache in a single fused pass."""

    validated = validate_process_v2_chunk_cache_plan(plan, repo_root=repo_root)
    task = _task_by_identity(validated, task_identity_sha256)
    output = _mounted(task["output_artifact_path"], artifact_root, "task.output_artifact_path")
    if output.exists():
        manifest = validate_process_v2_chunk_cache_source(output, repo_root=repo_root)
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
            "cache_implementation_sha256": validated["cache_implementation_sha256"],
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
        validate_process_v2_chunk_cache_source(
            staging, expected_manifest=manifest, repo_root=repo_root
        )
        if output.exists():
            validate_process_v2_chunk_cache_source(
                output, expected_manifest=manifest, repo_root=repo_root
            )
        else:
            try:
                os.rename(staging, output)
                published = True
            except OSError:
                if not output.exists():
                    raise
                validate_process_v2_chunk_cache_source(
                    output, expected_manifest=manifest, repo_root=repo_root
                )
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
    plan: Mapping[str, Any], *, artifact_root: Path, repo_root: Path | None = None
) -> set[str]:
    """Exact reusable source identities; any other object is a hard failure."""

    validated = validate_process_v2_chunk_cache_plan(plan, repo_root=repo_root)
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
        manifest = validate_process_v2_chunk_cache_source(
            source_root / identity, repo_root=repo_root
        )
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


def reduction_order(tasks: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Order tasks by ``(data_lane, split)``, and prove that key is a total order.

    Stated rather than assumed.  The plan happens to be built in a deterministic
    order today, but "deterministic" and "ordered by the key the artifact is
    read by" are different properties, and only the second survives a change to
    how the plan enumerates its sources.  A repeated key would make the sort
    ambiguous, so it is a refusal rather than a tie broken silently.
    """

    ordered = sorted(
        (dict(task) for task in tasks),
        key=lambda task: (str(task["data_lane"]), str(task["split"])),
    )
    keys = [(task["data_lane"], task["split"]) for task in ordered]
    if len(set(keys)) != len(keys):
        raise ProcessV2ChunkCacheError(
            "the chunk-cache reduction order is ambiguous: two sources share one "
            "(data_lane, split) cell"
        )
    return tuple(ordered)


def reduce_process_v2_chunk_cache(
    plan: Mapping[str, Any], *, artifact_root: Path, repo_root: Path | None = None
) -> dict[str, Any]:
    """Publish the completion after proving every planned source is present.

    The completion is written **last**, after the whole generation validates, so
    an interrupted or refused reduction leaves no committed generation at all
    and :func:`open_process_v2_chunk_cache` refuses the run root.
    """

    validated = validate_process_v2_chunk_cache_plan(plan, repo_root=repo_root)
    run_root = _mounted(validated["run_artifact_root"], artifact_root, "plan.run_artifact_root")
    published_plan = run_root / PLAN_FILENAME
    if (
        not published_plan.is_file()
        or published_plan.read_bytes() != canonical_bytes(validated) + b"\n"
    ):
        raise ProcessV2ChunkCacheError(
            "the chunk-cache reduction requires the exact published plan bytes"
        )
    complete = completed_process_v2_chunk_cache_task_ids(
        validated, artifact_root=artifact_root, repo_root=repo_root
    )
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
    for task in reduction_order(validated["tasks"]):
        identity = str(task["task_identity_sha256"])
        manifest = validate_process_v2_chunk_cache_source(
            source_root / identity, repo_root=repo_root
        )
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
        "cache_implementation_sha256": validated["cache_implementation_sha256"],
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
    validate_process_v2_chunk_cache_completion(completion, repo_root=repo_root)
    _publish_json_atomically(
        run_root / COMPLETION_FILENAME, completion, label="Process-V2 chunk-cache completion"
    )
    return completion


def validate_process_v2_chunk_cache_completion(
    value: object, *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Validate the completion's exact field set, every count and every hash."""

    if not isinstance(value, Mapping):
        raise ProcessV2ChunkCacheError("the chunk-cache completion must be an object")
    completion = dict(value)
    if set(completion) != _COMPLETION_FIELDS:
        raise ProcessV2ChunkCacheError(
            "chunk-cache completion fields disagree; missing="
            f"{sorted(_COMPLETION_FIELDS - set(completion))}, "
            f"extras={sorted(set(completion) - _COMPLETION_FIELDS)}"
        )
    require_authority_false(completion, label="the chunk-cache completion")
    if (
        completion["schema"] != COMPLETION_SCHEMA
        or completion["schema_version"] != COMPLETION_SCHEMA_VERSION
        or completion["status"] != COMPLETION_STATUS
        or completion["completion_sha256"] != _self_hash(completion, field="completion_sha256")
    ):
        raise ProcessV2ChunkCacheError("the chunk-cache completion identity disagrees")
    if completion["cache_implementation_sha256"] != cache_implementation_sha256(
        repo_root=repo_root
    ):
        raise ProcessV2ChunkCacheError(
            "the chunk-cache completion was sealed by a different implementation revision"
        )
    for field in (
        "cache_semantic_identity_sha256",
        "planned_semantic_identity_sha256",
        "cache_physical_identity_sha256",
        "plan_sha256",
        "source_inventory_sha256",
    ):
        _require_sha256(completion[field], field=f"completion.{field}")
    require_process_v2_artifact_path(
        completion["run_artifact_root"], field="completion.run_artifact_root"
    )
    if not str(completion["run_artifact_root"]).endswith(
        f"/{completion['cache_physical_identity_sha256']}"
    ):
        raise ProcessV2ChunkCacheError(
            "the chunk-cache completion run root is not addressed by its physical identity"
        )
    semantic = completion["cache_semantic_identity"]
    if (
        not isinstance(semantic, Mapping)
        or semantic.get("semantic_identity_sha256")
        != _self_hash(semantic, field="semantic_identity_sha256")
        or completion["cache_semantic_identity_sha256"] != semantic["semantic_identity_sha256"]
    ):
        raise ProcessV2ChunkCacheError("the chunk-cache completion semantic identity disagrees")
    records_per_chunk = completion["records_per_chunk"]
    if (
        type(records_per_chunk) is not int
        or not 1 <= records_per_chunk <= MAX_RECORDS_PER_CHUNK
    ):
        raise ProcessV2ChunkCacheError("the chunk-cache completion chunk size is outside its bound")
    inventory = completion["source_inventory"]
    if not isinstance(inventory, list) or not inventory:
        raise ProcessV2ChunkCacheError("the chunk-cache completion must name at least one source")
    seen: set[str] = set()
    cells: set[tuple[str, str]] = set()
    entries = 0
    chunk_count = 0
    for index, source in enumerate(inventory):
        if not isinstance(source, Mapping) or set(source) != _COMPLETION_SOURCE_FIELDS:
            raise ProcessV2ChunkCacheError(f"completion source {index} fields disagree")
        for field in (
            "task_identity_sha256",
            "v1_task_identity_sha256",
            "manifest_sha256",
            "chunk_inventory_sha256",
            "record_identity_sha256",
        ):
            _require_sha256(source[field], field=f"completion source {index} {field}")
        cell = (
            _require_nonempty_str(source["data_lane"], field=f"completion source {index} lane"),
            _require_nonempty_str(source["split"], field=f"completion source {index} split"),
        )
        if source["task_identity_sha256"] in seen or cell in cells:
            raise ProcessV2ChunkCacheError(
                f"the chunk-cache completion repeats source {index}"
            )
        seen.add(str(source["task_identity_sha256"]))
        cells.add(cell)
        entries += _require_nonnegative_int(
            source["entries"], field=f"completion source {index} entries"
        )
        chunk_count += _require_nonnegative_int(
            source["chunk_count"], field=f"completion source {index} chunk_count"
        )
    if [
        (str(source["data_lane"]), str(source["split"])) for source in inventory
    ] != sorted(cells):
        raise ProcessV2ChunkCacheError(
            "the chunk-cache completion inventory is not in (data_lane, split) order"
        )
    if (
        completion["source_count"] != len(inventory)
        or completion["entries"] != entries
        or completion["chunk_count"] != chunk_count
        or completion["source_inventory_sha256"]
        != canonical_sha256([dict(source) for source in inventory])
    ):
        raise ProcessV2ChunkCacheError("the chunk-cache completion census disagrees")
    return completion


def load_process_v2_chunk_cache_completion(
    plan: Mapping[str, Any], *, artifact_root: Path, repo_root: Path | None = None
) -> dict[str, Any]:
    validated = validate_process_v2_chunk_cache_plan(plan, repo_root=repo_root)
    run_root = _mounted(validated["run_artifact_root"], artifact_root, "plan.run_artifact_root")
    path = run_root / COMPLETION_FILENAME
    if not path.is_file():
        raise ProcessV2ChunkCacheIncomplete(f"the chunk-cache completion is absent: {path}")
    try:
        raw = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessV2ChunkCacheError(f"the chunk-cache completion is not JSON: {path}") from error
    completion = validate_process_v2_chunk_cache_completion(raw, repo_root=repo_root)
    if (
        completion["plan_sha256"] != validated["plan_sha256"]
        or completion["cache_physical_identity_sha256"]
        != validated["cache_physical_identity_sha256"]
        or completion["planned_semantic_identity_sha256"]
        != validated["cache_semantic_identity_sha256"]
    ):
        raise ProcessV2ChunkCacheError(f"the chunk-cache completion is stale for this plan: {path}")
    return completion


# ---- Opening a committed generation -------------------------------------------


@dataclass(frozen=True)
class ProcessV2ChunkCacheGeneration:
    """One committed chunk-cache generation, opened and validated whole.

    Constructing this is the *opening* boundary: the committed completion must
    exist, and every published chunk of every source is hashed here, once, so a
    later target-chunk read never has to.
    """

    completion: Mapping[str, Any]
    run_artifact_root: str
    source_artifact_paths: tuple[str, ...]
    source_manifests: tuple[Mapping[str, Any], ...]
    chunk_bytes_verified: bool

    def targets(self) -> tuple[ProcessV2ChunkTarget, ...]:
        """Every chunk of the generation, in ``(data_lane, split, entry_start)``."""

        targets: list[ProcessV2ChunkTarget] = []
        for path, manifest in zip(self.source_artifact_paths, self.source_manifests, strict=True):
            targets.extend(chunk_targets_for_source(manifest, source_artifact_path=path))
        return tuple(
            sorted(
                targets,
                key=lambda target: (target.data_lane, target.split, target.entry_start),
            )
        )


def load_committed_process_v2_chunk_cache_completion(
    run_artifact_root: str, *, artifact_root: Path, repo_root: Path | None = None
) -> dict[str, Any]:
    """Read the committed completion of one run root, or refuse the generation.

    The completion is the committed marker: the reducer writes it last, after
    the whole generation validates.  An interrupted generation therefore has no
    completion and cannot be opened -- it is *absent*, never *partial*.
    """

    run_root = _mounted(run_artifact_root, artifact_root, "run_artifact_root")
    path = run_root / COMPLETION_FILENAME
    if not path.is_file():
        raise ProcessV2ChunkCacheIncomplete(
            f"the chunk-cache generation is not committed; no {COMPLETION_FILENAME} at {run_root}"
        )
    try:
        raw = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessV2ChunkCacheError(f"the chunk-cache completion is not JSON: {path}") from error
    completion = validate_process_v2_chunk_cache_completion(raw, repo_root=repo_root)
    if completion["run_artifact_root"] != run_artifact_root:
        raise ProcessV2ChunkCacheError(
            f"the chunk-cache completion at {path} names another run root"
        )
    return completion


def open_process_v2_chunk_cache(
    completion: Mapping[str, Any],
    *,
    artifact_root: Path,
    repo_root: Path | None = None,
    verify_chunk_bytes: bool = True,
) -> ProcessV2ChunkCacheGeneration:
    """Open one committed generation and validate it whole.

    ``verify_chunk_bytes`` defaults to true because this is the boundary that
    owes the whole-generation guarantee.  Setting it false checks every manifest
    and every declared identity but not the chunk payloads, which is the right
    trade only for a caller that will hash each target chunk itself anyway.
    """

    validated = validate_process_v2_chunk_cache_completion(completion, repo_root=repo_root)
    run_artifact_root = str(validated["run_artifact_root"])
    paths: list[str] = []
    manifests: list[Mapping[str, Any]] = []
    for source in validated["source_inventory"]:
        identity = str(source["task_identity_sha256"])
        artifact_path = f"{run_artifact_root}/{SOURCE_DIRNAME}/{identity}"
        output = _mounted(artifact_path, artifact_root, "chunk cache source path")
        if verify_chunk_bytes:
            manifest = validate_process_v2_chunk_cache_source(output, repo_root=repo_root)
        else:
            manifest_path = output / MANIFEST_FILENAME
            if not manifest_path.is_file():
                raise ProcessV2ChunkCacheIncomplete(
                    f"the source chunk manifest is absent: {manifest_path}"
                )
            manifest = validate_process_v2_chunk_cache_manifest(
                json.loads(manifest_path.read_bytes()), repo_root=repo_root
            )
        disagreements = [
            field
            for field, declared, observed in (
                ("task_identity_sha256", identity, manifest["task_identity_sha256"]),
                ("data_lane", source["data_lane"], manifest["data_lane"]),
                ("split", source["split"], manifest["split"]),
                (
                    "v1_task_identity_sha256",
                    source["v1_task_identity_sha256"],
                    manifest["v1_task_identity_sha256"],
                ),
                ("manifest_sha256", source["manifest_sha256"], manifest["manifest_sha256"]),
                (
                    "chunk_inventory_sha256",
                    source["chunk_inventory_sha256"],
                    manifest["chunk_inventory_sha256"],
                ),
                (
                    "record_identity_sha256",
                    source["record_identity_sha256"],
                    manifest["record_identity_sha256"],
                ),
                ("entries", int(source["entries"]), int(manifest["entries"])),
                ("chunk_count", int(source["chunk_count"]), int(manifest["chunk_count"])),
                # A worker seals the *planned* semantic identity, which is all a
                # worker can know; the reducer publishes the *measured* one it
                # proved reduces to it. Comparing the manifest against the
                # measured value would refuse every correct cache.
                (
                    "cache_semantic_identity_sha256",
                    validated["planned_semantic_identity_sha256"],
                    manifest["cache_semantic_identity_sha256"],
                ),
                (
                    "cache_physical_identity_sha256",
                    validated["cache_physical_identity_sha256"],
                    manifest["cache_physical_identity_sha256"],
                ),
                (
                    "records_per_chunk",
                    int(validated["records_per_chunk"]),
                    int(manifest["records_per_chunk"]),
                ),
            )
            if declared != observed
        ]
        if disagreements:
            raise ProcessV2ChunkCacheError(
                f"published source cache {identity} disagrees with the committed completion "
                f"on {sorted(disagreements)}"
            )
        paths.append(artifact_path)
        manifests.append(manifest)
    return ProcessV2ChunkCacheGeneration(
        completion=validated,
        run_artifact_root=run_artifact_root,
        source_artifact_paths=tuple(paths),
        source_manifests=tuple(manifests),
        chunk_bytes_verified=bool(verify_chunk_bytes),
    )


# ---- Reading exactly one target chunk -----------------------------------------


@dataclass(frozen=True)
class ProcessV2ChunkTarget:
    """The exact chunk one worker is assigned, bound before anything is read.

    Every field is a *declared expectation*.  The reader re-derives each one
    from the published manifest and the chunk's own bytes and refuses on any
    disagreement, so a task cannot silently read a different chunk, a different
    entry range, or a chunk of a different generation.
    """

    source_artifact_path: str
    cache_source_task_identity_sha256: str
    cache_source_manifest_sha256: str
    cache_semantic_identity_sha256: str
    cache_physical_identity_sha256: str
    v1_task_identity_sha256: str
    data_lane: str
    split: str
    semantic_shard_sha256: str
    pinned_process_identity_sha256: str
    chunk_index: int
    chunk_filename: str
    chunk_file_sha256: str
    chunk_uncompressed_sha256: str
    entry_start: int
    entry_stop: int
    row_count: int

    def as_payload(self) -> dict[str, Any]:
        """The deterministic descriptor a plan task records."""

        return {
            "cache_source_artifact_path": self.source_artifact_path,
            "cache_source_task_identity_sha256": self.cache_source_task_identity_sha256,
            "cache_source_manifest_sha256": self.cache_source_manifest_sha256,
            "cache_semantic_identity_sha256": self.cache_semantic_identity_sha256,
            "cache_physical_identity_sha256": self.cache_physical_identity_sha256,
            "chunk_index": self.chunk_index,
            "chunk_filename": self.chunk_filename,
            "chunk_file_sha256": self.chunk_file_sha256,
            "chunk_uncompressed_sha256": self.chunk_uncompressed_sha256,
            "chunk_row_count": self.row_count,
        }


def chunk_targets_for_source(
    manifest: Mapping[str, Any], *, source_artifact_path: str
) -> tuple[ProcessV2ChunkTarget, ...]:
    """Every chunk of one validated source manifest, as bound targets."""

    require_process_v2_artifact_path(source_artifact_path, field="source_artifact_path")
    return tuple(
        ProcessV2ChunkTarget(
            source_artifact_path=source_artifact_path,
            cache_source_task_identity_sha256=str(manifest["task_identity_sha256"]),
            cache_source_manifest_sha256=str(manifest["manifest_sha256"]),
            cache_semantic_identity_sha256=str(manifest["cache_semantic_identity_sha256"]),
            cache_physical_identity_sha256=str(manifest["cache_physical_identity_sha256"]),
            v1_task_identity_sha256=str(manifest["v1_task_identity_sha256"]),
            data_lane=str(manifest["data_lane"]),
            split=str(manifest["split"]),
            semantic_shard_sha256=str(manifest["semantic_shard_sha256"]),
            pinned_process_identity_sha256=str(manifest["pinned_process_identity_sha256"]),
            chunk_index=int(chunk["chunk_index"]),
            chunk_filename=str(chunk["chunk_filename"]),
            chunk_file_sha256=str(chunk["chunk_file_sha256"]),
            chunk_uncompressed_sha256=str(chunk["uncompressed_sha256"]),
            entry_start=int(chunk["entry_start"]),
            entry_stop=int(chunk["entry_stop"]),
            row_count=int(chunk["row_count"]),
        )
        for chunk in manifest["chunks"]
    )


def _bind_target_manifest(
    source_output: Path, target: ProcessV2ChunkTarget, *, repo_root: Path | None
) -> Mapping[str, Any]:
    """Validate the source manifest structurally and bind the declared target.

    Opens exactly one file: the manifest.  No sibling chunk is opened, read or
    hashed, which is the property the whole target-only reader exists for.
    """

    manifest_path = Path(source_output) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ProcessV2ChunkCacheIncomplete(f"the source chunk manifest is absent: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessV2ChunkCacheError(
            f"the source chunk manifest is not JSON: {manifest_path}"
        ) from error
    manifest = validate_process_v2_chunk_cache_manifest(raw, repo_root=repo_root)
    if manifest["manifest_sha256"] != target.cache_source_manifest_sha256:
        raise ProcessV2ChunkCacheError(
            f"the source chunk manifest at {manifest_path} is not the one the target binds"
        )
    disagreements = [
        field
        for field, declared, observed in (
            (
                "task_identity_sha256",
                target.cache_source_task_identity_sha256,
                manifest["task_identity_sha256"],
            ),
            (
                "cache_semantic_identity_sha256",
                target.cache_semantic_identity_sha256,
                manifest["cache_semantic_identity_sha256"],
            ),
            (
                "cache_physical_identity_sha256",
                target.cache_physical_identity_sha256,
                manifest["cache_physical_identity_sha256"],
            ),
            (
                "v1_task_identity_sha256",
                target.v1_task_identity_sha256,
                manifest["v1_task_identity_sha256"],
            ),
            ("data_lane", target.data_lane, manifest["data_lane"]),
            ("split", target.split, manifest["split"]),
            (
                "semantic_shard_sha256",
                target.semantic_shard_sha256,
                manifest["semantic_shard_sha256"],
            ),
            (
                "pinned_process_identity_sha256",
                target.pinned_process_identity_sha256,
                manifest["pinned_process_identity_sha256"],
            ),
        )
        if declared != observed
    ]
    if disagreements:
        raise ProcessV2ChunkCacheError(
            f"the bound chunk target disagrees with its source manifest on {sorted(disagreements)}"
        )
    chunks = manifest["chunks"]
    if not 0 <= target.chunk_index < len(chunks):
        raise ProcessV2ChunkCacheError(
            f"chunk {target.chunk_index} is outside the {len(chunks)}-chunk source cache"
        )
    chunk = chunks[target.chunk_index]
    if dict(chunk) != {
        "chunk_index": target.chunk_index,
        "entry_start": target.entry_start,
        "entry_stop": target.entry_stop,
        "row_count": target.row_count,
        "uncompressed_sha256": target.chunk_uncompressed_sha256,
        "uncompressed_bytes": int(chunk["uncompressed_bytes"]),
        "chunk_filename": target.chunk_filename,
        "chunk_file_sha256": target.chunk_file_sha256,
    }:
        raise ProcessV2ChunkCacheError(
            f"chunk {target.chunk_index} does not carry the identity the target binds"
        )
    return manifest


def iter_process_v2_chunk_target_rows(
    source_output: Path,
    *,
    target: ProcessV2ChunkTarget,
    repo_root: Path | None = None,
) -> Iterator[tuple[int, bytes, dict[str, Any]]]:
    """Yield ``(entry_index, raw_line, record)`` for exactly one target chunk.

    The raw JSON layer, with no chemistry decoding: the chunk file is hashed
    once against its bound digest, decompressed once, and the verbatim source
    line is exposed alongside the parsed record.
    """

    _bind_target_manifest(source_output, target, repo_root=repo_root)
    path = Path(source_output) / target.chunk_filename
    if not path.is_file():
        raise ProcessV2ChunkCacheIncomplete(f"the target chunk is absent: {path}")
    if _sha256_file(path) != target.chunk_file_sha256:
        raise ProcessV2ChunkCacheError(f"the target chunk's physical hash disagrees: {path}")
    digest = hashlib.sha256()
    entry_index = target.entry_start
    with gzip.open(path, "rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                raise ProcessV2ChunkCacheError(f"chunk row {entry_index} is blank: {path}")
            if entry_index >= target.entry_stop:
                raise ProcessV2ChunkCacheError(
                    f"the target chunk holds more than its declared {target.row_count} rows"
                )
            digest.update(raw_line)
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ProcessV2ChunkCacheError(
                    f"chunk row {entry_index} is not JSON: {path}"
                ) from error
            if not isinstance(record, dict):
                raise ProcessV2ChunkCacheError(f"chunk row {entry_index} is not an object: {path}")
            yield entry_index, raw_line, record
            entry_index += 1
    if entry_index != target.entry_stop:
        raise ProcessV2ChunkCacheError(
            f"the target chunk holds {entry_index - target.entry_start} rows, "
            f"its manifest declares {target.row_count}"
        )
    if digest.hexdigest() != target.chunk_uncompressed_sha256:
        raise ProcessV2ChunkCacheError(f"target chunk record bytes disagree with their hash: {path}")


def read_process_v2_chunk_target(
    source_output: Path,
    *,
    target: ProcessV2ChunkTarget,
    expected_process_identity: Mapping[str, object] | None = None,
    sentinel_replay_entries: int = 0,
    recover_row_errors: bool = False,
    repo_root: Path | None = None,
) -> Iterator[SemanticPackedRowRead]:
    """Decode exactly one target chunk into the production semantic row type.

    Every row becomes the same :class:`SemanticPackedRowRead` the raw production
    reader yields, carrying the same global ``entry_index``, the same exact
    slot-addressed states decoded from the persisted arrays, the same ActionV4
    trace and the same immutable shard address.  Nothing is reconstructed from a
    canonical SMILES key: the cache stores the source line verbatim, and this
    decodes that line through the same ``decode_semantic_trace_record`` and
    ``decode_state`` the raw reader calls.

    ``recover_row_errors`` has the raw reader's meaning exactly: a per-row
    decode, lane/split or duplicate-identity failure becomes a reason-carrying
    row instead of an exception, and the reason strings are the raw reader's own
    so a consumer cannot tell the two apart.  Artifact-level failures always
    raise.
    """

    if type(sentinel_replay_entries) is not int or sentinel_replay_entries < 0:
        raise ValueError("sentinel_replay_entries must be a nonnegative integer")
    if type(recover_row_errors) is not bool:
        raise ValueError("recover_row_errors must be a boolean")
    rows = iter_process_v2_chunk_target_rows(source_output, target=target, repo_root=repo_root)
    seen_trace_ids: set[str] = set()
    for entry_index, _raw_line, record in rows:
        try:
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
            detail = f"semantic packed row {entry_index} is malformed: {error}"
            if not recover_row_errors:
                raise ProcessV2ChunkCacheError(detail) from error
            yield SemanticPackedRowRead(
                entry_index=entry_index, record=dict(record), addressed=None, error=detail
            )
            continue
        if record["data_lane"] != target.data_lane or record["split"] != target.split:
            detail = f"semantic packed row {entry_index} leaves its declared lane or split"
            if not recover_row_errors:
                raise ProcessV2ChunkCacheError(detail)
            yield SemanticPackedRowRead(
                entry_index=entry_index, record=dict(record), addressed=None, error=detail
            )
            continue
        trace_id = str(record["trace_id"])
        if trace_id in seen_trace_ids:
            detail = f"duplicate trace_id {trace_id!r} in semantic packed entry range"
            if not recover_row_errors:
                raise ProcessV2ChunkCacheError(detail)
            yield SemanticPackedRowRead(
                entry_index=entry_index, record=dict(record), addressed=None, error=detail
            )
            continue
        seen_trace_ids.add(trace_id)
        canonical_keys = record["canonical_state_keys"]
        yield SemanticPackedRowRead(
            entry_index=entry_index,
            record=dict(record),
            addressed=AddressedPackedTrace(
                address=PackedTraceAddress(
                    packed_shard_content_sha256=target.semantic_shard_sha256,
                    packed_shard_name=SEMANTIC_SHARD_FILENAME,
                    entry_index=entry_index,
                    trace_id=trace_id,
                    layer=str(record["data_lane"]),
                    partition=str(record["split"]),
                    source_key=str(canonical_keys[0]),
                    target_key=str(canonical_keys[-1]),
                    path_length=int(record["path_length"]),
                ),
                trace=trace,
                path=PackedTraceProgress(trace, states),
            ),
            error=None,
        )


def iter_process_v2_chunk_cache_records(
    plan: Mapping[str, Any],
    *,
    artifact_root: Path,
    task_identity_sha256: str,
    repo_root: Path | None = None,
) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield every ``(entry_index, record)`` of one cached source, in order."""

    validated = validate_process_v2_chunk_cache_plan(plan, repo_root=repo_root)
    task = _task_by_identity(validated, task_identity_sha256)
    artifact_path = str(task["output_artifact_path"])
    output = _mounted(artifact_path, artifact_root, "task.output_artifact_path")
    manifest = validate_process_v2_chunk_cache_source(output, repo_root=repo_root)
    for target in chunk_targets_for_source(manifest, source_artifact_path=artifact_path):
        for entry_index, _raw_line, record in iter_process_v2_chunk_target_rows(
            output, target=target, repo_root=repo_root
        ):
            yield entry_index, record


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
    "CACHE_IMPLEMENTATION_FILES",
    "CACHE_IMPLEMENTATION_REVISION_SCHEMA",
    "CACHE_IMPLEMENTATION_REVISION_SCHEMA_VERSION",
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
    "DEFAULT_REPO_ROOT",
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
    "ProcessV2ArtifactPathError",
    "ProcessV2ChunkCacheError",
    "ProcessV2ChunkCacheGeneration",
    "ProcessV2ChunkCacheIncomplete",
    "ProcessV2ChunkCacheIntegrityError",
    "ProcessV2ChunkTarget",
    "ProcessV2ConcurrencyError",
    "build_cache_implementation_revision",
    "cache_implementation_sha256",
    "cache_semantic_identity",
    "chunk_object_filename",
    "chunk_targets_for_source",
    "completed_process_v2_chunk_cache_task_ids",
    "execute_process_v2_chunk_cache_task",
    "iter_process_v2_chunk_cache_records",
    "iter_process_v2_chunk_target_rows",
    "load_committed_process_v2_chunk_cache_completion",
    "load_process_v2_chunk_cache_completion",
    "migration_identity_from_completion",
    "mount_process_v2_artifact_path",
    "open_process_v2_chunk_cache",
    "plan_process_v2_chunk_cache",
    "plan_submission_waves",
    "read_process_v2_chunk_target",
    "reduce_process_v2_chunk_cache",
    "reduction_order",
    "require_process_v2_artifact_path",
    "run_bounded_map",
    "scan_source_shard_once",
    "validate_cache_implementation_revision",
    "validate_map_container_bound",
    "validate_process_v2_chunk_cache_completion",
    "validate_process_v2_chunk_cache_manifest",
    "validate_process_v2_chunk_cache_plan",
    "validate_process_v2_chunk_cache_source",
    "write_chunk_cache_refusal_report",
    "write_process_v2_chunk_cache_plan",
]
