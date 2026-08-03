"""One worker, one chunk: what a target read touches, and what it produces.

Two properties are proven here, and the second is the one that makes the cache
usable as a *replacement* rather than an optimisation.

**A target read touches exactly two files.**  The old
``read_process_v2_chunk_records`` called the whole-source validator, which loops
every chunk in the manifest and SHA-256s each one.  Reading a source's chunks
one at a time therefore cost ``(N+1) x N`` full chunk hashes -- at the production
geometry, roughly 290 full gzip reads to read 17 chunks once.  The replacement
validates the manifest structurally, binds the declared target, and hashes only
that chunk.  Both the instrumented file opens and a stronger witness are used:
every sibling chunk is *deleted* and the target still reads.

**A cached row is the raw reader's row.**  ``read_process_v2_chunk_target``
yields the production ``SemanticPackedRowRead``, and this compares it field by
field against ``read_semantic_packed_artifact_range_rows`` over the same
half-open range on the same payload: the same global entry index, the same
verbatim record, the same immutable shard address, the same ActionV4 steps and
the same exact slot-addressed state arrays.  Nothing is reconstructed from a
canonical SMILES key -- the comparison would catch it if it were, because a
reconstruction cannot reproduce the persisted slot layout.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    MANIFEST_FILENAME,
    ProcessV2ChunkCacheError,
    ProcessV2ChunkTarget,
    execute_process_v2_chunk_cache_task,
    iter_process_v2_chunk_target_rows,
    open_process_v2_chunk_cache,
    plan_process_v2_chunk_cache,
    read_process_v2_chunk_target,
    reduce_process_v2_chunk_cache,
    validate_process_v2_chunk_cache_source,
    write_process_v2_chunk_cache_plan,
)
from compose_v4.data.editing_v2_process_v2_schema import canonical_bytes, canonical_sha256
from compose_v4.data.semantic_packed_trace_store import (
    read_semantic_packed_artifact_range_rows,
)
from compose_v4.data.semantic_trace_migration_materializer import (
    SEMANTIC_ARTIFACT_DIRNAME,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests"))

import test_editing_process_v2_rebind as v1_fixture  # noqa: E402
import test_editing_v2_process_v2_completion_binder as binder_fixture  # noqa: E402

_register_extra_fixture_traces = binder_fixture._register_extra_fixture_traces

# Five real V1-admitted traces in every cell, so a chunk is a strict subset of
# its source and a global entry index is not trivially its local one.
_DEEP_CELL = (
    "cyclize_hexane",
    "cyclize_fluoropentane",
    "trim_methylcyclohexane",
    "open_cyclohexane",
    "open_benzene",
)
_DEEP_TASKS = tuple(
    (lane, role, _DEEP_CELL) for lane, role, _names in binder_fixture._TASKS
)


# ---- Fixtures -----------------------------------------------------------------


def _built_cache(tmp_path: Path, *, records_per_chunk: int = 2):
    payload, _completion, expectation = binder_fixture.build_migration_run(
        tmp_path / "artifacts", tasks=_DEEP_TASKS
    )
    binding = binder_fixture.bind(payload, expectation)
    plan = plan_process_v2_chunk_cache(
        binding,
        output_artifact_prefix="/artifacts/chunk_target_fixture",
        records_per_chunk=records_per_chunk,
    )
    write_process_v2_chunk_cache_plan(plan, artifact_root=payload.artifact_root)
    for task in plan["tasks"]:
        execute_process_v2_chunk_cache_task(
            plan, task["task_identity_sha256"], artifact_root=payload.artifact_root
        )
    completion = reduce_process_v2_chunk_cache(plan, artifact_root=payload.artifact_root)
    generation = open_process_v2_chunk_cache(completion, artifact_root=payload.artifact_root)
    return payload, plan, generation


def _mounted(payload, artifact_path: str) -> Path:
    return payload.artifact_root / str(artifact_path).removeprefix("/artifacts/")


class _OpenRecorder:
    """Record every file opened under one artifact root, however it is opened.

    Scoped to the artifact root on purpose.  Re-deriving the narrow
    implementation revision re-hashes nine repository modules on every call,
    which is a source read and not an artifact read; counting those would make
    the assertion about the wrong thing.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, artifact_root: Path) -> None:
        self.paths: list[Path] = []
        self._root = Path(artifact_root).resolve()
        real_path_open = Path.open
        real_gzip_open = gzip.open

        def record(path: Any) -> None:
            resolved = Path(path).resolve()
            if resolved.is_relative_to(self._root):
                self.paths.append(resolved)

        def path_open(path_self, *args: Any, **kwargs: Any):
            record(path_self)
            return real_path_open(path_self, *args, **kwargs)

        def gzip_open(filename: Any, *args: Any, **kwargs: Any):
            record(filename)
            return real_gzip_open(filename, *args, **kwargs)

        monkeypatch.setattr(Path, "open", path_open)
        monkeypatch.setattr(gzip, "open", gzip_open)

    @property
    def names(self) -> set[str]:
        return {path.name for path in self.paths}


def _deterministic_gzip(rows) -> bytes:
    """The cache's own chunk encoding, so a rewritten chunk is still a chunk."""

    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as handle:
        for row in rows:
            handle.write(row)
    return output.getvalue()


def _row_projection(row) -> dict[str, Any]:
    """A comparable projection of one ``SemanticPackedRowRead``.

    Deliberately explicit rather than an ``==`` on the dataclass: the decoded
    trace holds ``MolecularGraph`` states whose numpy arrays do not compare with
    ``==``, so an equality that silently degraded to identity would pass while
    comparing nothing.
    """

    projection: dict[str, Any] = {
        "entry_index": row.entry_index,
        "record": row.record,
        "error": row.error,
        "addressed": row.addressed is not None,
    }
    if row.addressed is None:
        return projection
    projection["address"] = row.addressed.address
    projection["steps"] = [
        (step.rule_name, step.action) for step in row.addressed.trace.steps
    ]
    projection["states"] = [
        (
            state.atom_types.tobytes(),
            state.bonds.tobytes(),
            state.formal_charges.tobytes(),
            state.implicit_h_counts.tobytes(),
        )
        for state in row.addressed.path.states
    ]
    return projection


def _raw_oracle_rows(payload, target: ProcessV2ChunkTarget) -> list[Any]:
    """The bounded raw-packed-shard oracle, over the target's exact range."""

    v1_task_dir = (
        payload.payload_root / v1_fixture.TASK_DIRNAME / target.v1_task_identity_sha256
    )
    receipt = json.loads((v1_task_dir / v1_fixture.V1_RECEIPT_FILENAME).read_bytes())
    process_identity, builder_identity = v1_fixture._pinned_identities()
    return list(
        read_semantic_packed_artifact_range_rows(
            v1_task_dir / SEMANTIC_ARTIFACT_DIRNAME,
            expected_shard_sha256=receipt["semantic_shard_sha256"],
            expected_manifest_sha256=receipt["semantic_manifest_sha256"],
            expected_source_binding=receipt["source_binding"],
            expected_process_identity=process_identity,
            expected_builder_identity=builder_identity,
            entry_start=target.entry_start,
            entry_stop=target.entry_stop,
            sentinel_replay_entries=0,
            recover_row_errors=True,
        )
    )


# ---- Acceptance test 12: a target read opens no sibling chunk -----------------


def test_a_target_read_opens_only_its_manifest_and_its_own_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, _plan, generation = _built_cache(tmp_path)
    targets = generation.targets()
    source_path = targets[0].source_artifact_path
    siblings = [target for target in targets if target.source_artifact_path == source_path]
    assert len(siblings) >= 3, "the fixture must publish a multi-chunk source"
    target = siblings[1]
    source_dir = _mounted(payload, source_path)

    recorder = _OpenRecorder(monkeypatch, artifact_root=payload.artifact_root)
    rows = list(read_process_v2_chunk_target(source_dir, target=target))
    opened = recorder.names

    assert len(rows) == target.row_count
    assert opened == {MANIFEST_FILENAME, target.chunk_filename}
    other_names = {
        other.chunk_filename for other in siblings if other.chunk_index != target.chunk_index
    }
    assert other_names and not (opened & other_names)
    # And the original packed shard, which the whole cache exists to stop
    # reopening, is never touched either.
    assert not any("traces.jsonl.gz" == path.name for path in recorder.paths)


def test_the_whole_generation_validator_does_open_every_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The positive control: the instrument can see a sibling when one is read.

    Without this the previous test would pass just as well against a recorder
    that observes nothing.
    """

    payload, _plan, generation = _built_cache(tmp_path)
    targets = generation.targets()
    source_path = targets[0].source_artifact_path
    siblings = [target for target in targets if target.source_artifact_path == source_path]

    recorder = _OpenRecorder(monkeypatch, artifact_root=payload.artifact_root)
    validate_process_v2_chunk_cache_source(_mounted(payload, source_path))
    assert {target.chunk_filename for target in siblings} <= recorder.names


def test_a_target_reads_with_every_sibling_chunk_deleted(tmp_path: Path) -> None:
    """The witness that does not depend on an instrument being correct."""

    payload, _plan, generation = _built_cache(tmp_path)
    targets = generation.targets()
    source_path = targets[0].source_artifact_path
    siblings = [target for target in targets if target.source_artifact_path == source_path]
    target = siblings[-1]
    source_dir = _mounted(payload, source_path)

    expected = [_row_projection(row) for row in read_process_v2_chunk_target(source_dir, target=target)]
    for other in siblings:
        if other.chunk_index != target.chunk_index:
            (source_dir / other.chunk_filename).unlink()

    observed = [_row_projection(row) for row in read_process_v2_chunk_target(source_dir, target=target)]
    assert observed == expected
    # And the whole-source validator, which is the boundary that owes the
    # whole-generation guarantee, now correctly refuses the same directory.
    with pytest.raises(ProcessV2ChunkCacheError):
        validate_process_v2_chunk_cache_source(source_dir)


# ---- The target is a bound expectation, not a hint ----------------------------


def test_a_target_naming_another_chunks_hash_is_refused(tmp_path: Path) -> None:
    payload, _plan, generation = _built_cache(tmp_path)
    targets = generation.targets()
    source_path = targets[0].source_artifact_path
    siblings = [target for target in targets if target.source_artifact_path == source_path]
    source_dir = _mounted(payload, source_path)

    swapped = ProcessV2ChunkTarget(
        **{
            **siblings[0].__dict__,
            "chunk_file_sha256": siblings[1].chunk_file_sha256,
        }
    )
    with pytest.raises(ProcessV2ChunkCacheError, match="identity the target binds"):
        list(read_process_v2_chunk_target(source_dir, target=swapped))

    shifted = ProcessV2ChunkTarget(**{**siblings[0].__dict__, "entry_start": 99})
    with pytest.raises(ProcessV2ChunkCacheError, match="identity the target binds"):
        list(read_process_v2_chunk_target(source_dir, target=shifted))

    foreign = ProcessV2ChunkTarget(
        **{**siblings[0].__dict__, "cache_source_manifest_sha256": "0" * 64}
    )
    with pytest.raises(ProcessV2ChunkCacheError, match="not the one the target binds"):
        list(read_process_v2_chunk_target(source_dir, target=foreign))


def test_a_chunk_whose_decompressed_stream_is_not_its_declared_one_is_refused(
    tmp_path: Path,
) -> None:
    """The uncompressed digest is a separate proof from the physical one.

    Found by mutation: bypassing it left every focused suite green, because the
    only tampering the suite performed changed the compressed bytes and so was
    caught one check earlier. The reachable witness is a manifest resealed with
    a wrong ``uncompressed_sha256`` over an untouched chunk file -- a publisher
    that computed the record-stream digest wrongly, or a gzip that decompresses
    to something other than what was hashed at build time.
    """

    payload, _plan, generation = _built_cache(tmp_path)
    target = generation.targets()[0]
    source_dir = _mounted(payload, target.source_artifact_path)

    manifest = json.loads((source_dir / MANIFEST_FILENAME).read_bytes())
    manifest["chunks"][target.chunk_index]["uncompressed_sha256"] = "0" * 64
    manifest["chunk_inventory_sha256"] = canonical_sha256(
        [dict(chunk) for chunk in manifest["chunks"]]
    )
    body = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    resealed = {**body, "manifest_sha256": canonical_sha256(body)}
    (source_dir / MANIFEST_FILENAME).write_bytes(canonical_bytes(resealed) + b"\n")

    # The target is rebuilt from the resealed manifest, so the binding agrees
    # and only the stream digest can refuse the read.
    retargeted = ProcessV2ChunkTarget(
        **{
            **target.__dict__,
            "cache_source_manifest_sha256": resealed["manifest_sha256"],
            "chunk_uncompressed_sha256": "0" * 64,
        }
    )
    assert (source_dir / target.chunk_filename).exists()
    with pytest.raises(ProcessV2ChunkCacheError, match="record bytes disagree"):
        list(read_process_v2_chunk_target(source_dir, target=retargeted))


def test_a_target_chunk_missing_rows_is_refused_by_its_row_count(tmp_path: Path) -> None:
    payload, _plan, generation = _built_cache(tmp_path)
    target = generation.targets()[0]
    source_dir = _mounted(payload, target.source_artifact_path)

    rows = list(iter_process_v2_chunk_target_rows(source_dir, target=target))
    assert len(rows) == target.row_count >= 2
    truncated = _deterministic_gzip(row[1] for row in rows[:-1])
    chunk_path = source_dir / target.chunk_filename
    chunk_path.write_bytes(truncated)

    manifest = json.loads((source_dir / MANIFEST_FILENAME).read_bytes())
    chunk = manifest["chunks"][target.chunk_index]
    chunk["chunk_file_sha256"] = hashlib.sha256(truncated).hexdigest()
    manifest["chunk_inventory_sha256"] = canonical_sha256(
        [dict(entry) for entry in manifest["chunks"]]
    )
    body = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    resealed = {**body, "manifest_sha256": canonical_sha256(body)}
    (source_dir / MANIFEST_FILENAME).write_bytes(canonical_bytes(resealed) + b"\n")

    retargeted = ProcessV2ChunkTarget(
        **{
            **target.__dict__,
            "cache_source_manifest_sha256": resealed["manifest_sha256"],
            "chunk_file_sha256": chunk["chunk_file_sha256"],
        }
    )
    with pytest.raises(ProcessV2ChunkCacheError, match="rows, its manifest declares"):
        list(read_process_v2_chunk_target(source_dir, target=retargeted))


def test_a_tampered_target_chunk_is_refused_by_its_own_hash(tmp_path: Path) -> None:
    payload, _plan, generation = _built_cache(tmp_path)
    target = generation.targets()[0]
    source_dir = _mounted(payload, target.source_artifact_path)
    chunk_path = source_dir / target.chunk_filename
    chunk_path.write_bytes(chunk_path.read_bytes() + b"\x00")
    with pytest.raises(ProcessV2ChunkCacheError, match="physical hash disagrees"):
        list(read_process_v2_chunk_target(source_dir, target=target))


# ---- Acceptance test 13 (rows): the cache row is the raw reader's row ---------


def test_every_cached_row_equals_the_raw_oracle_row(tmp_path: Path) -> None:
    payload, _plan, generation = _built_cache(tmp_path)
    process_identity, _builder = v1_fixture._pinned_identities()

    compared = 0
    for target in generation.targets():
        source_dir = _mounted(payload, target.source_artifact_path)
        cached = [
            _row_projection(row)
            for row in read_process_v2_chunk_target(
                source_dir,
                target=target,
                expected_process_identity=process_identity,
                recover_row_errors=True,
            )
        ]
        oracle = [_row_projection(row) for row in _raw_oracle_rows(payload, target)]
        assert cached == oracle
        assert [row["entry_index"] for row in cached] == list(
            range(target.entry_start, target.entry_stop)
        )
        compared += len(cached)
    assert compared == int(generation.completion["entries"])
    assert compared > int(generation.completion["source_count"]), (
        "the comparison must cover more than one row per source"
    )


def test_the_global_entry_index_is_the_shard_index_not_the_chunk_index(
    tmp_path: Path,
) -> None:
    payload, _plan, generation = _built_cache(tmp_path)
    by_source: dict[str, list[int]] = {}
    for target in generation.targets():
        source_dir = _mounted(payload, target.source_artifact_path)
        indices = [
            row.entry_index for row in read_process_v2_chunk_target(source_dir, target=target)
        ]
        by_source.setdefault(target.source_artifact_path, []).extend(indices)
        assert indices == list(range(target.entry_start, target.entry_stop))
    for indices in by_source.values():
        assert indices == list(range(len(indices)))
        assert len(indices) == len(_DEEP_CELL)
    # A chunk past the first therefore starts above zero, which is the case a
    # chunk-local index would silently get wrong.
    assert any(target.entry_start > 0 for target in generation.targets())
