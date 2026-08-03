"""The chunk cache reads each source shard once, and proves it.

The defect this cache exists to remove is not hypothetical.  Every entry-range
reader available to the Process-V2 chain
(``semantic_packed_trace_store.read_semantic_packed_artifact_range_rows``)
hashes the whole compressed shard and then decompresses from byte zero to reach
its range, so ``R`` ranges over one shard cost ``R`` full-file hashes and about
``R/2`` full decompressions.  ``test_a_range_reader_rehashes_and_redecompresses``
measures that on a real production-built artifact; the rest of this module pins
the replacement.

What is proven here
-------------------

1. one ``open``, one sequential read of exactly the file's bytes, one
   decompression, per source shard, measured by instrumenting the opener;
2. peak memory bounded by the read block and one chunk, not by shard size;
3. the cached records are the source records, byte for byte and in order;
4. the semantic cache identity is invariant to chunk size and to worker
   schedule, while the physical identity moves with chunk size;
5. restart is exact: a killed worker publishes nothing, a rerun reuses;
6. the reducer refuses a missing, duplicated, colliding or foreign source cache;
7. an integrity mismatch publishes nothing under the run namespace and leaves a
   structured refusal in a separate diagnostic namespace.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from compose_v4.data.editing_v2_process_v2_chunk_cache import (
    DECOMPRESS_BLOCK_BYTES,
    DEFAULT_RECORDS_PER_CHUNK,
    MANIFEST_FILENAME,
    MAX_RECORDS_PER_CHUNK,
    PLAN_FILENAME,
    READ_BLOCK_BYTES,
    SOURCE_DIRNAME,
    ProcessV2ChunkCacheError,
    ProcessV2ChunkCacheIncomplete,
    ProcessV2ChunkCacheIntegrityError,
    completed_process_v2_chunk_cache_task_ids,
    execute_process_v2_chunk_cache_task,
    iter_process_v2_chunk_cache_records,
    load_process_v2_chunk_cache_completion,
    plan_process_v2_chunk_cache,
    reduce_process_v2_chunk_cache,
    run_bounded_map,
    scan_source_shard_once,
    validate_process_v2_chunk_cache_plan,
    write_chunk_cache_refusal_report,
    write_process_v2_chunk_cache_plan,
)
from compose_v4.data.editing_v2_process_v2_schema import canonical_bytes, canonical_sha256
from compose_v4.data.semantic_packed_trace_store import (
    MANIFEST_FILENAME as SEMANTIC_MANIFEST_FILENAME,
)
from compose_v4.data.semantic_packed_trace_store import (
    SHARD_FILENAME as SEMANTIC_SHARD_FILENAME,
)
from compose_v4.data.semantic_packed_trace_store import (
    read_semantic_packed_artifact_range_rows,
)
from compose_v4.data.semantic_trace_migration_materializer import (
    SEMANTIC_ARTIFACT_DIRNAME,
)
from compose_v4.rewrite import action_codec_v4
from compose_v4.rewrite.trace_shard_v3 import TRACE_SCHEMA, TRACE_SCHEMA_VERSION

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests"))

import test_editing_process_v2_rebind as v1_fixture  # noqa: E402
import test_editing_v2_process_v2_completion_binder as binder_fixture  # noqa: E402

_register_extra_fixture_traces = binder_fixture._register_extra_fixture_traces

PINNED_PROCESS_IDENTITY = "e" * 64
PINNED_CONTRACT = "d" * 64


# ---- A synthetic semantic artifact, for the pass and memory instruments -------


def _synthetic_semantic_dir(
    directory: Path,
    *,
    entries: int,
    data_lane: str = "observed_local_analogue",
    split: str = "train",
    payload_bytes: int = 512,
) -> Path:
    """Write a semantic artifact whose rows carry only the identity fields.

    The cache scanner never decodes chemistry, so a synthetic shard exercises
    exactly the code path a production shard does while letting the instruments
    vary entry count and row size independently.
    """

    directory.mkdir(parents=True, exist_ok=True)
    lines: list[bytes] = []
    for index in range(entries):
        row = {
            "schema": TRACE_SCHEMA,
            "schema_version": TRACE_SCHEMA_VERSION,
            "trace_id": f"semantic-{index:08d}",
            "data_lane": data_lane,
            "split": split,
            "path_length": 1,
            "record_sha256": hashlib.sha256(f"row-{index}".encode()).hexdigest(),
            "process_identity_sha256": PINNED_PROCESS_IDENTITY,
            "process_contract_sha256": PINNED_CONTRACT,
            "action_codec_schema_version": action_codec_v4.SCHEMA_VERSION,
            "action_codec_implementation_hash": action_codec_v4.codec_implementation_hash(),
            "filler": "x" * payload_bytes,
        }
        lines.append(canonical_bytes(row) + b"\n")
    shard = directory / SEMANTIC_SHARD_FILENAME
    with shard.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            for line in lines:
                compressed.write(line)
    stream = hashlib.sha256()
    for line in lines:
        stream.update(line)
    manifest_body: dict[str, Any] = {
        "schema": "compose.data.semantic_packed_trace",
        "shard_filename": SEMANTIC_SHARD_FILENAME,
        "shard_sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
        "record_stream_sha256": stream.hexdigest(),
        "entries": entries,
        "data_lane": data_lane,
        "split": split,
        "process_identity_sha256": PINNED_PROCESS_IDENTITY,
        "action_codec_implementation_hash": action_codec_v4.codec_implementation_hash(),
        "trace_schema": TRACE_SCHEMA,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
    }
    manifest = {**manifest_body, "manifest_sha256": canonical_sha256(manifest_body)}
    (directory / SEMANTIC_MANIFEST_FILENAME).write_bytes(canonical_bytes(manifest) + b"\n")
    return directory


class _CountingOpener:
    """Count opens of the shard and the bytes actually read from it."""

    def __init__(self) -> None:
        self.opens = 0
        self.reads = 0
        self.bytes_read = 0

    def __call__(self, path: Path, mode: str = "rb"):
        self.opens += 1
        counter = self

        class _Handle:
            def __init__(self) -> None:
                self._handle = open(path, mode)

            def read(self, size: int = -1) -> bytes:
                block = self._handle.read(size)
                counter.reads += 1
                counter.bytes_read += len(block)
                return block

            def __enter__(self):
                return self

            def __exit__(self, *_exc: object) -> None:
                self._handle.close()

        return _Handle()


def _scan(directory: Path, *, records_per_chunk: int, opener) -> Any:
    return scan_source_shard_once(
        directory,
        data_lane="observed_local_analogue",
        split="train",
        expected_shard_sha256=json.loads(
            (directory / SEMANTIC_MANIFEST_FILENAME).read_bytes()
        )["shard_sha256"],
        expected_manifest_sha256=hashlib.sha256(
            (directory / SEMANTIC_MANIFEST_FILENAME).read_bytes()
        ).hexdigest(),
        pinned_process_identity_sha256=PINNED_PROCESS_IDENTITY,
        records_per_chunk=records_per_chunk,
        opener=opener,
    )


# ---- 1. Exactly one fused pass ------------------------------------------------


def test_one_source_shard_is_opened_once_and_read_exactly_once(tmp_path: Path) -> None:
    directory = _synthetic_semantic_dir(tmp_path / "semantic", entries=4096)
    shard = directory / SEMANTIC_SHARD_FILENAME
    opener = _CountingOpener()

    scan = _scan(directory, records_per_chunk=256, opener=opener)

    assert opener.opens == 1, "the fused pass must open the raw shard exactly once"
    assert opener.bytes_read == shard.stat().st_size
    assert scan.compressed_bytes == shard.stat().st_size
    assert scan.entries == 4096
    assert len(scan.chunks) == 16
    # The physical digest came out of that same read, not a second one.
    assert scan.shard_file_sha256 == hashlib.sha256(shard.read_bytes()).hexdigest()


def test_peak_memory_is_bounded_by_the_block_and_one_chunk_not_the_shard(
    tmp_path: Path,
) -> None:
    small = _synthetic_semantic_dir(tmp_path / "small", entries=2_048)
    large = _synthetic_semantic_dir(tmp_path / "large", entries=32_768)

    peaks: list[int] = []
    scans = []
    for directory in (small, large):
        tracemalloc.start()
        scans.append(_scan(directory, records_per_chunk=128, opener=open))
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    assert scans[1].decompressed_bytes > 15 * scans[0].decompressed_bytes
    # Sixteen times the data must not cost anything like sixteen times the
    # memory. The rows here are deliberately highly compressible, which is the
    # case an unbounded `decompressobj.decompress` expands into the whole shard.
    assert peaks[1] < 4 * peaks[0]
    assert peaks[1] < scans[1].decompressed_bytes / 4
    # The structural bound: one read block, one bounded decompression block and
    # one chunk of rows, with slack for the interpreter's own allocations.
    chunk_bound = 128 * (512 + 512)
    assert peaks[1] < 4 * (READ_BLOCK_BYTES + DECOMPRESS_BLOCK_BYTES + chunk_bound)


def test_a_truncated_or_retagged_shard_is_refused_by_the_same_pass(
    tmp_path: Path,
) -> None:
    directory = _synthetic_semantic_dir(tmp_path / "semantic", entries=64)
    shard = directory / SEMANTIC_SHARD_FILENAME
    shard.write_bytes(shard.read_bytes()[: -1])
    with pytest.raises(ProcessV2ChunkCacheIntegrityError):
        _scan(directory, records_per_chunk=16, opener=open)


# ---- 2. The measured range-reader defect --------------------------------------


def test_a_range_reader_rehashes_and_redecompresses(tmp_path: Path) -> None:
    """The status quo, measured on a real production-built artifact.

    Not a performance assertion: it counts *work*, which is deterministic. Each
    range opens the shard twice (once for the physical hash, once for the gzip)
    while the fused pass opens it once in total.
    """

    payload = v1_fixture._build_v1_payload(
        tmp_path / "artifacts",
        tasks=((v1_fixture.REQUIRED_DATA_LANES[0], v1_fixture.REQUIRED_PARTITION_ROLES[0],
                ("cyclize_hexane", "cyclize_fluoropentane", "trim_methylcyclohexane",
                 "open_cyclohexane")),),
    )
    task_dir = payload.payload_root / v1_fixture.TASK_DIRNAME / payload.v1_task_identities[0]
    semantic_dir = task_dir / SEMANTIC_ARTIFACT_DIRNAME
    receipt = json.loads((task_dir / v1_fixture.V1_RECEIPT_FILENAME).read_bytes())
    process_identity, builder_identity = v1_fixture._pinned_identities()

    opened: list[str] = []
    real_path_open = Path.open
    real_gzip_open = gzip.open

    def counting_path_open(self, *args, **kwargs):  # noqa: ANN001
        if self.name == SEMANTIC_SHARD_FILENAME:
            opened.append("hash")
        return real_path_open(self, *args, **kwargs)

    def counting_gzip_open(path, *args, **kwargs):  # noqa: ANN001
        if Path(path).name == SEMANTIC_SHARD_FILENAME:
            opened.append("decompress")
        return real_gzip_open(path, *args, **kwargs)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(Path, "open", counting_path_open)
    monkey.setattr(gzip, "open", counting_gzip_open)
    try:
        for start in range(4):
            list(
                read_semantic_packed_artifact_range_rows(
                    semantic_dir,
                    expected_shard_sha256=receipt["semantic_shard_sha256"],
                    expected_manifest_sha256=receipt["semantic_manifest_sha256"],
                    expected_process_identity=process_identity,
                    expected_builder_identity=builder_identity,
                    entry_start=start,
                    entry_stop=start + 1,
                    sentinel_replay_entries=0,
                    recover_row_errors=False,
                )
            )
    finally:
        monkey.undo()

    assert opened.count("hash") >= 4, "each range hashes the whole compressed shard"
    assert opened.count("decompress") >= 4, "each range decompresses from byte zero"

    fused = _CountingOpener()
    scan = scan_source_shard_once(
        semantic_dir,
        data_lane=receipt["data_lane"],
        split=receipt["split"],
        expected_shard_sha256=receipt["semantic_shard_sha256"],
        expected_manifest_sha256=receipt["semantic_manifest_sha256"],
        pinned_process_identity_sha256=process_identity["process_identity_sha256"],
        records_per_chunk=2,
        opener=fused,
    )
    assert fused.opens == 1
    assert fused.bytes_read == (semantic_dir / SEMANTIC_SHARD_FILENAME).stat().st_size
    assert scan.entries == 4
    assert len(opened) >= 8 * fused.opens


# ---- 3. End to end over the exact twenty-cell completion ----------------------


def _bound_run(tmp_path: Path):
    payload, _completion, expectation = binder_fixture.build_migration_run(tmp_path / "artifacts")
    binding = binder_fixture.bind(payload, expectation)
    return payload, binding


def _plan(binding, *, records_per_chunk: int = 4):
    return plan_process_v2_chunk_cache(
        binding,
        output_artifact_prefix="/artifacts/chunk_cache_fixture",
        records_per_chunk=records_per_chunk,
    )


def _run_root(payload, plan) -> Path:
    return payload.artifact_root / str(plan["run_artifact_root"]).removeprefix("/artifacts/")


def _build_all(payload, plan, *, order=None) -> list[dict]:
    write_process_v2_chunk_cache_plan(plan, artifact_root=payload.artifact_root)
    tasks = list(plan["tasks"]) if order is None else [plan["tasks"][index] for index in order]
    return [
        execute_process_v2_chunk_cache_task(
            plan, task["task_identity_sha256"], artifact_root=payload.artifact_root
        )
        for task in tasks
    ]


def test_the_cache_holds_every_admitted_record_byte_for_byte(tmp_path: Path) -> None:
    payload, binding = _bound_run(tmp_path)
    plan = _plan(binding)
    _build_all(payload, plan)
    completion = reduce_process_v2_chunk_cache(plan, artifact_root=payload.artifact_root)

    assert completion["source_count"] == 20
    assert completion["entries"] == binding.rebind_source_entries
    assert completion["entries"] == binding.counts["admitted"] < binding.counts["source"]
    assert load_process_v2_chunk_cache_completion(
        plan, artifact_root=payload.artifact_root
    ) == completion

    for task in plan["tasks"]:
        semantic_dir = (
            payload.artifact_root
            / str(task["v1_task_artifact_path"]).removeprefix("/artifacts/")
            / SEMANTIC_ARTIFACT_DIRNAME
        )
        with gzip.open(semantic_dir / SEMANTIC_SHARD_FILENAME, "rb") as handle:
            source_rows = [json.loads(line) for line in handle if line.strip()]
        cached = list(
            iter_process_v2_chunk_cache_records(
                plan,
                artifact_root=payload.artifact_root,
                task_identity_sha256=task["task_identity_sha256"],
            )
        )
        assert [index for index, _row in cached] == list(range(len(source_rows)))
        assert [row for _index, row in cached] == source_rows
        # Exact slot-addressed records: the encoded states travel verbatim and
        # nothing is reconstructed from a canonical SMILES key.
        for _index, row in cached:
            assert row["states"] and len(row["states"]) == row["path_length"] + 1
            assert len(row["canonical_state_keys"]) == row["path_length"] + 1


def test_the_semantic_identity_ignores_chunk_size_while_the_address_moves(
    tmp_path: Path,
) -> None:
    payload, binding = _bound_run(tmp_path)
    small = _plan(binding, records_per_chunk=1)
    large = _plan(binding, records_per_chunk=MAX_RECORDS_PER_CHUNK)

    assert small["cache_semantic_identity_sha256"] == large["cache_semantic_identity_sha256"]
    assert small["cache_physical_identity_sha256"] != large["cache_physical_identity_sha256"]
    assert small["run_artifact_root"] != large["run_artifact_root"]

    _build_all(payload, small)
    _build_all(payload, large)
    small_completion = reduce_process_v2_chunk_cache(small, artifact_root=payload.artifact_root)
    large_completion = reduce_process_v2_chunk_cache(large, artifact_root=payload.artifact_root)

    assert (
        small_completion["cache_semantic_identity_sha256"]
        == large_completion["cache_semantic_identity_sha256"]
    )
    assert small_completion["chunk_count"] > large_completion["chunk_count"]
    assert small_completion["entries"] == large_completion["entries"]


@pytest.mark.parametrize("max_map_containers", [1, 20, 40])
def test_one_twenty_and_forty_worker_schedules_produce_the_same_artifact(
    tmp_path: Path, max_map_containers: int
) -> None:
    """The schedule is a physical choice and must leave no semantic trace.

    Each schedule runs the production ``run_bounded_map`` over the production
    task executor, so the wave geometry really differs; the completion bytes
    must not.
    """

    reference_payload, reference_binding = _bound_run(tmp_path / "reference")
    reference_plan = _plan(reference_binding)
    _build_all(reference_payload, reference_plan)
    reference = reduce_process_v2_chunk_cache(
        reference_plan, artifact_root=reference_payload.artifact_root
    )

    payload, binding = _bound_run(tmp_path / f"w{max_map_containers}")
    plan = _plan(binding)
    write_process_v2_chunk_cache_plan(plan, artifact_root=payload.artifact_root)
    submitted: list[tuple[str, ...]] = []
    run_bounded_map(
        [str(task["task_identity_sha256"]) for task in plan["tasks"]],
        max_map_containers=max_map_containers,
        submit=lambda wave: (
            submitted.append(wave)
            or [
                execute_process_v2_chunk_cache_task(
                    plan, task_id, artifact_root=payload.artifact_root
                )
                for task_id in wave
            ]
        ),
    )
    assert max(len(wave) for wave in submitted) <= max_map_containers
    assert len(submitted) == -(-20 // max_map_containers)

    completion = reduce_process_v2_chunk_cache(plan, artifact_root=payload.artifact_root)
    assert canonical_bytes(completion) == canonical_bytes(reference)
    assert (
        completion["cache_semantic_identity_sha256"]
        == reference["cache_semantic_identity_sha256"]
    )
    assert (
        completion["cache_physical_identity_sha256"]
        == reference["cache_physical_identity_sha256"]
    ), "the schedule is not part of the physical cache identity; the chunk size is"


def test_the_completion_marker_is_absent_until_the_reducer_publishes_it(
    tmp_path: Path,
) -> None:
    payload, binding = _bound_run(tmp_path)
    plan = _plan(binding)
    _build_all(payload, plan)
    with pytest.raises(ProcessV2ChunkCacheIncomplete, match="completion is absent"):
        load_process_v2_chunk_cache_completion(plan, artifact_root=payload.artifact_root)
    reduce_process_v2_chunk_cache(plan, artifact_root=payload.artifact_root)
    assert load_process_v2_chunk_cache_completion(plan, artifact_root=payload.artifact_root)


def test_a_different_corpus_does_change_the_semantic_identity(tmp_path: Path) -> None:
    """The negative half of invariance: the digest is not simply constant."""

    baseline_payload, baseline = _bound_run(tmp_path / "baseline")
    altered_payload, _completion, altered_expectation = binder_fixture.build_migration_run(
        tmp_path / "altered" / "artifacts",
        tasks=tuple(
            (lane, role, ("trim_methylcyclohexane",) + tuple(names[1:]))
            for lane, role, names in binder_fixture._TASKS
        ),
    )
    altered = binder_fixture.bind(altered_payload, altered_expectation)

    baseline_plan = _plan(baseline)
    altered_plan = _plan(altered)
    assert (
        baseline_plan["cache_semantic_identity_sha256"]
        != altered_plan["cache_semantic_identity_sha256"]
    )

    _build_all(baseline_payload, baseline_plan)
    _build_all(altered_payload, altered_plan)
    first = reduce_process_v2_chunk_cache(
        baseline_plan, artifact_root=baseline_payload.artifact_root
    )
    second = reduce_process_v2_chunk_cache(
        altered_plan, artifact_root=altered_payload.artifact_root
    )
    assert first["cache_semantic_identity_sha256"] != second["cache_semantic_identity_sha256"]


def test_worker_order_changes_nothing_at_all(tmp_path: Path) -> None:
    payload_a, binding_a = _bound_run(tmp_path / "a")
    payload_b, binding_b = _bound_run(tmp_path / "b")
    plan_a, plan_b = _plan(binding_a), _plan(binding_b)
    assert plan_a["cache_semantic_identity_sha256"] == plan_b["cache_semantic_identity_sha256"]

    _build_all(payload_a, plan_a)
    _build_all(payload_b, plan_b, order=list(reversed(range(len(plan_b["tasks"])))))
    first = reduce_process_v2_chunk_cache(plan_a, artifact_root=payload_a.artifact_root)
    second = reduce_process_v2_chunk_cache(plan_b, artifact_root=payload_b.artifact_root)
    assert canonical_bytes(first) == canonical_bytes(second)


# ---- 4. Interruption, restart and reuse ---------------------------------------


def test_a_crashed_worker_publishes_nothing_and_the_rerun_reuses_exactly(
    tmp_path: Path,
) -> None:
    payload, binding = _bound_run(tmp_path)
    plan = _plan(binding)
    write_process_v2_chunk_cache_plan(plan, artifact_root=payload.artifact_root)
    victim = plan["tasks"][6]

    class _Dies(Exception):
        pass

    crashing = _CountingOpener()

    def crash_after_first_block(path: Path, mode: str = "rb"):
        handle = crashing(path, mode)
        original_read = handle.read
        state = {"blocks": 0}

        def read(size: int = -1) -> bytes:
            state["blocks"] += 1
            if state["blocks"] > 1:
                raise _Dies("container terminated")
            return original_read(size)

        handle.read = read
        return handle

    with pytest.raises(_Dies):
        execute_process_v2_chunk_cache_task(
            plan,
            victim["task_identity_sha256"],
            artifact_root=payload.artifact_root,
            opener=crash_after_first_block,
        )
    source_root = _run_root(payload, plan) / SOURCE_DIRNAME
    assert not (source_root / victim["task_identity_sha256"]).exists()
    # A killed worker's private staging sibling must not block the retry.
    assert completed_process_v2_chunk_cache_task_ids(
        plan, artifact_root=payload.artifact_root
    ) == set()

    first = _build_all(payload, plan)
    assert all(result["reused"] is False for result in first)
    second = _build_all(payload, plan)
    assert all(result["reused"] is True for result in second)
    assert [result["manifest_sha256"] for result in first] == [
        result["manifest_sha256"] for result in second
    ]


def test_the_reducer_refuses_missing_duplicated_and_foreign_source_caches(
    tmp_path: Path,
) -> None:
    payload, binding = _bound_run(tmp_path)
    plan = _plan(binding)
    _build_all(payload, plan)
    source_root = _run_root(payload, plan) / SOURCE_DIRNAME
    outside = tmp_path / "stash"
    outside.mkdir()

    # Missing: one planned source cache moved out of the namespace entirely.
    victim = source_root / plan["tasks"][2]["task_identity_sha256"]
    stash = victim.rename(outside / "missing")
    with pytest.raises(ProcessV2ChunkCacheIncomplete, match="missing 1 of 20"):
        reduce_process_v2_chunk_cache(plan, artifact_root=payload.artifact_root)
    stash.rename(victim)

    # An unexpected object in the source namespace.
    intruder = source_root / "not-a-source"
    intruder.mkdir()
    with pytest.raises(ProcessV2ChunkCacheError, match="unexpected objects"):
        reduce_process_v2_chunk_cache(plan, artifact_root=payload.artifact_root)
    intruder.rmdir()

    # A duplicate of one source published under another source's address.
    donor = source_root / plan["tasks"][0]["task_identity_sha256"]
    collision = source_root / plan["tasks"][1]["task_identity_sha256"]
    saved = collision.rename(outside / "collided")
    collision.mkdir()
    for entry in sorted(donor.iterdir()):
        (collision / entry.name).write_bytes(entry.read_bytes())
    with pytest.raises(ProcessV2ChunkCacheError, match="does not belong to this plan"):
        reduce_process_v2_chunk_cache(plan, artifact_root=payload.artifact_root)
    for entry in sorted(collision.iterdir()):
        entry.unlink()
    collision.rmdir()
    saved.rename(collision)

    # A chunk whose bytes moved after publication.
    chunk = next(
        path
        for path in sorted((source_root / plan["tasks"][3]["task_identity_sha256"]).iterdir())
        if path.name != MANIFEST_FILENAME
    )
    original = chunk.read_bytes()
    chunk.write_bytes(original + b"\x00")
    with pytest.raises(ProcessV2ChunkCacheError, match="physical hash disagrees"):
        reduce_process_v2_chunk_cache(plan, artifact_root=payload.artifact_root)
    chunk.write_bytes(original)

    assert reduce_process_v2_chunk_cache(plan, artifact_root=payload.artifact_root)["source_count"] == 20


def test_the_reducer_requires_the_exact_published_plan_bytes(tmp_path: Path) -> None:
    payload, binding = _bound_run(tmp_path)
    plan = _plan(binding)
    _build_all(payload, plan)
    published = _run_root(payload, plan) / PLAN_FILENAME
    published.write_bytes(published.read_bytes()[:-1])
    with pytest.raises(ProcessV2ChunkCacheError, match="exact published plan bytes"):
        reduce_process_v2_chunk_cache(plan, artifact_root=payload.artifact_root)


def test_an_absent_source_cache_is_incomplete_not_a_smaller_valid_cache(
    tmp_path: Path,
) -> None:
    payload, binding = _bound_run(tmp_path)
    plan = _plan(binding)
    write_process_v2_chunk_cache_plan(plan, artifact_root=payload.artifact_root)
    for task in plan["tasks"][:19]:
        execute_process_v2_chunk_cache_task(
            plan, task["task_identity_sha256"], artifact_root=payload.artifact_root
        )
    with pytest.raises(ProcessV2ChunkCacheIncomplete, match="missing 1 of 20"):
        reduce_process_v2_chunk_cache(plan, artifact_root=payload.artifact_root)


# ---- 5. Integrity refusal publishes nothing -----------------------------------


def test_an_integrity_mismatch_publishes_no_object_and_reports_separately(
    tmp_path: Path,
) -> None:
    payload, binding = _bound_run(tmp_path)
    plan = _plan(binding)
    write_process_v2_chunk_cache_plan(plan, artifact_root=payload.artifact_root)
    task = plan["tasks"][4]
    semantic_dir = (
        payload.artifact_root
        / str(task["v1_task_artifact_path"]).removeprefix("/artifacts/")
        / SEMANTIC_ARTIFACT_DIRNAME
    )
    shard = semantic_dir / SEMANTIC_SHARD_FILENAME
    with gzip.open(shard, "rb") as handle:
        rows = [line for line in handle if line.strip()]
    tampered = json.loads(rows[0])
    tampered["trace_id"] = "tampered"
    rows[0] = canonical_bytes(tampered) + b"\n"
    with shard.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed:
            for line in rows:
                compressed.write(line)

    with pytest.raises(ProcessV2ChunkCacheIntegrityError) as failure:
        execute_process_v2_chunk_cache_task(
            plan, task["task_identity_sha256"], artifact_root=payload.artifact_root
        )
    source_root = _run_root(payload, plan) / SOURCE_DIRNAME
    published = list(source_root.iterdir()) if source_root.exists() else []
    assert published == [], "an integrity refusal must publish nothing under the run namespace"

    diagnostics = tmp_path / "diagnostics"
    report_path = write_chunk_cache_refusal_report(
        diagnostic_root=diagnostics,
        plan=plan,
        stage="build_one_source_cache",
        error=failure.value,
        detail={"task_identity_sha256": task["task_identity_sha256"]},
    )
    assert report_path.parent == diagnostics
    assert diagnostics not in _run_root(payload, plan).parents
    report = json.loads(report_path.read_bytes())
    assert report["cache_physical_identity_sha256"] == plan["cache_physical_identity_sha256"]
    assert report["training_authorized"] is False
    assert report_path.name == f"{report['refusal_sha256']}.json"


def test_the_plan_and_completion_grant_no_authority(tmp_path: Path) -> None:
    payload, binding = _bound_run(tmp_path)
    plan = _plan(binding)
    _build_all(payload, plan)
    completion = reduce_process_v2_chunk_cache(plan, artifact_root=payload.artifact_root)
    for document in (plan, completion):
        granted = [
            key for key, value in document.items() if key.endswith("_authorized") and value is not False
        ]
        assert granted == []
    assert "p50_authorized" not in plan
    assert "bounded_p50_authorized" in plan


def test_the_plan_refuses_an_out_of_range_chunk_size(tmp_path: Path) -> None:
    _payload, binding = _bound_run(tmp_path)
    for bad in (0, -1, MAX_RECORDS_PER_CHUNK + 1):
        with pytest.raises(ProcessV2ChunkCacheError, match="records_per_chunk"):
            _plan(binding, records_per_chunk=bad)
    assert validate_process_v2_chunk_cache_plan(_plan(binding, records_per_chunk=DEFAULT_RECORDS_PER_CHUNK))


def test_a_resealed_plan_that_moves_its_identity_is_refused(tmp_path: Path) -> None:
    _payload, binding = _bound_run(tmp_path)
    plan = _plan(binding)
    body = {key: item for key, item in plan.items() if key != "plan_sha256"}
    body["records_per_chunk"] = int(body["records_per_chunk"]) + 1
    resealed = {**body, "plan_sha256": canonical_sha256(body)}
    with pytest.raises(ProcessV2ChunkCacheError, match="identity or path disagrees"):
        validate_process_v2_chunk_cache_plan(resealed)


# ---- 6. A non-production local timing note ------------------------------------


def test_local_synthetic_throughput_is_recorded_not_asserted(tmp_path: Path) -> None:
    """A local synthetic timing, deliberately without a threshold.

    Timings on a laptop are not production throughput evidence, so this records
    a number for the report and asserts only correctness.
    """

    directory = _synthetic_semantic_dir(tmp_path / "semantic", entries=20_000)
    started = time.perf_counter()
    scan = _scan(directory, records_per_chunk=DEFAULT_RECORDS_PER_CHUNK, opener=open)
    elapsed = time.perf_counter() - started
    assert scan.entries == 20_000
    assert elapsed > 0
    print(
        json.dumps(
            {
                "non_production_local_synthetic": True,
                "entries": scan.entries,
                "compressed_bytes": scan.compressed_bytes,
                "decompressed_bytes": scan.decompressed_bytes,
                "seconds": round(elapsed, 4),
                "records_per_second": round(scan.entries / elapsed),
            }
        )
    )
