"""Indexed successor fibers are exact, immutable, and worker-safe."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import compose_v4.data.indexed_successor_fiber_cache as indexed_cache_module
from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.data.indexed_successor_fiber_cache import (
    IndexedSuccessorFiberCacheError,
    assert_indexed_cache_is_pickle_safe,
    open_indexed_successor_fiber_cache,
    write_indexed_successor_fiber_cache,
)
from compose_v4.data.packed_trace_store import PackedTraceAddress
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheLimits,
    SuccessorFiberCacheProvenance,
    SuccessorFiberCacheRecord,
)
from compose_v4.experiments.factorized_successor_training import (
    StateProductiveSupport,
    TeacherSuccessorAlias,
    TeacherSuccessorFiber,
)

_PACKED_SHA256 = "a" * 64
_SHARD_NAME = "shard_0000.jsonl.gz"


def _state(*, target: bool) -> MolecularGraph:
    atom_types = np.asarray(
        [1, 1 if target else 0, 0, 0],
        dtype=np.int32,
    )
    hydrogens = np.asarray(
        [3 if target else 4, 3 if target else 0, 0, 0],
        dtype=np.int32,
    )
    bonds = np.zeros((4, 4), dtype=np.int32)
    if target:
        bonds[0, 1] = bonds[1, 0] = 1
    return MolecularGraph(
        atom_types=atom_types,
        formal_charges=np.zeros(4, dtype=np.int32),
        implicit_h_counts=hydrogens,
        bonds=bonds,
    )


@pytest.fixture
def provenance() -> SuccessorFiberCacheProvenance:
    return SuccessorFiberCacheProvenance(
        operator_registry_hash="operator-v1",
        capability_hash="capability-v1",
        support_signature_sha256="1" * 64,
        canonicalization_version="canonical-state-key-v1",
        canonicalizer_contract_sha256="2" * 64,
        packed_corpus_schema="compose.data.packed_trace",
        packed_corpus_schema_version=1,
        packed_shard_content_sha256=_PACKED_SHA256,
        packed_manifest_sha256="3" * 64,
        packed_provenance_overlay_sha256="4" * 64,
        unified_packed_manifest_sha256="5" * 64,
        representability_overlay_sha256="6" * 64,
        coordinate_schema_version=1,
        tensorization_implementation_hash="tensorization-v1",
        fiber_compiler_implementation_hash="7" * 64,
    )


def _records(
    *,
    entry_index: int = 0,
    trace_id: str = "trace-0",
) -> tuple[SuccessorFiberCacheRecord, ...]:
    source = _state(target=False)
    target = _state(target=True)
    source_support = StateProductiveSupport(
        source_key="C",
        source_state_sha256=persistent_slot_state_sha256(source),
        virtual_aliases=(
            TeacherSuccessorAlias(
                "atom_restate",
                "atom_restate",
                (0, 0),
            ),
        ),
    )
    teacher = TeacherSuccessorFiber(
        source_key="C",
        target_key="CC",
        target_state_sha256=persistent_slot_state_sha256(target),
        aliases=(
            TeacherSuccessorAlias(
                "atom_insert",
                "grow_connected",
                (0, 0, 0),
            ),
            TeacherSuccessorAlias(
                "atom_insert",
                "grow_connected",
                (1, 0, 0),
            ),
        ),
        state_support=source_support,
    )

    def address(progress_index: int) -> SuccessorFiberCacheAddress:
        return SuccessorFiberCacheAddress(
            packed_shard_content_sha256=_PACKED_SHA256,
            packed_shard_name=_SHARD_NAME,
            entry_index=entry_index,
            layer="mmp_analogue",
            partition="train",
            trace_id=trace_id,
            trace_source_key="C",
            trace_target_key="CC",
            progress_index=progress_index,
            path_length=1,
        )

    return (
        SuccessorFiberCacheRecord(
            address=address(0),
            state_support=source_support,
            teacher_fiber=teacher,
        ),
        SuccessorFiberCacheRecord(
            address=address(1),
            state_support=StateProductiveSupport(
                source_key="CC",
                source_state_sha256=persistent_slot_state_sha256(target),
                virtual_aliases=(),
            ),
            teacher_fiber=None,
        ),
    )


def _packed_address(
    *,
    entry_index: int = 0,
    trace_id: str = "trace-0",
) -> PackedTraceAddress:
    return PackedTraceAddress(
        packed_shard_content_sha256=_PACKED_SHA256,
        packed_shard_name=_SHARD_NAME,
        entry_index=entry_index,
        trace_id=trace_id,
        layer="mmp_analogue",
        partition="train",
        source_key="C",
        target_key="CC",
        path_length=1,
    )


def _open(path: Path, metadata, provenance):
    return open_indexed_successor_fiber_cache(
        path,
        expected_provenance=provenance,
        expected_content_sha256=metadata.content_sha256,
        expected_file_sha256=metadata.file_sha256,
        expected_file_bytes=metadata.file_bytes,
        decoded_row_cache_size=1,
    )


def test_indexed_round_trip_is_exact_and_deterministic(
    tmp_path,
    provenance,
) -> None:
    first = _records(entry_index=0, trace_id="trace-0")
    second = _records(entry_index=1, trace_id="trace-1")
    path_a = tmp_path / "a.sqlite"
    path_b = tmp_path / "b.sqlite"
    metadata_a = write_indexed_successor_fiber_cache(
        path_a,
        (*second, *first),
        provenance=provenance,
    )
    metadata_b = write_indexed_successor_fiber_cache(
        path_b,
        (*first, *second),
        provenance=provenance,
    )

    assert path_a.read_bytes() == path_b.read_bytes()
    assert metadata_a == metadata_b
    assert metadata_a.record_count == 4
    assert metadata_a.active_trace_count == 2
    assert metadata_a.jump_record_count == 2
    assert metadata_a.terminal_record_count == 2
    assert metadata_a.teacher_alias_count == 4
    assert metadata_a.virtual_alias_count == 2

    cache = _open(path_a, metadata_a, provenance)
    assert cache.require_packed_address(
        _packed_address(),
        progress_index=0,
    ) == first[0]
    assert cache.require_packed_address(
        _packed_address(entry_index=1, trace_id="trace-1"),
        progress_index=1,
    ) == second[1]
    assert_indexed_cache_is_pickle_safe(cache)
    restored = pickle.loads(pickle.dumps(cache))
    assert restored.require_packed_address(
        _packed_address(),
        progress_index=0,
    ) == first[0]
    assert len(restored._decoded) == 1

    cache.require_packed_address(_packed_address(), progress_index=0)
    cache.require_packed_address(
        _packed_address(entry_index=1, trace_id="trace-1"),
        progress_index=0,
    )
    assert len(cache._decoded) == 1


def test_parent_validated_receipt_reopens_without_full_worker_scan(
    tmp_path,
    provenance,
    monkeypatch,
) -> None:
    path = tmp_path / "cache.sqlite"
    metadata = write_indexed_successor_fiber_cache(
        path,
        _records(),
        provenance=provenance,
    )
    parent = _open(path, metadata, provenance)
    parent.require_entry_census(
        packed_entry_count=1,
        excluded_entry_indices=(),
    )
    receipt = pickle.loads(
        pickle.dumps(parent.parent_validated_open_receipt)
    )
    schema_modes: list[bool] = []
    original_schema_validation = indexed_cache_module._validate_sqlite_schema

    def record_schema_mode(connection, *, run_integrity_check=True):
        schema_modes.append(run_integrity_check)
        return original_schema_validation(
            connection,
            run_integrity_check=run_integrity_check,
        )

    def forbid_full_scan(*args, **kwargs):
        raise AssertionError("worker attempted a full cache scan")

    monkeypatch.setattr(
        indexed_cache_module,
        "_validate_sqlite_schema",
        record_schema_mode,
    )
    monkeypatch.setattr(indexed_cache_module, "_sha256_file", forbid_full_scan)
    monkeypatch.setattr(
        indexed_cache_module,
        "_audit_indexed_records",
        forbid_full_scan,
    )
    worker = open_indexed_successor_fiber_cache(
        path,
        expected_provenance=provenance,
        expected_content_sha256=metadata.content_sha256,
        expected_file_sha256=metadata.file_sha256,
        expected_file_bytes=metadata.file_bytes,
        parent_validated_open_receipt=receipt,
    )

    assert schema_modes == [False]
    worker.require_entry_census(
        packed_entry_count=1,
        excluded_entry_indices=(),
    )
    assert worker.require_packed_address(
        _packed_address(),
        progress_index=0,
    ) == _records()[0]


def test_parent_validated_receipt_fails_on_identity_or_contract_drift(
    tmp_path,
    provenance,
    monkeypatch,
) -> None:
    path = tmp_path / "cache.sqlite"
    metadata = write_indexed_successor_fiber_cache(
        path,
        _records(),
        provenance=provenance,
    )
    parent = _open(path, metadata, provenance)
    receipt = parent.parent_validated_open_receipt

    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="worker open expectations",
    ):
        open_indexed_successor_fiber_cache(
            path,
            expected_provenance=provenance,
            expected_content_sha256="f" * 64,
            expected_file_sha256=metadata.file_sha256,
            expected_file_bytes=metadata.file_bytes,
            parent_validated_open_receipt=receipt,
        )

    def forbid_fallback(*args, **kwargs):
        raise AssertionError("receipt mismatch fell back to full validation")

    monkeypatch.setattr(
        indexed_cache_module,
        "_sha256_file",
        forbid_fallback,
    )
    path.touch()
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="identity changed",
    ):
        open_indexed_successor_fiber_cache(
            path,
            expected_provenance=provenance,
            expected_content_sha256=metadata.content_sha256,
            expected_file_sha256=metadata.file_sha256,
            expected_file_bytes=metadata.file_bytes,
            parent_validated_open_receipt=receipt,
        )


def test_indexed_cache_refuses_identity_drift_and_missing_rows(
    tmp_path,
    provenance,
) -> None:
    path = tmp_path / "cache.sqlite"
    metadata = write_indexed_successor_fiber_cache(
        path,
        _records(),
        provenance=provenance,
    )

    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="byte SHA-256",
    ):
        open_indexed_successor_fiber_cache(
            path,
            expected_provenance=provenance,
            expected_content_sha256=metadata.content_sha256,
            expected_file_sha256="f" * 64,
            expected_file_bytes=metadata.file_bytes,
        )
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="provenance",
    ):
        open_indexed_successor_fiber_cache(
            path,
            expected_provenance=replace(
                provenance,
                capability_hash="another-capability",
            ),
            expected_content_sha256=metadata.content_sha256,
            expected_file_sha256=metadata.file_sha256,
            expected_file_bytes=metadata.file_bytes,
        )

    cache = _open(path, metadata, provenance)
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="no exact",
    ):
        cache.require_packed_address(
            _packed_address(entry_index=7, trace_id="missing"),
            progress_index=0,
        )
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="full packed address",
    ):
        cache.require_packed_address(
            replace(_packed_address(), trace_id="wrong-trace"),
            progress_index=0,
        )
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="full packed address",
    ):
        cache.require_packed_address(
            replace(_packed_address(), source_key="wrong-source"),
            progress_index=0,
        )


def test_indexed_cache_validates_exact_source_state_and_file_identity(
    tmp_path,
    provenance,
) -> None:
    path = tmp_path / "cache.sqlite"
    metadata = write_indexed_successor_fiber_cache(
        path,
        _records(),
        provenance=provenance,
    )
    cache = _open(path, metadata, provenance)
    address = _packed_address()

    assert cache.require(
        address,
        progress_index=0,
        source_state=_state(target=False),
    ) == _records()[0]
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="exact state",
    ):
        cache.require(
            address,
            progress_index=0,
            source_state=_state(target=True),
        )

    fresh = _open(path, metadata, provenance)
    fresh.require_packed_address(address, progress_index=0)
    path.touch()
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="identity changed",
    ):
        fresh.require_packed_address(address, progress_index=0)


def test_indexed_cache_detects_in_place_mutation_with_restored_mtime(
    tmp_path,
    provenance,
) -> None:
    path = tmp_path / "cache.sqlite"
    metadata = write_indexed_successor_fiber_cache(
        path,
        _records(),
        provenance=provenance,
    )
    cache = _open(path, metadata, provenance)
    address = _packed_address()
    cache.require_packed_address(address, progress_index=0)
    before = path.stat()
    with sqlite3.connect(path) as connection:
        encoded = connection.execute(
            """
            SELECT aliases FROM records
            WHERE entry_index = 0 AND progress_index = 0
            """
        ).fetchone()[0]
        mutated = encoded.replace(b"[0,0,0]", b"[9,0,0]", 1)
        assert len(mutated) == len(encoded)
        connection.execute(
            """
            UPDATE records SET aliases = ?
            WHERE entry_index = 0 AND progress_index = 0
            """,
            (mutated,),
        )
        connection.commit()
    os.utime(
        path,
        ns=(before.st_atime_ns, before.st_mtime_ns),
    )
    after = path.stat()
    assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    assert after.st_ctime_ns != before.st_ctime_ns
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="identity changed",
    ):
        cache.require_packed_address(address, progress_index=0)


def test_indexed_open_recomputes_semantic_hash_and_census(
    tmp_path,
    provenance,
) -> None:
    path = tmp_path / "cache.sqlite"
    metadata = write_indexed_successor_fiber_cache(
        path,
        _records(),
        provenance=provenance,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            UPDATE records
            SET aliases = ?
            WHERE entry_index = 0 AND progress_index = 0
            """,
            (b'[["atom_insert","grow_connected",[9,0,0]]]',),
        )
        connection.commit()
    mutated_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="recomputed semantic hash",
    ):
        open_indexed_successor_fiber_cache(
            path,
            expected_provenance=provenance,
            expected_content_sha256=metadata.content_sha256,
            expected_file_sha256=mutated_sha256,
            expected_file_bytes=path.stat().st_size,
        )

    census_path = tmp_path / "census.sqlite"
    census_metadata = write_indexed_successor_fiber_cache(
        census_path,
        _records(),
        provenance=provenance,
    )
    with sqlite3.connect(census_path) as connection:
        payload = connection.execute(
            "SELECT payload FROM metadata WHERE singleton = 1"
        ).fetchone()[0]
        decoded = json.loads(payload)
        decoded["teacher_alias_count"] += 1
        connection.execute(
            "UPDATE metadata SET payload = ? WHERE singleton = 1",
            (
                json.dumps(
                    decoded,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                .encode(),
            ),
        )
        connection.commit()
    census_sha256 = hashlib.sha256(census_path.read_bytes()).hexdigest()
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="recomputed census",
    ):
        open_indexed_successor_fiber_cache(
            census_path,
            expected_provenance=provenance,
            expected_content_sha256=census_metadata.content_sha256,
            expected_file_sha256=census_sha256,
            expected_file_bytes=census_path.stat().st_size,
        )

    chain_path = tmp_path / "chain.sqlite"
    chain_metadata = write_indexed_successor_fiber_cache(
        chain_path,
        _records(),
        provenance=provenance,
    )
    with sqlite3.connect(chain_path) as connection:
        connection.execute(
            """
            UPDATE records SET path_length = 2
            WHERE entry_index = 0 AND progress_index = 0
            """
        )
        connection.commit()
    chain_sha256 = hashlib.sha256(chain_path.read_bytes()).hexdigest()
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="address/progress chain",
    ):
        open_indexed_successor_fiber_cache(
            chain_path,
            expected_provenance=provenance,
            expected_content_sha256=chain_metadata.content_sha256,
            expected_file_sha256=chain_sha256,
            expected_file_bytes=chain_path.stat().st_size,
        )


def test_indexed_open_enforces_schema_limits_and_hash_to_open_identity(
    tmp_path,
    provenance,
    monkeypatch,
) -> None:
    path = tmp_path / "cache.sqlite"
    metadata = write_indexed_successor_fiber_cache(
        path,
        _records(),
        provenance=provenance,
    )
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="resource limits",
    ):
        open_indexed_successor_fiber_cache(
            path,
            expected_provenance=provenance,
            expected_content_sha256=metadata.content_sha256,
            expected_file_sha256=metadata.file_sha256,
            expected_file_bytes=metadata.file_bytes,
            limits=SuccessorFiberCacheLimits(max_records=1),
        )
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="rank",
    ):
        open_indexed_successor_fiber_cache(
            path,
            expected_provenance=provenance,
            expected_content_sha256=metadata.content_sha256,
            expected_file_sha256=metadata.file_sha256,
            expected_file_bytes=metadata.file_bytes,
            limits=SuccessorFiberCacheLimits(max_coordinate_rank=2),
        )

    extra_schema = tmp_path / "extra-schema.sqlite"
    shutil.copy2(path, extra_schema)
    with sqlite3.connect(extra_schema) as connection:
        connection.execute(
            "CREATE INDEX unexpected_index ON records(trace_id)"
        )
        connection.commit()
    extra_sha256 = hashlib.sha256(extra_schema.read_bytes()).hexdigest()
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="unexpected schema objects",
    ):
        open_indexed_successor_fiber_cache(
            extra_schema,
            expected_provenance=provenance,
            expected_content_sha256=metadata.content_sha256,
            expected_file_sha256=extra_sha256,
            expected_file_bytes=extra_schema.stat().st_size,
        )

    replacement = tmp_path / "replacement.sqlite"
    shutil.copy2(path, replacement)
    original_hash = indexed_cache_module._sha256_file

    def replace_after_hash(source):
        digest = original_hash(source)
        os.replace(replacement, source)
        return digest

    monkeypatch.setattr(
        indexed_cache_module,
        "_sha256_file",
        replace_after_hash,
    )
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="identity changed",
    ):
        open_indexed_successor_fiber_cache(
            path,
            expected_provenance=provenance,
            expected_content_sha256=metadata.content_sha256,
            expected_file_sha256=metadata.file_sha256,
            expected_file_bytes=metadata.file_bytes,
        )


def test_indexed_cache_requires_exact_packed_entry_census(
    tmp_path,
    provenance,
) -> None:
    path = tmp_path / "cache.sqlite"
    metadata = write_indexed_successor_fiber_cache(
        path,
        _records(),
        provenance=provenance,
    )
    cache = _open(path, metadata, provenance)
    cache.require_entry_census(
        packed_entry_count=2,
        excluded_entry_indices=(1,),
    )
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="omits or invents",
    ):
        cache.require_entry_census(
            packed_entry_count=2,
            excluded_entry_indices=(0,),
        )
    with pytest.raises(
        IndexedSuccessorFiberCacheError,
        match="active-trace count",
    ):
        cache.require_entry_census(packed_entry_count=2)


def test_indexed_writer_never_overwrites_frozen_bytes(
    tmp_path,
    provenance,
) -> None:
    path = tmp_path / "cache.sqlite"
    records = _records()
    first = write_indexed_successor_fiber_cache(
        path,
        records,
        provenance=provenance,
    )
    assert write_indexed_successor_fiber_cache(
        path,
        records,
        provenance=provenance,
    ) == first
    original = path.read_bytes()

    with pytest.raises(FileExistsError, match="different bytes"):
        write_indexed_successor_fiber_cache(
            path,
            _records(entry_index=1, trace_id="other"),
            provenance=provenance,
        )
    assert path.read_bytes() == original
