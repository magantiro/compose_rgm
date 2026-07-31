"""Canonical-successor fibers survive storage without storing model scores."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from compose_v4.data.packed_trace_store import PackedTraceAddress
from compose_v4.data.successor_fiber_cache import (
    SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION,
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheError,
    SuccessorFiberCacheLimits,
    SuccessorFiberCacheProvenance,
    SuccessorFiberCacheRecord,
    deserialize_successor_fiber_cache,
    read_successor_fiber_cache,
    serialize_successor_fiber_cache,
    write_successor_fiber_cache,
)
from compose_v4.experiments.factorized_successor_training import (
    StateProductiveSupport,
    TeacherSuccessorAlias,
    TeacherSuccessorFiber,
)

_SHARD_SHA256 = "d" * 64
_SOURCE_STATE_SHA256 = "1" * 64
_TARGET_STATE_SHA256 = "2" * 64


def _alias(
    family: str,
    table: str,
    coordinate: tuple[int, ...],
) -> TeacherSuccessorAlias:
    return TeacherSuccessorAlias(family, table, coordinate)


@pytest.fixture
def provenance() -> SuccessorFiberCacheProvenance:
    return SuccessorFiberCacheProvenance(
        operator_registry_hash="a" * 16,
        capability_hash="b" * 16,
        support_signature_sha256="3" * 64,
        canonicalization_version="c" * 16,
        canonicalizer_contract_sha256="4" * 64,
        packed_corpus_schema="compose.data.packed_trace",
        packed_corpus_schema_version=1,
        packed_shard_content_sha256=_SHARD_SHA256,
        packed_manifest_sha256="5" * 64,
        packed_provenance_overlay_sha256="6" * 64,
        unified_packed_manifest_sha256="7" * 64,
        representability_overlay_sha256="8" * 64,
        coordinate_schema_version=1,
        tensorization_implementation_hash="9" * 16,
        fiber_compiler_implementation_hash="e" * 64,
    )


def _one_step_records(
    *,
    trace_id: str = "trace-7",
    entry_index: int = 7,
) -> tuple[SuccessorFiberCacheRecord, ...]:
    teacher_aliases = (
        _alias("atom_insert", "atom_insert_bond_order", (0, 1, 0)),
        _alias("atom_insert", "atom_insert_bond_order", (2, 1, 0)),
    )
    source_virtual = (
        _alias("atom_restate", "atom_restate", (0, 3)),
    )
    terminal_virtual = (
        _alias("bond_reorder", "bond_reorder", (0, 1, 1)),
    )
    source_support = StateProductiveSupport(
        source_key="[CH4]",
        source_state_sha256=_SOURCE_STATE_SHA256,
        virtual_aliases=source_virtual,
    )
    fiber = TeacherSuccessorFiber(
        source_key="[CH4]",
        target_key="CC",
        target_state_sha256=_TARGET_STATE_SHA256,
        aliases=teacher_aliases,
        state_support=source_support,
    )
    return (
        SuccessorFiberCacheRecord(
            address=SuccessorFiberCacheAddress(
                packed_shard_content_sha256=_SHARD_SHA256,
                packed_shard_name="shard_0000.jsonl.gz",
                entry_index=entry_index,
                layer="mmp",
                partition="train",
                trace_id=trace_id,
                trace_source_key="[CH4]",
                trace_target_key="CC",
                progress_index=0,
                path_length=1,
            ),
            state_support=source_support,
            teacher_fiber=fiber,
        ),
        SuccessorFiberCacheRecord(
            address=SuccessorFiberCacheAddress(
                packed_shard_content_sha256=_SHARD_SHA256,
                packed_shard_name="shard_0000.jsonl.gz",
                entry_index=entry_index,
                layer="mmp",
                partition="train",
                trace_id=trace_id,
                trace_source_key="[CH4]",
                trace_target_key="CC",
                progress_index=1,
                path_length=1,
            ),
            state_support=StateProductiveSupport(
                source_key="CC",
                source_state_sha256=_TARGET_STATE_SHA256,
                virtual_aliases=terminal_virtual,
            ),
            teacher_fiber=None,
        ),
    )


def _rehash(envelope: dict) -> None:
    payload = {
        key: envelope[key]
        for key in ("schema", "schema_version", "provenance", "records")
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    envelope["content_sha256"] = hashlib.sha256(encoded).hexdigest()


def test_exact_round_trip_includes_virtual_and_terminal_support(
    tmp_path,
    provenance,
):
    records = _one_step_records()
    path = tmp_path / "fibers.json"
    written = write_successor_fiber_cache(
        path,
        records,
        provenance=provenance,
    )
    loaded = read_successor_fiber_cache(
        path,
        expected_provenance=provenance,
        expected_content_sha256=written.content_sha256,
    )

    assert loaded == written
    assert loaded.records == records
    assert loaded.records[0].teacher_fiber.aliases == records[
        0
    ].teacher_fiber.aliases
    assert loaded.records[0].state_support.virtual_aliases == records[
        0
    ].state_support.virtual_aliases
    terminal = loaded.record_at(
        layer="mmp",
        partition="train",
        trace_id="trace-7",
        progress_index=1,
    )
    assert terminal.address.is_terminal
    assert terminal.teacher_fiber is None
    assert terminal.state_support.virtual_aliases
    assert loaded.records_for_trace(
        layer="mmp",
        partition="train",
        trace_id="trace-7",
    ) == records
    packed_address = PackedTraceAddress(
        packed_shard_content_sha256=_SHARD_SHA256,
        packed_shard_name="shard_0000.jsonl.gz",
        entry_index=7,
        trace_id="trace-7",
        layer="mmp",
        partition="train",
        source_key="[CH4]",
        target_key="CC",
        path_length=1,
    )
    assert loaded.require_packed_address(
        packed_address,
        progress_index=0,
    ) == records[0]
    with pytest.raises(KeyError, match="exact packed row"):
        loaded.require_packed_address(
            replace(
                packed_address,
                packed_shard_content_sha256="f" * 64,
            ),
            progress_index=0,
        )


def test_serialization_is_deterministic_under_record_input_order(provenance):
    first = _one_step_records(trace_id="first", entry_index=7)
    second = _one_step_records(trace_id="second", entry_index=8)
    encoded_a, cache_a = serialize_successor_fiber_cache(
        (*second, *first),
        provenance=provenance,
    )
    encoded_b, cache_b = serialize_successor_fiber_cache(
        (*first, *second),
        provenance=provenance,
    )
    assert encoded_a == encoded_b
    assert cache_a.content_sha256 == cache_b.content_sha256


def test_writer_never_overwrites_a_frozen_cache_path(tmp_path, provenance):
    path = tmp_path / "frozen.json"
    records = _one_step_records()
    first = write_successor_fiber_cache(
        path,
        records,
        provenance=provenance,
    )
    assert write_successor_fiber_cache(
        path,
        records,
        provenance=provenance,
    ) == first
    original_bytes = path.read_bytes()

    with pytest.raises(FileExistsError, match="different content"):
        write_successor_fiber_cache(
            path,
            _one_step_records(trace_id="different"),
            provenance=provenance,
        )
    assert path.read_bytes() == original_bytes


def test_writer_falls_back_when_volume_rejects_hard_links(
    tmp_path,
    provenance,
    monkeypatch,
):
    path = tmp_path / "modal-volume.json"

    def reject_hard_link(*args, **kwargs):
        raise PermissionError(1, "operation not permitted")

    monkeypatch.setattr("os.link", reject_hard_link)
    first = write_successor_fiber_cache(
        path,
        _one_step_records(),
        provenance=provenance,
    )

    assert path.is_file()
    assert write_successor_fiber_cache(
        path,
        _one_step_records(),
        provenance=provenance,
    ) == first
    with pytest.raises(FileExistsError, match="different content"):
        write_successor_fiber_cache(
            path,
            _one_step_records(trace_id="collision"),
            provenance=provenance,
        )


def test_zero_step_trace_serializes_terminal_support(provenance):
    record = SuccessorFiberCacheRecord(
        address=SuccessorFiberCacheAddress(
            packed_shard_content_sha256=_SHARD_SHA256,
            packed_shard_name="shard_0001.jsonl.gz",
            entry_index=0,
            layer="corruption",
            partition="validation",
            trace_id="already-done",
            trace_source_key="N",
            trace_target_key="N",
            progress_index=0,
            path_length=0,
        ),
        state_support=StateProductiveSupport(
            source_key="N",
            source_state_sha256="3" * 64,
            virtual_aliases=(),
        ),
        teacher_fiber=None,
    )
    encoded, expected = serialize_successor_fiber_cache(
        (record,),
        provenance=provenance,
    )
    observed = deserialize_successor_fiber_cache(
        encoded,
        expected_provenance=provenance,
        expected_content_sha256=expected.content_sha256,
    )
    assert observed == expected


def test_artifact_contains_no_weight_probability_rate_or_logit(provenance):
    encoded, _cache = serialize_successor_fiber_cache(
        _one_step_records(),
        provenance=provenance,
    )
    root = json.loads(encoded)

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    forbidden = {
        "weight",
        "weights",
        "probability",
        "probabilities",
        "rate",
        "rates",
        "logit",
        "logits",
    }
    assert forbidden.isdisjoint(keys(root))


def test_content_tampering_is_refused(provenance):
    encoded, cache = serialize_successor_fiber_cache(
        _one_step_records(),
        provenance=provenance,
    )
    envelope = json.loads(encoded)
    envelope["records"][0]["target_key"] = "CCC"
    tampered = json.dumps(envelope).encode()
    with pytest.raises(SuccessorFiberCacheError, match="content hash mismatch"):
        deserialize_successor_fiber_cache(
            tampered,
            expected_provenance=provenance,
            expected_content_sha256=cache.content_sha256,
        )


def test_unknown_probability_field_is_refused_even_if_rehashed(provenance):
    encoded, _ = serialize_successor_fiber_cache(
        _one_step_records(),
        provenance=provenance,
    )
    envelope = json.loads(encoded)
    envelope["records"][0]["probability"] = 0.9
    _rehash(envelope)
    tampered = json.dumps(envelope).encode()
    with pytest.raises(SuccessorFiberCacheError, match="unexpected=.*probability"):
        deserialize_successor_fiber_cache(
            tampered,
            expected_provenance=provenance,
            expected_content_sha256=envelope["content_sha256"],
        )


def test_provenance_mismatch_is_refused(provenance):
    encoded, cache = serialize_successor_fiber_cache(
        _one_step_records(),
        provenance=provenance,
    )
    drifted = SuccessorFiberCacheProvenance(
        **{
            **provenance.__dict__,
            "capability_hash": "f" * 16,
        }
    )
    with pytest.raises(
        SuccessorFiberCacheError,
        match="provenance mismatch.*capability_hash",
    ):
        deserialize_successor_fiber_cache(
            encoded,
            expected_provenance=drifted,
            expected_content_sha256=cache.content_sha256,
        )


def test_schema_version_mismatch_is_refused(provenance):
    encoded, _ = serialize_successor_fiber_cache(
        _one_step_records(),
        provenance=provenance,
    )
    envelope = json.loads(encoded)
    envelope["schema_version"] = SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION + 1
    _rehash(envelope)
    with pytest.raises(SuccessorFiberCacheError, match="schema version"):
        deserialize_successor_fiber_cache(
            json.dumps(envelope).encode(),
            expected_provenance=provenance,
            expected_content_sha256=envelope["content_sha256"],
        )


def test_missing_progress_and_target_chain_break_are_refused(provenance):
    records = _one_step_records()
    with pytest.raises(SuccessorFiberCacheError, match="every progress row"):
        serialize_successor_fiber_cache(
            records[:1],
            provenance=provenance,
        )

    wrong_terminal = SuccessorFiberCacheRecord(
        address=records[1].address,
        state_support=StateProductiveSupport(
            source_key="CC",
            source_state_sha256="4" * 64,
            virtual_aliases=(),
        ),
        teacher_fiber=None,
    )
    with pytest.raises(SuccessorFiberCacheError, match="chain breaks"):
        serialize_successor_fiber_cache(
            (records[0], wrong_terminal),
            provenance=provenance,
        )


def test_alias_order_and_resource_bounds_are_enforced(provenance):
    records = _one_step_records()
    fiber = records[0].teacher_fiber
    reversed_fiber = TeacherSuccessorFiber(
        source_key=fiber.source_key,
        target_key=fiber.target_key,
        target_state_sha256=fiber.target_state_sha256,
        aliases=tuple(reversed(fiber.aliases)),
        state_support=fiber.state_support,
    )
    reversed_record = SuccessorFiberCacheRecord(
        address=records[0].address,
        state_support=fiber.state_support,
        teacher_fiber=reversed_fiber,
    )
    with pytest.raises(SuccessorFiberCacheError, match="lexicographic order"):
        serialize_successor_fiber_cache(
            (reversed_record, records[1]),
            provenance=provenance,
        )

    limits = SuccessorFiberCacheLimits(max_records=1)
    with pytest.raises(SuccessorFiberCacheError, match="record count"):
        serialize_successor_fiber_cache(
            records,
            provenance=provenance,
            limits=limits,
        )
