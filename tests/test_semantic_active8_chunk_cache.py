"""Determinism and fail-closed tests for the semantic Active8 chunk cache."""

from __future__ import annotations

import gzip
import hashlib
import io
import json

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.semantic_active8_chunk_cache import (
    SemanticActive8ChunkCacheError,
    build_semantic_active8_chunk_cache,
    load_semantic_active8_chunk_cache,
    read_semantic_active8_chunk,
)
from compose_v4.data.semantic_packed_trace_store import (
    MANIFEST_FILENAME,
    SHARD_FILENAME,
    SemanticPackedStoreError,
    read_semantic_packed_artifact,
    write_semantic_packed_artifact,
)
from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import CycleCloseEdge
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard_v3 import encode_semantic_trace_record

LANE = "reversible_synthetic_walk"
SPLIT = "development"


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


def _record(*, trace_id: str) -> dict[str, object]:
    source = pad_molecular_graph(smiles_to_molecular_graph("CCCCCC"), 16)
    step = RewriteStep("cycle_close", CycleCloseEdge(0, 5, 1))
    target = editing_v2_semantic_rewrite_system().apply(
        source,
        step.rule_name,
        step.action,
    )
    return encode_semantic_trace_record(
        RewriteTrace(source, target, (step,), {"source_science": "unchanged"}),
        trace_id=trace_id,
        data_lane=LANE,
        split=SPLIT,
        source_address={"source_address_sha256": "6" * 64},
        lineage={"semantic_migration": "fixture"},
    )


def _source_binding(entries: int) -> dict[str, object]:
    return {
        "source_shard_name": "legacy_0000.jsonl.gz",
        "source_shard_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "source_overlay_sha256": "3" * 64,
        "source_unified_manifest_sha256": "4" * 64,
        "source_entry_count": entries,
    }


def _write_source(artifact, *, entries: int):
    records = [
        _record(trace_id=f"semantic-fixture-{index}") for index in range(entries)
    ]
    write_semantic_packed_artifact(
        artifact,
        records,
        data_lane=LANE,
        split=SPLIT,
        source_binding=_source_binding(entries),
        decision_binding={
            "decision_ledger_sha256": "5" * 64,
            "source_count": entries,
            "admitted_count": entries,
            "rejected_count": 0,
        },
    )
    return (
        _sha256(artifact / SHARD_FILENAME),
        _sha256(artifact / MANIFEST_FILENAME),
    )


def _build(artifact, output, *, entries: int, target: int = 2):
    shard_sha256, manifest_sha256 = _write_source(artifact, entries=entries)
    completion = build_semantic_active8_chunk_cache(
        artifact,
        expected_shard_sha256=shard_sha256,
        expected_manifest_sha256=manifest_sha256,
        expected_source_binding=_source_binding(entries),
        output_root=output,
        target_rows_per_chunk=target,
    )
    return completion, shard_sha256, manifest_sha256


def test_chunk_cache_is_deterministic_and_recovers_original_addresses(tmp_path) -> None:
    first_artifact = tmp_path / "first-source"
    second_artifact = tmp_path / "second-source"
    first_output = tmp_path / "first-cache"
    second_output = tmp_path / "second-cache"
    first, shard_sha256, manifest_sha256 = _build(
        first_artifact,
        first_output,
        entries=5,
    )
    second, second_shard, second_manifest = _build(
        second_artifact,
        second_output,
        entries=5,
    )
    assert second_shard == shard_sha256
    assert second_manifest == manifest_sha256
    assert second == first
    assert [
        (chunk["entry_start"], chunk["entry_stop"])
        for chunk in first["chunk_inventory"]
    ] == [(0, 2), (2, 4), (4, 5)]

    source_rows = gzip.decompress((first_artifact / SHARD_FILENAME).read_bytes())
    cached_rows = b"".join(
        gzip.decompress((first_output / chunk["chunk_object_path"]).read_bytes())
        for chunk in first["chunk_inventory"]
    )
    assert cached_rows == source_rows

    full = list(
        read_semantic_packed_artifact(
            first_artifact,
            expected_shard_sha256=shard_sha256,
            expected_manifest_sha256=manifest_sha256,
            sentinel_replay_entries=0,
        )
    )
    chunked = [
        row
        for chunk in first["chunk_inventory"]
        for row in read_semantic_active8_chunk(
            first_output,
            receipt_object_path=chunk["receipt_object_path"],
            expected_receipt_file_sha256=chunk["receipt_file_sha256"],
            expected_source_shard_sha256=shard_sha256,
            expected_source_manifest_sha256=manifest_sha256,
            sentinel_replay_entries=0,
        )
    ]
    assert [row.address for row in chunked] == [row.address for row in full]
    assert [row.address.entry_index for row in chunked] == list(range(5))
    assert (
        load_semantic_active8_chunk_cache(
            first_output,
            run_identity_sha256=first["run_identity_sha256"],
            expected_source_shard_sha256=shard_sha256,
            expected_source_manifest_sha256=manifest_sha256,
        )
        == first
    )


def test_completed_chunk_cache_restart_is_read_only(tmp_path) -> None:
    artifact = tmp_path / "source"
    output = tmp_path / "cache"
    completion, shard_sha256, manifest_sha256 = _build(
        artifact,
        output,
        entries=4,
    )
    before = {
        path.relative_to(output).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in output.rglob("*")
        if path.is_file()
    }
    restarted = build_semantic_active8_chunk_cache(
        artifact,
        expected_shard_sha256=shard_sha256,
        expected_manifest_sha256=manifest_sha256,
        expected_source_binding=_source_binding(4),
        output_root=output,
        target_rows_per_chunk=2,
    )
    after = {
        path.relative_to(output).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in output.rglob("*")
        if path.is_file()
    }
    assert restarted == completion
    assert after == before


def test_chunk_cache_build_uses_no_chemistry_replay(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "source"
    output = tmp_path / "cache"
    shard_sha256, manifest_sha256 = _write_source(artifact, entries=2)

    import compose_v4.rewrite.trace_shard_v3 as trace_v3

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("mechanical chunking invoked chemistry")

    monkeypatch.setattr(trace_v3, "editing_v2_semantic_rewrite_system", forbidden)
    monkeypatch.setattr(trace_v3, "canonical_state_key", forbidden)
    completion = build_semantic_active8_chunk_cache(
        artifact,
        expected_shard_sha256=shard_sha256,
        expected_manifest_sha256=manifest_sha256,
        output_root=output,
        target_rows_per_chunk=1,
    )
    assert completion["chunk_count"] == 2


def test_chunk_cache_rejects_source_tampering_before_publication(tmp_path) -> None:
    artifact = tmp_path / "source"
    output = tmp_path / "cache"
    shard_sha256, manifest_sha256 = _write_source(artifact, entries=1)
    shard_path = artifact / SHARD_FILENAME
    shard_path.write_bytes(shard_path.read_bytes() + b"tamper")
    with pytest.raises(SemanticPackedStoreError, match="expected SHA-256"):
        build_semantic_active8_chunk_cache(
            artifact,
            expected_shard_sha256=shard_sha256,
            expected_manifest_sha256=manifest_sha256,
            output_root=output,
        )
    assert not list(output.rglob("COMPLETE.json"))


def test_chunk_cache_rejects_noncanonical_source_rows(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "source"
    output = tmp_path / "cache"
    shard_sha256, manifest_sha256 = _write_source(artifact, entries=1)

    import compose_v4.data.semantic_active8_chunk_cache as chunk_cache

    with gzip.open(artifact / SHARD_FILENAME, "rb") as handle:
        record = json.loads(next(iter(handle)))
    noncanonical = json.dumps(record, separators=(", ", ": ")).encode() + b"\n"
    original_open = chunk_cache.gzip.open

    def substituted_open(path, *args, **kwargs):
        if path == artifact / SHARD_FILENAME:
            return io.BytesIO(noncanonical)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(chunk_cache.gzip, "open", substituted_open)
    with pytest.raises(SemanticActive8ChunkCacheError, match="not canonical JSONL"):
        build_semantic_active8_chunk_cache(
            artifact,
            expected_shard_sha256=shard_sha256,
            expected_manifest_sha256=manifest_sha256,
            output_root=output,
        )
    assert not list(output.rglob("COMPLETE.json"))


def test_chunk_cache_requires_full_source_stream_before_completion(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "source"
    output = tmp_path / "cache"
    shard_sha256, manifest_sha256 = _write_source(artifact, entries=2)

    import compose_v4.data.semantic_active8_chunk_cache as chunk_cache

    with gzip.open(artifact / SHARD_FILENAME, "rb") as handle:
        first_line = next(iter(handle))
    original_open = chunk_cache.gzip.open

    def short_open(path, *args, **kwargs):
        if path == artifact / SHARD_FILENAME:
            return io.BytesIO(first_line)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(chunk_cache.gzip, "open", short_open)
    with pytest.raises(SemanticActive8ChunkCacheError, match="census disagrees"):
        build_semantic_active8_chunk_cache(
            artifact,
            expected_shard_sha256=shard_sha256,
            expected_manifest_sha256=manifest_sha256,
            output_root=output,
            target_rows_per_chunk=1,
        )
    assert not list(output.rglob("COMPLETE.json"))
    assert list((output / "objects").rglob("*.jsonl.gz"))

    monkeypatch.undo()
    completion = build_semantic_active8_chunk_cache(
        artifact,
        expected_shard_sha256=shard_sha256,
        expected_manifest_sha256=manifest_sha256,
        output_root=output,
        target_rows_per_chunk=1,
    )
    assert completion["chunk_count"] == 2


@pytest.mark.parametrize("second_start", [1, 3])
def test_chunk_cache_loader_rejects_overlap_or_gap_in_completion(
    tmp_path,
    second_start,
) -> None:
    artifact = tmp_path / "source"
    output = tmp_path / "cache"
    completion, shard_sha256, manifest_sha256 = _build(
        artifact,
        output,
        entries=4,
    )
    completion_path = (
        output / "runs" / completion["run_identity_sha256"] / "COMPLETE.json"
    )
    tampered = json.loads(completion_path.read_text())
    tampered["chunk_inventory"][1]["entry_start"] = second_start
    tampered["chunk_inventory_sha256"] = _canonical_sha256(tampered["chunk_inventory"])
    completion_body = {
        key: value for key, value in tampered.items() if key != "completion_sha256"
    }
    tampered["completion_sha256"] = _canonical_sha256(completion_body)
    completion_path.write_bytes(_canonical_bytes(tampered))
    with pytest.raises(
        SemanticActive8ChunkCacheError,
        match="ranges do not partition",
    ):
        load_semantic_active8_chunk_cache(
            output,
            run_identity_sha256=completion["run_identity_sha256"],
            expected_source_shard_sha256=shard_sha256,
            expected_source_manifest_sha256=manifest_sha256,
        )


@pytest.mark.parametrize("object_field", ["chunk_object_path", "receipt_object_path"])
def test_chunk_cache_loader_rejects_tampered_content_objects(
    tmp_path,
    object_field,
) -> None:
    artifact = tmp_path / "source"
    output = tmp_path / "cache"
    completion, shard_sha256, manifest_sha256 = _build(
        artifact,
        output,
        entries=2,
    )
    object_path = output / completion["chunk_inventory"][0][object_field]
    object_path.write_bytes(object_path.read_bytes() + b"tamper")
    with pytest.raises(SemanticActive8ChunkCacheError, match="bytes disagree"):
        load_semantic_active8_chunk_cache(
            output,
            run_identity_sha256=completion["run_identity_sha256"],
            expected_source_shard_sha256=shard_sha256,
            expected_source_manifest_sha256=manifest_sha256,
        )


def test_empty_semantic_source_publishes_one_safe_empty_chunk(tmp_path) -> None:
    artifact = tmp_path / "empty-source"
    output = tmp_path / "empty-cache"
    completion, shard_sha256, manifest_sha256 = _build(
        artifact,
        output,
        entries=0,
    )
    assert completion["chunk_count"] == 1
    assert completion["chunk_inventory"][0]["entry_start"] == 0
    assert completion["chunk_inventory"][0]["entry_stop"] == 0
    chunk = completion["chunk_inventory"][0]
    assert (
        list(
            read_semantic_active8_chunk(
                output,
                receipt_object_path=chunk["receipt_object_path"],
                expected_receipt_file_sha256=chunk["receipt_file_sha256"],
                expected_source_shard_sha256=shard_sha256,
                expected_source_manifest_sha256=manifest_sha256,
            )
        )
        == []
    )


def test_chunk_cache_requires_positive_target_rows(tmp_path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_semantic_active8_chunk_cache(
            tmp_path / "absent",
            expected_shard_sha256="0" * 64,
            expected_manifest_sha256="1" * 64,
            output_root=tmp_path / "cache",
            target_rows_per_chunk=0,
        )
