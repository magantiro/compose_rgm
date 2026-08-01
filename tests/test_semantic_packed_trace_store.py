"""Determinism and fail-closed tests for semantic exact-state packing."""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.semantic_packed_trace_store import (
    COMPLETION_FILENAME,
    MANIFEST_FILENAME,
    SHARD_FILENAME,
    SemanticPackedStoreError,
    load_semantic_packed_manifest,
    read_semantic_packed_artifact,
    read_semantic_packed_artifact_range,
    validate_semantic_packed_entry_ranges,
    write_semantic_packed_artifact,
)
from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system
from compose_v4.rewrite.operators import CycleCloseEdge
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace
from compose_v4.rewrite.trace_shard import decode_state
from compose_v4.rewrite.trace_shard_v3 import encode_semantic_trace_record

LANE = "reversible_synthetic_walk"
SPLIT = "development"


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(*, trace_id: str = "semantic-fixture") -> dict[str, object]:
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


def _source_binding(*, entries: int = 1) -> dict[str, object]:
    return {
        "source_shard_name": "legacy_0000.jsonl.gz",
        "source_shard_sha256": "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "source_overlay_sha256": "3" * 64,
        "source_unified_manifest_sha256": "4" * 64,
        "source_entry_count": entries,
    }


def _decision_binding(*, admitted: int = 1, rejected: int = 0) -> dict[str, object]:
    return {
        "decision_ledger_sha256": "5" * 64,
        "source_count": admitted + rejected,
        "admitted_count": admitted,
        "rejected_count": rejected,
    }


def _write(artifact, records, *, source_entries=1, admitted=1, rejected=0):
    write_semantic_packed_artifact(
        artifact,
        records,
        data_lane=LANE,
        split=SPLIT,
        source_binding=_source_binding(entries=source_entries),
        decision_binding=_decision_binding(admitted=admitted, rejected=rejected),
    )
    return (
        _sha256(artifact / SHARD_FILENAME),
        _sha256(artifact / MANIFEST_FILENAME),
    )


def test_semantic_packed_bytes_are_deterministic_across_output_names(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_hashes = _write(first, [_record()])
    second_hashes = _write(second, [_record()])
    assert first_hashes == second_hashes
    for filename in (SHARD_FILENAME, MANIFEST_FILENAME, COMPLETION_FILENAME):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_semantic_packed_roundtrip_preserves_every_exact_array(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    record = _record()
    shard_sha256, manifest_sha256 = _write(artifact, [record])
    rows = list(
        read_semantic_packed_artifact(
            artifact,
            expected_shard_sha256=shard_sha256,
            expected_manifest_sha256=manifest_sha256,
            expected_source_binding=_source_binding(),
        )
    )
    assert len(rows) == 1
    assert rows[0].address.layer == LANE
    assert rows[0].address.partition == SPLIT
    for state, payload in zip(rows[0].path.states, record["states"]):
        expected = decode_state(payload)
        assert np.array_equal(state.atom_types, expected.atom_types)
        assert np.array_equal(state.formal_charges, expected.formal_charges)
        assert np.array_equal(state.implicit_h_counts, expected.implicit_h_counts)
        assert np.array_equal(state.bonds, expected.bonds)


def test_semantic_packed_hot_reader_skips_executor_and_canonicalizer(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "artifact"
    shard_sha256, manifest_sha256 = _write(artifact, [_record()])

    import compose_v4.rewrite.trace_shard_v3 as trace_v3

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("hot reader invoked a forbidden chemistry path")

    monkeypatch.setattr(trace_v3, "editing_v2_semantic_rewrite_system", forbidden)
    monkeypatch.setattr(trace_v3, "canonical_state_key", forbidden)
    rows = list(
        read_semantic_packed_artifact(
            artifact,
            expected_shard_sha256=shard_sha256,
            expected_manifest_sha256=manifest_sha256,
            sentinel_replay_entries=0,
        )
    )
    assert len(rows) == 1
    with pytest.raises(RuntimeError, match="forbidden chemistry"):
        list(
            read_semantic_packed_artifact(
                artifact,
                expected_shard_sha256=shard_sha256,
                expected_manifest_sha256=manifest_sha256,
                sentinel_replay_entries=1,
            )
        )


def test_semantic_packed_reader_requires_external_physical_hashes(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    shard_sha256, manifest_sha256 = _write(artifact, [_record()])
    with pytest.raises(SemanticPackedStoreError, match="expected SHA-256"):
        load_semantic_packed_manifest(
            artifact,
            expected_shard_sha256="0" * 64,
            expected_manifest_sha256=manifest_sha256,
        )
    with pytest.raises(SemanticPackedStoreError, match="physical SHA-256"):
        load_semantic_packed_manifest(
            artifact,
            expected_shard_sha256=shard_sha256,
            expected_manifest_sha256="0" * 64,
        )


def test_semantic_packed_missing_completion_is_never_readable(tmp_path) -> None:
    artifact = tmp_path / "artifact"
    shard_sha256, manifest_sha256 = _write(artifact, [_record()])
    (artifact / COMPLETION_FILENAME).unlink()
    with pytest.raises(SemanticPackedStoreError, match="inventory"):
        load_semantic_packed_manifest(
            artifact,
            expected_shard_sha256=shard_sha256,
            expected_manifest_sha256=manifest_sha256,
        )


def test_semantic_packed_empty_shard_is_complete_and_deterministic(tmp_path) -> None:
    artifact = tmp_path / "empty"
    shard_sha256, manifest_sha256 = _write(
        artifact,
        [],
        source_entries=0,
        admitted=0,
    )
    manifest = load_semantic_packed_manifest(
        artifact,
        expected_shard_sha256=shard_sha256,
        expected_manifest_sha256=manifest_sha256,
    )
    assert manifest["entries"] == 0
    assert (
        list(
            read_semantic_packed_artifact(
                artifact,
                expected_shard_sha256=shard_sha256,
                expected_manifest_sha256=manifest_sha256,
            )
        )
        == []
    )


def test_semantic_packed_rejects_duplicate_trace_ids_before_publication(
    tmp_path,
) -> None:
    with pytest.raises(SemanticPackedStoreError, match="duplicate trace_id"):
        _write(
            tmp_path / "duplicate",
            [_record(), _record()],
            source_entries=2,
            admitted=2,
        )
    assert not (tmp_path / "duplicate").exists()


def test_adjacent_semantic_ranges_are_address_equivalent_to_full_read(
    tmp_path,
) -> None:
    artifact = tmp_path / "ranged"
    records = [_record(trace_id=f"semantic-fixture-{index}") for index in range(5)]
    shard_sha256, manifest_sha256 = _write(
        artifact,
        records,
        source_entries=5,
        admitted=5,
    )
    full = list(
        read_semantic_packed_artifact(
            artifact,
            expected_shard_sha256=shard_sha256,
            expected_manifest_sha256=manifest_sha256,
            sentinel_replay_entries=0,
        )
    )
    ranges = validate_semantic_packed_entry_ranges(
        ((0, 2), (2, 4), (4, 5)),
        entries=5,
    )
    ranged = [
        row
        for entry_start, entry_stop in ranges
        for row in read_semantic_packed_artifact_range(
            artifact,
            expected_shard_sha256=shard_sha256,
            expected_manifest_sha256=manifest_sha256,
            entry_start=entry_start,
            entry_stop=entry_stop,
            sentinel_replay_entries=0,
        )
    ]
    assert [row.address for row in ranged] == [row.address for row in full]
    assert [row.address.entry_index for row in ranged] == list(range(5))
    for ranged_row, full_row in zip(ranged, full, strict=True):
        assert len(ranged_row.path.states) == len(full_row.path.states)
        for ranged_state, full_state in zip(
            ranged_row.path.states,
            full_row.path.states,
            strict=True,
        ):
            assert np.array_equal(ranged_state.atom_types, full_state.atom_types)
            assert np.array_equal(
                ranged_state.formal_charges,
                full_state.formal_charges,
            )
            assert np.array_equal(
                ranged_state.implicit_h_counts,
                full_state.implicit_h_counts,
            )
            assert np.array_equal(ranged_state.bonds, full_state.bonds)


def test_semantic_range_contract_rejects_gaps_overlaps_and_bad_bounds() -> None:
    assert validate_semantic_packed_entry_ranges(
        ((0, 2), (2, 5)),
        entries=5,
    ) == ((0, 2), (2, 5))
    assert validate_semantic_packed_entry_ranges(((0, 0),), entries=0) == ((0, 0),)
    with pytest.raises(ValueError, match="gap"):
        validate_semantic_packed_entry_ranges(((0, 2), (3, 5)), entries=5)
    with pytest.raises(ValueError, match="overlap"):
        validate_semantic_packed_entry_ranges(((0, 3), (2, 5)), entries=5)
    with pytest.raises(ValueError, match="does not cover"):
        validate_semantic_packed_entry_ranges(((0, 4),), entries=5)
    with pytest.raises(ValueError, match="empty.*only for an empty shard"):
        validate_semantic_packed_entry_ranges(((0, 0), (0, 5)), entries=5)
    with pytest.raises(ValueError, match="outside the manifest census"):
        validate_semantic_packed_entry_ranges(((0, 6),), entries=5)


def test_semantic_range_reader_rejects_out_of_bounds_and_nonempty_empty_range(
    tmp_path,
) -> None:
    artifact = tmp_path / "bounds"
    shard_sha256, manifest_sha256 = _write(artifact, [_record()])
    common = {
        "expected_shard_sha256": shard_sha256,
        "expected_manifest_sha256": manifest_sha256,
    }
    with pytest.raises(ValueError, match="nonnegative half-open"):
        read_semantic_packed_artifact_range(
            artifact,
            entry_start=-1,
            entry_stop=1,
            **common,
        )
    with pytest.raises(ValueError, match="outside the manifest census"):
        read_semantic_packed_artifact_range(
            artifact,
            entry_start=0,
            entry_stop=2,
            **common,
        )
    with pytest.raises(ValueError, match="empty.*only for an empty shard"):
        read_semantic_packed_artifact_range(
            artifact,
            entry_start=1,
            entry_stop=1,
            **common,
        )


def test_semantic_empty_shard_has_one_safe_empty_range(tmp_path) -> None:
    artifact = tmp_path / "empty-range"
    shard_sha256, manifest_sha256 = _write(
        artifact,
        [],
        source_entries=0,
        admitted=0,
    )
    rows = list(
        read_semantic_packed_artifact_range(
            artifact,
            expected_shard_sha256=shard_sha256,
            expected_manifest_sha256=manifest_sha256,
            entry_start=0,
            entry_stop=0,
        )
    )
    assert rows == []


def test_semantic_range_reader_rejects_physical_shard_tampering(tmp_path) -> None:
    artifact = tmp_path / "tampered-range"
    shard_sha256, manifest_sha256 = _write(artifact, [_record()])
    shard_path = artifact / SHARD_FILENAME
    shard_path.write_bytes(shard_path.read_bytes() + b"tamper")
    with pytest.raises(SemanticPackedStoreError, match="expected SHA-256"):
        read_semantic_packed_artifact_range(
            artifact,
            expected_shard_sha256=shard_sha256,
            expected_manifest_sha256=manifest_sha256,
            entry_start=0,
            entry_stop=1,
        )


def test_semantic_range_reader_requires_exact_source_binding(tmp_path) -> None:
    artifact = tmp_path / "source-binding-range"
    shard_sha256, manifest_sha256 = _write(artifact, [_record()])
    wrong_binding = _source_binding()
    wrong_binding["source_shard_sha256"] = "9" * 64
    with pytest.raises(SemanticPackedStoreError, match="source binding disagrees"):
        read_semantic_packed_artifact_range(
            artifact,
            expected_shard_sha256=shard_sha256,
            expected_manifest_sha256=manifest_sha256,
            expected_source_binding=wrong_binding,
            entry_start=0,
            entry_stop=1,
        )


def test_semantic_range_exhaustion_rejects_a_short_decoded_stream(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "short-range"
    records = [_record(trace_id=f"semantic-fixture-{index}") for index in range(2)]
    shard_sha256, manifest_sha256 = _write(
        artifact,
        records,
        source_entries=2,
        admitted=2,
    )

    import compose_v4.data.semantic_packed_trace_store as semantic_store

    with semantic_store.gzip.open(artifact / SHARD_FILENAME, "rb") as handle:
        first_line = next(iter(handle))

    def short_stream(*_args, **_kwargs):
        return io.BytesIO(first_line)

    monkeypatch.setattr(semantic_store.gzip, "open", short_stream)
    with pytest.raises(
        SemanticPackedStoreError, match="expected 2 rows but observed 1"
    ):
        list(
            read_semantic_packed_artifact_range(
                artifact,
                expected_shard_sha256=shard_sha256,
                expected_manifest_sha256=manifest_sha256,
                entry_start=0,
                entry_stop=2,
                sentinel_replay_entries=0,
            )
        )


def test_semantic_range_sentinel_replay_uses_original_shard_indices(
    tmp_path,
    monkeypatch,
) -> None:
    artifact = tmp_path / "sentinel-range"
    records = [_record(trace_id=f"semantic-fixture-{index}") for index in range(3)]
    shard_sha256, manifest_sha256 = _write(
        artifact,
        records,
        source_entries=3,
        admitted=3,
    )

    import compose_v4.rewrite.trace_shard_v3 as trace_v3

    original_runtime = trace_v3.editing_v2_semantic_rewrite_system
    replayed_rows: list[None] = []

    def counted_runtime():
        replayed_rows.append(None)
        return original_runtime()

    monkeypatch.setattr(
        trace_v3,
        "editing_v2_semantic_rewrite_system",
        counted_runtime,
    )
    common = {
        "expected_shard_sha256": shard_sha256,
        "expected_manifest_sha256": manifest_sha256,
        "sentinel_replay_entries": 2,
    }
    assert (
        len(
            list(
                read_semantic_packed_artifact_range(
                    artifact,
                    entry_start=0,
                    entry_stop=1,
                    **common,
                )
            )
        )
        == 1
    )
    assert (
        len(
            list(
                read_semantic_packed_artifact_range(
                    artifact,
                    entry_start=1,
                    entry_stop=3,
                    **common,
                )
            )
        )
        == 2
    )
    assert len(replayed_rows) == 2
