"""Exact-address semantic sidecars are complete, deterministic, and sealed."""

from __future__ import annotations

from dataclasses import replace

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.packed_trace_store import PackedTraceAddress
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.ringcore_semantic_axes import (
    BoundedSemanticCellCensus,
    context_from_path_record,
    label_validation_semantic_axes,
)
from compose_v4.experiments.ringcore_semantic_sidecar import (
    SemanticSidecarConfig,
    SemanticSidecarError,
    SemanticSidecarProvenance,
    SemanticSidecarSourceShard,
    build_semantic_cell_sidecar,
    read_semantic_cell_sidecar,
    semantic_axis_labeler_source_sha256,
    write_semantic_cell_sidecar,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteTrace, inverse_step
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target

_SHARD_SHA256 = "a" * 64
_MAXIMUM_CELLS = 100


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 6)


def _record(
    *,
    entry_index: int,
    trace_id: str,
    trace: RewriteTrace,
    partition: str = "validation",
) -> PathRecord:
    path = TraceProgressCTMC(trace)
    address = PackedTraceAddress(
        packed_shard_content_sha256=_SHARD_SHA256,
        packed_shard_name="semantic-validation.jsonl.gz",
        entry_index=entry_index,
        trace_id=trace_id,
        layer="corruption",
        partition=partition,
        source_key=canonical_state_key(path.state_at(0)),
        target_key=canonical_state_key(path.state_at(path.path_length)),
        path_length=path.path_length,
    )
    return PathRecord(
        target_key=address.target_key,
        path=path,
        corpus_address=address,
    )


@pytest.fixture(scope="module")
def records():
    carbon = _state("C")
    ethane = _state("CC")
    grow = compile_carbon_tree_to_target(carbon, ethane)
    delete = inverse_step(grow.source, grow.steps[0])
    shrink = RewriteTrace(
        source=grow.target,
        target=grow.source,
        steps=(delete,),
        metadata={"fixture": "shrink"},
    )
    return (
        _record(entry_index=0, trace_id="grow", trace=grow),
        _record(entry_index=1, trace_id="shrink", trace=shrink),
    )


def _semantic_census(records):
    census = BoundedSemanticCellCensus(maximum_nonempty_cells=_MAXIMUM_CELLS)
    for record in records:
        for progress_index in range(record.path.path_length + 1):
            census.add(
                label_validation_semantic_axes(
                    context_from_path_record(
                        record,
                        progress_index=progress_index,
                    )
                )
            )
    return census.payload()


def _provenance(records):
    return SemanticSidecarProvenance(
        source_census_artifact_sha256="1" * 64,
        unified_packed_manifest_sha256="2" * 64,
        representability_overlay_sha256="3" * 64,
        labeler_source_sha256=semantic_axis_labeler_source_sha256(),
        source_shards=(
            SemanticSidecarSourceShard(
                packed_shard_content_sha256=_SHARD_SHA256,
                packed_shard_name="semantic-validation.jsonl.gz",
                layer="corruption",
                partition="validation",
                packed_manifest_sha256="4" * 64,
                provenance_overlay_sha256="5" * 64,
                packed_entry_count=len(records),
                effective_trace_count=len(records),
                progress_row_count=sum(record.path.path_length + 1 for record in records),
            ),
        ),
    )


def _config(records):
    return SemanticSidecarConfig(
        maximum_nonempty_cells=_MAXIMUM_CELLS,
        expected_semantic_census_sha256=_semantic_census(records)["census_sha256"],
    )


def test_builder_retains_every_progress_and_terminal_row(records) -> None:
    artifact = build_semantic_cell_sidecar(
        records,
        config=_config(records),
        provenance=_provenance(records),
    )

    assert artifact.trace_count == 2
    assert len(artifact.rows) == 4
    assert tuple(row.progress_index for row in artifact.rows) == (0, 1, 0, 1)
    assert sum(row.semantic_cell_id is None for row in artifact.rows) == 2
    assert artifact.teacher_family_rows == {
        "atom_delete": 1,
        "atom_insert": 1,
    }
    assert len(artifact.semantic_cell_ids) == 4
    assert artifact.manifest()["training_authorized"] is False
    assert artifact.training_authorized is False
    with pytest.raises(SemanticSidecarError, match="never authorizes training"):
        artifact.assert_training_authorized()


def test_builder_is_deterministic_under_input_order(records) -> None:
    config = _config(records)
    provenance = _provenance(records)
    forward = build_semantic_cell_sidecar(
        records,
        config=config,
        provenance=provenance,
    )
    reverse = build_semantic_cell_sidecar(
        tuple(reversed(records)),
        config=config,
        provenance=provenance,
    )

    assert forward.source_sha256 == reverse.source_sha256
    assert forward.rows == reverse.rows
    assert forward.compressed == reverse.compressed
    assert forward.manifest_sha256 == reverse.manifest_sha256


def test_writer_reader_round_trip_and_tamper_detection(
    records,
    tmp_path,
) -> None:
    artifact = build_semantic_cell_sidecar(
        records,
        config=_config(records),
        provenance=_provenance(records),
    )
    sidecar = tmp_path / "semantic.jsonl.gz"
    manifest = tmp_path / "semantic.manifest.json"
    write_semantic_cell_sidecar(sidecar, manifest, artifact)
    loaded = read_semantic_cell_sidecar(
        sidecar,
        manifest,
        expected_manifest_sha256=artifact.manifest_sha256,
        expected_config_sha256=artifact.config.sha256,
        expected_provenance_sha256=artifact.provenance.sha256,
    )

    assert loaded.rows == artifact.rows
    assert loaded.semantic_cell_ids == artifact.semantic_cell_ids
    assert loaded.training_authorized is False
    original = sidecar.read_bytes()
    sidecar.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(SemanticSidecarError, match="compressed.*hash"):
        read_semantic_cell_sidecar(
            sidecar,
            manifest,
            expected_manifest_sha256=artifact.manifest_sha256,
            expected_config_sha256=artifact.config.sha256,
            expected_provenance_sha256=artifact.provenance.sha256,
        )


def test_builder_fails_closed_on_incomplete_source_or_census(records) -> None:
    with pytest.raises(
        SemanticSidecarError,
        match="complete declared shard census",
    ):
        build_semantic_cell_sidecar(
            records[:1],
            config=_config(records),
            provenance=_provenance(records),
        )
    wrong_config = replace(
        _config(records),
        expected_semantic_census_sha256="f" * 64,
    )
    with pytest.raises(
        SemanticSidecarError,
        match="semantic census disagrees",
    ):
        build_semantic_cell_sidecar(
            records,
            config=wrong_config,
            provenance=_provenance(records),
        )


def test_builder_rejects_nonvalidation_and_labeler_drift(records) -> None:
    train = replace(
        records[0],
        corpus_address=replace(
            records[0].corpus_address,
            partition="train",
        ),
    )
    with pytest.raises(
        SemanticSidecarError,
        match="address disagrees with its declared shard lane",
    ):
        build_semantic_cell_sidecar(
            (train, records[1]),
            config=_config(records),
            provenance=_provenance(records),
        )

    drifted = replace(
        _provenance(records),
        labeler_source_sha256="f" * 64,
    )
    with pytest.raises(
        SemanticSidecarError,
        match="does not bind the current labeler",
    ):
        build_semantic_cell_sidecar(
            records,
            config=_config(records),
            provenance=drifted,
        )


def test_builder_rejects_packed_endpoint_identity_drift(records) -> None:
    drifted = replace(
        records[0],
        corpus_address=replace(
            records[0].corpus_address,
            source_key=records[0].corpus_address.target_key,
        ),
    )
    with pytest.raises(
        SemanticSidecarError,
        match="endpoint identity disagrees",
    ):
        build_semantic_cell_sidecar(
            (drifted, records[1]),
            config=_config(records),
            provenance=_provenance(records),
        )


def test_reader_enforces_uncompressed_byte_bound(records, tmp_path) -> None:
    artifact = build_semantic_cell_sidecar(
        records,
        config=_config(records),
        provenance=_provenance(records),
    )
    sidecar = tmp_path / "semantic.jsonl.gz"
    manifest = tmp_path / "semantic.manifest.json"
    write_semantic_cell_sidecar(sidecar, manifest, artifact)

    with pytest.raises(
        SemanticSidecarError,
        match="uncompressed-byte bound",
    ):
        read_semantic_cell_sidecar(
            sidecar,
            manifest,
            expected_manifest_sha256=artifact.manifest_sha256,
            expected_config_sha256=artifact.config.sha256,
            expected_provenance_sha256=artifact.provenance.sha256,
            max_content_bytes=1,
        )
