"""Gate-0 successor probes are deterministic, exact-addressed diagnostics."""

from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.persistent_state_identity import (
    persistent_slot_state_sha256,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.packed_trace_store import PackedTraceAddress
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheAddress,
    SuccessorFiberCacheProvenance,
    SuccessorFiberCacheRecord,
    deserialize_successor_fiber_cache,
    fiber_compiler_implementation_hash,
)
from compose_v4.experiments import editing_step_zero_probe as probe
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.factorized_successor_training import (
    StateProductiveSupport,
    TeacherSuccessorAlias,
    TeacherSuccessorFiber,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.action_codec import canonical_family
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteTrace, inverse_step
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog

_SHARD_SHA256 = "a" * 64


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), 6)


def _address(
    *,
    entry_index: int,
    trace_id: str,
    path: TraceProgressCTMC,
    partition: str = "validation",
) -> PackedTraceAddress:
    return PackedTraceAddress(
        packed_shard_content_sha256=_SHARD_SHA256,
        packed_shard_name="gate-zero-validation.jsonl.gz",
        entry_index=entry_index,
        trace_id=trace_id,
        layer="corruption",
        partition=partition,
        source_key=canonical_state_key(path.state_at(0)),
        target_key=canonical_state_key(path.state_at(path.path_length)),
        path_length=path.path_length,
    )


def _record(
    *,
    entry_index: int,
    trace_id: str,
    trace: RewriteTrace,
    partition: str = "validation",
) -> PathRecord:
    path = TraceProgressCTMC(trace)
    address = _address(
        entry_index=entry_index,
        trace_id=trace_id,
        path=path,
        partition=partition,
    )
    return PathRecord(
        target_key=address.target_key,
        path=path,
        corpus_address=address,
    )


@pytest.fixture(scope="module")
def fixture_bundle():
    carbon = _state("C")
    ethane = _state("CC")
    grow = compile_carbon_tree_to_target(carbon, ethane)
    shrink = inverse_step(grow.source, grow.steps[0])
    round_trip = RewriteTrace(
        source=grow.source,
        target=grow.source,
        steps=(grow.steps[0], shrink),
        metadata={"fixture": "round_trip"},
    )
    grow_only = grow
    shrink_only = RewriteTrace(
        source=grow.target,
        target=grow.source,
        steps=(shrink,),
        metadata={"fixture": "shrink"},
    )
    records = (
        _record(
            entry_index=0,
            trace_id="round-trip",
            trace=round_trip,
        ),
        _record(
            entry_index=1,
            trace_id="grow-only",
            trace=grow_only,
        ),
        _record(
            entry_index=2,
            trace_id="shrink-only",
            trace=shrink_only,
        ),
    )
    torch.manual_seed(17)
    model = FactorizedTraceletRateModel(
        build_typed_ring_catalog((round_trip,)),
        hidden_dim=16,
        message_passing_steps=1,
        enable_ring_restates=True,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_ring_opening=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        atom_vocabulary=ORGANIC_VOCABULARY,
    ).eval()
    return model, records


def _semantic_cells(records):
    cells = {}
    for record in records:
        assert record.corpus_address is not None
        for progress_index in range(record.path.path_length + 1):
            key = (
                record.corpus_address.packed_shard_content_sha256,
                record.corpus_address.entry_index,
                progress_index,
            )
            if progress_index == record.path.path_length:
                cells[key] = None
            else:
                family = canonical_family(record.path.trace.steps[progress_index].rule_name)
                cells[key] = f"cell:{family}"
    return cells


def _provenance(model) -> SuccessorFiberCacheProvenance:
    return SuccessorFiberCacheProvenance(
        operator_registry_hash="operator-registry-v1",
        capability_hash=model.operator_capabilities.fingerprint(),
        support_signature_sha256="1" * 64,
        canonicalization_version="canonical-state-key-v1",
        canonicalizer_contract_sha256="2" * 64,
        packed_corpus_schema="compose.data.packed_trace",
        packed_corpus_schema_version=1,
        packed_shard_content_sha256=_SHARD_SHA256,
        packed_manifest_sha256="3" * 64,
        packed_provenance_overlay_sha256="4" * 64,
        unified_packed_manifest_sha256="5" * 64,
        representability_overlay_sha256="6" * 64,
        coordinate_schema_version=1,
        tensorization_implementation_hash="tensorization-v1",
        fiber_compiler_implementation_hash=(fiber_compiler_implementation_hash()),
    )


def _fake_compile(_model, record, *, time):
    assert time == probe.GATE_ZERO_SUCCESSOR_SUPPORT_TIME
    assert record.corpus_address is not None
    rows = []
    for progress_index in range(record.path.path_length + 1):
        source = record.path.state_at(progress_index)
        source_key = canonical_state_key(source)
        support = StateProductiveSupport(
            source_key=source_key,
            source_state_sha256=persistent_slot_state_sha256(source),
        )
        address = SuccessorFiberCacheAddress.from_packed_trace(
            record.corpus_address,
            progress_index=progress_index,
        )
        if progress_index == record.path.path_length:
            fiber = None
        else:
            target = record.path.state_at(progress_index + 1)
            family = canonical_family(record.path.trace.steps[progress_index].rule_name)
            fiber = TeacherSuccessorFiber(
                source_key=source_key,
                target_key=canonical_state_key(target),
                target_state_sha256=persistent_slot_state_sha256(target),
                aliases=(
                    TeacherSuccessorAlias(
                        family_name=family,
                        table_name=f"{family}_fixture",
                        coordinate=(0,),
                    ),
                ),
                state_support=support,
            )
        rows.append(
            SuccessorFiberCacheRecord(
                address=address,
                state_support=support,
                teacher_fiber=fiber,
            )
        )
    return tuple(rows)


def _build(monkeypatch, model, records, cells):
    calls = []

    def compile_spy(*args, **kwargs):
        calls.append(args[1].corpus_address)
        return _fake_compile(*args, **kwargs)

    monkeypatch.setattr(
        probe,
        "compile_successor_fiber_trace",
        compile_spy,
    )
    artifact = probe.build_gate_zero_successor_probe(
        model,
        records,
        semantic_cell_ids=cells,
        config=probe.GateZeroSuccessorProbeConfig(
            required_families=("atom_delete", "atom_insert"),
            required_semantic_cells=(
                "cell:atom_delete",
                "cell:atom_insert",
            ),
        ),
        provenance_by_shard={_SHARD_SHA256: _provenance(model)},
    )
    return artifact, calls


def test_probe_selects_one_complete_trace_and_uses_cache_schema(
    fixture_bundle,
    monkeypatch,
) -> None:
    model, records = fixture_bundle
    artifact, calls = _build(
        monkeypatch,
        model,
        records,
        _semantic_cells(records),
    )

    assert tuple(record.corpus_address.trace_id for record in artifact.selected_path_records) == (
        "round-trip",
    )
    assert len(calls) == 1
    assert len(artifact.rows) == 3
    assert [row.address.progress_index for row in artifact.rows] == [
        0,
        1,
        2,
    ]
    assert artifact.rows[-1].address.is_terminal
    assert artifact.rows[-1].semantic_cell_id is None
    assert set(artifact.family_witnesses) == {
        "atom_insert",
        "atom_delete",
    }
    assert set(artifact.semantic_cell_witnesses) == {
        "cell:atom_insert",
        "cell:atom_delete",
    }
    assert artifact.training_authorized is False
    with pytest.raises(
        probe.GateZeroSuccessorProbeError,
        match="never authorizes training",
    ):
        artifact.assert_training_authorized()

    shard = artifact.cache_shards[0]
    decoded = deserialize_successor_fiber_cache(
        shard.encoded,
        expected_provenance=_provenance(model),
        expected_content_sha256=shard.content_sha256,
    )
    assert decoded.records == artifact.cache_records
    manifest = artifact.manifest()
    assert manifest["training_authorized"] is False
    assert manifest["probe_sha256"] == artifact.probe_sha256
    assert (
        manifest["cache_shards"][0]["cache_encoded_sha256"]
        == hashlib.sha256(shard.encoded).hexdigest()
    )


def test_probe_is_deterministic_under_source_and_sidecar_order(
    fixture_bundle,
    monkeypatch,
) -> None:
    model, records = fixture_bundle
    cells = _semantic_cells(records)
    first, _calls = _build(monkeypatch, model, records, cells)
    reversed_cells = dict(reversed(tuple(cells.items())))
    second, _calls = _build(
        monkeypatch,
        model,
        tuple(reversed(records)),
        reversed_cells,
    )

    assert first.config_sha256 == second.config_sha256
    assert first.source_sha256 == second.source_sha256
    assert first.semantic_cell_sidecar_sha256 == second.semantic_cell_sidecar_sha256
    assert first.probe_sha256 == second.probe_sha256
    assert first.cache_shards[0].encoded == second.cache_shards[0].encoded


def test_probe_fails_closed_on_missing_obligation_or_sidecar_row(
    fixture_bundle,
    monkeypatch,
) -> None:
    model, records = fixture_bundle
    cells = _semantic_cells(records)
    config = probe.GateZeroSuccessorProbeConfig(
        required_families=("atom_insert",),
        required_semantic_cells=("absent-cell",),
    )
    with pytest.raises(
        probe.GateZeroSuccessorProbeError,
        match="cannot cover",
    ):
        probe.build_gate_zero_successor_probe(
            model,
            records,
            semantic_cell_ids=cells,
            config=config,
            provenance_by_shard={},
        )

    missing = dict(cells)
    missing.pop(next(iter(missing)))
    with pytest.raises(
        probe.GateZeroSuccessorProbeError,
        match="missing exact row",
    ):
        _build(monkeypatch, model, records, missing)


def test_probe_rejects_nonvalidation_and_provenance_drift(
    fixture_bundle,
    monkeypatch,
) -> None:
    model, records = fixture_bundle
    train_record = replace(
        records[0],
        corpus_address=replace(
            records[0].corpus_address,
            partition="train",
        ),
    )
    with pytest.raises(
        probe.GateZeroSuccessorProbeError,
        match="validation PathRecords only",
    ):
        probe.build_gate_zero_successor_probe(
            model,
            (train_record,),
            semantic_cell_ids={},
            config=probe.GateZeroSuccessorProbeConfig(
                required_families=("atom_insert",),
                required_semantic_cells=("cell:atom_insert",),
            ),
            provenance_by_shard={},
        )

    monkeypatch.setattr(
        probe,
        "compile_successor_fiber_trace",
        _fake_compile,
    )
    drifted = replace(_provenance(model), capability_hash="wrong")
    with pytest.raises(
        probe.GateZeroSuccessorProbeError,
        match="capability hash disagrees",
    ):
        probe.build_gate_zero_successor_probe(
            model,
            records,
            semantic_cell_ids=_semantic_cells(records),
            config=probe.GateZeroSuccessorProbeConfig(
                required_families=("atom_insert",),
                required_semantic_cells=("cell:atom_insert",),
            ),
            provenance_by_shard={_SHARD_SHA256: drifted},
        )


def test_probe_rejects_compiler_state_identity_drift(
    fixture_bundle,
    monkeypatch,
) -> None:
    model, records = fixture_bundle

    def compile_drifted(*args, **kwargs):
        compiled = _fake_compile(*args, **kwargs)
        first = compiled[0]
        assert first.teacher_fiber is not None
        wrong_target = persistent_slot_state_sha256(_state("CCC"))
        drifted_fiber = replace(
            first.teacher_fiber,
            target_state_sha256=wrong_target,
        )
        return (
            replace(
                first,
                teacher_fiber=drifted_fiber,
            ),
            *compiled[1:],
        )

    monkeypatch.setattr(
        probe,
        "compile_successor_fiber_trace",
        compile_drifted,
    )
    with pytest.raises(
        probe.GateZeroSuccessorProbeError,
        match="wrong exact teacher successor",
    ):
        probe.build_gate_zero_successor_probe(
            model,
            records,
            semantic_cell_ids=_semantic_cells(records),
            config=probe.GateZeroSuccessorProbeConfig(
                required_families=("atom_insert",),
                required_semantic_cells=("cell:atom_insert",),
            ),
            provenance_by_shard={_SHARD_SHA256: _provenance(model)},
        )


def test_probe_config_rejects_duplicates_and_disabled_macro() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        probe.GateZeroSuccessorProbeConfig(
            required_families=("atom_insert", "atom_insert"),
            required_semantic_cells=("cell",),
        )
    with pytest.raises(ValueError, match="disabled or unknown"):
        probe.GateZeroSuccessorProbeConfig(
            required_families=("ring_system_grow",),
            required_semantic_cells=("cell",),
        )
