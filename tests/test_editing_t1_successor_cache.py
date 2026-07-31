"""The T1 trace-union cache is immutable, exact, and load-only at runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from rdkit import rdBase

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
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
)
from compose_v4.experiments import editing_t1_successor_cache as t1_cache
from compose_v4.experiments import successor_fiber_cache_builder as builder
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.factorized_successor_training import (
    StateProductiveSupport,
    TeacherSuccessorAlias,
    TeacherSuccessorFiber,
)
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.operators import AtomInsert
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace import RewriteStep, RewriteTrace

_SHARD_SHA256 = "1" * 64
_MANIFEST_SHA256 = "2" * 64
_OVERLAY_SHA256 = "3" * 64
_SYSTEM = de_novo_rewrite_system()
_SLOTS = 6


def _state(smiles: str):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), _SLOTS)


def _path_record(
    *,
    entry_index: int,
    atom_type: int,
    implicit_h_count: int,
) -> PathRecord:
    source = _state("C")
    step = RewriteStep(
        "atom_insert",
        AtomInsert(
            slot=1,
            atom_type=atom_type,
            formal_charge=0,
            implicit_h_count=implicit_h_count,
            neighbors=((0, 1),),
        ),
    )
    target = _SYSTEM.apply(source, step.rule_name, step.action)
    trace = RewriteTrace(
        source=source,
        target=target,
        steps=(step,),
        metadata={},
    )
    address = PackedTraceAddress(
        packed_shard_content_sha256=_SHARD_SHA256,
        packed_shard_name="validation_0000.jsonl.gz",
        entry_index=entry_index,
        trace_id=f"trace-{entry_index}",
        layer="operator_aware_real_endpoint",
        partition="validation",
        source_key=canonical_state_key(source),
        target_key=canonical_state_key(target),
        path_length=1,
    )
    return PathRecord(
        target_key=address.target_key,
        path=TraceProgressCTMC(trace, system=_SYSTEM),
        corpus_address=address,
    )


@pytest.fixture
def selected_records() -> tuple[PathRecord, ...]:
    return (
        _path_record(
            entry_index=7,
            atom_type=ELEMENT_TO_IDX["N"],
            implicit_h_count=2,
        ),
        _path_record(
            entry_index=2,
            atom_type=ELEMENT_TO_IDX["C"],
            implicit_h_count=3,
        ),
    )


@pytest.fixture
def provenance() -> SuccessorFiberCacheProvenance:
    return SuccessorFiberCacheProvenance(
        operator_registry_hash="operator-v1",
        capability_hash="active8-v1",
        support_signature_sha256="4" * 64,
        canonicalization_version="canonical-v1",
        canonicalizer_contract_sha256="5" * 64,
        packed_corpus_schema="compose.data.packed_trace_store",
        packed_corpus_schema_version=1,
        packed_shard_content_sha256=_SHARD_SHA256,
        packed_manifest_sha256=_MANIFEST_SHA256,
        packed_provenance_overlay_sha256="6" * 64,
        unified_packed_manifest_sha256=_MANIFEST_SHA256,
        representability_overlay_sha256=_OVERLAY_SHA256,
        coordinate_schema_version=1,
        tensorization_implementation_hash="tensor-v1",
        fiber_compiler_implementation_hash="7" * 64,
    )


@pytest.fixture
def model() -> torch.nn.Module:
    torch.manual_seed(41)
    return torch.nn.Linear(2, 2).eval()


@pytest.fixture
def identity(model) -> t1_cache.T1SuccessorCacheIdentity:
    return t1_cache.T1SuccessorCacheIdentity(
        t1_runtime_contract_sha256="8" * 64,
        t1_implementation_sha256="9" * 64,
        gate_zero_runtime_contract_sha256="a" * 64,
        panel_artifact_sha256="b" * 64,
        panel_selection_sha256="c" * 64,
        panel_census_sha256="d" * 64,
        panel_capacity_strata_sha256="e" * 64,
        forensics_file_sha256="f" * 64,
        charge_policy_audit_file_sha256="0" * 64,
        charge_policy_exclusions_file_sha256="1" * 64,
        charge_policy_exclusion_payload_sha256="2" * 64,
        charge_policy_source_input_inventory_sha256="3" * 64,
        active8_inventory_manifest_file_sha256="4" * 64,
        active8_inventory_sha256="5" * 64,
        active8_effective_source_corpus_cache_sha256="6" * 64,
        active8_unified_packed_manifest_sha256=_MANIFEST_SHA256,
        active8_support_contract_sha256="a" * 64,
        unified_packed_manifest_sha256=_MANIFEST_SHA256,
        representability_overlay_sha256=_OVERLAY_SHA256,
        support_time=0.5,
        scratch_seed=1729,
        initial_model_state_sha256=state_dict_semantic_sha256(model.state_dict()),
        compiler_device="cpu",
        compiler_dtype="torch.float32",
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        rdkit_version=rdBase.rdkitVersion,
    )


def _cache_records(
    selected_records: tuple[PathRecord, ...],
) -> tuple[SuccessorFiberCacheRecord, ...]:
    rows: list[SuccessorFiberCacheRecord] = []
    for path_record in selected_records:
        address = path_record.corpus_address
        assert address is not None
        source = path_record.path.state_at(0)
        target = path_record.path.state_at(1)
        source_support = StateProductiveSupport(
            source_key=canonical_state_key(source),
            source_state_sha256=persistent_slot_state_sha256(source),
            virtual_aliases=(),
        )
        fiber = TeacherSuccessorFiber(
            source_key=source_support.source_key,
            target_key=canonical_state_key(target),
            target_state_sha256=persistent_slot_state_sha256(target),
            aliases=(
                TeacherSuccessorAlias(
                    family_name="atom_insert",
                    table_name="grow_connected",
                    coordinate=(0, 0, 0),
                ),
            ),
            state_support=source_support,
        )
        rows.extend(
            (
                SuccessorFiberCacheRecord(
                    address=SuccessorFiberCacheAddress.from_packed_trace(
                        address,
                        progress_index=0,
                    ),
                    state_support=source_support,
                    teacher_fiber=fiber,
                ),
                SuccessorFiberCacheRecord(
                    address=SuccessorFiberCacheAddress.from_packed_trace(
                        address,
                        progress_index=1,
                    ),
                    state_support=StateProductiveSupport(
                        source_key=canonical_state_key(target),
                        source_state_sha256=persistent_slot_state_sha256(target),
                        virtual_aliases=(),
                    ),
                    teacher_fiber=None,
                ),
            )
        )
    return tuple(rows)


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_records: tuple[PathRecord, ...],
    provenance: SuccessorFiberCacheProvenance,
    identity: t1_cache.T1SuccessorCacheIdentity,
    model: torch.nn.Module,
):
    compiled = _cache_records(selected_records)
    calls: list[tuple[PathRecord, ...]] = []

    def compile_union(model, records, **kwargs):
        del model
        calls.append(tuple(records))
        assert kwargs["time"] == identity.support_time
        return compiled

    monkeypatch.setattr(
        t1_cache,
        "compile_successor_fiber_trace_union",
        compile_union,
    )
    manifest, receipt = t1_cache.build_t1_successor_cache(
        tmp_path,
        model,
        selected_records,
        provenance_by_shard={_SHARD_SHA256: provenance},
        identity=identity,
    )
    assert len(calls) == 1
    return manifest, receipt, compiled


def test_public_trace_union_compiler_passes_all_traces_to_one_state_grouping(
    selected_records,
    monkeypatch,
) -> None:
    captured = {}

    def compile_occurrences(model, occurrences, **kwargs):
        captured["model"] = model
        captured["occurrences"] = occurrences
        captured["kwargs"] = kwargs
        return ()

    monkeypatch.setattr(builder, "_compile_occurrences", compile_occurrences)
    model = object()
    assert (
        builder.compile_successor_fiber_trace_union(
            model,
            reversed(selected_records),
            time=0.5,
        )
        == ()
    )
    occurrences = captured["occurrences"]
    assert captured["model"] is model
    assert captured["kwargs"]["time"] == 0.5
    assert [row.cache_address.entry_index for row in occurrences] == [2, 2, 7, 7]
    assert [row.cache_address.progress_index for row in occurrences] == [0, 1, 0, 1]

    with pytest.raises(
        builder.SuccessorFiberCacheBuildError,
        match="repeats an exact packed entry",
    ):
        builder.compile_successor_fiber_trace_union(
            model,
            (selected_records[0], selected_records[0]),
        )


def test_build_load_exact_lookup_and_runtime_receipts(
    tmp_path,
    monkeypatch,
    selected_records,
    provenance,
    identity,
    model,
) -> None:
    manifest, receipt, compiled = _build(
        tmp_path,
        monkeypatch,
        selected_records,
        provenance,
        identity,
        model,
    )

    assert manifest.training_authorized is False
    assert manifest.gate_decision is None
    assert manifest.entries[0].included_entry_indices == (2, 7)
    assert manifest.entries[0].terminal_record_count == 2
    assert (tmp_path / receipt.manifest_relative_path).is_file()
    assert (
        t1_cache.t1_selected_trace_set_sha256(reversed(selected_records))
        == receipt.selected_trace_set_sha256
    )

    store = t1_cache.load_t1_successor_cache(
        tmp_path,
        receipt,
        expected_identity=identity,
    )
    selected = selected_records[0]
    packed_address = selected.corpus_address
    assert packed_address is not None
    source = selected.path.state_at(0)
    observed = store.require_packed_address(
        packed_address,
        progress_index=0,
        source_state=source,
    )
    expected = next(
        row
        for row in compiled
        if row.address.entry_index == packed_address.entry_index
        and row.address.progress_index == 0
    )
    assert observed == expected
    assert (
        store.require(
            expected.address,
            source_state=source,
        )
        == expected
    )
    assert store.receipts_for((expected.address,)) == (
        t1_cache.T1SuccessorCacheShardReceipt(
            packed_shard_content_sha256=_SHARD_SHA256,
            cache_content_sha256=manifest.entries[0].cache_content_sha256,
            encoded_sha256=manifest.entries[0].cache_file_sha256,
            record_count=4,
        ),
    )
    assert store.validate_all_leaves() == (
        t1_cache.T1SuccessorCacheShardReceipt(
            packed_shard_content_sha256=_SHARD_SHA256,
            cache_content_sha256=manifest.entries[0].cache_content_sha256,
            encoded_sha256=manifest.entries[0].cache_file_sha256,
            record_count=4,
        ),
    )


def test_build_publishes_leaf_and_manifest_when_hard_links_are_unsupported(
    tmp_path,
    monkeypatch,
    selected_records,
    provenance,
    identity,
    model,
) -> None:
    def reject_hard_link(*_args, **_kwargs):
        raise PermissionError(1, "operation not permitted")

    monkeypatch.setattr("os.link", reject_hard_link)
    manifest, receipt, _compiled = _build(
        tmp_path,
        monkeypatch,
        selected_records,
        provenance,
        identity,
        model,
    )

    assert (tmp_path / manifest.entries[0].cache_relative_path).is_file()
    assert (tmp_path / receipt.manifest_relative_path).is_file()
    loaded = t1_cache.load_t1_successor_cache(
        tmp_path,
        receipt,
        expected_identity=identity,
    )
    assert loaded.manifest == manifest
    assert loaded.validate_all_leaves()[0].record_count == 4


def test_manifest_and_leaf_hashes_fail_closed_without_enumeration_fallback(
    tmp_path,
    monkeypatch,
    selected_records,
    provenance,
    identity,
    model,
) -> None:
    manifest, receipt, _ = _build(
        tmp_path,
        monkeypatch,
        selected_records,
        provenance,
        identity,
        model,
    )
    with pytest.raises(
        t1_cache.EditingT1SuccessorCacheError,
        match="another frozen identity",
    ):
        t1_cache.load_t1_successor_cache(
            tmp_path,
            receipt,
            expected_identity=replace(
                identity,
                initial_model_state_sha256="8" * 64,
            ),
        )

    store = t1_cache.load_t1_successor_cache(
        tmp_path,
        receipt,
        expected_identity=identity,
    )
    leaf = tmp_path / manifest.entries[0].cache_relative_path
    leaf.unlink()

    def forbidden_compilation(*args, **kwargs):
        raise AssertionError("runtime lookup attempted support enumeration")

    monkeypatch.setattr(
        t1_cache,
        "compile_successor_fiber_trace_union",
        forbidden_compilation,
    )
    with pytest.raises(
        t1_cache.EditingT1SuccessorCacheError,
        match="leaf is absent",
    ):
        store.validate_all_leaves()
    address = selected_records[0].corpus_address
    assert address is not None
    with pytest.raises(
        t1_cache.EditingT1SuccessorCacheError,
        match="leaf is absent",
    ):
        store.require_packed_address(address, progress_index=0)


def test_exact_trace_miss_never_falls_back_to_compilation(
    tmp_path,
    monkeypatch,
    selected_records,
    provenance,
    identity,
    model,
) -> None:
    _, receipt, _ = _build(
        tmp_path,
        monkeypatch,
        selected_records,
        provenance,
        identity,
        model,
    )
    store = t1_cache.load_t1_successor_cache(
        tmp_path,
        receipt,
        expected_identity=identity,
    )
    selected = selected_records[0]
    address = selected.corpus_address
    assert address is not None
    missing = replace(address, entry_index=5, trace_id="unselected")

    def forbidden_compilation(*args, **kwargs):
        raise AssertionError("cache miss attempted support enumeration")

    monkeypatch.setattr(
        t1_cache,
        "compile_successor_fiber_trace_union",
        forbidden_compilation,
    )
    with pytest.raises(
        t1_cache.EditingT1SuccessorCacheError,
        match="absent from the selected T1 trace union",
    ):
        store.require_packed_address(missing, progress_index=0)


def test_complete_trace_leaf_rejects_sparse_compiler_output(
    tmp_path,
    monkeypatch,
    selected_records,
    provenance,
    identity,
    model,
) -> None:
    compiled = _cache_records(selected_records)
    sparse = tuple(row for row in compiled if not row.address.is_terminal)
    monkeypatch.setattr(
        t1_cache,
        "compile_successor_fiber_trace_union",
        lambda *args, **kwargs: sparse,
    )
    with pytest.raises(
        t1_cache.EditingT1SuccessorCacheError,
        match="complete immutable trace leaf",
    ):
        t1_cache.build_t1_successor_cache(
            tmp_path,
            model,
            selected_records,
            provenance_by_shard={_SHARD_SHA256: provenance},
            identity=identity,
        )
    assert not (tmp_path / "manifests").exists()


def test_manifest_tampered_included_indices_fails_even_with_recomputed_self_hash(
    tmp_path,
    monkeypatch,
    selected_records,
    provenance,
    identity,
    model,
) -> None:
    _, receipt, _ = _build(
        tmp_path,
        monkeypatch,
        selected_records,
        provenance,
        identity,
        model,
    )
    path = tmp_path / receipt.manifest_relative_path
    payload = json.loads(path.read_bytes())
    payload["entries"][0]["included_entry_indices"] = [2, 5]
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    with pytest.raises(
        t1_cache.EditingT1SuccessorCacheError,
        match="manifest contract is invalid",
    ):
        t1_cache.deserialize_t1_successor_cache_manifest(
            encoded,
            expected_manifest_sha256=payload["manifest_sha256"],
            expected_identity=identity,
        )


def test_build_is_byte_deterministic_under_selected_trace_order(
    tmp_path,
    monkeypatch,
    selected_records,
    provenance,
    identity,
    model,
) -> None:
    compiled = _cache_records(selected_records)
    monkeypatch.setattr(
        t1_cache,
        "compile_successor_fiber_trace_union",
        lambda *args, **kwargs: compiled,
    )
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_manifest, first_receipt = t1_cache.build_t1_successor_cache(
        first_root,
        model,
        selected_records,
        provenance_by_shard={_SHARD_SHA256: provenance},
        identity=identity,
    )
    second_manifest, second_receipt = t1_cache.build_t1_successor_cache(
        second_root,
        model,
        reversed(selected_records),
        provenance_by_shard={_SHARD_SHA256: provenance},
        identity=identity,
    )

    assert first_manifest == second_manifest
    assert first_receipt == second_receipt
    assert (first_root / first_receipt.manifest_relative_path).read_bytes() == (
        second_root / second_receipt.manifest_relative_path
    ).read_bytes()


def test_manifest_receipt_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="safe POSIX-relative path"):
        t1_cache.T1SuccessorCacheManifestReceipt(
            manifest_relative_path="../manifest.json",
            manifest_sha256="1" * 64,
            manifest_file_sha256="2" * 64,
            manifest_file_bytes=1,
            selected_trace_set_sha256="3" * 64,
            initial_model_state_sha256="4" * 64,
        )
