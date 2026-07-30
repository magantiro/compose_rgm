"""The real Gate-0 runtime stays exact, bounded, and non-authorizing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.charge_policy import CHARGE_POLICY_VERSION
from compose_v4.data.packed_trace_store import (
    AddressedPackedTrace,
    PackedTraceAddress,
    manifest_path_for,
)
from compose_v4.data.provenance_overlay import overlay_path_for
from compose_v4.data.successor_fiber_cache import (
    SuccessorFiberCacheProvenance,
    fiber_compiler_implementation_hash,
)
from compose_v4.experiments import editing_gate_zero_runtime as runtime
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.editing_step_zero_gate import (
    InitializationParityReport,
    StepZeroSuccessorSupportReport,
    audit_successor_support_before_optimization,
)
from compose_v4.experiments.editing_step_zero_probe import (
    GateZeroSuccessorProbeConfig,
    build_gate_zero_successor_probe,
)
from compose_v4.experiments.ringcore_semantic_sidecar import (
    LoadedSemanticCellSidecar,
    SemanticSidecarRow,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import build_typed_ring_catalog


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _contract_payload(
    *,
    shard_manifest_sha256: str = "1" * 64,
    sidecar_config_sha256: str = "2" * 64,
    sidecar_provenance_sha256: str = "3" * 64,
    census_sha256: str = "4" * 64,
    unified_sha256: str = "5" * 64,
    representability_sha256: str = "6" * 64,
    nonempty_cells: int = 1,
) -> dict:
    body = {
        "schema": runtime.EDITING_GATE_ZERO_RUNTIME_CONTRACT_SCHEMA,
        "schema_version": runtime.EDITING_GATE_ZERO_RUNTIME_CONTRACT_VERSION,
        "status": runtime.EDITING_GATE_ZERO_RUNTIME_CONTRACT_STATUS,
        "training_authorized": False,
        "sidecar": {
            "manifest_sha256": shard_manifest_sha256,
            "config_sha256": sidecar_config_sha256,
            "provenance_sha256": sidecar_provenance_sha256,
            "semantic_census_sha256": census_sha256,
            "nonempty_semantic_cell_count": nonempty_cells,
            "unified_packed_manifest_sha256": unified_sha256,
            "representability_overlay_sha256": representability_sha256,
        },
        "model": {
            "seed": 17,
            "max_atoms": 40,
            "hidden_dim": 256,
            "message_passing_steps": 6,
            "mark_dim": 32,
            "atom_vocabulary_class_count": 15,
            "catalog_fingerprint": (runtime.PRODUCTION_RINGCORE_CATALOG_FINGERPRINT),
            "operator_capability_fingerprint": "e787c852410c6b63",
            "enable_ring_restates": True,
            "enable_cyclic_graft": True,
            "enable_heteroatom_scan": True,
            "enable_ring_opening": True,
            "enable_cycle_ops": True,
            "enable_ring_grow_macro": False,
            "enable_ring_system_delete": False,
        },
        "required_families": list(runtime.PRODUCTION_ACTIVE_FAMILIES),
        "audit_batch_size": 2,
    }
    return {**body, "contract_sha256": _canonical_sha256(body)}


def _state(smiles: str, *, slots: int = 6):
    return pad_molecular_graph(smiles_to_molecular_graph(smiles), slots)


def _record(*, shard_sha256: str) -> PathRecord:
    trace = compile_carbon_tree_to_target(_state("C"), _state("CC"))
    path = TraceProgressCTMC(trace)
    address = PackedTraceAddress(
        packed_shard_content_sha256=shard_sha256,
        packed_shard_name="shard_0000.jsonl.gz",
        entry_index=0,
        trace_id="fixture-trace",
        layer="corruption",
        partition="validation",
        source_key=canonical_state_key(path.state_at(0)),
        target_key=canonical_state_key(path.state_at(path.path_length)),
        path_length=path.path_length,
    )
    return PathRecord(
        target_key=address.target_key,
        path=path,
        corpus_address=address,
    )


def _tiny_model(record: PathRecord) -> FactorizedTraceletRateModel:
    torch.manual_seed(17)
    return FactorizedTraceletRateModel(
        build_typed_ring_catalog((record.path.trace,)),
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


def _cache_provenance(
    model: FactorizedTraceletRateModel,
    *,
    shard_sha256: str,
) -> SuccessorFiberCacheProvenance:
    return SuccessorFiberCacheProvenance(
        operator_registry_hash="operator-fixture",
        capability_hash=model.operator_capabilities.fingerprint(),
        support_signature_sha256="a" * 64,
        canonicalization_version="canonical-state-key-fixture",
        canonicalizer_contract_sha256="b" * 64,
        packed_corpus_schema="compose.data.packed_trace",
        packed_corpus_schema_version=1,
        packed_shard_content_sha256=shard_sha256,
        packed_manifest_sha256="c" * 64,
        packed_provenance_overlay_sha256="d" * 64,
        unified_packed_manifest_sha256="e" * 64,
        representability_overlay_sha256="f" * 64,
        coordinate_schema_version=1,
        tensorization_implementation_hash="tensorization-fixture",
        fiber_compiler_implementation_hash=fiber_compiler_implementation_hash(),
    )


def test_frozen_runtime_contract_is_self_hashed_and_non_authorizing(
    tmp_path: Path,
) -> None:
    payload = _contract_payload()
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload))
    contract = runtime.load_gate_zero_runtime_contract(path)

    assert contract.sha256 == payload["contract_sha256"]
    assert contract.training_authorized is False
    with pytest.raises(
        runtime.EditingGateZeroRuntimeError,
        match="never authorizes",
    ):
        contract.assert_training_authorized()

    tampered = dict(payload)
    tampered["audit_batch_size"] = 3
    path.write_text(json.dumps(tampered))
    with pytest.raises(
        runtime.EditingGateZeroRuntimeError,
        match="self-hash",
    ):
        runtime.load_gate_zero_runtime_contract(path)


def test_gate_zero_support_signature_binds_exact_charge_policy() -> None:
    record = _record(shard_sha256="1" * 64)
    model = _tiny_model(record)
    signature = runtime._support_signature_payload(model, max_atoms=6)

    assert signature["charge_policy"] == CHARGE_POLICY_VERSION


def test_exact_validation_loader_binds_every_fixture_byte(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "transfer"
    packed_path = root / "edit_packed_v1" / "corruption" / "validation" / "shard_0000.jsonl.gz"
    packed_path.parent.mkdir(parents=True)
    packed_path.write_bytes(b"packed-fixture")
    packed_sha256 = hashlib.sha256(packed_path.read_bytes()).hexdigest()
    packed_manifest = manifest_path_for(packed_path)
    packed_manifest.write_bytes(b"manifest-fixture")
    packed_manifest_sha256 = hashlib.sha256(packed_manifest.read_bytes()).hexdigest()
    packed_overlay = overlay_path_for(packed_path)
    packed_overlay.write_bytes(b"overlay-fixture")
    packed_overlay_sha256 = hashlib.sha256(packed_overlay.read_bytes()).hexdigest()
    unified = root / "UNIFIED_PACKED_MANIFEST.json"
    unified.write_bytes(b"unified-fixture")
    unified_sha256 = hashlib.sha256(unified.read_bytes()).hexdigest()
    representability = root / "REPRESENTABILITY_OVERLAY.json"
    representability.write_bytes(b"representability-fixture")
    representability_sha256 = hashlib.sha256(representability.read_bytes()).hexdigest()

    record = _record(shard_sha256=packed_sha256)
    assert record.corpus_address is not None
    rows = (
        SemanticSidecarRow(
            packed_shard_content_sha256=packed_sha256,
            entry_index=0,
            progress_index=0,
            semantic_cell_id="fixture-cell",
        ),
        SemanticSidecarRow(
            packed_shard_content_sha256=packed_sha256,
            entry_index=0,
            progress_index=1,
            semantic_cell_id=None,
        ),
    )
    sidecar_manifest_sha256 = "1" * 64
    sidecar_config_sha256 = "2" * 64
    sidecar_provenance_sha256 = "3" * 64
    census_sha256 = "4" * 64
    sidecar = LoadedSemanticCellSidecar(
        rows=rows,
        manifest={
            "manifest_sha256": sidecar_manifest_sha256,
            "source_sha256": "9" * 64,
            "semantic_census_sha256": census_sha256,
            "counts": {
                "traces": 1,
                "rows": 2,
                "nonempty_cells": 1,
                "teacher_family_rows": {family: 1 for family in runtime.PRODUCTION_ACTIVE_FAMILIES},
            },
            "provenance": {
                "unified_packed_manifest_sha256": unified_sha256,
                "representability_overlay_sha256": (representability_sha256),
                "source_shards": [
                    {
                        "packed_shard_content_sha256": packed_sha256,
                        "packed_shard_name": packed_path.name,
                        "layer": "corruption",
                        "partition": "validation",
                        "packed_manifest_sha256": (packed_manifest_sha256),
                        "provenance_overlay_sha256": (packed_overlay_sha256),
                        "packed_entry_count": 1,
                        "effective_trace_count": 1,
                        "progress_row_count": 2,
                        "excluded_entry_indices": [],
                    }
                ],
            },
        },
    )
    contract = runtime.GateZeroRuntimeContract(
        _contract_payload(
            shard_manifest_sha256=sidecar_manifest_sha256,
            sidecar_config_sha256=sidecar_config_sha256,
            sidecar_provenance_sha256=sidecar_provenance_sha256,
            census_sha256=census_sha256,
            unified_sha256=unified_sha256,
            representability_sha256=representability_sha256,
        )
    )
    monkeypatch.setattr(
        runtime,
        "read_semantic_cell_sidecar",
        lambda *_args, **_kwargs: sidecar,
    )
    addressed = AddressedPackedTrace(
        address=record.corpus_address,
        trace=record.path.trace,
        path=record.path,
    )
    monkeypatch.setattr(
        runtime,
        "read_addressed_packed_shard",
        lambda *_args, **_kwargs: iter((addressed,)),
    )
    loaded = runtime.load_frozen_validation_source(
        contract=contract,
        transfer_root=root,
        sidecar_path=tmp_path / "sidecar.gz",
        sidecar_manifest_path=tmp_path / "sidecar.manifest.json",
    )

    assert loaded.records == (record,)
    assert len(loaded.sidecar.rows) == 2
    assert loaded.bindings[0].spec.packed_shard_content_sha256 == packed_sha256
    model = _tiny_model(record)
    cache_provenance = runtime.build_exact_cache_provenance(
        model,
        loaded,
        selected_shard_digests=(packed_sha256,),
        max_atoms=40,
    )[packed_sha256]
    assert cache_provenance.packed_manifest_sha256 == packed_manifest_sha256
    assert cache_provenance.packed_provenance_overlay_sha256 == packed_overlay_sha256
    assert cache_provenance.operator_registry_hash == "9197401e8dc3a7ae"
    assert (
        cache_provenance.fiber_compiler_implementation_hash == fiber_compiler_implementation_hash()
    )

    packed_path.write_bytes(b"tampered")
    with pytest.raises(
        runtime.EditingGateZeroRuntimeError,
        match="packed shard.*hash mismatch",
    ):
        runtime.load_frozen_validation_source(
            contract=contract,
            transfer_root=root,
            sidecar_path=tmp_path / "sidecar.gz",
            sidecar_manifest_path=tmp_path / "sidecar.manifest.json",
        )


def test_probe_audit_batches_retain_every_progress_row() -> None:
    shard_sha256 = "7" * 64
    record = _record(shard_sha256=shard_sha256)
    model = _tiny_model(record)
    assert record.corpus_address is not None
    semantic_cells = {
        (
            shard_sha256,
            0,
            0,
        ): "fixture-cell",
        (
            shard_sha256,
            0,
            1,
        ): None,
    }
    probe = build_gate_zero_successor_probe(
        model,
        (record,),
        semantic_cell_ids=semantic_cells,
        config=GateZeroSuccessorProbeConfig(
            required_families=("atom_insert",),
            required_semantic_cells=("fixture-cell",),
        ),
        provenance_by_shard={
            shard_sha256: _cache_provenance(
                model,
                shard_sha256=shard_sha256,
            )
        },
    )
    batches = runtime.reconstruct_probe_audit_batches(
        model,
        probe,
        batch_size=1,
    )

    assert len(batches) == 2
    assert tuple(
        address.progress_index for batch in batches for address in batch.cache_addresses
    ) == (0, 1)
    assert batches[0].semantic_cell_ids == ("fixture-cell",)
    assert batches[1].semantic_cell_ids == (None,)
    assert float(batches[0].mark_batch.teacher_rates[0]) > 0.0
    assert float(batches[1].mark_batch.teacher_rates[0]) == 0.0
    report = audit_successor_support_before_optimization(
        model,
        batches,
        required_families=("atom_insert",),
        required_semantic_cells=("fixture-cell",),
    )
    assert report.row_count == 2
    assert report.nonterminal_row_count == 1
    assert report.terminal_row_count == 1


def test_evidence_freeze_is_self_hashed_and_never_overwrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = b'{"cache":"fixture"}'
    provenance = _cache_provenance(
        _tiny_model(_record(shard_sha256="7" * 64)),
        shard_sha256="7" * 64,
    )
    shard = SimpleNamespace(
        packed_shard_content_sha256="7" * 64,
        encoded_sha256=hashlib.sha256(encoded).hexdigest(),
        content_sha256="8" * 64,
        encoded=encoded,
        cache=SimpleNamespace(records=(1, 2), provenance=provenance),
    )
    contract = runtime.GateZeroRuntimeContract(_contract_payload())
    result = runtime.GateZeroRuntimeResult(
        contract=contract,
        source=SimpleNamespace(),
        model=SimpleNamespace(),
        initialization_parity=InitializationParityReport(
            regime="scratch",
            transfer_plan_sha256="a" * 64,
            exact_tensor_count=0,
            copied_row_count=0,
            fresh_row_count=0,
            fresh_tensor_count=1,
            exact_tensor_names=(),
            fresh_tensor_names=("weight",),
        ),
        probe=SimpleNamespace(cache_shards=(shard,)),
        support_report=StepZeroSuccessorSupportReport(
            batch_count=1,
            row_count=2,
            nonterminal_row_count=1,
            terminal_row_count=1,
            unique_source_state_count=2,
            teacher_examples_by_family={"atom_insert": 1},
            candidate_marks_by_family={"atom_insert": 1},
            productive_successors_by_family={"atom_insert": 1},
            examples_by_semantic_cell={"fixture-cell": 1},
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_evidence_body",
        lambda _result, *, cache_entries: {
            "schema": runtime.EDITING_GATE_ZERO_RUNTIME_EVIDENCE_SCHEMA,
            "schema_version": 1,
            "status": runtime.EDITING_GATE_ZERO_RUNTIME_STATUS,
            "training_authorized": False,
            "training_decision": None,
            "quality_thresholds": None,
            "cache_artifacts": cache_entries,
        },
    )
    cache_directory = tmp_path / "caches"
    manifest_path = tmp_path / "evidence.json"
    manifest = runtime.freeze_gate_zero_runtime_evidence(
        result,
        cache_directory=cache_directory,
        evidence_manifest_path=manifest_path,
    )

    body = {key: value for key, value in manifest.items() if key != "evidence_sha256"}
    assert manifest["evidence_sha256"] == _canonical_sha256(body)
    assert manifest["training_authorized"] is False
    assert manifest["training_decision"] is None
    assert (
        runtime.freeze_gate_zero_runtime_evidence(
            result,
            cache_directory=cache_directory,
            evidence_manifest_path=manifest_path,
        )
        == manifest
    )

    manifest_path.write_text("different")
    with pytest.raises(FileExistsError, match="different content"):
        runtime.freeze_gate_zero_runtime_evidence(
            result,
            cache_directory=cache_directory,
            evidence_manifest_path=manifest_path,
        )


def test_checked_in_contract_names_real_256_by_6_scratch_model() -> None:
    contract = runtime.load_gate_zero_runtime_contract(
        Path("configs/editing_gate_zero_runtime_v2.json")
    )
    assert contract.model["hidden_dim"] == 256
    assert contract.model["message_passing_steps"] == 6
    assert contract.model["atom_vocabulary_class_count"] == 15
    assert contract.model["enable_cycle_ops"] is True
    assert contract.model["enable_ring_grow_macro"] is False
    assert contract.model["enable_ring_system_delete"] is False
    assert contract.sidecar["nonempty_semantic_cell_count"] == 110


def test_runtime_contract_rejects_ring_grow_or_missing_family() -> None:
    payload = _contract_payload()
    body = {key: value for key, value in payload.items() if key != "contract_sha256"}
    body["model"] = {
        **body["model"],
        "enable_ring_grow_macro": True,
    }
    drifted = {**body, "contract_sha256": _canonical_sha256(body)}
    with pytest.raises(
        runtime.EditingGateZeroRuntimeError,
        match="production 256x6",
    ):
        runtime.GateZeroRuntimeContract(drifted)

    body["model"] = payload["model"]
    body["required_families"] = body["required_families"][:-1]
    missing = {**body, "contract_sha256": _canonical_sha256(body)}
    with pytest.raises(
        runtime.EditingGateZeroRuntimeError,
        match="eight active pilot families",
    ):
        runtime.GateZeroRuntimeContract(missing)
