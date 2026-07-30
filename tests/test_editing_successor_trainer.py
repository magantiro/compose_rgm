"""Launcher-level guards for bounded canonical-successor training."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from compose_v4.chem.molecular_graph import (
    ORGANIC_VOCABULARY,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.data.packed_trace_store import (
    PACKED_STORE_SCHEMA,
    PACKED_STORE_SCHEMA_VERSION,
    PackedTraceAddress,
)
from compose_v4.data.provenance_overlay import (
    tensorization_implementation_hash,
)
from compose_v4.data.sharded_successor_fiber_cache import (
    SuccessorFiberCacheCompatibility,
)
from compose_v4.data.successor_fiber_cache import (
    SUCCESSOR_FIBER_CACHE_SCHEMA,
    SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION,
    fiber_compiler_implementation_hash,
)
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.editing_successor_trainer import (
    CANONICAL_SUCCESSOR_P50_STEPS,
    CanonicalSuccessorRuntime,
    CanonicalSuccessorRuntimeConfig,
    EditingSuccessorTrainerError,
    PRODUCTION_ACTION_TABLE_VOCABULARY,
    _source_set_sha256,
    _validate_live_compatibility,
    _mark_collator,
    assert_records_covered_by_successor_cache,
    build_canonical_successor_loader_factory,
)
from compose_v4.experiments.factorized_successor_objective import (
    CanonicalSuccessorTrainingObjective,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    MARK_RULE_NAMES,
    FactorizedTraceletRateModel,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.trace_shard import (
    TRACE_SCHEMA,
    TRACE_SCHEMA_VERSION,
)
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.typed_ring_catalog import (
    build_typed_ring_catalog,
    ring_catalog_fingerprint,
)
from scripts.train_tracelet_cnof_gate import (
    _checkpoint_ring_restates_capability,
    _resolve_ring_restates_capability,
    _resolve_ring_system_delete_capability,
    _ring_core_v1_identity_applies,
    main as training_main,
)

_PACKED_SHA256 = "1" * 64


def _record(*, partition: str = "train") -> PathRecord:
    source = pad_molecular_graph(smiles_to_molecular_graph("C"), 6)
    target = pad_molecular_graph(smiles_to_molecular_graph("CC"), 6)
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=True,
        align_source=True,
        flexible_size=True,
    )
    path = TraceProgressCTMC(trace)
    address = PackedTraceAddress(
        packed_shard_content_sha256=_PACKED_SHA256,
        packed_shard_name="tiny.jsonl.gz",
        entry_index=0,
        trace_id="tiny-0",
        layer="mmp_analogue",
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


def _model(
    *,
    enable_ring_system_delete: bool = True,
) -> FactorizedTraceletRateModel:
    record = _record()
    catalog = build_typed_ring_catalog((record.path.trace,))
    torch.manual_seed(7)
    return FactorizedTraceletRateModel(
        catalog,
        hidden_dim=16,
        message_passing_steps=1,
        enable_ring_restates=True,
        enable_cyclic_graft=True,
        enable_heteroatom_scan=True,
        enable_ring_opening=True,
        enable_cycle_ops=True,
        enable_ring_grow_macro=False,
        enable_ring_system_delete=enable_ring_system_delete,
        atom_vocabulary=ORGANIC_VOCABULARY,
    )


def _spec(*, partition: str = "train", excluded=()):
    return SimpleNamespace(
        packed_shard_content_sha256=_PACKED_SHA256,
        packed_shard_name="tiny.jsonl.gz",
        layer="mmp_analogue",
        partition=partition,
        packed_entry_count=1,
        excluded_entry_indices=tuple(excluded),
    )


def _compatibility(model, *, capability_hash: str | None = None):
    from rdkit import rdBase

    from compose_v4.experiments import editing_successor_trainer as module

    operator_registry_source_sha256 = _source_set_sha256(
        module._OPERATOR_REGISTRY_SOURCES
    )
    operator_registry_hash = operator_registry_source_sha256[:16]
    canonicalizer_hash = _source_set_sha256(module._CANONICALIZER_CONTRACT_SOURCES)
    support_signature = {
        "schema": "compose.editing.gate_zero_support_signature",
        "schema_version": 1,
        "operator_registry_hash": operator_registry_hash,
        "operator_registry_source_sha256": operator_registry_source_sha256,
        "operator_capability_fingerprint": (model.operator_capabilities.fingerprint()),
        "ordered_family_vocabulary": list(MARK_RULE_NAMES),
        "ordered_action_table_vocabulary": list(PRODUCTION_ACTION_TABLE_VOCABULARY),
        "atom_vocabulary": [
            [str(element), int(valence)] for element, valence in model.atom_vocabulary.classes
        ],
        "max_atoms": 6,
        "atom_insert_arity_support": [0, 1],
        "catalog_fingerprint": ring_catalog_fingerprint(model.ring_catalog),
        "enable_ring_restates": model.enable_ring_restates,
        "enable_cyclic_graft": model.enable_cyclic_graft,
        "enable_heteroatom_scan": model.enable_heteroatom_scan,
        "enable_ring_opening": model.enable_ring_opening,
        "enable_cycle_ops": model.enable_cycle_ops,
        "enable_ring_grow_macro": model.enable_ring_grow_macro,
        "enable_ring_system_delete": model.enable_ring_system_delete,
        "embedded_jump_chain_policy": "productive_canonical_successors",
    }
    return SuccessorFiberCacheCompatibility.from_payload(
        {
            "successor_cache_schema": SUCCESSOR_FIBER_CACHE_SCHEMA,
            "successor_cache_schema_version": (SUCCESSOR_FIBER_CACHE_SCHEMA_VERSION),
            "support_signature": support_signature,
            "ordered_family_vocabulary": list(MARK_RULE_NAMES),
            "ordered_action_table_vocabulary": list(PRODUCTION_ACTION_TABLE_VOCABULARY),
            "coordinate_schema_version": 1,
            "operator_registry_hash": operator_registry_hash,
            "capability_hash": (
                model.operator_capabilities.fingerprint()
                if capability_hash is None
                else capability_hash
            ),
            "canonicalization_version": "canonical-state-key-v1",
            "canonicalizer_contract_sha256": canonicalizer_hash,
            "rdkit_version": rdBase.rdkitVersion,
            "executor_implementation_hash": "executor-v1",
            "action_enumerator_implementation_hash": "enumerator-v1",
            "fiber_compiler_implementation_hash": (fiber_compiler_implementation_hash()),
            "persistent_state_digest_schema": ("compose.chem.persistent_slot_state"),
            "persistent_state_digest_schema_version": 1,
            "packed_corpus_schema": PACKED_STORE_SCHEMA,
            "packed_corpus_schema_version": PACKED_STORE_SCHEMA_VERSION,
            "trace_codec_schema": TRACE_SCHEMA,
            "trace_codec_schema_version": TRACE_SCHEMA_VERSION,
            "tensorization_implementation_hash": (tensorization_implementation_hash()),
        }
    )


def test_cache_record_preflight_has_no_missing_shard_fallback() -> None:
    record = _record()
    inventory = SimpleNamespace(by_packed_digest={_PACKED_SHA256: _spec()})
    assert_records_covered_by_successor_cache(
        (record,),
        inventory,
        partition="train",
    )

    missing = SimpleNamespace(by_packed_digest={})
    with pytest.raises(
        EditingSuccessorTrainerError,
        match="absent from the successor-cache inventory",
    ):
        assert_records_covered_by_successor_cache(
            (record,),
            missing,
            partition="train",
        )

    excluded = SimpleNamespace(by_packed_digest={_PACKED_SHA256: _spec(excluded=(0,))})
    with pytest.raises(
        EditingSuccessorTrainerError,
        match="exact successor-cache shard envelope",
    ):
        assert_records_covered_by_successor_cache(
            (record,),
            excluded,
            partition="train",
        )


def test_live_model_capability_mismatch_fails_before_training() -> None:
    model = _model()
    valid = SimpleNamespace(compatibility=_compatibility(model))
    _validate_live_compatibility(valid, model, max_atoms=6)

    mismatched = SimpleNamespace(compatibility=_compatibility(model, capability_hash="different"))
    with pytest.raises(
        EditingSuccessorTrainerError,
        match="compatibility differs",
    ):
        _validate_live_compatibility(mismatched, model, max_atoms=6)


def test_ring_system_delete_choice_is_explicit_and_threads_to_collator() -> None:
    assert (
        _resolve_ring_system_delete_capability(
            None,
            canonical_successor_backend=False,
        )
        is True
    )
    with pytest.raises(ValueError, match="requires an explicit ring-system-delete"):
        _resolve_ring_system_delete_capability(
            None,
            canonical_successor_backend=True,
        )
    assert (
        _resolve_ring_system_delete_capability(
            False,
            canonical_successor_backend=True,
        )
        is False
    )
    assert _ring_core_v1_identity_applies(
        cycle_op_mix=True,
        disable_ring_grow_macro=True,
        enable_ring_restates=True,
        enable_ring_system_delete=True,
    )
    assert not _ring_core_v1_identity_applies(
        cycle_op_mix=True,
        disable_ring_grow_macro=True,
        enable_ring_restates=True,
        enable_ring_system_delete=False,
    )

    disabled = _model(enable_ring_system_delete=False)
    collator = _mark_collator(disabled, use_aromatic_bond_view=True)
    assert collator.compute_ring_system_delete is False
    valid_disabled = SimpleNamespace(compatibility=_compatibility(disabled))
    _validate_live_compatibility(valid_disabled, disabled, max_atoms=6)

    enabled_cache = SimpleNamespace(compatibility=_compatibility(_model()))
    with pytest.raises(
        EditingSuccessorTrainerError,
        match="support signature does not match",
    ):
        _validate_live_compatibility(enabled_cache, disabled, max_atoms=6)


def test_ring_restate_choice_is_explicit_with_historical_metadata_fallback() -> None:
    assert (
        _resolve_ring_restates_capability(
            None,
            canonical_successor_backend=False,
            corrupted_prior_mix=True,
        )
        is True
    )
    assert (
        _resolve_ring_restates_capability(
            None,
            canonical_successor_backend=False,
            corrupted_prior_mix=False,
        )
        is False
    )
    with pytest.raises(ValueError, match="requires an explicit ring-restate"):
        _resolve_ring_restates_capability(
            None,
            canonical_successor_backend=True,
            corrupted_prior_mix=True,
        )
    assert (
        _resolve_ring_restates_capability(
            False,
            canonical_successor_backend=True,
            corrupted_prior_mix=True,
        )
        is False
    )
    assert _checkpoint_ring_restates_capability(
        {"corrupted_prior_mix": True}
    )
    assert not _checkpoint_ring_restates_capability(
        {
            "corrupted_prior_mix": True,
            "enable_ring_restates": False,
        }
    )
    with pytest.raises(ValueError, match="literal Boolean"):
        _checkpoint_ring_restates_capability(
            {"enable_ring_restates": "false"}
        )
    for malformed in ("false", 0, 1, None):
        with pytest.raises(
            ValueError,
            match="corrupted_prior_mix.*literal Boolean",
        ):
            _checkpoint_ring_restates_capability(
                {"corrupted_prior_mix": malformed}
            )
    assert not _ring_core_v1_identity_applies(
        cycle_op_mix=True,
        disable_ring_grow_macro=True,
        enable_ring_restates=False,
        enable_ring_system_delete=True,
    )


def test_runtime_and_loader_authorize_only_exact_p50(
    tmp_path,
) -> None:
    config = CanonicalSuccessorRuntimeConfig(
        cache_root=tmp_path,
        inventory_path=tmp_path / "inventory.json",
        inventory_sha256="1" * 64,
        compatibility_sha256="2" * 64,
        source_corpus_inventory_sha256="3" * 64,
        unified_packed_manifest_path=tmp_path / "unified.json",
        representability_overlay_path=tmp_path / "overlay.json",
        semantic_sidecar_path=tmp_path / "cells.jsonl.gz",
        semantic_sidecar_manifest_path=tmp_path / "cells.manifest.json",
        semantic_sidecar_manifest_sha256="4" * 64,
        semantic_sidecar_config_sha256="5" * 64,
        semantic_sidecar_provenance_sha256="6" * 64,
    )
    runtime = CanonicalSuccessorRuntime(
        cache=None,
        inventory=SimpleNamespace(
            inventory_sha256="1" * 64,
            compatibility=SimpleNamespace(compatibility_sha256="2" * 64),
            source_corpus_inventory_sha256="3" * 64,
            unified_packed_manifest_sha256="4" * 64,
            representability_overlay_sha256="5" * 64,
            storage_backend="indexed_jsonl_gzip_v1",
        ),
        validation_sidecar=None,
        config=config,
    )
    assert runtime.full_training_authorized is False
    assert (
        runtime.checkpoint_metadata()["successor_training_scope"]
        == "bounded_development_p50_only"
    )
    with pytest.raises(
        EditingSuccessorTrainerError,
        match="cannot authorize a full run",
    ):
        runtime.assert_full_training_authorized()

    with pytest.raises(
        EditingSuccessorTrainerError,
        match="P500 is blocked",
    ):
        build_canonical_successor_loader_factory(
            runtime,
            None,
            (),
            steps=500,
            batch_size=1,
            seed=1,
            workers=0,
            data_prefetch_factor=1,
            late_time_fraction=0.5,
            operational_horizon=2.0,
            progress_stratification_fraction=0.0,
            use_aromatic_bond_view=True,
            ring_electronic_mode="factorized_local",
            target_property_conditions=None,
            condition_dropout_probability=0.0,
            record_index_sampler=None,
        )
    with pytest.raises(
        EditingSuccessorTrainerError,
        match="exactly P50",
    ):
        build_canonical_successor_loader_factory(
            runtime,
            None,
            (),
            steps=CANONICAL_SUCCESSOR_P50_STEPS - 1,
            batch_size=1,
            seed=1,
            workers=0,
            data_prefetch_factor=1,
            late_time_fraction=0.5,
            operational_horizon=2.0,
            progress_stratification_fraction=0.0,
            use_aromatic_bond_view=True,
            ring_electronic_mode="factorized_local",
            target_property_conditions=None,
            condition_dropout_probability=0.0,
            record_index_sampler=None,
        )


def test_hazard_configuration_never_changes_identity_selector() -> None:
    identity = CanonicalSuccessorTrainingObjective(
        mode="productive_identity",
        hazard_weight=0.0,
    )
    with_hazard = CanonicalSuccessorTrainingObjective(
        mode="productive_identity_plus_hazard",
        hazard_weight=0.1,
    )
    assert identity.selection_metric == ("production_weighted_canonical_successor_nll")
    assert with_hazard.selection_metric == identity.selection_metric
    assert identity.name != with_hazard.name


def _canonical_launcher_argv(
    *,
    skip_rollouts: bool,
    steps: int = 50,
    checkpoint: str = "not-written-before-launch-guards.pt",
) -> list[str]:
    argv = [
        "train_tracelet_cnof_gate.py",
        "not-read-before-launch-guards.smi",
        "--factorized-training-objective",
        "canonical_successor",
        "--training-backend",
        "factorized_marks",
        "--corrupted-prior-mix",
        "--cycle-op-mix",
        "--disable-ring-grow-macro",
        "--organic-vocabulary",
        "--max-atoms",
        "40",
        "--enable-ring-restates",
        "--enable-ring-system-delete",
        "--steps",
        str(steps),
        "--checkpoint",
        checkpoint,
    ]
    if skip_rollouts:
        argv.append("--skip-rollouts")
    return argv


def test_canonical_launcher_rejects_legacy_rollout_path(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        _canonical_launcher_argv(skip_rollouts=False),
    )
    with pytest.raises(ValueError, match="require --skip-rollouts"):
        training_main()


def test_canonical_launcher_null_p50_thresholds_fail_before_data_load(
    monkeypatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    gate = root / "configs" / "editing_training_v2_gate.json"
    gate_sha256 = hashlib.sha256(gate.read_bytes()).hexdigest()
    argv = _canonical_launcher_argv(skip_rollouts=True)
    argv.extend(
        (
            "--editing-training-gate-contract",
            str(gate),
            "--editing-training-gate-contract-sha256",
            gate_sha256,
            "--successor-p50-initialization-regime",
            "scratch",
        )
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(
        ValueError,
        match="blocks P50 before loader construction",
    ):
        training_main()


def test_canonical_launcher_rejects_preexisting_completion_target_before_data(
    monkeypatch,
    tmp_path,
) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_text("pre-existing bytes")
    monkeypatch.setattr(
        sys,
        "argv",
        _canonical_launcher_argv(
            skip_rollouts=True,
            checkpoint=str(checkpoint),
        ),
    )
    with pytest.raises(
        RuntimeError,
        match="targets already exist",
    ):
        training_main()
    assert checkpoint.read_text() == "pre-existing bytes"


def test_canonical_launcher_does_not_silently_promote_p50_to_p500(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        _canonical_launcher_argv(skip_rollouts=True, steps=500),
    )
    with pytest.raises(ValueError, match="exactly the P50 sentinel"):
        training_main()
