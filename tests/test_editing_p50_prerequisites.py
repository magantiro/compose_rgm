from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from compose_v4.data.charge_policy import CHARGE_POLICY_VERSION
from compose_v4.experiments.editing_gate_zero_runtime import (
    EDITING_GATE_ZERO_RUNTIME_EVIDENCE_SCHEMA,
    EDITING_GATE_ZERO_RUNTIME_EVIDENCE_VERSION,
    EDITING_GATE_ZERO_RUNTIME_STATUS,
)
from compose_v4.experiments.editing_p50_prerequisites import (
    P50_DISABLED_FAMILIES,
    P50_RECIPE_LAUNCH_FIELDS,
    P50_RECIPE_SCHEMA,
    P50_RECIPE_SCHEMA_VERSION,
    P50_RECIPE_STATUS,
    T1_P50_DECISION_PASS_STATUS,
    T1_P50_DECISION_SCHEMA,
    T1_P50_DECISION_SCHEMA_VERSION,
    EditingP50PrerequisiteError,
    validate_gate_zero_structural_evidence,
    validate_p50_recipe,
    validate_t1_p50_decision,
)
from compose_v4.experiments.successor_micro_overfit import (
    RINGCORE_EDITING_FAMILIES,
)
from scripts.train_tracelet_cnof_gate import _p50_launch_recipe_projection


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _admission():
    return SimpleNamespace(
        manifest_file_sha256="1" * 64,
        inventory_sha256="2" * 64,
        unified_packed_manifest_sha256="3" * 64,
        support_contract_sha256="4" * 64,
    )


def _gate_zero(admission) -> dict[str, object]:
    support_signature = {
        "charge_policy": CHARGE_POLICY_VERSION,
        "enable_ring_restates": True,
        "enable_cycle_ops": True,
        "enable_ring_opening": True,
        "enable_ring_grow_macro": False,
        "enable_ring_system_delete": False,
        "max_atoms": 40,
        "atom_insert_arity_support": [0, 1],
        "embedded_jump_chain_policy": "productive_canonical_successors",
        "ordered_family_vocabulary": [
            *RINGCORE_EDITING_FAMILIES,
            "ring_system_delete",
            "ring_system_grow",
        ],
    }
    positive = {family: 1 for family in RINGCORE_EDITING_FAMILIES}
    body = {
        "schema": EDITING_GATE_ZERO_RUNTIME_EVIDENCE_SCHEMA,
        "schema_version": EDITING_GATE_ZERO_RUNTIME_EVIDENCE_VERSION,
        "status": EDITING_GATE_ZERO_RUNTIME_STATUS,
        "training_authorized": False,
        "quality_thresholds": None,
        "training_decision": None,
        "optimizer_steps": 0,
        "runtime_contract_sha256": admission.support_contract_sha256,
        "source": {
            "semantic_sidecar_manifest_sha256": "5" * 64,
            "semantic_sidecar_source_sha256": "6" * 64,
            "semantic_census_sha256": "7" * 64,
            "unified_packed_manifest_sha256": (
                admission.unified_packed_manifest_sha256
            ),
            "representability_overlay_sha256": "8" * 64,
            "validation_trace_count": 8,
            "validation_progress_row_count": 16,
            "validation_shards": [{"packed_shard_content_sha256": "9" * 64}],
        },
        "model": {"initialization_regime": "scratch"},
        "cache_contract": {
            "support_signature": support_signature,
            "support_signature_sha256": _sha(support_signature),
            "operator_registry_hash": "a" * 16,
            "operator_registry_source_sha256": "a" * 64,
            "canonicalizer_contract_sha256": "b" * 64,
            "tensorization_implementation_hash": "c" * 64,
            "fiber_compiler_implementation_hash": "d" * 64,
        },
        "probe": {"probe_sha256": "e" * 64},
        "cache_artifacts": [{"cache_file_sha256": "f" * 64}],
        "support_audit": {
            "teacher_examples_by_family": positive,
            "candidate_marks_by_family": positive,
            "productive_successors_by_family": positive,
        },
    }
    return {**body, "evidence_sha256": _sha(body)}


def _t1_decision(admission, gate_zero_file_sha256: str) -> dict[str, object]:
    body = {
        "schema": T1_P50_DECISION_SCHEMA,
        "schema_version": T1_P50_DECISION_SCHEMA_VERSION,
        "status": T1_P50_DECISION_PASS_STATUS,
        "bounded_p50_authorized": True,
        "full_training_authorized": False,
        "thresholds_frozen": True,
        "prerequisites": {
            "source_corpus_inventory_file_sha256": (
                admission.manifest_file_sha256
            ),
            "source_corpus_inventory_sha256": admission.inventory_sha256,
            "gate_zero_structural_evidence_file_sha256": (
                gate_zero_file_sha256
            ),
            "unified_packed_manifest_sha256": (
                admission.unified_packed_manifest_sha256
            ),
            "support_contract_sha256": admission.support_contract_sha256,
        },
        "required_families": list(RINGCORE_EDITING_FAMILIES),
        "disabled_families": list(P50_DISABLED_FAMILIES),
        "arm_results_manifest_sha256": "a" * 64,
    }
    return {**body, "decision_sha256": _sha(body)}


def _launch() -> dict[str, object]:
    values: dict[str, object] = {
        field: f"value-{field}" for field in P50_RECIPE_LAUNCH_FIELDS
    }
    values.update(
        {
            "optimizer_steps": 50,
            "max_atoms": 40,
            "hidden_dim": 256,
            "message_passing_steps": 6,
            "mark_dim": 32,
            "factorized_training_objective": "canonical_successor",
            "rate_factorization": "hierarchical",
            "empirical_mark_prior_mode": "none",
            "ring_family_mass_mode": "boolean",
            "ring_template_factorization": "flat",
            "required_families": list(RINGCORE_EDITING_FAMILIES),
            "disabled_families": list(P50_DISABLED_FAMILIES),
            "resume": False,
        }
    )
    return values


def _recipe(
    admission,
    *,
    gate_zero_file_sha256: str,
    t1_file_sha256: str,
    launch: dict[str, object],
) -> dict[str, object]:
    body = {
        "schema": P50_RECIPE_SCHEMA,
        "schema_version": P50_RECIPE_SCHEMA_VERSION,
        "status": P50_RECIPE_STATUS,
        "bounded_p50_authorized": True,
        "full_training_authorized": False,
        "prerequisites": {
            "source_corpus_inventory_file_sha256": (
                admission.manifest_file_sha256
            ),
            "source_corpus_inventory_sha256": admission.inventory_sha256,
            "gate_zero_structural_evidence_file_sha256": (
                gate_zero_file_sha256
            ),
            "unified_packed_manifest_sha256": (
                admission.unified_packed_manifest_sha256
            ),
            "support_contract_sha256": admission.support_contract_sha256,
            "t1_successor_gate_decision_file_sha256": t1_file_sha256,
        },
        "launch": launch,
        "launch_sha256": _sha(launch),
        "planned_stream": {
            "ordered_address_stream_sha256": "8" * 64,
            "ordered_training_stream_sha256": "9" * 64,
        },
    }
    return {**body, "recipe_sha256": _sha(body)}


def test_gate_zero_evidence_is_semantically_bound_to_active8() -> None:
    admission = _admission()
    evidence = _gate_zero(admission)
    assert (
        validate_gate_zero_structural_evidence(
            evidence,
            source_admission=admission,
        )["evidence_sha256"]
        == evidence["evidence_sha256"]
    )

    stale = copy.deepcopy(evidence)
    stale["runtime_contract_sha256"] = "0" * 64
    stale["evidence_sha256"] = _sha(
        {key: value for key, value in stale.items() if key != "evidence_sha256"}
    )
    with pytest.raises(
        EditingP50PrerequisiteError,
        match="different support contracts",
    ):
        validate_gate_zero_structural_evidence(
            stale,
            source_admission=admission,
        )


def test_correctly_hashed_t1_no_go_cannot_authorize_p50() -> None:
    admission = _admission()
    decision = _t1_decision(admission, "5" * 64)
    no_go = copy.deepcopy(decision)
    no_go["status"] = "NO_GO"
    no_go["bounded_p50_authorized"] = False
    no_go["decision_sha256"] = _sha(
        {key: value for key, value in no_go.items() if key != "decision_sha256"}
    )
    with pytest.raises(
        EditingP50PrerequisiteError,
        match="not an exact passing",
    ):
        validate_t1_p50_decision(
            no_go,
            source_admission=admission,
            source_inventory_file_sha256=admission.manifest_file_sha256,
            gate_zero_evidence_file_sha256="5" * 64,
        )


def test_recipe_must_equal_the_live_launcher_projection() -> None:
    admission = _admission()
    launch = _launch()
    recipe = _recipe(
        admission,
        gate_zero_file_sha256="5" * 64,
        t1_file_sha256="6" * 64,
        launch=launch,
    )
    assert (
        validate_p50_recipe(
            recipe,
            source_admission=admission,
            source_inventory_file_sha256=admission.manifest_file_sha256,
            gate_zero_evidence_file_sha256="5" * 64,
            t1_decision_file_sha256="6" * 64,
            expected_launch=launch,
        )["recipe_sha256"]
        == recipe["recipe_sha256"]
    )

    changed = dict(launch)
    changed["optimizer_steps"] = 51
    with pytest.raises(
        EditingP50PrerequisiteError,
        match="differ from the frozen recipe",
    ):
        validate_p50_recipe(
            recipe,
            source_admission=admission,
            source_inventory_file_sha256=admission.manifest_file_sha256,
            gate_zero_evidence_file_sha256="5" * 64,
            t1_decision_file_sha256="6" * 64,
            expected_launch=changed,
        )


def test_live_recipe_projection_exactly_covers_frozen_fields(tmp_path) -> None:
    unified = tmp_path / "unified.json"
    overlay = tmp_path / "overlay.json"
    unified.write_text("unified")
    overlay.write_text("overlay")
    args = SimpleNamespace(
        steps=50,
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_steps=5,
        schedule_steps=50,
        minimum_learning_rate_fraction=0.1,
        seed=17,
        successor_p50_initialization_regime="scratch",
        factorized_training_objective="canonical_successor",
        successor_objective_mode="productive_identity",
        successor_hazard_weight=0.0,
        hidden_dim=256,
        message_passing_steps=6,
        rate_factorization="hierarchical",
        empirical_mark_prior_mode="none",
        empirical_mark_prior_smoothing=1.0,
        ring_family_mass_mode="boolean",
        ring_template_factorization="flat",
        trainable_parameter_scope="all",
        late_time_fraction=0.5,
        operational_horizon=7.0,
        progress_stratification_fraction=0.0,
        bond_representation="aromatic",
        ring_electronic_mode="production",
        teacher_ordering="causal_frontier",
        condition_dropout_probability=0.0,
        property_condition=[],
        use_bf16=True,
        corrupted_prior_mix=True,
        cycle_op_mix=True,
        disable_ring_grow_macro=True,
        organic_vocabulary=True,
        evaluation_every=50,
        evaluation_batch_size=64,
        validation_examples=256,
        validation_size=133,
        max_atoms=40,
        resume_checkpoint=None,
        early_stopping_patience=0,
        benchmark_steps=0,
        snapshot_checkpoints=False,
        successor_source_corpus_inventory_sha256="1" * 64,
        successor_cache_inventory_sha256="2" * 64,
        successor_cache_compatibility_sha256="3" * 64,
        successor_semantic_sidecar_manifest_sha256="4" * 64,
        unified_packed_manifest=unified,
        representability_overlay=overlay,
    )
    projection = _p50_launch_recipe_projection(args, denovo_weight=0.0)
    assert set(projection) == set(P50_RECIPE_LAUNCH_FIELDS)
    assert projection["optimizer_steps"] == 50
    assert projection["resume"] is False
