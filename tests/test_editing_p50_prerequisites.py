from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
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
    build_t1_arm_results_manifest,
    build_t1_p50_decision,
    planned_t1_arms,
    validate_gate_zero_structural_evidence,
    validate_p50_recipe,
    validate_t1_p50_decision,
)
from compose_v4.experiments.editing_t1_panel import (
    ACTIVE8_T1_IDENTITY_FIELDS,
    EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
    EDITING_T1_UNIQUE_PANEL_KIND,
    EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND,
)
from compose_v4.experiments.editing_t1_successor_runtime import (
    EDITING_T1_RESULT_SCHEMA,
    EDITING_T1_RESULT_STATUS,
    EDITING_T1_RESULT_VERSION,
    EDITING_T1_RUNTIME_CONTRACT_SCHEMA,
    EDITING_T1_RUNTIME_CONTRACT_STATUS,
    EDITING_T1_RUNTIME_CONTRACT_VERSION,
    EditingT1RuntimeContract,
    editing_t1_implementation_sha256,
    editing_t1_numeric_thresholds_sha256,
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
        effective_source_corpus_cache_sha256="0" * 64,
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
        "active8_admission": {
            "active8_inventory_manifest_file_sha256": (
                admission.manifest_file_sha256
            ),
            "active8_inventory_sha256": admission.inventory_sha256,
            "active8_effective_source_corpus_cache_sha256": (
                admission.effective_source_corpus_cache_sha256
            ),
            "active8_unified_packed_manifest_sha256": (
                admission.unified_packed_manifest_sha256
            ),
            "active8_support_contract_sha256": (
                admission.support_contract_sha256
            ),
            "partition": "validation",
            "source_shard_count": 1,
            "source_trace_count": 8,
            "source_progress_row_count": 16,
            "admitted_trace_count": 8,
            "excluded_trace_count": 0,
            "admitted_progress_row_count": 16,
            "admitted_nonterminal_row_count": 8,
            "admitted_terminal_row_count": 8,
            "admitted_nonempty_semantic_cell_count": 1,
            "admitted_teacher_examples_by_family": positive,
        },
        "source": {
            "semantic_sidecar_manifest_sha256": "5" * 64,
            "semantic_sidecar_source_sha256": "6" * 64,
            "semantic_census_sha256": "7" * 64,
            "unified_packed_manifest_sha256": (admission.unified_packed_manifest_sha256),
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
    numeric_thresholds = {
        "minimum_unique_state_teacher_successor_top1": 0.8,
        "minimum_unique_state_teacher_successor_probability": 0.6,
        "maximum_unique_state_teacher_successor_nll": 0.7,
        "maximum_global_repeated_state_excess_nll_over_empirical_entropy": (0.2),
    }
    body = {
        "schema": T1_P50_DECISION_SCHEMA,
        "schema_version": T1_P50_DECISION_SCHEMA_VERSION,
        "status": T1_P50_DECISION_PASS_STATUS,
        "bounded_p50_authorized": True,
        "full_training_authorized": False,
        "thresholds_frozen": True,
        "pre_result_runtime_contract_relative_path": "runtime.json",
        "pre_result_runtime_contract_file_sha256": "c" * 64,
        "pre_result_runtime_contract_sha256": "b" * 64,
        "numeric_thresholds": numeric_thresholds,
        "numeric_thresholds_sha256": _sha(numeric_thresholds),
        "prerequisites": {
            "source_corpus_inventory_file_sha256": (admission.manifest_file_sha256),
            "source_corpus_inventory_sha256": admission.inventory_sha256,
            "gate_zero_structural_evidence_file_sha256": (gate_zero_file_sha256),
            "unified_packed_manifest_sha256": (admission.unified_packed_manifest_sha256),
            "support_contract_sha256": admission.support_contract_sha256,
        },
        "required_families": list(RINGCORE_EDITING_FAMILIES),
        "disabled_families": list(P50_DISABLED_FAMILIES),
        "arm_results_manifest_relative_path": "manifest.json",
        "arm_results_manifest_file_sha256": "d" * 64,
        "arm_results_manifest_sha256": "a" * 64,
        "evaluation": {
            "planned_arm_count": 1,
            "validated_arm_count": 1,
            "core_unique_all_scope": {},
            "global_repeated_entropy_all_scope": {},
            "required_diagnostic_arm_count": 0,
            "failed_checks": [],
        },
    }
    return {**body, "decision_sha256": _sha(body)}


def _launch() -> dict[str, object]:
    values: dict[str, object] = {field: f"value-{field}" for field in P50_RECIPE_LAUNCH_FIELDS}
    values.update(
        {
            "optimizer_steps": 50,
            "max_atoms": 40,
            "hidden_dim": 256,
            "message_passing_steps": 6,
            "mark_dim": 32,
            "factorized_training_objective": "canonical_successor",
            "successor_objective_mode": "productive_identity",
            "successor_hazard_weight": 0.0,
            "rate_factorization": "hierarchical",
            "empirical_mark_prior_mode": "none",
            "ring_family_mass_mode": "boolean",
            "ring_template_factorization": "flat",
            "trainable_parameter_scope": "all",
            "condition_dropout_probability": 0.0,
            "property_conditions": [],
            "denovo_weight": 0.0,
            "corrupted_prior_mix": True,
            "cycle_op_mix": True,
            "disable_ring_grow_macro": True,
            "enable_ring_system_delete": False,
            "organic_vocabulary": True,
            "required_families": list(RINGCORE_EDITING_FAMILIES),
            "disabled_families": list(P50_DISABLED_FAMILIES),
            "source_corpus_inventory_sha256": _admission().inventory_sha256,
            "unified_packed_manifest_sha256": (_admission().unified_packed_manifest_sha256),
            "resume": False,
        }
    )
    return values


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def _runtime_contract(admission) -> dict[str, object]:
    thresholds = {
        "minimum_unique_state_teacher_successor_top1": 0.8,
        "minimum_unique_state_teacher_successor_probability": 0.6,
        "maximum_unique_state_teacher_successor_nll": 0.7,
        "maximum_global_repeated_state_excess_nll_over_empirical_entropy": 0.2,
    }
    body = {
        "schema": EDITING_T1_RUNTIME_CONTRACT_SCHEMA,
        "schema_version": EDITING_T1_RUNTIME_CONTRACT_VERSION,
        "status": EDITING_T1_RUNTIME_CONTRACT_STATUS,
        "training_authorized": False,
        "gate_zero_runtime_contract_sha256": admission.support_contract_sha256,
        "panel_artifact_sha256": "1" * 64,
        "panel_selection_sha256": "2" * 64,
        "panel_census_sha256": "3" * 64,
        "panel_capacity_strata_sha256": "4" * 64,
        "forensics_file_sha256": "5" * 64,
        "charge_policy_audit_file_sha256": "6" * 64,
        "charge_policy_exclusions_file_sha256": "7" * 64,
        "charge_policy_exclusion_payload_sha256": "8" * 64,
        "charge_policy_source_input_inventory_sha256": "9" * 64,
        "capacity_census_file_sha256": "a" * 64,
        "capacity_census_sha256": "b" * 64,
        "active8_inventory_manifest_file_sha256": (admission.manifest_file_sha256),
        "active8_inventory_sha256": admission.inventory_sha256,
        "active8_effective_source_corpus_cache_sha256": (
            admission.effective_source_corpus_cache_sha256
        ),
        "active8_unified_packed_manifest_sha256": (admission.unified_packed_manifest_sha256),
        "active8_support_contract_sha256": admission.support_contract_sha256,
        "implementation_sha256": editing_t1_implementation_sha256(),
        "families": list(RINGCORE_EDITING_FAMILIES),
        "panel_kinds": [
            EDITING_T1_UNIQUE_PANEL_KIND,
            EDITING_T1_GLOBAL_REPEATED_PANEL_KIND,
            EDITING_T1_WITHIN_FAMILY_REPEATED_PANEL_KIND,
        ],
        "required_high_candidate_families": list(RINGCORE_EDITING_FAMILIES),
        "required_aliased_teacher_families": [],
        "within_family_repeated_families": ["cycle_attach"],
        "optimization": {
            "steps": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "scopes": ["heads_only", "heads_plus_local_adapter", "all"],
            "report_points": [1],
        },
        "thresholds": thresholds,
        "thresholds_sha256": editing_t1_numeric_thresholds_sha256(thresholds),
    }
    return {**body, "contract_sha256": _sha(body)}


def _arm_result(
    contract: EditingT1RuntimeContract,
    arm: Mapping[str, str],
    *,
    metric_pass: bool = True,
    gradient_pass: bool = True,
) -> dict[str, object]:
    family = arm["family"]
    panel_kind = arm["panel_kind"]
    scope = arm["scope"]
    example_count = 64
    unique_address_count = (
        example_count if panel_kind == EDITING_T1_UNIQUE_PANEL_KIND else example_count // 2
    )
    final = {
        "n_examples": example_count,
        "teacher_successor_top1_recall": 0.9 if metric_pass else 0.1,
        "teacher_successor_probability": 0.8 if metric_pass else 0.1,
        "canonical_successor_nll": 0.3 if metric_pass else 2.0,
        "teacher_family_probability": 0.9,
        "teacher_family_nll": 0.1,
        "within_teacher_family_successor_probability": 0.9,
        "within_teacher_family_successor_nll": 0.1,
        "mean_raw_mark_count": 4.0,
        "mean_canonical_successor_count": 3.0,
        "mean_alias_multiplicity": 1.0,
        "repeated_state_excess_nll_over_empirical_entropy": (0.1 if metric_pass else 1.0),
    }
    receipt = {
        "packed_shard_content_sha256": "c" * 64,
        "cache_content_sha256": "d" * 64,
        "encoded_sha256": "e" * 64,
        "record_count": 1,
    }
    body = {
        "schema": EDITING_T1_RESULT_SCHEMA,
        "schema_version": EDITING_T1_RESULT_VERSION,
        "status": EDITING_T1_RESULT_STATUS,
        "training_authorized": False,
        "gate_decision": None,
        "numeric_thresholds_frozen": True,
        "numeric_thresholds_sha256": contract.numeric_thresholds_sha256,
        "contract_sha256": contract.sha256,
        "gate_zero_runtime_contract_sha256": contract.payload["gate_zero_runtime_contract_sha256"],
        "panel_artifact_sha256": contract.payload["panel_artifact_sha256"],
        "panel_selection_sha256": contract.payload["panel_selection_sha256"],
        "panel_census_sha256": contract.payload["panel_census_sha256"],
        "panel_capacity_strata_sha256": contract.payload["panel_capacity_strata_sha256"],
        "charge_policy_exclusion_payload_sha256": contract.payload[
            "charge_policy_exclusion_payload_sha256"
        ],
        "charge_policy_source_input_inventory_sha256": contract.payload[
            "charge_policy_source_input_inventory_sha256"
        ],
        **{field: contract.payload[field] for field in ACTIVE8_T1_IDENTITY_FIELDS},
        "family": family,
        "family_status": "CURRENT_ACTIVE_FAMILY",
        "panel_kind": panel_kind,
        "panel_role": "T1_REQUIRED_ARM",
        "scope": scope,
        "operator_identity": {
            "operator_capability_fingerprint": "fixture-active8",
            "enable_ring_system_delete": False,
            "enable_ring_grow_macro": False,
            "enable_cycle_ops": True,
        },
        "example_count": example_count,
        "unique_progress_address_count": unique_address_count,
        "repeated_progress_observation_count": (example_count - unique_address_count),
        "source_row_sha256s": [f"{index + 1:064x}" for index in range(example_count)],
        "scratch_initialization": {
            "regime": "scratch",
            "transfer_plan_sha256": "0" * 64,
            "initial_model_state_sha256": "1" * 64,
        },
        "cache_receipts": [receipt],
        "training_report": {
            "scope": scope,
            "steps": 1,
            "families": (
                sorted(RINGCORE_EDITING_FAMILIES) if family == "all_families" else [family]
            ),
            "optimizer_steps_with_nonzero_gradient": (1 if gradient_pass else 0),
            "required_components_without_gradient": ([] if gradient_pass else ["family_head"]),
            "component_gradient_update_counts": {"family_head": 1 if gradient_pass else 0},
            "history": [
                {
                    "step": 1.0,
                    "loss": 1.0,
                    "gradient_norm": 1.0 if gradient_pass else 0.0,
                }
            ],
            "initial": dict(final),
            "final": final,
        },
        "final_model_state_sha256": "2" * 64,
    }
    return {**body, "result_sha256": _sha(body)}


def _physical_t1_evidence(
    tmp_path: Path,
    *,
    fail_arm: tuple[str, str, str] | None = None,
    gradient_fail_arm: tuple[str, str, str] | None = None,
):
    admission = _admission()
    runtime_payload = _runtime_contract(admission)
    runtime_path = tmp_path / "runtime.json"
    _write_json(runtime_path, runtime_payload)
    runtime = EditingT1RuntimeContract(runtime_payload)
    result_paths: dict[tuple[str, str, str], Path] = {}
    retained_receipts: dict[tuple[str, str, str], dict[str, object]] = {}
    for index, arm in enumerate(planned_t1_arms(runtime)):
        key = (arm["family"], arm["panel_kind"], arm["scope"])
        result = _arm_result(
            runtime,
            arm,
            metric_pass=key != fail_arm,
            gradient_pass=key != gradient_fail_arm,
        )
        path = tmp_path / "results" / f"{index:03d}.json"
        _write_json(path, result)
        result_paths[key] = path
        retained_receipts[key] = {
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "result_sha256": result["result_sha256"],
            "cache_receipts": copy.deepcopy(result["cache_receipts"]),
        }
    manifest = build_t1_arm_results_manifest(
        runtime_contract_path=runtime_path,
        result_paths=result_paths,
        retained_result_receipts=retained_receipts,
        manifest_directory=tmp_path,
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return admission, runtime_path, manifest_path, result_paths, retained_receipts


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
            "source_corpus_inventory_file_sha256": (admission.manifest_file_sha256),
            "source_corpus_inventory_sha256": admission.inventory_sha256,
            "gate_zero_structural_evidence_file_sha256": (gate_zero_file_sha256),
            "unified_packed_manifest_sha256": (admission.unified_packed_manifest_sha256),
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

    wrong_inventory = copy.deepcopy(evidence)
    wrong_inventory["active8_admission"][
        "active8_inventory_sha256"
    ] = "f" * 64
    wrong_inventory["evidence_sha256"] = _sha(
        {
            key: value
            for key, value in wrong_inventory.items()
            if key != "evidence_sha256"
        }
    )
    with pytest.raises(
        EditingP50PrerequisiteError,
        match="identities disagree",
    ):
        validate_gate_zero_structural_evidence(
            wrong_inventory,
            source_admission=admission,
        )

    partial_trace = copy.deepcopy(evidence)
    partial_trace["active8_admission"]["admitted_trace_count"] = 7
    partial_trace["evidence_sha256"] = _sha(
        {
            key: value
            for key, value in partial_trace.items()
            if key != "evidence_sha256"
        }
    )
    with pytest.raises(
        EditingP50PrerequisiteError,
        match="census is internally inconsistent",
    ):
        validate_gate_zero_structural_evidence(
            partial_trace,
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
            decision_path=Path("decision.json"),
            source_admission=admission,
            source_inventory_file_sha256=admission.manifest_file_sha256,
            gate_zero_evidence_file_sha256="5" * 64,
        )


def test_t1_bare_threshold_boolean_cannot_authorize_p50() -> None:
    admission = _admission()
    decision = _t1_decision(admission, "5" * 64)
    bare = {
        key: value
        for key, value in decision.items()
        if key
        not in {
            "pre_result_runtime_contract_sha256",
            "numeric_thresholds",
            "numeric_thresholds_sha256",
            "decision_sha256",
        }
    }
    bare["decision_sha256"] = _sha(bare)
    with pytest.raises(
        EditingP50PrerequisiteError,
        match="missing or unknown fields",
    ):
        validate_t1_p50_decision(
            bare,
            decision_path=Path("decision.json"),
            source_admission=admission,
            source_inventory_file_sha256=admission.manifest_file_sha256,
            gate_zero_evidence_file_sha256="5" * 64,
        )


def test_t1_decision_rejects_threshold_map_hash_disagreement() -> None:
    admission = _admission()
    decision = _t1_decision(admission, "5" * 64)
    tampered = copy.deepcopy(decision)
    tampered["numeric_thresholds"]["minimum_unique_state_teacher_successor_top1"] = 0.9
    tampered["decision_sha256"] = _sha(
        {key: value for key, value in tampered.items() if key != "decision_sha256"}
    )
    with pytest.raises(
        EditingP50PrerequisiteError,
        match="disagree with their prospective hash",
    ):
        validate_t1_p50_decision(
            tampered,
            decision_path=Path("decision.json"),
            source_admission=admission,
            source_inventory_file_sha256=admission.manifest_file_sha256,
            gate_zero_evidence_file_sha256="5" * 64,
        )


def test_t1_decision_recomputes_complete_physical_v4_arm_matrix(
    tmp_path,
) -> None:
    admission, runtime_path, manifest_path, _results, _receipts = _physical_t1_evidence(tmp_path)
    decision = build_t1_p50_decision(
        runtime_contract_path=runtime_path,
        arm_results_manifest_path=manifest_path,
        decision_directory=tmp_path,
        source_admission=admission,
        source_inventory_file_sha256=admission.manifest_file_sha256,
        gate_zero_evidence_file_sha256="5" * 64,
    )
    decision_path = tmp_path / "decision.json"
    _write_json(decision_path, decision)
    observed = validate_t1_p50_decision(
        decision,
        decision_path=decision_path,
        source_admission=admission,
        source_inventory_file_sha256=admission.manifest_file_sha256,
        gate_zero_evidence_file_sha256="5" * 64,
    )
    assert observed["bounded_p50_authorized"] is True
    assert (
        observed["evaluation"]["planned_arm_count"] == observed["evaluation"]["validated_arm_count"]
    )
    assert set(observed["evaluation"]["core_unique_all_scope"]) == set(RINGCORE_EDITING_FAMILIES)


def test_t1_manifest_rejects_one_missing_required_diagnostic_arm(
    tmp_path,
) -> None:
    (
        _admission_value,
        runtime_path,
        _manifest_path,
        results,
        receipts,
    ) = _physical_t1_evidence(tmp_path)
    missing = dict(results)
    missing_key = next(iter(missing))
    missing.pop(missing_key)
    missing_receipts = dict(receipts)
    missing_receipts.pop(missing_key)
    with pytest.raises(
        EditingP50PrerequisiteError,
        match="complete planned arm matrix",
    ):
        build_t1_arm_results_manifest(
            runtime_contract_path=runtime_path,
            result_paths=missing,
            retained_result_receipts=missing_receipts,
            manifest_directory=tmp_path,
        )


def test_t1_manifest_requires_independently_retained_cache_receipts(
    tmp_path,
) -> None:
    (
        _admission_value,
        runtime_path,
        _manifest_path,
        results,
        receipts,
    ) = _physical_t1_evidence(tmp_path)
    key = next(iter(results))
    path = results[key]
    forged = json.loads(path.read_text())
    forged["cache_receipts"][0]["cache_content_sha256"] = "0" * 64
    forged_body = {field: value for field, value in forged.items() if field != "result_sha256"}
    forged["result_sha256"] = _sha(forged_body)
    _write_json(path, forged)
    forged_receipts = copy.deepcopy(receipts)
    forged_receipts[key]["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    forged_receipts[key]["result_sha256"] = forged["result_sha256"]

    with pytest.raises(
        EditingP50PrerequisiteError,
        match="retained run receipt",
    ):
        build_t1_arm_results_manifest(
            runtime_contract_path=runtime_path,
            result_paths=results,
            retained_result_receipts=forged_receipts,
            manifest_directory=tmp_path,
        )


def test_t1_manifest_rejects_results_missing_required_factor_metrics(
    tmp_path,
) -> None:
    (
        _admission_value,
        runtime_path,
        _manifest_path,
        results,
        receipts,
    ) = _physical_t1_evidence(tmp_path)
    key = next(iter(results))
    path = results[key]
    incomplete = json.loads(path.read_text())
    del incomplete["training_report"]["final"]["teacher_family_probability"]
    incomplete_body = {
        field: value for field, value in incomplete.items() if field != "result_sha256"
    }
    incomplete["result_sha256"] = _sha(incomplete_body)
    _write_json(path, incomplete)
    updated_receipts = copy.deepcopy(receipts)
    updated_receipts[key] = {
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "result_sha256": incomplete["result_sha256"],
        "cache_receipts": copy.deepcopy(incomplete["cache_receipts"]),
    }

    with pytest.raises(
        EditingP50PrerequisiteError,
        match="required successor metrics",
    ):
        build_t1_arm_results_manifest(
            runtime_contract_path=runtime_path,
            result_paths=results,
            retained_result_receipts=updated_receipts,
            manifest_directory=tmp_path,
        )


def test_self_declared_t1_pass_cannot_override_failed_metrics_or_gradients(
    tmp_path,
) -> None:
    failed_key = ("atom_insert", EDITING_T1_UNIQUE_PANEL_KIND, "all")
    admission, runtime_path, manifest_path, _results, _receipts = _physical_t1_evidence(
        tmp_path,
        fail_arm=failed_key,
    )
    measured_no_go = build_t1_p50_decision(
        runtime_contract_path=runtime_path,
        arm_results_manifest_path=manifest_path,
        decision_directory=tmp_path,
        source_admission=admission,
        source_inventory_file_sha256=admission.manifest_file_sha256,
        gate_zero_evidence_file_sha256="5" * 64,
    )
    assert measured_no_go["bounded_p50_authorized"] is False
    forged = copy.deepcopy(measured_no_go)
    forged["status"] = T1_P50_DECISION_PASS_STATUS
    forged["bounded_p50_authorized"] = True
    forged["decision_sha256"] = _sha(
        {key: value for key, value in forged.items() if key != "decision_sha256"}
    )
    decision_path = tmp_path / "decision.json"
    _write_json(decision_path, forged)
    with pytest.raises(
        EditingP50PrerequisiteError,
        match="physical v4 arm-result recomputation",
    ):
        validate_t1_p50_decision(
            forged,
            decision_path=decision_path,
            source_admission=admission,
            source_inventory_file_sha256=admission.manifest_file_sha256,
            gate_zero_evidence_file_sha256="5" * 64,
        )

    gradient_root = tmp_path / "gradient"
    with pytest.raises(
        EditingP50PrerequisiteError,
        match="gradient evidence",
    ):
        _physical_t1_evidence(
            gradient_root,
            gradient_fail_arm=failed_key,
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


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("organic_vocabulary", False),
        ("disable_ring_grow_macro", False),
        ("enable_ring_system_delete", True),
        ("corrupted_prior_mix", False),
        ("cycle_op_mix", False),
        ("denovo_weight", 0.1),
        ("trainable_parameter_scope", "heads_only"),
        ("property_conditions", ["reward"]),
        ("source_corpus_inventory_sha256", "0" * 64),
        ("unified_packed_manifest_sha256", "0" * 64),
    ),
)
def test_p50_recipe_rejects_support_or_regime_metadata_drift(
    field: str,
    bad_value: object,
) -> None:
    admission = _admission()
    launch = _launch()
    launch[field] = bad_value
    recipe = _recipe(
        admission,
        gate_zero_file_sha256="5" * 64,
        t1_file_sha256="6" * 64,
        launch=launch,
    )
    with pytest.raises(
        EditingP50PrerequisiteError,
        match="exact non-resumable Active8 successor pilot",
    ):
        validate_p50_recipe(
            recipe,
            source_admission=admission,
            source_inventory_file_sha256=admission.manifest_file_sha256,
            gate_zero_evidence_file_sha256="5" * 64,
            t1_decision_file_sha256="6" * 64,
            expected_launch=launch,
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
    projection = _p50_launch_recipe_projection(
        args,
        denovo_weight=0.0,
        enable_ring_system_delete=False,
    )
    assert set(projection) == set(P50_RECIPE_LAUNCH_FIELDS)
    assert projection["optimizer_steps"] == 50
    assert projection["enable_ring_system_delete"] is False
    assert projection["resume"] is False
