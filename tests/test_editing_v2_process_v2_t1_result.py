from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from compose_v4.data.editing_v2_process_v2_pipeline_schema import (
    GATE_ZERO_DECISION_SCHEMA,
    GATE_ZERO_DECISION_SCHEMA_VERSION,
    PIPELINE_STATUS_NO_AUTHORITY,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    authority_false_block,
    canonical_bytes,
    canonical_sha256,
)
from compose_v4.experiments.editing_v2_process_v2_p50_prerequisites import (
    ProcessV2P50PrerequisiteError,
    load_process_v2_p50_prerequisites,
    validate_process_v2_p50_prerequisite_relationships,
)
from compose_v4.experiments.editing_v2_process_v2_t1_result import (
    ProcessV2T1ResultError,
    build_process_v2_t1_capacity_decision,
    build_process_v2_t1_capacity_result,
    process_v2_t1_runner_implementation_sha256,
    validate_process_v2_t1_capacity_result,
    validate_process_v2_t1_capacity_decision,
)

ROOT = Path(__file__).resolve().parents[1]


def _policy() -> dict[str, object]:
    policy = json.loads(
        (ROOT / "configs/editing_v2_process_v2_t1_capacity_policy.json").read_text()
    )
    return {**policy, "policy_sha256": policy["contract_sha256"]}


def _inputs() -> tuple[dict[str, object], dict[str, object]]:
    policy = _policy()
    families = tuple(policy["required_families"])
    entries: list[dict[str, object]] = []
    for family_index, family in enumerate(families):
        for item_index in range(64):
            identifier = f"{family_index * 64 + item_index + 1:064x}"
            entries.append(
                {
                    "panel_entry_sha256": identifier,
                    "family": family,
                    "semantic_cell_id": f"process_v2::{family}:fixture",
                    "teacher_successor_probability": 0.9,
                    "canonical_successor_nll": -math.log(0.9),
                    "teacher_successor_rank": 1,
                    "teacher_successor_top1": True,
                }
            )
    metadata = [
        {
            "panel_entry_sha256": row["panel_entry_sha256"],
            "family": row["family"],
            "semantic_cell_id": row["semantic_cell_id"],
        }
        for row in entries
    ]
    environment_body = {
        "hardware_class": "test_gpu",
        "device_name": "test",
        "device_capability": "0.0",
        "accelerator_class": "gpu",
        "dtype": "float32",
        "mixed_precision": False,
        "batch_size": 64,
        "python_version": "test",
        "torch_version": "test",
        "cuda_version": "test",
        "cudnn_version": "test",
        "rdkit_version": "test",
        "numpy_version": "test",
    }
    provenance: dict[str, object] = {
        name: "a" * 64
        for name in (
            "capacity_policy_file_sha256",
            "capacity_policy_sha256",
            "prepared_completion_file_sha256",
            "prepared_completion_sha256",
            "prepared_input_file_sha256",
            "prepared_input_artifact_sha256",
            "panel_sha256",
            "gate_zero_decision_sha256",
            "active8_completion_sha256",
            "process_identity_sha256",
            "initial_model_state_sha256",
            "runner_implementation_sha256",
            "runner_source_revision_sha256",
        )
    }
    provenance.update(
        {
            "capacity_policy_sha256": policy["policy_sha256"],
            "panel_entry_inventory_sha256": canonical_sha256(
                sorted(row["panel_entry_sha256"] for row in entries)
            ),
            "panel_entry_metadata_sha256": canonical_sha256(metadata),
            "panel_entry_binding_count": len(entries),
            "execution_environment": {
                **environment_body,
                "environment_sha256": canonical_sha256(environment_body),
            },
        }
    )
    trajectory = [
        {
            "step": step,
            "minimum_entry_teacher_successor_probability": probability,
            "mean_entry_canonical_successor_nll": -math.log(probability),
            "model_state_sha256": f"{step + 1000:064x}",
        }
        for step, probability in ((0, 0.1), (1, 0.2), (10, 0.9))
    ]
    run_integrity = {
        "optimizer_steps_completed": 10,
        "evaluation_steps": [0, 1, 10],
        "selected_step": 10,
        "termination_reason": "all_thresholds_passed_early",
        "resume_requested": False,
        "resume_count": 0,
        "abort_triggered": False,
        "abort_reasons": [],
        "nonfinite_event_count": 0,
        "unsupported_teacher_count": 0,
        "missing_candidate_count": 0,
        "provenance_drift_detected": False,
        "stream_identity_drift_detected": False,
        "deterministic_algorithms_enabled": True,
        "dtype": "float32",
        "mixed_precision": False,
        "hazard_included": False,
        "address_stream_sha256": "b" * 64,
        "selected_model_state_sha256": trajectory[-1]["model_state_sha256"],
        "selected_checkpoint_file_sha256": "c" * 64,
        "optimizer_state_sha256": "d" * 64,
    }
    gradients = [
        {
            "family": family,
            "family_gate_gradient_finite": True,
            "family_gate_cumulative_l2": 1.0,
            "family_gate_nonzero_update_steps": 10,
            "action_route_gradient_finite": True,
            "action_route_cumulative_l2": 1.0,
            "action_route_nonzero_update_steps": 10,
        }
        for family in families
    ]
    inputs = {
        "provenance": provenance,
        "run_integrity": run_integrity,
        "evaluation_trajectory": trajectory,
        "entry_metrics": entries,
        "gradient_evidence": gradients,
    }
    return policy, inputs


def _p50_policy() -> dict[str, object]:
    return json.loads(
        (ROOT / "configs/editing_v2_process_v2_p50_recipe_policy.json").read_text()
    )


def _gate_zero_pass(*, process_sha256: str, active8_sha256: str) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": GATE_ZERO_DECISION_SCHEMA,
        "schema_version": GATE_ZERO_DECISION_SCHEMA_VERSION,
        "status": PIPELINE_STATUS_NO_AUTHORITY,
        **authority_false_block(),
        "decision": "PASS",
        "process_identity_sha256": process_sha256,
        "active8_completion_sha256": active8_sha256,
    }
    return {**body, "decision_sha256": canonical_sha256(body)}


def _p50_inputs(*, probability: float = 0.9) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    policy, inputs = _inputs()
    p50_policy = _p50_policy()
    process_sha256 = p50_policy["process_identity"]["process_identity_sha256"]
    active8_sha256 = "f" * 64
    gate = _gate_zero_pass(
        process_sha256=process_sha256,
        active8_sha256=active8_sha256,
    )
    inputs["provenance"].update(
        {
            "process_identity_sha256": process_sha256,
            "active8_completion_sha256": active8_sha256,
            "gate_zero_decision_sha256": gate["decision_sha256"],
        }
    )
    if probability != 0.9:
        for entry in inputs["entry_metrics"]:
            entry["teacher_successor_probability"] = probability
            entry["canonical_successor_nll"] = -math.log(probability)
        inputs["evaluation_trajectory"][-1][
            "minimum_entry_teacher_successor_probability"
        ] = probability
        inputs["evaluation_trajectory"][-1][
            "mean_entry_canonical_successor_nll"
        ] = -math.log(probability)
    result = build_process_v2_t1_capacity_result(**inputs, capacity_policy=policy)
    decision = build_process_v2_t1_capacity_decision(
        result,
        capacity_policy=policy,
        result_file_sha256="e" * 64,
    )
    return p50_policy, policy, gate, result, decision


def test_process_v2_result_recomputes_the_frozen_capacity_decision_inputs() -> None:
    policy, inputs = _inputs()
    result = build_process_v2_t1_capacity_result(**inputs, capacity_policy=policy)

    assert validate_process_v2_t1_capacity_result(result, capacity_policy=policy) == result
    assert all(result["threshold_checks"].values())
    assert result["provenance"]["panel_entry_binding_count"] == 512


def test_process_v2_result_refuses_resealed_panel_metadata_drift() -> None:
    policy, inputs = _inputs()
    result = build_process_v2_t1_capacity_result(**inputs, capacity_policy=policy)
    changed = copy.deepcopy(result)
    changed["entry_metrics"][0]["semantic_cell_id"] = "process_v2::tampered"
    body = {key: value for key, value in changed.items() if key != "result_sha256"}
    changed["result_sha256"] = canonical_sha256(body)

    with pytest.raises(ProcessV2T1ResultError, match="panel binding disagrees"):
        validate_process_v2_t1_capacity_result(changed, capacity_policy=policy)


def test_process_v2_decision_is_the_only_bounded_p50_authority() -> None:
    policy, inputs = _inputs()
    result = build_process_v2_t1_capacity_result(**inputs, capacity_policy=policy)
    decision = build_process_v2_t1_capacity_decision(
        result,
        capacity_policy=policy,
        result_file_sha256="e" * 64,
    )

    assert (
        validate_process_v2_t1_capacity_decision(
            decision,
            result=result,
            capacity_policy=policy,
            result_file_sha256="e" * 64,
            require_p50_go=True,
        )
        == decision
    )
    assert decision["t1_authorized"] is True
    assert decision["bounded_p50_authorized"] is True
    assert decision["training_authorized"] is False
    assert decision["long_training_authorized"] is False


def test_process_v2_decision_records_a_capacity_no_go() -> None:
    policy, inputs = _inputs()
    for entry in inputs["entry_metrics"]:
        entry["teacher_successor_probability"] = 0.5
        entry["canonical_successor_nll"] = -math.log(0.5)
    inputs["evaluation_trajectory"][-1]["minimum_entry_teacher_successor_probability"] = 0.5
    inputs["evaluation_trajectory"][-1]["mean_entry_canonical_successor_nll"] = -math.log(0.5)
    result = build_process_v2_t1_capacity_result(**inputs, capacity_policy=policy)
    decision = build_process_v2_t1_capacity_decision(
        result,
        capacity_policy=policy,
        result_file_sha256="e" * 64,
    )

    assert decision["bounded_p50_authorized"] is False
    assert decision["failed_checks"]
    with pytest.raises(ProcessV2T1ResultError, match="not a bounded-P50 GO"):
        validate_process_v2_t1_capacity_decision(
            decision,
            result=result,
            capacity_policy=policy,
            result_file_sha256="e" * 64,
            require_p50_go=True,
        )


def test_process_v2_runner_implementation_hash_is_exact_and_stable() -> None:
    first = process_v2_t1_runner_implementation_sha256(repo_root=ROOT)
    second = process_v2_t1_runner_implementation_sha256(repo_root=ROOT)

    assert first == second
    assert len(first) == 64


def test_process_v2_p50_prerequisites_accept_only_the_exact_t1_go_chain() -> None:
    p50_policy, policy, gate, result, decision = _p50_inputs()

    binding = validate_process_v2_p50_prerequisite_relationships(
        p50_policy=p50_policy,
        capacity_policy=policy,
        gate_zero_decision=gate,
        t1_result=result,
        t1_decision=decision,
        t1_result_file_sha256="e" * 64,
    )

    assert binding.optimizer_steps == 50
    assert binding.batch_size == 64
    assert binding.t1_initial_model_state_sha256 == result["provenance"][
        "initial_model_state_sha256"
    ]
    assert binding.t1_selected_model_state_sha256 == decision[
        "selected_model_state_sha256"
    ]
    assert (
        binding.t1_initial_model_state_sha256
        != binding.t1_selected_model_state_sha256
    )


def test_process_v2_p50_prerequisites_refuse_a_t1_no_go() -> None:
    p50_policy, policy, gate, result, no_go = _p50_inputs(probability=0.5)

    with pytest.raises(ProcessV2P50PrerequisiteError, match="not a bounded-P50 GO"):
        validate_process_v2_p50_prerequisite_relationships(
            p50_policy=p50_policy,
            capacity_policy=policy,
            gate_zero_decision=gate,
            t1_result=result,
            t1_decision=no_go,
            t1_result_file_sha256="e" * 64,
        )


def test_process_v2_p50_prerequisites_refuse_cross_run_gate_zero() -> None:
    p50_policy, policy, gate, result, decision = _p50_inputs()
    changed_gate = {
        **gate,
        "active8_completion_sha256": "1" * 64,
    }
    changed_gate["decision_sha256"] = canonical_sha256(
        {key: value for key, value in changed_gate.items() if key != "decision_sha256"}
    )

    with pytest.raises(ProcessV2P50PrerequisiteError, match="identity disagrees"):
        validate_process_v2_p50_prerequisite_relationships(
            p50_policy=p50_policy,
            capacity_policy=policy,
            gate_zero_decision=changed_gate,
            t1_result=result,
            t1_decision=decision,
            t1_result_file_sha256="e" * 64,
        )


def test_process_v2_p50_prerequisites_reopen_exact_artifact_bytes(
    tmp_path: Path,
) -> None:
    _p50, policy, gate, result, _decision = _p50_inputs()
    gate_path = tmp_path / "gate-zero.json"
    result_path = tmp_path / "t1-result.json"
    decision_path = tmp_path / "t1-decision.json"
    gate_path.write_bytes(canonical_bytes(gate) + b"\n")
    result_bytes = canonical_bytes(result) + b"\n"
    result_path.write_bytes(result_bytes)
    decision = build_process_v2_t1_capacity_decision(
        result,
        capacity_policy=policy,
        result_file_sha256=hashlib.sha256(result_bytes).hexdigest(),
    )
    decision_path.write_bytes(canonical_bytes(decision) + b"\n")

    binding = load_process_v2_p50_prerequisites(
        gate_zero_decision_path=gate_path,
        t1_result_path=result_path,
        t1_decision_path=decision_path,
        repo_root=ROOT,
    )

    assert binding.t1_decision_sha256 == decision["decision_sha256"]
    assert binding.p50_recipe_policy_sha256 == _p50_policy()["contract_sha256"]
