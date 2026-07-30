"""Fail-closed tests for the underspecified E6 A2.3 control problem."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from compose_v4.experiments.e6_a2_3_readiness import (
    A2_3_BLOCKED_STATUS,
    E6A23ReadinessError,
    artifact_self_hash,
    contract_self_hash,
    freeze_a2_3_readiness_artifact,
    load_a2_3_readiness_contract,
    run_a2_3_readiness_audit,
    validate_a2_3_readiness_artifact,
)

_ROOT = Path(__file__).resolve().parent.parent
_CONTRACT_PATH = _ROOT / "configs/exact_control_a2_3_readiness_v1.json"
_ARTIFACT_PATH = (
    _ROOT
    / "diagnostics/exactness/"
    "e6_a2_3_readiness_blocked_v1_2026-07-30.json"
)


@pytest.fixture(scope="module")
def contract():
    return load_a2_3_readiness_contract(_CONTRACT_PATH)


def test_contract_is_self_hashed_and_fail_closed(contract):
    assert contract["contract_sha256"] == contract_self_hash(contract)
    assert contract["paper_claim_authorized"] is False
    assert contract["solver_execution_authorized"] is False
    assert contract["confirmed_stage_1_kernel"]["level"] == (
        "canonical_molecular_successor"
    )
    assert contract["confirmed_stage_1_kernel"][
        "is_a2_2b_diagnostic_mark_law"
    ] is False


def test_contract_tampering_fails_closed(tmp_path):
    payload = json.loads(_CONTRACT_PATH.read_text())
    payload["solver_execution_authorized"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(E6A23ReadinessError, match="fail-closed|self-hash"):
        load_a2_3_readiness_contract(path)


def test_audit_reports_exact_missing_decisions_without_solver(contract):
    artifact = run_a2_3_readiness_audit(contract)
    assert artifact["status"] == A2_3_BLOCKED_STATUS
    assert artifact["control_solver_run"] is False
    assert artifact["solver_kernel_materialized"] is False
    assert artifact["terminal_law_metrics_computed"] is False
    assert artifact["paper_claim_authorized"] is False
    assert [item["decision_id"] for item in artifact["missing_decisions"]] == [
        "finite_control_horizon",
        "terminal_desirability",
        "initial_law",
        "solver_tolerances",
        "explicit_path_check",
    ]
    assert artifact["missing_task_artifacts"] == [
        "configs/tasks/exact_control.development.json",
        "configs/tasks/exact_control.final.json",
    ]


def test_e7_dynamic_horizon_is_not_imported(contract):
    artifact = run_a2_3_readiness_audit(contract)
    assert artifact["confirmed"]["e6_horizon_fields_present"] == {}
    assert artifact["invariants"]["e7_dynamic_horizon_not_imported"] is True


def test_named_tilt_families_are_not_misrepresented_as_g(contract):
    artifact = run_a2_3_readiness_audit(contract)
    assert artifact["confirmed"]["registry_tilt_names_are_unparameterized"]
    assert artifact["confirmed"]["e6_desirability_fields_present"] == {}
    assert artifact["invariants"][
        "unparameterized_tilt_names_not_treated_as_g"
    ] is True


def test_frozen_artifact_matches_current_inputs(contract):
    artifact = json.loads(_ARTIFACT_PATH.read_text())
    validate_a2_3_readiness_artifact(
        artifact,
        contract=contract,
        repo_root=_ROOT,
    )
    assert artifact["artifact_sha256"] == artifact_self_hash(artifact)
    assert artifact["confirmed"]["selected_benchmark"] == {
        "candidate_id": "carbon_6_slots",
        "n_states": 967,
        "n_edges": 14_432,
        "structural_graph_fingerprint": "84121ff86cbc1ba8",
        "state_index_sha256": (
            "4fe916579503e2b4750556e4dff479127a812d33b0df1f4f1dc2db23036d676c"
        ),
    }


def test_artifact_tampering_fails_closed(contract):
    artifact = json.loads(_ARTIFACT_PATH.read_text())
    tampered = copy.deepcopy(artifact)
    tampered["control_solver_run"] = True
    with pytest.raises(E6A23ReadinessError, match="self-hash"):
        validate_a2_3_readiness_artifact(
            tampered,
            contract=contract,
            repo_root=_ROOT,
        )


def test_freeze_is_idempotent_but_never_overwrites(contract, tmp_path):
    artifact = run_a2_3_readiness_audit(contract)
    output = tmp_path / "readiness.json"
    freeze_a2_3_readiness_artifact(
        artifact,
        output,
        contract=contract,
        repo_root=_ROOT,
    )
    frozen = output.read_bytes()
    freeze_a2_3_readiness_artifact(
        artifact,
        output,
        contract=contract,
        repo_root=_ROOT,
    )
    assert output.read_bytes() == frozen

    output.write_text("occupied by other bytes")
    with pytest.raises(E6A23ReadinessError, match="immutable.*collision"):
        freeze_a2_3_readiness_artifact(
            artifact,
            output,
            contract=contract,
            repo_root=_ROOT,
        )


def test_result_producer_contains_no_solver_or_old_evidence_import():
    source = (
        _ROOT / "src/compose_v4/experiments/e6_a2_3_readiness.py"
    ).read_text()
    assert "exact_doob_enumerable_benchmark" not in source
    assert "doob_guidance_ground_truth" not in source
    assert "e0_toy_h_exactness" not in source
    assert "def exact_doob" not in source
    assert "def backward_values" not in source
