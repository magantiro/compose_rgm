from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from modal_apps import run_process_v2_t1_app as recovery
from compose_v4.experiments.editing_v2_process_v2_t1_result import (
    build_process_v2_t1_capacity_result,
    validate_process_v2_t1_capacity_result,
)
from test_editing_v2_process_v2_t1_result import _inputs


def _recovery_inputs():
    policy, inputs = _inputs()
    policy = copy.deepcopy(policy)
    policy["optimization"] = {
        **policy["optimization"],
        "maximum_optimizer_steps": 10,
    }
    inputs = copy.deepcopy(inputs)
    inputs["provenance"]["panel_entry_metadata_sha256"] = "0" * 64
    inputs["run_integrity"]["resume_requested"] = True
    inputs["run_integrity"]["resume_count"] = 1
    return policy, inputs


def test_publication_recovery_repairs_only_order_and_optimizer_resume_label() -> None:
    policy, inputs = _recovery_inputs()

    result = recovery._publication_recovery_result_builder(
        build_result=build_process_v2_t1_capacity_result,
        capacity_policy=policy,
        **inputs,
    )

    assert validate_process_v2_t1_capacity_result(result, capacity_policy=policy) == result
    assert result["run_integrity"]["resume_requested"] is False
    assert result["run_integrity"]["resume_count"] == 0
    assert result["threshold_checks"] == build_process_v2_t1_capacity_result(
        provenance=result["provenance"],
        run_integrity=result["run_integrity"],
        evaluation_trajectory=inputs["evaluation_trajectory"],
        entry_metrics=inputs["entry_metrics"],
        gradient_evidence=inputs["gradient_evidence"],
        capacity_policy=policy,
    )["threshold_checks"]
    assert inputs["run_integrity"]["resume_count"] == 1
    assert inputs["provenance"]["panel_entry_metadata_sha256"] == "0" * 64


def test_publication_recovery_requires_exactly_one_completed_reopen() -> None:
    policy, inputs = _recovery_inputs()
    inputs["run_integrity"]["optimizer_steps_completed"] = 9

    with pytest.raises(RuntimeError, match="exactly one completed run"):
        recovery._publication_recovery_result_builder(
            build_result=build_process_v2_t1_capacity_result,
            capacity_policy=policy,
            **inputs,
        )


def _checkpoint_fixture():
    policy_sha = "a" * 64
    prepared_sha = "b" * 64
    completion_sha = "c" * 64
    manifest_sha = "d" * 64
    initial_sha = "e" * 64
    runner_sha = "f" * 64
    source_sha = "1" * 64
    runtime = SimpleNamespace(
        capacity_policy={
            "policy_sha256": policy_sha,
            "optimization": {"maximum_optimizer_steps": 500},
        },
        prepared=SimpleNamespace(
            artifact={
                "artifact_sha256": prepared_sha,
                "source_revision": {"source_revision_sha256": source_sha},
            }
        ),
        cache=SimpleNamespace(
            completion={
                "completion_sha256": completion_sha,
                "initial_model_state_sha256": initial_sha,
            },
            manifest={"manifest_sha256": manifest_sha},
        ),
    )
    loaded = {
        "CHECKPOINT_SCHEMA": "checkpoint",
        "CHECKPOINT_SCHEMA_VERSION": 3,
        "CHECKPOINT_STATUS": "complete",
        "NO_AUTHORITY": {"training_authorized": False},
    }
    payload = {
        "schema": "checkpoint",
        "schema_version": 3,
        "status": "complete",
        "training_authorized": False,
        "completed_steps": 500,
        "resume_count": 0,
        "identity": {
            "capacity_policy_sha256": policy_sha,
            "prepared_input_artifact_sha256": prepared_sha,
            "cache_completion_sha256": completion_sha,
            "cache_manifest_sha256": manifest_sha,
            "initial_model_state_sha256": initial_sha,
            "runner_implementation_sha256": runner_sha,
            "runner_source_revision_sha256": source_sha,
        },
    }
    return payload, loaded, runtime, runner_sha, source_sha


def test_publication_recovery_checkpoint_binds_completed_unresumed_run() -> None:
    payload, loaded, runtime, runner_sha, source_sha = _checkpoint_fixture()

    assert recovery._validate_publication_recovery_checkpoint(
        payload,
        loaded=loaded,
        runtime=runtime,
        expected_runner_implementation_sha256=runner_sha,
        expected_runner_source_revision_sha256=source_sha,
    ) == payload

    changed = {**payload, "resume_count": 1}
    with pytest.raises(RuntimeError, match="checkpoint disagrees"):
        recovery._validate_publication_recovery_checkpoint(
            changed,
            loaded=loaded,
            runtime=runtime,
            expected_runner_implementation_sha256=runner_sha,
            expected_runner_source_revision_sha256=source_sha,
        )
