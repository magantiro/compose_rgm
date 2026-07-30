from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


pytest.importorskip("modal", reason="Modal freeze utility tests require the cloud extra")
module = importlib.import_module("modal_apps.freeze_training_run")


def test_expected_snapshot_steps_are_complete_and_interval_aligned() -> None:
    assert module._expected_snapshot_steps(
        maximum_step=2000,
        snapshot_interval=500,
    ) == (500, 1000, 1500, 2000)
    with pytest.raises(ValueError, match="divisible"):
        module._expected_snapshot_steps(maximum_step=2001, snapshot_interval=500)


def test_snapshot_step_parser_fails_closed() -> None:
    assert module._snapshot_step(Path("checkpoint.step16000.pt")) == 16000
    for name in (
        "checkpoint.pt",
        "checkpoint.step0.pt",
        "checkpoint.step500.bin",
        "checkpoint.step-500.pt",
    ):
        with pytest.raises(ValueError, match="not a step snapshot"):
            module._snapshot_step(Path(name))


def test_run_label_rejects_paths_and_unsafe_characters() -> None:
    module._validate_run_label("compose-v4-ringcore-v1")
    for label in ("../run", "nested/run", "-run", "run label", ""):
        with pytest.raises(ValueError, match="run_label"):
            module._validate_run_label(label)


def test_checkpoint_summary_verifies_exact_recovery_step_and_schema() -> None:
    state = {"head.weight": torch.zeros((2, 3)), "head.bias": torch.zeros(2)}
    payload = {
        "checkpoint_kind": "exact_training_recovery",
        "completed_steps": 500,
        "current_state_dict": state,
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "best_state_dict": {name: value.clone() for name, value in state.items()},
        "best_metrics": {
            "selected_step": 250.0,
            "factorized_gm_loss": 2.0,
        },
        "history": [{"step": 500.0}],
        "initial_validation": {"factorized_gm_loss": 4.0},
        **{key: None for key in module.CHECKPOINT_METADATA_KEYS},
    }
    payload.update(
        {
            "training_steps": 2000,
            "optimizer_kind": "adamw_decoupled_v1",
            "provenance_sha256": "a" * 64,
        }
    )

    summary = module._checkpoint_summary(payload, expected_step=500)

    assert summary["completed_steps"] == 500
    assert summary["history_last_step"] == 500
    assert summary["state_tensor_count"] == 2
    assert len(summary["state_schema_sha256"]) == 64
    assert summary["metadata"]["provenance_sha256"] == "a" * 64

    payload["completed_steps"] = 501
    with pytest.raises(ValueError, match="filename step"):
        module._checkpoint_summary(payload, expected_step=500)


def test_stable_json_hash_is_key_order_invariant() -> None:
    assert module._stable_json_sha256({"a": 1, "b": 2}) == module._stable_json_sha256(
        {"b": 2, "a": 1}
    )
