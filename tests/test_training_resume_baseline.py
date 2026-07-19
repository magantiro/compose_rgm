from __future__ import annotations

import pytest

from scripts.train_tracelet_cnof_gate import _validation_baseline_for_run


def test_fresh_training_uses_observed_initial_validation() -> None:
    observed = {"factorized_gm_loss": 12.0, "family_accuracy": 0.1}

    baseline = _validation_baseline_for_run(observed, None)

    assert baseline == observed
    assert baseline is not observed


def test_resume_preserves_original_initial_validation() -> None:
    observed_after_resume = {
        "factorized_gm_loss": 4.0,
        "family_accuracy": 0.7,
    }
    original = {
        "factorized_gm_loss": 12.0,
        "family_accuracy": 0.1,
    }

    baseline = _validation_baseline_for_run(
        observed_after_resume,
        {"initial_validation": original},
    )

    assert baseline == original
    assert baseline != observed_after_resume


def test_resume_rejects_recovery_without_original_baseline() -> None:
    with pytest.raises(ValueError, match="lacks the original"):
        _validation_baseline_for_run(
            {"factorized_gm_loss": 4.0},
            {"completed_steps": 500},
        )
