from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_qed_successor_guidance import (
    CONTROL_EQUATION,
    _paired_summary,
    _select_with_uniform,
)


def test_common_uniform_selection_replays_different_soft_controls() -> None:
    assert _select_with_uniform(np.asarray([0.25, 0.25, 0.25, 0.25]), 0.6) == 2
    assert _select_with_uniform(np.asarray([0.05, 0.05, 0.8, 0.1]), 0.6) == 2
    assert "lambda_beta(x,t)=lambda_theta(x,t)" in CONTROL_EQUATION


def test_paired_summary_reports_qed_and_target_distance_changes() -> None:
    control = [
        {"qed": 0.4},
        {"qed": 0.8},
    ]
    candidate = [
        {"qed": 0.6},
        {"qed": 0.9},
    ]
    report = _paired_summary(control, candidate, target_qed=0.9)
    assert report["mean_paired_qed_delta"] == pytest.approx(0.15)
    assert report["qed_improved_states"] == 2
    assert report["mean_paired_target_error_improvement"] == pytest.approx(0.15)
    assert report["target_error_improved_states"] == 2
