import numpy as np
import pytest
import torch

from compose_v4.control.improvement_value import (
    ImprovementFitConfig,
    ImprovementTrace,
    MonotoneImprovementModel,
    build_targets,
    censored_targets,
    evaluate_improvement,
    fit_improvement_model,
    improvement_parameter_id,
    predict_improvement,
    source_group_split,
)


def trace(name, policy="policy-a", *, weight=None, terminated=False):
    return ImprovementTrace(
        source_id=name,
        trace_id=f"trace-{name}",
        behavior_policy_id=policy,
        features=(float(len(name)), 1.0),
        scores=(0.2, 0.5),
        incumbent=0.2,
        terminated=terminated,
        importance_weight=weight,
    )


def test_right_censoring_never_invents_a_failure():
    labels, mask = censored_targets([0.2, 0.5], 0.2, [1, 2], [0.1, 0.4], terminated=False)
    assert labels.tolist() == [[1, 0], [1, 0]]
    assert mask.tolist() == [[True, True], [True, False]]
    _, ended = censored_targets([0.2, 0.5], 0.2, [1, 2], [0.1, 0.4], terminated=True)
    assert ended.all()


def test_policy_identity_and_group_split_are_explicit():
    with pytest.raises(ValueError, match="behavior-policy"):
        build_targets(
            [trace("a"), trace("b", "policy-b")], [1], [0.1], behavior_policy_id="policy-a"
        )
    targets = build_targets(
        [trace("a"), trace("b", "policy-b", weight=0.4)],
        [1],
        [0.1],
        behavior_policy_id="policy-a",
    )
    assert targets.row_weights.tolist() == pytest.approx([1, 0.4])
    train, calibration = source_group_split(["a", "a", "b", "c"], seed=7)
    assert set(train).isdisjoint(calibration)
    assert {targets for targets in np.array(["a", "a", "b", "c"])[train]}.isdisjoint(
        np.array(["a", "a", "b", "c"])[calibration]
    )
    with pytest.raises(ValueError, match="at least two"):
        source_group_split(["only"])


def test_monotone_parameterization_and_fitting():
    torch.manual_seed(3)
    model = MonotoneImprovementModel(2, 3, 4, hidden=5)
    prediction = model(torch.randn(7, 2)).detach().numpy()
    assert np.all(np.diff(prediction, axis=1) >= -1e-7)
    assert np.all(np.diff(prediction, axis=2) <= 1e-7)

    rows = [
        ImprovementTrace(
            source_id=f"s{i}",
            trace_id=f"t{i}",
            behavior_policy_id="policy-a",
            features=(float(i % 2), float(i)),
            scores=(0.1, 0.8 if i % 2 else 0.15),
            incumbent=0.1,
            terminated=True,
        )
        for i in range(8)
    ]
    targets = build_targets(rows, [1, 2], [0.1, 0.4], behavior_policy_id="policy-a")
    fitted, receipt = fit_improvement_model(
        targets, ImprovementFitConfig(hidden=8, updates=30, seed=11)
    )
    fitted_prediction = predict_improvement(fitted, targets.features)
    assert receipt["identified_cells"] == 32
    assert receipt["model_id"]
    assert receipt["parameter_id"] == improvement_parameter_id(fitted)
    assert receipt["parameters"]
    assert np.all(np.diff(fitted_prediction, axis=1) >= -1e-7)
    assert np.all(np.diff(fitted_prediction, axis=2) <= 1e-7)
    report = evaluate_improvement(targets, fitted_prediction)
    assert report["target_coverage"] == 1
    assert 0 <= report["brier"] <= 1


def test_target_grid_rejects_boolean_thresholds():
    with pytest.raises(ValueError, match="thresholds"):
        censored_targets([0.1], 0.1, [1], [False], terminated=True)
