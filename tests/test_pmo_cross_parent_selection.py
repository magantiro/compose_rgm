import numpy as np
import pytest

from compose_v4.control.docking_value import identity
from tools.pmo_cross_parent_selection import decision, predict_frozen, transfer_arms


def test_transfer_allocation_is_frozen_distinct_and_order_invariant():
    pool = [f"m{i:02}" for i in range(40)]
    score = np.arange(40)
    a = transfer_arms(pool, score, 0)
    assert a == transfer_arms(pool[::-1], score[::-1], 0)
    assert a["predicted"] == pool[:23:-1]
    assert len(set(a["uniform"])) == 16
    assert a["uniform"] != transfer_arms(pool, score, 1)["uniform"]
    with pytest.raises(ValueError, match="16 distinct"):
        transfer_arms(pool[:15], score[:15], 0)


def test_prediction_reuses_coefficients_without_fitting(monkeypatch):
    model = {"train_indices": [0, 2], "mean": 0.4, "coefficients": [0.2, -0.1]}
    lock = {
        "model": {**model, "snapshot_sha256": identity(model)},
        "all_smiles": ["a", "b", "c"],
        "training_smiles": ["a", "c"],
    }

    def features(smiles):
        assert smiles == ["a", "c", "d"]
        return np.array([[1, 0.1, 0.8], [0.1, 1, 0.2], [0.8, 0.2, 1]]), None

    monkeypatch.setattr("tools.pmo_cross_parent_selection.state_features", features)
    np.testing.assert_allclose(predict_frozen(lock, ["d"]), [0.54])
    with pytest.raises(ValueError, match="overlap"):
        predict_frozen(lock, ["a"])
    lock["model"]["coefficients"][0] = 10
    with pytest.raises(ValueError, match="hash mismatch"):
        predict_frozen(lock, ["d"])


def test_better_mean_without_more_improvers_is_not_positive():
    row = {
        "predicted": {"mean": 0.6, "parent_improvers": 0},
        "uniform": {"mean": 0.5, "parent_improvers": 0},
    }
    assert decision([row, row]) == "do_not_promote_frozen_local_model"
    row["predicted"]["parent_improvers"] = 1
    assert decision([row, row]) == "integrate_local_channel_then_matched_broad_comparison"
