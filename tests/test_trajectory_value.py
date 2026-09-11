"""Focused achieved-return and matched decision invariants."""

import numpy as np
import pytest

from compose_v4.control.trajectory_value import achieved_returns, choices, features


def test_delayed_return_uses_only_actual_suffix_with_stop():
    assert achieved_returns([0.5, 0.2, 0.8], (1, 2)) == [[0.5, 0.8], [0.8, 0.8], [0.8, 0.8]]
    with pytest.raises(ValueError, match="invalid"):
        achieved_returns([0.2, np.nan])


def test_feature_graph_not_smiles_order_or_goal():
    assert np.array_equal(features("CCO", 11), features("OCC", 11))
    assert features("CCO", 11)[-1] == np.float32(11 / 64)
    with pytest.raises(ValueError, match="budget"):
        features("CCO", -1)


def test_duplicate_mass_failed_draws_and_paired_randomness():
    a, b = {"smiles": "CC", "score": 0.3}, {"smiles": "CCC", "score": 0.4}
    result = choices([a, a, None, b], [0.8, 0.8, None, 0.5], seed=10)
    for decision in result.values():
        assert np.allclose(decision["reference"], [2 / 3, 1 / 3])
        assert decision["kl"] <= 1 + 1e-10
        assert decision["candidate"] in (a, b)
        assert np.all(np.array(decision["probabilities"]) >= 0.1 * np.array(decision["reference"]))
    assert len({r["u"] for r in result.values()}) == 1
    assert result["future"]["values"] == [0.8, 0.5]
    assert choices([None], [None], seed=0)["future"]["status"] == "abstained"


def test_stop_score_floor_and_conflicting_alias_fail():
    candidate = {"smiles": "CC", "score": 0.7}
    assert choices([candidate], [0.2], seed=0)["future"]["values"] == [0.7]
    with pytest.raises(ValueError, match="aliases"):
        choices([candidate, candidate], [0.2, 0.3], seed=0)
