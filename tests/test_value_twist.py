"""Regression test for the V1 learned twist (scripts/griddd_value_twist.py).

Validates the module end-to-end on synthetic trajectories -- no base rollouts,
no oracle -- so it guards the reward-to-go labelling, the net, and training
without any heavy compute.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from griddd_value_twist import (
    ValueTwistNet,
    build_reward_to_go_labels,
    fingerprint_array,
    train_value_twist,
)


def test_reward_to_go_is_best_feasible_ahead() -> None:
    # benzene (0.9) is the peak; states before it should target 0.9, after it 0.5.
    traj = [[("CCO", 0.3, True), ("c1ccccc1", 0.9, True), ("CCCCCC", 0.5, True)]]
    features, targets = build_reward_to_go_labels(traj)
    assert features.shape[1] == 2048
    # states are emitted in reversed order: hexane, benzene, ethanol
    np.testing.assert_allclose(targets, [0.5, 0.9, 0.9], rtol=0, atol=1e-6)


def test_infeasible_states_do_not_raise_the_target() -> None:
    traj = [[("CCO", 0.3, True), ("c1ccccc1", 0.99, False), ("CCCCCC", 0.4, True)]]
    _, targets = build_reward_to_go_labels(traj)
    # the 0.99 was infeasible, so it must not count toward reward-to-go
    assert max(targets) < 0.99


def test_net_forward_shape_and_training_runs() -> None:
    net = ValueTwistNet(hidden=32)
    out = net(torch.zeros((3, 2048)))
    assert out.shape == (3,)

    traj = [[("CCO", 0.3, True), ("c1ccccc1", 0.9, True), ("CCCCCC", 0.5, True)]]
    features, targets = build_reward_to_go_labels(traj)
    trained, final_loss = train_value_twist(
        features, targets, epochs=25, batch_size=8, seed=0, hidden=32
    )
    assert np.isfinite(final_loss)
    value = trained.value("c1ccccc1")
    assert np.isfinite(value)


def test_fingerprint_none_on_invalid_smiles() -> None:
    assert fingerprint_array("not_a_molecule)))") is None
    assert fingerprint_array("c1ccccc1") is not None
