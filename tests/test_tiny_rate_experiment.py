from __future__ import annotations

import numpy as np
import torch

from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.experiments.tiny_rate import (
    build_coverage_batch,
    build_exact_marginal_batch,
    build_tiny_paths,
    sample_piecewise_ancestral,
    train_coverage_batch,
    train_exact_marginal_batch,
)
from compose_v4.model.rate_model import WholeGraphRateModel


def test_tiny_coverage_batch_and_short_training_gate() -> None:
    torch.manual_seed(2)
    torch.set_num_threads(1)
    paths = build_tiny_paths(seed=2)
    examples = build_coverage_batch(paths)
    assert len(paths) == 12
    assert len(examples) >= 30
    model = WholeGraphRateModel(hidden_dim=24, message_passing_steps=1)
    initial, final, _ = train_coverage_batch(model, examples, epochs=20)
    assert final["mean_loss"] < initial["mean_loss"]


def test_untrained_target_free_sampler_still_preserves_hard_invariants() -> None:
    torch.manual_seed(5)
    model = WholeGraphRateModel(hidden_dim=16, message_passing_steps=1).eval()
    rollout = sample_piecewise_ancestral(
        model,
        rng=np.random.default_rng(5),
        operational_horizon=0.5,
        time_step=0.1,
    )
    assert all(is_valid_state(state) for state in rollout.states)
    assert all(is_connected_or_null(state) for state in rollout.states)


def test_exact_marginal_batch_has_late_null_target_and_trains() -> None:
    torch.manual_seed(7)
    paths = build_tiny_paths(seed=7)
    examples = build_exact_marginal_batch(paths, times=(0.1, 0.9))
    null_examples = [
        example for example in examples if example.state.n_real_atoms == 0
    ]
    assert len(null_examples) == 2
    assert all(
        sum(rate for _key, rate in item.teacher_successor_rates) > 0
        for item in null_examples
    )
    model = WholeGraphRateModel(hidden_dim=16, message_passing_steps=1)
    initial, final, _ = train_exact_marginal_batch(
        model,
        examples,
        steps=20,
        batch_size=8,
        seed=7,
    )
    assert final["mean_loss"] < initial["mean_loss"]
