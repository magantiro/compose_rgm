"""Behavior checks for the QED particle operations."""

import numpy as np

from compose_v4.experiments.qed_particle_ops import (
    effective_sample_size,
    normalized_weights,
    should_resample,
    systematic_resample,
    terminal_output,
)


def test_log_weights_normalize_and_resample_only_below_half_ess() -> None:
    weights = normalized_weights(np.array([0.0, 0.0, -np.inf, -np.inf]))
    np.testing.assert_array_equal(weights, np.array([0.5, 0.5, 0.0, 0.0]))
    assert effective_sample_size(weights) == 2.0
    assert not should_resample(weights)
    assert should_resample(np.array([1.0, 0.0, 0.0, 0.0]))


def test_systematic_resampling_and_terminal_draw_are_seeded() -> None:
    weights = np.array([0.1, 0.7, 0.2])
    np.testing.assert_array_equal(
        systematic_resample(weights, np.random.default_rng(7)), np.array([1, 1, 2])
    )
    assert terminal_output(np.log(np.array([0.25, 0.75])), np.random.default_rng(7)) == (
        1,
        "OK",
    )


def test_extinct_population_has_no_terminal_particle() -> None:
    assert terminal_output(np.array([-np.inf, -np.inf]), np.random.default_rng(7)) == (
        None,
        "EXTINCT_NO_HIT",
    )
