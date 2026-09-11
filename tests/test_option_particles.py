import numpy as np
import pytest

from compose_v4.control.option_particles import advance, log_potentials


def test_potentials_telescope_with_nonuniform_roots_and_no_resampling():
    rng = np.random.default_rng(0)
    first = advance([-np.log(3)] * 3, [0] * 3, [0.2, 2, 0.7], [True] * 3, rng, resample=False)
    last = advance(
        first["log_weights"], first["log_potential"], [4, 1, 2], [True] * 3, rng, resample=False
    )
    expected = np.exp([4.0, 1.0, 2.0])
    assert np.allclose(last["weights"], expected / sum(expected))


def test_resampling_inherits_potential_and_resets_only_weight():
    result = advance(
        [-np.log(4)] * 4,
        [0] * 4,
        [10, 0, 0, 0],
        [True] * 4,
        np.random.default_rng(0),
        resample=True,
    )
    assert result["resampled"]
    assert result["indices"] == [0] * 4
    assert result["log_potential"] == [10] * 4
    assert result["log_weights"] == [-np.log(4)] * 4
    # Identical molecules still have four independent particle slots.
    next_step = advance(
        result["log_weights"],
        result["log_potential"],
        [10] * 4,
        [True] * 4,
        np.random.default_rng(1),
        resample=True,
    )
    assert next_step["ess"] == 4
    assert not next_step["resampled"]


def test_death_extinction_and_strict_ess_boundary():
    rng = np.random.default_rng(0)
    result = advance(
        [0] * 4, [0] * 4, [0, 0, None, None], [True, True, False, False], rng, resample=True
    )
    assert result["ess"] == 2
    assert not result["resampled"]
    assert result["log_weights"][2:] == [None, None]
    extinct = advance(
        result["log_weights"], result["log_potential"], [None] * 4, [False] * 4, rng, resample=True
    )
    assert extinct["status"] == "extinct"
    assert extinct["weights"] == [0] * 4


def test_exact_terminal_boundary_replaces_any_heuristic():
    for arm in ("reference", "immediate", "future"):
        assert log_potentials(arm, [0.2, None, 0.4], [0.8, None, 0.1], terminal=True, beta=10) == [
            2,
            None,
            4,
        ]
    assert log_potentials("reference", [0.2], [0.8], terminal=False, beta=10) == [0]
    assert log_potentials("future", [0.2], [0.8], terminal=False, beta=10) == [8]
    with pytest.raises(ValueError, match="finite score"):
        log_potentials("future", [0.2], [float("nan")], terminal=False, beta=10)
    with pytest.raises(ValueError, match="equal nonzero"):
        advance([0], [], [], [], np.random.default_rng(0), resample=False)
