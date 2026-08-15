"""Mechanical qualification of the frozen SMC mathematics. NO model required.

SMC is supposed to produce different trajectories from rejection, so it cannot
be qualified by parity against the previous sampler. The oracle is the
mathematics, and these are the exact checks that oracle admits.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.experiments.hphi_smc import (
    CANDIDATES_PER_SOURCE,
    ESS_FRACTION,
    N_PARTICLES,
    effective_sample_size,
    normalized_weights,
    sample_terminal_particle,
    should_resample,
    smc_synchronisation,
    systematic_resample,
)


def test_frozen_constants_match_the_preregistration():
    assert N_PARTICLES == 32
    assert ESS_FRACTION == 0.5
    assert CANDIDATES_PER_SOURCE == 20


def test_ess_is_exactly_n_for_uniform_weights():
    for n in (2, 8, 32):
        assert effective_sample_size(np.full(n, 1.0 / n)) == pytest.approx(n)


def test_ess_is_exactly_one_for_a_point_mass():
    w = np.zeros(32); w[7] = 1.0
    assert effective_sample_size(w) == pytest.approx(1.0)


def test_ess_is_between_one_and_n():
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = rng.random(32); w /= w.sum()
        assert 1.0 <= effective_sample_size(w) <= 32.0 + 1e-9


def test_resample_trigger_is_strict_at_half():
    n = 32
    # Construct weights with ESS exactly N/2 = 16: k uniform, rest zero.
    w = np.zeros(n); w[:16] = 1.0 / 16
    assert effective_sample_size(w) == pytest.approx(16.0)
    assert not should_resample(w, n), "ESS == N/2 must NOT trigger (strict <)"
    w2 = np.zeros(n); w2[:15] = 1.0 / 15
    assert should_resample(w2, n), "ESS just below N/2 must trigger"


def test_max_shift_does_not_change_normalized_weights():
    lw = np.array([-1000.5, -1002.0, -999.25, -1001.0])
    a = normalized_weights(lw)
    b = normalized_weights(lw + 500.0)          # shift is numerical only
    assert np.allclose(a, b, rtol=0, atol=0)
    assert a.sum() == pytest.approx(1.0)


def test_systematic_resample_returns_n_indices_in_range():
    rng = np.random.default_rng(1)
    w = rng.random(32); w /= w.sum()
    idx = systematic_resample(w, np.random.default_rng(2))
    assert len(idx) == 32 and idx.min() >= 0 and idx.max() < 32


def test_systematic_resample_is_reproducible_from_the_seed():
    w = np.full(32, 1.0 / 32)
    a = systematic_resample(w, np.random.default_rng(7))
    b = systematic_resample(w, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_systematic_resample_is_unbiased_in_expectation():
    # Counts should track weights: low variance, not merely correct on average.
    w = np.array([0.5, 0.25, 0.125, 0.125])
    counts = np.zeros(4)
    for s in range(4000):
        idx = systematic_resample(w, np.random.default_rng(s))
        counts += np.bincount(idx, minlength=4)
    assert np.allclose(counts / counts.sum(), w, atol=0.01)


def test_systematic_resample_beats_multinomial_variance():
    """The whole point of systematic resampling is lower variance."""
    w = np.full(16, 1.0 / 16)
    sys_var, mult_var = [], []
    for s in range(400):
        sys_var.append(np.bincount(systematic_resample(w, np.random.default_rng(s)),
                                   minlength=16).var())
        mult_var.append(np.bincount(np.random.default_rng(s).choice(16, 16, p=w),
                                    minlength=16).var())
    assert np.mean(sys_var) < np.mean(mult_var)


def test_uniform_weights_give_a_near_identity_resample():
    # With equal weights the comb picks each particle exactly once.
    w = np.full(32, 1.0 / 32)
    idx = systematic_resample(w, np.random.default_rng(3))
    assert sorted(idx.tolist()) == list(range(32))


def test_terminal_output_samples_from_the_normalized_measure():
    """The frozen output rule -- NOT best-of-population."""
    w = np.array([0.7, 0.2, 0.1])
    counts = np.zeros(3)
    for s in range(6000):
        counts[sample_terminal_particle(w, np.random.default_rng(s))] += 1
    assert np.allclose(counts / counts.sum(), w, atol=0.02)


def test_terminal_output_can_return_a_non_maximal_particle():
    """A best-of-population rule could never do this. That is the difference."""
    w = np.array([0.6, 0.4])
    picks = {sample_terminal_particle(w, np.random.default_rng(s))
             for s in range(50)}
    assert picks == {0, 1}, "output must not collapse onto the heaviest particle"


def test_synchronisation_resamples_only_when_degenerate():
    rng = np.random.default_rng(0)
    healthy = smc_synchronisation(np.zeros(32), rng)      # uniform -> ESS = 32
    assert not healthy.resampled and healthy.ess == pytest.approx(32.0)
    assert np.array_equal(healthy.indices, np.arange(32))

    lw = np.full(32, -50.0); lw[0] = 0.0                  # near point mass
    degenerate = smc_synchronisation(lw, rng)
    assert degenerate.resampled and degenerate.ess < 16.0


def test_resampling_collapses_unique_particles():
    """Duplicate states after resampling are what make exact computation reuse
    worthwhile later -- this pins that the collapse actually happens."""
    lw = np.full(32, -50.0); lw[0] = 0.0
    out = smc_synchronisation(lw, np.random.default_rng(1))
    assert out.resampled and out.n_unique < 32


# --- FROZEN EXTINCTION RULE (preregistration section 13.1) -------------------

def test_extinction_detected_when_every_weight_is_zero():
    from compose_v4.experiments.hphi_smc import (
        EXTINCT_NO_HIT, is_extinct, terminal_output)
    lw = np.full(32, -np.inf)
    assert is_extinct(lw)
    j, status = terminal_output(lw, np.random.default_rng(0))
    assert j is None and status == EXTINCT_NO_HIT


def test_a_single_surviving_particle_is_NOT_extinction():
    """One particle in B is enough support; it must be returned, not declared
    extinct. This is the boundary the rule turns on."""
    from compose_v4.experiments.hphi_smc import is_extinct, terminal_output
    lw = np.full(32, -np.inf)
    lw[7] = 0.0
    assert not is_extinct(lw)
    j, status = terminal_output(lw, np.random.default_rng(0))
    assert j == 7 and status == "OK"


def test_healthy_population_samples_normally():
    from compose_v4.experiments.hphi_smc import is_extinct, terminal_output
    lw = np.zeros(32)
    assert not is_extinct(lw)
    j, status = terminal_output(lw, np.random.default_rng(3))
    assert status == "OK" and 0 <= j < 32


def test_extinction_never_returns_a_particle_index():
    """Guards against a best-particle / last-nondegenerate fallback creeping in:
    an extinct run must yield NO index at all, so the caller is forced to use
    x_0 rather than any molecule the sampler happened to like."""
    from compose_v4.experiments.hphi_smc import terminal_output
    for seed in range(50):
        j, status = terminal_output(np.full(32, -np.inf),
                                    np.random.default_rng(seed))
        assert j is None and status == "EXTINCT_NO_HIT"


def test_extinction_does_not_raise_or_produce_nan():
    """The pre-freeze implementation produced NaN and raised ValueError here."""
    from compose_v4.experiments.hphi_smc import terminal_output
    j, status = terminal_output(np.full(32, -np.inf), np.random.default_rng(1))
    assert j is None and isinstance(status, str)
