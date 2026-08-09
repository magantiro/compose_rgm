"""Guards on exact finite-horizon Doob control and the evaluation-semantics preflight.

These are the two pieces where a silent defect is expensive.  The control math
can be wrong in ways that still produce plausible-looking distributions, so the
tests assert the telescoping identity to machine precision rather than to a
tolerance.  And the preflight exists because a tight graph rebuilt from SMILES
drops the whole ``atom_insert`` family without erroring -- a failure that
invalidated three completed experiments -- so a test that only proved it passes
on good input would be worthless.  The negative case is the point.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.experiments.editing_v2_bridge_control import (
    backward_values,
    controlled_kernel,
    propagate,
    terminal_tilt_residual,
)
from compose_v4.experiments.editing_v2_evaluation_semantics import (
    EvaluationSemanticsError,
    assert_production_state_semantics,
)


def _toy_chain(n: int = 12, seed: int = 5) -> np.ndarray:
    """A dense row-stochastic reference law; the control math is chemistry-free."""

    rng = np.random.default_rng(seed)
    R = rng.random((n, n)) ** 3
    R[rng.random((n, n)) < 0.4] = 0.0          # genuine zeros, so support matters
    R[np.arange(n), (np.arange(n) + 1) % n] += 1e-3
    return R / R.sum(axis=1, keepdims=True)


def test_terminal_tilt_is_exact_not_approximate():
    """The controlled endpoint law must EQUAL the reference law tilted by g."""

    R = _toy_chain()
    g = np.zeros(R.shape[0]); g[[3, 7, 9]] = 1.0
    report = terminal_tilt_residual(R, g, budget=5, start=0)

    assert report["reachable"]
    assert report["terminal_tilt_tv"] < 1e-12, report
    assert report["h_partition_gap"] < 1e-12, report
    assert report["max_row_sum_error"] < 1e-12, report
    assert report["backward_residual"] < 1e-12, report
    # A bridge may never create an edge the reference process does not have.
    assert report["support_violations"] == 0


def test_unreachable_target_leaves_the_controlled_row_undefined():
    """h_B == 0 must yield an all-zero (undefined) row, never an epsilon patch."""

    n = 8
    R = np.zeros((n, n))
    R[np.arange(n), np.arange(n)] = 1.0        # every state absorbing
    g = np.zeros(n); g[5] = 1.0                # unreachable from 0

    h = backward_values(R, g, budget=4)
    assert h[4][0] == 0.0

    P = controlled_kernel(R, h[3], h[4])
    assert np.all(P[0] == 0.0), "unreachable row must be undefined, not patched"

    report = terminal_tilt_residual(R, g, budget=4, start=0)
    assert report["reachable"] is False
    assert report["partition"] == 0.0


def test_retargeting_midway_reproduces_the_bridge_from_the_realised_state():
    """Switching the objective mid-trajectory must stay exactly on the bridge."""

    R = _toy_chain(seed=11)
    n = R.shape[0]
    budget, switch = 6, 2
    g1 = np.zeros(n); g1[[2, 5]] = 1.0
    g2 = np.zeros(n); g2[[4, 8, 10]] = 1.0

    h1 = backward_values(R, g1, budget)
    nu = np.zeros(n); nu[0] = 1.0
    for b in (budget, budget - 1):
        nu = nu @ controlled_kernel(R, h1[b - 1], h1[b])

    remaining = budget - switch
    h2 = backward_values(R, g2, remaining)
    continued = nu.copy()
    for b in range(remaining, 0, -1):
        continued = continued @ controlled_kernel(R, h2[b - 1], h2[b])

    R2 = np.linalg.matrix_power(R, remaining)
    denom = (R2 @ g2)[:, None]
    per_state = np.divide(R2 * g2[None, :], denom, out=np.zeros_like(R2), where=denom > 0)
    assert 0.5 * np.abs(continued - nu @ per_state).sum() < 1e-12


def test_conditioning_amplifies_a_rare_target_exactly():
    """Reach probability goes to one, and the partition function agrees with h."""

    R = _toy_chain(seed=3)
    g = np.zeros(R.shape[0]); g[6] = 1.0
    budget, start = 5, 0

    h = backward_values(R, g, budget)
    reference = float(np.linalg.matrix_power(R, budget)[start] @ g)
    assert 0.0 < reference < 0.5, "target should be rare enough to be interesting"

    mu = propagate(R, h, budget, start, R.shape[0])
    assert abs(float(mu @ g) - 1.0) < 1e-12
    assert abs(float(h[budget][start]) - reference) < 1e-12


def test_preflight_rejects_a_census_mismatch():
    """The negative case is the reason this preflight exists."""

    class _Model:
        pass

    import compose_v4.experiments.editing_v2_evaluation_semantics as mod

    original = mod.realized_family_census
    mod.realized_family_census = lambda model, state, time=0.5: {
        "atom_delete": 24, "atom_restate": 328,
    }
    try:
        # A tight graph loses atom_insert entirely; the recorded census has it.
        with pytest.raises(EvaluationSemanticsError, match="atom_insert"):
            assert_production_state_semantics(
                _Model(),
                [(object(), {"atom_delete": 24, "atom_restate": 328, "atom_insert": 663})],
            )
        # Agreement passes, and zero-count families are not spurious mismatches.
        report = assert_production_state_semantics(
            _Model(),
            [(object(), {"atom_delete": 24, "atom_restate": 328, "cycle_insert": 0})],
        )
        assert report["states_checked"] == 1
    finally:
        mod.realized_family_census = original


def test_preflight_refuses_to_pass_on_zero_samples():
    """An empty check must fail loudly rather than report success."""

    with pytest.raises(EvaluationSemanticsError, match="zero samples"):
        assert_production_state_semantics(object(), [])
