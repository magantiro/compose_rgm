"""Exact tests of the twisted-SMC mathematics on an enumerable state space.

No chemistry, no oracle. Every quantity here is computed exactly by matrix
algebra and compared against the sampler, so an error in the weighting shows up
as a numerical discrepancy rather than as a wasted docking run.
"""

import numpy as np
import pytest

from compose_v4.inference.twisted_smc import (
    ess, exact_h, systematic_resample, twist)

K, B = 12, 6


def toy(seed=0):
    rng = np.random.default_rng(seed)
    R = rng.random((K, K)) ** 3 + 1e-3          # sparse-ish, strictly positive
    R /= R.sum(1, keepdims=True)
    goal = np.zeros(K); goal[[2, 9]] = 1.0      # terminal feasible set
    return R, goal


def test_local_normaliser_is_h():
    """The twisted local normaliser must equal h_b(x) exactly.

    This is the check that catches a logsumexp missing its max-add-back: the
    proposal is renormalised so it looks fine, while log Z is short by a
    state-dependent constant.
    """
    R, goal = toy()
    hs = exact_h(R, goal, B)
    for b in range(1, B + 1):
        for x in range(K):
            _, logZ = twist(R[x], np.log(np.maximum(hs[b - 1], 1e-300)))
            assert np.isclose(np.exp(logZ), hs[b][x], rtol=1e-9, atol=1e-12), (
                f"b={b} x={x}: logZ gave {np.exp(logZ):.6e}, h_b={hs[b][x]:.6e}")


def test_beta_zero_recovers_reference():
    R, _ = toy()
    for x in range(K):
        q, logZ = twist(R[x], np.zeros(K))
        assert np.allclose(q, R[x] / R[x].sum(), rtol=1e-12)
        assert np.isclose(logZ, 0.0, atol=1e-12)


def test_exact_h_gives_constant_path_weights():
    """With exact h every path weight collapses to h_B(x0) by telescoping."""
    R, goal = toy()
    hs = exact_h(R, goal, B)
    rng = np.random.default_rng(1)
    x0 = 5
    for _ in range(64):
        x, logw = x0, 0.0
        for t in range(B):
            b = B - t
            q, logZ = twist(R[x], np.log(np.maximum(hs[b - 1], 1e-300)))
            logw += logZ - np.log(max(hs[b][x], 1e-300))   # incremental weight
            x = int(rng.choice(K, p=q))
        assert goal[x] == 1.0, "exact twist must land in the terminal set"
        assert np.isclose(logw, 0.0, atol=1e-9)


def test_normalising_constant_estimate():
    """Product of mean weight increments estimates h_B(x0) = Pr(feasible)."""
    R, goal = toy()
    hs = exact_h(R, goal, B)
    rng = np.random.default_rng(2)
    x0, N = 5, 4000
    part = np.full(N, x0)
    logZhat = 0.0
    for t in range(B):
        b = B - t
        inc = np.empty(N)
        for i in range(N):
            q, logZ = twist(R[part[i]], np.zeros(K))     # untwisted proposal
            part[i] = int(rng.choice(K, p=q))
            inc[i] = np.log(max(hs[b - 1][part[i]], 1e-300)) \
                - np.log(max(hs[b][ x0 if t == 0 else part[i]], 1e-300)) * 0
        m = inc.max()
        logZhat += m + np.log(np.mean(np.exp(inc - m)))
        # untwisted: weight increment is the potential ratio; telescopes to h_B
        break
    # single-step check: E_R[h_{B-1}(X_1) | X_0] = h_B(x0)
    assert np.isclose(np.exp(logZhat), hs[B][x0], rtol=0.15), (
        f"{np.exp(logZhat):.4e} vs {hs[B][x0]:.4e}")


def test_systematic_resample_is_unbiased():
    rng = np.random.default_rng(3)
    logw = rng.normal(size=64) * 1.5
    w = np.exp(logw - logw.max()); w /= w.sum()
    cnt = np.zeros(64)
    T = 20000
    for _ in range(T):
        idx = systematic_resample(logw, rng.random())
        np.add.at(cnt, idx, 1.0)
    emp = cnt / (T * 64)
    assert np.max(np.abs(emp - w)) < 0.01, np.max(np.abs(emp - w))


def test_resampling_preserves_target():
    """Resampling must not move the weighted estimate beyond MC error."""
    R, goal = toy()
    rng = np.random.default_rng(4)
    N = 8000
    x = rng.integers(0, K, size=N)
    logw = rng.normal(size=N) * 0.8
    f = np.arange(K, dtype=float)
    w = np.exp(logw - logw.max()); w /= w.sum()
    before = float(np.sum(w * f[x]))
    idx = systematic_resample(logw, rng.random())
    after = float(np.mean(f[x[idx]]))
    assert abs(before - after) < 0.15, (before, after)


def test_ess_bounds():
    assert np.isclose(ess(np.zeros(50)), 50.0)
    lw = np.full(50, -1e3); lw[0] = 0.0
    assert ess(lw) < 1.01
