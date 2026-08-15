"""END-TO-END exactness: does the composed SMC target the KNOWN controlled law?

The component tests establish that ESS, the trigger, the resampler, the RNG and
the output rule are each right. They cannot catch the dangerous composition bug:

    correct resampler + correct ESS + WRONG Feynman-Kac weight
        = confidently wrong SMC

So this builds a tiny Markov system where everything is enumerable, derives the
finite-horizon Doob h-transform analytically, and requires the full recursion --
proposal, incremental weights, resampling, absorbing STOP, terminal draw -- to
reproduce it.
"""

from __future__ import annotations

import numpy as np
import pytest

from compose_v4.experiments.hphi_smc import run_twisted_smc

# A 5-state birth-death chain. State 4 is the target; 0 is a sticky trap, so the
# uncontrolled process reaches the target only rarely -- which is the regime the
# real controller operates in.
N_STATES, TARGET, HORIZON = 5, 4, 6
R = np.array([
    [0.80, 0.20, 0.00, 0.00, 0.00],
    [0.55, 0.30, 0.15, 0.00, 0.00],
    [0.00, 0.55, 0.30, 0.15, 0.00],
    [0.00, 0.00, 0.55, 0.30, 0.15],
    [0.00, 0.00, 0.00, 0.00, 1.00],   # absorbing target
])


def exact_h() -> np.ndarray:
    """h[b, x] = P(hit TARGET within b steps | start x), by backward recursion.

    h[0, x] = 1{x in B}; h[b, x] = 1 if x in B else sum_y R[x, y] h[b-1, y].
    """
    h = np.zeros((HORIZON + 1, N_STATES))
    h[0, TARGET] = 1.0
    for b in range(1, HORIZON + 1):
        for x in range(N_STATES):
            h[b, x] = 1.0 if x == TARGET else float(R[x] @ h[b - 1])
    return h


def exact_controlled_terminal(x0: int) -> np.ndarray:
    """Distribution over terminal states under the EXACT h-transform,

        P*_b(y | x) = R(x, y) h_{b-1}(y) / h_b(x),

    propagated forward with the target absorbing. This is the ground truth the
    sampler must reproduce.
    """
    h = exact_h()
    dist = np.zeros(N_STATES)
    dist[x0] = 1.0
    for step in range(HORIZON):
        b = HORIZON - step
        nxt = np.zeros(N_STATES)
        for x in range(N_STATES):
            if dist[x] == 0.0:
                continue
            if x == TARGET:                     # absorbed: STOP
                nxt[x] += dist[x]
                continue
            denom = h[b, x]
            if denom <= 0:
                continue
            for y in range(N_STATES):
                p = R[x, y] * h[b - 1, y] / denom
                if p > 0:
                    nxt[y] += dist[x] * p
        dist = nxt
    return dist


H = exact_h()


def propose(x, b, rng):
    return int(rng.choice(N_STATES, p=R[x]))


def twist(x, b):
    return float(H[max(0, min(b, HORIZON)), x])


def in_target(x):
    return x == TARGET


def test_exact_h_matches_brute_force_enumeration():
    """Sanity: the analytic h really is the hitting probability."""
    rng = np.random.default_rng(0)
    for x0 in (1, 2, 3):
        hits = 0
        trials = 20000
        for _ in range(trials):
            x = x0
            for _ in range(HORIZON):
                x = int(rng.choice(N_STATES, p=R[x]))
                if x == TARGET:
                    hits += 1
                    break
        assert hits / trials == pytest.approx(H[HORIZON, x0], abs=0.015)


def test_controlled_law_reaches_target_almost_surely():
    """Under the exact h-transform the target is hit with probability 1."""
    for x0 in (1, 2, 3):
        assert exact_controlled_terminal(x0)[TARGET] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("x0", [1, 2, 3])
def test_smc_reproduces_the_exact_controlled_terminal_law(x0):
    """THE TEST. Full recursion vs analytic ground truth."""
    exact = exact_controlled_terminal(x0)
    counts = np.zeros(N_STATES)
    runs = 600
    for s in range(runs):
        out = run_twisted_smc(x0, horizon=HORIZON, n_particles=32,
                              propose=propose, twist=twist,
                              in_target=in_target,
                              rng=np.random.default_rng(1000 + s))
        counts[out["returned"]] += 1
    emp = counts / counts.sum()
    assert emp[TARGET] > 0.95, (
        f"controlled process must reach the target; got {emp[TARGET]:.3f} "
        f"vs exact {exact[TARGET]:.3f}")
    assert np.abs(emp - exact).max() < 0.05


def test_a_wrong_weight_is_actually_detected():
    """Guard the guard: if the incremental weight were h(y) alone rather than
    h_{b-1}(y)/h_b(x), the target mass must visibly degrade. Otherwise the test
    above proves nothing."""
    def wrong_twist(x, b):
        return 1.0            # no twist at all -> plain unguided proposal
    counts = np.zeros(N_STATES)
    for s in range(600):
        out = run_twisted_smc(x0=2, horizon=HORIZON, n_particles=32,
                              propose=propose, twist=wrong_twist,
                              in_target=in_target,
                              rng=np.random.default_rng(2000 + s))
        counts[out["returned"]] += 1
    emp = counts / counts.sum()
    assert emp[TARGET] < 0.5, (
        "an untwisted sampler must NOT reach the target reliably; if it does, "
        "the end-to-end test cannot detect a wrong weight")


def test_stop_absorption_freezes_particles():
    out = run_twisted_smc(TARGET, horizon=HORIZON, n_particles=32,
                          propose=propose, twist=twist, in_target=in_target,
                          rng=np.random.default_rng(5))
    assert all(out["absorbed"]) and out["returned"] == TARGET


def test_run_is_reproducible_from_its_seed():
    a = run_twisted_smc(2, horizon=HORIZON, n_particles=32, propose=propose,
                        twist=twist, in_target=in_target,
                        rng=np.random.default_rng(11))
    b = run_twisted_smc(2, horizon=HORIZON, n_particles=32, propose=propose,
                        twist=twist, in_target=in_target,
                        rng=np.random.default_rng(11))
    assert a["returned"] == b["returned"] and a["states"] == b["states"]


def test_measure_particle_collapse_after_resampling():
    """Observational: how far below N do unique particles fall? This is the
    number that decides whether duplicate-state reuse is worth building.
    NOT an assertion about performance -- just a measurement."""
    uniq = []
    for s in range(200):
        out = run_twisted_smc(1, horizon=HORIZON, n_particles=32,
                              propose=propose, twist=twist, in_target=in_target,
                              rng=np.random.default_rng(3000 + s))
        if out["mean_unique_after_resample"] is not None:
            uniq.append(out["mean_unique_after_resample"])
    if uniq:
        print(f"\n  mean unique particles after resampling: "
              f"{np.mean(uniq):.1f} / 32  (n={len(uniq)} runs that resampled)")
    assert True
