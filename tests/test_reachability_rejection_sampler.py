"""Stage A1: the rejection sampler reproduces the exact Doob kernel.

Correctness here is a MATHEMATICAL property of the sampler, independent of what
`R` and `h` happen to be, so it is verified on synthetic systems -- free,
immediate, and before any compute is spent on the 967-state system or on
training `h_phi`.

What is being checked, from `docs/LEARNED_REACHABILITY_CONTROLLER.md`:

  P^phi_b(y|x,z)  proportional to  R_theta(y|x) * h(b-1, y; z)

drawn by: sample a MARK from the base law, execute only that mark, reject
self-loops, accept with probability h(y). The two non-obvious properties are
that ALIASES aggregate correctly for free (because h is a function of the
canonical state), and that BATCHED proposals leave the law unchanged (because
proposal order and uniforms are fixed in advance).
"""

from __future__ import annotations

import numpy as np
import pytest


def exact_controlled(r_marks, mark_to_state, h, n_states, source=None):
    """The law we must reproduce: pushforward of R, tilted by h, normalized."""
    w = np.zeros(n_states)
    for m, p in enumerate(r_marks):
        y = mark_to_state[m]
        if source is not None and y == source:
            continue                      # self-loops are not successors
        w[y] += p * h[y]
    t = w.sum()
    return w / t if t > 0 else w


def sequential_sampler(r_marks, mark_to_state, h, rng, source=None, cap=100_000):
    """One draw: mark -> execute -> reject self-loop -> accept w.p. h(y)."""
    p = np.asarray(r_marks, float)
    p = p / p.sum()
    for _ in range(cap):
        m = int(rng.choice(len(p), p=p))
        y = mark_to_state[m]
        if source is not None and y == source:
            continue
        if rng.random() <= h[y]:
            return y
    raise RuntimeError("no acceptance within cap")


def batched_sampler(r_marks, mark_to_state, h, rng, batch=8, source=None,
                    cap=10_000):
    """Draw `batch` proposals AND uniforms in advance; accept the first in order.

    Must be distributionally identical to `sequential_sampler`: the sequential
    algorithm would have examined the same proposals in the same order against
    the same uniforms.
    """
    p = np.asarray(r_marks, float)
    p = p / p.sum()
    for _ in range(cap):
        marks = rng.choice(len(p), size=batch, p=p)
        us = rng.random(batch)
        for m, u in zip(marks, us):
            y = mark_to_state[int(m)]
            if source is not None and y == source:
                continue
            if u <= h[y]:
                return y
    raise RuntimeError("no acceptance within cap")


def _empirical(fn, n_states, draws, **kw):
    rng = np.random.default_rng(kw.pop("seed", 0))
    counts = np.zeros(n_states)
    for _ in range(draws):
        counts[fn(rng=rng, **kw)] += 1
    return counts / counts.sum()


def _tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())


# --------------------------------------------------------------------------
# The core claim
# --------------------------------------------------------------------------

def test_sampler_reproduces_the_exact_doob_kernel():
    rng = np.random.default_rng(0)
    n_marks, n_states = 40, 12
    r = rng.random(n_marks); r /= r.sum()
    m2s = rng.integers(0, n_states, n_marks)
    h = rng.random(n_states)

    exact = exact_controlled(r, m2s, h, n_states)
    emp = _empirical(sequential_sampler, n_states, 200_000,
                     r_marks=r, mark_to_state=m2s, h=h, seed=1)
    assert _tv(exact, emp) < 0.01, f"TV {_tv(exact, emp):.4f} -- sampler is not exact"


def test_ALIASES_aggregate_correctly_for_free():
    """Many marks -> one canonical state. The whole reason h is state-level.

    A mark-level control head could assign different values to aliases of the
    same molecule and silently break the quotient; a state-level h cannot.
    """
    # 10 marks all produce state 0; 1 mark produces state 1.
    r = np.array([0.05] * 10 + [0.5]); r /= r.sum()
    m2s = np.array([0] * 10 + [1])
    h = np.array([0.2, 0.9])

    exact = exact_controlled(r, m2s, h, 2)
    # By hand: state 0 mass 0.5*0.2=0.10, state 1 mass 0.5*0.9=0.45.
    assert exact[0] == pytest.approx(0.10 / 0.55, abs=1e-12)
    assert exact[1] == pytest.approx(0.45 / 0.55, abs=1e-12)

    emp = _empirical(sequential_sampler, 2, 100_000,
                     r_marks=r, mark_to_state=m2s, h=h, seed=2)
    assert _tv(exact, emp) < 0.01


def test_BATCHED_sampler_is_distributionally_identical():
    rng = np.random.default_rng(3)
    n_marks, n_states = 30, 8
    r = rng.random(n_marks); r /= r.sum()
    m2s = rng.integers(0, n_states, n_marks)
    h = rng.random(n_states)
    exact = exact_controlled(r, m2s, h, n_states)
    for batch in (1, 4, 16):
        emp = _empirical(batched_sampler, n_states, 120_000,
                         r_marks=r, mark_to_state=m2s, h=h, batch=batch, seed=4)
        assert _tv(exact, emp) < 0.012, f"batch={batch} TV {_tv(exact, emp):.4f}"


def test_self_loops_are_excluded_not_reweighted():
    """A virtual mark must be redrawn, leaving the law over TRUE successors."""
    r = np.array([0.6, 0.25, 0.15])
    m2s = np.array([0, 1, 2])            # state 0 IS the source
    h = np.array([0.9, 0.5, 0.5])
    exact = exact_controlled(r, m2s, h, 3, source=0)
    assert exact[0] == 0.0
    emp = _empirical(sequential_sampler, 3, 100_000,
                     r_marks=r, mark_to_state=m2s, h=h, source=0, seed=5)
    assert emp[0] == 0.0
    assert _tv(exact, emp) < 0.01


def test_h_equal_everywhere_recovers_the_base_pushforward():
    """A constant h must leave R_theta's successor law untouched."""
    rng = np.random.default_rng(6)
    r = rng.random(25); r /= r.sum()
    m2s = rng.integers(0, 6, 25)
    base = exact_controlled(r, m2s, np.ones(6), 6)
    tilted = exact_controlled(r, m2s, np.full(6, 0.37), 6)
    assert _tv(base, tilted) < 1e-12


def test_zero_h_states_are_unreachable_but_do_not_break_normalization():
    r = np.array([0.5, 0.3, 0.2])
    m2s = np.array([0, 1, 2])
    exact = exact_controlled(r, m2s, np.array([0.0, 0.8, 0.4]), 3)
    assert exact[0] == 0.0
    assert float(exact.sum()) == pytest.approx(1.0)
