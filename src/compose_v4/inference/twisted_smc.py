"""Twisted-SMC / Feynman-Kac primitives for long-horizon COMPOSE control.

Pure functions, no chemistry, no Modal -- so the mathematics can be tested on an
exactly enumerable state space before any oracle call is spent.

The object being sampled. For a frozen reference kernel R, a terminal goal
indicator g_z, and a remaining budget b, the finite-horizon value is

    h_b(x; z) = Pr_R[ g_z(X_b) = 1 | X_0 = x ],
    h_0(x)    = g_z(x),
    h_b(x)    = sum_y R(y|x) h_{b-1}(y).

The twisted (Doob h-transform) proposal is

    q_b(y|x) = R(y|x) h_{b-1}(y) / h_b(x),

whose LOCAL NORMALISER IS h_b(x) itself. Budget dependence therefore comes from
h_b, not from a hand-imposed pressure schedule: a bridge state is free to look
bad now provided its future reachability is good. With exact h the path weights
are constant at h_B(x_0), which is the sharpest available check on the weighting.

Generalised twist. In practice h is approximated by a learned value V, used as
`log_pot = beta * V`. beta = 1 with V = log h_hat recovers the exact transform;
beta = 0 recovers the reference process R untouched.
"""

from __future__ import annotations

import numpy as np


def twist(pr, log_pot):
    """One twisted transition. Returns (proposal probabilities, log normaliser).

    `pr` is the reference conditional R(.|x) over the enumerated support and
    `log_pot` is log of the potential on each successor (beta * V).

    The log-normaliser is a TRUE logsumexp. Subtracting the max for numerical
    safety WITHOUT adding it back leaves log Z short by a state-dependent
    constant, which cancels in the proposal (it is renormalised) but silently
    distorts every relative particle weight and therefore resampling. That bug
    is invisible at beta = 0 and only appears once the twist is switched on.
    """
    pr = np.asarray(pr, dtype=np.float64)
    log_pot = np.asarray(log_pot, dtype=np.float64)
    s = pr.sum()
    if not np.isfinite(s) or s <= 0:
        raise ValueError("reference conditional must be positive and finite")
    pr = pr / s
    m = float(np.max(log_pot))
    if not np.isfinite(m):                      # every successor killed
        return np.zeros_like(pr), -np.inf
    w = pr * np.exp(log_pot - m)
    tot = float(w.sum())
    if tot <= 0:
        return np.zeros_like(pr), -np.inf
    return w / tot, m + float(np.log(tot))      # <-- the "+ m" that must be here


def ess(logw):
    """Effective sample size from unnormalised log weights."""
    logw = np.asarray(logw, dtype=np.float64)
    if logw.size == 0:
        return 0.0
    a = logw - logw.max()
    w = np.exp(a)
    s = w.sum()
    if s <= 0:
        return 0.0
    w = w / s
    return float(1.0 / np.sum(w ** 2))


def systematic_resample(logw, u):
    """Systematic resampling. `u` is one uniform in [0,1). Returns indices.

    Unbiased in the sense that E[count_i] = N * w_i, which the toy test checks
    directly rather than assuming.
    """
    logw = np.asarray(logw, dtype=np.float64)
    n = logw.size
    a = logw - logw.max()
    w = np.exp(a)
    w = w / w.sum()
    pos = (float(u) + np.arange(n)) / n
    idx = np.searchsorted(np.cumsum(w), pos)
    return np.clip(idx, 0, n - 1)


def exact_h(R, goal, horizon):
    """Exact finite-horizon values h_0..h_B by backward recursion. Toy use."""
    R = np.asarray(R, dtype=np.float64)
    hs = [np.asarray(goal, dtype=np.float64)]
    for _ in range(horizon):
        hs.append(R @ hs[-1])
    return hs
