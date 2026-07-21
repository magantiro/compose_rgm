"""E0: the Doob h-transform enforces rule-closed conditioning EXACTLY on the
actual (size-capped) rewrite CTMC. Fast cap=2 slice."""

import pathlib
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph  # noqa: E402
from compose_v4.chem.state import pad_molecular_graph  # noqa: E402
from compose_v4.rewrite.kernel import canonical_state_key  # noqa: E402
from e0_toy_h_exactness import build_state_space, generator_matrix  # noqa: E402


def test_generator_valid_and_doob_transform_exact():
    keys, states, smiles, edges = build_state_space(cap=2, n_slots=5)
    Q, index = generator_matrix(keys, edges)
    n = len(keys)
    assert n > 5

    # exact CTMC generator: rows sum to 0, off-diagonals >= 0
    assert np.abs(Q.sum(axis=1)).max() < 1e-9
    off = Q - np.diag(np.diag(Q))
    assert off.min() >= -1e-12

    one_A = np.array(
        [1.0 if ("O" in smiles[k] or "o" in smiles[k]) else 0.0 for k in keys]
    )
    assert 0 < one_A.sum() < n  # non-trivial target

    T, M = 1.0, 200
    dt = T / M
    P_dt = expm(Q * dt)
    h = [None] * (M + 1)
    h[M] = one_A.copy()
    for k in range(M - 1, -1, -1):
        h[k] = P_dt @ h[k + 1]

    x0 = index[canonical_state_key(pad_molecular_graph(smiles_to_molecular_graph("C"), 5))]
    P_T = expm(Q * T)
    exact = P_T[x0] * one_A
    exact = exact / exact.sum()

    eps = 1e-12
    mu = np.zeros(n)
    mu[x0] = 1.0
    for k in range(M):
        hk, hk1 = h[k], h[k + 1]
        idx = np.where(hk > eps)[0]
        Ph = np.zeros((n, n))
        Ph[np.ix_(idx, np.arange(n))] = P_dt[idx] * (hk1[None, :] / hk[idx][:, None])
        mu = mu @ Ph
    mu = np.clip(mu, 0, None)
    mu = mu / mu.sum()

    # Doob-tilted kernels telescope to the exact conditional -> machine precision.
    assert np.abs(mu - exact).max() < 1e-9
    # validity-closed under conditioning: no mass leaks outside the target set.
    assert float(mu[one_A == 0].sum()) < 1e-9
