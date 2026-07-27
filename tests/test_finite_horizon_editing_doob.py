"""Editing is a fixed-K-step embedded jump chain P(x,y) (the hazard is discarded -- see the process
contract). Exact terminal-reward steering is therefore the finite-horizon Doob transform, with the value
function indexed by the REMAINING edit budget b:

    h_0(x) = g(x)                             # 0 steps left -> terminal value
    h_b(x) = sum_y P(x,y) h_{b-1}(y)          # b steps left  ( = (P^b g)(x) )
    P_b^g(x,y) = P(x,y) h_{b-1}(y) / h_b(x)    # controlled step with b steps left

From x0 with a K-step budget the controlled endpoint equals the exact terminal tilt

    Pr^g(X_K=y | X_0=x0) = Pr(X_K=y | x0) * g(y) / h_K(x0),   h_K(x0) = sum_y Pr(X_K=y|x0) g(y).

These tests build a SMALL enumerable molecular embedded chain (BFS over the real factorized fiber, capped
by heavy-atom count) and verify: the terminal-tilt identity; that DYNAMIC STEERING (swap g->g' at an
intermediate state, apply the continuation for the remaining budget) is exact; and that an early-stop
marginal is NOT the terminal tilt (so exactness is budget-specific -- a budget-conditioned controller is
required).
"""
from __future__ import annotations

import numpy as np

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.factorized_fiber import enumerate_factorized_cnof_fiber
from compose_v4.rewrite.kernel import canonical_state_key

_MAX_HEAVY = 3
_NSLOTS = 4


def _enumerate_embedded_chain(seed_smiles="C"):
    """BFS over the real factorized fiber (successors capped at _MAX_HEAVY heavy atoms) -> a finite,
    row-stochastic molecular embedded jump chain (uniform over legal successors, absorbing where none)."""
    seed = pad_molecular_graph(smiles_to_molecular_graph(seed_smiles), _NSLOTS)
    seed_key = canonical_state_key(seed)
    states = {seed_key: seed}
    edges: dict[str, dict[str, int]] = {}
    frontier = [seed_key]
    while frontier:
        key = frontier.pop()
        if key in edges:
            continue
        counts: dict[str, int] = {}
        fiber = enumerate_factorized_cnof_fiber(states[key], allow_bond_reroute=True)
        for transitions in fiber.by_family.values():
            for t in transitions:
                if int(t.successor.n_real_atoms) <= _MAX_HEAVY:
                    counts[t.successor_key] = counts.get(t.successor_key, 0) + 1
                    if t.successor_key not in states:
                        states[t.successor_key] = t.successor
                        frontier.append(t.successor_key)
        edges[key] = counts
    keys = sorted(states)
    index = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    P = np.zeros((n, n))
    for src, dsts in edges.items():
        i = index[src]
        tot = sum(dsts.values())
        if tot:
            for dst, m in dsts.items():
                P[i, index[dst]] = m / tot
    for i in range(n):
        if P[i].sum() == 0:
            P[i, i] = 1.0
    return keys, states, index, P


def _desirability(states, keys, weight, element):
    idx = ELEMENT_TO_IDX[element]
    return np.array([np.exp(weight * int((states[k].atom_types == idx).sum())) for k in keys])


def _values(P, g, K):
    h = [g.copy()]
    for _ in range(K):
        h.append(P @ h[-1])
    return h


def _controlled_endpoint(P, h, K, x0, n):
    mu = np.zeros(n)
    mu[x0] = 1.0
    for step in range(K):
        b = K - step
        Hb, Hb1 = h[b], h[b - 1]
        live = Hb > 0
        Pg = np.zeros((n, n))
        Pg[live] = P[live] * (Hb1[None, :] / Hb[live][:, None])
        Pg[~live, np.where(~live)[0]] = 1.0
        mu = mu @ Pg
    return mu


def _tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def test_finite_horizon_terminal_tilt_is_exact() -> None:
    keys, states, index, P = _enumerate_embedded_chain()
    n = len(keys)
    assert np.allclose(P.sum(1), 1.0)  # row-stochastic embedded chain
    x0 = index[canonical_state_key(pad_molecular_graph(smiles_to_molecular_graph("C"), _NSLOTS))]
    K = 5
    g = _desirability(states, keys, 1.5, "O")
    h = _values(P, g, K)
    ref = np.zeros(n)
    ref[x0] = 1.0
    ref = ref @ np.linalg.matrix_power(P, K)
    tilt = ref * g / float((ref * g).sum())
    mu = _controlled_endpoint(P, h, K, x0, n)
    assert np.isclose(h[K][x0], float((ref * g).sum()), rtol=1e-9)  # normalizer == h_K(x0)
    assert _tv(mu, tilt) < 1e-9  # controlled endpoint == exact terminal tilt


def test_dynamic_steering_continuation_is_exact() -> None:
    # Swap the objective g->g' at an intermediate state and apply the exact continuation transform for
    # the remaining budget: the continuation endpoint is the g'-tilt from that state.
    keys, states, index, P = _enumerate_embedded_chain()
    n = len(keys)
    x_mid = index[canonical_state_key(pad_molecular_graph(smiles_to_molecular_graph("CCO"), _NSLOTS))]
    bp = 4
    gprime = _desirability(states, keys, 1.2, "N")  # new objective: reward nitrogen
    hp = _values(P, gprime, bp)
    refp = np.zeros(n)
    refp[x_mid] = 1.0
    refp = refp @ np.linalg.matrix_power(P, bp)
    tiltp = refp * gprime / float((refp * gprime).sum())
    mup = _controlled_endpoint(P, hp, bp, x_mid, n)
    assert _tv(mup, tiltp) < 1e-9


def test_early_stop_is_not_the_terminal_law() -> None:
    # Exactness is budget-specific: the controlled marginal after fewer than K steps is NOT the terminal
    # tilt, so one cannot stop early and claim the terminal tilted law (a budget-conditioned controller
    # h_phi(b, x; ...) is required).
    keys, states, index, P = _enumerate_embedded_chain()
    n = len(keys)
    x0 = index[canonical_state_key(pad_molecular_graph(smiles_to_molecular_graph("C"), _NSLOTS))]
    K = 5
    g = _desirability(states, keys, 1.5, "O")
    h = _values(P, g, K)
    ref = np.zeros(n)
    ref[x0] = 1.0
    ref = ref @ np.linalg.matrix_power(P, K)
    tilt = ref * g / float((ref * g).sum())
    mu_early = _controlled_endpoint(P, h, K - 2, x0, n)  # stop 2 steps early
    assert _tv(mu_early, tilt) > 0.02
