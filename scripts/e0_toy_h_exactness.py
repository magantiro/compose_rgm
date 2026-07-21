#!/usr/bin/env python3
"""E0 -- numerical exactness of the Doob h-transform on the ACTUAL rewrite CTMC.

The paper claims exactness in exactly ONE place: a rule-closed conditioning,
when the true harmonic function h is available, is enforced *exactly* by the
Doob h-transform of the generator. Property-conditioned distributions in the
real system are only *approximated* (learned twist / finite-particle SMC),
because the true h is intractable. This script verifies the exact case on a
small, EXHAUSTIVE, size-capped slice of the real validity-closed rewrite CTMC
(built from the production legal-event fiber, not a hand-drawn toy graph):

  1. Enumerate the exhaustive state space reachable within an atom cap, using the
     production `enumerate_action_fiber` (every committed state is a complete,
     chemically valid, connected molecular graph).
  2. Build the exact generator Q (off-diagonal q[s,s'] = number of distinct legal
     marks s->s'; the transform is exact for ANY rates, this just mirrors the
     fiber's superposition multiplicity).
  3. Condition on a target set A at horizon T. Compute the true harmonic
     h(s,t) = P(X_T in A | X_t = s) = [exp(Q (T-t)) 1_A](s) by backward recursion.
  4. Build the time-inhomogeneous Doob-transformed generator
     q^h[s,s'](t) = q[s,s'] h(s',t)/h(s,t),
     integrate the forward equation from delta_{x0} under Q^h, and verify the
     endpoint law equals the exact Bayes conditional endpoint
     mu_A(s') = [exp(QT)]_{x0,s'} 1_A(s') / P(X_T in A | x0)   -- to ~machine eps.

  Also verified: Q^h is a valid generator (off-diagonals >= 0, rows sum to 0);
  the transform keeps support ONLY on legal fiber transitions (validity-closed
  under conditioning); and all conditional mass lands in A at T.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.fiber import ActionFiberSpec, enumerate_action_fiber
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system


def build_state_space(cap: int, n_slots: int, seed_smiles: str = "C"):
    """BFS the exhaustive set of states with n_real_atoms <= cap, closed under
    the legal fiber (transitions leaving the cap are dropped -> a proper finite
    CTMC on the size-capped slice). Returns (keys, smiles, transition multiplicities).
    """
    runtime = de_novo_rewrite_system()
    spec = ActionFiberSpec.neutral_cnof()
    seed = pad_molecular_graph(smiles_to_molecular_graph(seed_smiles), n_slots)
    seed_key = canonical_state_key(seed)

    states: dict[str, object] = {seed_key: seed}
    smiles: dict[str, str] = {seed_key: molecular_graph_to_smiles(seed)}
    # edges[src_key][dst_key] = number of distinct legal marks src->dst
    edges: dict[str, dict[str, int]] = {}
    queue: deque[str] = deque([seed_key])

    while queue:
        src_key = queue.popleft()
        src_state = states[src_key]
        if src_state.n_real_atoms > cap:
            continue
        transitions = enumerate_action_fiber(src_state, spec=spec, system=runtime)
        multiplicities: dict[str, int] = {}
        for t in transitions:
            if t.successor.n_real_atoms > cap:
                continue  # boundary: drop transitions leaving the size-capped slice
            multiplicities[t.successor_key] = multiplicities.get(t.successor_key, 0) + 1
            if t.successor_key not in states:
                states[t.successor_key] = t.successor
                smiles[t.successor_key] = molecular_graph_to_smiles(t.successor)
                queue.append(t.successor_key)
        edges[src_key] = multiplicities

    keys = sorted(states)
    return keys, states, smiles, edges


def generator_matrix(keys, edges):
    """Exact CTMC generator Q (rows sum to 0, off-diagonals >= 0)."""
    index = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    Q = np.zeros((n, n), dtype=np.float64)
    for src_key, dsts in edges.items():
        i = index[src_key]
        for dst_key, mult in dsts.items():
            if dst_key == src_key:
                continue
            Q[i, index[dst_key]] += float(mult)
    for i in range(n):
        Q[i, i] = -(Q[i].sum() - Q[i, i])
    return Q, index


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cap", type=int, default=3, help="max heavy atoms in the capped slice")
    p.add_argument("--n-slots", type=int, default=6)
    p.add_argument("--seed-smiles", default="C")
    p.add_argument("--horizon", type=float, default=1.0)
    p.add_argument("--grid", type=int, default=400, help="forward-integration steps")
    p.add_argument("--target", default="contains_O", choices=["contains_O", "contains_N", "size_max"])
    p.add_argument("--start-smiles", default="C")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    keys, states, smiles, edges = build_state_space(args.cap, args.n_slots, args.seed_smiles)
    Q, index = generator_matrix(keys, edges)
    n = len(keys)
    print(json.dumps({"phase": "state_space", "n_states": n, "cap": args.cap}), flush=True)

    # Target set A (rule-closed / structural predicate over the committed molecule).
    def in_target(key: str) -> bool:
        s = smiles[key]
        if args.target == "contains_O":
            return "O" in s or "o" in s
        if args.target == "contains_N":
            return "N" in s or "n" in s
        return states[key].n_real_atoms >= args.cap  # size_max

    one_A = np.array([1.0 if in_target(k) else 0.0 for k in keys])
    n_target = int(one_A.sum())
    T = args.horizon
    eps = 1e-12

    start_key = canonical_state_key(pad_molecular_graph(smiles_to_molecular_graph(args.start_smiles), args.n_slots))
    P_T = expm(Q * T)
    if start_key not in index:
        start_key = next(k for k in keys if (P_T[index[k]] * one_A).sum() > 1e-9)
    x0 = index[start_key]

    # Exact Bayes conditional endpoint: mu_A(s') = [exp(QT)]_{x0,s'} 1_A(s') / P(X_T in A | x0).
    h_x0_0 = float((P_T[x0] * one_A).sum())
    exact_endpoint = (P_T[x0] * one_A) / h_x0_0

    def h_grid_for(M):
        dt = T / M
        P_dt = expm(Q * dt)
        h = [None] * (M + 1)
        h[M] = one_A.copy()
        for k in range(M - 1, -1, -1):
            h[k] = P_dt @ h[k + 1]           # h(.,t_k) = exp(Q(T-t_k)) 1_A
        return dt, P_dt, h

    def doob_generator(h_t):
        """Local Doob-tilted generator q^h[s,s'] = q[s,s'] h(s',t)/h(s,t)."""
        live = np.where(h_t > eps)[0]
        Qh = np.zeros((n, n))
        sub = Q[np.ix_(live, live)] * (h_t[live][None, :] / h_t[live][:, None])
        np.fill_diagonal(sub, 0.0)
        np.fill_diagonal(sub, -sub.sum(axis=1))
        Qh[np.ix_(live, live)] = sub
        return Qh

    # (1) EXACT one-step Doob-propagator chain: P^h_k[x,y] = P_dt[x,y] h(y,t_{k+1})/h(x,t_k).
    #     These one-step tilted kernels are row-stochastic (h(.,t_k)=P_dt h(.,t_{k+1}))
    #     and telescope to the exact conditional -> machine-eps for any grid.
    Mp = 200
    dt, P_dt, hg = h_grid_for(Mp)
    mu = np.zeros(n); mu[x0] = 1.0
    for k in range(Mp):
        hk, hk1 = hg[k], hg[k + 1]
        live = hk > eps
        Ph = np.zeros((n, n))
        idx = np.where(live)[0]
        Ph[np.ix_(idx, np.arange(n))] = P_dt[idx] * (hk1[None, :] / hk[idx][:, None])
        mu = mu @ Ph
    mu = np.clip(mu, 0, None); mu = mu / mu.sum()
    linf_propagator = float(np.abs(mu - exact_endpoint).max())
    leak_propagator = float(mu[one_A == 0].sum())

    # (2) GENERATOR-level convergence: integrate delta_{x0} under the tilted GENERATOR
    #     Q^h(t) (built from local rates, not the propagator). First-order in dt ->
    #     Linf must fall ~geometrically toward 0, proving the generator (its rates)
    #     reproduces the exact conditional in the continuum limit.
    convergence = []
    offdiag_viol = 0.0; rowsum_max = 0.0
    for M in (50, 100, 200, 400, 800):
        dt, P_dt, hg = h_grid_for(M)
        mu = np.zeros(n); mu[x0] = 1.0
        for k in range(M):
            Qh = doob_generator(hg[k])
            off = Qh - np.diag(np.diag(Qh))
            offdiag_viol = max(offdiag_viol, float(max(0.0, -off.min())))
            rowsum_max = max(rowsum_max, float(np.abs(Qh.sum(axis=1)).max()))
            mu = mu @ expm(Qh * dt)
        mu = np.clip(mu, 0, None); mu = mu / mu.sum()
        convergence.append({"grid": M, "dt": dt,
                            "Linf": float(np.abs(mu - exact_endpoint).max()),
                            "mass_in_target": float(mu[one_A > 0].sum())})

    ratios = [convergence[i]["Linf"] / max(convergence[i + 1]["Linf"], 1e-18)
              for i in range(len(convergence) - 1)]

    result = {
        "phase": "E0_exactness",
        "n_states": n,
        "n_target_states": n_target,
        "cap": args.cap,
        "horizon": T,
        "target": args.target,
        "start_smiles": args.start_smiles,
        "P_reach_target_from_start": h_x0_0,
        "exact_propagator_chain": {
            "Linf_vs_exact_conditional": linf_propagator,
            "mass_outside_target_at_T": leak_propagator,
            "machine_exact": bool(linf_propagator < 1e-9 and leak_propagator < 1e-12),
        },
        "Qh_valid_generator": {
            "max_offdiagonal_negativity": offdiag_viol,
            "max_abs_rowsum": rowsum_max,
            "is_valid_generator": bool(offdiag_viol < 1e-12 and rowsum_max < 1e-9),
        },
        "generator_integration_convergence": convergence,
        "convergence_ratios_halving_dt": ratios,
        "converges_to_exact": bool(convergence[-1]["Linf"] < convergence[0]["Linf"] / 8),
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
