#!/usr/bin/env python3
"""EXACT-DOOB benchmark on a fully-enumerable molecular-rewrite CTMC (Phase D).

The paper's mathematical centerpiece is exact Doob control: on an operator-closed
space of connected, valence-valid molecular graphs, the Doob h-transform of the
reference generator reweights *existing* legal rates and, when the harmonic
function h is exact, realizes the specified terminal reweighting EXACTLY. This
script demonstrates that claim end-to-end on a genuinely enumerable slice of the
real rewrite CTMC (built from the production legal-event fiber, C/N/O micro
chemistry, capped heavy-atom count) so that every quantity is computed exactly:

  1. ENUMERATE the exhaustive connected valid-molecule state space reachable
     under the micro operators, bounded to <= `cap` heavy atoms and neutral
     C/N/O with single/double/triple bonds, by BFS over the legal fiber A(G)
     from a seed, deduplicating by `canonical_state_key`. Reports state + edge
     counts. Every state is a complete, chemically valid, connected molecule.
  2. REFERENCE GENERATOR Q: an explicit reference rate that is UNIFORM over legal
     marks -- each distinct legal mark fires at rate 1, so the off-diagonal
     q[x,y] equals the number of distinct legal marks x->y (superposition
     multiplicity) and the diagonal is the negative total outgoing rate.
  3. EXACT DOOB VALUE h_t(x) = E[g(X_T) | X_t = x] by the backward equation
     h(.,t) = exp(Q (T-t)) g, for BOTH a hard indicator desirability
     g(x) = 1{x contains a chosen atom / substructure} and a soft Boltzmann tilt
     g(x) = exp{beta * u(x)} with u = heavy-atom count.
  4. DOOB-TILTED GENERATOR Q^g_t(x,y) = Q(x,y) h_t(y)/h_t(x) (off-diagonals; the
     diagonal is re-summed to zero). We verify it is a valid generator and that
     the tilted endpoint law equals the analytic target p_T^g(x) prop
     p_T(x) g(x) to ~machine precision, via the exact one-step Doob transition
     kernel P^h_k(x,y) = P_dt(x,y) h_{t_{k+1}}(y)/h_{t_k}(x) (the h-transform
     theorem makes this the EXACT tilted kernel, so its telescoping product is
     p_T^g for any grid -- not a dt-approximation).
  5. CONTROLLER COMPARISON by total-variation distance to the exact tilted
     endpoint: exact-Doob vs local-greedy (concentrate rate on the successor
     maximizing the immediate reward change Delta-u) vs local-Boltzmann (reweight
     each legal rate by exp{beta * Delta-u}). All endpoints are computed exactly
     (matrix exponential of each control generator) -- no sampling noise.

Headline: exact-Doob reproduces the analytic tilted distribution to numerical
error, while local greedy / Boltzmann guidance attains the reward but the WRONG
distribution (large TV).

Run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python \
     scripts/exact_doob_enumerable_benchmark.py --cap 4
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.fiber import ActionFiberSpec, AtomState, enumerate_action_fiber
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system


# ---- State-space enumeration (BFS over the legal fiber) ----------------------


def cno_fiber_spec() -> ActionFiberSpec:
    """Neutral C/N/O micro-rewrite language: charge 0, implicit H in 0..4,
    single/double/triple bonds (the fiber's MICRO_BOND_CLASSES). This bounds the
    space to a tiny vocabulary so the reachable slice is fully enumerable."""

    atom_types = tuple(ELEMENT_TO_IDX[symbol] for symbol in ("C", "N", "O"))
    return ActionFiberSpec(
        atom_states=tuple(
            AtomState(atom_type, 0, hydrogen_count)
            for atom_type in atom_types
            for hydrogen_count in range(5)
        )
    )


def build_state_space(cap: int, n_slots: int, spec: ActionFiberSpec, seed_smiles: str = "C"):
    """BFS the exhaustive set of connected valid molecular graphs with
    n_real_atoms <= cap, closed under the legal fiber. Transitions leaving the
    cap are dropped, giving a proper finite CTMC on the size-capped slice.
    States are deduplicated by canonical (SMILES) key.

    Returns (keys, states, smiles, edges) where edges[src][dst] is the number of
    distinct legal marks src->dst (the superposition multiplicity)."""

    runtime = de_novo_rewrite_system()
    seed = pad_molecular_graph(smiles_to_molecular_graph(seed_smiles), n_slots)
    seed_key = canonical_state_key(seed)

    states: dict[str, object] = {seed_key: seed}
    smiles: dict[str, str] = {seed_key: molecular_graph_to_smiles(seed)}
    edges: dict[str, dict[str, int]] = {}
    queue: deque[str] = deque([seed_key])

    while queue:
        src_key = queue.popleft()
        src_state = states[src_key]
        if src_state.n_real_atoms > cap:
            continue
        transitions = enumerate_action_fiber(src_state, spec=spec, system=runtime)
        multiplicities: dict[str, int] = {}
        for transition in transitions:
            if transition.successor.n_real_atoms > cap:
                continue  # boundary: drop transitions that leave the size cap
            key = transition.successor_key
            multiplicities[key] = multiplicities.get(key, 0) + 1
            if key not in states:
                states[key] = transition.successor
                smiles[key] = molecular_graph_to_smiles(transition.successor)
                queue.append(key)
        edges[src_key] = multiplicities

    keys = sorted(states)
    return keys, states, smiles, edges


def generator_matrix(keys, edges):
    """Reference CTMC generator Q with UNIFORM per-mark rates: off-diagonal
    q[x,y] = number of distinct legal marks x->y; diagonal = -total outgoing.
    Rows sum to 0 and off-diagonals are >= 0 (a valid generator)."""

    index = {key: i for i, key in enumerate(keys)}
    n = len(keys)
    Q = np.zeros((n, n), dtype=np.float64)
    for src_key, dsts in edges.items():
        i = index[src_key]
        for dst_key, mult in dsts.items():
            if dst_key == src_key:
                continue
            Q[i, index[dst_key]] += float(mult)
    for i in range(n):
        Q[i, i] = -Q[i].sum()
    return Q, index


# ---- Terminal desirability g and its local reward surrogate u ----------------


def atom_count(state, element: str) -> int:
    """Number of real atoms of a given element in the molecular graph."""

    return int(np.sum(state.atom_types == ELEMENT_TO_IDX[element]))


def build_targets(keys, states, cap: int):
    """Return the benchmark's terminal desirabilities. Each entry supplies a
    positive desirability g(x) and a scalar reward u(x) that the myopic
    controllers (greedy / Boltzmann) hill-climb -- u is the local surrogate
    ALIGNED with g, so greedy/Boltzmann genuinely attain the reward and the only
    question is whether they attain the right DISTRIBUTION.

      soft_shrink : g = exp(beta * (cap - heavy count)); u = cap - heavy count.
                    A soft Boltzmann tilt toward SMALLER molecules -- i.e. AGAINST
                    the reference process's natural drift (uniform-over-marks
                    favours high-degree, larger molecules). This is the honest
                    regime where local guidance must fight the base dynamics.
      soft_grow   : g = exp(beta * heavy count);         u = heavy count.
                    A soft tilt toward LARGER molecules, i.e. WITH the reference
                    drift -- an honest contrast where the myopic Boltzmann tilt
                    is already a good approximation.
      hard_oxygen : g = 1{contains >= 1 oxygen};         u = oxygen count.
                    Hard conditioning on a substructure event.
      hard_N_and_O: g = 1{contains >=1 N AND >=1 O};     u = N-count + O-count.
                    A rarer two-atom conjunction event.
    """

    heavy = np.array([states[k].n_real_atoms for k in keys], dtype=np.float64)
    n_o = np.array([atom_count(states[k], "O") for k in keys], dtype=np.float64)
    n_n = np.array([atom_count(states[k], "N") for k in keys], dtype=np.float64)

    return {
        "soft_shrink": {
            "kind": "soft",
            "reward": float(cap) - heavy,
            "g_of_reward": lambda beta: np.exp(beta * (float(cap) - heavy)),
            "reward_name": "cap_minus_heavy_atom_count",
        },
        "soft_grow": {
            "kind": "soft",
            "reward": heavy,
            "g_of_reward": lambda beta: np.exp(beta * heavy),
            "reward_name": "heavy_atom_count",
        },
        "hard_oxygen": {
            "kind": "hard",
            "reward": n_o,
            "indicator": (n_o >= 1.0).astype(np.float64),
            "reward_name": "oxygen_count",
        },
        "hard_N_and_O": {
            "kind": "hard",
            "reward": n_n + n_o,
            "indicator": ((n_n >= 1.0) & (n_o >= 1.0)).astype(np.float64),
            "reward_name": "N_plus_O_count",
        },
    }


# ---- Control generators (all reweightings of the SAME reference rates) --------


def greedy_generator(Q, reward):
    """Local-greedy controller: from each state keep the reference rate ONLY on
    the successor(s) maximizing the immediate reward change Delta-u = u(y)-u(x).
    This is the zero-temperature limit of the local-Boltzmann tilt."""

    n = Q.shape[0]
    QG = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        succ = np.where(Q[i] > 0.0)[0]  # off-diagonal positive rates = successors
        if succ.size == 0:
            continue
        delta = reward[succ] - reward[i]
        winners = succ[delta >= delta.max() - 1e-9]
        QG[i, winners] = Q[i, winners]
    for i in range(n):
        QG[i, i] = -QG[i].sum()
    return QG


def boltzmann_generator(Q, reward, beta):
    """Local-Boltzmann controller: reweight each legal reference rate by the
    immediate-reward tilt exp{beta * (u(y)-u(x))}. The myopic analogue of the
    Doob tilt h(y)/h(x) -- it uses the instantaneous reward change instead of the
    future-aware harmonic function."""

    delta = reward[None, :] - reward[:, None]
    QB = Q * np.exp(beta * delta)
    np.fill_diagonal(QB, 0.0)
    for i in range(QB.shape[0]):
        QB[i, i] = -QB[i].sum()
    return QB


def doob_generator(Q, h_t, eps=1e-12):
    """Doob-tilted generator Q^g_t(x,y) = Q(x,y) h_t(y)/h_t(x) on the live set
    {h_t > 0}; diagonal re-summed to zero. Returned for the valid-generator
    check; the exact endpoint law is obtained from the one-step Doob kernel."""

    n = Q.shape[0]
    live = np.where(h_t > eps)[0]
    QH = np.zeros((n, n), dtype=np.float64)
    sub = Q[np.ix_(live, live)] * (h_t[live][None, :] / h_t[live][:, None])
    np.fill_diagonal(sub, 0.0)
    np.fill_diagonal(sub, -sub.sum(axis=1))
    QH[np.ix_(live, live)] = sub
    return QH


# ---- Distances / summaries ---------------------------------------------------


def total_variation(p, q) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def endpoint_from_generator(Qc, x0, T):
    """Exact endpoint law of a time-homogeneous control generator: row x0 of
    exp(Qc T) (a row-stochastic transition matrix)."""

    return expm(Qc * T)[x0]


# ---- Main --------------------------------------------------------------------


def run_target(name, spec, Q, P_T, P_dt, h_from_g, x0, T, M, beta, reward, kind):
    """Compute the exact tilted target, verify exactness, and compare controllers
    for a single terminal desirability. Returns a JSON-able dict."""

    n = Q.shape[0]
    eps = 1e-12

    # Terminal desirability g and the analytic tilted endpoint p_T^g prop p_T g.
    g = h_from_g
    Z = float((P_T[x0] * g).sum())              # = h(x0, 0) = E[g(X_T) | X_0=x0]
    p_tilt = (P_T[x0] * g) / Z

    # Backward equation: h(., t_k) = exp(Q (T - t_k)) g, built with the SAME P_dt.
    h_grid = [None] * (M + 1)
    h_grid[M] = g.copy()
    for k in range(M - 1, -1, -1):
        h_grid[k] = P_dt @ h_grid[k + 1]
    h0_x0 = float(h_grid[0][x0])                # must match Z (backward == forward)

    # Valid-generator check on the formed Q^g_t at two interior grid times.
    offdiag_negativity = 0.0
    rowsum_abs = 0.0
    for k in (0, M // 2):
        QH = doob_generator(Q, h_grid[k], eps)
        off = QH - np.diag(np.diag(QH))
        offdiag_negativity = max(offdiag_negativity, float(max(0.0, -off.min())))
        rowsum_abs = max(rowsum_abs, float(np.abs(QH.sum(axis=1)).max()))

    # EXACT tilted endpoint via the one-step Doob transition kernel
    # P^h_k(x,y) = P_dt(x,y) h_{t_{k+1}}(y)/h_{t_k}(x). Telescopes to p_T^g.
    mu = np.zeros(n)
    mu[x0] = 1.0
    for k in range(M):
        hk, hk1 = h_grid[k], h_grid[k + 1]
        live = hk > eps
        rows = np.where(live)[0]
        Ph = np.zeros((n, n))
        Ph[rows] = P_dt[rows] * (hk1[None, :] / hk[rows][:, None])
        mu = mu @ Ph
    mu = np.clip(mu, 0.0, None)
    mu = mu / mu.sum()
    doob_linf = float(np.abs(mu - p_tilt).max())
    doob_tv = total_variation(mu, p_tilt)

    # Controllers (all reweightings of the SAME reference rates).
    baseline = P_T[x0]                                   # unguided reference law
    greedy = endpoint_from_generator(greedy_generator(Q, reward), x0, T)
    boltz = endpoint_from_generator(boltzmann_generator(Q, reward, beta), x0, T)

    def mean_reward(dist):
        return float((dist * reward).sum())

    # "attained reward": E[u(X_T)]; for hard targets also the target-event prob.
    summary = {
        "controllers": {
            "reference_unguided": {
                "tv_to_exact_tilt": total_variation(baseline, p_tilt),
                "mean_reward": mean_reward(baseline),
            },
            "exact_doob": {
                "tv_to_exact_tilt": doob_tv,
                "mean_reward": mean_reward(mu),
            },
            "local_greedy": {
                "tv_to_exact_tilt": total_variation(greedy, p_tilt),
                "mean_reward": mean_reward(greedy),
            },
            "local_boltzmann": {
                "tv_to_exact_tilt": total_variation(boltz, p_tilt),
                "mean_reward": mean_reward(boltz),
            },
        },
    }
    if kind == "hard":
        indicator = (g > 0.5).astype(np.float64)
        summary["target_event_probability"] = {
            "reference_unguided": float((baseline * indicator).sum()),
            "exact_tilt_analytic": float((p_tilt * indicator).sum()),
            "exact_doob": float((mu * indicator).sum()),
            "local_greedy": float((greedy * indicator).sum()),
            "local_boltzmann": float((boltz * indicator).sum()),
        }

    result = {
        "target": name,
        "kind": kind,
        "beta": beta,
        "reward": spec["reward_name"],
        "P_reach_or_partition": Z,
        "backward_forward_h_match": abs(h0_x0 - Z),
        "n_support_p_tilt": int((p_tilt > 1e-9).sum()),
        "doob_generator_valid": {
            "max_offdiagonal_negativity": offdiag_negativity,
            "max_abs_rowsum": rowsum_abs,
            "is_valid_generator": bool(offdiag_negativity < 1e-12 and rowsum_abs < 1e-9),
        },
        "exactness_doob_vs_analytic_tilt": {
            "linf": doob_linf,
            "tv": doob_tv,
            "machine_exact": bool(doob_linf < 1e-9),
        },
        **summary,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, default=4, help="max heavy atoms in the enumerable slice")
    parser.add_argument("--n-slots", type=int, default=None, help="padded slots (default cap+1)")
    parser.add_argument("--seed-smiles", default="C")
    parser.add_argument("--start-smiles", default="C", help="CTMC start state X_0")
    parser.add_argument("--horizon", type=float, default=1.0)
    parser.add_argument("--grid", type=int, default=60, help="backward/propagator grid steps")
    parser.add_argument("--beta", type=float, default=1.5, help="soft/Boltzmann inverse temperature")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    n_slots = args.n_slots if args.n_slots is not None else args.cap + 1
    spec = cno_fiber_spec()

    keys, states, smiles, edges = build_state_space(args.cap, n_slots, spec, args.seed_smiles)
    Q, index = generator_matrix(keys, edges)
    n = len(keys)
    n_edges = int((Q > 0).sum())
    print(json.dumps({"phase": "state_space", "cap": args.cap, "n_slots": n_slots,
                      "n_states": n, "n_directed_edges": n_edges,
                      "vocab": "CNO_neutral", "bonds": "single/double/triple"}), flush=True)

    T = args.horizon
    M = args.grid
    dt = T / M
    P_T = expm(Q * T)
    P_dt = expm(Q * dt)

    start = pad_molecular_graph(smiles_to_molecular_graph(args.start_smiles), n_slots)
    x0 = index[canonical_state_key(start)]

    targets = build_targets(keys, states, args.cap)
    results = []
    for name, tspec in targets.items():
        if tspec["kind"] == "soft":
            g = tspec["g_of_reward"](args.beta)
        else:
            g = tspec["indicator"]
        res = run_target(name, tspec, Q, P_T, P_dt, g, x0, T, M, args.beta,
                         tspec["reward"], tspec["kind"])
        results.append(res)
        ctl = res["controllers"]
        print(json.dumps({
            "phase": "target", "target": name, "kind": tspec["kind"],
            "doob_tv": res["exactness_doob_vs_analytic_tilt"]["tv"],
            "doob_linf": res["exactness_doob_vs_analytic_tilt"]["linf"],
            "tv_reference": ctl["reference_unguided"]["tv_to_exact_tilt"],
            "tv_greedy": ctl["local_greedy"]["tv_to_exact_tilt"],
            "tv_boltzmann": ctl["local_boltzmann"]["tv_to_exact_tilt"],
        }), flush=True)

    payload = {
        "benchmark": "exact_doob_enumerable",
        "cap": args.cap,
        "n_slots": n_slots,
        "n_states": n,
        "n_directed_edges": n_edges,
        "horizon": T,
        "grid": M,
        "beta": args.beta,
        "start_smiles": args.start_smiles,
        "reference_rate": "uniform_over_legal_marks (q[x,y] = #marks x->y)",
        "results": results,
    }
    print(json.dumps({"phase": "summary_tv_table",
                      "rows": [{"target": r["target"],
                                "doob": r["controllers"]["exact_doob"]["tv_to_exact_tilt"],
                                "reference": r["controllers"]["reference_unguided"]["tv_to_exact_tilt"],
                                "greedy": r["controllers"]["local_greedy"]["tv_to_exact_tilt"],
                                "boltzmann": r["controllers"]["local_boltzmann"]["tv_to_exact_tilt"]}
                               for r in results]}, indent=2), flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
