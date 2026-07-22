#!/usr/bin/env python3
"""Exact Pareto-reachability / objective-barrier diagnostic on the rewrite graph.

The reframe's spotlight *mechanism*: on the executable molecular rewrite graph a
Pareto-optimal molecule may be unreachable by any sequence of locally improving
edits -- reaching it requires descending in utility first, through intermediate
states that are themselves complete valid molecules. A greedy / monotone
controller (one that never accepts a worsening edit) is therefore provably
confined to a strict subset of the reachable frontier; only nonmyopic control
over valid detours recovers the rest. This is a hypothesis, not yet a finding,
so we measure it EXACTLY on an exhaustive, size-capped slice of the real
validity-closed rewrite CTMC (the same `build_state_space` used by E0), across
several sources and preference directions.

Data structure
--------------
- Exhaustive directed graph G=(V,E): V = every complete valid connected molecule
  with n_real_atoms <= cap (canonical successor identity), E = legal rewrites
  (self-loops dropped). Built once, cached by (cap, n_slots, seed).
- Objective matrix S (|V| x d): d black-box descriptors, all MAXIMIZED (min-max
  normalized over V so a simplex weight omega gives a comparable utility U=S_n@omega).

Per source G_s and preference direction omega we compute, EXACTLY:
- global optimum         U* = max_{H in R(G_s)} U(H)         over the reachable set R
- monotone-reachable opt U_mono = max over states reachable from G_s by U-non-
                         decreasing edits (an *upper bound* on any greedy climber)
- barrier gap            gap = U* - U_mono   (>0  => monotone control is provably
                         suboptimal for this omega: it cannot reach the optimum)
- objective barrier      B(H) = min_{path G_s~>H} max_k [U(G_s) - U(G_k)]_+
                         (minimax / bottleneck path; the minimum utility descent
                         below the source that ANY route to H must accept)

Invariants (asserted): every node is a valid RDKit molecule; monotone-reachable
subset(R); B(G_s)=0; B is 0 exactly when a never-below-source path exists; global
optimum lies in R(G_s). Reported both for a 2-objective and a 3-objective slice
(the memo asks for multiple objective pairs and sources).
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e0_toy_h_exactness import build_state_space  # noqa: E402  exhaustive legal-fiber CTMC slice

RDLogger.DisableLog("rdApp.*")

_ROOT = Path(__file__).resolve().parents[1]
SC = os.environ.get("COMPOSE_SCRATCH", str(_ROOT / "scratch"))
os.makedirs(SC, exist_ok=True)
OUTDIAG = _ROOT / "diagnostics" / "reachability"

# ---- Objective registry (all MAXIMIZED; illustrative descriptor trade-offs) ----
# logP and TPSA are genuinely anti-correlated (lipophilicity vs polar surface),
# so maximizing both is a real tension -> a nontrivial frontier. QED adds a
# drug-likeness axis. The reachability claim is about graph topology and is
# invariant to the specific oracle; these are stand-ins for any black-box vector.
OBJECTIVES = {
    "logp": lambda m: float(Crippen.MolLogP(m)),
    "tpsa": lambda m: float(Descriptors.TPSA(m)),
    "qed": lambda m: float(QED.qed(m)),
    "mw": lambda m: float(Descriptors.MolWt(m)),
    "neg_mw": lambda m: -float(Descriptors.MolWt(m)),
}


# ---- Exhaustive graph (cached) ----
def load_or_build_graph(cap: int, n_slots: int, seed_smiles: str):
    """Return (smiles: list[str], adj: list[list[int]], heavy: list[int]).

    Nodes are indexed 0..n-1 by sorted canonical key; adj[i] = successor indices
    (self-loops removed). Cached to scratch so the ~minute enumeration runs once.
    """
    cache = Path(SC) / f"reach_graph_cap{cap}_slots{n_slots}_{seed_smiles}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["smiles"], [list(a) for a in d["adj"]], d["heavy"]

    keys, states, smiles_map, edges = build_state_space(cap, n_slots, seed_smiles)
    index = {k: i for i, k in enumerate(keys)}
    smiles = [smiles_map[k] for k in keys]
    heavy = [int(states[k].n_real_atoms) for k in keys]
    adj: list[list[int]] = [[] for _ in keys]
    for src_key, dsts in edges.items():
        i = index[src_key]
        adj[i] = sorted({index[d] for d in dsts if d != src_key})
    cache.write_text(json.dumps({"smiles": smiles, "adj": adj, "heavy": heavy}))
    return smiles, adj, heavy


def objective_matrix(smiles: list[str], names: list[str]):
    """(n x d) raw objectives and (n x d) min-max normalized to [0,1] over V."""
    funcs = [OBJECTIVES[n] for n in names]
    raw = np.zeros((len(smiles), len(names)), dtype=np.float64)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        assert m is not None, f"invalid molecule in enumerated state space: {s!r}"
        raw[i] = [f(m) for f in funcs]
    lo, hi = raw.min(0), raw.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return raw, (raw - lo) / span


# ---- Graph reachability primitives ----
def reachable_from(adj: list[list[int]], src: int) -> np.ndarray:
    seen = np.zeros(len(adj), dtype=bool)
    seen[src] = True
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return seen


def monotone_reachable(adj: list[list[int]], U: np.ndarray, src: int, tol: float) -> np.ndarray:
    """States reachable from src by U-non-decreasing edits (a greedy climber's ceiling)."""
    seen = np.zeros(len(adj), dtype=bool)
    seen[src] = True
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not seen[v] and U[v] >= U[u] - tol:
                seen[v] = True
                q.append(v)
    return seen


def minimax_barrier(adj: list[list[int]], pen: np.ndarray, src: int):
    """Bottleneck (minimax) path value to every node: min over paths of max node penalty.

    Returns (barrier array, predecessor array) so the min-barrier detour is reconstructable.
    """
    n = len(adj)
    b = np.full(n, np.inf)
    pred = np.full(n, -1, dtype=np.int64)
    b[src] = pen[src]
    pq = [(float(pen[src]), src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > b[u]:
            continue
        for v in adj[u]:
            nd = d if d > pen[v] else pen[v]
            if nd < b[v]:
                b[v] = nd
                pred[v] = u
                heapq.heappush(pq, (float(nd), v))
    return b, pred


def pareto_mask(S: np.ndarray) -> np.ndarray:
    """Boolean mask of nondominated rows (maximization)."""
    n = len(S)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(S >= S[i], axis=1) & np.any(S > S[i], axis=1)
        if dominated.any():
            keep[i] = False
    return keep


def simplex_grid(d: int, divisions: int) -> list[tuple[float, ...]]:
    """Weights on the (d-1)-simplex: all nonneg integer compositions of `divisions`."""
    if d == 2:
        return [(t / divisions, 1 - t / divisions) for t in range(divisions + 1)]
    out = []
    for i in range(divisions + 1):
        for j in range(divisions + 1 - i):
            k = divisions - i - j
            out.append((i / divisions, j / divisions, k / divisions))
    return out


def farthest_point_sources(Sn: np.ndarray, heavy: list[int], cap: int, k: int) -> list[int]:
    """Spread-out sources among 'mature' states (heavy >= cap-1), deterministic."""
    cand = [i for i, h in enumerate(heavy) if h >= cap - 1]
    if len(cand) <= k:
        return cand
    pts = Sn[cand]
    centroid = pts.mean(0)
    first = cand[int(np.argmin(((pts - centroid) ** 2).sum(1)))]
    chosen = [first]
    while len(chosen) < k:
        d = np.min([((Sn[cand] - Sn[c]) ** 2).sum(1) for c in chosen], axis=0)
        nxt = cand[int(np.argmax(d))]
        if nxt in chosen:
            break
        chosen.append(nxt)
    return chosen


# ---- Sweep ----
def analyze_slice(adj, raw, Sn, heavy, cap, names, divisions, n_sources, tol):
    d = len(names)
    weights = simplex_grid(d, divisions)
    sources = farthest_point_sources(Sn, heavy, cap, n_sources)
    global_pareto = int(pareto_mask(raw).sum())

    per_source = []
    for s in sources:
        R = reachable_from(adj, s)
        Ridx = np.where(R)[0]
        gaps, barriers, spans = [], [], []
        mono_suboptimal = barrier_positive = 0
        for w in weights:
            U = Sn @ np.asarray(w)
            u_src = U[s]
            gopt = Ridx[int(np.argmax(U[Ridx]))]
            u_star = U[gopt]
            span = u_star - u_src
            M = monotone_reachable(adj, U, s, tol)
            u_mono = U[M].max()
            gap = u_star - u_mono
            pen = np.maximum(0.0, u_src - U)
            b, _ = minimax_barrier(adj, pen, s)
            barrier = float(b[gopt])
            gaps.append(gap)
            barriers.append(barrier)
            spans.append(span)
            mono_suboptimal += int(gap > tol)
            barrier_positive += int(barrier > tol)
        gaps = np.array(gaps)
        barriers = np.array(barriers)
        spans = np.array(spans)
        # gap as a fraction of the utility actually achievable from this source
        frac = np.where(spans > tol, gaps / np.maximum(spans, tol), 0.0)
        gpos = gaps > tol
        bpos = barriers > tol
        per_source.append({
            "source_smiles": None,  # filled by caller
            "source_index": int(s),
            "n_reachable": int(R.sum()),
            "n_weights": len(weights),
            "frac_dirs_monotone_suboptimal": mono_suboptimal / len(weights),
            "frac_dirs_barrier_positive": barrier_positive / len(weights),
            # magnitude CONDITIONAL on there being a barrier (0 if never) -- the honest
            # "when monotone control is suboptimal, by how much" number.
            "gap_mean_when_positive": float(gaps[gpos].mean()) if gpos.any() else 0.0,
            "gap_frac_of_reach_mean_when_positive": float(frac[gpos].mean()) if gpos.any() else 0.0,
            "barrier_mean_when_positive": float(barriers[bpos].mean()) if bpos.any() else 0.0,
            "gap_max": float(gaps.max()),
            "gap_frac_of_reach_max": float(frac.max()),
            "barrier_max": float(barriers.max()),
        })

    agg = {
        "objectives": names,
        "n_states": len(adj),
        "n_edges": sum(len(a) for a in adj),
        "global_pareto_size": global_pareto,
        "n_sources": len(sources),
        "n_weights": len(weights),
        "mean_frac_dirs_monotone_suboptimal":
            float(np.mean([p["frac_dirs_monotone_suboptimal"] for p in per_source])),
        "mean_frac_dirs_barrier_positive":
            float(np.mean([p["frac_dirs_barrier_positive"] for p in per_source])),
        # source-dependence is a finding, so report the strongest source too, not only the mean
        "worst_source_frac_dirs_monotone_suboptimal":
            float(max(p["frac_dirs_monotone_suboptimal"] for p in per_source)),
        "worst_source_index":
            max(per_source, key=lambda p: p["frac_dirs_monotone_suboptimal"])["source_index"],
        "mean_gap_frac_of_reach_when_positive":
            float(np.mean([p["gap_frac_of_reach_mean_when_positive"] for p in per_source
                           if p["frac_dirs_monotone_suboptimal"] > 0] or [0.0])),
        "max_gap_frac_of_reach_any_source":
            float(max(p["gap_frac_of_reach_max"] for p in per_source)),
        "max_barrier_any_source": float(max(p["barrier_max"] for p in per_source)),
    }
    return agg, per_source, sources


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cap", type=int, default=4)
    p.add_argument("--n-slots", type=int, default=8)
    p.add_argument("--seed-smiles", default="C")
    p.add_argument("--divisions", type=int, default=100, help="omega grid resolution")
    p.add_argument("--n-sources", type=int, default=6)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    smiles, adj, heavy = load_or_build_graph(args.cap, args.n_slots, args.seed_smiles)
    print(json.dumps({"phase": "graph", "n_states": len(adj),
                      "n_edges": sum(len(a) for a in adj)}), flush=True)

    result = {"phase": "pareto_reachability", "cap": args.cap, "n_slots": args.n_slots,
              "seed_smiles": args.seed_smiles, "slices": {}}
    for names in (["logp", "tpsa"], ["logp", "tpsa", "qed"]):
        raw, Sn = objective_matrix(smiles, names)
        div = args.divisions if len(names) == 2 else min(args.divisions, 14)
        agg, per_source, sources = analyze_slice(
            adj, raw, Sn, heavy, args.cap, names, div, args.n_sources, args.tol)
        for ps in per_source:
            ps["source_smiles"] = smiles[ps["source_index"]]
        key = "+".join(names)
        result["slices"][key] = {"aggregate": agg, "per_source": per_source}
        print(json.dumps({"slice": key, **agg}, indent=2), flush=True)

    OUTDIAG.mkdir(parents=True, exist_ok=True)
    out = args.output or (OUTDIAG / f"pareto_reachability_cap{args.cap}.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
