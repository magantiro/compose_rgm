#!/usr/bin/env python3
"""Frontier recovery: does detour-accepting control recover what greedy cannot?

The reachability diagnostic (scripts/pareto_reachability.py) established the
*opportunity*: on the exact rewrite graph a monotone/greedy controller is
provably confined below the reachable Pareto frontier for a substantial fraction
of preference directions. This script tests the *payoff* on the SAME exhaustive
graph, where the true global Pareto frontier is computable exactly:

  - ceiling  : the exact global Pareto frontier over states reachable from a source.
  - greedy   : steepest-ascent hill-climb under U_omega, swept over the weight
               simplex (realistic local monotone control -- never accepts a
               worsening edit). Archive = union of climb endpoints.
  - detour   : an UNGUIDED annealed archiving explorer over the SAME legal moves
               that CAN accept downhill edits (Metropolis), archiving every
               visited valid molecule, swept over omega. This is deliberately a
               strawman: substrate + detours but NO learned rates and NO
               archive-aware resampling -- it isolates "detours WITHOUT guidance".

Two honest findings this exposes (see diagnostics/reachability/frontier_recovery):
  (1) STRUCTURAL: greedy has a hard ceiling strictly below the frontier
      (~35%/25% coverage; ~93% hypervolume) that NO budget can exceed, because the
      missing frontier is barrier-gated. Detour-capable control has no such ceiling
      (reaches 100% given budget). This is the exact statement of the opportunity.
  (2) EFFICIENCY: unguided detours are oracle-INEFFICIENT -- at matched budget the
      unguided explorer does NOT beat greedy; it only reaches 100% by near-
      exhaustive visitation. So the lever for efficient barrier-crossing is
      proposal quality (the learned rates Q_theta) + archive-aware resampling,
      NOT detours alone. That is precisely the memo's "local feasible proposal
      quality, not global fidelity" claim, and it is tested at scale on real leads
      with the learned model -- this toy only proves the ceiling is structural and
      that detours-without-guidance don't buy efficiency.

Every visited state is a complete valid molecule -- the comparison is undefined if
intermediates are not molecules, which is the point of the substrate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pareto_reachability import (  # noqa: E402
    farthest_point_sources,
    load_or_build_graph,
    objective_matrix,
    pareto_mask,
    reachable_from,
    simplex_grid,
)

_ROOT = Path(__file__).resolve().parents[1]
SC = os.environ.get("COMPOSE_SCRATCH", str(_ROOT / "scratch"))
os.makedirs(SC, exist_ok=True)
OUTDIAG = _ROOT / "diagnostics" / "reachability"

TOL = 1e-9


# ---- controllers on the exact graph ----
def hill_climb(adj, U, src):
    """Steepest-ascent monotone climb: never accept a worsening edit. Returns
    (endpoint, set of distinct states SCORED = every neighbour examined)."""
    cur = src
    scored = {src}
    while adj[cur]:
        scored.update(adj[cur])
        nxt = max(adj[cur], key=lambda v: U[v])
        if U[nxt] > U[cur] + TOL:
            cur = nxt
        else:
            break
    return cur, scored


def explore_until(adj, Us, src, rng, budget, t0, t1, max_steps):
    """Annealed Metropolis explorer over legal moves that CAN step downhill,
    rotating the preference direction and archiving every visited valid molecule.
    Runs until `budget` distinct molecules have been SCORED (oracle calls) or
    max_steps is hit. Returns (visited set = archive, distinct scored count)."""
    n_dir = Us.shape[1]
    cur = src
    scored = {src}
    visited = {src}
    step = 0
    while len(scored) < budget and step < max_steps:
        U = Us[:, step % n_dir]                      # rotate through weight directions
        nbrs = adj[cur]
        if not nbrs:                                 # dead end: warm-restart at source
            cur = src
            step += 1
            continue
        v = int(rng.choice(nbrs))
        scored.add(v)
        temp = t0 * (t1 / t0) ** (step / max_steps)  # geometric anneal high->low
        du = U[v] - U[cur]
        if du >= 0 or rng.random() < np.exp(du / temp):
            cur = v
        visited.add(cur)
        step += 1
    return visited, len(scored)


# ---- metrics ----
def coverage(archive_pts, frontier_pts):
    """Fraction of frontier points weakly dominated by some archived point."""
    if len(frontier_pts) == 0:
        return 1.0
    cov = 0
    for p in frontier_pts:
        if np.any(np.all(archive_pts >= p - 1e-9, axis=1)):
            cov += 1
    return cov / len(frontier_pts)


def hypervolume_2d(pts, ref):
    """2D hypervolume (maximization) above reference nadir `ref`."""
    P = pts[pareto_mask(pts)]
    P = P[np.argsort(-P[:, 0])]  # x descending
    hv = 0.0
    y_prev = ref[1]
    for x, y in P:
        if x <= ref[0] or y <= y_prev:
            continue
        hv += (x - ref[0]) * (y - y_prev)
        y_prev = y
    return hv


# ---- experiment ----
def run_source(adj, raw, Sn, src, weights, rng, t0, t1, max_steps):
    R = reachable_from(adj, src)
    Ridx = np.where(R)[0]
    sub = raw[Ridx]
    F = Ridx[pareto_mask(sub)]                       # exact frontier (global indices)
    frontier_pts = raw[F]
    ref = sub.min(0)                                 # nadir over reachable
    Us = Sn @ np.asarray(weights).T                  # (n_states x n_directions)

    # greedy: steepest-ascent climb per direction, swept to convergence
    greedy_arch, greedy_scored = set(), set()
    for k in range(Us.shape[1]):
        end, sc = hill_climb(adj, Us[:, k], src)
        greedy_arch.add(end)
        greedy_scored |= sc
    g_calls = len(greedy_scored)
    g_pts = raw[sorted(greedy_arch)]

    # ours: same legal moves, detour-accepting, at MATCHED budget then at full budget
    ours_m, m_calls = explore_until(adj, Us, src, rng, g_calls, t0, t1, max_steps)
    ours_f, f_calls = explore_until(adj, Us, src, rng, int(R.sum()), t0, t1, max_steps)
    om_pts = raw[sorted(ours_m)]
    of_pts = raw[sorted(ours_f)]

    out = {
        "source_index": int(src),
        "n_reachable": int(R.sum()),
        "frontier_size": int(len(F)),
        "greedy_coverage": coverage(g_pts, frontier_pts),
        "detour_matched_coverage": coverage(om_pts, frontier_pts),
        "detour_full_coverage": coverage(of_pts, frontier_pts),
        "greedy_oracle_calls": int(g_calls),
        "detour_matched_oracle_calls": int(m_calls),
        "detour_full_oracle_calls": int(f_calls),
    }
    if raw.shape[1] == 2:
        hv_ceiling = max(hypervolume_2d(frontier_pts, ref), 1e-12)
        out["greedy_hv_frac"] = hypervolume_2d(g_pts, ref) / hv_ceiling
        out["detour_matched_hv_frac"] = hypervolume_2d(om_pts, ref) / hv_ceiling
        out["detour_full_hv_frac"] = hypervolume_2d(of_pts, ref) / hv_ceiling
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cap", type=int, default=4)
    p.add_argument("--n-slots", type=int, default=8)
    p.add_argument("--seed-smiles", default="C")
    p.add_argument("--divisions", type=int, default=60)
    p.add_argument("--n-sources", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=20000, help="explorer step ceiling")
    p.add_argument("--t0", type=float, default=0.3)
    p.add_argument("--t1", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    smiles, adj, heavy = load_or_build_graph(args.cap, args.n_slots, args.seed_smiles)
    print(json.dumps({"phase": "graph", "n_states": len(adj)}), flush=True)

    result = {"phase": "frontier_recovery", "cap": args.cap, "slices": {}}
    for names in (["logp", "tpsa"], ["logp", "tpsa", "qed"]):
        raw, Sn = objective_matrix(smiles, names)
        div = args.divisions if len(names) == 2 else min(args.divisions, 12)
        weights = simplex_grid(len(names), div)
        sources = farthest_point_sources(Sn, heavy, args.cap, args.n_sources)
        rng = np.random.default_rng(args.seed)
        rows = [run_source(adj, raw, Sn, s, weights, rng, args.t0, args.t1, args.max_steps)
                for s in sources]
        for r in rows:
            r["source_smiles"] = smiles[r["source_index"]]
        agg = {
            "objectives": names,
            "mean_greedy_coverage": float(np.mean([r["greedy_coverage"] for r in rows])),
            "mean_detour_matched_coverage":
                float(np.mean([r["detour_matched_coverage"] for r in rows])),
            "mean_detour_full_coverage": float(np.mean([r["detour_full_coverage"] for r in rows])),
            # can be NEGATIVE: unguided detours are oracle-inefficient at matched budget
            "mean_matched_detour_minus_greedy":
                float(np.mean([r["detour_matched_coverage"] - r["greedy_coverage"] for r in rows])),
            "mean_greedy_oracle": float(np.mean([r["greedy_oracle_calls"] for r in rows])),
            "mean_detour_full_oracle": float(np.mean([r["detour_full_oracle_calls"] for r in rows])),
        }
        if names == ["logp", "tpsa"]:
            agg["mean_greedy_hv_frac"] = float(np.mean([r["greedy_hv_frac"] for r in rows]))
            agg["mean_detour_matched_hv_frac"] = \
                float(np.mean([r["detour_matched_hv_frac"] for r in rows]))
            agg["mean_detour_full_hv_frac"] = float(np.mean([r["detour_full_hv_frac"] for r in rows]))
        result["slices"]["+".join(names)] = {"aggregate": agg, "per_source": rows}
        print(json.dumps({"slice": "+".join(names), **agg}, indent=2), flush=True)

    OUTDIAG.mkdir(parents=True, exist_ok=True)
    out = args.output or (OUTDIAG / f"frontier_recovery_cap{args.cap}.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
