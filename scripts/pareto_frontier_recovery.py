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

FAIR accounting (this replaces an earlier confounded version): an oracle call is
evaluating a molecule's objective vector; a molecule scored once is cached. For
BOTH strategies the archive at budget K is the nondominated set over the first K
distinct molecules EVALUATED -- you keep what you score. We then ask two things,
letting the coverage-vs-budget curve (recorded per source) speak:
  (1) STRUCTURAL ceiling: greedy sweeps every weight direction and evaluates all
      candidate edits; its coverage at its own total budget is the best a monotone
      controller can do. Barrier-gated frontier is invisible to it (never scored).
  (2) MATCHED-BUDGET efficiency: at greedy's OWN oracle budget, does the detour
      walk cover more, less, or the same?

ROBUST findings (cap=4, 6 sources):
  - (1) holds and is EXACT: greedy ceilings at ~54% frontier coverage / ~96% HV;
    the barrier-gated remainder is greedy-inaccessible at ANY budget.
  - Detour-capable control reaches 100% coverage given budget (recoverable).
  - (2) has NO robust answer on this toy. The detour's matched-budget coverage is
    an exploration/exploitation dial: hot (t0=0.3) -> ~0.21 (explores, poor early,
    reaches 100% later); cold (t0=0.02) -> ~0.54 == greedy exactly (a cold
    Metropolis IS greedy). So "unguided detours are oracle-inefficient" is NOT a
    finding -- it was an artifact of one hot setting (retracted). Whether a
    controller gets the barrier-gated frontier at COMPETITIVE budget is decided by
    proposal quality, which this unstructured 739-node graph cannot probe; that is
    the learned-model, real-lead job (archive-aware SMC with Q_theta vs uniform).

Reproduce the sensitivity: sweep --t0 in {0.3,0.1,0.05,0.02}. Every evaluated
state is a complete valid molecule -- the comparison is undefined if intermediates
are not molecules, which is the whole point of the substrate.
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


# ---- search strategies (UNIFIED, fair oracle accounting) ----
# Both strategies return the ORDER in which distinct molecules are EVALUATED. An
# oracle call = computing a molecule's objective vector; a molecule scored once is
# cached and free thereafter. The archive at budget K is the nondominated set over
# the FIRST K distinct molecules evaluated -- once scored you keep it. This removes
# the earlier confound where greedy archived only endpoints and the detour walk
# archived only visited states.
def greedy_eval_order(adj, Us, src):
    """Steepest-ascent monotone search swept over every weight direction. To CHOOSE
    the steepest step it must evaluate all candidate edits, so every neighbour
    examined is an oracle call. Never accepts a worsening edit."""
    order, seen = [], set()

    def ev(i):
        if i not in seen:
            seen.add(i)
            order.append(i)

    ev(src)
    for k in range(Us.shape[1]):
        U = Us[:, k]
        cur = src
        while adj[cur]:
            for v in adj[cur]:
                ev(v)                               # scoring candidate edits = oracle calls
            nxt = max(adj[cur], key=lambda v: U[v])
            if U[nxt] > U[cur] + TOL:
                cur = nxt
            else:
                break
    return order


def detour_eval_order(adj, Us, src, rng, steps, t0, t1):
    """Annealed Metropolis search over the SAME legal moves that CAN step downhill,
    rotating the preference direction. Anneals over its OWN horizon `steps` (so it
    actually cools within budget). Returns the order distinct molecules are scored."""
    order, seen = [], set()

    def ev(i):
        if i not in seen:
            seen.add(i)
            order.append(i)

    ev(src)
    cur = src
    n_dir = Us.shape[1]
    denom = max(steps - 1, 1)
    for step in range(steps):
        U = Us[:, step % n_dir]
        nbrs = adj[cur]
        if not nbrs:
            cur = src
            continue
        v = int(rng.choice(nbrs))
        ev(v)
        temp = t0 * (t1 / t0) ** (step / denom)
        du = U[v] - U[cur]
        if du >= 0 or rng.random() < np.exp(du / temp):
            cur = v
    return order


# ---- metrics ----
def coverage(archive_pts, frontier_pts):
    """Fraction of frontier points weakly dominated by some archived point."""
    if len(frontier_pts) == 0:
        return 1.0
    if len(archive_pts) == 0:
        return 0.0
    cov = 0
    for p in frontier_pts:
        if np.any(np.all(archive_pts >= p - 1e-9, axis=1)):
            cov += 1
    return cov / len(frontier_pts)


def hypervolume_2d(pts, ref):
    """2D hypervolume (maximization) above reference nadir `ref`."""
    if len(pts) == 0:
        return 0.0
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
def run_source(adj, raw, Sn, src, weights, rng, t0, t1, detour_step_mult):
    R = reachable_from(adj, src)
    Ridx = np.where(R)[0]
    sub = raw[Ridx]
    F = Ridx[pareto_mask(sub)]                       # exact frontier (global indices)
    frontier_pts = raw[F]
    ref = sub.min(0)                                 # nadir over reachable
    n_reach = int(R.sum())
    Us = Sn @ np.asarray(weights).T                  # (n_states x n_directions)

    g_order = greedy_eval_order(adj, Us, src)        # greedy evaluation order
    bg = len(g_order)                                # greedy's TOTAL oracle budget (its ceiling)
    d_order = detour_eval_order(adj, Us, src, rng, detour_step_mult * n_reach, t0, t1)

    def cov_at(order, k):
        return coverage(raw[order[:k]], frontier_pts)

    def hv_at(order, k):
        return hypervolume_2d(raw[order[:k]], ref)

    # coverage vs oracle budget, so nothing is cherry-picked
    budgets = sorted(set(np.unique(np.linspace(1, n_reach, 20).astype(int)).tolist()) | {bg})
    greedy_curve = [cov_at(g_order, min(k, bg)) for k in budgets]  # greedy caps at its budget bg
    detour_curve = [cov_at(d_order, min(k, len(d_order))) for k in budgets]

    out = {
        "source_index": int(src),
        "n_reachable": n_reach,
        "frontier_size": int(len(F)),
        "greedy_oracle_budget": int(bg),
        # both archive nondominated-over-all-evaluated; greedy at its ceiling (all bg evals)
        "greedy_coverage": cov_at(g_order, bg),
        "detour_coverage_at_matched": cov_at(d_order, min(bg, len(d_order))),
        "detour_coverage_full": cov_at(d_order, min(n_reach, len(d_order))),
        "budgets": budgets,
        "greedy_curve": greedy_curve,
        "detour_curve": detour_curve,
    }
    if raw.shape[1] == 2:                            # hypervolume defined here for the 2-obj slice
        hv_ceiling = max(hypervolume_2d(frontier_pts, ref), 1e-12)
        out["greedy_hv_frac"] = hv_at(g_order, bg) / hv_ceiling
        out["detour_hv_at_matched"] = hv_at(d_order, min(bg, len(d_order))) / hv_ceiling
        out["detour_hv_full"] = hv_at(d_order, min(n_reach, len(d_order))) / hv_ceiling
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cap", type=int, default=4)
    p.add_argument("--n-slots", type=int, default=8)
    p.add_argument("--seed-smiles", default="C")
    p.add_argument("--divisions", type=int, default=60)
    p.add_argument("--n-sources", type=int, default=6)
    p.add_argument("--detour-step-mult", type=int, default=40,
                   help="detour walk length = mult * n_reachable (anneals over this horizon)")
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
        rows = [run_source(adj, raw, Sn, s, weights, rng, args.t0, args.t1, args.detour_step_mult)
                for s in sources]
        for r in rows:
            r["source_smiles"] = smiles[r["source_index"]]
        agg = {
            "objectives": names,
            "mean_greedy_oracle_budget": float(np.mean([r["greedy_oracle_budget"] for r in rows])),
            # ALL archives = nondominated over every molecule evaluated; MATCHED = detour at
            # greedy's own oracle budget. The sign of the matched delta is the honest verdict.
            "mean_greedy_coverage": float(np.mean([r["greedy_coverage"] for r in rows])),
            "mean_detour_coverage_at_matched":
                float(np.mean([r["detour_coverage_at_matched"] for r in rows])),
            "mean_detour_coverage_full": float(np.mean([r["detour_coverage_full"] for r in rows])),
            "mean_matched_detour_minus_greedy_coverage":
                float(np.mean([r["detour_coverage_at_matched"] - r["greedy_coverage"]
                               for r in rows])),
        }
        if names == ["logp", "tpsa"]:
            agg["mean_greedy_hv_frac"] = float(np.mean([r["greedy_hv_frac"] for r in rows]))
            agg["mean_detour_hv_at_matched"] = \
                float(np.mean([r["detour_hv_at_matched"] for r in rows]))
            agg["mean_matched_detour_minus_greedy_hv"] = \
                float(np.mean([r["detour_hv_at_matched"] - r["greedy_hv_frac"] for r in rows]))
        result["slices"]["+".join(names)] = {"aggregate": agg, "per_source": rows}
        print(json.dumps({"slice": "+".join(names), **agg}, indent=2), flush=True)

    OUTDIAG.mkdir(parents=True, exist_ok=True)
    out = args.output or (OUTDIAG / f"frontier_recovery_cap{args.cap}.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
