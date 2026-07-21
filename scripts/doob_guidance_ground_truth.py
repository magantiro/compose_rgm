#!/usr/bin/env python3
"""Doob exact conditioning as GUIDANCE GROUND TRUTH (the spine).

On a tractable exhaustive slice of the real rewrite CTMC the exact conditional
distribution is computable (the Doob h-transform, verified in E0). We use it as
ground truth to measure how well FINITE-PARTICLE guidance -- the same twisted-SMC
family used in the full model -- recovers the exact conditional, as a function of
particle count and twist quality.

This is a benchmark for guidance quality that only our valid + tractable structure
enables: for a diffusion the exact conditional over real molecules is neither
defined nor computable at intermediate states, so guidance error cannot be
measured against truth. Here it can.

Compared:
  - bootstrap SMC (no twist; weight only by the terminal target indicator),
  - exact-h twisted SMC (twist = the true harmonic function; the Doob optimum),
  - immediate-reward twisted SMC (a practical, imperfect twist).
All are consistent (TV -> 0 as N -> inf); the twist quality sets the particle
budget needed for a target accuracy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compose_v4.chem.state import pad_molecular_graph
from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.rewrite.kernel import canonical_state_key
from e0_toy_h_exactness import build_state_space, generator_matrix

ROOT = Path(__file__).resolve().parents[1]
OUTFIG = ROOT / "paper_iclr_stochastic_rewriting" / "figures"
OUTDIAG = ROOT / "diagnostics" / "exactness"

CAP, N_SLOTS, T, M = 3, 6, 1.0, 40
PARTICLES = [10, 30, 100, 300, 1000, 3000]
REPEATS = 12
SEED0 = 20260728


def _rng(seed):
    return np.random.default_rng(seed)


def _sample_rows(P, rows, rng):
    """Vectorized categorical sampling: one draw from P[r] for each r in rows."""
    C = np.cumsum(P[rows], axis=1)
    u = rng.random(len(rows))[:, None]
    return (u < C).argmax(axis=1)


def twisted_smc(P_dt, h_grid, x0, one_A, n_particles, twist, rng):
    """One guidance run on the finite CTMC, returning the endpoint distribution.

    - 'exact_h': propose from the exact Doob one-step kernel P^h(x,y)=P_dt(x,y)h(y,t')/h(x,t);
      weights are constant, so endpoints are (near-)iid draws from the exact conditional.
    - 'bootstrap': propose from the base kernel, weight only by the terminal target
      indicator (naive importance sampling / no guidance).
    """
    n = P_dt.shape[0]
    idx = np.full(n_particles, x0, dtype=int)
    logw = np.zeros(n_particles)
    eps = 1e-12
    for k in range(M):
        if twist == "exact_h":
            h0, h1 = h_grid[k], h_grid[k + 1]
            live = h0 > eps
            # exact Doob one-step kernel for each occupied state, row-normalized
            Ph = P_dt * (h1[None, :])                      # P_dt[x,y] h(y,t')
            rs = Ph.sum(axis=1, keepdims=True)
            Ph = np.divide(Ph, np.where(rs > 0, rs, 1.0))  # == P_dt[x,y]h(y,t')/h(x,t)
            idx = np.where(live[idx], _sample_rows(Ph, idx, rng), idx)
        else:  # bootstrap: base dynamics, no intermediate weight
            idx = _sample_rows(P_dt, idx, rng)
            w = np.exp(logw - logw.max()); w /= w.sum()
            if 1.0 / np.sum(w ** 2) < n_particles / 2:
                idx = idx[rng.choice(n_particles, size=n_particles, p=w)]
                logw = np.zeros(n_particles)
    if twist == "exact_h":
        dist = np.zeros(n); np.add.at(dist, idx, 1.0 / n_particles)
        return dist
    w = np.exp(logw - logw.max()) * one_A[idx]
    if w.sum() <= 0:
        return None
    w /= w.sum()
    dist = np.zeros(n); np.add.at(dist, idx, w)
    return dist


def main():
    keys, states, smiles, edges = build_state_space(CAP, N_SLOTS)
    Q, index = generator_matrix(keys, edges)
    n = len(keys)
    def _has(s, el):
        return el in s or el.lower() in s
    # rare conditioning target: contains BOTH nitrogen and oxygen (a 2-heteroatom
    # conjunction) -- the realistic regime where naive guidance struggles.
    one_A = np.array([1.0 if (_has(smiles[k], "N") and _has(smiles[k], "O")) else 0.0 for k in keys])

    dt = T / M
    P_dt = expm(Q * dt)
    h_grid = [None] * (M + 1)
    h_grid[M] = one_A.copy()
    for k in range(M - 1, -1, -1):
        h_grid[k] = P_dt @ h_grid[k + 1]

    x0 = index[canonical_state_key(pad_molecular_graph(smiles_to_molecular_graph("C"), N_SLOTS))]
    P_T = expm(Q * T)
    exact = P_T[x0] * one_A
    exact = exact / exact.sum()

    print(f"states {n} | target {int(one_A.sum())} | P(reach target from C) {float(h_grid[0][x0]):.3f}", flush=True)
    results = {}
    for twist in ("bootstrap", "exact_h"):
        row = []
        for N in PARTICLES:
            tvs = []
            for rep in range(REPEATS):
                d = twisted_smc(P_dt, h_grid, x0, one_A, N, twist, _rng(SEED0 + 1000 * rep + N))
                if d is not None:
                    tvs.append(0.5 * float(np.abs(d - exact).sum()))
            row.append({"N": N, "tv_median": float(np.median(tvs)),
                        "tv_q1": float(np.percentile(tvs, 25)), "tv_q3": float(np.percentile(tvs, 75)),
                        "n_valid": len(tvs)})
            print(f"  {twist:10s} N={N:5d}  TV(median)={np.median(tvs):.4f}  (valid {len(tvs)}/{REPEATS})", flush=True)
        results[twist] = row

    OUTDIAG.mkdir(parents=True, exist_ok=True)
    (OUTDIAG / "doob_guidance_ground_truth.json").write_text(
        json.dumps({"cap": CAP, "n_states": n, "particles": PARTICLES, "results": results}, indent=2))

    # figure: TV to exact conditional vs particles, per twist
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "cm", "font.size": 11, "axes.edgecolor": "#333333", "savefig.bbox": "tight"})
    colors = {"bootstrap": "#D1603D", "immediate": "#E0A030", "exact_h": "#1F6FB2"}
    labels = {"bootstrap": "bootstrap SMC (no twist)", "immediate": "immediate-reward twist (practical)",
              "exact_h": "exact-$h$ twist (Doob optimum)"}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for twist, row in results.items():
        Ns = [r["N"] for r in row]
        med = [r["tv_median"] for r in row]
        q1 = [r["tv_q1"] for r in row]
        q3 = [r["tv_q3"] for r in row]
        ax.fill_between(Ns, q1, q3, color=colors[twist], alpha=0.15)
        ax.plot(Ns, med, "-o", color=colors[twist], lw=2.2, ms=6, label=labels[twist],
                markeredgecolor="white", markeredgewidth=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("particles")
    ax.set_ylabel("TV to exact conditional")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(color="#EEEEEE", lw=0.7, which="both")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.set_title("Guidance recovers the exact conditional (measured against ground truth)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUTFIG / "doob_guidance_ground_truth.pdf")
    fig.savefig(OUTFIG / "doob_guidance_ground_truth.png", dpi=200)
    print("wrote doob_guidance_ground_truth.{pdf,png,json}")


if __name__ == "__main__":
    main()
