#!/usr/bin/env python3
"""Publication figures for the COMPOSE control paper.

Fig 1: scaffold-collapse curve -- generate-then-filter usable fraction collapses
       with scaffold size while COMPOSE stays at 100% (the headline result).
Fig 2: anytime curve -- best-so-far feasible reward vs oracle calls under an
       exactly enforced scaffold (E3), showing ~90% of gain by ~35 calls.

Outputs vector PDF (for LaTeX) and PNG (for inspection) into paper .../figures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "diagnostics" / "conditional_smc"
OUT = ROOT / "paper_iclr_stochastic_rewriting" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- shared style: serif to match the paper's Times, clean spines, cb-safe palette ---
COMPOSE_C = "#1F6FB2"   # blue
BASE_C = "#D1603D"      # terracotta
GAP_C = "#9AA7B0"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _murcko_size(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    s = MurckoScaffold.GetScaffoldForMol(m)
    return s.GetNumHeavyAtoms() if s is not None else 0


def figure_scaffold_collapse():
    d = json.load(open(DIAG / "smc_scaffold_control48.json"))
    rows = d["results"]
    sizes, fracs = [], []
    for x in rows:
        f = x.get("scaffold_preservation_fraction")
        if f is None:
            continue
        n = _murcko_size(x["lead_smiles"])
        if n is None:
            continue
        sizes.append(n)
        fracs.append(100.0 * f)
    sizes = np.array(sizes)
    fracs = np.array(fracs)

    # committed binned baseline (matches the paper table)
    bins = [("$\\leq 8$", sizes <= 8), ("$9$--$15$", (sizes >= 9) & (sizes <= 15)),
            ("$16$--$22$", (sizes >= 16) & (sizes <= 22)), ("$\\geq 23$", sizes >= 23)]
    centers, means, labels = [], [], []
    for lab, mask in bins:
        if mask.sum() == 0:
            continue
        centers.append(float(sizes[mask].mean()))
        means.append(float(fracs[mask].mean()))
        labels.append(lab)
    centers = np.array(centers)
    means = np.array(means)

    fig, ax = plt.subplots(figsize=(6.3, 3.7))
    # widening gap band
    ax.fill_between(centers, means, 100, color=GAP_C, alpha=0.16, zorder=1,
                    label="advantage of in-fiber enforcement")
    # per-lead baseline scatter
    ax.scatter(sizes, fracs, s=26, color=BASE_C, alpha=0.35, edgecolor="none",
               zorder=2, label="generate-then-filter (per lead)")
    # binned baseline trend
    ax.plot(centers, means, "-o", color=BASE_C, lw=2.4, ms=7, zorder=4,
            markeredgecolor="white", markeredgewidth=0.8,
            label="generate-then-filter (binned mean)")
    for cx, cy in zip(centers, means):
        ax.annotate(f"{cy:.0f}%", (cx, cy), textcoords="offset points",
                    xytext=(0, -14), ha="center", color=BASE_C, fontsize=9.5)
    # COMPOSE flat 100%
    ax.axhline(100, color=COMPOSE_C, lw=2.6, zorder=5)
    ax.text(sizes.max() * 0.99, 100.9, "COMPOSE (in-fiber): 100%",
            color=COMPOSE_C, ha="right", va="bottom", fontsize=10.5)

    ax.set_xlabel("Murcko scaffold size (heavy atoms)")
    ax.set_ylabel("Constraint satisfaction (%)")
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.xaxis.set_major_locator(MultipleLocator(5))
    _despine(ax)
    ax.grid(axis="y", color="#DDDDDD", lw=0.7, zorder=0)
    ax.legend(loc="lower left", frameon=False, fontsize=9, handletextpad=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "scaffold_collapse.pdf")
    fig.savefig(OUT / "scaffold_collapse.png", dpi=200)
    plt.close(fig)
    print("wrote scaffold_collapse.{pdf,png}  n_leads=", len(sizes))


def figure_anytime():
    d = json.load(open(DIAG / "scaffold_opt_panel12.json"))
    rows = [x for x in d["results"] if x.get("best_qed_trace")]
    # common oracle-call grid; hold last value (step function) beyond each trace
    max_calls = max(tr["best_qed_trace"][-1][0] for tr in rows)
    grid = np.arange(0, max_calls + 1)
    raw_curves, gain_curves = [], []
    for x in rows:
        tr = np.array(x["best_qed_trace"], dtype=float)
        calls, best = tr[:, 0], tr[:, 1]
        raw = np.interp(grid, calls, best, right=best[-1])
        raw_curves.append(raw)
        lead, final = best[0], best[-1]
        if final > lead:
            gain_curves.append(np.clip((raw - lead) / (final - lead), 0, 1))
    raw_curves = np.array(raw_curves)
    gain_curves = np.array(gain_curves)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 3.3))

    # left: raw best-so-far QED per lead + mean
    for c in raw_curves:
        axL.plot(grid, c, color=COMPOSE_C, lw=0.8, alpha=0.28)
    axL.plot(grid, raw_curves.mean(0), color=COMPOSE_C, lw=2.6, label="mean (12 leads)")
    axL.set_xlabel("oracle calls")
    axL.set_ylabel("best-so-far QED (scaffold fixed)")
    axL.set_xlim(0, max_calls)
    _despine(axL)
    axL.grid(color="#EEEEEE", lw=0.7)
    axL.legend(loc="lower right", frameon=False, fontsize=9)

    # right: median normalized gain (robust to a few late-jumping leads) + IQR band.
    # The 90%-crossing of the median curve is the reported "typical lead" statistic.
    med = np.median(gain_curves, 0)
    q1, q3 = np.percentile(gain_curves, [25, 75], axis=0)
    axR.fill_between(grid, q1, q3, color=COMPOSE_C, alpha=0.18, label="IQR (12 leads)")
    axR.plot(grid, med, color=COMPOSE_C, lw=2.6, label="median lead")
    c90 = int(grid[np.argmax(med >= 0.9)]) if (med >= 0.9).any() else None
    axR.axhline(0.9, color="#666666", lw=1.0, ls="--")
    if c90 is not None:
        axR.axvline(c90, color="#666666", lw=1.0, ls="--")
        axR.annotate(f"median lead: 90% of\nfinal gain by ~{c90} calls", (c90, 0.40),
                     textcoords="offset points", xytext=(12, 0), fontsize=9.5,
                     color="#333333", va="center")
    axR.set_xlabel("oracle calls")
    axR.set_ylabel("fraction of final gain")
    axR.set_xlim(0, max_calls)
    axR.set_ylim(0, 1.03)
    _despine(axR)
    axR.grid(color="#EEEEEE", lw=0.7)
    axR.legend(loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "anytime.pdf")
    fig.savefig(OUT / "anytime.png", dpi=200)
    plt.close(fig)
    print(f"wrote anytime.{{pdf,png}}  n_leads={len(rows)}  90%_gain_by={c90} calls")


if __name__ == "__main__":
    figure_scaffold_collapse()
    figure_anytime()
