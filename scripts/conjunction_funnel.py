#!/usr/bin/env python3
"""The conjunction feasibility funnel (killer experiment A).

A realistic multi-constraint drug-design spec is a CONJUNCTION: keep the lead's
scaffold AND avoid toxicophores AND stay in drug-like property windows. As
constraints are AND-ed, the joint feasible set shrinks multiplicatively, so a
generate-then-filter baseline (drawing from the same base model near the lead)
collapses toward zero usable yield -- while COMPOSE enforces the whole
conjunction in the legal fiber and stays at 100% by construction.

This evaluates the growing conjunction offline on the base model's dumped
feasible population (from --dump-population, measure_only) and reports the
collapse + the implied generate-then-filter cost (1 / final yield).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[1]
SC = os.environ.get("COMPOSE_SCRATCH", str(_ROOT / "scratch"))
os.makedirs(SC, exist_ok=True)
LEADS = str(_ROOT / "configs" / "benchmarks" / "cnof_leads.json")
OUTFIG = ROOT / "paper_iclr_stochastic_rewriting" / "figures"
OUTDIAG = ROOT / "diagnostics" / "conditional_smc"

NITRO = Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])]")
ALDEHYDE = Chem.MolFromSmarts("[CX3H1](=O)")

# The ordered design spec (each row is AND-ed onto the previous).
SPEC = [
    ("sim $\\geq$ 0.4", lambda m, c: True),  # the population IS the similarity fiber
    ("scaffold", lambda m, c: c["scaffold"] is not None and m.HasSubstructMatch(c["scaffold"])),
    ("no nitro", lambda m, c: not m.HasSubstructMatch(NITRO)),
    ("no aldehyde", lambda m, c: not m.HasSubstructMatch(ALDEHYDE)),
    ("MW $\\in$ [200,450]", lambda m, c: 200.0 <= Descriptors.MolWt(m) <= 450.0),
    ("logP $\\in$ [0,4]", lambda m, c: 0.0 <= Crippen.MolLogP(m) <= 4.0),
    ("QED $\\geq$ 0.6", lambda m, c: QED.qed(m) >= 0.6),
]
LABELS = [s[0] for s in SPEC]


def _ctx(lead_smiles):
    lead = Chem.MolFromSmiles(lead_smiles)
    scaf = MurckoScaffold.GetScaffoldForMol(lead) if lead is not None else None
    return {"lead": lead, "scaffold": scaf}


def prefix_satisfaction(pop_smiles, ctx):
    """Return, per prefix length k, the fraction of the population satisfying C1..Ck."""
    mols = [Chem.MolFromSmiles(s) for s in pop_smiles]
    mols = [m for m in mols if m is not None]
    if not mols:
        return None
    n = len(mols)
    yields = []
    survivors = mols
    for _, check in SPEC:
        survivors = [m for m in survivors if check(m, ctx)]
        yields.append(len(survivors) / n)
    return yields


def main():
    data = json.load(open(f"{SC}/funnel_baseline24.json"))
    rows = [r for r in data["results"] if r.get("population_smiles")]
    per_lead = []
    lead_satisfies_full = 0
    for r in rows:
        ctx = _ctx(r["lead_smiles"])
        if ctx["scaffold"] is None:
            continue
        y = prefix_satisfaction(r["population_smiles"], ctx)
        if y is None:
            continue
        per_lead.append(y)
        # sanity: does the lead itself satisfy the full spec? (feasible set non-empty)
        lead = ctx["lead"]
        if all(check(lead, ctx) for _, check in SPEC):
            lead_satisfies_full += 1
    per_lead = np.array(per_lead)  # (n_leads, 7)
    mean_yield = per_lead.mean(0)  # baseline usable fraction per prefix

    print(f"leads: {len(per_lead)} | leads whose own molecule meets the full spec: {lead_satisfies_full}/{len(per_lead)}")
    print(f"{'constraints AND-ed':<40}{'baseline yield':>16}{'ours':>8}")
    for k, lab in enumerate(LABELS):
        print(f"  C1..C{k+1}  {lab:<30}{mean_yield[k]*100:>14.2f}%{'100%':>8}")
    final = float(mean_yield[-1])
    implied = (1.0 / final) if final > 0 else float("inf")
    print(f"\nfull-spec baseline yield: {final*100:.3f}%  ->  generate-then-filter needs ~{implied:,.0f} draws per compliant hit")
    print("COMPOSE: 100% by construction (whole conjunction enforced in the legal fiber)")

    out = {
        "experiment": "conjunction_feasibility_funnel",
        "spec": LABELS,
        "n_leads": len(per_lead),
        "leads_meeting_full_spec": int(lead_satisfies_full),
        "baseline_yield_by_prefix": [float(x) for x in mean_yield],
        "ours_yield_by_prefix": [1.0] * len(LABELS),
        "full_spec_baseline_yield": final,
        "implied_generate_then_filter_draws_per_hit": implied,
    }
    (OUTDIAG / "conjunction_funnel.json").write_text(json.dumps(out, indent=2))

    # --- figure ---
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "cm", "font.size": 11, "axes.edgecolor": "#333333",
                         "savefig.bbox": "tight"})
    COMPOSE_C, BASE_C = "#1F6FB2", "#D1603D"
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    x = np.arange(1, len(LABELS) + 1)
    ax.fill_between(x, mean_yield * 100, 100, color="#9AA7B0", alpha=0.16, zorder=1,
                    label="advantage of in-fiber enforcement")
    for row in per_lead:
        ax.plot(x, np.clip(row, 1e-4, 1) * 100, color=BASE_C, lw=0.6, alpha=0.18, zorder=2)
    ax.plot(x, mean_yield * 100, "-o", color=BASE_C, lw=2.4, ms=6.5, zorder=4,
            markeredgecolor="white", markeredgewidth=0.8, label="generate-then-filter (base + reject)")
    ax.axhline(100, color=COMPOSE_C, lw=2.6, zorder=5)
    ax.text(len(LABELS), 118, "COMPOSE (in-fiber): 100%", color=COMPOSE_C, ha="right", va="bottom", fontsize=10.5)
    ax.annotate(f"full spec: {final*100:.2f}%\n($\\approx$1 in {implied:,.0f} draws)",
                (len(LABELS), max(mean_yield[-1] * 100, 0.02)), textcoords="offset points",
                xytext=(-6, 18), ha="right", color=BASE_C, fontsize=9.5)
    ax.set_yscale("log")
    ax.set_ylim(max(mean_yield[mean_yield > 0].min() * 100 * 0.4, 1e-2), 200)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\wedge$\n{l}" if i else l for i, l in enumerate(LABELS)], fontsize=8.5)
    ax.set_ylabel("usable yield (\\%, log scale)")
    ax.set_xlabel("design-spec constraints, conjoined left to right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E4E4E4", lw=0.7, which="both", zorder=0)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUTFIG / "conjunction_funnel.pdf")
    fig.savefig(OUTFIG / "conjunction_funnel.png", dpi=200)
    print("wrote conjunction_funnel.{pdf,png} and conjunction_funnel.json")


if __name__ == "__main__":
    main()
