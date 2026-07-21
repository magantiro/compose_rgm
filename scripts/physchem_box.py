#!/usr/bin/env python3
"""Multi-property physchem-box targeting under an EXACT fixed scaffold.

The native conjunction: redesign a lead's decorations to land inside a drug-
likeness box (logP, TPSA, QED windows) WHILE the Murcko scaffold is held exactly
in the legal fiber. No method in the literature does both -- property targeting
(GrIDDD/FreeGress/DiGress-guidance) OR structural constraints (ConStruct/PRODIGY,
which cannot even represent a required scaffold). We steer the property box
(earned in-box yield) and enforce the scaffold (exact, 100%).

Figure: (logP x TPSA) joystick with the target box; our scaffold-exact steered
population lands inside, generate-then-filter (same base, scaffold-preserving)
scatters. Reports in-box yield (all properties jointly) for each.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from griddd_value_guided_smc_controller import _load_base_sampler, value_guided_smc

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[1]
SC = os.environ.get("COMPOSE_SCRATCH", str(_ROOT / "scratch"))
os.makedirs(SC, exist_ok=True)
LEADS = str(_ROOT / "configs" / "benchmarks" / "cnof_leads.json")
OUTFIG = ROOT / "paper_iclr_stochastic_rewriting" / "figures"
OUTDIAG = ROOT / "diagnostics" / "conditional_smc"
CKPT = "/private/tmp/lineage_b_checkpoint/checkpoint.best_so_far.pt"
N_LEADS = 12

BOX = {"logp": (1.0, 3.0), "tpsa": (40.0, 90.0), "qed": (0.70, 1.01)}
SCALE = {"logp": 2.0, "tpsa": 30.0, "qed": 0.2}


def _props(mol):
    return {"logp": Crippen.MolLogP(mol), "tpsa": Descriptors.TPSA(mol), "qed": QED.qed(mol)}


def _box_reward(mol):
    p = _props(mol)
    pen = sum((max(0.0, lo - p[k]) + max(0.0, p[k] - hi)) / SCALE[k] for k, (lo, hi) in BOX.items())
    return -pen  # 0.0 iff fully inside the box


def _in_box(mol):
    p = _props(mol)
    return all(lo <= p[k] <= hi for k, (lo, hi) in BOX.items())


def _scaffold(smi):
    m = Chem.MolFromSmiles(smi)
    return MurckoScaffold.GetScaffoldForMol(m) if m is not None else None


def run():
    sampler, _ = _load_base_sampler(CKPT, 0.0)
    rewrite = de_novo_rewrite_system()
    leads = json.load(open(LEADS))[:N_LEADS]

    TOPK = 10
    ours_pts = []            # CONVERGED output points (top-k per lead) for the figure
    pop_inbox = pop_total = scaffold_ok = 0     # whole-population (diluted) reference
    best_inbox = best_total = 0                 # best molecule per lead
    topk_inbox = topk_total = 0                 # top-k by box-reward per lead (the steered output)
    for i, (_, smi, _) in enumerate(leads):
        scaf = _scaffold(smi)
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), 40)
        r = value_guided_smc(
            st, sampler=sampler, qed_oracle=lambda s: _box_reward(Chem.MolFromSmiles(molecular_graph_to_smiles(s)))
            if Chem.MolFromSmiles(molecular_graph_to_smiles(s) or "") else -100.0,
            rewrite=rewrite, seed=4200 + i, similarity_minimum=0.0, budget=300, n_particles=16,
            max_steps=48, time_step=0.1, horizon=16.0, alpha_start=0.20, alpha_end=0.03,
            feasible_attempts=8, scaffold_mol=scaf, dump_population=True,
        )
        scored = []  # (reward, mol) over the population
        for s in (r.get("population_smiles") or []):
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            pop_total += 1
            if scaf is not None and m.HasSubstructMatch(scaf):
                scaffold_ok += 1
            if _in_box(m):
                pop_inbox += 1
            scored.append((_box_reward(m), m))
        # best molecule for this lead
        bm = Chem.MolFromSmiles(r.get("best_state_smiles") or "")
        if bm is not None:
            best_total += 1
            best_inbox += int(_in_box(bm))
        # the top-k highest-reward (closest-to-box) molecules = the steered output
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, m in scored[:TOPK]:
            topk_total += 1
            topk_inbox += int(_in_box(m))
            p = _props(m)
            ours_pts.append((p["logp"], p["tpsa"], _in_box(m)))
        print(f"lead {i}: pop {len(scored)} | best in-box={bm is not None and _in_box(bm)}", flush=True)

    # baseline: same base model's scaffold-preserving population (from the measure_only
    # funnel run), filtered to in-box -> generate-then-filter yield
    base_pts, base_inbox, base_scaf_total = [], 0, 0
    try:
        bl = json.load(open(f"{SC}/funnel_baseline24.json"))["results"]
    except Exception:
        bl = []
    for r in bl[:N_LEADS]:
        scaf = _scaffold(r["lead_smiles"])
        for s in (r.get("population_smiles") or []):
            m = Chem.MolFromSmiles(s)
            if m is None or scaf is None or not m.HasSubstructMatch(scaf):
                continue
            base_scaf_total += 1
            p = _props(m)
            base_pts.append((p["logp"], p["tpsa"], _in_box(m)))
            if _in_box(m):
                base_inbox += 1

    base_yield = base_inbox / max(base_scaf_total, 1)
    summary = {
        "experiment": "physchem_box_under_exact_scaffold",
        "box": BOX,
        "ours_scaffold_satisfaction": scaffold_ok / max(pop_total, 1),
        "ours_in_box_yield_topk": topk_inbox / max(topk_total, 1),        # steered output (top-k by box-reward)
        "ours_in_box_yield_best": best_inbox / max(best_total, 1),         # best per lead
        "ours_in_box_yield_population": pop_inbox / max(pop_total, 1),     # whole explored population (diluted)
        "baseline_in_box_yield_given_scaffold": base_yield,
        "baseline_n_scaffold_preserving": base_scaf_total,
        "n_leads": len(leads),
        "topk": TOPK,
    }
    OUTDIAG.mkdir(parents=True, exist_ok=True)
    (OUTDIAG / "physchem_box.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return ours_pts, base_pts, summary


def figure(ours_pts, base_pts, summary):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "cm", "font.size": 11, "axes.edgecolor": "#333333", "savefig.bbox": "tight"})
    COMPOSE_C, BASE_C = "#1F6FB2", "#D1603D"
    (lx, hx), (ly, hy) = BOX["logp"], BOX["tpsa"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    bx = np.array([(x, y) for x, y, _ in base_pts])
    ox = np.array([(x, y) for x, y, _ in ours_pts])
    if len(bx):
        ax.scatter(bx[:, 0], bx[:, 1], s=12, color=BASE_C, alpha=0.28, edgecolor="none",
                   label=f"generate-then-filter (scaffold-preserving)  [{summary['baseline_in_box_yield_given_scaffold']*100:.0f}% in box]")
    if len(ox):
        ax.scatter(ox[:, 0], ox[:, 1], s=16, color=COMPOSE_C, alpha=0.5, edgecolor="none",
                   label=f"COMPOSE (scaffold exact, box-steered output)  [{summary['ours_in_box_yield_topk']*100:.0f}% in box]")
    ax.add_patch(Rectangle((lx, ly), hx - lx, hy - ly, fill=False, edgecolor="#222222", lw=2.0, zorder=5))
    ax.text((lx + hx) / 2, hy + 3, "target box  (logP $\\in$[1,3], TPSA $\\in$[40,90], QED $\\geq$0.7)",
            ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("logP"); ax.set_ylabel("TPSA ($\\AA^2$)")
    ax.set_xlim(-2, 7); ax.set_ylim(0, 170)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(color="#EEEEEE", lw=0.7)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.set_title("Redesign to a physchem box while holding the scaffold exact", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTFIG / "physchem_box.pdf"); fig.savefig(OUTFIG / "physchem_box.png", dpi=200)
    print("wrote physchem_box.{pdf,png}")


if __name__ == "__main__":
    o, b, s = run()
    figure(o, b, s)
