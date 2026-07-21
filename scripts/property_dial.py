#!/usr/bin/env python3
"""Property dial: conditional generation to a TARGET property value.

For each target logP, the value-guided SMC over the validity-closed rewrite CTMC
is steered (reward = -|logP - target|) to generate molecules whose logP hits the
target. Sweeping the target shows the generated distribution tracking the dial --
controllable property targeting on real, valid molecules at every step. No
similarity constraint here (pure targeting); every committed state is still a
complete valid molecule, so the property oracle is defined at every step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles, smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from griddd_value_guided_smc_controller import _load_base_sampler, value_guided_smc

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
SC = "/private/tmp/claude-502/-Users-rmaganti-Documents-Codex-2026-07-14-ok-so-compose-rgm-claude-generators/6d6fc94f-1f64-41f3-8db5-e0589f315b48/scratchpad"
OUTFIG = ROOT / "paper_iclr_stochastic_rewriting" / "figures"
OUTDIAG = ROOT / "diagnostics" / "conditional_smc"
CKPT = "/private/tmp/lineage_b_checkpoint/checkpoint.best_so_far.pt"
TARGETS = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
N_LEADS = 12


def _logp(smi):
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    return Crippen.MolLogP(m) if m is not None else None


def run_sweep():
    sampler, _ = _load_base_sampler(CKPT, 0.0)
    rewrite = de_novo_rewrite_system()
    leads = json.load(open(f"{SC}/cnof_leads.json"))[:N_LEADS]
    states = [pad_molecular_graph(smiles_to_molecular_graph(s), 40) for _, s, _ in leads]
    lead_logps = [_logp(s) for _, s, _ in leads]
    print(f"leads: {len(states)} | lead logP range {min(lead_logps):.2f}..{max(lead_logps):.2f}", flush=True)

    out = {}
    for target in TARGETS:
        def oracle(state, t=target):
            lp = _logp(molecular_graph_to_smiles(state))
            return -abs(lp - t) if lp is not None else -100.0

        best, topk = [], []
        for i, st in enumerate(states):
            r = value_guided_smc(
                st, sampler=sampler, qed_oracle=oracle, rewrite=rewrite,
                seed=7000 + int((target + 1) * 100) + i, similarity_minimum=0.0,
                budget=250, n_particles=16, max_steps=48, time_step=0.1, horizon=16.0,
                alpha_start=0.20, alpha_end=0.03, feasible_attempts=8, dump_population=True,
            )
            b = _logp(r.get("best_state_smiles"))
            if b is not None:
                best.append(b)
            # richer distribution: the 10 population molecules closest to target
            pop = [(_logp(s), s) for s in (r.get("population_smiles") or [])]
            pop = [(v, s) for v, s in pop if v is not None]
            pop.sort(key=lambda x: abs(x[0] - target))
            topk.extend(v for v, _ in pop[:10])
        out[str(target)] = {"best": best, "topk": topk}
        arr = np.array(topk) if topk else np.array([np.nan])
        print(f"target logP={target:+.0f}: generated median {np.median(arr):+.2f} "
              f"(IQR {np.percentile(arr,25):+.2f}..{np.percentile(arr,75):+.2f}, n={len(topk)})", flush=True)

    OUTDIAG.mkdir(parents=True, exist_ok=True)
    json.dump({"targets": TARGETS, "results": out, "lead_logps": lead_logps},
              open(f"{SC}/property_dial.json", "w"))
    return out


def make_figure(out):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "cm", "font.size": 11, "axes.edgecolor": "#333333",
                         "savefig.bbox": "tight"})
    COMPOSE_C = "#1F6FB2"
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    lo, hi = min(TARGETS) - 1.5, max(TARGETS) + 1.5
    ax.plot([lo, hi], [lo, hi], color="#999999", lw=1.2, ls="--", zorder=1, label="ideal (generated = target)")
    data = [np.array(out[str(t)]["topk"]) for t in TARGETS]
    parts = ax.violinplot(data, positions=TARGETS, widths=0.7, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(COMPOSE_C); b.set_alpha(0.35); b.set_edgecolor(COMPOSE_C)
    meds = [float(np.median(d)) for d in data]
    ax.plot(TARGETS, meds, "o-", color=COMPOSE_C, lw=2.2, ms=7, zorder=4,
            markeredgecolor="white", markeredgewidth=0.8, label="generated median")
    ax.set_xlabel("target logP (the dial)")
    ax.set_ylabel("generated logP")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xticks(TARGETS)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(color="#EEEEEE", lw=0.7)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Steer generation to any target logP", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTFIG / "property_dial.pdf")
    fig.savefig(OUTFIG / "property_dial.png", dpi=200)
    print("wrote property_dial.{pdf,png}")


if __name__ == "__main__":
    out = run_sweep()
    make_figure(out)
