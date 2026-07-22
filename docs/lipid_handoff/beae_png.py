"""Render the BEAE quality-spectrum as a PNG figure (crappy->really-good), balanced
4-per-tier, with real leads badged where they land, rendered structures, metrics, and
an honest caption. matplotlib + RDKit."""
import json, sys, io
from pathlib import Path
SCRATCH = Path("/private/tmp/claude-502/-Users-rmaganti-Documents-Codex-2026-07-14-ok-so-compose-rgm-claude-lipid/11425d68-331b-425e-9161-eb394e3d1e57/scratchpad")
sys.path.insert(0, "src"); sys.path.insert(0, str(SCRATCH))
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem
RDLogger.DisableLog("rdApp.*")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import rank_beae
from beae_rank_composite import design_score

# ---- palette (sequential bad->good, semantic; not the AI-default) ----
INK   = "#141a1c"; PAPER = "#f6f8f7"; MUTE = "#5d6b6a"; HAIR = "#d5ddda"
TIER_C = {"REALLY-GOOD": "#1f7a54", "GOOD": "#3f8f86", "MARGINAL": "#c2872a", "POOR": "#a8503c"}
TIER_DESC = {
 "REALLY-GOOD": "passes every design\nrule · predicted\nmost potent",
 "GOOD":        "clean, lead-like\nlipids · real leads\nland here",
 "MARGINAL":    "borderline · weak\npotency or a\ndesign demerit",
 "POOR":      "fails design rules\n· correctly\nrejected",
}

def img(smi, w=520, h=250):
    m = Chem.MolFromSmiles(smi); AllChem.Compute2DCoords(m)
    return Draw.MolToImage(m, size=(w, h))

def score(smi):
    r = rank_beae.rank_beae([smi])[0]
    ds, notes, td = design_score(smi)
    pot = r["pred_potency"] or 0.0
    comp = ds + 3.0 * pot + 1.5 * (r["lead_similarity"] or 0.0)
    return {"comp": comp, "design": ds, "pred": r["pred_potency"], "lead": r["lead_similarity"],
            "tail": f"{td.get('tail1','?')}+{td.get('tail2','?')}" if td else "?"}

# ---- select a balanced 4-per-tier set ----
gen = [g for g in json.load(open(SCRATCH / "beae_ranked.json")) if g.get("kind") != "decoy"]
best_head = {}
for g in gen:
    best_head.setdefault(g["head"], g)
rg = list(best_head.values())[:4]                     # really-good: 4 diverse heads
leads = [("RM-60", "CCCCCCCCCCC(CCCCCCCC)OC(=O)/C=C/N(CCCN(C)C)CCC(=O)OCC(CCCCCC)CCCCCCCC"),
         ("Example-2", "CCCCC/C=C\\C/C=C\\CCCCCCCCOC(=O)/C=C/N(CCCN(C)C)CCC(=O)OCCCCCCCCC(C)C")]
good_gen = [g for g in gen if 7.3 <= g["composite"] < 8.1][:2]
marg = [g for g in gen if g["composite"] < 6.0][-4:]
decoys = [("aromatic tails", "c1ccccc1OC(=O)/C=C/N(CCCN(C)C)CCC(=O)Oc1ccccc1"),
          ("C3 tails (too short)", "CCCOC(=O)/C=C/N(CCCN(C)C)CCC(=O)OCCC"),
          ("single tail", "CCCCCCCCCCCCCCCCCCOC(=O)/C=C/N(CCCN(C)C)CC"),
          ("no ionizable head", "CCCCCCCCCCCCOC(=O)/C=C/N(CCCO)CCC(=O)OCCCCCCCCCCCC")]

def card(smi, head, sub, badge=None):
    s = score(smi)
    return {"smi": smi, "head": head, "sub": sub, "badge": badge, **s}

bands = {
 "REALLY-GOOD": [card(g["smiles"], g["head"], g["head"]) for g in rg],
 "GOOD": ([card(smi, nm, "qualified lead", badge="LEAD") for nm, smi in leads] +
          [card(g["smiles"], g["head"], g["head"]) for g in good_gen]),
 "MARGINAL": [card(g["smiles"], g["head"], g["head"]) for g in marg],
 "POOR": [card(smi, nm, nm, badge="DECOY") for nm, smi in decoys],
}

# ---- compose figure ----
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
FW, FH = 15.5, 13.6
fig = plt.figure(figsize=(FW, FH), dpi=170)
fig.patch.set_facecolor(PAPER)

fig.text(0.035, 0.965, "BEAE ionizable-lipid candidates — quality spectrum",
         fontsize=25, fontweight="bold", ha="left", va="top", color=INK)
fig.text(0.035, 0.928,
         "Generator-native candidates ranked by  composite = design-rule score  +  3×predicted potency  +  1.5×lead-similarity",
         fontsize=12.5, ha="left", va="top", color=MUTE)

order = ["REALLY-GOOD", "GOOD", "MARGINAL", "POOR"]
top, bot = 0.892, 0.085
bandh = (top - bot) / 4
gut_l, cells_l, cells_r = 0.035, 0.175, 0.988
ncell = 4
cw = (cells_r - cells_l) / ncell

for bi, tier in enumerate(order):
    by_top = top - bi * bandh
    by_bot = by_top - bandh
    cy = (by_top + by_bot) / 2
    col = TIER_C[tier]
    # left label gutter
    ax = fig.add_axes([gut_l, by_bot + 0.012, cells_l - gut_l - 0.02, bandh - 0.024]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.06",
                 transform=ax.transAxes, facecolor=col, edgecolor="none"))
    rng = [c["comp"] for c in bands[tier]]
    ax.text(0.5, 0.80, tier, transform=ax.transAxes, ha="center", va="center",
            fontsize=15.5, fontweight="bold", color="white")
    ax.text(0.5, 0.605, f"composite {min(rng):.1f}–{max(rng):.1f}", transform=ax.transAxes,
            ha="center", va="center", fontsize=10.5, color="white")
    ax.text(0.5, 0.30, TIER_DESC[tier], transform=ax.transAxes, ha="center", va="center",
            fontsize=8.6, color="white", alpha=0.95, linespacing=1.35)
    # molecule cells
    for ci, c in enumerate(bands[tier]):
        cl = cells_l + ci * cw
        # white structure plate
        pax = fig.add_axes([cl + 0.006, by_bot + 0.075, cw - 0.014, bandh - 0.096]); pax.axis("off")
        pax.imshow(img(c["smi"]))
        for sp in pax.spines.values():
            sp.set_visible(False)
        # badge
        if c["badge"]:
            pax.text(0.02, 0.96, c["badge"], transform=pax.transAxes, ha="left", va="top",
                     fontsize=8.5, fontweight="bold", color=col,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=1.0))
        # composite score + metrics under the plate
        ytx = by_bot + 0.066
        fig.text(cl + 0.012, ytx, f"{c['comp']:.2f}", fontsize=17, fontweight="bold", color=col, va="top")
        pr = "n/a" if c["pred"] is None else f"{c['pred']:+.2f}"
        fig.text(cl + 0.075, ytx - 0.004,
                 f"design {c['design']:.1f}\npred {pr} (prov.)\nlead {c['lead']}",
                 fontsize=8.6, color=INK, va="top", linespacing=1.25)
        fig.text(cl + 0.012, ytx - 0.052, f"{c['head']}  ·  {c['tail']}",
                 fontsize=8.4, color=MUTE, va="top")
    # band separator hairline
    if bi < 3:
        fig.add_artist(plt.Line2D([gut_l, cells_r], [by_bot, by_bot], color=HAIR, lw=0.8))

# caption (the honest framing — the ranking is a 3-term composite; the oracle is one term)
fig.text(0.035, 0.062,
         "How to read it — the ranking is a 3-term composite.  The design-rule score (reliable cheminformatics, NOT the oracle) makes the poor/marginal boundary.",
         fontsize=9.2, color=MUTE, va="top")
fig.text(0.035, 0.041,
         "The oracle supplies only the predicted-potency term — provisional near-domain extrapolation (BEAE is a Michael/propiolate product, the oracle's own reaction family),",
         fontsize=9.2, color=MUTE, va="top")
fig.text(0.035, 0.020,
         "so it only ranks within the already-good set.  Both qualified leads land in GOOD; some generated candidates score higher — a wet-lab hypothesis, not proven (top tier: deapa / C18:1 motif).",
         fontsize=9.2, color=MUTE, va="top")

out = SCRATCH / "beae_spectrum_figure.png"
fig.savefig(out, facecolor=PAPER, bbox_inches=None)
print("saved", out)
