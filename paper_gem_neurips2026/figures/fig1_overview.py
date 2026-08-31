"""Figure 1: Overview of COMPOSE.

Row positions and molecule geometries are computed ONCE and drawn in all three
panels, so panel invariance is structural rather than eyeballed.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
from matplotlib.patches import Circle
from composefig import style as S, prims as P, mol as M, layout as L

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

NY   = 5
YY   = L.rows(NY, top=0.720, bot=0.180)
MOLS = [M.graph(n, s) for n, s in zip((7, 6, 7, 6, 8), (11, 12, 13, 14, 15))]
XMOL = M.graph(8, 3)
RTHETA = np.array([0.31, 0.23, 0.16, 0.19, 0.11])
PPHI   = np.array([0.09, 0.19, 0.46, 0.19, 0.02])
BADGE  = ["+1", "-1", "0", "0", "ring"]
FUT    = ["weak", "moderate", "high", "moderate", "dead end"]
NFUT   = [5, 9, 18, 9, 2]
GLYPH  = ["+", "-", "~", "~", "~", "↔", "○"]   # add, delete, restate x3, reroute, ring
OWN    = [0, 1, 2, 2, 3, 4, 4]
MERGE  = {2, 3}

fig, ax = L.canvas()
cols, PY, PH = L.columns(ax, 3, ratios=[1.0, 1.0, 1.32])

for (px, pw), head, word, sub in zip(cols,
        ["(a)  Executable support", "(b)  Learned reference process", "(c)  Finite-horizon control"],
        ["POSSIBLE", "PLAUSIBLE", "PURPOSEFUL"],
        ["exact executor, state-dependent support", "learned, goal-independent",
         "goal- and budget-conditioned"]):
    P.panel_head(ax, px + pw/2, head, word, sub)

def source(px, pw, x=0.13):
    sx = px + pw*x
    P.glow(ax, sx, 0.455, 0.062, S.INDIGO, a0=0.10, z=0)
    M.draw(ax, sx, 0.455, 0.036, XMOL)
    ax.text(sx, 0.560, "$x$", ha="center", va="center", color=S.NAVY,
            fontsize=15, fontweight="bold")
    return sx

def tiles(px, pw, frac, w=0.082, h=0.100, sc=0.029, side="right"):
    xs = px + pw*frac
    for k in range(NY):
        P.card(ax, xs-w/2, YY[k]-h/2, w, h, fc=S.TILE, r=S.R["tile"], lw=0.9, z=1)
        M.draw(ax, xs, YY[k], sc, MOLS[k])
        dx, ha = (w/2+0.009, "left") if side == "right" else (-w/2-0.009, "right")
        ax.text(xs+dx, YY[k], f"$y_{k+1}$", ha=ha, va="center", color=S.NAVY, fontsize=10.5)
    return xs

# ------------------------------------------------------------------- panel (a)
px, pw = cols[0]
MX = px + pw*0.435
P.wedge(ax, (px+pw*0.18, 0.455), (px+pw*0.70, 0.790), (px+pw*0.70, 0.115), S.INDIGO, a0=0.024)
sx = source(px, pw); xs = tiles(px, pw, 0.735)
MY = np.array([0.720, 0.610, 0.500, 0.412, 0.300, 0.205, 0.125])
MMID=(MY[2]+MY[3])/2
P.glow(ax, MX, MMID, 0.052, S.ORCHID, a0=0.09, z=0)
from matplotlib.patches import FancyBboxPatch as _FBP
ax.add_patch(_FBP((MX-0.021, MY[3]-0.021), 0.042, (MY[2]-MY[3])+0.042,
    boxstyle="round,pad=0,rounding_size=0.020", fc="white", ec=S.ORCHID,
    lw=1.6, alpha=0.95, zorder=2))
for i, my in enumerate(MY):
    hot = i in MERGE
    P.chip(ax, MX, my, GLYPH[i], hot=hot)
    P.arrow(ax, (sx+0.052, 0.455), (MX-0.020, my), 0.8,
            S.ORCHID if hot else S.INDIGO, 0.70 if hot else 0.38,
            rad=0.13, ls=P.DASH, head=5)
for i, o in enumerate(OWN):
    if i in MERGE:
        continue                                  # the capsule speaks for both
    P.arrow(ax, (MX+0.018, MY[i]), (xs-0.047, YY[o]), 0.8, S.INDIGO, 0.24,
            rad=0.09, ls=P.DASH, head=5)
P.glow(ax, xs, YY[2], 0.062, S.ORCHID, a0=0.055, z=0)
P.arrow(ax, (MX+0.024, MMID), (xs-0.044, YY[2]), 2.4, S.ORCHID, 0.95, rad=0.0, head=8)
ax.text(MX, 0.800, "edit marks\n$\\mathcal{A}(x)$", ha="center", va="center",
        color=S.ORCHID, fontsize=9.5)
ax.text(xs, 0.800, "molecular successors\n$\\mathcal{S}(x)$", ha="center", va="center",
        color=S.NAVY, fontsize=9.5)
for k in range(NY):
    P.pill(ax, xs-0.040, YY[k]+0.029, BADGE[k])
ax.text(px+pw/2, 0.082, "two equivalent marks collapse to one successor",
        ha="center", va="center", color=S.ORCHID, fontsize=S.FS["small"])

# ------------------------------------------------------------------- panel (b)
px, pw = cols[1]
P.wedge(ax, (px+pw*0.20, 0.455), (px+pw*0.66, 0.780), (px+pw*0.66, 0.130), S.INDIGO, a0=0.026)
sx = source(px, pw); xs = tiles(px, pw, 0.735)
order = np.argsort(-RTHETA)
for k in range(NY):
    P.arrow(ax, (sx+0.053, 0.455+(YY[k]-0.455)*0.10), (xs-0.047, YY[k]),
            0.8 + RTHETA[k]*21, S.INDIGO, 0.94, rad=0.15-0.075*k)
    ax.text(sx+0.076, 0.455+(YY[k]-0.455)*0.80, f"{RTHETA[k]:.2f}",
            ha="center", va="center", color=S.NAVY, fontsize=S.FS["num"], zorder=7,
            bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="none", alpha=0.90))
P.gradient_bar(fig, ax, px+pw*0.22, 0.078, pw*0.56, 0.016, label="$R_\\theta(y \\mid x)$")

# ------------------------------------------------------------------- panel (c)
px, pw = cols[2]
P.eq_box(ax, px+0.010, 0.786, pw-0.020, 0.048,
         "$P_\\phi(y \\mid x,z,b)\\ \\propto\\ R_\\theta(y \\mid x)\\ h_\\phi(y,z,b-1)$", fs=10.0)
P.wedge(ax, (px+pw*0.17, 0.455), (px+pw*0.41, 0.760), (px+pw*0.41, 0.140), S.INDIGO, a0=0.026)
sx = source(px, pw, x=0.105); xs = tiles(px, pw, 0.475, w=0.078, h=0.096, sc=0.027)
hx, hy = px+pw*0.115, 0.672
for dx, lab in ((-0.031, "goal $z$"), (0.031, "budget $b$")):
    P.arrow(ax, (hx+dx, hy+0.074), (hx+dx*0.42, hy+0.030), 1.0, S.ORCHID, 0.9, ls=P.DASH, head=5)
    ax.text(hx+dx*1.30, hy+0.090, lab, ha="center", va="center",
            color=S.ORCHID, fontsize=S.FS["small"])
P.note_box(ax, hx, hy, "$h_\\phi$")
P.arrow(ax, (hx+0.019, hy-0.029), (xs-0.028, 0.575), 1.0, S.ORCHID, 0.45,
        ls=P.DASH, rad=-0.24, head=5)
P.glow(ax, xs, YY[2], 0.075, S.ORCHID, a0=0.075, z=0)            # the promoted successor
for k in range(NY):
    P.arrow(ax, (sx+0.053, 0.455+(YY[k]-0.455)*0.22), (xs-0.045, YY[k]),
            0.7 + PPHI[k]*21, S.INDIGO, 0.95, rad=0.085-0.0425*k)
    fx = xs + 0.076
    M.branch_tree(ax, fx, YY[k], NFUT[k], 40+k,
                  length=0.030, alpha=0.26 if k != 2 else 0.60)
    if k == 2: ax.plot(fx+0.037, YY[k], marker="*", ms=15, color=S.ORCHID, zorder=6)
    if k == 4: ax.plot(fx+0.013, YY[k], marker="x", ms=7, color="#B9B4C8", zorder=6, mew=1.8)
    ax.text(fx+0.019, YY[k]-0.036, FUT[k], ha="center", va="center", fontsize=S.FS["tiny"],
            color=S.NAVY if k == 2 else S.INK, alpha=1.0 if k == 2 else 0.55,
            fontweight="bold" if k == 2 else "normal")
P.legend_rows(ax, px+pw*0.07, 0.096,
              [(1.0, 0.30, "reference $R_\\theta$ (frozen)"), (3.6, 0.95, "control $P_\\phi$")])

fig.savefig(OUT/"fig1.pdf", facecolor=S.GROUND)
fig.savefig(OUT/"fig1.png", facecolor=S.GROUND, dpi=200)
fig.savefig(OUT/"fig1_hires.png", facecolor=S.GROUND, dpi=600)
print(f"  wrote {OUT}/fig1.pdf + fig1.png")
