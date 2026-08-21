"""Figure 1: Overview of COMPOSE.  Built on composefig.

The successor row positions and the molecule geometries are computed ONCE and
drawn in all three panels, so the panel invariance that carries the figure is
structural rather than something a human has to maintain by eye.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
from matplotlib.patches import Circle
from composefig import style as S, prims as P, mol as M, layout as L

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")

# ---- the invariant geometry, computed once --------------------------------
NY   = 5
YY   = L.rows(NY)
MOLS = [M.graph(n, s) for n, s in zip((7, 6, 7, 6, 8), (11, 12, 13, 14, 15))]
XMOL = M.graph(8, 3)
RTHETA = np.array([0.31, 0.23, 0.16, 0.19, 0.11])
PPHI   = np.array([0.10, 0.20, 0.44, 0.19, 0.02])     # y3 promoted by reachability
BADGE  = ["+1", "-1", "0", "0", "ring"]
FUT    = ["weak", "moderate", "high", "moderate", "dead end"]
NFUT   = [5, 9, 16, 9, 2]

fig, ax = L.canvas()
cols, PY, PH = L.columns(ax, 3)

for (px, pw), head, word, sub in zip(cols,
        ["(a)  Executable support", "(b)  Learned reference process", "(c)  Finite-horizon control"],
        ["POSSIBLE", "PLAUSIBLE", "PURPOSEFUL"],
        ["exact executor, state-dependent support", "learned, goal-independent",
         "goal- and budget-conditioned"]):
    P.panel_head(ax, px + pw/2, head, word, sub)

def source(px, pw):
    sx = px + pw*0.13
    ax.add_patch(Circle((sx, 0.455), 0.052, color=S.INDIGO, alpha=0.10, zorder=1, lw=0))
    M.draw(ax, sx, 0.455, 0.036, XMOL)
    ax.text(sx, 0.556, "$x$", ha="center", va="center", color=S.NAVY, fontsize=14, fontweight="bold")
    return sx

def tiles(px, pw, frac=0.70, w=0.084, h=0.104, sc=0.030, label_side="right"):
    xs = px + pw*frac
    for k in range(NY):
        P.card(ax, xs-w/2, YY[k]-h/2, w, h, fc=S.TILE, r=S.R["tile"], lw=0.9, z=1)
        M.draw(ax, xs, YY[k], sc, MOLS[k])
        dx, ha = (w/2+0.008, "left") if label_side == "right" else (-w/2-0.008, "right")
        ax.text(xs+dx, YY[k], f"$y_{k+1}$", ha=ha, va="center", color=S.NAVY, fontsize=10.5)
    return xs

# ---------------------------------------------------------------- panel (a)
px, pw = cols[0]; sx = source(px, pw); xs = tiles(px, pw)
MX = px + pw*0.42
#            marks:  y1   y2   y3a  y3b   y4   y5a  y5b
MY   = np.array([0.755, 0.640, 0.500, 0.412, 0.300, 0.190, 0.120])
OWN  = [0, 1, 2, 2, 3, 4, 4]
MERGE = {2, 3}
for i, my in enumerate(MY):
    hot = i in MERGE
    ax.add_patch(Circle((MX, my), 0.0092 if hot else 0.0062,
                        color=S.ORCHID if hot else S.MUTE, zorder=3, lw=0))
    P.arrow(ax, (sx+0.055, 0.455), (MX-0.012, my), 0.8,
            S.ORCHID if hot else S.INDIGO, 0.70 if hot else 0.28,
            rad=0.14, ls=P.DASH, head=5)
for i, o in enumerate(OWN):
    hot = i in MERGE
    P.arrow(ax, (MX+0.010, MY[i]), (xs-0.048, YY[o]), 1.6 if hot else 0.8,
            S.ORCHID if hot else S.INDIGO, 0.95 if hot else 0.28,
            rad=0.0 if hot else 0.10, ls="-" if hot else P.DASH, head=6 if hot else 5)
ax.text(MX, 0.815, "edit marks\n$\\mathcal{A}(x)$", ha="center", va="center",
        color=S.ORCHID, fontsize=9.5)
ax.text(xs, 0.815, "molecular successors\n$\\mathcal{S}(x)$", ha="center", va="center",
        color=S.NAVY, fontsize=9.5)
for k in range(NY):
    P.pill(ax, xs-0.041, YY[k]+0.030, BADGE[k])
ax.text(px+pw/2, 0.086, "two equivalent marks, one successor",
        ha="center", va="center", color=S.ORCHID, fontsize=S.FS["small"])

# ---------------------------------------------------------------- panel (b)
px, pw = cols[1]; sx = source(px, pw); xs = tiles(px, pw)
for k in range(NY):
    P.arrow(ax, (sx+0.056, 0.455), (xs-0.048, YY[k]),
            0.8 + RTHETA[k]*20, S.INDIGO, 0.92, rad=0.15-0.075*k)
    ax.text(sx+0.074, 0.455+(YY[k]-0.455)*0.80, f"{RTHETA[k]:.2f}",
            ha="center", va="center", color=S.NAVY, fontsize=S.FS["num"], zorder=7,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.88))
P.gradient_bar(fig, ax, px+pw*0.22, 0.075, pw*0.56, 0.016, label="$R_\\theta(y \\mid x)$")

# ---------------------------------------------------------------- panel (c)
px, pw = cols[2]
P.eq_box(ax, px+0.012, 0.775, pw-0.024, 0.052,
         "$P_\\phi(y \\mid x,z,b)\\ \\propto\\ R_\\theta(y \\mid x)\\ h_\\phi(y,z,b-1)$")
sx = source(px, pw); xs = tiles(px, pw, frac=0.50, w=0.080, h=0.100, sc=0.028)
hx, hy = px+pw*0.135, 0.672
P.note_box(ax, hx, hy, "$h_\\phi$")
for dx, lab in ((-0.030, "goal $z$"), (0.030, "budget $b$")):
    P.arrow(ax, (hx+dx, hy+0.082), (hx+dx*0.45, hy+0.030), 1.0, S.ORCHID, 0.9, ls=P.DASH, head=5)
    ax.text(hx+dx*1.35, hy+0.098, lab, ha="center", va="center", color=S.ORCHID, fontsize=S.FS["small"])
P.arrow(ax, (hx+0.020, hy-0.028), (xs-0.030, 0.585), 1.0, S.ORCHID, 0.5, ls=P.DASH, rad=-0.22, head=5)
for k in range(NY):
    P.arrow(ax, (sx+0.056, 0.455), (xs-0.046, YY[k]),
            0.7 + PPHI[k]*20, S.INDIGO, 0.95, rad=0.15-0.075*k)
    fx = xs + 0.080
    M.branch_tree(ax, fx, YY[k], NFUT[k], 40+k, alpha=0.30 if k != 2 else 0.62)
    if k == 2: ax.plot(fx+0.038, YY[k], marker="*", ms=14, color=S.ORCHID, zorder=6)
    if k == 4: ax.plot(fx+0.014, YY[k], marker="x", ms=7, color="#B9B4C8", zorder=6, mew=1.8)
    ax.text(fx+0.020, YY[k]-0.038, FUT[k], ha="center", va="center", fontsize=S.FS["tiny"],
            color=S.NAVY if k == 2 else S.INK, alpha=1.0 if k == 2 else 0.6,
            fontweight="bold" if k == 2 else "normal")
P.legend_rows(ax, px+pw*0.10, 0.098,
              [(1.0, 0.32, "reference $R_\\theta$ (frozen)"), (3.4, 0.95, "control $P_\\phi$")])

fig.savefig(OUT/"fig1.pdf", facecolor=S.GROUND)
fig.savefig(OUT/"fig1.png", facecolor=S.GROUND, dpi=200)
print(f"  wrote {OUT}/fig1.pdf (vector) + fig1.png")
