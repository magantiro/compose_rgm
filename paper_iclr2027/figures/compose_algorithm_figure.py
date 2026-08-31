#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPOSE — algorithmic / architecture figure.

Renders a single self-contained publication figure describing COMPOSE as control
of a learned stochastic process over executable graph rewrites, organized around
the separation of three layers:

    EXECUTOR        A(x)                what is POSSIBLE
    REFERENCE LAW   R_theta             what is PLAUSIBLE
    CONTROLLER      P_phi via h_phi     what is PURPOSEFUL

Outputs
    compose_algorithm_figure.png   (400 dpi)
    compose_algorithm_figure.pdf   (vector, TrueType-embedded text)

Everything is vector geometry + text; no raster art, no external assets.

Run:
    python3 compose_algorithm_figure.py
"""

from __future__ import annotations

import math
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

# --------------------------------------------------------------------------------------
# Typography
# --------------------------------------------------------------------------------------

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 9,
        "savefig.transparent": False,
        "axes.linewidth": 0.0,
    }
)

MONO = ["Menlo", "Andale Mono", "Courier New", "DejaVu Sans Mono"]

# --------------------------------------------------------------------------------------
# Palette — "cold ink".
#
# Cool pale-slate ground, graphite structure, one signal color (crimson) reserved
# strictly for the goal-aware control layer.  The value ramp neutral -> steel ->
# crimson carries the thesis: possible -> plausible -> purposeful.  Deliberately
# unlike the warm cream / teal / orange reference scheme.
# --------------------------------------------------------------------------------------

PAPER = "#EBEFF3"        # cool pale-slate ground
WASH = "#FBFCFD"         # centre wash
CARD = "#F7F9FB"         # spec-card fill
INK = "#141A23"          # near-black blue — capsules
INK_TXT = "#E8EDF3"      # text on ink
INK_TXT2 = "#96A4B5"     # secondary text on ink

TXT = "#1D2531"          # primary text on paper
TXT2 = "#5B6A7B"         # secondary text
TXT3 = "#8794A3"         # tertiary / captions
HAIR = "#C2CCD6"         # hairlines

# layer 1 — executor (neutral graphite: what is merely possible)
G_STK = "#4E5F72"
G_FILL = "#DCE1E6"
G_MARK = "#75879B"

# layer 2 — reference law R_theta (steel blue: what is plausible)
S_STK = "#2A5D8C"
S_FILL = "#CDDBEB"
S_LT = "#7E9FBE"

# layer 3 — learned value h_phi + controller P_phi (crimson: what is purposeful)
C_STK = "#AF2149"
C_FILL = "#EFD9E0"
C_FILL_LT = "#F7EAEE"
C_LT = "#D98BA2"

DEAD = "#9AA6B3"         # guard-rejected marks / unreachable successors

# --------------------------------------------------------------------------------------
# Canvas geometry.  100 data units == 1 inch.
# --------------------------------------------------------------------------------------

W, H = 1700.0, 1096.0
FIGSIZE = (W / 100.0, H / 100.0)

FRAME_L, FRAME_R = 152.0, 1550.0
CARD_L, CARD_R = 152.0, 414.0
LANE_L, LANE_R = 424.0, 1550.0

TITLE_Y = 1082.0
HEAD_Y0, HEAD_Y1 = 988.0, 1068.0
AXIS_Y = 958.0
BUD_Y = 934.0
STATE_Y = 906.0

L1_Y0, L1_Y1 = 758.0, 890.0     # EXECUTOR
GA_Y0, GA_Y1 = 716.0, 758.0     # aggregation gap
L2_Y0, L2_Y1 = 612.0, 716.0     # R_theta
LH_Y0, LH_Y1 = 558.0, 602.0     # h_phi
L3_Y0, L3_Y1 = 444.0, 548.0     # P_phi
MS_Y0, MS_Y1 = 362.0, 424.0     # measurement band
PN_Y0, PN_Y1 = 176.0, 352.0     # bottom panels
NT_Y0, NT_Y1 = 100.0, 166.0     # notation strip
FT_Y0, FT_Y1 = 16.0, 88.0       # footer capsule

ANCHOR_CY = 0.5 * (L1_Y1 + L3_Y0)      # 667
L_ANCHOR = (80.0, ANCHOR_CY, 54.0)
R_ANCHOR = (1620.0, ANCHOR_CY, 54.0)

COLS = [560.0, 800.0, 1075.0, 1405.0]
ELLIPSES_X = [937.0, 1240.0]
STATE_LBL = [r"$x_0$", r"$x_1$", r"$x_k$", r"$x_K$"]
BUDGET_LBL = [r"$b=K$", r"$b=K-1$", r"$b=K-k$", r"$b=0$"]

GLYPH_CY = 845.0
MARK_CY = 782.0
BUB2_CY = 680.0
HCHIP_CY = 580.0
BUB3_CY = 512.0
SLOT_LBL_Y = 634.0

ZOOM_X0, ZOOM_X1 = 956.0, 1208.0       # emphasised column (x_k)

# --------------------------------------------------------------------------------------
# Illustrative structure of the successor sets.  These encode the *shape* of the
# mechanism only (bubble area is proportional to probability mass); no numeric value
# from these arrays is ever printed in the figure.
# --------------------------------------------------------------------------------------

SLOT_OFF_4 = [-66.0, -22.0, 22.0, 66.0]
SLOT_OFF_5 = [-92.0, -46.0, 0.0, 46.0, 106.0]    # 4 real successors + 1 unreachable

R_MASS = [
    [0.34, 0.14, 0.29, 0.23],
    [0.22, 0.38, 0.12, 0.28],
    [0.20, 0.34, 0.11, 0.35],
]
H_VAL = [
    [0.30, 0.75, 0.45, 0.95],
    [0.85, 0.25, 0.55, 0.40],
    [0.35, 0.30, 0.90, 0.72],
]

MARK_TYPES = [r"$+$", r"$0$", r"$+$", r"$-$", r"$0$"]   # birth / same / death
MARK_TO_SLOT = [0, 1, 1, 2, 3]                          # many marks -> one canonical y

BUB_RMAX = 24.0


def controlled(r_mass, h_val):
    """P_phi mass: R_theta reweighted by h_phi, renormalized over S(x)."""
    w = [r * h for r, h in zip(r_mass, h_val)]
    tot = sum(w)
    return [v / tot for v in w]


# --------------------------------------------------------------------------------------
# Molecular state glyphs.  Every state is a COMPLETE graph: all atoms present, all
# bonds drawn, nothing masked or partially specified.  Heavy-atom count grows by at
# most one per elementary edit; the emphasised ring system in x_K is what the
# temporally extended action BUILD_RING_SYSTEM installs.
# --------------------------------------------------------------------------------------


def ring(cx, cy, r, n, phase_deg):
    return [
        (
            cx + r * math.cos(math.radians(phase_deg + 360.0 * i / n)),
            cy + r * math.sin(math.radians(phase_deg + 360.0 * i / n)),
        )
        for i in range(n)
    ]


def _mol_x0():
    nodes = [(-1.05, 0.10), (-0.42, -0.28), (0.21, 0.10), (0.84, -0.28), (0.21, 0.83)]
    edges = [(0, 1), (1, 2), (2, 3), (2, 4)]
    return nodes, edges, [], []


def _mol_x1():
    nodes, edges, _, _ = _mol_x0()
    return nodes + [(1.47, 0.10)], edges + [(3, 5)], [5], []


def _mol_xk():
    r1 = ring(-0.45, 0.10, 0.52, 6, 90.0)
    chain = [(0.55, 0.42), (1.12, 0.08), (1.69, 0.42), (1.69, 1.05)]
    edges = [(i, (i + 1) % 6) for i in range(6)] + [(5, 6), (6, 7), (7, 8), (8, 9)]
    return r1 + chain, edges, [9], [(-0.45, 0.10, 0.30)]


def _mol_xK():
    r1 = ring(-1.00, 0.05, 0.50, 6, 90.0)
    r2 = ring(0.42, 0.10, 0.46, 5, 162.0)
    chain = [(1.42, 0.50), (1.95, 0.15), (1.95, -0.48)]
    edges = [(i, (i + 1) % 6) for i in range(6)]
    edges += [(6 + i, 6 + (i + 1) % 5) for i in range(5)]
    edges += [(5, 6), (9, 11), (11, 12), (12, 13)]
    return r1 + r2 + chain, edges, [6, 7, 8, 9, 10], [(-1.00, 0.05, 0.29)]


MOLS = [_mol_x0(), _mol_x1(), _mol_xk(), _mol_xK()]


def mol_half_width(mol, s):
    xs = [p[0] for p in mol[0]]
    return 0.5 * (max(xs) - min(xs)) * s


def draw_mol(ax, mol, cx, cy, s, node_r=4.0, lw=2.0, z=6, hi_color=C_STK):
    nodes, edges, hi, arom = mol
    xs = [p[0] for p in nodes]
    ys = [p[1] for p in nodes]
    ox = cx - 0.5 * (max(xs) + min(xs)) * s
    oy = cy - 0.5 * (max(ys) + min(ys)) * s

    def T(p):
        return (ox + p[0] * s, oy + p[1] * s)

    single = len(hi) == 1
    for (a, b) in edges:
        pa, pb = T(nodes[a]), T(nodes[b])
        both_hi = a in hi and b in hi
        ax.plot(
            [pa[0], pb[0]], [pa[1], pb[1]],
            color=hi_color if both_hi else G_STK, lw=lw,
            solid_capstyle="round", zorder=z + (1 if both_hi else 0),
        )
    for (acx, acy, ar) in arom:
        p = T((acx, acy))
        ax.add_patch(
            Circle(p, ar * s, fill=False, ec=G_STK, lw=lw * 0.62, alpha=0.55, zorder=z)
        )
    for i, p in enumerate(nodes):
        q = T(p)
        if i in hi:
            if single:
                ax.add_patch(
                    Circle(q, node_r * 1.65, facecolor="none", ec=hi_color,
                           lw=lw * 0.70, zorder=z + 2)
                )
            ax.add_patch(Circle(q, node_r, facecolor=hi_color, ec="none", zorder=z + 3))
        else:
            ax.add_patch(Circle(q, node_r, facecolor=G_STK, ec="none", zorder=z + 1))


# --------------------------------------------------------------------------------------
# Drawing primitives
# --------------------------------------------------------------------------------------


def rrect(ax, x0, y0, x1, y1, r, fc, ec="none", lw=0.0, z=1, alpha=1.0, ls="solid"):
    p = FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}",
        mutation_scale=1.0, facecolor=fc, edgecolor=ec, linewidth=lw,
        linestyle=ls, zorder=z, alpha=alpha,
    )
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, size=8.0, color=TXT, ha="left", va="center", weight="normal",
        style="normal", z=10, family=None, alpha=1.0, rotation=0):
    kw = {}
    if family is not None:
        kw["fontfamily"] = family
    return ax.text(
        x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight,
        fontstyle=style, zorder=z, alpha=alpha, rotation=rotation,
        linespacing=1.35, **kw,
    )


def measure(fig, ax, artist):
    """Width of a drawn text artist, in data units."""
    r = fig.canvas.get_renderer()
    bb = artist.get_window_extent(renderer=r)
    inv = ax.transData.inverted()
    (x0, _), (x1, _) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
    return x1 - x0


def heading(fig, ax, x, y, label, symbol, size, sym_color, gap=13.0):
    """A bold heading followed by its accent-colored symbol, laid out by measurement."""
    t = txt(ax, x, y, label, size=size, color=TXT, weight="bold")
    txt(ax, x + measure(fig, ax, t) + gap, y, symbol, size=size, color=sym_color,
        weight="bold")


def arrow(ax, p0, p1, color, lw=1.0, z=8, rad=0.0, head=6.0, ls="solid", alpha=1.0):
    a = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=head, lw=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z,
        linestyle=ls, alpha=alpha, joinstyle="round", capstyle="round",
    )
    ax.add_patch(a)
    return a


def halo(ax, cx, cy, r, spread=1.95, n=30, color="#FFFFFF", amax=0.62, z=2):
    a_each = 1.0 - (1.0 - amax) ** (1.0 / n)
    for i in range(n):
        f = i / (n - 1.0)
        rr = r * spread - (r * spread - r) * f
        ax.add_patch(Circle((cx, cy), rr, facecolor=color, ec="none",
                            alpha=a_each, zorder=z))


def cross(ax, cx, cy, s, color=DEAD, lw=1.1, z=12):
    ax.plot([cx - s, cx + s], [cy - s, cy + s], color=color, lw=lw,
            solid_capstyle="round", zorder=z)
    ax.plot([cx - s, cx + s], [cy + s, cy - s], color=color, lw=lw,
            solid_capstyle="round", zorder=z)


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------


def build():
    fig = plt.figure(figsize=FIGSIZE, dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- ground -----------------------------------------------------------------
    ax.add_patch(Rectangle((0, 0), W, H, facecolor=PAPER, ec="none", zorder=0))
    for i in range(34):
        f = i / 33.0
        ax.add_patch(
            Ellipse((905, 640), 2300 - 1500 * f, 1560 - 1000 * f,
                    facecolor=WASH, ec="none", alpha=0.013, zorder=0)
        )

    # ---- title -------------------------------------------------------------------
    t = txt(ax, FRAME_L, TITLE_Y, "COMPOSE", size=13.0, color=TXT, weight="bold")
    txt(ax, FRAME_L + measure(fig, ax, t) + 16, TITLE_Y,
        "control of a learned stochastic process over executable graph rewrites",
        size=11.0, color=TXT2)
    txt(ax, FRAME_R, TITLE_Y,
        "three separable layers:   possible  ·  plausible  ·  purposeful",
        size=9.2, color=TXT3, ha="right")

    # ---- header capsule: the controlled kernel -----------------------------------
    rrect(ax, FRAME_L, HEAD_Y0, FRAME_R, HEAD_Y1, 24, INK, z=3)
    txt(ax, FRAME_L + 26, HEAD_Y1 - 15, "CONTROLLED KERNEL", size=7.6,
        color=INK_TXT2)
    txt(
        ax, 0.5 * (FRAME_L + FRAME_R), 0.5 * (HEAD_Y0 + HEAD_Y1) - 2,
        r"$P_\phi(y\mid x,z,b)\;=\;"
        r"\frac{R_\theta(y\mid x)\;h_\phi(y,z,b-1)}"
        r"{\sum_{y'\in S(x)}R_\theta(y'\mid x)\;h_\phi(y',z,b-1)}$",
        size=17.5, color=INK_TXT, ha="center", z=11,
    )

    # ---- trajectory axis ---------------------------------------------------------
    ax.plot([LANE_L + 16, LANE_R - 10], [AXIS_Y, AXIS_Y], color=HAIR, lw=0.9, zorder=4)
    for c in COLS:
        ax.plot([c, c], [AXIS_Y, AXIS_Y - 9], color=HAIR, lw=0.9, zorder=4)
        idx = COLS.index(c)
        txt(ax, c, BUD_Y, BUDGET_LBL[idx], size=9.5, color=TXT2, ha="center")
        txt(ax, c, STATE_Y, STATE_LBL[idx], size=15.0, color=TXT, ha="center")
    for ex in ELLIPSES_X:
        txt(ax, ex, STATE_Y - 2, r"$\cdots$", size=13.0, color=TXT3, ha="center")
    txt(ax, CARD_R - 12, BUD_Y, "remaining budget", size=8.2, color=TXT3, ha="right")
    txt(ax, CARD_R - 12, STATE_Y, "state", size=8.2, color=TXT3, ha="right")

    # ---- lane bands + spec cards -------------------------------------------------
    for (y0, y1, fill, stk) in [
        (L1_Y0, L1_Y1, G_FILL, G_STK),
        (L2_Y0, L2_Y1, S_FILL, S_STK),
        (LH_Y0, LH_Y1, C_FILL_LT, C_STK),
        (L3_Y0, L3_Y1, C_FILL, C_STK),
    ]:
        rrect(ax, LANE_L, y0, LANE_R, y1, 9, fill, z=3)
        rrect(ax, CARD_L, y0, CARD_R, y1, 9, CARD, ec=HAIR, lw=0.7, z=3)
        ax.add_patch(Rectangle((CARD_L + 1.0, y0 + 6), 4.0, (y1 - y0) - 12,
                               facecolor=stk, ec="none", zorder=5))

    cx0 = CARD_L + 18

    # card 1 — executor
    txt(ax, cx0, 872, "EXECUTOR", size=12.5, color=TXT, weight="bold")
    txt(ax, cx0, 853, "defines what is POSSIBLE", size=9.3, color=G_STK, style="italic")
    txt(ax, cx0, 832, r"$A(x)=\{\,a\;:\;a$ matches $x$, and", size=8.5, color=TXT)
    txt(ax, cx0, 819, r"$T(x,a)$ passes all executor guards$\,\}$", size=8.5, color=TXT)
    txt(ax, cx0, 801, "guards — declared vocabularies, valences,", size=7.7, color=TXT2)
    txt(ax, cx0, 790, "graph connectivity, heavy-atom range", size=7.7, color=TXT2)
    txt(ax, cx0, 777, "edits move heavy-atom count by at most one:", size=7.7,
        color=TXT2)
    txt(ax, cx0, 766, r"birth $+$,   same-cardinality $0$,   death $-$", size=7.7,
        color=TXT2)

    # card 2 — reference law
    heading(fig, ax, cx0, 704, "REFERENCE LAW", r"$R_\theta$", 12.5, S_STK)
    txt(ax, cx0, 686, "defines what is PLAUSIBLE", size=9.3, color=S_STK,
        style="italic")
    txt(ax, cx0, 664,
        r"$R_\theta(y\mid x)\;\propto\!\!\sum_{a\,:\,T(x,a)\,\cong\,y}\!\!"
        r"p_\theta(a\mid x)$",
        size=8.6, color=TXT)
    txt(ax, cx0, 640, r"normalized over $S(x)$: the distinct,", size=7.7, color=TXT2)
    txt(ax, cx0, 629, r"non-self canonical successors $y_1\dots y_m$", size=7.7,
        color=TXT2)
    txt(ax, cx0, 618, r"goal-independent — receives no $z$, no $b$", size=7.7,
        color=S_STK)

    # card 3 — learned value
    heading(fig, ax, cx0, 592, "VALUE", r"$h_\phi$", 11.0, C_STK)
    txt(ax, cx0, 577, r"learned backward value:   $P_\phi\;\propto\;R_\theta\times "
                      r"h_\phi$", size=7.9, color=TXT2)
    txt(ax, cx0, 566, r"renormalized over $S(x)$", size=7.7, color=TXT2)

    # card 4 — controller
    heading(fig, ax, cx0, 534, "CONTROLLER", r"$P_\phi$", 12.5, C_STK)
    txt(ax, cx0, 516, "defines what is PURPOSEFUL", size=9.3, color=C_STK,
        style="italic")
    txt(ax, cx0, 496, r"reweights only $y\in S(x)$: control redistributes",
        size=8.0, color=TXT)
    txt(ax, cx0, 485, "probability over reachable futures; it cannot",
        size=8.0, color=TXT)
    txt(ax, cx0, 474, "create a transition the executor does not contain.",
        size=8.0, color=TXT)
    txt(ax, cx0, 458, r"draw $x_{k+1}\sim P_\phi(\cdot\mid x_k,z,b)$", size=8.2,
        color=C_STK)

    # ---- emphasised column bracket at x_k ---------------------------------------
    rrect(ax, ZOOM_X0, L3_Y0 - 6, ZOOM_X1, L1_Y1 + 4, 10, INK, z=4, alpha=0.038)
    rrect(ax, ZOOM_X0, L3_Y0 - 6, ZOOM_X1, L1_Y1 + 4, 10, "none", ec=S_LT, lw=0.8,
          z=4, alpha=0.85, ls=(0, (4, 4)))

    # ---- lane 1: complete states + legal edit sets -------------------------------
    txt(ax, LANE_L + 14, 878, "every state is a complete molecular graph",
        size=7.4, color=TXT3, style="italic")

    gs = 30.0
    half_w = [mol_half_width(m, gs) for m in MOLS]
    for i, c in enumerate(COLS):
        draw_mol(ax, MOLS[i], c, GLYPH_CY, gs, node_r=4.0, lw=2.0, z=6)

    for i in range(3):
        x0 = COLS[i] + half_w[i] + 12
        x1 = COLS[i + 1] - half_w[i + 1] - 12
        dashed = i >= 1
        arrow(ax, (x0, GLYPH_CY), (x1, GLYPH_CY), C_STK, lw=1.3, z=9, head=6.5,
              ls=(0, (4, 3)) if dashed else "solid", alpha=0.9)
        if dashed:
            xm = 0.5 * (x0 + x1)
            ax.add_patch(Rectangle((xm - 13, GLYPH_CY - 8), 26, 16,
                                   facecolor=G_FILL, ec="none", zorder=9))
            txt(ax, xm, GLYPH_CY - 1, r"$\cdots$", size=11.0, color=TXT3,
                ha="center", z=10)
    txt(ax, 0.5 * (COLS[0] + COLS[1]), GLYPH_CY + 15, r"$\sim P_\phi$", size=8.2,
        color=C_STK, ha="center")

    for i, c in enumerate(COLS[:3]):
        txt(ax, c, 803, r"$A(%s)$" % ["x_0", "x_1", "x_k"][i], size=8.0, color=TXT2,
            ha="center")
        for j in range(7):
            mx = c + (j - 3) * 20.0
            if j < 5:
                rrect(ax, mx - 7, MARK_CY - 7, mx + 7, MARK_CY + 7, 2.6, G_MARK, z=6)
                txt(ax, mx, MARK_CY, MARK_TYPES[j], size=6.6, color="#FFFFFF",
                    ha="center", z=7)
            else:
                rrect(ax, mx - 7, MARK_CY - 7, mx + 7, MARK_CY + 7, 2.6, "none",
                      ec=DEAD, lw=0.9, z=6, ls=(0, (2, 2)))
                cross(ax, mx, MARK_CY, 3.2, color=DEAD, lw=0.9, z=7)

    # terminal column: budget exhausted
    rrect(ax, COLS[3] - 84, 640, COLS[3] + 84, 716, 8, "none", ec=S_LT, lw=0.9, z=6,
          ls=(0, (3, 3)), alpha=0.9)
    txt(ax, COLS[3], 688, r"$b=0$", size=10.0, color=TXT2, ha="center")
    txt(ax, COLS[3], 667, "no further transitions:", size=7.7, color=TXT3, ha="center")
    txt(ax, COLS[3], 656, r"$A(x_K)$ is not exercised", size=7.7, color=TXT3,
        ha="center")
    ax.plot([COLS[3] - 9, COLS[3] + 9], [HCHIP_CY, HCHIP_CY], color=DEAD, lw=1.1,
            solid_capstyle="round", zorder=7)

    # ---- aggregation gap ---------------------------------------------------------
    txt(ax, CARD_R - 12, 746, r"push each mark through $T(x,\cdot)$", size=7.8,
        color=TXT2, ha="right")
    txt(ax, CARD_R - 12, 733, "and aggregate by canonical outcome", size=7.8,
        color=TXT2, ha="right")

    rrect(ax, 1247, 741, 1261, 755, 2.6, G_MARK, z=6)
    txt(ax, 1270, 748, r"a mark $a\in A(x)$", size=7.4, color=TXT2)
    rrect(ax, 1247, 723, 1261, 737, 2.6, "none", ec=DEAD, lw=0.9, z=6, ls=(0, (2, 2)))
    cross(ax, 1254, 730, 3.2, color=DEAD, lw=0.9, z=7)
    txt(ax, 1270, 730, "a candidate mark rejected by a guard", size=7.4, color=TXT3)

    # ---- lanes 2 / h / 3 ---------------------------------------------------------
    for i, c in enumerate(COLS[:3]):
        offs = SLOT_OFF_5 if i == 2 else SLOT_OFF_4
        r_mass, h_val = R_MASS[i], H_VAL[i]
        p_mass = controlled(r_mass, h_val)
        rmax, pmax = max(r_mass), max(p_mass)
        sel = p_mass.index(pmax)

        slot_x = [c + o for o in offs]
        r_rad = [BUB_RMAX * math.sqrt(m / rmax) for m in r_mass]
        p_rad = [BUB_RMAX * math.sqrt(m / pmax) for m in p_mass]

        for j in range(len(r_mass)):
            sx = slot_x[j]
            ax.plot([sx, sx], [BUB2_CY - r_rad[j] - 3, HCHIP_CY + 16], color=S_LT,
                    lw=0.7, ls=(0, (1.6, 2.2)), zorder=5)
            ax.plot([sx, sx], [HCHIP_CY - 16, BUB3_CY + p_rad[j] + 3], color=C_LT,
                    lw=0.7, ls=(0, (1.6, 2.2)), zorder=5)

        for j, slot in enumerate(MARK_TO_SLOT):
            mx = c + (j - 3) * 20.0
            sx = slot_x[slot]
            arrow(ax, (mx, MARK_CY - 9), (sx, BUB2_CY + r_rad[slot] + 4), S_LT,
                  lw=0.85, z=7, head=4.6, alpha=0.75, rad=0.12 if sx > mx else -0.12)

        for j in range(len(r_mass)):
            ax.add_patch(Circle((slot_x[j], BUB2_CY), r_rad[j], facecolor=S_STK,
                                ec="#FFFFFF", lw=1.0, zorder=8))
            txt(ax, slot_x[j], SLOT_LBL_Y, r"$y_%d$" % (j + 1), size=7.8, color=TXT2,
                ha="center")

        for j in range(len(h_val)):
            sx = slot_x[j]
            rrect(ax, sx - 11, HCHIP_CY - 13, sx + 11, HCHIP_CY + 13, 3.0, "#FFFFFF",
                  ec=C_LT, lw=0.9, z=7)
            rrect(ax, sx - 9.5, HCHIP_CY - 11.5, sx + 9.5,
                  HCHIP_CY - 11.5 + 24.0 * h_val[j], 2.2, C_STK, z=8, alpha=0.9)

        for j in range(len(p_mass)):
            ax.add_patch(Circle((slot_x[j], BUB3_CY), p_rad[j], facecolor=C_STK,
                                ec="#FFFFFF", lw=1.0, zorder=8))
        ax.add_patch(Circle((slot_x[sel], BUB3_CY), p_rad[sel] + 7.0, facecolor="none",
                            ec=C_STK, lw=1.4, zorder=9))

        if i == 2:
            gx = slot_x[4]
            for cy_ in (BUB2_CY, BUB3_CY):
                ax.add_patch(Circle((gx, cy_), 15.0, facecolor="none", ec=DEAD, lw=1.0,
                                    ls=(0, (2.4, 2.4)), zorder=8))
                ax.plot([gx + 17, 1214], [cy_, cy_], color=DEAD, lw=0.7,
                        ls=(0, (1.6, 2.2)), zorder=6)
            cross(ax, gx, BUB3_CY, 7.0, color=C_STK, lw=1.5, z=9)
            txt(ax, gx, SLOT_LBL_Y, r"$y^{\ast}$", size=7.8, color=TXT3, ha="center")
            txt(ax, 1220, BUB2_CY + 7, r"$y^{\ast}\notin S(x_k)$", size=7.4, color=TXT2)
            txt(ax, 1220, BUB2_CY - 6, r"$R_\theta(y^{\ast}\mid x_k)=0$", size=7.4,
                color=TXT2)
            txt(ax, 1220, BUB3_CY + 7, r"$P_\phi(y^{\ast}\mid x_k,z,b)=0$", size=7.4,
                color=C_STK)
            txt(ax, 1220, BUB3_CY - 6, r"for every $\phi$", size=7.4, color=C_STK)
            txt(ax, slot_x[sel], BUB3_CY - p_rad[sel] - 16, r"$x_{k+1}$", size=8.4,
                color=C_STK, ha="center")

    rrect(ax, COLS[3] - 62, BUB3_CY - 17, COLS[3] + 62, BUB3_CY + 17, 8, C_STK, z=7)
    txt(ax, COLS[3], BUB3_CY, r"$g_z(x_K)$", size=11.0, color="#FFFFFF", ha="center",
        z=8)
    txt(ax, COLS[3], BUB3_CY - 28, "terminal desirability", size=7.6, color=TXT2,
        ha="center")

    # ---- anchors -----------------------------------------------------------------
    lx, ly, lr = L_ANCHOR
    halo(ax, lx, ly, lr)
    ax.add_patch(Circle((lx, ly), lr, facecolor="#FDFEFE", ec=HAIR, lw=0.8, zorder=5))
    draw_mol(ax, MOLS[0], lx, ly, 19.0, node_r=2.9, lw=1.5, z=6)
    txt(ax, lx, ly - lr - 19, r"$x_0,\;z,\;b=K$", size=10.5, color=TXT, ha="center")
    txt(ax, lx, ly - lr - 36, "seed  ·  objective  ·  budget", size=7.6,
        color=TXT3, ha="center")

    rx, ry, rr_ = R_ANCHOR
    halo(ax, rx, ry, rr_)
    ax.add_patch(Circle((rx, ry), rr_, facecolor="#FDFEFE", ec=HAIR, lw=0.8, zorder=5))
    draw_mol(ax, MOLS[3], rx, ry, 15.5, node_r=2.6, lw=1.4, z=6)
    txt(ax, rx, ry - rr_ - 19, r"$x_K$", size=10.5, color=TXT, ha="center")
    txt(ax, rx, ry - rr_ - 36, "designed molecule out,", size=7.6, color=TXT3,
        ha="center")
    txt(ax, rx, ry - rr_ - 47, r"scored by $g_z\geq 0$", size=7.6, color=TXT3,
        ha="center")

    arrow(ax, (lx + lr + 8, ly), (CARD_L - 6, ly), TXT3, lw=1.0, z=6, head=6.0,
          alpha=0.75)
    arrow(ax, (LANE_R + 6, ly), (rx - rr_ - 8, ly), TXT3, lw=1.0, z=6, head=6.0,
          alpha=0.75)

    # ---- measurement band --------------------------------------------------------
    for (x0, x1, acc, tag, head, d1, d2) in [
        (FRAME_L, 640.0, S_STK, "MEASURED — REFERENCE LAW",
         r"$5.37 \rightarrow 3.90$ nats",
         "canonical-successor NLL against an empirical-family baseline, on 3,545 "
         "held-out transitions;",
         "86.7% of that gain comes from within-state successor identity, not global "
         "edit-family frequency."),
        (652.0, 1100.0, C_STK, "MEASURED — FUTURE-AWARE CONTROL",
         r"$40/65$    vs    $26/65$",
         "held-out targets recovered under future-aware control versus local greedy "
         "selection:",
         "+21.5 percentage points, [12.3, 33.5]; rescuing 14 of 39 local failures."),
        (1112.0, FRAME_R, C_STK, r"MEASURED — $h_\phi$ PRIORITIZATION",
         r"$34.6\%$",
         "of the continuation evaluations of all-candidate lookahead — the cost at "
         "which",
         r"$h_\phi$ prioritization retains that same recovery."),
    ]:
        rrect(ax, x0, MS_Y0, x1, MS_Y1, 7, CARD, ec=HAIR, lw=0.7, z=3)
        ax.add_patch(Rectangle((x0 + 1.0, MS_Y0 + 5), 4.0, (MS_Y1 - MS_Y0) - 10,
                               facecolor=acc, ec="none", zorder=5))
        txt(ax, x0 + 18, MS_Y1 - 10, tag, size=6.8, color=TXT3, weight="bold")
        txt(ax, x0 + 18, MS_Y1 - 26, head, size=12.0, color=acc, weight="bold")
        txt(ax, x0 + 18, MS_Y1 - 43, d1, size=7.5, color=TXT2)
        txt(ax, x0 + 18, MS_Y1 - 53, d2, size=7.5, color=TXT2)

    # ---- panel A: temporally extended actions ------------------------------------
    PA_L, PA_R = FRAME_L, 1000.0
    rrect(ax, PA_L, PN_Y0, PA_R, PN_Y1, 10, CARD, ec=HAIR, lw=0.7, z=3)
    pa = PA_L + 18
    txt(ax, pa, 340, "TEMPORALLY EXTENDED ACTION", size=8.6, color=TXT3, weight="bold")
    txt(ax, pa, 320, "BUILD_RING_SYSTEM(topology, size, stoichiometry, state)",
        size=11.0, color=TXT, family=MONO)
    txt(ax, pa, 305,
        "the controller learns factorized distributions over the four parameters:",
        size=7.8, color=TXT2)

    params = [
        ("topology", [r"$\in$ {linked, fused}"]),
        ("size", ["an integer over the", "executor-qualified range"]),
        ("stoichiometry", ["an exact count vector over ring",
                           "elements, summing to size"]),
        ("state", [r"$\in$ {aromatic, saturated}"]),
    ]
    pw = (PA_R - 18 - pa - 3 * 8) / 4.0
    for k, (name, lines) in enumerate(params):
        x0 = pa + k * (pw + 8)
        rrect(ax, x0, 256, x0 + pw, 294, 5, "#FFFFFF", ec=HAIR, lw=0.7, z=4)
        txt(ax, x0 + 9, 285, name, size=8.4, color=C_STK, weight="bold")
        for li, ln in enumerate(lines):
            txt(ax, x0 + 9, 271 - li * 11, ln, size=7.4, color=TXT2)

    txt(ax, pa, 246,
        r"the compiler enumerates concrete realizations $\xi$ (attachment site, "
        r"heteroatom position), which collapse onto distinct canonical endpoints "
        r"$y_\xi$:",
        size=7.8, color=TXT2)

    fan_cy = 203.0
    rrect(ax, pa, fan_cy - 11, pa + 92, fan_cy + 11, 5, INK, z=5)
    txt(ax, pa + 46, fan_cy, r"$a_{\mathrm{semantic}}$", size=9.0, color=INK_TXT,
        ha="center", z=6)
    xi_x, y_x = pa + 148, pa + 214
    xi_y = [fan_cy + (j - 1.5) * 13.0 for j in range(4)]
    ye_y = [fan_cy + (j - 1) * 16.0 for j in range(3)]
    for j, yy in enumerate(xi_y):
        arrow(ax, (pa + 94, fan_cy), (xi_x - 7, yy), G_MARK, lw=0.75, z=6, head=4.0,
              rad=0.16, alpha=0.85)
        rrect(ax, xi_x - 4.5, yy - 4.5, xi_x + 4.5, yy + 4.5, 2.0, G_MARK, z=6)
        arrow(ax, (xi_x + 6, yy), (y_x - 8, ye_y[[0, 0, 1, 2][j]]), S_LT, lw=0.75,
              z=6, head=4.0, rad=0.14, alpha=0.85)
    for yy in ye_y:
        ax.add_patch(Circle((y_x, yy), 6.0, facecolor=S_STK, ec="#FFFFFF", lw=0.8,
                            zorder=7))
    txt(ax, xi_x, fan_cy + 31, r"$\xi$", size=8.0, color=TXT2, ha="center")
    txt(ax, y_x, fan_cy + 31, r"$y_\xi$", size=8.0, color=TXT2, ha="center")

    txt(ax, y_x + 26, fan_cy + 11,
        r"$P(\xi\mid a_{\mathrm{semantic}},x,z,b)\;\propto\;"
        r"R_\theta(y_\xi\mid x)\,H(y_\xi,z,b)$", size=9.2, color=TXT)
    txt(ax, y_x + 26, fan_cy - 8,
        r"with $H=1$ when no realization-level value model is available",
        size=7.5, color=TXT2)

    rrect(ax, 806, fan_cy - 20, 982, fan_cy + 20, 6, C_FILL, ec=C_LT, lw=0.8, z=5)
    txt(ax, 894, fan_cy + 7, r"realizations $\xi$ are NOT", size=8.0, color=C_STK,
        ha="center", weight="bold")
    txt(ax, 894, fan_cy - 7, "controller actions", size=8.0, color=C_STK, ha="center",
        weight="bold")

    # ---- panel B: backward value -------------------------------------------------
    PB_L, PB_R = 1016.0, FRAME_R
    rrect(ax, PB_L, PN_Y0, PB_R, PN_Y1, 10, CARD, ec=HAIR, lw=0.7, z=3)
    pb = PB_L + 18
    txt(ax, pb, 340, "BACKWARD VALUE", size=8.6, color=TXT3, weight="bold")
    txt(ax, 0.5 * (PB_L + PB_R), 318,
        r"$h_b(x,z)\;=\;\mathrm{E}_{R_\theta}\!\left[\,g_z(X_b)\mid X_0=x\,\right]$",
        size=11.5, color=TXT, ha="center")
    txt(ax, 0.5 * (PB_L + PB_R), 297,
        r"$g_z\geq 0$ — terminal desirability for objective $z$", size=7.7,
        color=TXT2, ha="center")

    src = (pb + 26, 241.0)
    ax.add_patch(Circle(src, 6.5, facecolor=G_STK, ec="#FFFFFF", lw=0.9, zorder=7))
    txt(ax, src[0], src[1] + 18, r"$x$", size=8.4, color=TXT2, ha="center")
    gx0 = pb + 196
    gys = [210.0, 231.0, 252.0, 273.0]
    for gy in gys:
        arrow(ax, (src[0] + 8, src[1]), (gx0 - 4, gy), S_LT, lw=0.85, z=6, head=4.4,
              rad=0.20 if gy > src[1] else -0.20, alpha=0.85)
        rrect(ax, gx0, gy - 8, gx0 + 30, gy + 8, 3.0, S_FILL, ec=S_LT, lw=0.8, z=6)
        txt(ax, gx0 + 15, gy, r"$g_z$", size=7.6, color=S_STK, ha="center", z=7)
    hb_x0, hb_x1 = gx0 + 92, gx0 + 214
    for gy in gys:
        arrow(ax, (gx0 + 32, gy), (hb_x0 - 5, 241.0), C_LT, lw=0.85, z=6, head=4.4,
              rad=-0.16 if gy > 241 else 0.16, alpha=0.9)
    rrect(ax, hb_x0, 228, hb_x1, 254, 6, INK, z=6)
    txt(ax, 0.5 * (hb_x0 + hb_x1), 241, r"$h_b(x,z)$", size=10.0, color=INK_TXT,
        ha="center", z=7)
    txt(ax, 0.5 * (PB_L + PB_R), 189,
        r"$b$ rollout steps under $R_\theta$;$\;\;$ $h_\phi$ is a learned "
        r"approximation to $h_b$ at molecular scale",
        size=7.6, color=TXT2, ha="center")

    # ---- notation strip ----------------------------------------------------------
    rrect(ax, FRAME_L, NT_Y0, FRAME_R, NT_Y1, 8, CARD, ec=HAIR, lw=0.7, z=3)
    txt(ax, FRAME_L + 16, 152, "NOTATION", size=8.2, color=TXT3, weight="bold")
    for x0, lines in [
        (262.0, [
            r"$x$ — a state: a complete molecular graph, never masked",
            r"$a$ — a typed rewrite mark;   $T(x,a)$ — the graph after applying $a$",
            r"$\cong$ — equivalence under canonicalization",
            r"$k$ — step index;   $K$ — total budget;   $b=K-k$ — remaining budget",
        ]),
        (706.0, [
            r"$S(x)$ — distinct non-self canonical successors $y_1\dots y_m$ of $x$",
            r"$y^{\ast}$ — a molecule outside $S(x)$: no legal mark produces it",
            r"$z$ — the design objective;   $g_z\geq 0$ — terminal desirability",
            r"$\xi$ — a concrete realization;   $y_\xi$ — its canonical endpoint",
        ]),
        (1146.0, [
            r"$p_\theta(a\mid x)$ — the mark law, parameters $\theta$",
            r"$X_b$ — the state after $b$ further steps;   $\mathrm{E}_{R_\theta}$ — "
            r"expectation under $R_\theta$",
            r"$h_b$ — exact backward value;   $h_\phi$ — learned approximation, "
            r"parameters $\phi$",
            r"$H(y_\xi,z,b)$ — realization-level value;  $=1$ when none is available",
        ]),
    ]:
        for li, ln in enumerate(lines):
            txt(ax, x0, 152 - li * 13, ln, size=7.4, color=TXT2)

    # ---- footer capsule ----------------------------------------------------------
    rrect(ax, FRAME_L, FT_Y0, FRAME_R, FT_Y1, 22, INK, z=3)
    txt(ax, 0.5 * (FRAME_L + FRAME_R), 65,
        r"with exact backward values $h_b$,$\;$ $P_\phi$ is the finite-horizon Doob "
        r"$h$-transform of $R_\theta$", size=13.0, color=INK_TXT, ha="center", z=6)
    txt(ax, 0.5 * (FRAME_L + FRAME_R), 38,
        r"$h_\phi$ is a learned approximation at molecular scale; exact terminal "
        r"reweighting is a property of the exact transform and is not claimed "
        r"for $h_\phi$.", size=9.0, color=INK_TXT2, ha="center", z=6)

    return fig


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    fig = build()
    png = os.path.join(here, "compose_algorithm_figure.png")
    pdf = os.path.join(here, "compose_algorithm_figure.pdf")
    fig.savefig(png, dpi=400, facecolor=PAPER)
    fig.savefig(pdf, facecolor=PAPER)
    plt.close(fig)
    print("wrote", png)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
