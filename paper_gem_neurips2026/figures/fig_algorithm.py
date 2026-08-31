"""Algorithm figure: the COMPOSE transition kernel, drawn rigorously and boxless.

WHAT THIS FIGURE IS
-------------------
One molecular step, read left to right, showing every object the method defines
and every map between them:

    x  ->  p_theta on A(x)  ->  S(x)  ->  R_theta  ->  Doob tilt  ->  P^{z,b}

Boxless: no cards, no frames, no legend boxes, one hairline. Structure is column
alignment and whitespace. The only filled shapes are the bars.

WHY IT IS RIGOROUS
------------------
Every printed number is exact and the build asserts each identity, so the whole
step can be checked from the figure with a calculator:

    sum of p_theta over A(x)                    = 1.00
    the fiber G_{y3}(x)          3 x 0.06 = 0.18 = R_theta(y3)
    sum of R_theta over S(x)                    = 1.00
    h_b(x,z) = sum_y R_theta h_{b-1}            = 0.500 EXACTLY
    R_theta x (h_{b-1}/h_b) = P, ROW BY ROW, exact at 2 d.p.
    sum of P over S(x)                          = 1.00

The middle numeric column is the Doob factor h_{b-1}(y,z)/h_b(x,z) rather than
h_{b-1} itself. That is the quantity the algorithm actually multiplies by, it
makes the reweighting checkable row by row, and its pivot at 1.00 says directly
which successors are promoted and which are suppressed. h_b = 0.500 is stated
in the caption; every ratio here is h_{b-1}/0.500.

These are illustrative weights for one worked step, NOT measured results, chosen
so the arithmetic closes exactly. No measured quantity from the paper appears,
so the "no number outside a verified source" convention governing results is not
engaged.

THE CHEMISTRY IS REAL and generalises the paper's own worked example.
x = neopentyl alcohol has THREE symmetry-equivalent methyls, so deleting any one
gives the same canonical successor: a genuine 3-fold executable fiber. The figure
therefore SHOWS canonicalisation instead of asserting it -- the step fig1.tex's
drawing spec records as missing from the installed artwork ("the installed figure
draws x straight to the y_i"). Those three marks are symmetry-equivalent and so
carry EQUAL p_theta; splitting them unevenly would quietly claim the reference
model is not symmetry-respecting, which the paper nowhere asserts.

RESTRAINT IS DELIBERATE. Earlier drafts labelled every column with an italic role
("state", "edit marks", ...), tagged the equations possible/plausible/purposeful,
and carried rank, Delta-n and per-column sum annotations. All of it is cut. The
caption is the place for prose; a figure that captions its own every part reads
as decoration. What survives is what cannot be said in the caption: the fiber
geometry, the structures, and the three numeric columns.

SMILES are the source of truth and are never typeset into the figure
(house rule, cf. fig2.tex "NO SMILES IN THE FIGURE"):

    x   CC(C)(C)CO      2,2-dimethylpropan-1-ol         6 heavy
    y1  CCC(C)(C)CO     atom insert    dn = +1          7 heavy
    y2  CC(C)(C)CN      atom restate   dn =  0          6 heavy
    y3  CC(C)CO         atom delete    dn = -1          5 heavy   |G_y3(x)| = 3
    y4  CCC(C)(C)O      bond reroute   dn =  0          6 heavy
    y5  CC1(CO)CC1      cycle close    dn =  0          6 heavy

All |dn| <= 1, consistent with the audited executor families in
diagnostics/compose_delta_n_audit.json.

Drawn at 5.5 in = \\textwidth of gem_workshop.sty, 1:1, so point sizes here are
true point sizes on the page.

Output: figures/out/fig_algorithm.{pdf,png}
"""
import sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patheffects as pe

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                   pathlib.Path(__file__).parent / "out")
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------- design tokens
# Three type sizes, three greys, one accent. The accent traces a single thread --
# the fiber, its successor, and its reweighting -- and appears nowhere else.
INK    = "#1A1A1A"
GREY   = "#8E8E8E"
FAINT  = "#C6C6C6"
GHOST  = "#E7E7E7"
BAR    = "#7A7A7A"
ACCENT = "#9C3F26"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "text.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.transparent": False,
})

HEAD, VAL, SMALL = 8.6, 7.2, 6.2

# ================================================================= the numbers
LBL  = ["y_1", "y_2", "y_3", "y_4", "y_5"]
SMI  = ["CCC(C)(C)CO", "CC(C)(C)CN", "CC(C)CO", "CCC(C)(C)O", "CC1(CO)CC1"]
XSMI = "CC(C)(C)CO"
HOT  = 2                                   # the successor the fiber lands on

MARKS = [(0.31, 0), (0.23, 1), (0.06, 2), (0.06, 2), (0.06, 2), (0.17, 3), (0.11, 4)]
PTH = np.array([m[0] for m in MARKS])
OWN = [m[1] for m in MARKS]
assert abs(PTH.sum() - 1.0) < 1e-12, "p_theta must normalise over A(x)"

R = np.array([PTH[[i for i, o in enumerate(OWN) if o == k]].sum() for k in range(5)])
assert abs(R.sum() - 1.0) < 1e-12, "R_theta must normalise over S(x)"
assert abs(R[HOT] - 0.18) < 1e-12, "the 3-fold fiber must total 0.18"

H  = np.array([0.19, 0.52, 1.34, 0.44, 0.05])       # h_{b-1}(y_i, z)
HB = float(R @ H)                                    # h_b(x,z) = E_{R_theta}[h_{b-1}]
assert abs(HB - 0.5) < 1e-12, f"h_b must close at 0.500, got {HB}"

TILT = H / HB                                        # the Doob factor, as drawn
P = R * TILT
assert abs(P.sum() - 1.0) < 1e-12
assert abs(np.round(P, 2).sum() - 1.0) < 1e-12, "rounded P must still read 1.00"
# the figure's central claim: each row multiplies out, at the precision printed
assert np.allclose(np.round(np.round(R, 2) * np.round(TILT, 2), 2), np.round(P, 2)), \
    "R x tilt must equal P row by row at 2 d.p."
assert np.argmax(P) == HOT and np.argsort(-R).tolist().index(HOT) == 2

# =================================================================== geometry
FIGW, FIGH = 5.5, 2.70
ASP = FIGW / FIGH

X_MOL  = 0.048
X_MARK = 0.150
X_PVAL = 0.166
X_BRK  = 0.220
X_YLAB = 0.292
X_SUCC = 0.362
R_L, R_W = 0.470, 0.115
X_TILT = 0.678                       # right edge of the Doob-factor column
P_L, P_W = 0.760, 0.135

ROWS = np.linspace(0.820, 0.310, 5)
Y_HEAD, Y_RULE = 0.925, 0.882
Y_EQ = 0.125

fig = plt.figure(figsize=(FIGW, FIGH), dpi=400)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def txt(x, y, s, fs=VAL, c=INK, ha="center", va="center", z=6, **kw):
    return ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, zorder=z, **kw)


# =================================================================== molecules
def skeleton(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(smiles)
    AllChem.Compute2DCoords(m)
    c = m.GetConformer()
    pos = np.array([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y]
                    for i in range(m.GetNumAtoms())])
    pos -= (pos.min(0) + pos.max(0)) / 2          # centred, NOT normalised:
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble())
             for b in m.GetBonds()]
    het = {i: a.GetSymbol() for i, a in enumerate(m.GetAtoms())
           if a.GetAtomicNum() != 6}
    return pos, bonds, het


def draw_mol(cx, cy, s, data, lw=0.7, c=INK, fs=5.2, z=5):
    """Skeletal: carbons implicit, heteroatoms lettered."""
    pos, bonds, het = data
    u = s / GLOBAL
    Q = pos * np.array([u / ASP, u]) + np.array([cx, cy])
    for i, j, order in bonds:
        p, q = Q[i], Q[j]
        if order == 2:
            d = q - p
            n = np.array([-d[1] * ASP * ASP, d[0]])
            n = n / (np.hypot(*n) + 1e-12) * s * 0.06
            for k in (1, -1):
                ax.add_line(Line2D(*zip(p + n * k, q + n * k), color=c, lw=lw,
                                   zorder=z, solid_capstyle="round"))
        else:
            ax.add_line(Line2D(*zip(p, q), color=c, lw=lw, zorder=z,
                               solid_capstyle="round"))
    for i, sym in het.items():
        txt(Q[i, 0], Q[i, 1], sym, fs=fs, c=c, z=z + 2,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])


XMOL, YMOLS = skeleton(XSMI), [skeleton(s) for s in SMI]
# one scale across every structure, so atom count reads off the drawing and
# y_1 (7 heavy) is visibly larger than y_3 (5 heavy). This is what carries
# dn = +1 / 0 / -1 now that the Delta-n column is gone.
GLOBAL = max(np.abs(d[0]).max() for d in [XMOL] + YMOLS)

# ====================================================================== header
for cx, sym in [(X_MOL,               r"$x$"),
                (X_PVAL + 0.026,      r"$p_\theta(a\mid x)$"),
                (X_SUCC,              r"$\mathcal{S}(x)$"),
                (R_L + R_W / 2,       r"$R_\theta(y\mid x)$"),
                (X_TILT,             r"$h_{b-1}/h_b$"),
                (P_L + P_W / 2,       r"$P^{z,b}(y\mid x)$")]:
    txt(cx, Y_HEAD, sym, fs=HEAD)

ax.add_line(Line2D([0.012, 0.988], [Y_RULE] * 2, color=FAINT, lw=0.6, zorder=1))

# ==================================================================== the state
draw_mol(X_MOL, np.mean(ROWS), 0.056, XMOL, lw=0.7, fs=5.2)

# ================================================ marks, and the quotient onto S
# Marks sit on their own successor's row, so the only convergence drawn is the
# real one: the three symmetry-equivalent deletions collapsing onto y_3.
MY, SPAN = [], 0.048
for k in range(5):
    idx = [i for i, o in enumerate(OWN) if o == k]
    off = np.linspace(-SPAN, SPAN, len(idx)) if len(idx) > 1 else [0.0]
    MY.extend(ROWS[k] + d for d in off)

for i, (p, o) in enumerate(MARKS):
    hot = o == HOT
    c = ACCENT if hot else FAINT
    for a, b, rad, lw in [((X_MOL + 0.044, np.mean(ROWS)), (X_MARK - 0.010, MY[i]), 0.14, 0.5),
                          ((X_BRK + 0.008, MY[i]), (X_YLAB - 0.013, ROWS[o]), -0.12, 0.5)]:
        ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}",
                                     arrowstyle="-", lw=lw if not hot else lw + 0.1,
                                     color=c, alpha=1.0 if hot else 0.9,
                                     zorder=2, shrinkA=0, shrinkB=0))
    ax.add_line(Line2D([X_MARK - 0.007, X_MARK + 0.007], [MY[i]] * 2,
                       color=ACCENT if hot else GREY, lw=1.1, zorder=4,
                       solid_capstyle="round"))
    txt(X_PVAL, MY[i], f"{p:.2f}", fs=SMALL, c=INK if hot else GREY, ha="left")

fb = [MY[i] for i, o in enumerate(OWN) if o == HOT]
ax.add_line(Line2D([X_BRK] * 2, [min(fb), max(fb)], color=ACCENT, lw=0.5, zorder=3))
for yy in (min(fb), max(fb)):
    ax.add_line(Line2D([X_BRK - 0.005, X_BRK], [yy] * 2, color=ACCENT, lw=0.5, zorder=3))
txt(X_BRK - 0.002, max(fb) + 0.028, r"$\mathcal{G}_{y_3}\!(x)$", fs=SMALL, c=ACCENT)

# ================================================================= successors
for k in range(5):
    c = ACCENT if k == HOT else INK
    txt(X_YLAB, ROWS[k], rf"${LBL[k]}$", fs=VAL, c=c)
    draw_mol(X_SUCC, ROWS[k], 0.056, YMOLS[k], c=c)

# ============================================================ the two measures
def bars(x0, w, vals, ghost=None):
    for k, v in enumerate(vals):
        y, hot = ROWS[k], k == HOT
        if ghost is not None:
            ax.add_patch(Rectangle((x0, y - 0.0125), ghost[k] / P.max() * w, 0.025,
                                   fc=GHOST, ec="none", zorder=3))
        ax.add_patch(Rectangle((x0, y - 0.0085), v / P.max() * w, 0.017,
                               fc=ACCENT if hot else BAR, ec="none", zorder=4))
        txt(x0 + v / P.max() * w + 0.008, y, f"{v:.2f}", fs=VAL,
            c=INK if hot else GREY, ha="left")


bars(R_L, R_W, R)
bars(P_L, P_W, P, ghost=R)

for k, t in enumerate(TILT):
    txt(X_TILT, ROWS[k], f"{t:.2f}", fs=VAL,
        c=ACCENT if k == HOT else GREY)

# ==================================================================== equations
txt(0.253, Y_EQ,
    r"$R_\theta(y\mid x)\ \propto \sum_{a\in\mathcal{G}_y(x)} p_\theta(a\mid x)$",
    fs=HEAD - 0.4)
txt(0.723, Y_EQ,
    r"$P^{z,b}(y\mid x)\ =\ R_\theta(y\mid x)\;h_{b-1}(y,z)\,/\,h_b(x,z)$",
    fs=HEAD - 0.4)

for ext, kw in (("pdf", {}), ("png", dict(dpi=400))):
    fig.savefig(OUT / f"fig_algorithm.{ext}", facecolor="white", **kw)

print(f"  wrote {OUT/'fig_algorithm.pdf'}   {FIGW} x {FIGH} in")
print(f"  h_b = {HB:.4f}   R = {np.round(R,2).tolist()}")
print(f"  tilt = {np.round(TILT,2).tolist()}   P = {np.round(P,2).tolist()}")
