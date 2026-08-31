"""Figure 2 -- the empirical argument, in the aesthetic of fig_algorithm_tikz.py.

Same register as the algorithm figure: TikZ so the maths is real LaTeX in the
paper's Times, white ground, one hue family with a single warm accent, hairline
axes, no gridlines, no chart junk, no panel frames.

EVERY NUMBER IS READ FROM THE VERIFIED SOURCE AT BUILD TIME.
Nothing here is transcribed. The panels load diagnostics/*.json directly and the
script asserts each drawn quantity against it, so the figure cannot silently
drift from the record the way a hand-copied value can.

    (a) diagnostics/editing_v2_experiment1_reference_law_paper.json
    (c) diagnostics/editing_v2_sealed67_result.json   primary, n = 65
    (d) diagnostics/editing_v2_sealed67_result.json   primary, n = 65
    (e) diagnostics/retarget_heldout_confirmation.json   "Q2 value of history"

TWO THINGS THE SPEC GETS WRONG OR CANNOT SUPPORT. Both are surfaced rather than
papered over.

 1. fig2.tex gives similarity top-2 as (52.1, 56.9). The primary n = 65 arm
    measures 52.048 -> 52.0. The value 52.1 is the n = 67 SENSITIVITY arm
    (52.061). Every other coordinate in the spec is primary, so this one is
    mixing arms. We draw 52.0 and the assertion below pins it to the primary.

 2. PANEL (b) IS NOT DRAWN FROM BANKED DATA AND IS THEREFORE NOT DRAWN AS
    SPECIFIED. fig2.tex asks for the shared prefix, the divergent edit, and the
    intermediate structures of sealed pair 10, and notes the records "bank
    source/target/outcome but NOT the intermediate states". That is exactly what
    per_pair contains, so the intermediates do not exist to draw. Worse, the
    similarities the spec wants at the split (0.737 local vs 0.718 future-aware)
    appear NOWHERE in the record: the banked bests for pair 10 are greedy 0.7778
    (not recovered) and full 1.0 (recovered, 1 override).
    So panel (b) draws only what is banked -- the real source and target, the
    path length, the single override, and the two banked bests -- with the
    intermediate states shown as unlabelled nodes, because their count is known
    and their identity is not. Re-deriving pair 10's trajectory would let the
    panel be drawn as specified; until then, inventing those structures would be
    fabricating chemistry.

Output: figures/out/fig_results_{palette}.pdf  +  sections/fig_results.tex
"""
import os, json, pathlib, subprocess, sys
import numpy as np

HERE = pathlib.Path(__file__).parent
PAPER = HERE.parent
REPO = PAPER.parent
DIAG = REPO / "diagnostics"
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)

PALETTES = {
    "teal":   dict(navy="12464C", indigo="357C82", cyan="63AAA6", orchid="C2673A",
                   mute="BAD0CF", faint="DDE9E8", ink="1F383A"),
    "slate":  dict(navy="223047", indigo="4C6580", cyan="6E9BB0", orchid="B4762F",
                   mute="C2CBD5", faint="E1E6EC", ink="27313E"),
    "graphite": dict(navy="2B2B2B", indigo="5C5C5C", cyan="909090", orchid="A8452A",
                     mute="CBCBCB", faint="E6E6E6", ink="2B2B2B"),
    "forest": dict(navy="1E3B30", indigo="46705C", cyan="7BA88C", orchid="BE8B2B",
                   mute="C4D3C8", faint="E2EAE3", ink="223328"),
    "indigo": dict(navy="1B1B8F", indigo="5A5AC8", cyan="3FB8DC", orchid="BC5FD0",
                   mute="BFB6DA", faint="DFDAEE", ink="2A2A4A"),
}
PALETTE = os.environ.get("FIG_PALETTE", "teal")
COLORS = PALETTES[PALETTE]

# ============================================================ verified sources
REF = json.load(open(DIAG / "editing_v2_experiment1_reference_law_paper.json"))["overall"]
S67 = json.load(open(DIAG / "editing_v2_sealed67_result.json"))
RET = json.load(open(DIAG / "retarget_heldout_confirmation.json"))

# ---- (a) reference-law decomposition
A_KEYS = ["uniform_canonical", "empirical_family", "learned_family_uniform_id",
          "empirical_family_learned_id", "r_theta"]
A_LAB = ["uniform", "empirical\\\\family", "learned family\\\\uniform id",
         "empirical family\\\\learned id", "\\textsc{compose}"]
A_VAL = [REF[k] for k in A_KEYS]
A_GAIN = REF["empirical_family"] - REF["r_theta"]
A_IDSHARE = (REF["empirical_family"] - REF["empirical_family_learned_id"]) / A_GAIN
assert f"{A_GAIN:.2f}" == "1.46" and f"{A_IDSHARE*100:.1f}" == "86.7"

# ---- (c)/(d) sealed panel, PRIMARY arm (n = 65)
P, H1 = S67["primary"], S67["H1"]
N_PAIRS = P["_meta"]["pairs"]
C_LOCAL, C_FULL = P["greedy"]["recovered"], P["full"]["recovered"]
C_RESCUED, C_FAIL = H1["discordant_arm_only"], H1["base_failures"]
C_DIFF = H1["difference_in_rate"] * 100
C_CI = [v * 100 for v in H1["difference_ci95"]]
assert (N_PAIRS, C_LOCAL, C_FULL, C_RESCUED, C_FAIL) == (65, 26, 40, 14, 39)
assert f"{C_DIFF:.1f}" == "21.5" and [f"{v:.1f}" for v in C_CI] == ["12.3", "33.5"]

D_ARMS = [("greedy", "local", False), ("ref1", "$R_\\theta$ top-1", False),
          ("hphi1", "$h_\\phi$ top-1", False),
          ("sim2", "similarity top-2", True), ("full", "all-candidate", False)]
D_PTS = [(P[k]["fraction_of_universe_evaluated"] * 100, P[k]["rate"] * 100, lab, ti)
         for k, lab, ti in D_ARMS]
# the spec's 52.1 is the n=67 sensitivity arm; primary is 52.0. Pin the primary.
assert f"{P['sim2']['fraction_of_universe_evaluated']*100:.1f}" == "52.0"
assert f"{P['hphi1']['rate']*100:.1f}" == f"{P['full']['rate']*100:.1f}" == "61.5"

# ---- (e) dynamic retargeting: "Q2 value of history", verified_retarget vs restart
def q2(h):
    c = RET["by_history"][h]["comparisons"]["Q2 value of history"]
    assert c["arm"] == "verified_retarget" and c["base"] == "restart" and c["n"] == 40
    return c["mean_difference"], c["ci95"]
E_ROWS = [("potency-first", *q2("P")), ("developability-first", *q2("D"))]
assert f"{E_ROWS[0][1]:.3f}" == "0.302" and f"{E_ROWS[1][1]:.3f}" == "0.176"
E_N = RET["by_history"]["P"]["n"]

# ---- (b) what is actually banked for sealed pair 10
B = S67["per_pair"][10]
B_GREEDY, B_FULL = B["arms"]["greedy"], B["arms"]["full"]
assert B["index"] == 10 and B["verified_steps"] == 5
assert B_GREEDY["recovered"] is False and B_FULL["recovered"] is True
assert B_FULL["overrides"] == 1
B_STEPS = B["verified_steps"]

# ================================================================== geometry
# Every panel's plot rectangle is declared here, and label room is reserved
# OUTSIDE it, so annotations cannot silently run past the text block.
W, GH = 5.5, 4.42
L = []
A = L.append


def txt(x, y, s, fs=6.2, c="ink", ha="center", va="center", sans=True, op=None):
    # TikZ anchors compose as "<vertical> <horizontal>"; "center north" is not a
    # valid anchor, so build the pair explicitly rather than concatenating.
    h = {"center": "", "left": "west", "right": "east"}[ha]
    v = {"center": "", "top": "north", "bottom": "south"}[va]
    an = " ".join(t for t in (v, h) if t) or "center"
    f = r"\sffamily" if sans else ""
    o = f",opacity={op}" if op else ""
    A(rf"  \node[text={c},anchor={an},font={f}\fontsize{{{fs}}}{{{fs+1.2}}}\selectfont{o}] "
      rf"at ({x:.4f},{y:.4f}) {{{s}}};")


def rule(x0, y0, x1, y1, c="faint", lw=0.5, op=None, dash=None):
    d = f",dash pattern=on {dash}pt off {dash}pt" if dash else ""
    o = f",opacity={op}" if op else ""
    A(rf"  \draw[{c},line width={lw}pt{d}{o}] ({x0:.4f},{y0:.4f}) -- ({x1:.4f},{y1:.4f});")


def rect(x, y, w, h, c, op=None):
    o = f",opacity={op}" if op else ""
    A(rf"  \fill[{c}{o}] ({x:.4f},{y:.4f}) rectangle ({x+w:.4f},{y+h:.4f});")


def dot(x, y, c, r=0.032):
    A(rf"  \fill[{c}] ({x:.4f},{y:.4f}) circle ({r});")


def head(x, y, letter, title, sub):
    A(rf"  \node[text=navy,anchor=west,font=\sffamily\bfseries\fontsize{{7.2}}{{8}}\selectfont] "
      rf"at ({x:.4f},{y:.4f}) {{({letter})}};")
    txt(x + 0.16, y, title, fs=6.6, c="navy", ha="left")
    txt(x, y - 0.135, sub, fs=5.5, ha="left", op=0.75)


A(r"\begin{tikzpicture}[x=1in,y=1in,>=stealth,line join=round]")

# ------------------------------------------------------------------ panel (a)
# HORIZONTAL bars. Five category names cannot fit side by side under vertical
# bars at this width -- they collided under every font size that stayed legible.
# Rotated on their side the names get a whole column and read straight.
head(0.10, 4.16, "a", "Reference-law decomposition",
     "canonical-successor NLL (nats), lower is better")
AX, AW, AMAX = 0.98, 1.02, 6.6
AROWS = [3.90 - 0.112 * i for i in range(5)]
ANAMES = ["uniform", "empirical family", "learned fam.\\,/\\,uniform id",
          "emp.\\ fam.\\,/\\,learned id", "\\textsc{compose}"]
rule(AX, AROWS[-1] - 0.075, AX, AROWS[0] + 0.075, "faint", 0.6)
for i, (v, nm) in enumerate(zip(A_VAL, ANAMES)):
    yy = AROWS[i]
    rect(AX, yy - 0.036, v / AMAX * AW, 0.072,
         "indigo" if i == 4 else ("cyan" if i in (1, 3) else "mute"))
    txt(AX - 0.035, yy, nm, fs=5.0, ha="right", op=0.9)
    txt(AX + v / AMAX * AW + 0.03, yy, f"{v:.2f}", fs=5.2, ha="left", sans=False,
        op=None if i == 4 else 0.85)
# the 5.37 -> 4.10 span carries 86.7% of the full 1.46-nat gain
ya, yb = AROWS[1], AROWS[3]
bxr = AX + AW + 0.20
A(rf"  \draw[orchid,line width=0.6pt] ({bxr-0.03:.4f},{ya:.4f}) -- ({bxr:.4f},{ya:.4f}) -- "
  rf"({bxr:.4f},{yb:.4f}) -- ({bxr-0.03:.4f},{yb:.4f});")
A(rf"  \node[text=orchid,anchor=west,align=left,font=\sffamily\fontsize{{5.0}}{{5.8}}"
  rf"\selectfont] at ({bxr+0.03:.4f},{(ya+yb)/2:.4f}) "
  rf"{{\begin{{tabular}}{{@{{}}l@{{}}}}{A_IDSHARE*100:.1f}\% of the\\{A_GAIN:.2f} nat gain"
  rf"\end{{tabular}}}};")

# ------------------------------------------------------------------ panel (b)
head(2.86, 4.16, "b", "A rescued decision",
     f"sealed pair {B['index']}, {B_STEPS} verified edits, one override")
BX, BSY = 3.06, 3.66
NX = [BX + t for t in (0.0, 0.38, 0.76)]
for i in range(len(NX) - 1):
    rule(NX[i] + 0.030, BSY, NX[i + 1] - 0.030, BSY, "indigo", 1.4)
for nx in NX:
    dot(nx, BSY, "indigo", 0.026)
txt(NX[0], BSY - 0.095, "source", fs=5.0, op=0.8)
txt((NX[0] + NX[-1]) / 2, BSY + 0.075, "shared prefix", fs=5.0, op=0.8)
txt(NX[-1] + 0.02, BSY + 0.075, "override", fs=5.0, c="orchid", ha="left")
for yy, col, lw, lab in ((BSY + 0.22, "mute", 1.0,
                          f"local: best {B_GREEDY['best']:.2f}, not reached"),
                         (BSY - 0.22, "orchid", 1.4, "future-aware: target reached")):
    A(rf"  \draw[{col},line width={lw}pt] ({NX[-1]+0.030:.4f},{BSY:.4f}) "
      rf"to[out=0,in=180] ({NX[-1]+0.26:.4f},{yy:.4f});")
    dot(NX[-1] + 0.26, yy, col, 0.026)
    txt(NX[-1] + 0.31, yy, lab, fs=5.0, ha="left",
        c="orchid" if col == "orchid" else "ink", op=None if col == "orchid" else 0.85)
txt(2.86, 3.34, "intermediate states are not banked in the sealed record,", fs=4.7,
    ha="left", op=0.55)
txt(2.86, 3.26, "so they are drawn as unlabelled nodes rather than structures", fs=4.7,
    ha="left", op=0.55)

# ------------------------------------------------------------------ panel (c)
head(0.10, 2.90, "c", "Future reachability rescues failures",
     f"same denominator, $n={N_PAIRS}$ held-out pairs")
CX, CW, BH = 0.72, 1.44, 0.128
for r, (name, segs) in enumerate([
        ("local", [(C_LOCAL, "indigo"), (N_PAIRS - C_LOCAL, "faint")]),
        ("future-aware", [(C_LOCAL, "indigo"), (C_RESCUED, "orchid"),
                          (N_PAIRS - C_LOCAL - C_RESCUED, "faint")])]):
    yy = 2.52 - r * 0.22
    txt(CX - 0.035, yy + BH / 2, name, fs=5.2, ha="right", op=0.9)
    cx = CX
    for n_, col in segs:
        w_ = n_ / N_PAIRS * CW
        rect(cx, yy, w_, BH, col)
        txt(cx + w_ / 2, yy + BH / 2, f"{n_}", fs=5.0, sans=False,
            c="white" if col != "faint" else "ink", op=None if col != "faint" else 0.7)
        cx += w_
txt(CX, 2.18, f"{C_RESCUED} of {C_FAIL} local failures rescued", fs=5.0, ha="left", c="orchid")
txt(CX + CW, 2.52 + BH + 0.042, f"$+{C_DIFF:.1f}$ pp $[{C_CI[0]:.1f}, {C_CI[1]:.1f}]$",
    fs=5.4, ha="right", sans=False)

# ------------------------------------------------------------------ panel (d)
# no subtitle: the two axis labels already say what the axes are, and the
# subtitle was landing on top of the plot.
head(2.86, 2.90, "d", "Prioritizing continuation evaluations", "")
DX, DY, DW, DH = 3.22, 2.02, 1.58, 0.60
XLO, XHI, YLO, YHI = -6.0, 106.0, 36.0, 65.5
fx = lambda v: DX + (v - XLO) / (XHI - XLO) * DW
fy = lambda v: DY + (v - YLO) / (YHI - YLO) * DH
rule(DX, DY, DX, DY + DH, "faint", 0.6)
rule(DX, DY, DX + DW, DY, "faint", 0.6)
for gv in (0, 50, 100):
    rule(fx(gv), DY - 0.02, fx(gv), DY, "faint", 0.5)
    txt(fx(gv), DY - 0.036, f"{gv}", fs=4.8, va="top", sans=False, op=0.7)
for gv in (40, 50, 60):
    rule(DX - 0.02, fy(gv), DX, fy(gv), "faint", 0.5)
    txt(DX - 0.03, fy(gv), f"{gv}", fs=4.8, ha="right", sans=False, op=0.7)
txt(DX + DW / 2, DY - 0.155, "\\% of all-candidate evaluations", fs=4.8, op=0.8)
txt(DX - 0.16, DY + DH + 0.075, "\\% of targets recovered", fs=4.8,
    ha="left", op=0.8)
hp = next(p for p in D_PTS if "h_\\phi" in p[2])
fu = next(p for p in D_PTS if p[2] == "all-candidate")
rule(fx(hp[0]), fy(hp[1]), fx(fu[0]), fy(fu[1]), "orchid", 0.5, op=0.6, dash=1.3)
txt((fx(hp[0]) + fx(fu[0])) / 2, fy(hp[1]) + 0.048, "same aggregate recovery",
    fs=4.6, c="orchid")
DOFF = {"local": (0.045, 0.0, "left"),
        "$R_\\theta$ top-1": (0.045, -0.005, "left"),
        "$h_\\phi$ top-1": (-0.042, -0.052, "right"),
        "similarity top-2": (0.045, 0.0, "left"),
        "all-candidate": (-0.020, -0.072, "right")}
for x_, y_, lab, ti in D_PTS:
    col = "orchid" if "h_\\phi" in lab else ("mute" if ti else "indigo")
    dot(fx(x_), fy(y_), col, 0.029)
    if ti:
        A(rf"  \draw[ink,opacity=0.45,line width=0.4pt] ({fx(x_):.4f},{fy(y_):.4f}) circle (0.047);")
    ox, oy, ha_ = DOFF[lab]
    txt(fx(x_) + ox, fy(y_) + oy, lab, fs=4.8, ha=ha_, sans=False, op=0.9)
txt(DX + DW, DY + DH + 0.075, "circled: target-informed", fs=4.6, ha="right", op=0.6)

# ------------------------------------------------------------------ panel (e)
head(0.10, 1.52, "e", "Dynamic retargeting",
     f"paired post-switch utility, continue from $x_\\tau$ vs.\\ restart from $x_0$, "
     f"same remaining budget \\quad $n={E_N}$")
EX, EW, ELO, EHI = 1.44, 2.30, -0.05, 0.47
ex = lambda v: EX + (v - ELO) / (EHI - ELO) * EW
rule(ex(0), 0.72, ex(0), 1.22, "faint", 0.6)
txt(ex(0), 0.69, "0", fs=4.8, va="top", sans=False, op=0.7)
for i, (name, m, ci) in enumerate(E_ROWS):
    yy = 1.12 - i * 0.22
    txt(EX - 0.06, yy, name, fs=5.2, ha="right", op=0.9)
    rule(ex(ci[0]), yy, ex(ci[1]), yy, "indigo", 1.0)
    for e in ci:
        rule(ex(e), yy - 0.026, ex(e), yy + 0.026, "indigo", 1.0)
    dot(ex(m), yy, "orchid", 0.032)
    txt(ex(EHI) + 0.03, yy, f"$+{m:.3f}$ $[{ci[0]:.3f}, {ci[1]:.3f}]$", fs=5.2,
        ha="left", sans=False)
txt(ex(0.21), 0.60, "paired difference in post-switch utility", fs=4.8, op=0.8)

A(r"\end{tikzpicture}")
TIKZ = "\n".join(L)

defs = "\n".join(rf"\definecolor{{{k}}}{{HTML}}{{{v}}}" for k, v in COLORS.items())
(PAPER / "sections" / "fig_results.tex").write_text(
    "% GENERATED by figures/fig_results_tikz.py -- do not hand-edit.\n"
    "% Preamble needs: \\usepackage{tikz,xcolor}  \\usetikzlibrary{arrows}\n"
    f"{defs}\n\\begin{{figure}}[t]\n\\centering\n{TIKZ}\n"
    "\\caption{\\small\\textbf{Learning and intervening on the \\method{} molecular "
    "process.}}\n\\label{fig:mechanism}\n\\end{figure}\n")

prev = ("\\documentclass{article}\n"
        f"\\usepackage[paperwidth={W}in,paperheight={GH}in,margin=0pt]{{geometry}}\n"
        "\\usepackage{times,amsmath,amssymb,tikz,xcolor}\n\\usetikzlibrary{arrows}\n"
        f"{defs}\n\\pagestyle{{empty}}\\setlength{{\\parindent}}{{0pt}}\n"
        f"\\begin{{document}}\\noindent\n{TIKZ}\n\\end{{document}}\n")
wd = OUT / "_res"; wd.mkdir(exist_ok=True)
(wd / "r.tex").write_text(prev)
r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "r.tex"],
                   cwd=wd, capture_output=True, text=True)
if r.returncode != 0:
    sys.stderr.write(r.stdout[-4000:]); raise SystemExit("pdflatex failed")
(OUT / f"fig_results_{PALETTE}.pdf").write_bytes((wd / "r.pdf").read_bytes())
print(f"  wrote {OUT / f'fig_results_{PALETTE}.pdf'}")
print(f"  (a) gain {A_GAIN:.4f} -> {A_GAIN:.2f} nats, identity share {A_IDSHARE*100:.1f}%")
print(f"  (c) {C_LOCAL}/{N_PAIRS} -> {C_FULL}/{N_PAIRS}, +{C_DIFF:.1f} pp {[f'{v:.1f}' for v in C_CI]}")
print(f"  (d) sim2 x = {P['sim2']['fraction_of_universe_evaluated']*100:.3f} "
      f"(spec said 52.1, that is the n=67 arm)")
print(f"  (e) {E_ROWS[0][1]:+.3f} / {E_ROWS[1][1]:+.3f}, n={E_N}")
