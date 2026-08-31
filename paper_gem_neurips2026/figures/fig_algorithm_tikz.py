"""Generate the COMPOSE algorithm figure as a TikZ fragment.

WHY TIKZ AND NOT MATPLOTLIB
---------------------------
The equations are the point of this figure, and matplotlib's mathtext is a
reimplementation of TeX maths, not TeX. Emitting TikZ means the display maths is
set by the same engine, in the same Times, at the same size as the body text, so
the figure reads as part of the paper rather than as an image dropped into it.
It also ships vector, which fig1.tex's spec requires ("must ship as PDF or SVG,
never PNG ... if it cannot be exported as vector from the design tool, build it
in TikZ").

WHAT THE FIGURE SHOWS, AND WHY THIS IS THE ALGORITHM
----------------------------------------------------
A controlled TRAJECTORY, not a single step. Proposition 2 is a statement about
the K-step law, so a figure showing one state's fan-out shows a fragment of the
method. Here the reader sees the whole loop:

    at each state the executor exposes several legal successors (thin branches),
    the frozen reference law says which are plausible,
    the controller reweights them by what remains reachable in the budget left,
    one is committed, the budget decrements, and control is recomputed from the
    state actually reached.

The two governing laws sit in filled navy boxes above and below the process --
the reference law that is learned once and frozen, and the control law that is
recomputed at every state. That top/bottom framing is deliberate: it is the
Chatterjee-lab convention fig1.tex documents as "the signature move ... primary
equations sit in FILLED DEEP-NAVY ROUNDED BOXES WITH WHITE TEXT", and it puts
the goal-independent object and the goal-conditioned object on opposite sides of
the trajectory they act on.

ENCODING. Arrow WIDTH carries P^{z,b}. This is why no per-branch numbers are
printed: only a few successors of each state are drawn, so printed weights would
have to either sum to less than one (confusing) or imply the drawn support is
complete (false). Magnitude by width states the ordering without claiming
exhaustiveness, and the exact arithmetic lives in the caption.

CHEMISTRY. A real 3-edit trajectory, one executor family per step, every
intermediate a complete valid molecule. |dn| <= 1 throughout, consistent with
the audited families in diagnostics/compose_delta_n_audit.json:

    x_0  CCCCO         butan-1-ol            5 heavy
      atom_insert   dn = +1
    x_1  CCCCCO        pentan-1-ol           6 heavy
      bond_reroute  dn =  0
    x_2  CC(C)CCO      3-methylbutan-1-ol    6 heavy
      cycle_close   dn =  0
    x_3  CC1(CO)CC1    (1-methylcyclopropyl)methanol   6 heavy

Uncommitted alternatives drawn at each state, also real and legal:
    from x_0   CCC(C)O   butan-2-ol        CCCCN   butan-1-amine
    from x_1   CCCCCN    pentan-1-amine    CCC(C)CO  2-methylbutan-1-ol
    from x_2   CC(C)CCN  3-methylbutan-1-amine       CC(C)CC  2-methylbutane

Molecules are drawn as node-link graphs -- filled circles, thin bonds, no atom
labels -- which is the reference style, with heteroatoms carrying their own
palette colour so O and N remain distinguishable without lettering. Every layout
is real RDKit 2D geometry, and all structures share ONE scale so a 5-heavy
molecule draws smaller than a 6-heavy one and dn is visible rather than asserted.

Outputs
    sections/fig_algorithm.tex        the \\input-able TikZ fragment
    figures/out/fig_algorithm_tikz.pdf   cropped preview
"""
import os, pathlib, subprocess, sys
import numpy as np

HERE = pathlib.Path(__file__).parent
PAPER = HERE.parent
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------- palettes
# Every palette holds one hue family plus a single complementary accent. Roles,
# not names: `navy` fills the equation boxes, `indigo` is the committed path and
# carbon, `cyan` is nitrogen, `orchid` is oxygen AND the objective, `mute` is the
# legal-but-uncommitted support.
PALETTES = {
    # the reference lab's blue -> indigo -> magenta family
    "indigo": dict(navy="1B1B8F", indigo="5A5AC8", cyan="3FB8DC", orchid="BC5FD0",
                   mute="BFB6DA", faint="DFDAEE", ink="2A2A4A",
                   ground="F5F1FB", card="FFFFFF", edge="E7DFF4"),
    # deep slate blue with a warm bronze accent; sober, prints well in greyscale
    "slate":  dict(navy="223047", indigo="4C6580", cyan="6E9BB0", orchid="B4762F",
                   mute="C2CBD5", faint="E1E6EC", ink="27313E",
                   ground="F1F4F8", card="FFFFFF", edge="DEE5ED"),
    # deep teal with terracotta; distinctive without reading as a template
    "teal":   dict(navy="12464C", indigo="357C82", cyan="63AAA6", orchid="C2673A",
                   mute="BAD0CF", faint="DDE9E8", ink="1F383A",
                   ground="EEF5F4", card="FFFFFF", edge="D9E8E6"),
    # near-monochrome graphite, one rust accent carrying every semantic highlight
    "graphite": dict(navy="2B2B2B", indigo="5C5C5C", cyan="909090", orchid="A8452A",
                     mute="CBCBCB", faint="E6E6E6", ink="2B2B2B",
                     ground="F4F4F3", card="FFFFFF", edge="E3E3E1"),
    # deep forest with amber
    "forest": dict(navy="1E3B30", indigo="46705C", cyan="7BA88C", orchid="BE8B2B",
                   mute="C4D3C8", faint="E2EAE3", ink="223328",
                   ground="F0F5F1", card="FFFFFF", edge="DDE8DF"),
}
PALETTE = os.environ.get("FIG_PALETTE", "teal")
# fill    dark fill, white maths  -- the reference lab's signature
# tint    pale wash, dark maths   -- inverts the value relationship
# outline hairline rule, no fill  -- keeps the shape, drops the mass
# tab     pale wash + saturated edge bar -- the recommended one
# rule    no shape at all         -- a single accent rule as separator
FRAME = os.environ.get("FIG_FRAME", "tab")
COLORS = PALETTES[PALETTE]

# ------------------------------------------------------------------ the path
SPINE = ["CCCCO", "CCCCCO", "CCC(C)CO", "CC(C)CO", "OCC1CC1"]
STEP_FAMILY = ["atom insert", "bond reroute", "atom delete", "cycle close"]
DN = ["+1", "0", "-1", "0"]
# Alternatives must be DISTINCT from the spine and from each other: MOLS is keyed
# by SMILES, so reusing a spine molecule here silently overwrote its depiction
# with one aligned to a different source. x_3 gets a single branch so the right
# margin stays clear for g_z.
ALTS = [["CCC(C)O", "CCCCN"], ["CCCCCN", "CC(C)CCO"],
        ["CCC(C)CN", "CCC(C)C"], ["CC(C)CN"]]
assert len(set(SPINE + [a for r in ALTS for a in r])) == len(SPINE) + sum(len(r) for r in ALTS), \
    "every structure needs its own SMILES key"

# arrow widths carry P^{z,b}: committed edge is the mode of the controlled law
W_COMMIT, W_ALT = 2.0, (0.85, 0.55)


def _mcs_map(prev, cur):
    """Largest connected common core, as an index map cur -> prev."""
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    r = rdFMCS.FindMCS([prev, cur], ringMatchesRingOnly=False,
                       completeRingsOnly=False, timeout=10)
    patt = Chem.MolFromSmarts(r.smartsString)
    ma, mb = prev.GetSubstructMatch(patt), cur.GetSubstructMatch(patt)
    return patt, {mb[k]: ma[k] for k in range(min(len(ma), len(mb)))}


def _delta(prev, cur, kind):
    """What this operator actually changed, so the highlight is not a lie.

    The MCS core is 5 atoms at every step, so for the two dn = 0 steps it reports
    an 'unmapped' atom -- but that atom MOVED, it was not added. Haloing it would
    claim an atom-count change the transition does not make. The highlight is
    therefore operator-aware: insertion marks the new ATOM; reroute and ring
    closure mark the new BOND, which is what those operators actually do.
    """
    _, fwd = _mcs_map(prev, cur)
    new_atoms = {a for a in range(cur.GetNumAtoms()) if a not in fwd}
    new_bonds = set()
    for b in cur.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u in fwd and v in fwd:
            if prev.GetBondBetweenAtoms(fwd[u], fwd[v]) is None:
                new_bonds.add((u, v))
        else:
            new_bonds.add((u, v))
    if kind == "atom delete":
        return set(), set()          # the change is an ABSENCE; see the ghost below
    if kind == "atom insert":
        return new_atoms, set()
    if kind == "cycle close":
        ring = {(u, v) for u, v in new_bonds
                if cur.GetBondBetweenAtoms(u, v).IsInRing()}
        return set(), ring or new_bonds
    return set(), new_bonds


def build():
    """Depictions ALIGNED to the previous state, so the operator is visible.

    RDKit lays out every molecule independently, which gave x_0 and x_1 unrelated
    coordinates: the label said "atom insert" but nothing in the drawing showed
    which atom, or where. Aligning each depiction to its predecessor on the common
    core turns the edit into a local difference the reader can actually see, and
    each structure is then centred on that CORE, so cores land in the same place
    across panels instead of drifting with the bounding box.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    def pack(m, core=None):
        c = m.GetConformer()
        pos = np.array([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y]
                        for i in range(m.GetNumAtoms())])
        anchor = pos[sorted(core)].mean(0) if core else pos.mean(0)
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
        sym = [a.GetSymbol() for a in m.GetAtoms()]
        return pos - anchor, bonds, sym

    def align(m, ref):
        patt, fwd = _mcs_map(ref, m)
        try:
            AllChem.GenerateDepictionMatching2DStructure(m, ref, refPatt=patt)
        except Exception:
            AllChem.Compute2DCoords(m)
        return set(fwd)

    spine = [Chem.MolFromSmiles(x) for x in SPINE]
    AllChem.Compute2DCoords(spine[0])
    mols, hot, ghost = {}, {}, {}
    mols[SPINE[0]] = pack(spine[0])
    hot[SPINE[0]] = (set(), set())
    ghost[SPINE[0]] = []
    for k in range(1, len(spine)):
        prev, cur = spine[k - 1], spine[k]
        core = align(cur, prev)
        mols[SPINE[k]] = pack(cur, core)
        hot[SPINE[k]] = _delta(prev, cur, STEP_FAMILY[k - 1])
        ghost[SPINE[k]] = []
        if STEP_FAMILY[k - 1] == "atom delete":
            # Deletion is the one operator whose effect is invisible in the product.
            # Depictions share a frame after alignment, so the removed atom's old
            # position transfers directly: draw it where it used to be, plus the
            # bond it used to sit on. That is the only honest way to show a
            # subtraction -- shading something present would name the wrong atom.
            _, fwd = _mcs_map(prev, cur)
            back = {v: kk for kk, v in fwd.items()}
            pc, cc = prev.GetConformer(), cur.GetConformer()
            anchor = np.array([[cc.GetAtomPosition(i).x, cc.GetAtomPosition(i).y]
                               for i in sorted(fwd)]).mean(0)
            for a_ in range(prev.GetNumAtoms()):
                if a_ in back:
                    continue
                gp = np.array([pc.GetAtomPosition(a_).x, pc.GetAtomPosition(a_).y]) - anchor
                stubs = [back[n.GetIdx()] for n in prev.GetAtomWithIdx(a_).GetNeighbors()
                         if n.GetIdx() in back]
                ghost[SPINE[k]].append((gp, stubs))
    for k, row in enumerate(ALTS):               # align each fan to its own source
        for a in row:
            m = Chem.MolFromSmiles(a)
            core = align(m, spine[k])
            mols[a] = pack(m, core)
            hot[a] = (set(), set())
            ghost[a] = []
    return mols, hot, ghost


MOLS, HOT, GHOST = build()
GLOBAL = max(np.abs(p_).max() for p_, _, _ in MOLS.values())

ATOM_COLOR = {"C": "indigo", "O": "orchid", "N": "cyan"}


def span(smiles, size):
    """(left, right, half-height) AS DRAWN, halo included.

    Three separate numbers because these structures are far wider than tall AND
    are centred on their common core, so the extent left of centre differs from
    the extent right of it. A single radius put arrowheads inside some molecules
    and far from others; ignoring the halo put one arrowhead on top of it.
    """
    pos = MOLS[smiles][0] * (size / GLOBAL)
    lo, hi = float(pos[:, 0].min()), float(pos[:, 0].max())
    hh = float(np.abs(pos[:, 1]).max())
    for gp, _ in GHOST.get(smiles, []):
        gx, gy = gp * (size / GLOBAL)
        lo, hi = min(lo, float(gx) - 0.048), max(hi, float(gx) + 0.048)
        hh = max(hh, abs(float(gy)) + 0.048)
    ha, _ = HOT.get(smiles, (set(), set()))
    if ha:
        r = size * 0.20
        idx = sorted(ha)
        lo, hi = min(lo, float(pos[idx, 0].min()) - r), max(hi, float(pos[idx, 0].max()) + r)
        hh = max(hh, float(np.abs(pos[idx, 1]).max()) + r)
    return -lo, hi, hh


def emit_mol(smiles, cx, cy, size, bond_c, bond_w, r=0.026, mute=False, halo=False):
    """TikZ for one node-link molecule. No frame, no atom labels."""
    pos, bonds, sym = MOLS[smiles]
    q = pos * (size / GLOBAL) + np.array([cx, cy])
    o = []
    if halo:
        ha, hb = HOT.get(smiles, (set(), set()))
        for u, v in hb:                                # the bond the operator made
            o.append(f"  \\draw[navy,opacity=0.20,line width=3.6pt,line cap=round] "
                     f"({q[u,0]:.4f},{q[u,1]:.4f}) -- ({q[v,0]:.4f},{q[v,1]:.4f});")
        for a in ha:                                   # the atom the operator added
            o.append(f"  \\fill[navy,opacity=0.20] "
                     f"({q[a,0]:.4f},{q[a,1]:.4f}) circle ({size*0.20:.4f});")
        for gp, stubs in GHOST.get(smiles, []):        # the atom the operator removed
            gx, gy = gp * (size / GLOBAL) + np.array([cx, cy])
            for t in stubs:
                o.append(f"  \\draw[ink,opacity=0.34,line width=0.5pt,"
                         f"dash pattern=on 1.1pt off 1.1pt] "
                         f"({q[t,0]:.4f},{q[t,1]:.4f}) -- ({gx:.4f},{gy:.4f});")
            o.append(f"  \\draw[ink,opacity=0.34,line width=0.5pt,"
                     f"dash pattern=on 1.1pt off 1.1pt] "
                     f"({gx:.4f},{gy:.4f}) circle ({r*1.5:.4f});")
    for u, v in bonds:
        o.append(f"  \\draw[{bond_c},line width={bond_w}pt,line cap=round] "
                 f"({q[u,0]:.4f},{q[u,1]:.4f}) -- ({q[v,0]:.4f},{q[v,1]:.4f});")
    for k, (X, Y) in enumerate(q):
        c = "mute" if mute else ATOM_COLOR.get(sym[k], "ink")
        rr = r if sym[k] == "C" else r * 1.28
        o.append(f"  \\fill[{c}] ({X:.4f},{Y:.4f}) circle ({rr:.4f});")
    return o


# ================================================================== geometry
W, GH = 5.5, 2.88                         # 5.5in = \textwidth of gem_workshop.sty
BOX_L, BOX_R = 0.10, W - 0.10
TOP_B, TOP_T = 2.42, 2.78                 # equal-height boxes, read as a pair
BOT_B, BOT_T = 0.06, 0.42
SPINE_Y = 1.14
MOL_R, ALT_R = 0.34, 0.20
ARROW_GAP = 0.62      # > 0.495in, the widest label that sits in it
ALT_AT = [(0.42, 2.05), (0.84, 1.76)]     # (x-offset from the state, height)

# States placed from their MEASURED widths, so every arrow is the same length
# whatever the structures do. Five of them no longer fit on an even grid.
_sp = [span(s_, MOL_R) for s_ in SPINE]
XS, _x = [], 0.16
for _l, _r, _h in _sp:
    _x += _l
    XS.append(_x)
    _x += _r + ARROW_GAP
XS = [v + (W - (_x - ARROW_GAP + 0.16)) / 2 for v in XS]
HW = _sp

# The operator label no longer fits on one line beside four arrows, so family and
# dn stack, and the state row drops clear of them.
Y_FAM, Y_DN = SPINE_Y - 0.085, SPINE_Y - 0.165
Y_LABEL, Y_BUDGET = SPINE_Y - 0.33, SPINE_Y - 0.465

L = []
A = L.append


def callout(cx, cy, tex, border, fs=6.5):
    """Filled bordered label carrying a symbol from the equations, placed ON the
    thing it names, so every symbol in the two boxes has a referent."""
    A(rf"  \node[draw={border},fill=white,line width=0.45pt,rounded corners=1.8pt,"
      rf"inner sep=2.2pt,font=\fontsize{{{fs}}}{{{fs+1}}}\selectfont,text=ink] "
      rf"at ({cx},{cy}) {{${tex}$}};")


A(r"\begin{tikzpicture}[x=1in,y=1in,>=stealth,line join=round]")

def eqframe(y0, y1, above):
    """Draw whatever encloses an equation, and return the maths colour."""
    if FRAME == "fill":
        A(rf"\fill[navy,rounded corners=3.2pt] ({BOX_L},{y0}) rectangle ({BOX_R},{y1});")
        return "white"
    if FRAME == "tint":
        A(rf"\fill[navy!7,rounded corners=3.2pt] ({BOX_L},{y0}) rectangle ({BOX_R},{y1});")
        return "navy"
    if FRAME == "tab":
        # My pick. A pale wash zones the equation without competing with the
        # molecules, and a single saturated edge bar supplies the colour as
        # punctuation rather than as a field. Inverting the value relationship
        # (dark maths on light) is what breaks the resemblance to the reference,
        # whose signature is dark fill + white text, symmetric top and bottom.
        A(rf"\fill[navy!6,rounded corners=3.2pt] ({BOX_L},{y0}) rectangle ({BOX_R},{y1});")
        A(rf"\fill[navy,rounded corners=1.4pt] ({BOX_L},{y0}) rectangle ({BOX_L+0.042},{y1});")
        return "navy"
    if FRAME == "outline":
        A(rf"\draw[navy,line width=0.7pt,rounded corners=3.2pt] "
          rf"({BOX_L},{y0}) rectangle ({BOX_R},{y1});")
        return "navy"
    yr = y0 - 0.07 if above else y1 + 0.07
    A(rf"\draw[navy,opacity=0.55,line width=0.8pt] ({BOX_L},{yr}) -- ({BOX_R},{yr});")
    return "navy"


A(r"% ---- the executor fixes the support; R_theta is a law on it")
_c = eqframe(TOP_B, TOP_T, True)
A(rf"\node[text={_c},anchor=center] at ({W/2},{(TOP_B+TOP_T)/2}) "
  r"{$\textstyle a\in\mathcal{A}(x),\quad T(x,a)\simeq y\in\mathcal{S}(x),"
  r"\qquad \sum_{y\in\mathcal{S}(x)}R_\theta(y\mid x)=1$};")

A(r"% ---- control law: recomputed at every state from the remaining budget")
_c = eqframe(BOT_B, BOT_T, False)
A(rf"\node[text={_c},anchor=center] at ({W/2},{(BOT_B+BOT_T)/2}) "
  r"{$\displaystyle P^{z,b}(y\mid x)\;=\;R_\theta(y\mid x)\,"
  r"\frac{h_{b-1}(y,z)}{h_b(x,z)},\qquad "
  r"h_b(x,z)=\mathbb{E}_{R_\theta}\!\left[g_z(X_b)\mid X_0=x\right]$};")

A(r"% ---- legal successors the controller did not commit to")
for i_, alts in enumerate(ALTS):
    for j_, alt in enumerate(alts):
        dx, ay = ALT_AT[j_]
        ax = XS[i_] + dx
        al, _ar, _ah = span(alt, ALT_R)
        A(rf"  \draw[mute,line width={W_ALT[j_]}pt,->,opacity=0.9] "
          rf"({XS[i_]+HW[i_][1]*0.30},{SPINE_Y+HW[i_][2]+0.03}) "
          rf"to[out=72,in=200] ({ax-al-0.05},{ay});")
        L.extend(emit_mol(alt, ax, ay, ALT_R, "mute", 0.55, r=0.016, mute=True))

A(r"% ---- the committed trajectory; arrow width carries P^{z,b}")
for i_ in range(len(XS) - 1):
    A(rf"  \draw[indigo,line width={W_COMMIT}pt,->,line cap=round] "
      rf"({XS[i_]+HW[i_][1]+0.09},{SPINE_Y}) -- ({XS[i_+1]-HW[i_+1][0]-0.09},{SPINE_Y});")
    mx = (XS[i_] + HW[i_][1] + XS[i_ + 1] - HW[i_ + 1][0]) / 2
    A(rf"  \node[text=ink,anchor=north,font=\sffamily\fontsize{{5.9}}{{7}}\selectfont,opacity=0.85] "
      rf"at ({mx},{Y_FAM}) {{{STEP_FAMILY[i_]}}};")
    A(rf"  \node[text=ink,anchor=north,font=\fontsize{{5.9}}{{7}}\selectfont,opacity=0.55] "
      rf"at ({mx},{Y_DN}) {{$\Delta n={DN[i_]}$}};")

for i_, smi in enumerate(SPINE):
    L.extend(emit_mol(smi, XS[i_], SPINE_Y, MOL_R, "indigo", 1.0, r=0.026, halo=True))
    lbl = "x_K" if i_ == len(SPINE) - 1 else f"x_{i_}"
    A(rf"  \node[text=navy,anchor=center,font=\fontsize{{8.4}}{{9}}\selectfont] "
      rf"at ({XS[i_]},{Y_LABEL}) {{${lbl}$}};")
    b_ = "b=K" if i_ == 0 else (f"b=K-{i_}" if i_ < len(SPINE) - 1 else "b=0")
    A(rf"  \node[text=ink,anchor=center,font=\fontsize{{6.0}}{{7}}\selectfont,opacity=0.7] "
      rf"at ({XS[i_]},{Y_BUDGET}) {{${b_}$}};")

A(r"% ---- the maths, tied to the objects it describes")
A(rf"  \node[text=ink,anchor=center,font=\fontsize{{7.0}}{{8}}\selectfont,opacity=0.75] "
  rf"at ({XS[0]+ALT_AT[0][0]},{ALT_AT[0][1]+0.24}) {{$T(x,a)\simeq y$}};")

_pmx = (XS[0] + HW[0][1] + XS[1] - HW[1][0]) / 2
A(rf"  \draw[indigo,line width=0.45pt,opacity=0.9] ({_pmx},{SPINE_Y+0.112}) -- ({_pmx},{SPINE_Y+0.032});")
callout(_pmx, SPINE_Y + 0.155, r"P^{z,b}(y\mid x)", "indigo")

callout(XS[1] + 0.30, (SPINE_Y + ALT_AT[1][1]) / 2 + 0.06, r"R_\theta(y\mid x)", "mute")

_hx = XS[len(ALTS) - 1] + ALT_AT[0][0]
A(rf"  \draw[mute,line width=0.45pt,opacity=0.9] ({_hx+0.14},{ALT_AT[0][1]+0.20}) -- ({_hx+0.05},{ALT_AT[0][1]+0.11});")
callout(_hx + 0.34, ALT_AT[0][1] + 0.26, r"h_{b-1}(y,z)\,/\,h_b(x,z)", "mute", fs=6.2)

A(rf"  \draw[orchid,line width=0.7pt,dash pattern=on 2pt off 1.6pt,opacity=0.9] "
  rf"({XS[-1]},{SPINE_Y+HW[-1][2]+0.05}) -- ({XS[-1]},{ALT_AT[1][1]-0.06});")
A(rf"  \node[text=orchid,anchor=south,font=\fontsize{{8.4}}{{9}}\selectfont] "
  rf"at ({XS[-1]},{ALT_AT[1][1]-0.05}) {{$g_z$}};")

A(r"\end{tikzpicture}")

TIKZ = "\n".join(L)

# ---------------------------------------------------------------- emit fragment
defs = "\n".join(rf"\definecolor{{{k}}}{{HTML}}{{{v}}}" for k, v in COLORS.items())
frag = (
    "% GENERATED by figures/fig_algorithm_tikz.py -- do not hand-edit.\n"
    "% Requires in the preamble: \\usepackage{tikz,xcolor}\n"
    "%                           \\usetikzlibrary{arrows}\n"
    f"{defs}\n"
    "\\begin{figure}[t]\n\\centering\n"
    f"{TIKZ}\n"
    "\\caption{\\small\\textbf{The \\method{} molecular process under finite-horizon "
    "control.} At each molecular state the executor exposes the legal canonical "
    "successors $\\mathcal{S}(x)$ (thin branches). The frozen reference law "
    "$R_\\theta$ scores them without seeing the objective; the controller reweights "
    "them by the value of the molecular futures they leave reachable within the "
    "remaining budget $b$, and arrow width is proportional to $P^{z,b}$. One "
    "successor is committed, $b$ decrements, and control is recomputed from the "
    "state actually reached, so an objective change at any point resumes from "
    "$x_\\tau$ rather than restarting. The trajectory shown grows, rearranges and "
    "shrinks, then closes a ring inside one state space, with $|\\Delta n|\\le 1$ at "
    "every transition. Each depiction is aligned to its predecessor on their common "
    "core, and the shaded atom or bond marks what that operator changed. Deletion "
    "is the one edit that leaves no trace in the product, so the removed atom is "
    "drawn as a dashed ghost where it stood.}\n\\label{fig:algorithm}\n\\end{figure}\n"
)
(PAPER / "sections" / "fig_algorithm.tex").write_text(frag)

# ------------------------------------------------- all-palette comparison sheet
if os.environ.get("FIG_COMPARE"):
    blocks = []
    for name, pal in PALETTES.items():
        d = "\n".join(rf"\definecolor{{{k}}}{{HTML}}{{{v}}}" for k, v in pal.items())
        blocks.append(
            d + "\n\\noindent{\\sffamily\\footnotesize\\bfseries " + name +
            "}\\par\\vspace{2pt}\\noindent\n" + TIKZ + "\n\\par\\vspace{16pt}")
    sheet = ("\\documentclass{article}\n"
             "\\usepackage[paperwidth=5.7in,paperheight=15.6in,margin=6pt]{geometry}\n"
             "\\usepackage{times,amsmath,amssymb,tikz,xcolor}\n\\usetikzlibrary{arrows}\n"
             "\\pagestyle{empty}\\setlength{\\parindent}{0pt}\n\\begin{document}\n"
             + "\n".join(blocks) + "\n\\end{document}\n")
    cd = OUT / "_compare"; cd.mkdir(exist_ok=True)
    (cd / "cmp.tex").write_text(sheet)
    rc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "cmp.tex"],
                        cwd=cd, capture_output=True, text=True)
    if rc.returncode != 0:
        sys.stderr.write(rc.stdout[-4000:]); raise SystemExit("compare sheet failed")
    (OUT / "fig_algorithm_palettes.pdf").write_bytes((cd / "cmp.pdf").read_bytes())
    print(f"  wrote {OUT/'fig_algorithm_palettes.pdf'}  ({len(PALETTES)} palettes)")

# ------------------------------------------------ one finished PDF per palette
if os.environ.get("FIG_ALL"):
    for name, pal in PALETTES.items():
        d = "\n".join(rf"\definecolor{{{k}}}{{HTML}}{{{v}}}" for k, v in pal.items())
        doc = ("\\documentclass{article}\n"
               "\\usepackage[paperwidth=5.5in,paperheight=2.88in,margin=0pt]{geometry}\n"
               "\\usepackage{times,amsmath,amssymb,tikz,xcolor}\n\\usetikzlibrary{arrows}\n"
               + d + "\n\\pagestyle{empty}\\setlength{\\parindent}{0pt}\n"
               "\\begin{document}\\noindent\n" + TIKZ + "\n\\end{document}\n")
        wd = OUT / "_pal"; wd.mkdir(exist_ok=True)
        (wd / "p.tex").write_text(doc)
        rc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "p.tex"],
                            cwd=wd, capture_output=True, text=True)
        if rc.returncode != 0:
            sys.stderr.write(rc.stdout[-3000:]); raise SystemExit(f"{name} failed")
        (OUT / f"fig_algorithm_{name}.pdf").write_bytes((wd / "p.pdf").read_bytes())
        print(f"  wrote {OUT/f'fig_algorithm_{name}.pdf'}")

# ------------------------------------------------------------------- preview
prev = (
    "\\documentclass{article}\n"
    "\\usepackage[paperwidth=5.5in,paperheight=2.88in,margin=0pt]{geometry}\n"
    "\\usepackage{times,amsmath,amssymb,tikz,xcolor}\n"
    "\\usetikzlibrary{arrows}\n"
    f"{defs}\n"
    "\\pagestyle{empty}\\setlength{\\parindent}{0pt}\n"
    "\\begin{document}\\noindent\n"
    f"{TIKZ}\n"
    "\\end{document}\n"
)
tmp = OUT / "_preview"
tmp.mkdir(exist_ok=True)
(tmp / "prev.tex").write_text(prev)
r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "prev.tex"],
                   cwd=tmp, capture_output=True, text=True)
if r.returncode != 0:
    sys.stderr.write(r.stdout[-4000:])
    raise SystemExit("pdflatex failed")
(OUT / "fig_algorithm_tikz.pdf").write_bytes((tmp / "prev.pdf").read_bytes())
print(f"  wrote {PAPER/'sections'/'fig_algorithm.tex'}")
print(f"  wrote {OUT/'fig_algorithm_tikz.pdf'}")
