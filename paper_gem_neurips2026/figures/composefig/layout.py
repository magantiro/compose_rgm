"""Panel grids. Row positions are computed once so panel invariance is structural."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from . import style as S, prims as P

def canvas(w=16, h=9, dpi=200):
    S.apply()
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    fig.patch.set_facecolor(S.GROUND)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.012, 0.012), 0.976, 0.976,
        boxstyle="round,pad=0,rounding_size=0.02", fc=S.GROUND, ec="none", zorder=-1))
    return fig, ax

def columns(ax, n=3, left=0.028, right=0.972, gap=0.030, y=0.045, h=0.912, ratios=None):
    """n cards. ratios lets one panel carry more content than the others."""
    ratios = ratios or [1.0]*n
    avail = right - left - gap*(n-1)
    ws = [avail*r/sum(ratios) for r in ratios]
    out, x = [], left
    for w in ws:
        P.card(ax, x, y, w, h)
        out.append((x, w)); x += w + gap
    return out, y, h

def rows(n, top=0.735, bot=0.175):
    """Shared vertical positions. Draw these in EVERY panel -- that is the point."""
    return np.linspace(top, bot, n)
