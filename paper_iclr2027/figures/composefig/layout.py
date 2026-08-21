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

def columns(ax, n=3, left=0.028, right=0.972, gap=0.034, y=0.045, h=0.912):
    """n equal cards. Returns [(x, w), ...]."""
    w = (right - left - gap*(n-1)) / n
    xs = [left + i*(w+gap) for i in range(n)]
    for x in xs:
        P.card(ax, x, y, w, h)
    return [(x, w) for x in xs], y, h

def rows(n, top=0.735, bot=0.175):
    """Shared vertical positions. Draw these in EVERY panel -- that is the point."""
    return np.linspace(top, bot, n)
