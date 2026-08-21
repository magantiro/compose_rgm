"""Drawing primitives that carry the house style. Composable, no globals."""
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib as mpl
from . import style as S

def card(ax, x, y, w, h, fc=None, ec=None, r=None, lw=1.1, z=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r or S.R['card']}",
        fc=fc or S.CARD, ec=ec or S.EDGE, lw=lw, zorder=z))

def eq_box(ax, x, y, w, h, tex, fs=10.5, z=5):
    """The signature move: filled navy box, white maths."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={S.R['box']}", fc=S.NAVY, ec="none", zorder=z))
    ax.text(x + w/2, y + h/2, tex, ha="center", va="center",
            color="white", fontsize=fs, zorder=z+1)

def note_box(ax, x, y, tex, color=None, fs=13, pad=0.026, z=5):
    """White box, thin coloured border. Secondary annotations live here."""
    c = color or S.INDIGO
    ax.add_patch(FancyBboxPatch((x-pad, y-pad*0.92), pad*2, pad*1.84,
        boxstyle=f"round,pad=0,rounding_size={S.R['tile']}", fc="white", ec=c, lw=1.4, zorder=z))
    ax.text(x, y, tex, ha="center", va="center", color=S.NAVY, fontsize=fs, zorder=z+1)

def arrow(ax, p, q, lw=1.0, color=None, alpha=1.0, rad=0.0, ls="-", z=4, head=7):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}",
        arrowstyle=f"-|>,head_width={head*0.045},head_length={head*0.07}",
        lw=lw, color=color or S.INDIGO, alpha=alpha, linestyle=ls, zorder=z,
        shrinkA=0, shrinkB=0, joinstyle="round", capstyle="round"))

DASH = (0, (2.6, 2.2))

def pill(ax, x, y, text, w=0.030, h=0.021, fc=None, fs=7.4, z=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={S.R['pill']}",
        fc=fc or S.ORCHID, ec="none", alpha=0.16, zorder=z))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color=S.NAVY, fontsize=fs, zorder=z+1)

def gradient_bar(fig, ax, x, y, w, h, lo="low", hi="high", label=None, c=None):
    g = fig.add_axes([x, y, w, h]); g.axis("off")
    g.imshow(np.linspace(0, 1, 256).reshape(1, -1), aspect="auto",
        cmap=mpl.colors.LinearSegmentedColormap.from_list("g", ["#EDEBFA", c or S.INDIGO]))
    ax.text(x, y - 0.020, lo, ha="left", va="center", fontsize=S.FS["small"], color=S.INK)
    ax.text(x + w, y - 0.020, hi, ha="right", va="center", fontsize=S.FS["small"], color=S.INK)
    if label:
        ax.text(x + w/2, y + h + 0.022, label, ha="center", va="center",
                color=S.NAVY, fontsize=S.FS["label"])

def legend_rows(ax, x, y, rows, dx=0.042, dy=0.030):
    """rows = [(linewidth, alpha, text), ...]"""
    for i, (lw, al, txt) in enumerate(rows):
        yy = y - i*dy
        arrow(ax, (x, yy), (x + dx, yy), lw, S.INDIGO, al)
        ax.text(x + dx + 0.010, yy, txt, ha="left", va="center",
                fontsize=S.FS["small"], color=S.INK)

def panel_head(ax, cx, head, word, sub, y=0.935):
    ax.text(cx, y,        head, ha="center", va="center", color=S.NAVY,
            fontsize=S.FS["head"], fontweight="bold")
    ax.text(cx, y-0.043,  word, ha="center", va="center", color=S.INDIGO,
            fontsize=S.FS["word"], fontweight="bold")
    ax.text(cx, y-0.077,  sub,  ha="center", va="center", color=S.INK,
            fontsize=S.FS["sub"], alpha=0.75)
