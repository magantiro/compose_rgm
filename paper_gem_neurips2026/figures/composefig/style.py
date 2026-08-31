"""Design tokens. Every figure imports from here; nothing hard-codes a colour."""
import matplotlib as mpl

NAVY   = "#1B1B8F"   # panel letters, headings, filled equation boxes
INDIGO = "#6B6BD6"   # primary nodes and arrows
CYAN   = "#4EC3E0"   # secondary branch
ORCHID = "#C46BD6"   # highlight, goal, conditioning
GROUND = "#F7F0FB"   # page ground behind the cards
CARD   = "#FFFFFF"
TILE   = "#FCFAFE"
EDGE   = "#E4D9F2"
INK    = "#2A2A5A"
MUTE   = "#B9AFD6"

SEQ = [INDIGO, CYAN, ORCHID, NAVY, "#8E86E0", "#7FD4E8"]

FS = dict(head=15, word=21, sub=9.5, label=11, num=9, small=8.4, tiny=7.8)
R  = dict(card=0.022, tile=0.012, box=0.014, pill=0.008)   # corner radii

def apply():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "axes.linewidth": 0,
        "text.color": INK, "savefig.transparent": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,          # embed real fonts, not paths
    })
