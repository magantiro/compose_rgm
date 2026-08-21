"""Molecules. Abstract node-link graphs for concept figures; RDKit for real ones."""
import numpy as np
from matplotlib.patches import Circle
from . import style as S

def graph(n, seed, cyan_frac=0.25):
    """Deterministic small node-link graph, normalised to unit extent."""
    r = np.random.default_rng(seed)
    pos = np.zeros((n, 2))
    for i in range(1, n):
        j = r.integers(0, i)
        for _ in range(15):
            a = r.uniform(0, 2*np.pi)
            p = pos[j] + np.array([np.cos(a), np.sin(a)]) * 0.62
            if i == 1 or min(np.hypot(*(pos[:i] - p).T)) > 0.5:
                break
        pos[i] = p
    edges = [(int(r.integers(0, i)), i) for i in range(1, n)]
    cols = [S.INDIGO]*n
    for k in r.choice(n, size=max(1, int(n*cyan_frac)), replace=False):
        cols[k] = S.CYAN
    pos -= pos.mean(0)
    return pos / max(np.abs(pos).max(), 1e-6), edges, cols

def draw(ax, cx, cy, scale, data, alpha=1.0, lw=1.6, z=2):
    pos, edges, cols = data
    P = pos*scale + np.array([cx, cy])
    for a, b in edges:
        ax.plot(*zip(P[a], P[b]), color=S.INDIGO, lw=lw, alpha=alpha,
                zorder=z, solid_capstyle="round")
    for k, (x, y) in enumerate(P):
        ax.add_patch(Circle((x, y), scale*0.20, color=cols[k], alpha=alpha, zorder=z+1, lw=0))

def from_rdkit(smiles):
    """Real molecule -> (pos, edges, cols) in the same format as graph()."""
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
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    zc = {6: S.INDIGO, 7: S.CYAN, 8: S.ORCHID, 16: "#D9A63A"}
    cols = [zc.get(a.GetAtomicNum(), S.MUTE) for a in m.GetAtoms()]
    pos -= pos.mean(0)
    return pos / max(np.abs(pos).max(), 1e-6), edges, cols

def branch_tree(ax, x, y, n, seed, spread=0.016, length=0.032, alpha=0.30, z=2):
    """Faint fan of reachable futures. Size carries value."""
    r = np.random.default_rng(seed)
    tp = np.column_stack([r.uniform(0.005, length, n), r.normal(0, spread, n)])
    for j in range(n):
        ax.plot([x, x+tp[j, 0]], [y, y+tp[j, 1]], color=S.INDIGO, lw=0.7, alpha=alpha, zorder=z)
        ax.add_patch(Circle((x+tp[j, 0], y+tp[j, 1]), 0.0034,
                            color=S.INDIGO, alpha=alpha+0.05, zorder=z+1, lw=0))
