"""Train the multi-objective future-value model h_phi(x, lambda, b).

The target is a SOFT expectation, h_b(x,lambda) = E_{R_theta}[g_lambda(X_b)|X_0=x],
not a hitting probability, so this is a regression and not the binary
cross-entropy fit used by the QED reachability head.

M=32 rollout means are noisy but UNBIASED draws of that conditional mean.  We do
not require any individual label to be highly reliable; the regression pools
across states.  Per-target Monte Carlo variance is available and is used only as
an optional precision weight, never to redesign the model.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np, torch, torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
EMBED, NPREF = 256, 5
BMAX = 8                      # committed development horizon


def featurize(emb: np.ndarray, pref_onehot: np.ndarray, b: int) -> np.ndarray:
    """[graph embedding | preference one-hot | budget one-hot]."""
    bb = np.zeros(BMAX + 1, dtype=np.float32); bb[b] = 1.0
    return np.concatenate([emb, pref_onehot, bb]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="docs/INVGNN_HPHI_CORPUS.json")
    ap.add_argument("--embeddings", default="docs/INVGNN_HPHI_EMBEDDINGS.npz")
    ap.add_argument("--out", default="docs/INVGNN_HPHI_HEAD.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--precision-weight", action="store_true")
    a = ap.parse_args()

    C = json.loads((ROOT / a.corpus).read_text())
    E = np.load(ROOT / a.embeddings)
    rows = C["results"]
    budgets = C["budgets"]

    X, Y, W, SRC = [], [], [], []
    for r in rows:
        if r.get("status") != "OK" or r["state"] not in E:
            continue
        emb = E[r["state"]].astype(np.float32)
        G = np.asarray(r["g_rollout"], dtype=np.float64)      # (M, |b|, 5)
        mean, var = G.mean(0), G.var(0, ddof=1)
        for bi, b in enumerate(budgets):
            for p in range(NPREF):
                oh = np.zeros(NPREF, dtype=np.float32); oh[p] = 1.0
                X.append(featurize(emb, oh, b))
                Y.append(mean[bi, p])
                W.append(1.0 / (1.0 + var[bi, p] / G.shape[0]))
                SRC.append(r["state"])
    X = np.stack(X); Y = np.array(Y, dtype=np.float32)
    W = np.array(W, dtype=np.float32) if a.precision_weight else np.ones_like(Y)
    print(f"targets {len(Y):,} from {len(set(SRC)):,} states  |  width {X.shape[1]}")

    # split BY STATE: a state contributes many (b, lambda) rows, so a random row
    # split would put near-identical rows on both sides.
    states = sorted(set(SRC))
    h = lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
    val_states = {s for s in states if (h(s) % 1000) / 1000.0 < a.val_frac}
    m_val = np.array([s in val_states for s in SRC])
    print(f"val states {len(val_states)} / {len(states)}  |  val rows {m_val.sum():,}")

    xt = torch.tensor(X[~m_val]); yt = torch.tensor(Y[~m_val]); wt = torch.tensor(W[~m_val])
    xv = torch.tensor(X[m_val]);  yv = torch.tensor(Y[m_val])

    torch.manual_seed(20260819)
    head = nn.Sequential(nn.Linear(X.shape[1], 512), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state, bad = float("inf"), None, 0
    n = len(yt)
    for ep in range(a.epochs):
        head.train(); perm = torch.randperm(n)
        for i in range(0, n, 512):
            j = perm[i:i + 512]
            pred = head(xt[j]).squeeze(1)
            loss = (wt[j] * (pred - yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            v = float(((head(xv).squeeze(1) - yv) ** 2).mean())
        if v < best - 1e-6:
            best, best_state, bad = v, {k: t.clone() for k, t in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stop at epoch {ep}"); break
        if ep % 10 == 0:
            print(f"  epoch {ep:3d}  val MSE {v:.6f}  (best {best:.6f})")
    head.load_state_dict(best_state)
    torch.save({"state": head.state_dict(), "in_dim": int(X.shape[1]),
                "bmax": BMAX, "npref": NPREF, "val_mse": best,
                "val_states": sorted(val_states)}, ROOT / a.out)
    print(f"saved {a.out}  |  best val MSE {best:.6f}")


if __name__ == "__main__":
    main()
