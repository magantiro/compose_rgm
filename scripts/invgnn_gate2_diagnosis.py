"""Gate 2 as a DIAGNOSIS, not a retrain.

The previous h_phi lost to a trivial persistence baseline (Spearman 0.487 vs
0.680). The brute-force lookahead measurement has already ruled out the benign
explanation -- at b=8 the immediate score has median rank correlation 0.32-0.37
with TRUE future value -- so the amortiser is at fault and the amendment's gate
rule directs us to troubleshoot the value-learning implementation.

This runs four arms that differ in EXACTLY one thing each, so the comparison
identifies the fault instead of merely producing a better number:

  1 persistence   pred = g_lambda(x).  No training.  The baseline that won.
  2 emb           [R_theta embedding | pref | budget].  Reproduces the FAILED
                  configuration exactly, so arm 2 vs arm 1 is the original loss.
  3 emb+base      arm 2 plus F_psi_hat(x) and g_lambda(x).  Tests the FEATURE
                  hypothesis: the R_theta encoder was trained to predict
                  transitions, not properties, while g is a function of Morgan
                  bits.  Handing the model the baseline makes the task learning
                  the CORRECTION rather than rediscovering it.
  4 emb+base/log  arm 3 trained on log target.  Tests the SCALE hypothesis that
                  Addendum IV flagged: max/median ~190, so plain MSE is
                  dominated by a handful of states.  EXPLORATORY and labelled as
                  such -- it is reported as a finding to rule on, never
                  substituted for the preregistered arm.

Arms 2-4 share seed, architecture, optimiser, schedule and split. Only the named
factor moves.

Splitting is BY STATE into train/val/test. A state contributes 40 (budget,
preference) rows, so a random row split would put near-identical rows on both
sides and every arm would look excellent.

Reported metrics are Spearman AND R^2, per Addendum IV, because the target's
tail makes R^2 alone misleading. No Bellman residual is reported: the corpus
stores rollout TERMINALS, not the successors of each training state, so
h_b(x) = E_{y~R_theta}[h_{b-1}(y)] is not checkable from it. Saying so is
cheaper than faking it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
NPREF, BMAX, EMB = 5, 8, 256


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    ss = float(((y - y.mean()) ** 2).sum())
    return float("nan") if ss == 0 else 1.0 - float(((pred - y) ** 2).sum()) / ss


def train(Xtr, ytr, Xva, yva, seed=20260820, epochs=200, patience=20):
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Linear(Xtr.shape[1], 512), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-5)
    xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    xv, yv = torch.tensor(Xva), torch.tensor(yva)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 512):
            j = perm[i:i + 512]
            loss = ((head(xt[j]).squeeze(1) - yt[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            v = float(((head(xv).squeeze(1) - yv) ** 2).mean())
        if v < best - 1e-9:
            best, best_state, bad = v, {k: t.clone() for k, t in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out", default="diagnostics/invgnn_gate2_diagnosis.json")
    a = ap.parse_args()

    C = json.loads(Path(a.corpus).read_text())
    E = np.load(a.embeddings)
    budgets = C["budgets"]
    rows = [r for r in C["results"] if r.get("status") == "OK" and r["state"] in E.files]
    print(f"states {len(rows)}   budgets {budgets}   M {C['m_roll']}")

    S, B, P, Y, EMBV, FPSI, GV = [], [], [], [], [], [], []
    for r in rows:
        v = E[r["state"]].astype(np.float32)
        emb, fps, gl = v[:EMB], v[EMB:EMB + 2], v[EMB + 2:]
        h = np.asarray(r["g_rollout"], dtype=np.float64).mean(0)      # (|b|, 5)
        for bi, b in enumerate(budgets):
            for p in range(NPREF):
                S.append(r["state"]); B.append(b); P.append(p)
                Y.append(h[bi, p]); EMBV.append(emb); FPSI.append(fps); GV.append(gl)
    S = np.array(S); B = np.array(B); P = np.array(P)
    Y = np.array(Y, dtype=np.float32)
    EMBV = np.stack(EMBV); FPSI = np.stack(FPSI); GV = np.stack(GV)
    onehot = lambda idx, n: np.eye(n, dtype=np.float32)[idx]
    ctx = np.concatenate([onehot(P, NPREF), onehot(B - 1, BMAX)], 1)
    persistence = GV[np.arange(len(P)), P].astype(np.float32)         # g_lambda(x)

    # split BY STATE, 70/15/15
    states = sorted(set(S))
    hsh = lambda s: (int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 1000) / 1000.0
    role = {s: ("test" if hsh(s) < .15 else "val" if hsh(s) < .30 else "train") for s in states}
    m = {k: np.array([role[s] == k for s in S]) for k in ("train", "val", "test")}
    print(f"states train/val/test "
          f"{sum(role[s]=='train' for s in states)}/"
          f"{sum(role[s]=='val' for s in states)}/{sum(role[s]=='test' for s in states)}"
          f"   rows {m['train'].sum():,}/{m['val'].sum():,}/{m['test'].sum():,}")

    FEATS = {
        "emb":      np.concatenate([EMBV, ctx], 1),
        "emb+base": np.concatenate([EMBV, FPSI, GV, ctx], 1),
    }
    preds: dict[str, np.ndarray] = {"1 persistence": persistence}
    for name, key, logt in (("2 emb", "emb", False),
                            ("3 emb+base", "emb+base", False),
                            ("4 emb+base/log [exploratory]", "emb+base", True)):
        X = FEATS[key]
        yy = np.log(Y) if logt else Y
        head = train(X[m["train"]], yy[m["train"]], X[m["val"]], yy[m["val"]])
        with torch.no_grad():
            p_all = head(torch.tensor(X)).squeeze(1).numpy()
        preds[name] = np.exp(p_all) if logt else p_all
        print(f"  trained {name}  (width {X.shape[1]})")

    out: dict = {"n_states": len(states), "budgets": budgets,
                 "split": {k: int(sum(role[s] == k for s in states)) for k in ("train", "val", "test")},
                 "note": "Test rows only. Spearman is primary; R^2 is reported "
                         "alongside because the target's tail makes it fragile. "
                         "Arm 4 is exploratory and is not a substitute for arm 3.",
                 "arms": {}}
    te = m["test"]
    print(f"\n{'arm':<32}{'rho all':>9}{'R2 all':>9}   " + "".join(f"{'b='+str(b):>7}" for b in budgets))
    for name, pr in preds.items():
        rho_all, r2_all = spearman(pr[te], Y[te]), r2(pr[te], Y[te])
        per_b = [spearman(pr[te & (B == b)], Y[te & (B == b)]) for b in budgets]
        per_p = [spearman(pr[te & (P == p)], Y[te & (P == p)]) for p in range(NPREF)]
        out["arms"][name] = {"spearman_all": rho_all, "r2_all": r2_all,
                             "spearman_by_budget": dict(zip(map(str, budgets), per_b)),
                             "spearman_by_preference": per_p}
        print(f"{name:<32}{rho_all:>9.3f}{r2_all:>9.3f}   "
              + "".join(f"{v:>7.3f}" for v in per_b))

    Path(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ROOT / a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")

    b0 = out["arms"]["1 persistence"]["spearman_all"]
    print(f"\nGATE 2 READ: an arm must beat persistence ({b0:.3f}) to be worth keeping.")
    for name in list(preds)[1:]:
        d = out["arms"][name]["spearman_all"] - b0
        print(f"  {name:<32}{d:+.3f}  {'BEATS' if d > 0 else 'loses to'} persistence")


if __name__ == "__main__":
    main()
