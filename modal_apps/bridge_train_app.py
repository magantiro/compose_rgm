"""A task-independent goal-conditioned reachability potential over molecules.

WHY THIS EXISTS. QED worked because h_phi told the controller which states lead
to the target LATER -- a finite-budget reachability model, not a local score.
Every MOLLEO probe so far ran without any analogue: H40 gave more time with no
destination, top-K a narrower proposal with no destination, coverage more
diversity with no destination. ~30,000 blind molecules later, none exceeded
JNK3 0.16. That is what a missing map looks like, not a verdict on the kernel.

WHAT IS LEARNED, AND FOR FREE:

    s_psi(x, z, b) ~ log  P_theta(X_b ~ z | X_0 = x) / P(z)

x is the current molecule, z a molecular DESTINATION, b the remaining edits.
No QED, no JNK3, no MOLLEO oracle -- the labels come from hindsight on R_theta
trajectories we already paid for: in a rollout x_0..x_24, every x_{t+k} is by
construction reachable from x_t in k edits.

TWO THINGS THE LITERATURE SAYS TO GET RIGHT, AND WE DO:

  - negatives are CONTRASTIVE, never "unreachable" labels. Not observing a
    connection between two trajectories is not evidence of its absence, and
    treating it as such is a known source of severe value overestimation in
    offline goal-conditioned RL.
  - the split is by SOURCE MOLECULE. Splitting pairs would put x_t in train and
    x_{t+k} from the same rollout in validation, and the score would be read off
    memorised trajectories rather than learned reachability.

Dot-product form g(x,b) . h(z) so every other goal in the batch is a negative
at no extra cost, and the marginal P(z) cancels in the normalised successor law.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
app = modal.App("bridge-train")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
CORPUS = "hphi_rollout_corpus/train_1024x02_H24.json.gz"
EMB_DIR = "hphi_v2/embeddings"
OUT_DIR = "bridge_v1"
BUDGETS = (1, 2, 4, 8, 12, 16, 24)
VAL_FRACTION = 0.15
EMBED_DIM = 256


@app.function(image=image, cpu=(4.0, 4.0), memory=32768, timeout=4 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def assemble() -> dict[str, Any]:
    """Hindsight pairs (x_t, x_{t+b}, b) from banked trajectories. CPU only."""
    import numpy as np
    artifact_volume.reload()
    blob = json.loads(gzip.decompress(
        (Path(RUN_ROOT) / CORPUS).read_bytes()).decode())
    srcs = [r for r in blob["results"] if r.get("status") == "OK"]
    srcs.sort(key=lambda r: r["index"])
    n_val = int(len(srcs) * VAL_FRACTION)
    val_idx = {r["index"] for r in srcs[-n_val:]}      # held out BY SOURCE

    emb: dict[str, list[float]] = {}
    for f in sorted((Path(RUN_ROOT) / EMB_DIR).glob("shard_*.json.gz")):
        emb.update(json.loads(gzip.decompress(f.read_bytes()).decode()))
    print(f"{len(srcs)} sources, {len(emb):,} embeddings, "
          f"{len(val_idx)} sources held out", flush=True)

    def build(sel):
        X, Z, B = [], [], []
        for r in srcs:
            if (r["index"] in val_idx) != sel:
                continue
            for t in r["trajectories"]:
                p = t["path"]
                if any(s not in emb for s in p):
                    continue
                for i in range(len(p)):
                    for b in BUDGETS:
                        if i + b < len(p):
                            X.append(emb[p[i]]); Z.append(emb[p[i + b]])
                            B.append(b)
        return (np.asarray(X, dtype=np.float32),
                np.asarray(Z, dtype=np.float32),
                np.asarray(B, dtype=np.int64))

    Xtr, Ztr, Btr = build(False)
    Xva, Zva, Bva = build(True)
    print(f"train pairs {len(Xtr):,}   val pairs {len(Xva):,}", flush=True)
    out = Path(RUN_ROOT) / OUT_DIR / "pairs.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, Xtr=Xtr, Ztr=Ztr, Btr=Btr,
                        Xva=Xva, Zva=Zva, Bva=Bva,
                        n_val_sources=np.int64(len(val_idx)))
    artifact_volume.commit()
    return {"n_train": int(len(Xtr)), "n_val": int(len(Xva))}


@app.function(image=image, gpu="A10G", cpu=(2.0, 2.0), memory=32768,
              timeout=4 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def train(epochs: int = 30, dim: int = 128, batch: int = 512) -> dict[str, Any]:
    """GPU for the gradient steps only. Contrastive InfoNCE, in-batch negatives."""
    import numpy as np
    import torch
    import torch.nn as nn

    artifact_volume.reload()
    z = np.load(Path(RUN_ROOT) / OUT_DIR / "pairs.npz")
    Xtr, Ztr, Btr = z["Xtr"], z["Ztr"], z["Btr"]
    Xva, Zva, Bva = z["Xva"], z["Zva"], z["Bva"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nb = int(max(BUDGETS)) + 1

    class Enc(nn.Module):
        def __init__(self, extra: int):
            super().__init__()
            self.f = nn.Sequential(
                nn.Linear(EMBED_DIM + extra, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, dim))

        def forward(self, e, b=None):
            if b is not None:
                e = torch.cat([e, b], dim=-1)
            v = self.f(e)
            return v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    g, h = Enc(nb).to(dev), Enc(0).to(dev)
    opt = torch.optim.Adam(list(g.parameters()) + list(h.parameters()), lr=1e-3)
    logit_scale = nn.Parameter(torch.tensor(2.3, device=dev))
    opt.add_param_group({"params": [logit_scale]})

    def onehot(b):
        o = torch.zeros(len(b), nb, device=dev)
        o[torch.arange(len(b)), torch.as_tensor(b, device=dev)] = 1.0
        return o

    xt = torch.tensor(Xtr).to(dev); zt = torch.tensor(Ztr).to(dev)
    xv = torch.tensor(Xva).to(dev); zv = torch.tensor(Zva).to(dev)
    torch.manual_seed(0)
    best = {"val": -1.0, "state": None, "epoch": -1}
    hist = []
    t0 = time.perf_counter()
    for ep in range(epochs):
        g.train(); h.train()
        perm = torch.randperm(len(xt), device=dev)
        tot = 0.0
        for i in range(0, len(xt) - batch + 1, batch):
            idx = perm[i:i + batch]
            b = Btr[idx.cpu().numpy()]
            gv = g(xt[idx], onehot(b)); hv = h(zt[idx])
            logits = logit_scale.exp() * gv @ hv.T
            lab = torch.arange(len(idx), device=dev)
            loss = 0.5 * (nn.functional.cross_entropy(logits, lab)
                          + nn.functional.cross_entropy(logits.T, lab))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        # Validation: retrieval accuracy against 511 in-batch distractors.
        g.eval(); h.eval()
        accs, r10 = [], []
        with torch.no_grad():
            for i in range(0, len(xv) - batch + 1, batch):
                b = Bva[i:i + batch]
                gv = g(xv[i:i + batch], onehot(b)); hv = h(zv[i:i + batch])
                s = gv @ hv.T
                lab = torch.arange(len(gv), device=dev)
                accs.append(float((s.argmax(1) == lab).float().mean()))
                rk = (s >= s.gather(1, lab[:, None])).sum(1)
                r10.append(float((rk <= 10).float().mean()))
        va = float(np.mean(accs))
        hist.append({"epoch": ep, "train_loss": tot, "val_top1": va,
                     "val_top10": float(np.mean(r10))})
        if va > best["val"]:
            best = {"val": va, "epoch": ep,
                    "state": ({k: v.detach().cpu().clone() for k, v in g.state_dict().items()},
                              {k: v.detach().cpu().clone() for k, v in h.state_dict().items()})}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep {ep:>3} loss {tot:>10.1f} val_top1 {va:.4f} "
                  f"top10 {np.mean(r10):.4f}", flush=True)

    # Per-budget retrieval on the best checkpoint: the whole question is whether
    # it works at LONG range, so a pooled number would hide the answer.
    g.load_state_dict(best["state"][0]); h.load_state_dict(best["state"][1])
    g.to(dev).eval(); h.to(dev).eval()
    per_b = {}
    with torch.no_grad():
        for bb in BUDGETS:
            m = np.where(Bva == bb)[0]
            if len(m) < batch:
                continue
            acc, r10b = [], []
            for i in range(0, len(m) - batch + 1, batch):
                idx = m[i:i + batch]
                gv = g(xv[idx], onehot(Bva[idx])); hv = h(zv[idx])
                s = gv @ hv.T
                lab = torch.arange(len(gv), device=dev)
                acc.append(float((s.argmax(1) == lab).float().mean()))
                rk = (s >= s.gather(1, lab[:, None])).sum(1)
                r10b.append(float((rk <= 10).float().mean()))
            per_b[str(bb)] = {"n": int(len(m)), "top1": float(np.mean(acc)),
                              "top10": float(np.mean(r10b))}
            print(f"  b={bb:>2}  n={len(m):>7,}  top1 {np.mean(acc):.4f}  "
                  f"top10 {np.mean(r10b):.4f}", flush=True)
    p = Path(RUN_ROOT) / OUT_DIR
    torch.save({"g": g.state_dict(), "h": h.state_dict(), "dim": dim,
                "n_budget": nb, "embed_dim": EMBED_DIM}, p / "bridge.pt")
    rec = {"epochs": epochs, "best_epoch": best["epoch"],
           "best_val_top1": best["val"], "per_budget": per_b,
           "chance_top1": 1.0 / batch, "batch": batch,
           "history": hist, "seconds": round(time.perf_counter() - t0, 1)}
    (p / "BRIDGE.json").write_text(json.dumps(rec, indent=2))
    artifact_volume.commit()
    return rec


@app.function(image=image, cpu=(0.25, 0.25), memory=1024, timeout=8 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(stage: str) -> dict[str, Any]:
    out = {}
    if stage in ("assemble", "both"):
        artifact_volume.reload()
        if (Path(RUN_ROOT) / OUT_DIR / "pairs.npz").exists():
            print("pairs.npz present -- skipping assembly", flush=True)
        else:
            out["assemble"] = assemble.remote()
    if stage in ("train", "both"):
        out["train"] = train.remote()
    return out


@app.local_entrypoint()
def main(stage: str = "both") -> None:
    print("BRIDGE: goal-conditioned reachability from hindsight on banked "
          "R_theta trajectories. ZERO task-oracle calls.")
    print("Split by SOURCE. Negatives are contrastive, never 'unreachable'.")
    call = drive.spawn(stage)
    print(f"spawned: {call.object_id}")


@app.function(image=image, cpu=(4.0, 4.0), memory=32768, timeout=2 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def reeval(n_distract: int = 511, n_trials: int = 200) -> dict[str, Any]:
    """Retrieval with a FIXED protocol at every budget.

    The first evaluation drew negatives from contiguous slices of an array
    ordered by source/trajectory/position, so small budgets faced near-duplicate
    negatives from the same rollout while large budgets faced easier ones drawn
    across trajectories. Negative difficulty therefore varied with the exact
    variable under study, and the apparent improvement from b=8 to b=16 is not
    interpretable. Here every budget gets the same number of negatives, sampled
    uniformly at random from all validation goals, so the budgets are comparable.
    """
    import numpy as np
    import torch
    import torch.nn as nn

    artifact_volume.reload()
    z = np.load(Path(RUN_ROOT) / OUT_DIR / "pairs.npz")
    Xva, Zva, Bva = z["Xva"], z["Zva"], z["Bva"]
    ck = torch.load(Path(RUN_ROOT) / OUT_DIR / "bridge.pt", map_location="cpu")
    dim, nb = ck["dim"], ck["n_budget"]

    class Enc(nn.Module):
        def __init__(self, extra: int):
            super().__init__()
            self.f = nn.Sequential(
                nn.Linear(EMBED_DIM + extra, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, dim))

        def forward(self, e, b=None):
            if b is not None:
                e = torch.cat([e, b], dim=-1)
            v = self.f(e)
            return v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    g, h = Enc(nb), Enc(0)
    g.load_state_dict(ck["g"]); h.load_state_dict(ck["h"])
    g.eval(); h.eval()
    torch.set_grad_enabled(False)
    HV = h(torch.tensor(Zva))                     # all validation goals
    rng = np.random.default_rng(0)
    out: dict[str, Any] = {"n_distract": n_distract, "n_trials": n_trials,
                           "chance_top1": 1.0 / (n_distract + 1),
                           "chance_top10": 10.0 / (n_distract + 1),
                           "per_budget": {}}
    print(f"{'b':>3}{'n_pairs':>10}{'top1':>9}{'top10':>9}{'medrank':>9}"
          f"{'x chance':>10}")
    for bb in BUDGETS:
        m = np.where(Bva == bb)[0]
        if len(m) < 8:
            continue
        k = min(n_trials, len(m))
        pick = rng.choice(m, size=k, replace=False)
        oh = torch.zeros(k, nb); oh[torch.arange(k), bb] = 1.0
        gv = g(torch.tensor(Xva[pick]), oh)
        ranks = []
        for i in range(k):
            neg = rng.choice(len(Zva), size=n_distract, replace=False)
            cand = torch.cat([HV[pick[i]][None, :], HV[neg]], dim=0)
            s = cand @ gv[i]
            ranks.append(int((s >= s[0]).sum()))
        r = np.asarray(ranks)
        rec = {"n_pairs": int(len(m)), "n_eval": k,
               "top1": float((r == 1).mean()), "top10": float((r <= 10).mean()),
               "median_rank": float(np.median(r))}
        out["per_budget"][str(bb)] = rec
        print(f"{bb:>3}{len(m):>10,}{rec['top1']:>9.4f}{rec['top10']:>9.4f}"
              f"{rec['median_rank']:>9.0f}"
              f"{rec['top1']*(n_distract+1):>10.1f}")
    (Path(RUN_ROOT) / OUT_DIR / "BRIDGE_REEVAL.json").write_text(
        json.dumps(out, indent=2))
    artifact_volume.commit()
    return out
