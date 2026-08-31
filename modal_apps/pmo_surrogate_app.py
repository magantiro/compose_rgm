"""Task surrogate fitted on the SAME uncounted prescreen labels GenMol receives.

THE INFORMATION ARGUMENT. GenMol calls the task oracle on all 249,455 ZINC250k
molecules outside the counted budget and turns those labels into a task-specific
fragment vocabulary. Our matched-information lane so far used those labels only
to pick 100 starting molecules, throwing away 249,355 of them. That is not a
matched lane, it is a strictly weaker one.

This fits a lightweight ranking surrogate f_hat(x) on the same labels, so COMPOSE
consumes the same task information through its own representation instead of
through a fragment vocabulary. Nothing here touches the counted budget.

WHAT WOULD MAKE THIS USELESS. If f_hat cannot RANK held-out molecules, guiding
search with it is worse than useless -- it would confidently mislead. So the
held-out check is ranking-first: Spearman over the held-out set, and enrichment
at the top, which is what search actually consumes. A model with good MSE and bad
top-decile enrichment fails this test.

The objective-blind clean lane never sees any of this and is reported separately.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume,
)
from modal_apps.genmol_t4_opt_app import image as _opt_image

image = _opt_image.add_local_file(
    ROOT / "modal_apps/genmol_t4_opt_app.py",
    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)

app = modal.App("pmo-surrogate")


def _spearman(a, b):
    """Rank correlation, numpy only (scipy is not in this image)."""
    import numpy as np
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


@app.function(image=image, cpu=(4.0, 4.0), memory=int(16 * 1024),
              timeout=2 * 60 * 60, volumes={str(ARTIFACT_ROOT): artifact_volume})
def fit(task: str = "scaffold_hop", n_train: int = 50000, n_test: int = 10000,
        seed: int = 20260822) -> dict[str, Any]:
    import os, sys
    import numpy as np
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    import torch

    t0 = time.time()
    artifact_volume.reload()
    z = np.load(f"/artifacts/pmo_matched_init/labels_{task}.npz", allow_pickle=True)
    smiles, score = z["smiles"], z["score"].astype(np.float64)
    print(f"  {len(smiles):,} labels  score min {score.min():.4f} "
          f"median {np.median(score):.4f} max {score.max():.4f}", flush=True)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(smiles))[: n_train + n_test]
    tr, te = idx[:n_train], idx[n_train:]

    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    def feat(ix):
        X = np.zeros((len(ix), 2048), dtype=np.float32)
        for j, i in enumerate(ix):
            m = Chem.MolFromSmiles(str(smiles[i]))
            if m is None:
                continue
            a = np.zeros((2048,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), a)
            X[j] = a
        return X

    tf = time.time()
    Xtr, ytr = feat(tr), score[tr]
    Xte, yte = feat(te), score[te]
    print(f"  featurised {len(tr):,}+{len(te):,} in {time.time()-tf:.0f}s", flush=True)

    def report(name, pred, secs):
        sp = _spearman(yte, pred)
        k = max(1, len(yte) // 100)                       # top 1%
        true_top = set(np.argsort(-yte)[:k].tolist())
        pred_top = np.argsort(-pred)[:k].tolist()
        hit = len(true_top & set(pred_top)) / k
        enrich = hit / (k / len(yte))
        # what search actually consumes: the true quality of what the model ranks first
        mean_true_at_top = float(yte[pred_top].mean())
        r = {"model": name, "spearman": round(sp, 4),
             "top1pct_recall": round(hit, 4), "top1pct_enrichment": round(enrich, 1),
             "mean_true_score_of_predicted_top1pct": round(mean_true_at_top, 4),
             "held_out_mean": round(float(yte.mean()), 4),
             "held_out_max": round(float(yte.max()), 4),
             "seconds": round(secs, 1)}
        print(f"  {name:8s} spearman {sp:.4f}  top1% recall {hit:.3f}  "
              f"enrichment {enrich:.1f}x  mean-true-of-top1% {mean_true_at_top:.4f} "
              f"(held-out mean {yte.mean():.4f})", flush=True)
        return r

    out = {"task": task, "n_labels": int(len(smiles)),
           "n_train": int(len(tr)), "n_test": int(len(te)), "models": []}

    # ---- ridge, closed form ----
    tr0 = time.time()
    A = Xtr.T @ Xtr + 1.0 * np.eye(2048, dtype=np.float32)
    w = np.linalg.solve(A, Xtr.T @ (ytr - ytr.mean()).astype(np.float32))
    out["models"].append(report("ridge", Xte @ w + ytr.mean(), time.time() - tr0))

    # ---- small MLP ----
    tr0 = time.time()
    torch.manual_seed(seed); torch.set_num_threads(4)
    net = torch.nn.Sequential(
        torch.nn.Linear(2048, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr.astype(np.float32)).view(-1, 1)
    mu, sd = float(yt.mean()), float(yt.std()) or 1.0
    yn = (yt - mu) / sd
    for ep in range(25):
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xt[b]), yn[b])
            loss.backward(); opt.step(); tot += float(loss) * len(b)
        if (ep + 1) % 5 == 0:
            print(f"    mlp epoch {ep+1:>2}  train mse {tot/len(Xt):.5f}", flush=True)
    with torch.no_grad():
        pm = (net(torch.from_numpy(Xte)).view(-1).numpy() * sd + mu)
    out["models"].append(report("mlp", pm, time.time() - tr0))

    best = max(out["models"], key=lambda m: m["spearman"])
    out["chosen"] = best["model"]
    out["verdict"] = ("USABLE for search guidance" if best["spearman"] >= 0.5
                      else "NOT usable as a ranker; guiding search with this would mislead")
    out["seconds"] = round(time.time() - t0, 1)

    d = Path("/artifacts/pmo_surrogate"); d.mkdir(parents=True, exist_ok=True)
    (d / f"fit_{task}.json").write_text(json.dumps(out, indent=1))
    if best["model"] == "ridge":
        np.savez_compressed(d / f"model_{task}.npz", kind=np.array(["ridge"]),
                            w=w, b=np.array([ytr.mean()], dtype=np.float32))
    else:
        torch.save(net.state_dict(), d / f"model_{task}.pt")
        np.savez_compressed(d / f"model_{task}.npz", kind=np.array(["mlp"]),
                            mu=np.array([mu]), sd=np.array([sd]))
    artifact_volume.commit()
    print(f"\n  CHOSEN {best['model']}  spearman {best['spearman']:.4f}  -> {out['verdict']}",
          flush=True)
    return out


@app.function(image=image, cpu=(4.0, 4.0), memory=int(16 * 1024),
              timeout=60 * 60, volumes={str(ARTIFACT_ROOT): artifact_volume})
def ood(task: str = "scaffold_hop", corpus_max: float = 0.5261) -> dict[str, Any]:
    """Score the FROZEN prescreen surrogate on COMPOSE-generated molecules.

    THESE LABELS ARE NEVER TRAINED ON. They come from the oracle-greedy
    diagnostic, which spent true-oracle calls off-budget; folding them into the
    surrogate would hand the matched lane task information beyond the 249,455
    prescreen labels GenMol receives, and the lane would no longer be matched.
    They are used here strictly as a held-out diagnostic.

    The question is narrow and is the one that decides whether C is worth
    launching: a random ZINC split says the surrogate ranks the CORPUS well, but
    search consumes molecules COMPOSE invents. The decisive slice is the
    candidates scoring ABOVE the corpus maximum, because that is precisely where
    the surrogate has no training signal and where the whole gap to GenMol lives.
    """

    import os, sys
    import numpy as np
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    import torch

    t0 = time.time()
    artifact_volume.reload()
    z = np.load(f"/artifacts/pmo_ceiling/ood_{task}.npz", allow_pickle=True)
    smi, y, dep, par = z["smiles"], z["score"].astype(np.float64), z["depth"], z["parent"]
    print(f"  {len(smi):,} COMPOSE-generated molecules, "
          f"{(y > corpus_max).sum():,} above the {corpus_max} corpus max", flush=True)

    meta = np.load(f"/artifacts/pmo_surrogate/model_{task}.npz", allow_pickle=True)
    kind = str(meta["kind"][0])
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    X = np.zeros((len(smi), 2048), dtype=np.float32)
    for j, s in enumerate(smi):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        a = np.zeros((2048,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m), a)
        X[j] = a
    if kind == "mlp":
        net = torch.nn.Sequential(
            torch.nn.Linear(2048, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        net.load_state_dict(torch.load(f"/artifacts/pmo_surrogate/model_{task}.pt"))
        net.eval()
        with torch.no_grad():
            pred = net(torch.from_numpy(X)).view(-1).numpy() * float(meta["sd"][0]) + float(meta["mu"][0])
    else:
        pred = X @ meta["w"] + float(meta["b"][0])

    def block(name, mask):
        n = int(mask.sum())
        if n < 20:
            return {"slice": name, "n": n, "note": "too few to judge"}
        a, b = y[mask], pred[mask]
        k = max(1, n // 10)
        true_top = set(np.argsort(-a)[:k].tolist())
        rec = len(true_top & set(np.argsort(-b)[:k].tolist())) / k
        r = {"slice": name, "n": n, "spearman": round(_spearman(a, b), 4),
             "top10pct_recall": round(rec, 4),
             "true_mean": round(float(a.mean()), 4), "true_max": round(float(a.max()), 4)}
        print(f"  {name:28s} n={n:>6,}  spearman {r['spearman']:>7.4f}  "
              f"top10% recall {rec:.3f}  (true max {a.max():.4f})", flush=True)
        return r

    out = {"task": task, "n": int(len(smi)), "corpus_max": corpus_max,
           "trained_on_these": False, "slices": []}
    out["slices"].append(block("ALL COMPOSE successors", np.ones(len(y), bool)))
    out["slices"].append(block("below corpus max", y <= corpus_max))
    out["slices"].append(block("ABOVE corpus max", y > corpus_max))
    for d in sorted(set(dep.tolist())):
        out["slices"].append(block(f"depth {d}", dep == d))

    # within-fiber ranking: what the controller actually consumes each round
    fib = {}
    for i, pth in enumerate(par):
        fib.setdefault(str(pth), []).append(i)
    sps, ranks = [], []
    for _, ix in fib.items():
        if len(ix) < 10:
            continue
        ix = np.array(ix)
        sps.append(_spearman(y[ix], pred[ix]))
        order = ix[np.argsort(-pred[ix])]
        ranks.append(int(np.where(order == ix[np.argmax(y[ix])])[0][0]) + 1)
    out["within_fiber"] = {
        "n_fibers": len(sps),
        "median_spearman": round(float(np.median(sps)), 4) if sps else None,
        "median_rank_of_true_best": float(np.median(ranks)) if ranks else None,
        "median_pct_rank_of_true_best": round(
            100 * float(np.median(ranks)) / (len(y) / max(len(fib), 1)), 2) if ranks else None}
    print(f"  within-fiber: {len(sps)} fibers, median spearman "
          f"{out['within_fiber']['median_spearman']}, median rank of the true-best "
          f"successor {out['within_fiber']['median_rank_of_true_best']}", flush=True)

    ab = next((s for s in out["slices"] if s["slice"] == "ABOVE corpus max"), {})
    sp_ab = ab.get("spearman")
    out["verdict"] = (
        "extrapolation NOT trustworthy; C must adapt online" if sp_ab is None or sp_ab < 0.4
        else "extrapolation holds above the corpus max; frozen surrogate may suffice")
    out["seconds"] = round(time.time() - t0, 1)
    d_ = Path("/artifacts/pmo_surrogate"); d_.mkdir(parents=True, exist_ok=True)
    (d_ / f"ood_{task}.json").write_text(json.dumps(out, indent=1))
    artifact_volume.commit()
    print(f"\n  VERDICT: {out['verdict']}", flush=True)
    return out


@app.local_entrypoint()
def main(task: str = "scaffold_hop", n_train: int = 50000, n_test: int = 10000,
         out: str = "") -> None:
    o = fit.remote(task, n_train, n_test)
    p = Path(__file__).resolve().parents[1] / (out or f"diagnostics/pmo_surrogate_{task}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
