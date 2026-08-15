"""Train the REAL universal region-h_phi on the frozen 1,024x2xH24 corpus.

Corpus: sha256 647f8265f5143e2b15dc047de42e837203593c42beb7e45e2b94e55e41602581
R_theta: c979cdb3d7b0b403bfbf7bfb0aa5098b2588c6d4217770c2c58292b7c4e53de8 FROZEN

    h_b(x, z) = P( exists t <= b : X_t in B_z | X_0 = x )

Native finite-budget HITTING reachability, over the 20 preregistered regions and
budgets 0..24, with the exact boundary condition h = 1 whenever x is already in
B_z -- enforced, never learned, so in-region states are excluded from the
training target rather than labelled.

TWO ESTIMATORS OF THE SAME QUANTITY, as preregistered:

  MC future-event   every prefix takes the hitting outcome from that prefix
                    forward within its remaining budget; cross-entropy has the
                    true reachability probability as its population optimum

  Bellman           h_b(x) ~ E_{y ~ R_theta}[ h_{b-1}(y) ] on observed
                    transitions, with a STOPPED-GRADIENT target so the backup
                    cannot chase itself

R_theta is never touched. Only the head is trained; if capacity is short the
CONTROLLER grows, never the molecular dynamics.

INSTRUMENTATION. Counters are recorded now so the three resource axes --
performance, adaptation cost, inference cost -- can later be reported
separately rather than argued verbally.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"}
).add_local_dir(ROOT / "data/jin", str(REMOTE_ROOT / "data/jin"), copy=True)

app = modal.App("hphi-train")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
CORPUS = "hphi_rollout_corpus/train_1024x02_H24.json.gz"
OUT_DIR = "hphi_v2"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
ENCODE_SHARDS = 80
#: Held out BY SOURCE, so no state of a validation source is ever trained on.
VAL_FRACTION = 0.15
EPOCHS = 40
#: Stop when validation has not improved for this many epochs. The first run
#: overfit from epoch 0 and, having no best-checkpoint tracking, would have
#: persisted epoch 59 -- strictly worse than epoch 0 on held-out sources.
PATIENCE = 5
BELLMAN_WEIGHT = 0.3

_RT: dict[str, Any] = {}


def _runtime():
    if "model" in _RT:
        return _RT
    import sys

    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state,
    )
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME,
    )

    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    src = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]), repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        src, materialized_state=bundle)
    model = runtime.model
    ck = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["selected_model_state"], strict=True)
    model.eval(); torch.set_grad_enabled(False); torch.set_num_threads(1)
    _RT["model"] = model
    return _RT


@app.function(image=image, cpu=(1.0, 1.0), memory=8192, timeout=4 * 60 * 60,
              max_containers=ENCODE_SHARDS, retries=3,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def encode_shard(smiles: list[str]) -> dict[str, list[float]]:
    import sys

    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    model = _runtime()["model"]
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import _one_state_batch

    out: dict[str, list[float]] = {}
    for smi in smiles:
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            b = _one_state_batch(model, st, float(TIME_POINT), prepared_batch=None)
            with torch.no_grad():
                _n, g, _p = model._encode_batch(b)
            out[smi] = g[0].detach().cpu().numpy().tolist()
        except Exception as exc:  # noqa: BLE001
            print(f"encode failed {type(exc).__name__}", flush=True)
    return out


@app.function(image=image, cpu=(4.0, 4.0), memory=32768, timeout=8 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def train() -> dict[str, Any]:
    import sys

    import numpy as np
    import torch
    import torch.nn as nn

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.hphi_region_features import (
        BUDGET_MAX, INPUT_DIM, build_features, in_region,
    )
    from compose_v4.experiments.hphi_rollout import registered_regions

    t_start = time.perf_counter()
    artifact_volume.reload()
    blob = json.loads(gzip.decompress(
        (Path(RUN_ROOT) / CORPUS).read_bytes()).decode())
    srcs = [r for r in blob["results"] if r.get("status") == "OK"]
    srcs.sort(key=lambda r: r["index"])
    n_val = int(len(srcs) * VAL_FRACTION)
    val_src = {r["index"] for r in srcs[-n_val:]}       # held out BY SOURCE
    print(f"{len(srcs)} sources; {len(val_src)} held out by source", flush=True)

    need = sorted({s for r in srcs for t in r["trajectories"] for s in t["path"]})
    print(f"encoding {len(need):,} unique states across {ENCODE_SHARDS} shards",
          flush=True)
    # Embeddings come from the SEPARATE encode job (hphi_encode_app.py), which
    # persists per shard. This trainer never encodes -- that coupling is what
    # destroyed 93.5 core-hours twice.
    t_enc = time.perf_counter()
    sd = Path(RUN_ROOT) / "hphi_v2/embeddings"
    parts = sorted(sd.glob("shard_*.json.gz")) if sd.exists() else []
    if not parts:
        raise SystemExit(
            "no persisted embeddings. Run modal_apps/hphi_encode_app.py first; "
            "it is idempotent and resumable.")
    emb: dict[str, list[float]] = {}
    for f in parts:
        emb.update(json.loads(gzip.decompress(f.read_bytes()).decode()))
    enc_seconds = time.perf_counter() - t_enc
    missing = [s for s in need if s not in emb]
    print(f"loaded {len(emb):,} embeddings from {len(parts)} shards in "
          f"{enc_seconds:.0f}s; {len(missing):,} states missing", flush=True)
    if missing:
        print("  (states without embeddings are skipped, not re-encoded here)",
              flush=True)

    regions = registered_regions()
    Xtr, Ytr, Xva, Yva, Mva = [], [], [], [], []
    # Bellman pairs: (state, successor, budget, region) on OBSERVED transitions
    Btr = []
    for r in srcs:
        e_src = emb.get(r["source"])
        if e_src is None:
            continue
        e_src = np.asarray(e_src)
        is_val = r["index"] in val_src
        for t in r["trajectories"]:
            path, qed, sim = t["path"], t["qed"], t["similarity_to_source"]
            if any(s not in emb for s in path):
                continue
            H = len(path) - 1
            for i, s in enumerate(path):
                b = H - i
                if b > BUDGET_MAX:
                    continue
                for reg in regions:
                    if in_region(qed[i], sim[i], reg):
                        continue                # boundary: h = 1, not learned
                    hit = 0
                    for k in range(i, len(path)):
                        if (k - i) <= b and qed[k] >= reg[0] and sim[k] >= reg[1]:
                            hit = 1
                            break
                    f = build_features(np.asarray(emb[s]), e_src,
                                       qed[i], sim[i], reg, b)
                    if is_val:
                        Xva.append(f); Yva.append(hit); Mva.append((reg, b))
                    else:
                        Xtr.append(f); Ytr.append(hit)
                        if i + 1 < len(path) and b >= 1 and not in_region(
                                qed[i + 1], sim[i + 1], reg):
                            Btr.append(build_features(
                                np.asarray(emb[path[i + 1]]), e_src,
                                qed[i + 1], sim[i + 1], reg, b - 1))
    Xtr = np.asarray(Xtr, dtype=np.float32); Ytr = np.asarray(Ytr, dtype=np.float32)
    Xva = np.asarray(Xva, dtype=np.float32); Yva = np.asarray(Yva, dtype=np.float32)
    Btr = np.asarray(Btr, dtype=np.float32)
    print(f"train {len(Xtr):,}  val {len(Xva):,}  bellman pairs {len(Btr):,}",
          flush=True)

    torch.manual_seed(0)
    head = nn.Sequential(nn.Linear(INPUT_DIM, 512), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    xt = torch.tensor(Xtr); yt = torch.tensor(Ytr).unsqueeze(1)
    mu, sd = xt.mean(0, keepdim=True), xt.std(0, keepdim=True) + 1e-6
    xt = (xt - mu) / sd
    bt = (torch.tensor(Btr) - mu) / sd if len(Btr) else None
    xv = (torch.tensor(Xva) - mu) / sd; yv = torch.tensor(Yva).unsqueeze(1)

    t_tr = time.perf_counter()
    hist = []
    best = {"val": float("inf"), "epoch": -1, "state": None}
    torch.set_grad_enabled(True)
    for ep in range(EPOCHS):
        head.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), 4096):
            idx = perm[i:i + 4096]
            opt.zero_grad()
            loss = lossf(head(xt[idx]), yt[idx])
            if bt is not None and BELLMAN_WEIGHT > 0:
                j = idx[idx < len(bt)]
                if len(j):
                    with torch.no_grad():                 # STOPPED GRADIENT
                        tgt = torch.sigmoid(head(bt[j]))
                    loss = loss + BELLMAN_WEIGHT * nn.functional.binary_cross_entropy_with_logits(
                        head(xt[j]), tgt)
            loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        head.eval()
        with torch.no_grad():
            vl = float(lossf(head(xv), yv))
        hist.append({"epoch": ep, "train": tot / len(xt), "val": vl})
        if vl < best["val"]:
            best = {"val": vl, "epoch": ep,
                    "state": {k: v.detach().clone()
                              for k, v in head.state_dict().items()}}
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  ep {ep:>3}  train {tot/len(xt):.4f}  val {vl:.4f}"
                  f"  best {best['val']:.4f}@{best['epoch']}", flush=True)
        if ep - best["epoch"] >= PATIENCE:
            print(f"  EARLY STOP at ep {ep}; no val improvement for "
                  f"{PATIENCE} epochs", flush=True)
            break
    torch.set_grad_enabled(False)
    # RESTORE THE BEST-VALIDATION WEIGHTS. The final epoch is not the model.
    if best["state"] is not None:
        head.load_state_dict(best["state"])
        print(f"restored best checkpoint: epoch {best['epoch']}, "
              f"val {best['val']:.4f}", flush=True)
    head.eval()
    train_seconds = time.perf_counter() - t_tr

    with torch.no_grad():
        pv = torch.sigmoid(head(xv)).squeeze(1).numpy()
    base = float(Yva.mean())
    brier = float(np.mean((pv - Yva) ** 2))
    brier_const = float(np.mean((base - Yva) ** 2))
    by_q: dict[float, list[float]] = {}
    for p, (reg, b) in zip(pv, Mva):
        if abs(reg[1] - 0.40) < 1e-9:
            by_q.setdefault(reg[0], []).append(float(p))
    means = {f"{q:.2f}": float(np.mean(v)) for q, v in sorted(by_q.items())}

    out_p = Path(RUN_ROOT) / OUT_DIR
    out_p.mkdir(parents=True, exist_ok=True)
    head_s = torch.jit.script(head.eval())
    torch.jit.save(head_s, str(out_p / "head.pt"))
    (out_p / "norm.json").write_text(json.dumps(
        {"mu": mu.squeeze(0).tolist(), "sd": sd.squeeze(0).tolist()}))
    rec = {
        "schema": "compose.hphi.v2",
        "corpus_sha256": "647f8265f5143e2b15dc047de42e837203593c42beb7e45e2b94e55e41602581",
        "r_theta_sha256": "c979cdb3d7b0b403bfbf7bfb0aa5098b2588c6d4217770c2c58292b7c4e53de8",
        "r_theta_retrained": False,
        "target": "finite-budget HITTING reachability; boundary h=1 enforced",
        "n_train": len(Xtr), "n_val": len(Xva), "n_bellman": len(Btr),
        "val_sources": len(val_src), "held_out_by": "source",
        "epochs_max": EPOCHS, "patience": PATIENCE,
        "epochs_run": len(hist),
        "selected_epoch": best["epoch"], "selected_val_bce": best["val"],
        "bellman_weight": BELLMAN_WEIGHT,
        "history": hist,
        "val_base_rate": base,
        "val_brier": brier, "val_brier_constant": brier_const,
        "beats_constant": bool(brier < brier_const),
        "mean_pred_by_qed_at_sim040": means,
        "instrumentation": {
            "encode_seconds": round(enc_seconds, 1),
            "train_seconds": round(train_seconds, 1),
            "total_seconds": round(time.perf_counter() - t_start, 1),
            "unique_states_encoded": len(emb),
        },
    }
    (out_p / "HPHI_V2.json").write_text(json.dumps(rec, indent=2))
    artifact_volume.commit()
    print(f"\nval Brier {brier:.5f} vs constant {brier_const:.5f} "
          f"-> beats_constant={rec['beats_constant']}")
    print(f"mean pred by QED threshold at sim>=0.40: {means}")
    print(f"encode {enc_seconds:.0f}s  train {train_seconds:.0f}s")
    return rec


@app.local_entrypoint()
def main() -> None:
    print("Training the REAL region-h_phi on the frozen 1,024x2xH24 corpus.")
    print("R_theta FROZEN. Target: finite-budget hitting reachability.")
    print("MC future-event + Bellman consistency with stopped-gradient backup.")
    call = train.spawn()
    print(f"spawned: {call.object_id}")
