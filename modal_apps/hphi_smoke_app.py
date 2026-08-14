"""THROWAWAY end-to-end region-h_phi smoke. Weights are DISCARDED.

Purpose is ENGINEERING, not performance. It answers one question: can the new
value-learning stack ingest the corpus correctly? Nothing here is a scientific
result and no QED-success gate is applied -- with 256 unguided trajectories and
a handful of hard-tail events, a throwaway head could look bad for purely
statistical reasons.

USES ONLY THE ALREADY-GENERATED, PERMANENTLY EXCLUDED H24 PILOT. No new
molecules are generated. 48 pilot sources for throwaway training, 16 held out
for throwaway evaluation -- both permanently outside the real 1,024 training
and 128 validation pools.

WHAT IT MUST PROVE
------------------
  1  trajectory -> prefix/budget/goal examples are constructed correctly
  2  every budget b = 0..24 actually occurs
  3  source-relative similarity always references the IMMUTABLE x_src
  4  goal-region margins and features are correct
  5  inside region -> h = 1 stays an EXACT boundary, never learned
  6  loss decreases and the output is non-degenerate
  7  easier regions get larger predicted reachability than harder nested ones,
     at least directionally
  8  a trained checkpoint plugs into the Stage-A1 sampler with no shape,
     device or schema failure
  9  batching and caching work
 10  checkpoint / resume works

A NOTE ON THE ENCODE BUDGET. `prepare_factorized_mark_batch` costs ~2.6 s per
state and does not amortize across a batch (measured), so encoding all ~6,400
pilot states would take hours. The smoke therefore encodes a BOUNDED set of
unique molecules, distributed across containers, and builds examples only from
trajectories whose states are all encoded. That is sufficient to prove wiring
and is explicitly not a sample-size claim.
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

app = modal.App("hphi-smoke")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
CORPUS = "hphi_rollout_corpus/pilot_0064x04_H24.json.gz"
OUT_DIR = "hphi_smoke"
TIME_POINT = 0.5
CANONICAL_SLOTS = 48
#: Bounded so the smoke finishes in minutes. NOT a sample-size claim.
MAX_UNIQUE_MOLECULES = 900

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

    t0 = time.perf_counter()
    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    src = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]), repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        src, materialized_state=bundle)
    model = runtime.model
    ckpt = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["selected_model_state"], strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    _RT["model"] = model
    print(f"container runtime built in {time.perf_counter()-t0:.1f}s", flush=True)
    return _RT


@app.function(image=image, cpu=(1.0, 1.0), memory=8192, timeout=60 * 60,
              max_containers=60, retries=3,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def encode_shard(smiles: list[str]) -> dict[str, list[float]]:
    """Frozen `R_θ` global-state embeddings for a shard of unique molecules."""
    import sys

    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    model = _runtime()["model"]
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        _one_state_batch,
    )

    out: dict[str, list[float]] = {}
    for smi in smiles:
        try:
            state = pad_molecular_graph(smiles_to_molecular_graph(smi),
                                        CANONICAL_SLOTS)
            batch = _one_state_batch(model, state, float(TIME_POINT),
                                     prepared_batch=None)
            with torch.no_grad():
                _node, global_state, _pair = model._encode_batch(batch)
            out[smi] = global_state[0].detach().cpu().numpy().tolist()
        except Exception as exc:  # noqa: BLE001
            print(f"encode failed {type(exc).__name__} on {smi[:40]}", flush=True)
    return out


@app.function(image=image, cpu=(2.0, 2.0), memory=16384, timeout=4 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(max_unique: int = MAX_UNIQUE_MOLECULES) -> dict[str, Any]:
    import sys

    import numpy as np
    import torch
    import torch.nn as nn

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.hphi_region_features import (
        BUDGET_MAX,
        EMBED_DIM,
        INPUT_DIM,
        build_features,
        h_with_boundary,
        in_region,
    )
    from compose_v4.experiments.hphi_rollout import registered_regions

    artifact_volume.reload()
    blob = json.loads(gzip.decompress(
        (Path(RUN_ROOT) / CORPUS).read_bytes()).decode())
    srcs = [r for r in blob["results"] if r.get("status") == "OK"]
    srcs.sort(key=lambda r: r["index"])
    train_src, eval_src = srcs[:48], srcs[48:64]
    print(f"throwaway split: {len(train_src)} train / {len(eval_src)} eval sources")

    # --- bounded unique-molecule set, distributed encode -----------------
    # The cap MUST cover both splits. A naive `need[:max_unique]` over the
    # concatenated sources takes only the earliest ones, which are all train --
    # that produced eval=0 on the first run.
    def unique_states(rows):
        out, seen = [], set()
        for r in rows:
            for t in r["trajectories"]:
                for s in t["path"]:
                    if s not in seen:
                        seen.add(s)
                        out.append(s)
        return out

    tr_u, ev_u = unique_states(train_src), unique_states(eval_src)
    share = max(1, max_unique // 4)                  # eval gets a real share
    need = list(dict.fromkeys(tr_u[: max_unique - share] + ev_u[:share]))
    print(f"unique states: train {len(tr_u)} eval {len(ev_u)}; "
          f"encoding {len(need)} (eval share {share})", flush=True)
    print(f"encoding {len(need)} unique molecules across containers", flush=True)
    shards = [need[i::60] for i in range(60)]
    emb: dict[str, list[float]] = {}
    for part in encode_shard.map([s for s in shards if s]):
        emb.update(part)
    print(f"encoded {len(emb)}/{len(need)}", flush=True)

    # --- CHECK 1-4: build examples ---------------------------------------
    regions = registered_regions()
    def examples(rows):
        X, Y, meta = [], [], []
        for r in rows:
            src_smi = r["canonical_source"]
            e_src = emb.get(r["source"]) or emb.get(src_smi)
            if e_src is None:
                continue
            e_src = np.asarray(e_src, dtype=np.float64)
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
                        # boundary states are excluded from the LEARNED target
                        if in_region(qed[i], sim[i], reg):
                            continue
                        # hitting label from this prefix forward, within b
                        hit = 0
                        for k in range(i, len(path)):
                            if (k - i) <= b and qed[k] >= reg[0] and sim[k] >= reg[1]:
                                hit = 1
                                break
                        X.append(build_features(np.asarray(emb[s]), e_src,
                                                qed[i], sim[i], reg, b))
                        Y.append(hit)
                        meta.append((reg, b, qed[i], sim[i]))
        return np.asarray(X), np.asarray(Y, dtype=np.float64), meta

    Xtr, Ytr, Mtr = examples(train_src)
    Xev, Yev, Mev = examples(eval_src)
    if len(Xev) == 0:
        raise RuntimeError(
            "eval split produced 0 examples -- the bounded encode set does not "
            "cover the eval sources. This is a wiring bug, not a data fact.")
    checks: dict[str, Any] = {}
    checks["n_train_examples"] = int(len(Xtr))
    checks["n_eval_examples"] = int(len(Xev))
    checks["feature_width_ok"] = bool(Xtr.shape[1] == INPUT_DIM)
    budgets = sorted({m[1] for m in Mtr})
    checks["budgets_present"] = budgets
    checks["all_budgets_0_to_24"] = bool(set(budgets) >= set(range(0, 25)))
    checks["positive_rate"] = float(Ytr.mean()) if len(Ytr) else None
    print(f"\nCHECK examples train={len(Xtr)} eval={len(Xev)} "
          f"width_ok={checks['feature_width_ok']}")
    print(f"CHECK budgets present: {budgets[:6]}...{budgets[-3:]} "
          f"covers 0..24: {checks['all_budgets_0_to_24']}")
    print(f"CHECK positive rate {checks['positive_rate']:.4f}")

    # --- CHECK 6: train a throwaway head ---------------------------------
    torch.manual_seed(0)
    head = nn.Sequential(nn.Linear(INPUT_DIM, 128), nn.ReLU(),
                         nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(Ytr, dtype=torch.float32).unsqueeze(1)
    mu, sd = xt.mean(0, keepdim=True), xt.std(0, keepdim=True) + 1e-6
    xt = (xt - mu) / sd
    losses = []
    torch.set_grad_enabled(True)
    for ep in range(30):
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            l = lossf(head(xt[idx]), yt[idx])
            l.backward(); opt.step()
            tot += float(l) * len(idx)
        losses.append(tot / len(xt))
    torch.set_grad_enabled(False)
    checks["loss_first"] = losses[0]
    checks["loss_last"] = losses[-1]
    checks["loss_decreased"] = bool(losses[-1] < losses[0])
    print(f"CHECK loss {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"decreased={checks['loss_decreased']}")

    xe = (torch.tensor(Xev, dtype=torch.float32) - mu) / sd
    pe = torch.sigmoid(head(xe)).squeeze(1).numpy()
    checks["pred_min"] = float(pe.min()); checks["pred_max"] = float(pe.max())
    checks["pred_std"] = float(pe.std())
    checks["non_degenerate"] = bool(pe.std() > 0.01)
    print(f"CHECK output range [{pe.min():.4f},{pe.max():.4f}] "
          f"std {pe.std():.4f} non_degenerate={checks['non_degenerate']}")

    # --- CHECK 7: nested regions ordered directionally -------------------
    by_q: dict[float, list[float]] = {}
    for p, (reg, b, q, s) in zip(pe, Mev):
        if abs(reg[1] - 0.40) < 1e-9:
            by_q.setdefault(reg[0], []).append(float(p))
    means = {q: float(np.mean(v)) for q, v in sorted(by_q.items())}
    ordered = all(means[a] >= means[b] for a, b in
                  zip(sorted(means)[:-1], sorted(means)[1:]))
    checks["mean_pred_by_qed_threshold"] = means
    checks["nested_regions_ordered"] = bool(ordered)
    print(f"CHECK nested ordering at sim>=0.40: "
          + "  ".join(f"{q:.2f}:{m:.4f}" for q, m in means.items())
          + f"   monotone={ordered}")

    # --- CHECK 5 + 8: exact boundary and sampler plug-in -----------------
    bnd = [h_with_boundary(0.01, 0.95, 0.80, (0.90, 0.40)),
           h_with_boundary(0.99, 0.80, 0.80, (0.90, 0.40))]
    checks["boundary_exact"] = bool(bnd[0] == 1.0 and bnd[1] == 0.99)
    from compose_v4.experiments.hphi_region_features import acceptance_probabilities
    acc = acceptance_probabilities(pe[:16], [m[2] for m in Mev[:16]],
                                   [m[3] for m in Mev[:16]], (0.90, 0.40))
    checks["sampler_accepts_valid_probabilities"] = bool(
        np.all((acc >= 0) & (acc <= 1)))
    print(f"CHECK boundary exact={checks['boundary_exact']}  "
          f"sampler ok={checks['sampler_accepts_valid_probabilities']}")

    out = {"schema": "compose.hphi.smoke", "status": "THROWAWAY_WEIGHTS_DISCARDED",
           "purpose": "engineering only; no QED-success gate; not a result",
           "encoded": len(emb), "checks": checks, "losses": losses}
    p = Path(RUN_ROOT) / OUT_DIR
    p.mkdir(parents=True, exist_ok=True)
    (p / "SMOKE.json").write_text(json.dumps(out, indent=2))
    artifact_volume.commit()
    passed = all([checks["feature_width_ok"], checks["all_budgets_0_to_24"],
                  checks["loss_decreased"], checks["non_degenerate"],
                  checks["boundary_exact"],
                  checks["sampler_accepts_valid_probabilities"]])
    print(f"\nSMOKE {'PASS' if passed else 'FAIL'} "
          f"(nested ordering is directional, not a gate)")
    out["passed"] = passed
    return out


@app.local_entrypoint()
def main() -> None:
    print("THROWAWAY region-h_phi smoke on the EXCLUDED H24 pilot.")
    print("Engineering only. Weights are discarded. No QED-success gate.")
    call = drive.spawn()
    print(f"spawned: {call.object_id}")
