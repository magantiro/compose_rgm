"""Long-range discrimination on UNGUIDED held-out rollouts, old head vs new.

WHY THIS EXISTS ALONGSIDE THE BANKED COMPARISON. The banked SMC runs stop at
H = 24, so re-scoring them can only probe lookaheads up to 24 -- and every
lookahead in {4, 8, 12, 16} sits inside BOTH heads' training range. There the
H40 head has no structural advantage and strictly LESS data (512 trajectories
against 2,048), so that comparison is if anything stacked against it.

The H40 head's structural advantage is at budgets 25..40, where the H24 head
has no training data whatsoever and its one-hot input cannot even represent the
budget. That range is exactly what this measures, and nothing else can: it
needs trajectories longer than 24 steps.

QUANTITY. Precisely what both heads were trained to predict --
h_b(x, z) = P(exists t <= b : X_t in B_z) -- scored over the 20 preregistered
regions on held-out sources, with in-region states excluded because the
boundary h = 1 is enforced rather than learned. Pooling over regions is what
makes positives common enough to measure; the benchmark region alone is far too
rare under unguided dynamics to yield an AUC.

Both heads are clean here: the pilot-64 sources are permanently excluded and
disjoint from hphi_train_1024, so neither head trained on them.

CPU ONLY.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"}
)

app = modal.App("hphi-longrange-auc")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
LOOKAHEADS = (4, 8, 12, 16, 24, 32, 40)


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    n = 0.0
    for p in pos:
        for q in neg:
            n += 1.0 if p > q else (0.5 if p == q else 0.0)
    return n / (len(pos) * len(neg))


@app.function(image=image, cpu=(4.0, 4.0), memory=16384, timeout=4 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def evaluate(corpus: str, emb_dirs: str, new_dir: str,
             new_budget_max: int) -> dict[str, Any]:
    import sys
    import time

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.hphi_region_features import (
        build_features, in_region,
    )
    from compose_v4.experiments.hphi_rollout import registered_regions

    artifact_volume.reload()

    def load_head(d: str):
        h = torch.jit.load(str(Path(RUN_ROOT) / d / "head.pt"), map_location="cpu")
        h.eval()
        n = json.loads((Path(RUN_ROOT) / d / "norm.json").read_text())
        return h, np.asarray(n["mu"]), np.asarray(n["sd"])

    old_head, old_mu, old_sd = load_head("hphi_v2")
    new_head, new_mu, new_sd = load_head(new_dir)
    print(f"old width {old_mu.shape[0]}   new width {new_mu.shape[0]}", flush=True)

    blob = json.loads(gzip.decompress(
        (Path(RUN_ROOT) / corpus).read_bytes()).decode())
    srcs = [r for r in blob["results"] if r.get("status") == "OK"]
    emb: dict[str, list[float]] = {}
    for dname in [p.strip() for p in emb_dirs.split(",") if p.strip()]:
        sd_ = Path(RUN_ROOT) / dname
        for f in sorted(sd_.glob("shard_*.json.gz")):
            emb.update(json.loads(gzip.decompress(f.read_bytes()).decode()))
    print(f"{len(srcs)} sources; {len(emb):,} embeddings loaded", flush=True)

    regions = registered_regions()
    pos: dict = {L: defaultdict(list) for L in LOOKAHEADS}
    neg: dict = {L: defaultdict(list) for L in LOOKAHEADS}
    t0 = time.perf_counter()
    n_skipped = 0

    def batch_score(head, mu, sd, F: np.ndarray) -> np.ndarray:
        x = torch.tensor(((F - mu) / sd).astype(np.float32))
        with torch.no_grad():
            return torch.sigmoid(head(x)).squeeze(-1).numpy()

    for rn, r in enumerate(srcs):
        e_src = emb.get(r["source"])
        if e_src is None:
            n_skipped += 1
            continue
        e_src = np.asarray(e_src)
        for t in r["trajectories"]:
            path, qed, sim = t["path"], t["qed"], t["similarity_to_source"]
            if any(s not in emb for s in path):
                n_skipped += 1
                continue
            for L in LOOKAHEADS:
                rows_o, rows_n, labels = [], [], []
                for i in range(len(path)):
                    if i + L > len(path) - 1:
                        break
                    for reg in regions:
                        if in_region(qed[i], sim[i], reg):
                            continue        # boundary: h = 1, not learned
                        hit = 0
                        for k in range(i + 1, i + L + 1):
                            if qed[k] >= reg[0] and sim[k] >= reg[1]:
                                hit = 1
                                break
                        e_i = np.asarray(emb[path[i]])
                        rows_o.append(build_features(
                            e_i, e_src, qed[i], sim[i], reg, min(L, 24), 24))
                        rows_n.append(build_features(
                            e_i, e_src, qed[i], sim[i], reg,
                            min(L, new_budget_max), new_budget_max))
                        labels.append(hit)
                if not labels:
                    continue
                po = batch_score(old_head, old_mu, old_sd, np.asarray(rows_o))
                pn = batch_score(new_head, new_mu, new_sd, np.asarray(rows_n))
                for h_o, h_n, y in zip(po, pn, labels):
                    b = pos[L] if y else neg[L]
                    b["h_phi_old"].append(float(h_o))
                    b["h_phi_new"].append(float(h_n))
        if (rn + 1) % 8 == 0:
            print(f"  {rn+1}/{len(srcs)} sources  "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    print(f"\nskipped {n_skipped} trajectories/sources lacking embeddings")
    print(f"\n{'lookahead':<11}{'n_pos':>10}{'n_neg':>10}"
          f"{'h_phi_old':>12}{'h_phi_new':>12}{'delta':>10}")
    out: dict[str, Any] = {"auc": {}, "corpus": corpus, "new_dir": new_dir,
                           "new_budget_max": new_budget_max,
                           "n_skipped": n_skipped}
    for L in LOOKAHEADS:
        npos, nneg = len(pos[L]["h_phi_new"]), len(neg[L]["h_phi_new"])
        # AUC is over a huge number of pairs; subsample for tractability.
        rng = np.random.default_rng(0)
        def sub(v):
            return (list(rng.choice(v, 4000, replace=False))
                    if len(v) > 4000 else v)
        a_o = auc(sub(pos[L]["h_phi_old"]), sub(neg[L]["h_phi_old"]))
        a_n = auc(sub(pos[L]["h_phi_new"]), sub(neg[L]["h_phi_new"]))
        out["auc"][L] = {"h_phi_old": a_o, "h_phi_new": a_n,
                         "n_pos": npos, "n_neg": nneg}
        print(f"{L:<11}{npos:>10,}{nneg:>10,}{a_o:>12.4f}{a_n:>12.4f}"
              f"{a_n-a_o:>10.4f}")
    print("\nAt lookaheads > 24 the OLD head is clamped to budget 24 -- it "
          "cannot represent the horizon, so its column there is a handicapped\n"
          "reference, not a fair competitor. Below 24 both are in range.")
    return out


@app.local_entrypoint()
def main(corpus: str = "hphi_rollout_corpus/pilot_0064x02_H40.json.gz",
         emb_dirs: str = "hphi_v2/embeddings,hphi_v2/embeddings_h40,"
                         "hphi_v2/embeddings_pilot_h40",
         new_dir: str = "hphi_v2_h40", new_budget_max: int = 40) -> None:
    out = evaluate.remote(corpus, emb_dirs, new_dir, new_budget_max)
    Path("docs/LONGRANGE_AUC.json").write_text(json.dumps(out, indent=1))
    print("\nwrote docs/LONGRANGE_AUC.json")
