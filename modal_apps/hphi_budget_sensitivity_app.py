"""Does h_phi(x, b) actually depend on b?

Both readouts point the same way: the H24 head holds ~0.88 AUC from lookahead 4
out to 40, with no decay, while being fed a budget input that is WRONG beyond
24. A predictor that cannot tell 24 from 40 should not be indifferent to being
lied to about which one it is. The natural explanation is that the budget
one-hot barely moves the output, and that discrimination comes almost entirely
from the state embedding, QED and similarity.

If that holds, widening the one-hot from 25 to 41 slots was never going to fix
the long-range collapse, and the H40 line fails for a reason unrelated to
corpus size.

TWO MEASUREMENTS, on real states at the benchmark region:

  SENSITIVITY  sweep b over the head's whole range holding x fixed, and compare
               the resulting spread to the spread ACROSS states at fixed b. A
               ratio near zero means the budget input is nearly inert.

  MONOTONICITY h_b(x) is non-decreasing in b BY DEFINITION -- more budget cannot
               reduce a hitting probability. This is not a fit-quality
               question; a head that violates it is wrong about the quantity it
               names, so the violation rate is reported as its own number.

CPU ONLY. Inference on persisted embeddings; nothing is encoded here.
"""

from __future__ import annotations

import gzip
import json
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

app = modal.App("hphi-budget-sensitivity")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
REGION = (0.90, 0.40)


@app.function(image=image, cpu=(2.0, 2.0), memory=16384, timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def probe(emb_dirs: str, states_blob: str, n_states: int,
          new_dir: str, new_budget_max: int) -> dict[str, Any]:
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, rdFingerprintGenerator

    from compose_v4.experiments.hphi_region_features import build_features, in_region

    artifact_volume.reload()

    def load_head(d):
        h = torch.jit.load(str(Path(RUN_ROOT) / d / "head.pt"), map_location="cpu")
        h.eval()
        n = json.loads((Path(RUN_ROOT) / d / "norm.json").read_text())
        return h, np.asarray(n["mu"]), np.asarray(n["sd"])

    heads = {"H24_old": (*load_head("hphi_v2"), 24),
             "H40_new": (*load_head(new_dir), int(new_budget_max))}

    emb: dict[str, Any] = {}
    for dn in [p.strip() for p in emb_dirs.split(",") if p.strip()]:
        for fp in sorted((Path(RUN_ROOT) / dn).glob("shard_*.json.gz")):
            emb.update(json.loads(gzip.decompress(fp.read_bytes()).decode()))
    blob = json.loads(gzip.decompress(
        (Path(RUN_ROOT) / states_blob).read_bytes()).decode())
    print(f"{len(emb):,} embeddings; {len(blob['results'])} runs", flush=True)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rng = np.random.default_rng(0)
    picked = []
    for r in blob["results"]:
        src = r["source"]
        smol = Chem.MolFromSmiles(src)
        if smol is None or src not in emb:
            continue
        sfp = gen.GetFingerprint(smol)
        cands = [s for s in r["trajectories"][0]["path"]
                 if s != src and s in emb]
        if not cands:
            continue
        s = cands[int(rng.integers(len(cands)))]
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        q = float(QED.qed(m))
        sim = float(DataStructs.TanimotoSimilarity(sfp, gen.GetFingerprint(m)))
        if in_region(q, sim, REGION):
            continue            # boundary states are h=1 by construction
        picked.append((np.asarray(emb[s]), np.asarray(emb[src]), q, sim))
        if len(picked) >= n_states:
            break
    print(f"{len(picked)} evaluable states\n", flush=True)

    out: dict[str, Any] = {"n_states": len(picked), "region": list(REGION)}
    for name, (head, mu, sd, bmax) in heads.items():
        budgets = list(range(1, bmax + 1))
        curves = []
        for e_y, e_src, q, sim in picked:
            F = np.asarray([build_features(e_y, e_src, q, sim, REGION, b, bmax)
                            for b in budgets])
            x = torch.tensor(((F - mu) / sd).astype(np.float32))
            with torch.no_grad():
                curves.append(torch.sigmoid(head(x)).squeeze(-1).numpy())
        C = np.asarray(curves)                       # (states, budgets)
        within = float(np.mean(C.max(1) - C.min(1)))  # spread from b alone
        across = float(np.mean(C.std(0)))             # spread from x at fixed b
        # Non-decreasing in b is a DEFINITIONAL property, not a fit target.
        diffs = np.diff(C, axis=1)
        viol = float((diffs < -1e-6).mean())
        drop = float(np.mean(np.minimum(diffs, 0).sum(1)))
        out[name] = {"budget_max": bmax,
                     "mean_range_over_budget": within,
                     "mean_std_across_states": across,
                     "ratio_budget_to_state": within / across if across else None,
                     "frac_decreasing_steps": viol,
                     "mean_total_decrease": drop,
                     "mean_h_at_b1": float(C[:, 0].mean()),
                     "mean_h_at_bmax": float(C[:, -1].mean())}
        print(f"{name}: b sweeps h by {within:.4f} on average; states differ by "
              f"{across:.4f} at fixed b  (ratio {within/across:.3f})")
        print(f"    h(b=1)={C[:,0].mean():.4f} -> h(b={bmax})="
              f"{C[:,-1].mean():.4f}")
        print(f"    steps DECREASING in b: {viol*100:.1f}%  "
              f"(should be 0 -- more budget cannot lower a hitting prob)\n")
    return out


@app.local_entrypoint()
def main(n_states: int = 400, new_dir: str = "hphi_v2_h40",
         new_budget_max: int = 40,
         emb_dirs: str = "hphi_v2/embeddings,hphi_v2/embeddings_h40,"
                         "hphi_v2/embeddings_banked",
         states_blob: str = "hphi_rollout_corpus/banked_eval_states.json.gz") -> None:
    o = probe.remote(emb_dirs, states_blob, n_states, new_dir, new_budget_max)
    Path("docs/BUDGET_SENSITIVITY.json").write_text(json.dumps(o, indent=1))
    print("\nwrote docs/BUDGET_SENSITIVITY.json")
