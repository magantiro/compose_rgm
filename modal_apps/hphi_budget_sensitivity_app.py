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

               Measured in BOTH probability and logit space. A head saturated
               near 0 or 1 can barely move in probability while using the budget
               input heavily, so probability-space insensitivity alone proves
               nothing; near-zero LOGIT sensitivity is the strong evidence.

  MONOTONICITY h_b(x) is non-decreasing in b BY DEFINITION -- more budget cannot
               reduce a hitting probability. This is not a fit-quality
               question; a head that violates it is wrong about the quantity it
               names, so the violation rate is reported as its own number.

               Counted against a TOLERANCE, not at float resolution, so
               fitting wiggle is not confused with a genuinely backwards value
               function: the fraction of adjacent budget pairs that fall by
               more than TOL, the median and worst such fall, and the fraction
               of states showing any violation at all.

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
#: A drop smaller than this is fitting noise, not a backwards value function.
MONO_TOL = 0.01


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
                z = head(x).squeeze(-1)
                curves.append((torch.sigmoid(z).numpy(), z.numpy()))
        P = np.asarray([c[0] for c in curves])       # (states, budgets) probs
        Z = np.asarray([c[1] for c in curves])       # (states, budgets) logits

        def spread(M):
            within = float(np.mean(M.max(1) - M.min(1)))   # from b alone
            across = float(np.mean(M.std(0)))              # from x at fixed b
            return within, across, (within / across if across else None)

        wp, ap, rp = spread(P)
        wz, az, rz = spread(Z)

        # Non-decreasing in b is DEFINITIONAL. Count real drops, not wiggle.
        diffs = np.diff(P, axis=1)
        bad = diffs < -MONO_TOL
        falls = -diffs[bad]
        out[name] = {
            "budget_max": bmax,
            "prob": {"mean_range_over_budget": wp,
                     "mean_std_across_states": ap, "ratio": rp},
            "logit": {"mean_range_over_budget": wz,
                      "mean_std_across_states": az, "ratio": rz},
            "monotonicity": {
                "tol": MONO_TOL,
                "frac_adjacent_pairs_decreasing": float(bad.mean()),
                "frac_states_with_any_violation": float(bad.any(1).mean()),
                "median_decrease": (float(np.median(falls)) if falls.size else 0.0),
                "worst_decrease": (float(falls.max()) if falls.size else 0.0),
                "frac_pairs_decreasing_any_amount":
                    float((diffs < -1e-9).mean())},
            "mean_h_at_b1": float(P[:, 0].mean()),
            "mean_h_at_bmax": float(P[:, -1].mean())}
        print(f"{name}:")
        print(f"    prob : b sweeps {wp:.4f}, states differ {ap:.4f} "
              f"-> ratio {rp:.3f}")
        print(f"    logit: b sweeps {wz:.4f}, states differ {az:.4f} "
              f"-> ratio {rz:.3f}   <- the one that decides inertness")
        print(f"    h(b=1)={P[:,0].mean():.4f} -> h(b={bmax})="
              f"{P[:,-1].mean():.4f}")
        print(f"    decreasing by >{MONO_TOL}: {bad.mean()*100:.2f}% of pairs, "
              f"{bad.any(1).mean()*100:.1f}% of states affected")
        if falls.size:
            print(f"    median drop {np.median(falls):.4f}, "
                  f"worst {falls.max():.4f}")
        print()
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
