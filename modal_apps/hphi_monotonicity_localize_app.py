"""Are the frozen head's monotonicity violations LOAD-BEARING?

The frozen H24 head violates h_{b+1}(x) >= h_b(x) on 4.7% of states. That is a
definitional impossibility for a hitting probability, but a mathematical
imperfection is not automatically a defect worth fixing. Two things must be
true before it is worth touching the controller.

  CONCENTRATION  are the violations disproportionately in the HARD / failed
                 sources, i.e. where the controller actually loses?

  LOAD-BEARING   do they change what the controller DOES -- the ranking of
                 particles the twist induces at a step -- or do they wash out?

The second is the real question. The twist enters as an incremental weight
h(y, b-1)/h(x, b), so what matters operationally is how h orders the 32
particles at a step, not its absolute calibration. If an isotonic projection
over the budget axis leaves that ordering intact, the violations are cosmetic
and monotonicity work should be dropped.

METHOD. For every banked state already embedded, evaluate the frozen head over
its whole budget range, project that curve onto the non-decreasing cone by
PAVA, and compare. Rankings are compared WITHIN a (run, step), across the 32
particles alive there -- which is exactly the set the resampler competes.

CAVEAT, stated up front: this compares the twist's own contribution at a step.
It does not replay accumulated weights or the resampling RNG, so it bounds the
ordering change the projection induces rather than simulating a full rerun.

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

app = modal.App("hphi-monotonicity-localize")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
REGION = (0.90, 0.40)
LOOKAHEADS = (4, 8, 12, 16)
BUDGET_MAX = 24
MONO_TOL = 0.01


def pava(y):
    """L2 projection onto the non-decreasing cone (pool adjacent violators)."""
    import numpy as np

    y = np.asarray(y, dtype=float).copy()
    n = len(y)
    w = np.ones(n)
    lvl, wt, idx = [], [], []
    for i in range(n):
        lvl.append(y[i]); wt.append(w[i]); idx.append(1)
        while len(lvl) > 1 and lvl[-2] > lvl[-1]:
            v2, w2, c2 = lvl.pop(), wt.pop(), idx.pop()
            v1, w1, c1 = lvl.pop(), wt.pop(), idx.pop()
            lvl.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wt.append(w1 + w2); idx.append(c1 + c2)
    out = np.empty(n)
    p = 0
    for v, c in zip(lvl, idx):
        out[p:p + c] = v
        p += c
    return out


@app.function(image=image, cpu=(4.0, 4.0), memory=16384, timeout=2 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def localize(strata: dict, emb_dirs: str) -> dict[str, Any]:
    import sys
    import time

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, rdFingerprintGenerator

    from compose_v4.experiments.hphi_region_features import build_features, in_region

    artifact_volume.reload()
    head = torch.jit.load(str(Path(RUN_ROOT) / "hphi_v2" / "head.pt"),
                          map_location="cpu")
    head.eval()
    nrm = json.loads((Path(RUN_ROOT) / "hphi_v2" / "norm.json").read_text())
    mu, sd = np.asarray(nrm["mu"]), np.asarray(nrm["sd"])

    emb: dict[str, Any] = {}
    for dn in [p.strip() for p in emb_dirs.split(",") if p.strip()]:
        for fp in sorted((Path(RUN_ROOT) / dn).glob("shard_*.json.gz")):
            emb.update(json.loads(gzip.decompress(fp.read_bytes()).decode()))
    print(f"{len(emb):,} embeddings", flush=True)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    d = Path(RUN_ROOT) / "hphi_smc_64" / "replicates"
    files = sorted(d.glob("*.json"))

    budgets = list(range(1, BUDGET_MAX + 1))

    def curves(e_ys, e_src, qs, ss):
        """h(y, b) for b in 1..BUDGET_MAX, batched over states."""
        F, meta = [], []
        for e_y, q, s in zip(e_ys, qs, ss):
            for b in budgets:
                F.append(build_features(e_y, e_src, q, s, REGION, b))
            meta.append(in_region(q, s, REGION))
        X = torch.tensor(((np.asarray(F) - mu) / sd).astype(np.float32))
        with torch.no_grad():
            P = torch.sigmoid(head(X)).squeeze(-1).numpy()
        return P.reshape(len(e_ys), len(budgets)), meta

    # ---- per-state violations, tagged by stratum and run outcome ----------
    stats: dict = defaultdict(lambda: {"n": 0, "viol": 0, "pairs": 0,
                                       "viol_pairs": 0, "worst": 0.0})
    rank_rows: list = []
    t0 = time.perf_counter()
    n_states = 0

    for f in files:
        doc = json.loads(f.read_text())
        idx, rec = doc["index"], doc["record"]
        st = strata.get(str(idx), "hard")
        if st == "reliable":
            continue
        tr = rec["transitions"]
        if not tr:
            continue
        hit_steps = [t["step"] for t in tr if t.get("h_y_bm1") == 1.0]
        is_hit = bool(hit_steps)
        end = min(hit_steps) if is_hit else max(t["step"] for t in tr)
        source = tr[0]["x"]
        if source not in emb:
            continue
        e_src = np.asarray(emb[source])
        smol = Chem.MolFromSmiles(source)
        if smol is None:
            continue
        sfp = gen.GetFingerprint(smol)
        cache: dict = {}

        def props(smi):
            if smi not in cache:
                m = Chem.MolFromSmiles(smi)
                cache[smi] = (0.0, 0.0) if m is None else (
                    float(QED.qed(m)),
                    float(DataStructs.TanimotoSimilarity(sfp, gen.GetFingerprint(m))))
            return cache[smi]

        for L in LOOKAHEADS:
            s_step = end - L
            if s_step < 0:
                continue
            ys, bs = [], []
            for t in tr:
                if t["step"] != s_step or t.get("h_y_bm1") is None:
                    continue
                if t["y"] not in emb:
                    continue
                ys.append(t["y"]); bs.append(int(t["budget"]) - 1)
            if len(ys) < 4:
                continue
            qs, ss = zip(*[props(y) for y in ys])
            e_ys = [np.asarray(emb[y]) for y in ys]
            C, boundary = curves(e_ys, e_src, list(qs), list(ss))
            n_states += len(ys)

            key = f"{st}|{'hit' if is_hit else 'miss'}"
            a = stats[key]
            for row, bd in zip(C, boundary):
                if bd:
                    continue
                a["n"] += 1
                dif = np.diff(row)
                bad = dif < -MONO_TOL
                a["pairs"] += len(dif); a["viol_pairs"] += int(bad.sum())
                if bad.any():
                    a["viol"] += 1
                    a["worst"] = max(a["worst"], float(-dif[bad].min()))

            # ---- does the isotonic projection reorder the particles? -------
            orig = np.array([C[i, min(max(bs[i], 1), BUDGET_MAX) - 1]
                             for i in range(len(ys))])
            proj = np.array([pava(C[i])[min(max(bs[i], 1), BUDGET_MAX) - 1]
                             for i in range(len(ys))])
            if len(set(orig.tolist())) > 1:
                ro = np.argsort(np.argsort(orig))
                rp = np.argsort(np.argsort(proj))
                disc = 0
                m = len(orig)
                for i in range(m):
                    for j in range(i + 1, m):
                        if np.sign(orig[i] - orig[j]) != np.sign(proj[i] - proj[j]):
                            disc += 1
                k = max(1, m // 2)
                top_o = set(np.argsort(-orig)[:k].tolist())
                top_p = set(np.argsort(-proj)[:k].tolist())
                rank_rows.append({
                    "stratum": st, "hit": is_hit, "L": L, "m": m,
                    "discordant_frac": disc / (m * (m - 1) / 2),
                    "spearman": float(np.corrcoef(ro, rp)[0, 1]),
                    "top_half_changed": len(top_o ^ top_p) // 2,
                    "max_rel_change": float(np.max(np.abs(proj - orig)
                                                   / np.maximum(orig, 1e-12)))})
        print(f"  {f.name} {n_states:,} states {time.perf_counter()-t0:.0f}s",
              flush=True)

    print(f"\n{'group':<16}{'states':>8}{'viol states':>13}{'viol pairs':>12}"
          f"{'worst drop':>12}")
    out: dict[str, Any] = {"by_group": {}, "n_states": n_states}
    for k in sorted(stats):
        a = stats[k]
        if not a["n"]:
            continue
        out["by_group"][k] = {
            "n_states": a["n"],
            "frac_states_violating": a["viol"] / a["n"],
            "frac_pairs_violating": a["viol_pairs"] / max(a["pairs"], 1),
            "worst_drop": a["worst"]}
        print(f"{k:<16}{a['n']:>8}{a['viol']/a['n']*100:>12.1f}%"
              f"{a['viol_pairs']/max(a['pairs'],1)*100:>11.2f}%"
              f"{a['worst']:>12.4f}")

    if rank_rows:
        sp = np.array([r["spearman"] for r in rank_rows])
        df = np.array([r["discordant_frac"] for r in rank_rows])
        th = np.array([r["top_half_changed"] for r in rank_rows])
        print(f"\nRANKING IMPACT of the isotonic projection, over "
              f"{len(rank_rows)} (run, step) groups of ~32 particles:")
        print(f"  spearman(orig, projected): mean {sp.mean():.5f}  "
              f"min {sp.min():.5f}")
        print(f"  discordant pairs:          mean {df.mean()*100:.3f}%  "
              f"max {df.max()*100:.3f}%")
        print(f"  groups whose TOP HALF changes at all: "
              f"{(th>0).mean()*100:.1f}%  (mean swaps {th.mean():.2f})")
        out["ranking"] = {
            "n_groups": len(rank_rows),
            "spearman_mean": float(sp.mean()), "spearman_min": float(sp.min()),
            "discordant_frac_mean": float(df.mean()),
            "discordant_frac_max": float(df.max()),
            "frac_groups_top_half_changed": float((th > 0).mean()),
            "mean_top_half_swaps": float(th.mean())}
        print("\nIf spearman is ~1 and the top half never changes, the "
              "violations do not steer the controller and monotonicity work\n"
              "should be dropped rather than pursued.")
    return out


@app.local_entrypoint()
def main(emb_dirs: str = "hphi_v2/embeddings,hphi_v2/embeddings_banked") -> None:
    root = Path(__file__).resolve().parents[1]
    g = json.loads((root / "docs/HPHI_SMC_64_GATE_BANKED.json").read_text())
    l = json.loads((root / "docs/HPHI_COVERAGE_LADDER_BANKED.json").read_text())
    r3 = json.loads((root / "docs/HPHI_LADDER_RUNG3.json").read_text())
    rel = set(g["solved_sources"])
    marg = set(l["rung2"]["new_sources"]) | set(r3["conversions"])
    strata = {str(i): ("reliable" if i in rel else
                       "marginal" if i in marg else "hard") for i in range(64)}
    o = localize.remote(strata, emb_dirs)
    Path("docs/MONOTONICITY_LOCALIZE.json").write_text(json.dumps(o, indent=1))
    print("\nwrote docs/MONOTONICITY_LOCALIZE.json")
