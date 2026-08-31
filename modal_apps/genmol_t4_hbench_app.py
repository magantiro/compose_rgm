"""1%-scale timing microbenchmark for the T4 future-h controller.

WHY THIS RUNS BEFORE ANY REFACTOR. Two multi-hour surprises came from sizing
loops by estimate. This measures the primitives directly, at ~1% of a round, so
the refactor targets whatever is actually expensive rather than whatever looks
expensive. The specific question: inside fiber(), is the cost the R_theta forward
pass -- which batching would fix -- or the APPLY_CAP=300 system.apply loop, which
batching would not touch at all?

It reports, per stage:

    t_law      enumerate_factorized_marked_law  (one R_theta forward)
    t_apply    the up-to-300 system.apply calls that follow it
    t_gterm    g_terminal, i.e. the feasibility/desirability score
    fibers/h   how many fiber() calls one h_hat actually costs, after caching

and then projects a full round from the measured per-candidate h_hat cost.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import (
    APPLY_CAP, CANONICAL_SLOTS, QED_MIN, SA_MAX, TIME_POINT,
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume, _runtime,
)
from modal_apps.genmol_t4_opt_app import image as _opt_image

image = _opt_image.add_local_file(
    ROOT / "modal_apps/genmol_t4_opt_app.py",
    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)

app = modal.App("genmol-t4-hbench")

ROLLOUTS, H_BUDGET = 6, 40


@app.function(image=image, cpu=(1.0, 1.0), memory=int(6 * 1024),
              timeout=60 * 60, volumes={str(ARTIFACT_ROOT): artifact_volume})
def bench(idx: int = 6, n_fiber: int = 40, n_h: int = 12, horizon: int = 4) -> dict[str, Any]:
    import sys, os
    import numpy as np
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, QED
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)

    seeds = json.loads((REMOTE_ROOT / "docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    cell = seeds[idx]
    seed = cell["smiles"]
    t_load = time.time()
    rt = _runtime(); model, system = rt["model"], rt["system"]
    t_load = time.time() - t_load
    rng = np.random.default_rng(0)
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = gen.GetFingerprint(Chem.MolFromSmiles(seed))

    T = {"law": 0.0, "apply": 0.0, "gterm": 0.0}
    C = {"law": 0, "apply": 0, "gterm": 0, "fiber_calls": 0}
    TT: dict[str, dict] = {}

    def fiber(smi):
        e = TT.setdefault(smi, {})
        if "f" in e:
            return e["f"]
        C["fiber_calls"] += 1
        try:
            _t = time.time()
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            T["law"] += time.time() - _t; C["law"] += 1
        except Exception:
            e["f"] = {}; return e["f"]
        if not law.marks:
            e["f"] = {}; return e["f"]
        pr = np.array([m.probability for m in law.marks], float)
        out = {}
        _t = time.time()
        n_ap = 0
        for i in np.argsort(-pr)[:APPLY_CAP]:
            mk = law.marks[int(i)]
            try:
                y = canonical_state_key(system.apply(st, mk.executor_rule_name, mk.action))
            except Exception:
                continue
            n_ap += 1
            if y and y != smi and y not in out:
                out[y] = float(pr[int(i)])
        T["apply"] += time.time() - _t; C["apply"] += n_ap
        e["f"] = out
        return out

    def g_terminal(smi, tl):
        _t = time.time()
        m = Chem.MolFromSmiles(smi)
        v = 0.0
        if m is not None:
            q = QED.qed(m)
            sim = float(DataStructs.TanimotoSimilarity(seed_fp, gen.GetFingerprint(m)))
            v = max((QED_MIN - q) / QED_MIN, 0.0, (0.4 - sim) / 0.4)
        T["gterm"] += time.time() - _t; C["gterm"] += 1
        return float(np.exp(-v / 0.10))

    # ---- stage 1: cold fiber cost on distinct states ----
    root = fiber(seed)
    cands = list(root)[:n_fiber]
    t0 = time.time()
    for y in cands:
        fiber(y)
    t_fiber_block = time.time() - t0
    n_states_root = len(root)

    # ---- stage 2: h_hat cost per candidate ----
    def h_hat(y, b):
        if b <= 0:
            return g_terminal(y, None)
        spent, acc = 0, []
        for _ in range(ROLLOUTS):
            cur = y
            for _d in range(b):
                f = fiber(cur)
                if not f or spent >= H_BUDGET:
                    break
                ks = list(f); w = np.array([f[k] for k in ks]); w /= w.sum()
                cur = ks[int(rng.choice(len(ks), p=w))]
                spent += 1
            acc.append(g_terminal(cur, None))
        return float(np.mean(acc)) if acc else 0.0

    before = dict(C)
    t0 = time.time()
    for y in cands[:n_h]:
        h_hat(y, horizon)
    t_h_block = time.time() - t0
    fibers_per_h = (C["fiber_calls"] - before["fiber_calls"]) / max(n_h, 1)

    ms = lambda a, b: 1000 * a / max(b, 1)
    out = {
        "cell": {"idx": idx, "target": cell["target"], "chembl": cell["chembl"]},
        "runtime_load_s": round(t_load, 1),
        "root_successors": n_states_root,
        "n_fiber_timed": len(cands), "n_h_timed": min(n_h, len(cands)),
        "per_fiber_ms": round(ms(t_fiber_block, len(cands)), 1),
        "stage_ms": {"law": round(ms(T["law"], C["law"]), 2),
                     "apply_per_fiber": round(ms(T["apply"], C["law"]), 2),
                     "apply_per_call_us": round(1e6 * T["apply"] / max(C["apply"], 1), 1),
                     "gterm": round(ms(T["gterm"], C["gterm"]), 3)},
        "stage_share": {k: round(T[k] / max(sum(T.values()), 1e-9), 3) for k in T},
        "applies_total": C["apply"], "law_calls": C["law"],
        "per_h_hat_ms": round(ms(t_h_block, min(n_h, len(cands))), 1),
        "fibers_per_h_hat": round(fibers_per_h, 2),
    }
    # projection to a real round, using the measured per-candidate h cost
    for n_cand in (400, 1368, 2000):
        out.setdefault("projected_round_min", {})[str(n_cand)] = round(
            n_cand * out["per_h_hat_ms"] / 1000 / 60, 1)
    d = Path("/artifacts/t4_hbench"); d.mkdir(parents=True, exist_ok=True)
    (d / f"bench_{cell['target']}.json").write_text(json.dumps(out, indent=1))
    artifact_volume.commit()
    print(json.dumps(out, indent=1), flush=True)
    return out


@app.local_entrypoint()
def main(idx: int = 6, n_fiber: int = 40, n_h: int = 12, horizon: int = 4,
         out: str = "") -> None:
    o = bench.remote(idx, n_fiber, n_h, horizon)
    p = Path(__file__).resolve().parents[1] / (out or "diagnostics/genmol_t4_hbench.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
