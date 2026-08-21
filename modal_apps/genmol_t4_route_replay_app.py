"""Is a known-good route locally misranked, or found and then dropped?

The selection diagnostic showed two different Level-1 failures. parp1 enumerated
129 feasible molecules and docked 4: an oracle-allocation defect. fa7 enumerated
zero across ~7,600 successors while a route to feasibility exists at depth 9:
a lineage-selection failure.

"Lineage-selection failure" is as far as that evidence goes. It does NOT
establish that future lookahead is the fix, because two different causes produce
it:

    local misranking   the right next edit looks bad under the immediate rule,
                       so lookahead is genuinely earned
    lineage collapse   the right next edit ranks fine but its parent never
                       survives, so the fix is archive diversity, not a value
                       function

This distinguishes them for free. It reconstructs a certified route with parent
pointers, then walks that route asking, at every state on it, where Level 1
would have placed the known-good next step:

    rank of x_{j+1} among the enumerated fiber of x_j, by tilt weight
    its R_theta, its v, its tilt weight
    whether it falls inside the per-round docking allowance
    whether it was enumerated at all under APPLY_CAP

A route step that is not even enumerated, or sits at rank 400/750, is local
misranking. A route step at rank 8/750 that still loses is lineage collapse.

CPU only. NO DOCKING anywhere in this job.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import (
    APPLY_CAP, CANONICAL_SLOTS, QED_MIN, SA_MAX, TAU_V, TIME_POINT,
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume, _runtime,
)
from modal_apps.genmol_t4_opt_app import image as _opt_image

image = _opt_image.add_local_file(
    ROOT / "modal_apps/genmol_t4_opt_app.py",
    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)

app = modal.App("genmol-t4-route-replay")

DEPTH, BEAM, REACH_CAP = 16, 10, 200
CELLS = [
    {"idx":  6, "target": "fa7",   "chembl": "CHEMBL379809", "delta": 0.4},
    {"idx": 18, "target": "parp1", "chembl": "CHEMBL383578", "delta": 0.4},
    {"idx":  2, "target": "braf",  "chembl": "CHEMBL410295", "delta": 0.4},
]


@app.function(image=image, cpu=(1.0, 1.0), memory=int(6 * 1024),
              timeout=8 * 60 * 60, max_containers=6,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def replay(task: dict[str, Any]) -> dict[str, Any]:
    import sys, os
    import numpy as np
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, AllChem, RDConfig
    RDLogger.DisableLog("rdApp.*")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)

    rt = _runtime(); model, system = rt["model"], rt["system"]
    seed, delta = task["smiles"], task["delta"]
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = gen.GetFingerprint(Chem.MolFromSmiles(seed))
    t0 = time.perf_counter()

    def props(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        q = float(QED.qed(m)); s = float(sascorer.calculateScore(m))
        sim = float(DataStructs.TanimotoSimilarity(seed_fp, gen.GetFingerprint(m)))
        v = max(max(0.0, QED_MIN - q) / QED_MIN, max(0.0, s - SA_MAX) / SA_MAX,
                max(0.0, delta - sim) / delta)
        return {"qed": q, "sa": s, "sim": sim, "v": v}

    def fiber(smi, cap):
        """Enumerated successors of smi with their R_theta, top-cap by R_theta."""
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        except Exception:
            return {}
        if not law.marks:
            return {}
        pr = np.array([m.probability for m in law.marks], float)
        out = {}
        for i in np.argsort(-pr)[:cap]:
            mk = law.marks[int(i)]
            try:
                y = canonical_state_key(system.apply(st, mk.executor_rule_name, mk.action))
            except Exception:
                continue
            if y and y != smi and y not in out:
                out[y] = float(pr[int(i)])
        return out

    # ---- 1. certified route, with parent pointers ----
    p0 = props(seed)
    parent = {seed: None}
    frontier = [(seed, p0)]
    route, hit_depth = None, None
    for d in range(1, DEPTH + 1):
        cand = {}
        for smi, _ in frontier:
            for y, r in fiber(smi, REACH_CAP).items():
                if y in parent or y in cand:
                    continue
                p = props(y)
                if p is not None:
                    cand[y] = p; parent[y] = smi
        feas = {k: v for k, v in cand.items() if v["sim"] >= delta}
        if not feas:
            break
        adm = [(k, v) for k, v in feas.items() if v["v"] == 0.0]
        if adm:
            end = min(adm, key=lambda kv: -kv[1]["qed"])[0]
            route, hit_depth = [], d
            n = end
            while n is not None:
                route.append(n); n = parent[n]
            route.reverse()
            break
        frontier = sorted(feas.items(), key=lambda kv: kv[1]["v"])[:BEAM]

    out = {**{k: v for k, v in task.items() if k != "smiles"},
           "route_found": route is not None, "route_depth": hit_depth,
           "seconds_route": round(time.perf_counter() - t0, 1)}

    # ---- 2. replay the route through the Level-1 scoring rule ----
    steps = []
    if route:
        for j in range(len(route) - 1):
            x, nxt = route[j], route[j + 1]
            f = fiber(x, APPLY_CAP)
            keys = list(f)
            if nxt not in f:
                steps.append({"j": j, "n_fiber": len(keys), "enumerated": False,
                              "rank": None, "note": "route step NOT enumerated under APPLY_CAP"})
                continue
            pv = {y: props(y) for y in keys}
            keys = [y for y in keys if pv[y] is not None]
            w = np.array([f[y] * np.exp(-pv[y]["v"] / TAU_V) for y in keys])
            w = np.clip(w, 1e-30, None); w /= w.sum()
            order = np.argsort(-w)
            rank = int(np.where(np.array(keys)[order] == nxt)[0][0])
            steps.append({"j": j, "n_fiber": len(keys), "enumerated": True,
                          "rank": rank, "pct": round(100 * rank / len(keys), 1),
                          "weight": float(w[keys.index(nxt)]),
                          "rtheta": f[nxt], "v": round(pv[nxt]["v"], 4),
                          "in_top20": rank < 20})
    out["steps"] = steps
    enum = [s for s in steps if s["enumerated"]]
    out["n_steps"] = len(steps)
    out["n_not_enumerated"] = sum(1 for s in steps if not s["enumerated"])
    out["n_in_top20"] = sum(1 for s in enum if s["in_top20"])
    out["median_rank_pct"] = (sorted(s["pct"] for s in enum)[len(enum)//2]
                              if enum else None)
    out["verdict"] = (
        "no route found" if not route else
        "LOCAL MISRANKING: route steps rank poorly -> lookahead earned"
        if (out["n_not_enumerated"] or (out["median_rank_pct"] or 0) > 20) else
        "LINEAGE COLLAPSE: route steps rank well but are not retained -> fix archive diversity")
    try:
        d = Path("/artifacts/t4_route_replay"); d.mkdir(parents=True, exist_ok=True)
        (d / f"cell_{task['idx']}_{task['target']}.json").write_text(json.dumps(out, indent=1))
        artifact_volume.commit()
    except Exception as e:
        print(f"  !! persist failed idx {task['idx']}: {e}", flush=True)
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=2048, timeout=10 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(only: str = "") -> dict[str, Any]:
    seeds = json.loads((REMOTE_ROOT / "docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    cells = [c for c in CELLS if not only or c["target"] == only]
    tasks = [{**c, "smiles": seeds[c["idx"]]["smiles"],
              "qed": seeds[c["idx"]]["qed"], "sa": seeds[c["idx"]]["sa"]} for c in cells]
    print(f"{len(tasks)} cell(s), certified route then Level-1 replay, NO DOCKING\n", flush=True)
    out = []
    for r in replay.map(tasks, order_outputs=False, return_exceptions=True,
                        wrap_returned_exceptions=False):
        if not isinstance(r, dict):
            print(f"  !! {type(r).__name__}: {str(r)[:120]}", flush=True); continue
        out.append(r)
        print(f"  {r['target']:6s} route@{r['route_depth']}  steps {r['n_steps']}  "
              f"not-enumerated {r['n_not_enumerated']}  in-top20 {r['n_in_top20']}  "
              f"median rank {r['median_rank_pct']}%  -> {r['verdict']}", flush=True)
        for s in r["steps"]:
            if not s["enumerated"]:
                print(f"      step {s['j']}: NOT ENUMERATED of {s['n_fiber']}", flush=True)
            else:
                print(f"      step {s['j']}: rank {s['rank']:>4}/{s['n_fiber']:<4} "
                      f"({s['pct']:>5.1f}%)  R_theta {s['rtheta']:.2e}  v {s['v']:.3f}", flush=True)
    if not out:
        d = Path("/artifacts/t4_route_replay")
        if d.exists():
            out = [json.loads(f.read_text()) for f in sorted(d.glob("cell_*.json"))]
            print(f"  recovered {len(out)} cells from the volume", flush=True)
    if not out:
        raise RuntimeError("no cells completed")
    return {"cells": out}


@app.function(image=image, cpu=(1.0, 1.0), memory=1024, timeout=600,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def salvage() -> dict[str, Any]:
    d = Path("/artifacts/t4_route_replay")
    cells = [json.loads(f.read_text()) for f in sorted(d.glob("cell_*.json"))] if d.exists() else []
    print(f"  salvaged {len(cells)} cells", flush=True)
    return {"cells": cells, "salvaged": True}


@app.local_entrypoint()
def main(only: str = "") -> None:
    try:
        o = drive.remote(only)
    except Exception as e:
        print(f"  driver failed ({type(e).__name__}); salvaging")
        o = salvage.remote()
    p = Path(__file__).resolve().parents[1] / "diagnostics/genmol_t4_route_replay.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
