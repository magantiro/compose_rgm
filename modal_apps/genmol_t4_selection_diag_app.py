"""Where does Level 1 lose a route that exists?

The dev sweep reported n_feasible = 0 for several cells, but that counter is
computed from the ARCHIVE, and the archive only receives molecules that were
DOCKED. Feasibility itself is free: v = max((0.6-QED)+/0.6, (SA-4)+/4,
(delta-sim)+/delta) is computed for every enumerated candidate before any
docking call. So "zero feasible" may mean either

    (a) no feasible candidate was ever enumerated          -> real myopia
    (b) feasible candidates were enumerated and not docked -> selection defect

These demand opposite responses. (a) earns Level 2 continuation lookahead;
(b) means the frozen tilt rule under-prioritises feasibility and needs an
amendment, not an escalation. The banked sweep cannot distinguish them.

This replays the FROZEN Level-1 policy unchanged -- same fiber enumeration,
same APPLY_CAP, same parents rule, same tilt weights, same per-cell RNG -- and
adds three counters per round:

    n_cand         candidates enumerated
    n_feas_enum    of those, how many have v == 0
    n_feas_picked  of those, how many the tilt rule actually selected to dock

plus, when a feasible candidate is enumerated but NOT picked, its tilt rank and
weight against the ones that were. That is the direct evidence for (a) vs (b).

Docking is retained because the trajectory depends on it: parents are ranked by
docking score, so a no-dock replay would visit different states and diagnose a
different search. CPU only, ~200 dockings per cell.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import (
    APPLY_CAP, CANONICAL_SLOTS, QED_MIN, SA_MAX, TAU_V, TIME_POINT,
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume, _dock, _runtime,
)
from modal_apps.genmol_t4_opt_app import image as _opt_image

# the optimizer's image does not ship the optimizer module itself, so a job that
# imports from it fails at container import. Add it, and this file's own package
# marker, on top of that image.
image = _opt_image.add_local_file(
    ROOT / "modal_apps/genmol_t4_opt_app.py",
    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)

app = modal.App("genmol-t4-selection-diag")

# the Level-1 failures with a route the reachability procedure did find
CELLS = [
    {"idx": 18, "target": "parp1", "chembl": "CHEMBL383578", "delta": 0.4},
    {"idx":  6, "target": "fa7",   "chembl": "CHEMBL379809", "delta": 0.4},
    {"idx":  2, "target": "braf",  "chembl": "CHEMBL410295", "delta": 0.4},
]


@app.function(image=image, cpu=(1.0, 1.0), memory=int(6 * 1024),
              timeout=6 * 60 * 60, max_containers=8,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def diagnose(task: dict[str, Any]) -> dict[str, Any]:
    import sys
    import numpy as np
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, AllChem
    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem import RDConfig
    import os
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)

    rt = _runtime(); model, system = rt["model"], rt["system"]
    seed, delta, target = task["smiles"], task["delta"], task["target"]
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = gen.GetFingerprint(Chem.MolFromSmiles(seed))

    def props(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        q = float(QED.qed(m)); s = float(sascorer.calculateScore(m))
        sim = float(DataStructs.TanimotoSimilarity(seed_fp, gen.GetFingerprint(m)))
        v = max(max(0.0, QED_MIN - q) / QED_MIN, max(0.0, s - SA_MAX) / SA_MAX,
                max(0.0, delta - sim) / delta)
        return {"qed": q, "sa": s, "sim": sim, "v": v}

    rng = np.random.default_rng(task["seed_rng"])
    archive = [{"smiles": seed, **props(seed), "ds": None, "round": 0}]
    docked: dict[str, float] = {}
    rounds, per_round, parents_n = task["rounds"], task["per_round"], task["parents"]
    log = []

    for rd in range(1, rounds + 1):
        scored = sorted(archive, key=lambda a: (a["v"] > 0,
                        a["ds"] if (a["v"] == 0 and a["ds"] is not None) else 0.0, a["v"]))
        parents = scored[:parents_n]
        cand: dict[str, dict] = {}
        for p in parents:
            try:
                st = pad_molecular_graph(smiles_to_molecular_graph(p["smiles"]), CANONICAL_SLOTS)
                law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            except Exception:
                continue
            if not law.marks:
                continue
            pr = np.array([m.probability for m in law.marks], float)
            for i in np.argsort(-pr)[:APPLY_CAP]:
                mk = law.marks[int(i)]
                try:
                    y = canonical_state_key(system.apply(st, mk.executor_rule_name, mk.action))
                except Exception:
                    continue
                if not y or y == p["smiles"] or y in docked or y in cand:
                    continue
                cand[y] = {"rtheta": float(pr[int(i)])}
        if not cand:
            break
        for y in list(cand):
            pv = props(y)
            if pv is None:
                del cand[y]
            else:
                cand[y].update(pv)
        if not cand:
            break

        keys = list(cand)
        w = np.array([cand[y]["rtheta"] for y in keys], float)
        w = w * np.exp(-np.array([cand[y]["v"] for y in keys]) / TAU_V)
        w = np.clip(w, 1e-30, None); w /= w.sum()
        k = min(per_round, len(keys))
        pick = [keys[i] for i in rng.choice(len(keys), size=k, replace=False, p=w)]

        # ---- THE DIAGNOSTIC ----
        feas_enum = [y for y in keys if cand[y]["v"] == 0.0]
        feas_pick = [y for y in pick if cand[y]["v"] == 0.0]
        order = np.argsort(-w)
        rank = {keys[int(j)]: int(r) for r, j in enumerate(order)}
        missed = sorted(
            ({"rank": rank[y], "weight": float(w[keys.index(y)]),
              "rtheta": cand[y]["rtheta"], "qed": cand[y]["qed"],
              "sa": cand[y]["sa"], "sim": cand[y]["sim"]}
             for y in feas_enum if y not in pick),
            key=lambda d: d["rank"])[:5]
        log.append({"round": rd, "n_cand": len(keys),
                    "n_feas_enum": len(feas_enum), "n_feas_picked": len(feas_pick),
                    "min_v": round(float(min(cand[y]["v"] for y in keys)), 4),
                    "missed_feasible": missed})

        for j, y in enumerate(pick):
            ds = _dock(y, target, f"d{task['idx']}_{rd}_{j}")
            docked[y] = ds if ds is not None else 0.0
            if ds is not None:
                archive.append({"smiles": y, **cand[y], "ds": ds, "round": rd})

    feas = [a for a in archive if a["v"] == 0.0 and a["ds"] is not None]
    del rng
    tot_enum = sum(r["n_feas_enum"] for r in log)
    tot_pick = sum(r["n_feas_picked"] for r in log)
    out = {**{k: v for k, v in task.items() if k != "smiles"},
            "rounds_run": len(log), "n_docked": len(docked),
            "feasible_enumerated_total": tot_enum,
            "feasible_docked_total": tot_pick,
            "returned_feasible": len(feas),
            "verdict": ("SELECTION DEFECT: feasible candidates were enumerated and not docked"
                        if tot_enum > 0 and tot_pick == 0 else
                        "MYOPIA: no feasible candidate was ever enumerated"
                        if tot_enum == 0 else
                        "feasible found and docked"),
            "per_round": log}
    # PERSIST BEFORE RETURNING. Three runs have now been lost by holding
    # results in memory until a driver finished. A cell that completes is
    # written and committed here, so a later crash costs at most one cell.
    try:
        d = Path("/artifacts/t4_selection_diag"); d.mkdir(parents=True, exist_ok=True)
        (d / f"cell_{task['idx']}_{task['target']}.json").write_text(json.dumps(out, indent=1))
        artifact_volume.commit()
    except Exception as e:
        print(f"  !! persist failed for idx {task['idx']}: {e}", flush=True)
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=2048, timeout=8 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(rounds: int, per_round: int, parents: int, only: str = "") -> dict[str, Any]:
    seeds = json.loads((REMOTE_ROOT / "docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    cells = [c for c in CELLS if not only or c["target"] == only]
    tasks = []
    for c in cells:
        s = seeds[c["idx"]]
        tasks.append({**c, "smiles": s["smiles"], "qed": s["qed"], "sa": s["sa"],
                      "rounds": rounds, "per_round": per_round, "parents": parents,
                      "seed_rng": 20260820 + c["idx"]})
    print(f"{len(tasks)} Level-1 failure cell(s), {rounds}x{per_round} dockings each\n", flush=True)
    out = []
    for r in diagnose.map(tasks, order_outputs=False, return_exceptions=True,
                          wrap_returned_exceptions=False):
        if not isinstance(r, dict):
            print(f"  !! {type(r).__name__}: {str(r)[:120]}", flush=True); continue
        out.append(r)
        print(f"  {r['target']:6s} {r['chembl']:14s} "
              f"feasible enumerated {r['feasible_enumerated_total']:>4}  "
              f"docked {r['feasible_docked_total']:>3}  -> {r['verdict']}", flush=True)
        for pr in r["per_round"]:
            if pr["n_feas_enum"] and not pr["n_feas_picked"]:
                m = pr["missed_feasible"][0] if pr["missed_feasible"] else {}
                print(f"      round {pr['round']:>2}: {pr['n_feas_enum']} feasible of "
                      f"{pr['n_cand']} enumerated, 0 docked; best missed rank "
                      f"{m.get('rank','?')} weight {m.get('weight',0):.2e}", flush=True)
    if not out:                       # fall back to whatever cells persisted
        d = Path("/artifacts/t4_selection_diag")
        if d.exists():
            out = [json.loads(f.read_text()) for f in sorted(d.glob("cell_*.json"))]
            print(f"  driver had nothing; recovered {len(out)} cells from the volume",
                  flush=True)
    if not out:
        raise RuntimeError("no cells completed")
    return {"cells": out, "rounds": rounds, "per_round": per_round}


@app.function(image=image, cpu=(1.0, 1.0), memory=1024, timeout=600,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def salvage() -> dict[str, Any]:
    """Read back whatever cells persisted, independent of the driver."""
    d = Path("/artifacts/t4_selection_diag")
    cells = [json.loads(f.read_text()) for f in sorted(d.glob("cell_*.json"))] if d.exists() else []
    print(f"  salvaged {len(cells)} cells", flush=True)
    return {"cells": cells, "salvaged": True}


@app.local_entrypoint()
def main(rounds: int = 10, per_round: int = 20, only: str = "") -> None:
    try:
        o = drive.remote(rounds, per_round, 3, only)
    except Exception as e:                     # driver died; salvage the volume
        print(f"  driver failed ({type(e).__name__}); salvaging persisted cells")
        o = salvage.remote()
    p = Path(__file__).resolve().parents[1] / "diagnostics/genmol_t4_selection_diag.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
