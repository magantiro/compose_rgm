"""GATE 0.5 for GenMol Table 4: can R_theta reach the admissible region at all?

WHY THIS COMES BEFORE ANY OPTIMIZATION RUN. Table 4 requires the RETURNED lead to
satisfy QED >= 0.6, raw SA <= 4 and Tanimoto similarity >= delta to the seed. The
seeds do not satisfy this themselves: 9 of the 15 have QED < 0.6, fa7's three sit
at 0.284 / 0.186 / 0.156. So the task is to climb from QED 0.156 to QED 0.6 while
staying Tanimoto-similar to the QED-0.156 molecule. QED and similarity pull
against each other, which is the most likely reason GenMol reports a dash on fa7
at both delta.

That splits the constraints into two kinds, and the distinction is the whole
point of the measurement:

    sim >= delta   measured against a FIXED seed, so it can be held at every
                   step. Pathwise. Enforced here as a hard filter on successors.
    QED, SA        must be REACHED, not maintained. Endpoint constraints. The
                   trajectory must be permitted to pass through states that
                   violate them, or it cannot leave the seed at all.

This job spends no oracle calls and no docking. It asks only whether the region
is reachable and at what depth, because that depth is what sets the edit budget
for the optimization run. Guessing the budget instead of measuring it is how a
benchmark entry fails for a reason that has nothing to do with the method.

A NEGATIVE RESULT IS USEFUL. If fa7's seeds cannot reach the region within the
horizon, we will produce dashes exactly where GenMol does, and we learn that for
a few cents rather than after a full optimization run.

The beam is scored by QED subject to the similarity filter. That measures
REACHABILITY of the constraint region, not docking quality; no docking score
enters this job, so nothing here can be tuned toward the reported metric.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = (_base_image
         .add_local_file(ROOT / "docs/GENMOL_T4_SEEDS.json",
                         str(REMOTE_ROOT / "docs/GENMOL_T4_SEEDS.json"), copy=True)
         .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"}))

app = modal.App("genmol-t4-reach")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48

DEPTH = 6          # edits explored
BEAM = 10          # states carried per depth
APPLY_CAP = 200    # marks executed per state, top-probability first

_RT: dict[str, Any] = {}


def _runtime():
    if "model" in _RT:
        return _RT
    import sys

    import torch
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source)
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime, load_materialized_scorer_state)
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME)
    from compose_v4.experiments.production_successor_kernel import (
        _default_rewrite_system)
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
    model.eval()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    _RT.update({"model": model, "system": _default_rewrite_system(model)})
    return _RT


@app.function(image=image, cpu=(1.0, 1.0), memory=int(4.5 * 1024),
              timeout=4 * 60 * 60, max_containers=40,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def reach(task: dict[str, Any]) -> dict[str, Any]:
    import os
    import sys

    import numpy as np
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDConfig, RDLogger
    from rdkit.Chem import QED, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)

    rt = _runtime()
    model, system = rt["model"], rt["system"]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed, delta = task["smiles"], task["delta"]
    seed_fp = gen.GetFingerprint(Chem.MolFromSmiles(seed))

    def props(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return {"qed": float(QED.qed(m)),
                "sa": float(sascorer.calculateScore(m)),
                "sim": float(DataStructs.TanimotoSimilarity(
                    seed_fp, gen.GetFingerprint(m)))}

    def admissible(p):
        return p["qed"] >= 0.6 and p["sa"] <= 4.0 and p["sim"] >= delta

    t0 = time.perf_counter()
    frontier = [(seed, props(seed))]
    per_depth, hits, feas_counts = [], [], []
    first_depth = None

    for d in range(1, DEPTH + 1):
        cand: dict[str, Any] = {}
        for smi, _p in frontier:
            try:
                st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
                law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            except Exception:
                continue
            if not law.marks:
                continue
            pr = np.array([m.probability for m in law.marks], float)
            order = np.argsort(-pr)[:APPLY_CAP]
            for i in order:
                mk = law.marks[int(i)]
                try:
                    y = canonical_state_key(system.apply(
                        st, mk.executor_rule_name, mk.action))
                except Exception:
                    continue
                if not y or y == smi or y in cand:
                    continue
                p = props(y)
                if p is not None:
                    cand[y] = p
        if not cand:
            break
        feasible = {k: v for k, v in cand.items() if v["sim"] >= delta}
        feas_counts.append(len(feasible))
        adm = [(k, v) for k, v in feasible.items() if admissible(v)]
        if adm and first_depth is None:
            first_depth = d
        hits.extend([{"depth": d, "smiles": k, **v} for k, v in adm[:5]])
        pool = feasible or cand
        best_qed = max(v["qed"] for v in pool.values())
        per_depth.append({"depth": d, "n_candidates": len(cand),
                          "n_sim_feasible": len(feasible),
                          "n_admissible": len(adm),
                          "best_qed": round(best_qed, 4)})
        frontier = sorted(pool.items(), key=lambda kv: -kv[1]["qed"])[:BEAM]
        frontier = [(k, v) for k, v in frontier]

    return {**task, "first_admissible_depth": first_depth,
            "n_admissible_found": len(hits),
            "per_depth": per_depth,
            "median_sim_feasible": (float(np.median(feas_counts))
                                    if feas_counts else 0.0),
            "examples": hits[:3],
            "seconds": round(time.perf_counter() - t0, 1)}


@app.function(image=image, cpu=(1.0, 1.0), memory=2048, timeout=6 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive() -> dict[str, Any]:
    seeds = json.loads((REMOTE_ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    tasks = [{**s, "delta": d} for s in seeds for d in (0.4, 0.6)]
    print(f"{len(tasks)} (seed, delta) cells   depth<={DEPTH} beam={BEAM} "
          f"apply_cap={APPLY_CAP}   NO DOCKING\n", flush=True)
    out = [r for r in reach.map(tasks, order_outputs=True, return_exceptions=True,
                                wrap_returned_exceptions=False)
           if isinstance(r, dict)]
    print(f"{'target':8s}{'delta':>6}{'seedQED':>9}{'reach@':>8}{'#adm':>6}"
          f"{'med feasible':>14}{'bestQED':>9}", flush=True)
    for r in sorted(out, key=lambda x: (x["target"], x["delta"])):
        bq = max((d["best_qed"] for d in r["per_depth"]), default=0.0)
        fd = r["first_admissible_depth"]
        print(f"  {r['target']:8s}{r['delta']:>6.1f}{r['seed_qed']:>9.3f}"
              f"{(str(fd) if fd else 'NONE'):>8}{r['n_admissible_found']:>6}"
              f"{r['median_sim_feasible']:>14.0f}{bq:>9.3f}", flush=True)
    n_reach = sum(1 for r in out if r["first_admissible_depth"])
    print(f"\n  cells reaching the admissible region: {n_reach}/{len(out)}")
    print("  A cell that never reaches it will be a dash for us too. Compare")
    print("  those against GenMol's dashes before drawing any conclusion.")
    return {"depth": DEPTH, "beam": BEAM, "apply_cap": APPLY_CAP,
            "n_cells": len(out), "n_reaching": n_reach, "cells": out}


@app.local_entrypoint()
def main() -> None:
    o = drive.remote()
    p = Path(__file__).resolve().parents[1] / "diagnostics/genmol_t4_reachability.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
