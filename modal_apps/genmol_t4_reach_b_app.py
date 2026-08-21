"""GATE 0.5b: corrected reachability measurement for GenMol Table 4. Run ONCE.

WHAT WAS WRONG WITH 0.5a. The beam ranked candidates by QED alone while
feasibility is conjunctive over QED, SA and similarity. The two 5ht1b misses are
diagnostic: QED reached 0.88 and 0.81 while raw SA (seed 4.69) stayed the binding
constraint. The search optimized a requirement that was already satisfiable and
ignored the one that was not. That is an instrument defect, not a result.

A second defect, found while rewriting: 0.5a fell back to the unfiltered
candidate pool when the similarity filter emptied a level. That silently broke
the pathwise similarity restriction it claimed to enforce. Removed here; a branch
whose similarity-feasible successor set is empty now dies, which is the honest
measurement.

WHAT THIS RUN FIXES, FROZEN BEFORE LAUNCH.

  hard support restriction   sim(x, x0) >= delta at EVERY step
  beam rank                  v(x) = max{ [0.6 - QED(x)]_+ / 0.6 ,
                                         [SA(x)  - 4  ]_+ / 4   }
                             lower is better; v(x) = 0 means the endpoint
                             constraints are met.  The max is used because the
                             benchmark is conjunctive: it forces search onto
                             whichever requirement is most violated instead of
                             letting excellent QED compensate for bad SA.
  depth                      16, with the curve reported at H = {4, 8, 12, 16}
  beam                       10        apply cap  200

Depth 6 in 0.5a was not a benchmark constraint. Table 4 caps generated and
evaluated molecules, not molecular edits, so the horizon was ours to choose and
choosing it short understated reachability. The depth curve replaces a guess with
a measurement.

THIS IS A MEASUREMENT, NOT AN OPTIMIZATION. No docking score is computed, so
nothing here can be tuned toward the reported metric. The denominators above are
fixed before launch and this diagnostic is closed afterwards: no QED weighting
variants, no SA weight tuning, no beam-width rescue.

The question it answers, and the only one:
    does the frozen COMPOSE rewrite process contain routes into the T4 feasible
    region, and roughly how many edits do those routes need?

NOTE ON A RESTRICTION WE IMPOSE. The benchmark requires sim >= delta of the
RETURNED molecule only. Enforcing it at every step is our choice: it is
sufficient, and it is what makes support restriction meaningful. It could in
principle cost cells whose only route detours through dissimilar chemistry. That
is a property of the policy we are measuring, not of the benchmark, and it is
recorded here so the number is read correctly.
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
         .add_local_file(ROOT / "docs/GENMOL_T4_DEV_SEEDS.json",
                         str(REMOTE_ROOT / "docs/GENMOL_T4_DEV_SEEDS.json"), copy=True)
         .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"}))

app = modal.App("genmol-t4-reach-b")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48

DEPTH, BEAM, APPLY_CAP = 16, 10, 200
CURVE = (4, 8, 12, 16)
QED_MIN, SA_MAX = 0.6, 4.0

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
              timeout=8 * 60 * 60, max_containers=40,
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
        q = float(QED.qed(m))
        s = float(sascorer.calculateScore(m))
        sim = float(DataStructs.TanimotoSimilarity(seed_fp, gen.GetFingerprint(m)))
        # FROZEN feasibility distance: conjunctive, scale-free, 0 == feasible.
        v = max(max(0.0, QED_MIN - q) / QED_MIN, max(0.0, s - SA_MAX) / SA_MAX)
        return {"qed": q, "sa": s, "sim": sim, "v": v}

    t0 = time.perf_counter()
    p0 = props(seed)
    frontier = [(seed, p0)]
    per_depth, hits = [], []
    first_depth = None if p0["v"] > 0 else 0

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
            for i in np.argsort(-pr)[:APPLY_CAP]:
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
        # Hard pathwise similarity restriction. NO fallback: an empty
        # similarity-feasible set kills the branch, which is the honest answer.
        feasible = {k: v for k, v in cand.items() if v["sim"] >= delta}
        if not feasible:
            per_depth.append({"depth": d, "n_candidates": len(cand),
                              "n_sim_feasible": 0, "n_admissible": 0,
                              "best_v": None, "best_qed": None, "terminated": True})
            break
        adm = [(k, v) for k, v in feasible.items() if v["v"] == 0.0]
        if adm and first_depth is None:
            first_depth = d
        hits.extend([{"depth": d, "smiles": k, **v} for k, v in adm[:5]])
        per_depth.append({"depth": d, "n_candidates": len(cand),
                          "n_sim_feasible": len(feasible), "n_admissible": len(adm),
                          "best_v": round(min(v["v"] for v in feasible.values()), 4),
                          "best_qed": round(max(v["qed"] for v in feasible.values()), 4),
                          "terminated": False})
        frontier = sorted(feasible.items(), key=lambda kv: kv[1]["v"])[:BEAM]

    return {**task, "first_admissible_depth": first_depth,
            "n_admissible_found": len(hits), "per_depth": per_depth,
            "best_v_overall": min((p["best_v"] for p in per_depth
                                   if p["best_v"] is not None), default=None),
            "examples": hits[:3], "seconds": round(time.perf_counter() - t0, 1)}


@app.function(image=image, cpu=(1.0, 1.0), memory=2048, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(which: str = "bench") -> dict[str, Any]:
    f = "GENMOL_T4_SEEDS.json" if which == "bench" else "GENMOL_T4_DEV_SEEDS.json"
    raw = json.loads((REMOTE_ROOT / f"docs/{f}").read_text())
    seeds = raw if isinstance(raw, list) else raw["seeds"]
    tasks = [{**s, "delta": d} for s in seeds for d in (0.4, 0.6)]
    print(f"{len(tasks)} cells  depth<={DEPTH} beam={BEAM} cap={APPLY_CAP}  "
          f"v = max([0.6-QED]+/0.6, [SA-4]+/4)   NO DOCKING\n", flush=True)
    raw_out = list(reach.map(tasks, order_outputs=True, return_exceptions=True,
                             wrap_returned_exceptions=False))
    out = [r for r in raw_out if isinstance(r, dict)]
    errs = [r for r in raw_out if not isinstance(r, dict)]
    if errs:
        from collections import Counter
        print(f"  !! {len(errs)}/{len(tasks)} cells FAILED", flush=True)
        for m, k in Counter(f"{type(e).__name__}: {e}"[:180] for e in errs).most_common(5):
            print(f"     x{k}  {m}", flush=True)
    if not out:
        raise RuntimeError(f"all {len(tasks)} cells failed; refusing to write empty")

    result = {"depth": DEPTH, "beam": BEAM, "apply_cap": APPLY_CAP,
              "curve": {str(h): sum(1 for r in out
                                    if r["first_admissible_depth"] is not None
                                    and r["first_admissible_depth"] <= h)
                        for h in CURVE},
              "n_cells": len(out),
              "n_reaching": sum(1 for r in out
                                if r["first_admissible_depth"] is not None),
              "cells": out}
    try:
        _summarise(out, which)
    except Exception as e:                      # never lose the run to a printf
        print(f"  (summary print failed: {type(e).__name__}: {e}; "
              f"results are intact)", flush=True)
    return result


def _summarise(out, which):
    print(f"{'target':8s}{'DS':>6}{'delta':>6}{'seedQED':>8}{'seedSA':>7}"
          f"{'reach@':>8}{'#adm':>6}{'best v':>8}", flush=True)
    for r in sorted(out, key=lambda x: (x["target"], -(x.get("published_ds") or 0.0), x["delta"])):
        fd = r["first_admissible_depth"]
        bv = r["best_v_overall"]
        print(f"  {r['target']:8s}{-(r.get('published_ds') or 0.0):>6.1f}{r['delta']:>6.1f}"
              f"{(r.get('seed_qed', r.get('qed')) or 0.0):>8.3f}"
              f"{(r.get('seed_sa', r.get('sa')) or 0.0):>7.2f}"
              f"{(str(fd) if fd is not None else 'NONE'):>8}"
              f"{r['n_admissible_found']:>6}"
              f"{(f'{bv:.3f}' if bv is not None else '-'):>8}", flush=True)

    print(f"\n  DEPTH CURVE (cumulative cells reaching the feasible region)")
    for h in CURVE:
        n = sum(1 for r in out if r["first_admissible_depth"] is not None
                and r["first_admissible_depth"] <= h)
        print(f"    H <= {h:>2} : {n:>2}/{len(out)}")
    total = sum(1 for r in out if r["first_admissible_depth"] is not None)
    ref = "   (GenMol solves 26/30)" if which == "bench" else ""
    print(f"\n  total reaching by depth {DEPTH}: {total}/{len(out)}{ref}")
    print("  Cells not reached are cells this search did not find a route to;")
    print("  that is weaker than unreachable. Diagnostic closes here either way.")


@app.local_entrypoint()
def main(which: str = "bench", out: str = "") -> None:
    o = drive.remote(which)
    p = Path(__file__).resolve().parents[1] / (
        out or f"diagnostics/genmol_t4_reachability_{'b' if which=='bench' else 'dev'}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
