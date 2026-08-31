"""Can COMPOSE's executable transitions reach InVirtuoGen's published winners?

THE QUESTION THIS SETTLES. Every T4 controller arm has plateaued far below
InVirtuoGen's scores. Two very different explanations remain:

    SEARCH   their winning molecules lie inside COMPOSE's executable state space
             and our controller simply fails to find them.
    SUPPORT  those molecules are not reachable under our operator set at all,
             in which case no controller can ever get there.

Nothing we have run distinguishes these. This does, and it costs ZERO docking
calls -- their molecules already carry published docking scores.

ASYMMETRIC EVIDENCE, STATED UP FRONT. Finding a legal path is DEFINITIVE: the
target is reachable and the problem is search. Failing to find one is NOT proof
of unreachability -- it is bounded negative evidence, "not found within this beam
width and depth". Claiming a genuine support limitation would need an explicit
operator incompatibility or exhaustive bounded search. The artifact records
best-similarity-attained per depth so a near miss is distinguishable from a wall.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import (
    APPLY_CAP, CANONICAL_SLOTS, TIME_POINT,
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume, _runtime,
)
from modal_apps.genmol_t4_opt_app import image as _opt_image

image = _opt_image.add_local_file(
    ROOT / "modal_apps/genmol_t4_opt_app.py",
    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True
).add_local_file(ROOT / "diagnostics/ivg_winners.json",
                 str(REMOTE_ROOT / "docs/ivg_winners.json"), copy=True)

app = modal.App("genmol-t4-reach-ivg")

# DEPTH MUST EXCEED THE MINIMUM POSSIBLE PATH LENGTH, or a miss is uninformative.
# Each executable edit changes the heavy-atom count by at most one, so a winner
# |dHeavy| atoms away needs at least that many edits. Measured for these cells:
# 5ht1b s7 +15, parp1 s0 +14, braf s9 -14, braf s11 -10, braf s10 -8. The first
# run capped depth at 12 and therefore GUARANTEED a miss on three of five cells
# before searching -- those results carried no information. Depth is now derived
# per cell from the target, with headroom for edits that do not change atom count
# (restate, reorder, reroute).
WIDTH = 24
DEPTH_HEADROOM = 8          # extra depth beyond |dHeavy| for non-count-changing edits
DEPTH_CAP = 40
# One thread per beam member: each fiber() is single-threaded (OMP_NUM_THREADS=1),
# so WIDTH concurrent calls want WIDTH cores. At 6 threads a depth took 4 waves;
# at 24 it takes one. Depth itself cannot be parallelised -- depth d+1 needs d's
# beam -- so this is the only axis available inside a cell.
FIBER_THREADS = 24


@app.function(image=image, cpu=(28.0, 28.0), memory=int(24 * 1024),
              timeout=4 * 60 * 60, max_containers=16,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def reach(job: dict[str, Any]) -> dict[str, Any]:
    import os, sys
    import numpy as np
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)

    t0 = time.time()
    rt = _runtime(); model, system = rt["model"], rt["system"]
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    canon = lambda s: (lambda m: Chem.MolToSmiles(m) if m else None)(Chem.MolFromSmiles(s))

    seed = canon(job["seed_smiles"])
    targets = [canon(w["smiles"]) for w in job["winners"]]
    targets = [t for t in targets if t]
    tfps = [gen.GetFingerprint(Chem.MolFromSmiles(t)) for t in targets]

    def sim_to_targets(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return 0.0, -1
        f = gen.GetFingerprint(m)
        s = [DataStructs.TanimotoSimilarity(f, tf) for tf in tfps]
        i = int(np.argmax(s))
        return float(s[i]), i

    def fiber(smi):
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        except Exception:
            return []
        pr = np.array([m.probability for m in law.marks], float)
        out = []
        for i in np.argsort(-pr)[:APPLY_CAP]:
            mk = law.marks[int(i)]
            try:
                y = canonical_state_key(system.apply(st, mk.executor_rule_name, mk.action))
            except Exception:
                continue
            if y and y != smi:
                out.append(y)
        return out

    # per-cell depth from the hardest target we are chasing
    from rdkit.Chem import rdMolDescriptors
    hs = Chem.MolFromSmiles(seed).GetNumHeavyAtoms()
    need = max(abs(Chem.MolFromSmiles(t).GetNumHeavyAtoms() - hs) for t in targets)
    depth_max = min(DEPTH_CAP, need + DEPTH_HEADROOM)
    print(f"  [{job['cell']}] seed heavy={hs}, max |dHeavy| to a winner={need}, "
          f"searching to depth {depth_max}", flush=True)

    beam = [seed]
    seen = {seed}
    best_sim, best_i = sim_to_targets(seed)
    trace = [{"depth": 0, "best_sim": round(best_sim, 4), "beam": 1}]
    hit = None
    for d in range(1, depth_max + 1):
        # THREADED. The beam's 24 fiber enumerations are independent and this is a
        # diagnostic, not a reported number, so shared-model inference under
        # no_grad is an acceptable risk here in exchange for ~2-3x on the
        # dominant cost. Results are order-independent: everything lands in one
        # dedup set.
        from concurrent.futures import ThreadPoolExecutor
        cand = []
        with ThreadPoolExecutor(max_workers=FIBER_THREADS) as ex:
            for ys in ex.map(fiber, beam):
                for y in ys:
                    if y not in seen:
                        seen.add(y); cand.append(y)
        if not cand:
            break
        scored = []
        for y in cand:
            s, i = sim_to_targets(y)
            if y in targets:
                hit = {"depth": d, "smiles": y, "target_index": targets.index(y)}
                break
            scored.append((s, i, y))
        if hit:
            break
        # Similarity alone is a poor guide when the target is 14 atoms away: a
        # molecule can look similar while having no path to the right size. Rank
        # by similarity MINUS a penalty on remaining heavy-atom distance, so the
        # beam is pushed to actually traverse the size gap.
        tgt_h = [Chem.MolFromSmiles(t).GetNumHeavyAtoms() for t in targets]
        rescored = []
        for s_, i_, y_ in scored:
            m_ = Chem.MolFromSmiles(y_)
            gap = abs(m_.GetNumHeavyAtoms() - tgt_h[i_]) if m_ else 99
            rescored.append((s_ - 0.02 * gap, i_, y_, s_))
        rescored.sort(reverse=True)
        scored = [(s_orig, i_, y_) for _, i_, y_, s_orig in rescored]
        beam = [y for _, _, y in scored[:WIDTH]]
        if scored and scored[0][0] > best_sim:
            best_sim, best_i = scored[0][0], scored[0][1]
        # the best MOLECULE, not only its score: without it the structural gap
        # to the target cannot be diagnosed afterwards without re-running.
        trace.append({"depth": d, "best_sim": round(best_sim, 4),
                      "best_smiles": scored[0][2] if scored else None,
                      "best_heavy": (Chem.MolFromSmiles(scored[0][2]).GetNumHeavyAtoms()
                                     if scored else None),
                      "beam": len(beam), "expanded": len(cand),
                      "elapsed": round(time.time() - t0, 1)})
        print(f"  [{job['cell']}] depth {d}: expanded {len(cand):>5} best_sim {best_sim:.4f}",
              flush=True)
        # Persist EVERY depth. Preemption restarts a cell from depth 1 and we have
        # already watched it reset this audit mid-run; at depth 22 that means a
        # cell may never finish. One depth lost instead of all of them.
        try:
            d_ = Path("/artifacts/t4_reach_ivg"); d_.mkdir(parents=True, exist_ok=True)
            (d_ / f"{job['cell']}.partial.json").write_text(json.dumps(
                {"cell": job["cell"], "depth_reached": d, "best_sim": round(best_sim, 4),
                 "max_depth": depth_max, "min_edits_required": need,
                 "trace": trace, "complete": False}, indent=1))
            artifact_volume.commit()
        except Exception:
            pass

    out = {"cell": job["cell"], "target": job["target"], "delta": job["delta"],
           "seed": seed, "n_targets": len(targets),
           "EXACT_HIT": hit,
           "best_similarity_to_any_winner": round(best_sim, 4),
           "closest_winner_index": best_i,
           "width": WIDTH, "max_depth": depth_max,
           "min_edits_required": need, "seed_heavy": hs, "trace": trace,
           "evidence_note": ("A hit is DEFINITIVE reachability. No hit is bounded "
                             "negative evidence only: not found at width "
                             f"{WIDTH}, depth {depth_max} (>= the {need} edits "
                             "structurally required)."),
           "seconds": round(time.time() - t0, 1)}
    d_ = Path("/artifacts/t4_reach_ivg"); d_.mkdir(parents=True, exist_ok=True)
    (d_ / f"{job['cell']}.json").write_text(json.dumps(out, indent=1))
    artifact_volume.commit()
    print(f"  [{job['cell']}] DONE hit={bool(hit)} best_sim={best_sim:.4f} "
          f"{out['seconds']:.0f}s", flush=True)
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=2048, timeout=4 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive() -> dict[str, Any]:
    W = json.loads((REMOTE_ROOT / "docs/ivg_winners.json").read_text())
    seeds = json.loads((REMOTE_ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    jobs = []
    for cell, v in W.items():
        jobs.append({"cell": cell, "target": v["target"], "delta": v["delta"],
                     "seed_smiles": seeds[v["seed_idx"]]["smiles"],
                     "winners": v["winners"]})
    print(f"  {len(jobs)} cells: {[j['cell'] for j in jobs]}\n", flush=True)
    out = []
    for r in reach.map(jobs, order_outputs=False, return_exceptions=True,
                       wrap_returned_exceptions=False):
        if not isinstance(r, dict):
            print(f"  !! {type(r).__name__}: {str(r)[:160]}", flush=True); continue
        out.append(r)
    return {"cells": out}


@app.local_entrypoint()
def main(out: str = "") -> None:
    o = drive.remote()
    p = Path(__file__).resolve().parents[1] / (out or "diagnostics/genmol_t4_reach_ivg.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
