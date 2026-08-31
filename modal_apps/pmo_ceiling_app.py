"""Oracle-GREEDY REACHABILITY DIAGNOSTIC on scaffold_hop. Not a benchmark.

WHAT THIS DOES AND DOES NOT MEASURE -- read before citing any number from it.

It measures: how far locally perfect objective information gets us. The beam
ranks by IMMEDIATE true oracle score at every step.

It does NOT measure the maximum score reachable within COMPOSE's operator
support. Calling it a "support ceiling" would be wrong. A legitimate route may
look like

    0.63 -> 0.55 -> 0.51 -> 0.72 -> 0.91

and an oracle-GREEDY beam prunes that sacrificial branch at step two. Failing to
reach a score here is therefore consistent with two very different worlds: the
route does not exist, or the route exists but requires accepting local loss.
Distinguishing them is precisely what future-value control is for, so a low
result here ARGUES FOR testing h, it does not rule the support out.

THIS RUN IS NOT REPORTABLE AND MUST NEVER APPEAR IN A RESULTS TABLE. It cheats
deliberately: every enumerated successor is scored with the true scaffold_hop
oracle, off-budget, and the beam keeps the best. That answers one question no
legitimate run can answer cheaply --

    how far does locally perfect objective information get us?

The reason this matters. Our clean search plateaus near 0.48 and the full
249,455-molecule oracle prescreen tops out at top-10 0.5194, while GenMol reports
0.936. Two very different worlds are consistent with that:

    reaches ~0.9   -> routes exist AND greedy local information suffices to walk
                      them. The gap is then about exploiting task information,
                      not about lookahead.
    stalls ~0.5-6  -> greedy local information is NOT sufficient. Either the
                      routes need sacrificial steps, which is the case for
                      future-value control, or they are absent from the support.
                      This diagnostic cannot separate those two, and must not be
                      reported as though it had.

Establishing which world we are in costs ~64 fiber expansions. Doing it before
committing to controller development is the whole point.

SIZING. Per the standing rule, four fibers are timed first and the full run is
projected from that measurement before the beam starts.
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

image = (
    _opt_image
    .pip_install("PyTDC==0.3.6", "requests", "fuzzywuzzy", "seaborn", "networkx")
    .add_local_file(ROOT / "modal_apps/genmol_t4_opt_app.py",
                    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)
    .add_local_file(ROOT / "artifacts/oracles/molleo_task3_v1/jnk3_forest.npz",
                    "/frozen/jnk3_forest.npz", copy=True)
)

app = modal.App("pmo-ceiling")


def _shim():
    import sys, types
    six = types.ModuleType("rdkit.six")
    six.iteritems = lambda d, **k: iter(d.items())
    six.itervalues = lambda d, **k: iter(d.values())
    six.iterkeys = lambda d, **k: iter(d.keys())
    six.string_types = (str,)
    sys.modules["rdkit.six"] = six
    import rdkit
    rdkit.six = six


@app.function(image=image, cpu=(2.0, 2.0), memory=int(8 * 1024),
              timeout=3 * 60 * 60, max_containers=8,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def ceiling(task: str = "scaffold_hop", width: int = 8, depth: int = 8,
            n_seeds: int = 8) -> dict[str, Any]:
    import os, sys
    import numpy as np
    _shim(); os.chdir("/tmp")
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from tdc import Oracle

    t0 = time.time()
    rt = _runtime(); model, system = rt["model"], rt["system"]
    o = Oracle(name=task)
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fp = lambda s: gen.GetFingerprint(Chem.MolFromSmiles(s))

    TT: dict[str, dict] = {}
    C = {"fibers": 0, "scored": 0}

    def fiber(smi):
        if smi in TT:
            return TT[smi]
        C["fibers"] += 1
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        except Exception:
            TT[smi] = {}; return TT[smi]
        if not law.marks:
            TT[smi] = {}; return TT[smi]
        pr = np.array([m.probability for m in law.marks], float)
        out = {}
        for i in np.argsort(-pr)[:APPLY_CAP]:
            mk = law.marks[int(i)]
            try:
                y = canonical_state_key(system.apply(st, mk.executor_rule_name, mk.action))
            except Exception:
                continue
            if y and y != smi and y not in out:
                out[y] = float(pr[int(i)])
        TT[smi] = out
        return out

    bank = json.loads(
        Path(f"/artifacts/pmo_matched_init/bank_{task}.json").read_text())["bank"]

    # ---- 1%-scale timing microbenchmark, BEFORE the beam ----
    probe = [b["smiles"] for b in bank[:4]]
    tb = time.time()
    n_succ = sum(len(fiber(s)) for s in probe)
    t_fiber = (time.time() - tb) / max(len(probe), 1)
    per_succ = n_succ / max(len(probe), 1)
    ts = time.time(); _ = [float(o(s)) for s in probe]; t_oracle = (time.time() - ts) / 4
    proj = (width * depth * (t_fiber + per_succ * t_oracle)) / 60
    print(f"  MICROBENCH  fiber {t_fiber:.2f}s  successors/fiber {per_succ:.0f}  "
          f"oracle {t_oracle*1000:.1f}ms  ->  beam {width}x{depth} projected "
          f"{proj:.1f} min", flush=True)

    # ---- diverse seeds from the matched bank ----
    pool = [b for b in bank]
    seeds, seed_fps = [], []
    for b in pool:
        f = fp(b["smiles"])
        if all(DataStructs.TanimotoSimilarity(f, g) < 0.6 for g in seed_fps):
            seeds.append(b); seed_fps.append(f)
        if len(seeds) >= n_seeds:
            break
    print(f"  {len(seeds)} diverse seeds, scores "
          f"{[round(s['u'], 3) for s in seeds]}", flush=True)

    beam = [{"smi": s["smiles"], "u": float(s["u"]), "d": 0} for s in seeds]
    best = max(b["u"] for b in beam)
    by_depth = {0: best}
    trace = [{"depth": 0, "beam": len(beam), "best": best,
              "mean": float(np.mean([b["u"] for b in beam]))}]
    seen = {b["smi"] for b in beam}

    ood = {"smiles": [], "score": [], "depth": [], "parent": []}
    for d in range(1, depth + 1):
        cand: dict[str, float] = {}
        par: dict[str, str] = {}
        for p in beam:
            for y in fiber(p["smi"]):
                if y not in seen:
                    cand[y] = 0.0
                    par.setdefault(y, p["smi"])
        if not cand:
            print(f"  depth {d}: no new successors, beam exhausted", flush=True)
            break
        ks = list(cand)
        vals = []
        for y in ks:                       # TRUE oracle, off-budget, dev only
            try:
                vals.append(float(o(y)))
            except Exception:
                vals.append(0.0)
        C["scored"] += len(ks)
        # KEEP EVERY PAIR. A previous version of this file recorded only the
        # COUNT of scored successors and discarded all 18,031 (molecule, value)
        # pairs. Those pairs are the only out-of-distribution validation set we
        # have -- COMPOSE-generated molecules with true oracle values, including
        # the ones ABOVE the corpus maximum -- and regenerating them costs a full
        # re-run. Nothing expensive and deterministic leaves this loop unsaved.
        for y, v in zip(ks, vals):
            ood["smiles"].append(y); ood["score"].append(v)
            ood["depth"].append(d); ood["parent"].append(par.get(y, ""))
        order = np.argsort(-np.array(vals))
        nxt, nxt_fps = [], []
        for i in order:                    # best-first with a diversity guard
            y = ks[int(i)]
            f = fp(y)
            if f is None:
                continue
            if all(DataStructs.TanimotoSimilarity(f, g) < 0.85 for g in nxt_fps):
                nxt.append({"smi": y, "u": float(vals[int(i)]), "d": d})
                nxt_fps.append(f)
            if len(nxt) >= width:
                break
        seen.update(b["smi"] for b in nxt)
        beam = nxt or beam
        best = max(best, max(b["u"] for b in beam))
        by_depth[d] = best
        trace.append({"depth": d, "beam": len(beam), "best": best,
                      "mean": float(np.mean([b["u"] for b in beam])),
                      "candidates": len(ks),
                      "elapsed": round(time.time() - t0, 1)})
        print(f"  depth {d}: {len(ks):>5} successors scored  beam {len(beam)}  "
              f"best {best:.4f}  mean {trace[-1]['mean']:.4f}  "
              f"{time.time()-t0:.0f}s", flush=True)

    out = {"task": task, "reportable": False,
           "diagnostic": "oracle_greedy_reachability",
           "note": "ORACLE-GREEDY REACHABILITY DIAGNOSTIC. Cheats by scoring "
                   "every successor off-budget, and ranks GREEDILY by immediate "
                   "score. Never cite as a benchmark result, and never describe "
                   "the result as a support ceiling: a greedy beam prunes "
                   "sacrificial routes, so stalling here does NOT bound what the "
                   "operator support can reach.",
           "width": width, "depth": depth, "n_seeds": len(seeds),
           "microbench": {"t_fiber_s": round(t_fiber, 3),
                          "successors_per_fiber": round(per_succ, 1),
                          "t_oracle_ms": round(t_oracle * 1000, 2),
                          "projected_min": round(proj, 1)},
           "start_best": trace[0]["best"], "oracle_greedy_best": best,
           "beam_collapsed": None,
           "best_by_depth": {str(k): round(v, 5) for k, v in sorted(by_depth.items())},
           "trace": trace, "fibers": C["fibers"], "successors_scored": C["scored"],
           "seconds": round(time.time() - t0, 1)}
    d_ = Path("/artifacts/pmo_ceiling"); d_.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d_ / f"ood_{task}.npz",
                        smiles=np.array(ood["smiles"], dtype=object),
                        score=np.array(ood["score"], dtype=np.float32),
                        depth=np.array(ood["depth"], dtype=np.int16),
                        parent=np.array(ood["parent"], dtype=object))
    above = int((np.array(ood["score"]) > 0.5261).sum())
    out["ood_pairs"] = len(ood["smiles"])
    out["ood_above_corpus_max"] = above
    print(f"  persisted {len(ood['smiles']):,} OOD pairs "
          f"({above:,} above the 0.5261 corpus max)", flush=True)
    (d_ / f"ceiling_{task}.json").write_text(json.dumps(out, indent=1))
    artifact_volume.commit()
    print(f"\n  ORACLE-GREEDY {task}: {trace[0]['best']:.4f} -> {best:.4f} "
          f"in {C['fibers']} fibers, {C['scored']:,} off-budget scorings, "
          f"{out['seconds']:.0f}s", flush=True)
    return out


@app.local_entrypoint()
def main(task: str = "scaffold_hop", width: int = 8, depth: int = 8,
         n_seeds: int = 8, out: str = "") -> None:
    o = ceiling.remote(task, width, depth, n_seeds)
    p = Path(__file__).resolve().parents[1] / (out or f"diagnostics/pmo_ceiling_{task}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
