"""Reachability audit with CONTAINER fan-out instead of core threading.

WHY CONTAINERS, NOT THREADS. Each fiber() enumeration is single-threaded
(OMP_NUM_THREADS=1 for determinism), so in-container threading only helps up to
the core count and shares one torch model across threads -- a correctness risk on
the object that produces every number. Modal's own unit of parallelism is the
container, and `.map` keeps them WARM between calls, so R_theta loads once per
container and is then reused across every depth. The beam's 24 expansions become
24 concurrent containers rather than 24 threads on one.

Depth stays sequential -- depth d+1 needs d's beam -- so the fan-out is per depth.

The audit answers one question: do InVirtuoGen's published winners lie inside
COMPOSE's executable state space? A found path is definitive. A miss is bounded
negative evidence only, at this beam width and depth.

Depth is derived per cell: each edit changes heavy-atom count by at most one, so
a winner |dHeavy| away needs at least that many edits. The first version capped
depth at 12 and thereby guaranteed a miss on three of five cells.
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

app = modal.App("genmol-t4-reach-fanout")

# Modal caps concurrency at 100. 5 cells x 16 expand containers = 80, leaving
# headroom for the 5 drivers and any retry churn. Exceeding the cap makes calls
# queue rather than fail, which would silently serialise the fan-out.
WIDTH = 32              # beam members == concurrent containers per depth
PER_CONTAINER = 1       # one beam member per container: 24 concurrent per cell
DEPTH_HEADROOM = 8
DEPTH_CAP = 40


@app.function(image=image, cpu=(2.0, 2.0), memory=int(6 * 1024),
              timeout=60 * 60, max_containers=80, scaledown_window=900,
              enable_memory_snapshot=True,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def expand(smis: list[str]) -> dict[str, list[str]]:
    """Enumerate the executable successor set of each molecule. Warm-reused."""
    import os, sys
    import numpy as np
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    rt = _runtime(); model, system = rt["model"], rt["system"]
    out: dict[str, list[str]] = {}
    for smi in smis:
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        except Exception:
            out[smi] = []; continue
        pr = np.array([m.probability for m in law.marks], float)
        ys = []
        # FULL law, not the top-APPLY_CAP by model probability. This is a
        # REACHABILITY question -- "does a path exist" -- so truncating the
        # successor set by R_theta's probability answers a different question
        # than the one asked. It also invalidated the previous audit: parp1's 57
        # ring-forming marks all rank >= 305 against a cap of 300, so the audit
        # concluded the growth targets were unreachable (best similarity 0.5500)
        # using an expansion that could not form a single ring, while the
        # molecules it was searching for carry +3 rings.
        for i in np.argsort(-pr):
            mk = law.marks[int(i)]
            try:
                y = canonical_state_key(system.apply(st, mk.executor_rule_name, mk.action))
            except Exception:
                continue
            if y and y != smi:
                ys.append(y)
        out[smi] = ys
    return out


@app.function(image=image, cpu=(2.0, 2.0), memory=int(6 * 1024),
              timeout=6 * 60 * 60, volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive_cell(job: dict[str, Any]) -> dict[str, Any]:
    """Hold the beam; fan each depth out to `expand` containers."""
    import os, sys
    import numpy as np
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    t0 = time.time()
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    canon = lambda s: (lambda m: Chem.MolToSmiles(m) if m else None)(Chem.MolFromSmiles(s))

    seed = canon(job["seed_smiles"])
    targets = [t for t in (canon(w["smiles"]) for w in job["winners"]) if t]
    tmols = [Chem.MolFromSmiles(t) for t in targets]
    tfps = [gen.GetFingerprint(m) for m in tmols]
    th = [m.GetNumHeavyAtoms() for m in tmols]
    hs = Chem.MolFromSmiles(seed).GetNumHeavyAtoms()
    # HORIZON IN OPERATOR COUNT, NOT |dHeavy|.
    # |dHeavy| is the NET heavy-atom change and undercounts the work badly: it
    # hides deletions, ring closures, and atoms that are REPLACED rather than
    # net-added. Measured against the published molecules, parp1 s0 needs
    # 17 atom_insert + 3 atom_delete + 3 cycle_close = 23 operations while
    # |dHeavy| reports 14, and 5ht1b needs 24 against a reported 15. Every audit
    # and ladder in this project has therefore been calibrated ~40% too shallow,
    # which is exactly why the forward beam plateaued at 0.55 short of the target.
    from rdkit.Chem import rdFMCS
    def _ops(seed_smi, tgt_smi):
        a = Chem.MolFromSmiles(seed_smi); b = Chem.MolFromSmiles(tgt_smi)
        if a is None or b is None:
            return abs(b.GetNumHeavyAtoms() - a.GetNumHeavyAtoms()) if b and a else 0
        try:
            r = rdFMCS.FindMCS([a, b], timeout=30,
                               ringMatchesRingOnly=False, completeRingsOnly=False,
                               atomCompare=rdFMCS.AtomCompare.CompareElements,
                               bondCompare=rdFMCS.BondCompare.CompareOrder)
            n = r.numAtoms
        except Exception:
            n = 0
        from rdkit.Chem import rdMolDescriptors as _rdMD
        dr = _rdMD.CalcNumRings(b) - _rdMD.CalcNumRings(a)
        return (a.GetNumHeavyAtoms() - n) + (b.GetNumHeavyAtoms() - n) + max(0, dr)
    need = max(_ops(seed, w["smiles"]) for w in job["winners"])
    depth_max = min(DEPTH_CAP, need + DEPTH_HEADROOM)
    print(f"  [{job['cell']}] seed heavy={hs}, OPERATIONS required={need} "
          f"(|dHeavy| would say {max(abs(h - hs) for h in th)}), depth {depth_max}, "
          f"fan-out {WIDTH // PER_CONTAINER} containers/depth", flush=True)

    def score(y):
        m = Chem.MolFromSmiles(y)
        if m is None:
            return -9.0, 0, 0.0
        f = gen.GetFingerprint(m)
        s = [DataStructs.TanimotoSimilarity(f, tf) for tf in tfps]
        i = int(np.argmax(s))
        gap = abs(m.GetNumHeavyAtoms() - th[i])
        # similarity alone is a poor guide when the target is 14 atoms away:
        # a molecule can look similar with no route to the right size.
        return s[i] - 0.02 * gap, i, s[i]

    beam = [seed]; seen = {seed}
    best_sim = score(seed)[2]
    trace = [{"depth": 0, "best_sim": round(best_sim, 4)}]
    hit = None
    for d in range(1, depth_max + 1):
        chunks = [beam[i:i + PER_CONTAINER] for i in range(0, len(beam), PER_CONTAINER)]
        cand = []
        n_err = 0
        for res in expand.map(chunks, order_outputs=False, return_exceptions=True,
                              wrap_returned_exceptions=False):
            if not isinstance(res, dict):
                # NEVER swallow this. The first version of this loop discarded
                # every exception, so all 16 expand calls failing on a missing
                # volume mount looked like "DONE, no hit, 8 seconds" -- a silent
                # false negative reported as a result.
                n_err += 1
                if n_err <= 2:
                    print(f"  [{job['cell']}] !! expand failed: "
                          f"{type(res).__name__}: {str(res)[:200]}", flush=True)
                continue
            for _, ys in res.items():
                for y in ys:
                    if y not in seen:
                        seen.add(y); cand.append(y)
        if not cand:
            print(f"  [{job['cell']}] depth {d}: NO CANDIDATES "
                  f"({n_err}/{len(chunks)} expand calls failed) -- stopping. "
                  f"This is an ERROR if n_err > 0, not an exhausted search.",
                  flush=True)
            break
        for y in cand:
            if y in targets:
                hit = {"depth": d, "smiles": y}; break
        if hit:
            break
        sc = sorted(((score(y), y) for y in cand), reverse=True)
        beam = [y for _, y in sc[:WIDTH]]
        if sc and sc[0][0][2] > best_sim:
            best_sim = sc[0][0][2]
        trace.append({"depth": d, "best_sim": round(best_sim, 4),
                      "best_smiles": sc[0][1] if sc else None,
                      "expanded": len(cand), "elapsed": round(time.time() - t0, 1)})
        print(f"  [{job['cell']}] depth {d}: expanded {len(cand):>5} "
              f"best_sim {best_sim:.4f}  {time.time()-t0:.0f}s", flush=True)
        try:
            p = Path("/artifacts/t4_reach_fanout"); p.mkdir(parents=True, exist_ok=True)
            (p / f"{job['cell']}.json").write_text(json.dumps(
                {"cell": job["cell"], "seed": seed, "min_edits_required": need,
                 "max_depth": depth_max, "depth_reached": d, "EXACT_HIT": None,
                 "best_similarity_to_any_winner": round(best_sim, 4),
                 "trace": trace, "complete": False}, indent=1))
            artifact_volume.commit()
        except Exception:
            pass
    out = {"cell": job["cell"], "seed": seed, "min_edits_required": need,
           "max_depth": depth_max, "depth_reached": len(trace) - 1,
           "EXACT_HIT": hit, "best_similarity_to_any_winner": round(best_sim, 4),
           "evidence_note": (f"A hit is definitive. No hit means not found at width "
                             f"{WIDTH}, depth {depth_max} (>= the {need} edits "
                             f"structurally required); it is not proof of "
                             f"unreachability."),
           "trace": trace, "complete": True, "seconds": round(time.time() - t0, 1)}
    p = Path("/artifacts/t4_reach_fanout"); p.mkdir(parents=True, exist_ok=True)
    (p / f"{job['cell']}.json").write_text(json.dumps(out, indent=1))
    artifact_volume.commit()
    print(f"  [{job['cell']}] DONE hit={bool(hit)} best_sim={best_sim:.4f} "
          f"{out['seconds']:.0f}s", flush=True)
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=2048, timeout=8 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(only: str = "") -> dict[str, Any]:
    W = json.loads((REMOTE_ROOT / "docs/ivg_winners.json").read_text())
    seeds = json.loads((REMOTE_ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    want = [s.strip() for s in only.split(",") if s.strip()]
    jobs = [{"cell": c, "seed_smiles": seeds[v["seed_idx"]]["smiles"],
             "winners": v["winners"]} for c, v in W.items()
            if not want or any(w in c for w in want)]
    print(f"  {len(jobs)} cells x {WIDTH // PER_CONTAINER} expand containers each\n", flush=True)
    out = []
    for r in drive_cell.map(jobs, order_outputs=False, return_exceptions=True,
                            wrap_returned_exceptions=False):
        if isinstance(r, dict):
            out.append(r)
        else:
            print(f"  !! {type(r).__name__}: {str(r)[:160]}", flush=True)
    return {"cells": out}


@app.function(image=image, cpu=(1.0, 1.0), memory=1024, timeout=600,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def salvage() -> dict[str, Any]:
    """Recover whatever the volume holds. Every depth is persisted, so a dead
    driver costs at most the current depth -- never the whole audit."""
    artifact_volume.reload()
    d = Path("/artifacts/t4_reach_fanout")
    cells = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))] if d.exists() else []
    print(f"  salvaged {len(cells)} cells "
          f"({sum(1 for c in cells if c.get('complete'))} complete)", flush=True)
    return {"cells": cells, "salvaged": True}


@app.local_entrypoint()
def main(out: str = "", only: str = "") -> None:
    # SAVE PATH, THREE DEEP. Results have been lost tonight to a driver dying
    # after the work was done, so: every depth writes to the volume inside
    # drive_cell; a driver failure falls back to salvage(); and the local write
    # happens whichever path produced the payload.
    try:
        o = drive.remote(only)
    except Exception as e:
        print(f"  driver failed ({type(e).__name__}: {str(e)[:120]}); salvaging")
        try:
            o = salvage.remote()
        except Exception as e2:
            print(f"  salvage also failed ({type(e2).__name__}); nothing to write")
            o = {"cells": [], "error": f"{e}"}
    p = Path(__file__).resolve().parents[1] / (out or "diagnostics/genmol_t4_reach_fanout.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    n = len(o.get("cells", []))
    print(f"\nwrote {p}  ({n} cells)")
    if n == 0:
        print("  WARNING: zero cells. Read the volume directly: "
              "modal volume ls compose-v4-artifacts t4_reach_fanout")
