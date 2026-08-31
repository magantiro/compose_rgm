"""Bidirectional reachability: is the growth basin UNREACHABLE, or just unfound?

A forward beam of width 16 exploring ~400^14 paths and failing proves the BEAM is
narrow, not that the target is unreachable. Those two conclusions differ
completely: one is a control problem we can fix, the other is a hard limit on
COMPOSE's operator set and a far more important result.

The COMPOSE operator set is closed under inversion --
  atom_delete <-> atom_insert, atom_restate <-> atom_restate,
  bond_reorder <-> bond_reorder, cycle_insert <-> ring_system_delete
-- so the reachability graph is UNDIRECTED and a backward search from the target
is exactly a forward search from it. Meeting in the middle at depth k/2 replaces
b^k with 2*b^(k/2): for parp1 that is 400^14 -> 2*400^7, about the square root.

We retain every expanded state on each side, not just the beam, and test set
intersection. A shared canonical key IS a path: seed -> ... -> x -> ... -> target.

Uses the target ONLY as a diagnostic endpoint. Nothing here feeds the production
optimiser, which never sees these molecules.
"""

from __future__ import annotations

import json
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
    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)

app = modal.App("genmol-t4-bidir")

BEAM = 64          # beam per side per depth (4x the forward audit's 16)
FANOUT = 16        # containers per expansion
MAX_HALF = 9       # half-depth; 9+9 = 18 >= the 14-15 edits required


@app.function(image=image, cpu=(2.0, 2.0), memory=int(6 * 1024),
              timeout=60 * 60 * 3, max_containers=80, scaledown_window=900,
              enable_memory_snapshot=True,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def expand(smis: list[str]) -> dict[str, list[str]]:
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
        for i in np.argsort(-pr):          # FULL law: reachability, not ranking
            mk = law.marks[int(i)]
            try:
                y = canonical_state_key(system.apply(st, mk.executor_rule_name,
                                                     mk.action))
            except Exception:
                continue
            if y and y != smi:
                ys.append(y)
        out[smi] = ys
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=int(8 * 1024),
              timeout=60 * 60 * 8, max_containers=8,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive_cell(job: dict[str, Any]) -> dict[str, Any]:
    import time
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    seed, target, cell = job["seed"], job["target"], job["cell"]

    def fp(s):
        m = Chem.MolFromSmiles(s)
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None

    f_seed, f_tgt = fp(seed), fp(target)

    def sim(a, b):
        return float(DataStructs.TanimotoSimilarity(a, b)) if a and b else 0.0

    # visited sets retain EVERY expanded state, not just the beam -- the beam
    # only decides where to push next; intersection is tested over everything.
    seen_f = {seed: 0}
    seen_b = {target: 0}
    beam_f, beam_b = [seed], [target]
    t0 = time.time()
    rec = []
    for half in range(1, int(job.get("max_half", MAX_HALF)) + 1):
        for side in ("f", "b"):
            beam = beam_f if side == "f" else beam_b
            seen = seen_f if side == "f" else seen_b
            other = seen_b if side == "f" else seen_f
            goal_fp = f_tgt if side == "f" else f_seed
            chunks = [beam[i::FANOUT] for i in range(FANOUT)]
            chunks = [c for c in chunks if c]
            kids: list[str] = []
            for res in expand.map(chunks, order_outputs=False,
                                  return_exceptions=True,
                                  wrap_returned_exceptions=False):
                if not isinstance(res, dict):
                    continue
                for _, ys in res.items():
                    kids.extend(ys)
            fresh = []
            for y in kids:
                if y not in seen:
                    seen[y] = half
                    fresh.append(y)
            meet = [y for y in fresh if y in other]
            if meet:
                y = meet[0]
                out = {"cell": cell, "MET": True, "meet_smiles": y,
                       "path_len": seen_f.get(y, -1) + seen_b.get(y, -1),
                       "half": half, "side": side,
                       "n_forward": len(seen_f), "n_backward": len(seen_b),
                       "wall_s": time.time() - t0, "trace": rec}
                print(f"  [{cell}] *** FRONTIERS MET at half-depth {half} "
                      f"({side}); path length "
                      f"{seen_f.get(y,-1)}+{seen_b.get(y,-1)} ***", flush=True)
                _save(out, cell)
                return out
            scored = sorted(((sim(fp(y), goal_fp), y) for y in fresh),
                            key=lambda p: -p[0])[:BEAM]
            if side == "f":
                beam_f = [y for _, y in scored]
            else:
                beam_b = [y for _, y in scored]
            best = scored[0][0] if scored else 0.0
            rec.append({"half": half, "side": side, "fresh": len(fresh),
                        "best_sim_to_goal": round(best, 4),
                        "seen_f": len(seen_f), "seen_b": len(seen_b)})
            print(f"  [{cell}] half {half} {side}: fresh {len(fresh):>7d}  "
                  f"best->goal {best:.4f}  |F|={len(seen_f):>7d} "
                  f"|B|={len(seen_b):>7d}  {time.time()-t0:.0f}s", flush=True)
            _save({"cell": cell, "MET": False, "trace": rec,
                   "n_forward": len(seen_f), "n_backward": len(seen_b)}, cell)
    return {"cell": cell, "MET": False, "trace": rec,
            "n_forward": len(seen_f), "n_backward": len(seen_b),
            "wall_s": time.time() - t0}


def _save(obj, cell):
    try:
        d = ARTIFACT_ROOT / "t4_bidir"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{cell}.json").write_text(json.dumps(obj))
        artifact_volume.commit()
    except Exception as e:
        print(f"  !! save failed: {type(e).__name__}: {e}", flush=True)


@app.local_entrypoint()
def main(cells: str = "parp1,5ht1b", max_half: int = MAX_HALF, out: str = ""):
    seeds = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    win = json.loads((ROOT / "diagnostics/ivg_winners.json").read_text())
    want = [c.strip() for c in cells.split(",")]
    jobs = []
    for k, v in win.items():
        if v["target"] not in want:
            continue
        best = sorted(v.get("winners", []), key=lambda r: r["ds"])[:1]
        if not best:
            continue
        jobs.append({"cell": f"{v['target']}_s{v['seed_idx']}_d{v['delta']}",
                     "seed": seeds[v["seed_idx"]]["smiles"],
                     "target": best[0]["smiles"], "max_half": max_half})
    print(f"  {len(jobs)} cells, beam {BEAM}/side, half-depth {max_half} "
          f"({max_half}+{max_half}={2*max_half} edits), ZERO oracle calls",
          flush=True)
    got = []
    for r in drive_cell.map(jobs, order_outputs=False, return_exceptions=True,
                            wrap_returned_exceptions=False):
        if isinstance(r, dict):
            got.append(r)
            print(f"  {r['cell']}: MET={r.get('MET')} "
                  f"|F|={r.get('n_forward')} |B|={r.get('n_backward')}", flush=True)
        else:
            print(f"  !! {type(r).__name__}: {str(r)[:200]}", flush=True)
    if out:
        Path(ROOT / out).write_text(json.dumps(got, indent=2))
        print(f"wrote {ROOT / out}")
