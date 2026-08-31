"""Executor-only exact route search. DEVELOPMENT ORACLE.

Reachability is a property of the EXECUTOR, not of R_theta's learned
probabilities. The previous oracle paid one R_theta forward (6.7s) plus hundreds
of mark applications per step, which mixed two experiments and made route search
needlessly slow. `enumerate_process_v2_atom_deletes(state)` enumerates the legal
deletion fiber with no model at all, so backward construction costs only the
applies.

OBJECTIVE: exact graph difference in operator units, not fingerprints.
Shared-bit / Tanimoto guidance is provably misaligned here -- it was what made
the greedy walks cycle -- and the target is known, so there is no reason to
steer a development diagnostic by a proxy.

    D(x) = (heavy(x) - core) + (heavy(seed) - core)   [atoms to shed + to add]

Greedy where D decreases; bounded local search ONLY at a plateau.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import (
    CANONICAL_SLOTS, ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume,
    _runtime, _dock_many,
)
from modal_apps.genmol_t4_opt_app import image as _opt_image

image = _opt_image.add_local_file(
    ROOT / "modal_apps/genmol_t4_opt_app.py",
    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)

app = modal.App("genmol-t4-exact-route")


@app.function(image=image, cpu=(4.0, 4.0), memory=int(8 * 1024),
              timeout=60 * 60 * 3, max_containers=8, scaledown_window=900,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def exact_route(job: dict[str, Any]) -> dict[str, Any]:
    import os, sys, time, heapq
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key)
    from compose_v4.rewrite.process_v2_atom_delete import (
        enumerate_process_v2_atom_deletes)
    from compose_v4.rewrite.fiber import enumerate_action_fiber
    rt = _runtime(); system = rt["system"]

    seed_s, tgt_s = job["seed"], job["target"]
    core_q = Chem.MolFromSmarts(job["core_smarts"])
    h_seed = Chem.MolFromSmiles(seed_s).GetNumHeavyAtoms()
    seed_canon = Chem.MolToSmiles(Chem.MolFromSmiles(seed_s))

    def st_of(smi):
        return pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)

    def deletions(smi):
        """Legal moves. NO MODEL CALL.

        Deletion alone stalls: it shed 9 of parp1's 14 excess atoms and then hit
        a wall at excess=5, because the remaining atoms sit in ring systems that
        cannot be removed atom-by-atom without destroying the shared core --
        they need the ring opened first. So the backward move set is the FULL
        legal action fiber, not just atom_delete.
        """
        try:
            st = st_of(smi)
        except Exception:
            return []
        out = []
        if job.get("deletions_only"):
            for act in enumerate_process_v2_atom_deletes(st):
                try:
                    y = canonical_state_key(system.apply(st, "atom_delete", act))
                except Exception:
                    continue
                if y and y != smi:
                    out.append(y)
            return out
        try:
            for mt in enumerate_action_fiber(st, system=system):
                k = mt.successor_key
                if k and k != smi:
                    out.append(k)
        except Exception:
            return []
        return out

    def D(smi):
        """Exact operator-unit deficit to the seed, and core preservation."""
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        if core_q is not None and not m.HasSubstructMatch(core_q):
            return None                      # destroyed the shared core
        return m.GetNumHeavyAtoms() - h_seed

    cur = Chem.MolToSmiles(Chem.MolFromSmiles(tgt_s))
    route = [cur]
    t0 = time.time()
    obstruction = None
    for step in range(int(job.get("max_steps", 80))):
        if cur == seed_canon:
            break
        d_cur = D(cur)
        succ = [(y, D(y)) for y in deletions(cur)]
        succ = [(y, d) for y, d in succ if d is not None and d >= 0]
        best = min(succ, key=lambda p: p[1], default=None)
        if best is not None and d_cur is not None and best[1] < d_cur:
            cur = best[0]; route.append(cur)
            print(f"  [{job['cell']}] del {step:>2}: excess {d_cur}->{best[1]}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
            continue
        # PLATEAU: bounded local search, deletions only, depth up to L
        L = int(job.get("plateau_depth", 5))
        W = int(job.get("plateau_width", 8))
        seen = {cur}
        frontier = [(0, cur)]
        found = None
        for depth in range(1, L + 1):
            nxt = []
            for _, s in frontier[:W]:
                for y in deletions(s):
                    if y in seen:
                        continue
                    seen.add(y)
                    dy = D(y)
                    if dy is None or dy < 0:
                        continue
                    nxt.append((dy, y))
                    if d_cur is not None and dy < d_cur:
                        found = y; break
                if found:
                    break
            if found:
                break
            frontier = sorted(nxt)[:W]
            if not frontier:
                break
        if found:
            cur = found; route.append(cur)
            print(f"  [{job['cell']}] del {step:>2}: excess {d_cur}->{D(cur)} "
                  f"via depth<={L} local search  ({time.time()-t0:.0f}s)", flush=True)
            continue
        obstruction = (f"no deletion path of depth<={L} reduces excess "
                       f"below {d_cur} while preserving the core")
        print(f"  [{job['cell']}] OBSTRUCTION: {obstruction}", flush=True)
        break

    out = {"cell": job["cell"], "reached_seed": cur == seed_canon,
           "final_excess": D(cur), "n_states": len(route),
           "obstruction": obstruction,
           "forward_states": list(reversed(route)),
           "wall_s": time.time() - t0}
    try:
        d = ARTIFACT_ROOT / "t4_exact_route"; d.mkdir(parents=True, exist_ok=True)
        (d / f"{job['cell']}.json").write_text(json.dumps(out))
        artifact_volume.commit()
    except Exception as e:
        print(f"  !! save failed: {type(e).__name__}: {e}", flush=True)
    return out


@app.local_entrypoint()
def main(cell: str = "parp1_s0", max_steps: int = 80, plateau_depth: int = 5):
    cores = json.loads((ROOT / "diagnostics/route_cores.json").read_text())
    cc = cores[cell]
    r = exact_route.remote({"cell": cell, "seed": cc["seed"],
                            "target": cc["target"],
                            "core_smarts": cc["core_smarts"],
                            "max_steps": max_steps,
                            "plateau_depth": plateau_depth})
    print(f"\n  {cell}: reached_seed={r['reached_seed']} "
          f"final_excess={r['final_excess']} states={r['n_states']} "
          f"({r['wall_s']:.0f}s)")
    if r.get("obstruction"):
        print(f"  OBSTRUCTION: {r['obstruction']}")
