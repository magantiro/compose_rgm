"""Exact target-directed route oracle. DEVELOPMENT CELLS ONLY.

Separates two questions we have been conflating:

    (A) Can the chemistry be REACHED inside COMPOSE's executable state space?
    (B) Can our generic controller DISCOVER it?

A full target-blind SMC controller answers (B) badly and (A) not at all. This
answers (A) constructively, then uses the resulting route to diagnose (B)
precisely.

METHOD. Work BACKWARD, winner -> seed. Deletion has far lower branching than
construction, and for these cells the seed is (nearly) a subgraph of the winner,
which gives a cheap exact progress criterion: keep the seed as a substructure
while shedding heavy atoms. Every backward step must be a legal executor
transition. Invert the script to obtain the forward executable path.

The landmark is a TEACHER here, never a target: the production controller stays
target-blind, and the real evidence is transfer to untouched cells.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import (
    APPLY_CAP, CANONICAL_SLOTS, TIME_POINT,
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume, _runtime, _dock_many,
)
from modal_apps.genmol_t4_opt_app import image as _opt_image

image = _opt_image.add_local_file(
    ROOT / "modal_apps/genmol_t4_opt_app.py",
    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)

app = modal.App("genmol-t4-route-oracle")


@app.function(image=image, cpu=(4.0, 4.0), memory=int(8 * 1024),
              timeout=60 * 60 * 4, max_containers=8, scaledown_window=900,
              enable_memory_snapshot=True,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def solve_route(job: dict[str, Any]) -> dict[str, Any]:
    import os, sys, time, collections
    import numpy as np
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors as rdMD
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)

    rt = _runtime(); model, system = rt["model"], rt["system"]
    seed_s, tgt_s = job["seed"], job["target"]
    m_seed = Chem.MolFromSmiles(seed_s)
    seed_canon = Chem.MolToSmiles(m_seed)
    h_seed = m_seed.GetNumHeavyAtoms()
    # THE INVARIANT IS THE SHARED CORE, NOT THE SEED. Measured: the parp1
    # winner does NOT contain the seed as a substructure -- the strict
    # element+bond-order MCS is 16 of the seed's 19 atoms, so the transformation
    # deletes 3 seed atoms as well as adding 17. Requiring the seed itself to
    # survive made every backward step illegal at step 0. Both endpoints DO
    # match the MCS core, so that is the thing a route must not destroy.
    q_seed = Chem.MolFromSmarts(job["core_smarts"]) if job.get("core_smarts") \
        else Chem.MolFromSmarts(Chem.MolToSmarts(m_seed))

    def successors(smi):
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        except Exception:
            return []
        pr = np.array([m.probability for m in law.marks], float)
        order = np.argsort(-pr)
        out = []
        for rank, j in enumerate(order):
            mk = law.marks[int(j)]
            try:
                y = canonical_state_key(system.apply(st, mk.executor_rule_name,
                                                     mk.action))
            except Exception:
                continue
            if y and y != smi:
                out.append((y, mk.executor_rule_name, rank, float(pr[j])))
        return out

    def progress(smi):
        """Backward progress: heavy atoms still to shed, and whether the seed
        scaffold is still present. Cheap and exact -- no MCS per candidate."""
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        keeps = bool(q_seed is not None and m.HasSubstructMatch(q_seed))
        return (m.GetNumHeavyAtoms() - h_seed, keeps, m)

    cur = Chem.MolToSmiles(Chem.MolFromSmiles(tgt_s))
    route = []          # backward steps: winner -> ... -> seed
    t0 = time.time()
    stalls = 0
    for step in range(int(job.get("max_steps", 60))):
        if cur == seed_canon:
            break
        cand = successors(cur)
        p_cur = progress(cur)
        best = None
        for y, rule, rank, p in cand:
            pg = progress(y)
            if pg is None:
                continue
            excess, keeps, _m = pg
            # HARD: never destroy the seed scaffold. Losing it is unrecoverable
            # backward -- once gone, every candidate scores equally on `keeps`
            # and the tiebreak degenerates into "delete everything", which is
            # exactly what happened: the search ran to excess=-18, eighteen
            # atoms PAST the seed, and never reached it.
            if not keeps:
                continue
            # minimise |excess|, not excess: the target is excess == 0 (the seed
            # itself), and minimising the signed value drives deletion straight
            # through the seed and out the other side.
            key = (abs(excess), rank)
            if best is None or key < best[0]:
                best = (key, y, rule, rank, p, excess, keeps)
        if best is None or (p_cur is not None and best[5] >= p_cur[0]):
            # OBSTRUCTION: no single legal edit reduces the excess while keeping
            # the core. Do a SMALL depth-2 lookahead here only -- not a global
            # beam. Measured: the greedy backward walk reached excess=5 from 14
            # and then oscillated between two atom_restate moves, which is the
            # same plateau the forward route audit found, just mirrored.
            lk = None
            for y1, r1, k1, p1 in cand[:int(job.get("lookahead_width", 12))]:
                g1 = progress(y1)
                if g1 is None or not g1[1]:
                    continue
                for y2, r2, k2, p2 in successors(y1):
                    g2 = progress(y2)
                    if g2 is None or not g2[1] or g2[0] < 0:
                        continue
                    if p_cur is not None and g2[0] < p_cur[0]:
                        if lk is None or g2[0] < lk[0]:
                            lk = (g2[0], y1, r1, k1, p1, y2, r2, k2, p2)
                        break
                if lk is not None:
                    break
            if lk is not None:
                _, y1, r1, k1, p1, y2, r2, k2, p2 = lk
                for (yy, rr, kk, pp) in ((y1, r1, k1, p1), (y2, r2, k2, p2)):
                    ex = progress(yy)[0]
                    route.append({"step": step, "rule": rr, "rank": kk,
                                  "prob": pp, "excess_heavy": ex,
                                  "keeps_seed_scaffold": True, "smiles": yy,
                                  "improved": True, "via": "depth2_lookahead"})
                    print(f"  [{job['cell']}] back {step:>2}: {rr:22s} rank "
                          f"{kk:>4} p={pp:.2e} excess={ex:>3} scaffold=Y  "
                          f"(depth-2 lookahead)", flush=True)
                cur = y2
                stalls = 0
                continue
            route.append({"step": step, "OBSTRUCTION":
                          f"no 1- or 2-step legal move reduces excess "
                          f"(at excess={p_cur[0] if p_cur else '?'})"})
            break
        key, y, rule, rank, p, excess, keeps = best
        if excess < 0:
            route.append({"step": step, "OBSTRUCTION":
                          f"would overshoot the seed (excess {excess})"})
            break
        improved = (p_cur is None) or (excess < p_cur[0]) or (keeps and not p_cur[1])
        if not improved:
            stalls += 1
        else:
            stalls = 0
        route.append({"step": step, "rule": rule, "rank": rank, "prob": p,
                      "excess_heavy": excess, "keeps_seed_scaffold": keeps,
                      "smiles": y, "improved": bool(improved)})
        print(f"  [{job['cell']}] back {step:>2}: {rule:22s} rank {rank:>4} "
              f"p={p:.2e} excess={excess:>3} scaffold={'Y' if keeps else 'N'}"
              f"{'' if improved else '   (no progress)'}", flush=True)
        cur = y
        if stalls >= int(job.get("stall_cap", 6)):
            route.append({"step": step + 1, "OBSTRUCTION":
                          f"stalled {stalls} steps at excess={excess}"})
            break
    reached = (cur == seed_canon)
    out = {"cell": job["cell"], "REACHED_SEED": reached,
           "n_backward_steps": len([r for r in route if r.get("rule")]),
           "final_excess": progress(cur)[0] if progress(cur) else None,
           "route_backward": route,
           # forward path = reversed backward states, seed -> ... -> winner
           "forward_states": ([seed_canon] if reached else [])
                             + [r["smiles"] for r in reversed(route)
                                if r.get("smiles")],
           "wall_s": time.time() - t0}
    try:
        d = ARTIFACT_ROOT / "t4_route_oracle"; d.mkdir(parents=True, exist_ok=True)
        (d / f"{job['cell']}.json").write_text(json.dumps(out))
        artifact_volume.commit()
    except Exception as e:
        print(f"  !! save failed: {type(e).__name__}: {e}", flush=True)
    return out


@app.function(image=image, cpu=(4.0, 4.0), memory=int(8 * 1024),
              timeout=60 * 60 * 2, volumes={str(ARTIFACT_ROOT): artifact_volume})
def dock_route(job: dict[str, Any]) -> dict[str, Any]:
    """Binding versus route depth. ~6-10 calls, not 50-200.

    If the curve is flat until the final edits and then drops to -13, we have
    measured the delayed-credit bridge IN THE PROTEIN OBJECTIVE ITSELF, which is
    worth far more than another fingerprint heuristic.
    """
    states = job["states"]; every = int(job.get("every", 4))
    picks = states[::every]
    if states and states[-1] not in picks:
        picks.append(states[-1])
    ds = _dock_many(picks, job["target"], f"ro_{job['cell']}",
                    workers=4, cpu_per_dock=1)
    return {"cell": job["cell"], "target": job["target"],
            "depths": list(range(0, len(states), every))[:len(picks)],
            "smiles": picks, "ds": ds}


@app.local_entrypoint()
def main(cells: str = "parp1", max_steps: int = 60, dock_every: int = 4):
    seeds = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    win = json.loads((ROOT / "diagnostics/ivg_winners.json").read_text())
    cores = json.loads((ROOT / "diagnostics/route_cores.json").read_text())
    want = [c.strip() for c in cells.split(",")]
    jobs = []
    for k, v in win.items():
        if v["target"] not in want or v["delta"] != 0.4:
            continue
        b = sorted(v.get("winners", []), key=lambda r: r["ds"])[:1]
        if not b:
            continue
        cell = f"{v['target']}_s{v['seed_idx']}"
        cc = cores.get(cell)
        if not cc:
            continue
        print(f"  {cell}: shared core {cc['core_atoms']} atoms "
              f"(seed {cc['h_seed']}, winner {cc['h_target']})", flush=True)
        jobs.append({"cell": cell, "seed": cc["seed"], "target": cc["target"],
                     "max_steps": max_steps, "core_smarts": cc["core_smarts"]})
    print(f"  route oracle on {len(jobs)} development landmark(s)", flush=True)
    for r in solve_route.map(jobs, order_outputs=True, return_exceptions=True,
                             wrap_returned_exceptions=False):
        if not isinstance(r, dict):
            print(f"  !! {type(r).__name__}: {str(r)[:200]}"); continue
        print(f"\n  {r['cell']}: REACHED_SEED={r['REACHED_SEED']} "
              f"backward_steps={r['n_backward_steps']} "
              f"final_excess={r['final_excess']}")
        obs = [x for x in r["route_backward"] if x.get("OBSTRUCTION")]
        if obs:
            print(f"     OBSTRUCTION: {obs[0]['OBSTRUCTION']}")
        fs = r.get("forward_states") or []
        if len(fs) >= 3:
            tgt = [j for j in jobs if j["cell"] == r["cell"]][0]
            d = dock_route.remote({"cell": r["cell"], "states": fs,
                                   "target": r["cell"].split("_")[0],
                                   "every": dock_every})
            print(f"     binding vs route depth:")
            for dep, smi, sc in zip(d["depths"], d["smiles"], d["ds"]):
                print(f"       depth {dep:>3}: {sc}")
