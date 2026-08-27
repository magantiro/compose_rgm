"""PMO through the FROZEN COMPOSE controller (5e399bc, refine=2).

Minimal ORACLE ADAPTER. The controller is not touched and not redeployed: this
app calls the already-deployed `macro-basin::search_episode` by name, so the
episodes run the exact frozen binary the official T4 panel is running. Only the
OBJECTIVE changes -- docking becomes a PMO oracle.

Everything that defines the comparison is inherited verbatim from
pmo_pilot_app so the macro arm and the PRE_MACRO_FLOOR arm differ in ONE thing:

  * same oracle construction (PyTDC 0.3.6, frozen forest for jnk3)
  * same OracleMeter, PER_MOLECULE counting, same budget contract
  * same PMO_INIT_BANK.json initialisation
  * same distinct-molecule population and top-10 definition
  * same AUC-top10 integration

CHECKPOINTS are DENSE here (every 100 calls). The pilot's grid stops at 2000,
so at a 10,000-call budget np.trapz would draw one straight line across
2000->10000 and the two arms would not be comparable. This is an accounting fix
in the harness, not a change to either search.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import (
    CANONICAL_SLOTS, ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume, _runtime,
)
from modal_apps.pmo_pilot_app import image as _pilot_image, _shim

# The pilot's image bundles genmol_t4_opt_app.py but not pmo_pilot_app.py
# itself, and the container re-imports THIS module -- which imports the pilot
# for its image and _shim. Ship the pilot source too.
image = _pilot_image.add_local_file(
    ROOT / "modal_apps/pmo_pilot_app.py",
    str(REMOTE_ROOT / "modal_apps/pmo_pilot_app.py"), copy=True)
app = modal.App("pmo-macro")

# every 100 calls, the standard PMO AUC-top10 grid
CHECKPOINTS = tuple(range(100, 10001, 100))
N_LINEAGE = 12


@app.function(image=image, cpu=(2.0, 2.0), memory=int(8 * 1024),
              timeout=10 * 60 * 60, max_containers=4,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_task_macro(task: dict[str, Any]) -> dict[str, Any]:
    import sys, os
    import numpy as np
    _shim()
    os.chdir("/tmp")
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.benchmark.molleo_task3 import (OracleMeter, CountingRule,
                                                   BudgetExceeded)
    from compose_v4.control.constrained_search import (ACTION_SPACE, CEMController,
                                                       matched_prior,
                                                       realization_groups)

    name = task["task"]; budget = int(task["budget"])
    n_part = int(task.get("n_particles", 16))
    ep_len = int(task.get("episode_len", 5))
    rng = np.random.default_rng(task["seed_rng"])
    t0 = time.time()

    # ---- oracle: identical construction to the floor arm --------------------
    if name == "jnk3":
        from compose_v4.benchmark.oracles.forest import FrozenForest
        ff = FrozenForest("/frozen/jnk3_forest.npz")
        def raw(s):
            m = Chem.MolFromSmiles(s)
            if m is None:
                return (0.0,)
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
            a = np.zeros((1,), dtype=np.int8); DataStructs.ConvertToNumpyArray(fp, a)
            return (float(ff.probabilities(a.astype(np.float64).reshape(1, -1))[0]),)
        src = "frozen npz"
    else:
        from tdc import Oracle
        o = Oracle(name=name)
        raw = lambda s: (float(o(s)),)
        src = "PyTDC 0.3.6"

    canon = lambda s: (lambda m: Chem.MolToSmiles(m) if m else None)(Chem.MolFromSmiles(s))
    meter = OracleMeter(evaluate=raw, budget=budget,
                        counting_rule=CountingRule.PER_MOLECULE,
                        canonicalize=canon, n_objectives=1)

    # ---- identical initialisation to the floor ------------------------------
    init = json.loads((REMOTE_ROOT / "docs/PMO_INIT_BANK.json").read_text())["smiles"]
    pop: dict[str, dict] = {}
    depth, curve = {}, []
    try:
        for s in init:
            c = canon(s)
            if c and c not in pop:
                pop[c] = {"smi": c, "u": meter(c)[0], "d": 0}; depth[c] = 0
    except BudgetExceeded:
        pass

    def snapshot():
        vals = sorted((p["u"] for p in pop.values()), reverse=True)
        top = vals[:10]
        return {"calls": meter.spent, "best": vals[0] if vals else 0.0,
                "top10_mean": float(np.mean(top)) if top else 0.0,
                "n_distinct_pop": len(pop), "unique": meter.n_unique,
                "max_depth": max(depth.values()) if depth else 0}
    curve.append(snapshot())
    nxt = [c for c in CHECKPOINTS]

    # ---- THE FROZEN CONTROLLER, called by name, never redeployed ------------
    episode = modal.Function.from_name("macro-basin", "search_episode")
    ctrl = CEMController(len(ACTION_SPACE), rng,
                         init_probs=matched_prior(ACTION_SPACE, 0.24))
    _groups = realization_groups(ACTION_SPACE)

    rd, stop, n_ring_fired = 0, False, 0
    while meter.remaining > 0 and not stop:
        rd += 1
        ranked = sorted(pop.values(), key=lambda p: -p["u"])
        elite = ranked[:N_LINEAGE // 2]
        rest = ranked[N_LINEAGE // 2:]
        div = list(rng.choice(rest, size=min(N_LINEAGE - len(elite), len(rest)),
                              replace=False)) if rest else []
        parents = elite + list(div)
        if not parents:
            break

        jobs, prog_actions = [], []
        for pi in range(n_part):
            ai = ctrl.sample(ep_len)
            prog = [ACTION_SPACE[int(i)] for i in ai]
            prog_actions.append(list(ai))
            par = parents[pi % len(parents)]["smi"]
            # delta=0.0: PMO imposes no similarity-to-seed constraint. The
            # similarity-style tasks express that through the ORACLE, so a
            # pathwise delta would be a second, invented constraint.
            jobs.append(dict(smiles=par, seed=par, delta=0.0, program=prog,
                             start_depth=int(depth.get(par, 0)),
                             semantics="legacy", refine=2,
                             ring_policy=None,
                             seed_rng=int(rng.integers(0, 2**31))))
        try:
            res = list(episode.map(jobs, return_exceptions=True))
        except Exception as e:
            print(f"  episode map failed: {type(e).__name__} {e}", flush=True)
            break

        cand: dict[str, int] = {}
        traj_best: dict[int, list] = {}   # keyed by particle index
        for _pi2, (ai, r) in enumerate(zip(prog_actions, res)):
            if not isinstance(r, dict):
                rewards.append(0.0); continue
            for t in (r.get("trace") or []):
                if isinstance(t, str) and t.startswith("refine["):
                    n_ring_fired += 1
            got = []
            for c in (r.get("candidates") or []):
                y = c.get("smiles") if isinstance(c, dict) else None
                if y and y not in pop and y not in cand:
                    cand[y] = depth.get(r.get("smiles", ""), 0) + 1
                    got.append(y)
            y = r.get("smiles")
            if y and y not in pop and y not in cand:
                cand[y] = depth.get(jobs[0]["seed"], 0) + 1
                got.append(y)
            traj_best[_pi2] = got

        if not cand:
            if rd > 200:
                break
            continue
        scored = []
        for y, d in cand.items():
            if meter.remaining <= 0:
                stop = True; break
            try:
                u = meter(y)[0]
            except BudgetExceeded:
                stop = True; break
            pop[y] = {"smi": y, "u": u, "d": d}; depth[y] = d
            scored.append(u)
            while nxt and meter.spent >= nxt[0]:
                curve.append(snapshot()); nxt.pop(0)

        # CEM UPDATE -- mirrors drive_episodes exactly:
        #   flat action list, the trajectory's score repeated per action,
        #   update_by_rank with realization groups and site_shrink=0.25.
        # SIGN: update_by_rank takes elite = argsort(scores)[:k], i.e. LOWER IS
        # BETTER, because it was built for docking energies. PMO oracles are
        # higher-is-better, so the score is NEGATED here. Passing the raw
        # oracle value would train the controller to MINIMISE the objective.
        acts, scs = [], []
        for _pi3, ai in enumerate(prog_actions):
            got = traj_best.get(_pi3) or []
            vals = [pop[y]["u"] for y in got if y in pop]
            if not vals:
                continue
            tscore = -float(max(vals))          # negate: lower is better
            for i in ai:
                acts.append(int(i)); scs.append(tscore)
        if acts:
            ctrl.update_by_rank(acts, scs, groups=_groups, site_shrink=0.25)
        if len(pop) > 400:
            keep = sorted(pop.values(), key=lambda p: -p["u"])[:400]
            pop = {p["smi"]: p for p in keep}

    curve.append(snapshot())
    top10 = sorted((p["u"] for p in pop.values()), reverse=True)[:10]
    out = {"task": name, "arm": "MACRO_FROZEN_5e399bc_refine2",
           "oracle_source": src, "budget": budget,
           "calls_spent": meter.spent, "unique": meter.n_unique, "rounds": rd,
           "n_particles": n_part, "episode_len": ep_len,
           "refine_fired": n_ring_fired,
           "best": float(top10[0]) if top10 else 0.0,
           "top10_mean": float(np.mean(top10)) if top10 else 0.0,
           "auc_top10": float(np.trapz([c["top10_mean"] for c in curve],
                                       [c["calls"] for c in curve]) / max(meter.spent, 1)),
           "curve": curve,
           "max_depth": max(depth.values()) if depth else 0,
           "seconds": round(time.time() - t0, 1)}
    try:
        d = Path("/artifacts/pmo_macro"); d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}_b{budget}.json").write_text(json.dumps(out, indent=1))
        artifact_volume.commit()
    except Exception as e:
        print(f"  !! persist failed: {e}", flush=True)
    return out


@app.local_entrypoint()
def main(task: str = "thiothixene_rediscovery", budget: int = 100,
         n_particles: int = 8, episode_len: int = 5) -> None:
    o = run_task_macro.remote(dict(task=task, budget=budget, seed_rng=20260821,
                                   n_particles=n_particles, episode_len=episode_len))
    print(f"\n  {o['task']}  arm={o['arm']}")
    print(f"  calls {o['calls_spent']}/{o['budget']}  unique {o['unique']}  rounds {o['rounds']}")
    print(f"  best {o['best']:.4f}  top10 {o['top10_mean']:.4f}  AUC {o['auc_top10']:.4f}")
    print(f"  REFINE_RING fired {o['refine_fired']}  max_depth {o['max_depth']}  {o['seconds']}s")
    p = ROOT / f"diagnostics/pmo_macro_{task}_b{budget}.json"
    p.write_text(json.dumps(o, indent=1))
    print(f"  wrote {p}")
