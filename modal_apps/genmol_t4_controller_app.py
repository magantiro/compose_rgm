"""The strong COMPOSE controller for T4. One architecture, switchable control.

Level 1 stays frozen as a baseline. This is a separate engine built around the
same frozen R_theta, addressing the three defects the diagnostics actually
found rather than the one we first assumed:

  parp1  129 feasible molecules enumerated, 4 docked. One scalar weight was
         deciding BOTH molecular navigation and oracle allocation.
  fa7    route steps rank at median 7.2%, never outside APPLY_CAP, yet the
         lineage carrying them is discarded. Collapse, not misranking.
  braf   no route found by the beam procedure at all.

So the backbone is the fix, and adaptive h-control sits on top of it:

    frozen R_theta
      + persistent multi-lineage graph search with a transposition table
      + SEPARATE search and oracle archives
      + online docking surrogate driving acquisition
      + adaptive finite-horizon h-control                   [switchable]

CONTROL is a one-flag ablation inside this same engine, so the comparison is
free of implementation differences:

    control=immediate   score a successor by its own predicted desirability
    control=future_h    score it by E_{R_theta}[g_t(X_b) | X_0 = y], the
                        desirability of the futures it leaves reachable

The h-transform is APPROXIMATE here and is not claimed otherwise: g_t is built
from an online surrogate and h is a bounded Monte Carlo estimate, so Theorem 3
does not apply. It is an adaptive Doob-style controller, not the exact one.

Nothing is tuned per receptor, seed or delta. R_theta is never retrained.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.genmol_t4_opt_app import (
    APPLY_CAP, CANONICAL_SLOTS, QED_MIN, SA_MAX, TIME_POINT,
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume, _dock, _runtime,
)
from modal_apps.genmol_t4_opt_app import image as _opt_image

image = _opt_image.add_local_file(
    ROOT / "modal_apps/genmol_t4_opt_app.py",
    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)

app = modal.App("genmol-t4-controller")

# ---- frozen policy constants. Global, never per-cell. ----
N_LINEAGE      = 8       # active parents kept, vs 3 collapsing ones in Level 1
ROLLOUTS       = 6       # R_theta continuations per h estimate
H_BUDGET       = 40      # max successors expanded inside one h estimate
TAU_U          = 1.0     # temperature on predicted docking desirability
TAU_V          = 0.10    # graded feasibility temperature, search only
ENSEMBLE       = 8       # bootstrap members in the ridge surrogate
RIDGE          = 1.0
WARMUP         = 24      # diverse feasible dockings before the surrogate leads


@app.function(image=image, cpu=(1.0, 1.0), memory=int(6 * 1024),
              timeout=10 * 60 * 60, max_containers=24,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_cell(task: dict[str, Any]) -> dict[str, Any]:
    import sys, os
    import numpy as np
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, AllChem, RDConfig
    RDLogger.DisableLog("rdApp.*")
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)

    rt = _runtime(); model, system = rt["model"], rt["system"]
    seed, delta, target = task["smiles"], task["delta"], task["target"]
    control, budget = task["control"], task["budget"]
    do_dock = task.get("dock", True)
    rng = np.random.default_rng(task["seed_rng"])
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = gen.GetFingerprint(Chem.MolFromSmiles(seed))
    t0 = time.time()

    # ---------------- transposition table: one entry per canonical molecule ----
    TT: dict[str, dict] = {}

    def props(smi):
        e = TT.get(smi)
        if e is not None and "v" in e:
            return e
        m = Chem.MolFromSmiles(smi)
        if m is None:
            TT[smi] = {"bad": True}; return TT[smi]
        q = float(QED.qed(m)); s = float(sascorer.calculateScore(m))
        sim = float(DataStructs.TanimotoSimilarity(seed_fp, gen.GetFingerprint(m)))
        # the full constraint-shortfall VECTOR, not one scalar
        c = (max(0.0, QED_MIN - q) / QED_MIN,
             max(0.0, s - SA_MAX) / SA_MAX,
             max(0.0, delta - sim) / delta)
        e = {"qed": q, "sa": s, "sim": sim, "c": c, "v": max(c),
             "fp": np.array(gen.GetFingerprint(m), dtype=np.float32)}
        TT[smi] = e
        return e

    def fiber(smi):
        e = TT.setdefault(smi, {})
        if "fiber" in e:
            return e["fiber"]
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        except Exception:
            e["fiber"] = {}; return e["fiber"]
        if not law.marks:
            e["fiber"] = {}; return e["fiber"]
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
        e["fiber"] = out
        return out

    # ---------------- online docking surrogate: bootstrap ridge, numpy only ---
    class Surrogate:
        """Ranking-oriented. Thompson sampling = draw one bootstrap member."""
        def __init__(self):
            self.W = None
        def fit(self, X, y):
            n, d = X.shape
            if n < 8:
                self.W = None; return
            Ws = []
            for _ in range(ENSEMBLE):
                idx = rng.integers(0, n, n)
                Xb, yb = X[idx], y[idx]
                A = Xb.T @ Xb + RIDGE * np.eye(d, dtype=np.float32)
                Ws.append(np.linalg.solve(A, Xb.T @ yb))
            self.W = np.stack(Ws)
        def sample(self, X):
            if self.W is None:
                return np.zeros(len(X))
            return X @ self.W[rng.integers(0, len(self.W))]
        def mean(self, X):
            if self.W is None:
                return np.zeros(len(X))
            return X @ self.W.mean(0)

    sur = Surrogate()

    # ---------------- terminal desirability g_t ----------------
    def g_terminal(smi, tilde):
        """Hard C times predicted docking desirability, once we can estimate it.

        Before there is feasible terminal mass, a pure indicator makes every
        rollout zero and the planner is blind, so search (and ONLY search) uses
        a graded feasibility surrogate. Reported candidates are always subject
        to the exact endpoint constraints.
        """
        e = props(smi)
        if e.get("bad"):
            return 0.0
        if e["v"] > 0.0:
            return 0.0 if tilde is not None else float(np.exp(-e["v"] / TAU_V))
        if tilde is None:
            return 1.0
        return float(np.exp(tilde(smi) / TAU_U))

    def h_hat(y, b, tilde):
        """E_{R_theta}[g_t(X_b) | X_0 = y] by bounded cached continuations."""
        if b <= 0:
            return g_terminal(y, tilde)
        key = ("h", y, b, tilde is not None)
        if key in TT:
            return TT[key]
        spent, acc = 0, []
        for _ in range(ROLLOUTS):
            cur = y
            for _d in range(b):
                f = fiber(cur)
                if not f or spent >= H_BUDGET:
                    break
                ks = list(f); w = np.array([f[k] for k in ks]); w /= w.sum()
                cur = ks[int(rng.choice(len(ks), p=w))]
                spent += 1
            acc.append(g_terminal(cur, tilde))
        val = float(np.mean(acc)) if acc else 0.0
        TT[key] = val
        return val

    # ---------------- archives, deliberately separate ----------------
    A_search: list[dict] = [{"smi": seed, **{k: v for k, v in props(seed).items() if k != "fp"}}]
    A_oracle: list[dict] = []
    docked: dict[str, float] = {}
    Xs, ys = [], []
    log = []

    def refresh_surrogate():
        if len(ys) >= 8:
            sur.fit(np.stack(Xs), np.array(ys, dtype=np.float32))

    def tilde_fn():
        if sur.W is None or len(ys) < WARMUP:
            return None
        Wm = sur.W[rng.integers(0, len(sur.W))]
        return lambda s: float(props(s)["fp"] @ Wm)

    def keep_lineages(pool):
        """Multi-lineage frontier: Pareto on the shortfall vector, then diverse.

        Level 1 kept the top-3 by a single scalar and collapsed. Here a molecule
        that is worse overall but better on ANY component survives, which is
        what preserves the lineage the route replay showed being discarded.
        """
        pts = [(p, props(p["smi"])["c"]) for p in pool]
        front = []
        for p, c in pts:
            if not any(all(o <= x for o, x in zip(c2, c)) and o_better(c2, c)
                       for _, c2 in pts):
                front.append(p)
        if len(front) < N_LINEAGE:
            rest = sorted((p for p in pool if p not in front),
                          key=lambda p: props(p["smi"])["v"])
            front += rest[:N_LINEAGE - len(front)]
        if len(front) <= N_LINEAGE:
            return front
        chosen = [min(front, key=lambda p: props(p["smi"])["v"])]
        while len(chosen) < N_LINEAGE:
            best, bd = None, -1.0
            for p in front:
                if p in chosen:
                    continue
                d = min(1.0 - DataStructs.TanimotoSimilarity(
                    gen.GetFingerprint(Chem.MolFromSmiles(p["smi"])),
                    gen.GetFingerprint(Chem.MolFromSmiles(q["smi"]))) for q in chosen)
                if d > bd:
                    best, bd = p, d
            chosen.append(best)
        return chosen

    def o_better(a, b):
        return any(x < y for x, y in zip(a, b))

    # ---------------- main loop ----------------
    n_calls, rd = 0, 0
    while n_calls < budget or (not do_dock and rd < task.get("rounds", 10)):
        rd += 1
        parents = keep_lineages(A_search)
        cand: dict[str, float] = {}
        for p in parents:
            for y, r in fiber(p["smi"]).items():
                if y in docked or y in cand:
                    continue
                if not props(y).get("bad"):
                    cand[y] = r
        if not cand:
            break
        keys = list(cand)
        tl = tilde_fn()

        # ---- CONTROL: the one-flag ablation ----
        if control == "future_h":
            b = task.get("horizon", 4)
            score = np.array([cand[y] * h_hat(y, b, tl) for y in keys])
        else:                                    # immediate
            score = np.array([cand[y] * g_terminal(y, tl) for y in keys])
        score = np.clip(score, 1e-30, None)

        # ---- search archive: keep promising states, feasible or not ----
        top = np.argsort(-score)[:N_LINEAGE * 4]
        for i in top:
            A_search.append({"smi": keys[int(i)],
                             **{k: v for k, v in props(keys[int(i)]).items() if k != "fp"}})

        # ---- oracle archive: FEASIBLE FIRST, then acquisition ----
        feas = [y for y in keys if props(y)["v"] == 0.0]
        if do_dock:
            per = min(task.get("per_round", 20), budget - n_calls)
            if feas:
                fX = np.stack([props(y)["fp"] for y in feas])
                acq = sur.sample(fX) if sur.W is not None else rng.random(len(feas))
                pick = [feas[int(i)] for i in np.argsort(-acq)[:per]]
            else:
                pick = []
            if len(pick) < per:                  # fill from the soft frontier
                rest = [y for y in keys if y not in pick]
                w = score[[keys.index(y) for y in rest]]; w = w / w.sum()
                extra = rng.choice(len(rest), size=min(per - len(pick), len(rest)),
                                   replace=False, p=w)
                pick += [rest[int(i)] for i in extra]
            for y in pick:
                ds = _dock(y, target, f"h{task['idx']}_{rd}_{len(docked)}")
                n_calls += 1
                docked[y] = ds if ds is not None else 0.0
                if ds is not None:
                    Xs.append(props(y)["fp"]); ys.append(-ds)   # larger is better
                    if props(y)["v"] == 0.0:
                        A_oracle.append({"smi": y, "ds": ds, "round": rd,
                                         "call": n_calls})
            refresh_surrogate()

        log.append({"round": rd, "n_cand": len(keys), "n_feasible": len(feas),
                    "n_docked_total": n_calls, "n_lineages": len(parents),
                    "min_v": round(min(props(y)["v"] for y in keys), 4),
                    "surrogate_n": len(ys)})
        if not do_dock and rd >= task.get("rounds", 10):
            break

    best = min(A_oracle, key=lambda a: a["ds"]) if A_oracle else None
    out = {**{k: v for k, v in task.items() if k != "smiles"},
           "rounds": rd, "n_calls": n_calls,
           "n_feasible_docked": len(A_oracle),
           "first_feasible_call": A_oracle[0]["call"] if A_oracle else None,
           "best_ds": best["ds"] if best else None,
           "best_smiles": best["smi"] if best else None,
           "reached_feasible": any(l["n_feasible"] > 0 for l in log),
           "first_feasible_round": next((l["round"] for l in log if l["n_feasible"] > 0), None),
           "per_round": log, "seconds": round(time.time() - t0, 1)}
    try:
        d = Path("/artifacts/t4_controller"); d.mkdir(parents=True, exist_ok=True)
        (d / f"{task['tag']}_{task['idx']}_{task['target']}_{control}.json").write_text(
            json.dumps(out, indent=1))
        artifact_volume.commit()
    except Exception as e:
        print(f"  !! persist failed: {e}", flush=True)
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=2048, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(cells: str, control: str, budget: int, dock: bool, tag: str,
          horizon: int, rounds: int) -> dict[str, Any]:
    seeds = json.loads((REMOTE_ROOT / "docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    idxs = [int(x) for x in cells.split(",") if x != ""]
    arms = [c.strip() for c in control.split(",")]
    tasks = []
    for i in idxs:
        for arm in arms:
            for dl in (0.4,):
                tasks.append({"idx": i, "target": seeds[i]["target"],
                              "chembl": seeds[i]["chembl"], "smiles": seeds[i]["smiles"],
                              "qed": seeds[i]["qed"], "sa": seeds[i]["sa"], "delta": dl,
                              "control": arm, "budget": budget, "dock": dock,
                              "tag": tag, "horizon": horizon, "rounds": rounds,
                              "per_round": 20, "seed_rng": 20260821 + i})
    print(f"{len(tasks)} runs  arms={arms}  budget={budget}  dock={dock}\n", flush=True)
    out = []
    for r in run_cell.map(tasks, order_outputs=False, return_exceptions=True,
                          wrap_returned_exceptions=False):
        if not isinstance(r, dict):
            print(f"  !! {type(r).__name__}: {str(r)[:140]}", flush=True); continue
        out.append(r)
        print(f"  {r['target']:6s} {r['control']:9s} feas@rd {str(r['first_feasible_round']):>4}  "
              f"feas_docked {r['n_feasible_docked']:>3}  calls {r['n_calls']:>4}  "
              f"bestDS {r['best_ds'] if r['best_ds'] is not None else 'NONE'}", flush=True)
    if not out:
        d = Path("/artifacts/t4_controller")
        if d.exists():
            out = [json.loads(f.read_text()) for f in sorted(d.glob(f"{tag}_*.json"))]
            print(f"  recovered {len(out)} from volume", flush=True)
    if not out:
        raise RuntimeError("no runs completed")
    return {"runs": out, "tag": tag}


@app.function(image=image, cpu=(1.0, 1.0), memory=1024, timeout=600,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def salvage(tag: str = "") -> dict[str, Any]:
    d = Path("/artifacts/t4_controller")
    g = f"{tag}_*.json" if tag else "*.json"
    runs = [json.loads(f.read_text()) for f in sorted(d.glob(g))] if d.exists() else []
    print(f"  salvaged {len(runs)}", flush=True)
    return {"runs": runs, "salvaged": True}


@app.local_entrypoint()
def main(cells: str = "6,18,2", control: str = "immediate,future_h",
         budget: int = 200, dock: bool = True, tag: str = "h2",
         horizon: int = 4, rounds: int = 10, out: str = "") -> None:
    try:
        o = drive.remote(cells, control, budget, dock, tag, horizon, rounds)
    except Exception as e:
        print(f"  driver failed ({type(e).__name__}); salvaging")
        o = salvage.remote(tag)
    p = Path(__file__).resolve().parents[1] / (out or f"diagnostics/genmol_t4_controller_{tag}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
