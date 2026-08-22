"""PMO six-task diagnostic pilot. Generic R_theta search, NO h, NO surrogate.

The question is deliberately narrow:

    given the frozen R_theta and a competent but SIMPLE generic population
    search, which PMO landscapes can it already optimize, and where does it
    fail?

No adaptive h-control, no docking surrogate, no per-task tuning. If h were on
here and QED and scaffold_hop both worked, we could not tell whether R_theta
alone was already adequate. The pilot exists to make that separable.

What IS carried over from T4 is only non-task-specific hygiene the diagnostics
forced: canonical deduplication, multiple persistent lineages rather than one
collapsing parent chain, and never charging a duplicate oracle evaluation.

ORACLES. Five come from PyTDC. jnk3 does NOT: TDC ships it as a scikit-learn
0.21.3 pickle that no modern environment can unpickle. We use our frozen npz
instead, which the parity gate showed reproduces the TDC estimator to 0.000e+00
on a fixed panel, given identical features that themselves matched TDC's
construction to 0.000e+00 over 200 molecules. drd2 is EXCLUDED: its extraction
is not yet verified.

Every oracle call goes through OracleMeter, so the 10k contract is enforced and
duplicates are free exactly as PMO's own mol_buffer makes them free.
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
    .add_local_file(ROOT / "docs/PMO_INIT_BANK.json",
                    str(REMOTE_ROOT / "docs/PMO_INIT_BANK.json"), copy=True)
    .add_local_file(ROOT / "artifacts/oracles/molleo_task3_v1/jnk3_forest.npz",
                    "/frozen/jnk3_forest.npz", copy=True)
)

app = modal.App("pmo-pilot")

TASKS = ["qed", "jnk3", "albuterol_similarity", "celecoxib_rediscovery",
         "osimertinib_mpo", "scaffold_hop"]
CHECKPOINTS = (100, 250, 500, 1000, 2000)
N_LINEAGE, PER_ROUND = 12, 40


def _shim():
    """PyTDC 0.3.6 imports rdkit.six, removed from RDKit years ago."""
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
              timeout=8 * 60 * 60, max_containers=8,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_task(task: dict[str, Any]) -> dict[str, Any]:
    import sys, os
    import numpy as np
    _shim()
    os.chdir("/tmp")
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.benchmark.molleo_task3 import OracleMeter, CountingRule, BudgetExceeded

    name, budget = task["task"], task["budget"]
    rng = np.random.default_rng(task["seed_rng"])
    rt = _runtime(); model, system = rt["model"], rt["system"]
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    t0 = time.time()

    # ---- oracle ----
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
        src = "frozen npz (TDC pickle unloadable in modern env; parity 0.000e+00)"
    else:
        from tdc import Oracle
        o = Oracle(name=name)
        raw = lambda s: (float(o(s)),)
        src = "PyTDC 0.3.6"

    canon = lambda s: (lambda m: Chem.MolToSmiles(m) if m else None)(Chem.MolFromSmiles(s))
    meter = OracleMeter(evaluate=raw, budget=budget,
                        counting_rule=CountingRule.PER_MOLECULE,
                        canonicalize=canon, n_objectives=1)

    TT: dict[str, dict] = {}
    def fiber(smi):
        e = TT.setdefault(smi, {})
        if "f" in e:
            return e["f"]
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        except Exception:
            e["f"] = {}; return e["f"]
        if not law.marks:
            e["f"] = {}; return e["f"]
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
        e["f"] = out
        return out

    init = json.loads((REMOTE_ROOT / "docs/PMO_INIT_BANK.json").read_text())["smiles"]
    # POP IS KEYED BY CANONICAL SMILES. A list allowed the same molecule to be
    # appended twice when reached from two parents, which inflated top-10 into
    # the mean of one molecule with itself. PMO's metric is the mean of the ten
    # highest DISTINCT molecules.
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
                "n_distinct_pop": len(pop),
                "unique": meter.n_unique, "lineages": len(pop),
                "max_depth": max(depth.values()) if depth else 0}
    curve.append(snapshot())
    nxt = [c for c in CHECKPOINTS]

    rd, stop = 0, False
    while meter.remaining > 0 and not stop:
        rd += 1
        ranked = sorted(pop.values(), key=lambda p: -p["u"])
        elite = ranked[:N_LINEAGE // 2]
        rest = ranked[N_LINEAGE // 2:]
        div = list(rng.choice(rest, size=min(N_LINEAGE - len(elite), len(rest)),
                              replace=False)) if rest else []
        parents = elite + list(div)
        cand: dict[str, float] = {}
        for p in parents:
            for y, r in fiber(p["smi"]).items():
                if y not in cand and y not in pop:      # global seen-set
                    cand[y] = r
                    depth.setdefault(y, depth.get(p["smi"], 0) + 1)
        if not cand:
            break
        keys = list(cand)
        w = np.array([cand[y] for y in keys]); w = w / w.sum()
        k = min(PER_ROUND, len(keys), meter.remaining)
        pick = [keys[int(i)] for i in rng.choice(len(keys), size=k, replace=False, p=w)]
        for y in pick:
            try:
                u = meter(y)[0]
            except BudgetExceeded:
                stop = True; break
            pop[y] = {"smi": y, "u": u, "d": depth.get(y, 0)}
        if len(pop) > 400:                              # keep the best 400 DISTINCT
            keep = sorted(pop.values(), key=lambda p: -p["u"])[:400]
            pop = {p["smi"]: p for p in keep}
        while nxt and meter.spent >= nxt[0]:
            curve.append(snapshot()); nxt.pop(0)
    curve.append(snapshot())

    top10 = sorted((p["u"] for p in pop.values()), reverse=True)[:10]
    by_depth: dict[int, float] = {}
    for p in pop.values():
        by_depth[p["d"]] = max(by_depth.get(p["d"], 0.0), p["u"])
    out = {"task": name, "oracle_source": src, "budget": budget,
           "calls_spent": meter.spent, "unique": meter.n_unique,
           "rounds": rd, "best": float(top10[0]) if top10 else 0.0,
           "top10_mean": float(np.mean(top10)) if top10 else 0.0,
           "auc_top10": float(np.trapz([c["top10_mean"] for c in curve],
                                       [c["calls"] for c in curve]) / max(meter.spent, 1)),
           "curve": curve, "best_by_depth": {str(k): round(v, 5) for k, v in sorted(by_depth.items())},
           "max_depth": max(depth.values()) if depth else 0,
           "seconds": round(time.time() - t0, 1)}
    try:
        d = Path("/artifacts/pmo_pilot"); d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps(out, indent=1))
        artifact_volume.commit()
    except Exception as e:
        print(f"  !! persist failed {name}: {e}", flush=True)
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=2048, timeout=10 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(budget: int, tasks: str) -> dict[str, Any]:
    names = [t for t in (tasks.split(",") if tasks else TASKS)]
    ts = [{"task": n, "budget": budget, "seed_rng": 20260821} for n in names]
    print(f"{len(ts)} PMO tasks, {budget} oracle calls each, generic R_theta search, "
          f"NO h, NO surrogate\n", flush=True)
    out = []
    for r in run_task.map(ts, order_outputs=False, return_exceptions=True,
                          wrap_returned_exceptions=False):
        if not isinstance(r, dict):
            print(f"  !! {type(r).__name__}: {str(r)[:140]}", flush=True); continue
        out.append(r)
        print(f"  {r['task']:24s} best {r['best']:.4f}  top10 {r['top10_mean']:.4f}  "
              f"AUC {r['auc_top10']:.4f}  calls {r['calls_spent']}  depth {r['max_depth']}",
              flush=True)
    if not out:
        d = Path("/artifacts/pmo_pilot")
        if d.exists():
            out = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
            print(f"  recovered {len(out)} from volume", flush=True)
    if not out:
        raise RuntimeError("no tasks completed")
    return {"tasks": out}


@app.function(image=image, cpu=(1.0, 1.0), memory=1024, timeout=600,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def salvage() -> dict[str, Any]:
    d = Path("/artifacts/pmo_pilot")
    t = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))] if d.exists() else []
    print(f"  salvaged {len(t)}", flush=True)
    return {"tasks": t, "salvaged": True}


@app.local_entrypoint()
def main(budget: int = 2000, tasks: str = "") -> None:
    try:
        o = drive.remote(budget, tasks)
    except Exception as e:
        print(f"  driver failed ({type(e).__name__}); salvaging")
        o = salvage.remote()
    p = Path(__file__).resolve().parents[1] / "diagnostics/pmo_pilot.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
