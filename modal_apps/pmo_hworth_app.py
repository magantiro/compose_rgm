"""Is future-value control worth building FOR PMO? A diagnostic, not a controller.

THE QUESTION. Macro proposals already gave jnk3 +0.071 by sampling trajectory
lengths BLINDLY. Future-value control would only add something beyond that if it
can pick WHICH futures to pursue -- which requires that immediate score and
downstream achievable score actually disagree. This measures that disagreement
directly, and nothing else.

For frontier states, for each immediate successor y, we compare

    f(y)                      the immediate true oracle score
    V_L(y) = top-k over M     R_theta continuations of length L from y

If ranking by f(y) recovers the same ordering as V_L(y), there is no lookahead
signal to exploit and h would be machinery in service of nothing. If they
disagree -- especially if the best-V successors rank poorly under f -- that is
direct evidence h can beat blind macro sampling.

WHY jnk3. It is our largest important deficit (+0.235 at 100 calls) AND the task
where macro proposals already work, so it is where the marginal question "does
choosing futures beat sampling them" actually has stakes.

THE ORACLE IS USED OFF-BUDGET, DEVELOPMENT ONLY. Nothing here is reportable and
nothing here may train a controller: these labels are task information beyond
the matched prescreen regime.

SIZING. One container per frontier state, so the fan-out is the state count. Each
container microbenchmarks its first continuation and prints a projection before
committing to the rest.
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
    .add_local_file(ROOT / "artifacts/oracles/molleo_task3_v1/gsk3b_forest.npz",
                    "/frozen/gsk3b_forest.npz", copy=True)
)

app = modal.App("pmo-hworth")

N_SUCC, ROLLOUTS, DEPTHS = 8, 3, (2, 4)


def _spearman(a, b):
    import numpy as np
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = (np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra*rb).sum()/d) if d > 0 else 0.0


@app.function(image=image, cpu=(1.0, 1.0), memory=int(6 * 1024),
              timeout=2 * 60 * 60, max_containers=24,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def probe_state(job: dict[str, Any]) -> dict[str, Any]:
    import os, sys
    import numpy as np
    import types
    six = types.ModuleType("rdkit.six")
    six.iteritems = lambda d, **k: iter(d.items()); six.itervalues = lambda d, **k: iter(d.values())
    six.iterkeys = lambda d, **k: iter(d.keys()); six.string_types = (str,)
    sys.modules["rdkit.six"] = six
    import rdkit; rdkit.six = six
    os.chdir("/tmp"); sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)

    task, x, si = job["task"], job["smiles"], job["idx"]
    rng = np.random.default_rng(1000 + si)
    t0 = time.time()
    rt = _runtime(); model, system = rt["model"], rt["system"]

    if task in ("jnk3", "gsk3b"):
        from compose_v4.benchmark.oracles.forest import FrozenForest
        ff = FrozenForest(f"/frozen/{task}_forest.npz")
        def f(s):
            m = Chem.MolFromSmiles(s)
            if m is None: return 0.0
            a = np.zeros((1,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(
                AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), a)
            v = float(ff.probabilities(a.astype(np.float64).reshape(1, -1))[0])
            return 0.0 if v != v else v
    else:
        from tdc import Oracle
        o = Oracle(name=task)
        def f(s):
            try: v = float(o(s))
            except Exception: return 0.0
            return 0.0 if v != v else v

    LAW: dict[str, Any] = {}
    def law_of(smi):
        if smi not in LAW:
            try:
                st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
                lw = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
                pr = np.array([m.probability for m in lw.marks], float)
                ix = np.argsort(-pr)[:APPLY_CAP]
                LAW[smi] = (st, lw, ix, pr[ix]/pr[ix].sum())
            except Exception:
                LAW[smi] = None
        return LAW[smi]

    def step(smi):
        e = law_of(smi)
        if e is None: return None
        st, lw, ix, w = e
        mk = lw.marks[int(ix[int(rng.choice(len(ix), p=w))])]
        try: y = canonical_state_key(system.apply(st, mk.executor_rule_name, mk.action))
        except Exception: return None
        return y if y and y != smi else None

    # immediate successors of the frontier state
    e = law_of(x)
    if e is None:
        return {"idx": si, "task": task, "error": "no law for state"}
    st, lw, ix, w = e
    succ: dict[str, float] = {}
    for i in ix:
        mk = lw.marks[int(i)]
        try: y = canonical_state_key(system.apply(st, mk.executor_rule_name, mk.action))
        except Exception: continue
        if y and y != x and y not in succ:
            succ[y] = float(lw.marks[int(i)].probability)
        if len(succ) >= N_SUCC: break
    if len(succ) < 4:
        return {"idx": si, "task": task, "error": f"only {len(succ)} successors"}

    ys = list(succ)
    tb = time.time(); _ = step(ys[0]); t_step = time.time() - tb
    proj = len(ys) * ROLLOUTS * sum(DEPTHS) * t_step / 60
    print(f"  [s{si}] {len(ys)} successors, step {t_step:.2f}s -> projected {proj:.1f} min",
          flush=True)

    f_imm = np.array([f(y) for y in ys])
    V = {L: [] for L in DEPTHS}
    for y in ys:
        for L in DEPTHS:
            vals = []
            for _ in range(ROLLOUTS):
                cur = y
                for _d in range(L):
                    nx = step(cur)
                    if nx is None: break
                    cur = nx
                vals.append(f(cur))
            V[L].append(max(vals) if vals else 0.0)

    out = {"idx": si, "task": task, "state": x, "n_succ": len(ys),
           "f_immediate": [round(v, 5) for v in f_imm.tolist()],
           "seconds": round(time.time() - t0, 1)}
    for L in DEPTHS:
        v = np.array(V[L])
        out[f"V{L}"] = [round(z, 5) for z in v.tolist()]
        out[f"spearman_f_vs_V{L}"] = round(_spearman(f_imm, v), 4)
        # the operational question: where does the best-future successor rank
        # under immediate score? 1 means greedy already picks it.
        out[f"rank_of_best_V{L}_under_f"] = int(
            np.where(np.argsort(-f_imm) == int(np.argmax(v)))[0][0]) + 1
        out[f"best_V{L}"] = float(v.max())
        out[f"f_of_best_V{L}"] = float(f_imm[int(np.argmax(v))])
    out["best_f"] = float(f_imm.max())
    print(f"  [s{si}] spearman f vs V2 {out['spearman_f_vs_V2']:+.3f} "
          f"V4 {out['spearman_f_vs_V4']:+.3f}  bestV4 rank under f: "
          f"{out['rank_of_best_V4_under_f']}/{len(ys)}  ({out['seconds']:.0f}s)", flush=True)
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=2048, timeout=4 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(task: str, n_states: int) -> dict[str, Any]:
    import numpy as np
    artifact_volume.reload()
    # frontier states: the best COUNTED molecules from the L1 run on this task
    src = Path(f"/artifacts/pmo_control/{task}_L1_s1.json")
    states: list[str] = []
    if src.exists():
        r = json.loads(src.read_text())
        states = [s for s in r.get("top_smiles", [])]
    if not states:
        bank = json.loads(
            Path(f"/artifacts/pmo_matched_init/bank_{task}.json").read_text())["bank"]
        states = [b["smiles"] for b in bank[:n_states]]
        print(f"  no frontier smiles persisted; falling back to the bank top-{n_states}",
              flush=True)
    states = states[:n_states]
    jobs = [{"task": task, "smiles": s, "idx": i} for i, s in enumerate(states)]
    print(f"  {len(jobs)} frontier states, {N_SUCC} successors each, "
          f"{ROLLOUTS} rollouts x depths {DEPTHS}\n", flush=True)
    out = []
    for r in probe_state.map(jobs, order_outputs=False, return_exceptions=True,
                             wrap_returned_exceptions=False):
        if not isinstance(r, dict):
            print(f"  !! {type(r).__name__}: {str(r)[:160]}", flush=True); continue
        out.append(r)
    ok = [r for r in out if "error" not in r]
    agg = {}
    if ok:
        for L in DEPTHS:
            sp = [r[f"spearman_f_vs_V{L}"] for r in ok]
            rk = [r[f"rank_of_best_V{L}_under_f"] for r in ok]
            gain = [r[f"best_V{L}"] - r["best_f"] for r in ok]
            agg[f"L{L}"] = {
                "median_spearman_f_vs_V": round(float(np.median(sp)), 4),
                "median_rank_of_best_V_under_f": float(np.median(rk)),
                "frac_best_V_not_top1_under_f": round(
                    float(np.mean([x > 1 for x in rk])), 3),
                "median_gain_bestV_minus_bestf": round(float(np.median(gain)), 5),
                "frac_states_with_gain": round(float(np.mean([g > 0 for g in gain])), 3)}
            print(f"  L={L}: median spearman(f,V) {agg[f'L{L}']['median_spearman_f_vs_V']:+.3f}  "
                  f"median rank of best-V under f {agg[f'L{L}']['median_rank_of_best_V_under_f']}  "
                  f"gain {agg[f'L{L}']['median_gain_bestV_minus_bestf']:+.4f}", flush=True)
    res = {"task": task, "reportable": False, "n_states": len(ok),
           "aggregate": agg, "states": out}
    d = Path("/artifacts/pmo_hworth"); d.mkdir(parents=True, exist_ok=True)
    (d / f"hworth_{task}.json").write_text(json.dumps(res, indent=1))
    artifact_volume.commit()
    return res


@app.local_entrypoint()
def main(task: str = "jnk3", n_states: int = 20, out: str = "") -> None:
    o = drive.remote(task, n_states)
    p = Path(__file__).resolve().parents[1] / (out or f"diagnostics/pmo_hworth_{task}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
