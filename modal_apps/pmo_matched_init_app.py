"""Oracle-prescreened COMPOSE on PMO. Generic R_theta search, NO h, NO surrogate.

NAME THIS CAREFULLY. This arm is "oracle-prescreened" or "matched-information"
COMPOSE. It is NOT "the GenMol protocol", and the difference is not cosmetic.
GenMol scores ZINC250k, decomposes those molecules into fragments, propagates
molecule scores into a task-specific FRAGMENT vocabulary, and initializes from
the top fragments. We score the same 249,455 molecules with the same task oracle
and initialize from the top MOLECULES. What matches is the information -- 249,455
uncounted task-oracle evaluations before search -- consumed through each method's
native representation. Claiming an identical protocol would be false.

WHY THIS EXISTS. GenMol does not start PMO from a blind population. get_vocab.py
calls the task oracle on all of ZINC250k -- 250,000 UNCOUNTED calls per task,
against a 10,000 counted budget -- cuts every molecule into fragments, scores
each fragment by the mean oracle value of the molecules containing it, keeps the
top 10,000, and set_initial_population() takes df.iloc[:population_size], i.e.
the top 100. Those 100 arrive with scores already known and never touch
mol_buffer, which is what max_oracle_calls counts.

Our clean pilot started from an objective-blind bank. So the clean-vs-GenMol gap
conflates two things: initialization information and search competence. This app
separates them by giving COMPOSE the SAME information budget in COMPOSE's OWN
representation -- score ZINC250k with the task oracle outside the counted budget,
take the top BANK_N molecules -- and changing nothing else.

REPRESENTATION-NATIVE ON PURPOSE. We do not adopt GenMol's fragment-attachment
representation. Their unit is a fragment because their generator recombines
fragments; ours is a molecule because our kernel edits molecules. Matching the
information regime is the point; deforming COMPOSE into their action space would
confound the comparison rather than clean it.

INIT SCORES ARE NOT COUNTED, matching GenMol exactly: their population's scores
come precomputed from the vocab CSV. Counting ours would spend a fifth of a
500-call budget re-deriving numbers the prescreen already paid for, and would
make the arms differ by budget as well as by information.

Everything else -- frozen R_theta, N_LINEAGE, PER_ROUND, APPLY_CAP, the 400-state
cap, the seeds, OracleMeter with PER_MOLECULE counting -- is identical to
pmo_pilot_app, so clean@500 and matched@500 differ in initialization alone.
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

ZINC_LOCAL = ROOT / "local_runtime/zinc250k/250k_rndm_zinc_drugs_clean_3.csv"

image = (
    _opt_image
    .pip_install("PyTDC==0.3.6", "requests", "fuzzywuzzy", "seaborn", "networkx")
    .add_local_file(ROOT / "modal_apps/genmol_t4_opt_app.py",
                    str(REMOTE_ROOT / "modal_apps/genmol_t4_opt_app.py"), copy=True)
    .add_local_file(ROOT / "artifacts/oracles/molleo_task3_v1/jnk3_forest.npz",
                    "/frozen/jnk3_forest.npz", copy=True)
    .add_local_file(ROOT / "artifacts/oracles/molleo_task3_v1/gsk3b_forest.npz",
                    "/frozen/gsk3b_forest.npz", copy=True)
    .add_local_file(ZINC_LOCAL, "/zinc/zinc250k.csv", copy=True)
)

app = modal.App("pmo-matched-init")

TASKS = ["jnk3", "osimertinib_mpo", "scaffold_hop"]
CHECKPOINTS = (100, 250, 500)
N_LINEAGE, PER_ROUND = 12, 40
BANK_N = 100          # GenMol's population_size
SHARDS = 12           # prescreen shards per task; reduce is exact


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


def _top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    """PMO's top_auc, transcribed verbatim from the released optimizer.py.

    Two properties of the official metric matter and are easy to get wrong.
    First, `prev` starts at ZERO and `buffer` holds only COUNTED molecules --
    GenMol's initial population never enters mol_buffer -- so the curve ramps
    from zero rather than from the initialization value. Second, the released
    code calls this with finish=True HARDCODED, so the tail term always applies
    and a run shorter than max_oracle_calls is credited at its final top-n for
    the entire remaining budget. Any AUC we report against GenMol must come from
    this function, over a counted-only buffer, or it is a different quantity.
    """

    import numpy as np
    sum_, prev, called = 0, 0, 0
    ordered = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp = sorted(ordered[:idx], key=lambda kv: kv[1][0], reverse=True)[:top_n]
        now = float(np.mean([i[1][0] for i in temp]))
        sum_ += freq_log * (now + prev) / 2
        prev = now; called = idx
    temp = sorted(ordered, key=lambda kv: kv[1][0], reverse=True)[:top_n]
    now = float(np.mean([i[1][0] for i in temp]))
    sum_ += (len(buffer) - called) * (now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum_ += (max_oracle_calls - len(buffer)) * now
    return sum_ / max_oracle_calls


def _oracle(name):
    """Return (raw_fn, source_string). jnk3 uses our frozen npz; TDC ships it as
    a scikit-learn 0.21.3 pickle no modern environment can unpickle."""
    import numpy as np
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    if name in ("jnk3", "gsk3b"):
        import sys
        sys.path.insert(0, str(REMOTE_ROOT / "src"))
        from compose_v4.benchmark.oracles.forest import FrozenForest
        ff = FrozenForest(f"/frozen/{name}_forest.npz")
        def raw(s):
            m = Chem.MolFromSmiles(s)
            if m is None:
                return (0.0,)
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
            a = np.zeros((1,), dtype=np.int8); DataStructs.ConvertToNumpyArray(fp, a)
            return (float(ff.probabilities(a.astype(np.float64).reshape(1, -1))[0]),)
        return raw, "frozen npz (parity 0.000e+00)"
    from tdc import Oracle
    o = Oracle(name=name)
    return (lambda s: (float(o(s)),)), "PyTDC 0.3.6"


# --------------------------------------------------------------------------
# Phase 1: the prescreen. 249,455 UNCOUNTED oracle calls, top-BANK_N molecules.
#
# SHARDED. Every molecule is scored independently of every other, so splitting
# ZINC250k across workers and reducing to the global top-BANK_N is EXACTLY the
# serial result -- Top100(union of shard scores) == Top100(serial scores). This
# is execution parallelism only; no score, no tie, and no ordering changes.
# --------------------------------------------------------------------------
@app.function(image=image, cpu=(1.0, 1.0), memory=int(4 * 1024),
              timeout=2 * 60 * 60, max_containers=48)
def score_shard(job: dict[str, Any]) -> dict[str, Any]:
    import csv, os, sys
    _shim(); os.chdir("/tmp")
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    name, si, ns = job["task"], job["shard"], job["n_shards"]
    t0 = time.time()
    raw, _ = _oracle(name)

    with open("/zinc/zinc250k.csv", newline="") as fh:
        smis = [r["smiles"].strip() for r in csv.DictReader(fh)][si::ns]

    scored: dict[str, float] = {}
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        c = Chem.MolToSmiles(m)
        if c in scored:
            continue
        try:
            scored[c] = float(raw(c)[0])
        except Exception:
            continue
        # 1%-scale projection, printed before the bulk of the work is done, so a
        # mis-sized loop is visible in the first seconds instead of at the end.
        if i + 1 == 1000:
            r = (time.time() - t0) / 1000
            print(f"  [{name} s{si}] 1k in {time.time()-t0:.1f}s ({r*1000:.1f} ms/mol) "
                  f"-> shard {r*len(smis)/60:.1f} min", flush=True)
    print(f"  [{name} s{si}] {len(scored):,} scored in {time.time()-t0:.0f}s", flush=True)
    return {"task": name, "shard": si, "scores": scored,
            "seconds": round(time.time() - t0, 1)}


@app.function(image=image, cpu=(1.0, 1.0), memory=int(6 * 1024),
              timeout=4 * 60 * 60, max_containers=8,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def build_bank(name: str, n_shards: int = SHARDS) -> dict[str, Any]:
    import numpy as np
    t0 = time.time()

    # RESUME. A ZINC250k prescreen is expensive and deterministic, so it is persisted
    # once and reused. Without this, every relaunch -- including one that only
    # changes the search phase -- silently repays the whole prescreen.
    done = Path(f"/artifacts/pmo_matched_init/bank_{name}.json")
    labels = Path(f"/artifacts/pmo_matched_init/labels_{name}.npz")
    if done.exists() and labels.exists():
        o = json.loads(done.read_text())
        if len(o.get("bank", [])) >= BANK_N:
            print(f"  [{name}] bank already on volume, reusing "
                  f"({o['uncounted_calls']:,} uncounted calls)", flush=True)
            return o

    jobs = [{"task": name, "shard": i, "n_shards": n_shards} for i in range(n_shards)]
    scored: dict[str, float] = {}
    t_shards = []
    for r in score_shard.map(jobs, order_outputs=False, return_exceptions=True,
                             wrap_returned_exceptions=False):
        if not isinstance(r, dict):
            print(f"  !! shard {type(r).__name__}: {str(r)[:160]}", flush=True); continue
        scored.update(r["scores"]); t_shards.append(r["seconds"])
    if not scored:
        raise RuntimeError(f"no shard produced scores for {name}")

    order = sorted(scored.items(), key=lambda kv: -kv[1])
    bank = order[:BANK_N]
    v = np.array([x[1] for x in order], float)

    # PERSIST EVERY LABEL. The prescreen produces 249,455 (molecule, oracle value)
    # pairs at real cost and the bank uses 100 of them. Discarding the other
    # 249,355 throws away exactly the supervision a task surrogate needs, and
    # they are not recoverable without repaying the whole prescreen. Written as a
    # separate compressed artifact so it survives independently of the bank.
    lp = Path(f"/artifacts/pmo_matched_init/labels_{name}.npz")
    np.savez_compressed(lp,
                        smiles=np.array([k for k, _ in order], dtype=object),
                        score=np.array([u for _, u in order], dtype=np.float32))
    print(f"  [{name}] persisted {len(order):,} labels -> {lp.name} "
          f"({lp.stat().st_size/1e6:.1f} MB)", flush=True)
    _, src = None, ("frozen npz (parity 0.000e+00)" if name == "jnk3" else "PyTDC 0.3.6")
    out = {"task": name, "oracle_source": src,
           "uncounted_calls": len(scored),
           "n_shards": n_shards, "shard_seconds": t_shards,
           "labels_artifact": f"labels_{name}.npz",
           "bank_n": len(bank),
           "bank": [{"smiles": k, "u": u} for k, u in bank],
           "top1": float(v[0]), "top10_mean": float(v[:10].mean()),
           "bank_mean": float(v[:BANK_N].mean()),
           "median_all": float(np.median(v)),
           "seconds": round(time.time() - t0, 1)}
    d = Path("/artifacts/pmo_matched_init"); d.mkdir(parents=True, exist_ok=True)
    (d / f"bank_{name}.json").write_text(json.dumps(out, indent=1))
    artifact_volume.commit()
    print(f"  [{name}] BANK top1 {out['top1']:.4f}  top{BANK_N} {out['bank_mean']:.4f}  "
          f"top10 {out['top10_mean']:.4f}  |  {n_shards} shards, "
          f"{out['seconds']:.0f}s wall (slowest shard {max(t_shards):.0f}s)", flush=True)
    return out


# --------------------------------------------------------------------------
# Phase 2: identical generic search, matched init, 500 counted calls.
# --------------------------------------------------------------------------
@app.function(image=image, cpu=(2.0, 2.0), memory=int(8 * 1024),
              timeout=4 * 60 * 60, max_containers=4,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_task(task: dict[str, Any]) -> dict[str, Any]:
    import os, sys
    import numpy as np
    _shim(); os.chdir("/tmp")
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
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
    t0 = time.time()
    raw, src = _oracle(name)

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

    # A volume write committed by ANOTHER container is not visible here until the
    # mount is reloaded. build_bank runs in its own container, so a bank created
    # during this same app -- which is exactly what per-task pipelining causes --
    # is invisible without this call, and the search dies on FileNotFoundError.
    artifact_volume.reload()
    bank_meta = json.loads(Path(f"/artifacts/pmo_matched_init/bank_{name}.json").read_text())
    bank = bank_meta["bank"]

    # Init scores are KNOWN and UNCOUNTED, exactly as GenMol's vocab CSV supplies
    # them. meter.prime records them without charging the budget.
    pop: dict[str, dict] = {}
    ledger: dict[str, list] = {}
    depth = {}
    for b in bank:
        c = canon(b["smiles"])
        if c and c not in pop:
            pop[c] = {"smi": c, "u": float(b["u"]), "d": 0}; depth[c] = 0
            meter.prime(c, (float(b["u"]),))

    def snapshot():
        vals = sorted((p["u"] for p in pop.values()), reverse=True)
        top = vals[:10]
        return {"calls": meter.spent, "best": vals[0] if vals else 0.0,
                "top10_mean": float(np.mean(top)) if top else 0.0,
                "n_distinct_pop": len(pop),
                "unique": meter.n_unique, "lineages": len(pop),
                "max_depth": max(depth.values()) if depth else 0,
                "t": round(time.time() - t0, 1)}
    curve = [snapshot()]
    nxt = [c for c in CHECKPOINTS]

    def _partial(name, curve, meter, rd, pop, bank, bank_meta, src, budget):
        """Persist at every checkpoint. Two complete runs have already been lost
        by holding results in memory until the end, so nothing waits for the end."""
        try:
            vals = sorted((q["u"] for q in pop.values()), reverse=True)[:10]
            d = Path("/artifacts/pmo_matched_init"); d.mkdir(parents=True, exist_ok=True)
            (d / f"run_{name}.json").write_text(json.dumps(
                {"task": name, "arm": "oracle_prescreened", "oracle_source": src,
                 "budget": budget, "bank_n": len(bank),
                 "uncounted_prescreen_calls": bank_meta.get("uncounted_calls"),
                 "primed": meter.n_primed, "calls_spent": meter.spent,
                 "rounds": rd, "complete": False,
                 "best": float(vals[0]) if vals else 0.0,
                 "top10_mean": float(np.mean(vals)) if vals else 0.0,
                 "curve": curve}, indent=1))
            artifact_volume.commit()
        except Exception as e:
            print(f"  !! partial persist failed {name}: {e}", flush=True)

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
                if y not in cand and y not in pop:
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
            # COUNTED-ONLY ledger, in call order. Primed molecules are deliberately
            # absent: GenMol's initial population never enters mol_buffer either,
            # so including ours would inflate every checkpoint of the official AUC.
            if y not in ledger:
                ledger[y] = [float(u), len(ledger) + 1]
        if len(pop) > 400:
            keep = sorted(pop.values(), key=lambda p: -p["u"])[:400]
            pop = {p["smi"]: p for p in keep}
        while nxt and meter.spent >= nxt[0]:
            s = snapshot(); curve.append(s); nxt.pop(0)
            rate = s["t"] / max(s["calls"], 1)
            print(f"  [{name}] @{s['calls']:>4} calls  top10 {s['top10_mean']:.4f}  "
                  f"best {s['best']:.4f}  |  {s['t']:>6.1f}s  {rate:.3f} s/call  "
                  f"-> {rate * CHECKPOINTS[-1] / 60:.1f} min projected", flush=True)
            _partial(name, curve, meter, rd, pop, bank, bank_meta, src, budget)
    curve.append(snapshot())

    top10 = sorted((p["u"] for p in pop.values()), reverse=True)[:10]
    by_depth: dict[int, float] = {}
    for p in pop.values():
        by_depth[p["d"]] = max(by_depth.get(p["d"], 0.0), p["u"])
    out = {"task": name, "arm": "oracle_prescreened", "oracle_source": src, "budget": budget,
           "bank_n": len(bank),
           "uncounted_prescreen_calls": bank_meta.get("uncounted_calls"),
           "primed": meter.n_primed,
           "calls_spent": meter.spent, "unique": meter.n_unique,
           "rounds": rd, "best": float(top10[0]) if top10 else 0.0,
           "top10_mean": float(np.mean(top10)) if top10 else 0.0,
           "auc_top10": float(np.trapz([c["top10_mean"] for c in curve],
                                       [c["calls"] for c in curve]) / max(meter.spent, 1)),
           # The comparable-to-GenMol number. Counted-only buffer, finish=True,
           # normalised by 10,000 -- i.e. what PMO itself would print.
           "auc_top10_official_10k": _top_auc(ledger, 10, True, 100, 10000),
           "auc_top1_official_10k": _top_auc(ledger, 1, True, 100, 10000),
           "top10_counted_only": float(np.mean(sorted(
               (v[0] for v in ledger.values()), reverse=True)[:10])) if ledger else 0.0,
           "ledger_n": len(ledger),
           "curve": curve,
           "best_by_depth": {str(k): round(v, 5) for k, v in sorted(by_depth.items())},
           "max_depth": max(depth.values()) if depth else 0,
           "complete": True,
           "seconds": round(time.time() - t0, 1)}
    d = Path("/artifacts/pmo_matched_init"); d.mkdir(parents=True, exist_ok=True)
    (d / f"run_{name}.json").write_text(json.dumps(out, indent=1))
    artifact_volume.commit()
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=2048, timeout=8 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: str, budget: int, phase: str, n_shards: int = SHARDS) -> dict[str, Any]:
    """One independent pipeline per task: prescreen, then search, no barrier.

    A barrier between the two phases would hold jnk3's search -- whose prescreen
    finishes in ~5 minutes -- hostage to scaffold_hop's, which takes six times
    longer. The tasks share nothing, so each one runs its own bank-then-search
    chain and starts searching the instant ITS bank exists.
    """

    from concurrent.futures import ThreadPoolExecutor
    names = [t.strip() for t in tasks.split(",") if t.strip()]
    banks: list[dict] = []
    runs: list[dict] = []

    def one(name):
        b = r = None
        try:
            if phase in ("bank", "both"):
                b = build_bank.remote(name, n_shards)
                banks.append(b)
            if phase in ("run", "both"):
                r = run_task.remote({"task": name, "budget": budget,
                                     "seed_rng": 20260822})
                runs.append(r)
                print(f"  run  {r['task']:18s} best {r['best']:.4f}  "
                      f"top10 {r['top10_mean']:.4f}  AUC {r['auc_top10']:.4f}  "
                      f"calls {r['calls_spent']}  {r['seconds']:.0f}s", flush=True)
        except Exception as e:
            print(f"  !! {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=max(len(names), 1)) as ex:
        list(ex.map(one, names))
    return {"banks": banks, "runs": runs}


@app.function(image=image, cpu=(1.0, 1.0), memory=1024, timeout=600,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def salvage() -> dict[str, Any]:
    d = Path("/artifacts/pmo_matched_init")
    f = lambda g: [json.loads(p.read_text()) for p in sorted(d.glob(g))] if d.exists() else []
    o = {"banks": f("bank_*.json"), "runs": f("run_*.json"), "salvaged": True}
    print(f"  salvaged {len(o['banks'])} banks, {len(o['runs'])} runs", flush=True)
    return o


@app.local_entrypoint()
def main(tasks: str = "", budget: int = 500, phase: str = "both",
         n_shards: int = SHARDS, out: str = "") -> None:
    tasks = tasks or ",".join(TASKS)
    try:
        o = drive.remote(tasks, budget, phase, n_shards)
    except Exception as e:
        print(f"  driver failed ({type(e).__name__}); salvaging")
        o = salvage.remote()
    p = Path(__file__).resolve().parents[1] / (out or "diagnostics/pmo_matched_init.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1))
    print(f"\nwrote {p}")
