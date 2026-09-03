"""End-to-end population search: the whole COMPOSE loop, one cheap objective.

    population -> mu_exec(M|x) -> frozen guided rewrite -> task score -> resample

Every component below this line is already qualified and is NOT retuned here:
the structural committor with the KL trust region at kappa=1, the region
feasibility/cost prior calibrated on the local-to-global gate, and the
scale-balanced exploration floor. The only new thing is the loop that connects
them.

The objective is deliberately cheap -- QED, and SA-like heavy-atom sanity -- not
docking. The question at this stage is whether the machinery optimises ANYTHING
end to end, not whether it wins a benchmark. Docking arrives with T4.

Two properties are recorded because they are how this can silently go wrong:

  * scale usage. mu_exec rewards successes per second and cheap pendant edits
    win that trade, so a run that improves the objective purely by shaving
    methyls has not demonstrated local-to-global search. The exploration floor
    is meant to prevent that, and the per-scale breakdown is how we check.
  * the archive is scored on the SAME objective that drives selection, so a
    rising best-score is not evidence of generalisation, only of optimisation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
app = modal.App("compose-population-search")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
OUT_DIR = "/artifacts/region_committor"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
MEM_MIB = int(3 * 1024)
_RT: dict[str, Any] = {}


def _runtime():
    if "model" in _RT:
        return _RT
    import sys
    import torch
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source)
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state)
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME)
    from compose_v4.experiments.production_successor_kernel import (
        _default_rewrite_system)
    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    src = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]),
        repo_root=Path(REMOTE_ROOT))
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        src, materialized_state=bundle)
    model = runtime.model
    ck = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["selected_model_state"], strict=True)
    model.eval(); torch.set_grad_enabled(False); torch.set_num_threads(1)
    _RT.update({"model": model, "system": _default_rewrite_system(model)})
    return _RT


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=4 * 60 * 60,
              retries=2, max_containers=80,
              volumes={ARTIFACT_ROOT: artifact_volume})
def search_cell(job: dict) -> dict:
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    import numpy as np
    import torch
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.region import enumerate_regions
    from compose_v4.control import region_rewrite as RR
    from compose_v4.control import graph_geometry as GG
    from compose_v4.control.region_selector import sample_region
    from compose_v4.control.task_value import TaskValue
    from compose_v4.gates.med_chem_gate import is_valid

    rt = _runtime(); model, system = rt["model"], rt["system"]
    rng = np.random.default_rng(int(job.get("seed", 0)))
    kappa = float(job.get("kappa", 1.0))
    epsilon_region = float(job.get("epsilon_region", 0.2))
    iters = int(job.get("iters", 12))
    pop_size = int(job.get("pop_size", 8))
    particles = int(job.get("particles", 1))
    tau = float(job.get("tau", 0.05))
    tv = TaskValue.from_dict(job["value_table"]) if job.get("value_table") else None
    value_fn = tv.value_fn() if tv is not None else None   # None -> mu_exec only

    artifact_volume.reload()
    ck = torch.load(f"{OUT_DIR}/committor_bellman_v1.pt", map_location="cpu")
    n_feat = int(ck["n_features"])
    net = torch.nn.Sequential(torch.nn.Linear(n_feat, 48), torch.nn.ReLU(),
                              torch.nn.Linear(48, 1))
    net.load_state_dict(ck["state_dict"]); net.eval()
    torch.set_grad_enabled(False)

    def objective(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return float(QED.qed(m))

    start = job["seed_smiles"]
    s0 = objective(start)
    if s0 is None:
        return {"cell": job["cell"], "error": "unparseable seed"}
    # population of (smiles, score); starts as copies of the seed
    pop = [(start, s0)] * pop_size
    archive = {start: s0}
    events, t0 = [], time.time()

    for it in range(iters):
        # pick a parent by score (softmax over the population)
        sc = np.array([p[1] for p in pop], float)
        w = np.exp((sc - sc.max()) / 0.05)
        parent = pop[int(rng.choice(len(pop), p=w / w.sum()))][0]
        try:
            regions = enumerate_regions(parent)
        except Exception:
            continue
        regions = [r for r in regions if 1 <= r.size <= 24]
        if not regions:
            continue
        reg, rs = sample_region(regions, rng, epsilon=epsilon_region,
                                value_fn=value_fn, tau=tau)
        if reg is None:
            continue

        st0 = pad_molecular_graph(smiles_to_molecular_graph(parent), CANONICAL_SLOTS)
        raw = smiles_to_molecular_graph(parent)
        n_real = len(raw.atom_types)
        ctx = RR.context_from_region(reg)
        lin0 = RR.Lineage.initial(range(n_real))
        old_ids = frozenset(lin0.id_of[s] for s in reg.atoms if s in lin0.id_of)
        cache: dict = {}
        memo: dict = {}

        def slot_key(state):
            return (np.asarray(state.atom_types).tobytes(),
                    np.asarray(state.bonds).tobytes())

        def enum_fn(st):
            k = slot_key(st)
            if k not in cache:
                law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
                cache[k] = ([m.executor_rule_name for m in law.marks],
                            [m.action for m in law.marks],
                            np.array([m.probability for m in law.marks], float))
            return cache[k]

        def apply_fn(st, j):
            k = (slot_key(st), int(j))
            if k in memo:
                return memo[k]
            fams, acts, _ = enum_fn(st)
            try:
                y = system.apply(st, fams[j], acts[j])
            except Exception:
                y = None
            memo[k] = y
            return y

        hc: dict = {}

        def h_model(y, c, lin, budget, kind, _o=old_ids, _h=hc):
            key = (slot_key(y), int(budget), kind)
            if key not in _h:
                f = RR.features_from_shared(
                    RR.structural_features_shared(y, c, lin, _o), int(budget), kind)
                _h[key] = float(torch.sigmoid(
                    net(torch.tensor([f], dtype=torch.float32))).item())
            return _h[key]

        sched = ([("prune_old", max(1, reg.size)), ("grow_new", 2)]
                 if reg.interface == "pendant"
                 else [("grow_new", "until_handoff"),
                       ("prune_old", max(1, reg.size))])
        best = None
        for particle in range(particles):
            p = RR.propose(enum_fn, apply_fn, ctx, st0, schedule=sched,
                           rng=np.random.default_rng(int(rng.integers(0, 10 ** 6))),
                           lineage=lin0, original_region_ids=old_ids,
                           max_handoff_steps=int(job.get("max_handoff", 16)),
                           h_model=h_model, tilt="kl", kappa=kappa,
                           epsilon=float(job.get("epsilon", 0.1)))
            if p.status != "OK" or p.endpoint is None:
                continue
            key = canonical_state_key(p.endpoint)
            if not key or not is_valid(key):
                continue
            sc_new = objective(key)
            if sc_new is None:
                continue
            d = GG.structural_displacement(st0, p.endpoint, lin0, p.lineage)
            if best is None or sc_new > best[1]:
                best = (key, sc_new, d, p)
        ev = {"iter": it, "parent_score": objective(parent),
              "interface": reg.interface, "kind": reg.kind,
              "r_release": reg.released_fraction, "region_size": reg.size,
              "mu_exec": rs.mu_exec, "ok": best is not None,
              "sec": time.time() - t0}
        if best is not None:
            key, sc_new, d, p = best
            ev.update({"child_score": sc_new,
                       "delta": sc_new - objective(parent),
                       "r_coherent": d["largest_changed_fraction"],
                       "n_steps": len(p.steps)})
            archive[key] = max(archive.get(key, -1.0), sc_new)
            pop = sorted(pop + [(key, sc_new)], key=lambda t: -t[1])[:pop_size]
        events.append(ev)
        print(f"[pop] {job['cell']} it={it} {reg.interface}/{reg.kind} "
              f"r_rel={reg.released_fraction:.2f} ok={int(ev['ok'])} "
              f"best={max(s for s in archive.values()):.4f} "
              f"{time.time() - t0:.0f}s", flush=True)

    res = {"cell": job["cell"], "arm": job.get("arm", "mu_exec"),
           "seed_smiles": start, "seed_score": s0,
           "best_score": max(archive.values()),
           "improvement": max(archive.values()) - s0,
           "n_archive": len(archive), "events": events,
           "sec": time.time() - t0}
    d_out = Path(OUT_DIR) / "population"
    d_out.mkdir(parents=True, exist_ok=True)
    (d_out / f"{job['cell']}.json").write_text(json.dumps(res))
    artifact_volume.commit()
    return res


@app.function(image=image, cpu=(1.0, 1.0), memory=int(2 * 1024), timeout=15 * 60,
              volumes={ARTIFACT_ROOT: artifact_volume})
def harvest() -> list:
    artifact_volume.reload()
    d = Path(OUT_DIR) / "population"
    return [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))] if d.exists() else []


@app.local_entrypoint()
def main(n_seeds: int = 12, iters: int = 12, pop_size: int = 8,
         particles: int = 1, epsilon_region: float = 0.2):
    import sys as _sys
    _sys.path.insert(0, "tools")
    from preflight import assert_synced
    assert_synced(strict=False)   # abort before any container starts if the
                                  # mounted tree is not what git says it is
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    jobs = [{"cell": f"pop{i}", "seed_smiles": s["smiles"], "seed": 17 + i,
             "iters": iters, "pop_size": pop_size, "particles": particles,
             "epsilon_region": epsilon_region}
            for i, s in enumerate(dv[:n_seeds])]
    print(f"cells={len(jobs)} iters={iters} pop={pop_size} particles={particles} "
          f"epsilon_region={epsilon_region} objective=QED (cheap, not docking)")
    out = [o for o in search_cell.map(jobs) if o and "error" not in o]
    report(out)


@app.local_entrypoint()
def collect(indices: str = "7,6,19,12,2,18,13", iters: int = 15,
            pop_size: int = 6, particles: int = 1, tag: str = "fit"):
    """Generate (x, M) -> delta observations with mu_exec only.

    This is both the V_z training set and the control arm's own protocol, run on
    a DISJOINT set of seeds from the eventual test so the estimator is never
    fitted on the molecules it is judged on.
    """
    import sys as _sys
    _sys.path.insert(0, "tools")
    from preflight import assert_synced
    assert_synced(strict=False)
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    idx = [int(i) for i in indices.split(",")]
    jobs = [{"cell": f"{tag}{i}", "seed_smiles": dv[i]["smiles"], "seed": 31 + i,
             "iters": iters, "pop_size": pop_size, "particles": particles,
             "arm": "mu_exec"} for i in idx]
    print(f"collecting on seeds {idx} (QED "
          f"{[round(dv[i]['qed'], 3) for i in idx]})")
    out = [o for o in search_cell.map(jobs) if o and "error" not in o]
    ev = [e for o in out for e in o["events"]]
    acc = [e for e in ev if e.get("ok")]
    Path("diagnostics").mkdir(exist_ok=True)
    Path(f"diagnostics/population_{tag}.json").write_text(json.dumps(out, default=str))
    _sys.path.insert(0, "src")
    from compose_v4.control.task_value import TaskValue
    tv = TaskValue.from_events(ev)
    Path("diagnostics/task_value_qed.json").write_text(json.dumps(tv.to_dict()))
    print(f"\nproposals={len(ev)} accepted={len(acc)} "
          f"({len(acc)/max(1,len(ev)):.0%})  global mean delta={tv.global_mean:+.5f}")
    print("fitted cells (interface, scope band) -> V:")
    for (iface, band), (n, s_) in sorted(tv._cells.items()):
        print(f"  {iface:<10} band{band} n={n:<3} V={tv.value(iface, band*0.2+0.01):+.5f}")
    print("\nwrote diagnostics/task_value_qed.json")


@app.local_entrypoint()
def ab(indices: str = "3,9,8,5,21,4", iters: int = 15, pop_size: int = 6,
       particles: int = 1, tau: float = 0.05):
    """Matched A/B: mu_exec alone against mu_exec * exp(V_z/tau).

    Same seeds, same iteration budget, same particles, same everything below the
    region selector. The ONLY difference is whether the task-value tilt is
    applied, so a difference is attributable to task conditioning.
    """
    import sys as _sys
    _sys.path.insert(0, "tools")
    from preflight import assert_synced
    assert_synced(strict=False)
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    table = json.loads(Path("diagnostics/task_value_qed.json").read_text())
    idx = [int(i) for i in indices.split(",")]
    fit_seeds = json.loads(Path("diagnostics/population_fit.json").read_text())
    fit_smiles = {o["seed_smiles"] for o in fit_seeds}
    assert not (fit_smiles & {dv[i]["smiles"] for i in idx}), \
        "test seeds overlap the seeds V_z was fitted on"
    jobs = []
    for i in idx:
        base = {"seed_smiles": dv[i]["smiles"], "seed": 71 + i, "iters": iters,
                "pop_size": pop_size, "particles": particles, "tau": tau}
        jobs.append({**base, "cell": f"A{i}", "arm": "mu_exec"})
        jobs.append({**base, "cell": f"B{i}", "arm": "Q_taskvalue",
                     "value_table": table})
    print(f"A/B on seeds {idx}, {len(jobs)} cells, tau={tau}")
    out = [o for o in search_cell.map(jobs) if o and "error" not in o]
    Path("diagnostics/population_ab.json").write_text(json.dumps(out, default=str))
    import statistics as st
    for arm in ("mu_exec", "Q_taskvalue"):
        v = [o for o in out if o["arm"] == arm]
        imp = [o["improvement"] for o in v]
        ev = [e for o in v for e in o["events"]]
        acc = [e for e in ev if e.get("ok")]
        print(f"\n{arm:<13} cells={len(v)} improved={sum(1 for i in imp if i > 1e-6)}"
              f"  median={st.median(imp) if imp else 0:+.4f} max={max(imp, default=0):+.4f}"
              f"  accepted={len(acc)}/{len(ev)}")
        if acc:
            print(f"  mean delta of accepted rewrites: "
                  f"{st.mean([e['delta'] for e in acc]):+.5f}")
    pair = {}
    for o in out:
        pair.setdefault(o["seed_smiles"], {})[o["arm"]] = o["improvement"]
    wins = sum(1 for p in pair.values()
               if p.get("Q_taskvalue", 0) > p.get("mu_exec", 0))
    loss = sum(1 for p in pair.values()
               if p.get("mu_exec", 0) > p.get("Q_taskvalue", 0))
    print(f"\npaired by seed: Q better={wins}  mu_exec better={loss}  "
          f"tied={len(pair) - wins - loss}  of {len(pair)}")


@app.local_entrypoint()
def report_only():
    report(harvest.remote())


def report(out):
    if not out:
        print("no cells"); return
    import statistics as st
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/population_search.json").write_text(json.dumps(out, default=str))
    imp = [o["improvement"] for o in out]
    print(f"\ncells={len(out)}  QED improvement: median={st.median(imp):+.4f} "
          f"max={max(imp):+.4f}  improved={sum(1 for i in imp if i > 1e-6)}/{len(out)}")
    ev = [e for o in out for e in o["events"]]
    ok = [e for e in ev if e["ok"]]
    print(f"proposals={len(ev)} accepted={len(ok)} ({len(ok)/max(1,len(ev)):.0%})")
    print(f"\n{'scale band':<12} {'proposed':>9} {'succeeded':>10} {'med delta':>10}")
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
        b = [e for e in ev if lo <= e["r_release"] < hi]
        g = [e for e in b if e["ok"]]
        if not b:
            continue
        print(f"{lo}-{hi:<8} {len(b):>9} {len(g):>10} "
              f"{st.median([e.get('delta', 0) for e in g]) if g else 0:>+10.4f}")
    print(f"\n{'interface':<12} {'proposed':>9} {'succeeded':>10}")
    for k in ("pendant", "segment", "multi", "splitting"):
        b = [e for e in ev if e["interface"] == k]
        print(f"{k:<12} {len(b):>9} {sum(1 for e in b if e['ok']):>10}")
    gains = [e for e in ok if e.get("delta", 0) > 0]
    if gains:
        print(f"\nimproving rewrites by scale (the local-to-global check):")
        for e in sorted(gains, key=lambda e: -e["delta"])[:6]:
            print(f"  delta={e['delta']:+.4f} r_release={e['r_release']:.2f} "
                  f"r_coherent={e.get('r_coherent', 0):.2f} "
                  f"{e['interface']}/{e['kind']} steps={e.get('n_steps')}")
