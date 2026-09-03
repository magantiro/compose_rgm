"""Scale gate: does variable-scope rewriting produce broad COHERENT change?

The central question, and it is not "did we select a big region":

    does increasing scope genuinely unlock increasingly large coherent moves,
    including a substantial tail of half-scaffold-scale transformations?

For every attempt, three fractions of the ORIGINAL molecule:

    r_release   |M| / |x|                       what we asked to rewrite
    r_change    lineage-changed originals plus inserted/deleted, over |x|
    r_coherent  largest connected changed region / |x|

r_release is not required to equal r_change -- that would be an artificial
demand. What must show up is the RELATIONSHIP: larger scope producing larger
coherent change, with compile success stratified by scope so a method that makes
huge moves 2% of the time and UNSAT 98% is not mistaken for a global proposal.

Regions are sampled stratified by released-fraction band and interface class,
using the SAME frozen kernel as the sentinels: beta, W, epsilon and horizon are
not retuned here.

CPU only. No docking, no benchmark objective.
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
app = modal.App("region-scale-gate")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
MEM_MIB = int(3 * 1024)
BANDS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))
OUT_DIR = "/artifacts/region_committor"
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
def scale_cell(job: dict) -> dict:
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    import numpy as np
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.region import enumerate_regions
    from compose_v4.control import region_rewrite as RR
    from compose_v4.control import graph_geometry as GG
    from compose_v4.gates.med_chem_gate import is_valid
    import torch

    rt = _runtime(); model, system = rt["model"], rt["system"]
    smi = job["seed_smiles"]
    rng = np.random.default_rng(int(job.get("rng", 5)))
    st0 = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
    n_real = len(smiles_to_molecular_graph(smi).atom_types)
    cache: dict = {}
    calls = {"n": 0}
    kappa = float(job.get("kappa", 1.0))       # FROZEN from the qualification
    n_particles = int(job.get("particles", 8))

    def slot_key(state):
        # NOT canonical_state_key: actions carry slot coordinates, and two
        # canonically equal states with different layouts share a canonical key,
        # so a hit returns another state's actions.
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

    memo: dict = {}

    def apply_fn(st, j_):
        k = (slot_key(st), int(j_))
        if k in memo:
            return memo[k]
        calls["n"] += 1
        fams, acts, _ = enum_fn(st)
        try:
            y = system.apply(st, fams[j_], acts[j_])
        except Exception:
            y = None
        memo[k] = y
        return y

    # the qualified structural execution controller, frozen
    artifact_volume.reload()
    ck = torch.load(f"{OUT_DIR}/committor_bellman_v1.pt", map_location="cpu")
    n_feat = int(ck["n_features"])
    net = torch.nn.Sequential(torch.nn.Linear(n_feat, 48), torch.nn.ReLU(),
                              torch.nn.Linear(48, 1))
    net.load_state_dict(ck["state_dict"]); net.eval()
    torch.set_grad_enabled(False)

    regions = enumerate_regions(smi)
    per_band = int(job.get("per_band", 4))
    only_band = job.get("only_band")     # fan out per REGION, not per molecule:
    rows, t0 = [], time.time()           # attempts inside a cell are sequential,
    for lo, hi in BANDS:                 # so a per-molecule job serialises ~120
        if only_band is not None and f"{lo}-{hi}" != only_band:
            continue
        # every interface class, including multi -- the previous filter dropped
        # 42 of 108 regions per molecule, which a COVERAGE question cannot do
        pool = [r for r in regions if lo <= r.released_fraction < hi]
        if not pool:
            continue
        pick = rng.choice(len(pool), size=min(per_band, len(pool)), replace=False)
        if job.get("only_index") is not None:
            k_ = int(job["only_index"])
            if k_ >= len(pick):
                continue
            pick = [pick[k_]]
        for pi in pick:
            reg = pool[int(pi)]
            ctx = RR.context_from_region(reg)
            lin0 = RR.Lineage.initial(range(n_real))
            old_ids = frozenset(lin0.id_of[s] for s in reg.atoms if s in lin0.id_of)
            hcache: dict = {}

            def h_model(y, c, lin, budget, kind, _o=old_ids, _hc=hcache):
                key = (slot_key(y), int(budget), kind)
                if key not in _hc:
                    f = RR.features_from_shared(
                        RR.structural_features_shared(y, c, lin, _o),
                        int(budget), kind)
                    _hc[key] = float(torch.sigmoid(
                        net(torch.tensor([f], dtype=torch.float32))).item())
                return _hc[key]

            sched = ([("prune_old", max(1, reg.size)), ("grow_new", 2)]
                     if reg.interface == "pendant"
                     else [("grow_new", "until_handoff"),
                           ("prune_old", max(1, reg.size))])
            best = None
            n_ok = 0
            t_reg = time.time()
            calls_before = calls["n"]   # BEFORE the particles run: capturing it
                                        # afterwards made every cost read zero
            print(f"[region] {job['cell']} band={lo}-{hi} {reg.interface}/{reg.kind} "
                  f"size={reg.size} r_rel={reg.released_fraction:.2f} "
                  f"sched={sched}", flush=True)
            for particle in range(n_particles):
                p = RR.propose(enum_fn, apply_fn, ctx, st0, schedule=sched,
                               rng=np.random.default_rng(9000 + particle),
                               lineage=lin0, original_region_ids=old_ids,
                               max_handoff_steps=int(job.get("max_handoff", 16)),
                               h_model=h_model, tilt="kl", kappa=kappa,
                               epsilon=float(job.get("epsilon", 0.1)))
                print(f"[particle] {job['cell']} p{particle} status={p.status} "
                      f"stage={p.stage} steps={len(p.steps)} calls={calls['n']} "
                      f"{time.time() - t_reg:.0f}s", flush=True)
                if p.status != "OK" or p.endpoint is None:
                    continue
                if not is_valid(canonical_state_key(p.endpoint)):
                    continue
                n_ok += 1
                d = GG.structural_displacement(st0, p.endpoint, lin0, p.lineage)
                r_coh = d["largest_changed_fraction"]
                if best is None or r_coh > best[0]:
                    best = (r_coh, p, d)
            row = {"band": f"{lo}-{hi}", "interface": reg.interface,
                   "kind": reg.kind, "generator": reg.generator,
                   "region_has_ring": bool(reg.region_has_ring),
                   "r_release": reg.released_fraction, "region_size": reg.size,
                   "n_particles": n_particles, "n_ok": n_ok,
                   "any_ok": n_ok > 0,
                   "calls_region": calls["n"] - calls_before,
                   "sec_region": time.time() - t_reg}
            if best is not None:
                _, p, d = best
                surviving = [i for i in old_ids if i in p.lineage.slot_of]
                row.update({
                    "status": p.status, "stage": p.stage, "n_steps": len(p.steps),
                    "first_handoff_step": p.first_handoff_step,
                    "logq": p.conditional_path_logq,
                    "r_change": d["changed_fraction"]
                                + (d["n_inserted"] + d["n_deleted"]) / n_real,
                    "r_coherent": d["largest_changed_fraction"],
                    "coherence": d["coherence"],
                    "d_cycle_rank": d["d_cycle_rank"],
                    "d_ring_systems": d["d_ring_systems"],
                    "d_heavy": d["d_heavy"],
                    "old_region_removed": len(surviving) == 0,
                    "endpoint": canonical_state_key(p.endpoint),
                })
            rows.append(row)
            print(f"[scale] {job['cell']} band={lo}-{hi} {reg.interface}/{reg.kind} "
                  f"r_rel={reg.released_fraction:.2f} ok={n_ok}/{n_particles} "
                  f"r_coh={row.get('r_coherent', 0):.2f} {time.time() - t0:.0f}s",
                  flush=True)
    d_out = Path(OUT_DIR) / "scale_gate"
    d_out.mkdir(parents=True, exist_ok=True)
    (d_out / f"{job['cell']}.json").write_text(json.dumps(
        {"cell": job["cell"], "seed": smi, "rows": rows,
         "sec": time.time() - t0}))
    artifact_volume.commit()
    return {"cell": job["cell"], "seed": smi, "rows": rows, "sec": time.time() - t0}


@app.local_entrypoint()
def main(n_seeds: int = 12, per_band: int = 4, particles: int = 1,
         kappa: float = 1.0):
    """General local->global gate: coverage and structural scale.

    Two questions that must not be blended:

      COVERAGE   which regions and scales admit a useful executable rewrite at
                 all. Regions are sampled by released-fraction band across EVERY
                 interface class, NOT selected because a rewrite is known to
                 exist -- that selection was correct for qualifying execution
                 efficiency and is exactly wrong here.

      SCALE      does releasing more actually buy more coherent change?
                 r_release -> r_coherent is the local-to-global claim; releasing
                 60% and changing one atom would falsify it.

    ONE particle per region by default, fanned out per region. The endpoint is
    coverage of structural scales and actual coherent displacement, not precise
    per-region success rates, so eight expensive trajectories per region buys
    resolution nobody asked for. Bands that come back empty get more particles
    from `topup`, which is the only place extra compute is spent.

    Cost is part of the result, not overhead: a 0.8-released region legitimately
    needs 20+ primitive steps, and the scope -> steps -> calls -> seconds curve
    is what tells the population controller to make many cheap local proposals
    and few expensive global ones.

    The controller is frozen: the qualified committor with the KL trust region
    at kappa=1.0. Nothing here is tuned.
    """
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    for i, s_ in enumerate(dv):
        s_.setdefault("idx", i)
    jobs = [{"cell": f"dev{s_['idx']}_b{bi}_r{ri}", "seed_smiles": s_["smiles"],
             "per_band": per_band, "particles": particles, "kappa": kappa,
             "only_band": f"{lo}-{hi}", "only_index": ri,
             "rng": 5 + s_["idx"]}
            for s_ in dv[:n_seeds]
            for bi, (lo, hi) in enumerate(BANDS)
            for ri in range(per_band)]
    print(f"jobs={len(jobs)} (one region each) molecules={min(n_seeds, len(dv))} "
          f"bands={len(BANDS)} per_band={per_band} particles={particles} "
          f"kappa={kappa} (frozen)")
    res = [r for r in scale_cell.map(jobs) if r]
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/region_scale_gate.json").write_text(
        json.dumps({"results": res}, indent=2))
    rows = [r for c in res for r in c["rows"]]
    ok = [r for r in rows if r.get("any_ok")]
    print(f"\ncells={len(res)} regions attempted={len(rows)} with a valid rewrite={len(ok)}")

    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else float("nan")

    print(f"\n--- COVERAGE and SCALE by released-fraction band ---")
    print(f"{'band':<10} {'n':>4} {'any_ok':>7} {'rate':>6} "
          f"{'med r_change':>13} {'med r_coh':>10} {'max r_coh':>10}")
    for lo, hi in BANDS:
        b = [r for r in rows if r["band"] == f"{lo}-{hi}"]
        g = [r for r in b if r.get("any_ok")]
        print(f"{lo}-{hi:<6} {len(b):>4} {len(g):>7} "
              f"{(len(g)/len(b) if b else 0):>6.0%} "
              f"{med([x.get('r_change', 0) for x in g]):>13.2f} "
              f"{med([x.get('r_coherent', 0) for x in g]):>10.2f} "
              f"{max([x.get('r_coherent', 0) for x in g], default=0):>10.2f}")

    print(f"\n--- COST by scope (what a proposal of this scale costs) ---")
    print(f"{'band':<10} {'med steps':>10} {'med calls':>10} {'med sec':>9} "
          f"{'max sec':>9}")
    for lo, hi in BANDS:
        b = [r for r in rows if r["band"] == f"{lo}-{hi}"]
        if not b:
            continue
        print(f"{lo}-{hi:<6} {med([x.get('n_steps', 0) for x in b]):>10.0f} "
              f"{med([x.get('calls_region', 0) for x in b]):>10.0f} "
              f"{med([x.get('sec_region', 0) for x in b]):>9.0f} "
              f"{max([x.get('sec_region', 0) for x in b], default=0):>9.0f}")

    print(f"\n--- COVERAGE by interface class ---")
    for cls in ("pendant", "segment", "splitting", "multi"):
        b = [r for r in rows if r["interface"] == cls]
        g = [r for r in b if r.get("any_ok")]
        print(f"  {cls:<10} {len(g):>3}/{len(b):<4} "
              f"({(len(g)/len(b) if b else 0):>4.0%})  "
              f"med r_coh={med([x.get('r_coherent', 0) for x in g]):.2f}")

    print(f"\n--- COVERAGE by region kind ---")
    for k in sorted({r["kind"] for r in rows}):
        b = [r for r in rows if r["kind"] == k]
        g = [r for r in b if r.get("any_ok")]
        print(f"  {k:<12} {len(g):>3}/{len(b):<4} "
              f"({(len(g)/len(b) if b else 0):>4.0%})")

    big = [r for r in ok if r.get("r_coherent", 0) >= 0.3]
    print(f"\nscaffold-scale outcomes (r_coherent >= 0.30): {len(big)}")
    for r in sorted(big, key=lambda r: -r["r_coherent"])[:8]:
        print(f"  r_release={r['r_release']:.2f} -> r_coherent={r['r_coherent']:.2f} "
              f"coherence={r.get('coherence', 0):.2f} "
              f"d_rings={r.get('d_ring_systems')} {r['interface']}/{r['kind']}")
    from collections import Counter
    print("\nfailure stages:",
          dict(Counter(r.get("stage") for r in rows if not r.get("any_ok"))))
    empty = [f"{lo}-{hi}" for lo, hi in BANDS
             if [r for r in rows if r["band"] == f"{lo}-{hi}"]
             and not [r for r in rows if r["band"] == f"{lo}-{hi}" and r.get("any_ok")]]
    if empty:
        print(f"\nBANDS WITH ZERO SUCCESSES: {empty}")
        print("  -> these are the ONLY bands that get more particles:")
        for b in empty:
            print(f"     modal run modal_apps/region_scale_gate_app.py::topup "
                  f"--band '{b}' --particles 4")
    else:
        print("\nevery scope band produced at least one coherent rewrite")


@app.local_entrypoint()
def topup(band: str, particles: int = 4, n_seeds: int = 12, per_band: int = 4,
          kappa: float = 1.0):
    """Spend extra particles ONLY on a scope band that came back empty.

    A band with zero successes at one particle is ambiguous -- the mechanism may
    fail there, or one sample may simply have missed. This adjudicates that band
    and nothing else, which is where the extra compute belongs.
    """
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    for i, s_ in enumerate(dv):
        s_.setdefault("idx", i)
    jobs = [{"cell": f"topup_dev{s_['idx']}_r{ri}", "seed_smiles": s_["smiles"],
             "per_band": per_band, "particles": particles, "kappa": kappa,
             "only_band": band, "only_index": ri, "rng": 5 + s_["idx"]}
            for s_ in dv[:n_seeds] for ri in range(per_band)]
    print(f"top-up band={band} jobs={len(jobs)} particles={particles}")
    res = [r for r in scale_cell.map(jobs) if r]
    rows = [r for c in res for r in c["rows"]]
    ok = [r for r in rows if r.get("any_ok")]
    print(f"regions={len(rows)} with a valid rewrite={len(ok)}")
    if ok:
        v = sorted(x.get("r_coherent", 0) for x in ok)
        print(f"  r_coherent median={v[len(v)//2]:.2f} max={max(v):.2f}")
    Path("diagnostics").mkdir(exist_ok=True)
    Path(f"diagnostics/scale_gate_topup_{band.replace('.','p')}.json").write_text(
        json.dumps({"band": band, "results": res}, indent=2))


@app.function(image=image, cpu=(1.0, 1.0), memory=int(2 * 1024), timeout=15 * 60,
              volumes={ARTIFACT_ROOT: artifact_volume})
def harvest_cells() -> list:
    """Persisted scale-gate cells, so a dead client never loses measurements."""
    artifact_volume.reload()
    d = Path(OUT_DIR) / "scale_gate"
    return [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))] if d.exists() else []


@app.local_entrypoint()
def report():
    """Read whatever cells have persisted and print coverage, scale and cost."""
    cells = harvest_cells.remote()
    rows = [r for c in cells for r in c["rows"]]
    print(f"cells={len(cells)} regions={len(rows)}")
    if not rows:
        return

    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else 0.0

    print(f"\n{'band':<10} {'n':>3} {'ok':>3} {'med steps':>10} {'med calls':>10} "
          f"{'med sec':>8} {'med r_coh':>10} {'max r_coh':>10}")
    for lo, hi in BANDS:
        b = [r for r in rows if r["band"] == f"{lo}-{hi}"]
        if not b:
            continue
        g = [r for r in b if r.get("any_ok")]
        print(f"{lo}-{hi:<6} {len(b):>3} {len(g):>3} "
              f"{med([x.get('n_steps', 0) for x in g]):>10.0f} "
              f"{med([x.get('calls_region', 0) for x in b]):>10.0f} "
              f"{med([x.get('sec_region', 0) for x in b]):>8.0f} "
              f"{med([x.get('r_coherent', 0) for x in g]):>10.2f} "
              f"{max([x.get('r_coherent', 0) for x in g], default=0):>10.2f}")
    for r in rows:
        print(f"  {r['band']:<9} {r['interface']:<9}/{r['kind']:<11} "
              f"size={r['region_size']:<3} r_rel={r['r_release']:.2f} "
              f"ok={r['n_ok']}/{r['n_particles']} steps={r.get('n_steps','-')} "
              f"calls={r.get('calls_region','-')} sec={r.get('sec_region',0):.0f} "
              f"r_coh={r.get('r_coherent','-')}")
