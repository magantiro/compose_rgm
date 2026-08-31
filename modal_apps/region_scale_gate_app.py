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
BANDS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8))
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


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=3 * 60 * 60,
              retries=0, volumes={ARTIFACT_ROOT: artifact_volume})
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

    rt = _runtime(); model, system = rt["model"], rt["system"]
    smi = job["seed_smiles"]
    rng = np.random.default_rng(int(job.get("rng", 5)))
    st0 = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
    n_real = len(smiles_to_molecular_graph(smi).atom_types)
    cache: dict = {}

    def enum_fn(st):
        k = canonical_state_key(st)
        if k not in cache:
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            cache[k] = ([m.executor_rule_name for m in law.marks],
                        [m.action for m in law.marks],
                        np.array([m.probability for m in law.marks], float))
        return cache[k]

    def apply_fn(st, j):
        fams, acts, _ = enum_fn(st)
        try:
            return system.apply(st, fams[j], acts[j])
        except Exception:
            return None

    regions = enumerate_regions(smi)
    per_band = int(job.get("per_band", 4))
    rows, t0 = [], time.time()
    for lo, hi in BANDS:
        pool = [r for r in regions if lo <= r.released_fraction < hi
                and r.interface in ("pendant", "segment", "splitting")]
        if not pool:
            continue
        pick = rng.choice(len(pool), size=min(per_band, len(pool)), replace=False)
        for pi in pick:
            reg = pool[int(pi)]
            ctx = RR.context_from_region(reg)
            lin0 = RR.Lineage.initial(range(n_real))
            old_ids = frozenset(lin0.id_of[s] for s in reg.atoms if s in lin0.id_of)
            pot = (lambda s_, c_, l_: RR.connectivity_potential(s_, c_, l_, old_ids))
            sched = ([("prune_old", max(1, reg.size)), ("grow_new", 2)]
                     if reg.interface == "pendant"
                     else [("grow_new", "until_handoff"),
                           ("prune_old", max(1, reg.size))])
            p = RR.propose(enum_fn, apply_fn, ctx, st0, schedule=sched,
                           rng=np.random.default_rng(int(rng.integers(0, 10**6))),
                           lineage=lin0, original_region_ids=old_ids,
                           max_handoff_steps=int(job.get("max_handoff", 16)),
                           potential_fn=(None if reg.interface == "pendant" else pot),
                           beta=float(job.get("beta", 6.0)),
                           epsilon=float(job.get("epsilon", 0.1)))
            row = {"band": f"{lo}-{hi}", "interface": reg.interface,
                   "r_release": reg.released_fraction, "region_size": reg.size,
                   "status": p.status, "stage": p.stage,
                   "n_steps": len(p.steps),
                   "first_handoff_step": p.first_handoff_step,
                   "logq": p.conditional_path_logq}
            if p.status == "OK" and p.endpoint is not None:
                d = GG.structural_displacement(st0, p.endpoint, lin0, p.lineage)
                surviving = [i for i in old_ids if i in p.lineage.slot_of]
                row.update({
                    "r_change": d["changed_fraction"] + (d["n_inserted"] + d["n_deleted"]) / n_real,
                    "r_coherent": d["largest_changed_fraction"],
                    "coherence": d["coherence"],
                    "d_cycle_rank": d["d_cycle_rank"],
                    "d_ring_systems": d["d_ring_systems"],
                    "d_heavy": d["d_heavy"],
                    "old_region_removed": len(surviving) == 0,
                    "endpoint": canonical_state_key(p.endpoint),
                    "endpoint_valid": bool(is_valid(canonical_state_key(p.endpoint))),
                })
            rows.append(row)
    return {"cell": job["cell"], "seed": smi, "rows": rows, "sec": time.time() - t0}


@app.local_entrypoint()
def main(n_seeds: int = 22, per_band: int = 4):
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    for i, s in enumerate(dv):
        s.setdefault("idx", i)
    jobs = [{"cell": f"{s['target']}_dev{s['idx']}", "seed_smiles": s["smiles"],
             "per_band": per_band, "rng": 5 + s["idx"]} for s in dv[:n_seeds]]
    res = list(scale_cell.map(jobs))
    Path("diagnostics").mkdir(exist_ok=True)
    p = Path("diagnostics/region_scale_gate.json")
    p.write_text(json.dumps({"results": res}, indent=2))
    rows = [r for c in res for r in c["rows"]]
    print(f"cells={len(res)} attempts={len(rows)} wrote={p}\n")
    print(f"{'band':10s} {'n':>4s} {'OK':>4s} {'rate':>6s} "
          f"{'med r_change':>13s} {'med r_coherent':>15s} {'max r_coherent':>15s}")
    for lo, hi in BANDS:
        b = [r for r in rows if r["band"] == f"{lo}-{hi}"]
        ok = [r for r in b if r["status"] == "OK"]
        med = lambda k: (sorted(x[k] for x in ok)[len(ok) // 2] if ok else float("nan"))
        print(f"{lo}-{hi:<6} {len(b):>4d} {len(ok):>4d} "
              f"{(len(ok)/len(b) if b else 0):>6.0%} "
              f"{med('r_change'):>13.2f} {med('r_coherent'):>15.2f} "
              f"{max((x['r_coherent'] for x in ok), default=float('nan')):>15.2f}")
    from collections import Counter
    print("\nfailure reasons:", dict(Counter(r["stage"] for r in rows if r["status"] != "OK")))
