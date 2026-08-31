"""Candidate-pool feasibility CEILING and selection REGRET.

Adjudicates the frozen guided-realization result. For every dev cell it
re-enumerates the SAME deterministic realization pool (same panel, size,
topology, specs, max_realizations), aggregates to unique canonical endpoints,
and asks:

  * does ANY endpoint clear all three T4 gates (QED>=0.6, SA<=4, sim>=delta)?
  * how many do, and with what margins?
  * did the frozen guided selector pick one when one existed?

That splits the null two ways:

  no feasible candidate in the pool   -> the BROAD decision "grow a ring here"
                                         is wrong; no narrow steering fixes it
  feasible candidate exists but missed -> the soft-sigmoid product is the wrong
                                         selector; go constraint-first

Nothing here changes R_theta, the macros, H_z, or the development split. The
frozen experiment is read, not re-run. Also records the cheap-search cost that
the frozen run did not: unique canonical endpoints and realize() call counts.

CPU only. No docking.
"""

from __future__ import annotations

import json
import math
import time
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
app = modal.App("pool-ceiling-audit")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
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


def _sig(x):
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def _Hz(rec, delta, tq=0.05, ts=0.25, tm=0.05):
    q, s, m = rec.get("qed"), rec.get("sa"), rec.get("sim")
    if q is None or m is None:
        return 1e-12
    h = _sig((float(q) - 0.6) / tq) * _sig((float(m) - float(delta)) / tm)
    if s is not None:
        h *= _sig((4.0 - float(s)) / ts)
    return max(h, 1e-12)


def _feasible(rec, delta):
    q, s, m = rec.get("qed"), rec.get("sa"), rec.get("sim")
    return (q is not None and float(q) >= 0.6 and float(s or 9) <= 4.0
            and float(m or 0) >= float(delta))


def _admissible_stoichs(size, max_hetero=3):
    from compose_v4.control.semantic_actions import RING_ELEMENTS, normalize_stoich
    het = [e for e in RING_ELEMENTS if e != "C"]
    out = [normalize_stoich({}, size)]
    for n in range(1, min(max_hetero, size - 1) + 1):
        for combo in combinations_with_replacement(het, n):
            c: dict = {}
            for e in combo:
                c[e] = c.get(e, 0) + 1
            out.append(normalize_stoich(c, size))
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=3 * 60 * 60,
              retries=0, volumes={ARTIFACT_ROOT: artifact_volume})
def ceiling(job: dict) -> dict:
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law)
    from compose_v4.control import semantic_actions as SA
    from compose_v4.control.macro_engine import (
        fusable_edges, predict_fused_product, predict_pendant_product)
    rt = _runtime(); model = rt["model"]
    seed = job["seed_smiles"]; size = job["size"]; topo = job["topology"]
    t0 = time.time()
    st = pad_molecular_graph(smiles_to_molecular_graph(seed), CANONICAL_SLOTS)

    law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
    anchor_mass: dict = {}
    for m in law.marks:
        if m.executor_rule_name != "atom_insert":
            continue
        nb = list(getattr(m.action, "neighbors", ()) or ())
        if len(nb) == 1:
            anchor_mass[int(nb[0][0])] = anchor_mass.get(int(nb[0][0]), 0.0) + float(m.probability)

    stoichs = _admissible_stoichs(size, job.get("max_hetero", 3))
    carbon_only = [s for s in stoichs if len(s) == 1 and s[0][0] == "C"]
    n_realize_calls = 0

    def pool(specs):
        nonlocal n_realize_calls
        out = []
        for sto in specs:
            req = SA.RingRequest(topo, size, sto, "aromatic")
            n_realize_calls += 1
            try:
                reals, _ = SA.realize(req, st, seed,
                                      predict_fn=predict_pendant_product,
                                      fused_fn=predict_fused_product,
                                      edges_fn=fusable_edges,
                                      max_realizations=int(job.get("max_real", 64)))
            except Exception:
                continue
            for r in reals:
                r = dict(r); r["_stoich"] = [list(x) for x in sto]
                r["_prior"] = anchor_mass.get(int(r.get("anchor", -1)), 1e-9)
                out.append(r)
        return out

    out = {"cell": job["cell"], "seed": seed, "arms": {}}
    for name, specs in (("narrow_frozen", carbon_only), ("open", stoichs)):
        reals = pool(specs)
        groups = SA.aggregate_canonical(reals, [r["_prior"] for r in reals])
        recs = [g["record"] for g in groups]
        arm = {"n_realizations": len(reals), "n_unique_endpoints": len(groups)}
        for delta in (0.4, 0.6):
            feas = [r for r in recs if _feasible(r, delta)]
            best_h = max((_Hz(r, delta) for r in recs), default=0.0)
            # margins of the single best feasible candidate, by H_z
            bf = max(feas, key=lambda r: _Hz(r, delta)) if feas else None
            arm[f"d{delta}"] = {
                "n_feasible": len(feas),
                "ceiling_feasible": bool(feas),
                "best_Hz_in_pool": best_h,
                "best_feasible": ({"smiles": bf.get("smiles"), "qed": bf.get("qed"),
                                   "sa": bf.get("sa"), "sim": bf.get("sim"),
                                   "stoich": bf.get("_stoich"),
                                   "qed_margin": float(bf["qed"]) - 0.6,
                                   "sa_margin": 4.0 - float(bf.get("sa") or 9),
                                   "sim_margin": float(bf.get("sim") or 0) - delta}
                                  if bf else None),
            }
        out["arms"][name] = arm
    out["n_realize_calls"] = n_realize_calls
    out["n_kernel_enumerations"] = 1
    out["sec"] = time.time() - t0
    return out


@app.local_entrypoint()
def main(n_seeds: int = 22, size: int = 6, probe: int = 0):
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    for i, s in enumerate(dv):
        s.setdefault("idx", i)
    dev = dv[:probe] if probe else dv[:n_seeds]
    jobs = [{"cell": f"{s['target']}_dev{s['idx']}", "seed_smiles": s["smiles"],
             "size": size, "topology": "pendant"} for s in dev]
    res = list(ceiling.map(jobs))
    Path("diagnostics").mkdir(exist_ok=True)
    p = Path("diagnostics/pool_ceiling_audit.json")
    p.write_text(json.dumps({"results": res}, indent=2))
    print(f"cells={len(res)} wrote={p}")
    for delta in (0.4, 0.6):
        for arm in ("narrow_frozen", "open"):
            n = sum(1 for r in res if r["arms"][arm][f"d{delta}"]["ceiling_feasible"])
            tot = sum(r["arms"][arm][f"d{delta}"]["n_feasible"] for r in res)
            print(f"  delta={delta} {arm:14s} cells with >=1 feasible candidate: {n}/{len(res)}"
                  f"   total feasible endpoints: {tot}")
