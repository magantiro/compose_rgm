"""P(xi) ∝ R_theta(xi) · H_z(y_xi, b) -- task-aware realization control.

Tests the controller we actually intend to use. The realization SPACE is held
identical across arms; only the selection distribution changes.

  A  narrow_frozen   all-carbon spec only,          value_fn=None
  B  free_unguided   all admissible stoichiometries, value_fn=None
  C  free_guided     all admissible stoichiometries, value_fn=H_z   <-- the test

B reproduces the frozen finding (composition exposure without task awareness).
C adds H_z. A/B/C draw the SAME number of endpoints from the SAME pooled
realization set, so nothing is given more search for free.

H_z is the T4 terminal desirability g_z, computed from the cheap endpoint state
that `realize()` already returns -- no training, no oracle, no GPU:

    H_z(y) = sigma((QED-0.6)/tq) * sigma((4-SA)/ts) * sigma((sim-delta)/tm)

Selection uses `select_endpoint`, which applies H ONCE PER CANONICAL ENDPOINT
after fiber aggregation; applying it per realization would reintroduce the
multiplicity the aggregation removes.

The R_theta prior is the mass of atom_insert marks at each anchor -- deliberately
NOT `rtheta_semantic_prior`, which was measured to be mis-specified (it scored
the SIZE of a spec's element set, i.e. permissiveness, not plausibility).

CPU only. No docking. Development cells only. Nothing here is a benchmark number.
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
app = modal.App("guided-realization-arm")
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


def _make_Hz(delta, tq=0.05, ts=0.25, tm=0.05):
    """T4 terminal desirability from the record realize() already returns."""
    def H(rec):
        q = rec.get("qed"); s = rec.get("sa"); m = rec.get("sim")
        if q is None or m is None:
            return 1e-12
        h = _sig((float(q) - 0.6) / tq) * _sig((float(m) - float(delta)) / tm)
        if s is not None:
            h *= _sig((4.0 - float(s)) / ts)
        return max(h, 1e-12)
    return H


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
def guided_seed(job: dict) -> dict:
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    import numpy as np
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law)
    from compose_v4.control import semantic_actions as SA
    from compose_v4.control.macro_engine import (
        fusable_edges, predict_fused_product, predict_pendant_product)

    rt = _runtime(); model = rt["model"]
    seed = job["seed_smiles"]; size = job["size"]; topo = job["topology"]
    K = int(job["k_proposals"]); delta = float(job["delta"])
    st = pad_molecular_graph(smiles_to_molecular_graph(seed), CANONICAL_SLOTS)
    t0 = time.time()

    # ---- R_theta prior over ANCHORS: mass of atom_insert marks at each site --
    law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
    anchor_mass: dict = {}
    for m in law.marks:
        if m.executor_rule_name != "atom_insert":
            continue
        nb = list(getattr(m.action, "neighbors", ()) or ())
        if len(nb) != 1:
            continue
        anchor_mass[int(nb[0][0])] = anchor_mass.get(int(nb[0][0]), 0.0) + float(m.probability)

    stoichs = _admissible_stoichs(size, job.get("max_hetero", 3))
    carbon_only = [s for s in stoichs if len(s) == 1 and s[0][0] == "C"]

    def pool(specs):
        out = []
        for sto in specs:
            req = SA.RingRequest(topo, size, sto, "aromatic")
            try:
                reals, _why = SA.realize(req, st, seed,
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

    Hz = _make_Hz(delta)
    arms = {}
    for name, specs, vfn in (("narrow_frozen", carbon_only, None),
                             ("free_unguided", stoichs, None),
                             ("free_guided", stoichs, Hz)):
        reals = pool(specs)
        picks = []
        if reals:
            rng = np.random.default_rng(int(job.get("seed_rng", 11)))
            for _ in range(K):
                g, p, groups = SA.select_endpoint(
                    reals, prior=[r["_prior"] for r in reals],
                    value_fn=vfn, rng=rng)
                if g is None:
                    break
                rec = g["record"]
                picks.append({"smiles": rec.get("smiles"), "qed": rec.get("qed"),
                              "sa": rec.get("sa"), "sim": rec.get("sim"),
                              "stoich": rec.get("_stoich")})
        arms[name] = {"n_realizations_pooled": len(reals), "picks": picks}
    return {"cell": job["cell"], "seed": seed, "delta": delta,
            "n_specs": len(stoichs), "arms": arms, "sec": time.time() - t0}


@app.local_entrypoint()
def main(n_seeds: int = 6, k: int = 8, size: int = 6, delta: float = 0.4, probe: int = 0):
    # DECLARED development panel: DUD-E actives, canonical-SMILES disjoint from
    # the 15 GenMol benchmark seeds (0 overlap verified in the artifact), 2 per
    # (target, QED stratum). Using benchmark seeds for development would leak.
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    for i, s in enumerate(dv):
        s.setdefault("idx", i)
    dev = dv[:probe] if probe else dv[:n_seeds]
    jobs = [{"cell": f"{s['target']}_dev{s['idx']}", "seed_smiles": s["smiles"],
             "size": size, "topology": "pendant", "k_proposals": k,
             "delta": delta, "seed_rng": 11 + s["idx"]} for s in dev]
    res = list(guided_seed.map(jobs))
    Path("diagnostics").mkdir(exist_ok=True)
    p = Path(f"diagnostics/guided_realization_delta{delta}.json")
    p.write_text(json.dumps({"k": k, "delta": delta, "results": res}, indent=2))
    print(f"cells={len(res)} delta={delta} wrote={p}")
    for r in res:
        print(f"  {r['cell']:12s} specs={r['n_specs']}")
        for a, v in r["arms"].items():
            pk = v["picks"]
            feas = sum(1 for x in pk if x["qed"] and x["qed"] >= 0.6
                       and (x["sa"] or 9) <= 4 and (x["sim"] or 0) >= delta)
            het = sum(1 for x in pk if x["stoich"] and len(x["stoich"]) > 1)
            print(f"     {a:16s} pooled={v['n_realizations_pooled']:>4d} "
                  f"picks={len(pk):>2d} feasible={feas:>2d} hetero={het:>2d}")
