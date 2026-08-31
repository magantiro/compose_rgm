"""OPTION-BASIN ATLAS: an oracle option-selector ceiling on the T4 dev panel.

For every DECLARED option family in `macro_engine.MACRO_FAMILIES` (no invented
options), generate its canonical endpoint basin at a MATCHED cheap-search budget
and record whether any endpoint clears the T4 gates.

    B_o(x) = {canonical endpoints reachable under option o}
    F_o(x) = max_{y in B_o(x)} Phi_z(y)        Phi_z = hard feasibility + margins

Matched budget: every option gets the same cap on canonical endpoints generated
and the same beam/depth. An option that needs more steps than the depth allows
is reported as such rather than silently scored zero -- the ring-CONSTRUCTION
lane is deep by nature and its basin is measured separately in
`pool_ceiling_audit.json` (16,784 endpoints), which this atlas is meant to sit
beside, not replace.

Purpose: decide whether the 20/22 dev cells with no feasible growth endpoint
have SOME other option with a feasible one.

  many cells covered by another option -> selection problem; build the unified
                                          hierarchical proposal controller
  most cells zero under EVERY option    -> option library insufficient; stop
                                          controller work, find missing
                                          transformation classes

Nothing here deploys a controller or a benchmark-tuned macro. CPU only, no docking.
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
app = modal.App("option-basin-atlas")
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


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=3 * 60 * 60,
              retries=0, volumes={ARTIFACT_ROOT: artifact_volume})
def atlas(job: dict) -> dict:
    import sys, os
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, rdFingerprintGenerator, RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.macro_engine import MACRO_FAMILIES, contract_for
    from compose_v4.gates.med_chem_gate import is_valid

    rt = _runtime(); model, system = rt["model"], rt["system"]
    seed = job["seed_smiles"]
    CAP = int(job.get("cap", 600)); BEAM = int(job.get("beam", 14))
    DEPTH = int(job.get("depth", 3))
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    sm = Chem.MolFromSmiles(seed); sfp = gm.GetFingerprint(sm)
    t0 = time.time()

    def metrics(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return {"qed": float(QED.qed(m)), "sa": float(sascorer.calculateScore(m)),
                "sim": float(DataStructs.TanimotoSimilarity(sfp, gm.GetFingerprint(m)))}

    # ONE law per canonical state, SHARED across all 15 option families. They
    # start from the same seed and their basins overlap heavily; re-enumerating
    # per macro was 15x redundant and made the atlas infeasible.
    _lawcache: dict = {}

    def enum(st):
        k = canonical_state_key(st)
        if k not in _lawcache:
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            _lawcache[k] = ([m.executor_rule_name for m in law.marks],
                            [m.action for m in law.marks],
                            np.array([m.probability for m in law.marks], float))
        return _lawcache[k]

    st0 = pad_molecular_graph(smiles_to_molecular_graph(seed), CANONICAL_SLOTS)
    out = {"cell": job["cell"], "seed": seed, "options": {},
           "cap": CAP, "beam": BEAM, "depth": DEPTH}

    for macro, fams_allowed in MACRO_FAMILIES.items():
        allowed = set(fams_allowed)
        try:
            pred = contract_for(macro, seed)
        except Exception:
            pred = None
        basin: dict = {}
        frontier = [st0]
        kernel_calls = 0
        for _d in range(DEPTH):
            nxt = []
            for st in frontier:
                if len(basin) >= CAP:
                    break
                _pre = len(_lawcache)
                fams, acts, probs = enum(st)
                kernel_calls += (len(_lawcache) - _pre)   # count only cache MISSES
                order = sorted(range(len(fams)), key=lambda j: -float(probs[j]))
                taken = 0
                for j in order:
                    if fams[j] not in allowed or taken >= BEAM:
                        continue
                    try:
                        y = system.apply(st, fams[j], acts[j])
                    except Exception:
                        continue
                    smi = canonical_state_key(y)
                    if not smi or smi == canonical_state_key(st) or not is_valid(smi):
                        continue
                    if pred is not None:
                        try:
                            if not pred(smi):
                                continue
                        except Exception:
                            pass
                    taken += 1
                    if smi not in basin:
                        mm = metrics(smi)
                        if mm:
                            basin[smi] = mm
                            nxt.append(y)
                    if len(basin) >= CAP:
                        break
            frontier = nxt[:BEAM]
            if not frontier or len(basin) >= CAP:
                break
        rec = {"n_endpoints": len(basin), "kernel_calls": kernel_calls}
        for delta in (0.4, 0.6):
            feas = [(s, m) for s, m in basin.items()
                    if m["qed"] >= 0.6 and m["sa"] <= 4.0 and m["sim"] >= delta]
            best = max(feas, key=lambda t: t[1]["qed"]) if feas else None
            rec[f"d{delta}"] = {"n_feasible": len(feas), "ceiling": bool(feas),
                                "best": ({"smiles": best[0], **best[1]} if best else None)}
        out["options"][macro] = rec
    out["distinct_states_enumerated"] = len(_lawcache)
    out["sec"] = time.time() - t0
    return out


@app.local_entrypoint()
def main(n_seeds: int = 22, cap: int = 600, beam: int = 14, depth: int = 3, probe: int = 0):
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    for i, s in enumerate(dv):
        s.setdefault("idx", i)
    dev = dv[:probe] if probe else dv[:n_seeds]
    jobs = [{"cell": f"{s['target']}_dev{s['idx']}", "seed_smiles": s["smiles"],
             "cap": cap, "beam": beam, "depth": depth} for s in dev]
    res = list(atlas.map(jobs))
    Path("diagnostics").mkdir(exist_ok=True)
    p = Path("diagnostics/option_basin_atlas.json")
    p.write_text(json.dumps({"cap": cap, "beam": beam, "depth": depth,
                             "results": res}, indent=2))
    opts = list(res[0]["options"].keys())
    print(f"cells={len(res)} wrote={p}\n")
    for delta in (0.4, 0.6):
        print(f"--- delta={delta}: cells with >=1 FEASIBLE endpoint, per option ---")
        for o in opts:
            n = sum(1 for r in res if r["options"][o][f"d{delta}"]["ceiling"])
            ep = sum(r["options"][o]["n_endpoints"] for r in res)
            print(f"   {o:18s} {n:>3d}/{len(res)}   endpoints generated {ep}")
        covered = sum(1 for r in res
                      if any(r["options"][o][f"d{delta}"]["ceiling"] for o in opts))
        print(f"   >>> cells covered by ANY option: {covered}/{len(res)}\n")
