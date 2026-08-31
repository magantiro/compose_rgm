"""MULTISCALE REACHABILITY / UTILITY ATLAS on the declared T4 dev panel.

Answers two questions at MATCHED total primitive-edit budget B:

  COVERAGE  how far, structurally, can each search lane move?
  UTILITY   does anything in that reachable set clear the T4 gates?

Scale is MEASURED from the effect on molecular state, never inferred from an
operator's name:

  d0 = sigma0 changed   (ring-system topology: #systems, cycle rank, spiro,
                         bridgehead, rings-per-system)
  d1 = sigma1 changed   (+ ring sizes, aromatic state, size bin)
  plus dHeavy, dRings, Murcko-scaffold change, 1 - Tanimoto

Lanes, all given the SAME primitive-edit budget B:

  k=1   one declared macro, B edits
  k=2   two declared macros, B/2 edits each      } sequences SAMPLED from the
  k=3   three declared macros, B/3 edits each    } declared list, not hand-picked
  generic   unrestricted R_theta rollout of B edits with structural-exit
            encouragement -- a deliberately open-ended lane so the atlas can
            find useful chemistry nobody thought to name

Three distinguishable outcomes:
  existing options COMPOSE to useful chemistry -> no generic exit machinery needed
  generic lane reaches useful regions they miss -> global arm justified
  neither reaches useful regions               -> the OPERATOR VOCABULARY is the
                                                  gap; controller work will not fix it

CPU only. No docking. Diagnostic only -- deploys nothing.
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
app = modal.App("multiscale-atlas")
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


def _sigma(smi):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdMD
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import Counter
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    rings = [set(r) for r in m.GetRingInfo().AtomRings()]
    sysm = []
    for r in rings:
        merged, rest = [r], []
        for s in sysm:
            (merged if s & r else rest).append(s)
        sysm = rest + [set().union(*merged)]
    rps = sorted(sum(1 for r in rings if r <= s) for s in sysm)
    s0 = (len(sysm), m.GetRingInfo().NumRings(), rdMD.CalcNumSpiroAtoms(m),
          rdMD.CalcNumBridgeheadAtoms(m), tuple(rps))
    rdesc = sorted((len(r), all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r)) for r in rings)
    s1 = s0 + (tuple(rdesc), m.GetNumHeavyAtoms() // 5)
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        scaf = ""
    return {"s0": s0, "s1": s1, "scaffold": scaf,
            "heavy": m.GetNumHeavyAtoms(), "rings": m.GetRingInfo().NumRings()}


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
    from compose_v4.control.macro_engine import MACRO_FAMILIES
    from compose_v4.gates.med_chem_gate import is_valid

    rt = _runtime(); model, system = rt["model"], rt["system"]
    seed = job["seed_smiles"]; B = int(job.get("budget", 6))
    R = int(job.get("rollouts", 6)); STATE_CAP = int(job.get("state_cap", 220))
    rng = np.random.default_rng(int(job.get("rng", 3)))
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    sm = Chem.MolFromSmiles(seed); sfp = gm.GetFingerprint(sm)
    sig0 = _sigma(seed)
    t0 = time.time()
    _law: dict = {}

    def enum(st):
        k = canonical_state_key(st)
        if k not in _law:
            if len(_law) >= STATE_CAP:
                return None
            lw = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            _law[k] = ([m.executor_rule_name for m in lw.marks],
                       [m.action for m in lw.marks],
                       np.array([m.probability for m in lw.marks], float))
        return _law[k]

    def step(st, allowed, exit_bias=False):
        """One primitive edit restricted to `allowed` families, R_theta-sampled."""
        e = enum(st)
        if e is None:
            return None
        fams, acts, probs = e
        idx = [j for j, f in enumerate(fams) if f in allowed]
        if not idx:
            return None
        w = np.array([probs[j] for j in idx], float)
        if exit_bias:
            w = w ** 0.5                      # flatten: encourage leaving the mode
        w = w / w.sum()
        for _ in range(4):
            j = int(rng.choice(idx, p=w))
            try:
                y = system.apply(st, fams[j], acts[j])
            except Exception:
                continue
            smi = canonical_state_key(y)
            if smi and smi != canonical_state_key(st) and is_valid(smi):
                return y
        return None

    def run_lane(seq, exit_bias=False):
        """`seq` is a list of allowed-family sets; budget split evenly."""
        per = max(1, B // max(1, len(seq)))
        endpoints = {}
        for _r in range(R):
            st = pad_molecular_graph(smiles_to_molecular_graph(seed), CANONICAL_SLOTS)
            ok = True
            for allowed in seq:
                for _e in range(per):
                    nxt = step(st, allowed, exit_bias)
                    if nxt is None:
                        ok = False; break
                    st = nxt
                if not ok:
                    break
            smi = canonical_state_key(st)
            if smi and smi != seed:
                endpoints[smi] = True
        return list(endpoints)

    ALLF = set().union(*MACRO_FAMILIES.values())
    macros = list(MACRO_FAMILIES)
    lanes: dict = {}
    for mname in macros:                                   # k = 1
        lanes[f"k1:{mname}"] = [set(MACRO_FAMILIES[mname])]
    for k in (2, 3):                                       # k = 2,3 SAMPLED
        for i in range(int(job.get("n_seq", 8))):
            pick = [macros[int(x)] for x in rng.integers(0, len(macros), size=k)]
            lanes[f"k{k}:" + ">".join(pick)] = [set(MACRO_FAMILIES[p]) for p in pick]
    lanes["generic_exit"] = [ALLF]                          # open-ended lane

    out = {"cell": job["cell"], "seed": seed, "budget_primitive_edits": B,
           "rollouts_per_lane": R, "lanes": {}}
    for name, seq in lanes.items():
        eps = run_lane(seq, exit_bias=name.startswith("generic"))
        recs = []
        for smi in eps:
            sg = _sigma(smi)
            if not sg:
                continue
            m = Chem.MolFromSmiles(smi)
            recs.append({
                "smiles": smi,
                "d0": int(sg["s0"] != sig0["s0"]), "d1": int(sg["s1"] != sig0["s1"]),
                "d_scaffold": int(sg["scaffold"] != sig0["scaffold"]),
                "dHeavy": sg["heavy"] - sig0["heavy"], "dRings": sg["rings"] - sig0["rings"],
                "tan_dist": 1.0 - float(DataStructs.TanimotoSimilarity(sfp, gm.GetFingerprint(m))),
                "qed": float(QED.qed(m)), "sa": float(sascorer.calculateScore(m)),
                "sim": float(DataStructs.TanimotoSimilarity(sfp, gm.GetFingerprint(m))),
                "s0": str(sg["s0"]),
            })
        lane = {"n_endpoints": len(recs),
                "n_distinct_s0": len({r["s0"] for r in recs}),
                "frac_left_s0": (sum(r["d0"] for r in recs) / len(recs)) if recs else 0.0,
                "frac_left_scaffold": (sum(r["d_scaffold"] for r in recs) / len(recs)) if recs else 0.0,
                "max_tan_dist": max((r["tan_dist"] for r in recs), default=0.0)}
        for delta in (0.4, 0.6):
            feas = [r for r in recs if r["qed"] >= 0.6 and r["sa"] <= 4 and r["sim"] >= delta]
            lane[f"d{delta}"] = {"n_feasible": len(feas), "ceiling": bool(feas),
                                 "best_qed": max((r["qed"] for r in feas), default=None)}
        out["lanes"][name] = lane
    out["distinct_states_enumerated"] = len(_law)
    out["sec"] = time.time() - t0
    return out


@app.local_entrypoint()
def main(n_seeds: int = 22, budget: int = 6, rollouts: int = 6,
         n_seq: int = 8, state_cap: int = 220, probe: int = 0):
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    for i, s in enumerate(dv):
        s.setdefault("idx", i)
    dev = dv[:probe] if probe else dv[:n_seeds]
    jobs = [{"cell": f"{s['target']}_dev{s['idx']}", "seed_smiles": s["smiles"],
             "budget": budget, "rollouts": rollouts, "n_seq": n_seq,
             "state_cap": state_cap, "rng": 3 + s["idx"]} for s in dev]
    res = list(atlas.map(jobs))
    Path("diagnostics").mkdir(exist_ok=True)
    p = Path("diagnostics/multiscale_atlas.json")
    p.write_text(json.dumps({"budget": budget, "results": res}, indent=2))
    print(f"cells={len(res)} budget={budget} wrote={p}")
    print(f"median distinct states enumerated/cell: "
          f"{sorted(r['distinct_states_enumerated'] for r in res)[len(res)//2]}")
    print(f"median sec/cell: {sorted(r['sec'] for r in res)[len(res)//2]:.0f}")
