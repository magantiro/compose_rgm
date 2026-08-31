"""Operator-firing matrix: WHY does COMPOSE fail on the growth and shrink cells?

The SMILES show COMPOSE adds plenty of atoms but does not organise them into
ring/aromatic architecture (parp1 s0, 5ht1b s7), and cannot simplify large
seeds while holding similarity (braf strict-delta). Four hypotheses, and this
run separates them with NO docking and NO search:

    1  the useful operator is never legal            -> N_f == 0
    2  it is legal but R_theta barely fires it       -> tiny mass, rank > cap
    3  it fires, but mostly into bad chemistry       -> clean_mass << mass
    4  clean useful actions are available and ranked -> the CONTROLLER is at fault

Runs on Modal CPU only: R_theta is a forward pass, no GPU. It must run here
rather than locally because the Gate-0 decision's source_index_sha256 is
computed over a body whose first field is the ABSOLUTE artifact path, so the
lineage chain only validates where the volume is mounted at its own path.
"""

from __future__ import annotations

import collections
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
app = modal.App("operator-firing-matrix")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS, APPLY_CAP = 0.5, 48, 300
MEM_MIB = int(4.5 * 1024)

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


@app.function(image=image, cpu=(2.0, 2.0), memory=MEM_MIB, timeout=2 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def matrix(states: list) -> list:
    import numpy as np
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.gates.med_chem_gate import is_valid

    rt = _runtime(); model, system = rt["model"], rt["system"]
    out = []
    for label, smi, nxt in states:
        t0 = time.time()
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        pr = np.array([m.probability for m in law.marks], float)
        order = list(np.argsort(-pr))
        rank_of = {int(j): r for r, j in enumerate(order)}
        tot = float(pr.sum())
        fam = collections.defaultdict(lambda: dict(
            n=0, mass=0.0, best_rank=10**9, in_cap=0, applied=0,
            clean_n=0, clean_mass=0.0))
        target = None
        if nxt:
            tm = Chem.MolFromSmiles(nxt)
            target = Chem.MolToSmiles(tm) if tm is not None else None
        required = None
        for j in range(len(law.marks)):
            mk = law.marks[j]; f = mk.executor_rule_name
            p = float(pr[j]); r = rank_of[j]
            d = fam[f]
            d["n"] += 1; d["mass"] += p; d["best_rank"] = min(d["best_rank"], r)
            if r < APPLY_CAP:
                d["in_cap"] += 1
            try:
                y = canonical_state_key(system.apply(st, f, mk.action))
            except Exception:
                y = None
            if not y:
                continue
            d["applied"] += 1
            if is_valid(y):
                d["clean_n"] += 1; d["clean_mass"] += p
            if target is not None:
                ym = Chem.MolFromSmiles(y)
                if ym is not None and Chem.MolToSmiles(ym) == target:
                    if required is None or r < required["rank"]:
                        # within-family rank sets the top-k_f floor a
                        # family-stratified proposal support would need
                        wf = int(sum(1 for jj in range(len(law.marks))
                                     if law.marks[jj].executor_rule_name == f
                                     and pr[jj] > p))
                        required = dict(rank=r, prob=p, family=f,
                                        within_family_rank=wf,
                                        family_n=int(sum(
                                            1 for jj in range(len(law.marks))
                                            if law.marks[jj].executor_rule_name == f)),
                                        in_cap=bool(r < APPLY_CAP),
                                        clean=bool(is_valid(y)))
        rows = [dict(family=f, n=d["n"], mass=round(d["mass"] / tot, 6),
                     best_rank=d["best_rank"], in_cap=d["in_cap"],
                     applied=d["applied"],
                     clean_frac=(round(d["clean_n"] / d["applied"], 3)
                                 if d["applied"] else None),
                     clean_mass=round(d["clean_mass"] / tot, 6))
                for f, d in sorted(fam.items(), key=lambda kv: -kv[1]["mass"])]
        out.append(dict(label=label, smi=smi, n_marks=len(law.marks),
                        rows=rows, required=required,
                        seconds=round(time.time() - t0, 1)))
        print(f"{label}: {len(law.marks)} marks in {time.time()-t0:.0f}s", flush=True)
    return out


@app.local_entrypoint()
def run():
    root = Path(__file__).resolve().parents[1]
    seeds = {s["target"] + "_s" + str(s["idx"]): s["smiles"]
             for s in json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())}
    fs = json.loads((root / "diagnostics/parp1_witness_route.json").read_text())["forward_states"]
    states = [["parp1_s0 SEED (route 0)", fs[0], fs[1]],
              ["parp1_s0 route 6 (opened)", fs[6], fs[7]],
              ["parp1_s0 route 12 (pre-close)", fs[12], fs[13]],
              ["parp1_s0 route 1", fs[1], fs[2]],
              ["parp1_s0 route 3", fs[3], fs[4]],
              ["parp1_s0 route 15", fs[15], fs[16]]]
    res = matrix.remote(states)
    for r in res:
        print(f"\n=== {r['label']}  ({r['n_marks']} marks, {r['seconds']}s) ===")
        print(f"  {'family':24s} {'N':>4s} {'mass':>9s} {'rank':>5s} {'inCap':>5s} "
              f"{'clean%':>7s} {'cleanMass':>10s}")
        for x in r["rows"]:
            cf = "n/a" if x["clean_frac"] is None else f"{100*x['clean_frac']:.0f}%"
            print(f"  {x['family']:24s} {x['n']:4d} {x['mass']:9.5f} "
                  f"{x['best_rank']:5d} {x['in_cap']:5d} {cf:>7s} {x['clean_mass']:10.5f}")
        q = r["required"]
        if q:
            print(f"  ROUTE-REQUIRED: {q['family']} rank={q['rank']} p={q['prob']:.3e} "
                  f"in_cap={q['in_cap']} clean={q['clean']}")
        elif r["label"].startswith("parp1"):
            print("  ROUTE-REQUIRED: NOT reachable in one legal edit from this state")
    (root / "diagnostics/operator_firing_kf.json").write_text(json.dumps(res, indent=1))
    print("\nwrote diagnostics/operator_firing_kf.json")
