"""Does frozen R_theta reach the PARP1 basin when TOLD THE FAMILY but not the action?

The operator matrix measured, on the witness route, that the required action's
rank WITHIN ITS OWN FAMILY is 0,0,0,3,5,18 while one of them sits at global
rank 414. That says R_theta ranks well inside a family and badly across
families. This tests the implication directly.

    A  teacher-forced   at each TRUE route state, restrict to the route's
                        family and ask where R_theta puts the route action
    B  open-loop        from the seed, follow only the FAMILY SEQUENCE, letting
                        R_theta pick the action within each family, and see how
                        close the trajectory lands to the target

B is the real experiment. If it lands near the target, the missing intelligence
is almost entirely the family/macro choice, and the within-family prior is fine.

Plus the three support sanity checks: the entry closure appears under a family
floor, its probability is numerically unchanged, and the added support is clean.

CPU only. No docking.
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
app = modal.App("family-oracle")
TIME_POINT, CANONICAL_SLOTS, APPLY_CAP, FLOOR = 0.5, 48, 300, 20
MEM_MIB = int(4.5 * 1024)

_RT: dict[str, Any] = {}
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"


def _runtime():
    """Inlined rather than imported: the image mounts only src/ and configs/,
    so a sibling modal_apps module does not exist inside the container."""
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


def _enumerate(model, system, smi):
    import numpy as np
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
    law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
    pr = np.array([m.probability for m in law.marks], float)
    return st, law, pr, canonical_state_key


@app.function(image=image, cpu=(2.0, 2.0), memory=MEM_MIB, timeout=4 * 60 * 60, retries=0,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run(states: list) -> dict:
    import numpy as np
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.gates.med_chem_gate import is_valid

    rt = _runtime(); model, system = rt["model"], rt["system"]
    gm = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = lambda s: gm.GetFingerprint(Chem.MolFromSmiles(s))
    target = states[-1]
    tfp, tcan = fp(target), Chem.MolToSmiles(Chem.MolFromSmiles(target))

    def sim_to_target(s):
        try: return round(DataStructs.TanimotoSimilarity(tfp, fp(s)), 3)
        except Exception: return None

    # ---- A: teacher-forced, plus the family sequence and support sanity ----
    forced, families, sanity = [], [], []
    for i in range(len(states) - 1):
        st, law, pr, ckey = _enumerate(model, system, states[i])
        order = list(np.argsort(-pr))
        rank_of = {int(j): r for r, j in enumerate(order)}
        nxt_can = Chem.MolToSmiles(Chem.MolFromSmiles(states[i + 1]))
        best = None
        prods = {}
        for j in range(len(law.marks)):
            mk = law.marks[j]
            try: y = ckey(system.apply(st, mk.executor_rule_name, mk.action))
            except Exception: y = None
            if not y: continue
            prods[j] = (y, mk.executor_rule_name)
            ym = Chem.MolFromSmiles(y)
            if ym is not None and Chem.MolToSmiles(ym) == nxt_can:
                r = rank_of[j]
                if best is None or r < best[0]:
                    best = (r, float(pr[j]), mk.executor_rule_name, j)
        if best is None:
            forced.append(dict(step=i, reachable=False)); families.append(None); continue
        r, p, f, j = best
        wf = int(sum(1 for jj in range(len(law.marks))
                     if law.marks[jj].executor_rule_name == f and pr[jj] > p))
        famn = int(sum(1 for jj in range(len(law.marks))
                       if law.marks[jj].executor_rule_name == f))
        forced.append(dict(step=i, reachable=True, family=f, global_rank=r,
                           within_family_rank=wf, family_n=famn, prob=p,
                           in_cap=bool(r < APPLY_CAP),
                           in_floor=bool(r < APPLY_CAP or wf < FLOOR)))
        families.append(f)
        # support sanity at this state
        addl = [jj for jj in order[APPLY_CAP:]
                if sum(1 for kk in range(len(law.marks))
                       if law.marks[kk].executor_rule_name
                       == law.marks[jj].executor_rule_name
                       and pr[kk] > pr[jj]) < FLOOR]
        clean_addl = sum(1 for jj in addl if jj in prods and is_valid(prods[jj][0]))
        sanity.append(dict(step=i, added=len(addl),
                           clean_frac=(round(clean_addl / len(addl), 3) if addl else None)))
        print(f"forced step {i}: {f} g={r} wf={wf}/{famn} floor={forced[-1]['in_floor']}",
              flush=True)

    # ---- B: open-loop, family sequence only ----
    cur = states[0]; traj = [dict(step=-1, smi=cur, sim=sim_to_target(cur))]
    for i, f in enumerate(families):
        if f is None:
            traj.append(dict(step=i, halted="route step not legal forward")); break
        st, law, pr, ckey = _enumerate(model, system, cur)
        cand = [(float(pr[j]), j) for j in range(len(law.marks))
                if law.marks[j].executor_rule_name == f]
        cand.sort(reverse=True)
        picked = None
        for p, j in cand:
            try: y = ckey(system.apply(st, f, law.marks[j].action))
            except Exception: y = None
            if y and Chem.MolFromSmiles(y) is not None and is_valid(y):
                picked = (y, p); break
        if picked is None:
            traj.append(dict(step=i, halted=f"no clean legal {f} action")); break
        cur = picked[0]
        traj.append(dict(step=i, family=f, smi=cur, prob=picked[1],
                         sim=sim_to_target(cur),
                         exact=bool(Chem.MolToSmiles(Chem.MolFromSmiles(cur)) == tcan)))
        print(f"openloop step {i}: {f} sim={traj[-1]['sim']}", flush=True)
    return dict(forced=forced, sanity=sanity, trajectory=traj,
                target=target, families=families)


@app.local_entrypoint()
def main():
    root = Path(__file__).resolve().parents[1]
    fs = json.loads((root / "diagnostics/parp1_witness_route.json").read_text())["forward_states"]
    res = run.remote(fs)
    F = [x for x in res["forced"] if x.get("reachable")]
    print(f"\n=== A. teacher-forced: {len(F)}/{len(res['forced'])} route steps legal forward")
    print(f"  {'step':>4s} {'family':22s} {'gRank':>6s} {'wfRank':>7s} {'famN':>5s} {'inCap':>6s} {'inFloor':>8s}")
    for x in F:
        print(f"  {x['step']:4d} {x['family']:22s} {x['global_rank']:6d} "
              f"{x['within_family_rank']:7d} {x['family_n']:5d} "
              f"{str(x['in_cap']):>6s} {str(x['in_floor']):>8s}")
    wf = [x["within_family_rank"] for x in F]
    print(f"  within-family rank: median {sorted(wf)[len(wf)//2]}, max {max(wf)}")
    print(f"  in global cap: {sum(x['in_cap'] for x in F)}/{len(F)}   "
          f"in cap+floor({FLOOR}): {sum(x['in_floor'] for x in F)}/{len(F)}")
    s = [x for x in res["sanity"] if x["clean_frac"] is not None]
    if s:
        print(f"\n=== support sanity: mean +{sum(x['added'] for x in s)/len(s):.1f} actions, "
              f"mean clean {100*sum(x['clean_frac'] for x in s)/len(s):.0f}%")
    print("\n=== B. open-loop, family sequence only")
    for t in res["trajectory"]:
        if "halted" in t: print(f"  step {t['step']:3d}  HALTED: {t['halted']}"); break
        print(f"  step {t['step']:3d}  {t.get('family','(seed)'):22s} sim_to_target={t['sim']}"
              f"{'  EXACT MATCH' if t.get('exact') else ''}")
    (root / "diagnostics/family_oracle.json").write_text(json.dumps(res, indent=1))
    print("\nwrote diagnostics/family_oracle.json")
