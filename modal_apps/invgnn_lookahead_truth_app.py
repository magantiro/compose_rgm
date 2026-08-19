"""Is there a nonmyopic gap at all? A measurement, not a model.

THE QUESTION, before any more value learning:

    does looking ahead under the ACTUAL process R_theta carry information beyond
    scoring the next molecule with F_psi_hat?

h_phi failed its gate against a trivial persistence baseline (Spearman 0.487 vs
0.680), which admits two very different explanations: the amortiser is bad, or
the quantity it amortises barely differs from the immediate score. Training
another h_phi cannot distinguish them. Estimating the TRUE future value by brute
force can.

So for each held-out state we take the successors the controller would actually
choose among -- drawn from R_theta, not the whole fiber -- and estimate

    h_b(y, lambda) ~ (1/M) sum_m g_lambda(X_b^(m)),   X ~ R_theta from y

with M independent rollouts. Then we compare the decision the IMMEDIATE score
makes against the decision TRUE future value makes.

NO ORACLE CALLS. Terminal states are scored by the frozen surrogate, so this
consumes none of the benchmark budget.

DYNAMICS ARE NOT RANDOMISED. Rollouts are R_theta. Substituting uniform legal
edits would estimate E_U[g] instead of E_{R_theta}[g], and R_theta * h^U is not
the Doob control of the process we learned -- an easier-looking but different
problem. An epsilon-exploratory proposal WITH importance weights would be a
legitimate variance device later; it is deliberately not used here.
"""

from __future__ import annotations

import gzip, json, time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = (_base_image
         .add_local_file(ROOT / "docs/INVERSIONGNN_FPSI.pt",
                         str(REMOTE_ROOT / "docs/INVERSIONGNN_FPSI.pt"), copy=True)
         .add_local_file(ROOT / "docs/INVERSIONGNN_FROZEN_PROTOCOL.json",
                         str(REMOTE_ROOT / "docs/INVERSIONGNN_FROZEN_PROTOCOL.json"), copy=True)
         .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"}))

app = modal.App("invgnn-lookahead-truth")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
MEM_MIB = int(4.5 * 1024)
N_CAND, M_ROLL = 16, 32          # candidates per decision, rollouts per candidate
BUDGETS = (4, 8)

_RT: dict[str, Any] = {}


def _runtime():
    if "model" in _RT:
        return _RT
    import sys, torch
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import open_process_v2_t1_source
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime, load_materialized_scorer_state)
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import CHECKPOINT_FILENAME
    from compose_v4.experiments.production_successor_kernel import _default_rewrite_system
    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    src = open_process_v2_t1_source(Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]), repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(src, materialized_state=bundle)
    model = runtime.model
    ck = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["selected_model_state"], strict=True)
    model.eval(); torch.set_grad_enabled(False); torch.set_num_threads(1)
    _RT.update({"model": model, "system": _default_rewrite_system(model)})
    return _RT


@app.function(image=image, cpu=(2.0, 2.0), memory=MEM_MIB, timeout=6 * 60 * 60,
              max_containers=32, retries=1,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def probe(task: dict[str, Any]) -> dict[str, Any]:
    import sys
    import numpy as np, torch, torch.nn as nn
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.hphi_lazy_helpers import make_helpers
    from compose_v4.experiments.hphi_lazy_sampler import sample_one_transition
    from compose_v4.experiments.production_successor_kernel import (
        _coordinate_action, canonical_state_key)

    rt = _runtime(); model, system = rt["model"], rt["system"]
    helpers = make_helpers(model, time_point=float(TIME_POINT), canonical_slots=CANONICAL_SLOTS)
    _TF = {"grow_connected": "atom_insert"}
    P = json.loads((REMOTE_ROOT / "docs/INVERSIONGNN_FROZEN_PROTOCOL.json").read_text())
    prefs = np.array(P["preferences_unit_circle"], dtype=np.float32)
    taus = np.array(task["taus"], dtype=np.float32); Lstar = np.array(task["Lstar"], dtype=np.float32)
    ck = torch.load(REMOTE_ROOT / "docs/INVERSIONGNN_FPSI.pt", map_location="cpu")
    F = nn.Sequential(nn.Linear(2048,512), nn.ReLU(), nn.Dropout(0.1),
                      nn.Linear(512,256), nn.ReLU(), nn.Linear(256,2))
    F.load_state_dict(ck["state"]); F.eval()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def g_of(smis):
        A = np.zeros((len(smis),2048), dtype=np.float32)
        for i,s in enumerate(smis):
            m = Chem.MolFromSmiles(s)
            if m is not None: A[i] = gen.GetFingerprintAsNumPy(m).astype(np.float32)
        with torch.no_grad(): Fh = F(torch.tensor(A)).numpy()
        L = np.stack([np.max(prefs[k][None,:]*(1.0-Fh),axis=1) for k in range(5)],1)
        return np.exp(-(L-Lstar[None,:])/taus[None,:])          # (n,5)

    def step(smi, rng):
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        d = sample_one_transition(model, st, float(TIME_POINT), rng, helpers=helpers)
        if d.table is None or d.coordinate is None: return ""
        fam = _TF.get(d.table, d.table)
        batch = helpers["build_batch"](st, float(TIME_POINT))
        helpers["family_mask"](model, d.table, st, batch, None, None, None)
        rule, action = _coordinate_action(model, st, batch, family_name=fam,
                                          table_name=d.table, coordinate=tuple(d.coordinate))
        y = canonical_state_key(system.apply(st, rule, action))
        return "" if y == smi else y

    x = task["state"]; rng = np.random.default_rng(int(task["seed"]))
    t0 = time.perf_counter()
    cand = []
    for _ in range(N_CAND * 3):
        y = step(x, rng)
        if y and y not in cand: cand.append(y)
        if len(cand) >= N_CAND: break
    if len(cand) < 4:
        return {"state": x, "status": "TOO_FEW_CANDIDATES", "n": len(cand)}
    g_cand = g_of(cand)                                   # immediate desirability
    out = {"state": x, "status": "OK", "candidates": cand,
           "g_immediate": g_cand.tolist(), "h_true": {}, "n_rollout": M_ROLL}
    for b in BUDGETS:
        H = np.zeros((len(cand), 5), dtype=np.float64)
        for ci, y in enumerate(cand):
            ends = []
            for m in range(M_ROLL):
                r2 = np.random.default_rng(int(task["seed"])*7919 + ci*131 + m)
                cur = y
                for _s in range(b-1):
                    nxt = step(cur, r2)
                    if not nxt: break
                    cur = nxt
                ends.append(cur)
            H[ci] = g_of(ends).mean(axis=0)               # MC estimate of E[g_lambda(X_{b-1})]
        out["h_true"][str(b)] = H.tolist()
    out["seconds"] = round(time.perf_counter()-t0, 1)
    return out


@app.function(image=image, cpu=(0.25,0.25), memory=2048, timeout=8*60*60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict], out_name: str) -> dict:
    out, t0 = [], time.perf_counter()
    for r in probe.map(tasks, order_outputs=False, return_exceptions=True,
                       wrap_returned_exceptions=False):
        if isinstance(r, dict):
            out.append(r)
            print(f"  {r['status']} {r['state'][:26]} {r.get('seconds',0):.0f}s "
                  f"[{len(out)}/{len(tasks)}]", flush=True)
        else:
            print(f"  FAILED {type(r).__name__}: {r}", flush=True)
    p = Path(RUN_ROOT) / "invgnn_v1" / f"{out_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"results": out}))
    artifact_volume.commit()
    print(f"DONE {len(out)}/{len(tasks)} in {time.perf_counter()-t0:.0f}s")
    return {"n": len(out)}


@app.local_entrypoint()
def main(n_states: int = 24) -> None:
    """Panel and L* are precomputed into docs/INVGNN_LOOKAHEAD_PANEL.json --
    the modal CLI venv has no rdkit, so nothing chemical runs locally here."""
    root = Path(__file__).resolve().parents[1]
    D = json.loads((root / "docs/INVGNN_LOOKAHEAD_PANEL.json").read_text())
    states = D["panel"][:int(n_states)]
    tasks = [{"state": s, "seed": 400000 + i, "taus": D["taus"], "Lstar": D["Lstar"]}
             for i, s in enumerate(states)]
    print(f"LOOKAHEAD TRUTH: {len(tasks)} held-out states x {N_CAND} R_theta "
          f"candidates x {M_ROLL} rollouts x budgets {BUDGETS}.")
    print("Surrogate-scored, ZERO oracle calls. Dynamics are R_theta, NOT randomised.")
    call = drive.spawn(tasks, f"lookahead_truth_{len(tasks):02d}")
    print(f"spawned: {call.object_id}")
