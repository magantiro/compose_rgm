"""Amortized finite-horizon structural committor h_phi(x, M, tau, b).

Explicit tree expansion of h was measured infeasible: not one splitting trial
finished inside a 90-minute container, even with a working memo. The controller
principle is unchanged -- COMPOSE control is a KL-regularised change of path
measure relative to frozen R_theta via finite-horizon Doob/committor values --
so what changes is only HOW the value is obtained.

Labels are Monte Carlo committor estimates under the region-local BASE kernel
R_M (no tilt, no directed proposal), which is exactly the measure the Doob
transform is defined against:

    for a rollout x_0..x_L and each visited t, budget b:
        y = 1 if the structural terminal occurs in (t, t+b], else 0

Two structural terminals, no task objective and no operator identity anywhere:
establishment (d_old == 0) and completion (old region gone, handoff intact).

Training is a small logistic model on structural features -- CPU only, seconds,
no GPU. Inference is one dot product per successor, replacing a tree.
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
app = modal.App("committor-train")
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


def _pick_cases(smiles_list, max_region=8):
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.control.region import enumerate_regions
    out = {"pendant": None, "splitting_free": None, "splitting_saturated": None}
    for smi in smiles_list:
        try:
            g = smiles_to_molecular_graph(smi)
        except Exception:
            continue
        h = list(map(int, g.implicit_h_counts))
        for r in enumerate_regions(smi):
            if not (1 <= r.size <= max_region):
                continue
            terms = [int(t[1]) for t in r.boundary]
            free = all(h[t] > 0 for t in terms) if terms else False
            sat = all(h[t] == 0 for t in terms) if terms else False
            rec = {"smiles": smi, "atoms": sorted(int(a) for a in r.atoms),
                   "boundary": [[int(i), int(j), float(o)] for i, j, o in r.boundary],
                   "interface": r.interface, "size": r.size,
                   "released": r.released_fraction, "terminal_h": [h[t] for t in terms]}
            if r.interface == "pendant" and out["pendant"] is None:
                out["pendant"] = rec
            if r.interface == "splitting" and free and out["splitting_free"] is None:
                out["splitting_free"] = rec
            if r.interface == "splitting" and sat and out["splitting_saturated"] is None:
                out["splitting_saturated"] = rec
        if all(out.values()):
            break
    return out


# Rollouts are independent, so collection is embarrassingly parallel and the
# fan-out is set by the CHUNK COUNT, not by any Modal limit. At 8 units the
# measured wall clock was ~28 min against ~213 min of compute; 64 units brings
# it to ~5 min. max_containers respects the agreed 80 cap.
@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=90 * 60,
              retries=2, max_containers=80,
              volumes={ARTIFACT_ROOT: artifact_volume})
def collect(job: dict) -> dict:
    """Base-kernel rollouts -> (structural features, committor label) pairs."""
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    import numpy as np
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.region import Region
    from compose_v4.control import region_rewrite as RR
    from compose_v4.gates.med_chem_gate import is_valid

    rt = _runtime(); model, system = rt["model"], rt["system"]
    rec = _pick_cases(job["smiles"])[job["case_name"]]
    smi = rec["smiles"]
    st0 = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
    n_real = len(smiles_to_molecular_graph(smi).atom_types)
    region = Region(atoms=frozenset(rec["atoms"]),
                    boundary=tuple((i, j, o) for i, j, o in rec["boundary"]),
                    kind="", generator="train", n_atoms_total=n_real,
                    n_context_components=2, interface=rec["interface"])
    ctx = RR.context_from_region(region)
    lin0 = RR.Lineage.initial(range(n_real))
    old_ids = frozenset(lin0.id_of[s] for s in rec["atoms"] if s in lin0.id_of)
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

    rng = np.random.default_rng(int(job.get("rng", 0)))
    L = int(job.get("length", 14))
    max_b = int(job.get("max_b", 6))
    X, Y, t0 = [], [], time.time()
    n_roll = int(job.get("rollouts", 40))
    for _r in range(n_roll):
        # Progress logging: Modal streams container stdout, so a long unit is
        # observable instead of opaque until it returns. Without this the only
        # signal is the task count, which cannot distinguish slow from stalled.
        if _r and _r % 5 == 0:
            el = time.time() - t0
            print(f"[{job['case_name']}/{job.get('rng')}] rollout {_r}/{n_roll} "
                  f"{el:.0f}s elapsed, {el / _r:.1f}s/rollout, "
                  f"eta {(n_roll - _r) * el / _r:.0f}s, samples={len(Y)}",
                  flush=True)
        st, lin = st0, lin0
        traj = [(st, lin)]
        for _t in range(L):
            fams, acts, probs = enum_fn(st)
            idx, _w = RR.admissible_indices(fams, acts, ctx)
            if not idx:
                break
            w = np.array([float(probs[j]) for j in idx], float)
            if w.sum() <= 0:
                break
            w = w / w.sum()                       # BASE kernel R_M, untilted
            k = int(rng.choice(len(idx), p=w))
            j = idx[k]
            y = apply_fn(st, j)
            if y is None:
                break
            smi_y = canonical_state_key(y)
            if not smi_y or not is_valid(smi_y):
                break
            if not RR.context_preserved(st0, y, ctx.frozen, ctx.terminal_context_slots):
                break
            if not RR.graph_connected(y):
                break
            lin = lin.observe(fams[j], acts[j])
            st = y
            traj.append((st, lin))
        # Shared structural features once per visited state; the (b, tau) grid
        # only varies two scalars. Recomputing the three BFS passes for all 12
        # combinations was the dominant cost of collection.
        shared = [RR.structural_features_shared(s_, ctx, l_, old_ids)
                  for s_, l_ in traj]
        for kind, term in (("establishment", RR.establishment_terminal),
                           ("completion", RR.completion_terminal)):
            hits = [term(s_, ctx, l_, old_ids) for s_, l_ in traj]
            for t in range(len(traj)):
                for b in range(1, max_b + 1):
                    label = 1.0 if any(hits[t + 1: t + 1 + b]) else 0.0
                    X.append(RR.features_from_shared(shared[t], b, kind))
                    Y.append(label)
    return {"case": job["case_name"], "X": X, "Y": Y,
            "n": len(Y), "pos_rate": float(np.mean(Y)) if Y else 0.0,
            "sec": time.time() - t0}


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=30 * 60,
              volumes={ARTIFACT_ROOT: artifact_volume})
def fit(batches: list) -> dict:
    """Logistic committor on structural features. CPU, seconds, no GPU."""
    import numpy as np
    import torch
    X = np.array([x for b in batches for x in b["X"]], dtype=np.float32)
    Y = np.array([y for b in batches for y in b["Y"]], dtype=np.float32)
    case = np.array([b["case"] for b in batches for _ in b["Y"]])
    if len(Y) < 50:
        return {"status": "TOO_FEW_SAMPLES", "n": int(len(Y))}
    # RARE-EVENT CHECK, before anything is trained. The structural events the
    # controller exists to reach are rare under the untilted base kernel, so
    # passive Monte Carlo can return all-zero labels exactly where h matters
    # most. Report the positive rate stratified rather than in aggregate; an
    # aggregate rate can look healthy while the hard regime is empty.
    strat = {}
    is_comp = X[:, 10] == 1.0
    budget = X[:, 4]
    for cname in sorted(set(case.tolist())):
        for kind, mask_k in (("establishment", ~is_comp), ("completion", is_comp)):
            for b_ in sorted(set(budget.tolist())):
                m = (case == cname) & mask_k & (budget == b_)
                if m.sum() == 0:
                    continue
                strat[f"{cname}|{kind}|b={int(b_)}"] = {
                    "n": int(m.sum()), "pos_rate": float(Y[m].mean())}
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    n = len(Y)
    perm = np.random.default_rng(0).permutation(n)
    cut = int(0.8 * n)
    tr, va = perm[:cut], perm[cut:]
    xt = torch.tensor(Xn[tr]); yt = torch.tensor(Y[tr])
    xv = torch.tensor(Xn[va]); yv = torch.tensor(Y[va])
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss()
    torch.set_grad_enabled(True)
    best, best_state = 1e9, None
    for ep in range(300):
        opt.zero_grad()
        loss = lossf(net(xt).squeeze(-1), yt)
        loss.backward(); opt.step()
        with torch.no_grad():
            vl = float(lossf(net(xv).squeeze(-1), yv))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    torch.set_grad_enabled(False)
    with torch.no_grad():
        pv = torch.sigmoid(net(xv).squeeze(-1)).numpy()
    auc = float(((pv[yv.numpy() == 1][:, None] > pv[yv.numpy() == 0][None, :]).mean())
                if (yv.numpy() == 1).any() and (yv.numpy() == 0).any() else float("nan"))
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mu": mu, "sd": sd,
                "n_features": int(X.shape[1])}, f"{OUT_DIR}/committor_v1.pt")
    artifact_volume.commit()
    empty = [k for k, v in strat.items() if v["pos_rate"] == 0.0]
    return {"status": "OK", "n": int(n), "pos_rate": float(Y.mean()),
            "val_loss": best, "val_auc": auc, "path": f"{OUT_DIR}/committor_v1.pt",
            "stratified_pos_rate": strat,
            "n_empty_strata": len(empty), "empty_strata": empty[:20],
            "verdict": ("PASSIVE_MC_OK" if len(empty) <= len(strat) // 4 else
                        "RARE_EVENT_PROBLEM -> switch to importance-weighted labels")}


@app.local_entrypoint()
def main(rollouts: int = 40, length: int = 14, chunks: int = 4):
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    smiles = [s["smiles"] for s in dv[:6]]
    jobs = [{"case_name": c, "smiles": smiles, "rollouts": rollouts,
             "length": length, "rng": 10 * i + k}
            for i, c in enumerate(("splitting_free", "splitting_saturated"))
            for k in range(chunks)]
    print(f"collect units={len(jobs)}")
    batches = list(collect.map(jobs))
    for b in batches:
        print(f"  {b['case']:22s} n={b['n']:6d} pos_rate={b['pos_rate']:.3f} "
              f"sec={b['sec']:.0f}")
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/committor_batches.json").write_text(json.dumps(
        [{"case": b["case"], "X": b["X"], "Y": b["Y"]} for b in batches]))
    res = fit.remote(batches)
    Path("diagnostics/committor_fit.json").write_text(json.dumps(res, indent=2))
    print(f"\nverdict: {res.get('verdict')}")
    print(f"n={res.get('n')} pos_rate={res.get('pos_rate')} "
          f"val_auc={res.get('val_auc')} empty_strata={res.get('n_empty_strata')}")
    for k, v in sorted((res.get("stratified_pos_rate") or {}).items()):
        print(f"   {k:44s} n={v['n']:6d} pos={v['pos_rate']:.4f}")
