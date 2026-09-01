"""Bounded executor reachability: is it the proposal, or the operator support?

Two diagnostics, both FAMILY-STRATIFIED so a rare-but-necessary family is never
crowded out by a common one. R_theta ranks candidates only WITHIN a family, and
every family contributes its own quota, so "no path found" means the bounded
search genuinely failed rather than that the move had low probability.

  PRUNE     from a successful post-handoff free-splitting state, is there a
            context-preserving path that removes ALL superseded-region lineage
            while keeping the new connection (d_old == 0) intact?

  PLATEAU   from the saturated state, does ANY executor-valid sequence within
            the depth bound reduce d_old below its initial value, or reach
            handoff? Plateau paths are allowed: intermediate steps need not
            improve d_old, which is exactly what a greedy potential cannot see.

DIAGNOSTIC ONLY. This is not the deployed sampler and nothing here is tuned into
the proposal. Interpretation is fixed in advance:

    path found              -> proposal problem, the kernel missed a move
    no path within bound    -> operator/support problem at this depth

CPU only.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
app = modal.App("executor-reachability")
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


def _pick_cases(smiles_list, max_region=8):
    """Same selector the sentinels use, inlined so the container needs no
    modal_apps package. Pure RDKit plus region enumeration."""
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
            if r.size > max_region or r.size < 1:
                continue
            terms = [int(t[1]) for t in r.boundary]
            free = all(h[t] > 0 for t in terms) if terms else False
            sat = all(h[t] == 0 for t in terms) if terms else False
            rec = {"smiles": smi, "atoms": sorted(int(a) for a in r.atoms),
                   "boundary": [[int(i), int(j), float(o)] for i, j, o in r.boundary],
                   "interface": r.interface, "size": r.size,
                   "released": r.released_fraction,
                   "terminal_h": [h[t] for t in terms]}
            if r.interface == "pendant" and out["pendant"] is None:
                out["pendant"] = rec
            if r.interface == "splitting" and free and out["splitting_free"] is None:
                out["splitting_free"] = rec
            if r.interface == "splitting" and sat and out["splitting_saturated"] is None:
                out["splitting_saturated"] = rec
        if all(out.values()):
            break
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=2 * 60 * 60,
              retries=2, volumes={ARTIFACT_ROOT: artifact_volume})
def probe(job: dict) -> dict:
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    import numpy as np
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.region import Region, enumerate_regions
    from compose_v4.control import region_rewrite as RR
    from compose_v4.gates.med_chem_gate import is_valid

    rt = _runtime(); model, system = rt["model"], rt["system"]
    mode = job["mode"]                       # 'prune' | 'plateau'
    per_family = int(job.get("per_family", 4))
    depth = int(job.get("depth", 5))
    beam = int(job.get("beam", 40))
    t0 = time.time()

    # --- rebuild the exact case the sentinel used --------------------------
    # Inlined, not imported: the image mounts only src/ and configs/, so
    # `modal_apps` does not exist inside the container.
    cases = _pick_cases(job["smiles"])
    rec = cases[job["case_name"]]
    smi = rec["smiles"]
    st0 = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
    n_real = len(smiles_to_molecular_graph(smi).atom_types)
    region = Region(atoms=frozenset(rec["atoms"]),
                    boundary=tuple((i, j, o) for i, j, o in rec["boundary"]),
                    kind="", generator="probe", n_atoms_total=n_real,
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

    start_state, start_lin, note = st0, lin0, ""
    if mode == "prune":
        # reach a post-handoff state with the FROZEN directed kernel, then probe
        p = RR.propose(
            enum_fn, apply_fn, ctx, st0,
            schedule=[("grow_new", "until_handoff")],
            rng=np.random.default_rng(int(job.get("rng", 100))),
            lineage=lin0, original_region_ids=old_ids, max_handoff_steps=16,
            potentials={"grow_new": (lambda s_, c_, l_:
                                     RR.connectivity_potential(s_, c_, l_, old_ids))},
            betas={"grow_new": 6.0}, epsilon=0.1)
        if p.status != "OK" or p.endpoint is None:
            return {"mode": mode, "status": "NO_START_STATE", "stage": p.stage,
                    "sec": time.time() - t0}
        start_state, start_lin = p.endpoint, p.lineage
        note = f"handoff at step {p.first_handoff_step}"

    d0 = RR.old_dependence(start_state, ctx, start_lin, old_ids)

    def goal(st, lin):
        if mode == "prune":
            gone = not [i for i in old_ids if i in lin.slot_of]
            return gone and RR.old_dependence(st, ctx, lin, old_ids) == 0
        return RR.old_dependence(st, ctx, lin, old_ids) < d0

    # ---- family-stratified BFS -------------------------------------------
    seen = {canonical_state_key(start_state)}
    frontier = [(start_state, start_lin, [])]
    found, expanded = None, 0
    for d in range(depth):
        nxt = []
        for st, lin, path in frontier:
            fams, acts, probs = enum_fn(st)
            idx, _why = RR.admissible_indices(fams, acts, ctx)
            per = {}
            order = sorted(idx, key=lambda j: -float(probs[j]))
            for j in order:
                f = fams[j]
                if per.get(f, 0) >= per_family:      # QUOTA PER FAMILY
                    continue
                y = apply_fn(st, j)
                if y is None:
                    continue
                k = canonical_state_key(y)
                if not k or k in seen or not is_valid(k):
                    continue
                if not RR.context_preserved(start_state, y, ctx.frozen,
                                            ctx.terminal_context_slots):
                    continue
                if not RR.graph_connected(y):
                    continue
                per[f] = per.get(f, 0) + 1
                expanded += 1
                seen.add(k)
                lin2 = lin.observe(f, acts[j])
                np_ = path + [f]
                if goal(y, lin2):
                    found = {"depth": d + 1, "path": np_, "smiles": k,
                             "d_old": RR.old_dependence(y, ctx, lin2, old_ids)}
                    break
                nxt.append((y, lin2, np_))
            if found:
                break
        if found:
            break
        frontier = nxt[:beam]
        if not frontier:
            break
    return {"mode": mode, "case": job["case_name"], "note": note,
            "d_old_start": d0, "found": found,
            "states_expanded": expanded, "depth_bound": depth,
            "per_family": per_family, "beam": beam,
            "verdict": ("PATH_EXISTS -> proposal problem" if found
                        else "NO_PATH_IN_BOUND -> operator/support problem"),
            "sec": time.time() - t0}


@app.local_entrypoint()
def main(depth: int = 5, per_family: int = 4, beam: int = 40, n_seeds: int = 6):
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    smiles = [s["smiles"] for s in dv[:n_seeds]]
    jobs = [
        {"mode": "prune", "case_name": "splitting_free", "smiles": smiles,
         "depth": depth, "per_family": per_family, "beam": beam, "rng": 100},
        {"mode": "plateau", "case_name": "splitting_saturated", "smiles": smiles,
         "depth": depth, "per_family": per_family, "beam": beam},
    ]
    out = list(probe.map(jobs))
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/executor_reachability.json").write_text(json.dumps(out, indent=2))
    for r in out:
        print(f"\n=== {r['mode']} / {r.get('case')}  ({r['sec']:.0f}s)")
        if r.get("status") == "NO_START_STATE":
            print(f"    could not reach a post-handoff state: {r['stage']}"); continue
        print(f"    {r['note']}")
        print(f"    d_old at start = {r['d_old_start']}   states expanded = {r['states_expanded']}")
        print(f"    depth<={r['depth_bound']} per_family={r['per_family']} beam={r['beam']}")
        print(f"    {r['verdict']}")
        if r["found"]:
            print(f"    path depth {r['found']['depth']}: {' -> '.join(r['found']['path'])}")
            print(f"    {r['found']['smiles']}")
