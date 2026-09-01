"""Fitted finite-horizon structural committor by Bellman recursion.

Passive Monte Carlo is dead here: 48,588 samples, positive rate 0.0000 in all
24 strata. 320 base-kernel rollouts never reached establishment once, in a case
where the directed proposal reaches it 12/12 in 2-3 steps. So labels cannot come
from passively observed successes.

They also do not come from full-path importance weights yet -- with Q steering
that hard, P_R/Q is degenerate and its variance would become the bottleneck.
That machinery is kept for validation and eventual SMC correction, with ESS
reported wherever it is used.

Instead: the directed and family-stratified searches are used ONLY to DISCOVER
states worth learning about. The committor itself is defined against the actual
region-local base kernel and fitted by Bellman recursion:

    h_0(x) = g(x)                                    exact structural terminal
    t_b(x) = sum_y R_M(y|x) h_{b-1}(y)               one-step backup

Each state's canonical successor law is enumerated ONCE at collection and
stored, so fitting is pure offline value iteration -- no executor, no
re-enumeration. Training is staged b=1 -> 6 with a target network, so signal
propagates backward from exact terminals rather than bootstrapping from noise.

b is the short structural execution horizon in primitive edits, never an oracle
budget. No operator is named anywhere.
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
app = modal.App("committor-bellman")
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


def enumerate_training_regions(smiles_list, per_molecule: int = 3,
                               max_region: int = 8):
    """Many distinct (molecule, region) pairs, so positives are not all copies
    of one path motif. Molecules here MUST be disjoint from the sentinel set."""
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.control.region import enumerate_regions
    out = []
    for smi in smiles_list:
        try:
            g = smiles_to_molecular_graph(smi)
        except Exception:
            continue
        h = list(map(int, g.implicit_h_counts))
        taken = 0
        for r in enumerate_regions(smi):
            if taken >= per_molecule or not (1 <= r.size <= max_region):
                continue
            if r.interface != "splitting":
                continue
            terms = [int(t[1]) for t in r.boundary]
            if not terms:
                continue
            out.append({"smiles": smi,
                        "atoms": sorted(int(a) for a in r.atoms),
                        "boundary": [[int(i), int(j), float(o)]
                                     for i, j, o in r.boundary],
                        "interface": r.interface, "size": r.size,
                        "released": r.released_fraction,
                        "terminal_h": [h[t] for t in terms],
                        "saturated": all(h[t] == 0 for t in terms)})
            taken += 1
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=3 * 60 * 60,
              retries=2, max_containers=80,
              volumes={ARTIFACT_ROOT: artifact_volume})
def screen_region(job: dict) -> dict:
    """Cheap reachability screen: does a short goal search reach a terminal?

    Separated from collection because the expensive part is storing every
    retained state's canonical successor law, and the previous run spent 47
    minutes per unit doing that for regions with no terminals at all. Screening
    runs the search only.

    The outcome is the label we WANT for coverage: regions that are reachable
    supply positive transition tubes, regions that are not supply genuine
    zero-committor negatives. Neither is forced.
    """
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
    out = []
    t0 = time.time()
    d = Path(OUT_DIR) / "screen"
    d.mkdir(parents=True, exist_ok=True)
    unit_path = d / f"unit{int(job['unit']):04d}.json"
    deadline = float(job.get("soft_deadline_s", 2.5 * 3600))
    for rec in job["regions"]:
        if time.time() - t0 > deadline:      # return what we have rather than
            print(f"[screen] unit {job['unit']} soft deadline after "
                  f"{len(out)}/{len(job['regions'])} regions", flush=True)
            break                            # letting the container be killed
        smi = rec["smiles"]
        st0 = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        n_real = len(smiles_to_molecular_graph(smi).atom_types)
        region = Region(atoms=frozenset(rec["atoms"]),
                        boundary=tuple((i, j, o) for i, j, o in rec["boundary"]),
                        kind="", generator="screen", n_atoms_total=n_real,
                        n_context_components=2, interface=rec["interface"])
        ctx = RR.context_from_region(region)
        lin0 = RR.Lineage.initial(range(n_real))
        old_ids = frozenset(lin0.id_of[s_] for s_ in rec["atoms"] if s_ in lin0.id_of)
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

        found_depth, frontier = None, [(st0, lin0)]
        for d in range(int(job.get("depth", 5))):
            nxt = []
            for st, lin in frontier:
                fams, acts, probs = enum_fn(st)
                idx, _w = RR.admissible_indices(fams, acts, ctx)
                per = {}
                for j in sorted(idx, key=lambda k_: -float(probs[k_])):
                    f = fams[j]
                    if per.get(f, 0) >= int(job.get("per_family", 4)):
                        continue
                    y = apply_fn(st, j)
                    if y is None:
                        continue
                    k = canonical_state_key(y)
                    if not k or not is_valid(k):
                        continue
                    if not RR.context_preserved(st0, y, ctx.frozen,
                                                ctx.terminal_context_slots):
                        continue
                    if not RR.graph_connected(y):
                        continue
                    per[f] = per.get(f, 0) + 1
                    l2 = lin.observe(f, acts[j])
                    if RR.establishment_terminal(y, ctx, l2, old_ids):
                        found_depth = d + 1
                        break
                    nxt.append((y, l2))
                if found_depth:
                    break
            if found_depth:
                break
            frontier = nxt[: int(job.get("beam", 40))]
            if not frontier:
                break
        out.append({**rec, "reachable": found_depth is not None,
                    "reach_depth": found_depth})
        # persist after every region: a killed container then costs one region,
        # not the whole unit
        unit_path.write_text(json.dumps({"regions": out, "sec": time.time() - t0}))
        artifact_volume.commit()
        print(f"[region] u{job['unit']} {len(out)}/{len(job['regions'])} "
              f"size={rec['size']} sat={int(rec['saturated'])} "
              f"reach={'Y' if found_depth else 'n'} "
              f"d={found_depth} {time.time() - t0:.0f}s", flush=True)
    n_reach = sum(1 for r in out if r["reachable"])
    res = {"regions": out, "n_reachable": n_reach, "sec": time.time() - t0}
    # Persist server-side. `modal run --detach` only keeps the LAST triggered
    # function alive, so the local entrypoint that would collect these results
    # dies with the client; a screen that is not written here is lost work.
    unit_path.write_text(json.dumps(res))
    artifact_volume.commit()
    print(f"[screen] unit {job['unit']}: {len(out)} regions, {n_reach} reachable, "
          f"{time.time() - t0:.0f}s", flush=True)
    return res


@app.function(image=image, cpu=(1.0, 1.0), memory=int(2 * 1024), timeout=20 * 60,
              volumes={ARTIFACT_ROOT: artifact_volume})
def list_training_regions(smiles_list: list, per_molecule: int = 3) -> list:
    """Region enumeration must run REMOTELY. The environment that executes
    `modal run` has neither compose_v4 nor rdkit; only the container does. The
    local entrypoint may touch nothing but JSON and paths."""
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    return enumerate_training_regions(smiles_list, per_molecule=per_molecule)


@app.function(image=image, cpu=(1.0, 1.0), memory=int(8 * 1024), timeout=60 * 60,
              volumes={ARTIFACT_ROOT: artifact_volume})
def fit_from_volume(max_b: int = 6, epochs: int = 400) -> dict:
    """Read every persisted collection unit off the volume and fit."""
    artifact_volume.reload()
    d = Path(OUT_DIR) / "collect"
    units = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
    print(f"[fit] {len(units)} collection units off the volume", flush=True)
    return fit_bellman.local(units, max_b=max_b, epochs=epochs)


@app.function(image=image, cpu=(1.0, 1.0), memory=int(2 * 1024), timeout=20 * 60,
              volumes={ARTIFACT_ROOT: artifact_volume})
def reset_collect() -> int:
    """Clear persisted collection units so a new run cannot inherit stale ones."""
    artifact_volume.reload()
    d = Path(OUT_DIR) / "collect"
    n = 0
    if d.exists():
        for f in d.glob("*.json"):
            f.unlink(); n += 1
    artifact_volume.commit()
    return n


@app.function(image=image, cpu=(1.0, 1.0), memory=int(2 * 1024), timeout=20 * 60,
              volumes={ARTIFACT_ROOT: artifact_volume})
def collect_summary() -> list:
    """Per-unit summary of the persisted collection, without the records."""
    artifact_volume.reload()
    d = Path(OUT_DIR) / "collect"
    out = []
    for f in sorted(d.glob("*.json")):
        u = json.loads(f.read_text())
        out.append({k: u.get(k) for k in
                    ("case", "role", "region_smiles", "saturated", "released",
                     "n_states", "n_terminal_states", "n_terminal_successors",
                     "n_paths_found", "sec")})
    return out


@app.function(image=image, cpu=(1.0, 1.0), memory=int(2 * 1024), timeout=20 * 60,
              volumes={ARTIFACT_ROOT: artifact_volume})
def harvest_screen() -> list:
    """Read every persisted screen unit off the volume."""
    artifact_volume.reload()
    d = Path(OUT_DIR) / "screen"
    if not d.exists():
        return []
    return [r for f in sorted(d.glob("unit*.json"))
            for r in json.loads(f.read_text())["regions"]]


@app.function(image=image, cpu=(1.0, 1.0), memory=int(2 * 1024), timeout=20 * 60,
              volumes={ARTIFACT_ROOT: artifact_volume})
def list_many_regions(smiles_list: list, per_molecule: int = 30) -> list:
    """Wide region enumeration for the screening pass."""
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    return enumerate_training_regions(smiles_list, per_molecule=per_molecule)


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=90 * 60,
              retries=2, max_containers=80,
              volumes={ARTIFACT_ROOT: artifact_volume})
def collect_replay(job: dict) -> dict:
    """Discover states, then store each one's canonical successor law once.

    Discovery mixes: family-stratified BFS (reaches the rare structural events),
    ordinary base-kernel rollouts (the bulk of the state space), and states along
    both. The search only decides WHICH states to learn about; the committor is
    defined against R_M regardless.
    """
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
    rec = job["region"]            # explicit region; never re-derived from a
    smi = rec["smiles"]            # shared pool, so train/test cannot overlap
    st0 = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
    n_real = len(smiles_to_molecular_graph(smi).atom_types)
    region = Region(atoms=frozenset(rec["atoms"]),
                    boundary=tuple((i, j, o) for i, j, o in rec["boundary"]),
                    kind="", generator="replay", n_atoms_total=n_real,
                    n_context_components=2, interface=rec["interface"])
    ctx = RR.context_from_region(region)
    lin0 = RR.Lineage.initial(range(n_real))
    old_ids = frozenset(lin0.id_of[s] for s in rec["atoms"] if s in lin0.id_of)
    law_cache: dict = {}

    def enum_fn(st):
        k = canonical_state_key(st)
        if k not in law_cache:
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            law_cache[k] = ([m.executor_rule_name for m in law.marks],
                            [m.action for m in law.marks],
                            np.array([m.probability for m in law.marks], float))
        return law_cache[k]

    def apply_fn(st, j):
        fams, acts, _ = enum_fn(st)
        try:
            return system.apply(st, fams[j], acts[j])
        except Exception:
            return None

    rng = np.random.default_rng(int(job.get("rng", 0)))
    t0 = time.time()

    def is_term(st_, lin_):
        return (RR.establishment_terminal(st_, ctx, lin_, old_ids)
                or RR.completion_terminal(st_, ctx, lin_, old_ids))

    def admissible(st_):
        fams, acts, probs = enum_fn(st_)
        idx, _w = RR.admissible_indices(fams, acts, ctx)
        return fams, acts, probs, idx

    def step_ok(y):
        k = canonical_state_key(y)
        return bool(k and is_valid(k)
                    and RR.context_preserved(st0, y, ctx.frozen,
                                             ctx.terminal_context_slots)
                    and RR.graph_connected(y))

    # ---- GOAL-DIRECTED discovery: keep whole successful trajectories --------
    # Breadth-first sampling around x0 does not contain the events the committor
    # is defined by; a FIFO cap then drops the deep states, which are exactly
    # the terminals. Search here only chooses WHICH states to learn about --
    # every training target is still the R_M backup.
    paths, shells, negatives = [], [], []
    per_family = int(job.get("per_family", 4))
    # Match the reachability probe that actually found paths: depth 5, beam 40,
    # frontier kept in R_theta order. The previous version shuffled the frontier
    # before truncating to 16, which destroyed the family stratification at the
    # frontier level -- a random 16 of several hundred can drop the enabling
    # branch entirely, and every unit reported paths=0.
    # Diversity across searches comes from COMMITTING TO A DIFFERENT FIRST
    # FAMILY, which preserves stratification instead of randomising it away.
    fams0, acts0, probs0, idx0 = admissible(st0)
    first_families = []
    for j in sorted(idx0, key=lambda k_: -float(probs0[k_])):
        if fams0[j] not in first_families:
            first_families.append(fams0[j])
    for _seed in range(int(job.get("goal_searches", 4))):
        forced = (first_families[_seed] if _seed < len(first_families) else None)
        frontier = [(st0, lin0, [])]
        hit = None
        for _d in range(int(job.get("bfs_depth", 5))):
            nxt = []
            for st, lin, path in frontier:
                fams, acts, probs, idx = admissible(st)
                per = {}
                for j in sorted(idx, key=lambda k_: -float(probs[k_])):
                    f = fams[j]
                    if _d == 0 and forced is not None and f != forced:
                        continue          # this search commits to one opening family
                    if per.get(f, 0) >= per_family:
                        continue
                    y = apply_fn(st, j)
                    if y is None or not step_ok(y):
                        continue
                    per[f] = per.get(f, 0) + 1
                    l2 = lin.observe(f, acts[j])
                    node = (y, l2, path + [(st, lin)])
                    if is_term(y, l2):
                        hit = node
                        break
                    nxt.append(node)
                if hit:
                    break
            if hit:
                break
            frontier = nxt[: int(job.get("bfs_beam", 40))]
            if not frontier:
                break
        if hit:
            y, l2, path = hit
            paths.append([(s_, l_) for s_, l_ in path] + [(y, l2)])   # TERMINAL last
        else:
            for st, lin, _p in frontier[:4]:
                negatives.append((st, lin))                            # hard negatives

    # ---- neighbourhood shell around every state on a successful path -------
    for path in paths:
        for st, lin in path:
            fams, acts, probs, idx = admissible(st)
            per = {}
            for j in sorted(idx, key=lambda k_: -float(probs[k_])):
                f = fams[j]
                if per.get(f, 0) >= 2:
                    continue
                y = apply_fn(st, j)
                if y is None or not step_ok(y):
                    continue
                per[f] = per.get(f, 0) + 1
                shells.append((y, lin.observe(f, acts[j])))

    # ---- base-kernel rollouts, for ordinary-state coverage ------------------
    ordinary = []
    for _r in range(int(job.get("rollouts", 3))):
        st, lin = st0, lin0
        for _t in range(int(job.get("length", 8))):
            fams, acts, probs, idx = admissible(st)
            if not idx:
                break
            w = np.array([float(probs[j]) for j in idx], float)
            if w.sum() <= 0:
                break
            w = w / w.sum()
            j = idx[int(rng.choice(len(idx), p=w))]
            y = apply_fn(st, j)
            if y is None or not step_ok(y):
                break
            lin = lin.observe(fams[j], acts[j]); st = y
            ordinary.append((st, lin))

    # ---- PRIORITY order, so the cap can never drop a terminal --------------
    terminals = [(s_, l_) for p_ in paths for s_, l_ in p_ if is_term(s_, l_)]
    predecessors = [(s_, l_) for p_ in paths for s_, l_ in p_ if not is_term(s_, l_)]
    prioritised = (terminals + predecessors + shells + negatives
                   + [(st0, lin0)] + ordinary)

    seen, records = set(), []
    cap = int(job.get("state_cap", 120))
    for st, lin in prioritised:
        if len(records) >= cap:
            break
        k = canonical_state_key(st)
        sig = (k, tuple(sorted(lin.id_of.items())))
        if sig in seen:
            continue
        seen.add(sig)
        fams, acts, probs, idx = admissible(st)
        if not idx:
            continue
        w = np.array([float(probs[j]) for j in idx], float)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        succ = []
        for k_i, j in enumerate(idx):
            y = apply_fn(st, j)
            if y is None:
                continue
            l2 = lin.observe(fams[j], acts[j])
            succ.append({
                "shared": RR.structural_features_shared(y, ctx, l2, old_ids),
                "p": float(w[k_i]),
                "g_est": 1.0 if RR.establishment_terminal(y, ctx, l2, old_ids) else 0.0,
                "g_comp": 1.0 if RR.completion_terminal(y, ctx, l2, old_ids) else 0.0,
                "family": fams[j],
            })
        if not succ:
            continue
        records.append({
            "shared": RR.structural_features_shared(st, ctx, lin, old_ids),
            "g_est": 1.0 if RR.establishment_terminal(st, ctx, lin, old_ids) else 0.0,
            "g_comp": 1.0 if RR.completion_terminal(st, ctx, lin, old_ids) else 0.0,
            "succ": succ,
        })
    print(f"[{job['case_name']}/{job.get('rng')}] paths={len(paths)} "
          f"terminals={len(terminals)} preds={len(predecessors)} "
          f"shell={len(shells)} neg={len(negatives)} records={len(records)} "
          f"{time.time() - t0:.0f}s", flush=True)
    n_term = sum(1 for r in records if r["g_est"] or r["g_comp"])
    n_succ_term = sum(1 for r in records for s_ in r["succ"] if s_["g_est"] or s_["g_comp"])
    res = {"case": job["case_name"], "region_smiles": smi,
            "n_paths_found": len(paths),
            "region_atoms": rec["atoms"], "saturated": bool(rec.get("saturated")),
            "released": rec.get("released"), "records": records,
            "n_states": len(records), "n_terminal_states": n_term,
            "n_terminal_successors": n_succ_term,
            "role": job.get("role"),
            "sec": time.time() - t0}
    # Persist before returning: an hour of collection must not depend on the
    # client surviving. A heartbeat failure already dropped one connection.
    cd = Path(OUT_DIR) / "collect"
    cd.mkdir(parents=True, exist_ok=True)
    (cd / f"{job['case_name']}.json").write_text(json.dumps(res))
    artifact_volume.commit()
    return res


def _build_net(torch, n_feat: int):
    """One definition of the architecture, so the fit and every consumer of the
    checkpoint cannot drift apart."""
    return torch.nn.Sequential(torch.nn.Linear(n_feat, 48), torch.nn.ReLU(),
                               torch.nn.Linear(48, 1))


@app.function(image=image, cpu=(1.0, 1.0), memory=int(6 * 1024), timeout=60 * 60,
              volumes={ARTIFACT_ROOT: artifact_volume})
def fit_bellman(batches: list, max_b: int = 6, epochs: int = 400) -> dict:
    """Staged fitted value iteration with a target network, b = 1 -> max_b."""
    import numpy as np
    import torch
    from compose_v4.control.region_rewrite import features_from_shared

    recs = [r for b in batches
            for r in (b["records"] if isinstance(b, dict) else b)]
    if len(recs) < 20:
        return {"status": "TOO_FEW_STATES", "n": len(recs)}

    def build(kind):
        gk = "g_est" if kind == "establishment" else "g_comp"
        X, G, SP, SG, SX = [], [], [], [], []
        for r in recs:
            X.append(r["shared"]); G.append(r[gk])
            SP.append([s_["p"] for s_ in r["succ"]])
            SG.append([s_[gk] for s_ in r["succ"]])
            SX.append([s_["shared"] for s_ in r["succ"]])
        return X, np.array(G, np.float32), SP, SG, SX

    n_feat = len(features_from_shared(recs[0]["shared"], 1, "establishment"))
    net, tgt = _build_net(torch, n_feat), _build_net(torch, n_feat)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    report = {}
    torch.set_grad_enabled(True)
    # Both terminal kinds are trained JOINTLY at each horizon. Training them in
    # sequence let the completion pass overwrite the establishment committor --
    # the kind is a feature, so one network is fine, but only if it never sees
    # the two kinds in separate phases.
    data = {kind: build(kind) for kind in ("establishment", "completion")}
    for b in range(1, max_b + 1):
        xs, ys = [], []
        for kind, (X, G, SP, SG, SX) in data.items():
            with torch.no_grad():
                tv = []
                for i in range(len(X)):
                    if G[i] >= 1.0:
                        tv.append(1.0); continue
                    if b == 1:
                        tv.append(float(np.dot(SP[i], SG[i])))   # EXACT at b=1
                        continue
                    sx = torch.tensor(
                        [features_from_shared(s, b - 1, kind) for s in SX[i]],
                        dtype=torch.float32)
                    hv = torch.sigmoid(tgt(sx).squeeze(-1)).numpy()
                    hv = np.maximum(hv, np.array(SG[i]))          # terminals pin to 1
                    tv.append(float(np.dot(SP[i], hv)))
            tv = np.array(tv, np.float32)
            xs.extend(features_from_shared(x, b, kind) for x in X)
            ys.extend(tv.tolist())
            report[f"{kind}|b={b}"] = {"target_mean": float(tv.mean()),
                                       "target_max": float(tv.max()),
                                       "n_nonzero": int((tv > 1e-6).sum())}
        xb = torch.tensor(xs, dtype=torch.float32)
        yb = torch.tensor(ys, dtype=torch.float32)
        for _e in range(max(1, epochs // max_b)):
            opt.zero_grad()
            pred = torch.sigmoid(net(xb).squeeze(-1))
            loss = torch.nn.functional.mse_loss(pred, yb)
            loss.backward(); opt.step()
        tgt.load_state_dict(net.state_dict())
        for kind in data:
            report[f"{kind}|b={b}"]["mse"] = float(loss)
    torch.set_grad_enabled(False)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "n_features": n_feat,
                "max_b": max_b}, f"{OUT_DIR}/committor_bellman_v1.pt")
    artifact_volume.commit()
    return {"status": "OK", "n_states": len(recs), "stages": report,
            "path": f"{OUT_DIR}/committor_bellman_v1.pt"}


@app.local_entrypoint()
def screen(per_molecule: int = 30, batch: int = 12, train_lo: int = 6,
           train_hi: int = 22):
    """STAGE 1: screen many regions for short-horizon reachability.

    The replay distribution is then built as an explicit MIXTURE -- reachable
    regions supply positive transition tubes, unreachable ones supply genuine
    zero-committor negatives. We are not trying to make arbitrary regions
    positive; 2/48 (and 0/15 saturated) is a measured property, not a defect.
    """
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    sentinel = {s["smiles"] for s in dv[:6]}
    train = [s["smiles"] for s in dv[train_lo:train_hi]]
    assert not (set(train) & sentinel), "train/test molecule leak"
    regions = list_many_regions.remote(train, per_molecule=per_molecule)
    print(f"molecules={len(train)} regions={len(regions)}")
    jobs = [{"regions": regions[i:i + batch], "unit": i // batch}
            for i in range(0, len(regions), batch)]
    print(f"screen units={len(jobs)}")
    out = list(screen_region.map(jobs))
    scr = harvest_screen.remote() or [r for o in out for r in o["regions"]]
    reach = [r for r in scr if r["reachable"]]
    sat_reach = [r for r in reach if r["saturated"]]
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/region_screen.json").write_text(json.dumps(
        {"train_molecules": train, "regions": scr}))
    from collections import Counter
    print(f"\nscreened={len(scr)}  reachable={len(reach)} "
          f"({len(reach)/max(1,len(scr)):.1%})  saturated_reachable={len(sat_reach)}")
    print("  reach depth histogram:",
          dict(Counter(r["reach_depth"] for r in reach)))
    print("  distinct molecules with a reachable region:",
          len({r["smiles"] for r in reach}))
    print("  released-fraction of reachable regions:",
          [round(r["released"], 2) for r in reach[:15]])
    print(f"  slowest screen unit={max(o['sec'] for o in out):.0f}s")


def _build_mixture(screened, neg_per_pos=2, sat_zero=12, seed=0):
    """Compose the replay set explicitly, not by sampling nature.

    (i)  every constructively reachable region  -> positive transition tubes
    (ii) unreachable regions MATCHED to them on (saturated, size bucket)
         -> hard negatives that cannot be separated by region size alone
    (iii) a deliberate saturated/zero block     -> genuine zero-committor mass

    Intermediate reachability is not sampled directly: it arrives as the
    predecessors and local shells stored along each positive tube, whose R_M
    backups sit strictly between 0 and 1.
    """
    import random
    rng = random.Random(seed)
    pos = [r for r in screened if r["reachable"]]
    neg = [r for r in screened if not r["reachable"]]

    def bucket(r):
        return (bool(r["saturated"]), min(int(r["size"]), 8) // 3)

    pool = {}
    for r in neg:
        pool.setdefault(bucket(r), []).append(r)
    for v in pool.values():
        rng.shuffle(v)

    picked, seen = [], set()

    def take(r, role):
        key = (r["smiles"], tuple(r["atoms"]))
        if key in seen:
            return False
        seen.add(key)
        picked.append({**r, "role": role})
        return True

    for r in pos:
        take(r, "positive")
    for r in pos:                                    # matched hard negatives
        b = bucket(r)
        got = 0
        for cand in pool.get(b, []):
            if got >= neg_per_pos:
                break
            if take(cand, "hard_negative"):
                got += 1
        for cand in (pool.get((b[0], max(0, b[1] - 1)), []) +
                     pool.get((b[0], b[1] + 1), [])):
            if got >= neg_per_pos:
                break
            if take(cand, "hard_negative"):
                got += 1
    sat = [r for r in neg if r["saturated"]]         # deliberate zero block
    rng.shuffle(sat)
    by_mol = {}
    for r in sat:                                    # spread across molecules
        by_mol.setdefault(r["smiles"], []).append(r)
    from itertools import zip_longest             # round-robin, not truncate:
    order = [x for grp in zip_longest(*by_mol.values())   # zip() would drop
             for x in grp if x is not None]               # every long group
    for r in order:
        if sum(1 for p in picked if p["role"] == "zero") >= sat_zero:
            break
        take(r, "zero")
    return picked


@app.local_entrypoint()
def harvest(train_lo: int = 6, train_hi: int = 22):
    """Rebuild the local screen mirror from the volume after a client death."""
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    train = [s["smiles"] for s in dv[train_lo:train_hi]]
    scr = harvest_screen.remote()
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/region_screen.json").write_text(json.dumps(
        {"train_molecules": train, "regions": scr}))
    reach = [r for r in scr if r["reachable"]]
    print(f"harvested={len(scr)}  reachable={len(reach)}  "
          f"molecules={len({r['smiles'] for r in reach})}  "
          f"saturated_reachable={sum(1 for r in reach if r['saturated'])}")


@app.local_entrypoint()
def mixture(neg_per_pos: int = 2, sat_zero: int = 12, max_b: int = 6,
            epochs: int = 400, train_lo: int = 6, train_hi: int = 22):
    """STAGE 2: collect the mixture replay set, gate it, then fit.

    Search breadth is NOT changed from the screen -- the 2/48 positive rate is
    treated as a measured property of arbitrary splitting regions. What changes
    is which regions we spend collection on.
    """
    scr = json.loads(Path("diagnostics/region_screen.json").read_text())
    screened, train = scr["regions"], scr["train_molecules"]
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    assert not (set(train) & {s["smiles"] for s in dv[:6]}), "train/test leak"

    picked = _build_mixture(screened, neg_per_pos, sat_zero)
    from collections import Counter
    roles = Counter(p["role"] for p in picked)
    print(f"screened={len(screened)}  mixture={len(picked)}  {dict(roles)}")
    print("  positive molecules:",
          len({p['smiles'] for p in picked if p['role'] == 'positive'}))

    print(f"cleared {reset_collect.remote()} stale collection units")
    jobs = [{"region": p, "case_name": f"{p['role']}{i}", "rng": i,
             "role": p["role"]} for i, p in enumerate(picked)]
    out = [o for o in collect_replay.map(jobs) if o]
    role_of = {j["case_name"]: j["region"]["role"] for j in jobs}
    for o in out:
        o["role"] = role_of[o["case"]]

    n_term = sum(o["n_terminal_states"] for o in out)
    pos_mols = {o["region_smiles"] for o in out
                if o["role"] == "positive" and o["n_terminal_states"] > 0}
    n_neg_regions = sum(1 for o in out if o["role"] != "positive")
    n_states = sum(o["n_states"] for o in out)
    print(f"\nstates={n_states}  terminal states={n_term}  "
          f"terminal successors={sum(o['n_terminal_successors'] for o in out)}")
    for role in ("positive", "hard_negative", "zero"):
        sel = [o for o in out if o["role"] == role]
        print(f"  {role:<14} regions={len(sel):>3} states={sum(o['n_states'] for o in sel):>5} "
              f"term={sum(o['n_terminal_states'] for o in sel):>3} "
              f"mols={len({o['region_smiles'] for o in sel})}")
    print(f"  slowest unit={max((o['sec'] for o in out), default=0):.0f}s")

    ok = len(pos_mols) >= 3 and n_term >= 8 and n_neg_regions >= len(pos_mols)
    Path("diagnostics/committor_mixture.json").write_text(json.dumps(
        {"gate_passed": ok, "roles": dict(roles),
         "summary": [{k: o[k] for k in
                      ("case", "role", "region_smiles", "saturated", "released",
                       "n_states", "n_terminal_states", "n_terminal_successors",
                       "n_paths_found")} for o in out]}))
    if not ok:
        print(f"\nGATE FAILED (pos molecules={len(pos_mols)}/3, terminals="
              f"{n_term}/8) -> stopping before the fit.")
        return
    print("\nGATE PASSED -> fitting.")
    res = fit_bellman.remote(out, max_b=max_b, epochs=epochs)
    print(json.dumps(res, indent=2)[:3000])


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=3 * 60 * 60,
              retries=2, max_containers=80,
              volumes={ARTIFACT_ROOT: artifact_volume})
def rank_region(job: dict) -> dict:
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    import numpy as np
    import torch
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law)
    from compose_v4.control.region import Region
    from compose_v4.control import region_rewrite as RR
    from compose_v4.gates.med_chem_gate import is_valid

    artifact_volume.reload()
    ckpt = torch.load(f"{OUT_DIR}/committor_bellman_v1.pt", map_location="cpu")
    n_feat, max_b = int(ckpt["n_features"]), int(ckpt["max_b"])
    net = _build_net(torch, n_feat)          # same constructor as the fit
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    torch.set_grad_enabled(False)

    rt = _runtime(); model, system = rt["model"], rt["system"]
    rec = job["region"]; smi = rec["smiles"]
    st0 = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
    n_real = len(smiles_to_molecular_graph(smi).atom_types)
    region = Region(atoms=frozenset(rec["atoms"]),
                    boundary=tuple((i, j, o) for i, j, o in rec["boundary"]),
                    kind="", generator="rank", n_atoms_total=n_real,
                    n_context_components=2, interface=rec["interface"])
    ctx = RR.context_from_region(region)
    lin0 = RR.Lineage.initial(range(n_real))
    old_ids = frozenset(lin0.id_of[s] for s in rec["atoms"] if s in lin0.id_of)
    cache: dict = {}
    horizon = int(job.get("horizon", 5))

    def enum_fn(st):
        k = canonical_state_key(st)
        if k not in cache:
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            cache[k] = ([m.executor_rule_name for m in law.marks],
                        [m.action for m in law.marks],
                        np.array([m.probability for m in law.marks], float))
        return cache[k]

    def step(st, lin, j):
        fams, acts, _ = enum_fn(st)
        try:
            y = system.apply(st, fams[j], acts[j])
        except Exception:
            return None, None
        if y is None:
            return None, None
        k = canonical_state_key(y)
        if not k or not is_valid(k):
            return None, None
        if not RR.context_preserved(st0, y, ctx.frozen, ctx.terminal_context_slots):
            return None, None
        if not RR.graph_connected(y):
            return None, None
        return y, lin.observe(fams[j], acts[j])

    def h_phi(st, lin, budget):
        sh = RR.structural_features_shared(st, ctx, lin, old_ids)
        f = RR.features_from_shared(sh, int(budget), "establishment")
        return float(torch.sigmoid(
            net(torch.tensor([f], dtype=torch.float32))).item())

    def completes(st, lin, depth):
        """Frozen forward search: can a terminal still be reached in `depth`?"""
        frontier = [(st, lin)]
        for _ in range(depth):
            nxt = []
            for s_, l_ in frontier:
                fams, acts, probs = enum_fn(s_)
                idx, _w = RR.admissible_indices(fams, acts, ctx)
                per: dict = {}
                for j in sorted(idx, key=lambda k_: -float(probs[k_])):
                    f = fams[j]
                    if per.get(f, 0) >= 4:
                        continue
                    y, l2 = step(s_, l_, j)
                    if y is None:
                        continue
                    per[f] = per.get(f, 0) + 1
                    if RR.establishment_terminal(y, ctx, l2, old_ids):
                        return True
                    nxt.append((y, l2))
            frontier = nxt[:40]
            if not frontier:
                return False
        return False

    # walk the frozen directed proposal until it hits a plateau
    pairs, t0 = [], time.time()
    st, lin = st0, lin0
    for t in range(horizon):
        fams, acts, probs = enum_fn(st)
        idx, _w = RR.admissible_indices(fams, acts, ctx)
        if not idx:
            break
        base = RR.old_dependence(st, ctx, lin, old_ids)
        succ = []
        for j in idx:
            y, l2 = step(st, lin, j)
            if y is None:
                continue
            succ.append((j, y, l2, RR.old_dependence(y, ctx, l2, old_ids)))
        if not succ:
            break
        flat = [s for s in succ if s[3] >= base]     # no connectivity progress
        remaining = horizon - t - 1
        if len(flat) >= 2 and remaining >= 2:
            labelled = [(s, completes(s[1], s[2], remaining)) for s in flat]
            good = [s for s, ok in labelled if ok]
            bad = [s for s, ok in labelled if not ok]
            if good and bad:
                for g in good:
                    for b in bad:
                        pairs.append({
                            "t": t, "n_flat": len(flat),
                            "h_good": h_phi(g[1], g[2], remaining),
                            "h_bad": h_phi(b[1], b[2], remaining),
                            "fam_good": fams[g[0]], "fam_bad": fams[b[0]]})
                break
        nxt = min(succ, key=lambda s: (s[3], -float(probs[s[0]])))
        st, lin = nxt[1], nxt[2]

    n_ok = sum(1 for p in pairs if p["h_good"] > p["h_bad"])
    print(f"[rank] {job['case_name']} pairs={len(pairs)} correct={n_ok} "
          f"{time.time() - t0:.0f}s", flush=True)
    return {"case": job["case_name"], "smiles": smi,
            "saturated": bool(rec.get("saturated")), "pairs": pairs,
            "n_pairs": len(pairs), "n_correct": n_ok, "sec": time.time() - t0}


@app.function(image=image, cpu=(1.0, 1.0), memory=int(2 * 1024), timeout=20 * 60,
              volumes={ARTIFACT_ROOT: artifact_volume})
def list_test_regions(smiles_list: list, per_molecule: int = 6) -> list:
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    return enumerate_training_regions(smiles_list, per_molecule=per_molecule)


@app.local_entrypoint()
def rank(per_molecule: int = 6, test_lo: int = 0, test_hi: int = 6):
    """Held-out plateau-ranking test. See the module docstring on rank_region."""
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    test = [s["smiles"] for s in dv[test_lo:test_hi]]
    train = {s["smiles"] for s in dv[6:22]}
    assert not (set(test) & train), "held-out molecules overlap the fit set"
    regions = list_test_regions.remote(test, per_molecule=per_molecule)
    print(f"held-out molecules={len(test)} regions={len(regions)}")
    jobs = [{"region": r, "case_name": f"r{i}"} for i, r in enumerate(regions)]
    out = [o for o in rank_region.map(jobs) if o]
    tot = sum(o["n_pairs"] for o in out)
    cor = sum(o["n_correct"] for o in out)
    reg = sum(1 for o in out if o["n_pairs"] > 0)
    print(f"\nregions with a plateau pair={reg}/{len(out)}  pairs={tot}  "
          f"correct={cor} ({cor / max(1, tot):.1%})")
    if tot:
        import statistics as st_
        gaps = [p["h_good"] - p["h_bad"] for o in out for p in o["pairs"]]
        print(f"  median h_good - h_bad = {st_.median(gaps):+.4f}")
        print(f"  molecules contributing = "
              f"{len({o['smiles'] for o in out if o['n_pairs']})}")
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/committor_rank.json").write_text(json.dumps(
        {"n_pairs": tot, "n_correct": cor,
         "per_region": [{k_: o[k_] for k_ in
                         ("case", "smiles", "saturated", "n_pairs", "n_correct")}
                        for o in out]}))
