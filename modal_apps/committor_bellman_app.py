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


def _slot_key(state):
    """Exact slot layout, NOT the canonical key.

    canonical_state_key is a graph-isomorphism key, so two states that differ
    only in which slots hold which atoms share it. Mark actions carry slot
    COORDINATES, so index j denotes a different edit in those two states and a
    cache hit keyed canonically returns the wrong successor. Measured: 381
    wrong successors out of 909 on one region, with zero feature mismatches --
    the states came back valid, just not the ones asked for.
    """
    import numpy as np
    return (np.asarray(state.atom_types).tobytes(),
            np.asarray(state.bonds).tobytes())


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
    d = Path(OUT_DIR) / str(job.get("subdir", "screen"))
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
            k = _slot_key(st)          # NOT the canonical key: actions carry
            if k not in cache:         # slot coordinates (see _slot_key)
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
def harvest_screen(subdir: str = "screen") -> list:
    """Read every persisted screen unit off the volume."""
    artifact_volume.reload()
    d = Path(OUT_DIR) / subdir
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
        k = _slot_key(st)              # NOT the canonical key: actions carry
        if k not in law_cache:         # slot coordinates (see _slot_key)
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
        k = _slot_key(st)              # NOT the canonical key: actions carry
        if k not in cache:             # slot coordinates (see _slot_key)
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

    # Walk the frozen directed proposal, recording EVERY plateau contrast along
    # the way rather than stopping at the first. That is a power decision made
    # before seeing any held-out result, not a filter on outcomes.
    pairs, t0 = [], time.time()
    n_plateau, n_contrast = 0, 0
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
        flat = [s_ for s_ in succ if s_[3] >= base]   # no connectivity progress
        remaining = horizon - t - 1
        if len(flat) >= 2 and remaining >= 2:
            n_plateau += 1
            labelled = [(s_, completes(s_[1], s_[2], remaining)) for s_ in flat]
            good = [s_ for s_, ok in labelled if ok]
            bad = [s_ for s_, ok in labelled if not ok]
            if good and bad:                          # a genuine contrast
                n_contrast += 1
                for g in good:
                    for b in bad:
                        pairs.append({
                            "t": t, "n_flat": len(flat),
                            "h_good": h_phi(g[1], g[2], remaining),
                            "h_bad": h_phi(b[1], b[2], remaining),
                            "fam_good": fams[g[0]], "fam_bad": fams[b[0]]})
        nxt = min(succ, key=lambda s_: (s_[3], -float(probs[s_[0]])))
        st, lin = nxt[1], nxt[2]

    n_ok = sum(1 for p in pairs if p["h_good"] > p["h_bad"])
    print(f"[rank] {job['case_name']} plateaus={n_plateau} contrast={n_contrast} "
          f"pairs={len(pairs)} correct={n_ok} {time.time() - t0:.0f}s", flush=True)
    return {"case": job["case_name"], "smiles": smi,
            "saturated": bool(rec.get("saturated")), "pairs": pairs,
            "n_plateau": n_plateau, "n_contrast": n_contrast,
            "n_pairs": len(pairs), "n_correct": n_ok, "sec": time.time() - t0}


@app.function(image=image, cpu=(1.0, 1.0), memory=int(2 * 1024), timeout=20 * 60,
              volumes={ARTIFACT_ROOT: artifact_volume})
def list_test_regions(smiles_list: list, per_molecule: int = 6) -> list:
    import sys
    sys.path.insert(0, str(Path(REMOTE_ROOT) / "src"))
    return enumerate_training_regions(smiles_list, per_molecule=per_molecule)


@app.local_entrypoint()
def screen_dude(batch: int = 4, per_molecule: int = 2, n_molecules: int = 60,
                offset: int = 0, subdir: str = "screen_dude"):
    """Screen the fresh DUD-E holdout pool -- SAME DISTRIBUTION as the dev seeds.

    PREREGISTERED QUALIFICATION GATE, fixed before looking at any result:
        >= 10 reachable regions spanning >= 5 molecules.

    Screening stops once that is met; stragglers are not waited on. The full
    denominator screened is recorded either way, so the stop cannot flatter the
    rate -- though the rate is not the point here. These regions exist to
    supply a CONDITIONAL execution test, not a coverage benchmark.
    """
    pool = json.loads(Path("docs/DUDE_HOLDOUT_POOL.json").read_text())
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    used = {s["smiles"] for s in dv}
    avail = [m["smiles"] for m in pool["molecules"] if m["smiles"] not in used]
    smiles = avail[offset:offset + n_molecules]   # offset: extend the screen to
                                                  # new molecules toward the gate
    print(f"fresh DUD-E molecules={len(smiles)} (of {pool['n']} available)")
    regions = list_many_regions.remote(smiles, per_molecule=per_molecule)
    print(f"eligible splitting regions={len(regions)}")
    jobs = [{"regions": regions[i:i + batch], "unit": i // batch, "subdir": subdir}
            for i in range(0, len(regions), batch)]
    print(f"screen units={len(jobs)}  GATE: >=10 reachable across >=5 molecules")
    list(screen_region.map(jobs))
    scr = harvest_screen.remote(subdir)
    reach = [r for r in scr if r["reachable"]]
    mols = {r["smiles"] for r in reach}
    Path("diagnostics").mkdir(exist_ok=True)
    Path(f"diagnostics/screen_{subdir}.json").write_text(json.dumps(
        {"molecules": sorted({r["smiles"] for r in scr}), "regions": scr}))
    print(f"DENOMINATOR screened={len(scr)}  reachable={len(reach)} "
          f"({len(reach) / max(1, len(scr)):.1%}) across {len(mols)} molecules")
    print(f"  saturated reachable={sum(1 for r in reach if r['saturated'])}")
    print("  GATE MET" if (len(reach) >= 10 and len(mols) >= 5)
          else "  gate not met -- screen more molecules")


@app.local_entrypoint()
def exec_check(budget_calls: int = 6000, max_trials: int = 400,
               n_regions: int = 12, src: str = "diagnostics/screen_screen_pool.json"):
    """CONDITIONAL execution efficiency, not a reachability-rate benchmark.

    Regions are SELECTED BECAUSE a rewrite is known to exist: they come from
    the pool screen, on molecules disjoint from every molecule the committor
    trained on, and were found reachable by the frozen structural search. That
    selection is deliberate and is reported, because the question here is only:

        given that an executable rewrite exists, does R_M * h_phi find it more
        efficiently than R_M?

    Region-selection coverage is a different question, measured in the general
    variable-scope gate, not here.

    Matched on EXECUTOR CALLS alone -- one binding resource, no wall-clock
    valve, so neither arm can be stopped by a limit the other never reaches.
    """
    pool = json.loads(Path(src).read_text())
    reach = [r for r in pool["regions"] if r["reachable"]]
    train = {s["smiles"] for s in json.loads(
        Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]}
    reach = [r for r in reach if r["smiles"] not in train][:n_regions]
    assert reach, "no known-reachable held-out regions"
    print(f"SELECTED CONDITIONAL ON REACHABILITY: {len(reach)} regions, "
          f"{len({r['smiles'] for r in reach})} molecules, all disjoint from "
          f"the committor's training molecules")
    print(f"matched on executor calls only: {budget_calls} per arm per region")
    jobs = [{"region": r, "case_name": f"x{i}", "budget_calls": budget_calls,
             "max_trials": max_trials, "arm_deadline_s": 1e9}
            for i, r in enumerate(reach)]
    out = [o for o in arm_compare.map(jobs) if o]
    print(f"\nregions={len(out)}")
    for arm in ("R_M", "R_M_hphi"):
        est = sum(1 for o in out if o["arms"][arm]["n_establishment"] > 0)
        comp = sum(1 for o in out if o["arms"][arm]["n_complete"] > 0)
        tr = sum(o["arms"][arm]["trials"] for o in out)
        ca = sum(o["arms"][arm]["calls"] for o in out)
        dl = sum(1 for o in out if o["arms"][arm].get("hit_deadline"))
        print(f"  {arm:<10} establishment={est:>3}/{len(out)}  complete={comp:>3}/{len(out)}"
              f"  trials={tr:<6} calls={ca:<8} deadline_hit={dl}")
    won = [o["case"] for o in out if o["arms"]["R_M_hphi"]["n_establishment"] > 0
           and o["arms"]["R_M"]["n_establishment"] == 0]
    lost = [o["case"] for o in out if o["arms"]["R_M"]["n_establishment"] > 0
            and o["arms"]["R_M_hphi"]["n_establishment"] == 0]
    print(f"  h_phi establishes & R_M does not: {len(won)}  |  reverse: {len(lost)}")
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/exec_check.json").write_text(json.dumps(
        {"selection": "conditional on reachability, molecule-disjoint",
         "budget_calls": budget_calls, "per_region": out}, default=str))


@app.local_entrypoint()
def budget_probe(subdir: str = "budget_probe"):
    """Calibration probe: how much does apparent reachability depend on B?

    "Reachable" has meant "completable within 5 primitive edits under
    frozen-context rules". That was a fair unit test for future-aware plateau
    reasoning, where the known paths were 2-3 edits long. It is NOT a statement
    about whether a region is rewriteable: a scaffold rewrite may legitimately
    need 20+ primitive edits, and calling it unreachable at B=5 is a statement
    about the budget, not the molecule.

    So: same frozen search, same regions, only B varies over {5, 10, 20} plus a
    structural allowance

        B(M) = 2|V_M| - 1 + |boundary M|

    (|V_M| atoms, a connected region's spanning-tree lower bound on internal
    bonds, and one step per boundary bond). Measures only whether a complete
    rewrite exists and the step it first completes at. No retraining, no
    replication, no committor work.
    """
    scr = json.loads(Path("diagnostics/region_screen.json").read_text())
    regions = scr["regions"]
    free_ok = [r for r in regions if r["reachable"] and not r["saturated"]][:3]
    sat = [r for r in regions if r["saturated"]][:3]
    big = sorted(regions, key=lambda r: -r["size"])[:2]
    sel, seen = [], set()
    for r in free_ok + sat + big:
        k = (r["smiles"], tuple(r["atoms"]))
        if k not in seen:
            seen.add(k); sel.append(r)
    print(f"probe regions={len(sel)}  "
          f"(free-reachable={len(free_ok)} saturated={len(sat)} largest={len(big)})")
    jobs, u = [], 0
    for r in sel:
        bm = 2 * int(r["size"]) - 1 + len(r["boundary"])
        for depth in (5, 10, 20, bm):
            jobs.append({"regions": [r], "unit": u, "subdir": f"{subdir}_d{depth}",
                         "depth": int(depth)})
            u += 1
    print(f"units={len(jobs)}  (B in 5/10/20/structural, one region each)")
    out = list(screen_region.map(jobs))
    rows = []
    for j, o in zip(jobs, out):
        for rr in o["regions"]:
            rows.append({"smiles": rr["smiles"], "size": rr["size"],
                         "saturated": rr["saturated"], "B": j["depth"],
                         "reachable": rr["reachable"],
                         "first_completion": rr["reach_depth"]})
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/budget_probe.json").write_text(json.dumps(rows))
    from collections import defaultdict
    by_b = defaultdict(list)
    for r in rows:
        by_b[r["B"]].append(r)
    print("\nB      regions  complete  first-completion steps")
    for b in sorted(by_b):
        v = by_b[b]
        ok = [x for x in v if x["reachable"]]
        print(f"{b:<6} {len(v):<8} {len(ok):<9} "
              f"{sorted(x['first_completion'] for x in ok)}")
    print("\nper region (size, saturated) -> first completion by B")
    per = defaultdict(dict)
    for r in rows:
        per[(r["smiles"][:28], r["size"], r["saturated"])][r["B"]] = r["first_completion"]
    for k, v in per.items():
        print(f"  size={k[1]} sat={int(k[2])} {k[0]}: "
              + " ".join(f"B{b}={v[b]}" for b in sorted(v)))


@app.local_entrypoint()
def rescreen_reachable(batch: int = 2, subdir: str = "rescreen"):
    """Re-screen the regions the PRE-FIX screen called reachable.

    Same molecules, same regions, same frozen search -- only the law cache key
    changed. This separates two explanations for the fresh pool's 0.7%
    reachability against the training pool's 10.8%:

      distribution  the new molecules are simply harder, or
      kernel        the canonical-key collisions were manufacturing paths that
                    a correct transition law does not contain.

    The second would mean the committor's training positives were partly
    artifacts, so it has to be answered before any refit.
    """
    scr = json.loads(Path("diagnostics/region_screen.json").read_text())
    prev = [r for r in scr["regions"] if r["reachable"]]
    print(f"previously-reachable regions={len(prev)} "
          f"across {len({r['smiles'] for r in prev})} molecules")
    jobs = [{"regions": prev[i:i + batch], "unit": i // batch, "subdir": subdir}
            for i in range(0, len(prev), batch)]
    print(f"units={len(jobs)}")
    list(screen_region.map(jobs))
    now = harvest_screen.remote(subdir)
    still = [r for r in now if r["reachable"]]
    print(f"STILL REACHABLE under the corrected kernel: {len(still)}/{len(now)}")
    print(f"  molecules={len({r['smiles'] for r in still})}")
    print(f"  depths={sorted(r['reach_depth'] for r in still if r['reach_depth'])}")
    Path("diagnostics/rescreen_reachable.json").write_text(json.dumps(
        {"n_prev": len(prev), "n_still": len(still), "regions": now}))


@app.local_entrypoint()
def screen_pool(batch: int = 4, per_molecule: int = 2, n_molecules: int = 150,
                subdir: str = "screen_pool"):
    """Screen the fresh benchmark-disjoint pool with the corrected kernel.

    The reachable/screened denominator is reported over everything actually
    screened, so the coverage number cannot be flattered by which regions
    happen to finish first.
    """
    pool = json.loads(Path("docs/STRUCTURAL_QUAL_POOL.json").read_text())
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    train = {s["smiles"] for s in dv}
    smiles = [s for s in pool["smiles"] if s not in train][:n_molecules]
    print(f"pool molecules={len(smiles)}")
    regions = list_many_regions.remote(smiles, per_molecule=per_molecule)
    print(f"eligible splitting regions={len(regions)}")
    jobs = [{"regions": regions[i:i + batch], "unit": i // batch,
             "subdir": subdir} for i in range(0, len(regions), batch)]
    print(f"screen units={len(jobs)}")
    list(screen_region.map(jobs))
    scr = harvest_screen.remote(subdir)
    reach = [r for r in scr if r["reachable"]]
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/pool_screen.json").write_text(json.dumps(
        {"molecules": smiles, "regions": scr}))
    print(f"COVERAGE reachable/screened = {len(reach)}/{len(scr)} "
          f"= {len(reach) / max(1, len(scr)):.1%}")
    print(f"  molecules with a reachable region={len({r['smiles'] for r in reach})}")
    print(f"  saturated reachable={sum(1 for r in reach if r['saturated'])}")


@app.local_entrypoint()
def pool_status(subdir: str = "screen_pool"):
    """Read the running pool screen off the volume and apply the stop gate.

    Lets the run be stopped on sufficient evidence instead of on the last
    straggler -- the failure that cost hours on the arms run and killed the
    first Gate 2 attempt. Units in flight still persist, so stopping early
    discards nothing already collected.
    """
    scr = harvest_screen.remote(subdir)
    reach = [r for r in scr if r["reachable"]]
    mols = {r["smiles"] for r in reach}
    Path("diagnostics").mkdir(exist_ok=True)
    # per-subdir filename: a shared name silently overwrote the pool screen
    # with a rescreen of TRAINING regions, which would have been read back as
    # held-out data
    Path(f"diagnostics/screen_{subdir}.json").write_text(json.dumps(
        {"molecules": sorted({r["smiles"] for r in scr}), "regions": scr}))
    print(f"screened={len(scr)} reachable={len(reach)} molecules={len(mols)} "
          f"saturated_reachable={sum(1 for r in reach if r['saturated'])}")
    print(f"GATE {'MET' if (len(reach) >= 30 and len(mols) >= 8) else 'not met'} "
          f"(need >=30 reachable across >=8 molecules)")


@app.local_entrypoint()
def screen_heldout(batch: int = 4, test_lo: int = 0, test_hi: int = 6):
    """STAGE A of the held-out protocol: screen EVERY eligible region on the
    untouched molecules with the same frozen reachability search used
    everywhere else, and report the full denominator.

    This is the coverage question -- where does a ranking problem exist at all?
    It is answered before, and separately from, whether h_phi solves it. The
    committor is NOT retrained after this, and no search parameter is tuned on
    what comes back.
    """
    dv = json.loads(Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"]
    test = [s["smiles"] for s in dv[test_lo:test_hi]]
    train = {s["smiles"] for s in dv[6:22]}
    assert not (set(test) & train), "held-out molecules overlap the fit set"
    regions = list_many_regions.remote(test, per_molecule=100000)   # ALL eligible
    print(f"held-out molecules={len(test)}  eligible regions={len(regions)}")
    jobs = [{"regions": regions[i:i + batch], "unit": i // batch,
             "subdir": "screen_heldout"}
            for i in range(0, len(regions), batch)]
    print(f"screen units={len(jobs)}")
    list(screen_region.map(jobs))
    scr = harvest_screen.remote("screen_heldout")
    reach = [r for r in scr if r["reachable"]]
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/heldout_screen.json").write_text(json.dumps(
        {"molecules": test, "regions": scr}))
    from collections import Counter
    print(f"\nCOVERAGE  reachable/eligible = {len(reach)}/{len(scr)} "
          f"= {len(reach) / max(1, len(scr)):.1%}")
    print(f"  saturated eligible={sum(1 for r in scr if r['saturated'])} "
          f"saturated reachable={sum(1 for r in reach if r['saturated'])}")
    print(f"  molecules with a reachable region="
          f"{len({r['smiles'] for r in reach})}/{len(test)}")
    print(f"  reach depth histogram: {dict(Counter(r['reach_depth'] for r in reach))}")


@app.local_entrypoint()
def rank(max_regions: int = 0):
    """STAGE B: conditional ranking, on reachable held-out regions only.

    Reports two separate facts, never one blended number:
      coverage  -- how often a plateau contrast exists at all
      ranking   -- given a contrast, does h_phi order it correctly

    The committor is frozen. Nothing here retrains or retunes it.
    """
    scr = json.loads(Path("diagnostics/heldout_screen.json").read_text())
    eligible, mols = scr["regions"], scr["molecules"]
    train = {s["smiles"] for s in json.loads(
        Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"][6:22]}
    assert not (set(mols) & train), "held-out molecules overlap the fit set"
    reach = [r for r in eligible if r["reachable"]]
    if max_regions:
        reach = reach[:max_regions]
    print(f"eligible={len(eligible)} reachable={len(reach)} "
          f"({len(reach) / max(1, len(eligible)):.1%} of eligible)")
    jobs = [{"region": r, "case_name": f"h{i}"} for i, r in enumerate(reach)]
    out = [o for o in rank_region.map(jobs) if o]

    n_plateau = sum(o["n_plateau"] for o in out)
    n_contrast = sum(o["n_contrast"] for o in out)
    tot = sum(o["n_pairs"] for o in out)
    cor = sum(o["n_correct"] for o in out)
    gaps = [p["h_good"] - p["h_bad"] for o in out for p in o["pairs"]]
    ties = sum(1 for g in gaps if g == 0.0)
    auc = (cor + 0.5 * ties) / tot if tot else float("nan")
    print("\n--- COVERAGE (where a ranking problem exists) ---")
    print(f"  reachable/eligible      = {len(reach)}/{len(eligible)}")
    print(f"  regions with a plateau  = {sum(1 for o in out if o['n_plateau'])}/{len(out)}")
    print(f"  regions with a contrast = {sum(1 for o in out if o['n_contrast'])}/{len(out)}")
    print(f"  plateau states={n_plateau}  contrast states={n_contrast}")
    print("\n--- RANKING (given a contrast, is h_phi right) ---")
    if not tot:
        print("  NO CONTRAST PAIRS -- ranking undefined on this pool.")
        print("  Do not manufacture more tests; go to R_M vs R_M*h_phi end-to-end.")
    else:
        import statistics as st_
        print(f"  pairs={tot}  correct={cor}  accuracy={cor / tot:.1%}  AUC={auc:.3f}")
        print(f"  median h_good - h_bad = {st_.median(gaps):+.4f}")
        print(f"  molecules contributing = "
              f"{len({o['smiles'] for o in out if o['n_pairs']})}")
    Path("diagnostics/committor_rank.json").write_text(json.dumps(
        {"n_eligible": len(eligible), "n_reachable": len(reach),
         "n_plateau_states": n_plateau, "n_contrast_states": n_contrast,
         "n_pairs": tot, "n_correct": cor, "auc": auc,
         "per_region": [{k_: o[k_] for k_ in
                         ("case", "smiles", "saturated", "n_plateau",
                          "n_contrast", "n_pairs", "n_correct")} for o in out]}))


def make_memo_apply(system, enum_fn, key_fn, guard_fn, calls, cap=200000):
    """One executor application per (state, mark), reused everywhere.

    enumerate_factorized_marked_law does NOT carry successor states -- marks hold
    only (family, rule, action, coordinate, logprob) -- so something must apply
    them. What was wasteful is applying the SAME (state, mark) repeatedly: once
    to featurize it for h_phi, again for the sampled step, and again on every
    restart, since trials all begin at the same root. Only the law was memoized.

    `calls` counts genuine executor invocations, so a cache hit is correctly
    free and the reported executor cost stays honest.
    """
    memo: dict = {}

    def apply_fn(st, j):
        k = (_slot_key(st), int(j))
        if k in memo:
            return memo[k]
        calls["n"] += 1
        fams, acts, _ = enum_fn(st)
        try:
            y = system.apply(st, fams[j], acts[j])
        except Exception:
            y = None
        if y is not None and not guard_fn(y):
            y = None
        if len(memo) < cap:
            memo[k] = y
        return y

    apply_fn.memo = memo
    return apply_fn


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=3 * 60 * 60,
              retries=2, max_containers=80,
              volumes={ARTIFACT_ROOT: artifact_volume})
def arm_compare(job: dict) -> dict:
    """R_M versus R_M * h_phi on complete region replacement, matched on
    EXECUTOR CALLS rather than on trials.

    Trials are the wrong unit: the tilt evaluates every admissible successor at
    each step, while the base kernel applies only the one it samples. Matching
    trials would hand the tilt ~100x the compute. Matching executor calls means
    the undirected arm gets many more restarts out of the same budget, which is
    the honest comparison and the one that reflects practical runtime.

    Run on the WHOLE held-out pool, reachable or not. The frozen search solved
    1/52; regions it could not solve are exactly the population the committor is
    supposed to help with, so filtering them out would destroy the test.
    """
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
    n_feat = int(ckpt["n_features"])
    net = _build_net(torch, n_feat)
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    torch.set_grad_enabled(False)

    rt = _runtime(); model, system = rt["model"], rt["system"]
    rec = job["region"]; smi = rec["smiles"]
    st0 = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
    raw = smiles_to_molecular_graph(smi)
    n_real = len(raw.atom_types)
    region = Region(atoms=frozenset(rec["atoms"]),
                    boundary=tuple((i, j, o) for i, j, o in rec["boundary"]),
                    kind="", generator="arm", n_atoms_total=n_real,
                    n_context_components=2, interface=rec["interface"])
    ctx = RR.context_from_region(region)
    lin0 = RR.Lineage.initial(range(n_real))
    old_ids = frozenset(lin0.id_of[s] for s in rec["atoms"] if s in lin0.id_of)
    pb = RR.prune_budget(region, raw.bonds)
    schedule = [("grow_new", "until_handoff"), ("prune_old", pb)]
    budget_calls = int(job.get("budget_calls", 20000))
    max_trials = int(job.get("max_trials", 2000))
    cache: dict = {}
    calls = {"n": 0}

    def enum_fn(st):
        k = _slot_key(st)              # NOT the canonical key: actions carry
        if k not in cache:             # slot coordinates (see _slot_key)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            cache[k] = ([m.executor_rule_name for m in law.marks],
                        [m.action for m in law.marks],
                        np.array([m.probability for m in law.marks], float))
        return cache[k]

    def _guard(y):
        k = canonical_state_key(y)
        return bool(k and is_valid(k) and RR.graph_connected(y)
                    and RR.context_preserved(st0, y, ctx.frozen,
                                             ctx.terminal_context_slots))

    apply_fn = make_memo_apply(system, enum_fn, canonical_state_key, _guard, calls)

    hcache: dict = {}

    def h_model(y, c, lin, budget, kind):
        key = (canonical_state_key(y), int(budget), kind)
        if key not in hcache:
            sh = RR.structural_features_shared(y, c, lin, old_ids)
            f = RR.features_from_shared(sh, int(budget), kind)
            hcache[key] = float(torch.sigmoid(
                net(torch.tensor([f], dtype=torch.float32))).item())
        return hcache[key]

    out = {}
    for arm in ("R_M", "R_M_hphi"):
        calls["n"] = 0
        t0 = time.time()
        n_trials = 0
        est, comp, first_comp_trial = 0, 0, None
        stuck = 0
        deadline_s = float(job.get("arm_deadline_s", 1500))
        hit_deadline = False
        while calls["n"] < budget_calls and n_trials < max_trials:
            if time.time() - t0 > deadline_s:
                hit_deadline = True     # safety valve only; if it fires the
                break                   # call-matching for this arm is broken
            before_calls = calls["n"]
            p = RR.propose(
                enum_fn, apply_fn, ctx, st0, schedule=schedule,
                rng=np.random.default_rng(9000 + n_trials),
                lineage=lin0, original_region_ids=old_ids,
                max_handoff_steps=int(job.get("max_handoff", 16)),
                h_terminals=None, beta=0.0,
                h_model=(h_model if arm == "R_M_hphi" else None),
                target_ess=float(job.get("target_ess", 0.3)),
                epsilon=float(job.get("epsilon", 0.1)))
            n_trials += 1
            if n_trials <= 3 or n_trials % 25 == 0:
                print(f"[trial] {job['case_name']} {arm} t={n_trials} "
                      f"calls={calls['n']} status={p.status} stage={p.stage} "
                      f"{time.time() - t0:.0f}s", flush=True)
            end = p.endpoint if p.endpoint is not None else st0
            lin = p.lineage or lin0
            if RR.establishment_terminal(end, ctx, lin, old_ids):
                est += 1
            if RR.completion_terminal(end, ctx, lin, old_ids):
                comp += 1
                if first_comp_trial is None:
                    first_comp_trial = n_trials
            if calls["n"] == before_calls:      # trial spent nothing: the region
                stuck += 1                      # admits no action at all, so
                if stuck >= 5:                  # repeating it cannot help
                    break
        out[arm] = {"trials": n_trials, "calls": calls["n"],
                    "n_establishment": est, "n_complete": comp,
                    "first_complete_trial": first_comp_trial,
                    "hit_deadline": hit_deadline, "stuck": stuck,
                    "sec": time.time() - t0}
    print(f"[arm] {job['case_name']} "
          f"R_M: trials={out['R_M']['trials']} est={out['R_M']['n_establishment']} "
          f"comp={out['R_M']['n_complete']} | "
          f"hphi: trials={out['R_M_hphi']['trials']} "
          f"est={out['R_M_hphi']['n_establishment']} "
          f"comp={out['R_M_hphi']['n_complete']}", flush=True)
    res = {"case": job["case_name"], "smiles": smi,
           "saturated": bool(rec.get("saturated")),
           "reachable": bool(rec.get("reachable")),
           "released": rec.get("released"), "arms": out}
    ad = Path(OUT_DIR) / "arms"
    ad.mkdir(parents=True, exist_ok=True)
    (ad / f"{job['case_name']}.json").write_text(json.dumps(res))
    artifact_volume.commit()
    return res


@app.function(image=image, cpu=(1.0, 1.0), memory=int(2 * 1024), timeout=20 * 60,
              volumes={ARTIFACT_ROOT: artifact_volume})
def harvest_arms() -> list:
    artifact_volume.reload()
    d = Path(OUT_DIR) / "arms"
    return [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))] if d.exists() else []


@app.local_entrypoint()
def arms_report():
    """Report the arm comparison from whatever is persisted on the volume."""
    out = harvest_arms.remote()
    if not out:
        print("no arm results persisted yet"); return
    def nreg(arm, key):
        return sum(1 for o in out if o["arms"][arm][key] > 0)
    print(f"regions={len(out)}")
    for arm in ("R_M", "R_M_hphi"):
        dl = sum(1 for o in out if o["arms"][arm].get("hit_deadline"))
        print(f"  {arm:<10} establishment={nreg(arm,'n_establishment'):>3}  "
              f"COMPLETE={nreg(arm,'n_complete'):>3}  "
              f"trials={sum(o['arms'][arm]['trials'] for o in out):>6}  "
              f"calls={sum(o['arms'][arm]['calls'] for o in out):>8}  "
              f"deadline_hit={dl}")
    won = [o["case"] for o in out
           if o["arms"]["R_M_hphi"]["n_complete"] > 0 and o["arms"]["R_M"]["n_complete"] == 0]
    lost = [o["case"] for o in out
            if o["arms"]["R_M"]["n_complete"] > 0 and o["arms"]["R_M_hphi"]["n_complete"] == 0]
    print(f"  h_phi solves & R_M does not: {len(won)}  |  reverse: {len(lost)}")
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/committor_arms.json").write_text(json.dumps(out, default=str))


@app.local_entrypoint()
def arms(budget_calls: int = 20000, max_trials: int = 2000, limit: int = 0):
    """End-to-end: does tilting by the frozen committor buy complete region
    replacement that the base kernel does not reach at the same executor cost?"""
    scr = json.loads(Path("diagnostics/heldout_screen.json").read_text())
    regions, mols = scr["regions"], scr["molecules"]
    train = {s["smiles"] for s in json.loads(
        Path("docs/GENMOL_T4_DEV_SEEDS.json").read_text())["seeds"][6:22]}
    assert not (set(mols) & train), "held-out molecules overlap the fit set"
    if limit:
        regions = regions[:limit]          # pilot only; never for a reported run
    print(f"held-out regions={len(regions)} (whole pool, unfiltered)  "
          f"budget={budget_calls} executor calls per arm per region")
    jobs = [{"region": r, "case_name": f"a{i}", "budget_calls": budget_calls,
             "max_trials": max_trials} for i, r in enumerate(regions)]
    out = [o for o in arm_compare.map(jobs) if o]

    def agg(arm, key):
        return sum(o["arms"][arm][key] for o in out)

    def nreg(arm, key):
        return sum(1 for o in out if o["arms"][arm][key] > 0)
    print(f"\nregions={len(out)}")
    for arm in ("R_M", "R_M_hphi"):
        print(f"  {arm:<10} regions with establishment={nreg(arm,'n_establishment'):>3}"
              f"  regions with COMPLETE replacement={nreg(arm,'n_complete'):>3}"
              f"  trials={agg(arm,'trials'):>6}  calls={agg(arm,'calls'):>8}")
    both = [(o["case"], o["arms"]["R_M"]["n_complete"],
             o["arms"]["R_M_hphi"]["n_complete"]) for o in out]
    won = [c for c, a, b in both if b > 0 and a == 0]
    lost = [c for c, a, b in both if a > 0 and b == 0]
    print(f"  h_phi solves, R_M does not: {len(won)}  |  R_M solves, h_phi does not: {len(lost)}")
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/committor_arms.json").write_text(json.dumps(
        {"budget_calls": budget_calls, "per_region": out}, default=str))


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=60 * 60,
              retries=2, max_containers=80,
              volumes={ARTIFACT_ROOT: artifact_volume})
def verify_successor_cache(job: dict) -> dict:
    """Equivalence test for the memoized successor application.

    The speedup is worthless if it changes what the controller sees, so this
    compares the memoized path against the current uncached one and requires
    BIT-IDENTICAL results on both:
      - the canonical successor key for every (state, mark)
      - the full structural feature vector h_phi would be scored on

    Reports the executor-call saving alongside, so the amortization claim is
    measured rather than asserted.
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
    rec = job["region"]; smi = rec["smiles"]
    st0 = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
    n_real = len(smiles_to_molecular_graph(smi).atom_types)
    region = Region(atoms=frozenset(rec["atoms"]),
                    boundary=tuple((i, j, o) for i, j, o in rec["boundary"]),
                    kind="", generator="verify", n_atoms_total=n_real,
                    n_context_components=2, interface=rec["interface"])
    ctx = RR.context_from_region(region)
    lin0 = RR.Lineage.initial(range(n_real))
    old_ids = frozenset(lin0.id_of[s] for s in rec["atoms"] if s in lin0.id_of)
    law_cache: dict = {}

    def enum_fn(st):
        k = _slot_key(st)              # NOT the canonical key: actions carry
        if k not in law_cache:         # slot coordinates (see _slot_key)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            law_cache[k] = ([m.executor_rule_name for m in law.marks],
                            [m.action for m in law.marks],
                            np.array([m.probability for m in law.marks], float))
        return law_cache[k]

    def _guard(y):
        k = canonical_state_key(y)
        return bool(k and is_valid(k) and RR.graph_connected(y)
                    and RR.context_preserved(st0, y, ctx.frozen,
                                             ctx.terminal_context_slots))

    slow_calls, fast_calls = {"n": 0}, {"n": 0}

    def slow_apply(st, j):                      # current behaviour: no memo
        slow_calls["n"] += 1
        fams, acts, _ = enum_fn(st)
        try:
            y = system.apply(st, fams[j], acts[j])
        except Exception:
            return None
        return y if (y is not None and _guard(y)) else None

    fast_apply = make_memo_apply(system, enum_fn, canonical_state_key,
                                 _guard, fast_calls)

    # visit a spread of states, then re-visit them the way trials actually do
    frontier, states = [(st0, lin0)], []
    for _ in range(int(job.get("depth", 3))):
        nxt = []
        for st, lin in frontier:
            states.append((st, lin))
            fams, acts, probs = enum_fn(st)
            idx, _w = RR.admissible_indices(fams, acts, ctx)
            for j in sorted(idx, key=lambda k_: -float(probs[k_]))[:4]:
                y = slow_apply(st, j)
                if y is not None:
                    nxt.append((y, lin.observe(fams[j], acts[j])))
        frontier = nxt[:6]
        if not frontier:
            break

    n_key, n_feat_cmp, mismatch_key, mismatch_feat = 0, 0, 0, 0
    for rep in range(int(job.get("revisits", 3))):     # trials revisit states
        for st, lin in states:
            fams, acts, probs = enum_fn(st)
            idx, _w = RR.admissible_indices(fams, acts, ctx)
            for j in idx:
                a, b = slow_apply(st, j), fast_apply(st, j)
                if (a is None) != (b is None):
                    mismatch_key += 1
                    continue
                if a is None:
                    continue
                n_key += 1
                if canonical_state_key(a) != canonical_state_key(b):
                    mismatch_key += 1
                    continue
                if rep == 0:
                    l2 = lin.observe(fams[j], acts[j])
                    fa = RR.features_from_shared(
                        RR.structural_features_shared(a, ctx, l2, old_ids),
                        3, "establishment")
                    fb = RR.features_from_shared(
                        RR.structural_features_shared(b, ctx, l2, old_ids),
                        3, "establishment")
                    n_feat_cmp += 1
                    if fa != fb:
                        mismatch_feat += 1
    # census: how often does one canonical key cover several slot layouts?
    by_canon: dict = {}
    for st, _l in states:
        by_canon.setdefault(canonical_state_key(st), set()).add(_slot_key(st))
    colliding = sum(1 for v in by_canon.values() if len(v) > 1)
    extra = sum(len(v) - 1 for v in by_canon.values())
    res = {"case": job["case_name"], "smiles": smi,
           "canonical_keys": len(by_canon), "colliding_keys": colliding,
           "extra_layouts": extra,
           "states_visited": len(states),
           "successors_compared": n_key, "features_compared": n_feat_cmp,
           "key_mismatches": mismatch_key, "feature_mismatches": mismatch_feat,
           "slow_executor_calls": slow_calls["n"],
           "fast_executor_calls": fast_calls["n"]}
    print(f"[verify] {job['case_name']} succ={n_key} feat={n_feat_cmp} "
          f"key_mm={mismatch_key} feat_mm={mismatch_feat} "
          f"slow={slow_calls['n']} fast={fast_calls['n']}", flush=True)
    return res


@app.local_entrypoint()
def verify_cache(n: int = 6, depth: int = 3, revisits: int = 3):
    """Prove the memoized successor path is bit-identical before using it."""
    scr = json.loads(Path("diagnostics/heldout_screen.json").read_text())
    regions = scr["regions"][:n]
    jobs = [{"region": r, "case_name": f"v{i}", "depth": depth,
             "revisits": revisits} for i, r in enumerate(regions)]
    out = [o for o in verify_successor_cache.map(jobs) if o]
    succ = sum(o["successors_compared"] for o in out)
    feat = sum(o["features_compared"] for o in out)
    kmm = sum(o["key_mismatches"] for o in out)
    fmm = sum(o["feature_mismatches"] for o in out)
    slow = sum(o["slow_executor_calls"] for o in out)
    fast = sum(o["fast_executor_calls"] for o in out)
    ck = sum(o["canonical_keys"] for o in out)
    cc = sum(o["colliding_keys"] for o in out)
    print(f"\nCANONICAL-KEY COLLISION CENSUS: {cc}/{ck} canonical keys cover "
          f">1 slot layout ({cc / max(1, ck):.1%}); "
          f"extra layouts={sum(o['extra_layouts'] for o in out)}")
    print(f"\nregions={len(out)}  successors compared={succ}  features compared={feat}")
    print(f"  key mismatches={kmm}   feature mismatches={fmm}")
    print(f"  executor calls: uncached={slow}  memoized={fast}  "
          f"saving={1 - fast / max(1, slow):.1%}")
    print("  EQUIVALENT" if (kmm == 0 and fmm == 0 and succ > 0)
          else "  NOT EQUIVALENT -- do not use the cache")


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=60 * 60,
              retries=2, max_containers=80,
              volumes={ARTIFACT_ROOT: artifact_volume})
def verify_law_cache(job: dict) -> dict:
    """GATE 1: the slot-keyed law cache must equal uncached enumeration.

    The successor-apply check was not enough. The defect was in the LAW cache --
    a canonical-key hit returned another state's marks, actions and
    probabilities. So this compares, for every visited state, the cached lookup
    against a FRESH enumerate_factorized_marked_law call on that exact state:

      - the executor rule name of every mark, in order
      - the action of every mark, in order
      - every mark probability, bitwise

    Also counts canonical-key collisions actually encountered, so the corrected
    cache is exercised on the states where the old one would have been wrong.
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
    rec = job["region"]; smi = rec["smiles"]
    st0 = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
    n_real = len(smiles_to_molecular_graph(smi).atom_types)
    region = Region(atoms=frozenset(rec["atoms"]),
                    boundary=tuple((i, j, o) for i, j, o in rec["boundary"]),
                    kind="", generator="lawverify", n_atoms_total=n_real,
                    n_context_components=2, interface=rec["interface"])
    ctx = RR.context_from_region(region)
    lin0 = RR.Lineage.initial(range(n_real))
    cache: dict = {}

    def enum_cached(st):                      # the shipped path
        k = _slot_key(st)
        if k not in cache:
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            cache[k] = ([m.executor_rule_name for m in law.marks],
                        [m.action for m in law.marks],
                        np.array([m.probability for m in law.marks], float))
        return cache[k]

    def enum_fresh(st):                       # ground truth, never cached
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        return ([m.executor_rule_name for m in law.marks],
                [m.action for m in law.marks],
                np.array([m.probability for m in law.marks], float))

    # explore, then REVISIT so cache hits (not misses) are what gets compared
    frontier, states = [(st0, lin0)], []
    for _ in range(int(job.get("depth", 3))):
        nxt = []
        for st, lin in frontier:
            states.append((st, lin))
            fams, acts, probs = enum_cached(st)
            idx, _w = RR.admissible_indices(fams, acts, ctx)
            for j in sorted(idx, key=lambda k_: -float(probs[k_]))[:4]:
                try:
                    y = system.apply(st, fams[j], acts[j])
                except Exception:
                    continue
                if y is None:
                    continue
                k = canonical_state_key(y)
                if not k or not is_valid(k) or not RR.graph_connected(y):
                    continue
                nxt.append((y, lin.observe(fams[j], acts[j])))
        frontier = nxt[:6]
        if not frontier:
            break

    by_canon: dict = {}
    for st, _l in states:
        by_canon.setdefault(canonical_state_key(st), set()).add(_slot_key(st))
    collided = {c for c, v in by_canon.items() if len(v) > 1}

    n_states, n_marks = 0, 0
    mm_rule, mm_action, mm_prob, n_collided_checked = 0, 0, 0, 0
    for rep in range(int(job.get("revisits", 3))):
        for st, _l in states:
            cf, ca, cp = enum_cached(st)          # hit after the first pass
            ff, fa, fp = enum_fresh(st)
            n_states += 1
            if canonical_state_key(st) in collided:
                n_collided_checked += 1
            if len(cf) != len(ff):
                mm_rule += 1
                continue
            n_marks += len(cf)
            if cf != ff:
                mm_rule += 1
            if [repr(a) for a in ca] != [repr(a) for a in fa]:
                mm_action += 1
            if not np.array_equal(cp, fp):
                mm_prob += 1
    res = {"case": job["case_name"], "smiles": smi,
           "states_compared": n_states, "marks_compared": n_marks,
           "canonical_keys": len(by_canon), "colliding_keys": len(collided),
           "collided_states_checked": n_collided_checked,
           "rule_mismatches": mm_rule, "action_mismatches": mm_action,
           "prob_mismatches": mm_prob}
    print(f"[law] {job['case_name']} states={n_states} marks={n_marks} "
          f"collided_checked={n_collided_checked} "
          f"mm rule={mm_rule} action={mm_action} prob={mm_prob}", flush=True)
    return res


@app.local_entrypoint()
def verify_law(n: int = 10, depth: int = 3, revisits: int = 3):
    """GATE 1 report: cached law == uncached law, bitwise."""
    scr = json.loads(Path("diagnostics/heldout_screen.json").read_text())
    jobs = [{"region": r, "case_name": f"L{i}", "depth": depth,
             "revisits": revisits} for i, r in enumerate(scr["regions"][:n])]
    out = [o for o in verify_law_cache.map(jobs) if o]
    S = lambda k: sum(o[k] for o in out)
    print(f"\nregions={len(out)}  states compared={S('states_compared')}  "
          f"marks compared={S('marks_compared')}")
    print(f"  canonical keys={S('canonical_keys')}  colliding={S('colliding_keys')}"
          f"  states on a collided key={S('collided_states_checked')}")
    print(f"  mismatches: rule={S('rule_mismatches')} "
          f"action={S('action_mismatches')} prob={S('prob_mismatches')}")
    ok = (S('rule_mismatches') == 0 and S('action_mismatches') == 0
          and S('prob_mismatches') == 0 and S('states_compared') > 0)
    print("  GATE 1 PASS -- slot-keyed law cache is exact" if ok
          else "  GATE 1 FAIL -- do not proceed")
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/law_cache_equivalence.json").write_text(
        json.dumps({"pass": ok, "per_region": out}))


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=2 * 60 * 60,
              retries=2, max_containers=80,
              volumes={ARTIFACT_ROOT: artifact_volume})
def single_attempt(job: dict) -> dict:
    """ONE independent rewrite attempt: one region, one arm, one seed.

    Replaces the serial-restart design. Hundreds of restarts inside a container
    estimate a tiny per-region success probability very precisely, which is not
    the question -- and at ~8s per executor call it projected to 14 hours. The
    question is whether h_phi makes a KNOWN-reachable rewrite easier to execute
    at a practical budget, so attempts are independent jobs that fan out.

    A fixed executor-call ceiling bounds one attempt. No wall-clock valve.
    """
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

    arm = job["arm"]
    rt = _runtime(); model, system = rt["model"], rt["system"]
    rec = job["region"]; smi = rec["smiles"]
    st0 = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
    raw = smiles_to_molecular_graph(smi)
    n_real = len(raw.atom_types)
    region = Region(atoms=frozenset(rec["atoms"]),
                    boundary=tuple((i, j, o) for i, j, o in rec["boundary"]),
                    kind="", generator="race", n_atoms_total=n_real,
                    n_context_components=2, interface=rec["interface"])
    ctx = RR.context_from_region(region)
    lin0 = RR.Lineage.initial(range(n_real))
    old_ids = frozenset(lin0.id_of[s] for s in rec["atoms"] if s in lin0.id_of)
    pb = RR.prune_budget(region, raw.bonds)
    schedule = [("grow_new", "until_handoff"), ("prune_old", pb)]
    ceiling = int(job["max_calls"])
    calls = {"n": 0}
    cache: dict = {}

    def enum_fn(st):
        k = _slot_key(st)
        if k not in cache:
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            cache[k] = ([m.executor_rule_name for m in law.marks],
                        [m.action for m in law.marks],
                        np.array([m.probability for m in law.marks], float))
        return cache[k]

    def _guard(y):
        k = canonical_state_key(y)
        return bool(k and is_valid(k) and RR.graph_connected(y)
                    and RR.context_preserved(st0, y, ctx.frozen,
                                             ctx.terminal_context_slots))

    base_apply = make_memo_apply(system, enum_fn, canonical_state_key, _guard, calls)

    def apply_fn(st, j):
        if calls["n"] >= ceiling:      # hard allowance for ONE attempt
            return None
        return base_apply(st, j)

    net = None
    if arm == "R_M_hphi":
        artifact_volume.reload()
        ck = torch.load(f"{OUT_DIR}/committor_bellman_v1.pt", map_location="cpu")
        net = _build_net(torch, int(ck["n_features"]))
        net.load_state_dict(ck["state_dict"]); net.eval()
        torch.set_grad_enabled(False)
    hcache: dict = {}

    def h_model(y, c, lin, budget, kind):
        key = (_slot_key(y), int(budget), kind)
        if key not in hcache:
            f = RR.features_from_shared(
                RR.structural_features_shared(y, c, lin, old_ids), int(budget), kind)
            hcache[key] = float(torch.sigmoid(
                net(torch.tensor([f], dtype=torch.float32))).item())
        return hcache[key]

    t0 = time.time()
    p = RR.propose(enum_fn, apply_fn, ctx, st0, schedule=schedule,
                   rng=np.random.default_rng(int(job["seed"])),
                   lineage=lin0, original_region_ids=old_ids,
                   max_handoff_steps=int(job.get("max_handoff", 16)),
                   h_terminals=None, beta=0.0,
                   h_model=(h_model if arm == "R_M_hphi" else None),
                   target_ess=float(job.get("target_ess", 0.3)),
                   epsilon=float(job.get("epsilon", 0.1)))
    end = p.endpoint if p.endpoint is not None else st0
    lin = p.lineage or lin0
    res = {"case": job["case_name"], "arm": arm, "seed": int(job["seed"]),
           "smiles": smi, "region_atoms": rec["atoms"],
           "saturated": bool(rec.get("saturated")),
           "establishment": bool(RR.establishment_terminal(end, ctx, lin, old_ids)),
           "complete": bool(RR.completion_terminal(end, ctx, lin, old_ids)),
           "calls": calls["n"], "hit_ceiling": calls["n"] >= ceiling,
           "steps": len(p.steps),   # Proposal.steps is a list of step records
           "first_handoff_step": p.first_handoff_step,
           "status": p.status, "stage": p.stage, "sec": time.time() - t0}
    d = Path(OUT_DIR) / "race"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{job['case_name']}_{arm}_{job['seed']}.json").write_text(json.dumps(res))
    artifact_volume.commit()
    print(f"[race] {job['case_name']} {arm} s{job['seed']} "
          f"est={int(res['establishment'])} comp={int(res['complete'])} "
          f"calls={res['calls']} steps={res['steps']} ceil={int(res['hit_ceiling'])} "
          f"{res['sec']:.0f}s", flush=True)
    return res


@app.local_entrypoint()
def profile_attempt(max_calls: int = 4000, n: int = 2):
    """Measure what ONE full attempt actually costs post-cache-fix, on 1-2
    regions, one base and one guided. The ceiling for the race is chosen from
    this and then FROZEN -- never tuned on whether either arm succeeds."""
    scr = json.loads(Path("diagnostics/screen_dude_combined.json").read_text())
    reach = [r for r in scr["regions"] if r["reachable"]][:n]
    jobs = [{"region": r, "case_name": f"p{i}", "arm": a, "seed": 1,
             "max_calls": max_calls}
            for i, r in enumerate(reach) for a in ("R_M", "R_M_hphi")]
    out = [o for o in single_attempt.map(jobs) if o]
    for o in sorted(out, key=lambda o: (o["case"], o["arm"])):
        print(f"  {o['case']} {o['arm']:<10} calls={o['calls']:<6} steps={o['steps']:<3} "
              f"est={int(o['establishment'])} comp={int(o['complete'])} "
              f"ceiling_hit={int(o['hit_ceiling'])} {o['sec']:.0f}s")
    mx = max((o["calls"] for o in out), default=0)
    print(f"\nmax calls for one attempt={mx}  -> suggested frozen ceiling="
          f"{int(mx * 1.5)}")


@app.local_entrypoint()
def race(seeds: int = 2, seed0: int = 1, max_calls: int = 4150,
         n_regions: int = 16):
    """STAGE: independent attempts, paired by (region, seed), one wave.

    Matched per ATTEMPT with a frozen call ceiling, not on executor calls.
    The profile showed h_phi spends ~100x the calls of R_M but only ~2x the
    wall time -- its calls are cheap memoized applies while R_M's each trigger
    a fresh law enumeration -- so matching on calls would penalise guidance for
    being cheap per call. Cost is reported instead of equalised.
    """
    scr = json.loads(Path("diagnostics/screen_dude_combined.json").read_text())
    reach = [r for r in scr["regions"] if r["reachable"]][:n_regions]
    print(f"SELECTED CONDITIONAL ON REACHABILITY: {len(reach)} regions, "
          f"{len({r['smiles'] for r in reach})} molecules, disjoint from training")
    print(f"frozen ceiling={max_calls} calls/attempt, seeds={seed0}..{seed0+seeds-1}")
    jobs = [{"region": r, "case_name": f"r{i}", "arm": a, "seed": s,
             "max_calls": max_calls}
            for i, r in enumerate(reach)
            for a in ("R_M", "R_M_hphi")
            for s in range(seed0, seed0 + seeds)]
    print(f"jobs={len(jobs)} (one wave)")
    out = [o for o in single_attempt.map(jobs) if o]
    Path("diagnostics").mkdir(exist_ok=True)
    Path(f"diagnostics/race_seed{seed0}_{seed0+seeds-1}.json").write_text(
        json.dumps(out, default=str))
    import statistics as st_
    for arm in ("R_M", "R_M_hphi"):
        v = [o for o in out if o["arm"] == arm]
        est = sum(1 for o in v if o["establishment"])
        comp = sum(1 for o in v if o["complete"])
        print(f"  {arm:<10} attempts={len(v):<4} establishment={est:<4} complete={comp:<4} "
              f"median_calls={st_.median([o['calls'] for o in v]) if v else 0:.0f} "
              f"median_sec={st_.median([o['sec'] for o in v]) if v else 0:.0f}")
    # paired by (region, seed): discordant pairs are the evidence
    pair = {}
    for o in out:
        pair.setdefault((o["case"], o["seed"]), {})[o["arm"]] = o
    win = sum(1 for p in pair.values()
              if p.get("R_M_hphi", {}).get("establishment") and not p.get("R_M", {}).get("establishment"))
    loss = sum(1 for p in pair.values()
               if p.get("R_M", {}).get("establishment") and not p.get("R_M_hphi", {}).get("establishment"))
    tie = len(pair) - win - loss
    print(f"  paired (region,seed): h_phi only={win}  R_M only={loss}  tied={tie}  of {len(pair)}")
