"""Coverage-directed executable search vs R_theta diffusion. No task signal.

WHY COVERAGE AND NOT TREE SEARCH. Tree search allocates simulation by value, so
it presumes some value worth trusting. The fiber census removed that premise:
across ~16,000 exactly enumerated legal successors of random-ZINC states, not one
exceeded JNK3 0.16, and no fiber contained anything at 0.3. In that regime a
value-directed policy is pretending to know something. The correct objective
until informative labels exist is coverage, not optimization.

THE PRIMITIVE. Farthest-first traversal, the classical greedy k-center
algorithm with its 2-approximation on covering radius -- not an ad-hoc diversity
bonus:

    x* = argmax_x  min_{z in C} d(phi(x), phi(z))

with phi a frozen Morgan fingerprint and d Tanimoto distance. R_theta still
proposes every rollout, so nothing is pruned; coverage only decides which
endpoints are RETAINED as the next frontier. That distinction is what separates
this from the top-K idea that already failed: we are not deleting reachable
moves, only choosing where to stand next.

Returning to an archived molecule is free here -- a canonical graph is reopened
directly, with no trajectory replay -- which is the operation Go-Explore had to
work hardest to obtain.

THE ORACLE IS BLIND TO BOTH ARMS. JNK3 is scored only after both searches have
finished, as a diagnostic. Neither policy could see it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.add_local_dir(
    ROOT / "artifacts/oracles", str(REMOTE_ROOT / "artifacts/oracles"), copy=True,
).env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})

app = modal.App("molleo-coverage-sentinel")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
MEM_MIB = int(4.5 * 1024)
N_FRONTIER, N_ROLLOUT, DEPTH, N_GEN = 16, 8, 4, 4

_RT: dict[str, Any] = {}


def _runtime():
    if "model" in _RT:
        return _RT
    import sys
    import torch
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state,
    )
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME,
    )
    from compose_v4.experiments.production_successor_kernel import (
        _default_rewrite_system,
    )
    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    src = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]), repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        src, materialized_state=bundle)
    model = runtime.model
    ck = __import__("torch").load(
        Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
        map_location="cpu", weights_only=False)
    model.load_state_dict(ck["selected_model_state"], strict=True)
    model.eval(); torch.set_grad_enabled(False); torch.set_num_threads(1)
    _RT.update({"model": model, "system": _default_rewrite_system(model)})
    return _RT


@app.function(image=image, cpu=(2.0, 2.0), memory=MEM_MIB, timeout=6 * 60 * 60,
              max_containers=32, retries=1,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def search(task: dict[str, Any]) -> dict[str, Any]:
    import sys, time
    import numpy as np
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.hphi_lazy_helpers import make_helpers
    from compose_v4.experiments.hphi_lazy_sampler import sample_one_transition
    from compose_v4.experiments.production_successor_kernel import (
        _coordinate_action, canonical_state_key,
    )

    rt = _runtime()
    model, system = rt["model"], rt["system"]
    helpers = make_helpers(model, time_point=float(TIME_POINT),
                           canonical_slots=CANONICAL_SLOTS)
    _TF = {"grow_connected": "atom_insert"}
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    arm, rng = task["arm"], np.random.default_rng(int(task["seed"]))
    fp_cache: dict[str, Any] = {}

    def fp(s):
        if s not in fp_cache:
            m = Chem.MolFromSmiles(s)
            fp_cache[s] = None if m is None else gen.GetFingerprint(m)
        return fp_cache[s]

    def dist(a, b):
        fa, fb = fp(a), fp(b)
        if fa is None or fb is None:
            return 1.0
        return 1.0 - float(DataStructs.TanimotoSimilarity(fa, fb))

    def step(smi):
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        d = sample_one_transition(model, st, float(TIME_POINT), rng,
                                  helpers=helpers)
        if d.table is None or d.coordinate is None:
            return ""
        fam = _TF.get(d.table, d.table)
        batch = helpers["build_batch"](st, float(TIME_POINT))
        helpers["family_mask"](model, d.table, st, batch, None, None, None)
        rule, action = _coordinate_action(model, st, batch, family_name=fam,
                                          table_name=d.table,
                                          coordinate=tuple(d.coordinate))
        y = canonical_state_key(system.apply(st, rule, action))
        return "" if y == smi else y

    t0 = time.perf_counter()
    frontier = list(task["starts"])
    covered = list(frontier)          # C, the archive used for farthest-first
    visited = set(frontier)
    n_trans = 0
    gens = []
    for g in range(N_GEN):
        endpoints = []
        for x in frontier:
            for _ in range(N_ROLLOUT):
                cur = x
                for _d in range(DEPTH):
                    y = step(cur)
                    n_trans += 1
                    if not y:
                        break
                    cur = y
                    visited.add(y)
                if cur != x:
                    endpoints.append(cur)
        if not endpoints:
            break
        uniq = sorted(set(endpoints))
        if arm == "coverage":
            # Greedy farthest-first: repeatedly take the endpoint whose nearest
            # covered neighbour is furthest away.
            pick = []
            pool = list(uniq)
            dmin = {y: min(dist(y, z) for z in covered) for y in pool}
            for _ in range(min(N_FRONTIER, len(pool))):
                y = max(pool, key=lambda k: dmin[k])
                pick.append(y); pool.remove(y)
                for k in pool:
                    dk = dist(k, y)
                    if dk < dmin[k]:
                        dmin[k] = dk
        else:
            idx = rng.choice(len(uniq), size=min(N_FRONTIER, len(uniq)),
                             replace=False)
            pick = [uniq[int(i)] for i in idx]
        covered.extend(pick)
        frontier = pick
        radius = float(np.mean([min(dist(y, z) for z in covered[:-len(pick)])
                                for y in pick]))
        gens.append({"generation": g + 1, "n_endpoints": len(uniq),
                     "mean_new_frontier_dist": radius,
                     "n_visited": len(visited), "n_transitions": n_trans})
        print(f"  [{arm}/s{task['seed']}] gen {g+1}: endpoints {len(uniq)} "
              f"visited {len(visited)} dist {radius:.3f}", flush=True)

    def scaf(s):
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(smiles=s)
        except Exception:  # noqa: BLE001
            return ""
    return {"arm": arm, "seed": int(task["seed"]), "generations": gens,
            "n_transitions": n_trans, "visited": sorted(visited),
            "archive": covered,
            "n_scaffolds": len({scaf(s) for s in visited}),
            "seconds": round(time.perf_counter() - t0, 1)}


@app.function(image=image, cpu=(0.25, 0.25), memory=1024, timeout=8 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]], out_name: str) -> dict[str, Any]:
    import time
    out, started = [], time.perf_counter()
    for r in search.map(tasks, order_outputs=False, return_exceptions=True,
                        wrap_returned_exceptions=False):
        if isinstance(r, dict):
            out.append(r)
            print(f"  DONE {r['arm']}/s{r['seed']} trans {r['n_transitions']} "
                  f"visited {len(r['visited'])} scaffolds {r['n_scaffolds']} "
                  f"{r['seconds']:.0f}s [{len(out)}/{len(tasks)}]", flush=True)
        else:
            print(f"  FAILED {type(r).__name__}: {r}", flush=True)
    p = Path(RUN_ROOT) / "molleo_coverage" / f"{out_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"results": out}))
    artifact_volume.commit()
    print(f"DONE {len(out)}/{len(tasks)} in {time.perf_counter()-started:.0f}s")
    return {"n": len(out)}


@app.local_entrypoint()
def main(n_seeds: int = 3) -> None:
    root_dir = Path(__file__).resolve().parents[1]
    lab = json.loads((root_dir / "docs/MOLLEO_DEV_LABELS.json").read_text())
    starts = [m for m, _v in lab["labels"]][:N_FRONTIER]
    tasks = [{"arm": a, "seed": 500000 + s, "starts": starts}
             for s in range(int(n_seeds)) for a in ("random", "coverage")]
    print(f"COVERAGE SENTINEL: {len(tasks)} runs = 2 arms x {n_seeds} seeds.")
    print(f"{N_FRONTIER} frontier x {N_ROLLOUT} rollouts x depth {DEPTH} x "
          f"{N_GEN} generations = {N_FRONTIER*N_ROLLOUT*DEPTH*N_GEN} "
          f"transitions per run. Identical starts in both arms.")
    print("NO task oracle during navigation; JNK3 is scored post hoc only.")
    call = drive.spawn(tasks, f"coverage_{n_seeds}seeds")
    print(f"spawned: {call.object_id}")
