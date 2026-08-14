"""P0c on REAL canonical successor fibers: does the preference continuum collapse?

THE QUESTION, AND WHY IT DECIDES A BRANCH
------------------------------------------
Repaired P3/P4 left one asymmetry: COMPOSE wins hypervolume 12/0 against a fairly
funded generate-and-rank, yet loses on nondominated cardinality. Either

  (a) that is an artifact of sampling only five weights -- in which case sweeping
      the preference continuum fixes it and no new algorithm is needed; or
  (b) the controller maps the whole continuum onto a few distinct decisions -- in
      which case weight-grid density is NOT the limitation and set-level
      allocation is.

An endpoint-pool calculation suggested (b): 1,025 candidates collapsing to TWO
preference regions, median 5 across twelve pools. But endpoint pools are not
one-step successor fibers, and one-step sparsity does not prove trajectory-level
collapse over H=6. This probe settles it on the real object.

THE PARTITION IS EXACT, NOT SAMPLED
-----------------------------------
With `g = utopia - z`, the frozen scalarization is

    s_y(w) = max(w*g1, (1-w)*g2) + rho*(g1+g2)

The augmentation is CONSTANT in w -- a per-candidate offset -- so each candidate
is a max of two lines with a single structural breakpoint at `g2/(g1+g2)`, and
the controller's choice is the lower envelope of 2N segments. Breakpoints are
computed from candidate crossings and refined by bisection against the FROZEN
`_argmin_stable`, so the recorded partition is the one the controller would
actually act on, ties included.

WHAT THIS PROBE DOES NOT CLAIM
------------------------------
A sparse partition would NOT show "Chebyshev scalarization cannot sweep a Pareto
front". Tchebycheff scalarizations are used precisely because they can represent
non-convex Pareto points. The narrow, defensible statement is:

    under THIS frozen molecular state geometry, horizon, reference point and
    greedy closed-loop policy, varying the scalarization weight induces only a
    small number of distinct optimal successor and trajectory regions.

That is enough to motivate aspiration control. It does not indict Chebyshev.

FIBERS ARE PERSISTED THIS TIME
------------------------------
Every enumerated fiber and every objective vector is written to the shard. The
committed Pareto runs paid for this information and discarded it, so this probe
had to re-derive it. It will not have to again.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = (
    _base_image
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "4"})
    .add_local_file(
        ROOT / "diagnostics/retarget_goal_language_normalizers.json",
        str(REMOTE_ROOT / "diagnostics/retarget_goal_language_normalizers.json"), copy=True)
    .add_local_file(
        ROOT / "diagnostics/pareto_tradeoff_census.json",
        str(REMOTE_ROOT / "diagnostics/pareto_tradeoff_census.json"), copy=True)
    .add_local_dir(ROOT / "artifacts/oracles/drd2_svm_v1",
                   str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1"), copy=True)
)

app = modal.App("pareto-preference-partition")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
SMOKE_DIR = "pareto_control_smoke"
OUT_DIR = "pareto_preference_partition"
TIME_POINT = 0.5
CANONICAL_SLOTS = 48
BUDGET = 6
#: Operational guard on the recursive continuum sweep. If the tree needs more
#: unique expansions than this, the branching factor IS the answer -- record
#: TRUNCATED and the observed growth rather than burning the container.
MAX_EXPANSIONS = 240
CHECKPOINT_EVERY = 20


@app.function(
    image=image, cpu=8.0, memory=16 * 1024, timeout=6 * 60 * 60,
    max_containers=12, retries=5,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def partition_source(task: dict[str, Any]) -> dict[str, Any]:
    import sys

    import numpy as np
    import torch
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, Crippen, rdFingerprintGenerator

    sys.path.insert(0, str(REMOTE_ROOT / "src"))

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.drd2_oracle import load_default_oracle
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
    from compose_v4.experiments.pareto_control import Scalarization, _argmin_stable
    from compose_v4.experiments.production_successor_kernel import (
        canonical_successor_result,
    )

    RDLogger.DisableLog("rdApp.*")
    started = time.perf_counter()
    artifact_volume.reload()

    index = int(task["index"])
    out = Path(RUN_ROOT) / task.get("out_dir", OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    shard = out / f"{index:03d}.json.gz"
    if shard.exists():
        print(f"[{index:03d}] shard exists, skipping", flush=True)
        return {"index": index, "skipped": True}

    smoke = json.loads((Path(RUN_ROOT) / SMOKE_DIR / f"{index:03d}.json").read_text())
    start = smoke["start"]
    prefs = list(smoke["preferences"])

    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    source_obj = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]), repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        source_obj, materialized_state=bundle)
    model = runtime.model
    ckpt = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["selected_model_state"], strict=True)
    model.eval()
    torch.set_grad_enabled(False)

    census = json.loads((REMOTE_ROOT / "diagnostics/pareto_tradeoff_census.json").read_text())
    entry = next(p for p in census["pairs"] if p["pair"] == census["adopted_pair"])
    ka, kb = entry["objective_a"], entry["objective_b"]
    sc = census["frozen_scales"]
    utopia = np.array([sc["utopia_p99"][ka], sc["utopia_p99"][kb]])
    centre_s, s_s = sc["similarity_centre"], sc["similarity_iqr"]
    norms = json.loads((REMOTE_ROOT /
                        "diagnostics/retarget_goal_language_normalizers.json"
                        ).read_text())["normalizers"]
    centre_p, s_p = norms["drd2_logodds"]["median"], norms["drd2_logodds"]["iqr"]
    s_qed, s_logp = norms["qed"]["iqr"], norms["clogp"]["iqr"]
    oracle = load_default_oracle(
        str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json"))
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    source_fp = gen.GetFingerprint(Chem.MolFromSmiles(start))
    scalarize = Scalarization(utopia)

    zcache: dict[str, list[float]] = {}
    fibers: dict[str, list[list[Any]]] = {}
    kernel_calls = [0]

    def objective(key: str) -> list[float]:
        if key in zcache:
            return zcache[key]
        mol = Chem.MolFromSmiles(key)
        if mol is None:
            zcache[key] = [-10.0, -10.0]
            return zcache[key]
        vals = {"P": (float(oracle.margin_many([key])[0]) - centre_p) / s_p}
        try:
            qed = float(QED.qed(mol))
        except Exception:  # noqa: BLE001
            qed = float("nan")
        logp = float(Crippen.MolLogP(mol))
        m = np.clip([(qed - 0.6) / s_qed,
                     min(logp - 1.0, 4.0 - logp) / s_logp], -1.5, 1.5)
        vals["D"] = float(-0.25 * np.log(np.sum(np.exp(-m / 0.25))))
        vals["S"] = (DataStructs.TanimotoSimilarity(
            source_fp, gen.GetFingerprint(mol)) - centre_s) / s_s
        zcache[key] = [vals[ka], vals[kb]]
        return zcache[key]

    def fiber(key: str) -> list[list[Any]]:
        """Enumerate once, PERSIST, and never pay for it again."""
        if key in fibers:
            return fibers[key]
        kernel_calls[0] += 1
        state = pad_molecular_graph(smiles_to_molecular_graph(key), CANONICAL_SLOTS)
        res = canonical_successor_result(model, state, float(TIME_POINT))
        rows = [[s.key, float(s.probability)] for s in res.batch.successors]
        fibers[key] = rows
        return rows

    def partition_at(key: str) -> list[dict[str, Any]]:
        """EXACT preference regions at one state, under the frozen tie-break."""
        rows = fiber(key)
        if not rows:
            return []
        keys = [r[0] for r in rows]
        z = np.asarray([objective(k) for k in keys], float)

        def winner(w: float) -> int:
            return _argmin_stable(scalarize(z, w), keys)

        # Structural breakpoints: g2/(g1+g2) per candidate, plus the endpoints.
        g = utopia[None, :] - z
        denom = g[:, 0] + g[:, 1]
        cand = g[:, 1][np.abs(denom) > 1e-15] / denom[np.abs(denom) > 1e-15]
        events = np.unique(np.clip(np.r_[0.0, 1.0, cand[(cand > 0) & (cand < 1)]], 0, 1))
        regions, lo, cur = [], 0.0, winner(0.0)
        for i in range(len(events) - 1):
            a, b = events[i], events[i + 1]
            mid = 0.5 * (a + b)
            wm = winner(mid)
            if wm != cur:
                # Bisect to the exact boundary against the frozen selector.
                x0, x1 = lo, mid
                for _ in range(60):
                    xm = 0.5 * (x0 + x1)
                    if winner(xm) == cur:
                        x0 = xm
                    else:
                        x1 = xm
                regions.append({"lo": lo, "hi": x1, "winner": keys[cur]})
                lo, cur = x1, wm
        regions.append({"lo": lo, "hi": 1.0, "winner": keys[cur]})
        return regions

    # --- (a) partition at every state the frozen five-weight controller visited
    visited: list[str] = []
    for seq in smoke["arms"]["greedy_pref"]["action_sequences"]:
        states = [start] + list(seq)
        visited.extend(states[:-1])
    visited = sorted(set(visited))
    per_state = {}
    for i, key in enumerate(visited):
        regions = partition_at(key)
        per_state[key] = {"n_regions": len(regions),
                          "n_distinct_winners": len({r["winner"] for r in regions}),
                          "fiber_width": len(fibers.get(key, [])),
                          "regions": regions}
        if (i + 1) % CHECKPOINT_EVERY == 0:
            print(f"[{index:03d}] visited {i+1}/{len(visited)} "
                  f"kernel {kernel_calls[0]} {time.perf_counter()-started:.0f}s", flush=True)

    # --- (b) recursive sweep of the continuum from x_0, operationally bounded
    sweep = {"status": "COMPLETE", "leaves": [], "expansions": 0}
    frontier = [(start, 0.0, 1.0, 0, [start])]
    while frontier:
        key, lo, hi, depth, path = frontier.pop()
        if depth >= BUDGET:
            sweep["leaves"].append({"lo": lo, "hi": hi, "endpoint": key,
                                    "path": path})
            continue
        if len(fibers) > MAX_EXPANSIONS:
            sweep["status"] = "TRUNCATED_ON_EXPANSION_GUARD"
            break
        regions = partition_at(key)
        if not regions:
            sweep["leaves"].append({"lo": lo, "hi": hi, "endpoint": key,
                                    "path": path, "dead_end": True})
            continue
        for r in regions:
            a, b = max(lo, r["lo"]), min(hi, r["hi"])
            if b - a > 1e-9:
                frontier.append((r["winner"], a, b, depth + 1, path + [r["winner"]]))
        sweep["expansions"] = len(fibers)

    payload = {
        "schema": "compose.pareto.preference_partition",
        "status": "SMOKE_HELD_IN", "held_out_opened": False,
        "index": index, "source": smoke["source"], "start": start,
        "frozen_preferences": prefs,
        "seconds": round(time.perf_counter() - started, 1),
        "kernel_calls": kernel_calls[0],
        "unique_molecules_evaluated": len(zcache),
        "visited_state_partition": per_state,
        "continuum_sweep": sweep,
        # PERSISTED so nothing re-derives these again.
        "fibers": fibers,
        "objective_vectors": zcache,
        "claim_discipline": (
            "a sparse partition shows only that under THIS frozen state geometry, "
            "horizon, reference point and greedy policy the weight induces few "
            "distinct regions -- NOT that Chebyshev scalarization cannot sweep a "
            "Pareto front"),
    }
    shard.write_bytes(gzip.compress(json.dumps(payload, default=float).encode()))
    artifact_volume.commit()
    nleaf = len(sweep["leaves"])
    print(f"[{index:03d}] DONE  visited-state regions "
          f"med {sorted(v['n_regions'] for v in per_state.values())[len(per_state)//2]}  "
          f"sweep {sweep['status']} leaves={nleaf} expansions={sweep['expansions']}  "
          f"kernel={kernel_calls[0]}  {payload['seconds']:.0f}s", flush=True)
    return {"index": index, "leaves": nleaf, "status": sweep["status"]}


@app.function(image=image, cpu=0.25, memory=2048, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]]) -> int:
    artifact_volume.reload()
    done = 0
    for _ in partition_source.map(tasks, order_outputs=False, return_exceptions=True):
        done += 1
    return done


@app.local_entrypoint()
def main(sources: str = "0,1,2,3,4,5,6,7,8,9,10,11") -> None:
    idx = [int(s) for s in str(sources).split(",") if s.strip()]
    tasks = [{"index": i, "out_dir": OUT_DIR} for i in idx]
    print(f"P0c preference-partition probe on {len(idx)} committed sources")
    print("exact regions at every visited decision state, plus a bounded "
          f"recursive continuum sweep (guard {MAX_EXPANSIONS} expansions)")
    print("fibers and objective vectors are PERSISTED this time")
    call = drive.spawn(tasks)
    print(f"driver spawned: {call.object_id}")
