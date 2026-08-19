"""The 10K task-information corpus for the InversionGNN head-to-head.

BUDGET IS THE POINT. InversionGNN spends 10,000 oracle calls training a property
predictor. COMPOSE gets exactly the same 10,000 -- one call per DISTINCT
canonical molecule -- and converts them into future-value supervision instead of
an immediate-property map. Rollouts themselves cost nothing: R_theta is frozen
and unlabelled, so the whole budget buys labels.

DETERMINISM, so nothing can be chosen after seeing JNK3/GSK3b:
  - sources are taken in frozen sha256 order from the corpus pool recorded in
    docs/INVERSIONGNN_FROZEN_PROTOCOL.json, which is disjoint from the 100-molecule
    initialization bank;
  - H = 24 and 2 independent rollouts per source, so h_phi sees CONTRASTING
    futures from the same chemistry rather than one path per molecule;
  - states are deduplicated in (source index, trajectory, position) order and
    truncated at exactly 10,000 -- if dedup shortens a source's contribution the
    next source in the frozen order fills the gap;
  - sources are split 90/10 BEFORE any training example is expanded, and the
    B_lambda thresholds later come from the training split only.

H = 24 deliberately: that is the horizon the COMPOSE value machinery is already
qualified at. H40 is not reopened here.
"""

from __future__ import annotations

import gzip
import json
import time
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

app = modal.App("invgnn-corpus")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
OUT_DIR = "invgnn_v1"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
MEM_MIB = int(4.5 * 1024)
HORIZON, N_TRAJ = 24, 2
TARGET_UNIQUE = 10_000
VAL_EVERY = 10          # every 10th source held out -> 90/10 by SOURCE

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
    ck = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["selected_model_state"], strict=True)
    model.eval(); torch.set_grad_enabled(False); torch.set_num_threads(1)
    _RT.update({"model": model, "system": _default_rewrite_system(model)})
    return _RT


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=4 * 60 * 60,
              max_containers=64, retries=2,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def rollout(task: dict[str, Any]) -> dict[str, Any]:
    """N_TRAJ unlabelled R_theta rollouts of length HORIZON from one source."""
    import sys
    import numpy as np
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger
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
    t0 = time.perf_counter()

    def step(smi, rng):
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

    src = task["source"]
    try:
        canon = canonical_state_key(
            pad_molecular_graph(smiles_to_molecular_graph(src), CANONICAL_SLOTS))
    except Exception as exc:  # noqa: BLE001
        return {"index": task["index"], "status": f"BAD_SOURCE:{type(exc).__name__}"}
    trajs = []
    for r in range(N_TRAJ):
        rng = np.random.default_rng(int(task["seed"]) * 100 + r)
        path, cur = [canon], canon
        for _ in range(HORIZON):
            y = step(cur, rng)
            if not y:
                break
            cur = y
            path.append(y)
        trajs.append(path)
    return {"index": task["index"], "status": "OK", "source": canon,
            "trajectories": trajs,
            "seconds": round(time.perf_counter() - t0, 1)}


@app.function(image=image, cpu=(4.0, 4.0), memory=16384, timeout=4 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def label(results: list[dict]) -> dict[str, Any]:
    """Deduplicate deterministically, take exactly 10K, spend the budget once.

    One oracle call = one distinct canonical molecule, matching the accounting
    the benchmark charges InversionGNN for.
    """
    import sys
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.benchmark.oracles.task3 import Task3Objectives

    obj = Task3Objectives()
    ok = [r for r in results if r.get("status") == "OK"]
    ok.sort(key=lambda r: r["index"])
    order, seen, owner = [], set(), {}
    for r in ok:
        for ti, p in enumerate(r["trajectories"]):
            for pos, s in enumerate(p):
                if s not in seen:
                    seen.add(s); order.append(s)
                    owner[s] = {"src_index": r["index"], "traj": ti, "pos": pos,
                                "val": (r["index"] % VAL_EVERY == 0)}
                if len(order) >= TARGET_UNIQUE:
                    break
            if len(order) >= TARGET_UNIQUE:
                break
        if len(order) >= TARGET_UNIQUE:
            break
    print(f"{len(ok)} sources -> {len(order):,} unique states taken "
          f"(target {TARGET_UNIQUE:,})", flush=True)

    labels, failed = {}, 0
    t0 = time.perf_counter()
    for i, s in enumerate(order):
        try:
            v = obj(s)
            # RAW higher-is-better for THIS benchmark: undo the bundle's
            # 1-gsk3b minimisation transform. Verified against known actives.
            labels[s] = [float(v[1]), float(1.0 - v[3])]      # (jnk3, gsk3b)
        except Exception:  # noqa: BLE001
            failed += 1
        if (i + 1) % 2000 == 0:
            print(f"  labelled {i+1:,}/{len(order):,}", flush=True)
    print(f"labelled {len(labels):,}, failed {failed}, "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    blob = {"horizon": HORIZON, "n_traj": N_TRAJ,
            "target_unique": TARGET_UNIQUE, "val_every": VAL_EVERY,
            "objective_order": ["jnk3", "gsk3b"],
            "objective_note": "RAW higher-is-better; gsk3b = 1 - bundle_value",
            "n_sources_used": len({owner[s]['src_index'] for s in labels}),
            "n_labelled": len(labels), "n_failed": failed,
            "labels": labels, "owner": {s: owner[s] for s in labels},
            "trajectories": [{"index": r["index"], "source": r["source"],
                              "paths": r["trajectories"]} for r in ok]}
    p = Path(RUN_ROOT) / OUT_DIR / "corpus_10k.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(json.dumps(blob).encode()))
    artifact_volume.commit()
    return {"n_labelled": len(labels), "n_failed": failed,
            "n_sources_used": blob["n_sources_used"]}


@app.function(image=image, cpu=(0.25, 0.25), memory=2048, timeout=8 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    out, started = [], time.perf_counter()
    for r in rollout.map(tasks, order_outputs=False, return_exceptions=True,
                         wrap_returned_exceptions=False):
        if isinstance(r, dict):
            out.append(r)
            if len(out) % 25 == 0:
                print(f"  {len(out)}/{len(tasks)} sources "
                      f"{time.perf_counter()-started:.0f}s", flush=True)
        else:
            print(f"  FAILED {type(r).__name__}: {r}", flush=True)
    print(f"rollouts done: {len(out)}/{len(tasks)}", flush=True)
    return label.remote(out)


@app.local_entrypoint()
def main(n_sources: int = 280) -> None:
    root = Path(__file__).resolve().parents[1]
    P = json.loads((root / "docs/INVERSIONGNN_FROZEN_PROTOCOL.json").read_text())
    import csv, hashlib
    rows = list(csv.DictReader(open(root / P["zinc_pool"])))
    col = "smiles" if "smiles" in rows[0] else list(rows[0])[0]
    uniq = sorted({r[col].strip() for r in rows if r[col].strip()})
    init = set(P["init_bank"])
    pool = sorted((s for s in uniq if s not in init),
                  key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    srcs = pool[:int(n_sources)]
    tasks = [{"index": i, "source": s, "seed": 300000 + i}
             for i, s in enumerate(srcs)]
    print(f"INVGNN CORPUS: {len(tasks)} sources (frozen sha256 order, disjoint "
          f"from the 100-molecule init bank), H={HORIZON}, {N_TRAJ} rollouts "
          f"each.\nBuffer above the ~204 needed for 10,000 unique states; the "
          f"surplus only fills dedup shortfall and is truncated deterministically.")
    call = drive.spawn(tasks)
    print(f"spawned: {call.object_id}")
