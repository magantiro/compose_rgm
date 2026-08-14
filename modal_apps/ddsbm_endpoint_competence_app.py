"""COMPOSE on DDSBM's ZINC logP 2->4 task. TIER-1: COMPOSE runs alone.

Protocol frozen in `docs/DDSBM_ENDPOINT_COMPETENCE_PROTOCOL.md` before any
outcome existed. Nothing here may be tuned against DDSBM's published table.

THE OBJECTIVE IS A TRANSPORT ADAPTER, NOT A POINT TARGET

    tau(x0)   = logP(x0) + 2
    u(y; x0)  = -|logP(y) - tau(x0)|

DDSBM's ZINC experiment moves a source marginal at logP~N(2, 0.5) to a target at
N(4, 0.5) and scores Wasserstein-1 between MARGINALS. A point objective at 4
would collapse the generated spread toward the mode and inflate W1 against a
target whose sd is 0.5. Verified on the frozen test split before launch: source
mean 2.015 sd 0.499, tau mean 4.015 sd 0.499 -- the +2 shift reproduces the
prescribed target in BOTH moments. Absolute error, not squared, because W1 is an
L1 transport metric.

BARRED: the CSV's randomly paired PRB-SMI target. That coupling exists for
DDSBM's training; the scientific task is transport between marginals, and the
pair would inject per-source information not intrinsic to it. This app reads
REF-SMI and REF-LOGP only.

FROZEN, NO SWEEP: H = 6, COMPOSE's native horizon, taken from our framework
rather than optimised against their table. Greedy closed-loop only -- P3 showed
greedy carries the mechanism at ~22x lower cost, and a competence experiment does
not deploy the expensive controller to chase a benchmark. R_theta frozen; no
objective-specific parameter update anywhere.

REPRESENTABILITY GATE PASSED BEFORE LAUNCH: 5984/5984 = 100.00%, zero exclusions,
so this is a true head-to-head against DDSBM's published full-test numbers.

PERSIST EVERYTHING, ONCE. The P0c probe had to re-derive fibers the committed
Pareto runs had paid for and discarded. Not repeating that: every trajectory,
endpoint, property triple, counter and failure is written, so FCD/NSPDK and any
later analysis are computable WITHOUT another generation run.
"""

from __future__ import annotations

import csv
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

DDSBM_CSV = "local_runtime/ddsbm/DDSBM-main/data/raw/ZINC250k_logp_2_4_random_matched_no_nH.csv"

image = (
    _base_image
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "4"})
    .add_local_file(ROOT / DDSBM_CSV, str(REMOTE_ROOT / "ddsbm_pairs.csv"), copy=True)
)

app = modal.App("ddsbm-endpoint-competence")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
OUT_DIR = "ddsbm_endpoint_competence"
TIME_POINT = 0.5
CANONICAL_SLOTS = 48
HORIZON = 6
#: DDSBM's split: 29,920 pairs, last 20% is the 5,984-molecule test set.
TRAIN_ROWS = 23936
LOGP_SHIFT = 2.0
CHECKPOINT_EVERY = 25


@app.function(
    image=image, cpu=8.0, memory=16 * 1024, timeout=6 * 60 * 60,
    max_containers=24, retries=5,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_batch(task: dict[str, Any]) -> dict[str, Any]:
    import sys

    import numpy as np
    import torch
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, QED

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit.Contrib.SA_Score import sascorer  # noqa: E402

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
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
    from compose_v4.experiments.pareto_control import _argmin_stable
    from compose_v4.experiments.production_successor_kernel import (
        canonical_successor_result,
    )

    RDLogger.DisableLog("rdApp.*")
    started = time.perf_counter()
    artifact_volume.reload()

    shard_id = int(task["shard"])
    out = Path(RUN_ROOT) / task.get("out_dir", OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    shard_path = out / f"{shard_id:03d}.json.gz"
    partial = out / f"{shard_id:03d}.partial.json.gz"
    if shard_path.exists():
        print(f"[{shard_id:03d}] exists, skipping", flush=True)
        return {"shard": shard_id, "skipped": True}

    rows = list(csv.DictReader(open(REMOTE_ROOT / "ddsbm_pairs.csv")))[TRAIN_ROWS:]
    mine = rows[task["lo"]:task["hi"]]
    print(f"[{shard_id:03d}] {len(mine)} sources", flush=True)

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

    def props(smi: str) -> dict[str, float] | None:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return {"logp": float(Crippen.MolLogP(mol)),
                "qed": float(QED.qed(mol)),
                "sa": float(sascorer.calculateScore(mol)),
                "heavy": int(mol.GetNumHeavyAtoms())}

    results, resumed = [], 0
    if partial.exists():
        results = json.loads(gzip.decompress(partial.read_bytes()))["results"]
        resumed = len(results)
        print(f"[{shard_id:03d}] RESUMED at {resumed}", flush=True)

    kernel_calls = 0
    for i, row in enumerate(mine[resumed:], start=resumed):
        src = row["REF-SMI"]
        p0 = props(src)
        if p0 is None:
            results.append({"source": src, "status": "SOURCE_UNPARSEABLE"})
            continue
        tau = p0["logp"] + LOGP_SHIFT

        # Greedy closed-loop: commit the fiber argmin of |logP - tau| each step.
        cur, traj, valid_all, dead = src, [src], True, None
        cache: dict[str, float] = {}
        for step in range(HORIZON):
            try:
                state = pad_molecular_graph(smiles_to_molecular_graph(cur), CANONICAL_SLOTS)
                res = canonical_successor_result(model, state, float(TIME_POINT))
                kernel_calls += 1
            except Exception as e:  # noqa: BLE001
                dead = f"kernel:{type(e).__name__}"
                break
            keys = [s.key for s in res.batch.successors]
            if not keys:
                dead = "empty_fiber"
                break
            scores = []
            for k in keys:
                if k not in cache:
                    m = Chem.MolFromSmiles(k)
                    cache[k] = (abs(float(Crippen.MolLogP(m)) - tau)
                                if m is not None else 1e9)
                scores.append(cache[k])
            cur = keys[_argmin_stable(np.asarray(scores, float), keys)]
            traj.append(cur)
            if Chem.MolFromSmiles(cur) is None:
                valid_all = False

        pH = props(cur)
        results.append({
            "source": src, "endpoint": cur, "trajectory": traj,
            "tau": tau, "edits": len(traj) - 1, "dead_end": dead,
            "all_states_valid": valid_all,
            "endpoint_valid": pH is not None,
            "source_props": p0, "endpoint_props": pH,
            "ddsbm_ref_logp": float(row["REF-LOGP"]),
            "size_change": (pH["heavy"] - p0["heavy"]) if pH else None,
            "candidates_scored": len(cache),
        })
        if (i + 1) % CHECKPOINT_EVERY == 0:
            partial.write_bytes(gzip.compress(json.dumps(
                {"results": results}, default=float).encode()))
            artifact_volume.commit()
            print(f"[{shard_id:03d}] {i+1}/{len(mine)} kernel={kernel_calls} "
                  f"{time.perf_counter()-started:.0f}s", flush=True)

    payload = {
        "schema": "compose.ddsbm.endpoint_competence",
        "status": "TIER1_EXTERNAL_PROTOCOL", "shard": shard_id,
        "protocol": "docs/DDSBM_ENDPOINT_COMPETENCE_PROTOCOL.md",
        "objective": "u(y;x0) = -|logP(y) - (logP(x0) + 2)|",
        "horizon": HORIZON, "controller": "greedy_closed_loop",
        "r_theta": "frozen; no objective-specific update",
        "paired_target_used": False,
        "kernel_calls": kernel_calls,
        "seconds": round(time.perf_counter() - started, 1),
        "results": results,
    }
    shard_path.write_bytes(gzip.compress(json.dumps(payload, default=float).encode()))
    partial.unlink(missing_ok=True)
    artifact_volume.commit()
    print(f"[{shard_id:03d}] DONE {len(results)} kernel={kernel_calls} "
          f"{payload['seconds']:.0f}s", flush=True)
    return {"shard": shard_id, "n": len(results)}


@app.function(image=image, cpu=0.25, memory=2048, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]]) -> int:
    artifact_volume.reload()
    done = 0
    for _ in run_batch.map(tasks, order_outputs=False, return_exceptions=True):
        done += 1
    return done


@app.local_entrypoint()
def main(shards: int = 24) -> None:
    rows = list(csv.DictReader(open(ROOT / DDSBM_CSV)))
    n = len(rows) - TRAIN_ROWS
    assert n == 5984, f"expected DDSBM's 5,984 test sources, got {n}"
    step = (n + shards - 1) // shards
    tasks = [{"shard": s, "lo": s * step, "hi": min((s + 1) * step, n),
              "out_dir": OUT_DIR} for s in range(shards)]
    print(f"DDSBM ZINC logP 2->4, TIER-1: COMPOSE alone on {n} test sources")
    print(f"objective u = -|logP(y) - (logP(x0)+2)|  H={HORIZON}  greedy  frozen R_theta")
    print(f"{shards} shards of ~{step}; representability gate passed 5984/5984")
    call = drive.spawn(tasks)
    print(f"driver spawned: {call.object_id}")
