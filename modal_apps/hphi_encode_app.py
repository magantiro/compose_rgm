"""Encode corpus states with frozen R_theta. PERSIST PER SHARD. Then exit.

WHY THIS IS ITS OWN JOB
-----------------------
The first trainer did the 70-minute encode and the cheap, fragile training in
the SAME function. Encoding can never change -- it is frozen R_theta over a
frozen corpus, the same answer every time. Training obviously needed iteration.
Coupling them meant every training fix destroyed the encode, and it did: 4,206s
across 80 containers, 93.5 core-hours, thrown away twice.

So: encode -> persist -> EXIT. Training becomes a separate minutes-long step
that loads the cache and can be rerun freely at any hyperparameter.

EVERY SHARD WRITES ITS OWN FILE THE MOMENT IT FINISHES
-------------------------------------------------------
Not one merged write at the end. A merged write only survives if ALL 80 shards
return, which is precisely the failure that already happened. With per-shard
persistence, stopping mid-flight keeps everything completed so far, and a rerun
skips those shards entirely.

Idempotent: a shard whose file already exists is skipped without loading the
model, so reruns cost nothing for completed work.
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

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"}
)

app = modal.App("hphi-encode")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
CORPUS = "hphi_rollout_corpus/train_1024x02_H24.json.gz"
SHARD_DIR = "hphi_v2/embeddings"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
N_SHARDS = 80

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
    _RT["model"] = model
    return _RT


@app.function(image=image, cpu=(1.0, 1.0), memory=8192, timeout=6 * 60 * 60,
              max_containers=N_SHARDS, retries=3,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def encode_shard(task: dict[str, Any]) -> dict[str, Any]:
    """One shard. Writes its OWN file, commits, and returns only a summary."""
    import sys

    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    shard_id = int(task["shard"])
    out_p = Path(RUN_ROOT) / SHARD_DIR / f"shard_{shard_id:03d}.json.gz"

    artifact_volume.reload()
    if out_p.exists():                      # IDEMPOTENT: never redo paid work
        return {"shard": shard_id, "status": "ALREADY_DONE"}

    model = _runtime()["model"]
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import _one_state_batch

    t0 = time.perf_counter()
    out: dict[str, list[float]] = {}
    failed = 0
    for smi in task["smiles"]:
        try:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            b = _one_state_batch(model, st, float(TIME_POINT), prepared_batch=None)
            with torch.no_grad():
                _n, g, _p = model._encode_batch(b)
            out[smi] = g[0].detach().cpu().numpy().tolist()
        except Exception:  # noqa: BLE001
            failed += 1

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_bytes(gzip.compress(json.dumps(out).encode()))
    artifact_volume.commit()                # FLUSHED before this worker exits
    secs = time.perf_counter() - t0
    print(f"shard {shard_id:03d}: {len(out)} encoded, {failed} failed, "
          f"{secs:.0f}s -> persisted", flush=True)
    return {"shard": shard_id, "n": len(out), "failed": failed,
            "seconds": round(secs, 1), "status": "OK"}


@app.function(image=image, cpu=(1.0, 1.0), memory=4096, timeout=8 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive() -> dict[str, Any]:
    artifact_volume.reload()
    blob = json.loads(gzip.decompress(
        (Path(RUN_ROOT) / CORPUS).read_bytes()).decode())
    srcs = [r for r in blob["results"] if r.get("status") == "OK"]
    need = sorted({s for r in srcs for t in r["trajectories"] for s in t["path"]})
    print(f"{len(need):,} unique states over {len(srcs)} sources", flush=True)

    done = set()
    sd = Path(RUN_ROOT) / SHARD_DIR
    if sd.exists():
        done = {int(p.stem.split("_")[1]) for p in sd.glob("shard_*.json.gz")}
        if done:
            print(f"RESUMING: {len(done)}/{N_SHARDS} shards already persisted",
                  flush=True)

    tasks = [{"shard": i, "smiles": need[i::N_SHARDS]}
             for i in range(N_SHARDS) if i not in done]
    started, ok = time.perf_counter(), 0
    for r in encode_shard.map(tasks, order_outputs=False, return_exceptions=True):
        if isinstance(r, dict):
            ok += 1
            print(f"  {ok}/{len(tasks)} shards  "
                  f"{time.perf_counter()-started:.0f}s", flush=True)
    total = len({int(p.stem.split('_')[1])
                 for p in (Path(RUN_ROOT) / SHARD_DIR).glob("shard_*.json.gz")})
    print(f"DONE {total}/{N_SHARDS} shards persisted in "
          f"{time.perf_counter()-started:.0f}s", flush=True)
    return {"shards_persisted": total, "of": N_SHARDS}


@app.local_entrypoint()
def main() -> None:
    print("ENCODE ONLY. Every shard persists itself, then the job exits.")
    print("Idempotent and resumable: completed shards are never re-encoded.")
    call = drive.spawn()
    print(f"spawned: {call.object_id}")
