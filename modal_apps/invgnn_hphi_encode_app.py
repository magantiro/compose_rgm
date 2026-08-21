"""Encode the h_phi training states -> persist -> exit.

Its own job by design. The embeddings are deterministic and are the ONLY thing
blocking gate 2; holding them in memory until the end of a longer job is exactly
how banked work gets destroyed. This writes and commits before it returns.

WHY THIS EMITS THREE THINGS, NOT ONE.

The previously trained h_phi lost to a trivial persistence baseline (Spearman
0.487 vs 0.680). The brute-force lookahead measurement has since ruled out the
benign explanation: at b=8 the immediate score has median rank correlation
0.32-0.37 with TRUE future value and picks the true-best successor only 12-21%
of the time, so the quantity h_phi amortises differs from the immediate score by
a lot. That leaves the amortiser, which the amendment's gate rule directs us to
troubleshoot.

The most likely feature-side cause is that h_phi saw only the R_theta graph
embedding, which was trained to predict TRANSITIONS, not properties. But
g_lambda is a function of F_psi_hat, which reads Morgan bits. So we also persist
the state's own F_psi_hat output and its own g_lambda, i.e. the persistence
baseline itself. Handing the model the baseline turns the task into learning the
CORRECTION over it rather than rediscovering it from an embedding that may not
carry the information at all.

Persisting all three costs one pass instead of three and lets the feature set be
ablated later WITHOUT re-encoding: emb-only reproduces the failed configuration
exactly, which is what makes the comparison a diagnosis rather than a patch.

NO ORACLE CALLS. F_psi_hat is the frozen surrogate; nothing here touches the
benchmark's 10,000-call budget. CPU only.
"""

from __future__ import annotations

import json
import time
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
                         str(REMOTE_ROOT / "docs/INVERSIONGNN_FROZEN_PROTOCOL.json"),
                         copy=True)
         .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"}))

app = modal.App("invgnn-hphi-encode")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
MEM_MIB = int(4.5 * 1024)

#: [ R_theta graph embedding | F_psi_hat(x) | g_lambda(x) ]
DIMS = (256, 2, 5)

_RT: dict[str, Any] = {}


def _runtime():
    if "model" in _RT:
        return _RT
    import sys

    import torch
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source)
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime, load_materialized_scorer_state)
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME)
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
    model.eval()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    _RT["model"] = model
    return _RT


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=60 * 60,
              max_containers=100,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def encode(task: dict[str, Any]) -> dict[str, Any]:
    import sys

    import numpy as np
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import _one_state_batch

    model = _runtime()["model"]

    # Frozen surrogate and terminal desirability, byte-identical to the corpus
    # app that produced the training targets.  Reproduced rather than imported
    # so this job has no dependency on that module's import surface.
    P = json.loads((REMOTE_ROOT / "docs/INVERSIONGNN_FROZEN_PROTOCOL.json").read_text())
    prefs = np.array(P["preferences_unit_circle"], dtype=np.float32)
    taus = np.array(task["taus"], dtype=np.float32)
    Lstar = np.array(task["Lstar"], dtype=np.float32)
    ck = torch.load(REMOTE_ROOT / "docs/INVERSIONGNN_FPSI.pt", map_location="cpu")
    F = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.1),
                      nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2))
    F.load_state_dict(ck["state"])
    F.eval()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    out, t0, failed = [], time.perf_counter(), []
    for s in task["states"]:
        m = Chem.MolFromSmiles(s)
        if m is None:
            failed.append(s)
            continue
        fp = gen.GetFingerprintAsNumPy(m).astype(np.float32)
        with torch.no_grad():
            fh = F(torch.tensor(fp[None, :])).numpy()[0]          # (2,)
        L = np.array([float(np.max(prefs[k] * (1.0 - fh))) for k in range(5)])
        g = np.exp(-(L - Lstar) / taus)                            # (5,)

        stt = pad_molecular_graph(smiles_to_molecular_graph(s), CANONICAL_SLOTS)
        b = _one_state_batch(model, stt, float(TIME_POINT), prepared_batch=None)
        with torch.no_grad():
            _n, gemb, _p = model._encode_batch(b)
        emb = gemb[0].detach().cpu().numpy().astype(np.float32)    # (256,)
        if emb.shape[0] != DIMS[0]:
            raise RuntimeError(f"embedding width {emb.shape[0]} != {DIMS[0]}")
        out.append((s, np.concatenate([emb, fh.astype(np.float32),
                                       g.astype(np.float32)]).astype(np.float32)))
    return {"pairs": [(s, v.tolist()) for s, v in out], "failed": failed,
            "seconds": round(time.perf_counter() - t0, 1)}


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=2 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict], out_name: str) -> dict[str, Any]:
    import numpy as np

    store: dict[str, Any] = {}
    failed: list[str] = []
    t0 = time.perf_counter()
    for r in encode.map(tasks, order_outputs=False, return_exceptions=True,
                        wrap_returned_exceptions=False):
        if isinstance(r, dict):
            for s, v in r["pairs"]:
                store[s] = np.asarray(v, dtype=np.float32)
            failed.extend(r["failed"])
            print(f"  {len(store)} encoded  {time.perf_counter()-t0:.0f}s", flush=True)
        else:
            print(f"  FAILED {type(r).__name__}: {r}", flush=True)

    # Persist and commit BEFORE returning.  Nothing here is recomputable for
    # free, and a driver that dies holding results destroys the whole job.
    p = Path(RUN_ROOT) / "invgnn_v1" / f"{out_name}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, __dims__=np.array(DIMS, dtype=np.int32), **store)
    artifact_volume.commit()
    print(f"wrote {p}  ({p.stat().st_size:,} bytes)", flush=True)
    return {"n_encoded": len(store), "failed": failed, "dims": list(DIMS),
            "path": str(p), "seconds": round(time.perf_counter() - t0, 1)}


@app.local_entrypoint()
def main(out_name: str = "hphi_embeddings_0600", chunk: int = 50,
         states_file: str = "docs/INVGNN_HPHI_TRAIN_STATES.json") -> None:
    root = Path(__file__).resolve().parents[1]
    D = json.loads((root / states_file).read_text())
    states = list(D["states"])
    tasks = [{"states": states[i:i + chunk], "taus": D["taus"], "Lstar": D["Lstar"]}
             for i in range(0, len(states), chunk)]
    print(f"{len(states)} states in {len(tasks)} chunks of <= {chunk}")
    print("CPU only, no oracle calls. Encode -> persist -> exit.")
    o = drive.remote(tasks, out_name)
    print(json.dumps(o, indent=1))
