"""Where do the ~5 seconds per step actually go? Measure, do not theorise.

Every speed claim tonight has been arithmetic from guessed millisecond costs,
and it has been wrong by 3x and by 40x in both directions. This times each phase
of a real step on a real source with the real frozen model, and separately tests
whether pinning torch to one thread is what makes it slow.

Answers exactly two questions and then exits:
  1. what fraction of a step is law enumeration vs apply vs encode vs props?
  2. does giving torch more threads help, and by how much?
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
app = modal.App("hphi-graph-encode-qual")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
REGION = (0.90, 0.40)
STEPS = 6          # enough to see the per-step cost; not a trajectory


def _load(threads: int):
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
    model.eval(); torch.set_grad_enabled(False)
    torch.set_num_threads(threads)
    return model, _default_rewrite_system(model)



@app.function(image=image, cpu=(2.0, 2.0), memory=6144, timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def qualify(smiles: list[str]) -> dict[str, Any]:
    """BITWISE parity + timing, both paths in ONE process.

    In-process on purpose: comparing two Modal runs would confound the change
    with cross-container float nondeterminism, which is what made the
    chemistry-cache parity unreadable.
    """
    import sys, time as _t

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.hphi_graph_encode import (
        build_graph_only_batch, encode_graph_only,
    )
    from compose_v4.experiments.production_successor_kernel import _one_state_batch

    model, _sys = _load(2)
    states = [pad_molecular_graph(smiles_to_molecular_graph(s), CANONICAL_SLOTS)
              for s in smiles]

    # --- existing path ---
    t0 = _t.perf_counter(); ref = []
    for st in states:
        b = _one_state_batch(model, st, float(TIME_POINT), prepared_batch=None)
        with torch.no_grad():
            _n, g, _p = model._encode_batch(b)
        ref.append(g[0].detach().cpu().numpy())
    t_ref = _t.perf_counter() - t0

    # --- graph-only path, one at a time (same granularity) ---
    t0 = _t.perf_counter(); new = []
    for st in states:
        new.append(encode_graph_only(model, [st], float(TIME_POINT))[0])
    t_new = _t.perf_counter() - t0

    # --- graph-only, batched (only legitimate if per-state is already exact) ---
    t0 = _t.perf_counter()
    batched = encode_graph_only(model, states, float(TIME_POINT))
    t_batch = _t.perf_counter() - t0

    d1 = float(max(np.abs(a - b).max() for a, b in zip(ref, new)))
    d2 = float(max(np.abs(a - b).max() for a, b in zip(ref, batched)))
    out = {"n": len(states),
           "existing_s": round(t_ref, 3), "graph_only_s": round(t_new, 3),
           "graph_only_batched_s": round(t_batch, 3),
           "speedup_single": round(t_ref / t_new, 2) if t_new else None,
           "speedup_batched": round(t_ref / t_batch, 2) if t_batch else None,
           "max_abs_diff_single": d1, "max_abs_diff_batched": d2,
           "bitwise_single": d1 == 0.0, "bitwise_batched": d2 == 0.0}
    print(f"  existing    {t_ref:7.2f}s  ({1000*t_ref/len(states):.0f} ms/state)")
    print(f"  graph-only  {t_new:7.2f}s  ({1000*t_new/len(states):.0f} ms/state)"
          f"  speedup {t_ref/max(t_new,1e-9):.2f}x  maxdiff {d1:.3e}")
    print(f"  batched     {t_batch:7.2f}s  ({1000*t_batch/len(states):.0f} ms/state)"
          f"  speedup {t_ref/max(t_batch,1e-9):.2f}x  maxdiff {d2:.3e}")
    print(f"  BITWISE single={d1 == 0.0}  batched={d2 == 0.0}")
    return out


@app.local_entrypoint()
def main(n: int = 40) -> None:
    import json
    # Real molecular states from the banked SMC reference run.
    rec = json.loads((Path(__file__).resolve().parents[1]
                      / "artifacts/runs/smc_ref/replicates/000_smc_00.json"
                      ).read_text())["record"]
    seen, mols = set(), []
    for t in rec["transitions"]:
        for s in (t["x"], t["y"]):
            if s not in seen:
                seen.add(s); mols.append(s)
    mols = mols[:n]
    print(f"qualifying graph-only encode on {len(mols)} banked molecular states")
    print(json.dumps(qualify.remote(mols), indent=2))
