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
app = modal.App("hphi-law-profile")
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



@app.function(image=image, cpu=(2.0, 2.0), memory=6144, timeout=30 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def law_profile(smiles: list[str]) -> dict[str, Any]:
    """Where does enumerate_factorized_marked_law actually spend its time?"""
    import sys, time as _t

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.factorized_mark_conditional import (
        operator_capability_batch_kwargs,
    )
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law,
    )
    from compose_v4.model.factorized_tracelet_rate_model import (
        prepare_factorized_mark_batch,
    )

    model, _s = _load(2)
    states = [pad_molecular_graph(smiles_to_molecular_graph(s), CANONICAL_SLOTS)
              for s in smiles]

    # whole call
    t0 = _t.perf_counter()
    laws = [enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            for st in states]
    t_all = _t.perf_counter() - t0

    # just the batch builder it calls
    t0 = _t.perf_counter()
    for st in states:
        prepare_factorized_mark_batch(
            (st,), (float(TIME_POINT),), (None,), (None,), (0.0,),
            use_aromatic_bond_view=True, ring_catalog=model.ring_catalog,
            **operator_capability_batch_kwargs(model.operator_capabilities))
    t_prep = _t.perf_counter() - t0

    # just encode + action tables, on a prebuilt batch
    batches = [prepare_factorized_mark_batch(
        (st,), (float(TIME_POINT),), (None,), (None,), (0.0,),
        use_aromatic_bond_view=True, ring_catalog=model.ring_catalog,
        **operator_capability_batch_kwargs(model.operator_capabilities))
        for st in states]
    t0 = _t.perf_counter()
    for b in batches:
        with torch.no_grad():
            n_, g_, p_ = model._encode_batch(b)
            model._action_tables(b, n_, g_, p_)
    t_net = _t.perf_counter() - t0

    n = len(states)
    marks = float(np.mean([len(l.marks) for l in laws]))
    out = {"n": n, "mean_marks": marks,
           "law_total_ms": 1000*t_all/n, "prepare_ms": 1000*t_prep/n,
           "encode_plus_tables_ms": 1000*t_net/n,
           "remainder_ms": 1000*(t_all-t_prep-t_net)/n}
    print(f"  mean marks/law        {marks:.0f}")
    print(f"  FULL law call         {1000*t_all/n:8.1f} ms")
    print(f"    prepare batch       {1000*t_prep/n:8.1f} ms  ({100*t_prep/t_all:.0f}%)")
    print(f"    encode+action tables{1000*t_net/n:8.1f} ms  ({100*t_net/t_all:.0f}%)")
    print(f"    remainder (mark loop){1000*(t_all-t_prep-t_net)/n:7.1f} ms  "
          f"({100*(t_all-t_prep-t_net)/t_all:.0f}%)")
    return out


@app.local_entrypoint()
def main(n: int = 12) -> None:
    import json
    rec = json.loads((Path(__file__).resolve().parents[1]
                      / "artifacts/runs/smc_ref/replicates/000_smc_00.json").read_text())["record"]
    seen, mols = set(), []
    for t in rec["transitions"]:
        if t["x"] not in seen:
            seen.add(t["x"]); mols.append(t["x"])
    mols = mols[:n]
    print(f"profiling law enumeration internals on {len(mols)} banked states")
    print(json.dumps(law_profile.remote(mols), indent=2))
