"""Does the forward pass amortize across states? Systems only, no science.

The bounded kernel profile decomposed one enumeration (medians over 6 QED
sources):

    runtime.apply  (~600 graph rewrites)   4.33 s   61 %
    neural forward pass                    2.64 s   37 %
    canonical_state_key                    0.14 s    2 %
    coordinate loop                        0.01 s    0.1 %

Direct mark sampling removes the first, third and fourth -- exactly, since the
successor law is the pushforward of the normalized mark law. That leaves the
forward pass, and 2.64 s for a 6.18M-parameter model on ONE state is fixed
overhead, not arithmetic.

`prepare_factorized_mark_batch` already accepts a TUPLE of states;
`_one_state_batch` merely passes a 1-tuple. So the open question is whether
scoring N states in one forward costs materially less than N separate forwards.

WHY THIS IS NOT ASSUMED. `editing_v2_vectorized_successor_scoring` records the
opposite outcome on the TRAINING path: batch 32 -> 128 scaled backward 14.3x
for 4x the work and DROPPED throughput from 50.45 to 16.49 examples/s, because
the cost was autograd node count rather than FLOPs. That was backward, not
inference, but it is precisely why a 50x projection must be measured before it
is believed.

Reports per-state seconds against batch size. Inspects no benchmark outcome.
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
    ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"}
).add_local_file(ROOT / "data/jin/qed_test.txt",
                 str(REMOTE_ROOT / "qed_test.txt"), copy=True)

app = modal.App("batch-scaling-probe")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT = 0.5
CANONICAL_SLOTS = 48
SIZES = (1, 2, 4, 8, 16, 20, 32)


@app.function(image=image, cpu=(1.0, 1.0), memory=16384, timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def probe() -> dict[str, Any]:
    import sys

    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
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
    from compose_v4.experiments.production_successor_kernel import (
        _masked_family_logits,
        operator_capability_batch_kwargs,
        prepare_factorized_mark_batch,
    )

    artifact_volume.reload()
    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    src_obj = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]), repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        src_obj, materialized_state=bundle)
    model = runtime.model
    ckpt = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["selected_model_state"], strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)

    smis = [l.strip() for l in
            (REMOTE_ROOT / "qed_test.txt").read_text().splitlines() if l.strip()]
    states = [pad_molecular_graph(smiles_to_molecular_graph(s), CANONICAL_SLOTS)
              for s in smis[:max(SIZES)]]

    def score(batch_states):
        n = len(batch_states)
        t0 = time.perf_counter()
        b = prepare_factorized_mark_batch(
            tuple(batch_states), tuple(float(TIME_POINT) for _ in range(n)),
            tuple(None for _ in range(n)), tuple(None for _ in range(n)),
            tuple(0.0 for _ in range(n)),
            use_aromatic_bond_view=True, ring_catalog=model.ring_catalog,
            **operator_capability_batch_kwargs(model.operator_capabilities))
        t_prep = time.perf_counter() - t0
        t0 = time.perf_counter()
        with torch.no_grad():
            nd, gs, pr = model._encode_batch(b)
            masks, logits, alz = model._action_tables(
                b, nd, gs, pr, require_exact_ring_support=False)
            en = torch.isfinite(alz)
            fl = _masked_family_logits(model._family_base_logits(b, gs), alz, en,
                                       rate_factorization=model.rate_factorization)
            torch.log_softmax(fl, dim=-1)
        return t_prep, time.perf_counter() - t0

    score(states[:1])          # warm caches so size 1 is not penalised
    rows = []
    for n in SIZES:
        tp, tf = score(states[:n])
        rows.append({"batch": n, "prepare_s": round(tp, 3), "forward_s": round(tf, 3),
                     "total_s": round(tp + tf, 3),
                     "per_state_s": round((tp + tf) / n, 4)})
        print(f"batch={n:>3}  prepare={tp:>6.2f}s  forward={tf:>6.2f}s  "
              f"total={tp+tf:>6.2f}s  per-state={(tp+tf)/n:>6.3f}s", flush=True)

    base = rows[0]["per_state_s"]
    best = min(rows, key=lambda r: r["per_state_s"])
    out = {"schema": "compose.kernel.batch_scaling", "rows": rows,
           "per_state_at_1": base, "best_batch": best["batch"],
           "best_per_state_s": best["per_state_s"],
           "amortization_speedup": round(base / best["per_state_s"], 2)}
    (Path(RUN_ROOT) / "batch_scaling_probe.json").write_text(json.dumps(out, indent=2))
    artifact_volume.commit()
    print(f"\nbest batch {best['batch']}: {best['per_state_s']:.3f}s/state "
          f"vs {base:.3f}s at batch 1  ->  {base/best['per_state_s']:.1f}x amortization")
    return out


@app.local_entrypoint()
def main() -> None:
    print("batch-scaling probe: does the forward pass amortize across states?")
    call = probe.spawn()
    print(f"spawned: {call.object_id}")
