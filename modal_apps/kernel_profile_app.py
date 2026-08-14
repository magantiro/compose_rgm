"""Bounded profile of ONE fiber enumeration. Systems only, no science.

The master plan has carried "bounded kernel profile -- ~80% of remaining cost
is unmeasured" as an open item for a while. It is now load-bearing: the frozen
GrIDDD policy needs 96,000 enumerations at a measured 12.8 s each, so knowing
what those seconds are made of decides whether that number is inevitable.

`canonical_successor_result` has exactly two parts:

    marked_law = enumerate_factorized_marked_law(model, state, time)   # NEURAL
    for mark in marked_law.marks:                                      # PYTHON
        successor = runtime.apply(state, rule, action)                 #   rewrite
        key = canonical_state_key(successor)                           #   canonical

If the NEURAL half dominates, the enumeration cost is essentially irreducible
for a full-fiber policy and batching is the lever.

If the PYTHON MARK LOOP dominates, there is a much larger prize. Each mark
carries a normalized `log_probability`, and the successor law is the pushforward
of the mark law with self-loops removed. So

    sample mark ~ marked_law  ->  apply ONE rewrite  ->  canonicalize
    reject self-loops         ->  accept with probability QED(y)

draws EXACTLY from `pi(y|x) proportional to R_theta(y|x) * QED(y)` while
applying ~1 rewrite instead of ~600. Exact, parameter-free, same policy.

This app measures which case we are in. It inspects NO benchmark outcome.
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

app = modal.App("kernel-profile")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT = 0.5
CANONICAL_SLOTS = 48


@app.function(image=image, cpu=(1.0, 1.0), memory=8192, timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def profile(n_sources: int = 8) -> dict[str, Any]:
    import sys

    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED

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
        _default_rewrite_system,
        canonical_state_key,
        canonical_successor_result,
        enumerate_factorized_marked_law,
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
    rows = []
    system = _default_rewrite_system(model)

    for smi in smis[:n_sources]:
        state = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)

        # --- decompose the "neural" half: forward pass vs coordinate loop ---
        from compose_v4.experiments.production_successor_kernel import (
            _coordinate_action, _masked_family_logits, _one_state_batch,
            _TABLE_FAMILIES, MARK_RULE_TO_INDEX,
        )
        t0 = time.perf_counter()
        batch = _one_state_batch(model, state, float(TIME_POINT), prepared_batch=None)
        with torch.no_grad():
            nd, gs, pr = model._encode_batch(batch)
            masks, logits, alz = model._action_tables(
                batch, nd, gs, pr, require_exact_ring_support=False)
            en = torch.isfinite(alz)
            fl = _masked_family_logits(model._family_base_logits(batch, gs), alz, en,
                                       rate_factorization=model.rate_factorization)
            flp = torch.log_softmax(fl, dim=-1)[0]
        t_forward = time.perf_counter() - t0

        t_coord_action, t_tensor_idx, n_coords = 0.0, 0.0, 0
        for fam, tab in _TABLE_FAMILIES:
            fi = MARK_RULE_TO_INDEX[fam]
            tm, tl = masks[tab][0], logits[tab][0]
            coords = torch.nonzero(tm, as_tuple=False)
            n_coords += len(coords)
            for rc in coords:
                a = time.perf_counter()
                c = tuple(int(v) for v in rc)
                _coordinate_action(model, state, batch, family_name=fam,
                                   table_name=tab, coordinate=c)
                t_coord_action += time.perf_counter() - a
                a = time.perf_counter()
                float(tl[c]); float(flp[fi]); float(alz[0, fi])
                t_tensor_idx += time.perf_counter() - a

        t0 = time.perf_counter()
        law = enumerate_factorized_marked_law(model, state, float(TIME_POINT))
        t_neural = time.perf_counter() - t0

        t0 = time.perf_counter()
        keys, t_apply, t_canon = [], 0.0, 0.0
        for mark in law.marks:
            a = time.perf_counter()
            succ = system.apply(state, mark.executor_rule_name, mark.action)
            t_apply += time.perf_counter() - a
            a = time.perf_counter()
            keys.append(canonical_state_key(succ))
            t_canon += time.perf_counter() - a
        t_loop = time.perf_counter() - t0

        t0 = time.perf_counter()
        res = canonical_successor_result(model, state, float(TIME_POINT))
        t_full = time.perf_counter() - t0

        uniq = {k for k in keys}
        t0 = time.perf_counter()
        for k in list(uniq)[:200]:
            m = Chem.MolFromSmiles(k)
            if m is not None:
                QED.qed(m)
        t_qed200 = time.perf_counter() - t0

        rows.append({
            "smiles": smi, "n_marks": len(law.marks),
            "n_successors": len(res.batch.successors),
            "neural_s": round(t_neural, 3), "mark_loop_s": round(t_loop, 3),
            "fwd_s": round(t_forward, 3),
            "coord_action_s": round(t_coord_action, 3),
            "tensor_idx_s": round(t_tensor_idx, 3),
            "apply_s": round(t_apply, 3), "canon_s": round(t_canon, 3),
            "full_kernel_s": round(t_full, 3),
            "qed_per_200_s": round(t_qed200, 3),
        })
        print(f"marks={len(law.marks):>4} | fwd={t_forward:>5.2f} "
              f"coord_act={t_coord_action:>6.2f} tidx={t_tensor_idx:>5.2f} "
              f"|| apply={t_apply:>6.2f} canon={t_canon:>5.2f} || full={t_full:>6.2f}",
              flush=True)

    import statistics as st
    med_n = st.median(r["neural_s"] for r in rows)
    med_l = st.median(r["mark_loop_s"] for r in rows)
    out = {
        "schema": "compose.kernel.profile",
        "rows": rows,
        "median_neural_s": med_n,
        "median_mark_loop_s": med_l,
        "neural_fraction": med_n / (med_n + med_l) if (med_n + med_l) else None,
        "verdict": ("NEURAL-DOMINATED -> batching is the lever"
                    if med_n > med_l else
                    "MARK-LOOP-DOMINATED -> direct mark sampling avoids ~600 rewrites"),
    }
    outp = Path(RUN_ROOT) / "kernel_profile.json"
    outp.write_text(json.dumps(out, indent=2))
    artifact_volume.commit()
    print(f"\nmedian neural {med_n:.2f}s | median mark-loop {med_l:.2f}s")
    print(out["verdict"])
    return out


@app.local_entrypoint()
def main(n: int = 8) -> None:
    print(f"bounded kernel profile on {n} QED-task sources; systems only")
    call = profile.spawn(n)
    print(f"spawned: {call.object_id}")
