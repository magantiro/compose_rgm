"""What is the FLOOR for one exact transition once the admission masks are gone?

Measured before the rejection sampler is written, because the sampler is a large
build and its value is capped by the work it CANNOT remove. If that floor is
~300 ms the build is worth it; if it is ~2 s it is not, and the right move is to
attack whatever dominates the floor instead.

The sampler removes the two expensive semantic admission masks -- cycle-close at
2747 ms and atom-restate at 1626 ms of a 4458 ms law call -- replacing them with
a handful of per-candidate resolver calls. Everything else it still has to do:

    graph-only encode + the neural heads
    compute_topology_features
    _graph_application_masks
    process_v2_atom_delete_mask
    enumerate_ring_restate_semantic_groups
    the semantic cycle-OPEN admission mask     (a candidate for rejection too)
    a few exact resolver calls on sampled candidates

So this times each retained component on real states, sums them, and prices two
variants: cycle-open kept as an enumerated mask, and cycle-open also moved to
rejection. Reported against the 4458 ms full law.

Read-only. Builds no sampler and persists nothing.
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
app = modal.App("hphi-sampler-floor")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48


@app.function(image=image, cpu=(1.0, 1.0), memory=6144, timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def floor(srcs: list[str]) -> dict[str, Any]:
    import sys
    from collections import defaultdict

    import numpy as np
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
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import is_element, smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.hphi_graph_encode import build_graph_only_batch
    from compose_v4.experiments.production_successor_kernel import (
        enumerate_factorized_marked_law,
    )
    from compose_v4.model import factorized_tracelet_rate_model as F
    from compose_v4.rewrite.operators import (
        CycleCloseEdge, resolve_cycle_close_edge,
    )
    from compose_v4.rewrite.semantic_cycle_close import (
        prepare_semantic_cycle_close_context,
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

    t: dict[str, list[float]] = defaultdict(list)
    full: list[float] = []

    for smi in srcs:
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)

        t0 = time.perf_counter()
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        full.append(time.perf_counter() - t0)

        # --- components the sampler STILL has to pay --------------------
        t0 = time.perf_counter()
        batch = build_graph_only_batch([st], [float(TIME_POINT)])
        with torch.no_grad():
            node, glob, pair = model._encode_batch(batch)
        t["graph encode + _encode_batch"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        F.compute_topology_features(st)
        t["compute_topology_features"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        atom_topo, _c1, _r1 = F.compute_topology_features(st)
        F._graph_application_masks(st, atom_topo, compute_cyclic_graft=True)
        t["_graph_application_masks"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        F.process_v2_atom_delete_mask(st)
        t["process_v2_atom_delete_mask"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        try:
            F.enumerate_ring_restate_semantic_groups(
                st, system=F.de_novo_rewrite_system())
        except Exception:
            pass
        t["enumerate_ring_restate_semantic_groups"].append(
            time.perf_counter() - t0)

        t0 = time.perf_counter()
        F._semantic_cycle_open_admission_mask(st)
        t["_semantic_cycle_open_admission_mask"].append(time.perf_counter() - t0)

        # A handful of exact resolver calls -- what rejection actually costs.
        t0 = time.perf_counter()
        ctx = prepare_semantic_cycle_close_context(st)
        real = [int(v) for v in np.flatnonzero(is_element(st.atom_types))]
        done = 0
        for a in real:
            for b in real:
                if a < b and int(st.bonds[a, b]) == 0 and done < 8:
                    resolve_cycle_close_edge(st, CycleCloseEdge(a, b, 1),
                                             context=ctx)
                    done += 1
        t["8 exact cycle-close resolutions"].append(time.perf_counter() - t0)

        print(f"  {smi[:34]:<34} full law {full[-1]*1e3:7.1f} ms", flush=True)

    mean = lambda v: sum(v) / len(v)  # noqa: E731
    fl = mean(full)
    print(f"\n{'component':<42}{'ms':>9}{'% of full law':>15}")
    keep = 0.0
    for k, v in t.items():
        m = mean(v)
        keep += m
        print(f"  {k:<40}{m*1e3:9.1f}{m/fl*100:14.1f}%")
    open_mask = mean(t["_semantic_cycle_open_admission_mask"])
    print(f"  {'-'*40}{'-'*9}")
    print(f"  {'FLOOR, cycle-open kept as a mask':<40}{keep*1e3:9.1f}"
          f"{keep/fl*100:14.1f}%")
    print(f"  {'FLOOR, cycle-open also by rejection':<40}"
          f"{(keep-open_mask)*1e3:9.1f}{(keep-open_mask)/fl*100:14.1f}%")
    print(f"  {'FULL LAW TODAY':<40}{fl*1e3:9.1f}{100.0:14.1f}%")
    print(f"\n  projected speedup   {fl/keep:5.1f}x   "
          f"(and {fl/(keep-open_mask):5.1f}x if cycle-open is rejected too)")
    print("\n  Target from the brief: <500 ms good, <200 ms excellent.")
    return {"full_law_s": fl, "components": {k: mean(v) for k, v in t.items()},
            "floor_s": keep, "floor_no_open_s": keep - open_mask,
            "speedup": fl / keep, "speedup_no_open": fl / (keep - open_mask)}


@app.local_entrypoint()
def main(n_states: int = 10) -> None:
    srcs = [s.strip() for s in
            (Path(__file__).resolve().parents[1]
             / "data/jin/dev_panel_qed_64.txt").read_text().split("\n")
            if s.strip()][:n_states]
    out = floor.remote(srcs)
    Path("docs/SAMPLER_FLOOR.json").write_text(json.dumps(out, indent=1))
    print(f"\nfloor {out['floor_s']*1e3:.0f} ms "
          f"({out['floor_no_open_s']*1e3:.0f} ms without the cycle-open mask) "
          f"vs {out['full_law_s']*1e3:.0f} ms today")
