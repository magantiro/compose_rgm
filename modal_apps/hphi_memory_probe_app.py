"""Where do the 3.97 GiB go, and how much can be released after startup?

Memory is roughly HALF the bill for these runs: at Modal's rates a 1-core
container costs $0.047/hour and 6 GiB costs $0.048/hour. Peak RSS was measured
at 3.97 GiB and is FLAT in worker count -- 1 worker and 8 workers both land
there -- so it is fixed startup footprint, not per-slot state, and shrinking it
scales every run we do from here.

The suspicion is startup scaffolding that is not needed afterwards.
`open_process_v2_t1_source` authenticates the whole Active8 stream (751 MB
across 1641 files) and `load_materialized_scorer_state` reads a 49 MB bundle;
once the model is built and the checkpoint is loaded, none of that is required
to enumerate a law.

Measures RSS at each stage, drops the startup references, collects, and measures
again -- then runs a real law call to prove the model still works after the
release. A memory saving that breaks inference is not a saving.

Read-only. Persists nothing.
"""

from __future__ import annotations

import gc
import json
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
app = modal.App("hphi-memory-probe")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48


def _rss_gib() -> float:
    """Current RSS. /proc/self/statm is in PAGES, not kilobytes."""
    with open("/proc/self/statm") as fh:
        pages = int(fh.read().split()[1])
    return pages * 4096 / (1024 ** 3)


@app.function(image=image, cpu=(1.0, 1.0), memory=6144, timeout=30 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def probe(smiles: str) -> dict[str, Any]:
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
        _default_rewrite_system,
        enumerate_factorized_marked_law,
    )

    stages: list[tuple[str, float]] = [("imports", _rss_gib())]

    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    src = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]), repo_root=REMOTE_ROOT)
    stages.append(("after open_process_v2_t1_source", _rss_gib()))

    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    stages.append(("after load_materialized_scorer_state", _rss_gib()))

    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        src, materialized_state=bundle)
    model = runtime.model
    stages.append(("after build_runtime", _rss_gib()))

    ck = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["selected_model_state"], strict=True)
    model.eval(); torch.set_grad_enabled(False); torch.set_num_threads(1)
    system = _default_rewrite_system(model)
    stages.append(("after checkpoint + system", _rss_gib()))

    # --- release the startup scaffolding -----------------------------------
    del ck, bundle, src, runtime, _b, _c
    gc.collect()
    stages.append(("after del + gc.collect", _rss_gib()))

    # --- prove the model still works after the release ----------------------
    st = pad_molecular_graph(smiles_to_molecular_graph(smiles), CANONICAL_SLOTS)
    law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
    stages.append(("after a real law call", _rss_gib()))
    ok = bool(law.marks) and system is not None

    print(f"\n{'stage':<44}{'RSS GiB':>10}{'delta':>10}")
    prev = 0.0
    for name, v in stages:
        print(f"  {name:<42}{v:10.2f}{v-prev:10.2f}")
        prev = v
    peak = max(v for _, v in stages)
    after = stages[-1][1]
    print(f"\n  peak {peak:.2f} GiB   after release {stages[-2][1]:.2f} GiB"
          f"   steady {after:.2f} GiB")
    print(f"  law call after release: {len(law.marks)} marks  "
          f"{'OK' if ok else 'BROKEN'}")
    return {"stages": stages, "peak_gib": peak, "steady_gib": after,
            "law_ok": ok, "n_marks": len(law.marks)}


@app.local_entrypoint()
def main() -> None:
    smi = [s.strip() for s in
           (Path(__file__).resolve().parents[1]
            / "data/jin/dev_panel_qed_64.txt").read_text().split("\n")
           if s.strip()][0]
    out = probe.remote(smi)
    Path("docs/MEMORY_PROBE.json").write_text(json.dumps(out, indent=1))
    print(f"\npeak {out['peak_gib']:.2f} GiB, steady {out['steady_gib']:.2f} GiB, "
          f"law {'OK' if out['law_ok'] else 'BROKEN'}")
