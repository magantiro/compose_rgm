"""Is a persisted law a viable cross-attempt cache? Measure, then decide.

The in-process law cache is already banked at 2.28x, but it dies with the
container, so every one of the 20 benchmark attempts re-derives laws the earlier
attempts already paid for. Cross-attempt state overlap was measured at 34%
(median 35.7%, range 11.5-65.5%) between attempts 1 and 2, and should grow as
more attempts accumulate.

Caching only the probability vector does NOT work. `propose` samples an index
from the probabilities and then needs THAT MARK'S action to build the successor,
so a probabilities-only hit still forces a full enumeration to recover the
action -- which is the entire cost. The cache has to carry the marks themselves.

Three things decide whether that is viable, and all three are measurements:

  1. SIZE      bytes per law on the volume, times the states a 20-attempt run
               visits. A cache that does not fit is not a cache.
  2. SPEED     dump and load time against the ~1.4 s enumeration it replaces.
               A load that costs a meaningful fraction of an enumeration eats
               its own saving.
  3. EXACTNESS a round-tripped law must give bitwise-identical probabilities,
               the SAME sampled index under the same RNG state, and the SAME
               successor SMILES. Anything less silently changes trajectories.

Read-only. Persists nothing and changes no frozen artifact.
"""

from __future__ import annotations

import json
import pickle
import struct
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
app = modal.App("hphi-law-cache-probe")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48


def _load():
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
    return model, _default_rewrite_system(model)


@app.function(image=image, cpu=(1.0, 1.0), memory=6144, timeout=45 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def probe(srcs: list[str]) -> dict[str, Any]:
    import sys

    import numpy as np

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key,
        enumerate_factorized_marked_law,
    )

    model, system = _load()
    rows = []
    for i, smi in enumerate(srcs, 1):
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)

        t0 = time.perf_counter()
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        t_enum = time.perf_counter() - t0

        t0 = time.perf_counter()
        blob = pickle.dumps(law, protocol=pickle.HIGHEST_PROTOCOL)
        t_dump = time.perf_counter() - t0
        t0 = time.perf_counter()
        back = pickle.loads(blob)
        t_load = time.perf_counter() - t0

        # EXACTNESS. Same float64 bits, same sampled index under the same RNG
        # state, same successor SMILES. The index check matters most: identical
        # probabilities that arrive in a different ORDER would still redirect
        # the draw to a different mark.
        pa = np.array([m.probability for m in law.marks], float)
        pb = np.array([m.probability for m in back.marks], float)
        bits_equal = (len(pa) == len(pb)
                      and all(struct.pack("<d", x) == struct.pack("<d", y)
                              for x, y in zip(pa, pb)))
        idx_a = idx_b = -1
        succ_equal = None
        if len(pa) and bits_equal:
            na, nb = pa / pa.sum(), pb / pb.sum()
            idx_a = int(np.random.default_rng(12345).choice(len(na), p=na))
            idx_b = int(np.random.default_rng(12345).choice(len(nb), p=nb))
            ma, mb = law.marks[idx_a], back.marks[idx_b]
            ya = canonical_state_key(system.apply(st, ma.executor_rule_name, ma.action))
            yb = canonical_state_key(system.apply(st, mb.executor_rule_name, mb.action))
            succ_equal = bool(ya == yb)

        rows.append({"smiles": smi, "n_marks": len(law.marks),
                     "bytes": len(blob), "enum_s": t_enum,
                     "dump_s": t_dump, "load_s": t_load,
                     "bits_equal": bits_equal,
                     "index_equal": idx_a == idx_b,
                     "successor_equal": succ_equal})
        print(f"  {i:>3}/{len(srcs)} {len(law.marks):>4} marks  "
              f"{len(blob)/1024:7.1f} KiB  enum {t_enum*1e3:7.1f} ms  "
              f"load {t_load*1e3:6.2f} ms  "
              f"{'EXACT' if bits_equal and succ_equal else 'DIFFERS'}", flush=True)

    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n  # noqa: E731
    ok = all(r["bits_equal"] and r["index_equal"] and r["successor_equal"]
             for r in rows)
    print(f"\n{'='*66}")
    print(f"  mean law             {mean('n_marks'):8.0f} marks")
    print(f"  mean pickle          {mean('bytes')/1024:8.1f} KiB")
    print(f"  mean enumeration     {mean('enum_s')*1e3:8.1f} ms")
    print(f"  mean dump            {mean('dump_s')*1e3:8.2f} ms")
    print(f"  mean load            {mean('load_s')*1e3:8.2f} ms"
          f"   = {mean('load_s')/mean('enum_s')*100:.2f}% of an enumeration")
    print(f"  saving per hit       {(mean('enum_s')-mean('load_s'))*1e3:8.1f} ms")
    print(f"\n  EXACTNESS: {'PASSED' if ok else 'FAILED'}  "
          f"(float64 bits, sampled index, successor SMILES; {n} states)")
    return {"rows": rows, "exact": ok,
            "mean_bytes": mean("bytes"), "mean_enum_s": mean("enum_s"),
            "mean_load_s": mean("load_s")}


@app.local_entrypoint()
def main(n_states: int = 12) -> None:
    srcs = [s.strip() for s in
            (Path(__file__).resolve().parents[1]
             / "data/jin/dev_panel_qed_64.txt").read_text().split("\n")
            if s.strip()][:n_states]
    out = probe.remote(srcs)
    Path("docs/LAW_CACHE_PROBE.json").write_text(json.dumps(out, indent=1))
    print("\nwrote docs/LAW_CACHE_PROBE.json")
