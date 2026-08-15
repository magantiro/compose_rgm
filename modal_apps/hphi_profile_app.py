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
app = modal.App("hphi-profile")
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


@app.function(image=image, cpu=(8.0, 8.0), memory=16384, timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def profile(source: str) -> dict[str, Any]:
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger

    RDLogger.DisableLog("rdApp.*")
    from rdkit.Chem import QED, rdFingerprintGenerator

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        _one_state_batch, canonical_state_key, enumerate_factorized_marked_law,
    )

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    src_fp = gen.GetFingerprint(Chem.MolFromSmiles(source))
    out: dict[str, Any] = {"source": source, "by_threads": {}}

    for threads in (1, 4, 8):
        model, system = _load(threads)
        T = {"prepare": 0.0, "law": 0.0, "apply": 0.0, "encode": 0.0,
             "props": 0.0}
        n_marks: list[int] = []
        cur = source
        rng = np.random.default_rng(0)
        t_all = time.perf_counter()
        for _ in range(STEPS):
            t = time.perf_counter()
            st = pad_molecular_graph(smiles_to_molecular_graph(cur),
                                     CANONICAL_SLOTS)
            T["prepare"] += time.perf_counter() - t

            t = time.perf_counter()
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            T["law"] += time.perf_counter() - t
            if not law.marks:
                break
            n_marks.append(len(law.marks))

            p = np.array([m.probability for m in law.marks], float); p /= p.sum()
            idx = rng.choice(len(p), size=16, p=p)

            t = time.perf_counter()
            ys = [canonical_state_key(system.apply(
                st, law.marks[int(i)].executor_rule_name,
                law.marks[int(i)].action)) for i in idx]
            T["apply"] += time.perf_counter() - t

            t = time.perf_counter()
            for y in ys[:4]:
                m = Chem.MolFromSmiles(y)
                if m:
                    QED.qed(m)
                    DataStructs.TanimotoSimilarity(src_fp, gen.GetFingerprint(m))
            T["props"] += time.perf_counter() - t

            t = time.perf_counter()
            for y in ys[:4]:
                s2 = pad_molecular_graph(smiles_to_molecular_graph(y),
                                         CANONICAL_SLOTS)
                b = _one_state_batch(model, s2, float(TIME_POINT),
                                     prepared_batch=None)
                with torch.no_grad():
                    model._encode_batch(b)
            T["encode"] += time.perf_counter() - t

            nxt = next((y for y in ys if y != canonical_state_key(st)), None)
            if nxt is None:
                break
            cur = nxt
        total = time.perf_counter() - t_all
        out["by_threads"][threads] = {
            "total_s": round(total, 2),
            "per_step_s": round(total / max(1, STEPS), 2),
            "median_marks": int(np.median(n_marks)) if n_marks else 0,
            "phases_s": {k: round(v, 2) for k, v in T.items()},
            "phase_pct": {k: (round(100 * v / total, 1) if total else 0)
                          for k, v in T.items()},
        }
        print(f"threads={threads}: {total:.1f}s total, "
              f"{total/STEPS:.2f}s/step, phases={out['by_threads'][threads]['phase_pct']}",
              flush=True)
    return out


@app.local_entrypoint()
def main() -> None:
    src = [s.strip() for s in
           (Path(__file__).resolve().parents[1] / "data/jin/dev_panel_qed_64.txt"
            ).read_text().split("\n") if s.strip()][0]
    print(f"profiling {STEPS} steps on dev source 0 at 1 / 4 / 8 torch threads")
    r = profile.remote(src)
    print(json.dumps(r, indent=2))
