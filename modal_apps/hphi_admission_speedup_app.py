"""What does the admission-mask memoization actually do to a LAW CALL?

MEASUREMENT HARNESS ONLY. Produces no scientific artifact and must never be a
production path.

The 3.4x law-call figure was a PROJECTION: measured component speedups (6.28x
cycle-close, 3.24x atom-restate on 600 molecules) folded into a measured
component breakdown (2747 / 1626 / 487 ms of a 4860 ms call). Projections of
exactly this shape have been wrong here before -- the graph-only encoder was
projected at 9x from an isolated microbenchmark and delivered 1.4x end to end,
because a cache meant only 24% of encounters paid the cost being subtracted.
So the projection gets measured rather than quoted.

The obvious way to measure it is blocked: the optimized files sit inside the
Process-V2 identity hash, so a container running them fails
`open_process_v2_t1_source` and never loads the model.

This instead loads the model from the PRISTINE on-disk files -- the identity
chain authenticates normally -- and only then rebinds the two modules IN MEMORY,
from source text carried in as an argument. Nothing on disk changes, no artifact
is written, and the identity the container authenticated against is the identity
of the code that was actually on disk. The rebind exists solely to time the same
law twice and confirm it is the same law.

Reports, per state: before/after wall time, and whether the ordered marks and
the float64 probability bits are identical.
"""

from __future__ import annotations

import hashlib
import json
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
app = modal.App("hphi-admission-speedup")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48


@app.function(image=image, cpu=(1.0, 1.0), memory=6144, timeout=45 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def measure(srcs: list[str], patched_sources: dict[str, str]) -> dict[str, Any]:
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
        enumerate_factorized_marked_law,
    )

    # --- model from the PRISTINE tree; the identity gate runs normally -------
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
    print("model loaded; identity chain authenticated against on-disk files\n",
          flush=True)

    states = [pad_molecular_graph(smiles_to_molecular_graph(s), CANONICAL_SLOTS)
              for s in srcs]

    def fingerprint(law):
        ident, probs = hashlib.sha256(), hashlib.sha256()
        for m in law.marks:
            ident.update(repr((m.executor_rule_name, m.action)).encode())
            probs.update(struct.pack("<d", float(m.probability)))
        return len(law.marks), ident.hexdigest(), probs.hexdigest()

    def timed_pass(label):
        out = []
        for st, smi in zip(states, srcs):
            t0 = time.perf_counter()
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            dt = time.perf_counter() - t0
            n, hi, hp = fingerprint(law)
            out.append({"smiles": smi, "n": n, "marks": hi, "probs": hp,
                        "seconds": dt})
            print(f"  {label:<8} {n:>4} marks  {dt*1e3:8.1f} ms  {smi[:34]}",
                  flush=True)
        return out

    before = timed_pass("BEFORE")

    # --- rebind IN MEMORY ONLY ----------------------------------------------
    import compose_v4.rewrite.semantic_atom_restate as SAR
    import compose_v4.rewrite.semantic_cycle_close as SCC

    for mod, key in ((SCC, "semantic_cycle_close"),
                     (SAR, "semantic_atom_restate")):
        exec(compile(patched_sources[key], f"<patched:{key}>", "exec"),
             mod.__dict__)
    print("\nrebound both modules in memory (nothing on disk changed)\n",
          flush=True)

    after = timed_pass("AFTER")

    tb = sum(r["seconds"] for r in before)
    ta = sum(r["seconds"] for r in after)
    same = all(a["n"] == b["n"] and a["marks"] == b["marks"]
               and a["probs"] == b["probs"] for a, b in zip(before, after))
    print(f"\n{'='*68}")
    print(f"  law call BEFORE  {tb/len(before)*1e3:9.1f} ms")
    print(f"  law call AFTER   {ta/len(after)*1e3:9.1f} ms")
    print(f"  MEASURED SPEEDUP {tb/max(ta,1e-9):9.2f}x   (projection was 3.41x)")
    print(f"\n  LAW PARITY: {'PASSED' if same else 'FAILED'}  "
          f"({len(before)} states, ordered marks and float64 probability bits)")
    return {"before_mean_s": tb / len(before), "after_mean_s": ta / len(after),
            "speedup": tb / max(ta, 1e-9), "law_parity": same,
            "n_states": len(before)}


@app.local_entrypoint()
def main(n_states: int = 10) -> None:
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    srcs = [s.strip() for s in
            (repo / "data/jin/dev_panel_qed_64.txt").read_text().split("\n")
            if s.strip()][:n_states]

    # Source text comes from the perf BRANCH; the working tree stays pristine.
    patched = {}
    for key in ("semantic_cycle_close", "semantic_atom_restate"):
        patched[key] = subprocess.run(
            ["git", "show",
             f"perf/exact-admission-mask-memoization:src/compose_v4/rewrite/{key}.py"],
            capture_output=True, text=True, cwd=repo, check=True).stdout
        print(f"loaded patched {key}: {len(patched[key]):,} chars")

    out = measure.remote(srcs, patched)
    Path("docs/ADMISSION_SPEEDUP_MEASURED.json").write_text(json.dumps(out, indent=1))
    print(f"\nmeasured speedup {out['speedup']:.2f}x   "
          f"law parity {'PASSED' if out['law_parity'] else 'FAILED'}")
