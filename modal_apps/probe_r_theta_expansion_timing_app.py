"""Time ONE successor expansion of frozen R_theta, on the volume. Nothing else.

WHY THIS RUNS ON THE VOLUME AND NOT LOCALLY
-------------------------------------------
`source_index_sha256` is computed over a body whose first field is
`active8_run_root`, the absolute filesystem path, and the same path is
re-derived inside `reduce_gate_zero`. A byte-identical local mirror at
`local_runtime/active8/...` therefore cannot reproduce a hash of
`/artifacts/...` and the lineage check fails on LOCATION while the content is
right. Here the volume is mounted at its own path, so the chain validates with
no change to the lineage system at all. See `scripts/pareto_local_runtime.py`.

WHAT THIS IS FOR
----------------
The seeded-active A/B needs a price before it is launched, and the price is
dominated by one number nobody has measured: how long a single canonical
successor expansion takes on CPU. Eight molecules, one expansion each, is
enough to get a median and a spread. It is deliberately not a scientific run --
it produces a duration, not a result.

CPU ONLY, AND SIZED TO THE WORK. The expansions run one at a time and the image
pins OMP_NUM_THREADS=1, so this reserves 2 cores rather than 8. Reserving cores
that sit idle is how a cheap probe turns into an expensive one.
"""

from __future__ import annotations

import json
import statistics
import time
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import ARTIFACT_ROOT, REMOTE_ROOT, artifact_volume
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
app = modal.App("compose-v4-r-theta-expansion-timing")

ACTIVE8 = ("/artifacts/editing_v2/process_v2_active8/"
           "8ecc0e5e825a15200560c58960d4f662c9ca23be785c825c9c24aa73308144bb")
GATE_ZERO = ("/artifacts/editing_v2/process_v2_gate_zero_v6/"
             "bd4ac715f699c39c9012c3428475d13057bd7d3d18b2fb10ff716607633c4f2d"
             "/DECISION.json")
MATERIALIZED = "/artifacts/editing_v2/r_theta_run/materialized_scorer"
CHECKPOINT = "/artifacts/editing_v2/r_theta_run/runs/run_v2_01/R_THETA_CHECKPOINT.pt"

#: The first eight molecules of development seed 100's frozen initialization
#: set. Inlined rather than read from artifacts/ so the probe depends on nothing
#: but the volume and the image.
PANEL = (
    'CC(C)[C@@H](NC(=O)N1CCC[C@H]1C[NH+]1CCC[C@@H]1CO)c1ccccc1',
    'Cc1cc(CC(=O)N[C@H](C)c2nc(-c3ccc(Cl)cc3)no2)no1',
    'C[C@@H]1CCCN(C(=O)N[C@@H]2CC[C@H]([NH+](C)C)C2)[C@@H]1C',
    'CC1(C)CCC[C@H]1NC(=O)NC[C@H](CCO)c1ccccc1',
    'C[NH+](C)[C@H](CNS(=O)(=O)c1ccc(C#N)cc1)c1ccccc1',
    'Cc1cc(C[S@](=O)Cc2csc(C3CCCCC3)n2)no1',
    'COC(=O)[C@H]1CCCCC[C@H]1NC(=O)[C@@H](C)NC(=O)C1CCC1',
    'c1ccc(C[C@@H]2CCC[C@@H]2[NH2+][C@@H]2CCc3n[nH]cc3C2)cc1',
)


@app.function(
    image=image,
    cpu=2.0,
    memory=8 * 1024,
    timeout=20 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def time_expansions(panel: tuple[str, ...] = PANEL) -> dict[str, Any]:
    from pathlib import Path

    import torch

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state,
    )
    from compose_v4.experiments.production_successor_kernel import (
        canonical_successor_result,
    )

    artifact_volume.reload()
    started = time.perf_counter()
    source = open_process_v2_t1_source(
        Path(ACTIVE8),
        gate_zero_decision_path=Path(GATE_ZERO),
        artifact_root=ARTIFACT_ROOT,
        repo_root=REMOTE_ROOT,
    )
    bundle = load_materialized_scorer_state(Path(MATERIALIZED))
    runtime, _binding, _receipt = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=bundle)
    model = runtime.model
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["selected_model_state"], strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    build_seconds = time.perf_counter() - started

    rows = []
    for smiles in panel:
        graph = smiles_to_molecular_graph(smiles)
        padded = pad_molecular_graph(graph, graph.num_atoms + 8)
        mark = time.perf_counter()
        result = canonical_successor_result(model, padded, 0.0)
        elapsed = time.perf_counter() - mark
        batch = result.batch
        successors = len(batch.keys()) if hasattr(batch, "keys") else len(batch)
        rows.append({"smiles": smiles, "atoms": int(graph.num_atoms),
                     "successors": int(successors), "seconds": elapsed})
        print({"expansion": rows[-1]}, flush=True)

    seconds = [row["seconds"] for row in rows]
    report = {
        "phase": "r_theta_expansion_timing",
        "lineage": "validated on the volume, unmodified",
        "runtime_build_seconds": build_seconds,
        "expansions": rows,
        "median_seconds": statistics.median(seconds),
        "mean_seconds": statistics.fmean(seconds),
        "max_seconds": max(seconds),
        "median_successors": statistics.median(r["successors"] for r in rows),
        "cpu_reserved": 2.0,
    }
    print(json.dumps(report, indent=1), flush=True)
    return report


@app.local_entrypoint()
def main() -> None:
    report = time_expansions.remote()
    median = report["median_seconds"]
    print(json.dumps({
        "median_expansion_seconds": median,
        "median_successors": report["median_successors"],
        "runtime_build_seconds": report["runtime_build_seconds"],
        "implied_core_minutes_per_1000_expansions": median * 1000 / 60,
    }, indent=1))
