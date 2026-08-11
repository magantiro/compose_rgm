"""Experiment B / Claim 3: exact finite-horizon bridge control, on OUR kernel.

A CLOSURE EXPERIMENT, not a development program. The mathematics is already
implemented and tested in editing_v2_bridge_control (6 tests, covering exact
terminal tilt, unreachable targets, mid-trajectory retargeting and rare-target
amplification). What has never been done is running it against the FROZEN
R_theta on a slice built by the production editing kernel, and banking that.

WHY NOT THE OLDER BENCHMARK
---------------------------
scripts/exact_doob_enumerable_benchmark.py (2026-07-29) demonstrates the same
mathematics on a DIFFERENT process: it imports rewrite.fiber and
de_novo_rewrite_system and never touches the process identity, Active8 or
editing_v2. Its generator is uniform over legal marks rather than the learned
law. The theorem is process-agnostic, but Claim 3 is about controlling the
process the paper is actually about, so it is not the right instrument here.

WHY THE RESULT IS SELF-CHECKING
-------------------------------
Exact Doob has an analytic target: the B-step endpoint law of the controlled
process must equal R^B(x0, .) g / h_B(x0). Agreement at ~1e-16 is a stronger
statement than any code review, and the h-ratios telescoping along a path is
not a property broken code satisfies by accident.

THE CEMETERY
------------
A bounded slice is not closed under the editing kernel. Escaping mass is NOT
renormalised away -- that would silently redefine the process -- but flows to an
explicit absorbing cemetery with desirability zero, so reach probabilities stay
honest and a path that leaves cannot return to claim the target.
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
app = modal.App("compose-v4-experiment-b-exact-control")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"


@app.function(
    image=image,
    cpu=8.0,
    memory=32 * 1024,
    timeout=60 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_exact_control(
    seed_smiles: str = "CCO",
    heavy_atom_cap: int = 7,
    state_cap: int = 900,
    budget: int = 6,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from compose_v4.experiments.editing_v2_bridge_control import (
        CEMETERY,
        assemble_reference_matrix,
        backward_values,
        build_editing_closure,
        propagate,
        terminal_tilt_residual,
    )
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

    started = time.perf_counter()
    artifact_volume.reload()
    root = Path(RUN_ROOT)
    paths = json.loads((root / "run_inputs" / "RUN_PATHS.json").read_text())

    source = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]),
        repo_root=REMOTE_ROOT,
    )
    state = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=state)
    model = runtime.model
    checkpoint = torch.load(root / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                            map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["selected_model_state"], strict=True)
    model.eval()
    print(f"[{time.perf_counter()-started:6.1f}s] frozen R_theta step "
          f"{checkpoint['selected_step']:,}", flush=True)

    # ---- the enumerable slice, built BY THE KERNEL THAT WILL BE CONTROLLED --
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph

    def admissible(key: str) -> bool:
        graph = smiles_to_molecular_graph(key)
        if graph is None:
            return False
        return int((graph.atom_types >= 0).sum()) <= heavy_atom_cap

    states, rows, stop = build_editing_closure(
        model, seed_smiles, admissible=admissible, slots=heavy_atom_cap + 1,
        state_cap=state_cap, deadline_seconds=30 * 60)
    R, keys, index = assemble_reference_matrix(states, rows)
    print(f"[{time.perf_counter()-started:6.1f}s] closure: {len(states)} states "
          f"(+cemetery), stop={stop}", flush=True)

    # ---- a target set that needs several edits to reach --------------------
    def contains_nitrogen(key: str) -> bool:
        return "N" in key or "n" in key

    g = np.array([0.0 if k == CEMETERY else float(contains_nitrogen(k)) for k in keys])
    x0 = index.get(seed_smiles, index[sorted(states)[0]])
    print(f"  target: contains N -- {int(g.sum())} of {len(keys)-1} states", flush=True)

    # terminal_tilt_residual is the module's own tested routine and returns the
    # tilt TV, support violations, row-sum error and backward-equation residual
    # together. Hand-rolling these would re-derive tested logic and could differ
    # from it in exactly the ways the tests exist to catch.
    primary = terminal_tilt_residual(R, g, budget, x0)
    print(f"[{time.perf_counter()-started:6.1f}s] primary: {json.dumps(primary)}", flush=True)

    # ---- UNREACHABLE TARGET: h = 0, reported, never epsilon-patched --------
    impossible = np.zeros(len(keys))
    unreachable = terminal_tilt_residual(R, impossible, budget, x0)

    # ---- DYNAMIC RETARGETING from a REALISED intermediate state ------------
    half = max(1, budget // 2)
    h = backward_values(R, g, budget)
    midway = propagate(R, h, half, x0, len(keys))
    live = midway.copy()
    live[index[CEMETERY]] = -1.0
    switch = int(np.argmax(live))

    def contains_oxygen(key: str) -> bool:
        return "O" in key or "o" in key

    g2 = np.array([0.0 if k == CEMETERY else float(contains_oxygen(k)) for k in keys])
    retarget = terminal_tilt_residual(R, g2, budget - half, switch)
    print(f"[{time.perf_counter()-started:6.1f}s] retarget from {keys[switch]!r}: "
          f"{json.dumps(retarget)}", flush=True)

    result = {
        "schema": "compose.editing_v2.experiment_b_exact_control",
        "status": "DEVELOPMENT_RESULT_NOT_PAPER_BEARING",
        "checkpoint": {"run": "run_v2_01", "selected_step": int(checkpoint["selected_step"])},
        "slice": {"seed_smiles": seed_smiles, "heavy_atom_cap": heavy_atom_cap,
                  "states": len(states), "stop_reason": stop, "budget": budget,
                  "target": "contains nitrogen",
                  "target_states": int(g.sum()),
                  "reach_probability": primary.get("partition")},
        "primary": primary,
        "unreachable": unreachable,
        "retargeting": {"switch_state": keys[switch], "new_target": "contains oxygen",
                        "remaining_budget": budget - half, **retarget},
        "verdict": {
            "terminal_tilt_exact": primary.get("terminal_tilt_tv", 1.0) < 1e-9,
            "support_preserved": primary.get("support_violations", 1) == 0,
            "rows_sum_to_one": primary.get("max_row_sum_error", 1.0) < 1e-9,
            "backward_equation_holds": primary.get("backward_residual", 1.0) < 1e-9,
            "unreachable_reports_unreachable": unreachable.get("reachable") is False,
            "retargeting_exact": retarget.get("terminal_tilt_tv", 1.0) < 1e-9,
        },
        "reading": (
            "Exact Doob agreement with the analytic tilted law at machine "
            "precision is self-verifying: no code review is a stronger statement. "
            "Support preservation means the controller reweights EXISTING legal "
            "rates and never invents an edge. An unreachable target must report "
            "h=0 rather than be epsilon-patched. Retargeting must reproduce the "
            "exact g'-tilted law FROM THE REALISED SWITCH STATE, which is what "
            "makes mid-trajectory objective change coherent rather than heuristic."),
    }
    print("\nVERDICT " + json.dumps(result["verdict"], indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(seed_smiles: str = "CCO", heavy_atom_cap: int = 7,
         state_cap: int = 900, budget: int = 6) -> None:
    print(json.dumps(run_exact_control.remote(
        seed_smiles=seed_smiles, heavy_atom_cap=heavy_atom_cap,
        state_cap=state_cap, budget=budget), indent=2, sort_keys=True))
