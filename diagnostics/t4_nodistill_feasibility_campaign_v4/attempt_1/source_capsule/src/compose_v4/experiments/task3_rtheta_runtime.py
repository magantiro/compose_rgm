"""Where the frozen R_theta lives on the volume, and how to build it.

Kept in `src` rather than in a Modal app because the container image ships
`src/` and `configs/` plus a single launcher file -- so one app importing a
sibling app fails at runtime with ModuleNotFoundError, which is exactly how the
first fiber-cache launch died. Anything two apps share belongs here.

These paths are the container's, not a mirror's. `source_index_sha256` hashes an
absolute path (see `scripts/pareto_local_runtime.py`), so the lineage chain
validates at these locations and nowhere else.
"""

from __future__ import annotations

from pathlib import Path

ACTIVE8 = ("/artifacts/editing_v2/process_v2_active8/"
           "8ecc0e5e825a15200560c58960d4f662c9ca23be785c825c9c24aa73308144bb")
GATE_ZERO = ("/artifacts/editing_v2/process_v2_gate_zero_v6/"
             "bd4ac715f699c39c9012c3428475d13057bd7d3d18b2fb10ff716607633c4f2d"
             "/DECISION.json")
MATERIALIZED = "/artifacts/editing_v2/r_theta_run/materialized_scorer"
CHECKPOINT = "/artifacts/editing_v2/r_theta_run/runs/run_v2_01/R_THETA_CHECKPOINT.pt"

#: Padding slots above a source's atom count, matching the timing probe.
PAD_SLOTS = 8


def build_r_theta(artifact_root: Path, repo_root: Path):
    """The frozen R_theta, eval mode, grad disabled. Lineage validated en route."""

    import torch

    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state,
    )

    source = open_process_v2_t1_source(
        Path(ACTIVE8), gate_zero_decision_path=Path(GATE_ZERO),
        artifact_root=artifact_root, repo_root=repo_root)
    runtime, _binding, _receipt = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=load_materialized_scorer_state(Path(MATERIALIZED)))
    model = runtime.model
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["selected_model_state"], strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    return model


def expand_fiber(model, smiles: str) -> list[list]:
    """One canonical successor fiber as [[child_smiles, probability], ...]."""

    from compose_v4.chem.molecular_graph import (
        molecular_graph_to_smiles,
        smiles_to_molecular_graph,
    )
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_successor_result,
    )

    graph = smiles_to_molecular_graph(smiles)
    padded = pad_molecular_graph(graph, graph.n_atoms + PAD_SLOTS)
    fiber = canonical_successor_result(model, padded, 0.0).batch.successors
    rows = []
    for successor in fiber:
        child = molecular_graph_to_smiles(successor.state)
        if child:
            rows.append([child, float(successor.probability)])
    return rows
