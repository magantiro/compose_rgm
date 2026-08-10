"""Split the per-family held-out result into "not learned" vs "not generalizing".

WHY
---
MEASURED over six evals of the pilot: panel-native NLL plateaus near 4.4 while
the law-weighted mean climbs 5.37 -> 6.18, and the law-heavy families sit 1.0 to
2.6 nats WORSE than their own uniform baselines:

    atom_insert   +2.57      atom_restate  +2.19
    atom_delete   +1.75      bond_reroute  +1.02

Those four are 72.8% of the training draw. Two explanations imply opposite
decisions:

  NOT LEARNED YET      training NLL is also bad for them -> more epochs help
  NOT GENERALIZING     training NLL is good, held-out is bad -> more epochs
                       make it worse, and the corpus or the split is the issue

The aggregate loss cannot separate these because it is dominated by whatever
the sampler draws most. This scores the SAME checkpoint on a training-source
sample and on the held-out panel, per family, and reports the gap.

The training sample is drawn from the frozen stream so it is exactly what the
optimizer saw -- not a fresh draw that would confound the comparison with
sampling differences.
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
app = modal.App("compose-v4-r-theta-family-gap")


@app.function(
    image=image,
    gpu="A10G",
    cpu=8.0,
    memory=64 * 1024,
    timeout=25 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def diagnose_family_gap(
    run_root: str = "/artifacts/editing_v2/r_theta_run",
    output_name: str = "pilot_01",
    train_sample: int = 4000,
    batch_size: int = 32,
) -> dict[str, Any]:
    import torch

    from compose_v4.data.corpus_training_library import load_corpus_training_library
    from compose_v4.data.packed_collated_batch import PackedCollatedStore
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state,
    )
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME,
        evaluate_panel,
        summarize_panel,
    )
    from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
        _attach_successor_family_coordinates,
    )

    started = time.perf_counter()
    artifact_volume.reload()
    root = Path(run_root)
    inputs = root / "run_inputs"
    paths = json.loads((inputs / "RUN_PATHS.json").read_text())
    device = torch.device("cuda")

    source = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]),
        repo_root=REMOTE_ROOT,
    )
    state = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _binding, _c = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=state
    )
    model = runtime.model.to(device)

    checkpoint = torch.load(
        root / "runs" / output_name / CHECKPOINT_FILENAME,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    print(
        f"[{time.perf_counter() - started:6.1f}s] checkpoint step "
        f"{checkpoint['completed_steps']:,}",
        flush=True,
    )

    resolution = json.loads(
        (inputs / "editing_v2_split_precedence_resolution.json").read_text()
    )
    index = json.loads((inputs / "editing_v2_prepared_training_index.json").read_text())

    library = load_corpus_training_library(
        [Path(r) for r in paths["corpus_roots"]],
        excluded_sources=resolution["excluded_source_keys"]["train"],
        verify_state_roundtrip=False,
    )
    store = PackedCollatedStore(Path(paths["packed_store"]))
    template = torch.load(
        Path(paths["packed_store"]) / "TEMPLATE.pt", map_location="cpu", weights_only=False
    )
    panel_library = load_corpus_training_library(
        [Path(r) for r in paths["panel_corpus_roots"]],
        excluded_sources=resolution["excluded_source_keys"]["validation"],
        verify_state_roundtrip=False,
    )
    panel_store = PackedCollatedStore(Path(paths["packed_panel_store"]))
    panel_template = torch.load(
        Path(paths["packed_panel_store"]) / "TEMPLATE.pt",
        map_location="cpu",
        weights_only=False,
    )

    def builder(source_library, source_store, source_template):
        def build(entry_ids):
            states, fibers, entries = source_library.inputs_for(entry_ids)
            count = len(entry_ids)
            batch = source_store.rows_for(
                entry_ids,
                source_template,
                extra={
                    "states": tuple(states),
                    "times": torch.tensor(
                        [float.fromhex(e.support_time_hex) for e in entries],
                        dtype=torch.float32,
                    ),
                    "teacher_rates": torch.ones(count, dtype=torch.float32),
                    "importance_weights": torch.ones(count, dtype=torch.float32),
                },
            )
            batch = _attach_successor_family_coordinates(
                batch, [{"model_family": e.model_family} for e in entries]
            )
            return batch.to(device), fibers, entries

        return build

    # Exactly the rows the optimizer has already seen, in stream order, so the
    # comparison is not confounded by drawing differently from training.
    seen = int(checkpoint["completed_steps"]) * batch_size
    stream = index["sequence"][: max(seen, train_sample)][:train_sample]

    train_rows = evaluate_panel(
        model, entry_ids=stream, build_batch=builder(library, store, template),
        batch_size=batch_size,
    )
    panel_ids = json.loads((inputs / "editing_v2_eval_panel_ids.json").read_text())[
        "entry_ids"
    ]
    panel_rows = evaluate_panel(
        model, entry_ids=panel_ids,
        build_batch=builder(panel_library, panel_store, panel_template),
        batch_size=batch_size,
    )

    train_summary = summarize_panel(train_rows)
    panel_summary = summarize_panel(panel_rows)
    families = sorted(
        set(train_summary["by_family"]) | set(panel_summary["by_family"])
    )
    comparison = {}
    print(f"\n{'family':22} {'train':>8} {'held-out':>9} {'gap':>8}", flush=True)
    for name in families:
        on_train = train_summary["by_family"].get(name, {}).get("mean_nll")
        on_panel = panel_summary["by_family"].get(name, {}).get("mean_nll")
        gap = (on_panel - on_train) if (on_train is not None and on_panel is not None) else None
        comparison[name] = {
            "train_mean_nll": on_train,
            "held_out_mean_nll": on_panel,
            "generalization_gap": gap,
            "train_entries": train_summary["by_family"].get(name, {}).get("entries"),
            "held_out_entries": panel_summary["by_family"].get(name, {}).get("entries"),
        }
        if gap is not None:
            print(f"{name:22} {on_train:8.2f} {on_panel:9.2f} {gap:+8.2f}", flush=True)

    return {
        "schema": "compose.editing_v2.r_theta_family_gap",
        "status": "DIAGNOSTIC_EVIDENCE_ONLY_NO_AUTHORITY",
        "checkpoint_step": int(checkpoint["completed_steps"]),
        "train_sample": len(stream),
        "train_overall_mean_nll": train_summary["panel_native"]["mean_nll"],
        "held_out_overall_mean_nll": panel_summary["panel_native"]["mean_nll"],
        "by_family": comparison,
        "reading": (
            "A large positive gap means the checkpoint fits those training rows "
            "and does not transfer -- more epochs make it worse. A small gap "
            "with both values high means the family is simply not learned yet."
        ),
    }


@app.local_entrypoint()
def main(output_name: str = "pilot_01", train_sample: int = 4000) -> None:
    print(json.dumps(
        diagnose_family_gap.remote(output_name=output_name, train_sample=train_sample),
        indent=2, sort_keys=True,
    ))
