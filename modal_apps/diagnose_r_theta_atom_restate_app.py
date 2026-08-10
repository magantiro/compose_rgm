"""Allocation or within-family degradation? And the cohort number.

WHY THE JOINT MEAN CANNOT SETTLE IT
-----------------------------------
atom_restate ended epoch 1 more than 0.5 nats worse than its first
measurement, which gated steps 3,500 and 4,000 despite those holding the run's
best reference-law numbers. The joint per-family NLL says it got worse. It
cannot say WHICH factor moved, and the two imply different fixes:

    P(y|x)  =  P(F|x)  .  P(y|x,F)
               ^^^^^^     ^^^^^^^^
               family     identity
               gate       scorer

  family_nll rising    the gate is reallocating probability mass away from
                       atom_restate as other families improve. An allocation
                       effect -- arguably the model correctly learning that
                       atom_restate is rarer than it first assumed.

  identity_nll rising  the model is getting worse at choosing the SITE within
                       an atom_restate edit. Real degradation of the
                       capability, and the case that would justify acting.

Both states are scored: the SELECTED step-3,000 weights and the FINAL
step-4,251 weights, against the same reserve, so the movement is attributable
to training rather than to a different panel.

THE COHORT NUMBER
-----------------
The externally mined 16-shard panel was never measured during the run: its
diagnostic was gated on an epoch boundary nested inside the evaluate_every
block, and no evaluation ran on the final step at all because 4,251 is not a
multiple of 500. Measured here once, against the selected checkpoint. It is a
CROSS-COHORT DIAGNOSTIC and was never a selection input -- its support
composition is an accident of how those shards were mined, 42.6% at band 0
against the training population's 19.3%.

Runs at most two forward passes over 15,031 entries plus one over 14,131.
"""

from __future__ import annotations

import collections
import gzip
import json
import statistics
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
app = modal.App("compose-v4-r-theta-atom-restate")

SYNTHETIC_LANE = "reversible_synthetic_walk"
MINIMUM_CELL = 30


@app.function(
    image=image,
    gpu="A10G",
    cpu=8.0,
    memory=64 * 1024,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def diagnose_atom_restate(
    run_root: str = "/artifacts/editing_v2/r_theta_run",
    output_name: str = "run_v2_01",
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
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=state)
    model = runtime.model.to(device)
    checkpoint = torch.load(root / "runs" / output_name / CHECKPOINT_FILENAME,
                            map_location="cpu", weights_only=False)
    print(f"[{time.perf_counter()-started:6.1f}s] checkpoint step "
          f"{checkpoint['completed_steps']:,}, selected "
          f"{checkpoint['selected_step']:,}", flush=True)

    resolution = json.loads(
        (inputs / "editing_v2_split_precedence_resolution.json").read_text())
    library = load_corpus_training_library(
        [Path(r) for r in paths["corpus_roots"]],
        excluded_sources=resolution["excluded_source_keys"]["train"],
        verify_state_roundtrip=False)
    store = PackedCollatedStore(Path(paths["packed_store"]))
    template = torch.load(Path(paths["packed_store"]) / "TEMPLATE.pt",
                          map_location="cpu", weights_only=False)
    with gzip.open(
            inputs / "editing_v2_matched_validation_reserve_ids.json.gz", "rt") as handle:
        reserve = json.load(handle)
    reserve_ids = list(reserve["reserve_entry_ids"])
    bands = reserve["reserve_band_by_entry_id"]
    with gzip.open(inputs / "ENTRY_PROVENANCE.json.gz", "rt") as handle:
        provenance = json.load(handle)
    lanes = {i: ("synthetic" if provenance["lane_by_entry_id"].get(i) == SYNTHETIC_LANE
                 else "real") for i in reserve_ids}

    def builder(src_library, src_store, src_template):
        def build(entry_ids):
            states, fibers, entries = src_library.inputs_for(entry_ids)
            count = len(entry_ids)
            batch = src_store.rows_for(entry_ids, src_template, extra={
                "states": tuple(states),
                "times": torch.tensor(
                    [float.fromhex(e.support_time_hex) for e in entries],
                    dtype=torch.float32),
                "teacher_rates": torch.ones(count, dtype=torch.float32),
                "importance_weights": torch.ones(count, dtype=torch.float32)})
            batch = _attach_successor_family_coordinates(
                batch, [{"model_family": e.model_family} for e in entries])
            return batch.to(device), fibers, entries
        return build

    build_reserve = builder(library, store, template)

    def score(weights, label):
        model.load_state_dict(weights, strict=True)
        model.to(device)
        rows = evaluate_panel(model, entry_ids=reserve_ids,
                              build_batch=build_reserve, batch_size=batch_size,
                              lane_by_entry_id=lanes,
                              support_band_by_entry_id=bands)
        print(f"[{time.perf_counter()-started:6.1f}s] scored {label}: "
              f"{len(rows):,} entries", flush=True)
        by_family = collections.defaultdict(list)
        by_cell = collections.defaultdict(list)
        for row in rows:
            by_family[row["model_family"]].append(row)
            by_cell[row["capability_cell_id"]].append(row)

        def split(items):
            return {
                "entries": len(items),
                "joint_nll": statistics.mean(r["teacher_successor_nll"] for r in items),
                "family_nll": statistics.mean(r["family_nll"] for r in items),
                "identity_nll": statistics.mean(r["identity_nll"] for r in items),
            }
        return ({name: split(items) for name, items in sorted(by_family.items())},
                {name: split(items) for name, items in sorted(by_cell.items())
                 if len(items) >= MINIMUM_CELL})

    selected_families, selected_cells = score(
        checkpoint["selected_model_state"], f"selected step {checkpoint['selected_step']:,}")
    final_families, final_cells = score(
        checkpoint["model_state"], f"final step {checkpoint['completed_steps']:,}")

    print(f"\n{'family':22} {'joint':>17} {'family P(F|x)':>19} "
          f"{'identity P(y|x,F)':>21}")
    print(f"{'':22} {'sel':>8}{'fin':>9} {'sel':>9}{'fin':>10} {'sel':>10}{'fin':>11}")
    for name in sorted(final_families, key=lambda n: -(
            final_families[n]["joint_nll"] - selected_families[n]["joint_nll"])):
        s, f = selected_families[name], final_families[name]
        print(f"  {name:20} {s['joint_nll']:8.3f}{f['joint_nll']:9.3f} "
              f"{s['family_nll']:9.3f}{f['family_nll']:10.3f} "
              f"{s['identity_nll']:10.3f}{f['identity_nll']:11.3f}")

    # Attribution: of the joint movement, how much is each factor?
    attribution = {}
    for name in final_families:
        s, f = selected_families[name], final_families[name]
        joint = f["joint_nll"] - s["joint_nll"]
        family_move = f["family_nll"] - s["family_nll"]
        identity_move = f["identity_nll"] - s["identity_nll"]
        attribution[name] = {
            "joint_delta": round(joint, 4),
            "family_delta": round(family_move, 4),
            "identity_delta": round(identity_move, 4),
            "dominant_factor": ("family_gate" if abs(family_move) > abs(identity_move)
                                else "identity_scorer"),
        }
    print(f"\n{'family':22} {'joint d':>9} {'family d':>10} {'identity d':>12}  dominant")
    for name, item in sorted(attribution.items(), key=lambda kv: -kv[1]["joint_delta"]):
        print(f"  {name:20} {item['joint_delta']:+9.4f} {item['family_delta']:+10.4f} "
              f"{item['identity_delta']:+12.4f}  {item['dominant_factor']}")

    # The cohort number, measured once against the SELECTED weights.
    cohort: dict[str, Any] | None = None
    try:
        cohort_library = load_corpus_training_library(
            [Path(r) for r in paths["panel_corpus_roots"]],
            excluded_sources=resolution["excluded_source_keys"]["validation"],
            verify_state_roundtrip=False)
        cohort_store = PackedCollatedStore(Path(paths["packed_panel_store"]))
        cohort_template = torch.load(
            Path(paths["packed_panel_store"]) / "TEMPLATE.pt",
            map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["selected_model_state"], strict=True)
        model.to(device)
        rows = evaluate_panel(
            model, entry_ids=[e.entry_id for e in cohort_library.entries],
            build_batch=builder(cohort_library, cohort_store, cohort_template),
            batch_size=batch_size)
        cohort = {
            "entries": len(rows),
            "joint_nll": statistics.mean(r["teacher_successor_nll"] for r in rows),
            "family_nll": statistics.mean(r["family_nll"] for r in rows),
            "identity_nll": statistics.mean(r["identity_nll"] for r in rows),
        }
        print(f"\nexternal 16-shard cohort at the selected checkpoint: "
              f"joint {cohort['joint_nll']:.4f} over {cohort['entries']:,} entries",
              flush=True)
    except Exception as error:  # noqa: BLE001
        print(f"\ncohort evaluation failed: {error!r}", flush=True)

    return {
        "schema": "compose.editing_v2.r_theta_atom_restate_attribution",
        "status": "DIAGNOSTIC_EVIDENCE_ONLY_NO_AUTHORITY",
        "selected_step": int(checkpoint["selected_step"]),
        "final_step": int(checkpoint["completed_steps"]),
        "by_family_selected": selected_families,
        "by_family_final": final_families,
        "by_cell_selected": selected_cells,
        "by_cell_final": final_cells,
        "attribution": attribution,
        "external_cohort_at_selected": cohort,
        "external_cohort_note": (
            "Cross-cohort diagnostic, never a selection input. Its support "
            "composition is an accident of how those shards were mined -- 42.6% "
            "at band 0 against the training population's 19.3%."),
        "reading": (
            "family_delta dominant means the gate is reallocating mass away from "
            "the family as others improve, which is an allocation effect and may "
            "be correct learning. identity_delta dominant means the model is "
            "choosing worse sites WITHIN the family, which is real capability "
            "degradation and the case that would justify acting."),
    }


@app.local_entrypoint()
def main(output_name: str = "run_v2_01") -> None:
    print(json.dumps(diagnose_atom_restate.remote(output_name=output_name),
                     indent=2, sort_keys=True))
