"""Score the checkpoint on PROVENANCE-MATCHED pairs, decomposed by factor.

WHY
---
The pilot's per-family transfer gaps track the train->validation shift in
teacher policy far better than anything about the molecules:

    Spearman(gap, synthetic-share shift) = +0.762
    Spearman(gap, 2-hop unseen motif)    = -0.167
    Spearman(gap, motif total variation) = +0.000
    Spearman(gap, train motif frequency) = +0.405

The four families with large gaps are trained mostly on real endpoint-derived
transitions and evaluated mostly on reversible synthetic walks; the four that
transfer are synthetic in both pools. So the earlier "the model fails to
generalize atom chemistry" reading confounds source generalization with a
change in what the teacher is even asking.

A real-endpoint row asks "which legal edit moves toward this analogue
endpoint"; a synthetic-walk row asks "which legal edit did the corruption
sampler take". Those are different conditional distributions over the same
legal neighbourhood, and no amount of molecular similarity makes them the same
question.

WHAT THIS MEASURES
------------------
Train-vs-held-out gaps computed WITHIN each (family, provenance) stratum, so a
real-trained family is compared against real held-out rows and a
synthetic-trained one against synthetic. If the within-provenance gaps collapse
while the cross-provenance comparison stays large, the evaluation was the
problem rather than the model.

Each gap is also split into the two factors the scorer already returns:

    l_family   = -log P(F* | x)          teacher-family calibration
    l_identity = -log P(y* | x, F*)      within-family site selection

A +4 nat gap that is mostly the family term implies calibrating a gate; one
that is mostly the identity term implies the candidate scorer. Reporting only
the joint cannot distinguish them, which is why the joint was not enough.

STRATA THE PANEL CANNOT POPULATE ARE REFUSED, NOT ESTIMATED
-----------------------------------------------------------
MEASURED: held-out real-lane entries number 21 for atom_insert and 0 for
cycle_insert. Those strata are reported as unavailable rather than given a
number computed from a handful of rows.
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
app = modal.App("compose-v4-r-theta-provenance")

#: Below this a stratum mean is noise; report it as unavailable instead.
MINIMUM_STRATUM = 40


@app.function(
    image=image,
    gpu="A10G",
    cpu=8.0,
    memory=64 * 1024,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def diagnose_provenance(
    run_root: str = "/artifacts/editing_v2/r_theta_run",
    output_name: str = "pilot_01",
    per_stratum: int = 1200,
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
    runtime, _binding, _c = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=state
    )
    model = runtime.model.to(device)
    checkpoint = torch.load(
        root / "runs" / output_name / CHECKPOINT_FILENAME,
        map_location="cpu", weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    print(f"[{time.perf_counter()-started:6.1f}s] checkpoint step "
          f"{checkpoint['completed_steps']:,}", flush=True)

    with gzip.open(inputs / "ENTRY_PROVENANCE.json.gz", "rt") as handle:
        provenance = json.load(handle)
    synthetic_lane = provenance["synthetic_lane"]
    lane_by_entry = provenance["lane_by_entry_id"]

    def kind(entry_id: str) -> str:
        lane = lane_by_entry.get(entry_id)
        if lane is None:
            return "unknown"
        return "synthetic" if lane == synthetic_lane else "real"

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
    template = torch.load(Path(paths["packed_store"]) / "TEMPLATE.pt",
                          map_location="cpu", weights_only=False)
    panel_library = load_corpus_training_library(
        [Path(r) for r in paths["panel_corpus_roots"]],
        excluded_sources=resolution["excluded_source_keys"]["validation"],
        verify_state_roundtrip=False,
    )
    panel_store = PackedCollatedStore(Path(paths["packed_panel_store"]))
    panel_template = torch.load(Path(paths["packed_panel_store"]) / "TEMPLATE.pt",
                                map_location="cpu", weights_only=False)
    print(f"[{time.perf_counter()-started:6.1f}s] libraries loaded", flush=True)

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

    # Training rows the optimizer actually consumed, so the comparison is not
    # confounded by drawing differently from training.
    seen = int(checkpoint["completed_steps"]) * batch_size
    consumed = list(dict.fromkeys(index["sequence"][:seen]))
    train_by = collections.defaultdict(list)
    for entry_id in consumed:
        entry = library.entries[library.index_by_entry_id[entry_id]]
        train_by[(entry.model_family, kind(entry_id))].append(entry_id)
    panel_by = collections.defaultdict(list)
    for entry in panel_library.entries:
        panel_by[(entry.model_family, kind(entry.entry_id))].append(entry.entry_id)

    def score(ids, build):
        rows = evaluate_panel(model, entry_ids=ids, build_batch=build,
                              batch_size=batch_size)
        return {
            "entries": len(rows),
            "joint_nll": statistics.mean(r["teacher_successor_nll"] for r in rows),
            "family_nll": statistics.mean(r["family_nll"] for r in rows),
            "identity_nll": statistics.mean(r["identity_nll"] for r in rows),
        }

    build_train = builder(library, store, template)
    build_panel = builder(panel_library, panel_store, panel_template)
    strata = sorted({k for k in list(train_by) + list(panel_by) if k[1] != "unknown"})
    results: dict[str, Any] = {}
    print(f"\n{'family':22} {'prov':>10} {'n tr':>6} {'n ho':>6} "
          f"{'joint tr':>9} {'joint ho':>9} {'gap':>7} {'fam gap':>8} {'id gap':>7}",
          flush=True)
    for family, kind_name in strata:
        train_ids = train_by.get((family, kind_name), [])[:per_stratum]
        panel_ids = panel_by.get((family, kind_name), [])[:per_stratum]
        key = f"{family}|{kind_name}"
        if len(train_ids) < MINIMUM_STRATUM or len(panel_ids) < MINIMUM_STRATUM:
            results[key] = {"available": False,
                            "train_entries": len(train_ids),
                            "held_out_entries": len(panel_ids)}
            print(f"{family:22} {kind_name:>10} {len(train_ids):6,} {len(panel_ids):6,} "
                  f"{'--':>9} {'--':>9} {'unavailable':>7}", flush=True)
            continue
        on_train = score(train_ids, build_train)
        on_panel = score(panel_ids, build_panel)
        results[key] = {
            "available": True, "train": on_train, "held_out": on_panel,
            "joint_gap": on_panel["joint_nll"] - on_train["joint_nll"],
            "family_gap": on_panel["family_nll"] - on_train["family_nll"],
            "identity_gap": on_panel["identity_nll"] - on_train["identity_nll"],
        }
        r = results[key]
        print(f"{family:22} {kind_name:>10} {on_train['entries']:6,} "
              f"{on_panel['entries']:6,} {on_train['joint_nll']:9.2f} "
              f"{on_panel['joint_nll']:9.2f} {r['joint_gap']:+7.2f} "
              f"{r['family_gap']:+8.2f} {r['identity_gap']:+7.2f}", flush=True)

    return {
        "schema": "compose.editing_v2.r_theta_provenance_matched",
        "status": "DIAGNOSTIC_EVIDENCE_ONLY_NO_AUTHORITY",
        "checkpoint_step": int(checkpoint["completed_steps"]),
        "minimum_stratum": MINIMUM_STRATUM,
        "per_stratum_cap": per_stratum,
        "by_stratum": results,
        "reading": (
            "Within-provenance gaps near zero mean the earlier per-family gaps "
            "measured teacher-policy shift, not failure to generalize. A gap "
            "concentrated in family_gap indicts the family gate; one in "
            "identity_gap indicts the candidate scorer."
        ),
    }


@app.local_entrypoint()
def main(output_name: str = "pilot_01", per_stratum: int = 1200) -> None:
    print(json.dumps(
        diagnose_provenance.remote(output_name=output_name, per_stratum=per_stratum),
        indent=2, sort_keys=True))
