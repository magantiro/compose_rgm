"""Separate incomplete exposure from genuine transfer failure.

WHY THE EARLIER DIAGNOSTIC WAS NOT ENOUGH
-----------------------------------------
It compared rows the optimizer had consumed against held-out rows, at a
checkpoint that had seen 44,826 of 116,375 distinct entries -- 38.5%. For
multi-successor sources the median fraction of that source's own alternatives
drawn was 0.50, and only 14.2% had all of them drawn. Cross-entropy cannot
average over labels it has never sampled, so "one-hot suppressed the
alternatives" and "the alternatives had not appeared yet" produce the same
measurement. The gap was real; its attribution was not.

FOUR EXPOSURE CATEGORIES
------------------------
    seen_transition   the exact (x, y) was consumed
    unseen_sibling    x was consumed, but not THIS alternative successor
    unseen_source     same training population and provenance, x never consumed
    held_out          the provenance-matched validation population

Reading across them separates the hypotheses:

    seen good, sibling bad                  incomplete multi-successor exposure
                                            -> multi-positive supervision is
                                               directly motivated
    seen good, unseen_source AND held_out
        both bad                            transferable features not yet
                                            learned -> more exposure may fix it
    unseen_source good, held_out bad        residual population/split mismatch
    all bad                                 the family is simply not learned

WHY NO UNIFORM BASELINE IS NEEDED
---------------------------------
Raw NLL is not comparable ACROSS families with 59 to 420 median legal marks.
But every comparison here is within one family across exposure categories, so
the family's branching factor cancels. Adding a baseline would only reintroduce
the mark-count-versus-successor-count approximation.

Each cell is also split into the two factors the scorer returns, because a gap
in P(F|x) implies calibrating a gate and a gap in P(y|x,F) implies the
candidate scorer.
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
app = modal.App("compose-v4-r-theta-exposure")

#: Below this a cell mean is noise; report unavailable rather than a number.
MINIMUM_CELL = 40
CATEGORIES = ("seen_transition", "unseen_sibling", "unseen_source", "held_out")


@app.function(
    image=image,
    gpu="A10G",
    cpu=8.0,
    memory=64 * 1024,
    timeout=35 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def diagnose_exposure(
    run_root: str = "/artifacts/editing_v2/r_theta_run",
    output_name: str = "pilot_01",
    per_cell: int = 600,
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
        source, materialized_state=state
    )
    model = runtime.model.to(device)
    checkpoint = torch.load(root / "runs" / output_name / CHECKPOINT_FILENAME,
                            map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    step = int(checkpoint["completed_steps"])
    print(f"[{time.perf_counter()-started:6.1f}s] checkpoint step {step:,}", flush=True)

    with gzip.open(inputs / "ENTRY_PROVENANCE.json.gz", "rt") as handle:
        provenance = json.load(handle)
    synthetic_lane = provenance["synthetic_lane"]
    lane_by_entry = provenance["lane_by_entry_id"]

    def kind(entry_id: str) -> str:
        lane = lane_by_entry.get(entry_id)
        return "unknown" if lane is None else (
            "synthetic" if lane == synthetic_lane else "real")

    resolution = json.loads(
        (inputs / "editing_v2_split_precedence_resolution.json").read_text())
    index = json.loads((inputs / "editing_v2_prepared_training_index.json").read_text())
    library = load_corpus_training_library(
        [Path(r) for r in paths["corpus_roots"]],
        excluded_sources=resolution["excluded_source_keys"]["train"],
        verify_state_roundtrip=False)
    store = PackedCollatedStore(Path(paths["packed_store"]))
    template = torch.load(Path(paths["packed_store"]) / "TEMPLATE.pt",
                          map_location="cpu", weights_only=False)
    panel_library = load_corpus_training_library(
        [Path(r) for r in paths["panel_corpus_roots"]],
        excluded_sources=resolution["excluded_source_keys"]["validation"],
        verify_state_roundtrip=False)
    panel_store = PackedCollatedStore(Path(paths["packed_panel_store"]))
    panel_template = torch.load(Path(paths["packed_panel_store"]) / "TEMPLATE.pt",
                                map_location="cpu", weights_only=False)
    print(f"[{time.perf_counter()-started:6.1f}s] libraries loaded", flush=True)

    # Which exact transitions the optimizer consumed, and which SOURCES that
    # touched. The distinction is the point: a source can be well known while
    # one of its recorded alternatives has never been drawn.
    consumed = set(index["sequence"][: step * batch_size])
    source_of = {
        entry.entry_id: entry.teacher_fiber.state_support.source_state_sha256
        for entry in library.entries
    }
    touched_sources = {source_of[i] for i in consumed if i in source_of}

    buckets: dict[tuple[str, str, str], list[str]] = collections.defaultdict(list)
    for entry in library.entries:
        provenance_kind = kind(entry.entry_id)
        if provenance_kind == "unknown":
            continue
        if entry.entry_id in consumed:
            category = "seen_transition"
        elif source_of[entry.entry_id] in touched_sources:
            category = "unseen_sibling"
        else:
            category = "unseen_source"
        buckets[(entry.model_family, provenance_kind, category)].append(entry.entry_id)
    for entry in panel_library.entries:
        provenance_kind = kind(entry.entry_id)
        if provenance_kind != "unknown":
            buckets[(entry.model_family, provenance_kind, "held_out")].append(
                entry.entry_id)

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

    build_train = builder(library, store, template)
    build_panel = builder(panel_library, panel_store, panel_template)

    def score(ids, build):
        rows = evaluate_panel(model, entry_ids=ids, build_batch=build,
                              batch_size=batch_size)
        return {
            "entries": len(rows),
            "joint_nll": statistics.mean(r["teacher_successor_nll"] for r in rows),
            "family_nll": statistics.mean(r["family_nll"] for r in rows),
            "identity_nll": statistics.mean(r["identity_nll"] for r in rows),
        }

    results: dict[str, Any] = {}
    families = sorted({key[0] for key in buckets})
    print(f"\n{'family':20} {'prov':>10} {'category':>16} {'n':>6} "
          f"{'joint':>7} {'family':>7} {'identity':>9}", flush=True)
    for family in families:
        for provenance_kind in ("real", "synthetic"):
            for category in CATEGORIES:
                ids = buckets.get((family, provenance_kind, category), [])
                key = f"{family}|{provenance_kind}|{category}"
                if len(ids) < MINIMUM_CELL:
                    results[key] = {"available": False, "entries": len(ids)}
                    continue
                build = build_panel if category == "held_out" else build_train
                measured = score(ids[:per_cell], build)
                results[key] = {"available": True, **measured,
                                "population_entries": len(ids)}
                print(f"{family:20} {provenance_kind:>10} {category:>16} "
                      f"{measured['entries']:6,} {measured['joint_nll']:7.2f} "
                      f"{measured['family_nll']:7.2f} "
                      f"{measured['identity_nll']:9.2f}", flush=True)

    return {
        "schema": "compose.editing_v2.r_theta_exposure_stratified",
        "status": "DIAGNOSTIC_EVIDENCE_ONLY_NO_AUTHORITY",
        "checkpoint_step": step,
        "distinct_transitions_consumed": len(consumed),
        "sources_touched": len(touched_sources),
        "minimum_cell": MINIMUM_CELL,
        "per_cell_cap": per_cell,
        "by_cell": results,
        "reading": (
            "Within a family the branching factor cancels, so categories are "
            "directly comparable. seen good / unseen_sibling bad means "
            "incomplete multi-successor exposure and motivates multi-positive "
            "supervision. seen good with unseen_source and held_out both bad "
            "means transferable features are not yet learned and more exposure "
            "may help. unseen_source good with held_out bad means a residual "
            "population or split mismatch."
        ),
    }


@app.local_entrypoint()
def main(output_name: str = "pilot_01", per_cell: int = 600) -> None:
    print(json.dumps(
        diagnose_exposure.remote(output_name=output_name, per_cell=per_cell),
        indent=2, sort_keys=True))
