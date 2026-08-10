"""Does held-out error scale with analogue-series depth, or with scaffold rarity?

WHY
---
The validation shards are congeneric series (32 Murcko scaffolds per shard, ten
variants each, 1.4% singletons) while the training shards are diverse libraries
(1,337 scaffolds, 55% singletons). The proposed mechanism is that discriminating
among near-identical analogues -- where the teacher makes different choices for
molecules the model sees as nearly the same -- is a different capability from
transferring across diverse chemistry.

SCAFFOLD COVERAGE, AGAINST THE RIGHT REFERENCE SET
--------------------------------------------------
An earlier pass put coverage at 85.5% and concluded this was not chemical
novelty. That counted train-ROLE sources across 30 Active8 shards, but the
model trained on the compiled LIBRARY -- 106,759 of 564,316 role sources.
Against what the model actually saw, coverage is 56%: 6,224 of 14,140 panel
entries sit on a scaffold absent from the training corpus.

THE CONTROL, AND WHY IT ONLY WORKS IN ONE DIRECTION
---------------------------------------------------
Series depth and train scaffold support are nearly collinear here, Pearson
+0.974, so binning cannot separate them into independent axes. What saves the
test is that the two push in OPPOSITE directions on the same partition: more
support should ease prediction, more depth should harden it. So the reading is
asymmetric, and it is declared before the run rather than chosen after it.

WHAT WOULD CONFIRM IT
---------------------
Identity NLL RISING monotonically with panel_depth. Because support rises with
depth, a rising curve rises against a force that should lower it, which is
conservative evidence for the depth mechanism.

Identity NLL FALLING is uninformative: support helping and depth not hurting
are indistinguishable. Flat is a negative result -- series structure is not the
mechanism, it remains unidentified, and that is the outcome to report honestly
rather than narrate around.

Family and provenance are held fixed in the conditioned table so family mix
cannot masquerade as a depth effect.
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
app = modal.App("compose-v4-r-theta-series-depth")

MINIMUM_CELL = 40
#: Depth 0 is a real bucket -- 361 panel entries are the only member of their
#: scaffold. Omitting it dropped them into an unsortable "?" and crashed the
#: report after the scoring had already succeeded.
DEPTH_EDGES = ((0, 0), (1, 1), (2, 4), (5, 9), (10, 24), (25, 10**9))
SUPPORT_EDGES = ((0, 0), (1, 4), (5, 24), (25, 10**9))


def _label(value: int, edges) -> str:
    for low, high in edges:
        if low <= value <= high:
            return f"{low}" if low == high else (
                f"{low}-{high}" if high < 10**9 else f"{low}+")
    return f"?{value}"


def _order(label: str) -> int:
    """Sort bucket labels numerically, keeping any unbucketed value last."""
    if label.startswith("?"):
        return 10**12
    return int(label.split("-")[0].rstrip("+"))


@app.function(
    image=image,
    gpu="A10G",
    cpu=8.0,
    memory=64 * 1024,
    timeout=30 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def diagnose_series_depth(
    run_root: str = "/artifacts/editing_v2/r_theta_run",
    output_name: str = "pilot_01",
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
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    print(f"[{time.perf_counter()-started:6.1f}s] checkpoint step "
          f"{checkpoint['completed_steps']:,}", flush=True)

    with gzip.open(inputs / "ENTRY_PROVENANCE.json.gz", "rt") as handle:
        provenance = json.load(handle)
    synthetic_lane = provenance["synthetic_lane"]
    lane_by_entry = provenance["lane_by_entry_id"]
    with gzip.open(inputs / "PANEL_SERIES_DEPTH.json.gz", "rt") as handle:
        series = json.load(handle)["by_entry_id"]

    resolution = json.loads(
        (inputs / "editing_v2_split_precedence_resolution.json").read_text())
    panel_library = load_corpus_training_library(
        [Path(r) for r in paths["panel_corpus_roots"]],
        excluded_sources=resolution["excluded_source_keys"]["validation"],
        verify_state_roundtrip=False)
    panel_store = PackedCollatedStore(Path(paths["packed_panel_store"]))
    panel_template = torch.load(Path(paths["packed_panel_store"]) / "TEMPLATE.pt",
                                map_location="cpu", weights_only=False)

    def build(entry_ids):
        states, fibers, entries = panel_library.inputs_for(entry_ids)
        count = len(entry_ids)
        batch = panel_store.rows_for(entry_ids, panel_template, extra={
            "states": tuple(states),
            "times": torch.tensor(
                [float.fromhex(e.support_time_hex) for e in entries],
                dtype=torch.float32),
            "teacher_rates": torch.ones(count, dtype=torch.float32),
            "importance_weights": torch.ones(count, dtype=torch.float32)})
        batch = _attach_successor_family_coordinates(
            batch, [{"model_family": e.model_family} for e in entries])
        return batch.to(device), fibers, entries

    ids = [e.entry_id for e in panel_library.entries]
    rows = evaluate_panel(model, entry_ids=ids, build_batch=build,
                          batch_size=batch_size)
    print(f"[{time.perf_counter()-started:6.1f}s] scored {len(rows):,} panel entries",
          flush=True)

    for row in rows:
        info = series.get(row["entry_id"], {})
        row["panel_depth"] = int(info.get("panel_depth", 0))
        row["train_support"] = int(info.get("train_support", 0))
        lane = lane_by_entry.get(row["entry_id"])
        row["provenance"] = ("unknown" if lane is None else
                             ("synthetic" if lane == synthetic_lane else "real"))

    # Persist the scored rows BEFORE aggregating. Scoring is the GPU cost and
    # the tables are pure arithmetic over these rows, so a reporting bug should
    # never force a second pass -- the first attempt crashed in exactly that way.
    dump = root / "diagnostics"
    dump.mkdir(parents=True, exist_ok=True)
    with gzip.open(dump / f"SERIES_DEPTH_ROWS_{output_name}.json.gz", "wt") as handle:
        json.dump(rows, handle)
    artifact_volume.commit()
    print(f"[{time.perf_counter()-started:6.1f}s] wrote scored rows", flush=True)

    def aggregate(selected):
        return {
            "entries": len(selected),
            "identity_nll": statistics.mean(r["identity_nll"] for r in selected),
            "family_nll": statistics.mean(r["family_nll"] for r in selected),
            "joint_nll": statistics.mean(r["teacher_successor_nll"] for r in selected),
        }

    def table(key_fn, edges, title):
        grouped = collections.defaultdict(list)
        for row in rows:
            grouped[_label(key_fn(row), edges)].append(row)
        print(f"\n=== {title} (all families pooled) ===", flush=True)
        print(f"{'bucket':>10} {'n':>7} {'identity':>9} {'family':>8} {'joint':>7}",
              flush=True)
        out = {}
        for label in sorted(grouped, key=_order):
            cell = grouped[label]
            if len(cell) < MINIMUM_CELL:
                out[label] = {"available": False, "entries": len(cell)}
                continue
            measured = aggregate(cell)
            out[label] = {"available": True, **measured}
            print(f"{label:>10} {measured['entries']:7,} "
                  f"{measured['identity_nll']:9.2f} {measured['family_nll']:8.2f} "
                  f"{measured['joint_nll']:7.2f}", flush=True)
        return out

    pooled_depth = table(lambda r: r["panel_depth"], DEPTH_EDGES, "SERIES DEPTH")
    pooled_support = table(lambda r: r["train_support"], SUPPORT_EDGES,
                           "TRAIN SCAFFOLD SUPPORT")

    # Depth within fixed family and provenance -- the comparison that matters,
    # because pooling lets family mix masquerade as a depth effect.
    print("\n=== SERIES DEPTH within family x provenance ===", flush=True)
    print(f"{'family':20} {'prov':>10} {'depth':>8} {'n':>6} {'identity':>9}", flush=True)
    conditioned: dict[str, Any] = {}
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(row["model_family"], row["provenance"],
                 _label(row["panel_depth"], DEPTH_EDGES))].append(row)
    for family in sorted({r["model_family"] for r in rows}):
        for kind in ("real", "synthetic"):
            labels = sorted(
                {k[2] for k in grouped if k[0] == family and k[1] == kind},
                key=_order)
            for label in labels:
                cell = grouped[(family, kind, label)]
                key = f"{family}|{kind}|{label}"
                if len(cell) < MINIMUM_CELL:
                    conditioned[key] = {"available": False, "entries": len(cell)}
                    continue
                measured = aggregate(cell)
                conditioned[key] = {"available": True, **measured}
                print(f"{family:20} {kind:>10} {label:>8} {measured['entries']:6,} "
                      f"{measured['identity_nll']:9.2f}", flush=True)

    return {
        "schema": "compose.editing_v2.r_theta_series_depth",
        "status": "DIAGNOSTIC_EVIDENCE_ONLY_NO_AUTHORITY",
        "checkpoint_step": int(checkpoint["completed_steps"]),
        "panel_entries": len(rows),
        "by_series_depth": pooled_depth,
        "by_train_scaffold_support": pooled_support,
        "by_family_provenance_depth": conditioned,
        "reading": (
            "Identity NLL rising monotonically with panel_depth at fixed family, "
            "provenance and train_support confirms that dense analogue-series "
            "discrimination is the mechanism. Flat, or rising only as "
            "train_support falls, means series structure is not the mechanism "
            "and it remains unidentified."
        ),
    }


@app.local_entrypoint()
def main(output_name: str = "pilot_01") -> None:
    print(json.dumps(diagnose_series_depth.remote(output_name=output_name),
                     indent=2, sort_keys=True))
