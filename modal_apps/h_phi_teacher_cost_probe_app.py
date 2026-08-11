"""Cost probe for h_phi teacher data. Measures, does not generate a dataset.

FROZEN TRAINING LAW (settled before any architecture exists)
------------------------------------------------------------
    V_G(x, z, b) = the outcome greedy continuation achieves from molecule x
                   toward target z with b edits remaining

exactly as the rollout controller computed it -- so h_phi approximates the
teacher that was demonstrated useful, rather than a new objective invented for
the network's convenience. Two heads, matching how the controller ranks:

    recovery head    does greedy continuation reach z within the budget?
    similarity head  what target similarity does it eventually achieve?

WHAT THIS PROBE ANSWERS
-----------------------
    teacher labels per kernel call     -> what a dataset of size N costs
    positive-recovery prevalence       -> whether the recovery head has signal
                                          or is nearly all zeros
    cache hit rate                     -> how much sharing actually helps

Only then is dataset size chosen. Guessing it first is how the earlier cost
estimates went wrong.

DATA PROVENANCE
---------------
Pairs come from HELD-IN molecules. Both endpoints are training sources, and the
training-source and reserve key sets are disjoint (measured, zero overlap), so a
teacher pair cannot collide with the reserve-mined evaluation panel on either
endpoint. The sealed 67 are not touched here, and neither are the 24.

CPU ONLY.
"""

from __future__ import annotations

import collections
import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = (
    _base_image
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
    .add_local_file(
        ROOT / "diagnostics/editing_v2_heldin_teacher_pairs.json",
        str(REMOTE_ROOT / "diagnostics/editing_v2_heldin_teacher_pairs.json"),
        copy=True)
)
app = modal.App("compose-v4-h-phi-teacher-cost-probe")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT = 0.5
#: Candidate strata labelled at each decision state. Never all ~500 successors:
#: the point is a useful set, not an exhaustive one.
TOP_SIMILARITY = 4
TOP_REFERENCE = 2
RANDOM_LEGAL = 2


@app.function(
    image=image, cpu=2.0, memory=12 * 1024, timeout=2 * 60 * 60,
    max_containers=8, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def probe(task: dict[str, Any]) -> dict[str, Any]:
    """Label one held-in transformation and report what it cost."""

    import numpy as np
    import sys
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.drd2_oracle import tanimoto_to
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
        canonical_successor_result,
    )
    from compose_v4.rewrite.kernel import canonical_state_key

    started = time.perf_counter()
    artifact_volume.reload()
    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    source = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]),
        repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=bundle)
    model = runtime.model
    checkpoint = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                            map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["selected_model_state"], strict=True)
    model.eval()

    rng = np.random.default_rng(task["index"] + 7717)
    slots = int(task["slots"])
    budget = int(task["steps"])
    start_key = canonical_state_key(
        pad_molecular_graph(smiles_to_molecular_graph(task["source"]), slots))
    target_key = canonical_state_key(
        pad_molecular_graph(smiles_to_molecular_graph(task["target"]), slots))

    cache: dict[str, list[tuple[str, float]]] = {}
    calls = 0
    hits = 0

    def successors(key: str) -> list[tuple[str, float]]:
        nonlocal calls, hits
        if key in cache:
            hits += 1
            return cache[key]
        try:
            state = pad_molecular_graph(smiles_to_molecular_graph(key), slots)
        except Exception:  # noqa: BLE001
            cache[key] = []
            return []
        calls += 1
        with torch.no_grad():
            result = canonical_successor_result(model, state, float(TIME_POINT))
        cache[key] = [(s.key, float(s.probability)) for s in result.batch.successors]
        return cache[key]

    similarity: dict[str, float] = {}

    def sim(key: str) -> float:
        if key not in similarity:
            similarity[key] = (1.0 if key == target_key
                               else tanimoto_to(target_key, key))
        return similarity[key]

    # V_G cached by (state, remaining). The target is fixed per pair, so it does
    # not need to be part of the key here; a multi-target generator would add it.
    teacher: dict[tuple[str, int], tuple[bool, float]] = {}
    teacher_hits = 0

    def value(key: str, remaining: int) -> tuple[bool, float]:
        nonlocal teacher_hits
        if (key, remaining) in teacher:
            teacher_hits += 1
            return teacher[(key, remaining)]
        current, best = key, sim(key)
        recovered = current == target_key
        for _ in range(remaining):
            if recovered:
                break
            rows = successors(current)
            if not rows:
                break
            current = max(rows, key=lambda row: (sim(row[0]), row[0]))[0]
            best = max(best, sim(current))
            recovered = current == target_key
        out = (recovered, 1.0 if recovered else best)
        teacher[(key, remaining)] = out
        return out

    labels: list[dict[str, Any]] = []
    current = start_key
    for step in range(budget):
        remaining = budget - step
        rows = successors(current)
        if not rows:
            break
        keys = [r[0] for r in rows]
        immediate = np.array([sim(k) for k in keys])
        reference = np.array([r[1] for r in rows])
        greedy_index = int(np.argmax(immediate))

        picked = [greedy_index]
        seen = {greedy_index}
        for stratum, count in ((-immediate, TOP_SIMILARITY),
                               (-reference, TOP_REFERENCE)):
            for index in np.argsort(stratum)[:count]:
                if int(index) not in seen:
                    picked.append(int(index))
                    seen.add(int(index))
        rest = [i for i in range(len(keys)) if i not in seen]
        if rest:
            for index in rng.choice(rest, size=min(RANDOM_LEGAL, len(rest)),
                                    replace=False):
                picked.append(int(index))

        for index in picked:
            recovered, best = value(keys[index], remaining - 1)
            labels.append({
                "state": keys[index], "remaining": remaining - 1,
                "recovery": bool(recovered), "similarity": float(best),
                "immediate": float(immediate[index]),
                "is_greedy_action": bool(index == greedy_index),
            })
        current = keys[greedy_index]
        if current == target_key:
            break

    positives = sum(1 for row in labels if row["recovery"])
    seconds = time.perf_counter() - started
    print(f"  pair {task['index']}: {len(labels)} labels, {calls} kernel calls, "
          f"{positives} positive ({positives/max(len(labels),1):.0%}), "
          f"{seconds:.0f}s", flush=True)
    return {
        "index": task["index"], "steps": budget,
        "labels": len(labels), "positives": positives,
        "kernel_calls": calls, "cache_hits": hits,
        "teacher_cache_hits": teacher_hits,
        "seconds": round(seconds, 1),
        "label_rows": labels,
    }


@app.local_entrypoint()
def main(pairs: int = 6) -> None:
    rows = json.loads(
        (ROOT / "diagnostics/editing_v2_heldin_teacher_pairs.json").read_text()
    )["transformations"]
    # Spread across horizons so the cost estimate is not taken from the cheapest.
    chosen: list[dict[str, Any]] = []
    per = max(1, pairs // 3)
    for length in (4, 5, 6):
        band = [r for r in rows if r["steps"] == length][:per]
        chosen.extend(band)
    tasks = [{**row, "index": i} for i, row in enumerate(chosen)]
    print(f"probing {len(tasks)} held-in pairs "
          f"({collections.Counter(t['steps'] for t in tasks)})")

    results = [r for r in probe.map(tasks) if r]
    labels = sum(r["labels"] for r in results)
    calls = sum(r["kernel_calls"] for r in results)
    positives = sum(r["positives"] for r in results)
    seconds = sum(r["seconds"] for r in results)

    print(f"\n  labels {labels:,}   kernel calls {calls:,}   "
          f"labels per call {labels/max(calls,1):.2f}")
    print(f"  positive-recovery prevalence {positives}/{labels} "
          f"({positives/max(labels,1):.1%})")
    print(f"  cache hits {sum(r['cache_hits'] for r in results):,} enumeration, "
          f"{sum(r['teacher_cache_hits'] for r in results):,} teacher")
    print(f"  {seconds/max(len(results),1):.0f}s per pair, "
          f"{labels/max(len(results),1):.1f} labels per pair")

    per_pair = labels / max(len(results), 1)
    calls_per_pair = calls / max(len(results), 1)
    for size in (5_000, 20_000, 50_000):
        need = size / max(per_pair, 1e-9)
        print(f"  {size:,} labels -> ~{need:,.0f} pairs, "
              f"~{need*calls_per_pair:,.0f} kernel calls")

    destination = ROOT / "diagnostics/editing_v2_h_phi_teacher_cost_probe.json"
    destination.write_text(json.dumps({
        "schema": "compose.editing_v2.h_phi_teacher_cost_probe",
        "status": "COST_MEASUREMENT_ONLY_NO_DATASET_GENERATED",
        "training_law": (
            "V_G(x, z, b) = outcome of greedy continuation from x toward z with "
            "b edits remaining; two heads, recovery and terminal similarity"),
        "labels": labels, "kernel_calls": calls,
        "labels_per_call": labels / max(calls, 1),
        "positive_prevalence": positives / max(labels, 1),
        "per_pair": results,
    }, indent=2) + "\n")
    print(f"  wrote {destination}")
