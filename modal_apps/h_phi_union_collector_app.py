"""Union collector for h_phi teacher data: greedy states AND rollout-teacher states.

WHAT CHANGES, AND WHAT DOES NOT
-------------------------------
The teacher does NOT change. V_G(x, z, b) is still the outcome greedy
continuation achieves from x toward z with b edits remaining -- the same
quantity the rollout controller computed when it rescued six of greedy's twelve
failures.

What changes is only WHERE the teacher is asked for labels:

    D_states = greedy-visited states  UNION  rollout-teacher-visited states

The collector policy and the value target are different concepts. The collector
decides which states the network should learn about; V_G decides the correct
label at those states. Conflating them would quietly redefine the objective.

WHY THE UNION
-------------
Labelling only greedy-visited states teaches counterfactual ACTIONS at
on-policy STATES. But h_phi is queried at inference after it overrides greedy,
i.e. off the greedy trajectory -- exactly the region such a dataset never
contains. The states where h_phi most needs to be right would be the ones it
least saw. Rollout-teacher states are where successful overrides actually lead.

Greedy states still matter: greedy is the baseline and most decisions occur
there. Hence the union, not a replacement.

ACTION DISTRIBUTION, NOT JUST STATE DISTRIBUTION
------------------------------------------------
Fixing state coverage while leaving action coverage broken would reintroduce
the same failure one level down. So the candidate set at every collected state
includes the greedy action, high-immediate-similarity candidates,
high-R_theta-probability candidates, some random legal candidates, and -- at
rollout-visited states -- THE TEACHER-SELECTED ALTERNATIVE. Without that last
one, h_phi would never train on the actions the teacher actually chose.

NO RECURSIVE BRANCHING. Following every non-greedy successor rebuilds the
combinatorial explosion. The rollout teacher already identifies which off-greedy
branches are decision-relevant; its realized trajectory is the sample.

This run is a VERIFICATION of the generator, not a dataset and not a science
experiment. It re-measures decision coverage under the union so a collapse is
caught before ~$12 of generation.

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
app = modal.App("compose-v4-h-phi-union-collector")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT = 0.5
TOP_SIMILARITY = 4
TOP_REFERENCE = 2
RANDOM_LEGAL = 2


@app.function(
    image=image, cpu=2.0, memory=12 * 1024, timeout=60 * 60,
    max_containers=60, retries=2,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def collect(task: dict[str, Any]) -> dict[str, Any]:
    """Collect union-covered teacher labels for one held-in transformation."""

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

    # RESUME. A pair whose shard is already on the volume is never recomputed,
    # so a killed or preempted run is restarted by simply relaunching: it costs
    # only the pairs that were genuinely in flight. This is what makes a
    # multi-hour fan-out safe to interrupt.
    shard_dir = Path(RUN_ROOT) / "h_phi_teacher" / task.get("role", "train")
    shard_path = shard_dir / f"{task['index']:04d}.json"
    if shard_path.exists():
        done = json.loads(shard_path.read_text())
        print(f"  pair {task['index']}: resumed from volume "
              f"({done['labels']} labels), no recompute", flush=True)
        return done

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

    rng = np.random.default_rng(task["index"] + 4242)
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

    teacher: dict[tuple[str, int], tuple[bool, float]] = {}
    teacher_hits = 0

    def value(key: str, remaining: int) -> tuple[bool, float]:
        """V_G: unchanged from the probe and from the rollout controller."""

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

    def candidate_set(keys, immediate, reference):
        picked = [int(np.argmax(immediate))]
        seen = set(picked)
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
                seen.add(int(index))
        return picked, seen

    # ---- walk both policies, recording which visited what -----------------
    visited: dict[tuple[str, int], set[str]] = collections.defaultdict(set)
    teacher_choice: dict[tuple[str, int], str] = {}

    current = start_key
    for step in range(budget):
        remaining = budget - step
        visited[(current, remaining)].add("greedy")
        rows = successors(current)
        if not rows:
            break
        current = max(rows, key=lambda row: (sim(row[0]), row[0]))[0]
        if current == target_key:
            visited[(current, remaining - 1)].add("greedy")
            break

    current = start_key
    for step in range(budget):
        remaining = budget - step
        visited[(current, remaining)].add("rollout")
        rows = successors(current)
        if not rows:
            break
        keys = [r[0] for r in rows]
        immediate = np.array([sim(k) for k in keys])
        reference = np.array([r[1] for r in rows])
        greedy_index = int(np.argmax(immediate))
        picked, _seen = candidate_set(keys, immediate, reference)
        values = {i: value(keys[i], remaining - 1) for i in picked}
        greedy_value = values[greedy_index]
        challengers = {i: v for i, v in values.items() if v > greedy_value}
        chosen = (max(challengers, key=lambda i: (challengers[i][0],
                                                  challengers[i][1], -i))
                  if challengers else greedy_index)
        # Record what the teacher picked, so the candidate set at this state is
        # guaranteed to contain it when labels are emitted below.
        teacher_choice[(current, remaining)] = keys[chosen]
        current = keys[chosen]
        if current == target_key:
            visited[(current, remaining - 1)].add("rollout")
            break

    # ---- label the union --------------------------------------------------
    labels: list[dict[str, Any]] = []
    for (state_key, remaining), policies in sorted(visited.items()):
        if remaining <= 0:
            continue
        rows = successors(state_key)
        if not rows:
            continue
        keys = [r[0] for r in rows]
        immediate = np.array([sim(k) for k in keys])
        reference = np.array([r[1] for r in rows])
        greedy_index = int(np.argmax(immediate))
        picked, seen = candidate_set(keys, immediate, reference)

        # Action-distribution guard: the teacher's own choice must be scoreable.
        forced = teacher_choice.get((state_key, remaining))
        if forced is not None and forced in keys and keys.index(forced) not in seen:
            picked.append(keys.index(forced))

        block = []
        for index in picked:
            recovered, best = value(keys[index], remaining - 1)
            block.append({
                "pair_id": task["pair_id"],
                "decision_state": state_key,
                "candidate": keys[index],
                "remaining": remaining - 1,
                "recovery": bool(recovered),
                "similarity": float(best),
                "immediate": float(immediate[index]),
                "is_greedy_action": bool(index == greedy_index),
                "is_teacher_action": bool(forced is not None and keys[index] == forced),
                "visited_by": sorted(policies),
            })
        # Stratum is a property of the STATE, so it can only be stamped once all
        # of that state's candidates are labelled. Recorded explicitly at write
        # time rather than reconstructed later, so stratified subsampling cannot
        # silently disagree with what was generated.
        recoveries = [row["recovery"] for row in block]
        kind = ("contrastive" if (any(recoveries) and not all(recoveries))
                else "all_positive" if all(recoveries) else "all_negative")
        for row in block:
            row["state_kind"] = kind
        labels.extend(block)

    states = len({(row["decision_state"], row["remaining"]) for row in labels})
    positives = sum(1 for row in labels if row["recovery"])
    seconds = time.perf_counter() - started
    print(f"  pair {task['index']}: {states} states, {len(labels)} labels, "
          f"{calls} calls, {positives} positive, {seconds:.0f}s", flush=True)

    payload = {
        "index": task["index"], "pair_id": task["pair_id"], "steps": budget,
        "decision_states": states, "labels": len(labels), "positives": positives,
        "kernel_calls": calls, "cache_hits": hits, "teacher_cache_hits": teacher_hits,
        "seconds": round(seconds, 1), "label_rows": labels,
    }
    # Persist HERE, not only in the client. A multi-hour fan-out whose results
    # exist solely as return values is one network blip away from losing
    # everything -- the exact failure that cost an Experiment B run earlier.
    # Each container writes its own shard, so a disconnect costs only the pairs
    # still in flight.
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path.write_text(json.dumps(payload, indent=2) + "\n")
    artifact_volume.commit()
    return payload


@app.function(
    image=image, cpu=1.0, memory=4 * 1024, timeout=24 * 60 * 60,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def drive(tasks: list[dict[str, Any]], out_name: str) -> dict[str, Any]:
    """Fan out from INSIDE Modal, so nothing depends on the local client.

    `.map()` submits inputs from wherever it is called. Called locally, a
    dropped connection stops new inputs being submitted -- `--detach` keeps the
    app alive but leaves it with nothing to do. That is exactly what happened:
    a DNS failure killed the client and the run stalled at 29 of 290 pairs with
    ~20 idle containers.

    Running the map here makes submission server-side, so the job survives the
    laptop closing rather than merely surviving it gracefully.
    """

    results = [r for r in collect.map(tasks) if r]
    labels = sum(r["labels"] for r in results)
    states = sum(r["decision_states"] for r in results)
    calls = sum(r["kernel_calls"] for r in results)
    kinds: dict[str, int] = {}
    for r in results:
        for row in r["label_rows"]:
            kinds[row["state_kind"]] = kinds.get(row["state_kind"], 0) + 1

    summary = {
        "schema": "compose.editing_v2.h_phi_union_collector_check",
        "status": "GENERATED_SERVER_SIDE",
        "pairs": len(results), "decision_states": states,
        "labels": labels, "kernel_calls": calls,
        "label_strata": kinds,
        "per_pair": results,
    }
    out = Path(RUN_ROOT) / "h_phi_teacher" / out_name
    out.write_text(json.dumps(summary, indent=2) + "\n")
    artifact_volume.commit()
    print(f"driver done: {len(results)} pairs, {labels:,} labels, "
          f"{calls:,} kernel calls -> {out}", flush=True)
    return {k: v for k, v in summary.items() if k != "per_pair"}


@app.local_entrypoint()
def main(pairs: int = 12, role: str = "train",
         carve: str = "diagnostics/editing_v2_teacher_validation_carve.json",
         out: str = "diagnostics/editing_v2_h_phi_union_collector_check.json") -> None:
    carve_path = ROOT / carve
    if carve_path.exists():
        payload = json.loads(carve_path.read_text())
        if role not in payload:
            raise SystemExit(f"role {role!r} not in the carve; "
                             f"have {sorted(k for k in payload if isinstance(payload[k], list))}")
        rows = payload[role]
        print(f"carve {carve}: role={role}, {len(rows)} pairs available "
              f"(train/validation split by PAIR, made before generation)")
    else:
        rows = json.loads(
            (ROOT / "diagnostics/editing_v2_heldin_teacher_pairs.json").read_text()
        )["transformations"]
        print(f"no carve at {carve}; falling back to the raw pair list")

    # Balanced across horizons.
    chosen: list[dict[str, Any]] = []
    per = max(1, pairs // 3)
    for length in (4, 5, 6):
        band = [r for r in rows if r["steps"] == length]
        chosen.extend(band[:per])
    tasks = [{**row, "index": i, "role": role,
              "pair_id": f"{row['steps']}:{row['source']}>>{row['target']}"}
             for i, row in enumerate(chosen)]
    print(f"union-collecting {len(tasks)} held-in pairs "
          f"{dict(collections.Counter(t['steps'] for t in tasks))}")

    # Dispatch and return. The local client only kicks the driver off; it does
    # NOT feed inputs, so a dropped connection cannot stall the fan-out. Results
    # are written to the volume by the driver and by each pair's own shard;
    # reassemble locally with scripts/editing_v2_collect_teacher_shards.py.
    print("dispatching to the on-Modal driver (submission is server-side, so "
          "this survives the client disconnecting)")
    summary = drive.remote(tasks, f"aggregate_{role}.json")
    print(f"\n  {summary}")
    print(f"  aggregate written to the volume at "
          f"editing_v2/r_theta_run/h_phi_teacher/aggregate_{role}.json")
    print(f"  per-pair shards under editing_v2/r_theta_run/h_phi_teacher/{role}/")

