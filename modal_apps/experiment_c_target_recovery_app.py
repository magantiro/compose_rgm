"""Experiment C pilot: greedy vs future-aware control on known-reachable targets.

PREREGISTERED HYPOTHESIS
------------------------
Remaining-budget hypothesis: future-aware value control will improve exact
recovery of held-out, multi-step reachable molecular targets relative to myopic
control, with the advantage expected to increase with transformation horizon.

Deliberately NOT "will substantially outperform" -- the experiment has to earn
the adverb. A null is a publishable result here and is reported as one.

WHY THIS PANEL ANSWERS WHAT DRD2 COULD NOT
------------------------------------------
Every target is KNOWN REACHABLE within the declared budget: the pair was
nominated by a real shared-core relationship and a path was compiled and
replay-verified after the split. So a failure is a control failure, never "the
goal was impossible" -- which is the ambiguity that made the DRD2 null hard to
read.

Both endpoints are held out (the reserve source keys are disjoint from the
96,094-key training-source universe) and no supervised transition of any
accepted pair was used in training.

THE VALUE ESTIMATOR, AND WHY IT IS NOT RANDOM ROLLOUTS
-----------------------------------------------------
Sampling the reference law forward from a state five edits from the target
reaches that exact molecule with probability ~0, so a random-rollout value would
be pure noise and would produce a null for a measurement reason rather than a
scientific one -- exactly the failure mode the sparse DRD2 landscape caused.

So this uses the classic ROLLOUT ALGORITHM: each candidate is evaluated by
simulating a base policy to the horizon.  With a deterministic greedy base,

    h_b(y) = the outcome greedy actually achieves from y with b-1 edits left

which is (a) over the FULL remaining budget rather than a truncated depth,
(b) deterministic, so there is no rollout noise and no winner's curse to
correct, and (c) >= greedy by policy improvement -- so if it loses, it loses for
a real reason rather than a sampling one.

It also makes the sacrifice measurement exact: "the planner chose a
lower-immediate-similarity edit" is a fact about the run, not an estimate.

CPU ONLY.  No GPU, no h_phi, no SMC or Boltzmann arm -- this is the gate that
decides whether any of those get built.
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
    ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = (
    _base_image
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
    .add_local_file(
        ROOT / "diagnostics/editing_v2_fresh_panel_feasibility.json",
        str(REMOTE_ROOT / "diagnostics/editing_v2_fresh_panel_feasibility.json"),
        copy=True)
)
app = modal.App("compose-v4-experiment-c-target-recovery")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT = 0.5
#: Successors given a full-horizon rollout evaluation at each decision state.
CANDIDATES_TOP_SIMILARITY = 4
CANDIDATES_TOP_REFERENCE = 2


def _runtime(paths: dict[str, Any]):
    """Frozen R_theta plus the executor, identical for every arm."""

    import torch

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

    source = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]),
        repo_root=REMOTE_ROOT)
    state = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=state)
    checkpoint = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                            map_location="cpu", weights_only=False)
    runtime.model.load_state_dict(checkpoint["selected_model_state"], strict=True)
    runtime.model.eval()
    return runtime.model, int(checkpoint["selected_step"])


@app.function(
    image=image, cpu=2.0, memory=12 * 1024, timeout=4 * 60 * 60,
    max_containers=26, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def recover(task: dict[str, Any]) -> dict[str, Any]:
    """Run both arms on one held-out source->target transformation."""

    import numpy as np
    import sys
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.drd2_oracle import tanimoto_to
    from compose_v4.experiments.production_successor_kernel import (
        canonical_successor_result,
    )
    from compose_v4.rewrite.kernel import canonical_state_key

    started = time.perf_counter()
    artifact_volume.reload()
    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    model, step = _runtime(paths)

    source_smiles, target_smiles = task["source"], task["target"]
    budget = int(task["steps"])
    slots = int(task["slots"])
    start = pad_molecular_graph(smiles_to_molecular_graph(source_smiles), slots)
    target_key = canonical_state_key(
        pad_molecular_graph(smiles_to_molecular_graph(target_smiles), slots))
    source_key = canonical_state_key(start)
    print(f"[{time.perf_counter()-started:6.1f}s] {source_key} -> {target_key} "
          f"budget {budget}, step {step:,}", flush=True)

    cache: dict[str, list[tuple[str, float]]] = {}
    calls = 0

    def successors(key: str) -> list[tuple[str, float]]:
        nonlocal calls
        if key in cache:
            return cache[key]
        try:
            state = pad_molecular_graph(smiles_to_molecular_graph(key), slots)
        except Exception:  # noqa: BLE001 - unrepresentable state is a dead end
            cache[key] = []
            return []
        calls += 1
        with torch.no_grad():
            result = canonical_successor_result(model, state, float(TIME_POINT))
        rows = [(s.key, float(s.probability)) for s in result.batch.successors]
        cache[key] = rows
        return rows

    similarity: dict[str, float] = {}

    def sim(key: str) -> float:
        # Exact recovery is decided by canonical key equality, never by the
        # fingerprint, so a fingerprint collision cannot manufacture a success.
        if key not in similarity:
            similarity[key] = (1.0 if key == target_key
                               else tanimoto_to(target_key, key))
        return similarity[key]

    def greedy_step(key: str) -> str | None:
        rows = successors(key)
        if not rows:
            return None
        # Deterministic tie-break on the canonical key, so both arms see the
        # same greedy behaviour run to run.
        return max(rows, key=lambda row: (sim(row[0]), row[0]))[0]

    def greedy_from(key: str, remaining: int) -> dict[str, Any]:
        """Run the greedy base policy and report what it reached."""

        current, best, trail = key, sim(key), [key]
        recovered = current == target_key
        for _ in range(remaining):
            if recovered:
                break
            nxt = greedy_step(current)
            if nxt is None:
                break
            current = nxt
            trail.append(current)
            best = max(best, sim(current))
            if current == target_key:
                recovered = True
        return {"recovered": recovered, "best": 1.0 if recovered else best,
                "final": sim(current), "trail": trail}

    # ---- arm 1: greedy ---------------------------------------------------
    greedy = greedy_from(source_key, budget)
    greedy_calls = calls
    print(f"  greedy: recovered={greedy['recovered']} best={greedy['best']:.4f} "
          f"({calls} kernel calls)", flush=True)

    # ---- arm 2: full-horizon rollout lookahead ---------------------------
    # Same frozen R_theta, same legal support, same target, same budget.
    current = source_key
    trail = [current]
    decisions: list[dict[str, Any]] = []
    recovered = current == target_key
    best = sim(current)

    for step_index in range(budget):
        if recovered:
            break
        remaining = budget - step_index
        rows = successors(current)
        if not rows:
            break
        keys = [r[0] for r in rows]
        immediate = np.array([sim(k) for k in keys])
        reference = np.array([r[1] for r in rows])
        greedy_index = int(np.argmax(immediate))

        picked = [greedy_index]
        seen = {greedy_index}
        for stratum, count in ((-immediate, CANDIDATES_TOP_SIMILARITY),
                               (-reference, CANDIDATES_TOP_REFERENCE)):
            for index in np.argsort(stratum)[:count]:
                if int(index) not in seen:
                    picked.append(int(index))
                    seen.add(int(index))

        # h_b(y): what greedy actually achieves from y over the FULL remaining
        # budget. Deterministic, so one evaluation per candidate suffices.
        values = {}
        for index in picked:
            outcome = greedy_from(keys[index], remaining - 1)
            values[index] = (outcome["recovered"], outcome["best"])
        chosen = max(values, key=lambda i: (values[i][0], values[i][1], -i))

        decisions.append({
            "step": step_index,
            "remaining": remaining,
            "candidates": len(picked),
            "greedy_key": keys[greedy_index],
            "chosen_key": keys[chosen],
            "greedy_immediate": float(immediate[greedy_index]),
            "chosen_immediate": float(immediate[chosen]),
            "disagreed": bool(chosen != greedy_index),
            # Exact, not estimated: the planner took a worse-looking edit.
            "sacrificed_immediate": bool(
                immediate[chosen] < immediate[greedy_index]),
            "greedy_rollout_recovers": bool(values[greedy_index][0]),
            "chosen_rollout_recovers": bool(values[chosen][0]),
            "greedy_rollout_best": float(values[greedy_index][1]),
            "chosen_rollout_best": float(values[chosen][1]),
        })
        current = keys[chosen]
        trail.append(current)
        best = max(best, sim(current))
        if current == target_key:
            recovered = True

    lookahead = {"recovered": recovered, "best": 1.0 if recovered else best,
                 "final": sim(current), "trail": trail}
    print(f"  lookahead: recovered={lookahead['recovered']} "
          f"best={lookahead['best']:.4f} "
          f"disagreements={sum(d['disagreed'] for d in decisions)} "
          f"({calls} kernel calls total)", flush=True)

    payload = {
        "schema": "compose.editing_v2.c_target_recovery",
        "source": source_key, "target": target_key,
        "verified_steps": budget, "slots": slots,
        "source_heavy_atoms": task.get("size"),
        "reference_shape": task.get("shape"),
        "selected_step": step,
        "greedy": greedy, "lookahead": lookahead,
        "decisions": decisions,
        "kernel_calls_greedy": greedy_calls,
        "kernel_calls_total": calls,
        "kernel_calls_lookahead_marginal": calls - greedy_calls,
        "cost_caveat": (
            "Greedy runs first and its enumerations are cached, so the lookahead "
            "arm reuses them. Its marginal cost is honest; its standalone cost "
            "would be higher. Compare arms on marginal calls, not on a ratio."),
        "seconds": round(time.perf_counter() - started, 1),
    }
    out = Path(RUN_ROOT) / "c_recovery"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{task['index']:03d}.json").write_text(json.dumps(payload, indent=2) + "\n")
    artifact_volume.commit()
    return payload


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the pilot. Separated out so it can be exercised without Modal."""

    import collections

    results = [r for r in results if r]
    if not results:
        return {"verdict": "NO DATA"}
    g = sum(r["greedy"]["recovered"] for r in results)
    l = sum(r["lookahead"]["recovered"] for r in results)
    decisions = [d for r in results for d in r["decisions"]]
    disagreed = sum(d["disagreed"] for d in decisions)
    sacrificed = sum(d["sacrificed_immediate"] for d in decisions)
    # Did a sacrifice actually pay? Exact, because the rollout is deterministic.
    paid = sum(1 for d in decisions
               if d["sacrificed_immediate"]
               and (d["chosen_rollout_recovers"] > d["greedy_rollout_recovers"]
                    or d["chosen_rollout_best"] > d["greedy_rollout_best"]))
    by_len: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"n": 0, "greedy": 0, "lookahead": 0})
    for r in results:
        cell = by_len[str(r["verified_steps"])]
        cell["n"] += 1
        cell["greedy"] += int(r["greedy"]["recovered"])
        cell["lookahead"] += int(r["lookahead"]["recovered"])

    print(f"\n  EXACT RECOVERY   greedy {g}/{len(results)}   "
          f"lookahead {l}/{len(results)}")
    for horizon in sorted(by_len, key=int):
        cell = by_len[horizon]
        print(f"    {horizon} steps: greedy {cell['greedy']}/{cell['n']}  "
              f"lookahead {cell['lookahead']}/{cell['n']}")
    print(f"  mean best similarity  greedy "
          f"{sum(r['greedy']['best'] for r in results)/len(results):.4f}   "
          f"lookahead "
          f"{sum(r['lookahead']['best'] for r in results)/len(results):.4f}")
    print(f"  decisions {len(decisions)}: disagreed {disagreed}, "
          f"sacrificed immediate similarity {sacrificed}, of which paid off {paid}")
    print(f"  kernel calls  greedy {sum(r['kernel_calls_greedy'] for r in results):,}"
          f"   total {sum(r['kernel_calls_total'] for r in results):,}")

    # Policy-improvement invariant. The base policy is deterministic and greedy's
    # own choice is always in the candidate set, so value(chosen) >= value(greedy)
    # at every decision and lookahead CANNOT lose a recovery greedy achieves.
    # A violation is a bug -- budget accounting or candidate construction -- not
    # a scientific result, and must not be reported as one.
    regressions = [r for r in results
                   if r["greedy"]["recovered"] and not r["lookahead"]["recovered"]]
    if regressions:
        verdict = (f"INVARIANT VIOLATED -- lookahead lost {len(regressions)} "
                   f"recovery/recoveries greedy achieved, which policy improvement "
                   f"forbids. Treat as a bug, not a result.")
    elif l > g:
        verdict = (f"LOOKAHEAD AHEAD -- {l} vs {g} exact recoveries; "
                   f"future value has something to buy on reachable targets")
    elif l == g and g == len(results):
        verdict = (f"BOTH SOLVE EVERYTHING ({g}/{len(results)}) -- the panel is too "
                   f"easy to separate the arms; greedy suffices on these targets")
    elif l == g:
        verdict = (f"TIE at {g}/{len(results)} -- no recovery advantage; "
                   f"compare best-similarity and the horizon trend before reading "
                   f"further")
    else:
        verdict = (f"GREEDY AHEAD -- {g} vs {l}. See the invariant note: this "
                   f"should be unreachable")
    print(f"\n  {verdict}")
    return {
        "schema": "compose.editing_v2.c_target_recovery_summary",
        "hypothesis": (
            "Remaining-budget hypothesis: future-aware value control will improve "
            "exact recovery of held-out, multi-step reachable molecular targets "
            "relative to myopic control, with the advantage expected to increase "
            "with transformation horizon."),
        "pairs": len(results),
        "greedy_recovered": g, "lookahead_recovered": l,
        "by_verified_steps": {k: dict(v) for k, v in by_len.items()},
        "mean_best_greedy": sum(r["greedy"]["best"] for r in results) / len(results),
        "mean_best_lookahead": sum(r["lookahead"]["best"] for r in results) / len(results),
        "decisions": len(decisions), "disagreed": disagreed,
        "sacrificed_immediate": sacrificed, "sacrifice_paid_off": paid,
        "kernel_calls_greedy": sum(r["kernel_calls_greedy"] for r in results),
        "kernel_calls_total": sum(r["kernel_calls_total"] for r in results),
        "policy_improvement_regressions": len(regressions),
        "verdict": verdict,
    }


@app.local_entrypoint()
def main(pairs: int = 24) -> None:
    import collections

    # No chemistry here on purpose: `modal run` executes this entrypoint in the
    # Modal CLI's own virtualenv, which has no RDKit even when the system
    # interpreter does. Sizes and slot budgets are precomputed into the artifact.
    feasibility = json.loads(
        (ROOT / "diagnostics/editing_v2_fresh_panel_feasibility.json").read_text())
    rows = feasibility["transformations"]
    missing = [r for r in rows if "slots" not in r or "source_heavy_atoms" not in r]
    if missing:
        raise SystemExit(
            f"{len(missing)} rows lack precomputed slots/size; regenerate the "
            f"feasibility artifact before running")
    for row in rows:
        row["size"] = row["source_heavy_atoms"]

    # Stratified across verified path length, spread across molecular size
    # within each band. Deterministic; no outcome is consulted.
    per_band = max(1, pairs // 3)
    selected: list[dict[str, Any]] = []
    for length in (4, 5, 6):
        band = sorted([r for r in rows if r["steps"] == length],
                      key=lambda r: r["size"])
        take = min(per_band, len(band))
        if take == 1:
            picks = [0]
        else:
            picks = [round(i * (len(band) - 1) / (take - 1)) for i in range(take)]
        selected.extend(band[i] for i in dict.fromkeys(picks))

    print(f"{len(selected)} pairs: "
          f"{dict(sorted(collections.Counter(r['steps'] for r in selected).items()))}")
    tasks = [{**row, "index": i} for i, row in enumerate(selected)]
    results = list(recover.map(tasks))

    summary = summarise(results)
    destination = ROOT / "diagnostics/editing_v2_experiment_c_target_recovery.json"
    destination.write_text(
        json.dumps({"summary": summary, "per_pair": results}, indent=2) + "\n")
    print(f"  wrote {destination}")
