"""Experiment C0: is there a planning problem here at all?

THE QUESTION
------------
Before training a value network h_phi, establish that future value would change
decisions that matter.  If the action with the best estimated FUTURE value is
almost always the action with the best IMMEDIATE value, then greedy already is
the planner on this task, and amortising an expensive planner into h_phi buys
nothing.  This probe answers that for ~a tenth of the cost of the full six-arm
pilot, and gates it.

WHAT IS MEASURED, AND WHY IT IS MEASURED THIS WAY
-------------------------------------------------
At each decision state along a greedy trajectory, every mask-legal successor is
scored for immediate DRD2 (exact, no estimation), and a subset is scored for
future value by Monte-Carlo rollout under the frozen reference law.

Global rank correlation is deliberately NOT the deciding statistic.  With ~600
successors, immediate and future value can agree across hundreds of obviously
bad edits while disagreeing on the handful that matter.  The decisive statistics
are about the top action:

    top-1 disagreement    does MC value pick a different edit than greedy?
    greedy future regret  h(MC-best) - h(greedy-best)
    sacrifice-to-win      is the MC-best edit WORSE immediately and BETTER
                          after the remaining horizon?

The last one names the phenomenon directly: the best route sometimes requires
an edit that does not look best right now.

WINNER'S CURSE
--------------
h is estimated from few rollouts, so argmax over K noisy estimates is biased
upward -- it would inflate both disagreement and regret, i.e. bias the result
toward "planning helps", which is the dangerous direction.  So selection and
evaluation use INDEPENDENT rollout samples: MC-best is chosen on one sample and
the reported regret is computed on a fresh one.  Immediate value g needs no such
care; it is exact.

The candidate subset is drawn from the reference law, NOT from the top of the
g-ranking.  Preselecting by g would make a sacrificial edit -- low g, high h --
impossible to observe by construction.  Because only K of ~600 successors are
evaluated, the measured MC-best is a lower bound on the true one, so any
discordance found here understates the real planning signal.

CPU ONLY.  Executor and oracle work; no GPU.
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
        ROOT / "artifacts/oracles/drd2_svm_v1/drd2_svm_parameters.npz",
        str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1/drd2_svm_parameters.npz"),
        copy=True)
    .add_local_file(
        ROOT / "artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json",
        str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json"),
        copy=True)
)
app = modal.App("compose-v4-experiment-c0-planning-signal")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
ORACLE_DIR = REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1"

# ---- preregistered task definition ---------------------------------------
#: Source molecules must start below this DRD2 score.
SOURCE_CEILING = 0.05
#: A molecule counts as solved at or above this DRD2 score.
TARGET_FLOOR = 0.5
#: Hard similarity constraint to the SOURCE, applied to every arm alike.
SIMILARITY_FLOOR = 0.4
#: Productive edits available.
BUDGET = 6
#: Steps of rollout beyond a candidate when estimating its future value.
LOOKAHEAD_DEPTH = 2
#: Successors given a Monte-Carlo value estimate at each decision state.
CANDIDATES = 5
#: Rollouts per candidate for SELECTING the MC-best action.
ROLLOUTS_SELECT = 2
#: Fresh, independent rollouts for SCORING the selected action (winner's curse).
ROLLOUTS_EVAL = 4
#: Kernel time point, matching the training and partition conventions.
TIME_POINT = 0.5


@app.function(
    image=image, cpu=8.0, memory=64 * 1024, timeout=60 * 60,
    max_containers=1, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def select_sources(wanted: int = 16, scan_limit: int = 4000) -> dict[str, Any]:
    """Choose probe sources on OUTCOME-INDEPENDENT criteria only.

    Representable by COMPOSE, genuinely held out, below the DRD2 source ceiling,
    and not already solved.  Deliberately NO rollout-based reachability screen:
    removing sources that look hard under exploratory rollouts would discard
    exactly the instances where planning is supposed to pay, and "unsolvable in
    six edits" is a scientifically meaningful outcome rather than grounds for
    exclusion.
    """

    import gzip
    import sys

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.data.corpus_training_library import load_corpus_training_library
    from compose_v4.oracles.drd2_numpy import DRD2Oracle

    started = time.perf_counter()
    artifact_volume.reload()
    inputs = Path(RUN_ROOT) / "run_inputs"
    paths = json.loads((inputs / "RUN_PATHS.json").read_text())
    resolution = json.loads(
        (inputs / "editing_v2_split_precedence_resolution.json").read_text())

    library = load_corpus_training_library(
        [Path(r) for r in paths["corpus_roots"]],
        excluded_sources=resolution["excluded_source_keys"]["train"],
        verify_state_roundtrip=False)
    with gzip.open(
            inputs / "editing_v2_matched_validation_reserve_ids.json.gz", "rt") as handle:
        reserve = json.load(handle)
    reserve_ids = set(reserve["reserve_entry_ids"])
    print(f"[{time.perf_counter()-started:6.1f}s] library loaded", flush=True)

    # One representative entry per reserve SOURCE.
    representative: dict[str, str] = {}
    for entry in library.entries:
        if entry.entry_id not in reserve_ids:
            continue
        key = entry.teacher_fiber.state_support.source_key
        representative.setdefault(key, entry.entry_id)
    candidates = sorted(representative)[:scan_limit]
    print(f"  {len(representative):,} reserve sources; scanning {len(candidates):,}",
          flush=True)

    oracle = DRD2Oracle.from_manifest(ORACLE_DIR / "drd2_oracle_manifest.json")
    scores = oracle.score_many(candidates)

    # The probe containers need exactly one thing from the library: the slot
    # count its own padding assigned to this source.  Resolving it here means
    # they never load the corpus at all, which is most of their memory and most
    # of their startup.
    eligible = []
    for key, score in zip(candidates, scores):
        if score >= SOURCE_CEILING:
            continue
        entry_id = representative[key]
        state, _f, _e = library.inputs_for([entry_id])
        eligible.append({
            "source": key,
            "entry_id": entry_id,
            "drd2": float(score),
            "slots": int(state[0].atom_types.shape[0]),
            "heavy_atoms": int((state[0].atom_types >= 0).sum()),
        })
        if len(eligible) >= wanted * 4:
            break
    already_solved = int((scores >= TARGET_FLOOR).sum())
    print(f"  eligible (DRD2 < {SOURCE_CEILING}): {len(eligible):,}; "
          f"already solved: {already_solved}", flush=True)

    chosen = eligible[:wanted]
    payload = {
        "schema": "compose.editing_v2.c0_source_selection",
        "criteria": {
            "source_ceiling": SOURCE_CEILING, "target_floor": TARGET_FLOOR,
            "outcome_independent": True,
            "note": "no rollout-based reachability screen; unsolvable sources are "
                    "reported, not excluded",
        },
        "reserve_sources": len(representative),
        "scanned": len(candidates),
        "eligible": len(eligible),
        "already_solved": already_solved,
        "selected": chosen,
    }
    out = Path(RUN_ROOT) / "c0"
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    artifact_volume.commit()
    return payload


@app.function(
    image=image, cpu=2.0, memory=12 * 1024, timeout=4 * 60 * 60,
    max_containers=20, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def probe_source(task: dict[str, Any]) -> dict[str, Any]:
    """Greedy trajectory from one source, with an MC value probe at each state."""

    import numpy as np
    import sys
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
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
    from compose_v4.oracles.drd2_numpy import DRD2Oracle, tanimoto_to

    started = time.perf_counter()
    source_key = task["source"]
    rng = np.random.default_rng(task["seed"])
    artifact_volume.reload()
    inputs = Path(RUN_ROOT) / "run_inputs"
    paths = json.loads((inputs / "RUN_PATHS.json").read_text())

    runtime_source = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]),
        repo_root=REMOTE_ROOT)
    state_bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        runtime_source, materialized_state=state_bundle)
    model = runtime.model
    checkpoint = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                            map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["selected_model_state"], strict=True)
    model.eval()

    # The slot count is INHERITED from the library's own padding (resolved during
    # selection) rather than chosen here, so every state the kernel sees carries
    # the same insertion headroom it was trained with.  Rebuilding the start
    # state from its canonical key reproduces the library's state exactly while
    # letting this container skip loading the corpus.
    slots = int(task["slots"])
    start_state = pad_molecular_graph(smiles_to_molecular_graph(source_key), slots)
    oracle = DRD2Oracle.from_manifest(ORACLE_DIR / "drd2_oracle_manifest.json")
    print(f"[{time.perf_counter()-started:6.1f}s] ready: {source_key} "
          f"({task['heavy_atoms']} heavy atoms in {slots} slots, "
          f"step {checkpoint['selected_step']:,})", flush=True)

    enumeration_cache: dict[str, list[tuple[str, float]]] = {}
    kernel_calls = 0

    def successors(state, key: str) -> list[tuple[str, float]]:
        """Mask-legal successors of `state` with reference-law probabilities."""

        nonlocal kernel_calls
        if key in enumeration_cache:
            return enumeration_cache[key]
        kernel_calls += 1
        with torch.no_grad():
            result = canonical_successor_result(model, state, float(TIME_POINT))
        rows: list[tuple[str, float]] = []
        for successor in result.batch.successors:
            # The similarity constraint is to the SOURCE, not to the current
            # state, and binds every arm identically.
            if tanimoto_to(source_key, successor.key) >= SIMILARITY_FLOOR:
                rows.append((successor.key, float(successor.probability)))
        enumeration_cache[key] = rows
        return rows

    def rebuild(key: str):
        try:
            return pad_molecular_graph(smiles_to_molecular_graph(key), slots)
        except Exception:  # noqa: BLE001 - an unrepresentable successor is skipped
            return None

    def sample_index(probabilities: np.ndarray) -> int:
        total = probabilities.sum()
        if total <= 0:
            return int(rng.integers(len(probabilities)))
        return int(rng.choice(len(probabilities), p=probabilities / total))

    def rollout_value(key: str, depth: int) -> float:
        """max DRD2 along a `depth`-step masked-reference rollout starting at `key`.

        The task is to REACH an active molecule within the budget, not to be
        active exactly at the horizon, so the value of a path is the best
        molecule on it rather than its endpoint.
        """

        best = float(oracle.score(key))
        current_key = key
        for _ in range(depth):
            state = rebuild(current_key)
            if state is None:
                break
            rows = successors(state, current_key)
            if not rows:
                break
            keys = [r[0] for r in rows]
            probabilities = np.array([r[1] for r in rows], dtype=np.float64)
            current_key = keys[sample_index(probabilities)]
            best = max(best, float(oracle.score(current_key)))
        return best

    def estimate_value(key: str, depth: int, rollouts: int) -> float:
        if depth <= 0:
            return float(oracle.score(key))
        return float(np.mean([rollout_value(key, depth) for _ in range(rollouts)]))

    decisions: list[dict[str, Any]] = []
    current_key = source_key
    current_state = start_state
    trajectory = [source_key]
    solved_at = None

    for step in range(BUDGET):
        remaining = BUDGET - step
        rows = successors(current_state, current_key)
        if not rows:
            print(f"  step {step}: no mask-legal successors; stopping", flush=True)
            break

        keys = [r[0] for r in rows]
        reference = np.array([r[1] for r in rows], dtype=np.float64)
        immediate = oracle.score_many(keys)          # exact, every legal successor
        greedy_index = int(np.argmax(immediate))

        # Candidate subset: greedy's choice always, plus draws from the reference
        # law.  Never top-g preselection -- that would make a sacrificial edit
        # unobservable.
        pool = [i for i in range(len(keys)) if i != greedy_index]
        weights = reference[pool]
        take = min(CANDIDATES - 1, len(pool))
        if take > 0 and weights.sum() > 0:
            drawn = rng.choice(pool, size=take, replace=False,
                               p=weights / weights.sum())
        elif take > 0:
            drawn = rng.choice(pool, size=take, replace=False)
        else:
            drawn = np.array([], dtype=int)
        candidate_indices = [greedy_index] + [int(i) for i in drawn]

        depth = min(LOOKAHEAD_DEPTH, remaining - 1)
        selection = {i: estimate_value(keys[i], depth, ROLLOUTS_SELECT)
                     for i in candidate_indices}
        mc_index = max(selection, key=selection.get)

        # Fresh, independent rollouts for the reported comparison.
        fresh_mc = estimate_value(keys[mc_index], depth, ROLLOUTS_EVAL)
        fresh_greedy = (fresh_mc if mc_index == greedy_index
                        else estimate_value(keys[greedy_index], depth, ROLLOUTS_EVAL))

        decisions.append({
            "step": step,
            "remaining_budget": remaining,
            "lookahead_steps": depth,
            "state": current_key,
            "legal_successors": len(keys),
            "greedy_key": keys[greedy_index],
            "mc_key": keys[mc_index],
            "greedy_immediate": float(immediate[greedy_index]),
            "mc_immediate": float(immediate[mc_index]),
            "greedy_value_selection": float(selection[greedy_index]),
            "mc_value_selection": float(selection[mc_index]),
            "greedy_value_fresh": float(fresh_greedy),
            "mc_value_fresh": float(fresh_mc),
            "top1_disagreement": bool(mc_index != greedy_index),
            "fresh_regret": float(fresh_mc - fresh_greedy),
            "immediate_sacrifice": float(immediate[greedy_index]
                                         - immediate[mc_index]),
            "sacrifice_to_win": bool(
                mc_index != greedy_index
                and immediate[mc_index] < immediate[greedy_index]
                and fresh_mc > fresh_greedy),
            "best_immediate_available": float(immediate.max()),
        })
        print(f"  step {step}: {len(keys):4d} legal  g*={immediate.max():.4f}  "
              f"disagree={mc_index != greedy_index}  "
              f"regret={fresh_mc - fresh_greedy:+.4f}  "
              f"({time.perf_counter()-started:.0f}s, {kernel_calls} kernel calls)",
              flush=True)

        # Advance greedily: the probe asks what a planner would have done at the
        # states the greedy comparator actually reaches.
        current_key = keys[greedy_index]
        nxt = rebuild(current_key)
        if nxt is None:
            break
        current_state = nxt
        trajectory.append(current_key)
        if immediate[greedy_index] >= TARGET_FLOOR and solved_at is None:
            solved_at = step + 1
            break

    payload = {
        "schema": "compose.editing_v2.c0_probe",
        "source": source_key,
        "entry_id": task["entry_id"],
        "source_drd2": task["drd2"],
        "seed": task["seed"],
        "selected_step": int(checkpoint["selected_step"]),
        "config": {
            "budget": BUDGET, "similarity_floor": SIMILARITY_FLOOR,
            "target_floor": TARGET_FLOOR, "lookahead_depth": LOOKAHEAD_DEPTH,
            "candidates": CANDIDATES, "rollouts_select": ROLLOUTS_SELECT,
            "rollouts_eval": ROLLOUTS_EVAL, "time_point": TIME_POINT,
        },
        "greedy_trajectory": trajectory,
        "greedy_solved_at": solved_at,
        "decisions": decisions,
        "kernel_calls": kernel_calls,
        "seconds": round(time.perf_counter() - started, 1),
    }
    out = Path(RUN_ROOT) / "c0" / "probes"
    out.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in source_key)[:80]
    (out / f"{task['index']:03d}_{safe}.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    artifact_volume.commit()
    print(f"[{time.perf_counter()-started:6.1f}s] done: {len(decisions)} decisions, "
          f"{kernel_calls} kernel calls", flush=True)
    return payload


@app.local_entrypoint()
def main(sources: int = 16) -> None:
    selection = select_sources.remote(wanted=sources)
    chosen = selection["selected"]
    print(f"\n{len(chosen)} sources selected from {selection['eligible']:,} eligible "
          f"({selection['reserve_sources']:,} reserve sources scanned)")
    if not chosen:
        raise SystemExit("no eligible sources; nothing to probe")

    tasks = [{**row, "index": i, "seed": 20260811 + i}
             for i, row in enumerate(chosen)]
    results = [r for r in probe_source.map(tasks) if r]

    decisions = [d for r in results for d in r["decisions"]
                 if d["lookahead_steps"] >= 1]
    print(f"\n{len(decisions)} decision states with a non-trivial horizon "
          f"(from {len(results)} sources)")
    if not decisions:
        raise SystemExit("no decision states with lookahead; nothing to report")

    disagreements = sum(d["top1_disagreement"] for d in decisions)
    sacrifices = sum(d["sacrifice_to_win"] for d in decisions)
    regrets = [d["fresh_regret"] for d in decisions]
    positive = sum(1 for r in regrets if r > 0)
    mean_regret = sum(regrets) / len(regrets)

    print(f"\n  top-1 disagreement   {disagreements:4d} / {len(decisions)} "
          f"({disagreements/len(decisions):6.1%})")
    print(f"  sacrifice-to-win     {sacrifices:4d} / {len(decisions)} "
          f"({sacrifices/len(decisions):6.1%})")
    print(f"  fresh regret > 0     {positive:4d} / {len(decisions)} "
          f"({positive/len(decisions):6.1%})")
    print(f"  mean fresh regret    {mean_regret:+.5f}")
    print(f"  total kernel calls   {sum(r['kernel_calls'] for r in results):,}")

    # Self-calibrating null.  Among decisions where the MC action differs from
    # greedy AND is immediately worse, "no planning signal" predicts the fresh
    # regret is positive about half the time -- the fresh sample is independent
    # of the one that selected the action, so under the null its sign is a coin
    # flip.  A rate near 50% is noise however large the disagreement count is;
    # a rate well above 50% is the phenomenon.
    sacrificial = [d for d in decisions
                   if d["top1_disagreement"]
                   and d["mc_immediate"] < d["greedy_immediate"]]
    if sacrificial:
        won = sum(1 for d in sacrificial if d["fresh_regret"] > 0)
        print(f"\n  immediately-worse MC actions: {len(sacrificial)}")
        print(f"    of those, better after the horizon: {won} "
              f"({won/len(sacrificial):.1%})  [null predicts ~50%]")
        mean_sacrifice = sum(d["immediate_sacrifice"] for d in sacrificial) / len(sacrificial)
        print(f"    mean immediate DRD2 given up: {mean_sacrifice:+.5f}")

    verdict = ("GO -- future value changes consequential decisions"
               if disagreements / len(decisions) >= 0.15 and mean_regret > 0
               else "STOP / INCONCLUSIVE -- see the null comparison above")
    print(f"\n  {verdict}")

    summary = {
        "schema": "compose.editing_v2.c0_summary",
        "sources": len(results),
        "decision_states": len(decisions),
        "top1_disagreement": disagreements,
        "sacrifice_to_win": sacrifices,
        "fresh_regret_positive": positive,
        "mean_fresh_regret": mean_regret,
        "sacrificial_actions": len(sacrificial),
        "sacrificial_won": (sum(1 for d in sacrificial if d["fresh_regret"] > 0)
                            if sacrificial else 0),
        "kernel_calls": sum(r["kernel_calls"] for r in results),
        "verdict": verdict,
    }
    Path("diagnostics/editing_v2_experiment_c0_planning_signal.json").write_text(
        json.dumps({"summary": summary, "per_source": results}, indent=2) + "\n")
    print("  wrote diagnostics/editing_v2_experiment_c0_planning_signal.json")
