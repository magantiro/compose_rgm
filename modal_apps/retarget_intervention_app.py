"""Same-prefix goal intervention. H=6, switch at 3. CPU ONLY.

THE CLAIM THIS TESTS -- and the one it does NOT
-----------------------------------------------
Not "verified beats greedy". The calibration already showed that on a
target-free developability goal greedy suffices (28/30), and the sealed
exact-target panel already established future-aware control where the problem is
hard (40% -> 62%). This experiment establishes a different layer:

    once molecular history exists, the user can change what they want and
    COMPOSE continues from the molecule actually in hand rather than
    discarding the trajectory.

ONE FINAL GOAL, TWO HISTORIES
-----------------------------
The eventual requirement is identical in both conditions -- B = P AND D. Only
the information ORDER differs, which is what isolates the effect of history:

    P -> P&D   potency first, then a developability requirement arrives.
               The prefix should contain real progress on the HARD term.
    D -> P&D   developability first, then potency becomes required.
               The prefix may help, be neutral, or actively hurt.

This is better than manufacturing an antagonistic property pair: both histories
are natural, and because the destination is the same, any difference between
them is attributable to what was pursued first.

PHYSICAL SEPARATION OF THE PREFIX
---------------------------------
Prefixes are generated in a SEPARATE ENTRYPOINT, written to the volume, and
hashed into a committed artifact BEFORE any arm runs. Goal B is never
constructed in the prefix phase. This is not ceremony -- it is what proves the
prefix was not selected or steered with knowledge of the future goal.

THE FIVE ARMS, all scored by u_B and success_B
----------------------------------------------
    continue_A          ignore the switch; keep optimising A. Negative control:
                        does the outcome depend on changing the objective AT ALL?
    greedy_retarget     switch to u_B, act myopically from x_3.
    verified_retarget   switch to u_B, remaining-budget control from x_3.
    restart             DISCARD x_3, go back to x_0, same 3 remaining edits.
                        The history-value baseline -- the most important one.
    clairvoyant         B known from step 0, full 6 edits. The no-surprise
                        reference; the gap to it is the PRICE OF SURPRISE.

restart and clairvoyant use the same verified controller as verified_retarget,
so H2 and H3 are like-for-like comparisons and any difference is attributable to
the starting state and budget rather than to the controller.

INSTRUMENT DISCIPLINE
---------------------
Three defects in the calibration all had one shape: a statistic whose sign was
fixed before any data existed. Two guards here:

  * every arm records its landing key, so arm collapse is detectable rather
    than silent -- `--smoke` asserts the arms actually diverge before a panel
    is spent;
  * verified_retarget vs greedy_retarget is NOT a primary claim, because
    policy improvement guarantees its direction. The primary comparisons
    (retarget vs continue_A, retarget vs restart, retarget vs clairvoyant) all
    differ in STARTING STATE or OBJECTIVE, so none has a guaranteed sign.
"""

from __future__ import annotations

import hashlib
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
        ROOT / "diagnostics/retarget_goal_language_normalizers.json",
        str(REMOTE_ROOT / "diagnostics/retarget_goal_language_normalizers.json"),
        copy=True)
    .add_local_file(
        ROOT / "diagnostics/retarget_calibration_cohort.json",
        str(REMOTE_ROOT / "diagnostics/retarget_calibration_cohort.json"),
        copy=True)
    .add_local_dir(ROOT / "artifacts/oracles/drd2_svm_v1",
                   str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1"), copy=True)
)
app = modal.App("compose-v4-retarget-intervention")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT = 0.5
CANONICAL_SLOTS = 48

HORIZON = 6
SWITCH_AT = 3

#: Frozen by the held-in calibration. P>=0.5 was 10/30 reachable in 3 greedy
#: edits -- neither floor nor ceiling. The developability region is unchanged
#: from calibration and is NOT retuned.
POTENCY_THRESHOLD = 0.5
QED_FLOOR = 0.6
LOGP_BOX = (1.0, 4.0)
#: GOAL RANKING -- unbounded signed IQR-normalised margins, ranked
#: lexicographically by (worst margin, mean margin). No clip, no temperature.
#:
#: The clipped soft-min this replaces was defective, not merely inelegant: a
#: clip of 1.5 bound the ENTIRE interquartile range of the potency margin
#: (held-in quartiles -6.85/-5.52/-4.24 log-odds map to -2.62/-2.11/-1.62), so
#: distinct molecules received identical scores, the potency controller ranked
#: ties by canonical key, and the flat floor propagated into P&D because P is
#: its worst term. Clipping is unnecessary here anyway: the controller
#: ENUMERATES and RANKS candidates, so nothing differentiates through the
#: oracle and no smoothing is required.
#:
#: Max-min feasibility control. No property can compensate for a violated
#: conjunct; if potency is genuinely the binding constraint it SHOULD dominate
#: until it improves; developability re-enters as soon as it becomes the worst
#: margin, and acts as the tie-break before that.

CANDIDATES_TOP_IMMEDIATE = 4
CANDIDATES_TOP_REFERENCE = 2
CANDIDATES_RANDOM = 2

HISTORIES = ("P", "D")
ARMS = ("continue_A", "greedy_retarget", "verified_retarget",
        "restart", "clairvoyant")


def _runtime():
    """Model + oracle + goal algebra. Shared by both phases."""
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, QED

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.drd2_oracle import load_default_oracle
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

    RDLogger.DisableLog("rdApp.*")
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

    norms = json.loads((REMOTE_ROOT /
                        "diagnostics/retarget_goal_language_normalizers.json"
                        ).read_text())["normalizers"]
    s_drd2, s_qed, s_logp = (norms["drd2_logodds"]["iqr"], norms["qed"]["iqr"],
                             norms["clogp"]["iqr"])
    oracle = load_default_oracle(
        str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json"))
    potency_cut = float(np.log(POTENCY_THRESHOLD / (1 - POTENCY_THRESHOLD)))

    cache: dict[str, tuple[float, float, float]] = {}

    def props(key: str):
        if key not in cache:
            mol = Chem.MolFromSmiles(key)
            if mol is None:
                cache[key] = (float("nan"),) * 3
            else:
                try:
                    qed = float(QED.qed(mol))
                except Exception:  # noqa: BLE001
                    qed = float("nan")
                cache[key] = (float(oracle.margin_many([key])[0]), qed,
                              float(Crippen.MolLogP(mol)))
        return cache[key]

    def margins(key: str, goal: str) -> list[float]:
        m, qed, logp = props(key)
        if not all(np.isfinite(v) for v in (m, qed, logp)):
            return [-10.0]
        potency = (m - potency_cut) / s_drd2
        develop = [(qed - QED_FLOOR) / s_qed,
                   min(logp - LOGP_BOX[0], LOGP_BOX[1] - logp) / s_logp]
        if goal == "P":
            return [potency]
        if goal == "D":
            return develop
        return [potency, *develop]           # B = P AND D

    def utility(key: str, goal: str):
        """Lexicographic rank key: (worst margin, mean margin). Tuples compare
        lexicographically in Python, so `max` over this IS the stated rule."""
        m = margins(key, goal)
        return (float(min(m)), float(sum(m) / len(m)))

    def success(key: str, goal: str) -> bool:
        return min(margins(key, goal)) >= 0.0

    enumeration: dict[str, list[tuple[str, float]]] = {}
    counter = {"calls": 0}

    def successors(key: str, slots: int):
        if key in enumeration:
            return enumeration[key]
        try:
            state = pad_molecular_graph(smiles_to_molecular_graph(key), slots)
        except Exception:  # noqa: BLE001
            enumeration[key] = []
            return []
        counter["calls"] += 1
        with torch.no_grad():
            result = canonical_successor_result(model, state, float(TIME_POINT))
        enumeration[key] = [(s.key, float(s.probability))
                            for s in result.batch.successors]
        return enumeration[key]

    return props, margins, utility, success, successors, counter


@app.function(image=image, cpu=2.0, memory=12 * 1024, timeout=4 * 60 * 60,
              max_containers=30, retries=1,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def build_prefix(task: dict[str, Any]) -> dict[str, Any]:
    """PHASE 1. Three edits under goal A. Goal B is never constructed here."""
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.rewrite.kernel import canonical_state_key

    started = time.perf_counter()
    _props, _m, utility, success, successors, counter = _runtime()
    slots, goal_a = int(task["slots"]), task["history"]
    start = canonical_state_key(
        pad_molecular_graph(smiles_to_molecular_graph(task["source"]), slots))

    trajectory, current = [start], start
    for _ in range(SWITCH_AT):
        rows = successors(current, slots)
        if not rows:
            break
        current = max(rows, key=lambda r: (utility(r[0], goal_a), r[0]))[0]
        trajectory.append(current)

    payload = {"index": int(task["index"]), "history": goal_a,
               "source": task["source"], "start_key": start,
               "trajectory": trajectory, "switch_state": trajectory[-1],
               "prefix_edits": len(trajectory) - 1,
               "a_worst_start": utility(start, goal_a)[0],
               "a_worst_switch": utility(trajectory[-1], goal_a)[0],
               "a_mean_start": utility(start, goal_a)[1],
               "a_mean_switch": utility(trajectory[-1], goal_a)[1],
               "kernel_calls": counter["calls"],
               "seconds": round(time.perf_counter() - started, 1)}
    out = Path(RUN_ROOT) / "retarget_prefixes"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{goal_a}_{task['index']:03d}.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    artifact_volume.commit()
    print(f"[prefix {goal_a}/{task['index']:03d}] {counter['calls']} calls "
          f"{payload['seconds']:.0f}s  worst_A {payload['a_worst_start']:+.3f} -> "
          f"{payload['a_worst_switch']:+.3f}", flush=True)
    return payload


@app.function(image=image, cpu=2.0, memory=12 * 1024, timeout=5 * 60 * 60,
              max_containers=30, retries=1,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_arms(task: dict[str, Any]) -> dict[str, Any]:
    """PHASE 2. Five arms from a prefix that was committed before B existed."""
    import random

    import numpy as np

    started = time.perf_counter()
    _props, _m, utility, success, successors, counter = _runtime()
    slots = int(task["slots"])
    start, switch = task["start_key"], task["switch_state"]
    goal_a, post = task["history"], HORIZON - SWITCH_AT
    rng = random.Random(int(task["index"]) * 7919 + (0 if goal_a == "P" else 1))

    def greedy_step(key: str, goal: str):
        rows = successors(key, slots)
        if not rows:
            return None
        return max(rows, key=lambda r: (utility(r[0], goal), r[0]))[0]

    def greedy_run(key: str, budget: int, goal: str) -> str:
        current = key
        for _ in range(budget):
            nxt = greedy_step(current, goal)
            if nxt is None:
                break
            current = nxt
        return current

    def greedy_path(key: str, budget: int, goal: str) -> list[str]:
        """Same policy as greedy_run, but keeps the trace. Used only for the
        recorded arms -- V_G rollouts inside verified_run stay on greedy_run so
        the hot path allocates nothing extra."""
        path, current = [key], key
        for _ in range(budget):
            nxt = greedy_step(current, goal)
            if nxt is None:
                break
            current = nxt
            path.append(current)
        return path

    def verified_run(key: str, budget: int, goal: str):
        """Commit argmax V_G under strict improvement, then re-plan.
        Returns (path, overrides); the landing is path[-1]."""
        current, overrides, path = key, 0, [key]
        for step in range(budget):
            remaining = budget - step
            rows = successors(current, slots)
            if not rows:
                break
            keys = [r[0] for r in rows]
            reference = np.array([r[1] for r in rows], dtype=float)
            # RANK IS A TUPLE (worst, mean). It must stay a list of tuples --
            # np.array(..., dtype=float) would build an (n,2) array and every
            # comparison below would then be an ambiguous array truth value.
            rank = [utility(k, goal) for k in keys]
            order = sorted(range(len(keys)), key=lambda i: (rank[i], keys[i]),
                           reverse=True)
            greedy_index = order[0]
            picked, seen = [greedy_index], {greedy_index}
            for index in order[:CANDIDATES_TOP_IMMEDIATE]:
                if index not in seen:
                    picked.append(index); seen.add(index)
            for index in np.argsort(-reference, kind="stable")[:CANDIDATES_TOP_REFERENCE]:
                if int(index) not in seen:
                    picked.append(int(index)); seen.add(int(index))
            rest = [i for i in range(len(keys)) if i not in seen]
            if rest and CANDIDATES_RANDOM > 0:
                for index in rng.sample(rest, min(CANDIDATES_RANDOM, len(rest))):
                    picked.append(index); seen.add(index)
            futures = {i: utility(greedy_run(keys[i], remaining - 1, goal), goal)
                       for i in picked}
            best = max(picked, key=lambda i: (futures[i], keys[i]))
            chosen = best if futures[best] > futures[greedy_index] else greedy_index
            overrides += int(chosen != greedy_index)
            current = keys[chosen]
            path.append(current)
        return path, overrides

    paths: dict[str, list[str]] = {}
    overrides: dict[str, int] = {}
    # continue_A: the negative control -- keep optimising the OLD goal.
    paths["continue_A"] = greedy_path(switch, post, goal_a)
    paths["greedy_retarget"] = greedy_path(switch, post, "B")
    paths["verified_retarget"], overrides["verified_retarget"] = \
        verified_run(switch, post, "B")
    # restart: discard the realised history, same remaining budget.
    paths["restart"], overrides["restart"] = verified_run(start, post, "B")
    # clairvoyant: B known from step 0, full horizon. No surprise.
    paths["clairvoyant"], overrides["clairvoyant"] = \
        verified_run(start, HORIZON, "B")
    landings = {name: path[-1] for name, path in paths.items()}

    arms = {name: {"landing": key,
                   "b_worst_margin": utility(key, "B")[0],
                   "b_mean_margin": utility(key, "B")[1],
                   "b_success": bool(success(key, "B")),
                   "p_success": bool(success(key, "P")),
                   "d_success": bool(success(key, "D")),
                   "overrides": overrides.get(name),
                   # Full realised path. Lets prefix-edit survival, undo and
                   # reuse be quantified after the fact without a rerun.
                   "trajectory": paths[name]}
            for name, key in landings.items()}

    payload = {"index": int(task["index"]), "history": goal_a,
               "source": task["source"], "start_key": start,
               "switch_state": switch,
               "b_worst_at_switch": utility(switch, "B")[0],
               "b_mean_at_switch": utility(switch, "B")[1],
               "b_success_at_switch": bool(success(switch, "B")),
               "arms": arms,
               "distinct_landings": len({a["landing"] for a in arms.values()}),
               "kernel_calls": counter["calls"],
               "seconds": round(time.perf_counter() - started, 1)}
    out = Path(RUN_ROOT) / task.get("out_dir", "retarget_intervention")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{goal_a}_{task['index']:03d}.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    artifact_volume.commit()
    print(f"[{goal_a}/{task['index']:03d}] {counter['calls']} calls "
          f"{payload['seconds']:.0f}s  distinct={payload['distinct_landings']}/5  "
          + " ".join(f"{n}={'Y' if a['b_success'] else 'n'}"
                     for n, a in arms.items()), flush=True)
    return payload


@app.function(image=image, cpu=0.25, memory=2048, timeout=10 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(phase: str, tasks: list[dict[str, Any]]) -> int:
    """Fan out ON MODAL so a client disconnect cannot stall the run."""
    fn = build_prefix if phase == "prefix" else run_arms
    done = 0
    for _ in fn.map(tasks, order_outputs=False, return_exceptions=True,
                    wrap_returned_exceptions=False):
        done += 1
    return done


@app.local_entrypoint()
def main(phase: str = "prefix", sources: int = 30, smoke: bool = False) -> None:
    cohort = json.loads(
        (ROOT / "diagnostics/retarget_calibration_cohort.json").read_text())
    rows = cohort["sources"][: 4 if smoke else sources]

    if phase == "prefix":
        tasks = [{**r, "history": h} for h in HISTORIES for r in rows]
        print(f"PHASE 1 -- {len(tasks)} prefixes ({len(rows)} sources x "
              f"{len(HISTORIES)} histories), {SWITCH_AT} edits under goal A.")
        print("Goal B is NOT constructed in this phase.")
        print(f"completed {drive.remote('prefix', tasks)}")
        return

    prefixes = json.loads(
        (ROOT / "diagnostics/retarget_prefixes_committed.json").read_text())
    index = {(p["history"], p["index"]): p for p in prefixes["prefixes"]}
    tasks = []
    for h in HISTORIES:
        for r in rows:
            p = index.get((h, r["index"]))
            if p is None:
                continue
            tasks.append({**r, "history": h, "start_key": p["start_key"],
                          "switch_state": p["switch_state"],
                          "out_dir": ("retarget_intervention_smoke" if smoke
                                      else "retarget_intervention")})
    print(f"PHASE 2 -- {len(tasks)} branch points from committed prefixes "
          f"{prefixes['prefixes_sha256'][:16]}")
    print(f"arms: {', '.join(ARMS)}; horizon {HORIZON}, switch at {SWITCH_AT}")
    print(f"completed {drive.remote('arms', tasks)}")
