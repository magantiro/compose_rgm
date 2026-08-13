"""STAGE A2 -- cLogP corridor prevalence / viability. CPU ONLY. NOT LAUNCHED.

WHAT THIS IS, STATED HONESTLY
-----------------------------
Family B was selected for follow-up AFTER the three-family feasibility census
because it alone exhibited the intended reversible-excursion mechanism. Stage
A2 is developmental follow-up, not independent confirmation of the phenomenon.

The stage-A verdict stands as FAIL and is not retroactively revised. A2 asks a
different question than the one the 20-event bar answered:

    on NEW held-in sources, are endpoint-valid / path-invalid excursions
    frequent enough AND spread across enough distinct molecules to support a
    causal, source-level pathwise-control experiment -- and does the corridor
    mask leave a controller anything to act on?

FROZEN, UNCHANGED FROM STAGE A
------------------------------
corridor [2.3689, 4.4522] (the held-in cLogP interquartile range, read at
runtime from the frozen normalizers, never transcribed) - H=6 - the same
R_theta - the same violation and endpoint-recovery definitions - the same
rollout law (sample R_theta within the top-3 by u_B, goal B = P AND D).

No threshold is adjusted. No other family is measured. If A2 fails, no fourth
predicate is searched.

THE SOURCE IS THE INDEPENDENT UNIT
----------------------------------
12 sources x 6 rollouts is 72 trajectories but **12 observations**. Rollouts
from one source are repeated measures on the same molecule. Every headline
number this app feeds must be reported per source with cluster/bootstrap
uncertainty; the analysis script enforces that.

WHY THIS RUN IS DEFERRED
------------------------
Every A2 trajectory would be generated under the frozen R_theta that lane 1 is
currently deciding whether to discard. Spending container-hours now risks
measuring a model that is about to be replaced. This file is committed ready to
run and is NOT launched.

LAUNCH (only when authorised):
    PYTHONPATH=src:. MODAL_PROFILE=rahul-94866 \
      modal run --detach modal_apps/pathwise_corridor_prevalence_app.py --sources 12
Verify `modal app list` shows `ephemeral (detached)` before walking away.
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
        ROOT / "diagnostics/pathwise_a2_panel.json",
        str(REMOTE_ROOT / "diagnostics/pathwise_a2_panel.json"), copy=True)
    .add_local_dir(ROOT / "artifacts/oracles/drd2_svm_v1",
                   str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1"), copy=True)
)
app = modal.App("compose-v4-pathwise-corridor-prevalence")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
OUT_DIR = "pathwise_a2_corridor_prevalence"
TIME_POINT = 0.5

# ---- frozen, identical to stage A -----------------------------------------
HORIZON = 6
GOAL = "B"
ROLLOUTS = 6
SHORTLIST = 3
POTENCY_THRESHOLD = 0.5
QED_FLOOR = 0.6
LOGP_BOX = (1.0, 4.0)
# ---------------------------------------------------------------------------

#: Stage A measured 51 kernel calls per source for FIVE arms. A2 runs 6
#: stochastic rollouts and no lookahead, so ~35 is expected. 200 is a generous
#: ceiling that bounds the bill without truncating a healthy source.
KERNEL_CALL_BUDGET = 200


def _runtime():
    """Model + oracle + goal algebra. Identical construction to stage A."""
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED, Crippen

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

    def utility(key: str, goal: str):
        m, qed, logp = props(key)
        if not all(np.isfinite(v) for v in (m, qed, logp)):
            margins = [-10.0]
        else:
            potency = (m - potency_cut) / s_drd2
            develop = [(qed - QED_FLOOR) / s_qed,
                       min(logp - LOGP_BOX[0], LOGP_BOX[1] - logp) / s_logp]
            margins = ([potency] if goal == "P" else develop if goal == "D"
                       else [potency, *develop])
        return (float(min(margins)), float(sum(margins) / len(margins)))

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

    return (utility, successors, counter, smiles_to_molecular_graph,
            pad_molecular_graph)


@app.function(image=image, cpu=2.0, memory=12 * 1024, timeout=6 * 60 * 60,
              max_containers=12, retries=1,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_source(task: dict[str, Any]) -> dict[str, Any]:
    """Six unconstrained rollouts, plus the corridor mask census on every state."""
    import random

    from compose_v4.experiments.pathwise_arms import ArmContext, stochastic_path
    from compose_v4.experiments.pathwise_reversible_families import (
        audit_trajectory,
        clogp_corridor,
        clogp_of,
        corridor_excursions,
        state_is_feasible,
    )
    from compose_v4.rewrite.kernel import canonical_state_key

    started = time.perf_counter()
    utility, successors, counter, to_graph, pad = _runtime()
    slots = int(task["slots"])
    rng = random.Random(int(task["index"]) * 7919 + 20260814)
    start = canonical_state_key(pad(to_graph(task["source"]), slots))
    low, high = clogp_corridor()

    # PRECONDITION. Eligibility guaranteed the panel SMILES sits inside the
    # corridor; the kernel works on the canonical key. If canonicalisation moved
    # it out, this source can never "leave and return" and every count below
    # would be measuring something else.
    if not state_is_feasible("B_physchem_corridor", start):
        return {"schema": "compose.pathwise.a2_source", "status": "INVALID_INSTRUMENT",
                "index": int(task["index"]), "source": task["source"],
                "start_key": start, "start_clogp": clogp_of(start),
                "error": "canonical start key is outside the frozen corridor"}

    ctx = ArmContext(
        successors=lambda key: successors(key, slots),
        utility=lambda key: utility(key, GOAL),
        motif_smarts="",                       # unused: corridor family
        feasible=lambda key: state_is_feasible("B_physchem_corridor", key),
        rng=rng, horizon=HORIZON, rollouts=ROLLOUTS, shortlist=SHORTLIST,
    )

    rollouts: list[dict] = []
    for index in range(ROLLOUTS):
        if counter["calls"] >= KERNEL_CALL_BUDGET:
            break
        path = stochastic_path(ctx, start, mask=False, goal_aware=True)
        audit = audit_trajectory("B_physchem_corridor", path["trajectory"])
        rollouts.append({
            "rollout": index,
            "trajectory": path["trajectory"],
            "dead_end_step": path["dead_end_step"],
            "audit": audit,
            "excursions": corridor_excursions("B_physchem_corridor",
                                              path["trajectory"]),
            "clogp": [round(clogp_of(k), 4) for k in path["trajectory"]],
            "terminal_utility": utility(path["trajectory"][-1], GOAL),
        })

    # V4 -- the criterion stage A could never evaluate. Measured on every state
    # the UNCONSTRAINED rollouts actually visited, so the mask is scored on
    # states it did not choose for itself.
    mask_census: list[dict] = []
    seen: set[str] = set()
    for roll in rollouts:
        for step, key in enumerate(roll["trajectory"]):
            if key in seen:
                continue
            seen.add(key)
            rows = successors(key, slots)
            kept = [r for r in rows if state_is_feasible("B_physchem_corridor", r[0])]
            total_mass = sum(r[1] for r in rows)
            kept_mass = sum(r[1] for r in kept)
            mask_census.append({
                "step": step,
                "candidates": len(rows),
                "kept": len(kept),
                "retention": (len(kept) / len(rows)) if rows else None,
                "retained_reference_mass": (
                    kept_mass / total_mass) if total_mass > 0 else None,
                "empty_after_mask": bool(rows) and not kept,
                "state_feasible": state_is_feasible("B_physchem_corridor", key),
            })

    events = sum(1 for r in rollouts if r["audit"]["endpoint_valid_path_invalid"])
    violators = sum(1 for r in rollouts if r["audit"]["any_violation"])
    payload = {
        "schema": "compose.pathwise.a2_source",
        "status": "SMOKE_HELD_IN",
        "index": int(task["index"]),
        "source": task["source"],
        "start_key": start,
        "start_clogp": round(clogp_of(start), 4),
        "corridor": [low, high],
        "horizon": HORIZON,
        "goal": GOAL,
        "rollouts_requested": ROLLOUTS,
        "rollouts_completed": len(rollouts),
        "violating_rollouts": violators,
        "endpoint_valid_path_invalid_events": events,
        "source_has_event": bool(events),
        "rollouts": rollouts,
        "mask_census": mask_census,
        "kernel_calls": counter["calls"],
        "kernel_call_budget": KERNEL_CALL_BUDGET,
        "seconds": round(time.perf_counter() - started, 1),
    }
    out = Path(RUN_ROOT) / task.get("out_dir", OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{task['index']:03d}.json").write_text(json.dumps(payload, indent=2) + "\n")
    artifact_volume.commit()
    retention = sorted(c["retention"] for c in mask_census
                       if c["retention"] is not None)
    median_retention = retention[len(retention) // 2] if retention else float("nan")
    print(f"[{task['index']:03d}] {counter['calls']} calls "
          f"{payload['seconds']:.0f}s  violators={violators}/{len(rollouts)}  "
          f"events={events}  median_retention={median_retention:.3f}", flush=True)
    return payload


@app.function(image=image, cpu=0.25, memory=2048, timeout=10 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]], force: bool = False) -> dict[str, Any]:
    """Fan out ON MODAL, skipping already-committed shards.

    Resume filtering lives here, where the volume is already mounted: doing it
    inside `run_source` would make a relaunch pay a container start and a full
    checkpoint load per finished task just to learn it had nothing to do.
    """
    artifact_volume.reload()
    pending, skipped = [], []
    for task in tasks:
        shard = Path(RUN_ROOT) / task.get("out_dir", OUT_DIR) / f"{task['index']:03d}.json"
        if not force and shard.exists():
            try:
                status = json.loads(shard.read_text()).get("status")
            except Exception:  # noqa: BLE001
                status = None
            if status:
                skipped.append({"index": task["index"], "status": status})
                continue
        pending.append(task)
    if skipped:
        print(f"RESUME: skipping {len(skipped)} committed shards "
              f"{[s['index'] for s in skipped]}", flush=True)
    print(f"dispatching {len(pending)} of {len(tasks)} tasks", flush=True)

    done, failed = 0, []
    for task, result in zip(
            pending,
            run_source.map(pending, order_outputs=True, return_exceptions=True,
                           wrap_returned_exceptions=False),
            strict=True):
        if isinstance(result, BaseException):
            failed.append({"index": task["index"], "error": repr(result)})
            print(f"[{task['index']:03d}] FAILED {result!r}", flush=True)
            continue
        done += 1
    artifact_volume.commit()
    return {"completed": done, "skipped": skipped, "failed": failed,
            "dispatched": len(pending)}


@app.local_entrypoint()
def main(sources: int = 12, force: bool = False) -> None:
    panel_path = ROOT / "diagnostics/pathwise_a2_panel.json"
    panel = json.loads(panel_path.read_text())
    digest = hashlib.sha256(
        json.dumps(panel["sources"], sort_keys=True).encode()).hexdigest()
    if digest != panel["panel_sha256"]:
        raise SystemExit("panel hash mismatch -- the frozen A2 panel was edited")
    if panel.get("held_out_opened") is not False:
        raise SystemExit("panel is not marked held-in")

    rows = panel["sources"][:sources]
    tasks = [{**row, "out_dir": OUT_DIR} for row in rows]
    print(f"STAGE A2 -- corridor prevalence, {len(tasks)} NEW held-in sources, "
          f"panel {panel['panel_sha256'][:16]}")
    print(f"corridor {panel['constraint']['clogp_corridor']} (FROZEN, unchanged)")
    print(f"horizon {HORIZON}, goal {GOAL}, rollouts {ROLLOUTS}, "
          f"shortlist {SHORTLIST}  -- rollout law identical to stage A")
    print("THE SOURCE IS THE INDEPENDENT UNIT: "
          f"{len(tasks)} x {ROLLOUTS} = {len(tasks) * ROLLOUTS} trajectories, "
          f"{len(tasks)} observations.")
    print("HELD-OUT: not opened. Pool is training_source_keys only.")
    print(json.dumps(drive.remote(tasks, force), indent=2))
