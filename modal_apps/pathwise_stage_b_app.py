"""STAGE B -- corridor-constrained potency control. CPU ONLY. NOT LAUNCHED.

THE QUESTION
------------
Not "can the mask achieve zero violations". That is guaranteed by construction
and is barred from the results table. The question is:

    when terminally acceptable trajectories can pass through forbidden
    intermediate states, what is the COST and BENEFIT of enforcing the
    constraint throughout molecular evolution?

Task: increase DRD2 potency subject to 2.3689 <= cLogP(x_t) <= 4.4522 for all
t, H=6. The objective is potency ALONE, frozen in the retargeting lane, chosen
because it is independently motivated and already frozen -- not because it
maximises the pathwise effect.

THE 2x2
-------
                     navigate myopically   navigate with lookahead
  enforce at t=H     endpoint_greedy       endpoint_verified
  enforce at every t pathwise_greedy       pathwise_verified

All four run the SAME code path (`build_stage_b_arm`), differing only in the
declared `(mask, endpoint_only, controller)` triple. A terminal-cost difference
between `endpoint_greedy` and `pathwise_greedy` therefore cannot be an artefact
of two separately written policies. `unconstrained_potency` is recorded as
DESCRIPTIVE ONLY and never enters a causal contrast.

WHAT IS AND IS NOT A MEASUREMENT
--------------------------------
PRIMARY ESTIMAND, among endpoint-only trajectories that actually landed in C:

    P( exists t < H : x_t not in C  |  x_H in C )

It has genuine room to be zero -- the ring-motif family produced exactly that.

TERMINAL COST, under controller parity:
    Delta^G = U_P(pathwise greedy)   - U_P(endpoint greedy)
    Delta^V = U_P(pathwise verified) - U_P(endpoint verified)
The expected sign is NOT positive. A pathwise requirement removes options, so a
terminal cost is a legitimate finding: "eliminates large intermediate
excursions at modest terminal cost" is a good result, not a failed one.

GUARANTEED SIGN, and therefore NOT a primary claim: `pathwise_verified` minus
`pathwise_greedy`. Verified contains greedy's action and overrides only on
strict improvement, so the direction is fixed before any molecule exists. Only
effect SIZE and constrained-performance recovered are reported, never a sign
test, and if greedy is already at a ceiling the subclaim closes exactly as the
retargeting lane closed its own.

SUPPORT-TIGHT SOURCES -- predeclared, because the retention spread is known
-----------------------------------------------------------------------------
A2 found per-source median retention spanning 0.048 to 0.917. Predeclared here:
a source is SUPPORT_TIGHT when its median retained legal-successor fraction
falls below 0.10 -- the viability threshold already in use, not a new number.
Retention is measured along the DESCRIPTIVE unconstrained arm's states, so the
classification cannot depend on any constrained arm's outcome. All sources stay
in the primary intention-to-treat analysis; a predeclared sensitivity analysis
excluding them is reported alongside. The threshold is never redefined after
seeing which arm suffers.

Mask-empty frequency is reported at SOURCE level, not pooled.

HELD-OUT: not opened, and no held-out confirmation is designed yet.

LAUNCH (only when authorised):
    PYTHONPATH=src:. MODAL_PROFILE=rahul-94866 \
      modal run --detach modal_apps/pathwise_stage_b_app.py --sources 24
Verify `modal app list` shows `ephemeral (detached)`. Do NOT wrap the client in
`timeout`: killing a wrapped client cancels the detached job.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import modal

from compose_v4.experiments import pathwise_arm_names as _names
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
        ROOT / "diagnostics/pathwise_stage_b_panel.json",
        str(REMOTE_ROOT / "diagnostics/pathwise_stage_b_panel.json"), copy=True)
    .add_local_dir(ROOT / "artifacts/oracles/drd2_svm_v1",
                   str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1"), copy=True)
)
app = modal.App("compose-v4-pathwise-stage-b")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
OUT_DIR = "pathwise_stage_b"
TIME_POINT = 0.5

# ---- frozen -----------------------------------------------------------------
HORIZON = 6
GOAL = "P"                       # DRD2 potency ALONE
POTENCY_THRESHOLD = 0.5
CANDIDATES_TOP_IMMEDIATE = 4     # verbatim from retarget_intervention_app.py
CANDIDATES_TOP_REFERENCE = 2
CANDIDATES_RANDOM = 2
#: Predeclared. Same number as the V4a viability threshold already in use.
SUPPORT_TIGHT_THRESHOLD = 0.10
# -----------------------------------------------------------------------------

ARMS = _names.STAGE_B_ALL
CORRIDOR_ARMS = _names.STAGE_B_CORRIDOR_ARMS
DESCRIPTIVE_ARM = _names.STAGE_B_DESCRIPTIVE_ARM

#: Two verified arms dominate the cost. Stage A measured 51 calls/source for
#: five non-lookahead arms; two lookahead arms plus three greedy arms is
#: estimated at ~120-150. 400 bounds the bill without truncating a healthy
#: source. Checked BETWEEN arms only, so an arm is complete or absent.
KERNEL_CALL_BUDGET = 400


def _runtime():
    """Model + potency oracle. Identical construction to every earlier stage."""
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger

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
    s_drd2 = norms["drd2_logodds"]["iqr"]
    oracle = load_default_oracle(
        str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json"))
    potency_cut = float(np.log(POTENCY_THRESHOLD / (1 - POTENCY_THRESHOLD)))

    cache: dict[str, float] = {}

    def potency_margin(key: str) -> float:
        if key not in cache:
            mol = Chem.MolFromSmiles(key)
            if mol is None:
                cache[key] = -10.0
            else:
                cache[key] = (float(oracle.margin_many([key])[0]) - potency_cut) / s_drd2
        return cache[key]

    def utility(key: str):
        """U_P. A tuple so it ranks identically to every other stage."""
        margin = potency_margin(key)
        return (float(margin), float(margin))

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

    return (utility, potency_margin, successors, counter,
            smiles_to_molecular_graph, pad_molecular_graph)


@app.function(image=image, cpu=2.0, memory=12 * 1024, timeout=8 * 60 * 60,
              max_containers=24, retries=1,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_source(task: dict[str, Any]) -> dict[str, Any]:
    """The 2x2 plus the descriptive arm, on one source, one shared cache."""
    import random

    from compose_v4.experiments.pathwise_arms import ArmContext, build_stage_b_arm
    from compose_v4.experiments.pathwise_reversible_families import (
        clogp_corridor,
        clogp_of,
        corridor_excursions,
        state_is_feasible,
    )
    from compose_v4.rewrite.kernel import canonical_state_key

    started = time.perf_counter()
    (utility, potency_margin, successors, counter,
     to_graph, pad) = _runtime()
    slots = int(task["slots"])
    rng = random.Random(int(task["index"]) * 7919 + 20260815)
    start = canonical_state_key(pad(to_graph(task["source"]), slots))
    low, high = clogp_corridor()

    def feasible(key: str) -> bool:
        return state_is_feasible("B_physchem_corridor", key)

    if not feasible(start):
        return {"schema": "compose.pathwise.stage_b_source",
                "status": "INVALID_INSTRUMENT", "index": int(task["index"]),
                "source": task["source"], "start_key": start,
                "start_clogp": clogp_of(start),
                "error": "canonical start key is outside the frozen corridor"}

    ctx = ArmContext(
        successors=lambda key: successors(key, slots),
        utility=utility,
        motif_smarts="",
        feasible=feasible,
        rng=rng, horizon=HORIZON,
        candidates_top_immediate=CANDIDATES_TOP_IMMEDIATE,
        candidates_top_reference=CANDIDATES_TOP_REFERENCE,
        candidates_random=CANDIDATES_RANDOM,
    )

    arms: dict[str, Any] = {}
    order_run: list[str] = []
    skipped: list[str] = []
    budget = int(task.get("kernel_call_budget", KERNEL_CALL_BUDGET))
    # Descriptive arm FIRST: its states define the retention census, and the
    # census must not depend on any constrained arm having run.
    for name in (DESCRIPTIVE_ARM, *CORRIDOR_ARMS):
        if counter["calls"] >= budget:
            skipped.append(name)
            continue
        before = counter["calls"]
        result = build_stage_b_arm(name, ctx, start)
        trajectory = result["trajectory"]
        landing = trajectory[-1]
        flags = [feasible(key) for key in trajectory]
        violated = [i for i, ok in enumerate(flags[1:], start=1) if not ok]

        # How many actions were actually available at each committed state,
        # before and after the corridor mask. This is what tells a reader
        # whether a terminal failure was the controller giving up or the
        # support running out.
        feasible_actions = []
        for step, key in enumerate(trajectory):
            rows = successors(key, slots)
            kept = sum(1 for r in rows if feasible(r[0]))
            feasible_actions.append({"step": step, "candidates": len(rows),
                                     "feasible": kept})
        dead = result.get("dead_end_step")
        attribution = None
        if dead is not None and dead < len(feasible_actions):
            at_failure = feasible_actions[dead]
            attribution = (
                "no_legal_successor" if at_failure["candidates"] == 0
                else "empty_after_mask" if at_failure["feasible"] == 0
                else "controller_stopped_with_support_available")
        arms[name] = {
            "landing": landing,
            "trajectory": trajectory,
            "clogp": [round(clogp_of(k), 4) for k in trajectory],
            "edits": len(trajectory) - 1,
            "dead_end_step": dead,
            "completed": dead is None,
            "endpoint_terminal_infeasible": result.get(
                "endpoint_terminal_infeasible", False),
            "feasible_actions_by_step": feasible_actions,
            "terminal_failure_attribution": attribution,
            "overrides": result.get("overrides"),
            "top1_disagreements": result.get("top1_disagreements"),
            "U_P": utility(landing)[0],
            "U_P_at_source": utility(start)[0],
            "potency_gain": utility(landing)[0] - utility(start)[0],
            "endpoint_in_C": bool(flags[-1]),
            "any_intermediate_violation": bool(violated),
            "intermediate_violation_count": len(violated),
            "first_violation_index": violated[0] if violated else None,
            "excursions": corridor_excursions("B_physchem_corridor", trajectory),
            "marginal_kernel_calls": counter["calls"] - before,
        }
        order_run.append(name)

    # SUPPORT-TIGHT classification. Retention measured along the DESCRIPTIVE
    # arm's states only -- never along a constrained arm's own path, which
    # would only ever visit states the mask had already approved.
    census: list[dict] = []
    reference = arms.get(DESCRIPTIVE_ARM, {}).get("trajectory", [start])
    for step, key in enumerate(reference):
        rows = successors(key, slots)
        kept = [r for r in rows if feasible(r[0])]
        census.append({
            "step": step,
            "candidates": len(rows),
            "kept": len(kept),
            "retention": (len(kept) / len(rows)) if rows else None,
            "empty_after_mask": bool(rows) and not kept,
        })
    retentions = sorted(c["retention"] for c in census if c["retention"] is not None)
    median_retention = retentions[len(retentions) // 2] if retentions else None
    support_tight = (median_retention is not None
                     and median_retention < SUPPORT_TIGHT_THRESHOLD)

    # BUG DETECTOR, not a finding. A pathwise arm committing a violating state
    # means the mask leaked and the shard is void.
    leaks = {name: arms[name]["intermediate_violation_count"]
             for name in ("pathwise_greedy", "pathwise_verified")
             if name in arms and arms[name]["any_intermediate_violation"]}

    payload = {
        "schema": "compose.pathwise.stage_b_source",
        "status": "INVALID_INSTRUMENT" if leaks else "SMOKE_HELD_IN",
        "mask_leak": leaks or None,
        "index": int(task["index"]),
        "source": task["source"],
        "start_key": start,
        "start_clogp": round(clogp_of(start), 4),
        "corridor": [low, high],
        "horizon": HORIZON,
        "goal": GOAL,
        "arm_run_order": order_run,
        "arms_skipped_on_budget": skipped or None,
        "kernel_call_budget": budget,
        "support_census": census,
        "median_retention": median_retention,
        "support_tight": support_tight,
        "support_tight_threshold": SUPPORT_TIGHT_THRESHOLD,
        "mask_empty_states": sum(1 for c in census if c["empty_after_mask"]),
        "mask_empty_fraction_this_source": (
            sum(1 for c in census if c["empty_after_mask"]) / len(census)
            if census else None),
        "arms": arms,
        "distinct_landings": len({a["landing"] for a in arms.values()}),
        "kernel_calls": counter["calls"],
        "seconds": round(time.perf_counter() - started, 1),
    }
    out = Path(RUN_ROOT) / task.get("out_dir", OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{task['index']:03d}.json").write_text(json.dumps(payload, indent=2) + "\n")
    artifact_volume.commit()

    eg = arms.get("endpoint_greedy", {})
    print(f"[{task['index']:03d}] {counter['calls']} calls "
          f"{payload['seconds']:.0f}s  tight={support_tight} "
          f"ret={median_retention if median_retention is None else round(median_retention, 3)}  "
          f"endpoint_greedy_path_violation={eg.get('any_intermediate_violation')}  "
          f"landings={payload['distinct_landings']}/{len(arms)}", flush=True)
    return payload


@app.function(image=image, cpu=0.25, memory=2048, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]], force: bool = False) -> dict[str, Any]:
    """Fan out ON MODAL, skipping already-committed shards.

    Resume filtering lives here, where the volume is mounted, so a relaunch
    after an outage costs nothing. Stage B is the most expensive run in this
    lane; the panel is source-sharded, so it can also be authorised in halves
    with no wasted work.
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
def main(sources: int = 24, start: int = 0, force: bool = False) -> None:
    """`--start`/`--sources` allow the panel to be run in halves; the resume
    path means a second call costs nothing for shards already committed."""
    panel_path = ROOT / "diagnostics/pathwise_stage_b_panel.json"
    panel = json.loads(panel_path.read_text())
    digest = hashlib.sha256(
        json.dumps(panel["sources"], sort_keys=True).encode()).hexdigest()
    if digest != panel["panel_sha256"]:
        raise SystemExit("panel hash mismatch -- the frozen stage-B panel was edited")
    if panel.get("held_out_opened") is not False:
        raise SystemExit("panel is not marked held-in")

    rows = panel["sources"][start:start + sources]
    tasks = [{**row, "out_dir": OUT_DIR} for row in rows]
    print(f"STAGE B -- corridor-constrained potency, {len(tasks)} held-in sources "
          f"(index {start}..{start + len(tasks) - 1}), panel "
          f"{panel['panel_sha256'][:16]}")
    print(f"objective {GOAL} (DRD2 potency alone), corridor "
          f"{panel['constraint']['clogp_corridor']}, horizon {HORIZON}")
    print(f"arms: {', '.join(CORRIDOR_ARMS)}  (+ {DESCRIPTIVE_ARM}, DESCRIPTIVE)")
    print("THE SOURCE IS THE INDEPENDENT UNIT.")
    print("HELD-OUT: not opened. Pool is training_source_keys only.")
    print(json.dumps(drive.remote(tasks, force), indent=2))
