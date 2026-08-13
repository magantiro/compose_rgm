"""Pathwise structural constraints. H=6, exact labeled subgraph. CPU ONLY.

THE CLAIM THIS TESTS -- and the one it does NOT
-----------------------------------------------
Not "constrained control reaches the goal". The claim is about the SUPPORT of
the controlled process:

    because every COMPOSE state is a complete molecule and every transition is
    executable, a structural requirement can be imposed on every committed
    state, not merely on the endpoint. Endpoint-only filtering can return a
    valid final molecule after passing through states the requirement forbids;
    support masking makes those states unreachable by construction.

The protected motif is the largest fused ring system of the SOURCE, taken with
exact element / aromaticity / formal-charge atom labels and exact bond orders.
Preservation is subgraph monomorphism under those labels. It is NOT fingerprint
similarity and NOT endpoint recovery.

THE MEASUREMENT, AND THE THING THAT IS NOT A MEASUREMENT
--------------------------------------------------------
Pathwise arms have zero committed violations BY CONSTRUCTION. That number is a
definition, and reporting it as a finding would be the fourth instance of this
project's recurring defect -- a statistic whose sign was fixed before any data
existed. It is asserted here as a BUG DETECTOR and nothing more.

The measurements that can come out the other way are:

  * `endpoint_valid_path_invalid` on the UNCONSTRAINED and ENDPOINT_ONLY arms
    -- trajectories endpoint-only filtering would accept, having passed through
    a forbidden state. Zero if the constraint is vacuous. This is the gate.
  * `removed_fraction` -- the share of legal successors the mask deletes,
    measured along UNCONSTRAINED states so the mask is not scored on states it
    chose itself. Zero means vacuous; one means infeasible.
  * `endpoint_only` versus `pathwise_stochastic` terminal utility -- identical
    budget, identical policy, differing in exactly one thing: the mask. This is
    the price of the guarantee, and it has no guaranteed sign.

`pathwise_verified` versus `pathwise_greedy` DOES have a guaranteed sign:
greedy's action is always in the verified candidate set and strict improvement
never commits a lower V_G, so by induction verified >= greedy on both the
lexicographic utility and, since success is a threshold of the worst margin,
on binary success too. Only the MAGNITUDE of the gap and the TOP-1
DISAGREEMENT rate are admissible, and binary headroom must be reported over
the denominator of sources where greedy actually failed.

THE ARMS
--------
stage A (cheap: no lookahead rollouts)
    unconstrained_greedy    context, and the source of the vacuity gate
    endpoint_only           N stochastic unconstrained rollouts, keep the ones
                            with a motif-valid ENDPOINT, return the best by u_B
    pathwise_greedy         greedy over the masked support
    pathwise_stochastic     the budget-matched twin of endpoint_only: same N,
                            same policy, masked support
    mask_only_sampling      sample R_theta over the masked support with NO
                            goal -- separates feasibility from objective control

stage B (expensive: remaining-budget lookahead)
    unconstrained_verified  like-for-like partner for the cost of the guarantee
    pathwise_verified       planning inside the reduced feasible support

Stage A alone resolves the vacuity gate, the support-removal rate, feasibility
and the budget-matched price of the mask. Stage B is worth paying for only if
stage A's gate passes, so the two stages are separate invocations.

BUDGET ASYMMETRY, STATED IN THE DIRECTION IT CUTS
-------------------------------------------------
`endpoint_only` gets N rollouts and a best-of-N selection. `pathwise_greedy`
gets one. That asymmetry FAVOURS the arm this workstream argues against, which
is the safe direction; `pathwise_stochastic` exists so the utility comparison
also has a strictly budget-matched form.

HELD-OUT STATUS: this app reads the held-in smoke panel only. It has no code
path to `reserve_source_keys`.
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
        ROOT / "diagnostics/pathwise_constraints_smoke_panel.json",
        str(REMOTE_ROOT / "diagnostics/pathwise_constraints_smoke_panel.json"),
        copy=True)
    .add_local_dir(ROOT / "artifacts/oracles/drd2_svm_v1",
                   str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1"), copy=True)
)
app = modal.App("compose-v4-pathwise-constraints")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT = 0.5

HORIZON = 6
GOAL = "B"                    # B = P AND D, frozen from the retargeting lane

#: Rollouts for the two best-of-N arms. Frozen before any run.
ROLLOUTS = 6
#: Width of the goal-aware stochastic shortlist. SHORTLIST=1 IS greedy, so the
#: stochastic arms degenerate to the greedy arms in the limit -- there is no
#: separate policy family to tune.
SHORTLIST = 3

#: Verbatim from retarget_intervention_app.py so the verified controller is the
#: same instrument, not a reimplementation.
CANDIDATES_TOP_IMMEDIATE = 4
CANDIDATES_TOP_REFERENCE = 2
CANDIDATES_RANDOM = 2

#: COST CEILING per source, in kernel calls. Sized ~2.5x the stage-A estimate
#: (85 calls) and ~1.6x the stage-AB estimate (220), so it never truncates a
#: healthy run and does bound the bill if enumeration is slower than the
#: retargeting cohort suggested. At the measured ~16 s/call this caps one
#: container at about 1.6 hours of kernel time.
KERNEL_CALL_BUDGET = 360

POTENCY_THRESHOLD = 0.5
QED_FLOOR = 0.6
LOGP_BOX = (1.0, 4.0)

#: Single source of truth, shared with the local test suite. Imported from the
#: DEPENDENCY-FREE names module rather than from `pathwise_arms`, because
#: `modal run` imports this file in the LAUNCHER's interpreter, which has no
#: RDKit. Nothing at module scope here may touch the chemistry stack; the arm
#: BUILDERS are imported inside `run_source`, which runs in the image.
STAGE_A = _names.STAGE_A
STAGE_B = _names.STAGE_B
ALL_ARMS = _names.ALL_ARMS


def _runtime():
    """Model + oracle + goal algebra + constraint predicate."""
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
        return [potency, *develop]

    def utility(key: str, goal: str):
        m = margins(key, goal)
        return (float(min(m)), float(sum(m) / len(m)))

    def success(key: str, goal: str) -> bool:
        return min(margins(key, goal)) >= 0.0

    enumeration: dict[str, list[tuple[str, float]]] = {}
    counter = {"calls": 0}

    def successors(key: str, slots: int):
        """FULL legal support. The mask is applied downstream on these keys, so
        masking never costs an extra kernel call."""
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

    return (props, margins, utility, success, successors, counter,
            smiles_to_molecular_graph, pad_molecular_graph)


@app.function(image=image, cpu=2.0, memory=12 * 1024, timeout=6 * 60 * 60,
              max_containers=10, retries=1,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_source(task: dict[str, Any]) -> dict[str, Any]:
    """Every requested arm for one source, sharing one enumeration cache."""
    import random

    from compose_v4.experiments.pathwise_arms import ARM_BUILDERS, ArmContext
    from compose_v4.experiments.pathwise_constraints import (
        derive_protected_motif,
        mask_successors,
        path_violation_summary,
        preserves_motif,
    )
    from compose_v4.rewrite.kernel import canonical_state_key

    started = time.perf_counter()
    (_props, _margins, utility, success, successors, counter,
     to_graph, pad) = _runtime()

    slots = int(task["slots"])
    smarts = task["motif_smarts"]
    arms_wanted = list(task["arms"])
    rng = random.Random(int(task["index"]) * 7919 + 20260813)
    start = canonical_state_key(pad(to_graph(task["source"]), slots))

    def void(reason: str, **extra) -> dict[str, Any]:
        return {"schema": "compose.pathwise.source_result",
                "status": "INVALID_INSTRUMENT", "index": int(task["index"]),
                "source": task["source"], "start_key": start,
                "error": reason, **extra}

    # INSTRUMENT PRECONDITIONS.
    #
    # (1) The motif was derived from the panel SMILES locally; the kernel works
    #     on the canonical state key. If canonicalisation moved the molecule
    #     out of its own motif, every violation count downstream is noise.
    if not preserves_motif(smarts, start):
        return void("canonical start key does not contain its own motif")

    # (2) RDKit VERSION DRIFT. The frozen SMARTS was written under the local
    #     RDKit; this image pins a different one. Aromaticity perception is the
    #     part that could move, and it would silently widen or narrow the
    #     protected pattern rather than fail. Re-derive here and require the
    #     two rules to agree on geometry, so drift is loud instead of quiet.
    rederived = derive_protected_motif(start)
    if rederived is None:
        return void("motif re-derivation found no ring system in the start key")
    if rederived.atom_count != int(task["motif_atoms"]):
        return void(
            "motif geometry differs between the frozen panel and this runtime",
            frozen_motif_atoms=int(task["motif_atoms"]),
            rederived_motif_atoms=rederived.atom_count,
            rederived_smarts=rederived.smarts)

    # The arm policies are shared with the local test suite -- this app wires
    # the frozen kernel and the frozen goal language into them, it does not
    # reimplement them. See src/compose_v4/experiments/pathwise_arms.py.
    ctx = ArmContext(
        successors=lambda key: successors(key, slots),
        utility=lambda key: utility(key, GOAL),
        motif_smarts=smarts,
        rng=rng,
        horizon=HORIZON,
        rollouts=ROLLOUTS,
        shortlist=SHORTLIST,
        candidates_top_immediate=CANDIDATES_TOP_IMMEDIATE,
        candidates_top_reference=CANDIDATES_TOP_REFERENCE,
        candidates_random=CANDIDATES_RANDOM,
    )

    mask_census: list[dict] = []

    def record_mask(key: str, step: int, where: str) -> None:
        _kept, census = mask_successors(smarts, successors(key, slots))
        mask_census.append({"step": step, "where": where, **census})

    arms: dict[str, Any] = {}
    order_run: list[str] = []
    skipped: list[str] = []
    budget = int(task.get("kernel_call_budget", KERNEL_CALL_BUDGET))
    for name in ALL_ARMS:
        if name not in arms_wanted:
            continue
        # COST CIRCUIT BREAKER. Checked BETWEEN arms, never inside one, so an
        # arm is either complete or absent -- a half-run arm would be an
        # invalid instrument dressed as a result.
        if counter["calls"] >= budget:
            skipped.append(name)
            continue
        before = counter["calls"]
        result = ARM_BUILDERS[name](ctx, start)
        landing = result["trajectory"][-1]
        audit = path_violation_summary(smarts, result["trajectory"])
        arms[name] = {
            "landing": landing,
            "trajectory": result["trajectory"],
            "edits": len(result["trajectory"]) - 1,
            "dead_end_step": result.get("dead_end_step"),
            "completed": result.get("dead_end_step") is None,
            "selection_failed": result.get("selection_failed"),
            "rollouts_offered": result.get("rollouts_offered"),
            "rollouts_admissible": result.get("rollouts_admissible"),
            "overrides": result.get("overrides"),
            "top1_disagreements": result.get("top1_disagreements"),
            "b_worst_margin": utility(landing, "B")[0],
            "b_mean_margin": utility(landing, "B")[1],
            "b_success": bool(success(landing, "B")),
            "p_success": bool(success(landing, "P")),
            "d_success": bool(success(landing, "D")),
            "audit": audit,
            # marginal, in RUN ORDER: the shared cache means an arm run later
            # pays less. The order is recorded so this is interpretable.
            "marginal_kernel_calls": counter["calls"] - before,
        }
        if "rollouts" in result:
            arms[name]["rollout_audits"] = [
                {"audit": r["audit"], "utility": r["utility"],
                 "trajectory": r["trajectory"]}
                for r in result["rollouts"]
            ]
        order_run.append(name)

    # Mask census along an UNCONSTRAINED path: the mask is scored on states it
    # did not choose, so the removal rate is not selected for feasibility. A
    # census taken along a masked arm's own path would only ever visit states
    # the mask had already approved.
    reference_arm = next(
        (name for name in ("unconstrained_greedy", "unconstrained_verified",
                           "endpoint_only") if name in arms),
        None)
    reference_path = (arms[reference_arm]["trajectory"] if reference_arm
                      else [start])
    for step, key in enumerate(reference_path):
        record_mask(key, step, reference_arm or "start_only")

    # BUG DETECTORS, not findings. Zero violations on a masked arm is a
    # definition; a NON-zero count means the mask leaked and the run is void.
    leaks = {
        name: arm["audit"]["violation_count"]
        for name, arm in arms.items()
        if name.startswith(("pathwise_", "mask_only")) and arm["audit"]["any_violation"]
    }

    payload = {
        "schema": "compose.pathwise.source_result",
        "status": "INVALID_INSTRUMENT" if leaks else "SMOKE_HELD_IN",
        "mask_leak": leaks or None,
        "index": int(task["index"]),
        "source": task["source"],
        "start_key": start,
        "motif_smarts": smarts,
        "motif_atoms": task.get("motif_atoms"),
        "motif_fraction": task.get("motif_fraction"),
        "motif_rederived_smarts": rederived.smarts,
        "motif_rederived_atoms": rederived.atom_count,
        "horizon": HORIZON,
        "goal": GOAL,
        "rollouts": ROLLOUTS,
        "shortlist": SHORTLIST,
        "arm_run_order": order_run,
        "arms_skipped_on_budget": skipped or None,
        "kernel_call_budget": budget,
        "b_worst_at_source": utility(start, "B")[0],
        "b_success_at_source": bool(success(start, "B")),
        "arms": arms,
        "mask_census": mask_census,
        "distinct_landings": len({a["landing"] for a in arms.values()}),
        "kernel_calls": counter["calls"],
        "seconds": round(time.perf_counter() - started, 1),
    }
    out = Path(RUN_ROOT) / task.get("out_dir", "pathwise_constraints_smoke")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{task['index']:03d}.json").write_text(json.dumps(payload, indent=2) + "\n")
    artifact_volume.commit()

    gate = sum(1 for a in arms.values() if a["audit"]["endpoint_valid_path_invalid"])
    removed = [c["removed_fraction"] for c in mask_census]
    removed_mean = sum(removed) / len(removed) if removed else float("nan")
    print(f"[{task['index']:03d}] {counter['calls']} calls "
          f"{payload['seconds']:.0f}s  landings={payload['distinct_landings']}"
          f"/{len(arms)}  endpoint_valid_path_invalid_arms={gate}  "
          f"mask_removed_mean={removed_mean:.3f}", flush=True)
    return payload


@app.function(image=image, cpu=0.25, memory=2048, timeout=10 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]], force: bool = False) -> dict[str, Any]:
    """Fan out ON MODAL so a client disconnect cannot stall the run.

    RESUME. Already-committed shards are filtered out HERE, in the driver,
    where the volume is already mounted. Doing it inside `run_source` would
    make a resume pay a container start plus a full checkpoint load for every
    task that was already finished -- the expensive half of the work -- just to
    discover it had nothing to do.

    `drive` itself must still be launched under `modal run --detach`: a
    server-side fan-out stops `.map()` stalling when the client stops
    iterating, but only `--detach` stops the whole app being torn down when the
    client disconnects.
    """
    artifact_volume.reload()
    pending, skipped = [], []
    for task in tasks:
        shard = (Path(RUN_ROOT) / task.get("out_dir", "pathwise_constraints_smoke")
                 / f"{task['index']:03d}.json")
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
def main(stage: str = "A", sources: int = 6, force: bool = False) -> None:
    """stage A = the five cheap arms and the vacuity gate.
    stage B = add the two lookahead arms. Only worth paying for if A passes.
    stage AB = everything in one pass.

    LAUNCH WITH `modal run --detach`. Without it the app stays `ephemeral` and
    is torn down when the local process disconnects, taking the run with it.
    Verify `modal app list` shows `ephemeral (detached)` before walking away.
    """
    panel_path = ROOT / "diagnostics/pathwise_constraints_smoke_panel.json"
    panel = json.loads(panel_path.read_text())
    digest = hashlib.sha256(
        json.dumps(panel["sources"], sort_keys=True).encode()).hexdigest()
    if digest != panel["panel_sha256"]:
        raise SystemExit("panel hash mismatch -- the frozen panel was edited")
    if panel.get("held_out_opened") is not False:
        raise SystemExit("panel is not marked held-in")

    arms = {"A": STAGE_A, "B": STAGE_B, "AB": ALL_ARMS}[stage.upper()]
    rows = panel["sources"][:sources]
    tasks = [{**row, "arms": list(arms),
              "out_dir": f"pathwise_constraints_smoke_stage{stage.upper()}"}
             for row in rows]
    print(f"PATHWISE SMOKE stage {stage.upper()} -- {len(tasks)} held-in sources, "
          f"panel {panel['panel_sha256'][:16]}")
    print(f"arms: {', '.join(arms)}")
    print(f"horizon {HORIZON}, goal {GOAL}, rollouts {ROLLOUTS}, "
          f"shortlist {SHORTLIST}")
    print("HELD-OUT: not opened. Pool is training_source_keys only.")
    print(f"per-source kernel-call budget {KERNEL_CALL_BUDGET}; CPU only.")
    print(json.dumps(drive.remote(tasks, force), indent=2))
