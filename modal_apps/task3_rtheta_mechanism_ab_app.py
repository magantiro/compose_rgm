"""DEVELOPMENT-ONLY MECHANISM TEST over executable R_theta expansion.

    Given the SAME COMPOSE-generated molecular possibilities, does adaptive
    purpose specification spend evaluations better than fixed purpose?

Both arms share one proposal operator:

    x --[frozen R_theta + legal kernel]--> {y_1 ... y_~613}

    Arm A  chooses among {y_i} by a fixed scalar score
    Arm B  chooses among {y_i} by the current archive's gaps

Everything else is pinned: the same frozen seeded archive (sha256 recorded), the
same surrogate class and its data, the same additional metered budget, the same
seeds, the same expansion, the same candidates per iteration.

⚠️ NOT TASK 3 PERFORMANCE. The initialization is seeded and deliberately
contains JNK3 actives whose labels earlier runs paid for. It departs from the
benchmark's random-120 and no number here may be reported as Task 3
performance.

THE BUDGET RULE, WHICH IS THE POINT OF THE SEAM
-----------------------------------------------
The whole legal fiber -- all ~613 successors -- is constructed by R_theta and
scored by the LEARNED surrogate. Not one of them is sent through the benchmark
oracle. Only the handful actually selected per iteration is charged. Expansion
and internal scoring are search; the meter counts objective evaluations, and
`navigation_lockout()` makes evaluating anything else raise rather than
succeed.

WHAT THIS TESTS AND WHAT IT DOES NOT
-------------------------------------
It tests basic `R_theta expansion + adaptive selection` against the same
expansion with fixed selection. It deliberately has NO multi-step navigation, NO
state reuse and NO branching: one mechanism at a time, and those are only worth
adding once this one works.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import ARTIFACT_ROOT, REMOTE_ROOT, ROOT, artifact_volume
from modal_apps.run_process_v2_p50_app import image as _base_image

#: The frozen oracle bundle and the seeded archives travel in the image, so the
#: container depends on the volume only for R_theta itself.
image = (
    _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
    .add_local_dir(ROOT / "artifacts" / "oracles" / "molleo_task3_v1",
                   str(REMOTE_ROOT / "artifacts" / "oracles" / "molleo_task3_v1"),
                   copy=True)
    .add_local_dir(ROOT / "artifacts" / "oracles" / "drd2_svm_v1",
                   str(REMOTE_ROOT / "artifacts" / "oracles" / "drd2_svm_v1"),
                   copy=True)
    .add_local_file(ROOT / "artifacts" / "benchmarks"
                    / "task3_mechanism_archives_v1" / "archives.json",
                    str(REMOTE_ROOT / "archives.json"), copy=True)
)
app = modal.App("compose-v4-task3-rtheta-mechanism-ab")

ACTIVE8 = ("/artifacts/editing_v2/process_v2_active8/"
           "8ecc0e5e825a15200560c58960d4f662c9ca23be785c825c9c24aa73308144bb")
GATE_ZERO = ("/artifacts/editing_v2/process_v2_gate_zero_v6/"
             "bd4ac715f699c39c9012c3428475d13057bd7d3d18b2fb10ff716607633c4f2d"
             "/DECISION.json")
MATERIALIZED = "/artifacts/editing_v2/r_theta_run/materialized_scorer"
CHECKPOINT = "/artifacts/editing_v2/r_theta_run/runs/run_v2_01/R_THETA_CHECKPOINT.pt"
OUTPUT_ROOT = "/artifacts/editing_v2/task3_mechanism_rtheta"

#: Consecutive unproductive iterations before an arm gives up. The local policy
#: loop has had this since a test caught the same infinite loop there; the Modal
#: app did not, and it cost one arm of the first R_theta A/B.
STALL_LIMIT = 8
#: Padding slots above the source's atom count, matching the timing probe.
PAD_SLOTS = 8
#: KNOWN INEFFICIENCY, recorded rather than fixed mid-flight. R_theta is
#: deterministic, so expanding the same start twice returns the same fiber at
#: the same 12.5 s cost. The fixed arm re-selects its best-summed molecule until
#: a charged candidate displaces it, so it can re-expand one state several times.
#: A fiber cache keyed on the start molecule removes the waste entirely and is
#: the first thing to add before this is run at any larger budget. It does not
#: bias the comparison -- both arms pay the same way for the same behaviour --
#: and the per-run `expansions` count against distinct starts measures how much
#: was lost.
#: FIBER CACHING: an earned engineering candidate, NOT implemented, and the
#: first A/B says it is not yet earned. R_theta is deterministic, so re-expanding
#: a start would return the same ~613 successors at the same 12.5 s cost -- but
#: in this run `expansions` equalled `step` in every container, meaning no start
#: was ever revisited and a cache would have had zero hits. It becomes worth
#: building only when a development loop revisits states often enough to matter.
#: When it is built it must be qualified by EXACT FIBER EQUALITY against a fresh
#: expansion -- same canonical successors AND same R_theta probabilities. Note
#: that caching SMILES alone, as I first wrote it, cannot pass that check, since
#: it discards the probabilities.
#: How many starts to try before giving up on an iteration. A realized archive
#: molecule can fail to convert to a graph; that is not a reason to stop.
START_ATTEMPTS = 5


def fiber_composition(predicted, target, archive) -> dict[str, Any]:
    """What is IN the shared fiber, before anything is selected.

    This is what makes a null result diagnosable rather than ambiguous. If the
    region the adaptive arm is aiming at is simply not present among the ~613
    successors, that is a fact about expansion, and the controller must not be
    blamed for failing to select it.

    ⚠️ THESE ARE PREDICTED VALUES, so they answer "what could the controller
    SEE in this fiber", which is fiber composition CONVOLVED with surrogate
    error. The unconfounded question -- what is actually in the fiber -- needs
    true objectives on fiber members, which costs budget and is measured
    separately by `audit_fiber_composition`. Do not read this as ground truth
    about what R_theta exposes.
    """

    import numpy as np

    if not len(predicted):
        return {"fiber": 0}
    jnk3 = predicted[:, 1]
    gsk3b_coordinate = predicted[:, 3]
    aspiration = target.as_array()
    # "Represented" = at least one successor is predicted to meet the aspiration
    # on every axis. If this is 0 for the adaptive arm, the target was not on
    # offer and the selection had nothing to find.
    meets = np.all(predicted >= aspiration[None, :] - 1e-9, axis=1)
    return {
        "fiber": int(len(predicted)),
        "predicted_high_jnk3": int((jnk3 >= 0.4).sum()),
        "predicted_low_gsk3b": int((gsk3b_coordinate >= 0.9).sum()),
        "predicted_jointly_selective": int(
            ((jnk3 >= 0.4) & (gsk3b_coordinate >= 0.9)).sum()),
        "predicted_beats_archive_gain": int(
            sum(1 for row in predicted[:64] if archive.gain_of(row) > 0)),
        "target_represented": int(meets.sum()),
        "quantiles": {
            name: [round(float(q), 4) for q in
                   np.quantile(predicted[:, axis], [0.5, 0.9, 0.99, 1.0])]
            for axis, name in enumerate(("qed", "jnk3", "sa", "gsk3b", "drd2"))
        },
    }


def _build_r_theta():
    import torch

    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state,
    )

    source = open_process_v2_t1_source(
        Path(ACTIVE8), gate_zero_decision_path=Path(GATE_ZERO),
        artifact_root=ARTIFACT_ROOT, repo_root=REMOTE_ROOT)
    runtime, _binding, _receipt = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=load_materialized_scorer_state(Path(MATERIALIZED)))
    model = runtime.model
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["selected_model_state"], strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    return model


@app.function(
    image=image,
    cpu=2.0,
    memory=8 * 1024,
    timeout=75 * 60,
    max_containers=6,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_arm(spec: tuple[str, int, int, int]) -> dict[str, Any]:
    """One (arm, seed). Identical to the other arm except how it chooses."""

    arm_name, seed, budget, candidates = spec
    import numpy as np

    from compose_v4.benchmark.oracles.task3 import navigation_lockout
    from compose_v4.benchmark.task3_run import Task3Run
    from compose_v4.chem.molecular_graph import (
        molecular_graph_to_smiles,
        smiles_to_molecular_graph,
    )
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_successor_result,
    )
    from compose_v4.policy.task3.archive import ParetoArchive
    from compose_v4.policy.task3.steering import AdaptiveRegion, FixedScalarization
    from compose_v4.policy.task3.surrogate import TanimotoKNN

    started = time.perf_counter()
    fixture = json.loads((REMOTE_ROOT / "archives.json").read_text())
    entry = fixture["archives"][str(seed)]
    seeded = {s: tuple(v) for s, v in entry["molecules"].items()}

    archive = ParetoArchive()
    archive.add_many(seeded)
    surrogate = TanimotoKNN()
    surrogate.update(list(seeded), list(seeded.values()))
    steering = {"fixed-scalarization": FixedScalarization,
                "adaptive-region": AdaptiveRegion}[arm_name]()

    model = _build_r_theta()
    build_seconds = time.perf_counter() - started
    print({"phase": "runtime_built", "arm": arm_name, "seed": seed,
           "seconds": build_seconds, "archive_sha256": entry["sha256"]}, flush=True)

    root = Path(OUTPUT_ROOT) / f"{arm_name}_seed{seed}"
    run = Task3Run.open(root, seed=seed, budget=budget, policy=arm_name)
    expansions = 0
    stalled = 0
    composition: list[dict[str, Any]] = []
    fiber_sizes: list[int] = []
    proposals_seen = 0
    try:
        while run.remaining > 0:
            if stalled >= STALL_LIMIT:
                # An arm whose start never moves re-expands one state forever:
                # every successor is already charged, nothing new is proposed,
                # `remaining` never falls, and the container burns 12.5 s an
                # iteration until the timeout. Found the hard way -- one arm of
                # the first R_theta A/B died exactly here at step 60 of 75.
                print({"stalled_out": True, "arm": arm_name, "seed": seed,
                       "step": run.step, "charged": run.spent}, flush=True)
                break
            target = steering.target(archive, run.rng)
            if target is None:
                break
            # The whole fiber is built and scored INTERNALLY. The lockout makes
            # it impossible to price any of it against the benchmark budget.
            with navigation_lockout():
                proposed: list[str] = []
                for _ in range(START_ATTEMPTS):
                    start = steering.start(archive, target, run.rng)
                    if start is None:
                        break
                    try:
                        graph = smiles_to_molecular_graph(start)
                        padded = pad_molecular_graph(graph, graph.n_atoms + PAD_SLOTS)
                        result = canonical_successor_result(model, padded, 0.0)
                    except Exception as error:  # noqa: BLE001
                        print({"expansion_failed": start[:60],
                               "error": f"{type(error).__name__}: {error}"},
                              flush=True)
                        continue
                    expansions += 1
                    fiber = result.batch.successors
                    fiber_sizes.append(len(fiber))
                    for successor in fiber:
                        smiles = molecular_graph_to_smiles(successor.state)
                        if smiles:
                            proposed.append(smiles)
                    break
                proposed = [s for s in dict.fromkeys(proposed)
                            if s not in archive.values]
                proposals_seen += len(proposed)
                if not proposed:
                    stalled += 1
                    continue
                predicted = surrogate.predict(proposed)
                # Logged BEFORE selection, so a null result can be attributed.
                composition.append(fiber_composition(predicted, target, archive))
                order = np.argsort(-steering.rank(predicted, target))
            chosen = run.affordable([proposed[int(i)] for i in order[:candidates]])
            if not chosen:
                break
            values = run.evaluate(chosen)
            stalled = 0
            for smiles, value in zip(chosen, values):
                archive.add(smiles, value)
            surrogate.update(chosen, list(values))
            run.step += 1
            if run.step % 10 == 0:
                run.archive = [s for s, _ in archive.front()]
                run.checkpoint(policy_state={"arm": arm_name,
                                             "expansions": expansions})
                artifact_volume.commit()
                print({"arm": arm_name, "seed": seed, "step": run.step,
                       "charged": run.spent, "expansions": expansions,
                       "elapsed_s": round(time.perf_counter() - started, 1)},
                      flush=True)
        run.checkpoint(policy_state={"arm": arm_name, "done": True,
                                     "expansions": expansions})

        evaluated = np.asarray(list(run.meter.evaluated().values()))
        seeded_values = np.asarray(list(seeded.values()))
        from compose_v4.benchmark.molleo_task3 import hypervolume_qmc

        combined = (np.vstack([seeded_values, evaluated]) if len(evaluated)
                    else seeded_values)
        jnk3 = evaluated[:, 1] if len(evaluated) else np.zeros(1)
        gsk3b_activity = 1.0 - evaluated[:, 3] if len(evaluated) else np.ones(1)
        high = jnk3 >= 0.4
        report = {
            "arm": arm_name, "seed": seed,
            "archive_sha256": entry["sha256"],
            "charged": int(run.spent),
            "expansions": expansions,
            "median_fiber": float(np.median(fiber_sizes)) if fiber_sizes else 0.0,
            "proposals_scored_internally": proposals_seen,
            "fiber_composition_mean": {
                key: float(np.mean([c[key] for c in composition if key in c]))
                for key in ("fiber", "predicted_high_jnk3", "predicted_low_gsk3b",
                            "predicted_jointly_selective", "target_represented")
            } if composition else {},
            "stalled_out": stalled >= STALL_LIMIT,
            "iterations_where_target_was_represented": int(
                sum(1 for c in composition if c.get("target_represented", 0) > 0)),
            "fiber_composition_note": (
                "PREDICTED values: fiber composition convolved with surrogate "
                "error, not ground truth about what R_theta exposes"),
            "hv_seeded_only": hypervolume_qmc(seeded_values, log2_samples=17),
            "hv_after": hypervolume_qmc(combined, log2_samples=17),
            "front_size": len(archive.front()),
            "best_jnk3": float(jnk3.max()),
            "n_jnk3_over_0.4": int(high.sum()),
            "n_selective": int((high & (gsk3b_activity <= 0.1)).sum()),
            "seconds": time.perf_counter() - started,
            "runtime_build_seconds": build_seconds,
        }
        print(json.dumps(report), flush=True)
        artifact_volume.commit()
        return report
    finally:
        run.close()


@app.function(
    image=image,
    cpu=2.0,
    memory=8 * 1024,
    timeout=75 * 60,
    max_containers=3,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def audit_fiber_composition(spec: tuple[int, int, int]) -> dict[str, Any]:
    """What does R_theta ACTUALLY expose? Charged, because the answer costs labels.

    The composition logged during the A/B is predicted, so it cannot separate
    "the fiber lacks the region" from "the surrogate cannot see the region".
    This spends its budget measuring the fiber instead of optimising over it: it
    expands a start, charges a RANDOM sample of that fiber, and reports the true
    composition. Random, not ranked -- a ranked sample would measure the
    controller again, which is the thing being controlled for.

    It answers the first branch of the decomposition on its own:

        R_theta exposes useful chemistry?  -> if NO, the problem is expansion,
                                              not archive selection.
    """

    seed, expansions_wanted, per_fiber = spec
    import numpy as np

    from compose_v4.benchmark.oracles.task3 import navigation_lockout
    from compose_v4.benchmark.task3_run import Task3Run
    from compose_v4.chem.molecular_graph import (
        molecular_graph_to_smiles,
        smiles_to_molecular_graph,
    )
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_successor_result,
    )

    fixture = json.loads((REMOTE_ROOT / "archives.json").read_text())
    entry = fixture["archives"][str(seed)]
    seeded = {s: tuple(v) for s, v in entry["molecules"].items()}
    model = _build_r_theta()

    rng = np.random.default_rng(seed)
    # Start from the archive's own high-JNK3 molecules: the question is what
    # R_theta exposes AROUND useful chemistry, not around arbitrary molecules.
    ranked = sorted(seeded.items(), key=lambda kv: -kv[1][1])
    starts = [s for s, _ in ranked[:expansions_wanted]]

    root = Path(OUTPUT_ROOT) / f"fiber_audit_seed{seed}"
    run = Task3Run.open(root, seed=seed, budget=expansions_wanted * per_fiber,
                        policy="fiber-audit")
    rows = []
    try:
        for start in starts:
            with navigation_lockout():
                try:
                    graph = smiles_to_molecular_graph(start)
                    padded = pad_molecular_graph(graph, graph.n_atoms + PAD_SLOTS)
                    fiber = canonical_successor_result(model, padded, 0.0).batch.successors
                except Exception as error:  # noqa: BLE001
                    print({"expansion_failed": start[:60],
                           "error": f"{type(error).__name__}: {error}"}, flush=True)
                    continue
                smiles = [molecular_graph_to_smiles(s.state) for s in fiber]
                smiles = [s for s in dict.fromkeys(smiles) if s]
            if not smiles:
                continue
            index = rng.choice(len(smiles), size=min(per_fiber, len(smiles)),
                               replace=False)
            sample = run.affordable([smiles[int(i)] for i in index])
            if not sample:
                break
            values = np.asarray(run.evaluate(sample))
            jnk3 = values[:, 1]
            gsk3b_activity = 1.0 - values[:, 3]
            rows.append({
                "start": start,
                "start_jnk3": float(seeded[start][1]),
                "fiber": len(smiles),
                "sampled": len(sample),
                "true_high_jnk3": int((jnk3 >= 0.4).sum()),
                "true_selective": int(((jnk3 >= 0.4) & (gsk3b_activity <= 0.1)).sum()),
                "max_jnk3": float(jnk3.max()),
                "median_jnk3": float(np.median(jnk3)),
            })
            print(rows[-1], flush=True)
            artifact_volume.commit()

        sampled = sum(r["sampled"] for r in rows)
        report = {
            "phase": "fiber_composition_audit", "seed": seed,
            "expansions": len(rows), "molecules_sampled": sampled,
            "fraction_high_jnk3": (sum(r["true_high_jnk3"] for r in rows) / sampled
                                   if sampled else 0.0),
            "fraction_selective": (sum(r["true_selective"] for r in rows) / sampled
                                   if sampled else 0.0),
            "rows": rows,
            "note": ("TRUE objectives on a RANDOM sample of each fiber. Charged, "
                     "development-only, and spent on measuring what expansion "
                     "offers rather than on optimising."),
        }
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)
        artifact_volume.commit()
        return report
    finally:
        run.close()


@app.local_entrypoint()
def main(budget: int = 300, candidates: int = 4) -> None:
    specs = [(arm, seed, budget, candidates)
             for seed in (100, 101, 102)
             for arm in ("fixed-scalarization", "adaptive-region")]
    print(json.dumps({"phase": "launching", "runs": len(specs), "budget": budget,
                      "note": "DEVELOPMENT-ONLY -- not Task 3 performance"}))
    results = list(run_arm.map(specs))
    by_arm: dict[str, list[dict]] = {}
    for row in results:
        by_arm.setdefault(row["arm"], []).append(row)
    print("\n--- DEVELOPMENT-ONLY MECHANISM TEST (R_theta expansion) ---")
    for arm, rows in sorted(by_arm.items()):
        rows.sort(key=lambda r: r["seed"])
        lifts = [r["hv_after"] - r["hv_seeded_only"] for r in rows]
        print(f"{arm:<22} HV lift {sum(lifts)/len(lifts):+.4f}  "
              f"per-seed {[round(x, 4) for x in lifts]}  "
              f"selective {sum(r['n_selective'] for r in rows)}  "
              f"best jnk3 {max(r['best_jnk3'] for r in rows):.2f}  "
              f"expansions {sum(r['expansions'] for r in rows)}")
    Path("diagnostics").mkdir(exist_ok=True)
    Path("diagnostics/task3_rtheta_mechanism_ab.json").write_text(
        json.dumps({"STATUS": "DEVELOPMENT-ONLY -- NOT TASK 3 PERFORMANCE",
                    "results": results}, indent=1) + "\n")
