"""Held-in smoke for target-free Pareto / preference control. CPU ONLY.

*** NOT LAUNCHED. This file is a COSTED PLAN awaiting main-lane authorization. ***

Five arms x five preferences x six edits, on held-in sources only. It produces a
SMOKE_HELD_IN artifact and no claim. The held-out reserve is not opened here and
this app has no code path that could open it.

WHAT IT MEASURES, AND WHAT IT CANNOT
------------------------------------
It measures whether preference control produces preference-DEPENDENT futures at
matched budget: normalized hypervolume over committed endpoints, HV-AUC against
BOTH oracle-call conventions, preference coverage, nondominated-set size,
feasibility, and endpoint structural diversity.

It cannot, and does not, report:
  * sacrifice-to-win -- the lookahead action is argmin V_G, so scoring it by V_G
    has no falsifying range;
  * a sign test on verified-minus-greedy -- policy improvement makes the null
    false before any data exists;
  * a hypervolume comparison at unmatched budget -- an arm that generates more
    molecules inflates HV without controlling anything better.
`scripts/pareto_instrument_gate.py` enforces all three against the output.

DURABILITY, AND WHY IT IS BUILT THIS WAY
-----------------------------------------
`--detach` did NOT survive a client-side DNS failure twice in one day on this
project. Detaching is necessary and not sufficient. What actually protects work:

  * one DURABLE SHARD PER TASK, committed to the volume before the task returns;
  * a CHECKPOINT INSIDE each task, written after every arm, because a task here
    runs for over an hour and losing it whole is the expensive failure;
  * a RESUMABLE DRIVER that skips sources whose shard already exists;
  * an aggregate reconstructible from shards alone.

Launch procedure, once authorized:

    modal run --detach modal_apps/pareto_control_app.py --sources 12
    modal app list          # MUST show `ephemeral (detached)`

COST, MEASURED NOT GUESSED
--------------------------
Kernel calls per source (budget 6, five preferences, shortlist 8), before the
enumeration cache, which the committed calibration shows saves roughly 2x:

    unguided             5 seeds x 6 steps                        =  30
    greedy_pref          5 prefs x 6 steps                        =  30
    verified_pref        5 prefs x (6 + 8 x (5+4+3+2+1+0))        = 630
    gen_rank@greedy      kernel-matched to greedy_pref            =  30
    gen_rank@verified    kernel-matched to verified_pref          = 630
                                                          raw total 1350
                                          with observed ~2x caching  ~700

At `cpu=8.0` with `OMP_NUM_THREADS=4` an enumeration measured 5.2-8 s locally,
against 22 s on the calibration's `cpu=2.0, OMP_NUM_THREADS=1`. At 7 s/call that
is ~1.4 h per source; 12 sources in 12 containers is ~1.4 h wall and roughly
$18 of CPU. Shortlist 4 instead of 8 halves it to ~0.8 h and ~$10.
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
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}",
          "OMP_NUM_THREADS": "4"})
    .add_local_file(
        ROOT / "diagnostics/retarget_goal_language_normalizers.json",
        str(REMOTE_ROOT / "diagnostics/retarget_goal_language_normalizers.json"),
        copy=True)
    .add_local_file(
        ROOT / "diagnostics/pareto_control_cohort.json",
        str(REMOTE_ROOT / "diagnostics/pareto_control_cohort.json"), copy=True)
    .add_local_file(
        ROOT / "diagnostics/pareto_tradeoff_census.json",
        str(REMOTE_ROOT / "diagnostics/pareto_tradeoff_census.json"), copy=True)
    .add_local_dir(ROOT / "artifacts/oracles/drd2_svm_v1",
                   str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1"), copy=True)
)
app = modal.App("compose-v4-pareto-control")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
OUT_DIR = "pareto_control_smoke"
TIME_POINT = 0.5
CANONICAL_SLOTS = 48

#: Frozen by PROTOCOL.md section 7.1. Not tunable here.
BUDGET = 6
PREFERENCES = (0.1, 0.3, 0.5, 0.7, 0.9)
SHORTLIST = {"top_immediate": 4, "top_reference": 2, "n_random": 2}


@app.function(
    image=image, cpu=8.0, memory=16 * 1024, timeout=6 * 60 * 60,
    max_containers=12, retries=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_source(task: dict[str, Any]) -> dict[str, Any]:
    """One source, five arms, five preferences. Checkpoints after every arm."""
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, QED, rdFingerprintGenerator
    from rdkit import DataStructs

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
    from compose_v4.experiments.pareto_control import (
        MeteredProcess,
        Scalarization,
        branch_from_common_prefix,
        endpoint_diversity,
        generate_then_rank,
        greedy_preference_run,
        normalized_hypervolume,
        pareto_front_indices,
        preference_coverage,
        trajectories_for_kernel_budget,
        unguided_run,
        verified_preference_run,
    )
    from compose_v4.experiments.production_successor_kernel import (
        canonical_successor_result,
    )
    from compose_v4.rewrite.kernel import canonical_state_key

    RDLogger.DisableLog("rdApp.*")
    started = time.perf_counter()
    artifact_volume.reload()

    out = Path(RUN_ROOT) / task.get("out_dir", OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    shard = out / f"{task['index']:03d}.json"
    partial = out / f"{task['index']:03d}.partial.json"
    if shard.exists():
        print(f"[{task['index']:03d}] shard exists, skipping", flush=True)
        return json.loads(shard.read_text())

    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    source_obj = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]),
        repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        source_obj, materialized_state=bundle)
    model = runtime.model
    checkpoint = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                            map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["selected_model_state"], strict=True)
    model.eval()
    torch.set_grad_enabled(False)

    census = json.loads((REMOTE_ROOT / "diagnostics/pareto_tradeoff_census.json").read_text())
    pair = census["adopted_pair"]
    entry = next(p for p in census["pairs"] if p["pair"] == pair)
    key_a, key_b = entry["objective_a"], entry["objective_b"]
    scales = census["frozen_scales"]
    utopia = np.array([scales["utopia_p99"][key_a], scales["utopia_p99"][key_b]])
    reference = np.array([scales["reference_p5"][key_a], scales["reference_p5"][key_b]])
    centre_s, s_s = scales["similarity_centre"], scales["similarity_iqr"]

    norms = json.loads((REMOTE_ROOT /
                        "diagnostics/retarget_goal_language_normalizers.json"
                        ).read_text())["normalizers"]
    centre_p, s_p = norms["drd2_logodds"]["median"], norms["drd2_logodds"]["iqr"]
    s_qed, s_logp = norms["qed"]["iqr"], norms["clogp"]["iqr"]
    oracle = load_default_oracle(
        str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json"))

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    start = canonical_state_key(
        pad_molecular_graph(smiles_to_molecular_graph(task["source"]), CANONICAL_SLOTS))
    source_fp = gen.GetFingerprint(Chem.MolFromSmiles(start))

    def objective_vector(key: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(key)
        if mol is None:
            return np.array([-10.0, -10.0])
        values = {}
        values["P"] = (float(oracle.margin_many([key])[0]) - centre_p) / s_p
        try:
            qed = float(QED.qed(mol))
        except Exception:  # noqa: BLE001
            qed = float("nan")
        logp = float(Crippen.MolLogP(mol))
        m = np.clip([(qed - 0.6) / s_qed,
                     min(logp - 1.0, 4.0 - logp) / s_logp], -1.5, 1.5)
        values["D"] = float(-0.25 * np.log(np.sum(np.exp(-m / 0.25))))
        values["S"] = (DataStructs.TanimotoSimilarity(
            source_fp, gen.GetFingerprint(mol)) - centre_s) / s_s
        return np.array([values[key_a], values[key_b]])

    class _Process:
        def successors(self, key: str) -> list[tuple[str, float]]:
            try:
                state = pad_molecular_graph(
                    smiles_to_molecular_graph(key), CANONICAL_SLOTS)
            except Exception:  # noqa: BLE001
                return []
            result = canonical_successor_result(model, state, float(TIME_POINT))
            return [(s.key, float(s.probability)) for s in result.batch.successors]

    class _Objectives:
        def z(self, key: str) -> np.ndarray:
            return objective_vector(key)

    scalarize = Scalarization(utopia)
    payload: dict[str, Any] = {
        "index": int(task["index"]), "source": task["source"], "start": start,
        "pair": pair, "objective_a": key_a, "objective_b": key_b,
        "budget": BUDGET, "preferences": list(PREFERENCES),
        "status": "SMOKE_HELD_IN", "held_out_opened": False, "arms": {},
    }

    def checkpoint_now(stage: str) -> None:
        payload["stage"] = stage
        payload["seconds"] = round(time.perf_counter() - started, 1)
        partial.write_text(json.dumps(payload, indent=2, default=float) + "\n")
        artifact_volume.commit()
        print(f"[{task['index']:03d}] checkpoint {stage} "
              f"{payload['seconds']:.0f}s", flush=True)

    def summarize(name: str, runs: list) -> None:
        z = np.stack([r.endpoint_z for r in runs])
        keys = [r.endpoint for r in runs]
        fps = [gen.GetFingerprint(Chem.MolFromSmiles(k)) for k in keys]
        sim = np.array([[DataStructs.TanimotoSimilarity(a, b) for b in fps]
                        for a in fps])
        payload["arms"][name] = {
            "endpoints": keys,
            "endpoint_z": z.tolist(),
            "normalized_hypervolume": normalized_hypervolume(z, reference, utopia),
            "preference_coverage": preference_coverage(z, PREFERENCES, scalarize, keys),
            "nondominated_set_size": int(len(pareto_front_indices(z))),
            "feasible": [bool(r.complete) for r in runs],
            "endpoint_diversity": endpoint_diversity(sim),
            "source_similarity": [float(DataStructs.TanimotoSimilarity(source_fp, f))
                                  for f in fps],
            # D2 evidence: the committed action sequences themselves.
            "action_sequences": [list(r.actions) for r in runs],
            "overrode_greedy": [int(r.overrode_greedy) for r in runs],
            "decisions": [r.decisions for r in runs],
        }

    # --- Arm 1: unguided, preference-blind -------------------------------
    metered = MeteredProcess(_Process(), _Objectives())
    unguided = [unguided_run(metered, start, BUDGET, seed=1000 + i)
                for i in range(len(PREFERENCES))]
    summarize("unguided", unguided)
    payload["arms"]["unguided"]["cost"] = metered.ledger.as_dict()
    checkpoint_now("unguided")

    # --- Arm 3: greedy preference control --------------------------------
    metered = MeteredProcess(_Process(), _Objectives())
    greedy = [greedy_preference_run(metered, start, BUDGET, w, scalarize)
              for w in PREFERENCES]
    summarize("greedy_pref", greedy)
    greedy_cost = metered.ledger.as_dict()
    payload["arms"]["greedy_pref"]["cost"] = greedy_cost
    checkpoint_now("greedy_pref")

    # --- Arm 2a: generate-then-rank, kernel-matched to greedy -------------
    metered = MeteredProcess(_Process(), _Objectives())
    k_greedy = trajectories_for_kernel_budget(greedy_cost["kernel_calls"], BUDGET)
    summarize("gen_rank@greedy",
              generate_then_rank(metered, start, BUDGET, PREFERENCES, scalarize,
                                 k_greedy, seed=7))
    payload["arms"]["gen_rank@greedy"]["cost"] = metered.ledger.as_dict()
    payload["arms"]["gen_rank@greedy"]["n_trajectories"] = k_greedy
    checkpoint_now("gen_rank@greedy")

    # --- Arm 4: verified remaining-budget preference control --------------
    metered = MeteredProcess(_Process(), _Objectives())
    verified = [verified_preference_run(metered, start, BUDGET, w, scalarize,
                                        seed=int(task["index"]) * 31 + i, **SHORTLIST)
                for i, w in enumerate(PREFERENCES)]
    summarize("verified_pref", verified)
    verified_cost = metered.ledger.as_dict()
    payload["arms"]["verified_pref"]["cost"] = verified_cost
    checkpoint_now("verified_pref")

    # --- Arm 2b: generate-then-rank, kernel-matched to verified -----------
    metered = MeteredProcess(_Process(), _Objectives())
    k_verified = trajectories_for_kernel_budget(verified_cost["kernel_calls"], BUDGET)
    summarize("gen_rank@verified",
              generate_then_rank(metered, start, BUDGET, PREFERENCES, scalarize,
                                 k_verified, seed=11))
    payload["arms"]["gen_rank@verified"]["cost"] = metered.ledger.as_dict()
    payload["arms"]["gen_rank@verified"]["n_trajectories"] = k_verified
    checkpoint_now("gen_rank@verified")

    # --- Contrast P6: one realized history, five preference-dependent futures
    # The prefix is committed under a single balanced policy and finished BEFORE
    # any branch preference is injected, so all five branches share x_0..x_3
    # exactly. Without that separation this would be five trajectories rather
    # than one history with five futures.
    metered = MeteredProcess(_Process(), _Objectives())
    fan = branch_from_common_prefix(metered, start, 3, BUDGET - 3, PREFERENCES,
                                    scalarize, prefix_weight=0.5)
    payload["prefix_branching"] = {
        "prefix_states": fan["prefix"].states,
        "branch_point": fan["branch_point"],
        "prefix_weight": 0.5,
        "prefix_steps": 3,
        "branch_steps": BUDGET - 3,
        "branches": {str(w): {"states": r.states, "endpoint": r.endpoint,
                              "endpoint_z": r.endpoint_z.tolist(),
                              "complete": bool(r.complete)}
                     for w, r in fan["branches"].items()},
        "cost": metered.ledger.as_dict(),
    }
    checkpoint_now("prefix_branching")

    payload["seconds"] = round(time.perf_counter() - started, 1)
    shard.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    if partial.exists():
        partial.unlink()
    artifact_volume.commit()
    print(f"[{task['index']:03d}] DONE {payload['seconds']:.0f}s  "
          f"greedy HV {payload['arms']['greedy_pref']['normalized_hypervolume']:.4f}  "
          f"verified HV {payload['arms']['verified_pref']['normalized_hypervolume']:.4f}",
          flush=True)
    return payload


@app.function(image=image, cpu=0.25, memory=2048, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]]) -> int:
    """Fans out ON MODAL so a client disconnect cannot stall the run.

    RESUMABLE: `run_source` returns immediately for any source whose durable
    shard already exists, so a relaunch after a failure costs only the sources
    that had not finished. This is what actually protected work when --detach
    did not survive a client-side DNS failure.
    """
    artifact_volume.reload()
    done = 0
    for _ in run_source.map(tasks, order_outputs=False, return_exceptions=True):
        done += 1
    return done


@app.local_entrypoint()
def main(sources: int = 12, smoke: bool = False) -> None:
    """The cohort is read from a COMMITTED artifact frozen before this runs, so
    the source set is auditable and identical across reruns. No molecule is
    selected here."""
    cohort_path = ROOT / "diagnostics/pareto_control_cohort.json"
    if not cohort_path.exists():
        raise SystemExit(
            "diagnostics/pareto_control_cohort.json is missing. Freeze the cohort "
            "before launching: the source set must exist as a committed artifact "
            "so it cannot be reselected after seeing a result.")
    cohort = json.loads(cohort_path.read_text())
    chosen = [dict(row) for row in cohort["sources"]][: 2 if smoke else sources]
    for row in chosen:
        row["out_dir"] = OUT_DIR + ("_smoke" if smoke else "")

    census = json.loads((ROOT / "diagnostics/pareto_tradeoff_census.json").read_text())
    print(f"{'SMOKE' if smoke else 'HELD-IN RUN'}: {len(chosen)} held-in sources, "
          f"cohort {cohort['cohort_sha256'][:16]}")
    print(f"adopted pair: {census['adopted_pair']}")
    print(f"budget {BUDGET}, preferences {PREFERENCES}, five arms")
    print("STATUS: SMOKE_HELD_IN. The held-out reserve is not opened by this app.")
    print(f"completed {drive.remote(chosen)}")
