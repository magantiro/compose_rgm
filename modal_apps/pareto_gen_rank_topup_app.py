"""Repair the P3/P4 generate-and-rank baseline. BLINDED FEASIBILITY PROBE FIRST.

WHAT THIS APP DOES NOT DO
-------------------------
It does not run, re-run, or touch any COMPOSE arm. `pareto_control_app.py` is
untouched and its committed shards are read-only inputs here. Only the
generate-and-rank baseline is computed, because only the baseline was
underfunded.

THE TARGET IS READ, NEVER RECOMPUTED
------------------------------------
Each source's kernel target is `verified_pref.cost.kernel_calls` (or
`greedy_pref`'s) **from that source's committed smoke shard** -- source 000's is
368. Recomputing COMPOSE to obtain the target would silently convert "repair the
comparator against the frozen smoke" into a new experiment whose baseline no
longer corresponds to the COMPOSE arms it is contrasted with. The shard is the
authority; `_committed_target` refuses to proceed if it is absent.

THE PROBE IS BLINDED, STRUCTURALLY
----------------------------------
`--probe` passes `hypervolume=None`, so no hypervolume and no preference
selection are COMPUTED -- not computed-then-hidden. The emitted report is
`feasibility_only()`: status, trajectory count, kernel prefix, and the resource
counters. A go/no-go on operational feasibility must be unable to read the
science it would be tempted to condition on.

It wastes nothing. Endpoints and their already-paid-for objective vectors are
retained in the shard, so P3/P4 and every hypervolume are computed offline
afterwards, for all twelve sources together, with nothing re-run.

THE TWO ADMISSIBLE PROBE OUTCOMES, AND WHAT NEITHER LICENSES
-------------------------------------------------------------
MATCHED       the baseline can spend the intended kernel budget. Proceed to the
              remaining eleven, each against ITS OWN committed target.
UNREACHABLE   from the fixed source under the declared generate-and-rank
              procedure, the method exhausts new kernel work before it can
              consume closed-loop COMPOSE's budget. There is then no
              kernel-matched scalar P3 for that source and the comparison is the
              full resource frontier.

Neither outcome licenses modifying the matcher, generating more exotic
trajectories, moving the start state, or disabling the cache to force a match.
Forcing an open-loop method to consume kernel work its reachable structure
cannot consume would be fake parity.

DURABILITY. One shard per source committed before the task returns, a resumable
driver that skips finished shards, and a checkpoint every 50 trajectories --
because this arm's runtime is unbounded by construction and losing it whole is
the expensive failure.
"""

from __future__ import annotations

import gzip
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

# Identical image surface to the frozen smoke: same env, same mounted artifacts.
# OMP_NUM_THREADS=4 at cpu=8.0 is what the committed run used; leaving the base
# image's 1 would change nothing scientific but would triple the wall time.
image = (
    _base_image
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}",
          "OMP_NUM_THREADS": "4"})
    .add_local_file(
        ROOT / "diagnostics/retarget_goal_language_normalizers.json",
        str(REMOTE_ROOT / "diagnostics/retarget_goal_language_normalizers.json"),
        copy=True)
    .add_local_file(
        ROOT / "diagnostics/pareto_tradeoff_census.json",
        str(REMOTE_ROOT / "diagnostics/pareto_tradeoff_census.json"), copy=True)
    .add_local_dir(ROOT / "artifacts/oracles/drd2_svm_v1",
                   str(REMOTE_ROOT / "artifacts/oracles/drd2_svm_v1"), copy=True)
)

app = modal.App("pareto-gen-rank-topup")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
SMOKE_DIR = "pareto_control_smoke"
OUT_DIR = "pareto_gen_rank_topup"
TIME_POINT = 0.5
CANONICAL_SLOTS = 48

#: Identical to the frozen smoke. Not tunable here.
BUDGET = 6
PREFERENCES = (0.1, 0.3, 0.5, 0.7, 0.9)
#: Seeds as the committed smoke used them, so the pool prefix is comparable.
SEEDS = {"greedy_pref": 7, "verified_pref": 11}
CHECKPOINT_EVERY = 50


def _committed_target(shard: dict[str, Any], compose_arm: str) -> int:
    """The kernel budget to match, READ from the committed COMPOSE shard."""
    try:
        target = int(shard["arms"][compose_arm]["cost"]["kernel_calls"])
    except (KeyError, TypeError) as error:
        raise SystemExit(
            f"source {shard.get('index')}: no committed {compose_arm} kernel_calls. "
            "The target must come from the frozen smoke; recomputing COMPOSE "
            "would make this a new experiment.") from error
    if target <= 0:
        raise SystemExit(f"source {shard.get('index')}: committed target is {target}")
    return target


@app.function(
    image=image, cpu=8.0, memory=16 * 1024, timeout=6 * 60 * 60,
    max_containers=12, retries=5,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def topup_source(task: dict[str, Any]) -> dict[str, Any]:
    """One source, generate-and-rank only, metered to the committed target."""
    import sys

    import numpy as np
    import torch
    from rdkit import Chem, RDLogger

    sys.path.insert(0, str(REMOTE_ROOT / "src"))

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
        Trajectory,
        _argmin_stable,
        normalized_hypervolume,
        unguided_run,
    )
    from compose_v4.experiments.pareto_gen_rank_matched import (
        generate_then_rank_metered,
    )
    from compose_v4.experiments.production_successor_kernel import (
        canonical_successor_result,
    )
    from compose_v4.rewrite.kernel import canonical_state_key  # noqa: F401

    RDLogger.DisableLog("rdApp.*")
    started = time.perf_counter()
    artifact_volume.reload()

    index = int(task["index"])
    compose_arm = str(task["compose_arm"])
    probe = bool(task.get("probe", False))

    out = Path(RUN_ROOT) / task.get("out_dir", OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{index:03d}_{compose_arm}"
    shard_path = out / f"{tag}.json"
    if shard_path.exists():
        print(f"[{tag}] shard exists, skipping", flush=True)
        return json.loads(shard_path.read_text())

    smoke = json.loads(
        (Path(RUN_ROOT) / SMOKE_DIR / f"{index:03d}.json").read_text())
    target = _committed_target(smoke, compose_arm)
    start = smoke["start"]
    print(f"[{tag}] committed target {target} kernel calls (READ, not recomputed)",
          flush=True)

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

    from rdkit import DataStructs
    from rdkit.Chem import Crippen, QED, rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
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

    metered = MeteredProcess(_Process(), _Objectives())
    scalarize = Scalarization(utopia)
    partial = out / f"{tag}.partial.json"
    # THE RESUMABLE CHECKPOINT. The expensive state is the ENUMERATION CACHE --
    # one entry is a ~7 s kernel call -- so a checkpoint that stores only counts
    # is a progress log, not a checkpoint. A Modal preemption at t=800 restarted
    # this source at zero and cost 76 minutes, which is exactly what this file
    # exists to prevent.
    resume_path = out / f"{tag}.resume.json.gz"

    class _Run:
        """Rehydrated trajectory. Same duck-type the matcher and selection use."""

        def __init__(self, states, endpoint, endpoint_z, complete):
            self.states, self.endpoint = states, endpoint
            self.endpoint_z, self.complete = endpoint_z, complete

    resume = None
    if resume_path.exists():
        blob = json.loads(gzip.decompress(resume_path.read_bytes()))
        metered._successors = {k: [(a, float(b)) for a, b in v]
                               for k, v in blob["successors"].items()}
        metered._z = {k: np.asarray(v, dtype=float) for k, v in blob["z"].items()}
        metered.ledger.kernel_calls = int(blob["ledger"]["kernel_calls"])
        metered.ledger.raw_oracle_calls = int(blob["ledger"]["raw_oracle_calls"])
        metered.ledger.native_oracle_calls = int(blob["ledger"]["native_oracle_calls"])
        resume = {
            "pool": [_Run(tuple(r["states"]), r["endpoint"],
                          np.asarray(r["endpoint_z"], dtype=float), r["complete"])
                     for r in blob["runs"]],
            "prefix": blob["prefix"],
            "endpoint_z": [r["endpoint_z"] for r in blob["runs"]],
            "detector": blob.get("detector"),
        }
        print(f"[{tag}] RESUMED from checkpoint: t={len(resume['pool'])} "
              f"kernel {metered.ledger.kernel_calls}/{target} "
              f"({len(metered._successors)} cached enumerations)", flush=True)

    def _checkpoint(n: int, realized: int, tgt: int, prefix: list,
                    pool_now: list, detector: dict) -> None:
        """Durable AND RESUMABLE AND LOSSLESS.

        Persists the ENUMERATION CACHE (one entry = a ~7 s kernel call), the
        objective cache, the ledger, every completed trajectory, and the stall
        detector's loop-carried state. Everything the matcher reads across an
        iteration boundary is here; the RNG is not, because seeds are derived
        from pool position rather than from an advancing stream.
        """
        resume_path.write_bytes(gzip.compress(json.dumps({
            "successors": {k: [[a, b] for a, b in v]
                           for k, v in metered._successors.items()},
            "z": {k: list(map(float, v)) for k, v in metered._z.items()},
            "ledger": metered.ledger.as_dict(),
            "prefix": prefix,
            "runs": [{"states": list(r.states), "endpoint": r.endpoint,
                      "endpoint_z": list(map(float, r.endpoint_z)),
                      "complete": bool(r.complete)} for r in pool_now],
            "detector": detector,
        }).encode()))
        partial.write_text(json.dumps({
            "index": index, "compose_arm": compose_arm, "in_progress": True,
            "committed_target_kernel_calls": tgt,
            "n_trajectories_so_far": n, "realized_kernel_calls": realized,
            "seconds": round(time.perf_counter() - started, 1),
            "kernel_prefix": [{k: r[k] for k in
                               ("t", "kernel_calls", "native_oracle_calls",
                                "raw_oracle_calls")} for r in prefix],
        }, indent=2) + "\n")
        artifact_volume.commit()
        print(f"[{tag}] checkpoint t={n} kernel {realized}/{tgt} "
              f"{time.perf_counter() - started:.0f}s", flush=True)

    result = generate_then_rank_metered(
        metered, start, BUDGET, PREFERENCES, scalarize, target,
        seed=SEEDS[compose_arm], reference=reference, utopia=utopia,
        unguided_run=unguided_run,
        # THE FIREWALL. In probe mode the outcome is not computed at all.
        hypervolume=None if probe else normalized_hypervolume,
        argmin_stable=None if probe else _argmin_stable,
        trajectory_cls=None if probe else Trajectory,
        on_progress=_checkpoint, progress_every=CHECKPOINT_EVERY,
        resume=resume,
    )

    payload: dict[str, Any] = {
        "schema": "compose.pareto.gen_rank_topup",
        "index": index, "source": smoke["source"], "start": start,
        "pair": pair, "compose_arm": compose_arm,
        "committed_target_kernel_calls": target,
        "target_provenance": f"READ from {SMOKE_DIR}/{index:03d}.json "
                             f"arms.{compose_arm}.cost.kernel_calls",
        "compose_was_not_rerun": True,
        # Tells pareto_oracle_semantics that ONE request per candidate serves
        # both the ranking and the benchmark evaluation here, so the post-hoc
        # scoring must NOT be subtracted again. See that module's docstring.
        "shares_scoring_with_ranking": True,
        "blinded_probe": probe,
        "status": "SMOKE_HELD_IN", "held_out_opened": False,
        "seconds": round(time.perf_counter() - started, 1),
        "matching": result.as_dict(),
        # Retained so the outcome is computable offline with nothing re-run.
        "endpoints": result.endpoints,
        "endpoint_z": result.endpoint_z,
        "ledger": metered.ledger.as_dict(),
    }
    if not probe:
        payload["selected_endpoints"] = [t.endpoint for t in result.selected]
        payload["selected_endpoint_z"] = [list(map(float, t.endpoint_z))
                                          for t in result.selected]

    shard_path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    partial.unlink(missing_ok=True)
    resume_path.unlink(missing_ok=True)
    artifact_volume.commit()
    report = result.feasibility_only()
    print(f"[{tag}] {report['status']}  trajectories={report['n_trajectories']}  "
          f"kernel {report['realized_kernel_calls']}/{target}  "
          f"{payload['seconds']:.0f}s", flush=True)
    return payload


@app.function(image=image, cpu=0.25, memory=2048, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]]) -> int:
    artifact_volume.reload()
    done = 0
    for _ in topup_source.map(tasks, order_outputs=False, return_exceptions=True):
        done += 1
    return done


@app.local_entrypoint()
def main(sources: str = "0", compose_arm: str = "verified_pref",
         probe: bool = True) -> None:
    """Default is the BLINDED source-000 feasibility probe.

    Sources are given explicitly rather than as a count, so the eleven-source
    continuation is an auditable list rather than an off-by-one.
    """
    indices = [int(s) for s in str(sources).split(",") if s.strip() != ""]
    tasks = [{"index": i, "compose_arm": compose_arm, "probe": probe,
              "out_dir": OUT_DIR} for i in indices]
    print(f"gen_rank top-up: sources {indices}, matched to {compose_arm}")
    print(f"BLINDED PROBE: {probe}  "
          f"({'no hypervolume or selection is computed' if probe else 'full outcome computed'})")
    print("COMPOSE is NOT rerun. Targets are read from the committed smoke shards.")
    # SPAWN, not remote(). `--detach` did not survive a client-side DNS failure
    # here -- the third time this project has hit that -- so the client must not
    # need to stay alive at all. It fires the driver and exits; progress is read
    # from the VOLUME, never from this process.
    call = drive.spawn(tasks)
    print(f"driver spawned: {call.object_id}")
    print("watch the VOLUME, not this log: "
          "modal volume ls compose-v4-artifacts editing_v2/r_theta_run/pareto_gen_rank_topup")
