"""COMPOSE on DDSBM's ZINC logP 2->4 benchmark. TIER-1: COMPOSE runs alone.

Protocol frozen in `docs/DDSBM_ENDPOINT_COMPETENCE_PROTOCOL.md` before any
outcome existed. Nothing here may be tuned against DDSBM's published table.

THE OBJECTIVE IS A TRANSPORT ADAPTER, NOT A POINT TARGET

    tau(x0)   = logP(x0) + 2
    u(y; x0)  = -|logP(y) - tau(x0)|

DDSBM moves a source marginal at logP~N(2, 0.5) to a target at N(4, 0.5) and
scores Wasserstein-1 between MARGINALS. A point objective at 4 would collapse the
generated spread toward the mode and inflate W1 against a target whose sd is 0.5.
Verified on the frozen test split before launch: source mean 2.015 sd 0.499, tau
mean 4.015 sd 0.499. Absolute error, not squared, because W1 is an L1 metric.

BARRED: the CSV's randomly paired PRB-SMI. That coupling exists for DDSBM's
training; the task is transport between marginals. Only REF-SMI is read.

EXECUTION: HORIZONTAL, ONE CORE PER WORKER
------------------------------------------
The kernel's mark loop is PURE PYTHON and single-threaded. An earlier version
requested `cpu=8.0` per container, reserving eight cores while ~seven idled
through every expansion -- the cost was cores held, not compute done. The actual
work is ~50 core-hours whatever the shape.

So: `cpu=(1.0, 1.0)` hard request+limit so a worker cannot quietly burst,
`max_containers=80`, and **one source per mapped work item** rather than fixed
250-source shards -- `.map()` then feeds the next source to whichever worker frees
up, so one slow molecule cannot hold a shard hostage and determine wall time.

The runtime is built **once per container** and reused across warm inputs. Loading
`R_theta` per molecule would dominate everything else.

NOTHING IS COMPUTED IN THE TRAJECTORY LOOP except what the controller needs:
logP of each successor. QED, SA, ring counts and edit expressivity are derived
afterwards from the persisted trajectory.

METRICS: TASK-ALIGNED ONLY
--------------------------
FCD and NSPDK are **deliberately not computed**. They ask whether an aggregate
bag of molecules resembles a target dataset -- DDSBM's question as a
distribution-generative model, not ours. Including them would invite the reader
to think COMPOSE is imitating a distribution rather than demonstrating controlled
source-conditioned editing.

This is therefore **not** "a reproduction of DDSBM Table 1". It is evaluation on
DDSBM's held-out benchmark under shared task-aligned metrics: logP W1, source->
endpoint QED and SA drift, structural change, endpoint validity, and -- uniquely
for COMPOSE -- trajectory-wide validity and edit expressivity, which no row in
their table can report.
"""

from __future__ import annotations

import csv
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

DDSBM_CSV = "local_runtime/ddsbm/DDSBM-main/data/raw/ZINC250k_logp_2_4_random_matched_no_nH.csv"

image = (
    _base_image
    # One core per worker, so OMP must not oversubscribe it.
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
    .add_local_file(ROOT / DDSBM_CSV, str(REMOTE_ROOT / "ddsbm_pairs.csv"), copy=True)
)

app = modal.App("ddsbm-endpoint-competence")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
OUT_DIR = "ddsbm_endpoint_competence"
TIME_POINT = 0.5
CANONICAL_SLOTS = 48
HORIZON = 6
TRAIN_ROWS = 23936          #: DDSBM's split; the last 5,984 are test.
#: Commit a partial this often. 100 keeps worst-case loss ~2 min of the run
#: while staying cheap against a 5,984-source map.
CHECKPOINT_EVERY = 100
LOGP_SHIFT = 2.0

#: Built once per container, reused across every warm input.
_RT: dict[str, Any] = {}


def _runtime():
    """Load R_theta and the executor ONCE per container."""
    if "model" in _RT:
        return _RT
    import sys

    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
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

    t0 = time.perf_counter()
    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    source_obj = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]), repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        source_obj, materialized_state=bundle)
    model = runtime.model
    ckpt = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["selected_model_state"], strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    _RT["model"] = model
    _RT["init_seconds"] = round(time.perf_counter() - t0, 1)
    print(f"container runtime built in {_RT['init_seconds']}s", flush=True)
    return _RT


@app.function(
    image=image, cpu=(1.0, 1.0), memory=4096, timeout=60 * 60,
    max_containers=80, retries=3,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_source(task: dict[str, Any]) -> dict[str, Any]:
    """One source molecule. Returns a small record; the driver aggregates."""
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen

    RDLogger.DisableLog("rdApp.*")
    rt = _runtime()
    model = rt["model"]

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.pareto_control import _argmin_stable
    from compose_v4.experiments.production_successor_kernel import (
        canonical_successor_result,
    )

    t0 = time.perf_counter()
    src = task["smiles"]
    mol0 = Chem.MolFromSmiles(src)
    if mol0 is None:
        return {"index": task["index"], "source": src, "status": "SOURCE_UNPARSEABLE"}
    tau = float(Crippen.MolLogP(mol0)) + LOGP_SHIFT

    cur, traj, dead, kernel_calls = src, [src], None, 0
    logp_cache: dict[str, float] = {}
    for _ in range(HORIZON):
        try:
            state = pad_molecular_graph(smiles_to_molecular_graph(cur), CANONICAL_SLOTS)
            res = canonical_successor_result(model, state, float(TIME_POINT))
            kernel_calls += 1
        except Exception as e:  # noqa: BLE001
            dead = f"kernel:{type(e).__name__}"
            break
        keys = [s.key for s in res.batch.successors]
        if not keys:
            dead = "empty_fiber"
            break
        # The ONLY per-candidate work in the loop: the controller's objective.
        scores = []
        for k in keys:
            v = logp_cache.get(k)
            if v is None:
                m = Chem.MolFromSmiles(k)
                v = abs(float(Crippen.MolLogP(m)) - tau) if m is not None else 1e9
                logp_cache[k] = v
            scores.append(v)
        cur = keys[_argmin_stable(np.asarray(scores, float), keys)]
        traj.append(cur)

    return {"index": task["index"], "source": src, "endpoint": cur,
            "trajectory": traj, "tau": tau, "edits": len(traj) - 1,
            "dead_end": dead, "kernel_calls": kernel_calls,
            "candidates_scored": len(logp_cache),
            "seconds": round(time.perf_counter() - t0, 2),
            "status": "OK"}


def resume_from_partial(
    tasks: list[dict[str, Any]], partial: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split `tasks` into (already-done results, still-pending tasks).

    Pure and importable so it can be tested without Modal. Returns recovered
    results first so the caller can seed its accumulator.

    A torn or truncated partial is treated as ABSENT, not as corruption to
    propagate: gzip/JSON failures fall back to a full run. Losing a checkpoint
    costs time; trusting a torn one would corrupt the benchmark payload.

    Only records carrying an integer `index` count as complete, so a partial
    written mid-append cannot mask a source as done.
    """
    if not partial.exists():
        return [], list(tasks)
    try:
        blob = json.loads(gzip.decompress(partial.read_bytes()).decode())
        recovered = [r for r in blob.get("results", [])
                     if isinstance(r, dict) and isinstance(r.get("index"), int)]
    except Exception as exc:  # noqa: BLE001
        print(f"partial unreadable ({type(exc).__name__}); starting fresh", flush=True)
        return [], list(tasks)
    done = {int(r["index"]) for r in recovered}
    # Deduplicate by index in case a partial was written twice.
    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    for r in recovered:
        if int(r["index"]) not in seen:
            seen.add(int(r["index"]))
            unique.append(r)
    pending = [t for t in tasks if int(t["index"]) not in done]
    return unique, pending


@app.function(image=image, cpu=(1.0, 1.0), memory=4096, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]], out_name: str) -> dict[str, Any]:
    """Aggregate mapped results, RESUMING from a partial rather than restarting.

    The previous version was WRITE-ONLY: it committed a partial every 250
    results but nothing ever read it back, so a driver loss preserved the data
    and threw away the work. That is the same defect that cost 76 minutes on
    pathwise Stage B. Resume is now real -- completed source indices are read
    from the partial and removed from the work list, so a relaunch continues
    instead of recomputing.
    """
    artifact_volume.reload()
    out = Path(RUN_ROOT) / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    path, partial = out / f"{out_name}.json.gz", out / f"{out_name}.partial.json.gz"
    started = time.perf_counter()

    results, pending = resume_from_partial(tasks, partial)
    if len(results):
        print(f"RESUMED {len(results)} completed sources from {partial.name}; "
              f"{len(pending)} remain", flush=True)
    if not pending:
        print("nothing left to compute; finalizing from the partial", flush=True)

    #: Checkpoint on TOTAL completed, so the cadence does not restart on resume.
    last_ckpt = len(results)
    for r in run_source.map(pending, order_outputs=False, return_exceptions=True):
        if isinstance(r, dict):
            results.append(r)
        if len(results) - last_ckpt >= CHECKPOINT_EVERY:
            last_ckpt = len(results)
            partial.write_bytes(gzip.compress(json.dumps(
                {"results": results}, default=float).encode()))
            artifact_volume.commit()
            print(f"{len(results)}/{len(tasks)} {time.perf_counter()-started:.0f}s",
                  flush=True)
    payload = {
        "schema": "compose.ddsbm.endpoint_competence",
        "status": "EXTERNAL_BENCHMARK_TASK_ALIGNED_METRICS",
        "not_a_reproduction_of": ("DDSBM Table 1 -- FCD and NSPDK are "
                                  "deliberately omitted as distribution-model "
                                  "diagnostics, not task-aligned metrics"),
        "protocol": "docs/DDSBM_ENDPOINT_COMPETENCE_PROTOCOL.md",
        "objective": "u(y;x0) = -|logP(y) - (logP(x0) + 2)|",
        "horizon": HORIZON, "controller": "greedy_closed_loop",
        "r_theta": "frozen; no objective-specific update",
        "paired_target_used": False,
        "n": len(results),
        "resumed": len(tasks) != len(results) and partial.exists(),
        "seconds": round(time.perf_counter() - started, 1),
        "results": results,
    }
    path.write_bytes(gzip.compress(json.dumps(payload, default=float).encode()))
    partial.unlink(missing_ok=True)
    artifact_volume.commit()
    print(f"DONE {len(results)} in {payload['seconds']:.0f}s", flush=True)
    return {"n": len(results), "seconds": payload["seconds"]}


def _test_sources() -> list[dict[str, Any]]:
    rows = list(csv.DictReader(open(ROOT / DDSBM_CSV)))
    test = rows[TRAIN_ROWS:]
    assert len(test) == 5984, f"expected DDSBM's 5,984 test sources, got {len(test)}"
    return [{"index": i, "smiles": r["REF-SMI"]} for i, r in enumerate(test)]


@app.local_entrypoint()
def main(pilot: int = 0) -> None:
    """`--pilot N` runs N sources for TIMING ONLY. No metric is inspected."""
    tasks = _test_sources()
    if pilot:
        tasks = tasks[:pilot]
        print(f"RUNTIME-ONLY PILOT: {pilot} sources, one core per worker.")
        print("Verifying per-expansion latency at cpu=1.0. No outcome inspected.")
        out_name = f"pilot_{pilot:04d}"
    else:
        print(f"FULL RUN: {len(tasks)} DDSBM test sources")
        out_name = "full"
    print(f"objective u = -|logP(y) - (logP(x0)+2)|  H={HORIZON}  greedy  frozen R_theta")
    print("cpu=(1.0,1.0), max_containers=80, one source per work item")
    call = drive.spawn(tasks, out_name)
    print(f"driver spawned: {call.object_id}")
