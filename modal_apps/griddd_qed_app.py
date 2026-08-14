"""GrIDDD / Jin ZINC-250k QED constrained editing. COMPOSE runs alone.

Protocol frozen in `docs/GRIDDD_JIN_PROTOCOL.md` at commit fa130ee, BEFORE any
benchmark outcome existed. Nothing here may be tuned against GrIDDD's table.

THE POLICY -- stochastic one-step COMPOSE control
--------------------------------------------------
    pi(y|x) = R_theta(y|x) * QED(y) / sum_z R_theta(z|x) * QED(z)     z in F(x)

applied receding at each of 6 committed edits, over the COMPLETE legal fiber.
It is an EXACT one-step terminal tilt (h_0 = QED), NOT the exact H=6 Doob
controller, and must not be described as such. Parameter-free: no temperature,
no beta, no beam width, no shortlist.

The rollout h over the full fiber was measured at 145,440,000 kernel calls,
$24,241 and 0.7 years, and the 4/2/2 shortlist variant was rejected on
scientific grounds -- see the protocol.

FROZEN SAMPLING RULES
---------------------
20 independent rollouts, each reset to the original source; terminal endpoints
only; sha256 seed manifest, never Python's process-salted hash(); duplicates
consume attempts; similarity is EVALUATION-ONLY and never enters pi; no
success-triggered early stopping; failures stay in the denominator; and if the
normalizer is exactly zero, fall back to R_theta itself -- preregistered, not
invented after it happened.

THE ABLATION uses IDENTICAL seeds and policy; the only intervention is removing
successors that change heavy-atom count, BEFORE normalization.

MEMOIZATION is the only optimization taken. Fibers are cached by the complete
kernel input (canonical key + time point), so the 20 replicates share the source
fiber instead of recomputing it 20 times. Logical requests and true kernel calls
are recorded separately. See `docs/KERNEL_COST_CHARACTERIZATION.md` for why
nothing else is being optimized: the network is 1% of runtime, batching does not
amortize, and direct mark sampling is a 2.7x general improvement rather than an
enabler for this run.
"""

from __future__ import annotations

import gzip
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

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"}
).add_local_file(ROOT / "data/jin/qed_test.txt",
                 str(REMOTE_ROOT / "qed_test.txt"), copy=True)

app = modal.App("griddd-qed")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
OUT_DIR = "griddd_qed"
TIME_POINT = 0.5
CANONICAL_SLOTS = 48
HORIZON = 6
N_REPLICATES = 20
PROTOCOL_VERSION = "griddd-jin-qed-v1"

_RT: dict[str, Any] = {}


def replicate_seed(task: str, canonical_source: str, replicate: int) -> int:
    """Frozen seed manifest. NEVER Python's process-salted hash()."""
    payload = f"{PROTOCOL_VERSION}|{task}|{canonical_source}|{replicate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _runtime():
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
    src = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]), repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        src, materialized_state=bundle)
    model = runtime.model
    ckpt = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["selected_model_state"], strict=True)
    model.eval()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    _RT["model"] = model
    print(f"container runtime built in {time.perf_counter()-t0:.1f}s", flush=True)
    return _RT


@app.function(image=image, cpu=(1.0, 1.0), memory=6144, timeout=6 * 60 * 60,
              max_containers=80, retries=3,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_source(task: dict[str, Any]) -> dict[str, Any]:
    """20 independent trajectories from one source, under the frozen policy."""
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED

    RDLogger.DisableLog("rdApp.*")
    model = _runtime()["model"]

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.production_successor_kernel import (
        canonical_successor_result,
    )

    t0 = time.perf_counter()
    source = task["smiles"]
    size_fixed = bool(task.get("size_fixed", False))
    mol0 = Chem.MolFromSmiles(source)
    if mol0 is None:
        return {"index": task["index"], "source": source,
                "status": "SOURCE_UNPARSEABLE"}
    canonical_source = Chem.MolToSmiles(mol0)
    n_heavy_source = mol0.GetNumHeavyAtoms()

    # --- exact memoization, the only optimization taken -------------------
    fiber_cache: dict[str, list[tuple[str, float]]] = {}
    qed_cache: dict[str, float] = {}
    logical_requests = [0]
    kernel_calls = [0]

    def fiber(key: str) -> list[tuple[str, float]]:
        logical_requests[0] += 1
        hit = fiber_cache.get(key)
        if hit is not None:
            return hit
        state = pad_molecular_graph(smiles_to_molecular_graph(key), CANONICAL_SLOTS)
        res = canonical_successor_result(model, state, float(TIME_POINT))
        kernel_calls[0] += 1
        rows = [(s.key, float(s.probability)) for s in res.batch.successors]
        fiber_cache[key] = rows
        return rows

    def qed_of(key: str) -> float:
        v = qed_cache.get(key)
        if v is None:
            m = Chem.MolFromSmiles(key)
            v = float(QED.qed(m)) if m is not None else 0.0
            qed_cache[key] = v
        return v

    candidates: list[str] = []
    per_replicate: list[dict[str, Any]] = []
    zero_denominator_events = 0

    for rep in range(N_REPLICATES):
        rng = np.random.default_rng(replicate_seed("qed", canonical_source, rep))
        cur, dead = source, None
        for _ in range(HORIZON):
            try:
                rows = fiber(cur)
            except Exception as exc:  # noqa: BLE001
                dead = f"kernel:{type(exc).__name__}"
                break
            if size_fixed:
                # ABLATION: drop size-changing successors BEFORE normalization.
                rows = [
                    r for r in rows
                    if (lambda m: m is not None
                        and m.GetNumHeavyAtoms() == n_heavy_source)(
                        Chem.MolFromSmiles(r[0]))
                ]
            if not rows:
                dead = "empty_fiber"
                break
            keys = [r[0] for r in rows]
            w = np.array([r[1] * qed_of(r[0]) for r in rows], dtype=float)
            total = float(w.sum())
            if not np.isfinite(total) or total <= 0.0:
                # PREREGISTERED fallback: R_theta itself. Not a constant.
                zero_denominator_events += 1
                w = np.array([r[1] for r in rows], dtype=float)
                total = float(w.sum())
                if total <= 0.0:
                    dead = "zero_reference_mass"
                    break
            cur = keys[int(rng.choice(len(keys), p=w / total))]
        candidates.append(cur)                      # TERMINAL endpoint only
        per_replicate.append({"replicate": rep, "endpoint": cur, "dead_end": dead})

    return {
        "index": task["index"], "source": source,
        "canonical_source": canonical_source,
        "size_fixed": size_fixed,
        "candidates": candidates,                   # duplicates CONSUME attempts
        "n_candidates": len(candidates),
        "n_unique_candidates": len(set(candidates)),
        "per_replicate": per_replicate,
        "zero_denominator_events": zero_denominator_events,
        "logical_fiber_requests": logical_requests[0],
        "kernel_calls": kernel_calls[0],            # true work, after memoization
        "seconds": round(time.perf_counter() - t0, 2),
        "status": "OK",
    }


def resume_from_partial(tasks, partial: Path):
    """Same lossless-resume contract as the DDSBM driver."""
    if not partial.exists():
        return [], list(tasks)
    try:
        blob = json.loads(gzip.decompress(partial.read_bytes()).decode())
        rec = [r for r in blob.get("results", [])
               if isinstance(r, dict) and isinstance(r.get("index"), int)]
    except Exception as exc:  # noqa: BLE001
        print(f"partial unreadable ({type(exc).__name__}); starting fresh", flush=True)
        return [], list(tasks)
    seen, uniq = set(), []
    for r in rec:
        if int(r["index"]) not in seen:
            seen.add(int(r["index"]))
            uniq.append(r)
    return uniq, [t for t in tasks if int(t["index"]) not in seen]


@app.function(image=image, cpu=(1.0, 1.0), memory=4096, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]], out_name: str) -> dict[str, Any]:
    artifact_volume.reload()
    out = Path(RUN_ROOT) / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    path, partial = out / f"{out_name}.json.gz", out / f"{out_name}.partial.json.gz"
    started = time.perf_counter()
    results, pending = resume_from_partial(tasks, partial)
    if results:
        print(f"RESUMED {len(results)}; {len(pending)} remain", flush=True)
    last = len(results)
    for r in run_source.map(pending, order_outputs=False, return_exceptions=True):
        if isinstance(r, dict):
            results.append(r)
        if len(results) - last >= 50:
            last = len(results)
            partial.write_bytes(gzip.compress(json.dumps(
                {"results": results}, default=float).encode()))
            artifact_volume.commit()
            print(f"{len(results)}/{len(tasks)} "
                  f"{time.perf_counter()-started:.0f}s", flush=True)
    payload = {
        "schema": "compose.griddd.qed",
        "protocol": "docs/GRIDDD_JIN_PROTOCOL.md (frozen fa130ee)",
        "protocol_version": PROTOCOL_VERSION,
        "policy": "pi(y|x) = R_theta(y|x)*QED(y) / sum_z R_theta(z|x)*QED(z)",
        "policy_name": "stochastic one-step COMPOSE control",
        "not_": "NOT the exact H=6 Doob controller",
        "horizon": HORIZON, "replicates": N_REPLICATES,
        "r_theta": "frozen; no objective-specific update",
        "n": len(results),
        "seconds": round(time.perf_counter() - started, 1),
        "results": results,
    }
    path.write_bytes(gzip.compress(json.dumps(payload, default=float).encode()))
    partial.unlink(missing_ok=True)
    artifact_volume.commit()
    print(f"DONE {len(results)} in {payload['seconds']:.0f}s", flush=True)
    return {"n": len(results), "seconds": payload["seconds"]}


@app.local_entrypoint()
def main(limit: int = 0, size_fixed: bool = False) -> None:
    smis = [l.strip() for l in
            (ROOT / "data/jin/qed_test.txt").read_text().splitlines() if l.strip()]
    assert len(smis) == 800, f"expected 800 Jin QED sources, got {len(smis)}"
    if limit:
        smis = smis[:limit]
    tasks = [{"index": i, "smiles": s, "size_fixed": size_fixed}
             for i, s in enumerate(smis)]
    arm = "size_fixed" if size_fixed else "full"
    out_name = f"{arm}_{len(tasks):04d}"
    print(f"GrIDDD/Jin QED -- {len(tasks)} sources x {N_REPLICATES} replicates, arm={arm}")
    print("policy: pi ∝ R_theta(y|x)*QED(y) over the COMPLETE fiber, receding, H=6")
    print("frozen sha256 seeds; terminal endpoints only; duplicates consume attempts")
    call = drive.spawn(tasks, out_name)
    print(f"driver spawned: {call.object_id}")
