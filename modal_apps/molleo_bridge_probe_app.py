"""Persistent archive vs independent restarts. A/B on one frozen dev panel.

Tests one hypothesis: does remembering a promising valid intermediate beat
forgetting everything and restarting from x0?

    RESTART   every candidate from x0, as today
    RANDOM    branch from a UNIFORMLY chosen reusable archived state
    HPHI      branch from the reusable archived state maximising h_phi(x, H-d)

Same R_theta, same h_phi, same legal kernel, same N=32, same region, same
number of returned candidates.

RANDOM is the control that decides WHICH claim the v1 result supports. If
hphi > random > restart, then remembering intermediates helps AND h_phi
identifies which ones are worth remembering. If random is level with hphi, the
finding is the simpler "stop restarting from x0", which is still useful but a
different mechanism.

x0 IS ITSELF AN ARCHIVE STATE. A persistent archive means never FORGET, not
never restart: if x0 remains the best launch point the policy should choose it.
v1 forced a branch every candidate and lost a reliable source (4 -> 3) for
exactly that reason.

FROZEN BEFORE ANY DATA IS READ
------------------------------
BRANCH RULE. Among archived states, branch from the one maximising
h_phi(x, 24 - d). Nothing else -- no new learned model, no QED peeking beyond
the target definition, no tree heuristic. If the simple version works, a better
one can be designed afterwards; if it does not, a clever one would only be
harder to interpret.

The score needs no new computation. Every transition already records
h_y_bm1 = h_phi(y, b-1), and b-1 IS 24 - depth, so the rule reads off the
previous candidate's record directly.

DEPTH-RESPECTING BUDGET. A branch from depth d runs with H - d remaining, never
a fresh 24. This is correctness before it is fairness: h_phi is
budget-conditioned and was trained to estimate reachability within b REMAINING
steps, so a fresh horizon queries it outside its training semantics AND gives
the archive a deeper edit path from x0 than the restarts get -- a difference no
compute log would expose.

It also means the archive arm is budget-DISADVANTAGED: a branch at depth d costs
32 * (24 - d) particle-steps against a restart's 32 * 24. A win therefore comes
with less search work, which is the stronger form of the result. Work is logged
either way rather than assumed.

REPORTED BY STRATUM, never pooled. A gain on sources that already succeed is
close to worthless, and pooling would let the reliable four flatter a null
elsewhere.
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

# The frozen oracle bundle travels WITH the image (4.6 MB), so a container can
# never silently fall back to some other oracle -- Task3Objectives raises if the
# bundle is absent, and this is what satisfies it.
# Both oracle bundles travel with the image: the Task-3 forests/SA table and
# the separately-held DRD2 RBF-SVM. 5 MB total, and it means a container cannot
# silently fall back to a different oracle -- each loader raises if its bundle
# is missing, which is how the second one was found.
image = _base_image.add_local_dir(
    ROOT / "artifacts/oracles",
    str(REMOTE_ROOT / "artifacts/oracles"), copy=True,
).env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
app = modal.App("molleo-bridge-probe")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
OUT_DIR = "molleo_bridge_v1"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
HORIZON = 12
N_PARTICLES = 32
REGION = (0.90, 0.40)
N_CANDIDATES = 4
PROTOCOL = "hphi-horizon-v4"
MEM_MIB = int(4.5 * 1024)

_RT: dict[str, Any] = {}


def seed_for(arm: str, src: str, k: int) -> int:
    """Paired across arms: candidate 1 is IDENTICAL in both, by construction.

    The arm name enters only for candidates 2+, so any difference at candidate 1
    would be a bug rather than a finding.
    """
    tag = f"{PROTOCOL}|{src}|{k}" if k == 0 else f"{PROTOCOL}|{arm}|{src}|{k}"
    return int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "big")


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
    from compose_v4.experiments.production_successor_kernel import (
        _default_rewrite_system,
    )

    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    src = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]), repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        src, materialized_state=bundle)
    model = runtime.model
    ck = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["selected_model_state"], strict=True)
    model.eval(); torch.set_grad_enabled(False); torch.set_num_threads(1)
    _RT.update({"model": model, "system": _default_rewrite_system(model)})
    import gc

    del ck, bundle, src, runtime, _b, _c
    gc.collect()
    return _RT


def _record_name(task: dict) -> str:
    """ONE definition of the record filename.

    The two arms differ only in the head, so a name without it makes them
    collide: the second arm silently overwrites the first and the A/B compares
    a run against itself. The resume check and the write MUST use this same
    name, or resume looks for a file the writer never creates.
    """
    base = (f"{int(task['index']):03d}_H{task.get('horizon', HORIZON)}"
            f"_{task.get('head_dir', 'hphi_v2').replace('/', '-')}")
    if "k_start" in task or "k_end" in task:
        base += f"_k{int(task.get('k_start', 0))}-{int(task.get('k_end', 0))}"
    return base + ".json"


def _load_head(head_dir: str):
    """Per-task head. PROTOCOL and seed_for are untouched, so a run with
    head_dir='hphi_v2' MUST reproduce the banked recede_v5 record bit for bit;
    that is the parity gate for this whole comparison."""
    import json as _json

    import torch as _torch

    key = f"head::{head_dir}"
    if key not in _RT:
        h = _torch.jit.load(str(Path(RUN_ROOT) / head_dir / "head.pt"),
                            map_location="cpu")
        h.eval()
        n = _json.loads((Path(RUN_ROOT) / head_dir / "norm.json").read_text())
        _RT[key] = (h, n["mu"], n["sd"])
    return _RT[key]







ACTIVE = 0.50
BEAM = 4


@app.function(image=image, cpu=(2.0, 2.0), memory=MEM_MIB, timeout=6 * 60 * 60,
              max_containers=32, retries=1,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def bridge(task: dict[str, Any]) -> dict[str, Any]:
    """Target-AWARE planning from random ZINC toward a known active.

    This hands the planner information MOLLEO never provides: the identity of a
    held-out JNK3 active, and a dense similarity signal to it. It is deliberately
    an easier problem than the benchmark. The point is a bound:

        reaches the basin with the target supplied -> the basin is reachable and
            blind DISCOVERY is what fails;
        fails even with the target supplied       -> the active region is remote
            under COMPOSE's edit graph, and Task 3 from random-120 is out of
            reach for this kernel.

    Search is exact-fiber beam search on Tanimoto-to-target, so the only thing
    limiting it is the edit graph itself, not the proposal prior -- the fiber
    census already showed R_theta is not the bottleneck.
    """
    import sys, time
    import numpy as np

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.benchmark.oracles.task3 import Task3Objectives
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.hphi_lazy_helpers import make_helpers
    from compose_v4.experiments.production_successor_kernel import (
        _coordinate_action, canonical_state_key, enumerate_factorized_marked_law,
    )

    rt = _runtime()
    model, system = rt["model"], rt["system"]
    helpers = make_helpers(model, time_point=float(TIME_POINT),
                           canonical_slots=CANONICAL_SLOTS)
    obj = Task3Objectives()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    target = task["target"]
    tfp = gen.GetFingerprint(Chem.MolFromSmiles(target))
    horizon = int(task.get("horizon", 40))

    def sim(smi):
        m = Chem.MolFromSmiles(smi)
        return 0.0 if m is None else float(
            DataStructs.TanimotoSimilarity(tfp, gen.GetFingerprint(m)))

    def successors(x):
        st = pad_molecular_graph(smiles_to_molecular_graph(x), CANONICAL_SLOTS)
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        batch = helpers["build_batch"](st, float(TIME_POINT))
        out = set()
        for m in law.marks:
            try:
                helpers["family_mask"](model, m.table_name, st, batch,
                                       None, None, None)
                rule, action = _coordinate_action(
                    model, st, batch, family_name=m.family_name,
                    table_name=m.table_name, coordinate=tuple(m.coordinate))
                y = canonical_state_key(system.apply(st, rule, action))
                if y != x:
                    out.add(y)
            except Exception:  # noqa: BLE001
                continue
        return out

    t0 = time.perf_counter()
    beam = [task["start"]]
    seen = {task["start"]}
    trace = []
    best_jnk3 = float(obj(task["start"])[1])
    reached = False
    for depth in range(1, horizon + 1):
        cand: set = set()
        for x in beam:
            cand |= successors(x)
        cand -= seen
        if not cand:
            break
        scored = sorted(((sim(y), y) for y in cand), reverse=True)
        beam = [y for _s, y in scored[:BEAM]]
        seen |= set(beam)
        # JNK3 only on the frontier we keep plus the top few by similarity --
        # scoring every successor would cost more than the search.
        probe = [y for _s, y in scored[:8]]
        js = []
        for y in probe:
            try:
                js.append(float(obj(y)[1]))
            except Exception:  # noqa: BLE001
                pass
        if js:
            best_jnk3 = max(best_jnk3, max(js))
        row = {"depth": depth, "best_sim": float(scored[0][0]),
               "n_candidates": len(cand), "best_jnk3_seen": best_jnk3,
               "jnk3_of_best_sim": (js[0] if js else None)}
        trace.append(row)
        if best_jnk3 >= ACTIVE or scored[0][0] >= 0.999:
            reached = True
            break
    return {"start": task["start"], "target": target,
            "sim0": float(task["sim0"]), "target_jnk3": float(task["target_jnk3"]),
            "horizon": horizon, "reached": reached,
            "best_sim_final": trace[-1]["best_sim"] if trace else None,
            "best_jnk3": best_jnk3, "depths_run": len(trace), "trace": trace,
            "seconds": round(time.perf_counter() - t0, 1)}


@app.function(image=image, cpu=(0.25, 0.25), memory=1024, timeout=8 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]], out_name: str) -> dict[str, Any]:
    import time
    out, started = [], time.perf_counter()
    for r in bridge.map(tasks, order_outputs=False, return_exceptions=True,
                        wrap_returned_exceptions=False):
        if isinstance(r, dict):
            out.append(r)
            print(f"  sim {r['sim0']:.3f}->{r['best_sim_final']:.3f} "
                  f"bestJNK3 {r['best_jnk3']:.3f} reached={r['reached']} "
                  f"d={r['depths_run']} {r['seconds']:.0f}s "
                  f"[{len(out)}/{len(tasks)}]", flush=True)
        else:
            print(f"  FAILED {type(r).__name__}: {r}", flush=True)
    p = Path(RUN_ROOT) / "molleo_bridge" / f"{out_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"results": out}))
    artifact_volume.commit()
    print(f"DONE {len(out)}/{len(tasks)} in {time.perf_counter()-started:.0f}s")
    return {"n": len(out)}


@app.local_entrypoint()
def main(horizon: int = 40) -> None:
    root_dir = Path(__file__).resolve().parents[1]
    P = json.loads((root_dir / "docs/MOLLEO_BRIDGE_PAIRS.json").read_text())
    tasks = [dict(p, horizon=horizon) for p in P["pairs"]]
    print(f"BRIDGE PROBE: {len(tasks)} random-ZINC starts, each given its MOST "
          f"SIMILAR held-out active as an explicit target, H={horizon}, "
          f"exact-fiber beam search (width {BEAM}) on Tanimoto-to-target.")
    print("Strictly easier than MOLLEO: the target identity is supplied and the "
          "similarity signal is dense. A failure here bounds the benchmark.")
    call = drive.spawn(tasks, f"bridge_{len(tasks):02d}_H{horizon}")
    print(f"spawned: {call.object_id}")
