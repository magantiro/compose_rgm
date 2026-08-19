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
app = modal.App("molleo-basin-substrate")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
OUT_DIR = "molleo_basin_v1"
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





#: Unguided outward walk cap. No rejection on JNK3: rejecting steps that keep
#: JNK3 high would build the substrate out of the very signal the search is
#: supposed to lack, and would manufacture an easy problem.
WALK_CAP = 12
FLAT = 0.10
ACTIVE = 0.50


@app.function(image=image, cpu=(2.0, 2.0), memory=MEM_MIB, timeout=4 * 60 * 60,
              max_containers=64, retries=1,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def walk(task: dict[str, Any]) -> dict[str, Any]:
    """Anchor -> unguided R_theta walk -> flat start, with the REVERSE verified.

    Generating x_active -> ... -> x_flat proves that direction is executable. It
    does NOT prove x_flat -> ... -> x_active is: COMPOSE's legal edits are not
    guaranteed symmetric, and Gate 1's whole premise is that the active basin is
    reachable FROM the flat start. So every reverse step is checked explicitly by
    enumerating the legal fiber at the later state and executing marks until the
    earlier state is produced. A substrate whose reverse is not fully legal is
    discarded rather than quietly used.
    """
    import sys, time
    import numpy as np

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.benchmark.oracles.task3 import Task3Objectives
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.experiments.hphi_lazy_helpers import make_helpers
    from compose_v4.experiments.hphi_lazy_sampler import sample_one_transition
    from compose_v4.experiments.production_successor_kernel import (
        _coordinate_action, canonical_state_key, enumerate_factorized_marked_law,
    )

    rt = _runtime()
    model, system = rt["model"], rt["system"]
    helpers = make_helpers(model, time_point=float(TIME_POINT),
                           canonical_slots=CANONICAL_SLOTS)
    _TF = {"grow_connected": "atom_insert"}
    obj = Task3Objectives()
    rng = np.random.default_rng(int(task["seed"]))
    t0 = time.perf_counter()

    def st_of(smi):
        return pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)

    def step(smi):
        st = st_of(smi)
        d = sample_one_transition(model, st, float(TIME_POINT), rng,
                                  helpers=helpers)
        if d.table is None or d.coordinate is None:
            return ""
        fam = _TF.get(d.table, d.table)
        batch = helpers["build_batch"](st, float(TIME_POINT))
        helpers["family_mask"](model, d.table, st, batch, None, None, None)
        rule, action = _coordinate_action(model, st, batch, family_name=fam,
                                          table_name=d.table,
                                          coordinate=d.coordinate)
        y = canonical_state_key(system.apply(st, rule, action))
        return "" if y == smi else y

    def reverse_legal(frm: str, to: str) -> bool:
        """Is `to` a legal COMPOSE successor of `frm`? Exact, by construction."""
        st = st_of(frm)
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        batch = helpers["build_batch"](st, float(TIME_POINT))
        for m in law.marks:
            try:
                helpers["family_mask"](model, m.table_name, st, batch,
                                       None, None, None)
                rule, action = _coordinate_action(
                    model, st, batch, family_name=m.family_name,
                    table_name=m.table_name, coordinate=tuple(m.coordinate))
                if canonical_state_key(system.apply(st, rule, action)) == to:
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    anchor = task["anchor"]
    path = [anchor]
    j = [float(obj(anchor)[1])]
    for _ in range(WALK_CAP):
        y = step(path[-1])
        if not y:
            break
        path.append(y)
        j.append(float(obj(y)[1]))
        if j[-1] <= FLAT:
            break

    flat_idx = next((i for i, v in enumerate(j) if v <= FLAT), None)
    out = {"anchor": anchor, "seed": int(task["seed"]), "path": path,
           "jnk3": j, "flat_index": flat_idx,
           "reached_flat": flat_idx is not None,
           "seconds": round(time.perf_counter() - t0, 1)}
    if flat_idx is None or flat_idx == 0:
        out["reverse_ok"] = False
        out["reverse_verified_distance"] = None
        return out

    # Verify flat -> ... -> anchor, step by step, in the direction we need.
    ok = True
    checked = []
    for i in range(flat_idx, 0, -1):
        good = reverse_legal(path[i], path[i - 1])
        checked.append(bool(good))
        if not good:
            ok = False
            break
    out["reverse_steps_ok"] = checked
    out["reverse_ok"] = bool(ok)
    out["reverse_verified_distance"] = flat_idx if ok else None

    # ONE-STEP QUALIFICATION. A verified d-step route proves a path exists; it
    # does NOT prove there is no single edit from the flat start straight into
    # an active. If one exists, local search wins trivially and the substrate
    # cannot separate local from non-local search. So enumerate the EXACT legal
    # fiber at the flat start, execute every mark, and score JNK3 on the unique
    # successors. Substrate qualification only -- these calls buy no search.
    if ok:
        flat = path[flat_idx]
        st = st_of(flat)
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        batch = helpers["build_batch"](st, float(TIME_POINT))
        succ: set = set()
        for m in law.marks:
            try:
                helpers["family_mask"](model, m.table_name, st, batch,
                                       None, None, None)
                rule, action = _coordinate_action(
                    model, st, batch, family_name=m.family_name,
                    table_name=m.table_name, coordinate=tuple(m.coordinate))
                y = canonical_state_key(system.apply(st, rule, action))
                if y != flat:
                    succ.add(y)
            except Exception:  # noqa: BLE001
                continue
        js = []
        for y in succ:
            try:
                js.append(float(obj(y)[1]))
            except Exception:  # noqa: BLE001
                continue
        out["fiber_size"] = len(law.marks)
        out["n_unique_successors"] = len(succ)
        out["onestep_jnk3_max"] = (max(js) if js else None)
        out["onestep_jnk3_mean"] = (float(np.mean(js)) if js else None)
        out["onestep_jnk3_p95"] = (float(np.percentile(js, 95)) if js else None)
        out["onestep_jnk3_n_ge_0.5"] = int(sum(1 for v in js if v >= ACTIVE))
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


@app.function(image=image, cpu=(0.25, 0.25), memory=1024, timeout=6 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]], out_name: str) -> dict[str, Any]:
    import time
    out, started = [], time.perf_counter()
    for r in walk.map(tasks, order_outputs=False, return_exceptions=True,
                      wrap_returned_exceptions=False):
        if isinstance(r, dict):
            out.append(r)
            print(f"  anchor {r['anchor'][:20]} flat@{r['flat_index']} "
                  f"revOK={r['reverse_ok']} d={r['reverse_verified_distance']} "
                  f"1stepMax={r.get('onestep_jnk3_max')} "
                  f"{r['seconds']:.0f}s  [{len(out)}/{len(tasks)}]", flush=True)
        else:
            print(f"  FAILED {type(r).__name__}: {r}", flush=True)
    p = Path(RUN_ROOT) / "molleo_basin" / f"{out_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"results": out}))
    artifact_volume.commit()
    print(f"DONE {len(out)}/{len(tasks)} in {time.perf_counter()-started:.0f}s")
    return {"n": len(out)}


@app.local_entrypoint()
def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    anc = json.loads((root_dir / "docs/MOLLEO_BASIN_ANCHORS.json").read_text())
    tasks = [{"anchor": s, "seed": 700000 + i}
             for i, (s, _v) in enumerate(anc["anchors"])]
    print(f"BASIN SUBSTRATE: {len(tasks)} held-out JNK3 anchors, unguided "
          f"R_theta walks capped at {WALK_CAP}, flat threshold {FLAT}.")
    print("Every retained substrate has its REVERSE path verified legal "
          "step by step; the forward walk alone does not establish it.")
    call = drive.spawn(tasks, "walks_96")
    print(f"spawned: {call.object_id}")
