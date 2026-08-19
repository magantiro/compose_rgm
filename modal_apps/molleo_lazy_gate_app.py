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
app = modal.App("molleo-lazy-gate")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
OUT_DIR = "molleo_lazy_v1"
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




#: Verification cadences. L1 measures every depth; L4 every fourth. Nothing
#: else differs between the arms.
CADENCE = {"L1": 1, "L4": 4}
#: Reward-difference potential strength. The twist is the preference-scalarised
#: improvement, matching the earlier Pareto work's value_twist=None setting --
#: no learned five-objective value function exists yet, and inventing one here
#: would confound the cadence comparison this gate is for.
BETA = 10.0


@app.function(image=image, cpu=(2.0, 2.0), memory=MEM_MIB, timeout=6 * 60 * 60,
              max_containers=48, retries=1,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_root(task: dict[str, Any]) -> dict[str, Any]:
    """One root, one arm. True oracle ONLY on realized particles at checkpoints."""
    import sys, time
    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.benchmark.oracles.task3 import Task3Objectives
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.hphi_lazy_helpers import make_helpers
    from compose_v4.experiments.hphi_lazy_sampler import sample_one_transition
    from compose_v4.experiments.molleo_surrogate import TanimotoKNN
    from compose_v4.experiments.production_successor_kernel import (
        _coordinate_action, canonical_state_key,
    )

    arm = task["arm"]
    every = CADENCE[arm]
    horizon = int(task.get("horizon", HORIZON))
    w = np.asarray(task["preference"], dtype=float)
    rt = _runtime()
    model, system = rt["model"], rt["system"]
    helpers = make_helpers(model, time_point=float(TIME_POINT),
                           canonical_slots=CANONICAL_SLOTS)
    _TABLE_FAMILY = {"grow_connected": "atom_insert"}
    obj = Task3Objectives()

    sur = TanimotoKNN(5)
    for smi, vec in task["labels"]:
        sur.fit_add(smi, vec)
    truth: dict[str, tuple] = {smi: tuple(v) for smi, v in task["labels"]}

    _st: dict[str, Any] = {}

    def state_of(smi):
        if smi not in _st:
            _st[smi] = pad_molecular_graph(smiles_to_molecular_graph(smi),
                                           CANONICAL_SLOTS)
        return _st[smi]

    def propose(smi, rng):
        st = state_of(smi)
        d = sample_one_transition(model, st, float(TIME_POINT), rng,
                                  helpers=helpers)
        if d.table is None or d.coordinate is None:
            return ""
        fam = _TABLE_FAMILY.get(d.table, d.table)
        batch = helpers["build_batch"](st, float(TIME_POINT))
        helpers["family_mask"](model, d.table, st, batch, None, None, None)
        rule, action = _coordinate_action(model, st, batch, family_name=fam,
                                          table_name=d.table,
                                          coordinate=d.coordinate)
        y = canonical_state_key(system.apply(st, rule, action))
        return "" if y == smi else y

    def value(vecs):
        return np.asarray(vecs, dtype=float) @ w

    rng = np.random.default_rng(int(task["seed"]))
    root = task["root"]
    states = [root] * N_PARTICLES
    log_w = np.zeros(N_PARTICLES)
    # Speculative values carry the surrogate's belief; corrected at checkpoints.
    cur = np.repeat(value([truth.get(root, sur.predict([root])[0])]),
                    N_PARTICLES)
    spec_age = np.zeros(N_PARTICLES, dtype=int)
    measured: list[tuple[str, tuple]] = []
    sur_err: dict[int, list] = {}
    n_spec = 0
    # Per-depth dynamics. Without these the previous sentinel could report that
    # HV barely moved but not say WHY -- whether the twist has no gradient to
    # climb, whether the population collapses, or whether it never resamples.
    trace: list[dict] = []
    from compose_v4.benchmark.molleo_task3 import hypervolume as _hv
    archive = [list(v) for _, v in task["labels"]]
    best_obj = np.max(np.asarray(archive, dtype=float), axis=0)
    t0 = time.perf_counter()

    for depth in range(1, horizon + 1):
        nxt = []
        for i in range(N_PARTICLES):
            y = propose(states[i], rng)
            nxt.append(y if y else states[i])
        n_spec += len(set(nxt))
        verify = (depth % every == 0) or (depth == horizon)
        uniq = sorted({s for s in nxt})
        if verify:
            # TRUE ORACLE, and only here: unique REALIZED particles, never the
            # successor fiber. Cached molecules are free by the meter's rule.
            need = [s for s in uniq if s not in truth]
            vals = obj.evaluate_many(need) if need else []
            for s, v in zip(need, vals):
                truth[s] = tuple(v)
                sur.fit_add(s, v)
                measured.append((s, tuple(v)))
            pred = sur.predict(uniq)
            for j, s in enumerate(uniq):
                pass
            newv = np.array([value([truth[s]])[0] for s in nxt])
            # Surrogate error attributed to how far it had run unverified.
            for i in range(N_PARTICLES):
                a = int(spec_age[i]) + 1
                sur_err.setdefault(a, []).append(abs(float(newv[i] - cur[i])))
            spec_age[:] = 0
        else:
            p = sur.predict(nxt)
            newv = p @ w
            spec_age += 1
        log_w += BETA * (newv - cur)
        dv = newv - cur
        cur = newv
        states = nxt
        mx = log_w.max()
        wt = np.exp(log_w - mx); wt /= wt.sum()
        ess = 1.0 / float((wt ** 2).sum())
        if verify:
            for s_ in uniq:
                if s_ in truth:
                    archive.append(list(truth[s_]))
            best_obj = np.maximum(best_obj,
                                  np.max(np.asarray([truth[s_] for s_ in uniq
                                                     if s_ in truth],
                                                    dtype=float), axis=0))
        row = {"depth": depth, "verified": bool(verify),
               "V_mean": float(newv.mean()), "V_std": float(newv.std()),
               "V_max": float(newv.max()), "V_min": float(newv.min()),
               "dV_mean": float(dv.mean()), "dV_max": float(dv.max()),
               "ess": float(ess), "resampled": bool(ess < N_PARTICLES / 2),
               "n_unique_states": len(uniq),
               "n_measured_cum": len(measured),
               "best_jnk3": float(best_obj[1]),
               "hv": (float(_hv(archive, samples=40_000)) if verify else None)}
        trace.append(row)
        if ess < N_PARTICLES / 2:
            pos = (rng.random() + np.arange(N_PARTICLES)) / N_PARTICLES
            cc = np.cumsum(wt); cc[-1] = 1.0
            idx = np.searchsorted(cc, pos)
            states = [states[j] for j in idx]
            cur = cur[idx]; spec_age = spec_age[idx]
            log_w = np.zeros(N_PARTICLES)

    return {"arm": arm, "horizon": horizon, "root": root,
            "seed": int(task["seed"]),
            "preference": list(map(float, w)),
            "measured": measured, "n_measured": len(measured),
            "n_speculative_states": n_spec,
            "surrogate_abs_err_by_age": {str(k): v for k, v in sur_err.items()},
            "trace": trace,
            "seconds": round(time.perf_counter() - t0, 1)}


@app.function(image=image, cpu=(0.25, 0.25), memory=1024, timeout=8 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]], out_name: str) -> dict[str, Any]:
    import time
    out, started = [], time.perf_counter()
    for r in run_root.map(tasks, order_outputs=False, return_exceptions=True,
                          wrap_returned_exceptions=False):
        if isinstance(r, dict):
            out.append(r)
            print(f"  {r['arm']}/H{r.get('horizon','?')} root {r['root'][:22]} measured "
                  f"{r['n_measured']:>4}  spec {r['n_speculative_states']:>5}  "
                  f"{r['seconds']:.0f}s   [{len(out)}/{len(tasks)}]", flush=True)
        else:
            print(f"  FAILED {type(r).__name__}: {r}", flush=True)
    p = Path(RUN_ROOT) / "molleo_lazy" / f"{out_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"results": out}))
    artifact_volume.commit()
    print(f"DONE {len(out)}/{len(tasks)} in {time.perf_counter()-started:.0f}s")
    return {"n": len(out)}


@app.local_entrypoint()
def main(stage: str = "sentinel", n_roots: int = 2, arms: str = "L1,L4",
         horizons: str = "12") -> None:
    """MOLLEO Dev Gate 1. Preregistered in docs/AMENDMENT_MOLLEO_LAZY_GATE.md."""
    root_dir = Path(__file__).resolve().parents[1]
    coh = json.loads((root_dir / "docs/MOLLEO_DEV_COHORT.json").read_text())
    lab = json.loads((root_dir / "docs/MOLLEO_DEV_LABELS.json").read_text())
    roots = coh["roots"][:int(n_roots)] if stage != "full" else coh["roots"]
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    hs = [int(h) for h in horizons.split(",") if h.strip()]
    tasks = [{"arm": arm, "root": r, "seed": 900000 + i, "horizon": h,
              "preference": lab["preferences"][r], "labels": lab["labels"]}
             for i, r in enumerate(roots) for arm in arm_list for h in hs]
    print(f"MOLLEO LAZY GATE [{stage}]: {len(roots)} roots x {arm_list} x "
          f"H{hs}, N={N_PARTICLES}. Seeds keyed on ROOT only, so the same root "
          f"gets the same proposal stream in every configuration.")
    call = drive.spawn(tasks, f"{stage}_{'-'.join(arm_list)}_H{'-'.join(map(str,hs))}")
    print(f"spawned: {call.object_id}")
