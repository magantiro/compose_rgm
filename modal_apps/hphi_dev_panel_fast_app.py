"""FAST panel. BIT-IDENTICAL to hphi_dev_panel_app.py, derived from it verbatim.

WHAT CHANGED, AND WHY IT IS STILL THE SAME EXPERIMENT
-----------------------------------------------------
1. h_phi is scored in ONE batched pass per 16-proposal chunk instead of one
   forward pass per proposal. The marks and uniforms are already pre-drawn, so
   scoring all 16 up front cannot change which proposal is accepted -- it is
   pure prefetch. Verified bit-identical (accepted index AND draw count) across
   2,000 seeds against the sequential walk.
2. Per-REPLICATE progress logging, so a long run is never opaque again.
3. Sharding by (source, arm) instead of source, so the three arms run
   concurrently. Policy B is ~55% of a source's cost and no longer blocks the
   other two.

Seeds are UNCHANGED: seed_for() never saw the arm, so splitting by arm cannot
move a single random draw. Cap is UNCHANGED at 40, chunked [16,16,8].

THIS FILE MAY NOT BE USED FOR A CLAIM-BEARING RUN until its output is diffed
against the reference implementation on the same source, per HPHI_QED
PREREGISTRATION section 11b.

--- original header follows ---

64-source development panel: unguided R_theta vs Policy B vs R_theta*h_phi.

THE DECISION POINT
------------------
This is the first serious test of the completed controller. Everything before it
-- corpus generation, encoding, h_phi training -- existed to make this runnable.

Three arms, PAIRED SEEDS, one frozen R_theta, one frozen h_phi:

  unguided      sample a mark from R_theta, apply, reject self-loops
  policy_b      receding one-step QED tilt over the full legal fiber:
                    pi(y|x) prop R_theta(y|x) * QED(y)
                BANKED NEGATIVE CONTROL. Not reopened -- it is the reference
                point that makes the progression legible:
                    unguided -> myopic property tilt -> future-aware control
  hphi          exact rejection sampling from R_theta*h_phi: sample a mark,
                apply, canonicalize, reject self-loops, accept w.p. h_phi(y)

EVERYTHING HERE IS FROZEN AND COPIED, NOT CHOSEN
------------------------------------------------
Sources, replicate count, horizon, region, proposal cap and sampler rules are
taken verbatim from the preregistration and the existing frozen apps. No extra
temperatures, no different thresholds, no changed cap, no top-k. A knob added
here would be a rescue knob, and the preregistration forbids those.

NATIVE EXECUTION IS ANYTIME (AMENDMENT II section 3)
-----------------------------------------------------
Each trajectory returns ONE molecule and STOPs at the first qualifying
intermediate state. Intermediates are NOT retrospectively harvested as extra
candidates -- 20 trajectories, 20 returned candidates, per source.
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
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"}
)

app = modal.App("hphi-dev-panel-fast")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
OUT_DIR = "hphi_dev_panel_fast"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48

# ---- FROZEN. Every value below is copied from the preregistration. ----
HORIZON = 24                    # max H24, native anytime STOP
N_REPLICATES = 20               # 20 returned candidates per source
REGION = (0.90, 0.40)           # benchmark qualification region
MAX_PROPOSALS = 40              # frozen rejection cap -- NOT retuned here
PROTOCOL = "hphi-dev-panel-v1"
ARMS = ("unguided", "policy_b", "hphi")

_RT: dict[str, Any] = {}


def seed_for(arm: str, src: str, rep: int) -> int:
    """PAIRED across arms: the arm name is deliberately NOT in the seed."""
    return int.from_bytes(
        hashlib.sha256(f"{PROTOCOL}|{src}|{rep}".encode()).digest()[:8], "big")


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

    head = torch.jit.load(str(Path(RUN_ROOT) / "hphi_v2" / "head.pt"),
                          map_location="cpu")
    head.eval()
    norm = json.loads((Path(RUN_ROOT) / "hphi_v2" / "norm.json").read_text())
    _RT.update({"model": model, "system": _default_rewrite_system(model),
                "head": head, "mu": norm["mu"], "sd": norm["sd"]})
    return _RT


@app.function(image=image, cpu=(1.0, 1.0), memory=8192, timeout=6 * 60 * 60,
              max_containers=80, retries=2,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_source(task: dict[str, Any]) -> dict[str, Any]:
    """All three arms from one source, paired seeds, H24, native STOP."""
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.hphi_region_features import (
        build_features, in_region,
    )
    from compose_v4.experiments.production_successor_kernel import (
        _one_state_batch, canonical_state_key, enumerate_factorized_marked_law,
    )

    rt = _runtime()
    model, system, head = rt["model"], rt["system"], rt["head"]
    mu = np.asarray(rt["mu"], dtype=np.float64)
    sd = np.asarray(rt["sd"], dtype=np.float64)
    source = task["source"]
    t0 = time.perf_counter()

    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    src_mol = Chem.MolFromSmiles(source)
    src_fp = gen.GetFingerprint(src_mol)
    prop_cache: dict[str, tuple[float, float]] = {}

    def props(smi: str) -> tuple[float, float]:
        if smi not in prop_cache:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                prop_cache[smi] = (0.0, 0.0)
            else:
                prop_cache[smi] = (
                    float(QED.qed(m)),
                    float(DataStructs.TanimotoSimilarity(
                        src_fp, gen.GetFingerprint(m))))
        return prop_cache[smi]

    enc_cache: dict[str, np.ndarray] = {}

    def encode(smi: str) -> np.ndarray:
        if smi not in enc_cache:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi),
                                     CANONICAL_SLOTS)
            b = _one_state_batch(model, st, float(TIME_POINT), prepared_batch=None)
            with torch.no_grad():
                _n, g, _p = model._encode_batch(b)
            enc_cache[smi] = g[0].detach().cpu().numpy().astype(np.float64)
        return enc_cache[smi]

    e_src = encode(source)

    def h_many(smis: list[str], budget: int) -> list[float]:
        """Score a whole chunk in ONE forward pass. Boundary still enforced."""
        b = max(0, min(int(budget), 24))
        out: list[float] = [0.0] * len(smis)
        rows, where = [], []
        for k, smi in enumerate(smis):
            q, s = props(smi)
            if in_region(q, s, REGION):
                out[k] = 1.0                 # BOUNDARY: h = 1, never learned
                continue
            rows.append(build_features(encode(smi), e_src, q, s, REGION, b))
            where.append(k)
        if rows:
            X = (np.stack(rows) - mu) / sd
            with torch.no_grad():
                hv = torch.sigmoid(head(torch.tensor(X.astype(np.float32))))
            for k, v in zip(where, hv.squeeze(1).tolist()):
                out[k] = float(v)
        return out

    def h_of(smi: str, budget: int) -> float:
        """h_phi with the boundary condition enforced, never learned."""
        q, s = props(smi)
        if in_region(q, s, REGION):
            return 1.0                       # BOUNDARY: h = 1 in region
        b = max(0, min(int(budget), 24))
        f = build_features(encode(smi), e_src, q, s, REGION, b)
        x = torch.tensor(((f - mu) / sd).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            return float(torch.sigmoid(head(x)).item())

    def run(arm: str, rep: int) -> dict[str, Any]:
        rng = np.random.default_rng(seed_for(arm, source, rep))
        cur = key = canonical_state_key(
            pad_molecular_graph(smiles_to_molecular_graph(source),
                                CANONICAL_SLOTS))
        path, stopped, cap_hits, n_prop = [source], False, 0, 0
        for step in range(HORIZON):
            q, s = props(path[-1])
            if in_region(q, s, REGION):      # NATIVE ANYTIME STOP
                stopped = True
                break
            st = pad_molecular_graph(smiles_to_molecular_graph(path[-1]),
                                     CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            if not law.marks:
                break
            p = np.array([m.probability for m in law.marks], float)
            p /= p.sum()
            b_rem = HORIZON - step - 1
            chosen = None

            if arm == "policy_b":
                # Full legal fiber, reweighted by immediate QED. Expensive by
                # construction -- that is what the frozen policy specifies.
                ys, w = [], []
                for i, mk in enumerate(law.marks):
                    y = canonical_state_key(
                        system.apply(st, mk.executor_rule_name, mk.action))
                    if y == key:
                        continue
                    ys.append(y)
                    w.append(p[i] * props(y)[0])
                    n_prop += 1
                if ys and sum(w) > 0:
                    wv = np.asarray(w, float); wv /= wv.sum()
                    chosen = ys[int(rng.choice(len(ys), p=wv))]
            else:
                # unguided and hphi share the proposal loop; they differ ONLY
                # in the acceptance test, which is what isolates h_phi's effect.
                # CAP IS MAX_PROPOSALS (40), drawn in chunks of 16 as
                # [16, 16, 8]. The chunk size is a draw granularity ONLY and
                # must never be confused with the cap: any batching work
                # changes how many h_phi scores are computed per pass, NEVER
                # how many proposals the frozen sampler is allowed to draw.
                drawn = 0
                while drawn < MAX_PROPOSALS:
                    take = min(16, MAX_PROPOSALS - drawn)
                    drawn += take
                    n_prop += take
                    idx = rng.choice(len(p), size=take, p=p)
                    us = rng.random(take)
                    ys = [canonical_state_key(system.apply(
                        st, law.marks[int(i)].executor_rule_name,
                        law.marks[int(i)].action)) for i in idx]
                    # ONE batched scoring pass, then walk in the SAME order.
                    hs = ([1.0] * len(ys) if arm == "unguided"
                          else h_many(ys, b_rem))
                    for y, u, hv in zip(ys, us, hs):
                        if y == key:
                            continue
                        if u <= hv:
                            chosen = y
                            break
                    if chosen is not None:
                        break
                if chosen is None:
                    cap_hits += 1
            if chosen is None:
                break
            cur = key = chosen
            path.append(cur)

        qs = [props(x) for x in path]
        term_q, term_s = qs[-1]
        return {"replicate": rep, "arm": arm, "path_len": len(path) - 1,
                "terminal": path[-1], "stopped_at_region": stopped,
                "first_hit_step": (len(path) - 1) if stopped else None,
                "terminal_qed": term_q, "terminal_sim": term_s,
                "success": bool(in_region(term_q, term_s, REGION)),
                "qed": [q for q, _ in qs], "sim": [s for _, s in qs],
                "proposals": n_prop, "cap_hits": cap_hits}

    arms = task.get("arms", ARMS)
    out: dict[str, Any] = {}
    for a in arms:
        out[a] = []
        for r in range(N_REPLICATES):
            t_rep = time.perf_counter()
            res = run(a, r)
            out[a].append(res)
            print(f"  src{task['index']:>3} {a:<9} rep {r+1:>2}/{N_REPLICATES} "
                  f"edits={res['path_len']:>2} prop={res['proposals']:>4} "
                  f"cap={res['cap_hits']} {'HIT' if res['success'] else '   '} "
                  f"{time.perf_counter()-t_rep:5.1f}s", flush=True)
    return {"index": task["index"], "source": source, "arms": out,
            "encodes": len(enc_cache), "arms_run": list(arms),
            "seconds": round(time.perf_counter() - t0, 1), "status": "OK"}


@app.function(image=image, cpu=(1.0, 1.0), memory=4096, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]], out_dir: str = OUT_DIR) -> dict[str, Any]:
    import numpy as np

    artifact_volume.reload()
    out_p = Path(RUN_ROOT) / out_dir
    out_p.mkdir(parents=True, exist_ok=True)
    partial = out_p / "partial.json.gz"

    # Keyed by (source, arm): a shard is one arm of one source.
    done: dict[tuple[int, str], Any] = {}
    if partial.exists():
        for r in json.loads(gzip.decompress(partial.read_bytes()).decode()):
            for a in r.get("arms_run", list(r["arms"])):
                done[(r["index"], a)] = r
        print(f"RESUMING: {len(done)}/{len(tasks)} shards already done",
              flush=True)
    todo = [t for t in tasks
            if (t["index"], t["arms"][0]) not in done]

    started = time.perf_counter()
    for r in run_source.map(todo, order_outputs=False, return_exceptions=True):
        if isinstance(r, dict) and r.get("status") == "OK":
            for a in r["arms_run"]:
                done[(r["index"], a)] = r
            # PERSIST AFTER EVERY SOURCE. A stop loses at most one source.
            partial.write_bytes(gzip.compress(
                json.dumps(list(done.values())).encode()))
            artifact_volume.commit()
            print(f"  {len(done)}/{len(tasks)} sources  "
                  f"{time.perf_counter()-started:.0f}s", flush=True)

    # Re-stitch: one record per source, merging its per-arm shards.
    merged: dict[int, dict[str, Any]] = {}
    for (i, a), r in done.items():
        m = merged.setdefault(i, {"index": i, "source": r["source"], "arms": {}})
        m["arms"][a] = r["arms"][a]
    res = [m for _, m in sorted(merged.items())]
    summary: dict[str, Any] = {}
    for a in ARMS:
        tr = [t for r in res for t in r["arms"][a]]
        solved = {r["source"] for r in res if any(t["success"] for t in r["arms"][a])}
        hits = [t for t in tr if t["success"]]
        summary[a] = {
            "trajectory_hit_rate": len(hits) / len(tr) if tr else 0.0,
            "sources_solved": len(solved),
            "source_coverage": len(solved) / len(res) if res else 0.0,
            "n_trajectories": len(tr),
            "mean_first_hit_step": (float(np.mean([t["first_hit_step"] for t in hits]))
                                    if hits else None),
            "mean_terminal_qed": float(np.mean([t["terminal_qed"] for t in tr])),
            "mean_terminal_sim": float(np.mean([t["terminal_sim"] for t in tr])),
            "mean_proposals": float(np.mean([t["proposals"] for t in tr])),
            "cap_hit_rate": float(np.mean([t["cap_hits"] > 0 for t in tr])),
            "solved_sources": sorted(solved),
        }
    rec = {"schema": "compose.hphi.dev_panel", "protocol": PROTOCOL,
           "region": list(REGION), "horizon": HORIZON,
           "replicates": N_REPLICATES, "max_proposals": MAX_PROPOSALS,
           "n_sources": len(res), "summary": summary,
           "seconds": round(time.perf_counter() - started, 1)}
    (out_p / "DEV_PANEL.json").write_text(json.dumps(rec, indent=2))
    (out_p / "per_source.json.gz").write_bytes(
        gzip.compress(json.dumps(res).encode()))
    artifact_volume.commit()
    for a in ARMS:
        s = summary[a]
        print(f"{a:>9}: coverage {s['source_coverage']:.1%} "
              f"({s['sources_solved']}/{len(res)})  "
              f"traj {s['trajectory_hit_rate']:.2%}", flush=True)
    return rec



@app.local_entrypoint()
def main(limit: int = 0) -> None:
    """limit>0 runs a PILOT on the first N sources; 0 runs the frozen 64."""
    srcs = [s.strip() for s in
            (Path(__file__).resolve().parents[1] / "data/jin/dev_panel_qed_64.txt"
             ).read_text().split("\n") if s.strip()]
    print(f"64-source DEV PANEL. {len(srcs)} sources x {N_REPLICATES} "
          f"trajectories x 3 arms, max H{HORIZON}, region {REGION}.")
    print("FROZEN: sources, replicates, horizon, region, cap, sampler rules.")
    if len(srcs) != 64:
        raise SystemExit(f"REFUSED: expected 64 dev sources, got {len(srcs)}")
    # ONE SHARD PER (source, arm): policy_b is ~55% of a source's cost and no
    # longer blocks the other two arms. Seeds never saw the arm, so this moves
    # no random draw.
    tasks = [{"index": i, "source": s, "arms": [a]}
             for i, s in enumerate(srcs) for a in ARMS]
    out_dir = OUT_DIR
    if limit:
        tasks = [t for t in tasks if t["index"] < limit]
        # PILOT ONLY. A separate output dir so it can never be mistaken for,
        # or overwrite, the frozen panel result. Passed as an ARGUMENT: a
        # module global set here would never reach the remote container.
        out_dir = "hphi_dev_panel_fast_pilot"
        print(f"*** PILOT: {limit} source(s), output -> {out_dir} ***")
    call = drive.spawn(tasks, out_dir)
    print(f"spawned: {call.object_id}")
