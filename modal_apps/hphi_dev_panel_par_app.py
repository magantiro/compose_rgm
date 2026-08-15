"""Replicate-parallel runner. SCHEDULING ONLY -- the science is byte-identical.

WHAT THIS CHANGES: which independent replicates execute simultaneously.
WHAT THIS MUST NOT CHANGE: within-replicate sampling, proposal ordering, cap
behaviour, STOP logic, controller evaluation, or the RNG stream.

The trajectory function is COPIED VERBATIM out of hphi_dev_panel_app.py at build
time, never retyped, so the sampler cannot drift. seed_for() is the frozen one
and depends only on (source, replicate) -- never on worker count, launch order,
completion order, retries, or resume. Each replicate seeds its OWN Generator; no
worker touches shared or global randomness. Results are reassembled by
(source, arm, replicate), never by completion order.

ONE COMPLETED REPLICATE = ONE DURABLE UNIT OF SCIENTIFIC EVIDENCE.
Each replicate is persisted the instant it finishes, written to a temp file and
atomically renamed, with a manifest record carrying its seed, checksum, and the
git commit / model hashes it was produced under. Resume skips only fully
committed, checksum-valid replicates, so a killed job reproduces the same final
artifact as an uninterrupted one. Aggregates are always rebuilt FROM the
persisted records; they are never the only stored evidence.

NOT QUALIFIED UNTIL exact trajectory-level replay parity against the frozen
reference plus a forced-kill/resume test. Any unexplained discrepancy: park it
and run the reference.

--- derived from ---

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

app = modal.App("hphi-dev-panel-par")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
OUT_DIR = "hphi_dev_panel_par"
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



_G: dict[str, Any] = {}


def _replicate_unit(unit: tuple[str, int]) -> dict[str, Any]:
    """One (arm, replicate) trajectory in a forked worker. Reads _G only."""
    import numpy as np  # noqa: F401  (used by the copied closure)
    import torch

    torch.set_num_threads(1)
    arm, rep = unit
    t0 = time.perf_counter()
    rec = _G["run"](arm, rep)
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    rec["seed"] = seed_for(arm, _G["source"], rep)
    return rec


def _persist(out_p, idx: int, source: str, rec: dict[str, Any],
             provenance: dict[str, Any]) -> None:
    """Atomic write-then-rename so an interruption cannot leave a valid-looking
    partial. The checksum covers the record only, so resume can verify it."""
    import hashlib
    import os

    payload = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    doc = {"index": idx, "source": source, "arm": rec["arm"],
           "replicate": rec["replicate"], "seed": rec["seed"],
           "sha256": hashlib.sha256(payload.encode()).hexdigest(),
           "provenance": provenance, "record": rec}
    name = f"{idx:03d}_{rec['arm']}_{rec['replicate']:02d}.json"
    final = out_p / "replicates" / name
    tmp = final.with_suffix(".json.tmp")
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(doc))
    os.replace(tmp, final)                      # ATOMIC


def _valid_records(out_p) -> dict[tuple[int, str, int], dict[str, Any]]:
    """Load persisted replicates, keeping only checksum-valid ones."""
    import hashlib

    d = out_p / "replicates"
    keep: dict[tuple[int, str, int], dict[str, Any]] = {}
    if not d.exists():
        return keep
    for f in d.glob("*.json"):
        try:
            doc = json.loads(f.read_text())
            payload = json.dumps(doc["record"], sort_keys=True,
                                 separators=(",", ":"))
            if hashlib.sha256(payload.encode()).hexdigest() != doc["sha256"]:
                print(f"  CHECKSUM MISMATCH, ignoring {f.name}", flush=True)
                continue
            keep[(doc["index"], doc["arm"], doc["replicate"])] = doc
        except Exception:  # noqa: BLE001  -- a torn file is simply not valid
            print(f"  UNREADABLE, ignoring {f.name}", flush=True)
    return keep


@app.function(image=image, cpu=(8.0, 8.0), memory=32768, timeout=6 * 60 * 60,
              max_containers=64, retries=2,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_source(task: dict[str, Any]) -> dict[str, Any]:
    """All (arm, replicate) units for one source, forked across 8 workers."""
    import multiprocessing as mp
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
    idx = int(task["index"])
    out_p = Path(RUN_ROOT) / task.get("out_dir", OUT_DIR)
    workers = int(task.get("workers", 8))
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

    def h_of(smi: str, budget: int) -> float:
        q, s = props(smi)
        if in_region(q, s, REGION):
            return 1.0
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
                    for i, u in zip(idx, us):
                        mk = law.marks[int(i)]
                        y = canonical_state_key(
                            system.apply(st, mk.executor_rule_name, mk.action))
                        if y == key:
                            continue
                        if arm == "unguided" or u <= h_of(y, b_rem):
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

    provenance = {"protocol": PROTOCOL, "region": list(REGION),
                  "horizon": HORIZON, "max_proposals": MAX_PROPOSALS,
                  "git_commit": task.get("git_commit", "unknown"),
                  "corpus_sha256":
                      "647f8265f5143e2b15dc047de42e837203593c42beb7e45e2b94e55e41602581",
                  "r_theta_sha256":
                      "c979cdb3d7b0b403bfbf7bfb0aa5098b2588c6d4217770c2c58292b7c4e53de8"}

    artifact_volume.reload()
    have = _valid_records(out_p)
    # QUALIFICATION-ONLY subsetting. Never used for a claim-bearing run: the
    # frozen panel is all ARMS x all N_REPLICATES. This exists so replay parity
    # can be established on a handful of units instead of a 3.7-hour source.
    q_arms = task.get("qual_arms") or list(ARMS)
    q_reps = task.get("qual_reps") or list(range(N_REPLICATES))
    units = [(a, r) for a in q_arms for r in q_reps if (idx, a, r) not in have]
    if not units:
        print(f"src{idx:>3}: all {len(ARMS)*N_REPLICATES} replicates already "
              f"persisted and checksum-valid", flush=True)
        return {"index": idx, "source": source, "status": "OK",
                "replicates_run": 0, "seconds": 0.0}

    _G.update({"run": run, "source": source})
    done = 0
    ctx = mp.get_context("fork")     # model shared copy-on-write
    with ctx.Pool(workers) as pool:
        # imap_unordered: completion order varies, but every record is keyed by
        # (source, arm, replicate) and persisted individually, so the artifact
        # is order-independent by construction.
        for rec in pool.imap_unordered(_replicate_unit, units):
            _persist(out_p, idx, source, rec, provenance)
            artifact_volume.commit()
            done += 1
            print(f"  src{idx:>3} {rec['arm']:<9} rep {rec['replicate']:>2} "
                  f"edits={rec['path_len']:>2} prop={rec['proposals']:>4} "
                  f"cap={rec['cap_hits']} "
                  f"{'HIT' if rec['success'] else '   '} "
                  f"{rec['seconds']:6.1f}s  [{done}/{len(units)}]", flush=True)
    return {"index": idx, "source": source, "status": "OK",
            "replicates_run": done,
            "seconds": round(time.perf_counter() - t0, 1)}


@app.function(image=image, cpu=(2.0, 2.0), memory=8192, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]], out_dir: str = OUT_DIR) -> dict[str, Any]:
    import numpy as np

    artifact_volume.reload()
    out_p = Path(RUN_ROOT) / out_dir
    out_p.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for t in tasks:
        t["out_dir"] = out_dir
    for r in run_source.map(tasks, order_outputs=False, return_exceptions=True):
        if isinstance(r, dict) and r.get("status") == "OK":
            print(f"  source {r['index']} done ({r['replicates_run']} reps, "
                  f"{r['seconds']}s)", flush=True)

    # AGGREGATES ARE REBUILT FROM THE PERSISTED RECORDS, never accumulated in
    # memory -- the records are the evidence, the summary is derived.
    artifact_volume.reload()
    have = _valid_records(out_p)
    by_src: dict[int, dict[str, list]] = {}
    for (i, a, _r), doc in sorted(have.items()):
        by_src.setdefault(i, {}).setdefault(a, []).append(doc["record"])
    summary: dict[str, Any] = {}
    for a in ARMS:
        tr = [t for s in by_src.values() for t in s.get(a, [])]
        solved = {i for i, s in by_src.items() if any(t["success"] for t in s.get(a, []))}
        hits = [t for t in tr if t["success"]]
        summary[a] = {
            "trajectory_hit_rate": (len(hits) / len(tr)) if tr else 0.0,
            "sources_solved": len(solved),
            "source_coverage": (len(solved) / len(by_src)) if by_src else 0.0,
            "n_trajectories": len(tr),
            "mean_first_hit_step": (float(np.mean([t["first_hit_step"] for t in hits]))
                                    if hits else None),
            "mean_terminal_qed": float(np.mean([t["terminal_qed"] for t in tr])) if tr else None,
            "mean_terminal_sim": float(np.mean([t["terminal_sim"] for t in tr])) if tr else None,
            "mean_proposals": float(np.mean([t["proposals"] for t in tr])) if tr else None,
            "cap_hit_rate": float(np.mean([t["cap_hits"] > 0 for t in tr])) if tr else None,
            "solved_sources": sorted(solved),
        }
    rec = {"schema": "compose.hphi.dev_panel", "protocol": PROTOCOL,
           "runner": "replicate-parallel", "region": list(REGION),
           "horizon": HORIZON, "replicates": N_REPLICATES,
           "max_proposals": MAX_PROPOSALS, "n_sources": len(by_src),
           "n_replicate_records": len(have), "summary": summary,
           "seconds": round(time.perf_counter() - started, 1)}
    (out_p / "DEV_PANEL.json").write_text(json.dumps(rec, indent=2))
    artifact_volume.commit()
    for a in ARMS:
        s = summary[a]
        print(f"{a:>9}: coverage {s['source_coverage']:.1%} "
              f"({s['sources_solved']}/{len(by_src)})  "
              f"traj {s['trajectory_hit_rate']:.2%}  "
              f"cap {s['cap_hit_rate']}", flush=True)
    return rec


@app.local_entrypoint()
def main(limit: int = 0, out_dir: str = OUT_DIR, workers: int = 8,
         qual_arms: str = "", qual_reps: str = "") -> None:
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()[:12]
    srcs = [s.strip() for s in
            (Path(__file__).resolve().parents[1] / "data/jin/dev_panel_qed_64.txt"
             ).read_text().split("\n") if s.strip()]
    if len(srcs) != 64:
        raise SystemExit(f"REFUSED: expected 64 dev sources, got {len(srcs)}")
    qa = [a for a in qual_arms.split(",") if a] or None
    qr = [int(r) for r in qual_reps.split(",") if r] or None
    tasks = [{"index": i, "source": s, "workers": workers, "git_commit": commit,
              "qual_arms": qa, "qual_reps": qr}
             for i, s in enumerate(srcs)]
    if limit:
        tasks = tasks[:limit]
        print(f"*** LIMITED: {limit} source(s) -> {out_dir} ***")
    print(f"replicate-parallel runner, {workers} workers/source, commit {commit}")
    print("scheduling only: seeds, ordering, cap, STOP and RNG are the frozen ones")
    call = drive.spawn(tasks, out_dir)
    print(f"spawned: {call.object_id}")
