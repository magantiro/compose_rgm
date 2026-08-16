"""FAST molecular twisted-SMC. EXACT speedups only, qualified against the reference.

Derived by patching hphi_smc_app.py so the sampler cannot drift. One change:

  CARRIED-VALUE REUSE. h(x_t, b) at step t+1 is EXACTLY the h(y, b-1) already
  computed at step t, because b-1 == b\'. Carrying it halves h_phi evaluations
  with no approximation. It travels with the state through resampling, and is
  invalidated on dead ends. Verified arithmetically before implementation.

NOT QUALIFIED until its record matches the reference on a small fixed set.

--- derived from ---

SLOW REFERENCE molecular twisted-SMC. Correctness only -- NO optimization.

THE TWO RULES THAT MAKE THIS THE INTENDED ALGORITHM
----------------------------------------------------
1. Every particle proposal comes from the frozen R_theta law ALONE. h_phi is
   NOT used to select proposals.
2. h_phi enters ONLY through the Feynman-Kac incremental weight

       G_b(x, y) = h_phi(y, b-1) / h_phi(x, b)

Using h_phi to both select AND weight would double-count control and silently
sample a different process. Because the product telescopes to
h_0(x_H)/h_H(x_0), and the terminal boundary h_0(x) = 1[x in B] is EXACT, an
approximate h_phi steers particle allocation WITHOUT redefining what counts as
success. That is why a learned twist is safe here, not merely tolerable.

FROZEN (preregistration section 13): N = 32 particles, H = 24, region
(0.90, 0.40), absorbing STOP, systematic resampling at strict ESS < N/2, one
molecule sampled from the terminal normalized particle measure, 20 INDEPENDENT
runs per source. The 32 particles are inference state, never candidates.

Records per-transition h values, per-step ESS/resample decisions and the
resampling indices so the mechanical checks run OFFLINE against the artifact
rather than trusting this code.
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

app = modal.App("hphi-smc-fast")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
OUT_DIR = "hphi_smc_fast"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48

# ---- FROZEN. Every value below is copied from the preregistration. ----
HORIZON = 24                    # max H24, native anytime STOP
N_REPLICATES = 20               # 20 returned candidates per source
REGION = (0.90, 0.40)           # benchmark qualification region
MAX_PROPOSALS = 40              # frozen rejection cap -- NOT retuned here
PROTOCOL = "hphi-smc-fast-v1"
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


def _slot_unit(slot: int) -> dict[str, Any]:
    """One independent SMC run = one returned candidate."""
    import torch

    torch.set_num_threads(1)
    t0 = time.perf_counter()
    rec = _G["run_smc"](slot)
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    return rec


#: Fields that are RUNTIME/PROVENANCE, not science. They are recorded, but are
#: deliberately EXCLUDED from the scientific checksum because they legitimately
#: differ between an uninterrupted run and a killed-then-resumed one (wall time,
#: worker identity, retry count). Hashing them would make the resume-equivalence
#: test fail for reasons that have nothing to do with the experiment.
RUNTIME_FIELDS = ("seconds",)


def scientific_payload(rec: dict[str, Any]) -> dict[str, Any]:
    """The canonical scientific object: source, arm, replicate, seed, accepted
    states/path, proposal counts, cap events, STOP, first hit, and metrics."""
    return {k: v for k, v in rec.items() if k not in RUNTIME_FIELDS}


def scientific_sha256(rec: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(
        scientific_payload(rec), sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def _persist(out_p, idx: int, source: str, rec: dict[str, Any],
             provenance: dict[str, Any]) -> None:
    """Atomic write-then-rename so an interruption cannot leave a valid-looking
    partial. The checksum covers the SCIENTIFIC payload only."""
    import os

    doc = {"index": idx, "source": source, "arm": rec["arm"],
           "replicate": rec["replicate"], "seed": rec["seed"],
           "sha256": scientific_sha256(rec),
           "runtime": {k: rec.get(k) for k in RUNTIME_FIELDS},
           "provenance": provenance, "record": rec}
    name = f"{idx:03d}_{rec['arm']}_{rec['replicate']:02d}.json"
    final = out_p / "replicates" / name
    tmp = final.with_suffix(".json.tmp")
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(doc))
    os.replace(tmp, final)                      # ATOMIC


def _valid_records(out_p) -> dict[tuple[int, str, int], dict[str, Any]]:
    """Load persisted replicates, keeping only checksum-valid ones."""
    d = out_p / "replicates"
    keep: dict[tuple[int, str, int], dict[str, Any]] = {}
    if not d.exists():
        return keep
    for f in d.glob("*.json"):
        try:
            doc = json.loads(f.read_text())
            if scientific_sha256(doc["record"]) != doc["sha256"]:
                print(f"  CHECKSUM MISMATCH, ignoring {f.name}", flush=True)
                continue
            keep[(doc["index"], doc["arm"], doc["replicate"])] = doc
        except Exception:  # noqa: BLE001  -- a torn file is simply not valid
            print(f"  UNREADABLE, ignoring {f.name}", flush=True)
    return keep


# cpu=(request, limit): reserve ONE core, burst to 8. A hard (8, 8)
# reservation bills 8 cores even when only one slot has work.
# ONE CPU PER SLOT. Each slot is an independent serial SMC run; there is
# no particle parallelism, so a second core would sit idle and billed.
@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB,
              timeout=6 * 60 * 60,
              max_containers=64, retries=2,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_source(task: dict[str, Any]) -> dict[str, Any]:
    """20 independent SMC runs for one source."""
    import multiprocessing as mp
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, rdFingerprintGenerator

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.hphi_region_features import (
        build_features, in_region,
    )
    from compose_v4.experiments.hphi_smc import (
        EXTINCT_NO_HIT, N_PARTICLES, effective_sample_size,
        normalized_weights, should_resample, systematic_resample,
        terminal_output,
    )
    from compose_v4.experiments.factorized_mark_conditional import (
        operator_capability_batch_kwargs,
    )
    from compose_v4.model.factorized_tracelet_rate_model import (
        prepare_factorized_mark_batch,
    )
    from compose_v4.experiments.production_successor_kernel import (
        canonical_state_key, enumerate_factorized_marked_law,
    )

    rt = _runtime()
    model, system, head = rt["model"], rt["system"], rt["head"]
    mu = np.asarray(rt["mu"], dtype=np.float64)
    sd = np.asarray(rt["sd"], dtype=np.float64)
    source = task["source"]
    idx = int(task["index"])
    out_p = Path(RUN_ROOT) / task.get("out_dir", OUT_DIR)
    t0 = time.perf_counter()

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    src_fp = gen.GetFingerprint(Chem.MolFromSmiles(source))
    prop_cache: dict[str, tuple[float, float]] = {}
    enc_cache: dict[str, np.ndarray] = {}
    # CACHE-KEY AUDIT (static, done before relying on this):
    #   key = (id(ring_catalog), use_aromatic_bond_view, compute_cyclic_graft,
    #          compute_ring_opening, compute_ring_system_delete,
    #          5 semantics strings, molecular_state_cache_key(state))
    # Two gaps, both SAFE HERE but conditionally so:
    #   * `compute_ring_restates` is NOT in the key though it gates
    #     macro_system. Safe only because our capability flags come from
    #     operator_capability_batch_kwargs(model.operator_capabilities) and are
    #     constant for a given model.
    #   * id(ring_catalog) is object identity, so a collected-and-reallocated
    #     catalog could false-hit. Safe only because we hold the model for the
    #     container's lifetime.
    # If either assumption changes, this cache stops being exact.
    #
    # CHEMISTRY FEATURE CACHE. The 561-line batch builder puts every expensive
    # call -- admission masks, macro actions, ring-system deletes -- behind a
    # single `if features is None` guard, so a cache hit skips ALL of it. It is
    # keyed on the state and the features are deterministic, so this is exact by
    # construction, not an approximation. Measured hit rate on the banked
    # reference run: 69.5% (particles converge, and 39% of proposals are states
    # already seen). The encode path simply never passed one before.
    chem_cache: dict[Any, Any] = {}

    def props(smi: str) -> tuple[float, float]:
        if smi not in prop_cache:
            m = Chem.MolFromSmiles(smi)
            prop_cache[smi] = (0.0, 0.0) if m is None else (
                float(QED.qed(m)),
                float(DataStructs.TanimotoSimilarity(
                    src_fp, gen.GetFingerprint(m))))
        return prop_cache[smi]

    def encode(smi: str) -> np.ndarray:
        if smi not in enc_cache:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi),
                                     CANONICAL_SLOTS)
            b = prepare_factorized_mark_batch(
                (st,), (float(TIME_POINT),), (None,), (None,), (0.0,),
                use_aromatic_bond_view=True, ring_catalog=model.ring_catalog,
                chemistry_feature_cache=chem_cache,
                **operator_capability_batch_kwargs(model.operator_capabilities))
            with torch.no_grad():
                _n, g, _p = model._encode_batch(b)
            enc_cache[smi] = g[0].detach().cpu().numpy().astype(np.float64)
        return enc_cache[smi]

    e_src = encode(source)

    def h_phi(smi: str, budget: int) -> float:
        """The twist. EXACT boundary h = 1 in-region, matching h_0 = 1[x in B]."""
        q, s = props(smi)
        if in_region(q, s, REGION):
            return 1.0
        if budget <= 0:
            return 0.0
        f = build_features(encode(smi), e_src, q, s, REGION,
                           max(0, min(int(budget), 24)))
        x = torch.tensor(((f - mu) / sd).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            return float(torch.sigmoid(head(x)).item())

    def propose(smi: str, rng) -> str:
        """Sample ONE successor from the frozen R_theta law. NO h_phi here."""
        st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
        if not law.marks:
            return ""
        p = np.array([m.probability for m in law.marks], float)
        p /= p.sum()
        mk = law.marks[int(rng.choice(len(p), p=p))]
        y = canonical_state_key(system.apply(st, mk.executor_rule_name,
                                             mk.action))
        return "" if y == smi else y

    src_canon = canonical_state_key(
        pad_molecular_graph(smiles_to_molecular_graph(source), CANONICAL_SLOTS))

    def run_smc(slot: int) -> dict[str, Any]:
        # OBSERVATIONAL LOGGING ONLY. Nothing below reads these values back
        # into the algorithm; removing every print would leave the sampler
        # bit-identical.
        t_slot = time.perf_counter()
        print(f"    [src{idx} slot{slot:>2}] START N={N_PARTICLES} H={HORIZON}",
              flush=True)
        rng = np.random.default_rng(seed_for("smc", source, slot))
        q0, s0 = props(src_canon)
        states = [src_canon] * N_PARTICLES
        absorbed = [in_region(q0, s0, REGION)] * N_PARTICLES
        log_w = np.zeros(N_PARTICLES)
        transitions: list[dict[str, Any]] = []
        sync: list[dict[str, Any]] = []

        # CARRIED h. carried[i] holds h(states[i], current budget) when known.
        # At step t+1 the needed h(x_{t+1}, b') is EXACTLY the h(y, b-1) already
        # computed at step t, since b-1 == b'. Exact, not an approximation.
        carried: list[float | None] = [None] * N_PARTICLES
        for step in range(HORIZON):
            b = HORIZON - step
            for i in range(N_PARTICLES):
                if absorbed[i]:
                    continue
                x = states[i]
                hx = carried[i] if carried[i] is not None else h_phi(x, b)
                if hx <= 0.0:
                    log_w[i] = -np.inf
                    absorbed[i] = True
                    carried[i] = None
                    continue
                y = propose(x, rng)
                if not y:
                    log_w[i] = -np.inf
                    absorbed[i] = True
                    carried[i] = None
                    continue
                hy = h_phi(y, b - 1)
                inc = (np.log(hy) if hy > 0 else -np.inf) - np.log(hx)
                log_w[i] += inc
                states[i] = y
                carried[i] = hy          # becomes h(x, b') at the next step
                qy, sy = props(y)
                if in_region(qy, sy, REGION):
                    absorbed[i] = True
                if (i + 1) % 8 == 0:
                    print(f"    [src{idx} slot{slot:>2}] step {step:>2} "
                          f"particle {i + 1:>2}/{N_PARTICLES} "
                          f"h(x)={hx:.4f} h(y)={hy:.4f} "
                          f"{time.perf_counter() - t_slot:6.1f}s", flush=True)
                transitions.append({
                    "step": step, "particle": i, "budget": b, "x": x, "y": y,
                    "h_x_b": hx, "h_y_bm1": hy,
                    "log_G": (float(inc) if np.isfinite(inc) else None),
                    "absorbed_after": absorbed[i]})
            if np.all(np.isneginf(log_w)):
                break
            w = normalized_weights(log_w)
            entry: dict[str, Any] = {
                "step": step, "ess": float(effective_sample_size(w)),
                "resampled": bool(should_resample(w, N_PARTICLES)),
                "n_absorbed": int(sum(absorbed))}
            if entry["resampled"]:
                ridx = systematic_resample(w, rng)
                states = [states[j] for j in ridx]
                absorbed = [absorbed[j] for j in ridx]
                carried = [carried[j] for j in ridx]   # travels with the state
                log_w = np.zeros(N_PARTICLES)
                entry["indices"] = [int(v) for v in ridx]
                entry["n_unique"] = len({int(v) for v in ridx})
                entry["n_unique_states"] = len(set(states))
            sync.append(entry)
            print(f"    [src{idx} slot{slot:>2}] step {step:>2}/{HORIZON} "
                  f"ESS {entry['ess']:5.1f}/{N_PARTICLES} "
                  f"absorbed {entry['n_absorbed']:>2} "
                  f"{'RESAMPLE' if entry['resampled'] else '        '} "
                  f"uniq_states {entry.get('n_unique_states', '-'):>3} "
                  f"{time.perf_counter() - t_slot:6.1f}s", flush=True)
            if all(absorbed):
                print(f"    [src{idx} slot{slot:>2}] all particles absorbed "
                      f"at step {step}", flush=True)
                break

        j, status = terminal_output(log_w, rng)
        if status == EXTINCT_NO_HIT:
            # Z_H == 0: no particle entered B, so the target-conditioned
            # measure has no sampled support. Return the canonical source
            # SOLELY to satisfy the benchmark's fixed 20-output interface.
            # This slot is ALWAYS a benchmark failure.
            ret = src_canon
            print(f"    [src{idx} slot{slot:>2}] {EXTINCT_NO_HIT}: no particle "
                  f"entered the region; returning x_0", flush=True)
        else:
            ret = states[j]
        w = normalized_weights(log_w) if status != EXTINCT_NO_HIT \
            else np.zeros(N_PARTICLES)
        rq, rs = props(ret)
        return {"slot": slot, "returned": ret,
                "returned_index": (int(j) if j is not None else None),
                "status": status, "extinct": status == EXTINCT_NO_HIT,
                "n_terminal_nonzero": int(np.sum(np.asarray(w) > 0)),
                "first_absorption_step": next(
                    (t2["step"] for t2 in transitions if t2["absorbed_after"]),
                    None),
                "terminal_qed": rq, "terminal_sim": rs,
                "success": bool(in_region(rq, rs, REGION)),
                "final_weights": [float(v) for v in w],
                "final_ess": (float(effective_sample_size(w))
                              if status != EXTINCT_NO_HIT else 0.0),
                "final_states": states, "final_absorbed": absorbed,
                "n_absorbed": int(sum(absorbed)),
                "sync": sync, "transitions": transitions,
                "seed": seed_for("smc", source, slot),
                "n_particles": N_PARTICLES, "horizon": HORIZON}

    provenance = {
        "protocol": PROTOCOL, "region": list(REGION), "horizon": HORIZON,
        "n_particles": N_PARTICLES, "git_commit": task.get("git_commit", "?"),
        "output_rule": "sampled from the normalized terminal particle measure; "
                       "NOT best-of-population",
        "proposal_law": "frozen R_theta ALONE; h_phi enters only via the "
                        "Feynman-Kac incremental weight"}

    artifact_volume.reload()
    have = _valid_records(out_p)
    n_slots = int(task.get("n_slots", 20))
    slots = [s for s in range(n_slots) if (idx, "smc", s) not in have]
    if not slots:
        print(f"src{idx:>3}: all {n_slots} slots already persisted", flush=True)
        return {"index": idx, "source": source, "status": "OK", "slots_run": 0}

    _G["run_smc"] = run_smc
    done = 0
    ctx = mp.get_context("fork")
    n_workers = max(1, min(int(task.get("workers", 8)), len(slots)))
    print(f"src{idx}: {len(slots)} slots -> {n_workers} workers "
          f"(never more workers than work)", flush=True)
    with ctx.Pool(n_workers) as pool:
        for rec in pool.imap_unordered(_slot_unit, slots):
            rec["arm"] = "smc"
            rec["replicate"] = rec["slot"]
            _persist(out_p, idx, source, rec, provenance)
            artifact_volume.commit()
            done += 1
            nres = sum(e["resampled"] for e in rec["sync"])
            print(f"  src{idx:>3} slot {rec['slot']:>2} -> "
                  f"QED {rec['terminal_qed']:.3f} sim {rec['terminal_sim']:.3f} "
                  f"{'HIT' if rec['success'] else '---'} "
                  f"absorbed {rec['n_absorbed']:>2}/32 resample {nres:>2} "
                  f"{rec['seconds']:7.1f}s [{done}/{len(slots)}]", flush=True)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    print(f"src{idx}: PEAK RSS {peak:.2f} GiB (requested {MEM_GIB} GiB)", flush=True)
    return {"index": idx, "source": source, "status": "OK", "slots_run": done,
            "peak_rss_gib": round(peak, 2),
            "seconds": round(time.perf_counter() - t0, 1)}


@app.function(image=image, cpu=(0.25, 0.25), memory=768, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]], out_dir: str = OUT_DIR) -> dict[str, Any]:
    artifact_volume.reload()
    (Path(RUN_ROOT) / out_dir).mkdir(parents=True, exist_ok=True)
    for t in tasks:
        t["out_dir"] = out_dir
    started = time.perf_counter()
    for r in run_source.map(tasks, order_outputs=False, return_exceptions=True):
        if isinstance(r, dict) and r.get("status") == "OK":
            print(f"  source {r['index']} done ({r.get('slots_run')} slots)",
                  flush=True)
    return {"seconds": round(time.perf_counter() - started, 1),
            "n_sources": len(tasks)}


@app.local_entrypoint()
def main(limit: int = 1, out_dir: str = OUT_DIR, workers: int = 8,
         n_slots: int = 20, subset: str = "") -> None:
    import subprocess

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()[:12]
    srcs = [s.strip() for s in
            (Path(__file__).resolve().parents[1]
             / "data/jin/dev_panel_qed_64.txt").read_text().split("\n")
            if s.strip()]
    tasks = [{"index": i, "source": s, "workers": workers, "n_slots": n_slots,
              "git_commit": commit} for i, s in enumerate(srcs)]
    if subset:
        keep = {int(v) for v in subset.split(",")}
        tasks = [t for t in tasks if t["index"] in keep]
        print(f"SUBSET: {len(tasks)} sources {sorted(keep)}")
    else:
        tasks = tasks[:limit]
    print(f"SLOW REFERENCE molecular SMC: {len(tasks)} source(s) x {n_slots} "
          f"independent runs x N=32 particles, H={HORIZON}, region {REGION}")
    print("proposals from frozen R_theta ALONE; h_phi enters only via the "
          "Feynman-Kac weight")
    call = drive.spawn(tasks, out_dir)
    print(f"spawned: {call.object_id}")
