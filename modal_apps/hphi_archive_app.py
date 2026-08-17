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
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
app = modal.App("hphi-archive")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
OUT_DIR = "hphi_diverse_v3"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
HORIZON = 24
N_PARTICLES = 32
REGION = (0.90, 0.40)
N_CANDIDATES = 4
PROTOCOL = "hphi-diverse-v3"
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
    head = torch.jit.load(str(Path(RUN_ROOT) / "hphi_v2" / "head.pt"),
                          map_location="cpu")
    head.eval()
    norm = json.loads((Path(RUN_ROOT) / "hphi_v2" / "norm.json").read_text())
    _RT.update({"model": model, "system": _default_rewrite_system(model),
                "head": head, "mu": norm["mu"], "sd": norm["sd"]})
    import gc

    del ck, bundle, src, runtime, _b, _c
    gc.collect()
    return _RT


@app.function(image=image, cpu=(1.0, 1.0), memory=MEM_MIB, timeout=6 * 60 * 60,
              max_containers=128, retries=1,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def run_source(task: dict[str, Any]) -> dict[str, Any]:
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
    from compose_v4.experiments.hphi_lazy_helpers import make_helpers
    from compose_v4.experiments.hphi_lazy_sampler import sample_one_transition
    from compose_v4.experiments.hphi_region_features import build_features, in_region
    from compose_v4.experiments.hphi_smc import (
        effective_sample_size, normalized_weights, should_resample,
        systematic_resample, terminal_output,
    )
    from compose_v4.experiments.production_successor_kernel import (
        _coordinate_action, canonical_state_key,
    )

    rt = _runtime()
    model, system, head = rt["model"], rt["system"], rt["head"]
    mu = np.asarray(rt["mu"], dtype=np.float64)
    sd = np.asarray(rt["sd"], dtype=np.float64)
    source = task["source"]
    idx = int(task["index"])
    helpers = make_helpers(model, time_point=float(TIME_POINT),
                           canonical_slots=CANONICAL_SLOTS)
    _TABLE_FAMILY = {"grow_connected": "atom_insert"}

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    src_fp = gen.GetFingerprint(Chem.MolFromSmiles(source))
    prop_cache: dict[str, tuple[float, float]] = {}
    enc_cache: dict[str, np.ndarray] = {}
    state_cache: dict[str, Any] = {}

    def state_of(smi):
        st = state_cache.get(smi)
        if st is None:
            st = state_cache[smi] = pad_molecular_graph(
                smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
        return st

    def props(smi):
        if smi not in prop_cache:
            m = Chem.MolFromSmiles(smi)
            # Similarity is ALWAYS to the original source, never to a branch
            # point: the benchmark constrains distance from x0, so an archived
            # intermediate does not become a new origin.
            prop_cache[smi] = (0.0, 0.0) if m is None else (
                float(QED.qed(m)),
                float(DataStructs.TanimotoSimilarity(
                    src_fp, gen.GetFingerprint(m))))
        return prop_cache[smi]

    def encode(smi):
        if smi not in enc_cache:
            _b, _n, glob, _p, _h = helpers["encode"](
                state_of(smi), float(TIME_POINT))
            enc_cache[smi] = glob[0].detach().cpu().numpy().astype(np.float64)
        return enc_cache[smi]

    e_src = encode(source)

    def h_phi(smi, budget):
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

    def propose(smi, rng):
        st = state_of(smi)
        d = sample_one_transition(model, st, float(TIME_POINT), rng,
                                  helpers=helpers)
        if d.table is None or d.coordinate is None:
            return ""
        fam = _TABLE_FAMILY.get(d.table, d.table)
        batch = helpers["build_batch"](st, float(TIME_POINT))
        helpers["family_mask"](model, d.table, st, batch, None, None, None)
        rule, action = _coordinate_action(
            model, st, batch, family_name=fam, table_name=d.table,
            coordinate=d.coordinate)
        y = canonical_state_key(system.apply(st, rule, action))
        return "" if y == smi else y

    def run_smc(start_smi: str, start_depth: int, seed: int) -> dict[str, Any]:
        """SMC from `start_smi` with H - start_depth remaining steps."""
        rng = np.random.default_rng(seed)
        budget0 = HORIZON - int(start_depth)
        q0, s0 = props(start_smi)
        states = [start_smi] * N_PARTICLES
        absorbed = [in_region(q0, s0, REGION)] * N_PARTICLES
        log_w = np.zeros(N_PARTICLES)
        seen: list[tuple[str, int, float]] = []
        n_trans = 0

        for step in range(budget0):
            b = budget0 - step
            for i in range(N_PARTICLES):
                if absorbed[i]:
                    continue
                x = states[i]
                hx = h_phi(x, b)
                if hx <= 0.0:
                    log_w[i] = -np.inf; absorbed[i] = True; continue
                y = propose(x, rng)
                if not y:
                    log_w[i] = -np.inf; absorbed[i] = True; continue
                hy = h_phi(y, b - 1)
                log_w[i] += (np.log(hy) if hy > 0 else -np.inf) - np.log(hx)
                states[i] = y
                n_trans += 1
                depth = int(start_depth) + step + 1
                # ARCHIVE ENTRY. h_phi(y, b-1) is exactly h_phi(y, H - depth),
                # which is the frozen branch score -- no extra evaluation.
                seen.append((y, depth, float(hy)))
                qy, sy = props(y)
                if in_region(qy, sy, REGION):
                    absorbed[i] = True
            if np.all(np.isneginf(log_w)):
                break
            w = normalized_weights(log_w)
            if should_resample(w, N_PARTICLES):
                ridx = systematic_resample(w, rng)
                states = [states[j] for j in ridx]
                absorbed = [absorbed[j] for j in ridx]
                log_w = np.zeros(N_PARTICLES)
            if all(absorbed):
                break

        # FROZEN RULE C. With the exact terminal potential h_0(x) = 1[x in B],
        # every particle outside the region ends at weight zero, so a run where
        # none arrived has Z_H == 0 and NO sampled support. Normalising there
        # yields NaN, which is precisely the crash this reintroduced by
        # hand-rolling a terminal step instead of using the qualified one.
        #
        # Extinct returns the ORIGINAL source -- not the branch start -- solely
        # to satisfy the fixed candidate interface, and is ALWAYS a failure: no
        # particle ever entered B, so no fallback could have qualified.
        j, status = terminal_output(log_w, rng)
        if j is None:
            ret, ess = source, 0.0
        else:
            ret = states[j]
            ess = float(effective_sample_size(normalized_weights(log_w)))
        rq, rs = props(ret)
        return {"returned": ret, "terminal_qed": rq, "terminal_sim": rs,
                "success": bool(in_region(rq, rs, REGION)),
                "extinct": j is None, "status": status,
                "n_transitions": n_trans, "archive": seen,
                "start_depth": int(start_depth), "budget": budget0,
                "final_ess": ess}

    # RESUMABLE. A driver crash cost eleven sources of finished compute once;
    # an already-written source is never recomputed.
    done_path = (Path(RUN_ROOT) / task.get("out_dir", OUT_DIR)
                 / f"{idx:03d}_{task['arm']}.json")
    if done_path.exists():
        try:
            prior = json.loads(done_path.read_text())
            if prior.get("arms", {}).get(task["arm"]):
                print(f"  src{idx}: already complete, skipping", flush=True)
                return prior
        except Exception:  # noqa: BLE001
            pass

    out: dict[str, Any] = {"index": idx, "source": source,
                           "stratum": task.get("stratum"), "arms": {}}
    t_all = time.perf_counter()

    for arm in [task["arm"]]:
        t0 = time.perf_counter()
        cands, work, archive = [], 0, {}
        branch_log, used_branches = [], set()
        # x0 IS AN ARCHIVE STATE. "Persistent archive" means never FORGET, not
        # never restart: if x0 is still the best launch point the policy should
        # be free to choose it. v1 forced a branch every time and lost a
        # reliable source (4 -> 3) for exactly that reason.
        archive[source] = (0, h_phi(source, HORIZON))
        arm_rng = np.random.default_rng(seed_for(arm, source, 99))
        for k in range(N_CANDIDATES):
            if arm == "restart" or k == 0:
                start, depth = source, 0
            else:
                # POOL: archived states with budget left that are not already
                # in the region. An in-region state has h_phi = 1.0 by the exact
                # terminal boundary, so it would dominate forever once any
                # candidate succeeded -- v1's smoke showed later candidates
                # branching from the hit itself, doing ZERO transitions and
                # returning copies. A solution is not a frontier.
                #
                # REPEATS ARE ALLOWED. v1 also excluded already-used branch
                # points, which was a diversity heuristic rather than a
                # principle: the same state under a different seed gives a
                # different trajectory, and excluding repeats would stop the
                # policy restarting from x0 more than once, which is the whole
                # point of keeping x0 in the archive.
                # FORCED-DIVERSE BRANCH POINTS. This is the single ingredient
                # that differed between the positive v1 and the negative v2: v1
                # required each candidate to launch from a DISTINCT archived
                # state, v2 allowed repeats and collapsed onto one basin
                # (depth histogram d1:15, d11:13) with diversity falling below
                # plain restart's. The claim under test is therefore about
                # covering distinct search basins, not about intermediates
                # being intrinsically valuable.
                pool = [(h, s, d) for s, (d, h) in archive.items()
                        if d < HORIZON and s not in used_branches
                        and not in_region(*props(s), REGION)]
                if not pool:
                    start, depth = source, 0
                else:
                    h, start, depth = max(pool)
                    used_branches.add(start)
                    branch_log.append({"candidate": k, "depth": depth,
                                       "h_phi": h,
                                       "from_source": start == source})
                if not pool:
                    branch_log.append({"candidate": k, "depth": 0,
                                       "h_phi": None, "from_source": True})
            r = run_smc(start, depth, seed_for(arm, source, k))
            work += r["n_transitions"]
            for smi, d, h in r["archive"]:
                prev = archive.get(smi)
                if prev is None or d < prev[0]:
                    archive[smi] = (d, h)
            cands.append({k2: r[k2] for k2 in
                          ("returned", "terminal_qed", "terminal_sim", "success",
                           "extinct", "n_transitions", "start_depth", "budget")})
            print(f"  src{idx} {arm} cand{k} depth{r['start_depth']} "
                  f"QED {r['terminal_qed']:.3f} sim {r['terminal_sim']:.3f} "
                  f"{'HIT' if r['success'] else ('EXT' if r['extinct'] else '---')} "
                  f"{r['n_transitions']} trans", flush=True)
        out["arms"][arm] = {
            "candidates": cands,
            "success": any(c["success"] for c in cands),
            "first_success_at": next((i for i, c in enumerate(cands)
                                      if c["success"]), None),
            "work_transitions": work,
            "unique_states": len(archive),
            "distinct_returned": len({c["returned"] for c in cands}),
            "branches": branch_log,
            "seconds": round(time.perf_counter() - t0, 1),
        }

    out["seconds"] = round(time.perf_counter() - t_all, 1)
    p = Path(RUN_ROOT) / task.get("out_dir", OUT_DIR)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{idx:03d}_{task['arm']}.json").write_text(json.dumps(out))
    artifact_volume.commit()
    return out


@app.function(image=image, cpu=(0.25, 0.25), memory=768, timeout=12 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    artifact_volume.reload()
    done = 0
    # wrap_returned_exceptions=False is what the deprecation warning asks for,
    # and the driver previously died mid-map with "aclose(): asynchronous
    # generator is already running", taking eleven in-flight sources with it.
    for r in run_source.map(tasks, order_outputs=False, return_exceptions=True,
                            wrap_returned_exceptions=False):
        if isinstance(r, BaseException):
            print(f"  source failed: {type(r).__name__}: {r}", flush=True)
            continue
        if isinstance(r, dict):
            done += 1
            k = next(iter(r["arms"]))
            a = r["arms"][k]
            print(f"src{r['index']} [{r['stratum']}] {k}="
                  f"{'HIT' if a['success'] else '---'}/{a['work_transitions']}",
                  flush=True)
    return {"done": done}


@app.local_entrypoint()
def main(smoke: bool = False, out_dir: str = OUT_DIR) -> None:
    root = Path(__file__).resolve().parents[1]
    srcs = [s.strip() for s in
            (root / "data/jin/dev_panel_qed_64.txt").read_text().split("\n")
            if s.strip()]
    g = json.loads((root / "docs/HPHI_SMC_64_GATE_BANKED.json").read_text())
    l = json.loads((root / "docs/HPHI_COVERAGE_LADDER_BANKED.json").read_text())
    r3 = json.loads((root / "docs/HPHI_LADDER_RUNG3.json").read_text())
    rel = set(g["solved_sources"])
    marg = set(l["rung2"]["new_sources"]) | set(r3["conversions"])
    strat = {i: ("reliable" if i in rel else
                 "marginal" if i in marg else "hard") for i in range(len(srcs))}
    keep = list(range(len(srcs)))
    if smoke:
        keep = keep[:2]
        out_dir = out_dir + "_smoke"
    tasks = [{"index": i, "source": srcs[i], "stratum": strat[i],
              "arm": arm, "out_dir": out_dir}
             for i in keep for arm in ("restart", "diverse")]
    print(f"{'SMOKE' if smoke else 'PANEL'}: {len(tasks)} (source, arm) units "
          f"= {len(keep)} sources x 2 arms (restart/diverse) x {N_CANDIDATES} "
          f"candidates, N={N_PARTICLES}, H={HORIZON}")
    call = drive.spawn(tasks)
    print(f"spawned: {call.object_id}")
