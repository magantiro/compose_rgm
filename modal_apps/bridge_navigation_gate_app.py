"""R_theta vs endpoint-Doob bridge: can COMPOSE navigate to a distant molecule?

THE GATE. Retrieval said the bridge ranks true futures far above chance (top-10
61.5% at b=24, 51x). Retrieval is a surrogate. This asks whether the controller
can actually STEER: hidden held-out pair (x_0, z = x_b) from a frozen R_theta
trajectory, and two arms given identical budgets.

    A   y ~ R_theta(.|x)
    B   q(y|x,z,b) proportional to R_theta(y|x) exp s_psi(y, z, b-1)

If InfoNCE learned s_psi ~ log p(z|x,b)/p(z), then p(z) cancels in the
normalisation and B is the endpoint-conditioned Doob bridge of the frozen
process. COEFFICIENT IS 1 -- the learned temperature already sets the scale and
the density-ratio derivation fixes it. No bridge-strength sweep: tuning it after
seeing navigation would relocate the experiment back into hyperparameters.

MATCHED BY CONSTRUCTION. Both arms draw the SAME number of R_theta successors
per step (K) and differ only in which one is kept -- A uniformly, which is
exactly an R_theta draw, B by the bridge weight. Same goals, paired seeds, same
horizon.

APPROXIMATION, STATED: with K candidates the reweighting is a K-sample
approximation to q, exact only as K grows. K is small here because each
candidate costs a graph encode; a positive result understates what exact
reweighting would give, a null is confounded with K.

Success is EXACT canonical identity at the requested horizon, plus first-hit and
an INDEPENDENT terminal similarity (Morgan Tanimoto) -- never the bridge's own
embedding, which would be marking its own homework.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT, REMOTE_ROOT, artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

image = _base_image.env(
    {"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
app = modal.App("bridge-navigation-gate")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
CORPUS = "hphi_rollout_corpus/train_1024x02_H24.json.gz"
OUT_DIR = "bridge_v1"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
MEM_MIB = int(4.5 * 1024)
EMBED_DIM = 256
K_CAND, N_ROLLOUT = 8, 8
VAL_FRACTION = 0.15

_RT: dict[str, Any] = {}


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
    return _RT


@app.function(image=image, cpu=(2.0, 2.0), memory=MEM_MIB, timeout=6 * 60 * 60,
              max_containers=64, retries=1,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def navigate(task: dict[str, Any]) -> dict[str, Any]:
    import sys
    import numpy as np
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.hphi_lazy_helpers import make_helpers
    from compose_v4.experiments.hphi_lazy_sampler import sample_one_transition
    from compose_v4.experiments.production_successor_kernel import (
        _coordinate_action, canonical_state_key, _one_state_batch,
    )

    rt = _runtime()
    model, system = rt["model"], rt["system"]
    helpers = make_helpers(model, time_point=float(TIME_POINT),
                           canonical_slots=CANONICAL_SLOTS)
    _TF = {"grow_connected": "atom_insert"}
    ck = torch.load(Path(RUN_ROOT) / OUT_DIR / "bridge.pt", map_location="cpu")
    dim, nb = ck["dim"], ck["n_budget"]

    class Enc(nn.Module):
        def __init__(self, extra: int):
            super().__init__()
            self.f = nn.Sequential(
                nn.Linear(EMBED_DIM + extra, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, dim))

        def forward(self, e, b=None):
            if b is not None:
                e = torch.cat([e, b], dim=-1)
            v = self.f(e)
            return v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    gnet, hnet = Enc(nb), Enc(0)
    gnet.load_state_dict(ck["g"]); hnet.load_state_dict(ck["h"])
    gnet.eval(); hnet.eval()

    _st: dict[str, Any] = {}
    _emb: dict[str, Any] = {}

    def st_of(s):
        if s not in _st:
            _st[s] = pad_molecular_graph(smiles_to_molecular_graph(s),
                                         CANONICAL_SLOTS)
        return _st[s]

    def emb(s):
        if s not in _emb:
            b = _one_state_batch(model, st_of(s), float(TIME_POINT),
                                 prepared_batch=None)
            _n, gl, _p = model._encode_batch(b)
            _emb[s] = gl[0].detach().cpu().numpy().astype(np.float32)
        return _emb[s]

    def draw(smi, rng):
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
                                          coordinate=tuple(d.coordinate))
        y = canonical_state_key(system.apply(st, rule, action))
        return "" if y == smi else y

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    z, b0, x0 = task["goal"], int(task["budget"]), task["start"]
    zfp = gen.GetFingerprint(Chem.MolFromSmiles(z))
    hz = hnet(torch.tensor(emb(z))[None, :])

    def sim(s):
        m = Chem.MolFromSmiles(s)
        return 0.0 if m is None else float(
            DataStructs.TanimotoSimilarity(zfp, gen.GetFingerprint(m)))

    res: dict[str, Any] = {"goal": z, "start": x0, "budget": b0, "arms": {}}
    for arm in ("rtheta", "bridge"):
        hits, first, sims, traces = 0, [], [], []
        for r in range(N_ROLLOUT):
            rng = np.random.default_rng(int(task["seed"]) * 1000 + r)
            cur, hit_at, tr = x0, None, []
            for step in range(1, b0 + 1):
                cand = []
                for _ in range(K_CAND):
                    y = draw(cur, rng)
                    if y:
                        cand.append(y)
                if not cand:
                    break
                if arm == "rtheta":
                    cur = cand[int(rng.integers(len(cand)))]
                else:
                    rem = b0 - step
                    oh = torch.zeros(len(cand), nb)
                    oh[torch.arange(len(cand)), min(rem, nb - 1)] = 1.0
                    gv = gnet(torch.tensor(np.stack([emb(c) for c in cand])), oh)
                    s = (gv @ hz.T).squeeze(1).numpy()
                    p = np.exp(s - s.max()); p /= p.sum()
                    cur = cand[int(rng.choice(len(cand), p=p))]
                tr.append(sim(cur))
                if cur == z and hit_at is None:
                    hit_at = step
            if cur == z:
                hits += 1
            first.append(hit_at)
            sims.append(sim(cur))
            traces.append(tr)
        res["arms"][arm] = {
            "exact_hits": hits, "n_rollout": N_ROLLOUT,
            "first_hit": first,
            "terminal_sim_mean": float(np.mean(sims)),
            "terminal_sim_max": float(np.max(sims)),
            "sim_traces": traces,
        }
    return res


@app.function(image=image, cpu=(0.25, 0.25), memory=1024, timeout=8 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]], out_name: str) -> dict[str, Any]:
    out, started = [], time.perf_counter()
    for r in navigate.map(tasks, order_outputs=False, return_exceptions=True,
                          wrap_returned_exceptions=False):
        if isinstance(r, dict):
            out.append(r)
            a, b = r["arms"]["rtheta"], r["arms"]["bridge"]
            print(f"  b={r['budget']:>2} Rtheta hit {a['exact_hits']}/8 "
                  f"sim {a['terminal_sim_mean']:.3f} | bridge hit "
                  f"{b['exact_hits']}/8 sim {b['terminal_sim_mean']:.3f} "
                  f"[{len(out)}/{len(tasks)}]", flush=True)
        else:
            print(f"  FAILED {type(r).__name__}: {r}", flush=True)
    p = Path(RUN_ROOT) / "bridge_nav" / f"{out_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"results": out}))
    artifact_volume.commit()
    print(f"DONE {len(out)}/{len(tasks)} in {time.perf_counter()-started:.0f}s")
    return {"n": len(out)}


@app.function(image=image, cpu=(2.0, 2.0), memory=16384, timeout=60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def make_goals(n_per_budget: int = 32) -> list[dict]:
    """Held-out (x_0, z=x_b) pairs. Sources excluded from bridge training."""
    import numpy as np
    artifact_volume.reload()
    blob = json.loads(gzip.decompress(
        (Path(RUN_ROOT) / CORPUS).read_bytes()).decode())
    srcs = [r for r in blob["results"] if r.get("status") == "OK"]
    srcs.sort(key=lambda r: r["index"])
    n_val = int(len(srcs) * VAL_FRACTION)
    val = srcs[-n_val:]                     # the bridge never saw these
    rng = np.random.default_rng(11)
    tasks = []
    for b in (8, 16, 24):
        pool = []
        for r in val:
            for t in r["trajectories"]:
                p = t["path"]
                if len(p) > b and p[0] != p[b]:
                    pool.append((p[0], p[b]))
        idx = rng.choice(len(pool), size=min(n_per_budget, len(pool)),
                         replace=False)
        for i, j in enumerate(idx):
            x0, z = pool[int(j)]
            tasks.append({"start": x0, "goal": z, "budget": b,
                          "seed": 800000 + b * 1000 + i})
    print(f"goals: {len(tasks)} over budgets 8/16/24 from {len(val)} held-out "
          f"sources", flush=True)
    return tasks


@app.local_entrypoint()
def main(n_per_budget: int = 32) -> None:
    tasks = make_goals.remote(n_per_budget)
    print(f"NAVIGATION GATE: {len(tasks)} goals x 2 arms x {N_ROLLOUT} rollouts, "
          f"K={K_CAND} candidates/step, bridge coefficient 1 (no tuning).")
    call = drive.spawn(tasks, f"nav_{n_per_budget}")
    print(f"spawned: {call.object_id}")
