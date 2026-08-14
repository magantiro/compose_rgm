"""THE STEERING TEST: does R_theta * h_phi actually turn the car?

The final pre-scale qualification. The smoke showed h_phi is informative -- it
uses the goal descriptor, covers every budget, respects the exact boundary, and
recovered the difficulty ordering 0.75 > 0.80 > 0.85 > 0.90 > 0.95 on held-out
sources. None of that proves it knows WHICH MOLECULAR ROADS TO TAKE. A model
can learn base-rate difficulty and still be useless for control.

FROZEN BEFORE THE RUN
---------------------
  sources     held-out sources of the already-excluded H24 pilot
  R_theta     identical frozen checkpoint
  arms        unguided R_theta   vs   R_theta * h_phi
  seeds       PAIRED across arms
  inference   Stage-A1 exact rejection sampler, native first-hit STOP
  regions     QED >= 0.80 and >= 0.85 at sim >= 0.40 qualify
  0.90/0.40   reported DESCRIPTIVELY, never a gate
  no tuning   no temperature, no top-k, no shortlist, no horizon change

PASS is a COHERENT DIRECTIONAL EFFECT, not an arbitrary percentage: controlled
above unguided on 0.80 and 0.85, preferably earlier first hits, no similarity
collapse, no support or validity violation.

THE DECISION-LEVEL DIAGNOSTIC, and it is free
----------------------------------------------
The rejection sampler's proposals ARE draws from R_theta, so the rejected ones
give the base expectation at no extra cost:

    h_phi(y_chosen)   vs   E_{y ~ R_theta}[ h_phi(y) ]

If the chosen successor beats the base expectation at a substantial fraction of
decisions, the sampler is genuinely taking edits the learned GPS considers more
reachable -- rather than an endpoint effect arising by accident. If the endpoint
effect is weak, this separates the two failure modes:

  1  h_phi does not distinguish successors WITHIN a state despite learning
     global goal difficulty
  2  the sampler/control integration does not translate those distinctions
     into trajectory behaviour

HORIZON. This runs at H6, a frozen reported operating point, because the exact
rejection sampler pays one R_theta encode per PROPOSAL and acceptance is
expected to be low. Trajectory length is not what is being qualified here --
steering is.
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
).add_local_dir(ROOT / "data/jin", str(REMOTE_ROOT / "data/jin"), copy=True)

app = modal.App("hphi-steering-test")
RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
CORPUS = "hphi_rollout_corpus/pilot_0064x04_H24.json.gz"
OUT_DIR = "hphi_steering"
TIME_POINT, CANONICAL_SLOTS = 0.5, 48
HORIZON = 6
TRAJ_PER_SOURCE = 6
#: Bounds the rejection loop. Hitting it is RECORDED, never silently patched.
MAX_PROPOSALS = 40
REGION = (0.80, 0.40)          # the primary qualification region

_RT: dict[str, Any] = {}


def seed_for(src: str, rep: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"steer-v1|{src}|{rep}".encode()).digest()[:8], "big")


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
    _RT["model"] = model
    _RT["system"] = _default_rewrite_system(model)
    return _RT


@app.function(image=image, cpu=(1.0, 1.0), memory=8192, timeout=4 * 60 * 60,
              max_containers=64, retries=2,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def steer_source(task: dict[str, Any]) -> dict[str, Any]:
    """Both arms from one source, PAIRED seeds, H6, exact rejection sampler."""
    import sys

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import QED, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    rt = _runtime()
    model, system = rt["model"], rt["system"]
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.hphi_region_features import (
        build_features, h_with_boundary, in_region,
    )
    from compose_v4.experiments.production_successor_kernel import (
        _one_state_batch, canonical_state_key, enumerate_factorized_marked_law,
    )

    head = torch.jit.load(str(Path(RUN_ROOT) / OUT_DIR / "head.pt"))
    norm = json.loads((Path(RUN_ROOT) / OUT_DIR / "norm.json").read_text())
    mu = np.asarray(norm["mu"]); sd = np.asarray(norm["sd"])

    source = task["smiles"]
    mol0 = Chem.MolFromSmiles(source)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp0 = gen.GetFingerprint(mol0)
    enc_cache: dict[str, Any] = {}
    prop_cache: dict[str, tuple[float, float]] = {}
    t0 = time.perf_counter()

    def encode(smi: str):
        v = enc_cache.get(smi)
        if v is None:
            st = pad_molecular_graph(smiles_to_molecular_graph(smi), CANONICAL_SLOTS)
            b = _one_state_batch(model, st, float(TIME_POINT), prepared_batch=None)
            with torch.no_grad():
                _n, g, _p = model._encode_batch(b)
            v = g[0].detach().cpu().numpy().astype(np.float64)
            enc_cache[smi] = v
        return v

    def props(smi: str):
        v = prop_cache.get(smi)
        if v is None:
            m = Chem.MolFromSmiles(smi)
            v = ((float(QED.qed(m)),
                  float(DataStructs.TanimotoSimilarity(fp0, gen.GetFingerprint(m))))
                 if m is not None else (0.0, 0.0))
            prop_cache[smi] = v
        return v

    e_src = encode(source)

    def h_of(smi: str, budget: int) -> float:
        q, s = props(smi)
        if in_region(q, s, REGION):
            return 1.0
        f = build_features(encode(smi), e_src, q, s, REGION, budget)
        x = torch.tensor(((f - mu) / sd), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            raw = float(torch.sigmoid(head(x)).item())
        return h_with_boundary(raw, q, s, REGION)

    def run(controlled: bool, rep: int):
        rng = np.random.default_rng(seed_for(source, rep))
        cur, key = source, canonical_state_key(
            pad_molecular_graph(smiles_to_molecular_graph(source), CANONICAL_SLOTS))
        path = [cur]
        decisions, cap_hits, encodes0 = [], 0, len(enc_cache)
        stopped = False
        for step in range(HORIZON):
            q, s = props(cur)
            if in_region(q, s, REGION):          # native first-hit STOP
                stopped = True
                break
            st = pad_molecular_graph(smiles_to_molecular_graph(cur), CANONICAL_SLOTS)
            law = enumerate_factorized_marked_law(model, st, float(TIME_POINT))
            if not law.marks:
                break
            p = np.array([m.probability for m in law.marks], float); p /= p.sum()
            b_rem = HORIZON - step - 1
            chosen, hs = None, []
            for _ in range(MAX_PROPOSALS):
                i = int(rng.choice(len(p), p=p))
                mk = law.marks[i]
                y = canonical_state_key(system.apply(st, mk.executor_rule_name,
                                                     mk.action))
                if y == key:
                    continue                      # virtual
                if not controlled:
                    chosen = y
                    break
                hv = h_of(y, b_rem)
                hs.append(hv)                     # proposals ARE R_theta draws
                if rng.random() <= hv:
                    chosen = y
                    break
            if chosen is None:
                cap_hits += 1
                break
            if controlled and hs:
                decisions.append({"h_chosen": hs[-1],
                                  "h_base_mean": float(np.mean(hs)),
                                  "n_proposals": len(hs)})
            cur = chosen
            key = chosen
            path.append(cur)
        qs = [props(x) for x in path]
        return {"replicate": rep, "path": path, "stopped_early": stopped,
                "qed": [q for q, _ in qs], "sim": [s for _, s in qs],
                "decisions": decisions, "cap_hits": cap_hits,
                "encodes": len(enc_cache) - encodes0}

    unguided = [run(False, r) for r in range(TRAJ_PER_SOURCE)]
    controlled = [run(True, r) for r in range(TRAJ_PER_SOURCE)]
    return {"index": task["index"], "source": source,
            "unguided": unguided, "controlled": controlled,
            "seconds": round(time.perf_counter() - t0, 1), "status": "OK"}


@app.local_entrypoint()
def main() -> None:
    print("STEERING TEST -- does R_theta * h_phi turn the car?")
    print(f"held-out pilot sources, PAIRED seeds, H{HORIZON}, region {REGION}")
    print("0.90 reported descriptively; no tuning of any kind")
    print("NOTE: requires head.pt + norm.json written by the qualification "
          "trainer; run that first.")
