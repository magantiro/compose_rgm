"""Learned/heuristic PRIORITISATION + verified rollout, on Experiment C's universe.

THE QUESTION
------------
Not "can a learned value replace the planner" -- that was tested and failed.
Instead:

    which subset of the already-defined candidate set deserves the expensive
    full-horizon continuation?

Every hybrid arm shortlists K non-greedy challengers, ALWAYS keeps greedy, and
lets exact rollout make the final decision under the strict-improvement rule.
So under the rollout teacher's own value, no hybrid arm can do worse than
greedy -- the policy-improvement property direct override discarded.

THE ACTION UNIVERSE IS FROZEN TO EXPERIMENT C's
-----------------------------------------------
    greedy  UNION  top-4 similarity  UNION  top-2 R_theta      (deduplicated)

VERIFIED against the source of the 18/24 claim: CANDIDATES_TOP_SIMILARITY=4,
CANDIDATES_TOP_REFERENCE=2, and NO random stratum. The teacher collector added
2 random candidates; Experiment C did not. Scoring all ~500 legal successors
would introduce action-distribution shift -- h_phi was trained to discriminate
among curated plausible candidates and has never seen the tail -- so a failure
there could not be attributed to prioritisation rather than to
out-of-distribution deployment. Hence no expansion.

Because the universe holds only ~7 candidates and 5 of them are
similarity-selected (greedy is itself the similarity argmax), only K=1 and K=2
are genuinely discriminating; K=4 would already be most of the set.

K COUNTS NON-GREEDY CHALLENGERS
-------------------------------
    K=1 -> greedy + 1 challenger  -> at most 2 continuations, against ~7
    K=2 -> greedy + 2 challengers -> at most 3 continuations, against ~7

so the compute claim is transparent.

WHAT IS COUNTED AS EXPENSIVE
----------------------------
One V_G evaluation is one greedy continuation to the horizon. Evaluations are
counted LOGICALLY -- what a deployment would pay -- not net of the cache that
makes running eight arms in one container affordable.

CPU ONLY. Sealed 67 untouched.
"""

from __future__ import annotations

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

image = (
    _base_image
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
    .add_local_dir(ROOT / "artifacts/h_phi_frozen_v1",
                   str(REMOTE_ROOT / "artifacts/h_phi_frozen_v1"), copy=True)
    .add_local_file(
        ROOT / "diagnostics/editing_v2_controller_panel_seal.json",
        str(REMOTE_ROOT / "diagnostics/editing_v2_controller_panel_seal.json"),
        copy=True)
)
app = modal.App("compose-v4-h-phi-verified-hybrid")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT = 0.5
CANONICAL_SLOTS = 48
#: Experiment C's universe, verified against the app that produced 18/24.
CANDIDATES_TOP_SIMILARITY = 4
CANDIDATES_TOP_REFERENCE = 2
#: K counts NON-GREEDY challengers; greedy is always included separately.
ARMS = (("greedy", None), ("full", None),
        ("sim", 1), ("sim", 2), ("ref", 1), ("ref", 2), ("hphi", 1), ("hphi", 2))


@app.function(
    image=image, cpu=2.0, memory=12 * 1024, timeout=4 * 60 * 60,
    max_containers=26, retries=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_pair(task: dict[str, Any]) -> dict[str, Any]:
    import sys

    import numpy as np
    import torch
    from torch import nn

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.drd2_oracle import tanimoto_to
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
    from compose_v4.experiments.factorized_mark_conditional import (
        operator_capability_batch_kwargs,
    )
    from compose_v4.experiments.production_successor_kernel import (
        canonical_successor_result,
    )
    from compose_v4.model.factorized_tracelet_rate_model import (
        prepare_factorized_mark_batch,
    )
    from compose_v4.rewrite.kernel import canonical_state_key

    started = time.perf_counter()
    artifact_volume.reload()
    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    source = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]),
        repo_root=REMOTE_ROOT)
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=bundle)
    model = runtime.model
    checkpoint = torch.load(Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                            map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["selected_model_state"], strict=True)
    model.eval()

    frozen_dir = REMOTE_ROOT / "artifacts/h_phi_frozen_v1"
    contract = json.loads((frozen_dir / "H_PHI_FROZEN.json").read_text())
    if contract["feature_contract"]["canonical_slots"] != CANONICAL_SLOTS:
        raise RuntimeError("h_phi slot width disagrees with inference")

    class HPhi(nn.Module):
        def __init__(self, dim, hidden=256):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                       nn.Linear(hidden, hidden), nn.ReLU())
            self.recovery = nn.Linear(hidden, 1)
            self.similarity = nn.Linear(hidden, 1)

        def forward(self, x):
            h = self.trunk(x)
            return self.recovery(h).squeeze(-1), self.similarity(h).squeeze(-1)

    ensemble = []
    for member in contract["members"]:
        blob = torch.load(frozen_dir / member["path"], map_location="cpu",
                          weights_only=False)
        net = HPhi(blob["input_dim"], blob["hidden"])
        net.load_state_dict(blob["state_dict"])
        net.eval()
        ensemble.append(net)
    budget_max = int(contract["feature_contract"]["budget_max"])

    slots = int(task["slots"])
    budget = int(task["steps"])
    start_key = canonical_state_key(
        pad_molecular_graph(smiles_to_molecular_graph(task["source"]), slots))
    target_key = canonical_state_key(
        pad_molecular_graph(smiles_to_molecular_graph(task["target"]), slots))

    enumeration: dict[str, list[tuple[str, float]]] = {}
    similarity_cache: dict[str, float] = {}
    embedding: dict[str, Any] = {}
    value_cache: dict[tuple[str, int], tuple[bool, float]] = {}
    calls = 0

    def successors(key: str):
        nonlocal calls
        if key in enumeration:
            return enumeration[key]
        try:
            state = pad_molecular_graph(smiles_to_molecular_graph(key), slots)
        except Exception:  # noqa: BLE001
            enumeration[key] = []
            return []
        calls += 1
        with torch.no_grad():
            result = canonical_successor_result(model, state, float(TIME_POINT))
        enumeration[key] = [(s.key, float(s.probability))
                            for s in result.batch.successors]
        return enumeration[key]

    def sim(key: str) -> float:
        if key not in similarity_cache:
            similarity_cache[key] = (1.0 if key == target_key
                                     else tanimoto_to(target_key, key))
        return similarity_cache[key]

    def value(key: str, remaining: int):
        """V_G: greedy continuation to the horizon. THE expensive object."""

        if (key, remaining) in value_cache:
            return value_cache[(key, remaining)]
        current, best = key, sim(key)
        recovered = current == target_key
        for _ in range(remaining):
            if recovered:
                break
            rows = successors(current)
            if not rows:
                break
            current = max(rows, key=lambda row: (sim(row[0]), row[0]))[0]
            best = max(best, sim(current))
            recovered = current == target_key
        out = (recovered, 1.0 if recovered else best)
        value_cache[(key, remaining)] = out
        return out

    def embed(keys):
        missing = [k for k in keys if k not in embedding]
        if not missing:
            return
        graphs, usable = [], []
        for key in missing:
            try:
                graphs.append(pad_molecular_graph(
                    smiles_to_molecular_graph(key), CANONICAL_SLOTS))
                usable.append(key)
            except Exception:  # noqa: BLE001
                embedding[key] = None
        for start in range(0, len(usable), 32):
            chunk = usable[start:start + 32]
            states = tuple(graphs[usable.index(k)] for k in chunk)
            batch = prepare_factorized_mark_batch(
                states, tuple(float(TIME_POINT) for _ in chunk),
                tuple(None for _ in chunk), tuple(None for _ in chunk),
                tuple(0.0 for _ in chunk),
                use_aromatic_bond_view=True, ring_catalog=model.ring_catalog,
                **operator_capability_batch_kwargs(model.operator_capabilities))
            with torch.no_grad():
                _n, global_state, _p = model._encode_batch(batch)
            for key, vector in zip(chunk, global_state.cpu().numpy()):
                embedding[key] = vector.astype(np.float32)

    embed([target_key])
    e_target = embedding.get(target_key)
    if e_target is None:
        raise RuntimeError("target not representable at the canonical width")

    def learned_scores(keys, remaining):
        embed(keys)
        rows, usable = [], []
        for key in keys:
            e_y = embedding.get(key)
            if e_y is None:
                continue
            one_hot = np.zeros(budget_max, dtype=np.float32)
            one_hot[min(int(remaining), budget_max - 1)] = 1.0
            rows.append(np.concatenate([
                e_y, e_target, e_y - e_target, e_y * e_target,
                np.array([sim(key)], dtype=np.float32), one_hot]))
            usable.append(key)
        if not rows:
            return {}
        x = torch.from_numpy(np.asarray(rows, dtype=np.float32))
        with torch.no_grad():
            logits = torch.stack([m(x)[0] for m in ensemble]).mean(0).numpy()
        return dict(zip(usable, logits))

    def universe(keys, immediate, reference):
        """Experiment C's exact candidate construction."""

        greedy_index = max(range(len(keys)), key=lambda i: (immediate[i], keys[i]))
        picked = [greedy_index]
        seen = {greedy_index}
        for stratum, count in ((-immediate, CANDIDATES_TOP_SIMILARITY),
                               (-reference, CANDIDATES_TOP_REFERENCE)):
            for index in np.argsort(stratum)[:count]:
                if int(index) not in seen:
                    picked.append(int(index))
                    seen.add(int(index))
        return greedy_index, picked

    def arm(mode: str, k: int | None):
        current, best = start_key, sim(start_key)
        recovered = current == target_key
        evaluations = 0
        overrides = 0
        universe_seen = 0
        for step in range(budget):
            if recovered:
                break
            rows = successors(current)
            if not rows:
                break
            keys = [r[0] for r in rows]
            immediate = np.array([sim(k2) for k2 in keys])
            reference = np.array([r[1] for r in rows])
            greedy_index, pool = universe(keys, immediate, reference)
            universe_seen += len(pool)

            if mode == "greedy":
                chosen = greedy_index
            else:
                if mode == "full":
                    shortlist = list(pool)
                else:
                    others = [i for i in pool if i != greedy_index]
                    if mode == "sim":
                        order = sorted(others, key=lambda i: (-immediate[i], keys[i]))
                    elif mode == "ref":
                        order = sorted(others, key=lambda i: (-reference[i], keys[i]))
                    else:
                        scores = learned_scores([keys[i] for i in others],
                                                budget - step - 1)
                        order = sorted(others,
                                       key=lambda i: -scores.get(keys[i], -1e9))
                    shortlist = [greedy_index] + order[:k]
                # Logical cost: what a deployment pays, not net of cache.
                evaluations += len(shortlist)
                values = {i: value(keys[i], budget - step - 1) for i in shortlist}
                greedy_value = values[greedy_index]
                challengers = {i: v for i, v in values.items() if v > greedy_value}
                chosen = (max(challengers,
                              key=lambda i: (challengers[i][0], challengers[i][1], -i))
                          if challengers else greedy_index)
                if chosen != greedy_index:
                    overrides += 1
            current = keys[chosen]
            best = max(best, sim(current))
            recovered = current == target_key
        return {"recovered": recovered, "best": 1.0 if recovered else best,
                "continuation_evaluations": evaluations, "overrides": overrides,
                "universe_candidates_seen": universe_seen}

    results = {}
    for mode, k in ARMS:
        name = mode if k is None else f"{mode}{k}"
        results[name] = arm(mode, k)

    seconds = time.perf_counter() - started
    print(f"  pair {task['index']}: "
          + "  ".join(f"{n}={'Y' if r['recovered'] else 'n'}/{r['continuation_evaluations']}e"
                      for n, r in results.items())
          + f"  ({calls} calls, {seconds:.0f}s)", flush=True)

    payload = {"index": task["index"], "source": start_key, "target": target_key,
               "verified_steps": budget, "arms": results,
               "kernel_calls": calls, "seconds": round(seconds, 1)}
    out = Path(RUN_ROOT) / "h_phi_verified_hybrid"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{task['index']:03d}.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    artifact_volume.commit()
    return payload


@app.function(image=image, cpu=1.0, memory=4 * 1024, timeout=8 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    results = [r for r in run_pair.map(tasks) if r]
    out = Path(RUN_ROOT) / "h_phi_verified_hybrid" / "aggregate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pairs": len(results), "per_pair": results},
                              indent=2) + "\n")
    artifact_volume.commit()
    return {"pairs": len(results)}


@app.local_entrypoint()
def main() -> None:
    seal = json.loads(
        (ROOT / "diagnostics/editing_v2_controller_panel_seal.json").read_text())
    tasks = [{**row, "index": i} for i, row in enumerate(seal["development"])]
    print(f"verified hybrid on {len(tasks)} DEVELOPMENT pairs "
          f"({seal['commitment']['development_sha256'][:16]}); sealed 67 untouched")
    print(f"universe = greedy + top-{CANDIDATES_TOP_SIMILARITY} similarity + "
          f"top-{CANDIDATES_TOP_REFERENCE} R_theta  (Experiment C's exact set)")
    print(f"arms: {[m if k is None else f'{m}{k}' for m, k in ARMS]}")
    print(drive.remote(tasks))
