"""Closed-loop h_phi on the 24 development pairs, across the fixed threshold grid.

WHY THIS RUN EXISTS
-------------------
The offline analysis showed no override threshold with a positive state-level
rescue-minus-harm balance on endpoint-disjoint validation. That is a warning,
not the result: the objective is PAIR-LEVEL exact target recovery, and offline
state counts can misestimate it because

  - h_phi changes which states are visited, so the validation states (collected
    from greedy and rollout trajectories) are not the states it would meet;
  - one early harmful decision can dominate an entire trajectory;
  - several rescues inside one pair still recover only one target;
  - states within a pair are not independent.

So the learned controller is run closed-loop, exactly as the 24 development
pairs were reserved for.

THE GRID IS FIXED IN ADVANCE
----------------------------
Thresholds are the same grid already swept offline (0, 0.5, 1, 1.5, 2, 3, 4, 6).
No threshold is invented after seeing pair outcomes; choosing among these is the
single job of the 24, and it happens after this run reports all of them.

THE SEALED 67 ARE NOT TOUCHED.

CONSISTENCY THAT MATTERS
------------------------
Greedy's action is max by (similarity, canonical key) -- Experiment C's
convention. The teacher collector used np.argmax, which breaks ties by first
index and disagrees on 3 of 415 validation states; using it here would compare
against a greedy policy that was never run.

h_phi embeddings are computed at CANONICAL_SLOTS with the training time point.
The R_theta embedding is NOT padding-invariant, so a different width would
silently shift every feature away from what the model was trained on.

All eight thresholds run inside one container per pair, sharing the enumeration
and embedding caches: trajectories agree early and diverge late, so the shared
prefix is computed once.

CPU ONLY.
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
    .add_local_file(
        ROOT / "diagnostics/editing_v2_teacher_cohort_freeze.json",
        str(REMOTE_ROOT / "diagnostics/editing_v2_teacher_cohort_freeze.json"),
        copy=True)
)
app = modal.App("compose-v4-h-phi-closed-loop")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT = 0.5
CANONICAL_SLOTS = 48
CANDIDATES_TOP_SIMILARITY = 4
CANDIDATES_TOP_REFERENCE = 2
CANDIDATES_RANDOM = 2
#: Fixed before this run; identical to the offline sweep.
THRESHOLDS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@app.function(
    image=image, cpu=2.0, memory=12 * 1024, timeout=4 * 60 * 60,
    max_containers=26, retries=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def run_pair(task: dict[str, Any]) -> dict[str, Any]:
    """Greedy and h_phi-at-every-threshold on one held-out transformation."""

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

    # ---- frozen h_phi ensemble ------------------------------------------
    frozen_dir = REMOTE_ROOT / "artifacts/h_phi_frozen_v1"
    contract = json.loads((frozen_dir / "H_PHI_FROZEN.json").read_text())
    if contract["feature_contract"]["canonical_slots"] != CANONICAL_SLOTS:
        raise RuntimeError(
            f"h_phi was trained at {contract['feature_contract']['canonical_slots']} "
            f"slots but inference uses {CANONICAL_SLOTS}; the embedding is not "
            f"padding-invariant so this would silently shift every feature")

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

    rng = np.random.default_rng(task["index"] + 90210)
    slots = int(task["slots"])
    budget = int(task["steps"])
    start_key = canonical_state_key(
        pad_molecular_graph(smiles_to_molecular_graph(task["source"]), slots))
    target_key = canonical_state_key(
        pad_molecular_graph(smiles_to_molecular_graph(task["target"]), slots))

    enumeration: dict[str, list[tuple[str, float]]] = {}
    embedding: dict[str, np.ndarray] = {}
    similarity_cache: dict[str, float] = {}
    calls = 0

    def successors(key: str) -> list[tuple[str, float]]:
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

    def embed(keys: list[str]) -> None:
        """Embed at the TRAINING width; anything else shifts every feature."""

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
        raise RuntimeError("target is not representable at the canonical width")

    def score(keys: list[str], remaining: int):
        """Ensemble-mean recovery logit and similarity for each candidate."""

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
            return {}, {}
        x = torch.from_numpy(np.asarray(rows, dtype=np.float32))
        with torch.no_grad():
            logits = torch.stack([m(x)[0] for m in ensemble]).mean(0).numpy()
            sims = torch.stack([m(x)[1] for m in ensemble]).mean(0).numpy()
        return dict(zip(usable, logits)), dict(zip(usable, sims))

    def candidate_set(keys, immediate, reference):
        greedy_index = max(range(len(keys)),
                           key=lambda i: (immediate[i], keys[i]))
        picked = [greedy_index]
        seen = {greedy_index}
        for stratum, count in ((-immediate, CANDIDATES_TOP_SIMILARITY),
                               (-reference, CANDIDATES_TOP_REFERENCE)):
            for index in np.argsort(stratum)[:count]:
                if int(index) not in seen:
                    picked.append(int(index))
                    seen.add(int(index))
        rest = [i for i in range(len(keys)) if i not in seen]
        if rest:
            for index in rng.choice(rest, size=min(CANDIDATES_RANDOM, len(rest)),
                                    replace=False):
                picked.append(int(index))
        return greedy_index, picked

    def rollout(threshold: float | None) -> dict[str, Any]:
        """threshold None = pure greedy; otherwise h_phi may override."""

        current, best, overrides = start_key, sim(start_key), 0
        recovered = current == target_key
        trail = [current]
        for step in range(budget):
            if recovered:
                break
            rows = successors(current)
            if not rows:
                break
            keys = [r[0] for r in rows]
            immediate = np.array([sim(k) for k in keys])
            reference = np.array([r[1] for r in rows])
            greedy_index, picked = candidate_set(keys, immediate, reference)
            chosen = greedy_index
            if threshold is not None:
                subset = [keys[i] for i in picked]
                logits, sims = score(subset, budget - step - 1)
                greedy_key = keys[greedy_index]
                if greedy_key in logits:
                    ranked = sorted(subset,
                                    key=lambda k: (logits.get(k, -1e9),
                                                   sims.get(k, -1e9)),
                                    reverse=True)
                    top = ranked[0]
                    if logits[top] - logits[greedy_key] > threshold:
                        chosen = keys.index(top)
                        overrides += 1
            current = keys[chosen]
            trail.append(current)
            best = max(best, sim(current))
            recovered = current == target_key
        return {"recovered": recovered, "best": 1.0 if recovered else best,
                "final": sim(current), "overrides": overrides,
                "steps": len(trail) - 1}

    greedy = rollout(None)
    arms = {str(t): rollout(t) for t in THRESHOLDS}
    seconds = time.perf_counter() - started
    print(f"  pair {task['index']}: greedy={greedy['recovered']}  "
          + "  ".join(f"t{t}={arms[str(t)]['recovered']}"
                      f"/{arms[str(t)]['overrides']}o" for t in THRESHOLDS)
          + f"  ({calls} calls, {seconds:.0f}s)", flush=True)

    payload = {"index": task["index"], "source": start_key, "target": target_key,
               "verified_steps": budget, "greedy": greedy, "by_threshold": arms,
               "kernel_calls": calls, "seconds": round(seconds, 1)}
    out = Path(RUN_ROOT) / task.get("out_dir", "h_phi_closed_loop")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{task['index']:03d}.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    artifact_volume.commit()
    return payload


@app.function(image=image, cpu=1.0, memory=4 * 1024, timeout=8 * 60 * 60,
              volumes={str(ARTIFACT_ROOT): artifact_volume})
def drive(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Fan out server-side so a client disconnect cannot stall the run."""

    results = [r for r in run_pair.map(tasks) if r]
    out = Path(RUN_ROOT) / task.get("out_dir", "h_phi_closed_loop") / "aggregate.json"
    out.write_text(json.dumps({"pairs": len(results), "per_pair": results},
                              indent=2) + "\n")
    artifact_volume.commit()
    return {"pairs": len(results)}


@app.local_entrypoint()
def main(panel: str = "development", pairs: int = 24) -> None:
    """panel='development' (the 24) or 'heldin' (targets h_phi TRAINED on).

    The held-in panel is the sharp diagnostic: if h_phi beats greedy closed-loop
    on targets it trained on, the failure is cross-target TRANSFER. If it fails
    even there despite ~88% offline training rescue, then pointwise offline
    prediction is misaligned with sequential deployment regardless of
    generalisation -- and pairwise training alone would be unlikely to fix it.
    """

    if panel == "development":
        seal = json.loads(
            (ROOT / "diagnostics/editing_v2_controller_panel_seal.json").read_text())
        rows = seal["development"]
        out_dir, tag = "h_phi_closed_loop", seal["commitment"]["development_sha256"]
    else:
        cohort = json.loads(
            (ROOT / "diagnostics/editing_v2_teacher_cohort_freeze.json").read_text())
        # Stratified across horizons, matching the development panel's 8/8/8.
        # Taking the head of the cohort list would give all 4-step pairs, since
        # it is ordered by band -- not comparable to what greedy faces on the 24.
        rows = []
        per = max(1, pairs // 3)
        for length in (4, 5, 6):
            rows.extend([r for r in cohort["train"] if r["steps"] == length][:per])
        out_dir, tag = "h_phi_closed_loop_heldin", cohort["commitment"]["train_sha256"]
    tasks = [{**row, "index": i, "out_dir": out_dir}
             for i, row in enumerate(rows[:pairs])]
    print(f"closed-loop on {len(tasks)} {panel.upper()} pairs ({tag[:16]}); "
          f"the sealed 67 are untouched")
    print(f"thresholds fixed in advance: {THRESHOLDS}")
    print(drive.remote(tasks, out_dir))
    print(f"  results under editing_v2/r_theta_run/{out_dir}/")
