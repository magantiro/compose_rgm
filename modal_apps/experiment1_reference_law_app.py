"""Experiment 1 / Claim 1: did learning produce a useful transition law?

Five arms, ONE evaluator, ONE frozen sample, the SAME canonical-successor
interface. Every arm scores the probability the teacher's observed canonical
successor receives, so the numbers are directly comparable.

    uniform_canonical            1 / |N+(x)|
    empirical_family             sum over families k reaching y of
                                 qhat(k) / |N_k(x)|
    learned_family_uniform_id    P_theta(F|x) / |N_F(x)|
    empirical_family_learned_id  qhat(F) . P_theta(y|x,F)
    r_theta                      P_theta(F|x) . P_theta(y|x,F)

qhat is the frozen training law's realized family distribution, RESTRICTED to
families with legal productive successors at x and renormalized there. That is
the strong form of the baseline: it gets the correct global operator
frequencies, the exact legal support, state-dependent availability, and alias
aggregation. What it does NOT get is learned molecular context -- so
r_theta > empirical_family means the network learned state-dependent chemistry
rather than corpus-level family frequencies, which is the actual claim.

The families reaching y come from the teacher fiber's own aliases, so a
successor reachable through several operators contributes from each, summed at
the CANONICAL SUCCESSOR level rather than double counted.

DEVELOPMENT RESULT. Scored on the step-12,500 development reference checkpoint,
which is not paper-bearing: it predates the code-commit binding. Every number
here is for deciding whether the claim holds and what the final protocol should
be, and gets rerun under that protocol before it is claimed.
"""

from __future__ import annotations

import collections
import gzip
import json
import math
import statistics
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

image = _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
app = modal.App("compose-v4-experiment1-reference-law")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
#: Below this a cell mean is noise; reported as unavailable rather than as a number.
MINIMUM_CELL = 20


@app.function(
    image=image,
    cpu=8.0,
    memory=64 * 1024,
    timeout=60 * 60,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def evaluate_reference_law(batch_size: int = 32) -> dict[str, Any]:
    import torch

    from compose_v4.data.corpus_training_library import load_corpus_training_library
    from compose_v4.data.packed_collated_batch import PackedCollatedStore
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state,
    )
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME,
        evaluate_panel,
    )
    from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
        _attach_successor_family_coordinates,
    )

    started = time.perf_counter()
    artifact_volume.reload()
    root = Path(RUN_ROOT)
    inputs = root / "run_inputs"
    paths = json.loads((inputs / "RUN_PATHS.json").read_text())
    device = torch.device("cpu")

    # ---- partitions ------------------------------------------------------
    partition: dict[str, dict[str, Any]] = {}
    shard_failures = []
    for shard in sorted((root / "partitions").glob("shard-*.json")):
        payload = json.loads(shard.read_text())
        shard_failures.extend(payload.get("failures", []))
        for row in payload["rows"]:
            partition[row["source"]] = row
    print(f"[{time.perf_counter()-started:6.1f}s] partitions for {len(partition):,} "
          f"sources ({len(shard_failures)} compile failures)", flush=True)
    if not partition:
        raise RuntimeError("no partitions found; run compile_successor_partitions first")

    # ---- qhat, from the REALIZED FROZEN LAW, not raw corpus counts -------
    law = json.loads((inputs / "editing_v2_sampling_law_v2.json").read_text())
    q = dict(law["realized_coefficients"]["by_family"])
    print(f"  q(k) over {len(q)} families from law {law['frozen_sha256'][:12]}", flush=True)

    # ---- model + sample --------------------------------------------------
    source = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]),
        repo_root=REMOTE_ROOT,
    )
    state = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=state)
    model = runtime.model.to(device)
    checkpoint = torch.load(root / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
                            map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["selected_model_state"], strict=True)
    model.eval()
    print(f"[{time.perf_counter()-started:6.1f}s] development checkpoint step "
          f"{checkpoint['selected_step']:,}", flush=True)

    resolution = json.loads(
        (inputs / "editing_v2_split_precedence_resolution.json").read_text())
    library = load_corpus_training_library(
        [Path(r) for r in paths["corpus_roots"]],
        excluded_sources=resolution["excluded_source_keys"]["train"],
        verify_state_roundtrip=False)
    store = PackedCollatedStore(Path(paths["packed_store"]))
    template = torch.load(Path(paths["packed_store"]) / "TEMPLATE.pt",
                          map_location="cpu", weights_only=False)
    with gzip.open(
            inputs / "editing_v2_matched_validation_reserve_ids.json.gz", "rt") as handle:
        reserve = json.load(handle)
    bands = reserve["reserve_band_by_entry_id"]
    reserve_ids = set(reserve["reserve_entry_ids"])

    # Sampled entries: those in the reserve whose SOURCE has a partition.
    entries = [e for e in library.entries
               if e.entry_id in reserve_ids
               and e.teacher_fiber.state_support.source_key in partition]
    ids = [e.entry_id for e in entries]
    print(f"[{time.perf_counter()-started:6.1f}s] scoring {len(ids):,} entries", flush=True)

    def build(entry_ids):
        states, fibers, chosen = library.inputs_for(entry_ids)
        count = len(entry_ids)
        batch = store.rows_for(entry_ids, template, extra={
            "states": tuple(states),
            "times": torch.tensor([float.fromhex(e.support_time_hex) for e in chosen],
                                  dtype=torch.float32),
            "teacher_rates": torch.ones(count, dtype=torch.float32),
            "importance_weights": torch.ones(count, dtype=torch.float32)})
        batch = _attach_successor_family_coordinates(
            batch, [{"model_family": e.model_family} for e in chosen])
        return batch.to(device), fibers, chosen

    rows = evaluate_panel(model, entry_ids=ids, build_batch=build, batch_size=batch_size)
    print(f"[{time.perf_counter()-started:6.1f}s] scored", flush=True)

    fiber_of = {e.entry_id: e.teacher_fiber for e in entries}
    source_of = {e.entry_id: e.teacher_fiber.state_support.source_key for e in entries}

    scored, skipped = [], 0
    for row in rows:
        entry_id = row["entry_id"]
        part = partition[source_of[entry_id]]
        per_family = part["per_family_successor_count"]
        total = int(part["canonical_successor_count"])
        if total <= 0:
            skipped += 1
            continue
        family = row["model_family"]
        reaching = {a.family_name for a in fiber_of[entry_id].aliases}

        # qhat: restricted to families LEGAL AT x, renormalized there.
        legal = {k: q.get(k, 0.0) for k in per_family if per_family[k] > 0}
        mass = sum(legal.values())
        qhat = {k: v / mass for k, v in legal.items()} if mass > 0 else {}

        p_uniform = 1.0 / total
        p_empirical = sum(qhat.get(k, 0.0) / per_family[k]
                          for k in reaching if per_family.get(k, 0) > 0)
        family_probability = math.exp(-row["family_nll"])
        identity_probability = math.exp(-row["identity_nll"])
        p_learned_family_uniform_id = (
            family_probability / per_family[family] if per_family.get(family, 0) > 0 else 0.0)
        p_empirical_family_learned_id = qhat.get(family, 0.0) * identity_probability
        p_r_theta = row["teacher_successor_probability"]

        def nll(p):
            return -math.log(p) if p > 0 else float("inf")

        scored.append({
            "entry_id": entry_id, "family": family,
            "cell": row["capability_cell_id"], "band": bands.get(entry_id),
            "canonical_successor_count": total,
            "uniform_canonical": nll(p_uniform),
            "empirical_family": nll(p_empirical),
            "learned_family_uniform_id": nll(p_learned_family_uniform_id),
            "empirical_family_learned_id": nll(p_empirical_family_learned_id),
            "r_theta": nll(p_r_theta),
        })

    ARMS = ("uniform_canonical", "empirical_family", "learned_family_uniform_id",
            "empirical_family_learned_id", "r_theta")

    def summarize(items):
        out = {"entries": len(items)}
        for arm in ARMS:
            finite = [i[arm] for i in items if math.isfinite(i[arm])]
            out[arm] = statistics.mean(finite) if finite else None
            out[f"{arm}_infinite"] = len(items) - len(finite)
        return out

    overall = summarize(scored)
    print(f"\n{'arm':32} {'mean NLL':>10}  {'vs R_theta':>11}")
    for arm in ARMS:
        gap = "" if arm == "r_theta" else f"{overall[arm] - overall['r_theta']:+11.4f}"
        print(f"  {arm:30} {overall[arm]:10.4f}  {gap}")

    def group(key):
        buckets = collections.defaultdict(list)
        for item in scored:
            buckets[item[key]].append(item)
        return {name: summarize(items) for name, items in sorted(buckets.items())
                if len(items) >= MINIMUM_CELL}

    return {
        "schema": "compose.editing_v2.experiment1_reference_law",
        "status": "DEVELOPMENT_RESULT_NOT_PAPER_BEARING",
        "checkpoint": {"run": "run_v2_01", "selected_step": int(checkpoint["selected_step"]),
                       "note": "development reference checkpoint; predates the code-commit binding"},
        "law_frozen_sha256": law["frozen_sha256"],
        "sources_with_partitions": len(partition),
        "partition_compile_failures": len(shard_failures),
        "entries_scored": len(scored),
        "entries_skipped": skipped,
        "minimum_cell": MINIMUM_CELL,
        "overall": overall,
        "by_family": group("family"),
        "by_support_band": group("band"),
        "by_capability_cell": group("cell"),
        "reading": (
            "r_theta below uniform_canonical shows learning beats the executable "
            "kernel alone. r_theta below empirical_family is the claim that "
            "matters: the network learned state-dependent chemistry rather than "
            "corpus-level operator frequencies. The two middle arms decompose it "
            "into choosing WHICH edit versus choosing WHERE to apply it."),
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(evaluate_reference_law.remote(), indent=2, sort_keys=True))
