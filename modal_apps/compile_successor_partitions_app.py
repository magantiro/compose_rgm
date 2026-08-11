"""Exact productive canonical-successor partitions for the Experiment-1 sample.

WHAT THIS PRODUCES, AND WHY NOTHING ELSE CAN
--------------------------------------------
Claim 1 needs P_unif(y|x) = 1/|N+(x)| over DISTINCT PRODUCTIVE CANONICAL
successors, and the empirical-family arm needs the per-family counts |N_k(x)|.
Neither is stored anywhere: the library carries teacher fibers only, the raw
chunks carry per-family MARK counts (many marks collapse to one successor), and
`StateProductiveSupport.virtual_aliases` is the self-event set, not the support.

The one available shortcut was tested and rejected. On the 1,873-row frozen
capacity census, canonical_successor_count equals raw_legal_mark_count for only
17.4% of states; the ratio has mean 0.899 and falls to 0.482. Using raw mark
counts would bias the uniform arm by ~0.1 nats typically and 0.7 at worst, and
it would bias IN OUR FAVOUR -- raw > canonical makes uniform look worse and
inflates R_theta's advantage. So the partition is computed exactly.

CPU ONLY. No GPU: this is executor work, not model scoring. The model is
required only because the marked law is enumerated through it.

Runs on the VOLUME's Active8. The local copy authenticates against gate-zero v7
and the volume's against v6, and they differ -- so counts computed locally would
not correspond to the corpus R_theta trained on.
"""

from __future__ import annotations

import collections
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

image = _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
app = modal.App("compose-v4-successor-partitions")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"


@app.function(
    image=image,
    cpu=2.0,
    memory=16 * 1024,
    timeout=60 * 60,
    max_containers=40,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def compile_shard(shard: dict[str, Any]) -> dict[str, Any]:
    import torch

    from compose_v4.data.corpus_training_library import load_corpus_training_library
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state,
    )
    from compose_v4.experiments.factorized_successor_training import (
        compile_state_successor_map,
    )

    started = time.perf_counter()
    artifact_volume.reload()
    inputs = Path(RUN_ROOT) / "run_inputs"
    paths = json.loads((inputs / "RUN_PATHS.json").read_text())

    source = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]),
        repo_root=REMOTE_ROOT,
    )
    state = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=state)
    model = runtime.model
    resolution = json.loads(
        (inputs / "editing_v2_split_precedence_resolution.json").read_text())
    library = load_corpus_training_library(
        [Path(r) for r in paths["corpus_roots"]],
        excluded_sources=resolution["excluded_source_keys"]["train"],
        verify_state_roundtrip=False)
    print(f"[{time.perf_counter()-started:6.1f}s] shard {shard['index']}: runtime ready",
          flush=True)

    # One entry id per source -- |N+(x)| is a property of the SOURCE, so compiling
    # per entry would repeat the same work for every successor of that source.
    wanted = set(shard["sources"])
    representative: dict[str, str] = {}
    for entry in library.entries:
        key = entry.teacher_fiber.state_support.source_key
        if key in wanted and key not in representative:
            representative[key] = entry.entry_id

    rows, failures = [], []
    for position, (src, entry_id) in enumerate(sorted(representative.items())):
        states, _f, _e = library.inputs_for([entry_id])
        t0 = time.perf_counter()
        try:
            partition = compile_state_successor_map(model, states[0], time=0.5)
        except Exception as error:  # noqa: BLE001 - one bad state must not lose the shard
            failures.append({"source": src, "error": f"{type(error).__name__}: {error}"[:200]})
            continue
        per_family: collections.Counter = collections.Counter()
        for group in partition.successor_groups:
            for family in {mark.alias.family_name for mark in group.marks}:
                per_family[family] += 1
        rows.append({
            "source": src,
            "entry_id": entry_id,
            "canonical_successor_count": len(partition.successor_groups),
            "virtual_mark_count": len(partition.virtual_marks),
            "productive_alias_count": sum(len(g.marks) for g in partition.successor_groups),
            "per_family_successor_count": dict(per_family),
            "seconds": round(time.perf_counter() - t0, 3),
        })
        if position % 25 == 0:
            print(f"  shard {shard['index']}: {position+1}/{len(representative)} "
                  f"({time.perf_counter()-started:.0f}s)", flush=True)

    out = Path(RUN_ROOT) / "partitions"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "compose.editing_v2.successor_partition_shard",
        "status": "DEVELOPMENT_EVIDENCE_FROM_THE_DEVELOPMENT_CHECKPOINT",
        "shard_index": shard["index"],
        "requested_sources": len(shard["sources"]),
        "compiled": len(rows),
        "failures": failures,
        "rows": rows,
    }
    (out / f"shard-{shard['index']:04d}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    artifact_volume.commit()
    print(f"[{time.perf_counter()-started:6.1f}s] shard {shard['index']} done: "
          f"{len(rows)} compiled, {len(failures)} failed", flush=True)
    return {"index": shard["index"], "compiled": len(rows), "failed": len(failures),
            "seconds": round(time.perf_counter() - started, 1)}


@app.local_entrypoint()
def main(shards: int = 32) -> None:
    sample = json.loads(
        Path("diagnostics/editing_v2_experiment1_sample.json").read_text())
    sources = sample["sources"]
    chunks = [{"index": i, "sources": sources[i::shards]} for i in range(shards)]
    print(f"{len(sources):,} sources over {shards} shards "
          f"(~{len(sources)//shards} each)", flush=True)
    done = list(compile_shard.map(chunks))
    compiled = sum(d["compiled"] for d in done)
    failed = sum(d["failed"] for d in done)
    print(f"\ncompiled {compiled:,} / {len(sources):,} sources; {failed} failures")
    print(f"slowest shard: {max(d['seconds'] for d in done):.0f}s")
