"""Precompute frozen R_theta embeddings for every molecule in the teacher data.

WHY A FROZEN ENCODER
--------------------
The question is whether the useful remaining-budget value can be AMORTISED from
representations we already learned -- not whether a molecular encoder can be
trained from 14k labels. Those are different questions, and training an encoder
here would answer the second while appearing to answer the first.

So R_theta's own encoder is used unchanged, as a feature extractor, for both the
candidate molecule y and the target z. `_encode_batch` returns
(node, global_state, pair); `global_state` is the permutation-invariant
graph-level embedding, already carrying pooled node states plus structural
features (atom count, edge count, cycle rank, hydrogens).

Doing this ONCE turns h_phi training into a problem over vectors: the whole
2k/5k/10k sweep then runs locally in minutes, with no kernel calls and no
Modal, which is what makes an honest learning curve cheap to produce.

WHAT h_phi SEES
---------------
    h_phi(y, z, b)

y is the state AFTER the candidate action, z the target, b the remaining budget
FROM y. The controlled process is Markov, so the pre-action state x is not
needed -- the planner itself only ever evaluates V_G(y, z, b-1).

CPU ONLY. One forward pass per molecule; no successor enumeration anywhere.
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
    .add_local_file(
        ROOT / "diagnostics/editing_v2_h_phi_teacher_train_labels.json",
        str(REMOTE_ROOT / "diagnostics/editing_v2_h_phi_teacher_train_labels.json"),
        copy=True)
    .add_local_file(
        ROOT / "diagnostics/editing_v2_h_phi_teacher_validation_labels.json",
        str(REMOTE_ROOT / "diagnostics/editing_v2_h_phi_teacher_validation_labels.json"),
        copy=True)
)
app = modal.App("compose-v4-h-phi-encode-states")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT = 0.5
#: Molecules per encoder forward pass.
BATCH = 64
#: FIXED canonical slot width for every molecule, training and inference alike.
#:
#: The embedding is NOT padding-invariant: `structural` carries n_real/n_slots,
#: so slot count enters the features directly. Measured on one molecule,
#: |e(n) - e(n+4)| = 1.2e-2 and |e(n) - e(n+12)| = 2.4e-2, against 4.7e-2 for a
#: large change in t -- comparable, not negligible.
#:
#: Padding each batch to its own maximum would therefore give the same molecule
#: different features depending on batch composition, and inference (padding to
#: a pair's own slot count) would differ again. h_phi must see one width
#: everywhere. 48 covers the largest molecule in the data (42 heavy atoms) with
#: margin.
CANONICAL_SLOTS = 48


@app.function(
    image=image, cpu=4.0, memory=16 * 1024, timeout=4 * 60 * 60,
    max_containers=8, volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def encode(shard: dict[str, Any]) -> dict[str, Any]:
    """Embed a list of canonical molecule keys with the frozen R_theta encoder."""

    import sys

    import numpy as np
    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
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
    from compose_v4.model.factorized_tracelet_rate_model import (
        prepare_factorized_mark_batch,
    )

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
    print(f"[{time.perf_counter()-started:6.1f}s] frozen R_theta step "
          f"{checkpoint['selected_step']:,}", flush=True)

    keys = list(shard["keys"])
    graphs: dict[str, Any] = {}
    unparsable: list[str] = []
    for key in keys:
        try:
            graphs[key] = smiles_to_molecular_graph(key)
        except Exception:  # noqa: BLE001 - recorded, never fatal
            unparsable.append(key)
    print(f"  parsed {len(graphs):,}/{len(keys):,} ({len(unparsable)} unparsable)",
          flush=True)

    oversized = [k for k, g in graphs.items() if g.n_atoms > CANONICAL_SLOTS]
    if oversized:
        raise RuntimeError(
            f"{len(oversized)} molecules exceed CANONICAL_SLOTS={CANONICAL_SLOTS}; "
            f"raise it rather than padding them inconsistently")
    ordered = sorted(graphs)
    embeddings: dict[str, list[float]] = {}
    for start in range(0, len(ordered), BATCH):
        chunk = ordered[start:start + BATCH]
        # One width for every molecule everywhere -- see CANONICAL_SLOTS.
        states = tuple(pad_molecular_graph(graphs[k], CANONICAL_SLOTS)
                       for k in chunk)
        batch = prepare_factorized_mark_batch(
            states, tuple(float(TIME_POINT) for _ in chunk),
            tuple(None for _ in chunk), tuple(None for _ in chunk),
            tuple(0.0 for _ in chunk),
            use_aromatic_bond_view=True, ring_catalog=model.ring_catalog,
            **operator_capability_batch_kwargs(model.operator_capabilities))
        with torch.no_grad():
            _node, global_state, _pair = model._encode_batch(batch)
        for key, vector in zip(chunk, global_state.cpu().numpy()):
            embeddings[key] = [float(v) for v in vector]
        if start % (BATCH * 20) == 0:
            print(f"  [{time.perf_counter()-started:6.1f}s] {start + len(chunk):,}"
                  f"/{len(ordered):,}", flush=True)

    dim = len(next(iter(embeddings.values()))) if embeddings else 0
    out = Path(RUN_ROOT) / "h_phi_teacher" / "embeddings"
    out.mkdir(parents=True, exist_ok=True)
    np_path = out / f"shard-{shard['index']:03d}.npz"
    np.savez_compressed(
        np_path,
        canonical_slots=np.array([CANONICAL_SLOTS]),
        time_point=np.array([TIME_POINT]),
        keys=np.array(sorted(embeddings), dtype=object),
        vectors=np.array([embeddings[k] for k in sorted(embeddings)],
                         dtype=np.float32))
    artifact_volume.commit()
    print(f"[{time.perf_counter()-started:6.1f}s] shard {shard['index']}: "
          f"{len(embeddings):,} embeddings of dim {dim}", flush=True)
    return {"index": shard["index"], "embedded": len(embeddings),
            "unparsable": len(unparsable), "dim": dim,
            "seconds": round(time.perf_counter() - started, 1)}


@app.local_entrypoint()
def main(shards: int = 8) -> None:
    keys: set[str] = set()
    for name in ("editing_v2_h_phi_teacher_train_labels.json",
                 "editing_v2_h_phi_teacher_validation_labels.json"):
        payload = json.loads((ROOT / "diagnostics" / name).read_text())
        for pair in payload["per_pair"]:
            for row in pair["label_rows"]:
                keys.add(row["candidate"])
                keys.add(row["decision_state"])
    ordered = sorted(keys)
    print(f"{len(ordered):,} unique molecules to embed across {shards} shards")

    tasks = [{"index": i, "keys": ordered[i::shards]} for i in range(shards)]
    done = list(encode.map(tasks))
    print(f"\n  embedded {sum(d['embedded'] for d in done):,}; "
          f"unparsable {sum(d['unparsable'] for d in done)}; "
          f"dim {done[0]['dim'] if done else '?'}")
    print(f"  slowest shard {max(d['seconds'] for d in done):.0f}s")
    print("  embeddings under editing_v2/r_theta_run/h_phi_teacher/embeddings/")
