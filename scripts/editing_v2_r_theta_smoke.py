#!/usr/bin/env python
"""Prove the corpus R_theta training loop before any GPU is paid for.

This is the correctness smoke, not a performance decision. It answers: does a
fresh-initialized scorer consume the FROZEN manifest stream through the corpus
library, produce a finite loss, and take an optimizer step that leaves the
frozen hazard head untouched?

WHAT IT DELIBERATELY DOES NOT ANSWER
------------------------------------
Whether a GPU is warranted. Local CPU step time says nothing useful about the
data-wait fraction of an accelerator step: the forward/backward shrinks by a
large factor on GPU while the loader does not, so a data wait that is
negligible here can dominate there. The numbers printed below are the loader
side only; the GPU profiling smoke measures the ratio that matters.

THE INITIAL STATE IS LOADED, NEVER RE-DERIVED
---------------------------------------------
The scratch initialization is reproducible from ``initialization_seed`` only on
the platform that produced the frozen constant -- PyTorch's CPU ``normal_`` is
vectorized per architecture, so this macOS arm64 host rebuilds the identical
architecture with different numbers. Re-deriving here would train a model that
is not the frozen initial state and would split-brain against the corpus, which
was compiled under it. So the materialized state is loaded and its digest
verified.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from compose_v4.data.corpus_training_library import (  # noqa: E402
    load_corpus_training_library,
)
from compose_v4.experiments.editing_v2_process_v2_t1_panel import (  # noqa: E402
    open_process_v2_t1_source,
)
from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (  # noqa: E402
    build_process_v2_score_revised_scratch_runtime,
    load_materialized_scorer_state,
)
from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (  # noqa: E402
    TOTAL_HAZARD_PREFIX,
    _attach_successor_family_coordinates,
)
from compose_v4.experiments.factorized_mark_conditional import (  # noqa: E402
    FactorizedMarkCollator,
    FactorizedMarkExample,
)
from compose_v4.experiments.factorized_successor_training import (  # noqa: E402
    factorized_successor_identity_loss,
    forward_teacher_successor_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active8-root", required=True)
    parser.add_argument("--gate-zero", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--materialized-state", required=True)
    parser.add_argument("--corpus-root", action="append", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--packed-store", default="",
                        help="read collated rows from this packed store instead "
                             "of re-running RDKit collation per batch")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    started = time.perf_counter()
    source = open_process_v2_t1_source(
        Path(args.active8_root),
        gate_zero_decision_path=Path(args.gate_zero),
        artifact_root=Path(args.artifact_root),
        repo_root=REPO_ROOT,
    )
    state = load_materialized_scorer_state(Path(args.materialized_state))
    runtime, binding, _containment = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=state
    )
    model = runtime.model
    collator = FactorizedMarkCollator.from_capabilities(
        model.operator_capabilities,
        use_aromatic_bond_view=True,
        ring_catalog=model.ring_catalog,
    )
    model_runtime = binding["model_runtime"]
    print(f"model built                {time.perf_counter() - started:6.1f}s")
    print(f"  initialization_seed      {model_runtime['initialization_seed']}")
    print(f"  initial_model_state      {model_runtime['initial_model_state_sha256']}")
    print(f"  model_identity           {model_runtime['model_identity_sha256']}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  parameters               {sum(p.numel() for p in model.parameters()):,}")
    print(f"  trainable                {sum(p.numel() for p in trainable):,}")

    freeze = json.loads(Path(args.freeze).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest["sampling_law_sha256"] != freeze["sampling_law_sha256"]:
        raise SystemExit("manifest does not bind the frozen sampling law")
    index = json.loads(Path(args.index).read_text())
    if index["manifest_sha256"] != manifest["manifest_sha256"]:
        raise SystemExit("stream index was built for another manifest")
    sequence = index["sequence"]

    resolution = json.loads(
        Path(args.freeze).with_name("editing_v2_split_precedence_resolution.json").read_text()
    )
    library = load_corpus_training_library(
        [Path(r) for r in args.corpus_root],
        excluded_sources=resolution["excluded_source_keys"]["train"],
        verify_state_roundtrip=False,
    )
    print(f"library loaded             {time.perf_counter() - started:6.1f}s  "
          f"{len(library):,} entries")

    store = None
    template = None
    if args.packed_store:
        from compose_v4.data.packed_collated_batch import PackedCollatedStore

        store = PackedCollatedStore(Path(args.packed_store))
        template = torch.load(
            Path(args.packed_store) / "TEMPLATE.pt", map_location="cpu", weights_only=False
        )
        missing = [i for i in set(sequence) if i not in store.row_by_entry_id]
        if missing:
            raise SystemExit(f"{len(missing):,} stream ids are absent from the packed store")
        print(f"packed store               {len(store):,} rows, "
              f"{store.layout.record_bytes:,} B/row")

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    model.train()

    data_seconds = 0.0
    step_seconds = 0.0
    losses: list[float] = []
    hazard_touched: list[str] = []
    loop_started = time.perf_counter()

    for step in range(args.steps):
        start = step * args.batch_size
        selected = sequence[start : start + args.batch_size]
        if len(selected) < args.batch_size:
            raise SystemExit("stream exhausted before the requested step count")

        data_started = time.perf_counter()
        states, fibers, entries = library.inputs_for(selected)
        if store is not None:
            # The collated tensors were computed once at compile time; this is
            # a gather, not chemistry. MEASURED: 25.71 ms against 108.74 s to
            # re-collate the same 32 rows, and the resulting loss is bitwise
            # identical.
            batch = store.rows_for(
                selected,
                template,
                extra={
                    "states": tuple(states),
                    "times": torch.tensor(
                        [float.fromhex(e.support_time_hex) for e in entries],
                        dtype=torch.float32,
                    ),
                    "teacher_rates": torch.ones(len(selected), dtype=torch.float32),
                    "importance_weights": torch.ones(len(selected), dtype=torch.float32),
                },
            )
        else:
            batch = collator([
                FactorizedMarkExample(
                    state=state,
                    time=float.fromhex(entry.support_time_hex),
                    teacher_action=None,
                    teacher_rule_name=None,
                    teacher_rate=1.0,
                    importance_weight=1.0,
                )
                for state, entry in zip(states, entries, strict=True)
            ])
        batch = _attach_successor_family_coordinates(
            batch,
            [{"model_family": entry.model_family} for entry in entries],
        )
        # A no-op on CPU, but the host->device copy is exactly what the GPU
        # profiling smoke needs attributed to the data side rather than compute.
        batch = batch.to(model.device)
        data_seconds += time.perf_counter() - data_started

        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_teacher_successor_batch(model, batch, fibers)
        loss = factorized_successor_identity_loss(prediction, batch)
        if not bool(torch.isfinite(loss)):
            raise SystemExit(f"nonfinite loss at step {step + 1}")
        loss.backward()

        # The hazard head is frozen. A gradient here means the objective
        # reached a parameter it must not train, and nothing else would notice.
        for name, parameter in model.named_parameters():
            if name.startswith(TOTAL_HAZARD_PREFIX):
                if parameter.grad is not None:
                    hazard_touched.append(name)
            elif parameter.grad is not None and not bool(
                torch.isfinite(parameter.grad).all()
            ):
                raise SystemExit(f"nonfinite gradient at step {step + 1}: {name}")
        optimizer.step()
        step_seconds += time.perf_counter() - step_started
        losses.append(float(loss.detach()))

        if step == 0 or (step + 1) % 20 == 0:
            print(f"  step {step + 1:4d}  loss {losses[-1]:8.4f}")

    wall = time.perf_counter() - loop_started
    if hazard_touched:
        raise SystemExit(
            f"frozen hazard parameters received a gradient: {sorted(set(hazard_touched))}"
        )

    examples_seen = args.steps * args.batch_size
    print()
    print(f"steps                      {args.steps}")
    print(f"examples                   {examples_seen:,}")
    print(f"wall                       {wall:8.1f} s")
    print(f"  data                     {data_seconds:8.1f} s  "
          f"({data_seconds / wall * 100:5.1f}%)")
    print(f"  compute                  {step_seconds:8.1f} s  "
          f"({step_seconds / wall * 100:5.1f}%)")
    print(f"throughput                 {examples_seen / wall:8.1f} examples/s")
    print(f"first loss                 {losses[0]:8.4f}")
    print(f"last  loss                 {losses[-1]:8.4f}")
    print(f"mean of last 10            {sum(losses[-10:]) / len(losses[-10:]):8.4f}")
    print()
    print("frozen hazard received no gradient; every gradient finite.")
    print("NOTE: the data/compute split above is CPU-local and does NOT decide")
    print("whether a GPU is warranted -- compute shrinks on an accelerator and")
    print("the loader does not. The GPU smoke measures that ratio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
