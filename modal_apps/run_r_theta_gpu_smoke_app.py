"""Profile the corpus R_theta training step on a GPU before committing to a run.

This exists because the local CPU smoke cannot answer the question that decides
the full run. Forward/backward shrinks by a large factor on an accelerator while
the input path does not, so a data wait that is invisible at seconds-per-step
can dominate at milliseconds-per-step. Only a GPU measurement settles it.

So this measures the RATIO, not just the speed, splitting the step finely
enough to name the next optimization target if one exists:

    data       library.inputs_for + packed gather   (CPU, does not shrink)
    host->dev  batch.to(device)                     (grows in relative terms)
    forward    forward_teacher_successor_batch + loss
    backward   loss.backward()
    optimizer  grad checks + optimizer.step()

THE INPUT PATH IS A GATHER, NOT CHEMISTRY
-----------------------------------------
An earlier version of this app collated per batch, which was wrong. MEASURED:
fresh collation of 32 rows costs 108.74 s against 2.38 s of model math -- 97.2%
of the step -- because it runs RDKit admission masks over every candidate
successor. Collating inside a GPU container rents an accelerator to run a
valence checker, and the cost does not shrink on the device.

Batches therefore come from the packed store, a row-addressed copy of the
tensors the compile step already produced: a 32-row gather is 25.71 ms and the
resulting loss is bitwise identical to the collated one. If this smoke still
shows a bottleneck, it will be in ``forward``, where
``_factorized_action_probability_tables`` builds the legal-mark tables -- not in
the dataset, which is now three orders of magnitude too small to matter.

It writes no checkpoint and commits nothing to the volume. It is a measurement.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (  # reuse the exact environment
    ARTIFACT_ROOT,
    artifact_volume,
    image,
)

app = modal.App("compose-v4-r-theta-gpu-smoke")

#: Twenty minutes is generous for a few hundred steps and bounds the spend.
SMOKE_TIMEOUT_SECONDS = 20 * 60
#: Published Modal A10G rate, for the projection only. The receipt records it
#: so a stale constant is visible rather than silently wrong.
A10G_USD_PER_HOUR = 1.10


@app.function(
    image=image,
    gpu="A10G",
    cpu=8.0,
    memory=64 * 1024,
    timeout=SMOKE_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def profile_r_theta_step(
    *,
    run_root: str,
    corpus_roots: list[str],
    packed_store: str,
    steps: int = 200,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
) -> dict[str, Any]:
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
    from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
        TOTAL_HAZARD_PREFIX,
        _attach_successor_family_coordinates,
    )
    from compose_v4.experiments.factorized_successor_training import (
        factorized_successor_identity_loss,
        forward_teacher_successor_batch,
    )

    artifact_volume.reload()
    root = Path(run_root)
    inputs = root / "run_inputs"
    repo_root = Path("/root")

    if not torch.cuda.is_available():
        raise RuntimeError("the GPU smoke was scheduled without a visible device")
    device = torch.device("cuda")

    started = time.perf_counter()
    source = open_process_v2_t1_source(
        Path(json.loads((inputs / "RUN_PATHS.json").read_text())["active8_root"]),
        gate_zero_decision_path=Path(
            json.loads((inputs / "RUN_PATHS.json").read_text())["gate_zero"]
        ),
        artifact_root=Path(
            json.loads((inputs / "RUN_PATHS.json").read_text())["artifact_root"]
        ),
        repo_root=repo_root,
    )
    state = load_materialized_scorer_state(root / "materialized_scorer")
    runtime, binding, _containment = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=state
    )
    model = runtime.model.to(device)
    model_seconds = time.perf_counter() - started

    freeze = json.loads((inputs / "editing_v2_v2_dataset_freeze.json").read_text())
    manifest = json.loads(
        (inputs / "editing_v2_prepared_training_manifest.json").read_text()
    )
    if manifest["sampling_law_sha256"] != freeze["sampling_law_sha256"]:
        raise RuntimeError("manifest does not bind the frozen sampling law")
    index = json.loads(
        (inputs / "editing_v2_prepared_training_index.json").read_text()
    )
    if index["manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("stream index was built for another manifest")
    resolution = json.loads(
        (inputs / "editing_v2_split_precedence_resolution.json").read_text()
    )
    sequence = index["sequence"]

    library_started = time.perf_counter()
    library = load_corpus_training_library(
        [Path(r) for r in corpus_roots],
        excluded_sources=resolution["excluded_source_keys"]["train"],
        verify_state_roundtrip=False,
    )
    library_seconds = time.perf_counter() - library_started

    # Every id the frozen stream will ever draw must resolve BEFORE spending
    # GPU time, not on the step that happens to reach a missing one.
    unresolved = [
        entry_id
        for entry_id in set(sequence)
        if entry_id not in library.index_by_entry_id
    ]
    if unresolved:
        raise RuntimeError(
            f"{len(unresolved):,} stream ids do not resolve against the library"
        )

    panel = json.loads((inputs / "editing_v2_eval_panel.json").read_text())

    # The collated tensors come from the packed store, never from RDKit.
    # MEASURED locally: re-collating 32 rows is 108.74 s against a 25.71 ms
    # gather, and the resulting loss is bitwise identical. Collating inside a
    # GPU container would rent an accelerator to run a valence checker.
    store = PackedCollatedStore(Path(packed_store))
    template = torch.load(
        Path(packed_store) / "TEMPLATE.pt", map_location="cpu", weights_only=False
    )
    absent = [i for i in set(sequence) if i not in store.row_by_entry_id]
    if absent:
        raise RuntimeError(f"{len(absent):,} stream ids are absent from the packed store")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    model.train()
    torch.cuda.reset_peak_memory_stats()

    # The hazard head is frozen. Checking only that it received no GRADIENT is
    # weaker than checking it did not MOVE: a stray optimizer group, a weight
    # decay term, or an in-place write would change it with no gradient ever
    # appearing. Snapshot the values and compare them bit-for-bit at the end.
    hazard_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(TOTAL_HAZARD_PREFIX)
    }
    if not hazard_before:
        raise RuntimeError("no frozen hazard parameters found; the check is vacuous")

    data_seconds = 0.0
    transfer_seconds = 0.0
    forward_seconds = 0.0
    backward_seconds = 0.0
    optimizer_seconds = 0.0
    losses: list[float] = []
    hazard_touched: set[str] = set()
    nonfinite_grads: set[str] = set()

    # A handful of untimed warmup steps: the first CUDA kernels include
    # allocator and autotune costs that would otherwise be charged to compute
    # and make the ratio look worse than steady state.
    warmup = min(5, max(1, steps // 20))
    loop_started = None

    for step in range(warmup + steps):
        timed = step >= warmup
        if timed and loop_started is None:
            torch.cuda.synchronize()
            loop_started = time.perf_counter()

        start = step * batch_size
        selected = sequence[start : start + batch_size]
        if len(selected) < batch_size:
            raise RuntimeError("stream exhausted before the requested step count")

        mark = time.perf_counter()
        states, fibers, entries = library.inputs_for(selected)
        host_batch = store.rows_for(
            selected,
            template,
            extra={
                "states": tuple(states),
                "times": torch.tensor(
                    [float.fromhex(e.support_time_hex) for e in entries],
                    dtype=torch.float32,
                ),
                "teacher_rates": torch.ones(batch_size, dtype=torch.float32),
                "importance_weights": torch.ones(batch_size, dtype=torch.float32),
            },
        )
        host_batch = _attach_successor_family_coordinates(
            host_batch,
            [{"model_family": entry.model_family} for entry in entries],
        )
        if timed:
            data_seconds += time.perf_counter() - mark

        mark = time.perf_counter()
        batch = host_batch.to(device)
        torch.cuda.synchronize()
        if timed:
            transfer_seconds += time.perf_counter() - mark

        optimizer.zero_grad(set_to_none=True)
        mark = time.perf_counter()
        prediction = forward_teacher_successor_batch(model, batch, fibers)
        loss = factorized_successor_identity_loss(prediction, batch)
        torch.cuda.synchronize()
        if timed:
            forward_seconds += time.perf_counter() - mark
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss at step {step + 1}")

        mark = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        if timed:
            backward_seconds += time.perf_counter() - mark

        mark = time.perf_counter()
        for name, parameter in model.named_parameters():
            if name.startswith(TOTAL_HAZARD_PREFIX):
                if parameter.grad is not None:
                    hazard_touched.add(name)
            elif parameter.grad is not None and not bool(
                torch.isfinite(parameter.grad).all()
            ):
                nonfinite_grads.add(name)
        optimizer.step()
        torch.cuda.synchronize()
        if timed:
            optimizer_seconds += time.perf_counter() - mark
            losses.append(float(loss.detach()))

    wall = time.perf_counter() - loop_started
    if hazard_touched:
        raise RuntimeError(f"frozen hazard received a gradient: {sorted(hazard_touched)}")
    if nonfinite_grads:
        raise RuntimeError(f"nonfinite gradients: {sorted(nonfinite_grads)}")
    hazard_moved = sorted(
        name
        for name, before in hazard_before.items()
        if not torch.equal(before, dict(model.named_parameters())[name].detach())
    )
    if hazard_moved:
        raise RuntimeError(f"frozen hazard parameters CHANGED: {hazard_moved}")

    examples_seen = steps * batch_size
    per_second = examples_seen / wall
    epoch_seconds = manifest["epoch_size"] / per_second
    receipt = {
        "schema": "compose.editing_v2.r_theta_gpu_smoke",
        "schema_version": 1,
        "status": "GPU_SMOKE_EVIDENCE_ONLY_NO_AUTHORITY",
        "gpu": "A10G",
        "steps": steps,
        "warmup_steps": warmup,
        "batch_size": batch_size,
        "examples": examples_seen,
        "setup": {
            "model_seconds": round(model_seconds, 2),
            "library_seconds": round(library_seconds, 2),
            "library_entries": len(library),
        },
        "milliseconds_per_step": {
            "data_fetch_collate": round(data_seconds / steps * 1000, 2),
            "host_to_device": round(transfer_seconds / steps * 1000, 2),
            "forward_and_loss": round(forward_seconds / steps * 1000, 2),
            "backward": round(backward_seconds / steps * 1000, 2),
            "optimizer": round(optimizer_seconds / steps * 1000, 2),
            "total_step": round(wall / steps * 1000, 2),
        },
        "timing_share": {
            "data_fetch_collate": round(data_seconds / wall, 4),
            "host_to_device": round(transfer_seconds / wall, 4),
            "forward_and_loss": round(forward_seconds / wall, 4),
            "backward": round(backward_seconds / wall, 4),
            "optimizer": round(optimizer_seconds / wall, 4),
        },
        # The number that decides whether to optimize the input path at all.
        "gpu_data_wait_fraction": round(
            (data_seconds + transfer_seconds) / wall, 4
        ),
        "throughput_examples_per_second": round(per_second, 2),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "invariants": {
            "loss_finite_every_step": True,
            "gradients_finite_every_step": True,
            "frozen_hazard_received_no_gradient": True,
            "frozen_hazard_bitwise_unchanged": True,
            "frozen_hazard_parameters_checked": sorted(hazard_before),
            "stream_ids_unresolved": 0,
            "stream_ids_checked": len(set(sequence)),
        },
        "loss": {
            "first": round(losses[0], 5),
            "last": round(losses[-1], 5),
            "mean_last_10": round(sum(losses[-10:]) / len(losses[-10:]), 5),
        },
        "projection": {
            "epoch_examples": manifest["epoch_size"],
            "epoch_seconds": round(epoch_seconds, 1),
            "epoch_hours": round(epoch_seconds / 3600, 3),
            "assumed_usd_per_hour": A10G_USD_PER_HOUR,
            "epoch_usd": round(epoch_seconds / 3600 * A10G_USD_PER_HOUR, 2),
        },
        "binding": {
            "initialization_seed": binding["model_runtime"]["initialization_seed"],
            "initial_model_state_sha256":
                binding["model_runtime"]["initial_model_state_sha256"],
            "library_sha256": freeze["library_sha256"],
            "split_sha256": freeze["split_sha256"],
            "sampling_law_sha256": freeze["sampling_law_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "packed_store_rows": len(store),
            "packed_store_records_sha256": store.header["records_sha256"],
            "eval_panel_sha256": panel["panel_sha256"],
            "eval_panel_entries": panel["entry_count"],
        },
    }

    step_ms = receipt["milliseconds_per_step"]
    print("")
    print(f"data fetch / collate     {step_ms['data_fetch_collate']:8.2f} ms")
    print(f"host -> GPU              {step_ms['host_to_device']:8.2f} ms")
    print(f"forward + loss           {step_ms['forward_and_loss']:8.2f} ms")
    print(f"backward                 {step_ms['backward']:8.2f} ms")
    print(f"optimizer                {step_ms['optimizer']:8.2f} ms")
    print(f"total step               {step_ms['total_step']:8.2f} ms")
    print(f"GPU data-wait fraction   {receipt['gpu_data_wait_fraction'] * 100:8.2f} %")
    print(f"peak VRAM                {receipt['peak_vram_gb']:8.3f} GB")
    print(f"examples / sec           {receipt['throughput_examples_per_second']:8.2f}")
    print("")
    print(f"initial state hash       {receipt['binding']['initial_model_state_sha256']}")
    print(f"sampling law             {receipt['binding']['sampling_law_sha256']}")
    print(f"eval panel               {receipt['binding']['eval_panel_sha256']}")
    print(f"hazard before/after      identical ({len(hazard_before)} tensors)")
    print(f"loss / gradients         finite at every step")
    print(f"unresolved stream ids    0 of {len(set(sequence)):,}")
    print(f"projected epoch          {receipt['projection']['epoch_hours']:.3f} h "
          f"(${receipt['projection']['epoch_usd']:.2f} at "
          f"${A10G_USD_PER_HOUR:.2f}/h)")

    # Deliberately no artifact_volume.commit(): this run writes nothing.
    return receipt
