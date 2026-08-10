"""Train R_theta on the frozen COMPOSE corpus, resumably, on one A10G.

WHY THIS IS SEGMENTED
---------------------
MEASURED: 634 ms per step at batch 32, so a 144,870-example epoch is ~48
minutes and a multi-epoch run outlives any single container. This function
therefore trains for a bounded segment, stops itself BEFORE the container wall
with margin to spare, writes a checkpoint, and returns. Successive calls
resume. A run that ends on a kill instead of a checkpoint loses everything
since the last write, and at these step times that is expensive.

The margin is not decoration. Writing the checkpoint costs real seconds
(~25 MB of model plus optimizer state), so the loop must stop early enough to
finish writing, not merely early enough to notice the deadline.

WHAT IT REFUSES
---------------
Resuming against a different corpus, law, panel, manifest, packed store or
initialization -- the checkpoint carries all of them and the identity digest is
compared before any weights load. A run that silently continues under changed
inputs produces a result nothing can be attributed to.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (  # reuse the exact environment
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

# See run_r_theta_gpu_smoke_app: the base image ships modal_apps under
# REMOTE_ROOT but only puts REMOTE_ROOT/src on PYTHONPATH.
image = _base_image.env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})

app = modal.App("compose-v4-r-theta-training")

#: One hour per segment. Long enough to amortize the ~190 s setup, short
#: enough that a lost segment is cheap.
SEGMENT_TIMEOUT_SECONDS = 60 * 60
#: Stop training this long before the wall so the checkpoint completes.
WALL_MARGIN_SECONDS = 180.0


@app.function(
    image=image,
    gpu="A10G",
    cpu=8.0,
    memory=64 * 1024,
    timeout=SEGMENT_TIMEOUT_SECONDS,
    max_containers=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def train_segment(
    run_root: str = "/artifacts/editing_v2/r_theta_run",
    output_name: str = "run_01",
    maximum_steps: int = 4527,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    evaluate_every: int = 500,
    checkpoint_every: int = 250,
    resume: bool = True,
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
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME,
        TOTAL_HAZARD_PREFIX,
        RunIdentity,
        RThetaTrainingError,
        assert_training_invariants,
        evaluate_panel,
        load_checkpoint,
        selection_criterion,
        summarize_panel,
        write_checkpoint,
    )
    from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
        _attach_successor_family_coordinates,
    )
    from compose_v4.experiments.factorized_successor_training import (
        factorized_successor_identity_loss,
        forward_teacher_successor_batch,
    )

    segment_started = time.perf_counter()
    artifact_volume.reload()
    root = Path(run_root)
    inputs = root / "run_inputs"
    paths = json.loads((inputs / "RUN_PATHS.json").read_text())
    output_root = root / "runs" / output_name

    if not torch.cuda.is_available():
        raise RThetaTrainingError("training was scheduled without a visible GPU")
    device = torch.device("cuda")

    source = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]),
        repo_root=REMOTE_ROOT,
    )
    state = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, binding, _containment = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=state
    )
    model = runtime.model.to(device)
    print(f"[{time.perf_counter() - segment_started:6.1f}s] model on {device}", flush=True)

    freeze = json.loads((inputs / "editing_v2_v2_dataset_freeze.json").read_text())
    manifest = json.loads(
        (inputs / "editing_v2_prepared_training_manifest.json").read_text()
    )
    panel_artifact = json.loads((inputs / "editing_v2_eval_panel.json").read_text())
    index = json.loads((inputs / "editing_v2_prepared_training_index.json").read_text())
    resolution = json.loads(
        (inputs / "editing_v2_split_precedence_resolution.json").read_text()
    )
    if manifest["sampling_law_sha256"] != freeze["sampling_law_sha256"]:
        raise RThetaTrainingError("manifest does not bind the frozen sampling law")
    if index["manifest_sha256"] != manifest["manifest_sha256"]:
        raise RThetaTrainingError("stream index was built for another manifest")
    sequence = index["sequence"]

    library = load_corpus_training_library(
        [Path(r) for r in paths["corpus_roots"]],
        excluded_sources=resolution["excluded_source_keys"]["train"],
        verify_state_roundtrip=False,
    )
    store = PackedCollatedStore(Path(paths["packed_store"]))
    template = torch.load(
        Path(paths["packed_store"]) / "TEMPLATE.pt", map_location="cpu", weights_only=False
    )
    panel_library = load_corpus_training_library(
        [Path(r) for r in paths["panel_corpus_roots"]],
        excluded_sources=resolution["excluded_source_keys"]["validation"],
        verify_state_roundtrip=False,
    )
    panel_store = PackedCollatedStore(Path(paths["packed_panel_store"]))
    panel_template = torch.load(
        Path(paths["packed_panel_store"]) / "TEMPLATE.pt",
        map_location="cpu",
        weights_only=False,
    )
    panel_ids = json.loads(
        (inputs / "editing_v2_eval_panel_ids.json").read_text()
    )["entry_ids"]
    print(
        f"[{time.perf_counter() - segment_started:6.1f}s] library {len(library):,} / "
        f"panel {len(panel_library):,}",
        flush=True,
    )

    def make_builder(source_library, source_store, source_template):
        def build(entry_ids):
            states, fibers, entries = source_library.inputs_for(entry_ids)
            count = len(entry_ids)
            batch = source_store.rows_for(
                entry_ids,
                source_template,
                extra={
                    "states": tuple(states),
                    "times": torch.tensor(
                        [float.fromhex(e.support_time_hex) for e in entries],
                        dtype=torch.float32,
                    ),
                    "teacher_rates": torch.ones(count, dtype=torch.float32),
                    "importance_weights": torch.ones(count, dtype=torch.float32),
                },
            )
            batch = _attach_successor_family_coordinates(
                batch, [{"model_family": e.model_family} for e in entries]
            )
            return batch.to(device), fibers, entries

        return build

    build_training = make_builder(library, store, template)
    build_panel = make_builder(panel_library, panel_store, panel_template)

    identity = RunIdentity(
        initialization_seed=int(binding["model_runtime"]["initialization_seed"]),
        initial_model_state_sha256=str(binding["model_runtime"]["initial_model_state_sha256"]),
        library_sha256=str(freeze["library_sha256"]),
        split_sha256=str(freeze["split_sha256"]),
        sampling_law_sha256=str(freeze["sampling_law_sha256"]),
        manifest_sha256=str(manifest["manifest_sha256"]),
        packed_store_records_sha256=str(store.header["records_sha256"]),
        eval_panel_sha256=str(panel_artifact["panel_sha256"]),
    )
    stream_sha256 = json.loads((inputs / "editing_v2_prepared_training_index.json").read_text())[
        "manifest_sha256"
    ]

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    completed_steps = 0
    resume_count = 0
    selected_step = 0
    selected_criterion_value = None
    selected_state = None
    trajectory: list[dict[str, Any]] = []

    checkpoint_path = output_root / CHECKPOINT_FILENAME
    if resume and checkpoint_path.exists():
        payload = load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            identity=identity,
            expected_stream_sha256=stream_sha256,
        )
        completed_steps = int(payload["completed_steps"])
        resume_count = int(payload["resume_count"]) + 1
        selected_step = int(payload["selected_step"])
        selected_criterion_value = (
            tuple(payload["selected_criterion"]) if payload["selected_criterion"] else None
        )
        selected_state = payload.get("selected_model_state") or None
        trajectory = [dict(item) for item in payload["trajectory"]]
        model.to(device)
        print(f"resumed at step {completed_steps:,} (resume #{resume_count})", flush=True)

    hazard_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(TOTAL_HAZARD_PREFIX)
    }
    if not hazard_before:
        raise RThetaTrainingError("no frozen hazard parameters found; the check is vacuous")

    model.train()
    deadline = segment_started + SEGMENT_TIMEOUT_SECONDS - WALL_MARGIN_SECONDS
    stop_reason = "MAXIMUM_STEPS_REACHED"
    losses: list[float] = []
    # A 2.9-hour run that only speaks every 500 steps is indistinguishable
    # from a hung one, and on a paid device that ambiguity costs money.
    log_every = 50
    # Accumulate STEP time only. Charging the panel eval and the 87.8 MB
    # checkpoint write to training throughput made the alarm fire at every
    # eval, and an alarm that cries wolf 54 times is worse than none.
    window_seconds = 0.0
    window_losses: list[float] = []
    # Anomaly baselines. Streaming numbers is not the same as saying something
    # is wrong, and a 2.9-hour run should announce trouble rather than leave it
    # to be noticed in a plot afterwards.
    previous_window_loss: float | None = None
    reference_throughput: float | None = None
    warnings: list[str] = []

    def warn(message: str) -> None:
        warnings.append(f"step {completed_steps}: {message}")
        print(f"  WARN  {message}", flush=True)
    print(
        f"training from step {completed_steps:,} to {maximum_steps:,} "
        f"(batch {batch_size}, eval every {evaluate_every}, "
        f"checkpoint every {checkpoint_every})",
        flush=True,
    )

    while completed_steps < maximum_steps:
        if time.perf_counter() > deadline:
            stop_reason = "SEGMENT_WALL_MARGIN"
            break
        start = (completed_steps * batch_size) % max(1, len(sequence) - batch_size)
        selected = sequence[start : start + batch_size]
        if len(selected) < batch_size:
            stop_reason = "STREAM_EXHAUSTED"
            break

        step_started = time.perf_counter()
        batch, fibers, _entries = build_training(selected)
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_teacher_successor_batch(model, batch, fibers)
        loss = factorized_successor_identity_loss(prediction, batch)
        if not bool(torch.isfinite(loss)):
            raise RThetaTrainingError(f"nonfinite loss at step {completed_steps + 1}")
        loss.backward()
        for name, parameter in model.named_parameters():
            if name.startswith(TOTAL_HAZARD_PREFIX) and parameter.grad is not None:
                raise RThetaTrainingError(f"frozen hazard received a gradient: {name}")
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise RThetaTrainingError(
                    f"nonfinite gradient at step {completed_steps + 1}: {name}"
                )
        optimizer.step()
        completed_steps += 1
        window_seconds += time.perf_counter() - step_started
        value = float(loss.detach())
        losses.append(value)
        window_losses.append(value)

        if completed_steps % log_every == 0:
            elapsed = window_seconds
            print(
                f"  step {completed_steps:6,}/{maximum_steps:,}  "
                f"loss {sum(window_losses) / len(window_losses):7.4f}  "
                f"{log_every * batch_size / elapsed:6.1f} ex/s  "
                f"segment {time.perf_counter() - segment_started:6.0f}s",
                flush=True,
            )
            window_loss = sum(window_losses) / len(window_losses)
            throughput = log_every * batch_size / elapsed
            # Skip the opening windows as a baseline: the first carries CUDA
            # allocator and autotune warmup and reads ~20% slow, which would
            # set the threshold too low to catch a real regression.
            if completed_steps <= 2 * log_every:
                pass
            elif reference_throughput is None:
                reference_throughput = throughput
            elif throughput < 0.6 * reference_throughput:
                warn(
                    f"throughput {throughput:.1f} ex/s is below 60% of the "
                    f"opening {reference_throughput:.1f} ex/s"
                )
            if previous_window_loss is not None and window_loss > 1.5 * previous_window_loss:
                warn(
                    f"training loss rose {previous_window_loss:.4f} -> "
                    f"{window_loss:.4f} over one window"
                )
            previous_window_loss = window_loss
            window_seconds = 0.0
            window_losses = []

        if completed_steps % evaluate_every == 0:
            rows = evaluate_panel(
                model, entry_ids=panel_ids, build_batch=build_panel, batch_size=batch_size
            )
            summary = summarize_panel(
                rows,
                deployment_family_share=manifest["coefficient_proof"]["family_realized"],
            )
            criterion = selection_criterion(summary, step=completed_steps)
            trajectory.append(
                {
                    "step": completed_steps,
                    "panel_native_mean_nll": summary["panel_native"]["mean_nll"],
                    "panel_native_minimum_probability":
                        summary["panel_native"]["minimum_probability"],
                    "deployment_weighted_mean_nll": summary["deployment_weighted_mean_nll"],
                    "criterion": list(criterion),
                }
            )
            if trajectory[:-1]:
                earlier = trajectory[-2]["panel_native_mean_nll"]
                current = summary["panel_native"]["mean_nll"]
                if current > earlier * 1.10:
                    warn(
                        f"held-out NLL worsened {earlier:.4f} -> {current:.4f}; "
                        "this is where overfitting or divergence shows first"
                    )
            if summary["panel_native"]["minimum_probability"] <= 0.0:
                warn("a held-out transition has zero probability; the model has "
                     "abandoned it entirely")
            if selected_criterion_value is None or criterion > selected_criterion_value:
                selected_criterion_value = criterion
                selected_step = completed_steps
                selected_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            print(
                f"[step {completed_steps:6,}] panel NLL "
                f"{summary['panel_native']['mean_nll']:.4f}  "
                f"deployment {summary['deployment_weighted_mean_nll']:.4f}  "
                f"min p {summary['panel_native']['minimum_probability']:.3e}",
                flush=True,
            )

        if completed_steps % checkpoint_every == 0:
            assert_training_invariants(model, hazard_before=hazard_before, step=completed_steps)
            written = write_checkpoint(
                checkpoint_path,
                model=model, optimizer=optimizer, identity=identity,
                completed_steps=completed_steps, selected_step=selected_step,
                selected_criterion=selected_criterion_value, selected_state=selected_state,
                trajectory=trajectory, stream_sha256=stream_sha256, resume_count=resume_count,
            )
            artifact_volume.commit()
            print(
                f"  checkpoint step {completed_steps:6,}  "
                f"{written['file_bytes'] / 1e6:.1f} MB  "
                f"sha {written['file_sha256'][:12]}  best step {selected_step:,}",
                flush=True,
            )

    assert_training_invariants(model, hazard_before=hazard_before, step=completed_steps)
    receipt = write_checkpoint(
        checkpoint_path,
        model=model, optimizer=optimizer, identity=identity,
        completed_steps=completed_steps, selected_step=selected_step,
        selected_criterion=selected_criterion_value, selected_state=selected_state,
        trajectory=trajectory, stream_sha256=stream_sha256, resume_count=resume_count,
    )
    artifact_volume.commit()
    print(
        f"segment end: {stop_reason} at step {completed_steps:,} "
        f"after {time.perf_counter() - segment_started:.0f}s; "
        f"best step {selected_step:,}; warnings {len(warnings)}",
        flush=True,
    )
    for message in warnings:
        print(f"  WARN {message}", flush=True)
    return {
        "schema": "compose.editing_v2.r_theta_training_segment",
        "status": "R_THETA_SEGMENT_EVIDENCE_ONLY_NO_AUTHORITY",
        "stop_reason": stop_reason,
        "completed_steps": completed_steps,
        "maximum_steps": maximum_steps,
        "batch_size": batch_size,
        "segment_seconds": round(time.perf_counter() - segment_started, 1),
        "identity": identity.payload(),
        "identity_sha256": identity.sha256(),
        "checkpoint": receipt,
        "trajectory": trajectory,
        "recent_loss_mean": (sum(losses[-50:]) / len(losses[-50:])) if losses else None,
        "warnings": warnings,
    }


@app.local_entrypoint()
def main(
    output_name: str = "run_01",
    maximum_steps: int = 4527,
    batch_size: int = 32,
    evaluate_every: int = 500,
    checkpoint_every: int = 250,
    max_segments: int = 12,
) -> None:
    """Drive segments until the step budget is met.

    Each call resumes the previous checkpoint, so the only cost of a segment
    boundary is one setup (~190 s). ``max_segments`` is a runaway guard: if a
    segment ever returns without advancing, this stops rather than looping
    forever on a paid device.
    """

    previous = -1
    for segment in range(max_segments):
        receipt = train_segment.remote(
            output_name=output_name,
            maximum_steps=maximum_steps,
            batch_size=batch_size,
            evaluate_every=evaluate_every,
            checkpoint_every=checkpoint_every,
        )
        done = int(receipt["completed_steps"])
        print(
            f"\n=== segment {segment + 1}: {receipt['stop_reason']} at "
            f"{done:,}/{maximum_steps:,} steps in "
            f"{receipt['segment_seconds']:.0f}s ===\n",
            flush=True,
        )
        if receipt.get("warnings"):
            print(f"!! {len(receipt['warnings'])} warning(s) this segment:")
            for message in receipt["warnings"]:
                print(f"   {message}")
        if done >= maximum_steps:
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return
        if done <= previous:
            raise SystemExit(
                f"segment made no progress ({previous:,} -> {done:,}); stopping "
                "rather than burning GPU on a loop"
            )
        previous = done
    raise SystemExit(f"hit max_segments={max_segments} before the step budget")
