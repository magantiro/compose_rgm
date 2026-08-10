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

import gzip
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
        capability_baseline,
        capability_floor,
        collapsed_capabilities,
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

    # Defined before the artifact loads: the external-cohort guard below warns
    # on failure, and a NameError there would crash the run on the exact path
    # that exists to keep a diagnostic from crashing the run.
    warnings: list[str] = []
    completed_steps = 0

    def warn(message: str) -> None:
        warnings.append(f"step {completed_steps}: {message}")
        print(f"  WARN  {message}", flush=True)

    freeze = json.loads((inputs / "editing_v2_v2_dataset_freeze.json").read_text())
    resolution = json.loads(
        (inputs / "editing_v2_split_precedence_resolution.json").read_text()
    )

    # ---- v2 identities: the post-split reserve, law and sequence ------------
    # The reserve is carved from WITHIN the train library, so both sides share
    # one packed store and one corpus root. The separate 16-shard panel is
    # retained as an external cross-cohort test and is not read here -- it is
    # never a selection input.
    law = json.loads((inputs / "editing_v2_sampling_law_v2.json").read_text())
    manifest = json.loads(
        (inputs / "editing_v2_prepared_training_manifest_v2.json").read_text()
    )
    index = json.loads(
        (inputs / "editing_v2_prepared_training_index_v2.json").read_text()
    )
    with gzip.open(
        inputs / "editing_v2_matched_validation_reserve_ids.json.gz", "rt"
    ) as handle:
        reserve = json.load(handle)
    gate = json.loads((inputs / "editing_v2_post_split_freeze_gate.json").read_text())

    if gate["status"] != "FROZEN":
        raise RThetaTrainingError(
            f"pre-freeze gate is {gate['status']}, not FROZEN; refusing to train")
    if manifest["law_frozen_sha256"] != law["frozen_sha256"]:
        raise RThetaTrainingError("manifest does not bind this sampling law")
    if gate["law_frozen_sha256"] != law["frozen_sha256"]:
        raise RThetaTrainingError("the gate that passed was run against another law")
    if gate["manifest_frozen_sha256"] != manifest["frozen_sha256"]:
        raise RThetaTrainingError("the gate that passed was run against another manifest")
    if (law.get("matched_validation_reserve_source_digest")
            != manifest.get("matched_validation_reserve_source_digest")):
        raise RThetaTrainingError("law and manifest bind different reserves")
    sequence = index["sequence"]

    panel_ids = list(reserve["reserve_entry_ids"])
    reserve_band_by_entry_id = dict(reserve["reserve_band_by_entry_id"])
    training_ids = set(reserve["training_entry_ids"])
    if set(sequence) - training_ids:
        raise RThetaTrainingError("the training sequence reaches outside the split")
    if set(panel_ids) & set(sequence):
        raise RThetaTrainingError("a reserve entry appears in the training sequence")

    with gzip.open(inputs / "ENTRY_PROVENANCE.json.gz", "rt") as handle:
        provenance = json.load(handle)
    synthetic_lane = provenance["synthetic_lane"]
    reserve_lane_by_entry_id = {
        entry_id: ("synthetic" if provenance["lane_by_entry_id"].get(entry_id)
                   == synthetic_lane else "real")
        for entry_id in panel_ids
    }
    # The law defines the reference process; its realized draw share per
    # "family|lane" is what weights the selection number. Strata absent here,
    # or given zero, are reported as diagnostics and never vote.
    reference_law_stratum_share = {
        f'{s["family"]}|{s["provenance"]}': float(s["draw_share"])
        for s in law["strata"]
    }

    library = load_corpus_training_library(
        [Path(r) for r in paths["corpus_roots"]],
        excluded_sources=resolution["excluded_source_keys"]["train"],
        verify_state_roundtrip=False,
    )
    store = PackedCollatedStore(Path(paths["packed_store"]))
    template = torch.load(
        Path(paths["packed_store"]) / "TEMPLATE.pt", map_location="cpu", weights_only=False
    )
    # MEASURED: the packed train store holds exactly 151,059 rows -- the same
    # 4 duplicate ids and 19 precedence drops the carve found independently --
    # and its row set is exactly training + reserve. So one store serves both
    # sides and nothing needs repacking for the new split.
    panel_library, panel_store, panel_template = library, store, template

    # The externally mined 16-shard cohort. A DIAGNOSTIC, never a selection
    # input: its support composition is an accident of how those shards were
    # mined (42.6% at band 0 against the training population's 19.3%), which is
    # exactly why it cannot drive early stopping. Loaded here so an epoch
    # boundary can report it without a second container.
    try:
        cohort_library = load_corpus_training_library(
            [Path(r) for r in paths["panel_corpus_roots"]],
            excluded_sources=resolution["excluded_source_keys"]["validation"],
            verify_state_roundtrip=False,
        )
        cohort_store = PackedCollatedStore(Path(paths["packed_panel_store"]))
        cohort_template = torch.load(
            Path(paths["packed_panel_store"]) / "TEMPLATE.pt",
            map_location="cpu", weights_only=False,
        )
        cohort_ids = [e.entry_id for e in cohort_library.entries]
    except Exception as error:  # noqa: BLE001 - a diagnostic must not stop a paid run
        warn(f"external cohort unavailable, continuing without it: {error!r}")
        cohort_library = cohort_store = cohort_template = None
        cohort_ids = []
    missing = [i for i in panel_ids if i not in library.index_by_entry_id]
    if missing:
        raise RThetaTrainingError(
            f"{len(missing):,} reserve entries are absent from the train library")
    print(
        f"[{time.perf_counter() - segment_started:6.1f}s] library {len(library):,} = "
        f"train {len(training_ids):,} + reserve {len(panel_ids):,}",
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
    build_cohort = (make_builder(cohort_library, cohort_store, cohort_template)
                    if cohort_library is not None else None)
    # One pass over the realized sequence, which is 136,027 rows now --
    # not the 144,870 the old manifest carried.
    epoch_steps = max(1, -(-len(sequence) // batch_size))

    identity = RunIdentity(
        initialization_seed=int(binding["model_runtime"]["initialization_seed"]),
        initial_model_state_sha256=str(binding["model_runtime"]["initial_model_state_sha256"]),
        library_sha256=str(freeze["library_sha256"]),
        # The reserve digest IS the split identity now: the frozen freeze file
        # describes the pre-reserve division and would bind the wrong one.
        split_sha256=str(law["matched_validation_reserve_source_digest"]),
        sampling_law_sha256=str(law["frozen_sha256"]),
        manifest_sha256=str(manifest["frozen_sha256"]),
        packed_store_records_sha256=str(store.header["records_sha256"]),
        eval_panel_sha256=str(gate["reserve_source_digest"]),
    )
    # The digest of the SEQUENCE itself, not of the manifest that produced it.
    # A resume must verify the order it is about to consume, and two manifests
    # can agree while their sequences differ.
    stream_sha256 = str(manifest["sequence_sha256"])

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    completed_steps = 0
    resume_count = 0
    collapse_baseline = None
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
        # Restore rather than recapture: a baseline taken from a
        # mid-training summary would compare each capability against an
        # already degraded reference, and the gate would detect nothing.
        collapse_baseline = payload.get("collapse_baseline")
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

        # ALSO on the final step. maximum_steps need not be a multiple of
        # evaluate_every -- one epoch is 4,251 steps against an interval of 500
        # -- so the run previously ended with its last evaluation 251 steps in
        # the past. The final model state was never scored and so could never
        # be selected, and the epoch-boundary cohort diagnostic, nested inside
        # this block, could never fire either.
        if (completed_steps % evaluate_every == 0
                or completed_steps >= maximum_steps):
            rows = evaluate_panel(
                model, entry_ids=panel_ids, build_batch=build_panel,
                batch_size=batch_size,
                lane_by_entry_id=reserve_lane_by_entry_id,
                support_band_by_entry_id=reserve_band_by_entry_id,
            )
            summary = summarize_panel(
                rows,
                deployment_family_share=law["realized_coefficients"]["by_family"],
                # The law defines the reference process R_theta estimates, so it
                # supplies the selection weights. Strata it gives zero mass are
                # reported under zero_mass_stratum and never vote.
                reference_law_stratum_share=reference_law_stratum_share,
            )
            if collapse_baseline is None:
                # First evaluation of the run defines "before training".
                collapse_baseline = capability_baseline(summary)
                print(f"  collapse baseline captured over "
                      f"{len(collapse_baseline)} capabilities", flush=True)
            collapsed = collapsed_capabilities(summary, baseline=collapse_baseline)
            if collapsed:
                warn(f"capability collapse at step {completed_steps}: "
                     f"{list(collapsed)}")
            criterion = selection_criterion(
                summary, step=completed_steps, baseline=collapse_baseline)

            cohort_mean_nll = None
            # Trigger on an epoch boundary OR the final step. Requiring
            # `% epoch_steps == 0` INSIDE the `% evaluate_every == 0` block
            # meant both had to divide the step: with epoch_steps 4,251
            # (= 3 x 13 x 109) and evaluate_every 500 (= 2^2 x 5^3) the gcd is
            # 1, so the two conditions could not coincide before step 2,125,500
            # and the diagnostic never ran at all.
            at_epoch_boundary = (
                completed_steps % epoch_steps == 0
                or completed_steps >= maximum_steps
                or completed_steps // epoch_steps
                > (completed_steps - evaluate_every) // epoch_steps
            )
            if cohort_ids and at_epoch_boundary:
                try:
                    cohort_rows = evaluate_panel(
                        model, entry_ids=cohort_ids, build_batch=build_cohort,
                        batch_size=batch_size,
                    )
                    cohort_mean_nll = sum(
                        r["teacher_successor_nll"] for r in cohort_rows
                    ) / len(cohort_rows)
                    print(f"  external cohort (diagnostic, not selection): "
                          f"{cohort_mean_nll:.4f} over {len(cohort_rows):,}",
                          flush=True)
                except Exception as error:  # noqa: BLE001
                    warn(f"external cohort evaluation failed: {error!r}")
            trajectory.append(
                {
                    "step": completed_steps,
                    "panel_native_mean_nll": summary["panel_native"]["mean_nll"],
                    "panel_native_minimum_probability":
                        summary["panel_native"]["minimum_probability"],
                    "deployment_weighted_mean_nll": summary["deployment_weighted_mean_nll"],
                    # The selection number. Recorded first-class so the
                    # trajectory shows what actually chose the checkpoint.
                    "reference_law_weighted_mean_nll":
                        summary["reference_law_weighted_mean_nll"],
                    "eligible": bool(criterion[0]),
                    "collapsed_capabilities": list(collapsed),
                    "capability_floor": capability_floor(summary),
                    "by_support_band_mean_nll": {
                        name: value["mean_nll"]
                        for name, value in summary["by_support_band"].items()
                    },
                    "zero_mass_stratum_mean_nll": {
                        name: value["mean_nll"]
                        for name, value in summary["zero_mass_stratum"].items()
                    },
                    "external_cohort_mean_nll": cohort_mean_nll,
                    "criterion": list(criterion),
                    # Keep the per-family and per-cell means. Without them the
                    # aggregates can move in opposite directions -- as they did
                    # here, panel-native improving while deployment-weighted
                    # worsened -- with nothing recorded to say which families
                    # are responsible. Computing a diagnostic every 250 steps
                    # and discarding it is worse than not computing it.
                    "by_family_factor_nll": {
                        name: {"family": value.get("mean_family_nll"),
                               "identity": value.get("mean_identity_nll")}
                        for name, value in summary["by_family"].items()
                    },
                    "by_family_mean_nll": {
                        name: value["mean_nll"]
                        for name, value in summary["by_family"].items()
                    },
                    "by_cell_mean_nll": {
                        name: value["mean_nll"]
                        for name, value in summary["by_capability_cell"].items()
                    },
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
            families = summary["by_family"]
            # REF-LAW leads the line because it is the number that selects.
            # The first version of this print carried panel-native and
            # deployment only, so the log showed everything except the
            # quantity the checkpoint was actually chosen by.
            bands = summary["by_support_band"]
            print(
                f"[step {completed_steps:6,}] REF-LAW "
                f"{summary['reference_law_weighted_mean_nll']:.4f}  "
                f"{'ELIGIBLE' if criterion[0] else 'GATED'}  "
                f"native {summary['panel_native']['mean_nll']:.4f}  "
                f"deployment {summary['deployment_weighted_mean_nll']:.4f}",
                flush=True,
            )
            if bands:
                print(
                    "           bands     "
                    + "  ".join(f"{name} {value['mean_nll']:.3f}"
                               for name, value in bands.items()),
                    flush=True,
                )
            if summary["zero_mass_stratum"]:
                print(
                    "           zero-mass "
                    + "  ".join(
                        f"{name.split('|')[0][:6]}.{name.split('|')[1][:4]} "
                        f"{value['mean_nll']:.2f}"
                        for name, value in summary["zero_mass_stratum"].items()),
                    flush=True,
                )
            print(
                "           families  "
                + "  ".join(
                    f"{name.split('_')[0][:5]}.{name.split('_')[-1][:3]} "
                    f"{value['mean_nll']:.2f}"
                    for name, value in sorted(families.items())
                ),
                flush=True,
            )
            # The aggregates moved in opposite directions in this run, so watch
            # the law-heavy families explicitly rather than only the means.
            if len(trajectory) > 1:
                earlier_families = trajectory[-2].get("by_family_mean_nll", {})
                regressed = sorted(
                    name
                    for name, value in families.items()
                    if name in earlier_families
                    and value["mean_nll"] > earlier_families[name] * 1.05
                )
                if regressed:
                    warn(f"held-out NLL rose >5% for families: {regressed}")
            if (
                summary["deployment_weighted_mean_nll"] is not None
                and len(trajectory) > 1
                and trajectory[-2]["deployment_weighted_mean_nll"] is not None
                and summary["deployment_weighted_mean_nll"]
                > trajectory[-2]["deployment_weighted_mean_nll"] * 1.03
            ):
                warn(
                    "deployment-weighted NLL rose "
                    f"{trajectory[-2]['deployment_weighted_mean_nll']:.4f} -> "
                    f"{summary['deployment_weighted_mean_nll']:.4f} while "
                    f"panel-native is {summary['panel_native']['mean_nll']:.4f}"
                )

        if completed_steps % checkpoint_every == 0:
            assert_training_invariants(model, hazard_before=hazard_before, step=completed_steps)
            written = write_checkpoint(
                checkpoint_path,
                model=model, optimizer=optimizer, identity=identity,
                completed_steps=completed_steps, selected_step=selected_step,
                selected_criterion=selected_criterion_value, selected_state=selected_state,
                trajectory=trajectory, stream_sha256=stream_sha256,
                resume_count=resume_count, collapse_baseline=collapse_baseline,
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
        collapse_baseline=collapse_baseline,
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
