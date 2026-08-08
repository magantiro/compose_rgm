"""Corpus-scale editing training over published prep slices.

WHY THIS IS A NEW FUNCTION RATHER THAN A WIDER P50
--------------------------------------------------
``run_process_v2_p50`` refuses anything but exactly fifty optimizer steps of
batch sixty-four, against a 3,200-row stream it re-derives and re-authenticates.
Those bounds are deliberate -- P50 *is* the frozen bounded pilot -- so widening
them would defeat a guard rather than reuse one.  This module is separate for
that reason, and reuses every scoring, freezing, indexing and abort helper
instead of restating them.

WHAT IT FIXES RELATIVE TO THE BARRED 16,000-STEP RUN
-----------------------------------------------------
That run finished cleanly and improved in aggregate while ring opening and graft
were driven to ~0, and it got to step 16,000 undetected because monitoring
tracked only aggregate loss.  Two properties answer that directly:

* **the corpus is streamed, not selected.**  The barred checkpoints were data
  starved -- 6,922 unique records over ~28 passes.  Published slices are read
  shard by shard, so unique data seen grows with run length.
* **capability is re-measured on a cadence and a losing family aborts.**  Per
  step gradient monitoring cannot see this failure: a head receiving finite
  nonzero gradient every step can still be trained to zero capability.  Gradient
  proves motion, not direction.

WHAT IT DOES NOT DO
-------------------
It publishes no authorized artifact, selects no checkpoint, and never moves the
total hazard.  Measured, so the reason is on record: under
``factorized_successor_identity_loss`` the total-hazard head receives NO
gradient at all, even fully unfrozen -- the hazard is frozen by the OBJECTIVE.
The ``requires_grad_`` freeze and the per-step identity assertion this module
keeps (both mirroring P50) are therefore defence in depth against a future
change to the loss, not the mechanism.  This trains the productive jump law
only.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from compose_v4.data.editing_v2_process_v2_schema import (
    canonical_sha256,
    verify_self_hash,
)
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.editing_v2_process_v2_chunk_compile import (
    BATCH_FILENAME,
    CHUNKS_DIRNAME,
    ENTRIES_FILENAME,
    RECEIPT_FILENAME,
    TRAIN_ROLE,
)
from compose_v4.experiments.editing_v2_process_v2_p50_runtime import (
    _COMPILE_SUPPORT_TIME,
    _teacher_fiber_from_payload,
    _time_hex,
    capability_regressions,
)
from compose_v4.experiments.editing_v2_semantic_t1_capacity_runner import (
    ACTION_ROUTE_PREFIXES,
    TOTAL_HAZARD_PREFIX,
    _assert_hazard_identity,
    _attach_successor_family_coordinates,
    _clone_state_dict,
    _hazard_state,
    _index_factorized_batch,
)
from compose_v4.experiments.factorized_successor_training import (
    factorized_successor_identity_loss,
    forward_teacher_successor_batch,
)
from compose_v4.experiments.successor_micro_overfit import (
    configure_micro_overfit_parameters,
)
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel

VALIDATION_ROLE = "validation"
CHECKPOINT_DIRNAME = "checkpoints"


class ProcessV2TrainingError(RuntimeError):
    """A corpus-scale editing run cannot proceed."""


@dataclass(frozen=True)
class LoadedSlice:
    """One published prep slice, authenticated and ready to score."""

    directory: Path
    task_identity_sha256: str
    partition_role: str
    entry_offset: int
    entries: tuple[Mapping[str, Any], ...]
    fibers: tuple[Any, ...]
    batch: Any

    @property
    def entry_count(self) -> int:
        return len(self.entries)


def iter_slice_directories(prep_root: Path) -> tuple[Path, ...]:
    """Every published slice under a prep root, in deterministic order.

    Sorted, so a run's data order is a function of the corpus and the seed
    rather than of filesystem enumeration order.
    """

    chunks = Path(prep_root) / CHUNKS_DIRNAME
    if not chunks.is_dir():
        raise ProcessV2TrainingError(f"prep root holds no {CHUNKS_DIRNAME}/: {prep_root}")
    found = sorted(
        directory for directory in chunks.glob("*/*") if (directory / RECEIPT_FILENAME).is_file()
    )
    if not found:
        raise ProcessV2TrainingError(f"prep root holds no published slice: {prep_root}")
    return tuple(found)


def load_published_slice(directory: Path, *, expected_role: str | None = None) -> LoadedSlice:
    """Load one slice, refusing content its own receipt does not describe.

    The receipt is the authority: it is self-hashed and binds both payloads by
    content hash, so the byte check happens BEFORE ``torch.load`` unpickles
    anything.  ``expected_role`` is a parameter rather than a post-filter so a
    sealed role cannot be opened by passing the wrong root.
    """

    directory = Path(directory)
    try:
        receipt = json.loads((directory / RECEIPT_FILENAME).read_text())
        entries_bytes = (directory / ENTRIES_FILENAME).read_bytes()
        batch_bytes = (directory / BATCH_FILENAME).read_bytes()
    except (OSError, ValueError) as error:
        raise ProcessV2TrainingError(f"unreadable prep slice: {directory}") from error
    try:
        verify_self_hash(receipt, field="receipt_sha256", label="a prep slice receipt")
    except Exception as error:  # noqa: BLE001 - the schema module raises its own type
        raise ProcessV2TrainingError(f"{directory}: {error}") from error

    if hashlib.sha256(entries_bytes).hexdigest() != str(
        receipt["entries_file_sha256"]
    ) or hashlib.sha256(batch_bytes).hexdigest() != str(receipt["batch_file_sha256"]):
        raise ProcessV2TrainingError(f"prep slice bytes disagree with its receipt: {directory}")

    role = str(receipt["partition_role"])
    if expected_role is not None and role != expected_role:
        raise ProcessV2TrainingError(
            f"prep slice is role {role!r}, not {expected_role!r}: {directory}"
        )

    entries_payload = json.loads(entries_bytes)
    if canonical_sha256(entries_payload) != str(receipt["entries_payload_sha256"]):
        raise ProcessV2TrainingError(f"prep slice entries payload disagrees: {directory}")
    entries = tuple(entries_payload["entries"])
    if int(receipt["entry_count"]) != len(entries):
        raise ProcessV2TrainingError(f"prep slice entry count disagrees: {directory}")

    batch = torch.load(io.BytesIO(batch_bytes), weights_only=False)
    if int(getattr(batch, "batch_size", -1)) != len(entries):
        raise ProcessV2TrainingError(
            f"prep slice tensors hold {getattr(batch, 'batch_size', None)} rows "
            f"for {len(entries)} entries: {directory}"
        )
    # The chunk compile stores the RAW collator output -- unlike every P50 batch,
    # its successor-family coordinate is not attached.  Attaching once per slice
    # is both correct and cheap; `_index_factorized_batch` carries the field
    # through, so minibatches inherit it.
    batch = _attach_successor_family_coordinates(batch, entries)
    fibers = tuple(_teacher_fiber_from_payload(entry["teacher_successor_fiber"]) for entry in entries)
    return LoadedSlice(
        directory=directory,
        task_identity_sha256=str(receipt["task_identity_sha256"]),
        partition_role=role,
        entry_offset=int(receipt["entry_offset"]),
        entries=entries,
        fibers=fibers,
        batch=batch,
    )


# ---- Validation panel ----


@dataclass(frozen=True)
class _PanelShard:
    batch: Any
    fibers: tuple[Any, ...]
    entries: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ValidationPanel:
    """A fixed held-out panel, balanced per family.

    Balance is the point.  The failure this guards against is a rare family
    collapsing while the aggregate improves, so a panel drawn in corpus
    proportion would hide exactly the signal it exists to expose.
    """

    shards: tuple[_PanelShard, ...]
    families: tuple[str, ...]
    example_count: int
    family_counts: Mapping[str, int]


def build_validation_panel(
    validation_root: Path,
    *,
    examples_per_family: int = 64,
    expected_role: str = VALIDATION_ROLE,
) -> ValidationPanel:
    """Draw a fixed, per-family-balanced panel from published validation slices."""

    if type(examples_per_family) is not int or examples_per_family <= 0:
        raise ProcessV2TrainingError("examples_per_family must be a positive int")
    taken: dict[str, int] = {}
    shards: list[_PanelShard] = []
    for directory in iter_slice_directories(Path(validation_root)):
        loaded = load_published_slice(directory, expected_role=expected_role)
        indices: list[int] = []
        for index, entry in enumerate(loaded.entries):
            family = str(entry["model_family"])
            if taken.get(family, 0) < examples_per_family:
                taken[family] = taken.get(family, 0) + 1
                indices.append(index)
        if not indices:
            continue
        # Successive checks must differ only by the model, so the panel has to
        # sit at the one time it was collated at.  A panel collated elsewhere
        # would make baseline and observed incomparable without ever erroring.
        if not bool(
            torch.all(loaded.batch.times == torch.tensor(_COMPILE_SUPPORT_TIME).to(loaded.batch.times))
        ):
            raise ProcessV2TrainingError(
                f"validation slice is not collated at the compile support time: {directory}"
            )
        shards.append(
            _PanelShard(
                batch=_index_factorized_batch(loaded.batch, indices),
                fibers=tuple(loaded.fibers[index] for index in indices),
                entries=tuple(loaded.entries[index] for index in indices),
            )
        )
    if not shards:
        raise ProcessV2TrainingError(f"validation root yielded no panel rows: {validation_root}")
    return ValidationPanel(
        shards=tuple(shards),
        families=tuple(sorted(taken)),
        example_count=sum(taken.values()),
        family_counts=dict(sorted(taken.items())),
    )


def significant_capability_regressions(
    baseline: Mapping[str, Any],
    observed: Mapping[str, Any],
    ceilings: Mapping[str, float],
    *,
    minimum_sigma: float = 2.0,
) -> list[dict[str, Any]]:
    """Families whose WITHIN-family NLL rose past their ceiling AND the noise.

    Two corrections to the naive gate, both learned by getting them wrong.

    Wrong quantity: the joint successor probability charges a family for its own
    scarcity, so correctly calibrating a 1.2% prior looks exactly like forgetting
    the operation.  At step 400 the joint metric flagged ring and bond_reorder
    while their within-family capability was flat or improving, and said nothing
    about bond_reroute, the one family genuinely degrading on held-out data.

    No noise floor: a thirty-two example panel carries a standard error of
    +-0.03 to +-0.27 nats per family, so deltas of +0.06 and +0.26 were read as
    regressions when both sat at one sigma.  A regression must clear its own
    uncertainty before it is called one.
    """

    base = baseline["within_by_family"]
    obs = observed["within_by_family"]
    samples = observed.get("within_samples", {})
    base_samples = baseline.get("within_samples", {})
    rows: list[dict[str, Any]] = []
    for family in sorted(ceilings):
        if family not in base or family not in obs:
            raise ProcessV2TrainingError(
                f"capability check lacks a within-family measurement for {family!r}"
            )
        regression = float(obs[family]) - float(base[family])
        paired = base_samples.get(family) and samples.get(family)
        if paired and len(base_samples[family]) == len(samples[family]):
            # Paired per-example differences: the panel is fixed, so pairing
            # removes between-example variance and measures the shift itself.
            deltas = [
                samples[family][i] - base_samples[family][i]
                for i in range(len(samples[family]))
            ]
            n = len(deltas)
            mean = sum(deltas) / n
            variance = sum((x - mean) ** 2 for x in deltas) / (n - 1) if n > 1 else 0.0
            standard_error = math.sqrt(variance / n) if n > 1 else 0.0
        else:
            mean, standard_error = regression, 0.0
        sigma = abs(mean) / standard_error if standard_error > 0 else float("inf")
        if regression > float(ceilings[family]) and sigma >= float(minimum_sigma):
            rows.append(
                {
                    "family": family,
                    "within_family_regression_nats": regression,
                    "standard_error_nats": standard_error,
                    "sigma": sigma,
                    "ceiling_nats": float(ceilings[family]),
                }
            )
    return rows


def evaluate_validation_panel(
    model: FactorizedTraceletRateModel, panel: ValidationPanel, *, batch_size: int = 64
) -> dict[str, Any]:
    """Per-family canonical-successor NLL on the fixed panel.

    Evaluated at ``_COMPILE_SUPPORT_TIME``, the time the panel was collated at
    and the same time ``_evaluation_rows`` uses, so successive checks differ
    only by the model.
    """

    rows: list[tuple[str, float]] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for shard in panel.shards:
                for start in range(0, len(shard.entries), batch_size):
                    indices = tuple(range(start, min(start + batch_size, len(shard.entries))))
                    batch = _index_factorized_batch(shard.batch, indices)
                    prediction = forward_teacher_successor_batch(
                        model,
                        batch.to(model.device),
                        tuple(shard.fibers[index] for index in indices),
                    )
                    joint = -prediction.selected_productive_successor_log_probability.detach().cpu()
                    # WITHIN-family is the capability signal.  The joint quantity
                    # folds in how often the model thinks a family occurs at all,
                    # so a model correctly learning that ring edits are 1.2% of
                    # the data scores ring worse on it -- indistinguishable from
                    # forgetting the operation.  Measured at step 400: ring's
                    # joint rose 1.509 nats while its within-family capability
                    # IMPROVED by 0.063.  The controller chooses the family at
                    # inference, so the family prior is precisely what gets
                    # overridden; within-family is what determines whether the
                    # editor works.
                    within = (
                        -prediction.selected_within_teacher_family_log_probability.detach().cpu()
                    )
                    for position, index in enumerate(indices):
                        rows.append(
                            (
                                str(shard.entries[index]["model_family"]),
                                float(joint[position]),
                                float(within[position]),
                            )
                        )
    finally:
        model.train(was_training)

    by_family: dict[str, float] = {}
    within_by_family: dict[str, float] = {}
    within_samples: dict[str, list[float]] = {}
    for family in sorted({family for family, _joint, _within in rows}):
        selected = [j for name, j, _w in rows if name == family]
        by_family[family] = sum(selected) / len(selected)
        w = [x for name, _j, x in rows if name == family]
        within_samples[family] = w
        within_by_family[family] = sum(w) / len(w)
    return {
        "within_by_family": within_by_family,
        "within_samples": within_samples,
        "example_count": len(rows),
        "overall_mean_successor_nll": sum(j for _n, j, _w in rows) / len(rows),
        "by_family": by_family,
    }


# ---- Training ----


def _minibatch_plan(
    entry_count: int, *, batch_size: int, rng: random.Random
) -> tuple[tuple[int, ...], ...]:
    """Shuffle one slice into whole minibatches, dropping only a short tail."""

    order = list(range(entry_count))
    rng.shuffle(order)
    return tuple(
        tuple(order[start : start + batch_size])
        for start in range(0, entry_count - batch_size + 1, batch_size)
    )


def _sampled_times(
    entries: Sequence[Mapping[str, Any]], *, stream_index: int, seed: int
) -> tuple[float, ...]:
    """Per-example U(0,1) times from the audited P50 derivation.

    Reused rather than reimplemented so training times here are drawn by the
    same law the bounded pilot was measured under.  Times do not enter the
    collated chemistry -- the P50 materialized path overrides them on a cached
    batch for exactly that reason -- so overriding them on a loaded slice is
    sound.
    """

    return tuple(
        float.fromhex(_time_hex(stream_index=stream_index + offset, candidate=entry, seed=seed))
        for offset, entry in enumerate(entries)
    )


def run_process_v2_training(
    model: FactorizedTraceletRateModel,
    prep_root: Path,
    *,
    optimizer_steps: int,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.0,
    gradient_clip_norm: float = 10.0,
    seed: int = 31,
    panel: ValidationPanel | None = None,
    capability_check_every_steps: int = 0,
    maximum_family_nll_regression: Mapping[str, float] | None = None,
    checkpoint_every_steps: int = 0,
    checkpoint_root: Path | None = None,
    resume_from: Path | None = None,
    resume_weights_only: bool = False,
    parameter_scope: str = "all",
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train the productive jump law over published prep slices.

    ``capability_check_every_steps`` re-measures per-family NLL on the fixed
    panel and ABORTS when a required family regresses past its ceiling.  It
    defaults to 0 only so the function is testable without a panel; a real run
    must set it, because it is the single guard the barred 16,000-step run
    lacked.
    """

    if type(optimizer_steps) is not int or optimizer_steps <= 0:
        raise ProcessV2TrainingError("optimizer_steps must be a positive int")
    if type(batch_size) is not int or batch_size <= 0:
        raise ProcessV2TrainingError("batch_size must be a positive int")
    if type(capability_check_every_steps) is not int or capability_check_every_steps < 0:
        raise ProcessV2TrainingError("capability cadence must be a nonnegative int")
    # Fail closed: a cadence without ceilings evaluates and never refuses, which
    # reads as protection while providing none.
    if bool(capability_check_every_steps) != (maximum_family_nll_regression is not None):
        raise ProcessV2TrainingError(
            "the capability cadence and its per-family ceilings must be set together"
        )
    if capability_check_every_steps and panel is None:
        raise ProcessV2TrainingError("a capability cadence requires a validation panel")
    if checkpoint_every_steps and checkpoint_root is None:
        raise ProcessV2TrainingError("a checkpoint cadence requires a checkpoint root")

    slice_dirs = iter_slice_directories(Path(prep_root))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)

    hazard_initial = _hazard_state(model)
    named_parameters = dict(model.named_parameters())
    if parameter_scope == "all":
        for name, parameter in named_parameters.items():
            parameter.requires_grad_(True)
    else:
        # A restricted scope freezes the shared trunk and trains only the family
        # heads plus their local adapters.  This exists because rare families do
        # not lose to gradient starvation alone -- they lose because the trunk is
        # rewritten beneath them.  `ring_system_restate` reads `pair_project.`
        # and two restate embeddings; `atom_insert` at twenty-four percent of the
        # data rewrites that surface while ring gets one percent of the gradient,
        # so ring's inputs drift out from under a head that cannot hold them.
        # Freezing the trunk makes the representation stationary, which is the
        # regime the T1 capacity ladder recorded for this family.
        configure_micro_overfit_parameters(
            model, tuple(ACTION_ROUTE_PREFIXES), scope=parameter_scope
        )
    # The hazard stays frozen under EVERY scope: "all" would otherwise unfreeze
    # it, and the objective is hazard-free by construction.
    trainable: list[torch.Tensor] = []
    for name, parameter in named_parameters.items():
        if name.startswith(TOTAL_HAZARD_PREFIX):
            parameter.requires_grad_(False)
        if parameter.requires_grad:
            trainable.append(parameter)
    if not trainable:
        raise ProcessV2TrainingError(f"scope {parameter_scope!r} selected no trainable parameter")
    optimizer = torch.optim.AdamW(
        trainable, lr=float(learning_rate), weight_decay=float(weight_decay)
    )

    resumed_from_step = 0
    if resume_from is not None:
        if resume_weights_only:
            # A PHASE CHANGE, not an ordinary resume.  AdamW moments are indexed
            # by the trainable parameter set, so they cannot cross a scope
            # change: phase one trains the whole model, phase two trains heads
            # only, and the optimizer groups do not correspond.  Restoring
            # weights while starting fresh moments is the correct semantics, and
            # it is opt-in precisely so it can never happen by accident -- a
            # silent fresh optimizer would look like a resume and behave like a
            # restart, which is what `load_training_checkpoint` exists to refuse.
            payload = torch.load(Path(resume_from), weights_only=False)
            model.load_state_dict(payload["model_state"])
            resumed_from_step = int(payload["optimizer_step"])
            rng.setstate(payload["sampler_rng_state"])
            torch.set_rng_state(payload["torch_rng_state"])
            hazard_initial = _hazard_state(model)
        else:
            resumed_from_step, sampler_state = load_training_checkpoint(
                Path(resume_from), model, optimizer
            )
            rng.setstate(sampler_state)
        if resumed_from_step >= optimizer_steps:
            raise ProcessV2TrainingError(
                f"the checkpoint is already at step {resumed_from_step}, "
                f"which is not before the requested {optimizer_steps}"
            )
        # The hazard baseline must come from the RESUMED weights, or the
        # identity assertion would compare against a scratch model.
        hazard_initial = _hazard_state(model)

    baseline: dict[str, Any] | None = None
    ceilings: dict[str, float] = {}
    if maximum_family_nll_regression is not None:
        assert panel is not None
        ceilings = {
            str(family): float(value) for family, value in maximum_family_nll_regression.items()
        }
        if any(not math.isfinite(value) or value < 0.0 for value in ceilings.values()):
            raise ProcessV2TrainingError("capability ceilings must be finite and nonnegative")
        baseline = evaluate_validation_panel(model, panel, batch_size=batch_size)
        missing = set(ceilings) - set(baseline["within_by_family"])
        if missing:
            # "not measured" and "did not regress" must never look alike.
            raise ProcessV2TrainingError(
                f"the validation panel cannot measure ceilinged families: {sorted(missing)}"
            )

    family_exposure: dict[str, dict[str, Any]] = {
        family: {
            "observed_example_count": 0,
            # Both counters, because "nonzero on 5 steps" is uninterpretable
            # without the number of steps the family was actually present in.
            "exposure_steps": 0,
            "finite_nonzero_action_route_gradient_exposure_steps": 0,
            "cumulative_exposure_step_action_route_gradient_l2": 0.0,
        }
        for family in ACTION_ROUTE_PREFIXES
    }
    trajectory: list[dict[str, Any]] = []
    capability_checks: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    losses: list[float] = []
    slices_read = 0
    unique_entries_seen: set[str] = set()
    step = resumed_from_step
    epochs = 0
    aborted: dict[str, Any] | None = None

    model.train()
    while step < optimizer_steps and aborted is None:
        epochs += 1
        # Shuffle slice order per epoch, not just rows within a slice.  Fixed
        # slice order would make every epoch present the corpus in the same
        # sequence, which correlates a family's exposure with the point in the
        # epoch it sits at.
        epoch_slices = list(slice_dirs)
        rng.shuffle(epoch_slices)
        for directory in epoch_slices:
            if step >= optimizer_steps or aborted is not None:
                break
            loaded = load_published_slice(directory, expected_role=TRAIN_ROLE)
            slices_read += 1
            for indices in _minibatch_plan(loaded.entry_count, batch_size=batch_size, rng=rng):
                if step >= optimizer_steps:
                    break
                entries = tuple(loaded.entries[index] for index in indices)
                fibers = tuple(loaded.fibers[index] for index in indices)
                batch = _index_factorized_batch(loaded.batch, indices)
                times = _sampled_times(entries, stream_index=step * batch_size, seed=seed)
                batch = replace(
                    batch,
                    times=torch.tensor(times, dtype=batch.times.dtype, device=batch.times.device),
                )

                optimizer.zero_grad(set_to_none=True)
                device_batch = batch.to(model.device)
                prediction = forward_teacher_successor_batch(model, device_batch, fibers)
                loss = factorized_successor_identity_loss(prediction, device_batch)
                if not bool(torch.isfinite(loss)):
                    raise ProcessV2TrainingError(f"nonfinite loss at optimizer step {step + 1}")
                loss.backward()

                for name, parameter in named_parameters.items():
                    if name.startswith(TOTAL_HAZARD_PREFIX) and parameter.grad is not None:
                        raise ProcessV2TrainingError(
                            f"frozen hazard parameter received a gradient: {name}"
                        )
                    if (
                        parameter.grad is not None
                        and not name.startswith(TOTAL_HAZARD_PREFIX)
                        and not bool(torch.isfinite(parameter.grad).all())
                    ):
                        raise ProcessV2TrainingError(
                            f"nonfinite gradient on {name} at optimizer step {step + 1}"
                        )

                present: dict[str, int] = {}
                for entry in entries:
                    family = str(entry["model_family"])
                    present[family] = present.get(family, 0) + 1
                for family, count in present.items():
                    route_gradients = [
                        parameter.grad
                        for name, parameter in named_parameters.items()
                        if name.startswith(ACTION_ROUTE_PREFIXES[family])
                        and parameter.requires_grad
                        and parameter.grad is not None
                    ]
                    route_l2 = math.sqrt(
                        sum(float(gradient.detach().norm()) ** 2 for gradient in route_gradients)
                    )
                    if not route_gradients or not math.isfinite(route_l2):
                        raise ProcessV2TrainingError(
                            f"nonfinite or absent {family} action-route gradient "
                            f"at optimizer step {step + 1}"
                        )
                    exposure = family_exposure[family]
                    exposure["observed_example_count"] += count
                    exposure["exposure_steps"] += 1
                    exposure["finite_nonzero_action_route_gradient_exposure_steps"] += int(
                        route_l2 > 0.0
                    )
                    exposure["cumulative_exposure_step_action_route_gradient_l2"] += route_l2

                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable, float(gradient_clip_norm)
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise ProcessV2TrainingError(
                        f"nonfinite gradient norm at optimizer step {step + 1}"
                    )
                optimizer.step()
                _assert_hazard_identity(model, hazard_initial)

                step += 1
                losses.append(float(loss.detach().cpu()))
                unique_entries_seen.update(str(entry["p50_entry_sha256"]) for entry in entries)
                trajectory.append(
                    {
                        "optimizer_step": step,
                        "mean_batch_canonical_successor_nll": losses[-1],
                        "global_gradient_l2": float(gradient_norm),
                    }
                )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "process_v2_training_optimizer_step",
                            "completed": step,
                            "total": optimizer_steps,
                            "loss": losses[-1],
                        }
                    )

                if checkpoint_every_steps and step % checkpoint_every_steps == 0:
                    checkpoints.append(
                        _write_checkpoint(
                            model,
                            root=Path(checkpoint_root),
                            step=step,
                            optimizer=optimizer,
                            rng=rng,
                        )
                    )

                if capability_check_every_steps and step % capability_check_every_steps == 0:
                    assert panel is not None and baseline is not None
                    observed = evaluate_validation_panel(model, panel, batch_size=batch_size)
                    regressions = significant_capability_regressions(
                        baseline, observed, ceilings
                    )
                    capability_checks.append(
                        {
                            "optimizer_step": step,
                            "overall_mean_successor_nll": float(
                                observed["overall_mean_successor_nll"]
                            ),
                            "by_family": {
                                family: float(value)
                                for family, value in sorted(observed["by_family"].items())
                            },
                            "regressions": regressions,
                        }
                    )
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "phase": "process_v2_training_capability_check",
                                "optimizer_step": step,
                                "regressions": len(regressions),
                            }
                        )
                    if regressions:
                        worst = max(regressions, key=lambda row: row["within_family_regression_nats"])
                        aborted = {
                            "optimizer_step": step,
                            "family": worst["family"],
                            "within_family_regression_nats": float(
                                worst["within_family_regression_nats"]
                            ),
                            "standard_error_nats": float(worst["standard_error_nats"]),
                            "sigma": float(worst["sigma"]),
                            "ceiling_nats": float(worst["ceiling_nats"]),
                        }
                        break

    final = (
        evaluate_validation_panel(model, panel, batch_size=batch_size)
        if panel is not None
        else None
    )
    _assert_hazard_identity(model, hazard_initial)
    return {
        "optimizer_steps_requested": optimizer_steps,
        "optimizer_steps_completed": step,
        "resumed_from_optimizer_step": resumed_from_step,
        "epochs_started": epochs,
        "slices_available": len(slice_dirs),
        "slices_read": slices_read,
        "unique_training_examples_seen": len(unique_entries_seen),
        "losses": losses,
        "trajectory": trajectory,
        "family_exposure": family_exposure,
        "baseline_validation": baseline,
        "final_validation": final,
        "capability_checks": capability_checks,
        "capability_abort": aborted,
        "checkpoints": checkpoints,
        "final_model_state_sha256": state_dict_semantic_sha256(model.state_dict()),
        "parameter_scope": parameter_scope,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "hazard_frozen": True,
        "artifact_published": False,
        "checkpoint_selection_authorized": False,
    }


def _write_checkpoint(
    model: FactorizedTraceletRateModel,
    *,
    root: Path,
    step: int,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
) -> dict[str, Any]:
    """Write one resumable training checkpoint to disk.

    To disk rather than into the returned dict, because a long run keeping every
    state dict in memory is itself a failure mode.

    The OPTIMIZER and RNG states are stored alongside the weights because a
    checkpoint without them is not a resume point: AdamW's moments and the data
    order would restart, so a preempted run would repeat work rather than make
    forward progress -- the exact failure a retry budget cannot fix on its own.
    """

    directory = Path(root) / CHECKPOINT_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    state = _clone_state_dict(model)
    path = directory / f"step-{step:09d}.pt"
    torch.save(
        {
            "optimizer_step": step,
            "model_state": state,
            "optimizer_state": optimizer.state_dict(),
            "sampler_rng_state": rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "model_state_sha256": state_dict_semantic_sha256(state),
        },
        path,
    )
    return {
        "optimizer_step": step,
        "path": str(path),
        "model_state_sha256": state_dict_semantic_sha256(state),
        "resumable": True,
        "checkpoint_selection_authorized": False,
    }


def load_training_checkpoint(
    path: Path, model: FactorizedTraceletRateModel, optimizer: torch.optim.Optimizer
) -> tuple[int, Any]:
    """Restore a run from a checkpoint, refusing a partial restore.

    Returns the completed step count and the sampler RNG state.  A checkpoint
    missing any component raises rather than silently resuming with fresh
    optimizer moments, which would look like a resume and behave like a restart.
    """

    try:
        payload = torch.load(Path(path), weights_only=False)
    except (OSError, ValueError) as error:
        raise ProcessV2TrainingError(f"unreadable training checkpoint: {path}") from error
    required = (
        "optimizer_step",
        "model_state",
        "optimizer_state",
        "sampler_rng_state",
        "torch_rng_state",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ProcessV2TrainingError(
            f"training checkpoint cannot resume, it lacks {missing}: {path}"
        )
    model.load_state_dict(payload["model_state"])
    if state_dict_semantic_sha256(_clone_state_dict(model)) != str(
        payload["model_state_sha256"]
    ):
        raise ProcessV2TrainingError(f"training checkpoint weights disagree: {path}")
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    return int(payload["optimizer_step"]), payload["sampler_rng_state"]


__all__ = [
    "CHECKPOINT_DIRNAME",
    "load_training_checkpoint",
    "LoadedSlice",
    "ProcessV2TrainingError",
    "VALIDATION_ROLE",
    "ValidationPanel",
    "build_validation_panel",
    "capability_regressions",
    "significant_capability_regressions",
    "evaluate_validation_panel",
    "iter_slice_directories",
    "load_published_slice",
    "run_process_v2_training",
]
