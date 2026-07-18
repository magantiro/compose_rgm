"""Deterministic, prefetched training for the dense marked-rewrite generator."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass, replace
from math import cos, exp, pi
from time import perf_counter
from typing import Any, Callable, Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.experiments.tracelet_conditional import _sample_tracelet_progress
from compose_v4.model.factorized_tracelet_rate_model import (
    FactorizedMarkBatch,
    FactorizedTraceletRateModel,
    MARK_RULE_TO_INDEX,
    factorized_mark_bregman_loss,
    prepare_factorized_mark_batch,
)
from compose_v4.rewrite.ring_system_fiber import warm_ring_system_candidate_indices
from compose_v4.rewrite.typed_ring_catalog import TypedRingCatalog


@dataclass(frozen=True)
class FactorizedMarkExample:
    state: MolecularGraph
    time: float
    teacher_action: Any | None
    teacher_rule_name: str | None
    teacher_rate: float
    importance_weight: float
    ring_grow_support_mask: tuple[bool, ...] | None = None
    ring_grow_support_is_exact: bool = False
    ring_grow_enablement_is_exact: bool = False


class FactorizedMarkDataset(Dataset[FactorizedMarkExample]):
    """Index-deterministic examples safe under arbitrary worker prefetching.

    Example ``i`` is a pure function of ``(seed, i)``.  A resumed run starts at
    ``completed_steps * batch_size`` and therefore sees exactly the same future
    training stream without serializing worker RNG or prefetch queues.
    """

    def __init__(
        self,
        records: tuple[PathRecord, ...],
        *,
        start_index: int,
        length: int,
        seed: int,
        late_time_fraction: float,
        operational_horizon: float,
        progress_stratification_fraction: float,
        ring_catalog: TypedRingCatalog | None = None,
        ring_electronic_mode: str = "factorized_local",
    ) -> None:
        if not records:
            raise ValueError("factorized mark training records must be non-empty")
        if start_index < 0 or length < 0:
            raise ValueError("dataset indices and length must be non-negative")
        if not 0.0 <= late_time_fraction <= 1.0:
            raise ValueError("late-time fraction must lie in [0, 1]")
        if operational_horizon <= 0.0:
            raise ValueError("operational horizon must be positive")
        if not 0.0 <= progress_stratification_fraction <= 1.0:
            raise ValueError("progress stratification must lie in [0, 1]")
        self.records = records
        self.start_index = int(start_index)
        self.length = int(length)
        self.seed = int(seed)
        self.late_time_fraction = float(late_time_fraction)
        self.operational_horizon = float(operational_horizon)
        self.progress_stratification_fraction = float(progress_stratification_fraction)
        self.ring_catalog = ring_catalog
        self.ring_electronic_mode = str(ring_electronic_mode)
        self._ring_support_model: FactorizedTraceletRateModel | None = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> FactorizedMarkExample:
        if not 0 <= index < self.length:
            raise IndexError(index)
        absolute_index = self.start_index + int(index)
        rng = np.random.default_rng(np.random.SeedSequence((self.seed, absolute_index)))
        record = self.records[int(rng.integers(len(self.records)))]
        if rng.random() < self.late_time_fraction:
            operational_time = float(rng.uniform(0.0, self.operational_horizon))
            time = 1.0 - exp(-operational_time)
        else:
            time = float(rng.uniform(0.01, 0.99))
        progress, importance_weight = _sample_tracelet_progress(
            record.path,
            time=time,
            rng=rng,
            stratification_fraction=self.progress_stratification_fraction,
        )
        if progress < record.path.path_length:
            step = record.path.trace.steps[progress]
            if step.rule_name not in MARK_RULE_TO_INDEX:
                raise ValueError(
                    f"compiled teacher uses unsupported dense family: {step.rule_name}"
                )
            teacher_action = step.action
            teacher_rule_name = step.rule_name
            teacher_rate = record.path.operational_jump_rate(progress)
        else:
            teacher_action = None
            teacher_rule_name = None
            teacher_rate = 0.0
        state = record.path.state_at(progress)
        ring_grow_support_mask = None
        ring_grow_support_is_exact = False
        ring_grow_enablement_is_exact = False
        if self.ring_catalog is not None:
            if self._ring_support_model is None:
                # Exact semantic support is chemistry-only. A tiny parameter
                # shell lets persistent DataLoader workers reuse the same
                # placement/decoder caches without duplicating the compiler.
                self._ring_support_model = FactorizedTraceletRateModel(
                    self.ring_catalog,
                    hidden_dim=1,
                    message_passing_steps=1,
                    mark_dim=1,
                    ring_electronic_mode=self.ring_electronic_mode,
                )
            if teacher_rule_name == "ring_system_grow":
                ring_grow_support_mask = self._ring_support_model._ring_grow_support(state)
                ring_grow_support_is_exact = True
            else:
                ring_grow_support_mask = (
                    self._ring_support_model._ring_grow_enablement_certificate(state)
                )
            ring_grow_enablement_is_exact = True
        return FactorizedMarkExample(
            state=state,
            time=time,
            teacher_action=teacher_action,
            teacher_rule_name=teacher_rule_name,
            teacher_rate=teacher_rate,
            importance_weight=importance_weight,
            ring_grow_support_mask=ring_grow_support_mask,
            ring_grow_support_is_exact=ring_grow_support_is_exact,
            ring_grow_enablement_is_exact=ring_grow_enablement_is_exact,
        )


@dataclass(frozen=True)
class FactorizedMarkCollator:
    use_aromatic_bond_view: bool
    ring_catalog: TypedRingCatalog | None = None

    def __call__(self, examples: list[FactorizedMarkExample]) -> FactorizedMarkBatch:
        batch = prepare_factorized_mark_batch(
            tuple(example.state for example in examples),
            tuple(example.time for example in examples),
            tuple(example.teacher_action for example in examples),
            tuple(example.teacher_rule_name for example in examples),
            tuple(example.teacher_rate for example in examples),
            tuple(example.importance_weight for example in examples),
            use_aromatic_bond_view=self.use_aromatic_bond_view,
            ring_catalog=self.ring_catalog,
        )
        exact_masks = tuple(example.ring_grow_support_mask for example in examples)
        if all(mask is not None for mask in exact_masks):
            batch = replace(
                batch,
                ring_grow_support_mask=torch.tensor(
                    tuple(mask for mask in exact_masks if mask is not None),
                    dtype=torch.bool,
                ),
                ring_grow_support_is_exact=torch.tensor(
                    tuple(example.ring_grow_support_is_exact for example in examples),
                    dtype=torch.bool,
                ),
                ring_grow_enablement_is_exact=torch.tensor(
                    tuple(
                        example.ring_grow_enablement_is_exact for example in examples
                    ),
                    dtype=torch.bool,
                ),
            )
        return batch


def factorized_mark_loader(
    records: tuple[PathRecord, ...],
    *,
    steps: int,
    batch_size: int,
    start_step: int,
    seed: int,
    workers: int,
    late_time_fraction: float,
    operational_horizon: float,
    progress_stratification_fraction: float,
    use_aromatic_bond_view: bool,
    pin_memory: bool,
    ring_catalog: TypedRingCatalog | None = None,
    prefetch_factor: int = 2,
    ring_electronic_mode: str = "factorized_local",
) -> DataLoader[FactorizedMarkBatch]:
    if not 0 <= start_step <= steps:
        raise ValueError("start step lies outside the training horizon")
    if prefetch_factor <= 0:
        raise ValueError("prefetch factor must be positive")
    remaining_steps = steps - start_step
    if ring_catalog is not None:
        warm_ring_system_candidate_indices(ring_catalog)
    dataset = FactorizedMarkDataset(
        records,
        start_index=start_step * batch_size,
        length=remaining_steps * batch_size,
        seed=seed,
        late_time_fraction=late_time_fraction,
        operational_horizon=operational_horizon,
        progress_stratification_fraction=progress_stratification_fraction,
        ring_catalog=ring_catalog,
        ring_electronic_mode=ring_electronic_mode,
    )
    options: dict[str, Any] = {}
    if workers > 0:
        options.update(
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=FactorizedMarkCollator(use_aromatic_bond_view, ring_catalog),
        pin_memory=pin_memory,
        drop_last=True,
        **options,
    )


def sample_factorized_mark_batch(
    records: tuple[PathRecord, ...],
    *,
    batch_size: int,
    seed: int,
    late_time_fraction: float,
    operational_horizon: float,
    progress_stratification_fraction: float = 0.0,
    use_aromatic_bond_view: bool = True,
    workers: int = 0,
    ring_catalog: TypedRingCatalog | None = None,
    ring_electronic_mode: str = "factorized_local",
) -> FactorizedMarkBatch:
    if workers < 0:
        raise ValueError("evaluation workers must be non-negative")
    if ring_catalog is not None:
        warm_ring_system_candidate_indices(ring_catalog)
    dataset = FactorizedMarkDataset(
        records,
        start_index=0,
        length=batch_size,
        seed=seed,
        late_time_fraction=late_time_fraction,
        operational_horizon=operational_horizon,
        progress_stratification_fraction=progress_stratification_fraction,
        ring_catalog=ring_catalog,
        ring_electronic_mode=ring_electronic_mode,
    )
    collator = FactorizedMarkCollator(use_aromatic_bond_view, ring_catalog)
    if workers == 0:
        return collator([dataset[index] for index in range(batch_size)])
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, 32),
        shuffle=False,
        num_workers=workers,
        collate_fn=collator,
        pin_memory=False,
        drop_last=False,
    )
    return _concatenate_factorized_mark_batches(tuple(loader))


def _concatenate_factorized_mark_batches(
    batches: tuple[FactorizedMarkBatch, ...],
) -> FactorizedMarkBatch:
    if not batches:
        raise ValueError("cannot concatenate an empty batch sequence")

    def tensors(name: str) -> Tensor:
        return torch.cat([getattr(batch, name) for batch in batches], dim=0)

    support_masks = tuple(batch.ring_grow_support_mask for batch in batches)
    if any(mask is None for mask in support_masks):
        ring_grow_support_mask = None
    else:
        ring_grow_support_mask = torch.cat(
            [mask for mask in support_masks if mask is not None],
            dim=0,
        )
    delete_actions = tuple(batch.ring_delete_actions for batch in batches)
    ring_delete_actions = (
        None
        if any(actions is None for actions in delete_actions)
        else tuple(
            action_group
            for actions in delete_actions
            if actions is not None
            for action_group in actions
        )
    )

    return FactorizedMarkBatch(
        states=tuple(state for batch in batches for state in batch.states),
        atom_types=tensors("atom_types"),
        formal_charges=tensors("formal_charges"),
        implicit_h_counts=tensors("implicit_h_counts"),
        bonds=tensors("bonds"),
        neural_bonds=tensors("neural_bonds"),
        times=tensors("times"),
        atom_topology=tensors("atom_topology"),
        closure_topology=tensors("closure_topology"),
        ring_system_topology=tensors("ring_system_topology"),
        atom_delete_mask=tensors("atom_delete_mask"),
        cycle_edge_mask=tensors("cycle_edge_mask"),
        cyclic_pair_mask=tensors("cyclic_pair_mask"),
        graft_mask=tensors("graft_mask"),
        graft_remove_neighbors=tensors("graft_remove_neighbors"),
        graft_successor_groups=tensors("graft_successor_groups"),
        teacher_actions=tuple(action for batch in batches for action in batch.teacher_actions),
        teacher_rule_names=tuple(name for batch in batches for name in batch.teacher_rule_names),
        teacher_rates=tensors("teacher_rates"),
        importance_weights=tensors("importance_weights"),
        ring_restate_actions=tuple(
            actions for batch in batches for actions in batch.ring_restate_actions
        ),
        ring_grow_support_mask=ring_grow_support_mask,
        ring_delete_actions=ring_delete_actions,
        ring_grow_support_is_exact=torch.cat(
            [
                batch.ring_grow_support_is_exact
                if isinstance(batch.ring_grow_support_is_exact, Tensor)
                else torch.full(
                    (batch.batch_size,),
                    bool(batch.ring_grow_support_is_exact),
                    dtype=torch.bool,
                )
                for batch in batches
            ],
            dim=0,
        ),
        ring_grow_enablement_is_exact=torch.cat(
            [
                batch.ring_grow_enablement_is_exact
                if isinstance(batch.ring_grow_enablement_is_exact, Tensor)
                else torch.full(
                    (batch.batch_size,),
                    bool(batch.ring_grow_enablement_is_exact),
                    dtype=torch.bool,
                )
                for batch in batches
            ],
            dim=0,
        ),
    )


@torch.no_grad()
def factorized_mark_metrics(
    model: FactorizedTraceletRateModel,
    batch: FactorizedMarkBatch,
    *,
    use_bf16: bool,
) -> dict[str, float]:
    device = model.device
    batch = batch.to(device, non_blocking=device.type == "cuda")
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else nullcontext()
    )
    with context:
        prediction = model.forward_mark_batch(batch)
        loss = factorized_mark_bregman_loss(prediction, batch)
    teacher_family = torch.tensor(
        [MARK_RULE_TO_INDEX[name] if name is not None else -1 for name in batch.teacher_rule_names],
        dtype=torch.long,
        device=device,
    )
    nonterminal = teacher_family >= 0
    family_hits = (
        prediction.family_log_probabilities.argmax(dim=-1)[nonterminal]
        == teacher_family[nonterminal]
    )
    terminal = ~nonterminal
    return {
        "factorized_gm_loss": float(loss),
        "mean_teacher_mark_probability": (
            float(prediction.selected_mark_log_probability[nonterminal].exp().mean())
            if bool(nonterminal.any())
            else 0.0
        ),
        "family_accuracy": (float(family_hits.float().mean()) if family_hits.numel() else 0.0),
        "mean_terminal_hazard": (
            float(prediction.total_hazard[terminal].mean()) if bool(terminal.any()) else 0.0
        ),
    }


def train_factorized_mark_model(
    model: FactorizedTraceletRateModel,
    train_records: tuple[PathRecord, ...],
    validation_batch: FactorizedMarkBatch,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    workers: int,
    late_time_fraction: float,
    operational_horizon: float,
    progress_stratification_fraction: float,
    use_aromatic_bond_view: bool,
    use_bf16: bool,
    ring_catalog: TypedRingCatalog | None = None,
    data_prefetch_factor: int = 2,
    ring_electronic_mode: str = "factorized_local",
    evaluation_points: int = 10,
    evaluation_interval: int | None = None,
    warmup_steps: int = 0,
    minimum_learning_rate_fraction: float = 1.0,
    early_stopping_patience: int = 0,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    checkpoint_interval: int = 0,
    checkpoint_callback: Callable[[dict[str, object]], None] | None = None,
    resume_state: dict[str, object] | None = None,
    profile_timing: bool = False,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    if steps <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("steps, batch size, and learning rate must be positive")
    if workers < 0:
        raise ValueError("data workers must be non-negative")
    if data_prefetch_factor <= 0:
        raise ValueError("data prefetch factor must be positive")
    if evaluation_interval is not None and evaluation_interval <= 0:
        raise ValueError("evaluation interval must be positive")
    if not 0 <= warmup_steps <= steps:
        raise ValueError("warmup steps must lie in [0, steps]")
    if not 0.0 < minimum_learning_rate_fraction <= 1.0:
        raise ValueError("minimum learning-rate fraction must lie in (0, 1]")
    if early_stopping_patience < 0:
        raise ValueError("early-stopping patience must be non-negative")
    optimizer = torch.optim.AdamW(
        factorized_adamw_parameter_groups(model, weight_decay=weight_decay),
        lr=learning_rate,
    )
    if resume_state is None:
        start_step = 0
        model.eval()
        best_metrics = factorized_mark_metrics(
            model,
            validation_batch,
            use_bf16=use_bf16,
        )
        best_metrics["selected_step"] = 0.0
        best_state = _clone_model_state(model)
        history: list[dict[str, float]] = []
        evaluations_without_improvement = 0
    else:
        required = {
            "completed_steps",
            "current_state_dict",
            "optimizer_state_dict",
            "best_state_dict",
            "best_metrics",
            "history",
            "optimizer_kind",
            "evaluations_without_improvement",
        }
        missing = sorted(required - resume_state.keys())
        if missing:
            raise ValueError(f"factorized recovery state is missing: {missing}")
        if resume_state["optimizer_kind"] != "adamw_decoupled_v1":
            raise ValueError("factorized recovery checkpoint uses an incompatible optimizer")
        start_step = int(resume_state["completed_steps"])
        model.load_state_dict(resume_state["current_state_dict"])  # type: ignore[arg-type]
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])  # type: ignore[arg-type]
        torch_rng_state = resume_state.get("torch_rng_state")
        if isinstance(torch_rng_state, Tensor):
            torch.set_rng_state(torch_rng_state.detach().cpu().to(torch.uint8))
        cuda_rng_states = resume_state.get("cuda_rng_states")
        if cuda_rng_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [state.detach().cpu().to(torch.uint8) for state in cuda_rng_states]  # type: ignore[union-attr]
            )
        best_state = {
            name: value.detach().clone()
            for name, value in resume_state["best_state_dict"].items()  # type: ignore[union-attr]
        }
        best_metrics = {
            str(key): float(value)
            for key, value in resume_state["best_metrics"].items()  # type: ignore[union-attr]
        }
        history = [
            {str(key): float(value) for key, value in row.items()}
            for row in resume_state["history"]  # type: ignore[union-attr]
        ]
        evaluations_without_improvement = int(resume_state["evaluations_without_improvement"])

    loader = factorized_mark_loader(
        train_records,
        steps=steps,
        batch_size=batch_size,
        start_step=start_step,
        seed=seed,
        workers=workers,
        late_time_fraction=late_time_fraction,
        operational_horizon=operational_horizon,
        progress_stratification_fraction=progress_stratification_fraction,
        use_aromatic_bond_view=use_aromatic_bond_view,
        pin_memory=model.device.type == "cuda",
        ring_catalog=ring_catalog,
        prefetch_factor=data_prefetch_factor,
        ring_electronic_mode=ring_electronic_mode,
    )
    timing_loop_started = perf_counter()
    iterator: Iterator[FactorizedMarkBatch] = iter(loader)
    cumulative_data_wait = 0.0
    cumulative_update_time = 0.0
    maximum_data_wait = 0.0
    data_wait_history: list[float] = []
    resolved_evaluation_interval = (
        evaluation_interval
        if evaluation_interval is not None
        else max(steps // evaluation_points, 1)
    )
    model.train()
    for step in range(start_step, steps):
        completed_steps = step + 1
        current_learning_rate = cosine_warmup_learning_rate(
            base_learning_rate=learning_rate,
            completed_step=completed_steps,
            total_steps=steps,
            warmup_steps=warmup_steps,
            minimum_fraction=minimum_learning_rate_fraction,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate
        started = perf_counter()
        cpu_batch = next(iterator)
        loaded_at = perf_counter()
        batch = cpu_batch.to(model.device, non_blocking=model.device.type == "cuda")
        transferred_at = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16 and model.device.type == "cuda"
            else nullcontext()
        )
        with context:
            prediction = model.forward_mark_batch(batch)
            loss = factorized_mark_bregman_loss(prediction, batch)
        if profile_timing:
            _synchronize(model.device)
        forwarded_at = perf_counter()
        loss.backward()
        if profile_timing:
            _synchronize(model.device)
        backward_at = perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if profile_timing:
            _synchronize(model.device)
        optimized_at = perf_counter()
        data_wait = loaded_at - started
        update_time = optimized_at - started
        cumulative_data_wait += data_wait
        cumulative_update_time += update_time
        maximum_data_wait = max(maximum_data_wait, data_wait)
        data_wait_history.append(data_wait)
        evaluated = (
            step == start_step
            or completed_steps % resolved_evaluation_interval == 0
            or completed_steps == steps
        )
        should_early_stop = False
        if evaluated:
            model.eval()
            metrics = factorized_mark_metrics(
                model,
                validation_batch,
                use_bf16=use_bf16,
            )
            metrics["step"] = float(completed_steps)
            metrics["train_batch_loss"] = float(loss.detach())
            metrics["learning_rate"] = float(current_learning_rate)
            improved = metrics["factorized_gm_loss"] < best_metrics["factorized_gm_loss"]
            if improved:
                best_metrics = {
                    key: value
                    for key, value in metrics.items()
                    if key not in {"step", "train_batch_loss"}
                }
                best_metrics["selected_step"] = float(completed_steps)
                best_state = _clone_model_state(model)
                evaluations_without_improvement = 0
            elif completed_steps >= max(warmup_steps, 1):
                evaluations_without_improvement += 1
            metrics["evaluations_without_improvement"] = float(evaluations_without_improvement)
            should_early_stop = (
                early_stopping_patience > 0
                and evaluations_without_improvement >= early_stopping_patience
            )
            metrics["early_stopped"] = float(should_early_stop)
            history.append(metrics)
            if progress_callback is not None:
                observed_updates = completed_steps - start_step
                timing_metrics = {
                    "timing/profile_synchronized": float(profile_timing),
                    "timing/data_wait_seconds": data_wait,
                    "timing/cumulative_data_wait_seconds": cumulative_data_wait,
                    "timing/mean_data_wait_seconds": (
                        cumulative_data_wait / observed_updates
                    ),
                    "timing/p95_data_wait_seconds": float(
                        np.percentile(data_wait_history, 95)
                    ),
                    "timing/max_data_wait_seconds": maximum_data_wait,
                }
                if profile_timing:
                    elapsed = max(optimized_at - timing_loop_started, 1e-12)
                    timing_metrics.update(
                        {
                            "timing/transfer_seconds": transferred_at - loaded_at,
                            "timing/forward_seconds": forwarded_at - transferred_at,
                            "timing/backward_seconds": backward_at - forwarded_at,
                            "timing/optimizer_seconds": optimized_at - backward_at,
                            "timing/update_seconds": update_time,
                            "timing/data_wait_fraction": (
                                cumulative_data_wait
                                / max(cumulative_update_time, 1e-12)
                            ),
                            "timing/updates_per_second": observed_updates / elapsed,
                            "timing/examples_per_second": (
                                observed_updates * batch_size / elapsed
                            ),
                        }
                    )
                progress_callback(
                    {
                        **metrics,
                        **timing_metrics,
                    }
                )
            model.train()
        should_checkpoint = checkpoint_callback is not None and (
            completed_steps == steps
            or should_early_stop
            or (checkpoint_interval > 0 and completed_steps % checkpoint_interval == 0)
        )
        if should_checkpoint:
            checkpoint_callback(
                _factorized_recovery_state(
                    model=model,
                    optimizer=optimizer,
                    completed_steps=completed_steps,
                    best_state=best_state,
                    best_metrics=best_metrics,
                    history=history,
                    evaluations_without_improvement=(evaluations_without_improvement),
                )
            )
        if should_early_stop:
            break
    model.load_state_dict(best_state)
    model.eval()
    return history, best_metrics


def _clone_model_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def factorized_adamw_parameter_groups(
    model: nn.Module,
    *,
    weight_decay: float,
) -> list[dict[str, object]]:
    """Partition dense weights from lookup and scale parameters for AdamW.

    Mark dictionaries are large, sparsely activated embedding tables.  Coupled
    Adam L2 regularization normalizes their otherwise tiny decay gradients and
    can collapse an inactive entry by roughly one learning-rate unit per step.
    Decoupled decay avoids that pathology, and embeddings, biases, normalization
    scales, and other vector parameters receive no decay at all.
    """

    if weight_decay < 0.0:
        raise ValueError("weight decay must be non-negative")
    no_decay_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.LayerNorm)):
            no_decay_ids.update(id(parameter) for parameter in module.parameters(recurse=False))

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        excluded = id(parameter) in no_decay_ids or parameter.ndim < 2 or name.endswith(".bias")
        (no_decay if excluded else decay).append(parameter)

    assigned = {id(parameter) for parameter in (*decay, *no_decay)}
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if assigned != expected or len(decay) + len(no_decay) != len(expected):
        raise RuntimeError("optimizer parameter partition is incomplete or duplicated")
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def cosine_warmup_learning_rate(
    *,
    base_learning_rate: float,
    completed_step: int,
    total_steps: int,
    warmup_steps: int,
    minimum_fraction: float,
) -> float:
    """Deterministic linear-warmup/cosine-decay schedule, safe on resume."""

    if base_learning_rate <= 0.0 or total_steps <= 0:
        raise ValueError("learning rate and total steps must be positive")
    if not 1 <= completed_step <= total_steps:
        raise ValueError("completed step lies outside the training horizon")
    if not 0 <= warmup_steps <= total_steps:
        raise ValueError("warmup steps must lie in [0, total_steps]")
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("minimum fraction must lie in (0, 1]")
    if warmup_steps > 0 and completed_step <= warmup_steps:
        return base_learning_rate * completed_step / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(
        max((completed_step - warmup_steps) / decay_steps, 0.0),
        1.0,
    )
    cosine_fraction = 0.5 * (1.0 + cos(pi * progress))
    scale = minimum_fraction + (1.0 - minimum_fraction) * cosine_fraction
    return base_learning_rate * scale


def _factorized_recovery_state(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    best_state: dict[str, Tensor],
    best_metrics: dict[str, float],
    history: list[dict[str, float]],
    evaluations_without_improvement: int,
) -> dict[str, object]:
    return {
        "optimizer_kind": "adamw_decoupled_v1",
        "completed_steps": int(completed_steps),
        "current_state_dict": _clone_model_state(model),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "torch_rng_state": torch.get_rng_state().detach().cpu(),
        "cuda_rng_states": (
            [state.detach().cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
        "best_state_dict": {name: value.detach().clone() for name, value in best_state.items()},
        "best_metrics": copy.deepcopy(best_metrics),
        "history": copy.deepcopy(history),
        "evaluations_without_improvement": int(evaluations_without_improvement),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


__all__ = [
    "FactorizedMarkCollator",
    "FactorizedMarkDataset",
    "FactorizedMarkExample",
    "factorized_mark_loader",
    "factorized_mark_metrics",
    "factorized_adamw_parameter_groups",
    "cosine_warmup_learning_rate",
    "sample_factorized_mark_batch",
    "train_factorized_mark_model",
]
