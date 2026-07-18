"""Conditional Generator Matching and ancestral sampling with ring tracelets."""

from __future__ import annotations

import copy
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from math import exp
from multiprocessing import get_context
from time import perf_counter
from collections.abc import Callable
from typing import Iterator, Protocol

import numpy as np
import torch
from torch import Tensor, nn

from compose_v4.chem.molecular_graph import MolecularGraph, smiles_to_molecular_graph
from compose_v4.chem.state import empty_molecular_graph, pad_molecular_graph
from compose_v4.chem.source_prior import (
    DegreeBoundedCarbonTreePrior,
    MolecularSourcePrior,
    NullSourcePrior,
)
from compose_v4.experiments.cnof_conditional import PathRecord
from compose_v4.gm.loss import multi_successor_rate_bregman_loss, rate_bregman_loss
from compose_v4.model.rate_model import FiberRatePrediction
from compose_v4.rewrite.kernel import canonical_state_key, de_novo_rewrite_system
from compose_v4.rewrite.causal_trace import CausalTraceCTMC, CausalTraceSample
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.tracelet_compiler import compile_null_to_target_tracelets
from compose_v4.rewrite.tree_transport import compile_carbon_tree_to_target
from compose_v4.rewrite.tracelet_fiber import (
    TraceletFiber,
    enumerate_tracelet_cnof_fiber,
)
from compose_v4.rewrite.typed_ring_catalog import TypedRingCatalog


class TraceletRatePredictor(Protocol):
    def predict_tracelet_fiber(
        self,
        state: MolecularGraph,
        time: float,
        *,
        fiber: TraceletFiber | None = None,
    ) -> FiberRatePrediction: ...


@dataclass(frozen=True)
class TraceletConditionalRateExample:
    state: MolecularGraph
    time: float
    teacher_successor_key: str | None
    teacher_rate: float
    fiber: TraceletFiber
    teacher_rule_name: str | None = None
    importance_weight: float = 1.0
    teacher_successor_rates: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class TraceletRollout:
    final_state: MolecularGraph
    event_times: tuple[float, ...]
    event_rules: tuple[str, ...]
    exhausted_event_budget: bool


@dataclass(frozen=True)
class _PendingTraceletExample:
    path: TraceProgressCTMC
    progress: int
    time: float
    importance_weight: float
    state_key: str
    n_slots: int
    causal_sample: CausalTraceSample | None


def canonical_tracelet_representative(
    state: MolecularGraph,
    state_key: str | None = None,
) -> MolecularGraph:
    """Return one deterministic padded representative of a chemical state.

    Fiber and rate caches are keyed by canonical SMILES.  Reconstructing the
    representative from that same key makes transition ordering independent of
    whichever slot permutation happened to populate a cache first.
    """

    key = canonical_state_key(state) if state_key is None else state_key
    if key == "<NULL>":
        return empty_molecular_graph(state.n_atoms)
    return pad_molecular_graph(smiles_to_molecular_graph(key), state.n_atoms)


_FIBER_WORKER_RING_CATALOG: TypedRingCatalog | None = None
_PATH_WORKER_N_SLOTS = 0
_PATH_WORKER_TYPED_RING_PAYLOADS = False
_PATH_WORKER_RING_CATALOG: TypedRingCatalog | None = None
_PATH_WORKER_CHECKPOINT_INTERVAL: int | None = None


def _initialize_fiber_worker(ring_catalog: TypedRingCatalog | None) -> None:
    global _FIBER_WORKER_RING_CATALOG
    _FIBER_WORKER_RING_CATALOG = ring_catalog
    torch.set_num_threads(1)


def _enumerate_fiber_worker(
    task: tuple[str, int],
) -> tuple[str, MolecularGraph, TraceletFiber]:
    state_key, n_slots = task
    if state_key == "<NULL>":
        representative = empty_molecular_graph(n_slots)
    else:
        representative = pad_molecular_graph(
            smiles_to_molecular_graph(state_key),
            n_slots,
        )
    fiber = enumerate_tracelet_cnof_fiber(
        representative,
        ring_catalog=_FIBER_WORKER_RING_CATALOG,
    )
    return state_key, representative, fiber


def _initialize_path_worker(
    n_slots: int,
    typed_ring_payloads: bool,
    ring_catalog: TypedRingCatalog | None,
    checkpoint_interval: int | None,
) -> None:
    global _PATH_WORKER_N_SLOTS
    global _PATH_WORKER_TYPED_RING_PAYLOADS
    global _PATH_WORKER_RING_CATALOG
    global _PATH_WORKER_CHECKPOINT_INTERVAL
    _PATH_WORKER_N_SLOTS = int(n_slots)
    _PATH_WORKER_TYPED_RING_PAYLOADS = bool(typed_ring_payloads)
    _PATH_WORKER_RING_CATALOG = ring_catalog
    _PATH_WORKER_CHECKPOINT_INTERVAL = checkpoint_interval
    torch.set_num_threads(1)


def _build_tracelet_path_record_worker(text: str) -> PathRecord:
    target = pad_molecular_graph(
        smiles_to_molecular_graph(text),
        _PATH_WORKER_N_SLOTS,
    )
    trace = compile_null_to_target_tracelets(
        target,
        typed_ring_payloads=_PATH_WORKER_TYPED_RING_PAYLOADS,
        ring_catalog=_PATH_WORKER_RING_CATALOG,
    )
    return PathRecord(
        canonical_state_key(target),
        TraceProgressCTMC(
            trace,
            checkpoint_interval=_PATH_WORKER_CHECKPOINT_INTERVAL,
        ),
    )


def _build_tree_transport_path_record_worker(
    task: tuple[str, MolecularGraph, str],
) -> PathRecord:
    text, source, transport_mode = task
    target = pad_molecular_graph(
        smiles_to_molecular_graph(text),
        _PATH_WORKER_N_SLOTS,
    )
    trace = compile_carbon_tree_to_target(
        source,
        target,
        use_bond_reroute=transport_mode != "primitive",
        align_source=transport_mode == "size_matched_graft",
        flexible_size=transport_mode == "flexible_size_graft",
        typed_ring_payloads=_PATH_WORKER_TYPED_RING_PAYLOADS,
        ring_catalog=_PATH_WORKER_RING_CATALOG,
    )
    return PathRecord(
        canonical_state_key(target),
        TraceProgressCTMC(
            trace,
            checkpoint_interval=_PATH_WORKER_CHECKPOINT_INTERVAL,
        ),
    )


@contextmanager
def tracelet_fiber_executor(
    workers: int,
    *,
    ring_catalog: TypedRingCatalog | None,
) -> Iterator[Executor | None]:
    """Create persistent CPU workers for exact legal-fiber construction."""

    if workers < 0:
        raise ValueError("fiber workers must be non-negative")
    if workers <= 1:
        yield None
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_initialize_fiber_worker,
        initargs=(ring_catalog,),
    ) as executor:
        yield executor


@contextmanager
def tracelet_path_executor(
    workers: int,
    *,
    n_slots: int,
    typed_ring_payloads: bool,
    ring_catalog: TypedRingCatalog | None,
    checkpoint_interval: int | None,
) -> Iterator[Executor | None]:
    """Create one persistent worker pool across resumable path shards."""

    if workers < 0:
        raise ValueError("path workers must be non-negative")
    if workers <= 1:
        _initialize_path_worker(
            n_slots,
            typed_ring_payloads,
            ring_catalog,
            checkpoint_interval,
        )
        yield None
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_initialize_path_worker,
        initargs=(
            n_slots,
            typed_ring_payloads,
            ring_catalog,
            checkpoint_interval,
        ),
    ) as executor:
        yield executor


def build_tracelet_path_records(
    smiles: tuple[str, ...],
    *,
    n_slots: int,
    typed_ring_payloads: bool = False,
    ring_catalog: TypedRingCatalog | None = None,
    workers: int = 0,
    checkpoint_interval: int | None = None,
    executor: Executor | None = None,
) -> tuple[PathRecord, ...]:
    if workers < 0:
        raise ValueError("path workers must be non-negative")
    if executor is not None:
        chunksize = max(min(len(smiles) // 128, 64), 1)
        return tuple(
            executor.map(
                _build_tracelet_path_record_worker,
                smiles,
                chunksize=chunksize,
            )
        )
    if workers > 1:
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=get_context("spawn"),
                initializer=_initialize_path_worker,
                initargs=(
                    n_slots,
                    typed_ring_payloads,
                    ring_catalog,
                    checkpoint_interval,
                ),
            ) as executor:
                chunksize = max(min(len(smiles) // max(workers * 8, 1), 64), 1)
                return tuple(
                    executor.map(
                        _build_tracelet_path_record_worker,
                        smiles,
                        chunksize=chunksize,
                    )
                )
        except (OSError, PermissionError):
            # Restricted local sandboxes may deny POSIX semaphore creation.
            # The deterministic serial path is semantically identical.
            _initialize_path_worker(
                n_slots,
                typed_ring_payloads,
                ring_catalog,
                checkpoint_interval,
            )
            return tuple(_build_tracelet_path_record_worker(text) for text in smiles)
    records = []
    for text in smiles:
        target = pad_molecular_graph(smiles_to_molecular_graph(text), n_slots)
        trace = compile_null_to_target_tracelets(
            target,
            typed_ring_payloads=typed_ring_payloads,
            ring_catalog=ring_catalog,
        )
        records.append(
            PathRecord(
                canonical_state_key(target),
                TraceProgressCTMC(trace, checkpoint_interval=checkpoint_interval),
            )
        )
    return tuple(records)


def build_tree_transport_path_records(
    smiles: tuple[str, ...],
    *,
    n_slots: int,
    source_prior: MolecularSourcePrior,
    seed: int,
    couplings_per_target: int = 1,
    transport_mode: str = "primitive",
    typed_ring_payloads: bool = False,
    ring_catalog: TypedRingCatalog | None = None,
    workers: int = 0,
    checkpoint_interval: int | None = None,
    executor: Executor | None = None,
) -> tuple[PathRecord, ...]:
    """Compile structured-noise trees into data endpoints.

    A single frozen source tree per molecule badly under-samples the declared
    source coupling, especially in small screens.  Repeating each endpoint
    with fresh draws exposes training to the same source law used by ancestral
    sampling. ``size_matched_graft`` couples source and target sizes, preserving
    the empirical source-size marginal while greatly shortening topology
    transport; source labels are aligned only by a permutation.
    ``flexible_size_graft`` draws source size independently and supervises
    genuine grow/shrink events before the atom-retaining Graft compiler.
    """

    if couplings_per_target <= 0:
        raise ValueError("couplings_per_target must be positive")
    if workers < 0:
        raise ValueError("path workers must be non-negative")
    if transport_mode not in {
        "primitive",
        "size_matched_graft",
        "flexible_size_graft",
    }:
        raise ValueError(f"unknown tree transport mode: {transport_mode}")
    if (
        transport_mode in {"size_matched_graft", "flexible_size_graft"}
        and not isinstance(source_prior, DegreeBoundedCarbonTreePrior)
    ):
        raise TypeError("size-matched Graft transport requires a carbon-tree prior")

    rng = np.random.default_rng(seed)

    def source_for(target_size: int | None) -> MolecularGraph:
        if transport_mode in {"primitive", "flexible_size_graft"}:
            return source_prior.sample(rng, n_slots=n_slots)
        if target_size is None:
            raise RuntimeError("size-matched transport requires a target size")
        return source_prior.sample_size(
            rng,
            n_slots=n_slots,
            size=target_size,
        )

    use_size_matching = transport_mode == "size_matched_graft"
    target_sizes = (
        tuple(smiles_to_molecular_graph(text).n_real_atoms for text in smiles)
        if use_size_matching
        else (None,) * len(smiles)
    )
    if executor is not None:
        tasks = tuple(
            (text, source_for(target_size), transport_mode)
            for text, target_size in zip(smiles, target_sizes)
            for _ in range(couplings_per_target)
        )
        chunksize = max(min(len(tasks) // 128, 64), 1)
        return tuple(
            executor.map(
                _build_tree_transport_path_record_worker,
                tasks,
                chunksize=chunksize,
            )
        )
    if workers > 1:
        tasks = tuple(
            (text, source_for(target_size), transport_mode)
            for text, target_size in zip(smiles, target_sizes)
            for _ in range(couplings_per_target)
        )
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=get_context("spawn"),
                initializer=_initialize_path_worker,
                initargs=(
                    n_slots,
                    typed_ring_payloads,
                    ring_catalog,
                    checkpoint_interval,
                ),
            ) as executor:
                chunksize = max(min(len(tasks) // max(workers * 8, 1), 64), 1)
                return tuple(
                    executor.map(
                        _build_tree_transport_path_record_worker,
                        tasks,
                        chunksize=chunksize,
                    )
                )
        except (OSError, PermissionError):
            _initialize_path_worker(
                n_slots,
                typed_ring_payloads,
                ring_catalog,
                checkpoint_interval,
            )
            return tuple(
                _build_tree_transport_path_record_worker(task) for task in tasks
            )
    records = []
    for text, target_size in zip(smiles, target_sizes):
        target = pad_molecular_graph(smiles_to_molecular_graph(text), n_slots)
        target_key = canonical_state_key(target)
        for _ in range(couplings_per_target):
            source = source_for(target_size)
            trace = compile_carbon_tree_to_target(
                source,
                target,
                use_bond_reroute=transport_mode != "primitive",
                align_source=transport_mode == "size_matched_graft",
                flexible_size=transport_mode == "flexible_size_graft",
                typed_ring_payloads=typed_ring_payloads,
                ring_catalog=ring_catalog,
            )
            records.append(
                PathRecord(
                    target_key,
                    TraceProgressCTMC(
                        trace,
                        checkpoint_interval=checkpoint_interval,
                    ),
                )
            )
    return tuple(records)


def sample_tracelet_conditional_batch(
    records: tuple[PathRecord, ...],
    *,
    batch_size: int,
    rng: np.random.Generator,
    fiber_cache: dict[str, tuple[MolecularGraph, TraceletFiber]],
    late_time_fraction: float = 0.0,
    operational_horizon: float = 7.0,
    max_fiber_cache: int = 512,
    progress_stratification_fraction: float = 0.0,
    ring_catalog: TypedRingCatalog | None = None,
    causal_teachers: bool = False,
    causal_cache: dict[str, CausalTraceCTMC] | None = None,
    fiber_executor: Executor | None = None,
) -> tuple[TraceletConditionalRateExample, ...]:
    if not records or batch_size <= 0:
        raise ValueError("records must be non-empty and batch_size positive")
    if not 0.0 <= late_time_fraction <= 1.0:
        raise ValueError("late_time_fraction must lie in [0, 1]")
    if operational_horizon <= 0.0:
        raise ValueError("operational_horizon must be positive")
    if not 0.0 <= progress_stratification_fraction <= 1.0:
        raise ValueError("progress_stratification_fraction must lie in [0, 1]")
    if causal_teachers and progress_stratification_fraction != 0.0:
        raise ValueError("causal teachers do not yet support family stratification")

    pending_examples = []
    cached_causal = causal_cache if causal_cache is not None else {}
    for record_index in rng.integers(0, len(records), size=batch_size):
        record = records[int(record_index)]
        path = record.path
        if rng.random() < late_time_fraction:
            operational_time = float(rng.uniform(0.0, operational_horizon))
            time = 1.0 - exp(-operational_time)
        else:
            time = float(rng.uniform(0.01, 0.99))
        progress, importance_weight = _sample_tracelet_progress(
            path,
            time=time,
            rng=rng,
            stratification_fraction=progress_stratification_fraction,
        )
        causal_sample = None
        if causal_teachers:
            causal = cached_causal.get(record.target_key)
            if causal is None:
                causal = CausalTraceCTMC(path.trace, system=path.system)
                cached_causal[record.target_key] = causal
            causal_sample = causal.sample_at_progress(
                progress,
                time=time,
                rng=rng,
                operational=True,
            )
            raw_state = causal_sample.state
        else:
            raw_state = path.state_at(progress)
        state_key = canonical_state_key(raw_state)
        pending_examples.append(
            _PendingTraceletExample(
                path=path,
                progress=progress,
                time=time,
                importance_weight=importance_weight,
                state_key=state_key,
                n_slots=raw_state.n_atoms,
                causal_sample=causal_sample,
            )
        )

    resolved_fibers = {
        item.state_key: fiber_cache[item.state_key]
        for item in pending_examples
        if item.state_key in fiber_cache
    }
    missing_tasks = tuple(
        dict.fromkeys(
            (item.state_key, item.n_slots)
            for item in pending_examples
            if item.state_key not in resolved_fibers
        )
    )
    if fiber_executor is None:
        built = []
        for state_key, n_slots in missing_tasks:
            if state_key == "<NULL>":
                representative = empty_molecular_graph(n_slots)
            else:
                representative = pad_molecular_graph(
                    smiles_to_molecular_graph(state_key),
                    n_slots,
                )
            built.append(
                (
                    state_key,
                    representative,
                    enumerate_tracelet_cnof_fiber(
                        representative,
                        ring_catalog=ring_catalog,
                    ),
                )
            )
    else:
        built = list(
            fiber_executor.map(
                _enumerate_fiber_worker,
                missing_tasks,
                chunksize=1,
            )
        )
    for state_key, representative, fiber in built:
        cached = (representative, fiber)
        resolved_fibers[state_key] = cached
        if len(fiber_cache) < max_fiber_cache:
            fiber_cache[state_key] = cached

    examples = []
    for pending in pending_examples:
        path = pending.path
        progress = pending.progress
        causal_sample = pending.causal_sample
        state, fiber = resolved_fibers[pending.state_key]
        if progress < path.path_length:
            if causal_sample is not None:
                rate_by_key: dict[str, float] = {}
                for successor, rate in zip(
                    causal_sample.frontier_successors,
                    causal_sample.teacher_rates,
                ):
                    key = canonical_state_key(successor)
                    rate_by_key[key] = rate_by_key.get(key, 0.0) + float(rate)
                teacher_successor_rates = tuple(sorted(rate_by_key.items()))
                teacher_key = teacher_successor_rates[0][0]
                teacher_rate = float(sum(rate_by_key.values()))
                teacher_rule_name = (
                    causal_sample.frontier_steps[0].rule_name
                    if len(causal_sample.frontier_steps) == 1
                    else "<CAUSAL_FRONTIER>"
                )
            else:
                teacher_key = canonical_state_key(path.state_at(progress + 1))
                teacher_rate = path.operational_jump_rate(progress)
                teacher_rule_name = path.trace.steps[progress].rule_name
                teacher_successor_rates = ((teacher_key, teacher_rate),)
            inference_keys = {item.successor_key for item in fiber.transitions}
            if any(key not in inference_keys for key, _ in teacher_successor_rates):
                raise RuntimeError("tracelet teacher successor is outside inference fiber")
        else:
            teacher_key = None
            teacher_rate = 0.0
            teacher_rule_name = None
            teacher_successor_rates = ()
        examples.append(
            TraceletConditionalRateExample(
                state=state,
                time=pending.time,
                teacher_successor_key=teacher_key,
                teacher_rate=teacher_rate,
                fiber=fiber,
                teacher_rule_name=teacher_rule_name,
                importance_weight=pending.importance_weight,
                teacher_successor_rates=teacher_successor_rates,
            )
        )
    return tuple(examples)


def tracelet_conditional_batch_loss(
    model: TraceletRatePredictor,
    examples: tuple[TraceletConditionalRateExample, ...],
    *,
    action_kl_weight: float = 0.0,
    hazard_tilt_weight: float = 0.0,
) -> Tensor:
    cache_factory = getattr(model, "shared_tracelet_encoding_cache", None)
    context = cache_factory() if callable(cache_factory) else nullcontext()
    with context:
        return _tracelet_conditional_batch_loss_impl(
            model,
            examples,
            action_kl_weight=action_kl_weight,
            hazard_tilt_weight=hazard_tilt_weight,
        )


def _tracelet_conditional_batch_loss_impl(
    model: TraceletRatePredictor,
    examples: tuple[TraceletConditionalRateExample, ...],
    *,
    action_kl_weight: float,
    hazard_tilt_weight: float,
) -> Tensor:
    if action_kl_weight < 0.0 or hazard_tilt_weight < 0.0:
        raise ValueError("regularization weights must be non-negative")
    losses = []
    action_kls = []
    hazard_tilts = []
    for example in examples:
        prediction = model.predict_tracelet_fiber(
            example.state,
            example.time,
            fiber=example.fiber,
        )
        teacher_pairs = _teacher_pairs(example)
        if teacher_pairs:
            predicted_rates = torch.stack(
                [prediction.successor_rate(key) for key, _ in teacher_pairs]
            )
            teacher_rates = prediction.marked_rates.new_tensor(
                [rate for _, rate in teacher_pairs]
            )
            example_loss = multi_successor_rate_bregman_loss(
                prediction.total_hazard,
                predicted_rates,
                teacher_rates,
            )
        else:
            zero = prediction.marked_rates.new_zeros(())
            example_loss = rate_bregman_loss(
                prediction.total_hazard,
                zero,
                zero,
            )
        losses.append(example_loss * example.importance_weight)
        if prediction.action_kl_to_prior is not None:
            action_kls.append(
                prediction.action_kl_to_prior * example.importance_weight
            )
        if prediction.squared_log_hazard_tilt is not None:
            hazard_tilts.append(
                prediction.squared_log_hazard_tilt * example.importance_weight
            )
    loss = torch.stack(losses).mean()
    if action_kls:
        loss = loss + action_kl_weight * torch.stack(action_kls).mean()
    if hazard_tilts:
        loss = loss + hazard_tilt_weight * torch.stack(hazard_tilts).mean()
    return loss


@torch.no_grad()
def tracelet_conditional_metrics(
    model: TraceletRatePredictor,
    examples: tuple[TraceletConditionalRateExample, ...],
) -> dict[str, float]:
    cache_factory = getattr(model, "shared_tracelet_encoding_cache", None)
    context = cache_factory() if callable(cache_factory) else nullcontext()
    with context:
        return _tracelet_conditional_metrics_impl(model, examples)


def _tracelet_conditional_metrics_impl(
    model: TraceletRatePredictor,
    examples: tuple[TraceletConditionalRateExample, ...],
) -> dict[str, float]:
    losses = []
    support_mass = []
    top_hits = []
    terminal_hazards = []
    action_kls = []
    hazard_tilts = []
    family_rows: dict[str, dict[str, list[float]]] = {}
    for example in examples:
        prediction = model.predict_tracelet_fiber(
            example.state,
            example.time,
            fiber=example.fiber,
        )
        teacher_pairs = _teacher_pairs(example)
        if not teacher_pairs:
            predicted_rates = prediction.marked_rates.new_empty((0,))
            teacher_rates = prediction.marked_rates.new_empty((0,))
            terminal_hazards.append(float(prediction.total_hazard))
        else:
            teacher_keys = {key for key, _ in teacher_pairs}
            predicted_rates = torch.stack(
                [prediction.successor_rate(key) for key, _ in teacher_pairs]
            )
            teacher_rates = prediction.marked_rates.new_tensor(
                [rate for _, rate in teacher_pairs]
            )
            selected = predicted_rates.sum()
            support_mass.append(
                float(selected) / max(float(prediction.total_hazard), 1e-12)
            )
            rates = prediction.successor_rate_dict()
            top_key = max(rates, key=lambda key: float(rates[key]))
            top_hits.append(float(top_key in teacher_keys))
        if teacher_pairs:
            example_loss = multi_successor_rate_bregman_loss(
                prediction.total_hazard,
                predicted_rates,
                teacher_rates,
            )
        else:
            zero = prediction.marked_rates.new_zeros(())
            example_loss = rate_bregman_loss(
                prediction.total_hazard,
                zero,
                zero,
            )
        losses.append(float(example_loss))
        family = example.teacher_rule_name or "<TERMINAL>"
        row = family_rows.setdefault(
            family,
            {"loss": [], "support_mass": [], "top_hit": []},
        )
        row["loss"].append(float(example_loss))
        if teacher_pairs:
            row["support_mass"].append(float(support_mass[-1]))
            row["top_hit"].append(float(top_hits[-1]))
        if prediction.action_kl_to_prior is not None:
            action_kls.append(float(prediction.action_kl_to_prior))
        if prediction.squared_log_hazard_tilt is not None:
            hazard_tilts.append(float(prediction.squared_log_hazard_tilt))
    metrics = {
        "conditional_gm_loss": float(np.mean(losses)),
        "mean_teacher_successor_mass": float(np.mean(support_mass)) if support_mass else 0.0,
        "conditional_top_successor_accuracy": float(np.mean(top_hits)) if top_hits else 0.0,
        "mean_terminal_hazard": (
            float(np.mean(terminal_hazards)) if terminal_hazards else 0.0
        ),
    }
    if action_kls:
        metrics["mean_action_kl_to_prior"] = float(np.mean(action_kls))
    if hazard_tilts:
        metrics["mean_squared_log_hazard_tilt"] = float(np.mean(hazard_tilts))
    for family, row in sorted(family_rows.items()):
        prefix = f"family/{family}"
        metrics[f"{prefix}/examples"] = float(len(row["loss"]))
        metrics[f"{prefix}/conditional_gm_loss"] = float(np.mean(row["loss"]))
        if row["support_mass"]:
            metrics[f"{prefix}/mean_teacher_successor_mass"] = float(
                np.mean(row["support_mass"])
            )
            metrics[f"{prefix}/top_successor_accuracy"] = float(
                np.mean(row["top_hit"])
            )
    return metrics


def train_tracelet_conditional_model(
    model: nn.Module,
    train_records: tuple[PathRecord, ...],
    validation_examples: tuple[TraceletConditionalRateExample, ...],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    fiber_cache: dict[str, tuple[MolecularGraph, TraceletFiber]],
    weight_decay: float = 0.0,
    action_kl_weight: float = 0.0,
    hazard_tilt_weight: float = 0.0,
    late_time_fraction: float = 0.5,
    operational_horizon: float = 7.0,
    progress_stratification_fraction: float = 0.0,
    ring_catalog: TypedRingCatalog | None = None,
    causal_teachers: bool = False,
    causal_cache: dict[str, CausalTraceCTMC] | None = None,
    evaluation_points: int = 10,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    checkpoint_interval: int = 0,
    checkpoint_callback: Callable[[dict[str, object]], None] | None = None,
    resume_state: dict[str, object] | None = None,
    fiber_executor: Executor | None = None,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    if steps <= 0 or batch_size <= 0 or learning_rate <= 0.0:
        raise ValueError("steps, batch_size, and learning_rate must be positive")
    if evaluation_points <= 0:
        raise ValueError("evaluation_points must be positive")
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative")
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    start_step = 0
    if resume_state is None:
        model.eval()
        best_metrics = tracelet_conditional_metrics(model, validation_examples)
        best_metrics["selected_step"] = 0.0
        best_state = _clone_model_state(model)
        history: list[dict[str, float]] = []
    else:
        required = {
            "completed_steps",
            "current_state_dict",
            "optimizer_state_dict",
            "numpy_rng_state",
            "best_state_dict",
            "best_metrics",
            "history",
        }
        missing = sorted(required - resume_state.keys())
        if missing:
            raise ValueError(f"resume state is missing keys: {missing}")
        start_step = int(resume_state["completed_steps"])
        if not 0 <= start_step <= steps:
            raise ValueError(
                f"resume completed_steps={start_step} is outside [0, {steps}]"
            )
        model.load_state_dict(resume_state["current_state_dict"])  # type: ignore[arg-type]
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])  # type: ignore[arg-type]
        rng.bit_generator.state = copy.deepcopy(resume_state["numpy_rng_state"])
        torch_rng_state = resume_state.get("torch_rng_state")
        if isinstance(torch_rng_state, Tensor):
            torch.set_rng_state(_cpu_byte_rng_state(torch_rng_state))
        cuda_rng_states = resume_state.get("cuda_rng_states")
        if cuda_rng_states is not None and torch.cuda.is_available():
            if not isinstance(cuda_rng_states, (list, tuple)):
                raise ValueError("cuda_rng_states must be a sequence of tensors")
            torch.cuda.set_rng_state_all(
                [_cpu_byte_rng_state(state) for state in cuda_rng_states]
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

    evaluation_interval = max(steps // evaluation_points, 1)
    model.train()
    for step in range(start_step, steps):
        step_started = perf_counter()
        batch = sample_tracelet_conditional_batch(
            train_records,
            batch_size=batch_size,
            rng=rng,
            fiber_cache=fiber_cache,
            late_time_fraction=late_time_fraction,
            operational_horizon=operational_horizon,
            progress_stratification_fraction=progress_stratification_fraction,
            ring_catalog=ring_catalog,
            causal_teachers=causal_teachers,
            causal_cache=causal_cache,
            fiber_executor=fiber_executor,
        )
        sampled_at = perf_counter()
        optimizer.zero_grad()
        loss = tracelet_conditional_batch_loss(
            model,
            batch,
            action_kl_weight=action_kl_weight,
            hazard_tilt_weight=hazard_tilt_weight,
        )
        _synchronize_model_device(model)
        forwarded_at = perf_counter()
        loss.backward()
        _synchronize_model_device(model)
        backward_at = perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        _synchronize_model_device(model)
        optimized_at = perf_counter()
        completed_steps = step + 1
        evaluated = (
            step == 0
            or completed_steps % evaluation_interval == 0
            or completed_steps == steps
        )
        if evaluated:
            model.eval()
            metrics = tracelet_conditional_metrics(model, validation_examples)
            metrics["step"] = float(completed_steps)
            metrics["train_batch_loss"] = float(loss.detach())
            timing_metrics = {
                "timing/sample_seconds": sampled_at - step_started,
                "timing/forward_seconds": forwarded_at - sampled_at,
                "timing/backward_seconds": backward_at - forwarded_at,
                "timing/optimizer_seconds": optimized_at - backward_at,
                "timing/update_seconds": optimized_at - step_started,
            }
            history.append(metrics)
            if metrics["conditional_gm_loss"] < best_metrics["conditional_gm_loss"]:
                best_metrics = {
                    key: value
                    for key, value in metrics.items()
                    if key not in {"step", "train_batch_loss"}
                }
                best_metrics["selected_step"] = float(completed_steps)
                best_state = _clone_model_state(model)
            if progress_callback is not None:
                progress_callback({**metrics, **timing_metrics})
            model.train()
        should_checkpoint = checkpoint_callback is not None and (
            completed_steps == steps
            or (
                checkpoint_interval > 0
                and completed_steps % checkpoint_interval == 0
            )
        )
        if should_checkpoint:
            checkpoint_callback(
                _training_recovery_state(
                    model=model,
                    optimizer=optimizer,
                    rng=rng,
                    completed_steps=completed_steps,
                    best_state=best_state,
                    best_metrics=best_metrics,
                    history=history,
                )
            )
    model.load_state_dict(best_state)
    model.eval()
    return history, best_metrics


def _clone_model_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }


def _synchronize_model_device(model: nn.Module) -> None:
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _cpu_byte_rng_state(state: object) -> Tensor:
    """Normalize a serialized RNG state for PyTorch's restore APIs.

    Recovery checkpoints are loaded onto the training device so model and
    optimizer tensors are ready for use.  That also moves serialized CUDA RNG
    states onto CUDA, while ``set_rng_state_all`` specifically requires CPU
    byte tensors.  Normalize both CPU and CUDA RNG payloads at the API boundary.
    """

    if not isinstance(state, Tensor):
        raise ValueError("serialized RNG states must be tensors")
    return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _training_recovery_state(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    completed_steps: int,
    best_state: dict[str, Tensor],
    best_metrics: dict[str, float],
    history: list[dict[str, float]],
) -> dict[str, object]:
    """Snapshot every state variable required for an exact training resume."""

    return {
        "recovery_format_version": 1,
        "completed_steps": int(completed_steps),
        "current_state_dict": _clone_model_state(model),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "numpy_rng_state": copy.deepcopy(rng.bit_generator.state),
        "torch_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_states": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
        "best_state_dict": {
            name: value.detach().clone() for name, value in best_state.items()
        },
        "best_metrics": dict(best_metrics),
        "history": [dict(row) for row in history],
    }


def _teacher_pairs(
    example: TraceletConditionalRateExample,
) -> tuple[tuple[str, float], ...]:
    """Return the complete-successor teacher support, including legacy rows."""

    if example.teacher_successor_rates:
        return example.teacher_successor_rates
    if example.teacher_successor_key is None:
        return ()
    return ((example.teacher_successor_key, float(example.teacher_rate)),)


def _sample_tracelet_progress(
    path: TraceProgressCTMC,
    *,
    time: float,
    rng: np.random.Generator,
    stratification_fraction: float,
) -> tuple[int, float]:
    """Sample path progress with an exact family-stratified importance weight.

    The base Generator Matching law is ``path.marginal(time)``.  The auxiliary
    proposal first chooses one teacher rule family present on the path
    uniformly, then one progress position in that family uniformly; the
    terminal position is treated as its own family.  Sampling from a mixture of
    the base law and this proposal exposes rare macro events more often.  The
    returned likelihood ratio preserves the original GM expectation exactly.
    """

    fraction = float(stratification_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("stratification_fraction must lie in [0, 1]")
    if fraction == 0.0:
        return path.sample_progress(time, rng), 1.0

    marginal = path.marginal(time)
    groups: dict[str, list[int]] = {}
    for progress, step in enumerate(path.trace.steps):
        groups.setdefault(step.rule_name, []).append(progress)
    groups["<TERMINAL>"] = [path.path_length]
    family_names = tuple(groups)

    if rng.random() < fraction:
        family = family_names[int(rng.integers(0, len(family_names)))]
        positions = groups[family]
        progress = int(positions[int(rng.integers(0, len(positions)))])
    else:
        progress = path.sample_progress(time, rng)

    family = (
        path.trace.steps[progress].rule_name
        if progress < path.path_length
        else "<TERMINAL>"
    )
    stratified_probability = 1.0 / (len(family_names) * len(groups[family]))
    proposal_probability = (
        (1.0 - fraction) * float(marginal[progress])
        + fraction * stratified_probability
    )
    if proposal_probability <= 0.0:
        raise RuntimeError("progress proposal assigned zero probability")
    importance_weight = float(marginal[progress]) / proposal_probability
    return progress, importance_weight


@torch.no_grad()
def sample_tracelet_ancestral(
    model: TraceletRatePredictor,
    *,
    rng: np.random.Generator,
    n_slots: int,
    operational_horizon: float = 7.0,
    time_step: float = 0.1,
    max_events: int = 32,
    fiber_cache: dict[str, tuple[MolecularGraph, TraceletFiber]] | None = None,
    rate_cache: dict[tuple[str, float], tuple[np.ndarray, float]] | None = None,
    max_fiber_cache: int = 1024,
    max_rate_cache: int = 4096,
    source_prior: MolecularSourcePrior | None = None,
) -> TraceletRollout:
    cached_fibers = fiber_cache if fiber_cache is not None else {}
    cached_rates = rate_cache if rate_cache is not None else {}
    state = (source_prior or NullSourcePrior()).sample(rng, n_slots=n_slots)
    event_times = []
    event_rules = []
    operational_time = 0.0
    direct_mark_sampler = getattr(model, "sample_rewrite_mark", None)
    direct_runtime = de_novo_rewrite_system() if callable(direct_mark_sampler) else None
    while operational_time < operational_horizon and len(event_times) < max_events:
        interval_end = min(operational_time + time_step, operational_horizon)
        frozen_time = 1.0 - exp(-(operational_time + interval_end) / 2.0)
        while operational_time < interval_end and len(event_times) < max_events:
            if callable(direct_mark_sampler):
                sampled = direct_mark_sampler(state, frozen_time, rng)
                total_hazard = float(sampled.total_hazard)
                if total_hazard <= 1e-12:
                    operational_time = interval_end
                    break
                waiting_time = float(rng.exponential(1.0 / total_hazard))
                remaining = interval_end - operational_time
                if waiting_time >= remaining:
                    operational_time = interval_end
                    break
                operational_time += waiting_time
                assert direct_runtime is not None
                state = direct_runtime.apply(
                    state,
                    sampled.rule_name,
                    sampled.action,
                )
                event_times.append(operational_time)
                event_rules.append(sampled.rule_name)
                continue
            state_key = canonical_state_key(state)
            cached = cached_fibers.get(state_key)
            if cached is None:
                representative = canonical_tracelet_representative(state, state_key)
                catalog = getattr(model, "ring_catalog", None)
                cached = (
                    representative,
                    enumerate_tracelet_cnof_fiber(
                        representative,
                        ring_catalog=catalog,
                        allow_bond_reroute=bool(
                            getattr(model, "enable_bond_reroute", False)
                        ),
                    ),
                )
                if len(cached_fibers) < max_fiber_cache:
                    cached_fibers[state_key] = cached
            representative, fiber = cached
            if not fiber.transitions:
                operational_time = interval_end
                break
            rate_key = (state_key, round(frozen_time, 12))
            cached_prediction = cached_rates.get(rate_key)
            if cached_prediction is None:
                prediction = model.predict_tracelet_fiber(
                    representative,
                    frozen_time,
                    fiber=fiber,
                )
                total_hazard = float(prediction.total_hazard)
                probabilities = (
                    (prediction.marked_rates / prediction.total_hazard)
                    .detach()
                    .cpu()
                    .numpy()
                    if total_hazard > 0.0
                    else np.zeros(len(fiber.transitions), dtype=np.float64)
                )
                if len(cached_rates) < max_rate_cache:
                    cached_rates[rate_key] = (probabilities, total_hazard)
            else:
                probabilities, total_hazard = cached_prediction
            if total_hazard <= 1e-12:
                operational_time = interval_end
                break
            waiting_time = float(rng.exponential(1.0 / total_hazard))
            remaining = interval_end - operational_time
            if waiting_time >= remaining:
                operational_time = interval_end
                break
            operational_time += waiting_time
            selected = fiber.transitions[
                int(rng.choice(len(fiber.transitions), p=probabilities))
            ]
            state = selected.successor
            event_times.append(operational_time)
            event_rules.append(selected.rule_name)
    return TraceletRollout(
        state,
        tuple(event_times),
        tuple(event_rules),
        len(event_times) >= max_events,
    )
