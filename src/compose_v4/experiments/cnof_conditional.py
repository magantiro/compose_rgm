"""Conditional Generator Matching gate on a larger C/N/O/F corpus."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import exp

import numpy as np
import torch
from torch import Tensor

from compose_v4.chem.molecular_graph import (
    ELEMENT_TO_IDX,
    MolecularGraph,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import (
    empty_molecular_graph,
    is_connected_or_null,
    is_valid_state,
    pad_molecular_graph,
)
from compose_v4.gm.loss import rate_bregman_loss
from compose_v4.model.rate_model import FactorizedRateModel
from compose_v4.rewrite.compiler import compile_null_to_target
from compose_v4.rewrite.factorized_fiber import (
    FactorizedFiber,
    enumerate_factorized_cnof_fiber,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.progress import TraceProgressCTMC


@dataclass(frozen=True)
class PathRecord:
    target_key: str
    path: TraceProgressCTMC


@dataclass(frozen=True)
class ConditionalRateExample:
    state: MolecularGraph
    time: float
    teacher_successor_key: str | None
    teacher_rate: float
    fiber: FactorizedFiber


@dataclass(frozen=True)
class FactorizedRollout:
    final_state: MolecularGraph
    event_times: tuple[float, ...]
    event_rules: tuple[str, ...]
    exhausted_event_budget: bool


def build_path_records(
    smiles: tuple[str, ...],
    *,
    n_slots: int,
    seed: int,
    traces_per_molecule: int = 1,
    defer_bond_orders: bool = False,
) -> tuple[PathRecord, ...]:
    if traces_per_molecule <= 0:
        raise ValueError("traces_per_molecule must be positive")
    rng = np.random.default_rng(seed)
    records = []
    for text in smiles:
        target = pad_molecular_graph(smiles_to_molecular_graph(text), n_slots)
        target_key = canonical_state_key(target)
        for _ in range(traces_per_molecule):
            trace = compile_null_to_target(
                target,
                rng=rng,
                defer_bond_orders=defer_bond_orders,
            )
            records.append(PathRecord(target_key, TraceProgressCTMC(trace)))
    return tuple(records)


def sample_conditional_batch(
    records: tuple[PathRecord, ...],
    *,
    batch_size: int,
    rng: np.random.Generator,
    fiber_cache: dict[str, tuple[MolecularGraph, FactorizedFiber]],
    late_time_fraction: float = 0.0,
    operational_horizon: float = 7.0,
) -> tuple[ConditionalRateExample, ...]:
    if not records or batch_size <= 0:
        raise ValueError("records must be non-empty and batch_size positive")
    if not 0.0 <= late_time_fraction <= 1.0:
        raise ValueError("late_time_fraction must lie in [0, 1]")
    if operational_horizon <= 0.0:
        raise ValueError("operational_horizon must be positive")
    examples = []
    for record_index in rng.integers(0, len(records), size=batch_size):
        path = records[int(record_index)].path
        if rng.random() < late_time_fraction:
            operational_time = float(rng.uniform(0.0, operational_horizon))
            time = 1.0 - exp(-operational_time)
        else:
            time = float(rng.uniform(0.01, 0.99))
        progress = path.sample_progress(time, rng)
        raw_state = path.states[progress]
        state_key = canonical_state_key(raw_state)
        cached = fiber_cache.get(state_key)
        if cached is None:
            representative = raw_state
            fiber = enumerate_factorized_cnof_fiber(representative)
            cached = (representative, fiber)
            fiber_cache[state_key] = cached
        state, fiber = cached
        if progress < path.path_length:
            teacher_key = canonical_state_key(path.states[progress + 1])
            teacher_rate = path.operational_jump_rate(progress)
            if teacher_key not in {
                transition.successor_key for transition in fiber.transitions
            }:
                raise RuntimeError("conditional teacher is outside factorized fiber")
        else:
            teacher_key = None
            teacher_rate = 0.0
        examples.append(
            ConditionalRateExample(
                state=state,
                time=time,
                teacher_successor_key=teacher_key,
                teacher_rate=teacher_rate,
                fiber=fiber,
            )
        )
    return tuple(examples)


def conditional_batch_loss(
    model: FactorizedRateModel,
    examples: tuple[ConditionalRateExample, ...],
    *,
    action_kl_weight: float = 0.0,
    hazard_tilt_weight: float = 0.0,
) -> Tensor:
    if action_kl_weight < 0.0 or hazard_tilt_weight < 0.0:
        raise ValueError("regularization weights must be non-negative")
    losses = []
    action_kls = []
    hazard_tilts = []
    for example in examples:
        prediction = model.predict_factorized_fiber(
            example.state,
            example.time,
            fiber=example.fiber,
        )
        teacher_prediction = (
            prediction.successor_rate(example.teacher_successor_key)
            if example.teacher_successor_key is not None
            else prediction.marked_rates.new_zeros(())
        )
        losses.append(
            rate_bregman_loss(
                prediction.total_hazard,
                teacher_prediction,
                prediction.marked_rates.new_tensor(example.teacher_rate),
            )
        )
        if prediction.action_kl_to_prior is not None:
            action_kls.append(prediction.action_kl_to_prior)
        if prediction.squared_log_hazard_tilt is not None:
            hazard_tilts.append(prediction.squared_log_hazard_tilt)
    loss = torch.stack(losses).mean()
    if action_kls:
        loss = loss + action_kl_weight * torch.stack(action_kls).mean()
    if hazard_tilts:
        loss = loss + hazard_tilt_weight * torch.stack(hazard_tilts).mean()
    return loss


@torch.no_grad()
def conditional_metrics(
    model: FactorizedRateModel,
    examples: tuple[ConditionalRateExample, ...],
) -> dict[str, float]:
    losses = []
    support_mass = []
    top_hits = []
    terminal_hazards = []
    action_kls = []
    hazard_tilts = []
    for example in examples:
        prediction = model.predict_factorized_fiber(
            example.state,
            example.time,
            fiber=example.fiber,
        )
        if example.teacher_successor_key is None:
            selected = prediction.marked_rates.new_zeros(())
            terminal_hazards.append(float(prediction.total_hazard))
        else:
            selected = prediction.successor_rate(example.teacher_successor_key)
            support_mass.append(
                float(selected) / max(float(prediction.total_hazard), 1e-12)
            )
            rates = prediction.successor_rate_dict()
            top_key = max(rates, key=lambda key: float(rates[key]))
            top_hits.append(float(top_key == example.teacher_successor_key))
        losses.append(
            float(
                rate_bregman_loss(
                    prediction.total_hazard,
                    selected,
                    prediction.marked_rates.new_tensor(example.teacher_rate),
                )
            )
        )
        if prediction.action_kl_to_prior is not None:
            action_kls.append(float(prediction.action_kl_to_prior))
        if prediction.squared_log_hazard_tilt is not None:
            hazard_tilts.append(float(prediction.squared_log_hazard_tilt))
    metrics = {
        "conditional_gm_loss": float(np.mean(losses)),
        "mean_teacher_successor_mass": float(np.mean(support_mass)),
        "conditional_top_successor_accuracy": float(np.mean(top_hits)),
        "mean_terminal_hazard": float(np.mean(terminal_hazards)),
    }
    if action_kls:
        metrics["mean_action_kl_to_prior"] = float(np.mean(action_kls))
    if hazard_tilts:
        metrics["mean_squared_log_hazard_tilt"] = float(np.mean(hazard_tilts))
    return metrics


def fixed_conditional_examples(
    records: tuple[PathRecord, ...],
    *,
    samples_per_record: int,
    seed: int,
    fiber_cache: dict[str, tuple[MolecularGraph, FactorizedFiber]],
    late_time_fraction: float = 0.0,
    operational_horizon: float = 7.0,
) -> tuple[ConditionalRateExample, ...]:
    rng = np.random.default_rng(seed)
    examples = []
    for record in records:
        examples.extend(
            sample_conditional_batch(
                (record,),
                batch_size=samples_per_record,
                rng=rng,
                fiber_cache=fiber_cache,
                late_time_fraction=late_time_fraction,
                operational_horizon=operational_horizon,
            )
        )
    return tuple(examples)


def train_conditional_model(
    model: FactorizedRateModel,
    train_records: tuple[PathRecord, ...],
    validation_examples: tuple[ConditionalRateExample, ...],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    action_kl_weight: float = 0.0,
    hazard_tilt_weight: float = 0.0,
    seed: int,
    fiber_cache: dict[str, tuple[MolecularGraph, FactorizedFiber]],
    late_time_fraction: float = 0.0,
    operational_horizon: float = 7.0,
    evaluation_points: int = 20,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if evaluation_points <= 0:
        raise ValueError("evaluation_points must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if action_kl_weight < 0.0 or hazard_tilt_weight < 0.0:
        raise ValueError("regularization weights must be non-negative")
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history = []
    model.eval()
    best_metrics = conditional_metrics(model, validation_examples)
    best_metrics["selected_step"] = 0.0
    best_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    model.train()
    for step in range(steps):
        batch = sample_conditional_batch(
            train_records,
            batch_size=batch_size,
            rng=rng,
            fiber_cache=fiber_cache,
            late_time_fraction=late_time_fraction,
            operational_horizon=operational_horizon,
        )
        optimizer.zero_grad()
        loss = conditional_batch_loss(
            model,
            batch,
            action_kl_weight=action_kl_weight,
            hazard_tilt_weight=hazard_tilt_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(steps // evaluation_points, 1) == 0:
            model.eval()
            metrics = conditional_metrics(model, validation_examples)
            metrics["step"] = float(step + 1)
            metrics["train_batch_loss"] = float(loss.detach())
            history.append(metrics)
            if metrics["conditional_gm_loss"] < best_metrics["conditional_gm_loss"]:
                best_metrics = {
                    key: value
                    for key, value in metrics.items()
                    if key not in {"step", "train_batch_loss"}
                }
                best_metrics["selected_step"] = float(step + 1)
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
            model.train()
    model.load_state_dict(best_state)
    model.eval()
    return history, best_metrics


@torch.no_grad()
def sample_factorized_ancestral(
    model: FactorizedRateModel,
    *,
    rng: np.random.Generator,
    n_slots: int,
    operational_horizon: float = 7.0,
    time_step: float = 0.1,
    max_events: int = 32,
    fiber_cache: dict[str, tuple[MolecularGraph, FactorizedFiber]] | None = None,
    rate_cache: dict[tuple[str, float], tuple[np.ndarray, float]] | None = None,
) -> FactorizedRollout:
    cached_fibers = fiber_cache if fiber_cache is not None else {}
    cached_rates = rate_cache if rate_cache is not None else {}
    state = empty_molecular_graph(n_slots)
    event_times = []
    event_rules = []
    operational_time = 0.0
    while operational_time < operational_horizon and len(event_times) < max_events:
        interval_end = min(operational_time + time_step, operational_horizon)
        frozen_time = 1.0 - exp(-(operational_time + interval_end) / 2.0)
        while operational_time < interval_end and len(event_times) < max_events:
            state_key = canonical_state_key(state)
            cached = cached_fibers.get(state_key)
            if cached is None:
                representative = state
                fiber = enumerate_factorized_cnof_fiber(representative)
                cached = (representative, fiber)
                cached_fibers[state_key] = cached
            representative, fiber = cached
            if not fiber.transitions:
                operational_time = interval_end
                break
            rate_key = (state_key, round(frozen_time, 12))
            cached_prediction = cached_rates.get(rate_key)
            if cached_prediction is None:
                prediction = model.predict_factorized_fiber(
                    representative,
                    frozen_time,
                    fiber=fiber,
                )
                total_hazard = float(prediction.total_hazard)
                probabilities = (
                    prediction.marked_rates / prediction.total_hazard
                ).detach().cpu().numpy()
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
            transition_index = int(rng.choice(len(fiber.transitions), p=probabilities))
            selected = fiber.transitions[transition_index]
            state = selected.successor
            event_times.append(operational_time)
            event_rules.append(selected.rule_name)
    return FactorizedRollout(
        final_state=state,
        event_times=tuple(event_times),
        event_rules=tuple(event_rules),
        exhausted_event_budget=len(event_times) >= max_events,
    )


def corpus_rollout_metrics(
    rollouts: tuple[FactorizedRollout, ...],
    *,
    train_smiles: tuple[str, ...],
    reference_smiles: tuple[str, ...],
) -> dict[str, object]:
    train_keys = {
        canonical_state_key(smiles_to_molecular_graph(text)) for text in train_smiles
    }
    reference_graphs = tuple(
        smiles_to_molecular_graph(text) for text in reference_smiles
    )
    reference_keys = {canonical_state_key(graph) for graph in reference_graphs}
    final_keys = [canonical_state_key(item.final_state) for item in rollouts]
    counts = Counter(final_keys)
    generated_sizes = [item.final_state.n_real_atoms for item in rollouts]
    reference_sizes = [graph.n_real_atoms for graph in reference_graphs]
    max_size = max((*generated_sizes, *reference_sizes), default=0)
    generated_hist = np.bincount(generated_sizes, minlength=max_size + 1) / len(rollouts)
    reference_hist = (
        np.bincount(reference_sizes, minlength=max_size + 1) / len(reference_graphs)
    )
    size_tv = 0.5 * float(np.abs(generated_hist - reference_hist).sum())
    generated_structure = [_graph_statistics(item.final_state) for item in rollouts]
    reference_structure = [_graph_statistics(graph) for graph in reference_graphs]
    generated_cycles = [item[1] for item in generated_structure]
    reference_cycles = [item[1] for item in reference_structure]
    max_cycle_rank = max((*generated_cycles, *reference_cycles), default=0)
    generated_cycle_hist = (
        np.bincount(generated_cycles, minlength=max_cycle_rank + 1) / len(rollouts)
    )
    reference_cycle_hist = (
        np.bincount(reference_cycles, minlength=max_cycle_rank + 1)
        / len(reference_graphs)
    )
    cnof_indices = tuple(ELEMENT_TO_IDX[symbol] for symbol in ("C", "N", "O", "F"))
    generated_element_counts = np.sum(
        [item[2] for item in generated_structure], axis=0
    )
    reference_element_counts = np.sum(
        [item[2] for item in reference_structure], axis=0
    )
    generated_bond_counts = np.sum([item[3] for item in generated_structure], axis=0)
    reference_bond_counts = np.sum([item[3] for item in reference_structure], axis=0)
    reference_max_cycle_rank = max(reference_cycles, default=0)
    event_rule_counts = Counter(
        rule for rollout in rollouts for rule in rollout.event_rules
    )
    total_events = sum(event_rule_counts.values())
    return {
        "samples": len(rollouts),
        "valid_fraction": float(
            np.mean([is_valid_state(item.final_state) for item in rollouts])
        ),
        "connected_or_null_fraction": float(
            np.mean([is_connected_or_null(item.final_state) for item in rollouts])
        ),
        "non_null_fraction": float(
            np.mean([item.final_state.n_real_atoms > 0 for item in rollouts])
        ),
        "unique_fraction": len(counts) / len(rollouts),
        "train_support_fraction": float(np.mean([key in train_keys for key in final_keys])),
        "full_reference_support_fraction": float(
            np.mean([key in reference_keys for key in final_keys])
        ),
        "novel_to_train_fraction": float(
            np.mean([key not in train_keys and key != "<NULL>" for key in final_keys])
        ),
        "mean_atoms": float(np.mean(generated_sizes)),
        "reference_mean_atoms": float(np.mean(reference_sizes)),
        "atom_count_total_variation": size_tv,
        "mean_bonds": float(np.mean([item[0] for item in generated_structure])),
        "reference_mean_bonds": float(
            np.mean([item[0] for item in reference_structure])
        ),
        "mean_cycle_rank": float(np.mean(generated_cycles)),
        "reference_mean_cycle_rank": float(np.mean(reference_cycles)),
        "cycle_rank_total_variation": 0.5
        * float(np.abs(generated_cycle_hist - reference_cycle_hist).sum()),
        "cycle_rank_above_reference_max_fraction": float(
            np.mean([rank > reference_max_cycle_rank for rank in generated_cycles])
        ),
        "element_distribution_total_variation": _count_total_variation(
            generated_element_counts[list(cnof_indices)],
            reference_element_counts[list(cnof_indices)],
        ),
        "bond_order_distribution_total_variation": _count_total_variation(
            generated_bond_counts[1:4],
            reference_bond_counts[1:4],
        ),
        "mean_events": float(np.mean([len(item.event_times) for item in rollouts])),
        "event_rule_fractions": {
            rule: event_rule_counts[rule] / max(total_events, 1)
            for rule in (
                "atom_insert",
                "atom_delete",
                "atom_restate",
                "bond_insert",
                "bond_delete",
                "bond_reorder",
            )
        },
        "event_budget_exhaustion_fraction": float(
            np.mean([item.exhausted_event_budget for item in rollouts])
        ),
        "top_final_states": counts.most_common(20),
    }


def _graph_statistics(graph: MolecularGraph) -> tuple[int, int, np.ndarray, np.ndarray]:
    real = is_element(graph.atom_types)
    vertices = np.flatnonzero(real)
    bonds = graph.bonds[np.ix_(vertices, vertices)]
    edge_count = int(np.count_nonzero(np.triu(bonds, k=1)))
    cycle_rank = max(edge_count - int(len(vertices)) + int(len(vertices) > 0), 0)
    element_counts = np.bincount(graph.atom_types[real], minlength=12)
    bond_counts = np.bincount(
        bonds[np.triu_indices(len(vertices), k=1)],
        minlength=5,
    )
    return edge_count, cycle_rank, element_counts, bond_counts


def _count_total_variation(left: np.ndarray, right: np.ndarray) -> float:
    left_total = float(left.sum())
    right_total = float(right.sum())
    if left_total == 0.0 or right_total == 0.0:
        return float(left_total != right_total)
    return 0.5 * float(np.abs(left / left_total - right / right_total).sum())
