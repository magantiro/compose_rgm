"""Tiny-corpus overfit and target-free rollout gate for rewrite rates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import exp

import numpy as np
import torch
from torch import Tensor

from compose_v4.chem.molecular_graph import MolecularGraph, smiles_to_molecular_graph
from compose_v4.chem.state import (
    empty_molecular_graph,
    is_connected_or_null,
    is_valid_state,
    pad_molecular_graph,
)
from compose_v4.gm.loss import (
    multi_successor_rate_bregman_loss,
    rate_bregman_loss,
)
from compose_v4.model.rate_model import WholeGraphRateModel
from compose_v4.rewrite.compiler import compile_null_to_target
from compose_v4.rewrite.fiber import (
    ActionFiberSpec,
    MarkedTransition,
    enumerate_action_fiber,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.progress import TraceProgressCTMC


TINY_CNOF_SMILES = (
    "C",
    "N",
    "O",
    "F",
    "CC",
    "CO",
    "CN",
    "C=C",
    "C#N",
    "CCO",
    "CCN",
    "C1CC1",
)


@dataclass(frozen=True)
class RateTrainingExample:
    state: MolecularGraph
    time: float
    teacher_successor_key: str | None
    teacher_rate: float
    transitions: tuple[MarkedTransition, ...]
    target_smiles: str
    progress: int
    path_length: int


@dataclass(frozen=True)
class MarginalRateTrainingExample:
    """Exact tiny-system marginal generator target at one state and time."""

    state: MolecularGraph
    time: float
    teacher_successor_rates: tuple[tuple[str, float], ...]
    state_probability: float
    transitions: tuple[MarkedTransition, ...]


@dataclass(frozen=True)
class Rollout:
    final_state: MolecularGraph
    states: tuple[MolecularGraph, ...]
    event_times: tuple[float, ...]
    exhausted_event_budget: bool


def build_tiny_paths(
    *,
    seed: int = 20260714,
    n_slots: int = 3,
) -> tuple[tuple[str, TraceProgressCTMC], ...]:
    rng = np.random.default_rng(seed)
    paths = []
    for smiles in TINY_CNOF_SMILES:
        target = pad_molecular_graph(smiles_to_molecular_graph(smiles), n_slots)
        trace = compile_null_to_target(target, rng=rng)
        paths.append((smiles, TraceProgressCTMC(trace)))
    return tuple(paths)


def build_coverage_batch(
    paths: tuple[tuple[str, TraceProgressCTMC], ...],
    *,
    spec: ActionFiberSpec | None = None,
) -> tuple[RateTrainingExample, ...]:
    """Cover every prefix once with a valid non-singular teacher rate."""

    language = spec or ActionFiberSpec.neutral_cnof()
    examples = []
    fiber_cache: dict[tuple[bytes, ...], tuple[MarkedTransition, ...]] = {}
    for target_smiles, path in paths:
        length = path.path_length
        for progress, state in enumerate(path.states):
            # This time gives every prefix nonzero conditional probability. The
            # batch is a deterministic coverage gate, not an unbiased corpus
            # estimator.
            time = (progress + 0.5) / (length + 1.0)
            fingerprint = _state_fingerprint(state)
            transitions = fiber_cache.get(fingerprint)
            if transitions is None:
                transitions = enumerate_action_fiber(state, spec=language)
                fiber_cache[fingerprint] = transitions
            if progress < length:
                teacher_key = canonical_state_key(path.states[progress + 1])
                teacher_rate = path.operational_jump_rate(progress)
            else:
                teacher_key = None
                teacher_rate = 0.0
            examples.append(
                RateTrainingExample(
                    state=state,
                    time=time,
                    teacher_successor_key=teacher_key,
                    teacher_rate=teacher_rate,
                    transitions=transitions,
                    target_smiles=target_smiles,
                    progress=progress,
                    path_length=length,
                )
            )
    return tuple(examples)


def build_exact_marginal_batch(
    paths: tuple[tuple[str, TraceProgressCTMC], ...],
    *,
    times: tuple[float, ...] = (0.02, 0.15, 0.35, 0.6, 0.82, 0.97),
    spec: ActionFiberSpec | None = None,
) -> tuple[MarginalRateTrainingExample, ...]:
    """Exactly marginalize conditional trace generators on the tiny graph.

    This is a diagnostic oracle available because the complete tiny path
    mixture is enumerable. Large-corpus training uses conditional samples whose
    expectation has the same pointwise marginal-generator target.
    """

    if not paths or not times:
        raise ValueError("paths and times must be non-empty")
    if any(not 0.0 < float(time) < 1.0 for time in times):
        raise ValueError("marginal training times must lie strictly inside (0, 1)")
    language = spec or ActionFiberSpec.neutral_cnof()
    fiber_cache: dict[str, tuple[MarkedTransition, ...]] = {}
    examples = []

    for time in times:
        # canonical chemical state -> [representative, mass, weighted rates]
        grouped: dict[
            str,
            list[object],
        ] = {}
        for _target_smiles, path in paths:
            probabilities = path.marginal(time)
            for progress, state in enumerate(path.states):
                conditional_mass = float(probabilities[progress]) / len(paths)
                if conditional_mass <= 0.0:
                    continue
                state_key = canonical_state_key(state)
                if state_key not in grouped:
                    grouped[state_key] = [state, 0.0, {}]
                accumulator = grouped[state_key]
                accumulator[1] = float(accumulator[1]) + conditional_mass
                if progress < path.path_length:
                    successor_key = canonical_state_key(path.states[progress + 1])
                    weighted_rates = accumulator[2]
                    assert isinstance(weighted_rates, dict)
                    weighted_rates[successor_key] = (
                        float(weighted_rates.get(successor_key, 0.0))
                        + conditional_mass * path.operational_jump_rate(progress)
                    )

        for state_key, (state, marginal_mass, weighted_rates) in grouped.items():
            assert isinstance(state, MolecularGraph)
            marginal_mass = float(marginal_mass)
            assert isinstance(weighted_rates, dict)
            transitions = fiber_cache.get(state_key)
            if transitions is None:
                transitions = enumerate_action_fiber(state, spec=language)
                fiber_cache[state_key] = transitions
            available = {transition.successor_key for transition in transitions}
            missing = set(weighted_rates) - available
            if missing:
                raise ValueError(
                    f"teacher successors are outside the declared action fiber: {missing}"
                )
            teacher_rates = tuple(
                sorted(
                    (key, float(weighted_rate) / marginal_mass)
                    for key, weighted_rate in weighted_rates.items()
                )
            )
            examples.append(
                MarginalRateTrainingExample(
                    state=state,
                    time=float(time),
                    teacher_successor_rates=teacher_rates,
                    state_probability=marginal_mass,
                    transitions=transitions,
                )
            )
    return tuple(examples)


def batch_loss(
    model: WholeGraphRateModel,
    examples: tuple[RateTrainingExample, ...],
) -> Tensor:
    losses = []
    for example in examples:
        prediction = model.predict_fiber(
            example.state,
            example.time,
            transitions=example.transitions,
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
    return torch.stack(losses).mean()


@torch.no_grad()
def batch_metrics(
    model: WholeGraphRateModel,
    examples: tuple[RateTrainingExample, ...],
) -> dict[str, float]:
    losses = []
    positive_mass = []
    teacher_rate_errors = []
    terminal_hazards = []
    top_successor_hits = []
    for example in examples:
        prediction = model.predict_fiber(
            example.state,
            example.time,
            transitions=example.transitions,
        )
        rates_by_successor = prediction.successor_rate_dict()
        if example.teacher_successor_key is None:
            teacher_prediction = prediction.marked_rates.new_zeros(())
            terminal_hazards.append(float(prediction.total_hazard))
        else:
            teacher_prediction = prediction.successor_rate(
                example.teacher_successor_key
            )
            total = float(prediction.total_hazard)
            positive_mass.append(float(teacher_prediction) / max(total, 1e-12))
            teacher_rate_errors.append(
                abs(float(teacher_prediction) - example.teacher_rate)
            )
            top_key = max(rates_by_successor, key=lambda key: float(rates_by_successor[key]))
            top_successor_hits.append(float(top_key == example.teacher_successor_key))
        losses.append(
            float(
                rate_bregman_loss(
                    prediction.total_hazard,
                    teacher_prediction,
                    prediction.marked_rates.new_tensor(example.teacher_rate),
                )
            )
        )
    return {
        "mean_loss": float(np.mean(losses)),
        "mean_teacher_successor_mass": float(np.mean(positive_mass)),
        "mean_teacher_rate_absolute_error": float(np.mean(teacher_rate_errors)),
        "mean_terminal_hazard": float(np.mean(terminal_hazards)),
        "conditional_top_successor_accuracy": float(np.mean(top_successor_hits)),
    }


def train_coverage_batch(
    model: WholeGraphRateModel,
    examples: tuple[RateTrainingExample, ...],
    *,
    epochs: int = 600,
    learning_rate: float = 2e-3,
) -> tuple[dict[str, float], dict[str, float], list[float]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    initial = batch_metrics(model, examples)
    history = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = batch_loss(model, examples)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if epoch == 0 or (epoch + 1) % max(epochs // 20, 1) == 0:
            history.append(float(loss.detach()))
    model.eval()
    final = batch_metrics(model, examples)
    return initial, final, history


def marginal_batch_loss(
    model: WholeGraphRateModel,
    examples: tuple[MarginalRateTrainingExample, ...],
) -> Tensor:
    losses = []
    for example in examples:
        prediction = model.predict_fiber(
            example.state,
            example.time,
            transitions=example.transitions,
        )
        if example.teacher_successor_rates:
            predicted = torch.stack(
                [
                    prediction.successor_rate(key)
                    for key, _rate in example.teacher_successor_rates
                ]
            )
            teacher = prediction.marked_rates.new_tensor(
                [rate for _key, rate in example.teacher_successor_rates]
            )
            loss = multi_successor_rate_bregman_loss(
                prediction.total_hazard,
                predicted,
                teacher,
            )
        else:
            loss = prediction.total_hazard
        losses.append(loss)
    return torch.stack(losses).mean()


@torch.no_grad()
def marginal_batch_metrics(
    model: WholeGraphRateModel,
    examples: tuple[MarginalRateTrainingExample, ...],
) -> dict[str, float]:
    losses = []
    hazard_errors = []
    vector_l1_errors = []
    teacher_support_mass = []
    top_hits = []
    zero_target_hazards = []
    for example in examples:
        prediction = model.predict_fiber(
            example.state,
            example.time,
            transitions=example.transitions,
        )
        teacher_total = sum(rate for _key, rate in example.teacher_successor_rates)
        if example.teacher_successor_rates:
            predicted = torch.stack(
                [
                    prediction.successor_rate(key)
                    for key, _rate in example.teacher_successor_rates
                ]
            )
            teacher = prediction.marked_rates.new_tensor(
                [rate for _key, rate in example.teacher_successor_rates]
            )
            loss = multi_successor_rate_bregman_loss(
                prediction.total_hazard,
                predicted,
                teacher,
            )
            selected_predicted = float(predicted.sum())
            total_predicted = float(prediction.total_hazard)
            teacher_support_mass.append(
                selected_predicted / max(total_predicted, 1e-12)
            )
            vector_l1_errors.append(
                float(torch.abs(predicted - teacher).sum())
                + max(total_predicted - selected_predicted, 0.0)
            )
            teacher_keys = [key for key, _rate in example.teacher_successor_rates]
            predicted_rates = prediction.successor_rate_dict()
            predicted_top = max(
                predicted_rates,
                key=lambda key: float(predicted_rates[key]),
            )
            teacher_top = teacher_keys[int(torch.argmax(teacher))]
            top_hits.append(float(predicted_top == teacher_top))
        else:
            loss = prediction.total_hazard
            vector_l1_errors.append(float(prediction.total_hazard))
            zero_target_hazards.append(float(prediction.total_hazard))
        losses.append(float(loss))
        hazard_errors.append(abs(float(prediction.total_hazard) - teacher_total))
    return {
        "mean_loss": float(np.mean(losses)),
        "mean_total_hazard_absolute_error": float(np.mean(hazard_errors)),
        "mean_rate_vector_l1_error": float(np.mean(vector_l1_errors)),
        "mean_teacher_support_mass": float(np.mean(teacher_support_mass)),
        "top_successor_accuracy": float(np.mean(top_hits)),
        "mean_zero_target_hazard": float(np.mean(zero_target_hazards)),
    }


def train_exact_marginal_batch(
    model: WholeGraphRateModel,
    examples: tuple[MarginalRateTrainingExample, ...],
    *,
    steps: int = 800,
    batch_size: int = 24,
    learning_rate: float = 2e-3,
    seed: int = 20260714,
) -> tuple[dict[str, float], dict[str, float], list[float]]:
    """Importance-sample exact marginal targets uniformly across states.

    Uniform group sampling changes optimization emphasis but not the pointwise
    optimum. It deliberately covers rare states such as late-time null during
    this diagnostic overfit gate.
    """

    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch size must be positive")
    if not examples:
        raise ValueError("examples must be non-empty")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    initial = marginal_batch_metrics(model, examples)
    history = []
    model.train()
    for step in range(steps):
        indices = rng.choice(
            len(examples),
            size=min(batch_size, len(examples)),
            replace=False,
        )
        minibatch = tuple(examples[int(index)] for index in indices)
        optimizer.zero_grad()
        loss = marginal_batch_loss(model, minibatch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(steps // 20, 1) == 0:
            history.append(float(loss.detach()))
    model.eval()
    final = marginal_batch_metrics(model, examples)
    return initial, final, history


@torch.no_grad()
def sample_piecewise_ancestral(
    model: WholeGraphRateModel,
    *,
    rng: np.random.Generator,
    n_slots: int = 3,
    operational_horizon: float = 5.0,
    time_step: float = 0.1,
    max_events: int = 24,
    spec: ActionFiberSpec | None = None,
    fiber_cache: dict[
        tuple[bytes, ...], tuple[MarkedTransition, ...]
    ] | None = None,
    rate_cache: dict[
        tuple[tuple[bytes, ...], float], tuple[np.ndarray, float]
    ] | None = None,
) -> Rollout:
    """Sample a target-free CTMC with rates frozen within small time bins."""

    if operational_horizon <= 0 or time_step <= 0 or max_events <= 0:
        raise ValueError("horizon, time step, and event budget must be positive")
    language = spec or ActionFiberSpec.neutral_cnof()
    cached_fibers = fiber_cache if fiber_cache is not None else {}
    cached_rates = rate_cache if rate_cache is not None else {}
    state = empty_molecular_graph(n_slots)
    states = [state]
    event_times = []
    operational_time = 0.0

    while operational_time < operational_horizon and len(event_times) < max_events:
        interval_end = min(operational_time + time_step, operational_horizon)
        frozen_time = 1.0 - exp(-(operational_time + interval_end) / 2.0)
        while operational_time < interval_end and len(event_times) < max_events:
            fingerprint = _state_fingerprint(state)
            transitions = cached_fibers.get(fingerprint)
            if transitions is None:
                transitions = enumerate_action_fiber(state, spec=language)
                cached_fibers[fingerprint] = transitions
            if not transitions:
                operational_time = interval_end
                break
            rate_key = (fingerprint, round(frozen_time, 12))
            cached_prediction = cached_rates.get(rate_key)
            if cached_prediction is None:
                prediction = model.predict_fiber(
                    state,
                    frozen_time,
                    transitions=transitions,
                )
                total_hazard = float(prediction.total_hazard)
                probabilities = (
                    prediction.marked_rates / prediction.total_hazard
                ).detach().cpu().numpy()
                cached_rates[rate_key] = (probabilities, total_hazard)
            else:
                probabilities, total_hazard = cached_prediction
            if total_hazard <= 0.0:
                operational_time = interval_end
                break
            waiting_time = float(rng.exponential(1.0 / total_hazard))
            remaining = interval_end - operational_time
            if waiting_time >= remaining:
                operational_time = interval_end
                break
            operational_time += waiting_time
            index = int(rng.choice(len(transitions), p=probabilities))
            state = transitions[index].successor
            states.append(state)
            event_times.append(operational_time)

    return Rollout(
        final_state=state,
        states=tuple(states),
        event_times=tuple(event_times),
        exhausted_event_budget=len(event_times) >= max_events,
    )


def rollout_metrics(
    rollouts: tuple[Rollout, ...],
    *,
    reference_smiles: tuple[str, ...] = TINY_CNOF_SMILES,
) -> dict[str, object]:
    reference_sequence = [
        canonical_state_key(smiles_to_molecular_graph(smiles))
        for smiles in reference_smiles
    ]
    reference_counts = Counter(reference_sequence)
    reference_keys = set(reference_counts)
    final_keys = [canonical_state_key(rollout.final_state) for rollout in rollouts]
    counts = Counter(final_keys)
    sample_count = max(len(rollouts), 1)
    reference_count = max(len(reference_sequence), 1)
    union = set(counts) | reference_keys
    total_variation = 0.5 * sum(
        abs(
            counts.get(key, 0) / sample_count
            - reference_counts.get(key, 0) / reference_count
        )
        for key in union
    )
    return {
        "samples": len(rollouts),
        "valid_fraction": float(
            np.mean([is_valid_state(rollout.final_state) for rollout in rollouts])
        ),
        "connected_or_null_fraction": float(
            np.mean(
                [is_connected_or_null(rollout.final_state) for rollout in rollouts]
            )
        ),
        "non_null_fraction": float(
            np.mean([rollout.final_state.n_real_atoms > 0 for rollout in rollouts])
        ),
        "reference_support_fraction": float(
            np.mean([key in reference_keys for key in final_keys])
        ),
        "reference_mode_coverage_fraction": float(
            np.mean([key in counts for key in reference_keys])
        ),
        "reference_total_variation": float(total_variation),
        "distinct_final_states": len(counts),
        "unique_fraction": len(counts) / sample_count,
        "event_budget_exhaustion_fraction": float(
            np.mean([rollout.exhausted_event_budget for rollout in rollouts])
        ),
        "mean_events": float(np.mean([len(rollout.event_times) for rollout in rollouts])),
        "top_final_states": counts.most_common(20),
    }


def _state_fingerprint(state: MolecularGraph) -> tuple[bytes, ...]:
    return (
        state.atom_types.tobytes(),
        state.formal_charges.tobytes(),
        state.implicit_h_counts.tobytes(),
        state.bonds.tobytes(),
    )
